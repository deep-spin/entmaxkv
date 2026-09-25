"""Exact decode over the paged KV cache (alpha 1.5 and 2.0).

The kernels live in ``entmaxkv.kernels.paged_decode_utils`` and are shared
verbatim with entmax_vllm. This is the fallback when sparse decoding cannot
save work (short contexts, or a budget that already covers the context).
"""

import math

import torch

from entmaxkv.kernels.paged_decode_utils import (
    DECODE_HIST_BINS,
    DECODE_PARTITIONS,
    _histogram_init,
    _paged_histogram,
    _paged_output,
    _paged_refine,
    _paged_scores_max,
    _reduce_max,
    _reduce_output,
    _refine_update,
    default_decode_niter,
)


def entmax_decode_paged_native(
    q: torch.Tensor,             # (B, H_q, D)
    kv_cache: torch.Tensor,      # (2, num_blocks, block_size, H_kv, D)
    block_table: torch.Tensor,   # (B, max_pages) int32
    seq_lens: torch.Tensor,      # (B,) int32
    alpha: float,
    alibi_slopes: torch.Tensor,  # (H_q,) by QUERY head
    niter: int | None = None,
    out: torch.Tensor | None = None,
    return_debug: bool = False,
    workspace: "ExactDecodeWorkspace | None" = None,
) -> torch.Tensor:
    """Exact GQA decode directly over the paged cache.

    No dense K/V gather and no score tensor are created. ``block_table`` is
    traversed in logical token order and only supplies physical load addresses.
    ``niter`` defaults to :func:`default_decode_niter` for ``alpha`` when unset.
    """
    if niter is None:
        niter = default_decode_niter(alpha)
    if q.ndim != 3:
        raise ValueError(f"q must have shape [B, H_q, D], got {tuple(q.shape)}")
    if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
        raise ValueError("kv_cache must be [2, num_blocks, block_size, H_kv, D]")
    batch, nheads, dim = q.shape
    _, _, block_size, n_kv_heads, cache_dim = kv_cache.shape
    if dim != cache_dim or nheads % n_kv_heads:
        raise ValueError("incompatible Q and paged KV head geometry")
    if alpha not in (1.5, 2.0):
        raise ValueError("native paged decode supports alpha 1.5 and 2.0")
    if dim not in (32, 64, 128):
        raise ValueError(
            "native paged decode supports head dimensions 32, 64, and 128")
    if niter < 1:
        raise ValueError("native paged decode requires at least one refinement")
    if block_table.shape[0] != batch or seq_lens.numel() != batch:
        raise ValueError("batch mismatch between q, block_table, and seq_lens")
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise TypeError("block_table and seq_lens must be int32")

    slopes = alibi_slopes.to(device=q.device, dtype=torch.float32)
    if slopes.shape != (nheads,):
        raise ValueError(f"alibi_slopes must have shape ({nheads},)")
    if out is None:
        out = torch.empty_like(q)
    max_pages = block_table.shape[1]
    partitions, bins = DECODE_PARTITIONS, DECODE_HIST_BINS
    if workspace is None:
        workspace = ExactDecodeWorkspace()
    ws = workspace.get(batch, nheads, max_pages, dim, q.device)
    partial_max, global_max = ws["partial_max"], ws["global_max"]
    histogram, tau = ws["histogram"], ws["tau"]
    tau_lo, tau_hi = ws["tau_lo"], ws["tau_hi"]
    partial0, partial1, partial2 = (ws["partial0"], ws["partial1"],
                                    ws["partial2"])
    page_mask, partial_out = ws["page_mask"], ws["partial_out"]
    partial_mass, mass = ws["partial_mass"], ws["mass"]
    k_cache, v_cache = kv_cache.unbind(0)
    common = dict(
        stride_qb=q.stride(0), stride_qh=q.stride(1),
        stride_kblock=k_cache.stride(0), stride_ktok=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_btb=block_table.stride(0), alpha=float(alpha),
        sm_scale=1.0 / math.sqrt(dim), N_HEADS=nheads,
        N_KV_HEADS=n_kv_heads, HEAD_DIM=dim, BLOCK_SIZE=block_size,
        MAX_PAGES=max_pages, PARTITIONS=partitions,
        num_warps=4,
    )
    grid_bh = (batch, nheads)
    grid_bhp = (batch, nheads, partitions)
    _paged_scores_max[grid_bhp](
        q, k_cache, block_table, seq_lens, slopes, partial_max, **common,
    )
    _reduce_max[grid_bh](partial_max, global_max, N_HEADS=nheads,
                         PARTITIONS=partitions, num_warps=1)
    _paged_histogram[grid_bhp](
        q, k_cache, block_table, seq_lens, slopes, global_max, histogram,
        BINS=bins, **common,
    )
    _histogram_init[grid_bh](
        histogram, global_max, seq_lens, tau, tau_lo, tau_hi, alpha=float(alpha),
        N_HEADS=nheads, PARTITIONS=partitions, BINS=bins, num_warps=1,
    )
    for iteration in range(int(niter)):
        _paged_refine[grid_bhp](
            q, k_cache, block_table, seq_lens, slopes, global_max, tau,
            page_mask, partial0, partial1, partial2,
            BUILD_MASK=iteration == 0, **common,
        )
        _refine_update[grid_bh](
            partial0, partial1, partial2, tau, tau_lo, tau_hi,
            alpha=float(alpha), N_HEADS=nheads, PARTITIONS=partitions,
            num_warps=1,
        )
    _paged_output[grid_bhp](
        q, k_cache, v_cache, block_table, seq_lens, slopes, tau, page_mask,
        partial_out, partial_mass,
        stride_vblock=v_cache.stride(0), stride_vtok=v_cache.stride(1),
        stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
        **common,
    )
    _reduce_output[grid_bh](
        partial_out, partial_mass, out, mass,
        stride_ob=out.stride(0), stride_oh=out.stride(1),
        N_HEADS=nheads, HEAD_DIM=dim, PARTITIONS=partitions, num_warps=4,
    )
    if return_debug:
        return out, {"tau": tau, "mass": mass, "page_mask": page_mask}
    return out


class ExactDecodeWorkspace:
    """Persistent, shape-keyed scratch for exact paged decode."""

    def __init__(self):
        self._cache: dict = {}

    def get(self, batch: int, nheads: int, max_pages: int, dim: int,
            device: torch.device) -> dict:
        key = (batch, nheads, max_pages, dim, str(device))
        ws = self._cache.get(key)
        if ws is not None:
            return ws
        partitions, bins = DECODE_PARTITIONS, DECODE_HIST_BINS
        f32 = dict(device=device, dtype=torch.float32)
        partial_shape = (batch, nheads, partitions)
        ws = {
            "partial_max": torch.empty(partial_shape, **f32),
            "global_max": torch.empty((batch, nheads), **f32),
            "histogram": torch.empty((*partial_shape, bins), device=device,
                                     dtype=torch.int32),
            "tau": torch.empty((batch, nheads), **f32),
            "tau_lo": torch.empty((batch, nheads), **f32),
            "tau_hi": torch.empty((batch, nheads), **f32),
            "partial0": torch.empty(partial_shape, **f32),
            "partial1": torch.empty(partial_shape, **f32),
            "partial2": torch.empty(partial_shape, **f32),
            "page_mask": torch.empty((batch, nheads, max_pages),
                                     device=device, dtype=torch.int8),
            "partial_out": torch.empty((*partial_shape, dim), **f32),
            "partial_mass": torch.empty(partial_shape, **f32),
            "mass": torch.empty((batch, nheads), **f32),
        }
        self._cache[key] = ws
        return ws
