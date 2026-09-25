"""Entmax decode restricted to a selected set of KV pages.

These are the ``paged_decode_utils`` stages with two structural changes:

1. The page loop walks SELECTED SLOTS instead of every logical page. Each slot
   carries a logical page index (or -1 for padding); the logical index drives
   the ALiBi position and resolves through the block table to a physical
   block.
2. K is read exactly once. The first stage computes every selected token's
   score and writes it to an fp32 scratch buffer (-inf for padding slots and
   the partial tail); the histogram and every tau refinement then stream that
   buffer instead of re-gathering K and recomputing QK, and the output stage
   reads it alongside V. The stored value is the one each stage would have
   recomputed, so the result changes only by fp32 summation order in the
   tiled reductions; the memory traffic drops from 2 + niter + 1 K passes
   to one.

The scoring contract is otherwise untouched — GQA without materialising K/V,
query-head ALiBi slopes, the (alpha-1)-scaled score domain, entmax-1.5 and
sparsemax, and masking of invalid tokens in the partial tail. Normalisation is
over the selected tokens, so the tau bracket uses a per-head candidate count
(SEL_LENS) rather than the request's seq_len.

The result is exact with respect to the selected set, approximate with respect
to the full cache.
"""

import math

import torch
import triton
import triton.language as tl

from entmaxkv.kernels.paged_decode_utils import (
    DECODE_HIST_BINS,
    DECODE_PARTITIONS,
    _reduce_max,
    _reduce_output,
    _refine_update,
    default_decode_niter,
)

# Tokens per 2-D tile in the score-buffer stages (SLOT_TILE * BLOCK_SIZE).
SCORE_TILE_TOKENS = 512


@triton.jit
def _sel_scores(
    Q, K, BLOCK_TABLE, SELECTED, SEQ_LENS, SLOPES, SCORES, PARTIAL_MAX,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_btb: tl.constexpr, stride_selb: tl.constexpr,
    stride_selh: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr,
    N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    TOPK_PAGES: tl.constexpr, PARTITIONS: tl.constexpr,
):
    """The only pass over K: store every selected score, reduce the max."""
    req = tl.program_id(0)
    head = tl.program_id(1)
    part = tl.program_id(2)
    hz = req * N_HEADS + head
    kv_head = head // (N_HEADS // N_KV_HEADS)
    seq_len = tl.load(SEQ_LENS + req).to(tl.int32)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_SIZE)
    q = tl.load(Q + req * stride_qb + head * stride_qh + offs_d)
    q = (q * ((alpha - 1.0) * sm_scale)).to(Q.dtype.element_ty)
    slope = tl.load(SLOPES + head).to(tl.float32) * (alpha - 1.0)
    q_pos = seq_len - 1
    sel_base = SELECTED + req * stride_selb + head * stride_selh
    score_max = -float("inf")
    slots_per_part: tl.constexpr = tl.cdiv(TOPK_PAGES, PARTITIONS)
    for local in range(slots_per_part):
        slot = part * slots_per_part + local
        in_range = slot < TOPK_PAGES
        logical = tl.load(sel_base + slot, mask=in_range,
                          other=-1).to(tl.int32)
        live = logical >= 0
        pos = logical * BLOCK_SIZE + offs_t
        valid = live & (pos < seq_len)
        physical = tl.load(BLOCK_TABLE + req * stride_btb + logical,
                           mask=live, other=0).to(tl.int64)
        k_ptr = (K + physical * stride_kblock
                 + offs_t[:, None] * stride_ktok
                 + kv_head * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptr, mask=valid[:, None], other=0.0)
        score = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        score += slope * (pos - q_pos).to(tl.float32)
        score = tl.where(valid, score, -float("inf"))
        tl.store(SCORES + (hz * TOPK_PAGES + slot) * BLOCK_SIZE + offs_t,
                 score, mask=in_range)
        score_max = tl.maximum(score_max, tl.max(score))
    tl.store(PARTIAL_MAX + hz * PARTITIONS + part, score_max)


@triton.jit
def _load_score_tile(SCORES, hz, part, tile_start,
                     TOPK_PAGES: tl.constexpr, PARTITIONS: tl.constexpr,
                     BLOCK_SIZE: tl.constexpr, SLOT_TILE: tl.constexpr):
    """[SLOT_TILE, BLOCK_SIZE] stored scores of this partition's slots
    ``tile_start ..``; slots outside the partition or the budget read -inf."""
    slots_per_part: tl.constexpr = tl.cdiv(TOPK_PAGES, PARTITIONS)
    local = tile_start + tl.arange(0, SLOT_TILE)
    slot = part * slots_per_part + local
    live = (local < slots_per_part) & (slot < TOPK_PAGES)
    ptr = (SCORES + (hz * TOPK_PAGES + slot)[:, None] * BLOCK_SIZE
           + tl.arange(0, BLOCK_SIZE)[None, :])
    return tl.load(ptr, mask=live[:, None], other=-float("inf")), slot, live


@triton.jit
def _sel_histogram(
    SCORES, GLOBAL_MAX, HISTOGRAM,
    N_HEADS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    TOPK_PAGES: tl.constexpr, PARTITIONS: tl.constexpr,
    BINS: tl.constexpr, SLOT_TILE: tl.constexpr,
):
    req, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_HEADS + head
    bins = tl.arange(0, BINS)
    t0 = tl.load(GLOBAL_MAX + hz) - 1.0
    hist = tl.zeros((BINS,), tl.int32)
    slots_per_part: tl.constexpr = tl.cdiv(TOPK_PAGES, PARTITIONS)
    for tile_start in range(0, slots_per_part, SLOT_TILE):
        score, _, _ = _load_score_tile(SCORES, hz, part, tile_start,
                                       TOPK_PAGES, PARTITIONS, BLOCK_SIZE,
                                       SLOT_TILE)
        valid = score > -float("inf")
        projected = tl.where(valid, score - t0, -1.0)
        bin_idx = tl.minimum((projected * BINS).to(tl.int32), BINS - 1)
        included = valid & (bin_idx >= 0)
        onehot = ((bin_idx[:, :, None] == bins[None, None, :])
                  & included[:, :, None]).to(tl.int32)
        hist += tl.sum(tl.sum(onehot, axis=0), axis=0)
    base = (hz * PARTITIONS + part) * BINS
    tl.store(HISTOGRAM + base + bins, hist)


@triton.jit
def _sel_histogram_init(HISTOGRAM, GLOBAL_MAX, SEL_LENS, TAU, TAU_LO, TAU_HI,
                        stride_lb: tl.constexpr,
                        alpha: tl.constexpr, N_HEADS: tl.constexpr,
                        PARTITIONS: tl.constexpr, BINS: tl.constexpr):
    """As ``_histogram_init``, but the Peters et al. bracket's L is the size of
    the CANDIDATE set for this (request, head), not the full sequence."""
    req, head = tl.program_id(0), tl.program_id(1)
    hz = req * N_HEADS + head
    sel_len = tl.load(SEL_LENS + req * stride_lb + head).to(tl.float32)
    bins = tl.arange(0, BINS)
    hist = tl.zeros((BINS,), tl.int32)
    for part in range(PARTITIONS):
        base = (hz * PARTITIONS + part) * BINS
        hist += tl.load(HISTOGRAM + base + bins)
    maximum = tl.load(GLOBAL_MAX + hz)
    sum_z, sum_k, sum_z2 = 1.0, 1.0, 1.0
    for sj in range(BINS - 1, -1, -1):
        count = tl.sum(hist * (bins == sj).to(tl.int32)).to(tl.float32)
        if sj == BINS - 1:
            count -= 1.0
        edge = sj / BINS
        new_z = sum_z + count * edge
        new_k = sum_k + count
        new_z2 = sum_z2 + count * edge * edge
        if alpha == 2.0:
            accept = (new_z - new_k * edge) < 1.0
        else:
            accept = (new_k * edge * edge - 2.0 * new_z * edge
                      + new_z2) < 1.0
        sum_z = tl.where(accept, new_z, sum_z)
        sum_k = tl.where(accept, new_k, sum_k)
        sum_z2 = tl.where(accept, new_z2, sum_z2)
    if alpha == 2.0:
        lower = maximum + (sum_z - 1.0) / sum_k - 1.0
    else:
        disc = tl.maximum(sum_z * sum_z - sum_k * (sum_z2 - 1.0), 0.0)
        lower = maximum + (sum_z - tl.sqrt(disc)) / sum_k - 1.0
    bracket_lo = maximum - 1.0
    bracket_hi = maximum - tl.exp((1.0 - alpha) * tl.log(sel_len))
    initial = tl.minimum(tl.maximum(lower + 0.5 / BINS, bracket_lo), bracket_hi)
    tl.store(TAU_LO + hz, bracket_lo)
    tl.store(TAU_HI + hz, bracket_hi)
    tl.store(TAU + hz, initial)


@triton.jit
def _sel_refine(
    SCORES, GLOBAL_MAX, TAU, SLOT_MASK, PARTIAL0, PARTIAL1, PARTIAL2,
    alpha: tl.constexpr, N_HEADS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    TOPK_PAGES: tl.constexpr, PARTITIONS: tl.constexpr,
    BUILD_MASK: tl.constexpr, SLOT_TILE: tl.constexpr,
):
    req, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_HEADS + head
    tau = tl.load(TAU + hz)
    # Conservative lower bound on tau*, as in the exact path: Eq. 5's lower end
    # can never exclude a token the true threshold would admit.
    support_lo = tl.load(GLOBAL_MAX + hz) - 1.0
    acc0, acc1, acc2 = 0.0, 0.0, 0.0
    slots_per_part: tl.constexpr = tl.cdiv(TOPK_PAGES, PARTITIONS)
    for tile_start in range(0, slots_per_part, SLOT_TILE):
        score, slot, live = _load_score_tile(SCORES, hz, part, tile_start,
                                             TOPK_PAGES, PARTITIONS,
                                             BLOCK_SIZE, SLOT_TILE)
        if BUILD_MASK:
            candidate = tl.max(score, axis=1) > support_lo
            tl.store(SLOT_MASK + hz * TOPK_PAGES + slot,
                     candidate.to(tl.int8), mask=live)
        active = score > tau
        x = tl.where(active, score - tau, 0.0)
        if alpha == 1.5:
            acc0 += tl.sum(tl.sum(x * x, axis=1))
            acc1 += tl.sum(tl.sum(x, axis=1))
            acc2 += tl.sum(tl.sum(active.to(tl.float32), axis=1))
        else:
            acc0 += tl.sum(tl.sum(x, axis=1))
            acc1 += tl.sum(tl.sum(active.to(tl.float32), axis=1))
    base = hz * PARTITIONS + part
    tl.store(PARTIAL0 + base, acc0)
    tl.store(PARTIAL1 + base, acc1)
    tl.store(PARTIAL2 + base, acc2)


@triton.jit
def _sel_output(
    V, BLOCK_TABLE, SELECTED, SCORES, TAU, SLOT_MASK,
    PARTIAL_OUT, PARTIAL_MASS,
    stride_vblock: tl.constexpr, stride_vtok: tl.constexpr,
    stride_vh: tl.constexpr, stride_vd: tl.constexpr,
    stride_btb: tl.constexpr, stride_selb: tl.constexpr,
    stride_selh: tl.constexpr,
    alpha: tl.constexpr,
    N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    TOPK_PAGES: tl.constexpr, PARTITIONS: tl.constexpr,
):
    req, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_HEADS + head
    kv_head = head // (N_HEADS // N_KV_HEADS)
    offs_d = tl.arange(0, HEAD_DIM)
    tau = tl.load(TAU + hz)
    sel_base = SELECTED + req * stride_selb + head * stride_selh
    acc = tl.zeros((HEAD_DIM,), tl.float32)
    mass = 0.0
    offs_t = tl.arange(0, BLOCK_SIZE)
    slots_per_part: tl.constexpr = tl.cdiv(TOPK_PAGES, PARTITIONS)
    for local in range(slots_per_part):
        slot = part * slots_per_part + local
        in_range = slot < TOPK_PAGES
        logical = tl.load(sel_base + slot, mask=in_range,
                          other=-1).to(tl.int32)
        live = logical >= 0
        candidate_page = tl.load(SLOT_MASK + hz * TOPK_PAGES + slot,
                                 mask=in_range, other=0) != 0
        score = tl.load(SCORES + (hz * TOPK_PAGES + slot) * BLOCK_SIZE
                        + offs_t, mask=in_range & candidate_page,
                        other=-float("inf"))
        # Only tokens inside the support contribute, so only their V rows are
        # read; padding and the partial tail are already -inf in SCORES.
        active = score > tau
        x = tl.where(active, score - tau, 0.0)
        if alpha == 1.5:
            prob = x * x
        else:
            prob = x
        physical = tl.load(BLOCK_TABLE + req * stride_btb + logical,
                           mask=live, other=0).to(tl.int64)
        v_ptr = (V + physical * stride_vblock
                 + offs_t[:, None] * stride_vtok
                 + kv_head * stride_vh + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptr, mask=active[:, None], other=0.0)
        acc += tl.sum(prob[:, None] * v, axis=0)
        mass += tl.sum(prob)
    base = (hz * PARTITIONS + part) * HEAD_DIM
    tl.store(PARTIAL_OUT + base + offs_d, acc)
    tl.store(PARTIAL_MASS + hz * PARTITIONS + part, mass)


def entmax_decode_selected_pages(
    q: torch.Tensor,               # (B, H_q, D)
    kv_cache: torch.Tensor,        # (2, num_blocks, block_size, H_kv, D)
    block_table: torch.Tensor,     # (B, max_pages) int32
    selected_pages: torch.Tensor,  # (B, H_q, topk_pages) int32, logical, -1 pad
    sel_lens: torch.Tensor,        # (B, H_q) int32, candidate token counts
    seq_lens: torch.Tensor,        # (B,) int32
    alpha: float,
    alibi_slopes: torch.Tensor,    # (H_q,) by QUERY head
    niter: int | None = None,
    out: torch.Tensor | None = None,
    workspace: "SelectedDecodeWorkspace | None" = None,
    return_debug: bool = False,
    partitions: int = DECODE_PARTITIONS,
) -> torch.Tensor:
    """Entmax decode over ``selected_pages`` only.

    Exact with respect to the selected candidate set and approximate with
    respect to the full cache. Mirrors ``entmax_decode_paged_native`` stage for
    stage while walking selected slots instead of every logical page, and
    reading K once into a score buffer the later stages reuse.
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
        raise ValueError("selected-page decode supports alpha 1.5 and 2.0")
    if dim not in (32, 64, 128):
        raise ValueError(
            "selected-page decode supports head dimensions 32, 64, and 128")
    if niter < 1:
        raise ValueError("selected-page decode requires at least one refinement")
    if partitions < 1 or partitions & (partitions - 1):
        raise ValueError(
            "selected-page decode partitions must be a positive power of 2")
    if selected_pages.shape[:2] != (batch, nheads):
        raise ValueError("selected_pages must be [B, H_q, topk_pages]")
    if sel_lens.shape != (batch, nheads):
        raise ValueError("sel_lens must be [B, H_q]")
    if selected_pages.dtype != torch.int32 or sel_lens.dtype != torch.int32:
        raise TypeError("selected_pages and sel_lens must be int32")
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise TypeError("block_table and seq_lens must be int32")

    slopes = alibi_slopes.to(device=q.device, dtype=torch.float32)
    if slopes.shape != (nheads,):
        raise ValueError(f"alibi_slopes must have shape ({nheads},)")
    if out is None:
        out = torch.empty_like(q)
    topk_pages = selected_pages.shape[2]
    bins = DECODE_HIST_BINS
    if workspace is None:
        workspace = SelectedDecodeWorkspace()
    ws = workspace.get(batch, nheads, topk_pages, dim, partitions, block_size,
                       q.device)
    slots_per_part = triton.cdiv(topk_pages, partitions)
    slot_tile = min(triton.next_power_of_2(slots_per_part),
                    max(1, SCORE_TILE_TOKENS // block_size))

    k_cache, v_cache = kv_cache.unbind(0)
    sel_common = dict(
        stride_btb=block_table.stride(0),
        stride_selb=selected_pages.stride(0),
        stride_selh=selected_pages.stride(1),
        alpha=float(alpha), N_HEADS=nheads, N_KV_HEADS=n_kv_heads,
        HEAD_DIM=dim, BLOCK_SIZE=block_size, TOPK_PAGES=topk_pages,
        PARTITIONS=partitions, num_warps=4,
    )
    score_common = dict(
        N_HEADS=nheads, BLOCK_SIZE=block_size, TOPK_PAGES=topk_pages,
        PARTITIONS=partitions, SLOT_TILE=slot_tile, num_warps=4,
    )
    grid_bh = (batch, nheads)
    grid_bhp = (batch, nheads, partitions)
    _sel_scores[grid_bhp](
        q, k_cache, block_table, selected_pages, seq_lens, slopes,
        ws["scores"], ws["partial_max"],
        stride_qb=q.stride(0), stride_qh=q.stride(1),
        stride_kblock=k_cache.stride(0), stride_ktok=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        sm_scale=1.0 / math.sqrt(dim), **sel_common,
    )
    _reduce_max[grid_bh](ws["partial_max"], ws["global_max"], N_HEADS=nheads,
                         PARTITIONS=partitions, num_warps=1)
    _sel_histogram[grid_bhp](
        ws["scores"], ws["global_max"], ws["histogram"], BINS=bins,
        **score_common,
    )
    _sel_histogram_init[grid_bh](
        ws["histogram"], ws["global_max"], sel_lens, ws["tau"],
        ws["tau_lo"], ws["tau_hi"], stride_lb=sel_lens.stride(0),
        alpha=float(alpha), N_HEADS=nheads, PARTITIONS=partitions,
        BINS=bins, num_warps=1,
    )
    for iteration in range(int(niter)):
        _sel_refine[grid_bhp](
            ws["scores"], ws["global_max"], ws["tau"], ws["slot_mask"],
            ws["partial0"], ws["partial1"], ws["partial2"],
            alpha=float(alpha), BUILD_MASK=iteration == 0, **score_common,
        )
        _refine_update[grid_bh](
            ws["partial0"], ws["partial1"], ws["partial2"], ws["tau"],
            ws["tau_lo"], ws["tau_hi"], alpha=float(alpha), N_HEADS=nheads,
            PARTITIONS=partitions, num_warps=1,
        )
    _sel_output[grid_bhp](
        v_cache, block_table, selected_pages, ws["scores"], ws["tau"],
        ws["slot_mask"], ws["partial_out"], ws["partial_mass"],
        stride_vblock=v_cache.stride(0), stride_vtok=v_cache.stride(1),
        stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
        **sel_common,
    )
    _reduce_output[grid_bh](
        ws["partial_out"], ws["partial_mass"], out, ws["mass"],
        stride_ob=out.stride(0), stride_oh=out.stride(1),
        N_HEADS=nheads, HEAD_DIM=dim, PARTITIONS=partitions, num_warps=4,
    )
    if return_debug:
        return out, {"tau": ws["tau"], "mass": ws["mass"],
                     "slot_mask": ws["slot_mask"]}
    return out


class SelectedDecodeWorkspace:
    """Persistent, shape-keyed scratch for the selected-page decode stages.

    Buffers are allocated once per distinct shape and reused, so a decode step
    allocates nothing. Every stride the kernels take is a ``tl.constexpr``, so
    reuse only pays off if the logical shapes stay stable — hence the key.
    """

    def __init__(self):
        self._cache: dict = {}

    def get(self, batch: int, nheads: int, topk_pages: int, dim: int,
            partitions: int, block_size: int, device: torch.device) -> dict:
        key = (batch, nheads, topk_pages, dim, partitions, block_size,
               str(device))
        ws = self._cache.get(key)
        if ws is not None:
            return ws
        bins = DECODE_HIST_BINS
        f32 = dict(device=device, dtype=torch.float32)
        partial_shape = (batch, nheads, partitions)
        ws = {
            # Every selected token's (alpha-1)-scaled score, -inf where
            # invalid; written once by _sel_scores, read by every later stage.
            "scores": torch.empty((batch, nheads, topk_pages, block_size),
                                  **f32),
            "partial_max": torch.empty(partial_shape, **f32),
            "global_max": torch.empty((batch, nheads), **f32),
            "histogram": torch.empty((*partial_shape, bins), device=device,
                                     dtype=torch.int32),
            "tau": torch.empty((batch, nheads), **f32),
            "tau_lo": torch.empty((batch, nheads), **f32),
            "tau_hi": torch.empty((batch, nheads), **f32),
            "partial0": torch.empty(partial_shape, **f32),
            "partial1": torch.empty(partial_shape, **f32),
            # _refine_update reads PARTIAL2 unconditionally; at alpha=2.0
            # _sel_refine never writes it and its ddf coefficient is a
            # hardcoded 0.0, so it is dead — zero it anyway, deliberately.
            "partial2": torch.zeros(partial_shape, **f32),
            "slot_mask": torch.empty((batch, nheads, topk_pages), device=device,
                                     dtype=torch.int8),
            "partial_out": torch.empty((*partial_shape, dim), **f32),
            "partial_mass": torch.empty(partial_shape, **f32),
            "mass": torch.empty((batch, nheads), **f32),
        }
        self._cache[key] = ws
        return ws
