import math
from typing import Optional

import torch

from entmaxkv.kernels.adadecode_paged import sparse_attention_decode_paged as _paged_kernel
from entmaxkv.kernels.adadecode_paged_gaussian_tau import (
    sparse_attention_decode_paged_gaussian_tau as _gaussian_tau_kernel,
)
from entmaxkv.gaussian_utils import (
    clamp_tau_to_selected_page_statistics,
    compute_gaussian_aware_statistics,
)
from entmaxkv.kernels.selection_pack import (
    cache_seqlens_from_page_counts_triton,
)
from entmaxkv.kv_cache import PagedKVCache


def sparse_attention_decode_gaussian_aware_entmax(
    q: torch.Tensor,
    kv_cache: PagedKVCache,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.5,
    safety_margin_z: float = 0.0,
    max_quantile: float = 0.995,
    alibi_slopes: torch.Tensor = None,
    scale: Optional[float] = None,
    splits: int = 32,
    niter: int = 2,
    append_cache: bool = False,
    tau_mode: str = "corrected",
    tau_correction_iters: Optional[int] = None,
    clamp_tau: bool = True,
    threshold_excess_margin_fraction: Optional[float] = 0.2,
    tau_clamp_page_max_quantile: float = 0.50,
    q_seqlens: torch.Tensor = None,
):
    """
    Gaussian-aware page selection feeding a paged entmax kernel.

    Page selection uses the Gaussian upper-bound criterion; the selected page
    indices (plus the last page) are passed directly to the paged
    kernel, avoiding any gather of KV data into a contiguous buffer.

    tau_mode:
        "exact" runs the decode computation the selected pages
        "fixed" uses the gaussian tau estimate directly.
        "corrected" uses gaussian tau after N correction iterations.
    """
    batch, num_heads, _, head_dim = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if append_cache:
        kv_cache.append(k_new, v_new)

    seq_len   = kv_cache.k_cache.shape[2]
    page_size = kv_cache.page_size

    tau_mode = tau_mode.lower()
    if tau_mode not in ("exact", "fixed", "corrected"):
        raise ValueError(f"Unsupported tau_mode={tau_mode!r}; expected 'exact', 'fixed', or 'corrected'")
    needs_tau = tau_mode != "exact"
    gaussian_stats = compute_gaussian_aware_statistics(
        q=q,
        k_mean=kv_cache.k_mean,
        k_std=kv_cache.k_std,
        seq_len=kv_cache.k_cache.shape[2],
        page_size=kv_cache.page_size,
        alpha=alpha,
        alibi_slopes=alibi_slopes,
        threshold_excess_margin_fraction=threshold_excess_margin_fraction,
        tau_clamp_page_max_quantile=tau_clamp_page_max_quantile,
    )

    raw, num_selected_per_head = kv_cache.select_gaussian_aware(
        q, alpha=alpha, safety_margin_z=safety_margin_z, max_quantile=max_quantile,
        gaussian_stats=gaussian_stats,
    )
    tau_hat = gaussian_stats["tau_hat"] if needs_tau else None

    page_indices = raw.to(torch.int32)  # [B, H, max_sel]

    if needs_tau and clamp_tau:
        tau_hat = clamp_tau_to_selected_page_statistics(
            tau_hat,
            gaussian_stats,
            page_indices,
            num_selected_per_head,
            page_size,
            alpha,
        )

    # cache_seqlens [B, H_kv]: compacted token count per head.
    last_page_size = seq_len - (math.ceil(seq_len / page_size) - 1) * page_size if seq_len > page_size else seq_len
    cache_seqlens = cache_seqlens_from_page_counts_triton(
        num_selected_per_head,
        page_size,
        last_page_size,
    )

    if tau_mode == "exact":
        _paged_kernel(
            q=q,
            k_cache=kv_cache.k_cache,
            v_cache=kv_cache.v_cache,
            out=out,
            cache_seqlens=cache_seqlens,
            page_indices=page_indices,
            page_size=page_size,
            alibi_slopes=alibi_slopes,
            is_causal=True,
            alpha=alpha,
            niter=niter,
            max_splits=splits,
            q_seqlens=q_seqlens,
        )
    else:
        if tau_correction_iters is None:
            if tau_mode == "fixed":
                tau_correction_iters = 0
            else:
                tau_correction_iters = 1 if alibi_slopes is None else 2 
        _gaussian_tau_kernel(
            q=q,
            k_cache=kv_cache.k_cache,
            v_cache=kv_cache.v_cache,
            out=out,
            cache_seqlens=cache_seqlens,
            page_indices=page_indices,
            page_size=page_size,
            tau=tau_hat,
            alibi_slopes=alibi_slopes,
            alpha=alpha,
            max_splits=splits,
            q_seqlens=q_seqlens,
            tau_correction_iters=tau_correction_iters,
        )

    return out
