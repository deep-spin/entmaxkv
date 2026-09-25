"""Query-conditioned Gaussian score moments over paged key metadata.

ALiBi is added to each page mean BEFORE the global reduction, so the global
mean and sigma describe the scores the decoder actually sees.

The live page count is a runtime argument: only the output row stride is a
kernel constant, so a growing context never triggers a recompile.
"""

import math

import torch
import triton
import triton.language as tl

# Pages reduced per iteration of the global-moment loop.
GLOBAL_TILE_PAGES = 1024


@triton.jit(do_not_specialize=["n_pages"])
def _paged_gaussian_score_stats(
    Q, K_MEAN, K_STD, BLOCK_TABLE, SEQ_LENS, SLOPES, MU, VARIANCE, n_pages,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_mb: tl.constexpr, stride_mh: tl.constexpr,
    stride_btb: tl.constexpr, stride_ob: tl.constexpr,
    stride_oh: tl.constexpr, sm_scale: tl.constexpr,
    N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    """Compute a tile of pages instead of serialising every page per head."""
    req = tl.program_id(0)
    head = tl.program_id(1)
    page_tile = tl.program_id(2)
    kv_head = head // (N_HEADS // N_KV_HEADS)
    seq_len = tl.load(SEQ_LENS + req).to(tl.int32)
    tail = (seq_len - 1) // BLOCK_SIZE
    q_pos = seq_len - 1

    pages = page_tile * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    live_pages = (pages < tail) & (pages < n_pages)
    physical = tl.load(BLOCK_TABLE + req * stride_btb + pages,
                       mask=live_pages, other=0).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + req * stride_qb + head * stride_qh + offs_d
                ).to(tl.float32)
    q_scaled = q * sm_scale
    q_variance = q_scaled * q_scaled

    metadata = (physical[:, None] * stride_mb + kv_head * stride_mh
                + offs_d[None, :])
    load_mask = live_pages[:, None]
    mean = tl.load(K_MEAN + metadata, mask=load_mask,
                   other=0.0).to(tl.float32)
    mu = tl.sum(mean * q_scaled[None, :], axis=1)

    # Consume mean before loading std to keep only one page x dim tile live.
    std = tl.load(K_STD + metadata, mask=load_mask,
                  other=0.0).to(tl.float32)
    variance = tl.sum(std * std * q_variance[None, :], axis=1)
    slope = tl.load(SLOPES + head).to(tl.float32)
    page_mean_pos = pages * BLOCK_SIZE + (BLOCK_SIZE - 1) * 0.5
    mu += slope * (page_mean_pos - q_pos)

    out = req * stride_ob + head * stride_oh + pages
    tl.store(MU + out, mu, mask=pages < n_pages)
    tl.store(VARIANCE + out, variance, mask=pages < n_pages)


@triton.jit(do_not_specialize=["n_pages"])
def _global_gaussian_stats(
    MU, VARIANCE, SEQ_LENS, MEAN, SIGMA_GLOBAL, TOKEN_COUNTS,
    HISTORICAL_PAGES, n_pages,
    stride_ib: tl.constexpr, stride_ih: tl.constexpr,
    stride_ob: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    """Reduce global moments and turn page variance into sigma in-place."""
    req = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_PAGES)
    historical = (tl.load(SEQ_LENS + req).to(tl.int32) - 1) // BLOCK_SIZE
    row = req * stride_ib + head * stride_ih
    sum_mu, sum_mu2, sum_var = 0.0, 0.0, 0.0
    for start in range(0, n_pages, BLOCK_PAGES):
        pages = start + offsets
        active = pages < historical
        mu = tl.load(MU + row + pages, mask=active, other=0.0)
        variance = tl.load(VARIANCE + row + pages, mask=active, other=0.0)
        sum_mu += tl.sum(mu, axis=0)
        sum_mu2 += tl.sum(mu * mu, axis=0)
        sum_var += tl.sum(variance, axis=0)
        # VARIANCE is dead after this reduction. Reuse it as the per-page
        # sigma buffer consumed by the mixture solver and selector.
        tl.store(VARIANCE + row + pages, tl.sqrt(tl.maximum(variance, 0.0)),
                 mask=pages < n_pages)
    count = tl.maximum(historical, 1).to(tl.float32)
    mean = sum_mu / count
    variance_of_means = tl.maximum(sum_mu2 / count - mean * mean, 0.0)
    sigma_global = tl.sqrt(sum_var / count + variance_of_means)

    out = req * stride_ob + head
    tl.store(MEAN + out, mean)
    tl.store(SIGMA_GLOBAL + out, tl.maximum(sigma_global, 1.0e-6))
    tl.store(TOKEN_COUNTS + out, historical * BLOCK_SIZE)
    tl.store(HISTORICAL_PAGES + out, historical)


def gaussian_score_stats(
    q: torch.Tensor,             # (B, H_q, D)
    k_mean: torch.Tensor,        # (num_blocks, H_kv, D) fp32
    k_std: torch.Tensor,
    block_table: torch.Tensor,   # (B, max_pages) int32
    seq_lens: torch.Tensor,      # (B,) int32
    slopes: torch.Tensor,        # (H_q,) fp32, by QUERY head
    block_size: int,
    mu_out: torch.Tensor,        # (B, H_q, >= n_pages) fp32
    sigma_out: torch.Tensor,
    mean_out: torch.Tensor,      # (B, H_q) fp32
    sigma_global_out: torch.Tensor,
    token_counts_out: torch.Tensor,     # (B, H_q) int32
    historical_pages_out: torch.Tensor,
    n_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fill page and global score moments without PyTorch tensor operations.

    ``n_pages`` (host int) is the live block-table width,
    ``ceil(max(seq_lens) / block_size)``; defaults to the full table.
    """
    batch, nheads, dim = q.shape
    if n_pages is None:
        n_pages = block_table.shape[1]
    block_pages = 16
    _paged_gaussian_score_stats[(batch, nheads,
                                 triton.cdiv(n_pages, block_pages))](
        q, k_mean, k_std, block_table, seq_lens, slopes, mu_out, sigma_out,
        n_pages,
        stride_qb=q.stride(0), stride_qh=q.stride(1),
        stride_mb=k_mean.stride(0), stride_mh=k_mean.stride(1),
        stride_btb=block_table.stride(0), stride_ob=mu_out.stride(0),
        stride_oh=mu_out.stride(1), sm_scale=1.0 / math.sqrt(dim),
        N_HEADS=nheads, N_KV_HEADS=k_mean.shape[1], HEAD_DIM=dim,
        BLOCK_SIZE=block_size, BLOCK_PAGES=block_pages, num_warps=4)

    _global_gaussian_stats[(batch, nheads)](
        mu_out, sigma_out, seq_lens, mean_out, sigma_global_out,
        token_counts_out, historical_pages_out, n_pages,
        stride_ib=mu_out.stride(0), stride_ih=mu_out.stride(1),
        stride_ob=mean_out.stride(0), BLOCK_SIZE=block_size,
        BLOCK_PAGES=GLOBAL_TILE_PAGES, num_warps=4)
    return mu_out, sigma_out, mean_out, sigma_global_out
