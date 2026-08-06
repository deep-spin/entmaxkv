import math

import torch
import triton
import triton.language as tl


@triton.jit
def _page_gaussian_stats_kernel(
    Q,
    K_MEAN,
    K_STD,
    MU_OUT,
    VAR_OUT,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kp,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_sp,
    stride_sd,
    stride_ob,
    stride_oh,
    stride_op,
    num_pages: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    page_tile = tl.program_id(2)

    page_offsets = page_tile * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < num_pages

    q_base = Q + batch_id * stride_qb + head_id * stride_qh
    k_base = K_MEAN + batch_id * stride_kb + head_id * stride_kh
    s_base = K_STD + batch_id * stride_sb + head_id * stride_sh

    mu_acc = tl.zeros([BLOCK_PAGES], dtype=tl.float32)
    var_acc = tl.zeros([BLOCK_PAGES], dtype=tl.float32)

    for dim_start in range(0, head_dim, BLOCK_DIM):
        dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < head_dim

        q = tl.load(q_base + dim_offsets * stride_qd, mask=dim_mask, other=0.0).to(tl.float32)
        q_scaled = q * scale
        load_mask = page_mask[:, None] & dim_mask[None, :]

        # Consume the mean tile before loading the std tile. Keeping both
        # BLOCK_PAGES x BLOCK_DIM tiles live at once increases register
        # pressure and can cause spills for D=128 and larger page tiles.
        k_ptrs = page_offsets[:, None] * stride_kp + dim_offsets[None, :] * stride_kd
        k_mean = tl.load(k_base + k_ptrs, mask=load_mask, other=0.0).to(tl.float32)
        mu_acc += tl.sum(k_mean * q_scaled[None, :], axis=1)

        q_var_scaled = q_scaled * q_scaled
        s_ptrs = page_offsets[:, None] * stride_sp + dim_offsets[None, :] * stride_sd
        k_std = tl.load(s_base + s_ptrs, mask=load_mask, other=0.0).to(tl.float32)
        var_acc += tl.sum(k_std * k_std * q_var_scaled[None, :], axis=1)

    out_base = MU_OUT + batch_id * stride_ob + head_id * stride_oh
    var_base = VAR_OUT + batch_id * stride_ob + head_id * stride_oh
    tl.store(out_base + page_offsets * stride_op, mu_acc, mask=page_mask)
    tl.store(var_base + page_offsets * stride_op, var_acc, mask=page_mask)


@triton.jit
def _global_gaussian_stats_kernel(
    MU_PAGE,
    VAR_PAGE,
    MU_GLOBAL,
    MEAN_VARIANCE,
    VARIANCE_OF_MEANS,
    stride_pb,
    stride_ph,
    stride_pp,
    stride_ob,
    stride_oh,
    pages_for_stats: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    WRITE_SIGMA: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_PAGES)
    mask = offsets < pages_for_stats
    page_base = MU_PAGE + batch_id * stride_pb + head_id * stride_ph
    var_base = VAR_PAGE + batch_id * stride_pb + head_id * stride_ph

    mu = tl.load(page_base + offsets * stride_pp, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_base + offsets * stride_pp, mask=mask, other=0.0).to(tl.float32)

    denom = pages_for_stats + 0.0
    mu_sum = tl.sum(mu, axis=0)
    mu_sq_sum = tl.sum(mu * mu, axis=0)
    var_sum = tl.sum(var, axis=0)

    mean = mu_sum / denom
    mean_var = var_sum / denom
    var_means = tl.maximum(mu_sq_sum / denom - mean * mean, 0.0)

    # The variance array is dead after this reduction in the decode path.
    # Reuse its storage for sigma and avoid a separate elementwise torch.sqrt
    # launch over every page.
    if WRITE_SIGMA:
        tl.store(var_base + offsets * stride_pp, tl.sqrt(var), mask=mask)

    out_idx = batch_id * stride_ob + head_id * stride_oh
    tl.store(MU_GLOBAL + out_idx, mean)
    tl.store(MEAN_VARIANCE + out_idx, mean_var)
    tl.store(VARIANCE_OF_MEANS + out_idx, var_means)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def compute_page_gaussian_stats_triton(
    q: torch.Tensor,
    k_mean: torch.Tensor,
    k_std: torch.Tensor,
    pages_for_stats: int,
    *,
    block_pages: int = 16,
    num_warps: int = 4,
    return_sigma: bool = False,
):
    """
    Compute per-page Gaussian score moments with fused Triton kernels.

    Returns:
        mu_scores_per_page, score_variance_per_page (or sigma when
        return_sigma=True), mean_variance, variance_of_means, raw_mu_global.
        All outputs are float32.
    """
    if not q.is_cuda:
        raise ValueError("Triton Gaussian page stats require CUDA tensors")
    if q.shape[0] != k_mean.shape[0] or q.shape[1] != k_mean.shape[1]:
        raise ValueError("q and k_mean must have matching batch/head dimensions")
    if k_mean.shape != k_std.shape:
        raise ValueError("k_mean and k_std must have identical shapes")
    if pages_for_stats <= 0:
        raise ValueError("pages_for_stats must be positive")
    if pages_for_stats > k_mean.shape[2]:
        raise ValueError("pages_for_stats exceeds the available pages")
    if q.shape[-1] != k_mean.shape[-1]:
        raise ValueError("q and page statistics must have matching head dimensions")
    if block_pages <= 0 or block_pages & (block_pages - 1):
        raise ValueError("block_pages must be a positive power of two")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")

    batch, num_heads, _, head_dim = q.shape
    num_pages = pages_for_stats
    scale = 1.0 / math.sqrt(head_dim)

    # Do not compact the page slice. Excluding the in-progress last page makes
    # this slice non-contiguous across heads/batches, and materializing it here
    # copies almost all page statistics on every decode step. The kernel is
    # stride-aware, so reading the original storage directly is both correct
    # and substantially cheaper for long contexts.
    k_mean_view = k_mean[:, :, :pages_for_stats, :]
    k_std_view = k_std[:, :, :pages_for_stats, :]

    mu_scores = torch.empty(
        (batch, num_heads, pages_for_stats), device=q.device, dtype=torch.float32
    )
    score_variance = torch.empty_like(mu_scores)

    block_dim = min(_next_power_of_2(head_dim), 256)
    grid_pages = (batch, num_heads, triton.cdiv(num_pages, block_pages))
    _page_gaussian_stats_kernel[grid_pages](
        q,
        k_mean_view,
        k_std_view,
        mu_scores,
        score_variance,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_mean_view.stride(0),
        k_mean_view.stride(1),
        k_mean_view.stride(2),
        k_mean_view.stride(3),
        k_std_view.stride(0),
        k_std_view.stride(1),
        k_std_view.stride(2),
        k_std_view.stride(3),
        mu_scores.stride(0),
        mu_scores.stride(1),
        mu_scores.stride(2),
        num_pages,
        head_dim,
        scale,
        BLOCK_PAGES=block_pages,
        BLOCK_DIM=block_dim,
        num_warps=num_warps,
    )

    mu_global = torch.empty((batch, num_heads, 1), device=q.device, dtype=torch.float32)
    mean_variance = torch.empty_like(mu_global)
    variance_of_means = torch.empty_like(mu_global)
    reduce_block_pages = _next_power_of_2(pages_for_stats)
    reduce_warps = 1 if reduce_block_pages <= 64 else 4 if reduce_block_pages <= 2048 else 8
    _global_gaussian_stats_kernel[(batch, num_heads)](
        mu_scores,
        score_variance,
        mu_global,
        mean_variance,
        variance_of_means,
        mu_scores.stride(0),
        mu_scores.stride(1),
        mu_scores.stride(2),
        mu_global.stride(0),
        mu_global.stride(1),
        pages_for_stats,
        BLOCK_PAGES=reduce_block_pages,
        WRITE_SIGMA=return_sigma,
        num_warps=reduce_warps,
    )

    return mu_scores, score_variance, mean_variance, variance_of_means, mu_global
