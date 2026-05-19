import torch
import triton
import triton.language as tl
from statistics import NormalDist


@triton.jit
def _pack_page_mask_kernel(
    PAGE_MASK,
    OUT,
    COUNTS,
    stride_mb,
    stride_mh,
    stride_mp,
    stride_ob,
    stride_oh,
    stride_op,
    stride_cb,
    stride_ch,
    num_pages: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_PAGES)
    valid = offsets < num_pages
    mask = tl.load(
        PAGE_MASK + batch_id * stride_mb + head_id * stride_mh + offsets * stride_mp,
        mask=valid,
        other=0,
    ).to(tl.int32)

    ranks = tl.cumsum(mask, axis=0) - 1
    count = tl.sum(mask, axis=0)
    tl.store(COUNTS + batch_id * stride_cb + head_id * stride_ch, count)
    tl.store(
        OUT + batch_id * stride_ob + head_id * stride_oh + ranks * stride_op,
        offsets,
        mask=valid & (mask != 0),
    )


def pack_page_mask_triton(page_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not page_mask.is_cuda:
        raise ValueError("Triton page-mask pack requires a CUDA tensor")
    if page_mask.dim() != 3:
        raise ValueError("page_mask must have shape [batch, heads, pages]")

    page_mask_c = page_mask.contiguous()
    batch, num_heads, num_pages = page_mask_c.shape
    out = torch.empty((batch, num_heads, num_pages), dtype=torch.int32, device=page_mask.device)
    counts = torch.empty((batch, num_heads), dtype=torch.int32, device=page_mask.device)
    block_pages = triton.next_power_of_2(num_pages)
    _pack_page_mask_kernel[(batch, num_heads)](
        page_mask_c,
        out,
        counts,
        page_mask_c.stride(0),
        page_mask_c.stride(1),
        page_mask_c.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        counts.stride(0),
        counts.stride(1),
        num_pages,
        BLOCK_PAGES=block_pages,
        num_warps=8 if block_pages >= 1024 else 4,
    )
    return out, counts


@triton.jit
def _select_gaussian_threshold_pack_kernel(
    MU_PAGE,
    SIGMA_PAGE,
    TAU_HAT,
    SIGMA_GLOBAL,
    OUT,
    COUNTS,
    TAU_FLOOR,
    stride_mb,
    stride_mh,
    stride_mp,
    stride_tb,
    stride_th,
    stride_sb,
    stride_sh,
    stride_ob,
    stride_oh,
    stride_op,
    stride_cb,
    stride_ch,
    stride_fb,
    stride_fh,
    pages_for_stats: tl.constexpr,
    num_pages: tl.constexpr,
    alpha_minus_one: tl.constexpr,
    safety_margin_z: tl.constexpr,
    threshold_excess_margin_fraction: tl.constexpr,
    z_page: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_PAGES)
    stats_valid = offsets < pages_for_stats
    page_valid = offsets < num_pages
    is_last_page = (offsets == (num_pages - 1)) & (num_pages > 1)

    mu = tl.load(
        MU_PAGE + batch_id * stride_mb + head_id * stride_mh + offsets * stride_mp,
        mask=stats_valid,
        other=0.0,
    ).to(tl.float32)
    sigma = tl.load(
        SIGMA_PAGE + batch_id * stride_mb + head_id * stride_mh + offsets * stride_mp,
        mask=stats_valid,
        other=0.0,
    ).to(tl.float32)

    tau_hat = tl.load(TAU_HAT + batch_id * stride_tb + head_id * stride_th).to(tl.float32)
    sigma_global = tl.load(SIGMA_GLOBAL + batch_id * stride_sb + head_id * stride_sh).to(tl.float32)

    page_score_sum = tl.sum(tl.where(stats_valid, mu + 1.5 * sigma, 0.0), axis=0)
    page_score_upper = page_score_sum / pages_for_stats

    threshold_score = tau_hat / alpha_minus_one
    sigma_delta_score = safety_margin_z * sigma_global
    threshold_excess = tl.maximum(threshold_score - page_score_upper, 0.0)
    tau_delta_score = threshold_excess_margin_fraction * threshold_excess
    selection_delta_score = tl.maximum(sigma_delta_score, tau_delta_score)
    tau_floor = tau_hat - selection_delta_score * alpha_minus_one
    tl.store(TAU_FLOOR + batch_id * stride_fb + head_id * stride_fh, tau_floor)

    s_bar = mu + z_page * sigma
    selected = page_valid & ((alpha_minus_one * s_bar > tau_floor) | is_last_page)
    selected_i32 = selected.to(tl.int32)
    ranks = tl.cumsum(selected_i32, axis=0) - 1
    count = tl.sum(selected_i32, axis=0)

    tl.store(COUNTS + batch_id * stride_cb + head_id * stride_ch, count)
    tl.store(
        OUT + batch_id * stride_ob + head_id * stride_oh + ranks * stride_op,
        offsets,
        mask=selected,
    )


def select_gaussian_threshold_pack_triton(
    mu_scores_per_page: torch.Tensor,
    sigma_scores_per_page: torch.Tensor,
    tau_hat: torch.Tensor,
    sigma_global: torch.Tensor,
    page_size: int,
    alpha: float,
    safety_margin_z: float,
    max_quantile: float,
    threshold_excess_margin_fraction: float,
    pages_for_stats: int,
    num_pages: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not mu_scores_per_page.is_cuda:
        raise ValueError("Triton Gaussian threshold pack requires CUDA tensors")
    if mu_scores_per_page.dim() != 3:
        raise ValueError("mu_scores_per_page must have shape [batch, heads, pages]")
    if pages_for_stats <= 0 or num_pages <= 0:
        raise ValueError("pages_for_stats and num_pages must be positive")

    mu_c = mu_scores_per_page.contiguous()
    sigma_c = sigma_scores_per_page.contiguous()
    tau_c = tau_hat.squeeze(-1) if tau_hat.dim() == 3 and tau_hat.shape[-1] == 1 else tau_hat
    sigma_global_c = (
        sigma_global.squeeze(-1)
        if sigma_global.dim() == 3 and sigma_global.shape[-1] == 1
        else sigma_global
    )
    tau_c = tau_c.to(device=mu_c.device, dtype=torch.float32).contiguous()
    sigma_global_c = sigma_global_c.to(device=mu_c.device, dtype=torch.float32).contiguous()

    batch, num_heads, _ = mu_c.shape
    out = torch.empty((batch, num_heads, num_pages), dtype=torch.int32, device=mu_c.device)
    counts = torch.empty((batch, num_heads), dtype=torch.int32, device=mu_c.device)
    tau_floor = torch.empty((batch, num_heads), dtype=torch.float32, device=mu_c.device)

    block_pages = triton.next_power_of_2(num_pages)
    per_sample_prob = max_quantile ** (1.0 / page_size)
    z_page = NormalDist().inv_cdf(per_sample_prob)

    _select_gaussian_threshold_pack_kernel[(batch, num_heads)](
        mu_c,
        sigma_c,
        tau_c,
        sigma_global_c,
        out,
        counts,
        tau_floor,
        mu_c.stride(0),
        mu_c.stride(1),
        mu_c.stride(2),
        tau_c.stride(0),
        tau_c.stride(1),
        sigma_global_c.stride(0),
        sigma_global_c.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        counts.stride(0),
        counts.stride(1),
        tau_floor.stride(0),
        tau_floor.stride(1),
        pages_for_stats,
        num_pages,
        float(alpha - 1.0),
        float(safety_margin_z),
        float(threshold_excess_margin_fraction),
        float(z_page),
        BLOCK_PAGES=block_pages,
        num_warps=8 if block_pages >= 1024 else 4,
    )
    return out, counts, tau_floor
