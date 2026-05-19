from statistics import NormalDist

import torch
import triton
import triton.language as tl


@triton.jit
def _tau_clamp_selected_pages_kernel(
    TAU,
    MU_PAGE,
    SIGMA_PAGE,
    PAGE_INDICES,
    NUM_SELECTED,
    TAU_FLOOR,
    OUT,
    stride_tb,
    stride_th,
    stride_mb,
    stride_mh,
    stride_mp,
    stride_ib,
    stride_ih,
    stride_is,
    stride_nb,
    stride_nh,
    stride_fb,
    stride_fh,
    max_slots: tl.constexpr,
    pages_for_stats: tl.constexpr,
    alpha_minus_one: tl.constexpr,
    z_clamp: tl.constexpr,
    eps: tl.constexpr,
    HAS_TAU_FLOOR: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_SLOTS)
    slot_mask = offsets < max_slots

    n_selected = tl.load(NUM_SELECTED + batch_id * stride_nb + head_id * stride_nh).to(tl.int32)
    selected = tl.load(
        PAGE_INDICES + batch_id * stride_ib + head_id * stride_ih + offsets * stride_is,
        mask=slot_mask,
        other=0,
    ).to(tl.int32)
    valid = slot_mask & (offsets < n_selected) & (selected >= 0) & (selected < pages_for_stats)
    safe_selected = tl.minimum(tl.maximum(selected, 0), pages_for_stats - 1)

    mu = tl.load(
        MU_PAGE + batch_id * stride_mb + head_id * stride_mh + safe_selected * stride_mp,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    sigma = tl.load(
        SIGMA_PAGE + batch_id * stride_mb + head_id * stride_mh + safe_selected * stride_mp,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    page_cap = mu + z_clamp * sigma * 0.3333333333333333
    max_selected_page_cap = tl.max(page_cap, axis=0)

    tau_upper = alpha_minus_one * max_selected_page_cap - eps
    if HAS_TAU_FLOOR:
        tau_floor = tl.load(TAU_FLOOR + batch_id * stride_fb + head_id * stride_fh).to(tl.float32)
        tau_upper = tl.maximum(tau_upper, tau_floor)

    tau = tl.load(TAU + batch_id * stride_tb + head_id * stride_th).to(tl.float32)
    has_valid = max_selected_page_cap > -3.0e38
    clamped = tl.where(has_valid, tl.minimum(tau, tau_upper), tau)
    tl.store(OUT + batch_id * stride_tb + head_id * stride_th, clamped)


def clamp_tau_to_selected_page_statistics_triton(
    tau: torch.Tensor,
    mu_scores_per_page: torch.Tensor,
    sigma_scores_per_page: torch.Tensor,
    page_indices: torch.Tensor,
    num_selected_per_head: torch.Tensor,
    tau_floor: torch.Tensor | None,
    page_size: int,
    alpha: float,
    clamp_quantile: float,
    pages_for_stats: int,
    eps: float,
) -> torch.Tensor:
    if not tau.is_cuda:
        raise ValueError("Triton tau clamp requires CUDA tensors")
    if pages_for_stats <= 0:
        return tau.contiguous()

    tau_2d = tau.squeeze(-1) if tau.dim() == 3 and tau.shape[-1] == 1 else tau
    original_shape = tau.shape
    tau_c = tau_2d.contiguous()
    mu_c = mu_scores_per_page.contiguous()
    sigma_c = sigma_scores_per_page.contiguous()
    page_indices_c = page_indices.contiguous()
    num_selected_c = num_selected_per_head.contiguous()
    tau_floor_c = None
    if tau_floor is not None:
        tau_floor_c = tau_floor.squeeze(-1) if tau_floor.dim() == 3 and tau_floor.shape[-1] == 1 else tau_floor
        tau_floor_c = tau_floor_c.to(device=tau_c.device, dtype=torch.float32).contiguous()

    batch, num_heads = tau_c.shape
    max_slots = page_indices_c.shape[-1]
    if max_slots <= 0:
        out = tau_c.clone()
        return out.unsqueeze(-1) if original_shape != out.shape else out

    block_slots = triton.next_power_of_2(max_slots)
    per_sample_prob = clamp_quantile ** (1.0 / page_size)
    z_clamp = NormalDist().inv_cdf(per_sample_prob)

    out = torch.empty_like(tau_c, dtype=torch.float32)
    _tau_clamp_selected_pages_kernel[(batch, num_heads)](
        tau_c,
        mu_c,
        sigma_c,
        page_indices_c,
        num_selected_c,
        tau_floor_c if tau_floor_c is not None else tau_c,
        out,
        tau_c.stride(0),
        tau_c.stride(1),
        mu_c.stride(0),
        mu_c.stride(1),
        mu_c.stride(2),
        page_indices_c.stride(0),
        page_indices_c.stride(1),
        page_indices_c.stride(2),
        num_selected_c.stride(0),
        num_selected_c.stride(1),
        tau_floor_c.stride(0) if tau_floor_c is not None else tau_c.stride(0),
        tau_floor_c.stride(1) if tau_floor_c is not None else tau_c.stride(1),
        max_slots,
        pages_for_stats,
        float(alpha - 1.0),
        float(z_clamp),
        float(eps),
        HAS_TAU_FLOOR=tau_floor_c is not None,
        BLOCK_SLOTS=block_slots,
        num_warps=8 if block_slots >= 1024 else 4,
    )

    if original_shape != out.shape:
        out = out.unsqueeze(-1)
    return out.contiguous()
