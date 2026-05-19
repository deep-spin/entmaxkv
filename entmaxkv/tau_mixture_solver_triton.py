import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _normal_cdf_tl(x):
    return 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _normal_pdf_tl(x):
    return tl.exp(-0.5 * x * x) * 0.3989422804014327


@triton.jit
def _mixture_constraint_derivatives_tl(
    tau,
    mu_pages,
    sigma_pages,
    page_counts,
    page_mask,
    alpha: tl.constexpr,
):
    a = alpha - 1.0
    sigma_y = tl.maximum(a * sigma_pages, 1.0e-10)
    mu_y = a * mu_pages - tau
    t = mu_y / sigma_y
    phi = _normal_pdf_tl(t)
    Phi = _normal_cdf_tl(t)

    m0 = Phi
    m1 = mu_y * Phi + sigma_y * phi
    m2 = (mu_y * mu_y + sigma_y * sigma_y) * Phi + mu_y * sigma_y * phi

    if alpha == 2.0:
        expectation = m1
        first_derivative = -m0
        second_derivative = phi / sigma_y
    elif alpha == 1.5:
        expectation = m2
        first_derivative = -2.0 * m1
        second_derivative = 2.0 * m0
    else:
        m3 = (
            (mu_y * mu_y * mu_y + 3.0 * mu_y * sigma_y * sigma_y) * Phi
            + (sigma_y * sigma_y * sigma_y + 2.0 * mu_y * mu_y * sigma_y) * phi
        )
        expectation = m3
        first_derivative = -3.0 * m2
        second_derivative = 6.0 * m1

    weighted_expectation = tl.where(page_mask, page_counts * expectation, 0.0)
    weighted_first = tl.where(page_mask, page_counts * first_derivative, 0.0)
    weighted_second = tl.where(page_mask, page_counts * second_derivative, 0.0)

    f = tl.sum(weighted_expectation, axis=0) - 1.0
    f_prime = tl.sum(weighted_first, axis=0)
    f_second = tl.sum(weighted_second, axis=0)
    return f, f_prime, f_second


@triton.jit
def _mixture_constraint_tl(
    tau,
    mu_pages,
    sigma_pages,
    page_counts,
    page_mask,
    alpha: tl.constexpr,
):
    f, _, _ = _mixture_constraint_derivatives_tl(
        tau, mu_pages, sigma_pages, page_counts, page_mask, alpha
    )
    return f


@triton.jit
def _tau_hat_page_mixture_kernel(
    MU_PAGES,
    SIGMA_PAGES,
    PAGE_COUNTS,
    OUT,
    num_rows: tl.constexpr,
    pages: tl.constexpr,
    alpha: tl.constexpr,
    iterations: tl.constexpr,
    tol: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_PAGES)
    page_mask = offsets < pages
    base = row * pages + offsets

    mu = tl.load(MU_PAGES + base, mask=page_mask, other=0.0).to(tl.float32)
    sigma = tl.maximum(
        tl.load(SIGMA_PAGES + base, mask=page_mask, other=1.0).to(tl.float32), 1.0e-6
    )

    if HAS_COUNTS:
        counts_raw = tl.load(PAGE_COUNTS + base, mask=page_mask, other=0.0).to(tl.float32)
        counts = tl.where(page_mask, counts_raw, 0.0)
    else:
        counts = tl.where(page_mask, 1.0, 0.0)

    active = counts > 0.0
    has_active = tl.sum(tl.where(active, 1.0, 0.0), axis=0) > 0.0

    a = alpha - 1.0
    total_tokens = tl.maximum(tl.sum(counts, axis=0), 1.0)
    log_n_factor = tl.sqrt(2.0 * tl.log(total_tokens + 1.0))

    y_mean = a * mu
    y_sigma = a * sigma
    max_y = tl.max(tl.where(active, y_mean, -float("inf")), axis=0)
    max_y = tl.where(has_active, max_y, 0.0)
    max_sigma = tl.maximum(tl.max(tl.where(active, y_sigma, 0.0), axis=0), 1.0e-6)

    radius = max_sigma * (log_n_factor + 8.0)
    tau_low = max_y - radius
    tau_high = max_y + radius

    for _ in range(8):
        f_low = _mixture_constraint_tl(tau_low, mu, sigma, counts, page_mask, alpha)
        f_high = _mixture_constraint_tl(tau_high, mu, sigma, counts, page_mask, alpha)
        tau_low = tl.where(f_low <= 0.0, tau_low - radius, tau_low)
        tau_high = tl.where(f_high > 0.0, tau_high + radius, tau_high)
        radius = radius * 2.0

    tau = 0.5 * (tau_low + tau_high)
    for _ in range(iterations):
        f_tau, f_prime, f_second = _mixture_constraint_derivatives_tl(
            tau, mu, sigma, counts, page_mask, alpha
        )

        midpoint = 0.5 * (tau_low + tau_high)
        denom = 2.0 * f_prime * f_prime - f_tau * f_second
        denom_safe = tl.where(tl.abs(denom) > 1.0e-18, denom, 1.0)
        f_prime_safe = tl.minimum(f_prime, -1.0e-12)
        newton_step = f_tau / f_prime_safe
        halley_step = 2.0 * f_tau * f_prime / denom_safe
        step = tl.where(tl.abs(denom) > 1.0e-18, halley_step, newton_step)
        tau_candidate = tau - step

        valid = (
            (tau_candidate == tau_candidate)
            & (tl.abs(tau_candidate) < 1.0e20)
            & (tau_candidate > tau_low)
            & (tau_candidate < tau_high)
        )
        tau_candidate = tl.where(valid, tau_candidate, midpoint)

        f_candidate = _mixture_constraint_tl(tau_candidate, mu, sigma, counts, page_mask, alpha)
        tau_low = tl.where(f_candidate > 0.0, tau_candidate, tau_low)
        tau_high = tl.where(f_candidate <= 0.0, tau_candidate, tau_high)
        tau = tl.where(tl.abs(f_candidate) <= tol, tau_candidate, 0.5 * (tau_low + tau_high))

    tau = tl.where(has_active, tau, 0.0)
    tl.store(OUT + row, tau, mask=row < num_rows)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def solve_for_tau_hat_page_gaussian_mixture_triton(
    mu_pages: torch.Tensor,
    sigma_pages: torch.Tensor,
    page_counts: Optional[torch.Tensor] = None,
    alpha: float = 1.5,
    max_iter: int = 40,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Triton-parallel page-mixture tau solver.

    Each Triton program solves one independent prefix row and reduces across
    the last/page dimension in vector lanes. Inputs follow
    solve_for_tau_hat_page_gaussian_mixture and the output shape is
    ``mu_pages.shape[:-1] + (1,)``.
    """
    if mu_pages.shape != sigma_pages.shape:
        raise ValueError(
            f"mu_pages and sigma_pages must have the same shape, got "
            f"{tuple(mu_pages.shape)} and {tuple(sigma_pages.shape)}"
        )
    if mu_pages.dim() < 1:
        raise ValueError("mu_pages must have a page dimension")
    if not mu_pages.is_cuda or not sigma_pages.is_cuda:
        raise ValueError("Triton page-mixture tau solve requires CUDA tensors")
    if alpha not in (2, 1.5, 4 / 3):
        raise ValueError(f"Unsupported alpha for page-mixture tau solve: {alpha}")

    pages = mu_pages.shape[-1]
    if pages <= 0:
        raise ValueError("mu_pages must have a non-empty page dimension")

    original_dtype = mu_pages.dtype
    mu_c = mu_pages.contiguous()
    sigma_c = sigma_pages.contiguous()
    counts_c = None
    has_counts = page_counts is not None
    if has_counts:
        counts_c = torch.broadcast_to(
            page_counts.to(device=mu_pages.device, dtype=torch.float32), mu_pages.shape
        ).contiguous()

    num_rows = math.prod(mu_pages.shape[:-1]) if mu_pages.dim() > 1 else 1
    mu_flat = mu_c.view(num_rows, pages)
    sigma_flat = sigma_c.view(num_rows, pages)
    out = torch.empty((num_rows,), device=mu_pages.device, dtype=torch.float32)

    block_pages = _next_power_of_2(pages)
    num_warps = 1 if block_pages <= 64 else 4 if block_pages <= 2048 else 8
    _tau_hat_page_mixture_kernel[(num_rows,)](
        mu_flat,
        sigma_flat,
        counts_c.view(num_rows, pages) if has_counts else mu_flat,
        out,
        num_rows,
        pages,
        float(alpha),
        max_iter,
        float(tol),
        HAS_COUNTS=has_counts,
        BLOCK_PAGES=block_pages,
        num_warps=num_warps,
    )

    return out.view(*mu_pages.shape[:-1], 1).to(dtype=original_dtype)
