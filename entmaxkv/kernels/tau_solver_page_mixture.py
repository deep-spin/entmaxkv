import math
from typing import Optional, Tuple

import torch


def _gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _gaussian_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)


def _truncated_moments(
    mu_y: torch.Tensor,
    sigma_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_y = sigma_y.clamp(min=1e-10)
    t = mu_y / sigma_y
    phi = _gaussian_pdf(t)
    Phi = _gaussian_cdf(t)

    m0 = Phi
    m1 = mu_y * Phi + sigma_y * phi
    m2 = (mu_y.square() + sigma_y.square()) * Phi + mu_y * sigma_y * phi
    m3 = (
        (mu_y.pow(3) + 3.0 * mu_y * sigma_y.square()) * Phi
        + (sigma_y.pow(3) + 2.0 * mu_y.square() * sigma_y) * phi
    )
    return m0, m1, m2, m3


def _evaluate_mixture_constraint_with_derivatives(
    tau: torch.Tensor,
    mu_pages: torch.Tensor,
    sigma_pages: torch.Tensor,
    page_counts: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = alpha - 1.0
    mu_y = a * mu_pages - tau
    sigma_y = a * sigma_pages.clamp(min=1e-6)
    m0, m1, m2, m3 = _truncated_moments(mu_y, sigma_y)

    if alpha == 2:
        expectation = m1
        first_derivative = -m0
        second_derivative = _gaussian_pdf(mu_y / sigma_y.clamp(min=1e-10)) / sigma_y.clamp(min=1e-10)
    elif alpha == 1.5:
        expectation = m2
        first_derivative = -2.0 * m1
        second_derivative = 2.0 * m0
    elif alpha == 4 / 3:
        expectation = m3
        first_derivative = -3.0 * m2
        second_derivative = 6.0 * m1
    else:
        raise ValueError(f"Unsupported alpha for page-mixture tau solve: {alpha}")

    f = (page_counts * expectation).sum(dim=-1, keepdim=True) - 1.0
    f_prime = (page_counts * first_derivative).sum(dim=-1, keepdim=True)
    f_second = (page_counts * second_derivative).sum(dim=-1, keepdim=True)
    return f, f_prime, f_second


def solve_for_tau_hat_page_gaussian_mixture(
    mu_pages: torch.Tensor,
    sigma_pages: torch.Tensor,
    page_counts: Optional[torch.Tensor] = None,
    alpha: float = 1.5,
    max_iter: int = 40,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Solve entmax tau for a mixture of per-page Gaussian score models.

    This is the ALiBi-aware distributional constraint:

        sum_p count_p * E[((alpha - 1) * S_p - tau)_+ ** beta] = 1

    where beta = 1 / (alpha - 1) and

        S_p ~ Normal(mu_pages[..., p], sigma_pages[..., p]^2).

    ALiBi should be included by shifting mu_pages before calling this function:

        mu_pages_shifted = mu_qk_pages + alibi_bias_pages

    The per-page sigma should remain the raw QK uncertainty; deterministic
    ALiBi position bias should not inflate it.

    Args:
        mu_pages: Per-page score means, shape [B, H, P].
        sigma_pages: Per-page raw QK score stddevs, shape [B, H, P].
        page_counts: Token count represented by each page, broadcastable to
            [B, H, P]. Defaults to one token per mixture component.
        alpha: Supported values are 2, 1.5, and 4/3.
        max_iter: Number of bracketed Halley iterations.
        tol: Root tolerance for the normalization constraint.

    Returns:
        tau with shape [B, H, 1].
    """
    if mu_pages.shape != sigma_pages.shape:
        raise ValueError(
            f"mu_pages and sigma_pages must have the same shape, got "
            f"{tuple(mu_pages.shape)} and {tuple(sigma_pages.shape)}"
        )
    if mu_pages.dim() < 1:
        raise ValueError("mu_pages must have a page dimension")
    if alpha not in (2, 1.5, 4 / 3):
        raise ValueError(f"Unsupported alpha for page-mixture tau solve: {alpha}")

    original_dtype = mu_pages.dtype
    work_dtype = torch.float32 if mu_pages.is_cuda else torch.float64
    mu_pages = mu_pages.to(dtype=work_dtype)
    sigma_pages = sigma_pages.to(dtype=work_dtype).clamp(min=1e-6)

    if page_counts is None:
        page_counts = torch.ones_like(mu_pages)
    else:
        page_counts = page_counts.to(device=mu_pages.device, dtype=work_dtype)
        page_counts = torch.broadcast_to(page_counts, mu_pages.shape)

    active = page_counts > 0
    has_active = active.any(dim=-1, keepdim=True)
    if not bool(has_active.any().item()):
        return torch.zeros_like(mu_pages[..., :1], dtype=original_dtype)

    a = alpha - 1.0
    total_tokens = page_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    log_n_factor = torch.sqrt(2.0 * torch.log(total_tokens + 1.0))

    y_mean = a * mu_pages
    y_sigma = a * sigma_pages
    max_y = torch.where(active, y_mean, torch.full_like(y_mean, -float("inf"))).amax(dim=-1, keepdim=True)
    max_y = torch.where(has_active, max_y, torch.zeros_like(max_y))
    max_sigma = torch.where(active, y_sigma, torch.zeros_like(y_sigma)).amax(dim=-1, keepdim=True).clamp(min=1e-6)

    radius = max_sigma * (log_n_factor + 8.0)
    tau_low = max_y - radius
    tau_high = max_y + radius

    # Make the bracket robust for strongly ALiBi-shifted mixtures.
    for _ in range(8):
        f_low, _, _ = _evaluate_mixture_constraint_with_derivatives(
            tau_low, mu_pages, sigma_pages, page_counts, alpha
        )
        f_high, _, _ = _evaluate_mixture_constraint_with_derivatives(
            tau_high, mu_pages, sigma_pages, page_counts, alpha
        )
        tau_low = torch.where(f_low <= 0.0, tau_low - radius, tau_low)
        tau_high = torch.where(f_high > 0.0, tau_high + radius, tau_high)
        radius = radius * 2.0

    tau = 0.5 * (tau_low + tau_high)
    for _ in range(max_iter):
        f_tau, f_prime, f_second = _evaluate_mixture_constraint_with_derivatives(
            tau, mu_pages, sigma_pages, page_counts, alpha
        )

        midpoint = 0.5 * (tau_low + tau_high)
        denom = 2.0 * f_prime.square() - f_tau * f_second
        newton_step = f_tau / f_prime.clamp(max=-1e-12)
        halley_step = 2.0 * f_tau * f_prime / denom.clamp(min=-1e30, max=1e30)
        tau_candidate = tau - torch.where(denom.abs() > 1e-18, halley_step, newton_step)

        invalid = (
            torch.isnan(tau_candidate)
            | torch.isinf(tau_candidate)
            | (tau_candidate <= tau_low)
            | (tau_candidate >= tau_high)
        )
        tau_candidate = torch.where(invalid, midpoint, tau_candidate)

        f_candidate, _, _ = _evaluate_mixture_constraint_with_derivatives(
            tau_candidate, mu_pages, sigma_pages, page_counts, alpha
        )
        tau_low = torch.where(f_candidate > 0.0, tau_candidate, tau_low)
        tau_high = torch.where(f_candidate <= 0.0, tau_candidate, tau_high)
        tau = torch.where(f_candidate.abs() <= tol, tau_candidate, 0.5 * (tau_low + tau_high))

    tau = torch.where(has_active, tau, torch.zeros_like(tau))
    return tau.to(dtype=original_dtype)


def alibi_page_mean_bias(
    alibi_slopes: torch.Tensor,
    pages_for_stats: int,
    seq_len: int,
    page_size: int,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Return deterministic ALiBi bias at each page mean position.

    Output shape is [1, H, P], matching per-page Gaussian stats. Add this to
    raw per-page means before calling solve_for_tau_hat_page_gaussian_mixture.
    """
    slopes = alibi_slopes
    if dtype is not None:
        slopes = slopes.to(dtype=dtype)
    if slopes.dim() == 1:
        slopes = slopes.unsqueeze(0)
    slopes = slopes.unsqueeze(-1)
    q_pos = seq_len - 1
    page_mean_pos = (
        torch.arange(pages_for_stats, device=slopes.device, dtype=slopes.dtype) * page_size
        + (page_size - 1) / 2.0
    ).clamp(max=seq_len - 1)
    return slopes * (page_mean_pos - q_pos)
