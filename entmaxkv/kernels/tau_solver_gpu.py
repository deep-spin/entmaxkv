import math

import torch
import triton
import triton.language as tl


@triton.jit
def _normal_cdf_tl(x):
    return 0.5 * (1.0 + tl.erf(x * 0.7071067811865476)) #INV_SQRT_2


@triton.jit
def _normal_pdf_tl(x):
    return tl.exp(-0.5 * x * x) * 0.3989422804014327 #INV_SQRT_2PI


@triton.jit
def _constraint_tl(tau, mu_g, sigma_g, n: tl.constexpr, alpha: tl.constexpr):
    a = alpha - 1.0
    beta = 1.0 / a
    sigma_y = tl.maximum(a * sigma_g, 1.0e-10)
    mu_y = a * mu_g - tau
    t = mu_y / sigma_y

    if alpha == 2.0:
        expectation = mu_y * _normal_cdf_tl(t) + sigma_y * _normal_pdf_tl(t)
    elif alpha == 1.5:
        phi_t = _normal_pdf_tl(t)
        Phi_t = _normal_cdf_tl(t)
        expectation = (mu_y * mu_y + sigma_y * sigma_y) * Phi_t + mu_y * sigma_y * phi_t
    elif alpha == 1.3333333333333333:
        phi_t = _normal_pdf_tl(t)
        Phi_t = _normal_cdf_tl(t)
        expectation = (
            (mu_y * mu_y * mu_y + 3.0 * mu_y * sigma_y * sigma_y) * Phi_t
            + (sigma_y * sigma_y * sigma_y + 2.0 * mu_y * mu_y * sigma_y) * phi_t
        )
    else:
        t_pos = tl.maximum(t, 0.0)
        expectation = tl.exp(tl.log(sigma_y) * beta) * tl.exp(tl.log(t_pos) * beta)

    return n * expectation - 1.0


@triton.jit
def _constraint_derivatives_tl(tau, mu_g, sigma_g, n: tl.constexpr, alpha: tl.constexpr):
    a = alpha - 1.0
    sigma_y = tl.maximum(a * sigma_g, 1.0e-10)
    mu_y = a * mu_g - tau
    t = mu_y / sigma_y
    phi_t = _normal_pdf_tl(t)
    Phi_t = _normal_cdf_tl(t)

    m0 = Phi_t
    m1 = mu_y * Phi_t + sigma_y * phi_t
    m2 = (mu_y * mu_y + sigma_y * sigma_y) * Phi_t + mu_y * sigma_y * phi_t

    if alpha == 2.0:
        expectation = m1
        first_derivative = -n * m0
        second_derivative = n * phi_t / sigma_y
    elif alpha == 1.5:
        expectation = m2
        first_derivative = -2.0 * n * m1
        second_derivative = 2.0 * n * m0
    else:
        m3 = (
            (mu_y * mu_y * mu_y + 3.0 * mu_y * sigma_y * sigma_y) * Phi_t
            + (sigma_y * sigma_y * sigma_y + 2.0 * mu_y * mu_y * sigma_y) * phi_t
        )
        expectation = m3
        first_derivative = -3.0 * n * m2
        second_derivative = 6.0 * n * m1

    return n * expectation - 1.0, first_derivative, second_derivative


@triton.jit
def _tau_hat_single_gaussian_kernel(
    MU_G,
    SIGMA_G,
    OUT,
    total: tl.constexpr,
    n: tl.constexpr,
    alpha: tl.constexpr,
    log_n_factor: tl.constexpr,
    iterations: tl.constexpr,
    tol: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    mu_g = tl.load(MU_G + offsets, mask=mask, other=0.0).to(tl.float32)
    sigma_g = tl.maximum(tl.load(SIGMA_G + offsets, mask=mask, other=1.0).to(tl.float32), 1.0e-6)

    a = alpha - 1.0
    radius = a * sigma_g * (log_n_factor + 8.0)
    tau_low = a * mu_g - radius
    tau_high = a * mu_g + radius
    tau = 0.5 * (tau_low + tau_high)

    for _ in range(iterations):
        f_tau, f_prime, f_second = _constraint_derivatives_tl(tau, mu_g, sigma_g, n, alpha)
        midpoint = 0.5 * (tau_low + tau_high)
        denom = 2.0 * f_prime * f_prime - f_tau * f_second
        newton_step = f_tau / tl.minimum(f_prime, -1.0e-12)
        halley_step = 2.0 * f_tau * f_prime / denom
        step = tl.where(tl.abs(denom) > 1.0e-18, halley_step, newton_step)
        tau_candidate = tau - step

        valid = (
            (tau_candidate == tau_candidate)
            & (tl.abs(tau_candidate) < 1.0e20)
            & (tau_candidate > tau_low)
            & (tau_candidate < tau_high)
        )
        tau_candidate = tl.where(valid, tau_candidate, midpoint)

        f_candidate = _constraint_tl(tau_candidate, mu_g, sigma_g, n, alpha)
        tau_low = tl.where(f_candidate > 0.0, tau_candidate, tau_low)
        tau_high = tl.where(f_candidate <= 0.0, tau_candidate, tau_high)
        tau = tl.where(tl.abs(f_candidate) <= tol, tau_candidate, 0.5 * (tau_low + tau_high))

    tl.store(OUT + offsets, tau, mask=mask)


def solve_for_tau_hat_single_gaussian_gpu(
    mu_g: torch.Tensor,
    sigma_g: torch.Tensor,
    n: int,
    alpha: float,
    iterations: int,
    tol: float,
) -> torch.Tensor:
    """
    One-launch CUDA solver: each lane solves one independent [B,H] tau problem.
    """
    original_shape = mu_g.shape
    mu_flat = mu_g.contiguous().view(-1)
    sigma_flat = sigma_g.contiguous().view(-1)
    out = torch.empty_like(mu_flat)
    total = mu_flat.numel()
    block = min(1024, triton.next_power_of_2(total))
    grid = (triton.cdiv(total, block),)
    _tau_hat_single_gaussian_kernel[grid](
        mu_flat,
        sigma_flat,
        out,
        total,
        n,
        float(alpha),
        math.sqrt(2.0 * math.log(n + 1.0)),
        iterations,
        float(tol),
        BLOCK=block,
    )
    return out.view(original_shape)
