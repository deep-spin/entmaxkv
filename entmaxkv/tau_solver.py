import torch
import math
from typing import Tuple

try:
    from entmaxkv.kernels.tau_solver_gpu import solve_for_tau_hat_single_gaussian_gpu
    _HAS_GPU_SOLVER = True
except Exception:
    solve_for_tau_hat_single_gaussian_gpu = None
    _HAS_GPU_SOLVER = False


def _gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _gaussian_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF."""
    return torch.exp(-0.5 * x.pow(2)) / math.sqrt(2.0 * math.pi)


def _gaussian_cdf_scalar(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _gaussian_pdf_scalar(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _evaluate_constraint(
    tau: torch.Tensor,
    mu_g: torch.Tensor,
    sigma_g: torch.Tensor,
    n: int,
    alpha: float
) -> torch.Tensor:
    """
    Evaluate f(τ) = n E[g_α(S; τ)] - 1 using closed-form expressions.

    Args:
        tau: Threshold value [batch, heads, 1]
        mu_g: Global mean score [batch, heads, 1]
        sigma_g: Global std score [batch, heads, 1]
        n: Number of tokens
        alpha: Entmax alpha parameter

    Returns:
        Constraint function value f(τ) = n E[g_α(S; τ)] - 1
    """
    a = alpha - 1.0
    beta = 1.0 / a

    # Compute Y = aS - τ where S ~ N(μ_g, σ_g²)
    # Then Y ~ N(μ_Y, σ_Y²) with:
    mu_Y = a * mu_g - tau
    sigma_Y = a * sigma_g

    # Compute t = μ_Y / σ_Y
    t = mu_Y / sigma_Y.clamp(min=1e-10)

    # Compute closed-form expressions based on α
    if alpha == 2:
        # Case α = 2 (β = 1): Equation 108
        # E[g_α(S; τ)] = uY * Φ(t) +  σ φ(t)
        expectation = (mu_g  - tau) * _gaussian_cdf((mu_g  - tau)/ sigma_Y) + sigma_Y * _gaussian_pdf((mu_g  - tau)/ sigma_Y)
    elif alpha == 1.5:
        # Case α = 1.5 (β = 2): Equation 110
        # E[g_α(S; τ)] = (μ_Y² + σ_Y²) Φ(t) + μ_Y σ_Y φ(t)
        phi_t = _gaussian_pdf(t)
        Phi_t = _gaussian_cdf(t)
        expectation = (mu_Y.pow(2) + sigma_Y.pow(2)) * Phi_t + mu_Y * sigma_Y * phi_t
    elif alpha == 4/3:
        # Case α = 4/3 (β = 3): Equation 112
        # E[g_α(S; τ)] = (μ_Y³ + 3μ_Y σ_Y²) Φ(t) + (σ_Y³ + 2μ_Y² σ_Y) φ(t)
        phi_t = _gaussian_pdf(t)
        Phi_t = _gaussian_cdf(t)
        expectation = (mu_Y.pow(3) + 3.0 * mu_Y * sigma_Y.pow(2)) * Phi_t + (sigma_Y.pow(3) + 2.0 * mu_Y.pow(2) * sigma_Y) * phi_t

    else:
        # Fallback: use simple approximation
        # E[[Y]_+^β] ≈ max(t, 0)^β for large |t|
        truncated_moment = torch.relu(t).pow(beta)
        expectation = sigma_Y.pow(beta) * truncated_moment

    return n * expectation - 1.0


def _truncated_moments(
    mu_Y: torch.Tensor,
    sigma_Y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return E[[Y]_+^k] for k in {0,1,2,3}, where Y ~ N(mu_Y, sigma_Y^2).
    """
    sigma_Y = sigma_Y.clamp(min=1e-10)
    t = mu_Y / sigma_Y
    phi_t = _gaussian_pdf(t)
    Phi_t = _gaussian_cdf(t)

    m0 = Phi_t
    m1 = mu_Y * Phi_t + sigma_Y * phi_t
    m2 = (mu_Y.pow(2) + sigma_Y.pow(2)) * Phi_t + mu_Y * sigma_Y * phi_t
    m3 = (
        (mu_Y.pow(3) + 3.0 * mu_Y * sigma_Y.pow(2)) * Phi_t
        + (sigma_Y.pow(3) + 2.0 * mu_Y.pow(2) * sigma_Y) * phi_t
    )
    return m0, m1, m2, m3


def _evaluate_constraint_with_derivatives(
    tau: torch.Tensor,
    mu_g: torch.Tensor,
    sigma_g: torch.Tensor,
    n: int,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate f(τ), f'(τ), and f''(τ) for supported closed-form entmax alphas.
    """
    a = alpha - 1.0
    mu_Y = a * mu_g - tau
    sigma_Y = a * sigma_g
    m0, m1, m2, m3 = _truncated_moments(mu_Y, sigma_Y)

    if alpha == 2:
        expectation = m1
        first_derivative = -n * m0
        second_derivative = n * _gaussian_pdf((mu_Y / sigma_Y.clamp(min=1e-10))) / sigma_Y.clamp(min=1e-10)
    elif alpha == 1.5:
        expectation = m2
        first_derivative = -2.0 * n * m1
        second_derivative = 2.0 * n * m0
    elif alpha == 4 / 3:
        expectation = m3
        first_derivative = -3.0 * n * m2
        second_derivative = 6.0 * n * m1
    else:
        raise ValueError(f"Unsupported alpha for derivative-based tau solve: {alpha}")

    return n * expectation - 1.0, first_derivative, second_derivative


def _evaluate_constraint_scalar(
    tau: float,
    mu_g: float,
    sigma_g: float,
    n: int,
    alpha: float,
) -> float:
    a = alpha - 1.0
    beta = 1.0 / a
    sigma_Y = max(a * sigma_g, 1e-10)
    mu_Y = a * mu_g - tau
    t = mu_Y / sigma_Y

    if alpha == 2:
        z = (mu_g - tau) / sigma_Y
        expectation = (mu_g - tau) * _gaussian_cdf_scalar(z) + sigma_Y * _gaussian_pdf_scalar(z)
    elif alpha == 1.5:
        phi_t = _gaussian_pdf_scalar(t)
        Phi_t = _gaussian_cdf_scalar(t)
        expectation = (mu_Y * mu_Y + sigma_Y * sigma_Y) * Phi_t + mu_Y * sigma_Y * phi_t
    elif alpha == 4 / 3:
        phi_t = _gaussian_pdf_scalar(t)
        Phi_t = _gaussian_cdf_scalar(t)
        expectation = (
            (mu_Y ** 3 + 3.0 * mu_Y * sigma_Y * sigma_Y) * Phi_t
            + (sigma_Y ** 3 + 2.0 * mu_Y * mu_Y * sigma_Y) * phi_t
        )
    else:
        truncated_moment = max(t, 0.0) ** beta
        expectation = (sigma_Y ** beta) * truncated_moment

    return n * expectation - 1.0


def _evaluate_constraint_with_derivatives_scalar(
    tau: float,
    mu_g: float,
    sigma_g: float,
    n: int,
    alpha: float,
) -> Tuple[float, float, float]:
    a = alpha - 1.0
    sigma_Y = max(a * sigma_g, 1e-10)
    mu_Y = a * mu_g - tau
    t = mu_Y / sigma_Y
    phi_t = _gaussian_pdf_scalar(t)
    Phi_t = _gaussian_cdf_scalar(t)

    m0 = Phi_t
    m1 = mu_Y * Phi_t + sigma_Y * phi_t
    m2 = (mu_Y * mu_Y + sigma_Y * sigma_Y) * Phi_t + mu_Y * sigma_Y * phi_t

    if alpha == 2:
        expectation = m1
        first_derivative = -n * m0
        second_derivative = n * phi_t / sigma_Y
    elif alpha == 1.5:
        expectation = m2
        first_derivative = -2.0 * n * m1
        second_derivative = 2.0 * n * m0
    else:
        raise ValueError(f"Unsupported alpha for scalar derivative-based tau solve: {alpha}")

    return n * expectation - 1.0, first_derivative, second_derivative


def _solve_for_tau_hat_scalar_cpu(
    mu_g: torch.Tensor,
    sigma_g: torch.Tensor,
    n: int,
    alpha: float,
    iterations: int,
    tol: float,
) -> torch.Tensor:
    """
    Solve tiny [B,H,1] problems with Python scalar math instead of tiny Torch kernels.
    """
    original_device = mu_g.device
    original_dtype = mu_g.dtype
    mu_cpu = mu_g.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
    sigma_cpu = sigma_g.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()

    a = alpha - 1.0
    log_n_factor = math.sqrt(2.0 * math.log(n + 1.0))
    out = []

    for mu_i, sigma_i in zip(mu_cpu, sigma_cpu):
        sigma_i = max(sigma_i, 1e-6)
        radius = a * sigma_i * (log_n_factor + 8.0)
        tau_low = a * mu_i - radius
        tau_high = a * mu_i + radius
        tau = 0.5 * (tau_low + tau_high)

        for _ in range(iterations):
            f_tau, f_prime, f_second = _evaluate_constraint_with_derivatives_scalar(
                tau, mu_i, sigma_i, n, alpha
            )
            if abs(f_tau) <= tol:
                break

            midpoint = 0.5 * (tau_low + tau_high)
            denom = 2.0 * f_prime * f_prime - f_tau * f_second
            use_halley = abs(denom) > 1e-18
            step = (2.0 * f_tau * f_prime / denom) if use_halley else (
                f_tau / f_prime if abs(f_prime) > 1e-18 else 0.0
            )
            tau_candidate = tau - step

            if (
                not math.isfinite(tau_candidate)
                or tau_candidate <= tau_low
                or tau_candidate >= tau_high
            ):
                tau_candidate = midpoint

            f_candidate = _evaluate_constraint_scalar(tau_candidate, mu_i, sigma_i, n, alpha)
            if abs(f_candidate) <= tol:
                tau = tau_candidate
                break

            if f_candidate > 0.0:
                tau_low = tau_candidate
            else:
                tau_high = tau_candidate
            tau = 0.5 * (tau_low + tau_high)

        out.append(tau)

    tau_tensor = torch.tensor(out, dtype=torch.float64).reshape(mu_g.shape)
    return tau_tensor.to(device=original_device, dtype=original_dtype)


def _solve_for_tau_hat_cpu_bisection(
    mu_g: torch.Tensor,
    sigma_g: torch.Tensor,
    n: int,
    alpha: float,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> torch.Tensor:
    """
    Legacy CPU bisection path kept as a robust fallback and verbose/debug path.
    """
    original_dtype = mu_g.dtype
    original_device = mu_g.device
    mu_g = mu_g.cpu().double()
    sigma_g = sigma_g.cpu().double()

    a = alpha - 1.0
    log_n_factor = math.sqrt(2.0 * math.log(n + 1))
    tau_low = a * mu_g - 5.0 * a * sigma_g
    tau_high = a * mu_g + a * sigma_g * (log_n_factor + 5.0)

    if verbose:
        print(f"\n=== Solving for tau_hat (alpha={alpha}, n={n}) ===")
        print(f"Global mu: {mu_g[0, 0, 0].item():.6f}, Global sigma: {sigma_g[0, 0, 0].item():.6f}")
        print(f"Initial bounds: [{tau_low[0, 0, 0].item():.6f}, {tau_high[0, 0, 0].item():.6f}]")

    f_low = _evaluate_constraint(tau_low, mu_g, sigma_g, n, alpha)
    f_high = _evaluate_constraint(tau_high, mu_g, sigma_g, n, alpha)

    if verbose:
        print(f"f(tau_low) = {f_low[0,0,0].item():.6f}, f(tau_high) = {f_high[0,0,0].item():.6f}")

    while (f_low < 0).any():
        tau_low = tau_low - 2.0 * sigma_g
        f_low = _evaluate_constraint(tau_low, mu_g, sigma_g, n, alpha)

    while (f_high > 0).any():
        tau_high = tau_high + 2.0 * sigma_g
        f_high = _evaluate_constraint(tau_high, mu_g, sigma_g, n, alpha)

    if verbose:
        print(f"Adjusted bounds: [{tau_low[0, 0, 0].item():.6f}, {tau_high[0, 0, 0].item():.6f}]")

    for iteration in range(max_iter):
        tau_mid = (tau_low + tau_high) / 2.0
        f_mid = _evaluate_constraint(tau_mid, mu_g, sigma_g, n, alpha)

        if verbose and iteration % 5 == 0:
            print(f"Iter {iteration}: tau_mid={tau_mid[0,0,0].item():.6f}, f(tau_mid)={f_mid[0,0,0].item():.6f}")

        if torch.abs(f_mid).max() < tol:
            if verbose:
                print(f"Converged at iteration {iteration}")
            tau = tau_mid
            break

        tau_low = torch.where(f_mid > 0, tau_mid, tau_low)
        tau_high = torch.where(f_mid <= 0, tau_mid, tau_high)
    else:
        tau = (tau_low + tau_high) / 2.0

    if verbose:
        print(f"Final tau_hat: {tau[0, 0, 0].item():.6f}")
        final_f = _evaluate_constraint(tau, mu_g, sigma_g, n, alpha)
        print(f"Final n*E[g_alpha] - 1 = {final_f[0,0,0].item():.6f} (target: 0.0)")
        print(f"Threshold for mask (tau/(alpha-1)): {(tau / (alpha - 1.0))[0,0,0].item():.6f}\n")

    return tau.to(device=original_device, dtype=original_dtype)


def solve_for_tau_hat_single_gaussian(
    mu_g: torch.Tensor,
    sigma_g: torch.Tensor,
    n: int,
    alpha: float,
    max_iter: int = 120,
    tol: float = 1e-6,
    verbose: bool = False
) -> torch.Tensor:
    """

    Solve for τ̂ using bisection method on the distributional entmax constraint:
    n E[g_α(S; τ)] = 1

    where S ~ N(μ_g, σ_g²) is a SINGLE Gaussian approximating the score distribution.

    The function f(τ) = n E[g_α(S; τ)] - 1 is monotonically decreasing in τ,
    so we can use bisection to find the root.

    Args:
        mu_g: Global mean score [batch, heads, 1]
        sigma_g: Global std score [batch, heads, 1]
        n: Number of tokens
        alpha: Entmax alpha parameter
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        verbose: Print debug information

    Returns:
        tau_hat: Threshold value [batch, heads, 1]
    """
    original_dtype = mu_g.dtype
    if _HAS_GPU_SOLVER and mu_g.is_cuda and alpha in (2, 1.5) and not verbose:
        tau = solve_for_tau_hat_single_gaussian_gpu(
            mu_g=mu_g,
            sigma_g=sigma_g,
            n=n,
            alpha=alpha,
            iterations=max_iter,
            tol=tol,
        )
        return tau.to(dtype=original_dtype)

    mu_g = mu_g.double()
    sigma_g = sigma_g.double()

    if alpha in (2, 1.5) and not verbose:
        tau = _solve_for_tau_hat_scalar_cpu(
            mu_g=mu_g,
            sigma_g=sigma_g,
            n=n,
            alpha=alpha,
            iterations=max_iter,
            tol=tol,
        )
        return tau.to(dtype=original_dtype)

    return _solve_for_tau_hat_cpu_bisection(
        mu_g=mu_g.to(dtype=original_dtype),
        sigma_g=sigma_g.to(dtype=original_dtype),
        n=n,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )


def compute_page_selection_probability(
    mu_scores: torch.Tensor,
    sigma_scores: torch.Tensor,
    tau_hat: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Compute probability that each page exceeds the threshold.

    P(score^(p) > τ̂/(α-1)) where score^(p) ~ N(μ^(p), (σ^(p))²)

    Args:
        mu_scores: Per-page expected scores [batch, heads, pages]
        sigma_scores: Per-page score standard deviations [batch, heads, pages]
        tau_hat: Threshold value [batch, heads, 1]
        alpha: Entmax alpha parameter

    Returns:
        Probability that each page exceeds threshold [batch, heads, pages]
    """
    threshold = tau_hat / (alpha - 1.0)

    # Compute z-score for each page
    z = (threshold - mu_scores) / sigma_scores.clamp(min=1e-6)

    # Handle potential NaN in z-scores
    z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)

    # P(s > threshold) = 1 - Φ(z)
    prob_support = 1.0 - _gaussian_cdf(z)

    # Handle NaN in probabilities (fallback to uniform probabilities)
    prob_support = torch.where(
        torch.isnan(prob_support),
        torch.ones_like(prob_support) * 0.5,
        prob_support
    )

    return prob_support