import math
import torch
from typing import Optional, Tuple

from entmaxkv.tau_solver import solve_for_tau_hat_single_gaussian
from entmaxkv.kernels.tau_solver_page_mixture import solve_for_tau_hat_page_gaussian_mixture

try:
    from entmaxkv.kernels.tau_mixture_solver_triton import (
        solve_for_tau_hat_page_gaussian_mixture_triton,
    )
except Exception:
    solve_for_tau_hat_page_gaussian_mixture_triton = None


def solve_for_tau_hat_page_gaussian_mixture_auto(
    mu_pages: "torch.Tensor",
    sigma_pages: "torch.Tensor",
    page_counts: "torch.Tensor" = None,
    alpha: float = 1.5,
    max_iter: int = 40,
    tol: float = 1e-6,
    uniform_page_count: float = 1.0,
) -> "torch.Tensor":
    can_use_triton_tau = (
        solve_for_tau_hat_page_gaussian_mixture_triton is not None
        and mu_pages.is_cuda
        and sigma_pages.is_cuda
        and (page_counts is None or page_counts.is_cuda)
    )
    if can_use_triton_tau:
        return solve_for_tau_hat_page_gaussian_mixture_triton(
            mu_pages,
            sigma_pages,
            page_counts=page_counts,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            uniform_page_count=uniform_page_count,
        )
    if page_counts is None:
        page_counts = torch.full(
            mu_pages.shape, float(uniform_page_count),
            device=mu_pages.device, dtype=mu_pages.dtype,
        )
    return solve_for_tau_hat_page_gaussian_mixture(
        mu_pages, sigma_pages, page_counts=page_counts, alpha=alpha,
        max_iter=max_iter, tol=tol,
    )

try:
    from entmaxkv.kernels.gaussian_page_stats import (
        compute_page_gaussian_stats_triton,
    )
except Exception:
    compute_page_gaussian_stats_triton = None

try:
    from entmaxkv.kernels.tau_clamp import (
        clamp_tau_to_selected_page_statistics_triton,
    )
except Exception:
    clamp_tau_to_selected_page_statistics_triton = None


def compute_gaussian_aware_statistics(
    q: "torch.Tensor",
    k_mean: "torch.Tensor",
    k_std: "torch.Tensor",
    seq_len: int,
    page_size: int,
    alpha: float = 1.5,
    alibi_slopes: "torch.Tensor" = None,
    use_triton_stats: bool = True,
    threshold_excess_margin_fraction: float = 0.2,
    tau_clamp_page_max_quantile: float = 0.50,
    safety_margin_z: Optional[float] = None,
    max_quantile: Optional[float] = None,
):
    """
    Compute Gaussian score statistics and the distributional entmax tau estimate.

    Args:
        q: Query tensor [B, H, 1, D].
        k_mean: Per-page key mean [B, H, pages, D].
        k_std: Per-page key std [B, H, pages, D].
        seq_len: Current full cache sequence length.
        page_size: Cache page size.

    Returns:
        stats dict with per-page statistics plus tau_hat [B, H].
    """
    _, _, _, head_dim = q.shape

    scale = 1.0 / math.sqrt(head_dim)
    num_pages = k_mean.shape[2]

    pages_for_stats = num_pages - 1 if num_pages > 1 else num_pages
    k_mean_for_stats = k_mean[:, :, :pages_for_stats, :]
    k_std_for_stats = k_std[:, :, :pages_for_stats, :]
    last_page_tokens = seq_len - (num_pages - 1) * page_size if num_pages > 1 else 0
    seq_len_for_stats = seq_len - last_page_tokens

    can_use_triton_stats = (
        use_triton_stats
        and compute_page_gaussian_stats_triton is not None
        and q.is_cuda
        and k_mean.is_cuda
        and k_std.is_cuda
        and q.shape[0] == k_mean.shape[0]
        and q.shape[1] == k_mean.shape[1]
    )

    if can_use_triton_stats:
        (
            mu_scores_per_page,
            score_variance_per_page,
            mean_variance,
            variance_of_means,
            raw_mu_global,
        ) = compute_page_gaussian_stats_triton(
            q=q,
            k_mean=k_mean,
            k_std=k_std,
            pages_for_stats=pages_for_stats,
            return_sigma=True,
        )
        sigma_scores_per_page = score_variance_per_page
    else:
        q_scaled = q * scale
        q_var_scaled = q.square() * (scale * scale)

        mu_scores_per_page = (q_scaled * k_mean_for_stats).sum(dim=-1)
        score_variance_per_page = (q_var_scaled * k_std_for_stats.square()).sum(dim=-1)

        mean_variance = score_variance_per_page.mean(dim=-1, keepdim=True)
        variance_of_means, raw_mu_global = torch.var_mean(
            mu_scores_per_page, dim=-1, keepdim=True, unbiased=False
        )
        sigma_scores_per_page = torch.sqrt(score_variance_per_page)

    if alibi_slopes is not None:
        slopes = alibi_slopes.to(q.device, dtype=mu_scores_per_page.dtype)
        if slopes.dim() == 1:
            slopes = slopes.unsqueeze(0)
        slopes = slopes.unsqueeze(-1)
        q_pos = seq_len - 1
        page_mean_pos = (
            torch.arange(pages_for_stats, device=q.device).float() * page_size
            + (page_size - 1) / 2.0
        ).clamp(max=seq_len_for_stats - 1)
        alibi_bias_per_page = slopes * (page_mean_pos - q_pos)
        mu_scores_per_page = mu_scores_per_page + alibi_bias_per_page
    
    mu_global = torch.nan_to_num(raw_mu_global, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_global = torch.nan_to_num(
        torch.sqrt(mean_variance + variance_of_means),
        nan=1e-6,
        posinf=1e6,
        neginf=1e-6,
    ).clamp(min=1e-6)

    if safety_margin_z is None or max_quantile is None:
        sigma_scalar = sigma_global.mean()
        low_sigma = bool((sigma_scalar < 1.5).item())
        if safety_margin_z is None:
            safety_margin_z = (0.01 if low_sigma else 0.05) * float(sigma_scalar.item())
        if max_quantile is None:
            max_quantile = 0.99 if low_sigma else 0.995

    if alibi_slopes is not None:
        tau_hat = solve_for_tau_hat_page_gaussian_mixture_auto(
            mu_scores_per_page,
            sigma_scores_per_page,
            alpha=alpha,
            uniform_page_count=float(page_size),
        )  # [B, H, 1] — squeezed at stats dict assignment below
    else:
        tau_hat = solve_for_tau_hat_single_gaussian(
            mu_global, sigma_global, seq_len_for_stats, alpha
        )

    stats = {
        "mu_scores_per_page": mu_scores_per_page,
        "sigma_scores_per_page": sigma_scores_per_page,
        "mu_global": mu_global,
        "sigma_global": sigma_global,
        "tau_hat": tau_hat.squeeze(-1).to(device=q.device, dtype=torch.float32).contiguous(),
        "seq_len_for_stats": seq_len_for_stats,
        "pages_for_stats": pages_for_stats,
        "num_pages": num_pages,
        "safety_margin_z": safety_margin_z,
        "threshold_excess_margin_fraction": threshold_excess_margin_fraction,
        "tau_clamp_page_max_quantile": tau_clamp_page_max_quantile,
        "max_quantile": max_quantile,
        "mean_variance": mean_variance,
        "variance_of_means": variance_of_means,
    }

    return stats


def clamp_tau_to_selected_page_statistics(
    tau: "torch.Tensor",
    gaussian_stats: dict,
    page_indices: "torch.Tensor",
    num_selected_per_head: "torch.Tensor",
    page_size: int,
    alpha: float,
    eps: float = 1.0e-4,
) -> "torch.Tensor":
    """
    Clamp tau downward so it cannot exceed a per-head upper bound derived from the
    selected pages, preventing valid tokens from being zeroed out by the entmax mask.

    The upper bound for each head is:

        tau_stat_upper = (alpha - 1) * page_cap_max - eps

    where page_cap_max is the maximum, over selected pages, of a score quantile:

        page_cap[p] = mu[p] + z_clamp * sigma[p] / 3

    and z_clamp is the normal quantile corresponding to `clamp_quantile` raised to
    the power 1/page_size (treating page membership as independent Bernoulli draws).
    This cap estimates the highest score a token in the selected pages is likely to
    attain; (alpha-1)*cap is therefore the largest tau that still keeps at least one
    token in the entmax support.

    If `tau_floor` is provided it acts as a lower bound on the clamped value, ensuring
    tau does not drop below a minimum useful threshold.

    Args:
        tau: Threshold tensor, shape [B, H] or [B, H, 1].
        gaussian_stats: Dict with keys:
            - mu_scores_per_page:  [B, H, n_pages] per-page score means.
            - sigma_scores_per_page: [B, H, n_pages] per-page score std devs.
            - pages_for_stats: int, number of pages that carry valid statistics.
            - tau_clamp_page_max_quantile: float, target per-page max quantile
              (default 0.50).
            - tau_floor: optional lower bound tensor, same shape as tau.
        page_indices: [B, H, k] indices of the k selected pages per head.
        num_selected_per_head: [B, H] number of valid entries in page_indices.
        page_size: Number of tokens per page (used to convert page quantile to
            per-token quantile).
        alpha: Entmax alpha parameter.
        eps: Small margin subtracted from the upper bound so the boundary token
            is kept strictly in the support.

    Returns:
        Clamped tau, same shape as the input tau.
    """
    if tau is None:
        return tau

    original_shape = tau.shape
    tau_2d = tau.squeeze(-1) if tau.dim() == 3 and tau.shape[-1] == 1 else tau

    mu_scores_per_page = gaussian_stats["mu_scores_per_page"]
    sigma_scores_per_page = gaussian_stats["sigma_scores_per_page"]
    clamp_quantile = gaussian_stats.get("tau_clamp_page_max_quantile", 0.50)
    pages_for_stats = gaussian_stats["pages_for_stats"]
    tau_floor = gaussian_stats.get("tau_floor")

    if (
        clamp_tau_to_selected_page_statistics_triton is not None
        and tau_2d.is_cuda
        and mu_scores_per_page.is_cuda
        and sigma_scores_per_page.is_cuda
        and page_indices.is_cuda
        and num_selected_per_head.is_cuda
    ):
        return clamp_tau_to_selected_page_statistics_triton(
            tau=tau_2d,
            mu_scores_per_page=mu_scores_per_page,
            sigma_scores_per_page=sigma_scores_per_page,
            page_indices=page_indices,
            num_selected_per_head=num_selected_per_head,
            tau_floor=tau_floor,
            page_size=page_size,
            alpha=alpha,
            clamp_quantile=clamp_quantile,
            pages_for_stats=pages_for_stats,
            eps=eps,
        )

    per_sample_prob = clamp_quantile ** (1.0 / page_size)
    z_clamp = math.sqrt(2.0) * torch.erfinv(
        torch.tensor(
            2.0 * per_sample_prob - 1.0,
            device=mu_scores_per_page.device,
            dtype=torch.float32,
        )
    )
    page_cap = mu_scores_per_page + z_clamp.to(mu_scores_per_page.dtype) * sigma_scores_per_page/3

    selected = page_indices.long()
    slots = torch.arange(selected.shape[-1], device=selected.device).view(1, 1, -1)
    valid = (slots < num_selected_per_head.long().unsqueeze(-1)) & (selected < pages_for_stats)
    safe_selected = selected.clamp(min=0, max=max(pages_for_stats - 1, 0))
    gathered = torch.gather(page_cap, dim=2, index=safe_selected)
    gathered = torch.where(valid, gathered, torch.full_like(gathered, -float("inf")))

    has_valid = valid.any(dim=-1)
    max_selected_page_cap = gathered.amax(dim=-1)
    tau_stat_upper = (alpha - 1.0) * max_selected_page_cap - eps
    if tau_floor is not None:
        tau_floor = tau_floor.to(device=tau_stat_upper.device, dtype=tau_stat_upper.dtype)
        tau_floor = tau_floor.squeeze(-1) if tau_floor.dim() == 3 and tau_floor.shape[-1] == 1 else tau_floor
        tau_stat_upper = torch.maximum(tau_stat_upper, tau_floor)
    clamped = torch.where(has_valid, torch.minimum(tau_2d, tau_stat_upper), tau_2d)

    if original_shape != clamped.shape:
        clamped = clamped.unsqueeze(-1)
    return clamped.contiguous()
