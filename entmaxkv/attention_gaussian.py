from typing import Optional

import torch

from entmaxkv.decoding import (
    decode_exact,
    decode_on_selection,
    decode_views,
    query_slopes,
    workspace,
)
from entmaxkv.kernels.gaussian.decode import (
    GaussianDecodeWorkspace,
    entmax_decode_gaussian_pages,
)
from entmaxkv.kv_cache import PagedKVCache
from entmaxkv.selectors import GaussianPageSelector, gaussian_is_worthwhile

TAU_MODES = ("exact", "fixed", "corrected")


def sparse_attention_decode_gaussian_aware_entmax(
    q: torch.Tensor,
    kv_cache: PagedKVCache,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.5,
    safety_margin_z: float = 0.0,
    max_quantile: float = 0.995,
    threshold_excess_margin_fraction: float = 0.1,
    alibi_slopes: Optional[torch.Tensor] = None,
    niter: int = 3,
    append_cache: bool = False,
    tau_mode: str = "corrected",
    min_context: int = 0,
) -> torch.Tensor:
    """
    Gaussian-aware page selection feeding a paged entmax decoder.

    Per-page key moments give each page a Gaussian score model. tau-hat solves
    the distributional entmax constraint (page mixture for ALiBi heads, single
    Gaussian for NoPE heads), and every page whose estimated maximum can
    cross the safety-lowered threshold is selected, plus the tail page.

    tau_mode:
        "exact"     solves tau exactly on the selected pages.
        "fixed"     uses tau-hat directly (no correction).
        "corrected" starts from tau-hat clamped into [max - 1, max] of the
                    exact candidate-set maximum and runs ``niter`` bracketed
                    hybrid (Halley/Newton/secant/bisection) corrections.

    The whole batch decodes exactly when any row is shorter than
    ``min_context`` or fits in a single page.
    """
    tau_mode = tau_mode.lower()
    if tau_mode not in TAU_MODES:
        raise ValueError(f"Unsupported tau_mode={tau_mode!r}; "
                         f"expected one of {TAU_MODES}")
    if niter < 0:
        raise ValueError("niter must be >= 0")
    if min_context < 0:
        raise ValueError("min_context must be >= 0")
    if append_cache:
        kv_cache.append(k_new, v_new)
    q3, out3 = decode_views(q, out)
    slopes = query_slopes(alibi_slopes, q.shape[1], q.device)
    ps = kv_cache.page_size
    if not all(gaussian_is_worthwhile(n, min_context, ps)
               for n in kv_cache.seq_lens_host):
        decode_exact(q3, kv_cache, alpha, slopes, None, out3)
        return out
    if kv_cache.k_mean is None:
        raise ValueError(
            "Gaussian decoding needs a cache built with 'gaussian' stats")

    key = ("gaussian", float(alpha), float(safety_margin_z),
           float(max_quantile), float(threshold_excess_margin_fraction))
    selector = workspace(kv_cache, key, lambda: GaussianPageSelector(
        ps, alpha, safety_margin_z, max_quantile,
        threshold_excess_margin_fraction))
    sel = selector.select(q3, kv_cache.k_mean, kv_cache.k_std,
                          kv_cache.block_table, kv_cache.seq_lens, slopes,
                          live_pages=kv_cache.live_pages)

    if tau_mode == "exact":
        decode_on_selection(q3, kv_cache, sel.pages, sel.sel_lens, alpha,
                            slopes, None, out3)
        return out
    entmax_decode_gaussian_pages(
        q3, kv_cache.kv, kv_cache.block_table, sel.pages, sel.sel_lens,
        kv_cache.seq_lens, sel.tau, slopes, alpha=alpha,
        niter=0 if tau_mode == "fixed" else niter, out=out3,
        workspace=workspace(kv_cache, "gaussian_decode",
                            GaussianDecodeWorkspace))
    return out
