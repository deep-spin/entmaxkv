"""Page selectors: Quest-style top-k and Gaussian-aware.

Both return a :class:`SelectedPages` keyed by QUERY head: logical page indices
(ascending for Gaussian, ranked for top-k), the mandatory tail page last, and
-1 padding. The decoders resolve logical -> block_table -> physical themselves.
Each selector owns persistent, shape-keyed scratch, so a decode step allocates
nothing and never synchronises with the host.
"""

from dataclasses import dataclass

import torch

from entmaxkv.kernels.gaussian.page_stats import gaussian_score_stats
from entmaxkv.kernels.gaussian.selection import select_and_pack
from entmaxkv.kernels.gaussian.tau_solver import (
    solve_page_mixture,
    solve_single_gaussian,
)
from entmaxkv.kernels.page_criticality import page_criticality
from entmaxkv.kernels.page_topk import select_pages


def topk_is_worthwhile(seq_len: int, topk_pages: int, min_context: int,
                       page_size: int) -> bool:
    """Sparse top-k saves work only if the budget does not cover the row."""
    if seq_len < min_context:
        return False
    return -(-seq_len // page_size) > topk_pages


def gaussian_is_worthwhile(seq_len: int, min_context: int,
                           page_size: int) -> bool:
    return seq_len >= min_context and seq_len > page_size


@dataclass
class SelectedPages:
    """Logical page indices per (request, query head), -1 padded."""

    pages: torch.Tensor      # int32[B, H_q, width]
    sel_lens: torch.Tensor   # int32[B, H_q], candidate token counts
    tau: torch.Tensor | None = None  # Gaussian estimate, (alpha-1) tau space


class TopKPageSelector:
    """Quest-style criticality + top-k page selection.

    ``topk_pages`` counts the TOTAL attended pages per query head, tail
    inclusive (so it must be at least 2).
    """

    def __init__(self, topk_pages: int, page_size: int):
        if topk_pages < 2:
            raise ValueError("topk_pages must be at least 2 (tail inclusive)")
        self.topk_pages = topk_pages
        self.page_size = page_size
        self._cache: dict = {}

    def _scratch(self, batch, nheads, max_pages, device):
        key = (batch, nheads, max_pages, str(device))
        ws = self._cache.get(key)
        if ws is not None:
            return ws
        remote_k = self.topk_pages - 1
        ws = {
            "page_scores": torch.empty((batch, nheads, max_pages),
                                       device=device, dtype=torch.float32),
            "topk_vals": torch.empty((batch, nheads, remote_k), device=device,
                                     dtype=torch.float32),
            "topk_idx": torch.empty((batch, nheads, remote_k), device=device,
                                    dtype=torch.int64),
            "selected": torch.empty((batch, nheads, self.topk_pages),
                                    device=device, dtype=torch.int32),
            "sel_lens": torch.empty((batch, nheads), device=device,
                                    dtype=torch.int32),
        }
        self._cache[key] = ws
        return ws

    def select(self, query, k_min, k_max, block_table, seq_lens, slopes,
               live_pages: int) -> SelectedPages:
        """``live_pages`` (host int) bounds the scored block-table prefix."""
        batch, nheads, _ = query.shape
        if live_pages < self.topk_pages - 1:
            raise ValueError(
                f"only {live_pages} live pages for a remote budget of "
                f"{self.topk_pages - 1}; the caller should decode exactly")
        ws = self._scratch(batch, nheads, block_table.shape[1], query.device)
        page_criticality(query, k_min, k_max, block_table, seq_lens, slopes,
                         self.page_size, out=ws["page_scores"],
                         live_pages=live_pages)
        select_pages(ws["page_scores"][:, :, :live_pages], seq_lens,
                     self.topk_pages, self.page_size, ws["topk_vals"],
                     ws["topk_idx"], ws["selected"], ws["sel_lens"])
        return SelectedPages(pages=ws["selected"], sel_lens=ws["sel_lens"])


class GaussianPageSelector:
    """Select variable candidate sets from per-physical-page key moments.

    tau-hat is chosen PER HEAD: heads with a nonzero ALiBi slope use the
    page-mixture model (scores drift with position), NoPE heads (slope 0) use
    a single Gaussian over the whole history.
    """

    def __init__(self, page_size: int, alpha: float = 1.5,
                 safety_margin_z: float = 0.0, max_quantile: float = 0.995,
                 threshold_excess_margin_fraction: float = 0.1):
        if not alpha > 1.0:
            raise ValueError("Gaussian selection requires alpha > 1")
        if safety_margin_z < 0:
            raise ValueError("safety_margin_z must be >= 0")
        if not 0.0 < max_quantile < 1.0:
            raise ValueError("max_quantile must be between 0 and 1")
        if threshold_excess_margin_fraction < 0:
            raise ValueError("threshold_excess_margin_fraction must be >= 0")
        self.page_size = page_size
        self.alpha = float(alpha)
        self.safety_margin_z = safety_margin_z
        self.max_quantile = max_quantile
        self.threshold_excess_margin_fraction = \
            threshold_excess_margin_fraction
        self._cache: dict = {}

    def _scratch(self, batch, heads, width, device):
        key = (batch, heads, width, str(device))
        ws = self._cache.get(key)
        if ws is None:
            shape = (batch, heads, width)
            f32 = dict(device=device, dtype=torch.float32)
            i32 = dict(device=device, dtype=torch.int32)
            ws = {
                "mu": torch.empty(shape, **f32),
                "sigma": torch.empty(shape, **f32),
                "tau_single": torch.empty((batch, heads), **f32),
                "tau_mixture": torch.empty((batch, heads), **f32),
                "tau": torch.empty((batch, heads), **f32),
                "tau_mixture_workspace": {},
                "mean": torch.empty((batch, heads), **f32),
                "sigma_global": torch.empty((batch, heads), **f32),
                "token_counts": torch.empty((batch, heads), **i32),
                "historical_pages": torch.empty((batch, heads), **i32),
                "selected": torch.empty(shape, **i32),
                "page_counts": torch.empty((batch, heads), **i32),
                "sel_lens": torch.empty((batch, heads), **i32),
            }
            self._cache[key] = ws
        return ws

    def select(self, query, k_mean, k_std, block_table, seq_lens, slopes,
               live_pages: int) -> SelectedPages:
        batch, heads, _ = query.shape
        width = block_table.shape[1]
        ws = self._scratch(batch, heads, width, query.device)
        mu, sigma, mean, sigma_global = gaussian_score_stats(
            query, k_mean, k_std, block_table, seq_lens, slopes,
            self.page_size, ws["mu"], ws["sigma"], ws["mean"],
            ws["sigma_global"], ws["token_counts"], ws["historical_pages"],
            n_pages=live_pages)
        tau_single = solve_single_gaussian(
            mean, sigma_global, ws["token_counts"], alpha=self.alpha,
            out=ws["tau_single"])
        head_has_alibi = slopes.ne(0).contiguous()
        tau_mixture = solve_page_mixture(
            mu, sigma, ws["historical_pages"], self.page_size,
            alpha=self.alpha, out=ws["tau_mixture"],
            active_heads=head_has_alibi,
            workspace=ws["tau_mixture_workspace"], n_pages=live_pages)
        tau = torch.where(head_has_alibi.view(1, heads), tau_mixture,
                          tau_single, out=ws["tau"])
        pages, _, sel_lens = select_and_pack(
            mu, sigma, tau, sigma_global, seq_lens, self.page_size,
            self.safety_margin_z, self.max_quantile,
            self.threshold_excess_margin_fraction, alpha=self.alpha,
            out=ws["selected"], counts=ws["page_counts"],
            sel_lens=ws["sel_lens"], n_pages=live_pages)
        return SelectedPages(pages=pages, sel_lens=sel_lens, tau=tau)
