import torch
from typing import Optional

try:
    from entmaxkv.kernels.page_criticality import triton_estimate_page_criticality
except Exception:
    triton_estimate_page_criticality = None

try:
    from entmaxkv.kernels.selection_pack import select_gaussian_threshold_pack_triton
except Exception:
    select_gaussian_threshold_pack_triton = None



class PagedKVCache:
    """
    KV Cache with paged metadata tracking for criticality estimation.

    Maintains min/max/mean/std statistics per page for efficient query-aware sparsity.
    """

    def __init__(self, page_size: int = 16):
        self.page_size = page_size
        self.k_cache = None
        self.v_cache = None
        self.k_min = None
        self.k_max = None
        self.k_mean = None
        self.k_std = None

        self.last_num_selected_per_head = None

        # Persistent decode-step buffers, reused across calls to avoid
        # per-step allocation churn.
        self._page_indices_buf: Optional[torch.Tensor] = None
        self._page_indices_last_page: Optional[int] = None
        self._paged_decode_workspaces: dict = {}

    def initialize(self, k: torch.Tensor, v: torch.Tensor):
        """Initialize cache with prefill keys/values."""
        batch, num_heads, seq_len, head_dim = k.shape

        self.k_cache = k.clone()
        self.v_cache = v.clone()

        # Compute initial min/max/mean/std statistics per page
        num_pages = (seq_len + self.page_size - 1) // self.page_size
        self.k_min = torch.zeros(batch, num_heads, num_pages, head_dim,
                                 dtype=k.dtype, device=k.device)
        self.k_max = torch.zeros(batch, num_heads, num_pages, head_dim,
                                 dtype=k.dtype, device=k.device)
        self.k_mean = torch.zeros(batch, num_heads, num_pages, head_dim,
                                  dtype=k.dtype, device=k.device)
        self.k_std = torch.zeros(batch, num_heads, num_pages, head_dim,
                                 dtype=k.dtype, device=k.device)

        for page_idx in range(num_pages):
            start_idx = page_idx * self.page_size
            end_idx = min(start_idx + self.page_size, seq_len)
            page_keys = k[:, :, start_idx:end_idx, :]

            self.k_min[:, :, page_idx, :] = page_keys.min(dim=2)[0]
            self.k_max[:, :, page_idx, :] = page_keys.max(dim=2)[0]
            self.k_mean[:, :, page_idx, :] = page_keys.mean(dim=2)

            # Compute std with unbiased=False to handle pages with few tokens
            page_std = page_keys.std(dim=2, unbiased=False)
            # Clamp to avoid snumerical issues
            self.k_std[:, :, page_idx, :] = page_std.clamp(min=1e-6)

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        Append new tokens to cache and update metadata.

        Args:
            k_new: [batch, num_heads, 1, head_dim]
            v_new: [batch, num_heads, 1, head_dim]
        """
        # Append to cache
        self.k_cache = torch.cat([self.k_cache, k_new], dim=2)
        self.v_cache = torch.cat([self.v_cache, v_new], dim=2)

        # Update min/max/mean/std for the last page
        seq_len = self.k_cache.shape[2]
        page_idx = (seq_len - 1) // self.page_size

        if page_idx >= self.k_min.shape[2]:
            batch, num_heads, _, head_dim = self.k_cache.shape
            new_page_min = torch.zeros(batch, num_heads, 1, head_dim,
                                       dtype=k_new.dtype, device=k_new.device)
            new_page_max = torch.zeros(batch, num_heads, 1, head_dim,
                                       dtype=k_new.dtype, device=k_new.device)
            new_page_mean = torch.zeros(batch, num_heads, 1, head_dim,
                                        dtype=k_new.dtype, device=k_new.device)
            new_page_std = torch.zeros(batch, num_heads, 1, head_dim,
                                       dtype=k_new.dtype, device=k_new.device)
            self.k_min = torch.cat([self.k_min, new_page_min], dim=2)
            self.k_max = torch.cat([self.k_max, new_page_max], dim=2)
            self.k_mean = torch.cat([self.k_mean, new_page_mean], dim=2)
            self.k_std = torch.cat([self.k_std, new_page_std], dim=2)

        # Update min/max/mean/std for current page
        start_idx = page_idx * self.page_size
        end_idx = seq_len
        page_keys = self.k_cache[:, :, start_idx:end_idx, :]

        self.k_min[:, :, page_idx, :] = page_keys.min(dim=2)[0]
        self.k_max[:, :, page_idx, :] = page_keys.max(dim=2)[0]
        self.k_mean[:, :, page_idx, :] = page_keys.mean(dim=2)

        # Compute std with unbiased=False to handle single-element pages
        # For single element pages, std will be 0 which is correct
        page_std = page_keys.std(dim=2, unbiased=False)
        # Clamp to avoid exact zeros which can cause numerical issues
        self.k_std[:, :, page_idx, :] = page_std.clamp(min=1e-6)

    def estimate_page_criticality(self, q: torch.Tensor,
                                    use_triton: bool = False,
                                    alibi_slopes=None,
                                    q_pos: int = 0,
                                    exclude_last_page: bool = False) -> torch.Tensor:
        """
        Estimate criticality scores for each page given a query.

        score = sum_i max(q_i * min_i, q_i * max_i) [/ sqrt(d_head)]
                [+ alibi_slopes[h] * (min((p+1)*page_size-1, seq_len-1) - q_pos)]

        Args:
            q: Query tensor [batch, heads, 1, head_dim] or [batch, heads, pages, head_dim]
            apply_scaling: If True, apply attention scaling 1/sqrt(d_head) (default: False)
            use_triton: If True, use fused Triton kernel (default: False)
            alibi_slopes: [num_heads] float32 raw slopes; bonus computed in-kernel or Python-side.
            q_pos: absolute position of the query token (seq_len - 1).
            exclude_last_page: drop the in-progress last page from scoring.

        Returns:
            page_scores: Upper bound scores per page [batch, heads, pages]
        """
        if use_triton:
            return triton_estimate_page_criticality(
                q, self.k_min, self.k_max,
                alibi_slopes=alibi_slopes,
                q_pos=q_pos,
                page_size=self.page_size,
                seq_len=self.k_cache.shape[2],
                exclude_last_page=exclude_last_page,
            )

        k_min = self.k_min[:, :, :-1, :] if exclude_last_page else self.k_min
        k_max = self.k_max[:, :, :-1, :] if exclude_last_page else self.k_max
        upper_bound = torch.maximum(q * k_min, q * k_max)
        page_scores = upper_bound.sum(dim=-1)

        if alibi_slopes is not None:
            seq_len = self.k_cache.shape[2]
            num_pages = page_scores.shape[2]
            page_last_pos = torch.clamp(
                torch.arange(num_pages, device=q.device, dtype=torch.float32) * self.page_size + self.page_size - 1,
                max=seq_len - 1,
            )
            alibi_bonus = alibi_slopes.float().view(1, -1, 1) * (page_last_pos - q_pos)
            page_scores = page_scores + alibi_bonus

        return page_scores

    def get_page_indices_buf(
        self, batch: int, num_heads: int, num_pages_to_select: int, last_page_idx: int
    ) -> torch.Tensor:
        """Return a persistent [B, H, n_selected+1] page-indices buffer.

        Avoids a torch.empty + fill_ every decode step. During autoregressive
        decode the last page changes only once per page_size steps, so the
        fill_ for the sentinel column is skipped on the other steps.
        """
        target_shape = (batch, num_heads, num_pages_to_select + 1)
        buf = self._page_indices_buf
        if buf is None or tuple(buf.shape) != target_shape or buf.device != self.k_cache.device:
            buf = torch.empty(target_shape, dtype=torch.int32, device=self.k_cache.device)
            self._page_indices_buf = buf
            self._page_indices_last_page = None
        if self._page_indices_last_page != last_page_idx:
            buf[:, :, num_pages_to_select].fill_(last_page_idx)
            self._page_indices_last_page = last_page_idx
        return buf

    def get_paged_decode_workspace(
        self, batch: int, nheads: int, dim: int, max_splits: int, bins: int,
        dtype: torch.dtype, device: torch.device,
    ) -> dict:
        """Return a persistent workspace for one exact paged-kernel shape."""
        device = torch.device(device)
        key = (batch, nheads, dim, max_splits, bins, dtype, device.type, device.index)
        workspace = self._paged_decode_workspaces.get(key)
        if workspace is None:
            from entmaxkv.kernels.adadecode_paged import (
                make_adadecode_paged_workspace,
            )
            workspace = make_adadecode_paged_workspace(
                batch, nheads, dim, max_splits, bins, device=device, dtype=dtype
            )
            self._paged_decode_workspaces[key] = workspace
        return workspace

    def select_gaussian_aware(self, q: torch.Tensor, alpha: float = 1.5, safety_margin_z: float = 0.0,
                               max_quantile: float = 0.99,
                               gaussian_stats: dict = None):
        """
        Returns selected_page_indices: [B, H, max_selected_pages].
        Slots beyond num_selected_per_head[b, h] are zero (uninitialized) — callers must
        use last_num_selected_per_head to know the valid length per head.
        The last page (most recent) is always included via an inf sentinel.
        Also stores self.last_page_scores and self.last_selected_page_indices.

        safety_margin_z: dimensionless z-score margin. Lowers the effective score threshold by
            safety_margin_z * σ_global, i.e. Δ = safety_margin_z * σ_global * (α-1) in τ-space.
            Scale-invariant: 0.5 means "include tokens within half a global σ of the entmax cut-off".

        alibi_slopes: [H] or [B, H] tensor of ALiBi slopes. When provided:
            - μ_global is shifted by slope*(mean_k_pos - q_pos)
        """
        mu_scores_per_page = gaussian_stats["mu_scores_per_page"]
        sigma_scores_per_page = gaussian_stats["sigma_scores_per_page"]
        sigma_global = gaussian_stats["sigma_global"]
        pages_for_stats = gaussian_stats["pages_for_stats"]
        num_pages = gaussian_stats["num_pages"]
        safety_margin_z = gaussian_stats["safety_margin_z"]
        threshold_excess_margin_fraction = gaussian_stats.get("threshold_excess_margin_fraction", 0.1)
        max_quantile = gaussian_stats["max_quantile"]
        self.last_effective_safety_margin = safety_margin_z
        self.last_effective_max_quantile  = max_quantile

        assert select_gaussian_threshold_pack_triton is not None, "select_gaussian_threshold_pack_triton kernel not available"
        assert q.is_cuda, "q must be on CUDA"
        assert mu_scores_per_page.is_cuda, "mu_scores_per_page must be on CUDA"
        assert sigma_scores_per_page.is_cuda, "sigma_scores_per_page must be on CUDA"

        selected_page_indices, num_selected_per_head, tau_floor = (
            select_gaussian_threshold_pack_triton(
                mu_scores_per_page=mu_scores_per_page,
                sigma_scores_per_page=sigma_scores_per_page,
                tau_hat=gaussian_stats["tau_hat"],
                sigma_global=sigma_global,
                page_size=self.page_size,
                alpha=alpha,
                safety_margin_z=safety_margin_z,
                max_quantile=max_quantile,
                threshold_excess_margin_fraction=threshold_excess_margin_fraction,
                pages_for_stats=pages_for_stats,
                num_pages=num_pages,
            )
        )
        gaussian_stats["tau_floor"] = tau_floor.contiguous()
        self.last_num_selected_per_head = num_selected_per_head
        return selected_page_indices, num_selected_per_head

