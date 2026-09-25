"""Paged KV cache in vLLM's block layout, with per-page key statistics.

Storage matches vLLM's attention backend:

    kv           [2, num_blocks, page_size, H_kv, D]    (K = kv[0], V = kv[1])
    block_table  int32 [B, max_pages]                    logical -> physical
    seq_lens     int32 [B]                               ragged rows allowed

entmaxkv owns the whole pool, so the block table is the identity
``block_table[b, p] = b * max_pages + p``. Page statistics are indexed by
PHYSICAL block and refreshed in place, only for the blocks a write touches:

    k_min, k_max    [num_blocks, H_kv, D]  cache dtype   (top-k criticality)
    k_mean, k_std   [num_blocks, H_kv, D]  fp32          (Gaussian selection)

Appends write one slot per row in place. Capacity grows geometrically when a
row outgrows it; every kernel constant derived from strides changes at that
point, so pass ``max_seq_len`` to size the pool up front and avoid the
one-off recompiles.
"""

from typing import Iterable, Optional, Sequence

import torch

from entmaxkv.kernels.gaussian.page_metadata import (
    allocate_gaussian_page_metadata,
    update_gaussian_page_metadata,
)
from entmaxkv.kernels.page_metadata import (
    allocate_page_metadata,
    update_completed_page_metadata,
)

STATS = ("topk", "gaussian")
# Extra pages allocated past the prefill when no max_seq_len is given.
DEFAULT_SLACK_PAGES = 64


class PagedKVCache:
    """KV cache with paged key statistics for query-aware page selection."""

    def __init__(self, page_size: int = 16,
                 stats: Iterable[str] = STATS,
                 max_seq_len: Optional[int] = None):
        if page_size < 1 or page_size & (page_size - 1):
            raise ValueError("page_size must be a positive power of two")
        stats = tuple(stats)
        unknown = set(stats) - set(STATS)
        if unknown:
            raise ValueError(f"unknown stats {sorted(unknown)}; "
                             f"expected a subset of {STATS}")
        self.page_size = page_size
        self.stats = stats
        self.requested_max_seq_len = max_seq_len
        self.kv: Optional[torch.Tensor] = None
        self.block_table: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.k_min = self.k_max = None
        self.k_mean = self.k_std = None
        self._seq_lens_host: list[int] = []
        self._row_base: Optional[torch.Tensor] = None
        # Decode scratch owned by this cache (selectors, kernel workspaces).
        self.workspaces: dict = {}

    # ---- geometry -----------------------------------------------------------

    @property
    def batch(self) -> int:
        return self.block_table.shape[0]

    @property
    def max_pages(self) -> int:
        return self.block_table.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.kv.shape[3]

    @property
    def head_dim(self) -> int:
        return self.kv.shape[4]

    @property
    def seq_lens_host(self) -> tuple[int, ...]:
        """Row lengths, tracked on the host so decode never syncs for them."""
        return tuple(self._seq_lens_host)

    @property
    def max_seq_len(self) -> int:
        return max(self._seq_lens_host)

    @property
    def live_pages(self) -> int:
        """Block-table columns any row currently uses."""
        return -(-self.max_seq_len // self.page_size)

    @property
    def k_cache(self) -> torch.Tensor:
        """[num_blocks, page_size, H_kv, D] view of K."""
        return self.kv[0]

    @property
    def v_cache(self) -> torch.Tensor:
        return self.kv[1]

    # ---- allocation ---------------------------------------------------------

    def _allocate(self, batch, heads, dim, dtype, device, max_pages):
        ps = self.page_size
        self.kv = torch.zeros((2, batch * max_pages, ps, heads, dim),
                              dtype=dtype, device=device)
        rows = torch.arange(batch, device=device, dtype=torch.int32)
        cols = torch.arange(max_pages, device=device, dtype=torch.int32)
        self.block_table = (rows[:, None] * max_pages + cols[None, :]
                            ).contiguous()
        self._row_base = rows.to(torch.int64) * max_pages
        if "topk" in self.stats:
            self.k_min, self.k_max = allocate_page_metadata(self.kv)
        if "gaussian" in self.stats:
            self.k_mean, self.k_std = allocate_gaussian_page_metadata(self.kv)

    def _grow(self, needed_pages: int) -> None:
        """Re-home every row into a larger pool (one O(S) copy, amortised)."""
        old_pages = self.max_pages
        new_pages = max(needed_pages, old_pages + old_pages // 2)
        old = {"kv": self.kv, "k_min": self.k_min, "k_max": self.k_max,
               "k_mean": self.k_mean, "k_std": self.k_std}
        batch, heads, dim = self.batch, self.num_kv_heads, self.head_dim
        self._allocate(batch, heads, dim, self.kv.dtype, self.kv.device,
                       new_pages)
        ps = self.page_size
        self.kv.view(2, batch, new_pages, ps, heads, dim)[:, :, :old_pages] \
            .copy_(old["kv"].view(2, batch, old_pages, ps, heads, dim))
        for name in ("k_min", "k_max", "k_mean", "k_std"):
            if old[name] is not None:
                getattr(self, name).view(batch, new_pages, heads, dim)[
                    :, :old_pages].copy_(
                        old[name].view(batch, old_pages, heads, dim))
        self.workspaces.clear()

    def _refresh_stats(self, slot_mapping: torch.Tensor) -> None:
        if self.k_min is not None:
            update_completed_page_metadata(self.kv, slot_mapping,
                                           self.k_min, self.k_max)
        if self.k_mean is not None:
            update_gaussian_page_metadata(self.kv, slot_mapping,
                                          self.k_mean, self.k_std)

    # ---- writes -------------------------------------------------------------

    def initialize(self, k: torch.Tensor, v: torch.Tensor,
                   seq_lens: Optional[Sequence[int]] = None) -> None:
        """Load a prefill.

        Args:
            k, v: [B, H_kv, S, D].
            seq_lens: optional per-row lengths (<= S) for ragged batches;
                positions past a row's length are ignored.
        """
        if k.ndim != 4 or k.shape != v.shape:
            raise ValueError("k and v must both be [B, H_kv, S, D]")
        batch, heads, seq, dim = k.shape
        if seq_lens is None:
            lens = [seq] * batch
        else:
            lens = [int(x) for x in (seq_lens.tolist()
                                     if torch.is_tensor(seq_lens)
                                     else seq_lens)]
        if len(lens) != batch or min(lens) < 1 or max(lens) > seq:
            raise ValueError("seq_lens must hold B lengths in [1, S]")
        ps = self.page_size
        pages = -(-seq // ps)
        # Room for decode: at least one more token, plus slack or the
        # requested ceiling.
        needed = -(-(max(lens) + 1) // ps)
        if self.requested_max_seq_len is not None:
            capacity = max(needed, -(-self.requested_max_seq_len // ps))
        else:
            capacity = needed + DEFAULT_SLACK_PAGES
        capacity = max(capacity, pages)
        self._allocate(batch, heads, dim, k.dtype, k.device, capacity)
        self.workspaces.clear()

        positions = torch.arange(pages * ps, device=k.device)
        lens_dev = torch.tensor(lens, device=k.device, dtype=torch.int64)
        valid = (positions[None, :] < lens_dev[:, None])       # [B, pages*ps]
        view = self.kv.view(2, batch, capacity, ps, heads, dim)
        for idx, src in enumerate((k, v)):
            padded = src.new_zeros((batch, pages * ps, heads, dim))
            padded[:, :seq] = src.transpose(1, 2)
            padded.mul_(valid[:, :, None, None].to(src.dtype))
            view[idx, :, :pages].copy_(
                padded.view(batch, pages, ps, heads, dim))

        self.seq_lens = lens_dev.to(torch.int32)
        self._seq_lens_host = lens
        # One representative slot per touched block: the refresh kernels
        # reduce the whole block regardless of which slot named it.
        page_ids = torch.arange(pages, device=k.device)
        touched = page_ids[None, :] < ((lens_dev + ps - 1) // ps)[:, None]
        blocks = (self._row_base[:, None] + page_ids[None, :])[touched]
        self._refresh_stats((blocks * ps).contiguous())

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append one token per row. k_new, v_new: [B, H_kv, 1, D]."""
        if self.kv is None:
            raise RuntimeError("initialize() the cache before appending")
        expected = (self.batch, self.num_kv_heads, 1, self.head_dim)
        if tuple(k_new.shape) != expected or tuple(v_new.shape) != expected:
            raise ValueError(f"k_new and v_new must be {expected}")
        needed = -(-(self.max_seq_len + 1) // self.page_size)
        if needed > self.max_pages:
            self._grow(needed)
        ps = self.page_size
        lens = self.seq_lens.to(torch.int64)
        slots = (self._row_base + lens // ps) * ps + lens % ps
        heads, dim = self.num_kv_heads, self.head_dim
        self.kv[0].view(-1, heads, dim).index_copy_(0, slots, k_new[:, :, 0])
        self.kv[1].view(-1, heads, dim).index_copy_(0, slots, v_new[:, :, 0])
        self._refresh_stats(slots)
        self.seq_lens += 1
        self._seq_lens_host = [n + 1 for n in self._seq_lens_host]

    # ---- inspection ---------------------------------------------------------

    def dense(self) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, H_kv, max_seq_len, D] copies of K and V (zeros past a row's
        length). For tests and references; never on the decode path."""
        batch, heads, dim, ps = (self.batch, self.num_kv_heads,
                                 self.head_dim, self.page_size)
        seq = self.max_seq_len
        pages = -(-seq // ps)
        view = self.kv.view(2, batch, self.max_pages, ps, heads, dim)
        dense = view[:, :, :pages].reshape(2, batch, pages * ps, heads, dim)
        dense = dense[:, :, :seq].transpose(2, 3).clone()
        positions = torch.arange(seq, device=dense.device)
        valid = positions[None, :] < self.seq_lens.to(torch.int64)[:, None]
        dense.mul_(valid[None, :, None, :, None].to(dense.dtype))
        return dense[0], dense[1]
