"""Routing from (alpha, head_dim) to the decode kernels.

* alpha in {1.5, 2.0} with head_dim in {32, 64, 128}: the one-K-pass kernels
  (``entmax_decode_paged_native`` for exact, ``entmax_decode_selected_pages``
  over a selection).
* Anything else (e.g. alpha = 4/3 or 1.25): the generic-alpha AdaDecode
  stages, over all pages for exact decode.
"""

import torch

from entmaxkv.kernels.adadecode_paged import (
    AdaDecodePagedWorkspace,
    entmax_decode_selected_pages_generic,
)
from entmaxkv.kernels.decode import (
    ExactDecodeWorkspace,
    entmax_decode_paged_native,
)
from entmaxkv.kernels.selected_decode import (
    SelectedDecodeWorkspace,
    entmax_decode_selected_pages,
)

GENERIC_DEFAULT_NITER = 10


def uses_fast_kernels(alpha: float, head_dim: int) -> bool:
    return alpha in (1.5, 2.0) and head_dim in (32, 64, 128)


def workspace(cache, name, factory):
    ws = cache.workspaces.get(name)
    if ws is None:
        ws = cache.workspaces[name] = factory()
    return ws


def decode_views(q: torch.Tensor, out: torch.Tensor):
    """[B, H, 1, D] public tensors -> [B, H, D] kernel views (no copies)."""
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError("q must be [B, H, 1, D] (single-token decode)")
    if out.shape != q.shape:
        raise ValueError("out must have the same shape as q")
    q3, out3 = q[:, :, 0], out[:, :, 0]
    if q3.stride(2) != 1 or out3.stride(2) != 1:
        raise ValueError("q and out must have a unit last-dimension stride")
    return q3, out3


def query_slopes(alibi_slopes, nheads: int, device) -> torch.Tensor:
    """ALiBi slopes by QUERY head, fp32; zeros when ALiBi is disabled."""
    if alibi_slopes is None:
        return torch.zeros((nheads,), device=device, dtype=torch.float32)
    slopes = alibi_slopes.to(device=device, dtype=torch.float32).reshape(-1)
    if slopes.shape != (nheads,):
        raise ValueError(
            f"alibi_slopes must hold one slope per QUERY head ({nheads})")
    return slopes.contiguous()


def decode_on_selection(q3, cache, pages, sel_lens, alpha, slopes, niter,
                        out3):
    """Exact entmax over a selected page set."""
    kv, table, lens = cache.kv, cache.block_table, cache.seq_lens
    if uses_fast_kernels(alpha, q3.shape[-1]):
        return entmax_decode_selected_pages(
            q3, kv, table, pages, sel_lens, lens, alpha, slopes, niter=niter,
            out=out3,
            workspace=workspace(cache, "selected", SelectedDecodeWorkspace))
    return entmax_decode_selected_pages_generic(
        q3, kv, table, pages, sel_lens, lens, alpha, slopes,
        niter=GENERIC_DEFAULT_NITER if niter is None else niter, out=out3,
        workspace=workspace(cache, "generic", AdaDecodePagedWorkspace))


def decode_exact(q3, cache, alpha, slopes, niter, out3):
    """Exact entmax over every cached token."""
    if uses_fast_kernels(alpha, q3.shape[-1]):
        return entmax_decode_paged_native(
            q3, cache.kv, cache.block_table, cache.seq_lens, alpha, slopes,
            niter=niter, out=out3,
            workspace=workspace(cache, "exact", ExactDecodeWorkspace))
    pages, sel_lens = _all_pages(cache, q3.shape[1])
    return entmax_decode_selected_pages_generic(
        q3, cache.kv, cache.block_table, pages, sel_lens, cache.seq_lens,
        alpha, slopes, niter=GENERIC_DEFAULT_NITER if niter is None else niter,
        out=out3,
        workspace=workspace(cache, "generic", AdaDecodePagedWorkspace))


def _all_pages(cache, nheads):
    """Selection covering every page of every row (tail last, -1 padded)."""
    ps = cache.page_size
    ws = workspace(cache, "all_pages", dict)
    key = (cache.batch, nheads, cache.max_pages)
    if ws.get("key") != key:
        ws.clear()
        ws["key"] = key
        ws["arange"] = torch.arange(cache.max_pages, device=cache.kv.device,
                                    dtype=torch.int32)
        ws["pages"] = torch.empty((cache.batch, nheads, cache.max_pages),
                                  device=cache.kv.device, dtype=torch.int32)
        ws["sel_lens"] = torch.empty((cache.batch, nheads),
                                     device=cache.kv.device, dtype=torch.int32)
    used = (cache.seq_lens + ps - 1) // ps
    ar = ws["arange"]
    ws["pages"].copy_(torch.where(ar[None, :] < used[:, None], ar, -1)[:, None])
    ws["sel_lens"].copy_(cache.seq_lens[:, None])
    return ws["pages"], ws["sel_lens"]
