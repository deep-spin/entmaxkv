import torch
from typing import Optional

from entmaxkv.decoding import (
    decode_exact,
    decode_on_selection,
    decode_views,
    query_slopes,
    workspace,
)
from entmaxkv.kv_cache import PagedKVCache
from entmaxkv.selectors import TopKPageSelector, topk_is_worthwhile


def sparse_attention_decode_paged(
    q: torch.Tensor,
    kv_cache: PagedKVCache,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    out: torch.Tensor,
    topk_pages: int,
    alibi_slopes: Optional[torch.Tensor] = None,
    alpha: float = 1.5,
    niter: Optional[int] = None,
    append_cache: bool = True,
    min_context: int = 0,
) -> torch.Tensor:
    """
    Decode-phase entmax attention over the top-k pages per query head.

    Pages are ranked by a Quest-style upper bound on their best score
    (per-page key extrema, 1/sqrt(d)-scaled, ALiBi at the page's best logical
    position). Selection is per QUERY head; K/V stay compact under GQA.

    Args:
        q, out: [B, H_q, 1, D].
        k_new, v_new: [B, H_kv, 1, D], appended first when ``append_cache``.
        topk_pages: TOTAL pages attended per query head, including the
            always-attended tail page (>= 2).
        alibi_slopes: [H_q] slopes by query head, or None.
        niter: tau refinements (default: 3 for alpha=1.5, 4 for alpha=2,
            10 on the generic-alpha path).
        min_context: rows shorter than this decode exactly.

    The whole batch decodes exactly when any row is below ``min_context`` or
    has no more pages than the budget: sparse selection would not save work.
    The result is exact with respect to the selected pages.
    """
    if topk_pages < 2:
        raise ValueError("topk_pages is tail inclusive and must be >= 2")
    if min_context < 0:
        raise ValueError("min_context must be >= 0")
    if append_cache:
        kv_cache.append(k_new, v_new)
    q3, out3 = decode_views(q, out)
    slopes = query_slopes(alibi_slopes, q.shape[1], q.device)
    ps = kv_cache.page_size
    if not all(topk_is_worthwhile(n, topk_pages, min_context, ps)
               for n in kv_cache.seq_lens_host):
        decode_exact(q3, kv_cache, alpha, slopes, niter, out3)
        return out
    if kv_cache.k_min is None:
        raise ValueError("top-k decoding needs a cache built with 'topk' stats")
    selector = workspace(kv_cache, ("topk", topk_pages),
                         lambda: TopKPageSelector(topk_pages, ps))
    sel = selector.select(q3, kv_cache.k_min, kv_cache.k_max,
                          kv_cache.block_table, kv_cache.seq_lens, slopes,
                          live_pages=kv_cache.live_pages)
    decode_on_selection(q3, kv_cache, sel.pages, sel.sel_lens, alpha, slopes,
                        niter, out3)
    return out
