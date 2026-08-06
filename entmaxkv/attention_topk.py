import torch
from typing import Optional

from entmaxkv.kernels.adadecode_paged import sparse_attention_decode_paged as _paged_kernel
from entmaxkv.kv_cache import PagedKVCache


def sparse_attention_decode_paged(
    q: torch.Tensor,
    kv_cache: PagedKVCache,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    out: torch.Tensor,
    token_budget: int,
    cache_seqlens: torch.Tensor,
    q_seqlens: torch.Tensor,
    alibi_slopes: torch.Tensor = None,
    alpha: float = 1.5,
    splits: int = 32,
    niter: int = 10,
    append_cache: bool = True,
    use_triton_criticality: bool = True,
    workspace: dict = None,
):
    """
    Decode-phase attention with top-k page selection.

    Page-native variant: K/V are never gathered into a contiguous buffer.
    The kernel receives PAGE_INDICES [B, H, n_pages] and accesses the full
    cache directly, computing orig_row = page_id * page_size + in_page_offset
    on the fly. 
    """
    batch, num_heads, _, head_dim = q.shape
    num_kv_heads = kv_cache.k_cache.shape[1]

    if append_cache:
        kv_cache.append(k_new, v_new)

    seq_len    = kv_cache.k_cache.shape[2]
    page_size  = kv_cache.page_size
    total_pages = (seq_len + page_size - 1) // page_size
    num_pages_to_select = min(token_budget // page_size, total_pages - 1)

    last_page_idx = total_pages - 1

    page_scores = kv_cache.estimate_page_criticality(
        q,
        use_triton=use_triton_criticality,
        alibi_slopes=alibi_slopes,
        q_pos=seq_len - 1,
        exclude_last_page=True,
    )

    _, top_k_page_indices = torch.topk(page_scores, num_pages_to_select, dim=2)
    top_k_page_indices = top_k_page_indices.to(torch.int32)

    # Append last page as the final entry: [B, H, n_selected+1].
    # Re-use a cached buffer when kv_cache supports it (OptimizedKVCache),
    # otherwise fall back to a fresh allocation.
    if hasattr(kv_cache, 'get_page_indices_buf'):
        page_indices = kv_cache.get_page_indices_buf(
            batch, num_heads, num_pages_to_select, last_page_idx
        )
    else:
        page_indices = torch.empty(
            (batch, num_heads, num_pages_to_select + 1),
            dtype=torch.int32,
            device=q.device,
        )
        page_indices[:, :, num_pages_to_select].fill_(last_page_idx)
    page_indices[:, :, :num_pages_to_select].copy_(top_k_page_indices)

    if cache_seqlens.dim() == 1:
        cache_seqlens = cache_seqlens[:, None].expand(batch, num_kv_heads)
    else:
        assert cache_seqlens.shape == (batch, num_kv_heads), "cache_seqlens must be [B, H_kv]"

    if workspace is None and hasattr(kv_cache, "get_paged_decode_workspace"):
        workspace = kv_cache.get_paged_decode_workspace(
            batch, num_heads, head_dim, splits, 16, q.dtype, q.device
        )

    _paged_kernel(
        q=q,
        k_cache=kv_cache.k_cache,
        v_cache=kv_cache.v_cache,
        out=out,
        cache_seqlens=cache_seqlens,
        page_indices=page_indices,
        page_size=page_size,
        alibi_slopes=alibi_slopes,
        is_causal=True,
        alpha=alpha,
        niter=niter,
        max_splits=splits,
        q_seqlens=q_seqlens,
        workspace=workspace,
    )

    return out
