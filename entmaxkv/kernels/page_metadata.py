"""Per-page key extrema for Quest-style page criticality.

For every *complete* physical KV block we keep ``k_min``/``k_max`` over its
BLOCK_SIZE cached keys. These bound the best score a query can achieve inside
the page, which is what the criticality estimator needs.

Metadata is indexed by PHYSICAL block id, so it follows the block table: the
same kernels serve entmaxkv's identity table and vLLM's shuffled one, and a
recycled block's stale statistics are overwritten the moment it is written
again.

Every touched block gets statistics, including a partially filled tail. A tail's
extrema fold in slots that have not been written yet, but a tail is never scored
(the decoder force-includes it), and by the time a page is historical all of its
slots hold real keys.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _refresh_touched_page_stats(
    SLOT_MAPPING, K_CACHE, K_MIN, K_MAX,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_mblock: tl.constexpr, stride_mh: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """One program per (token, kv_head), reducing each contiguous block once.

    Draft KV can complete a block and later be rejected.  The correcting
    target write may then land in any slot, so restricting refreshes to the
    last slot leaves stale extrema for an otherwise complete historical page.
    Packed writes place a request's tokens in position order, so adjacent slots
    in the same block are duplicates. Only their first program performs the
    reduction. Non-adjacent aliases remain safe: they store identical results.
    """
    tok = tl.program_id(0)
    kv_head = tl.program_id(1)
    slot = tl.load(SLOT_MAPPING + tok).to(tl.int64)
    prev_slot = tl.load(SLOT_MAPPING + tok - 1, mask=tok > 0,
                        other=-1).to(tl.int64)
    # reshape_and_cache_flash skips padding tokens by marking the slot -1.
    if (slot >= 0) & ((tok == 0) | (slot // BLOCK_SIZE !=
                                    prev_slot // BLOCK_SIZE)):
        block = slot // BLOCK_SIZE
        offs_d = tl.arange(0, HEAD_DIM)
        offs_t = tl.arange(0, BLOCK_SIZE)
        k_ptr = (K_CACHE + block * stride_kblock
                 + offs_t[:, None] * stride_ktok
                 + kv_head * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptr)
        base = K_MIN + block * stride_mblock + kv_head * stride_mh + offs_d
        tl.store(base, tl.min(k, axis=0))
        base = K_MAX + block * stride_mblock + kv_head * stride_mh + offs_d
        tl.store(base, tl.max(k, axis=0))


def update_completed_page_metadata(
    kv_cache: torch.Tensor,      # (2, num_blocks, block_size, H_kv, D)
    slot_mapping: torch.Tensor,  # (num_tokens,) int64, flat slot ids
    k_min: torch.Tensor,         # (num_blocks, H_kv, D)
    k_max: torch.Tensor,
) -> None:
    """Refresh k_min/k_max for every physical block touched by this write.

    Call immediately after ``reshape_and_cache_flash``.  Partial-tail metadata
    may include unwritten slots, but tails are force-included rather than
    scored; once historical, all slots have been written.
    """
    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return
    k_cache = kv_cache[0]
    _, block_size, n_kv_heads, dim = k_cache.shape
    _refresh_touched_page_stats[(num_tokens, n_kv_heads)](
        slot_mapping, k_cache, k_min, k_max,
        stride_kblock=k_cache.stride(0), stride_ktok=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_mblock=k_min.stride(0), stride_mh=k_min.stride(1),
        HEAD_DIM=dim, BLOCK_SIZE=block_size,
        num_warps=1,
    )


def allocate_page_metadata(kv_cache: torch.Tensor):
    """Allocate the (k_min, k_max) store for one layer's KV cache.

    Costs 2/BLOCK_SIZE of the K cache, i.e. ~6.25% of K+V at block_size 16.
    Initialised so an unwritten block can never look attractive: an empty page
    scores -inf-ish rather than winning a top-k slot.
    """
    _, num_blocks, _, n_kv_heads, dim = kv_cache.shape
    shape = (num_blocks, n_kv_heads, dim)
    opts = dict(device=kv_cache.device, dtype=kv_cache.dtype)
    return torch.zeros(shape, **opts), torch.zeros(shape, **opts)
