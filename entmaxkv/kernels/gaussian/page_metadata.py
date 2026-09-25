"""Physical-page key moments for Gaussian-aware selection.

The metadata follows physical cache blocks, just like the top-k extrema store.
Touched pages are recomputed after each cache write.  A partial page may contain
unwritten slots, but selectors never consume statistics for the mandatory tail.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _refresh_touched_page_moments(
    SLOT_MAPPING, K_CACHE, K_MEAN, K_STD,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_mblock: tl.constexpr, stride_mh: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    tok = tl.program_id(0)
    kv_head = tl.program_id(1)
    slot = tl.load(SLOT_MAPPING + tok).to(tl.int64)
    prev = tl.load(SLOT_MAPPING + tok - 1, mask=tok > 0,
                   other=-1).to(tl.int64)
    if (slot >= 0) & ((tok == 0) | (slot // BLOCK_SIZE !=
                                    prev // BLOCK_SIZE)):
        block = slot // BLOCK_SIZE
        offs_d = tl.arange(0, HEAD_DIM)
        offs_t = tl.arange(0, BLOCK_SIZE)
        ptr = (K_CACHE + block * stride_kblock
               + offs_t[:, None] * stride_ktok
               + kv_head * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(ptr).to(tl.float32)
        mean = tl.sum(k, axis=0) / BLOCK_SIZE
        variance = tl.maximum(tl.sum(k * k, axis=0) / BLOCK_SIZE
                              - mean * mean, 0.0)
        base = K_MEAN + block * stride_mblock + kv_head * stride_mh + offs_d
        tl.store(base, mean)
        base = K_STD + block * stride_mblock + kv_head * stride_mh + offs_d
        tl.store(base, tl.sqrt(variance))


def allocate_gaussian_page_metadata(kv_cache: torch.Tensor):
    """Allocate fp32 mean/std arrays shaped ``[blocks, H_kv, D]``."""
    _, num_blocks, _, n_kv_heads, dim = kv_cache.shape
    shape = (num_blocks, n_kv_heads, dim)
    opts = dict(device=kv_cache.device, dtype=torch.float32)
    return torch.zeros(shape, **opts), torch.zeros(shape, **opts)


def update_gaussian_page_metadata(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_mean: torch.Tensor,
    k_std: torch.Tensor,
) -> None:
    """Refresh moments for every physical block touched by this cache write."""
    if slot_mapping.numel() == 0:
        return
    k_cache = kv_cache[0]
    _, block_size, n_kv_heads, dim = k_cache.shape
    _refresh_touched_page_moments[(slot_mapping.numel(), n_kv_heads)](
        slot_mapping, k_cache, k_mean, k_std,
        stride_kblock=k_cache.stride(0), stride_ktok=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_mblock=k_mean.stride(0), stride_mh=k_mean.stride(1),
        HEAD_DIM=dim, BLOCK_SIZE=block_size, num_warps=1)
