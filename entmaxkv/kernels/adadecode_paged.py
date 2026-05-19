import math
import torch
import triton
import triton.language as tl

# Re-use shared helpers from the original file (tau init, halley update, etc.)
from entmaxkv.kernels.adadecode import (
    _decode_stage2_reduce_max,
    _decode_stage4_reduce_hist,
    _decode_stage5_init,
    _decode_stage5b_halley_update,
    _decode_stage6b_reduce_partials,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compact index → original cache row
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _compact_to_orig(c_idx, page_indices_base, PAGE_SIZE: tl.constexpr):
    """
    Map a compacted sequential index to the original K/V cache row index.

    page_indices_base : pointer to PAGE_INDICES[b, h, 0]  (int32)
    PAGE_SIZE         : compile-time page size (tl.constexpr)
    """
    page_rank   = c_idx // PAGE_SIZE
    in_page_off = c_idx %  PAGE_SIZE
    orig_page   = tl.load(page_indices_base + page_rank).to(tl.int32)
    return orig_page * PAGE_SIZE + in_page_off


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – per-split local maxima
# grid = (B, H, MAX_SPLITS)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['H_DIM', 'USE_ALIBI', 'PAGE_SIZE'],
)
@triton.jit
def _paged_stage1_local_max(
    Q, K_cache,
    MAX_VALS,
    PAGE_INDICES,
    Cache_seqlens,
    Q_Seqlens,
    ALIBI_SLOPES,
    ##
    alpha: tl.constexpr,
    sm_scale: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ##
    N_H: tl.constexpr,
    N_KVH: tl.constexpr,
    H_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ##
    stride_qh: tl.constexpr,
    stride_kz: tl.constexpr,   # batch stride in K_cache
    stride_kh: tl.constexpr,   # head stride in K_cache
    stride_csz: tl.constexpr,  # batch stride in Cache_seqlens
    stride_csh: tl.constexpr,  # kv-head stride in Cache_seqlens
    stride_ah: tl.constexpr,
    stride_piz: tl.constexpr,  # batch stride in PAGE_INDICES
    stride_pih: tl.constexpr,  # head stride in PAGE_INDICES
    ##
    BLOCK_N: tl.constexpr,
):
    _scalar = (alpha - 1) * sm_scale
    input_dtype = Q.dtype.element_ty

    off_z    = tl.program_id(0)
    off_h    = tl.program_id(1)
    split_id = tl.program_id(2)
    off_hz   = off_z * N_H + off_h
    kv_h     = off_h if N_H == N_KVH else off_h // (N_H // N_KVH)

    cache_seqlen     = tl.load(Cache_seqlens + off_z * stride_csz + kv_h * stride_csh).to(tl.int32)
    seqlen_per_split = tl.cdiv(cache_seqlen, MAX_SPLITS)
    split_start      = split_id * seqlen_per_split
    split_end        = tl.minimum(split_start + seqlen_per_split, cache_seqlen)
    if split_start >= cache_seqlen:
        tl.store(MAX_VALS + off_hz * MAX_SPLITS + split_id, -1.0e6)
        return

    offs_k = tl.arange(0, H_DIM)
    q      = tl.load(Q + off_hz * stride_qh + offs_k) * _scalar
    q      = q.to(input_dtype)

    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(ALIBI_SLOPES + off_h) * (alpha - 1)
    q_idx = tl.load(Q_Seqlens + off_z).to(tl.int32) - 1

    page_indices_base = PAGE_INDICES + off_z * stride_piz + kv_h * stride_pih
    k_head_base       = K_cache + off_z * stride_kz + kv_h * stride_kh

    offs_n = tl.arange(0, BLOCK_N)
    local_max    = -1.0e6
    valid_blocks = tl.cdiv(split_end - split_start, BLOCK_N)

    for c_block in range(valid_blocks):
        c_idxs = split_start + c_block * BLOCK_N + offs_n
        c_mask = c_idxs < split_end

        page_ranks   = c_idxs // PAGE_SIZE
        in_page_offs = c_idxs %  PAGE_SIZE
        orig_pages   = tl.load(page_indices_base + page_ranks, mask=c_mask, other=0).to(tl.int32)
        orig_rows    = orig_pages * PAGE_SIZE + in_page_offs   # actual cache row

        k_ptrs = k_head_base + orig_rows[:, None] * H_DIM + offs_k[None, :]
        k = tl.load(k_ptrs, mask=c_mask[:, None], other=0.0).to(input_dtype)

        qk = tl.sum(q[None, :] * k, axis=1)
        if USE_ALIBI:
            position_diff = -(q_idx - orig_rows)
            qk += alibi_slope * position_diff

        qk = tl.where(c_mask, qk, float("-inf"))
        local_max = tl.maximum(local_max, tl.max(qk))

    tl.store(MAX_VALS + off_hz * MAX_SPLITS + split_id, local_max)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – per-split histograms
# grid = (B, H, MAX_SPLITS)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['H_DIM', 'USE_ALIBI', 'PAGE_SIZE'],
)
@triton.jit
def _paged_stage3_build_hist(
    Q, K_cache,
    GLOBAL_MAXS,
    HIST_SPLIT,
    PAGE_INDICES,
    Cache_seqlens,
    Q_Seqlens,
    ALIBI_SLOPES,
    ##
    alpha: tl.constexpr,
    sm_scale: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ##
    N_H: tl.constexpr,
    N_KVH: tl.constexpr,
    H_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    BINS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ##
    stride_qh: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_csz: tl.constexpr,
    stride_csh: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_piz: tl.constexpr,
    stride_pih: tl.constexpr,
    ##
    BLOCK_N: tl.constexpr,
):
    _scalar = (alpha - 1) * sm_scale
    input_dtype = Q.dtype.element_ty

    off_z    = tl.program_id(0)
    off_h    = tl.program_id(1)
    split_id = tl.program_id(2)
    off_hz   = off_z * N_H + off_h
    kv_h     = off_h if N_H == N_KVH else off_h // (N_H // N_KVH)

    cache_seqlen     = tl.load(Cache_seqlens + off_z * stride_csz + kv_h * stride_csh).to(tl.int32)
    seqlen_per_split = tl.cdiv(cache_seqlen, MAX_SPLITS)
    split_start      = split_id * seqlen_per_split
    split_end        = tl.minimum(split_start + seqlen_per_split, cache_seqlen)
    if split_start >= cache_seqlen:
        bins = tl.arange(0, BINS)
        base = ((off_hz * MAX_SPLITS + split_id) * BINS)
        tl.store(HIST_SPLIT + base + bins, tl.zeros((BINS,), dtype=tl.int32))
        return

    offs_k = tl.arange(0, H_DIM)
    q      = tl.load(Q + off_hz * stride_qh + offs_k) * _scalar
    q      = q.to(input_dtype)

    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(ALIBI_SLOPES + off_h) * (alpha - 1)
    q_idx = tl.load(Q_Seqlens + off_z).to(tl.int32) - 1

    global_max = tl.load(GLOBAL_MAXS + off_hz)
    t0 = global_max - 1.0

    page_indices_base = PAGE_INDICES + off_z * stride_piz + kv_h * stride_pih
    k_head_base       = K_cache + off_z * stride_kz + kv_h * stride_kh

    hist   = tl.zeros((BINS,), dtype=tl.int32)
    bins   = tl.arange(0, BINS)
    offs_n = tl.arange(0, BLOCK_N)
    valid_blocks = tl.cdiv(split_end - split_start, BLOCK_N)

    for c_block in range(valid_blocks):
        c_idxs = split_start + c_block * BLOCK_N + offs_n
        c_mask = c_idxs < split_end

        page_ranks   = c_idxs // PAGE_SIZE
        in_page_offs = c_idxs %  PAGE_SIZE
        orig_pages   = tl.load(page_indices_base + page_ranks, mask=c_mask, other=0).to(tl.int32)
        orig_rows    = orig_pages * PAGE_SIZE + in_page_offs

        k_ptrs = k_head_base + orig_rows[:, None] * H_DIM + offs_k[None, :]
        k = tl.load(k_ptrs, mask=c_mask[:, None], other=0.0).to(input_dtype)

        qk = tl.sum(q[None, :] * k, axis=1)
        if USE_ALIBI:
            qk += alibi_slope * (-(q_idx - orig_rows))

        proj  = qk - t0
        b     = (proj * BINS).to(tl.int32)
        b     = tl.minimum(b, BINS - 1)
        valid = c_mask & (b >= 0)

        eq   = (b[:, None] == bins[None, :])
        cnts = tl.sum(eq & valid[:, None], axis=0).to(tl.int32)
        hist += cnts

    base = ((off_hz * MAX_SPLITS + split_id) * BINS)
    tl.store(HIST_SPLIT + base + bins, hist)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5a – Halley accumulate
# grid = (B, H, MAX_SPLITS)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['H_DIM', 'USE_ALIBI', 'PAGE_SIZE'],
)
@triton.jit
def _paged_stage5a_halley_accumulate(
    Q, K_cache,
    ACC0_SPLIT, ACC1_SPLIT, ACC2_SPLIT,
    TAUS,
    PAGE_INDICES,
    Cache_seqlens,
    Q_Seqlens,
    ALIBI_SLOPES,
    ##
    alpha: tl.constexpr,
    sm_scale: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ##
    N_H: tl.constexpr,
    N_KVH: tl.constexpr,
    H_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ##
    stride_qh: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_csz: tl.constexpr,
    stride_csh: tl.constexpr,
    stride_th: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_piz: tl.constexpr,
    stride_pih: tl.constexpr,
    ##
    BLOCK_N: tl.constexpr,
):
    _scalar = (alpha - 1) * sm_scale
    coeff_0 = 1 / (alpha - 1)
    coeff_1 = coeff_0 - 1
    coeff_2 = coeff_0 - 2
    input_dtype = Q.dtype.element_ty

    off_z    = tl.program_id(0)
    off_h    = tl.program_id(1)
    split_id = tl.program_id(2)
    off_hz   = off_z * N_H + off_h
    kv_h     = off_h if N_H == N_KVH else off_h // (N_H // N_KVH)

    cache_seqlen     = tl.load(Cache_seqlens + off_z * stride_csz + kv_h * stride_csh).to(tl.int32)
    seqlen_per_split = tl.cdiv(cache_seqlen, MAX_SPLITS)
    split_start      = split_id * seqlen_per_split
    split_end        = tl.minimum(split_start + seqlen_per_split, cache_seqlen)
    if split_start >= cache_seqlen:
        base = off_hz * MAX_SPLITS + split_id
        tl.store(ACC0_SPLIT + base, 0.0)
        tl.store(ACC1_SPLIT + base, 0.0)
        tl.store(ACC2_SPLIT + base, 0.0)
        return

    offs_k = tl.arange(0, H_DIM)
    q      = tl.load(Q + off_hz * stride_qh + offs_k) * _scalar
    q      = q.to(input_dtype)

    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(ALIBI_SLOPES + off_h) * (alpha - 1)
    q_idx = tl.load(Q_Seqlens + off_z).to(tl.int32) - 1

    t = tl.load(TAUS + off_hz * stride_th)

    page_indices_base = PAGE_INDICES + off_z * stride_piz + kv_h * stride_pih
    k_head_base       = K_cache + off_z * stride_kz + kv_h * stride_kh

    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0

    offs_n = tl.arange(0, BLOCK_N)
    valid_blocks = tl.cdiv(split_end - split_start, BLOCK_N)

    for c_block in range(valid_blocks):
        c_idxs = split_start + c_block * BLOCK_N + offs_n
        c_mask = c_idxs < split_end

        page_ranks   = c_idxs // PAGE_SIZE
        in_page_offs = c_idxs %  PAGE_SIZE
        orig_pages   = tl.load(page_indices_base + page_ranks, mask=c_mask, other=0).to(tl.int32)
        orig_rows    = orig_pages * PAGE_SIZE + in_page_offs

        k_ptrs = k_head_base + orig_rows[:, None] * H_DIM + offs_k[None, :]
        k = tl.load(k_ptrs, mask=c_mask[:, None], other=0.0).to(input_dtype)

        qk = tl.sum(q[None, :] * k, axis=1)
        if USE_ALIBI:
            qk += alibi_slope * (-(q_idx - orig_rows))

        qk_mask   = (qk > t) & c_mask
        qk_mask_f = qk_mask.to(tl.float32)
        qk_act    = (qk - t) * qk_mask_f

        if alpha == 2.0:
            acc0 += tl.sum(qk_act)
            acc1 += tl.sum(qk_mask_f)
        elif alpha == 1.5:
            acc0 += tl.sum(qk_act * qk_act)
            acc1 += tl.sum(qk_act)
            acc2 += tl.sum(qk_mask_f)
        else:
            log2_act = tl.log2(qk_act)
            acc0 += tl.sum(tl.where(qk_mask, tl.exp2(log2_act * coeff_0), 0.0))
            acc1 += tl.sum(tl.where(qk_mask, tl.exp2(log2_act * coeff_1), 0.0))
            acc2 += tl.sum(tl.where(qk_mask, tl.exp2(log2_act * coeff_2), 0.0))

    base = off_hz * MAX_SPLITS + split_id
    tl.store(ACC0_SPLIT + base, acc0)
    tl.store(ACC1_SPLIT + base, acc1)
    tl.store(ACC2_SPLIT + base, acc2)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6a – partial outputs per split
# grid = (B, H, MAX_SPLITS)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['H_DIM', 'USE_ALIBI', 'PAGE_SIZE'],
)
@triton.jit
def _paged_stage6a_partial_out(
    Q, K_cache, V_cache,
    PARTIAL_OUT,
    TAUS,
    PAGE_INDICES,
    Cache_seqlens,
    Q_Seqlens,
    ALIBI_SLOPES,
    ##
    alpha: tl.constexpr,
    sm_scale: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ##
    N_H: tl.constexpr,
    N_KVH: tl.constexpr,
    H_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ##
    stride_qh: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_csz: tl.constexpr,
    stride_csh: tl.constexpr,
    stride_th: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_piz: tl.constexpr,
    stride_pih: tl.constexpr,
    ##
    BLOCK_N: tl.constexpr,
):
    _scalar = (alpha - 1) * sm_scale
    coeff_0 = 1 / (alpha - 1)
    input_dtype = Q.dtype.element_ty

    off_z    = tl.program_id(0)
    off_h    = tl.program_id(1)
    split_id = tl.program_id(2)
    off_hz   = off_z * N_H + off_h
    kv_h     = off_h if N_H == N_KVH else off_h // (N_H // N_KVH)

    cache_seqlen     = tl.load(Cache_seqlens + off_z * stride_csz + kv_h * stride_csh).to(tl.int32)
    q_idx            = tl.load(Q_Seqlens + off_z).to(tl.int32) - 1
    seqlen_per_split = tl.cdiv(cache_seqlen, MAX_SPLITS)
    split_start      = split_id * seqlen_per_split
    split_end        = tl.minimum(split_start + seqlen_per_split, cache_seqlen)
    if split_start >= cache_seqlen:
        return

    offs_k = tl.arange(0, H_DIM)
    q      = tl.load(Q + off_hz * stride_qh + offs_k) * _scalar
    q      = q.to(input_dtype)

    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(ALIBI_SLOPES + off_h) * (alpha - 1)

    t = tl.load(TAUS + off_hz * stride_th)

    page_indices_base = PAGE_INDICES + off_z * stride_piz + kv_h * stride_pih
    k_head_base       = K_cache + off_z * stride_kz + kv_h * stride_kh
    v_head_base       = V_cache + off_z * stride_vz + kv_h * stride_vh

    acc = tl.zeros([H_DIM], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    valid_blocks = tl.cdiv(split_end - split_start, BLOCK_N)

    for c_block in range(valid_blocks):
        c_idxs = split_start + c_block * BLOCK_N + offs_n
        c_mask = c_idxs < split_end

        page_ranks   = c_idxs // PAGE_SIZE
        in_page_offs = c_idxs %  PAGE_SIZE
        orig_pages   = tl.load(page_indices_base + page_ranks, mask=c_mask, other=0).to(tl.int32)
        orig_rows    = orig_pages * PAGE_SIZE + in_page_offs

        k_ptrs = k_head_base + orig_rows[:, None] * H_DIM + offs_k[None, :]
        k = tl.load(k_ptrs, mask=c_mask[:, None], other=0.0).to(input_dtype)

        qk = tl.sum(q[None, :] * k, axis=1)
        if USE_ALIBI:
            qk += alibi_slope * (-(q_idx - orig_rows))

        qk_mask = (qk > t) & c_mask

        has_nonzero = tl.sum(qk_mask.to(tl.int32)) > 0
        if has_nonzero:
            v_ptrs = v_head_base + orig_rows[:, None] * H_DIM + offs_k[None, :]
            v = tl.load(v_ptrs, mask=c_mask[:, None], other=0.0).to(input_dtype)

            qk_act    = qk - t
            qk_mask_f = qk_mask.to(tl.float32)

            if alpha == 2.0:
                qk_proj = qk_act * qk_mask_f
            elif alpha == 1.5:
                qk_proj = qk_act * qk_act * qk_mask_f
            else:
                qk_proj = tl.where(qk_mask, tl.exp2(tl.log2(qk_act) * coeff_0), 0.0)

            acc += tl.sum(v * qk_proj[:, None], axis=0)

    partial_out_base = PARTIAL_OUT + (off_hz * MAX_SPLITS + split_id) * H_DIM
    tl.store(partial_out_base + offs_k, acc.to(input_dtype))


# ============================================================ #
# Orchestrator: paged sparse attention decode     
# ============================================================ #

def quest_sparse_attention_decode_paged(
    q: torch.Tensor,            # [B, H, 1, D]
    k_cache: torch.Tensor,      # [B, H_kv, S, D]  — full cache, never copied
    v_cache: torch.Tensor,      # [B, H_kv, S, D]
    out: torch.Tensor,          # [B, H, 1, D]
    cache_seqlens: torch.Tensor,  # [B, H_kv] int32 — compacted token count per KV head
    page_indices: torch.Tensor,   # [B, H_kv, n_pages] int32 — selected page ids
    page_size: int,
    alibi_slopes: torch.Tensor = None,
    is_causal: bool = True,
    alpha: float = 1.5,
    niter: int = 10,
    max_splits: int = 32,
    bins: int = 16,
    q_seqlens: torch.Tensor = None,
    taus_out: torch.Tensor = None,  # optional [B, H] buffer to capture converged tau
):
    """
    Page-native six-stage adadecode pipeline.

    PAGE_INDICES is keyed by KV head. Under GQA the kernels map query head
    h to KV head h // n_rep, so callers can pass raw KV-head caches and page
    tables without materializing repeated heads.
    cache_seqlens[b, kv_h] = n_selected_pages_for_head * page_size + last_page_size.
    """
    batch, nheads, _, dim = q.shape
    num_kv_heads = k_cache.shape[1]
    assert q.shape[2] == 1
    assert page_indices.dtype == torch.int32
    assert nheads % num_kv_heads == 0, "Query heads must be divisible by KV heads"
    assert v_cache.shape[:2] == (batch, num_kv_heads), "K/V cache head shapes must match"
    assert page_indices.shape[0] == batch, "page_indices batch mismatch"
    assert page_indices.shape[1] == num_kv_heads, "page_indices must be keyed by KV heads"
    if cache_seqlens.dim() == 1:
        cache_seqlens = cache_seqlens[:, None].expand(batch, num_kv_heads)
    assert cache_seqlens.shape == (batch, num_kv_heads), "cache_seqlens must be [B, H_kv]"

    sm_scale   = 1.0 / math.sqrt(dim)
    use_alibi  = alibi_slopes is not None
    stride_ah  = alibi_slopes.stride(0) if use_alibi and alibi_slopes.dim() > 0 else 0

    # q_seqlens holds the original full-sequence length for each batch element.
    # Used to compute actual query token position for ALiBi biases.
    # If None, fall back to cache_seqlens (no-op when ALiBi is disabled).
    if q_seqlens is None:
        q_seqlens = cache_seqlens

    # Intermediates
    max_vals   = torch.full((batch, nheads, max_splits), float('-inf'),
                            device=q.device, dtype=torch.float32)
    global_maxs = torch.empty((batch, nheads), device=q.device, dtype=torch.float32)

    hist_split  = torch.zeros((batch, nheads, max_splits, bins),
                              device=q.device, dtype=torch.int32)
    hist_global = torch.empty((batch, nheads, bins), device=q.device, dtype=torch.int32)

    partial_out = torch.zeros((batch, nheads, max_splits, dim),
                              device=q.device, dtype=q.dtype)
    taus   = torch.empty((batch, nheads), device=q.device, dtype=torch.float32)
    t_los  = torch.empty((batch, nheads), device=q.device, dtype=torch.float32)
    t_his  = torch.empty((batch, nheads), device=q.device, dtype=torch.float32)
    acc0_split = torch.empty((batch, nheads, max_splits), device=q.device, dtype=torch.float32)
    acc1_split = torch.empty((batch, nheads, max_splits), device=q.device, dtype=torch.float32)
    acc2_split = torch.empty((batch, nheads, max_splits), device=q.device, dtype=torch.float32)

    # Strides
    stride_qh  = q.stride(1)
    stride_kz  = k_cache.stride(0)
    stride_kh  = k_cache.stride(1)
    stride_vz  = v_cache.stride(0)
    stride_vh  = v_cache.stride(1)
    stride_th  = taus.stride(1)
    stride_oh  = out.stride(1)
    stride_csz = cache_seqlens.stride(0)
    stride_csh = cache_seqlens.stride(1)
    stride_piz = page_indices.stride(0)
    stride_pih = page_indices.stride(1)

    # ---------------- Stage 1 ---------------- #
    grid1 = (batch, nheads, max_splits)
    _paged_stage1_local_max[grid1](
        Q=q, K_cache=k_cache,
        MAX_VALS=max_vals,
        PAGE_INDICES=page_indices,
        Cache_seqlens=cache_seqlens,
        Q_Seqlens=q_seqlens,
        ALIBI_SLOPES=alibi_slopes,
        alpha=alpha, sm_scale=sm_scale,
        USE_ALIBI=use_alibi,
        N_H=nheads, N_KVH=num_kv_heads, H_DIM=dim, MAX_SPLITS=max_splits, PAGE_SIZE=page_size,
        stride_qh=stride_qh, stride_kz=stride_kz, stride_kh=stride_kh,
        stride_csz=stride_csz, stride_csh=stride_csh,
        stride_ah=stride_ah, stride_piz=stride_piz, stride_pih=stride_pih,
    )

    # ---------------- Stage 2 ---------------- #
    grid2 = (batch, nheads)
    _decode_stage2_reduce_max[grid2](
        MAX_VALS=max_vals, GLOBAL_MAXS=global_maxs,
        N_H=nheads, MAX_SPLITS=max_splits,
    )

    # ---------------- Stage 3 ---------------- #
    grid3 = (batch, nheads, max_splits)
    _paged_stage3_build_hist[grid3](
        Q=q, K_cache=k_cache,
        GLOBAL_MAXS=global_maxs,
        HIST_SPLIT=hist_split,
        PAGE_INDICES=page_indices,
        Cache_seqlens=cache_seqlens,
        Q_Seqlens=q_seqlens,
        ALIBI_SLOPES=alibi_slopes,
        alpha=alpha, sm_scale=sm_scale,
        USE_ALIBI=use_alibi,
        N_H=nheads, N_KVH=num_kv_heads, H_DIM=dim, MAX_SPLITS=max_splits, BINS=bins, PAGE_SIZE=page_size,
        stride_qh=stride_qh, stride_kz=stride_kz, stride_kh=stride_kh,
        stride_csz=stride_csz, stride_csh=stride_csh,
        stride_ah=stride_ah, stride_piz=stride_piz, stride_pih=stride_pih,
    )

    # ---------------- Stage 4 ---------------- #
    grid4 = (batch, nheads)
    _decode_stage4_reduce_hist[grid4](
        HIST_SPLIT=hist_split, HIST_GLOBAL=hist_global,
        N_H=nheads, MAX_SPLITS=max_splits, BINS=bins,
    )

    # ---------------- Stage 5 ---------------- #
    gridBH  = (batch, nheads)
    gridBHS = (batch, nheads, max_splits)

    # 5-init: compute (t_lo, t_hi, t) from GLOBAL_MAXS + HIST_GLOBAL
    _decode_stage5_init[gridBH](
        GLOBAL_MAXS=global_maxs, HIST_GLOBAL=hist_global,
        TAUS=taus, T_LOS=t_los, T_HIS=t_his,
        alpha=alpha, BINS=bins, N_H=nheads, stride_th=stride_th,
    )

    # Halley iterations: (accumulate over splits) -> (update per head)
    # usually just one is enough
    for _ in range(niter):
        _paged_stage5a_halley_accumulate[gridBHS](
            Q=q, K_cache=k_cache,
            ACC0_SPLIT=acc0_split, ACC1_SPLIT=acc1_split, ACC2_SPLIT=acc2_split,
            TAUS=taus,
            PAGE_INDICES=page_indices,
            Cache_seqlens=cache_seqlens,
            Q_Seqlens=q_seqlens,
            ALIBI_SLOPES=alibi_slopes,
            alpha=alpha, sm_scale=sm_scale,
            USE_ALIBI=use_alibi,
            N_H=nheads, N_KVH=num_kv_heads, H_DIM=dim, MAX_SPLITS=max_splits, PAGE_SIZE=page_size,
            stride_qh=stride_qh, stride_kz=stride_kz, stride_kh=stride_kh,
            stride_csz=stride_csz, stride_csh=stride_csh,
            stride_th=stride_th, stride_ah=stride_ah, stride_piz=stride_piz, stride_pih=stride_pih,
        )
        _decode_stage5b_halley_update[gridBH](
            TAUS=taus, T_LOS=t_los, T_HIS=t_his,
            ACC0_SPLIT=acc0_split, ACC1_SPLIT=acc1_split, ACC2_SPLIT=acc2_split,
            alpha=alpha, N_H=nheads, MAX_SPLITS=max_splits, stride_th=stride_th,
        )

    if taus_out is not None:
        taus_out.copy_(taus)

    # ---------------- Stage 6a ---------------- #
    grid6a = (batch, nheads, max_splits)
    _paged_stage6a_partial_out[grid6a](
        Q=q, K_cache=k_cache, V_cache=v_cache,
        PARTIAL_OUT=partial_out,
        TAUS=taus,
        PAGE_INDICES=page_indices,
        Cache_seqlens=cache_seqlens,
        Q_Seqlens=q_seqlens,
        ALIBI_SLOPES=alibi_slopes,
        alpha=alpha, sm_scale=sm_scale,
        USE_ALIBI=use_alibi,
        N_H=nheads, N_KVH=num_kv_heads, H_DIM=dim, MAX_SPLITS=max_splits, PAGE_SIZE=page_size,
        stride_qh=stride_qh, stride_kz=stride_kz, stride_kh=stride_kh,
        stride_vz=stride_vz, stride_vh=stride_vh,
        stride_csz=stride_csz, stride_csh=stride_csh,
        stride_th=stride_th, stride_ah=stride_ah, stride_piz=stride_piz, stride_pih=stride_pih,
    )

    # ---------------- Stage 6b ---------------- #
    _decode_stage6b_reduce_partials[grid4](
        PARTIAL_OUT=partial_out, OUT=out,
        N_H=nheads, H_DIM=dim, MAX_SPLITS=max_splits, stride_oh=stride_oh,
    )

    return out
