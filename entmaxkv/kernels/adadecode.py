import math
import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import float2int_rd


# ------------------------------- #
# Helpers
# ------------------------------- #

@triton.jit
def halley_bisect_update(t, t_lo, t_hi, acc_0, acc_1, acc_2, coeff_0, coeff_1):
    """Halley-bisection update for finding tau"""
    EPS: tl.constexpr = 1e-6

    # Function eval
    ff = acc_0 - 1.0
    # First derivative
    df = -coeff_0 * acc_1
    # Second derivative
    ddf = coeff_0 * coeff_1 * acc_2

    # Update bounds
    t_lo = tl.where((ff > 0), t, t_lo)
    t_hi = tl.where((ff < 0), t, t_hi)

    # Halley's update
    new_t = t - (2 * ff * df) / (2 * df * df - ff * ddf)

    # Is halley's inside the bounds?
    is_good = (new_t > t_lo - EPS) & (new_t < t_hi + EPS)
    t = tl.where(is_good, new_t, 0.5 * (t_lo + t_hi))

    return t, t_lo, t_hi


@triton.jit
def init_tau_alpha2_tinit(hist0, bins, global_max, BINS: tl.constexpr):
    sum_z = 1.0
    sum_k = 1.0
    for sj in range(BINS - 1, -1, -1):
        c_bin = tl.sum(hist0 * (bins == sj).to(tl.int32)).to(tl.float32)
        if sj == (BINS - 1): c_bin -= 1.0
        c_tau = sj / BINS
        new_z = sum_z + c_bin * c_tau
        new_k = sum_k + c_bin
        flag  = (new_z - new_k * c_tau) < 1.0
        sum_z = tl.where(flag, new_z, sum_z)
        sum_k = tl.where(flag, new_k, sum_k)
    return global_max + (sum_z - 1.0) / sum_k - 1.0


@triton.jit
def init_tau_alpha15_tinit(hist0, bins, global_max, BINS: tl.constexpr):
    sum_z  = 1.0
    sum_k  = 1.0
    sum_z2 = 1.0
    for sj in range(BINS - 1, -1, -1):
        c_bin = tl.sum(hist0 * (bins == sj).to(tl.int32)).to(tl.float32)
        if sj == (BINS - 1): c_bin -= 1.0
        c_tau = sj / BINS
        new_z  = sum_z  + c_bin * c_tau
        new_k  = sum_k  + c_bin
        new_z2 = sum_z2 + c_bin * (c_tau * c_tau)
        flag = new_k * c_tau * c_tau - 2.0 * new_z * c_tau + new_z2 < 1.0
        sum_z  = tl.where(flag, new_z,  sum_z)
        sum_k  = tl.where(flag, new_k,  sum_k)
        sum_z2 = tl.where(flag, new_z2, sum_z2)
    disc = sum_z * sum_z - sum_k * (sum_z2 - 1.0)
    return global_max + (sum_z - tl.sqrt(disc)) / sum_k - 1.0


@triton.jit
def init_tau_generic_find_bin(hist0, bins, global_max, alpha, BINS: tl.constexpr):
    counts_f = hist0.to(tl.float32)
    counts_f = tl.where(bins == (BINS - 1), counts_f - 1.0, counts_f)  # top-bin correction
    p = 1.0 / (alpha - 1.0)

    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), BINS - 1, tl.int32)
    found = tl.full((), 0, tl.int32)
    b_idx = tl.full((), 0, tl.int32)

    for _ in range(5):  # enough for 16 bins
        mid = ((lo + hi) * 0.5).to(tl.int32)

        off  = (bins - mid).to(tl.int32)
        mask = off >= 0
        off_f = off.to(tl.float32)
        zL = off_f / BINS
        zU = (off_f + 1.0) / BINS

        posL = zL > 0
        zL_s = tl.where(posL, zL, 1.0)
        termL = tl.where(mask, tl.where(posL, tl.exp2(tl.log2(zL_s) * p), 0.0), 0.0)
        termU = tl.where(mask, tl.exp2(tl.log2(zU) * p), 0.0)

        fL = tl.sum(counts_f * termL) - 1.0
        fU = tl.sum(counts_f * termU) - 1.0

        go_right = fL > 0.0
        go_left  = fU < 0.0
        bracket  = (go_right == 0) & (go_left == 0)

        update = (found == 0)
        b_idx = tl.where(update & bracket, mid, b_idx)
        found = tl.where(update & bracket, 1,   found)
        lo = tl.where(update & go_right, mid + 1, lo)
        hi = tl.where(update & go_left,  mid - 1, hi)

    return tl.where(found == 1, b_idx, tl.minimum(lo, BINS - 1))


# ----------------------------------- #
# Stage 1: Local maxima per split     #
# grid = (B, H, S)                    #
# ----------------------------------- #

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['H_DIM', 'USE_ALIBI'],
)
@triton.jit
def _decode_stage1_local_max(
    Q, K_cache, V_cache,
    MAX_VALS,
    TOKEN_POSITIONS,
    Cache_seqlens,
    Q_Seqlens,
    ALIBI_SLOPES,
    ##
    alpha: tl.constexpr,
    sm_scale: tl.constexpr,
    ##
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    ##
    stride_qh: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_ph: tl.constexpr,
    ##
    BLOCK_N: tl.constexpr,
):
    """Per-split local maxima only, identical behavior to your stage1."""
    input_dtype = Q.dtype.element_ty
    _scalar = (alpha - 1) * sm_scale

    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    split_id = tl.program_id(2)
    off_hz = off_z * N_H + off_h

    cache_seqlen = tl.load(Cache_seqlens + off_z).to(tl.int32)
    seqlen_per_split = tl.cdiv(cache_seqlen, MAX_SPLITS)
    split_start = split_id * seqlen_per_split
    split_end = tl.minimum(split_start + seqlen_per_split, cache_seqlen)
    if split_start >= cache_seqlen:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, H_DIM)

    # Load Q
    q_ptrs = Q + off_hz * stride_qh + offs_k
    q = tl.load(q_ptrs) * _scalar
    q = q.to(input_dtype)

    # ALiBi
    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(ALIBI_SLOPES + off_h) * (alpha - 1)

    pos_base = TOKEN_POSITIONS + off_hz * stride_ph

    # q_idx is the original position of the query token in the full sequence
    q_idx = tl.load(Q_Seqlens + off_z).to(tl.int32) - 1
    k_base = K_cache + off_hz * stride_kh

    local_max = -1.0e6
    valid_blocks = tl.cdiv(split_end - split_start, BLOCK_N)

    for c_block in range(valid_blocks):
        k_idxs = split_start + c_block * BLOCK_N + offs_n
        k_ptrs = k_base + k_idxs[:, None] * H_DIM + offs_k[None, :]
        k_mask = k_idxs < split_end
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(input_dtype)

        qk = tl.sum(q[None, :] * k, axis=1)
        if USE_ALIBI:
            actual_positions = tl.load(pos_base + k_idxs, mask=k_mask, other=0)
            position_diff = -(q_idx - actual_positions)
            qk += alibi_slope * position_diff

        qk = tl.where(k_mask, qk, float("-inf"))
        local_max = tl.maximum(local_max, tl.max(qk))

    tl.store(MAX_VALS + off_hz * MAX_SPLITS + split_id, local_max)


# --------------------------------------------------- #
# Stage 2: Reduce local maxima → global max per head  #
# grid = (B, H)                                       #
# --------------------------------------------------- #

@triton.jit
def _decode_stage2_reduce_max(
    MAX_VALS, GLOBAL_MAXS,
    ##
    N_H: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_hz = off_z * N_H + off_h

    gmax = -1.0e6
    for i in range(MAX_SPLITS):
        gmax = tl.maximum(gmax, tl.load(MAX_VALS + off_hz * MAX_SPLITS + i))
    tl.store(GLOBAL_MAXS + off_hz, gmax)


# ---------------------------------------------------- #
# Stage 3: Build per-split histograms                  #
# grid = (B, H, S)                                     #
# ---------------------------------------------------- #

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['H_DIM', 'USE_ALIBI'],
)
@triton.jit
def _decode_stage3_build_hist(
    Q, K_cache,
    GLOBAL_MAXS,
    HIST_SPLIT,                # [B, H, S, BINS] contiguous in BINS
    TOKEN_POSITIONS,
    Cache_seqlens,
    Q_Seqlens,
    ALIBI_SLOPES,
    ##
    alpha: tl.constexpr,
    sm_scale: tl.constexpr,
    ##
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    BINS: tl.constexpr,        # e.g., 16
    ##
    stride_qh: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_ph: tl.constexpr,
    ##
    BLOCK_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    _scalar = (alpha - 1) * sm_scale

    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    split_id = tl.program_id(2)
    off_hz = off_z * N_H + off_h

    cache_seqlen = tl.load(Cache_seqlens + off_z).to(tl.int32)
    seqlen_per_split = tl.cdiv(cache_seqlen, MAX_SPLITS)
    split_start = split_id * seqlen_per_split
    split_end = tl.minimum(split_start + seqlen_per_split, cache_seqlen)
    if split_start >= cache_seqlen:
        # Zero out this split's hist (optional; assume host pre-zeroed)
        return

    # Load q
    offs_k = tl.arange(0, H_DIM)
    q = tl.load(Q + off_hz * stride_qh + offs_k) * _scalar
    q = q.to(input_dtype)

    pos_base = TOKEN_POSITIONS + off_hz * stride_ph

    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(ALIBI_SLOPES + off_h) * (alpha - 1)

    # q_idx is the original position of the query token in the full sequence
    q_idx = tl.load(Q_Seqlens + off_z).to(tl.int32) - 1
    k_base = K_cache + off_hz * stride_kh

    global_max = tl.load(GLOBAL_MAXS + off_hz)
    t0 = global_max - 1.0

    hist = tl.zeros((BINS,), dtype=tl.int32)
    bins = tl.arange(0, BINS)

    offs_n = tl.arange(0, BLOCK_N)
    valid_blocks = tl.cdiv(split_end - split_start, BLOCK_N)

    for c_block in range(valid_blocks):
        k_idxs = split_start + c_block * BLOCK_N + offs_n
        k_mask = k_idxs < split_end

        k_ptrs = k_base + k_idxs[:, None] * H_DIM + offs_k[None, :]
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(input_dtype)

        qk = tl.sum(q[None, :] * k, axis=1)
        if USE_ALIBI:
            actual_positions = tl.load(pos_base + k_idxs, mask=k_mask, other=0)
            pos_diff = -(q_idx - actual_positions)
            qk += alibi_slope * pos_diff

        mask = k_mask
        proj = qk - t0
        b = (proj * BINS).to(tl.int32)
        b = tl.minimum(b, BINS - 1)
        valid = mask & (b >= 0)

        eq = (b[:, None] == bins[None, :])
        cnts = tl.sum(eq & valid[:, None], axis=0).to(tl.int32)
        hist += cnts

    # Store this split histogram
    base = ((off_hz * MAX_SPLITS + split_id) * BINS)
    tl.store(HIST_SPLIT + base + bins, hist)


# --------------------------------------------- #
# Stage 4: Reduce histograms to per-head global #
# grid = (B, H)                                  #
# --------------------------------------------- #

@triton.jit
def _decode_stage4_reduce_hist(
    HIST_SPLIT,                # [B, H, S, BINS]
    HIST_GLOBAL,               # [B, H, BINS]
    ##
    N_H: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    BINS: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_hz = off_z * N_H + off_h

    bins = tl.arange(0, BINS)
    acc = tl.zeros((BINS,), dtype=tl.int32)

    for s in range(MAX_SPLITS):
        base = ((off_hz * MAX_SPLITS + s) * BINS)
        acc += tl.load(HIST_SPLIT + base + bins)

    tl.store(HIST_GLOBAL + off_hz * BINS + bins, acc)


# -------------------- #
# Stage 5: split into  #
#   5-init  (B, H)     #
#   5a-acc  (B, H, S)  #
#   5b-upd  (B, H)     #
# -------------------- #

@triton.jit
def _decode_stage5_init(   # grid = (B, H)
    GLOBAL_MAXS,
    HIST_GLOBAL,
    TAUS, T_LOS, T_HIS,
    ##
    alpha: tl.constexpr,
    BINS: tl.constexpr,
    ##
    N_H: tl.constexpr,
    ##
    stride_th: tl.constexpr,   # TAUS/T_LOS/T_HIS share shape [B, H]
):
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_hz = off_z * N_H + off_h

    global_max = tl.load(GLOBAL_MAXS + off_hz)
    t0 = global_max - 1.0

    bins = tl.arange(0, BINS)
    hist0 = tl.load(HIST_GLOBAL + off_hz * BINS + bins)

    # init bracket
    if alpha == 2.0:
        t_init = init_tau_alpha2_tinit(hist0, bins, global_max, BINS)
        t_lo = t_init
        t_hi = t_init + 1.0 / BINS
        t    = 0.5 * (t_lo + t_hi)
    elif alpha == 1.5:
        t_init = init_tau_alpha15_tinit(hist0, bins, global_max, BINS)
        t_lo = t_init
        t_hi = t_init + 1.0 / BINS
        t    = 0.5 * (t_lo + t_hi)
    else:
        b_sel  = init_tau_generic_find_bin(hist0, bins, global_max, alpha, BINS)
        e_lo   = t0 + b_sel.to(tl.float32) / BINS
        b_next = tl.minimum(b_sel + 1, BINS)
        e_hi   = t0 + b_next.to(tl.float32) / BINS
        t_lo = e_lo
        t_hi = e_hi
        t    = 0.5 * (e_lo + e_hi)

    tl.store(TAUS + off_hz * stride_th, t)
    tl.store(T_LOS + off_hz * stride_th, t_lo)
    tl.store(T_HIS + off_hz * stride_th, t_hi)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['H_DIM', 'USE_ALIBI'],
)
@triton.jit
def _decode_stage5a_halley_accumulate(   # grid = (B, H, S)
    Q, K_cache,
    ACC0_SPLIT, ACC1_SPLIT, ACC2_SPLIT,   # [B,H,S]
    TAUS,
    TOKEN_POSITIONS,
    Cache_seqlens,
    Q_Seqlens,
    ALIBI_SLOPES,
    ##
    alpha: tl.constexpr,
    sm_scale: tl.constexpr,
    ##
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    ##
    stride_qh: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_th: tl.constexpr,    # TAUS stride on dim=1
    stride_ah: tl.constexpr,
    stride_ph: tl.constexpr,
    ##
    BLOCK_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    _scalar = (alpha - 1) * sm_scale
    coeff_0 = 1 / (alpha - 1)
    coeff_1 = coeff_0 - 1
    coeff_2 = coeff_0 - 2

    off_z   = tl.program_id(0)
    off_h   = tl.program_id(1)
    split_id= tl.program_id(2)
    off_hz  = off_z * N_H + off_h

    cache_seqlen = tl.load(Cache_seqlens + off_z).to(tl.int32)

    seqlen_per_split = tl.cdiv(cache_seqlen, MAX_SPLITS)
    split_start = split_id * seqlen_per_split
    split_end   = tl.minimum(split_start + seqlen_per_split, cache_seqlen)
    if split_start >= cache_seqlen:
        # Store zeros if this split is empty
        base = off_hz * MAX_SPLITS + split_id
        tl.store(ACC0_SPLIT + base, 0.0)
        tl.store(ACC1_SPLIT + base, 0.0)
        tl.store(ACC2_SPLIT + base, 0.0)
        return

    # data
    offs_k = tl.arange(0, H_DIM)
    q = tl.load(Q + off_hz * stride_qh + offs_k) * _scalar
    q = q.to(input_dtype)

    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(ALIBI_SLOPES + off_h) * (alpha - 1)

    pos_base = TOKEN_POSITIONS + off_hz * stride_ph

    # q_idx is the original position of the query token in the full sequence
    q_idx = tl.load(Q_Seqlens + off_z).to(tl.int32) - 1
    k_base = K_cache + off_hz * stride_kh

    # current tau
    t = tl.load(TAUS + off_hz * stride_th)

    # accumulators for this split
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0

    offs_n = tl.arange(0, BLOCK_N)
    valid_blocks = tl.cdiv(split_end - split_start, BLOCK_N)

    for c_block in range(valid_blocks):
        k_idxs = split_start + c_block * BLOCK_N + offs_n
        k_mask = k_idxs < split_end

        k_ptrs = k_base + k_idxs[:, None] * H_DIM + offs_k[None, :]
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(input_dtype)

        qk = tl.sum(q[None, :] * k, axis=1)
        if USE_ALIBI:
            actual_positions = tl.load(pos_base + k_idxs, mask=k_mask, other=0)
            qk += alibi_slope * (-(q_idx - actual_positions))

        qk_mask = qk > t
        # if IS_CAUSAL: qk_mask &= (k_idxs <= q_idx)
        qk_mask &= k_mask

        qk_mask_f = qk_mask.to(tl.float32)
        qk_act = (qk - t) * qk_mask_f

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


@triton.jit
def _decode_stage5b_halley_update(   # grid = (B, H)
    TAUS, T_LOS, T_HIS,
    ACC0_SPLIT, ACC1_SPLIT, ACC2_SPLIT,   # [B,H,S]
    ##
    alpha: tl.constexpr,
    ##
    N_H: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    ##
    stride_th: tl.constexpr,   # TAUS/T_LOS/T_HIS stride on dim=1
):
    coeff_0 = 1 / (alpha - 1)
    coeff_1 = coeff_0 - 1

    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_hz = off_z * N_H + off_h

    # reduce per-split accumulators
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    for s in range(MAX_SPLITS):
        base = off_hz * MAX_SPLITS + s
        acc0 += tl.load(ACC0_SPLIT + base)
        acc1 += tl.load(ACC1_SPLIT + base)
        acc2 += tl.load(ACC2_SPLIT + base)

    t    = tl.load(TAUS + off_hz * stride_th)
    t_lo = tl.load(T_LOS + off_hz * stride_th)
    t_hi = tl.load(T_HIS + off_hz * stride_th)

    t, t_lo, t_hi = halley_bisect_update(t, t_lo, t_hi, acc0, acc1, acc2, coeff_0, coeff_1)

    tl.store(TAUS + off_hz * stride_th, t)
    tl.store(T_LOS + off_hz * stride_th, t_lo)
    tl.store(T_HIS + off_hz * stride_th, t_hi)



# ----------------------------------------------------------------- #
# Stage 6a: Partial outputs per split (same semantics as your S3)   #
# grid = (B, H, S)                                                  #
# ----------------------------------------------------------------- #

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['H_DIM', 'USE_ALIBI'],
)
@triton.jit
def _decode_stage6a_partial_out(
    Q,
    K_cache,
    V_cache,
    PARTIAL_OUT,
    TAUS,
    TOKEN_POSITIONS,
    Cache_seqlens,
    Q_Seqlens,
    ALIBI_SLOPES,
    ##
    alpha: tl.constexpr,
    sm_scale: tl.constexpr,
    ##
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    ##
    stride_qh: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_th: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_ph: tl.constexpr,
    ##
    BLOCK_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    _scalar = (alpha - 1) * sm_scale
    coeff_0 = 1 / (alpha - 1)

    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    split_id = tl.program_id(2)
    off_hz = off_z * N_H + off_h

    cache_seqlen = tl.load(Cache_seqlens + off_z).to(tl.int32)
    # q_idx is the original position of the query token in the full sequence
    q_idx = tl.load(Q_Seqlens + off_z).to(tl.int32) - 1

    seqlen_per_split = tl.cdiv(cache_seqlen, MAX_SPLITS)
    split_start = split_id * seqlen_per_split
    split_end = tl.minimum(split_start + seqlen_per_split, cache_seqlen)
    if split_start >= cache_seqlen:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, H_DIM)

    q = tl.load(Q + off_hz * stride_qh + offs_k) * _scalar
    q = q.to(input_dtype)

    alibi_slope = 0.0
    if USE_ALIBI:
        alibi_slope = tl.load(ALIBI_SLOPES + off_h) * (alpha - 1)
    
    pos_base = TOKEN_POSITIONS + off_hz * stride_ph


    k_base = K_cache + off_hz * stride_kh
    v_base = V_cache + off_hz * stride_vh

    acc = tl.zeros([H_DIM], dtype=tl.float32)
    valid_blocks = tl.cdiv(split_end - split_start, BLOCK_N)

    t = tl.load(TAUS + off_hz * stride_th)

    for c_block in range(valid_blocks):
        k_idxs = split_start + c_block * BLOCK_N + offs_n
        k_ptrs = k_base + k_idxs[:, None] * H_DIM + offs_k[None, :]
        kv_mask = k_idxs < split_end
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(input_dtype)

        qk = tl.sum(q[None, :] * k, axis=1)

        if USE_ALIBI:
            actual_positions = tl.load(pos_base + k_idxs, mask=kv_mask, other=0)
            position_diff = -(q_idx - actual_positions)
            qk += alibi_slope * position_diff

        qk_mask = qk > t
        # if IS_CAUSAL:
        #     qk_mask &= (k_idxs <= q_idx)
        qk_mask &= kv_mask

        has_nonzero = tl.sum(qk_mask.to(tl.int32)) > 0
        if has_nonzero:
            v_ptrs = v_base + k_idxs[:, None] * H_DIM + offs_k[None, :]
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(input_dtype)

            qk_act = qk - t
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


# ------------------------------------------------------------ #
# Stage 6b: Reduce partial outputs, grid = (B, H)              #
# ------------------------------------------------------------ #

@triton.jit
def _decode_stage6b_reduce_partials(
    PARTIAL_OUT, OUT,
    ##
    N_H: tl.constexpr,
    H_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    ##
    stride_oh: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_hz = off_z * N_H + off_h

    offs_k = tl.arange(0, H_DIM)
    acc = tl.zeros([H_DIM], dtype=tl.float32)

    for split_id in range(MAX_SPLITS):
        base = PARTIAL_OUT + (off_hz * MAX_SPLITS + split_id) * H_DIM
        acc += tl.load(base + offs_k)

    out_ptrs = OUT + off_hz * stride_oh + offs_k
    tl.store(out_ptrs, acc)


# ============================================================ #
# Orchestrator: sparse_attention_decode_sixstage()      
# ============================================================ #

def sparse_attention_decode_orchestrator(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    cache_seqlens: torch.Tensor,
    token_positions: torch.Tensor,
    alibi_slopes: torch.Tensor = None,
    is_causal: bool = True,
    alpha: float = 1.5,
    niter: int = 10,
    max_splits: int = 32,
    bins: int = 16,
    q_seqlens: torch.Tensor = None,
    profile: dict = None,
):
    """
    Six-stage pipeline:
      1) per-split local maxima
      2) reduce maxima to per-head global_max
      3) per-split histograms (using global_max)
      4) reduce histograms to per-head histogram
      5) Halley refinement (per-head) -> TAUS
      6a) per-split partial outputs
      6b) reduce partials to OUT
    """
    batch, nheads, _, dim = q.shape
    assert q.shape[2] == 1

    sm_scale = 1.0 / math.sqrt(dim)
    use_alibi = alibi_slopes is not None
    stride_ah = alibi_slopes.stride(0) if use_alibi and alibi_slopes.dim() > 0 else 0

    # q_seqlens holds the original full-sequence length for each batch element.
    # Used to compute actual query token position for ALiBi biases.
    # If None, fall back to cache_seqlens (no-op when ALiBi is disabled).
    if q_seqlens is None:
        q_seqlens = cache_seqlens

    profile_events = []

    def _profile_start(name):
        if profile is None:
            return None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return name, start

    def _profile_end(handle):
        if handle is None:
            return
        name, start = handle
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        profile_events.append((name, start, end))

    # Intermediates
    _total_profile = _profile_start("total")
    _alloc_profile = _profile_start("allocate_workspace")
    max_vals = torch.full((batch, nheads, max_splits), float('-inf'), device=q.device, dtype=torch.float32)
    global_maxs = torch.empty((batch, nheads), device=q.device, dtype=torch.float32)

    hist_split = torch.zeros((batch, nheads, max_splits, bins), device=q.device, dtype=torch.int32)
    hist_global = torch.empty((batch, nheads, bins), device=q.device, dtype=torch.int32)

    partial_out = torch.zeros((batch, nheads, max_splits, dim), device=q.device, dtype=q.dtype)
    taus = torch.empty((batch, nheads), device=q.device, dtype=torch.float32)

    # New buffers for stage 5
    t_los = torch.empty((batch, nheads), device=q.device, dtype=torch.float32)
    t_his = torch.empty((batch, nheads), device=q.device, dtype=torch.float32)
    acc0_split = torch.empty((batch, nheads, max_splits), device=q.device, dtype=torch.float32)
    acc1_split = torch.empty((batch, nheads, max_splits), device=q.device, dtype=torch.float32)
    acc2_split = torch.empty((batch, nheads, max_splits), device=q.device, dtype=torch.float32)
    _profile_end(_alloc_profile)
    
    # Strides
    stride_qh = q.stride(1)
    stride_kh = k_cache.stride(1)
    stride_vh = v_cache.stride(1)
    stride_th = taus.stride(1)
    stride_oh = out.stride(1)
    stride_ph = token_positions.stride(1)

    # Validate dimensions
    assert k_cache.shape[0] == batch, f"k_cache batch mismatch: {k_cache.shape[0]} != {batch}"
    assert k_cache.shape[1] == nheads, f"k_cache heads mismatch: {k_cache.shape[1]} != {nheads}"
    assert k_cache.shape[3] == dim, f"k_cache dim mismatch: {k_cache.shape[3]} != {dim}"

    # ---------------- Stage 1 ---------------- #
    grid1 = (batch, nheads, max_splits)
    _stage_profile = _profile_start("stage1_local_max")
    _decode_stage1_local_max[grid1](
        Q=q, K_cache=k_cache, V_cache=v_cache,
        MAX_VALS=max_vals,
        TOKEN_POSITIONS=token_positions,
        Cache_seqlens=cache_seqlens,
        Q_Seqlens=q_seqlens,
        ALIBI_SLOPES=alibi_slopes,
        alpha=alpha,
        sm_scale=sm_scale,
        IS_CAUSAL=is_causal,
        USE_ALIBI=use_alibi,
        N_H=nheads,
        H_DIM=dim,
        MAX_SPLITS=max_splits,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_vh=stride_vh,
        stride_ah=stride_ah,
        stride_ph=stride_ph,
    )
    _profile_end(_stage_profile)

    # ---------------- Stage 2 ---------------- #
    grid2 = (batch, nheads)
    _stage_profile = _profile_start("stage2_reduce_max")
    _decode_stage2_reduce_max[grid2](
        MAX_VALS=max_vals,
        GLOBAL_MAXS=global_maxs,
        N_H=nheads,
        MAX_SPLITS=max_splits,
    )
    _profile_end(_stage_profile)

    # ---------------- Stage 3 ---------------- #
    grid3 = (batch, nheads, max_splits)
    _stage_profile = _profile_start("stage3_build_hist")
    _decode_stage3_build_hist[grid3](
        Q=q, K_cache=k_cache,
        GLOBAL_MAXS=global_maxs,
        HIST_SPLIT=hist_split,
        TOKEN_POSITIONS=token_positions,
        Cache_seqlens=cache_seqlens,
        Q_Seqlens=q_seqlens,
        ALIBI_SLOPES=alibi_slopes,
        alpha=alpha,
        sm_scale=sm_scale,
        IS_CAUSAL=is_causal,
        USE_ALIBI=use_alibi,
        N_H=nheads,
        H_DIM=dim,
        MAX_SPLITS=max_splits,
        BINS=bins,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_ah=stride_ah,
        stride_ph=stride_ph,
    )
    _profile_end(_stage_profile)

    # ---------------- Stage 4 ---------------- #
    grid4 = (batch, nheads)
    _stage_profile = _profile_start("stage4_reduce_hist")
    _decode_stage4_reduce_hist[grid4](
        HIST_SPLIT=hist_split,
        HIST_GLOBAL=hist_global,
        N_H=nheads,
        MAX_SPLITS=max_splits,
        BINS=bins,
    )
    _profile_end(_stage_profile)

    # ---------------- Stage 5 ---------------- #
    gridBH  = (batch, nheads)
    gridBHS = (batch, nheads, max_splits)
    
    # 5-init: compute (t_lo, t_hi, t) from GLOBAL_MAXS + HIST_GLOBAL
    _stage_profile = _profile_start("stage5_init")
    _decode_stage5_init[gridBH](
        GLOBAL_MAXS=global_maxs,
        HIST_GLOBAL=hist_global,
        TAUS=taus, T_LOS=t_los, T_HIS=t_his,
        alpha=alpha, BINS=bins,
        N_H=nheads,
        stride_th=stride_th,
    )
    _profile_end(_stage_profile)
    
    # Halley iterations: (accumulate over splits) -> (update per head)
    # usually just one is enough
    for _ in range(niter):
        _stage_profile = _profile_start("stage5a_halley_accumulate")
        _decode_stage5a_halley_accumulate[gridBHS](
            Q=q, K_cache=k_cache,
            ACC0_SPLIT=acc0_split, ACC1_SPLIT=acc1_split, ACC2_SPLIT=acc2_split,
            TAUS=taus,
            TOKEN_POSITIONS=token_positions,
            Cache_seqlens=cache_seqlens,
            Q_Seqlens=q_seqlens,
            ALIBI_SLOPES=alibi_slopes,
            alpha=alpha,
            sm_scale=sm_scale,
            IS_CAUSAL=is_causal,
            USE_ALIBI=use_alibi,
            N_H=nheads,
            H_DIM=dim,
            MAX_SPLITS=max_splits,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_th=stride_th,
            stride_ah=stride_ah,
            stride_ph=stride_ph,
        )
        _profile_end(_stage_profile)
        _stage_profile = _profile_start("stage5b_halley_update")
        _decode_stage5b_halley_update[gridBH](
            TAUS=taus, T_LOS=t_los, T_HIS=t_his,
            ACC0_SPLIT=acc0_split, ACC1_SPLIT=acc1_split, ACC2_SPLIT=acc2_split,
            alpha=alpha,
            N_H=nheads,
            MAX_SPLITS=max_splits,
            stride_th=stride_th,
        )
        _profile_end(_stage_profile)

    # ---------------- Stage 6a ---------------- #
    grid6a = (batch, nheads, max_splits)
    _stage_profile = _profile_start("stage6a_partial_out")
    _decode_stage6a_partial_out[grid6a](
        Q=q, K_cache=k_cache, V_cache=v_cache,
        PARTIAL_OUT=partial_out,
        TAUS=taus,
        TOKEN_POSITIONS=token_positions,
        Cache_seqlens=cache_seqlens,
        Q_Seqlens=q_seqlens,
        ALIBI_SLOPES=alibi_slopes,
        alpha=alpha,
        sm_scale=sm_scale,
        IS_CAUSAL=is_causal,
        USE_ALIBI=use_alibi,
        N_H=nheads,
        H_DIM=dim,
        MAX_SPLITS=max_splits,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_vh=stride_vh,
        stride_th=stride_th,
        stride_ah=stride_ah,
        stride_ph=stride_ph,
    )
    _profile_end(_stage_profile)

    # ---------------- Stage 6b ---------------- #
    _stage_profile = _profile_start("stage6b_reduce_partials")
    _decode_stage6b_reduce_partials[grid4](
        PARTIAL_OUT=partial_out, 
        OUT=out,
        N_H=nheads, 
        H_DIM=dim, 
        MAX_SPLITS=max_splits,
        stride_oh=stride_oh,
    )
    _profile_end(_stage_profile)
    _profile_end(_total_profile)

    if profile is not None:
        torch.cuda.synchronize()
        for name, start, end in profile_events:
            profile[name] = profile.get(name, 0.0) + start.elapsed_time(end)
        profile["_count"] = profile.get("_count", 0) + 1

    return out
