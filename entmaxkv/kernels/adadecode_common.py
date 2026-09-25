"""AdaDecode helper kernels shared by the dense and paged decoders.

"""

import triton
import triton.language as tl


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

