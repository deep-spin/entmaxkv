"""Triton kernels backing ``entmax_decode_paged_native``.

Adapts AdaSplash-2's max/histogram/refinement/output structure to
one-query decode. Work is partitioned over logical pages for occupancy;
all K/V accesses resolve those pages through the request's block table.
"""

import triton
import triton.language as tl

DECODE_PARTITIONS = 16
DECODE_HIST_BINS = 8


def default_decode_niter(alpha: float) -> int:
    """alpha=2.0 (sparsemax) Newton steps are linear, not quadratic, so it
    needs one more refinement than alpha=1.5 to converge mass to 1.0."""
    return 3 if alpha == 1.5 else 4


@triton.jit
def _paged_scores_max(
    Q, K, BLOCK_TABLE, SEQ_LENS, SLOPES, PARTIAL_MAX,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_btb: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr,
    N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr, PARTITIONS: tl.constexpr,
):
    req = tl.program_id(0)
    head = tl.program_id(1)
    part = tl.program_id(2)
    kv_head = head // (N_HEADS // N_KV_HEADS)
    seq_len = tl.load(SEQ_LENS + req).to(tl.int32)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_SIZE)
    q = tl.load(Q + req * stride_qb + head * stride_qh + offs_d)
    q = (q * ((alpha - 1.0) * sm_scale)).to(Q.dtype.element_ty)
    slope = tl.load(SLOPES + head).to(tl.float32) * (alpha - 1.0)
    q_pos = seq_len - 1
    score_max = -float("inf")
    pages_per_part: tl.constexpr = tl.cdiv(MAX_PAGES, PARTITIONS)
    for local_page in range(pages_per_part):
        logical_page = part * pages_per_part + local_page
        pos = logical_page * BLOCK_SIZE + offs_t
        valid = pos < seq_len
        physical_block = tl.load(
            BLOCK_TABLE + req * stride_btb + logical_page,
            mask=logical_page < MAX_PAGES, other=0,
        ).to(tl.int64)
        k_ptr = (K + physical_block * stride_kblock
                 + offs_t[:, None] * stride_ktok
                 + kv_head * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptr, mask=valid[:, None], other=0.0)
        score = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        score += slope * (pos - q_pos).to(tl.float32)
        score_max = tl.maximum(score_max,
                               tl.max(tl.where(valid, score, -float("inf"))))
    tl.store(PARTIAL_MAX + (req * N_HEADS + head) * PARTITIONS + part,
             score_max)


@triton.jit
def _reduce_max(PARTIAL_MAX, GLOBAL_MAX,
                N_HEADS: tl.constexpr, PARTITIONS: tl.constexpr):
    req = tl.program_id(0)
    head = tl.program_id(1)
    parts = tl.arange(0, PARTITIONS)
    base = (req * N_HEADS + head) * PARTITIONS
    tl.store(GLOBAL_MAX + req * N_HEADS + head,
             tl.max(tl.load(PARTIAL_MAX + base + parts)))


@triton.jit
def _paged_histogram(
    Q, K, BLOCK_TABLE, SEQ_LENS, SLOPES, GLOBAL_MAX, HISTOGRAM,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_btb: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr,
    N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr, PARTITIONS: tl.constexpr,
    BINS: tl.constexpr,
):
    req, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head = head // (N_HEADS // N_KV_HEADS)
    seq_len = tl.load(SEQ_LENS + req).to(tl.int32)
    offs_d, offs_t = tl.arange(0, HEAD_DIM), tl.arange(0, BLOCK_SIZE)
    bins = tl.arange(0, BINS)
    q = tl.load(Q + req * stride_qb + head * stride_qh + offs_d)
    q = (q * ((alpha - 1.0) * sm_scale)).to(Q.dtype.element_ty)
    slope = tl.load(SLOPES + head).to(tl.float32) * (alpha - 1.0)
    q_pos = seq_len - 1
    t0 = tl.load(GLOBAL_MAX + req * N_HEADS + head) - 1.0
    hist = tl.zeros((BINS,), tl.int32)
    pages_per_part: tl.constexpr = tl.cdiv(MAX_PAGES, PARTITIONS)
    for local_page in range(pages_per_part):
        logical_page = part * pages_per_part + local_page
        pos = logical_page * BLOCK_SIZE + offs_t
        valid = pos < seq_len
        physical = tl.load(BLOCK_TABLE + req * stride_btb + logical_page,
                           mask=logical_page < MAX_PAGES, other=0).to(tl.int64)
        ptr = (K + physical * stride_kblock + offs_t[:, None] * stride_ktok
               + kv_head * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(ptr, mask=valid[:, None], other=0.0)
        score = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        score += slope * (pos - q_pos).to(tl.float32)
        projected = score - t0
        bin_idx = tl.minimum((projected * BINS).to(tl.int32), BINS - 1)
        included = valid & (bin_idx >= 0)
        hist += tl.sum((bin_idx[:, None] == bins[None, :])
                       & included[:, None], axis=0).to(tl.int32)
    base = ((req * N_HEADS + head) * PARTITIONS + part) * BINS
    tl.store(HISTOGRAM + base + bins, hist)


@triton.jit
def _histogram_init(HISTOGRAM, GLOBAL_MAX, SEQ_LENS, TAU, TAU_LO, TAU_HI,
                    alpha: tl.constexpr, N_HEADS: tl.constexpr,
                    PARTITIONS: tl.constexpr, BINS: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    hz = req * N_HEADS + head
    seq_len = tl.load(SEQ_LENS + req).to(tl.float32)
    bins = tl.arange(0, BINS)
    hist = tl.zeros((BINS,), tl.int32)
    for part in range(PARTITIONS):
        base = (hz * PARTITIONS + part) * BINS
        hist += tl.load(HISTOGRAM + base + bins)
    maximum = tl.load(GLOBAL_MAX + hz)
    sum_z, sum_k, sum_z2 = 1.0, 1.0, 1.0
    for sj in range(BINS - 1, -1, -1):
        count = tl.sum(hist * (bins == sj).to(tl.int32)).to(tl.float32)
        if sj == BINS - 1:
            count -= 1.0
        edge = sj / BINS
        new_z = sum_z + count * edge
        new_k = sum_k + count
        new_z2 = sum_z2 + count * edge * edge
        if alpha == 2.0:
            accept = (new_z - new_k * edge) < 1.0
        else:
            accept = (new_k * edge * edge - 2.0 * new_z * edge
                      + new_z2) < 1.0
        sum_z = tl.where(accept, new_z, sum_z)
        sum_k = tl.where(accept, new_k, sum_k)
        sum_z2 = tl.where(accept, new_z2, sum_z2)
    if alpha == 2.0:
        lower = maximum + (sum_z - 1.0) / sum_k - 1.0
    else:
        disc = tl.maximum(sum_z * sum_z - sum_k * (sum_z2 - 1.0), 0.0)
        lower = maximum + (sum_z - tl.sqrt(disc)) / sum_k - 1.0
    # Peters et al. (2019) / AdaSplash-2 Eq. 5 bracket: with L total scores,
    # m - 1 <= tau* <= m - L**(1-alpha) in the (alpha-1)*s scale (m = max score).
    bracket_lo = maximum - 1.0
    bracket_hi = maximum - tl.exp((1.0 - alpha) * tl.log(seq_len))
    initial = tl.minimum(tl.maximum(lower + 0.5 / BINS, bracket_lo), bracket_hi)
    tl.store(TAU_LO + hz, bracket_lo)
    tl.store(TAU_HI + hz, bracket_hi)
    tl.store(TAU + hz, initial)


@triton.jit
def _paged_refine(
    Q, K, BLOCK_TABLE, SEQ_LENS, SLOPES, GLOBAL_MAX, TAU, PAGE_MASK,
    PARTIAL0, PARTIAL1, PARTIAL2,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_btb: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr,
    N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr, PARTITIONS: tl.constexpr,
    BUILD_MASK: tl.constexpr,
):
    req, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_HEADS + head
    kv_head = head // (N_HEADS // N_KV_HEADS)
    seq_len = tl.load(SEQ_LENS + req).to(tl.int32)
    offs_d, offs_t = tl.arange(0, HEAD_DIM), tl.arange(0, BLOCK_SIZE)
    q = tl.load(Q + req * stride_qb + head * stride_qh + offs_d)
    q = (q * ((alpha - 1.0) * sm_scale)).to(Q.dtype.element_ty)
    slope = tl.load(SLOPES + head).to(tl.float32) * (alpha - 1.0)
    tau = tl.load(TAU + hz)
    # The page mask needs a LOWER bound on tau*, so it can never exclude a token
    # that the true threshold would admit. Eq. 5's lower end, m - 1, is that
    # bound; its upper end (m - L**(1-alpha)) is only valid as a solver bracket
    # and would prune real support if used here.
    support_lo = tl.load(GLOBAL_MAX + hz) - 1.0
    q_pos = seq_len - 1
    acc0, acc1, acc2 = 0.0, 0.0, 0.0
    pages_per_part: tl.constexpr = tl.cdiv(MAX_PAGES, PARTITIONS)
    for local_page in range(pages_per_part):
        logical_page = part * pages_per_part + local_page
        pos = logical_page * BLOCK_SIZE + offs_t
        valid = pos < seq_len
        physical = tl.load(BLOCK_TABLE + req * stride_btb + logical_page,
                           mask=logical_page < MAX_PAGES, other=0).to(tl.int64)
        ptr = (K + physical * stride_kblock + offs_t[:, None] * stride_ktok
               + kv_head * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(ptr, mask=valid[:, None], other=0.0)
        score = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        score += slope * (pos - q_pos).to(tl.float32)
        if BUILD_MASK:
            candidate = tl.max(tl.where(valid, score, -float("inf"))) > support_lo
            tl.store(PAGE_MASK + hz * MAX_PAGES + logical_page,
                     candidate.to(tl.int8), mask=logical_page < MAX_PAGES)
        active = valid & (score > tau)
        x = tl.where(active, score - tau, 0.0)
        if alpha == 1.5:
            acc0 += tl.sum(x * x)
            acc1 += tl.sum(x)
            acc2 += tl.sum(active.to(tl.float32))
        else:
            acc0 += tl.sum(x)
            acc1 += tl.sum(active.to(tl.float32))
    base = hz * PARTITIONS + part
    tl.store(PARTIAL0 + base, acc0)
    tl.store(PARTIAL1 + base, acc1)
    tl.store(PARTIAL2 + base, acc2)


@triton.jit
def _refine_update(PARTIAL0, PARTIAL1, PARTIAL2, TAU, TAU_LO, TAU_HI,
                   alpha: tl.constexpr, N_HEADS: tl.constexpr,
                   PARTITIONS: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    hz = req * N_HEADS + head
    parts = tl.arange(0, PARTITIONS)
    base = hz * PARTITIONS
    acc0 = tl.sum(tl.load(PARTIAL0 + base + parts))
    acc1 = tl.sum(tl.load(PARTIAL1 + base + parts))
    acc2 = tl.sum(tl.load(PARTIAL2 + base + parts))
    tau, lo, hi = tl.load(TAU + hz), tl.load(TAU_LO + hz), tl.load(TAU_HI + hz)
    f = acc0 - 1.0
    lo = tl.where(f > 0.0, tau, lo)
    hi = tl.where(f < 0.0, tau, hi)
    if alpha == 1.5:
        df, ddf = -2.0 * acc1, 2.0 * acc2
    else:
        df, ddf = -acc1, 0.0
    candidate = tau - (2.0 * f * df) / (2.0 * df * df - f * ddf)
    good = (candidate > lo - 1.0e-6) & (candidate < hi + 1.0e-6)
    tau = tl.where(good, candidate, 0.5 * (lo + hi))
    tl.store(TAU + hz, tau)
    tl.store(TAU_LO + hz, lo)
    tl.store(TAU_HI + hz, hi)


@triton.jit
def _paged_output(
    Q, K, V, BLOCK_TABLE, SEQ_LENS, SLOPES, TAU, PAGE_MASK,
    PARTIAL_OUT, PARTIAL_MASS,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_vblock: tl.constexpr, stride_vtok: tl.constexpr,
    stride_vh: tl.constexpr, stride_vd: tl.constexpr,
    stride_btb: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr,
    N_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr, PARTITIONS: tl.constexpr,
):
    req, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_HEADS + head
    kv_head = head // (N_HEADS // N_KV_HEADS)
    seq_len = tl.load(SEQ_LENS + req).to(tl.int32)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + req * stride_qb + head * stride_qh + offs_d)
    q = (q * ((alpha - 1.0) * sm_scale)).to(Q.dtype.element_ty)
    slope = tl.load(SLOPES + head).to(tl.float32) * (alpha - 1.0)
    tau = tl.load(TAU + hz)
    q_pos = seq_len - 1
    acc = tl.zeros((HEAD_DIM,), tl.float32)

    mass = 0.0
    offs_t = tl.arange(0, BLOCK_SIZE)
    pages_per_part: tl.constexpr = tl.cdiv(MAX_PAGES, PARTITIONS)
    for local_page in range(pages_per_part):
        logical_page = part * pages_per_part + local_page
        candidate_page = tl.load(PAGE_MASK + hz * MAX_PAGES + logical_page,
                                 mask=logical_page < MAX_PAGES, other=0) != 0
        pos = logical_page * BLOCK_SIZE + offs_t
        valid = pos < seq_len
        physical_block = tl.load(
            BLOCK_TABLE + req * stride_btb + logical_page,
            mask=logical_page < MAX_PAGES, other=0,
        ).to(tl.int64)
        k_ptr = (K + physical_block * stride_kblock
                 + offs_t[:, None] * stride_ktok
                 + kv_head * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptr, mask=(valid & candidate_page)[:, None], other=0.0)
        score = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        score += slope * (pos - q_pos).to(tl.float32)
        x = tl.where(valid & candidate_page, tl.maximum(score - tau, 0.0), 0.0)
        if alpha == 1.5:
            prob = x * x
        else:
            prob = x
        v_ptr = (V + physical_block * stride_vblock
                 + offs_t[:, None] * stride_vtok
                 + kv_head * stride_vh + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptr, mask=(valid & candidate_page)[:, None], other=0.0)
        acc += tl.sum(prob[:, None] * v, axis=0)
        mass += tl.sum(prob)
    base = (hz * PARTITIONS + part) * HEAD_DIM
    tl.store(PARTIAL_OUT + base + offs_d, acc)
    tl.store(PARTIAL_MASS + hz * PARTITIONS + part, mass)


@triton.jit
def _reduce_output(PARTIAL_OUT, PARTIAL_MASS, OUT, MASS,
                   stride_ob: tl.constexpr, stride_oh: tl.constexpr,
                   N_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
                   PARTITIONS: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    hz = req * N_HEADS + head
    offs_d = tl.arange(0, HEAD_DIM)
    acc = tl.zeros((HEAD_DIM,), tl.float32)
    mass = 0.0
    for part in range(PARTITIONS):
        base = (hz * PARTITIONS + part) * HEAD_DIM
        acc += tl.load(PARTIAL_OUT + base + offs_d)
        mass += tl.load(PARTIAL_MASS + hz * PARTITIONS + part)
    tl.store(OUT + req * stride_ob + head * stride_oh + offs_d, acc)
    tl.store(MASS + hz, mass)
