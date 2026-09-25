"""Dedicated selected-page decode using the AdaQuest Gaussian-tau setup.

Scores for the selected tokens are materialized once in FP32; tau correction
and the output contraction reuse them. Moments are written for general alpha
with beta = 1/(alpha-1): the solver drives sum x**beta to 1, where
x = [(alpha-1)s - tau]_+, using d/dtau = -beta * sum x**(beta-1) and
d2/dtau2 = beta(beta-1) * sum x**(beta-2).
"""

import math

import torch
import triton
import triton.language as tl


GAUSSIAN_DECODE_SPLITS = 32


@triton.jit
def _pow_pos(x, exponent: tl.constexpr):
    """x ** exponent for x >= 0, with 0 ** anything == 0."""
    return tl.where(x > 0.0, tl.exp2(tl.log2(tl.maximum(x, 1.0e-30)) * exponent),
                    0.0)


@triton.jit
def _moments(x, alpha: tl.constexpr):
    """(sum-able) x**beta, x**(beta-1), x**(beta-2) for x >= 0."""
    if alpha == 1.5:
        return x * x, x, (x > 0.0).to(tl.float32)
    elif alpha == 2.0:
        return x, (x > 0.0).to(tl.float32), tl.zeros_like(x)
    elif alpha == 1.3333333333333333:
        return x * x * x, x * x, x
    else:
        beta: tl.constexpr = 1.0 / (alpha - 1.0)
        return (_pow_pos(x, beta), _pow_pos(x, beta - 1.0),
                _pow_pos(x, beta - 2.0))


@triton.jit
def _probability(x, alpha: tl.constexpr):
    if alpha == 1.5:
        return x * x
    elif alpha == 2.0:
        return x
    elif alpha == 1.3333333333333333:
        return x * x * x
    else:
        return _pow_pos(x, 1.0 / (alpha - 1.0))


@triton.jit
def _copy_tau(INPUT, OUTPUT, total: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    tl.store(OUTPUT + offsets, tl.load(INPUT + offsets, mask=mask), mask=mask)


@triton.jit
def _gaussian_local_max(
    Q, K, BLOCK_TABLE, SELECTED, SEL_LENS, SEQ_LENS, SLOPES, PARTIAL_MAX,
    SCORES, SCORE_CAPACITY: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_btb: tl.constexpr, stride_selb: tl.constexpr,
    stride_selh: tl.constexpr, stride_lb: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr, N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MAX_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    req, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_HEADS + head
    kv_head = head // (N_HEADS // N_KV_HEADS)
    selected_tokens = tl.load(SEL_LENS + req * stride_lb + head).to(tl.int32)
    tokens_per_split = tl.cdiv(selected_tokens, MAX_SPLITS)
    split_start = split * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, selected_tokens)
    base = hz * MAX_SPLITS + split
    if split_start >= selected_tokens:
        tl.store(PARTIAL_MAX + base, -1.0e6)
        return

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + req * stride_qb + head * stride_qh + offs_d)
    q = (q * ((alpha - 1.0) * sm_scale)).to(Q.dtype.element_ty)
    slope = tl.load(SLOPES + head).to(tl.float32) * (alpha - 1.0)
    seq_len = tl.load(SEQ_LENS + req).to(tl.int32)
    q_pos = seq_len - 1
    selected_base = SELECTED + req * stride_selb + head * stride_selh
    offs_n = tl.arange(0, BLOCK_N)
    local_max = -float("inf")
    for block in range(tl.cdiv(split_end - split_start, BLOCK_N)):
        compact = split_start + block * BLOCK_N + offs_n
        valid = compact < split_end
        rank, token = compact // BLOCK_SIZE, compact % BLOCK_SIZE
        logical = tl.load(selected_base + rank, mask=valid, other=0).to(tl.int32)
        physical = tl.load(BLOCK_TABLE + req * stride_btb + logical,
                           mask=valid, other=0).to(tl.int64)
        pos = logical * BLOCK_SIZE + token
        ptr = (K + physical[:, None] * stride_kblock
               + token[:, None] * stride_ktok + kv_head * stride_kh
               + offs_d[None, :] * stride_kd)
        k = tl.load(ptr, mask=valid[:, None], other=0.0)
        score = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        score += slope * (pos - q_pos).to(tl.float32)
        tl.store(SCORES + hz * SCORE_CAPACITY + compact, score, mask=valid)
        local_max = tl.maximum(
            local_max, tl.max(tl.where(valid, score, -float("inf"))))
    tl.store(PARTIAL_MAX + base, local_max)


@triton.jit
def _gaussian_bracket_init(PARTIAL_MAX, ESTIMATE, GLOBAL_MAX, TAU, TAU_LO,
                           TAU_HI, TAU_PREV, F_PREV,
                           stride_tb: tl.constexpr, N_HEADS: tl.constexpr,
                           MAX_SPLITS: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    hz = req * N_HEADS + head
    splits = tl.arange(0, MAX_SPLITS)
    maximum = tl.max(tl.load(PARTIAL_MAX + hz * MAX_SPLITS + splits))
    lo, hi = maximum - 1.0, maximum
    estimate = tl.load(ESTIMATE + req * stride_tb + head).to(tl.float32)
    tau = tl.minimum(tl.maximum(estimate, lo), hi)
    tl.store(GLOBAL_MAX + hz, maximum)
    tl.store(TAU + hz, tau)
    tl.store(TAU_LO + hz, lo)
    tl.store(TAU_HI + hz, hi)
    tl.store(TAU_PREV + hz, tau)
    tl.store(F_PREV + hz, 0.0)


@triton.jit
def _gaussian_newton_accumulate(
    SCORES, SEL_LENS, TAU, PARTIAL_MASS, PARTIAL_LINEAR, PARTIAL_ACTIVE,
    stride_lb: tl.constexpr, alpha: tl.constexpr, N_HEADS: tl.constexpr,
    SCORE_CAPACITY: tl.constexpr, MAX_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    req, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_HEADS + head
    selected_tokens = tl.load(SEL_LENS + req * stride_lb + head).to(tl.int32)
    tokens_per_split = tl.cdiv(selected_tokens, MAX_SPLITS)
    split_start = split * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, selected_tokens)
    base = hz * MAX_SPLITS + split
    if split_start >= selected_tokens:
        tl.store(PARTIAL_MASS + base, 0.0)
        tl.store(PARTIAL_LINEAR + base, 0.0)
        tl.store(PARTIAL_ACTIVE + base, 0.0)
        return

    tau = tl.load(TAU + hz)
    offs_n = tl.arange(0, BLOCK_N)
    mass, linear, active_count = 0.0, 0.0, 0.0
    for block in range(tl.cdiv(split_end - split_start, BLOCK_N)):
        compact = split_start + block * BLOCK_N + offs_n
        valid = compact < split_end
        score = tl.load(SCORES + hz * SCORE_CAPACITY + compact,
                        mask=valid, other=0.0)
        x = tl.where(valid, tl.maximum(score - tau, 0.0), 0.0)
        m0, m1, m2 = _moments(x, alpha)
        mass += tl.sum(m0)
        linear += tl.sum(m1)
        active_count += tl.sum(m2)
    tl.store(PARTIAL_MASS + base, mass)
    tl.store(PARTIAL_LINEAR + base, linear)
    tl.store(PARTIAL_ACTIVE + base, active_count)


@triton.jit
def _gaussian_halley_update(PARTIAL_MASS, PARTIAL_LINEAR, PARTIAL_ACTIVE, TAU,
                            alpha: tl.constexpr, N_HEADS: tl.constexpr,
                            MAX_SPLITS: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    hz = req * N_HEADS + head
    splits = tl.arange(0, MAX_SPLITS)
    base = hz * MAX_SPLITS
    mass = tl.sum(tl.load(PARTIAL_MASS + base + splits), axis=0)
    linear = tl.sum(tl.load(PARTIAL_LINEAR + base + splits), axis=0)
    active = tl.sum(tl.load(PARTIAL_ACTIVE + base + splits), axis=0)
    tau = tl.load(TAU + hz)
    beta: tl.constexpr = 1.0 / (alpha - 1.0)
    f = mass - 1.0
    df = -beta * linear
    ddf = beta * (beta - 1.0) * active
    denominator = 2.0 * df * df - f * ddf
    halley = tau - (2.0 * f * df) / denominator
    newton = tau - f / df
    candidate = tl.where(tl.abs(denominator) > 1.0e-12, halley,
                         tl.where(tl.abs(df) > 1.0e-12, newton, tau))
    candidate = tl.where((candidate == candidate)
                         & (tl.abs(candidate) < 1.0e20), candidate, tau)
    tl.store(TAU + hz, candidate)


@triton.jit
def _gaussian_bracketed_update(
    PARTIAL_MASS, PARTIAL_LINEAR, PARTIAL_ACTIVE, TAU, TAU_LO, TAU_HI,
    TAU_PREV, F_PREV, alpha: tl.constexpr, N_HEADS: tl.constexpr,
    MAX_SPLITS: tl.constexpr, ALLOW_SECANT: tl.constexpr,
):
    """AdaQuest robust_hybrid_update, written for general alpha."""
    req, head = tl.program_id(0), tl.program_id(1)
    hz = req * N_HEADS + head
    splits = tl.arange(0, MAX_SPLITS)
    base = hz * MAX_SPLITS
    mass = tl.sum(tl.load(PARTIAL_MASS + base + splits), axis=0)
    linear = tl.sum(tl.load(PARTIAL_LINEAR + base + splits), axis=0)
    active = tl.sum(tl.load(PARTIAL_ACTIVE + base + splits), axis=0)
    tau = tl.load(TAU + hz)
    lo, hi = tl.load(TAU_LO + hz), tl.load(TAU_HI + hz)
    previous, f_previous = tl.load(TAU_PREV + hz), tl.load(F_PREV + hz)
    beta: tl.constexpr = 1.0 / (alpha - 1.0)
    f, df, ddf = mass - 1.0, -beta * linear, beta * (beta - 1.0) * active
    lo = tl.where(f > 0.0, tau, lo)
    hi = tl.where(f < 0.0, tau, hi)
    candidate = 0.5 * (lo + hi)

    if ALLOW_SECANT:
        secant_denominator = f - f_previous
        secant = tau - f * (tau - previous) / secant_denominator
        secant_valid = ((tl.abs(secant_denominator) > 1.0e-12)
                        & (secant == secant) & (tl.abs(secant) < 1.0e20)
                        & (secant > lo - 1.0e-6)
                        & (secant < hi + 1.0e-6))
        candidate = tl.where(secant_valid, secant, candidate)

    newton = tau - f / df
    newton_valid = ((tl.abs(df) > 1.0e-12) & (newton == newton)
                    & (tl.abs(newton) < 1.0e20)
                    & (newton > lo - 1.0e-6) & (newton < hi + 1.0e-6))
    candidate = tl.where(newton_valid, newton, candidate)
    denominator = 2.0 * df * df - f * ddf
    halley = tau - (2.0 * f * df) / denominator
    halley_valid = ((tl.abs(denominator) > 1.0e-12) & (halley == halley)
                    & (tl.abs(halley) < 1.0e20)
                    & (halley > lo - 1.0e-6) & (halley < hi + 1.0e-6))
    candidate = tl.where(halley_valid, halley, candidate)
    candidate = tl.where(tl.abs(f) <= 1.0e-6, tau, candidate)
    tl.store(TAU_PREV + hz, tau)
    tl.store(F_PREV + hz, f)
    tl.store(TAU + hz, candidate)
    tl.store(TAU_LO + hz, lo)
    tl.store(TAU_HI + hz, hi)


@triton.jit
def _gaussian_partial_output(
    V, BLOCK_TABLE, SELECTED, SEL_LENS, TAU, SCORES,
    PARTIAL_OUT, PARTIAL_MASS, SCORE_CAPACITY: tl.constexpr,
    stride_vblock: tl.constexpr, stride_vtok: tl.constexpr,
    stride_vh: tl.constexpr, stride_vd: tl.constexpr,
    stride_btb: tl.constexpr, stride_selb: tl.constexpr,
    stride_selh: tl.constexpr, stride_lb: tl.constexpr,
    alpha: tl.constexpr, N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MAX_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    req, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_HEADS + head
    kv_head = head // (N_HEADS // N_KV_HEADS)
    selected_tokens = tl.load(SEL_LENS + req * stride_lb + head).to(tl.int32)
    tokens_per_split = tl.cdiv(selected_tokens, MAX_SPLITS)
    split_start = split * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, selected_tokens)
    offs_d = tl.arange(0, HEAD_DIM)
    out_base = (hz * MAX_SPLITS + split) * HEAD_DIM
    if split_start >= selected_tokens:
        tl.store(PARTIAL_OUT + out_base + offs_d, 0.0)
        tl.store(PARTIAL_MASS + hz * MAX_SPLITS + split, 0.0)
        return

    tau = tl.load(TAU + hz)
    selected_base = SELECTED + req * stride_selb + head * stride_selh
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((HEAD_DIM,), tl.float32)
    mass = 0.0
    for block in range(tl.cdiv(split_end - split_start, BLOCK_N)):
        compact = split_start + block * BLOCK_N + offs_n
        valid = compact < split_end
        score = tl.load(SCORES + hz * SCORE_CAPACITY + compact,
                        mask=valid, other=0.0)
        x = tl.where(valid, tl.maximum(score - tau, 0.0), 0.0)
        prob = _probability(x, alpha)
        has_support = tl.sum((prob > 0.0).to(tl.int32), axis=0) > 0
        if has_support:
            rank, token = compact // BLOCK_SIZE, compact % BLOCK_SIZE
            logical = tl.load(selected_base + rank, mask=valid, other=0).to(tl.int32)
            physical = tl.load(BLOCK_TABLE + req * stride_btb + logical,
                               mask=valid, other=0).to(tl.int64)
            v_ptr = (V + physical[:, None] * stride_vblock
                     + token[:, None] * stride_vtok + kv_head * stride_vh
                     + offs_d[None, :] * stride_vd)
            v = tl.load(v_ptr, mask=valid[:, None], other=0.0)
            acc += tl.sum(prob[:, None] * v, axis=0)
        mass += tl.sum(prob)
    tl.store(PARTIAL_OUT + out_base + offs_d, acc)
    tl.store(PARTIAL_MASS + hz * MAX_SPLITS + split, mass)


@triton.jit
def _gaussian_reduce_output(PARTIAL_OUT, PARTIAL_MASS, OUT, MASS,
                            stride_ob: tl.constexpr, stride_oh: tl.constexpr,
                            N_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
                            MAX_SPLITS: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    hz = req * N_HEADS + head
    offs_d = tl.arange(0, HEAD_DIM)
    acc = tl.zeros((HEAD_DIM,), tl.float32)
    mass = 0.0
    for split in range(MAX_SPLITS):
        base = (hz * MAX_SPLITS + split) * HEAD_DIM
        acc += tl.load(PARTIAL_OUT + base + offs_d)
        mass += tl.load(PARTIAL_MASS + hz * MAX_SPLITS + split)
    tl.store(OUT + req * stride_ob + head * stride_oh + offs_d, acc)
    tl.store(MASS + hz, mass)


def entmax_decode_gaussian_pages(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_pages: torch.Tensor,
    sel_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    tau: torch.Tensor,
    alibi_slopes: torch.Tensor,
    *,
    alpha: float = 1.5,
    niter: int = 1,
    out: torch.Tensor | None = None,
    workspace: "GaussianDecodeWorkspace | None" = None,
    return_debug: bool = False,
    safeguard_tau: bool = True,
) -> torch.Tensor:
    """Decode over a Gaussian selection using AdaQuest's Gaussian-tau path.

    Scores for selected tokens are materialized once in FP32 and reused by
    correction iterations and output contraction. Optional correction uses
    AdaQuest's exact-max bracket and robust hybrid update by default.
    """
    if q.ndim != 3:
        raise ValueError("q must be [B, H_q, D]")
    if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
        raise ValueError("kv_cache must be [2, blocks, block_size, H_kv, D]")
    batch, heads, dim = q.shape
    _, _, block_size, kv_heads, cache_dim = kv_cache.shape
    if dim != cache_dim or heads % kv_heads:
        raise ValueError("incompatible Q and paged KV head geometry")
    if selected_pages.shape[:2] != (batch, heads):
        raise ValueError("selected_pages must be [B, H_q, pages]")
    if sel_lens.shape != (batch, heads) or tau.shape != (batch, heads):
        raise ValueError("sel_lens and tau must be [B, H_q]")
    if niter < 0:
        raise ValueError("niter must be non-negative")
    if not alpha > 1.0:
        raise ValueError("Gaussian decode requires alpha > 1")
    alpha = float(alpha)
    if out is None:
        out = torch.empty_like(q)
    if workspace is None:
        workspace = GaussianDecodeWorkspace()
    ws = workspace.get(batch, heads, dim, q.dtype, q.device)
    k, v = kv_cache.unbind(0)
    slopes = alibi_slopes.to(device=q.device, dtype=torch.float32)
    estimated_tau = tau.to(device=q.device, dtype=torch.float32).contiguous()
    splits = GAUSSIAN_DECODE_SPLITS
    grid_bh, grid_bhs = (batch, heads), (batch, heads, splits)
    score_capacity = selected_pages.shape[2] * block_size
    scores = workspace.get_scores(batch, heads, score_capacity, q.device)
    max_args = dict(
        stride_qb=q.stride(0), stride_qh=q.stride(1),
        stride_kblock=k.stride(0), stride_ktok=k.stride(1),
        stride_kh=k.stride(2), stride_kd=k.stride(3),
        stride_btb=block_table.stride(0),
        stride_selb=selected_pages.stride(0),
        stride_selh=selected_pages.stride(1), stride_lb=sel_lens.stride(0),
        alpha=alpha, sm_scale=1.0 / math.sqrt(dim), N_HEADS=heads,
        N_KV_HEADS=kv_heads, HEAD_DIM=dim, BLOCK_SIZE=block_size,
        MAX_SPLITS=splits, BLOCK_N=64, SCORE_CAPACITY=score_capacity,
        num_warps=4)
    use_bracket = safeguard_tau and niter > 0
    _gaussian_local_max[grid_bhs](
        q, k, block_table, selected_pages, sel_lens, seq_lens, slopes,
        ws["partial_max"], scores, **max_args)
    if use_bracket:
        _gaussian_bracket_init[grid_bh](
            ws["partial_max"], estimated_tau, ws["global_max"], ws["tau"],
            ws["tau_lo"], ws["tau_hi"], ws["tau_prev"], ws["f_prev"],
            stride_tb=estimated_tau.stride(0), N_HEADS=heads,
            MAX_SPLITS=splits, num_warps=1)
    else:
        tau_block = min(1024, triton.next_power_of_2(estimated_tau.numel()))
        _copy_tau[(triton.cdiv(estimated_tau.numel(), tau_block),)](
            estimated_tau, ws["tau"], total=estimated_tau.numel(),
            BLOCK=tau_block, num_warps=1)
    for iteration in range(niter):
        _gaussian_newton_accumulate[grid_bhs](
            scores, sel_lens, ws["tau"], ws["partial_mass"],
            ws["partial_linear"], ws["partial_active"],
            stride_lb=sel_lens.stride(0), alpha=alpha, N_HEADS=heads,
            SCORE_CAPACITY=score_capacity, MAX_SPLITS=splits, BLOCK_N=64,
            num_warps=4)
        if use_bracket:
            _gaussian_bracketed_update[grid_bh](
                ws["partial_mass"], ws["partial_linear"],
                ws["partial_active"], ws["tau"], ws["tau_lo"],
                ws["tau_hi"], ws["tau_prev"], ws["f_prev"],
                alpha=alpha, N_HEADS=heads, MAX_SPLITS=splits,
                ALLOW_SECANT=iteration > 0, num_warps=1)
        else:
            _gaussian_halley_update[grid_bh](
                ws["partial_mass"], ws["partial_linear"],
                ws["partial_active"], ws["tau"], alpha=alpha, N_HEADS=heads,
                MAX_SPLITS=splits, num_warps=1)
    _gaussian_partial_output[grid_bhs](
        v, block_table, selected_pages, sel_lens, ws["tau"], scores,
        ws["partial_out"], ws["partial_mass"],
        stride_vblock=v.stride(0), stride_vtok=v.stride(1),
        stride_vh=v.stride(2), stride_vd=v.stride(3),
        stride_btb=block_table.stride(0),
        stride_selb=selected_pages.stride(0),
        stride_selh=selected_pages.stride(1), stride_lb=sel_lens.stride(0),
        alpha=alpha, N_HEADS=heads, N_KV_HEADS=kv_heads, HEAD_DIM=dim,
        BLOCK_SIZE=block_size, MAX_SPLITS=splits, BLOCK_N=64,
        SCORE_CAPACITY=score_capacity, num_warps=4)
    _gaussian_reduce_output[grid_bh](
        ws["partial_out"], ws["partial_mass"], out, ws["mass"],
        stride_ob=out.stride(0), stride_oh=out.stride(1), N_HEADS=heads,
        HEAD_DIM=dim, MAX_SPLITS=splits, num_warps=4)
    if return_debug:
        return out, {"tau": ws["tau"], "mass": ws["mass"]}
    return out


class GaussianDecodeWorkspace:

    def __init__(self):
        self._cache: dict = {}
        self._scores: dict = {}

    def get_scores(self, batch: int, heads: int, capacity: int,
                   device: torch.device) -> torch.Tensor:
        key = (batch, heads, capacity, str(device))
        scores = self._scores.get(key)
        if scores is None:
            scores = torch.empty((batch, heads, capacity), device=device,
                                 dtype=torch.float32)
            self._scores[key] = scores
        return scores

    def get(self, batch: int, heads: int, dim: int, dtype: torch.dtype,
            device: torch.device) -> dict:
        key = (batch, heads, dim, dtype, str(device))
        ws = self._cache.get(key)
        if ws is None:
            shape = (batch, heads, GAUSSIAN_DECODE_SPLITS)
            kwargs = dict(device=device, dtype=torch.float32)
            ws = {
                "tau": torch.empty((batch, heads), **kwargs),
                "partial_max": torch.empty(shape, **kwargs),
                "global_max": torch.empty((batch, heads), **kwargs),
                "tau_lo": torch.empty((batch, heads), **kwargs),
                "tau_hi": torch.empty((batch, heads), **kwargs),
                "tau_prev": torch.empty((batch, heads), **kwargs),
                "f_prev": torch.empty((batch, heads), **kwargs),
                "partial_mass": torch.empty(shape, **kwargs),
                "partial_linear": torch.empty(shape, **kwargs),
                "partial_active": torch.empty(shape, **kwargs),
                # AdaQuest stores split outputs in the input dtype before the
                # fp32 final reduction.
                "partial_out": torch.empty((*shape, dim), device=device,
                                           dtype=dtype),
                "mass": torch.empty((batch, heads), **kwargs),
            }
            self._cache[key] = ws
        return ws
