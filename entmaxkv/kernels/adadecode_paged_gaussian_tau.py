import math

import torch
import triton
import triton.language as tl

from entmaxkv.kernels.adadecode_paged import (
    _paged_stage5a_halley_accumulate,
)


@triton.jit
def _decode_stage5b_halley_update_unbounded(
    TAUS,
    ACC0_SPLIT,
    ACC1_SPLIT,
    ACC2_SPLIT,
    alpha: tl.constexpr,
    N_H: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    stride_th: tl.constexpr,
):
    coeff_0 = 1 / (alpha - 1)
    coeff_1 = coeff_0 - 1
    EPS_DENOM: tl.constexpr = 1e-12

    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_hz = off_z * N_H + off_h

    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    for s in range(MAX_SPLITS):
        base = off_hz * MAX_SPLITS + s
        acc0 += tl.load(ACC0_SPLIT + base)
        acc1 += tl.load(ACC1_SPLIT + base)
        acc2 += tl.load(ACC2_SPLIT + base)

    t = tl.load(TAUS + off_hz * stride_th)
    ff = acc0 - 1.0
    df = -coeff_0 * acc1
    ddf = coeff_0 * coeff_1 * acc2
    halley_denom = 2.0 * df * df - ff * ddf
    newton_denom = df

    t_halley = t - (2.0 * ff * df) / halley_denom
    t_newton = t - ff / newton_denom
    halley_ok = tl.abs(halley_denom) > EPS_DENOM
    newton_ok = tl.abs(newton_denom) > EPS_DENOM
    t_next = tl.where(halley_ok, t_halley, tl.where(newton_ok, t_newton, t))
    t_next = tl.where((t_next == t_next) & (tl.abs(t_next) < 1.0e20), t_next, t)
    tl.store(TAUS + off_hz * stride_th, t_next)


# Inlined from quest_adadecode_paged_optimized.py — self-contained Triton kernel.
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
def _paged_stage6_atomic_out(
    Q, K_cache, V_cache,
    OUT,
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
    stride_oh: tl.constexpr,
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

    out_ptrs = OUT + off_hz * stride_oh + offs_k
    tl.atomic_add(out_ptrs, acc.to(input_dtype), sem="relaxed")


def make_gaussian_tau_workspace(
    batch: int,
    nheads: int,
    max_splits: int,
    *,
    device,
):
    return {
        "acc0_split": torch.empty((batch, nheads, max_splits), device=device, dtype=torch.float32),
        "acc1_split": torch.empty((batch, nheads, max_splits), device=device, dtype=torch.float32),
        "acc2_split": torch.empty((batch, nheads, max_splits), device=device, dtype=torch.float32),
    }


def quest_sparse_attention_decode_paged_gaussian_tau(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    tau: torch.Tensor,
    alibi_slopes: torch.Tensor = None,
    alpha: float = 1.5,
    max_splits: int = 32,
    q_seqlens: torch.Tensor = None,
    tau_correction_iters: int = 0,
    workspace: dict = None,
):
    """
    Decode-only paged entmax using gaussian-estimated tau.

    tau_correction_iters=0 is the fixed-tau fast path. Values >0 use tau as
    the initial value and run exact Halley/Newton correction passes over the
    selected tokens before computing the output.
    """
    batch, nheads, _, dim = q.shape
    num_kv_heads = k_cache.shape[1]
    assert q.shape[2] == 1
    assert page_indices.dtype == torch.int32
    assert nheads % num_kv_heads == 0, "Query heads must be divisible by KV heads"
    assert tau.shape == (batch, nheads), "tau must be [B, H]"
    if cache_seqlens.dim() == 1:
        cache_seqlens = cache_seqlens[:, None].expand(batch, num_kv_heads)
    assert cache_seqlens.shape == (batch, num_kv_heads), "cache_seqlens must be [B, H_kv]"

    if q_seqlens is None:
        q_seqlens = cache_seqlens

    if workspace is None:
        workspace = make_gaussian_tau_workspace(
            batch, nheads, max_splits, device=q.device
        )

    acc0_split = workspace["acc0_split"]
    acc1_split = workspace["acc1_split"]
    acc2_split = workspace["acc2_split"]

    sm_scale = 1.0 / math.sqrt(dim)
    use_alibi = alibi_slopes is not None
    stride_ah = alibi_slopes.stride(0) if use_alibi and alibi_slopes.dim() > 0 else 0

    stride_qh = q.stride(1)
    stride_kz = k_cache.stride(0)
    stride_kh = k_cache.stride(1)
    stride_vz = v_cache.stride(0)
    stride_vh = v_cache.stride(1)
    stride_oh = out.stride(1)
    stride_csz = cache_seqlens.stride(0)
    stride_csh = cache_seqlens.stride(1)
    stride_piz = page_indices.stride(0)
    stride_pih = page_indices.stride(1)

    tau = tau.contiguous()
    stride_th = tau.stride(1)

    grid_bh = (batch, nheads)
    grid_bhs = (batch, nheads, max_splits)
    for _ in range(tau_correction_iters):
        _paged_stage5a_halley_accumulate[grid_bhs](
            Q=q,
            K_cache=k_cache,
            ACC0_SPLIT=acc0_split,
            ACC1_SPLIT=acc1_split,
            ACC2_SPLIT=acc2_split,
            TAUS=tau,
            PAGE_INDICES=page_indices,
            Cache_seqlens=cache_seqlens,
            Q_Seqlens=q_seqlens,
            ALIBI_SLOPES=alibi_slopes,
            alpha=alpha,
            sm_scale=sm_scale,
            USE_ALIBI=use_alibi,
            N_H=nheads,
            N_KVH=num_kv_heads,
            H_DIM=dim,
            MAX_SPLITS=max_splits,
            PAGE_SIZE=page_size,
            stride_qh=stride_qh,
            stride_kz=stride_kz,
            stride_kh=stride_kh,
            stride_csz=stride_csz,
            stride_csh=stride_csh,
            stride_th=stride_th,
            stride_ah=stride_ah,
            stride_piz=stride_piz,
            stride_pih=stride_pih,
        )
        _decode_stage5b_halley_update_unbounded[grid_bh](
            TAUS=tau,
            ACC0_SPLIT=acc0_split,
            ACC1_SPLIT=acc1_split,
            ACC2_SPLIT=acc2_split,
            alpha=alpha,
            N_H=nheads,
            MAX_SPLITS=max_splits,
            stride_th=stride_th,
        )

    out.zero_()
    _paged_stage6_atomic_out[grid_bhs](
        Q=q,
        K_cache=k_cache,
        V_cache=v_cache,
        OUT=out,
        TAUS=tau,
        PAGE_INDICES=page_indices,
        Cache_seqlens=cache_seqlens,
        Q_Seqlens=q_seqlens,
        ALIBI_SLOPES=alibi_slopes,
        alpha=alpha,
        sm_scale=sm_scale,
        USE_ALIBI=use_alibi,
        N_H=nheads,
        N_KVH=num_kv_heads,
        H_DIM=dim,
        MAX_SPLITS=max_splits,
        PAGE_SIZE=page_size,
        stride_qh=stride_qh,
        stride_kz=stride_kz,
        stride_kh=stride_kh,
        stride_vz=stride_vz,
        stride_vh=stride_vh,
        stride_csz=stride_csz,
        stride_csh=stride_csh,
        stride_th=stride_th,
        stride_oh=stride_oh,
        stride_ah=stride_ah,
        stride_piz=stride_piz,
        stride_pih=stride_pih,
    )

    return out
