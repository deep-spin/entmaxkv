"""Generic-alpha entmax decode over a selected page set (paged layout).

This is the fallback for alpha outside {1.5, 2.0}, which the one-K-pass
``selected_decode`` kernels do not cover. It keeps AdaDecode's six-stage
structure (max -> histogram -> tau init -> Halley/bisection -> output) and its
generic-alpha tau initialisation, but consumes the same inputs as
``selected_decode``:

* ``kv_cache`` is ``[2, num_blocks, block_size, H_kv, D]`` and every access
  resolves ``logical page -> block_table[req, logical] -> physical block``;
* ``selected_pages`` is keyed by QUERY head (``[B, H_q, n]``, -1 padded, tail
  last) and ``sel_lens[b, h]`` counts its tokens, so GQA siblings may select
  different pages while K/V stay compact;
* ALiBi slopes are indexed by query head and applied at logical positions.

Work is split over the compacted token index ``c``: page rank ``c // BLOCK``,
in-page offset ``c % BLOCK``. Every stage recomputes scores from K.
"""

import math

import torch
import triton
import triton.language as tl

from entmaxkv.kernels.adadecode_common import (
    _decode_stage2_reduce_max,
    _decode_stage4_reduce_hist,
    _decode_stage5_init,
    _decode_stage5b_halley_update,
    _decode_stage6b_reduce_partials,
)

ADADECODE_SPLITS = 32
ADADECODE_BINS = 16
ADADECODE_BLOCK_N = 64


@triton.jit
def _split_bounds(SEL_LENS, req, head, split, stride_lb: tl.constexpr,
                  MAX_SPLITS: tl.constexpr):
    selected_tokens = tl.load(SEL_LENS + req * stride_lb + head).to(tl.int32)
    per_split = tl.cdiv(selected_tokens, MAX_SPLITS)
    start = split * per_split
    end = tl.minimum(start + per_split, selected_tokens)
    return start, end, selected_tokens


@triton.jit
def _block_scores(q, slope, q_pos, compact, valid, sel_base, bt_base,
                  K, kv_head, offs_d,
                  stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
                  stride_kh: tl.constexpr, stride_kd: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr):
    """(alpha-1)-scaled scores for compact indices; -inf where invalid."""
    rank, token = compact // BLOCK_SIZE, compact % BLOCK_SIZE
    logical = tl.load(sel_base + rank, mask=valid, other=0).to(tl.int32)
    physical = tl.load(bt_base + logical, mask=valid, other=0).to(tl.int64)
    rows = physical * stride_kblock + token * stride_ktok + kv_head * stride_kh
    k = tl.load(K + rows[:, None] + offs_d[None, :] * stride_kd,
                mask=valid[:, None], other=0.0)
    score = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
    pos = logical * BLOCK_SIZE + token
    score += slope * (pos - q_pos).to(tl.float32)
    return tl.where(valid, score, -float("inf"))


@triton.jit
def _load_query(Q, SLOPES, SEQ_LENS, req, head, offs_d,
                stride_qb: tl.constexpr, stride_qh: tl.constexpr,
                alpha: tl.constexpr, sm_scale: tl.constexpr):
    q = tl.load(Q + req * stride_qb + head * stride_qh + offs_d)
    q = (q * ((alpha - 1.0) * sm_scale)).to(Q.dtype.element_ty)
    slope = tl.load(SLOPES + head).to(tl.float32) * (alpha - 1.0)
    q_pos = tl.load(SEQ_LENS + req).to(tl.int32) - 1
    return q, slope, q_pos


@triton.jit
def _paged_stage1_local_max(
    Q, K, BLOCK_TABLE, SELECTED, SEL_LENS, SEQ_LENS, SLOPES, MAX_VALS,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_btb: tl.constexpr, stride_selb: tl.constexpr,
    stride_selh: tl.constexpr, stride_lb: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr,
    N_H: tl.constexpr, N_KVH: tl.constexpr, H_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MAX_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    req, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_H + head
    start, end, total = _split_bounds(SEL_LENS, req, head, split, stride_lb,
                                      MAX_SPLITS)
    if start >= total:
        tl.store(MAX_VALS + hz * MAX_SPLITS + split, -1.0e6)
        return
    kv_head = head // (N_H // N_KVH)
    offs_d = tl.arange(0, H_DIM)
    q, slope, q_pos = _load_query(Q, SLOPES, SEQ_LENS, req, head, offs_d,
                                  stride_qb, stride_qh, alpha, sm_scale)
    sel_base = SELECTED + req * stride_selb + head * stride_selh
    bt_base = BLOCK_TABLE + req * stride_btb
    offs_n = tl.arange(0, BLOCK_N)
    local_max = -1.0e6
    for block in range(tl.cdiv(end - start, BLOCK_N)):
        compact = start + block * BLOCK_N + offs_n
        score = _block_scores(q, slope, q_pos, compact, compact < end,
                              sel_base, bt_base, K, kv_head, offs_d,
                              stride_kblock, stride_ktok, stride_kh,
                              stride_kd, BLOCK_SIZE)
        local_max = tl.maximum(local_max, tl.max(score))
    tl.store(MAX_VALS + hz * MAX_SPLITS + split, local_max)


@triton.jit
def _paged_stage3_build_hist(
    Q, K, BLOCK_TABLE, SELECTED, SEL_LENS, SEQ_LENS, SLOPES,
    GLOBAL_MAXS, HIST_SPLIT,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_btb: tl.constexpr, stride_selb: tl.constexpr,
    stride_selh: tl.constexpr, stride_lb: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr,
    N_H: tl.constexpr, N_KVH: tl.constexpr, H_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MAX_SPLITS: tl.constexpr,
    BINS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    req, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_H + head
    bins = tl.arange(0, BINS)
    base = (hz * MAX_SPLITS + split) * BINS
    start, end, total = _split_bounds(SEL_LENS, req, head, split, stride_lb,
                                      MAX_SPLITS)
    if start >= total:
        tl.store(HIST_SPLIT + base + bins, tl.zeros((BINS,), dtype=tl.int32))
        return
    kv_head = head // (N_H // N_KVH)
    offs_d = tl.arange(0, H_DIM)
    q, slope, q_pos = _load_query(Q, SLOPES, SEQ_LENS, req, head, offs_d,
                                  stride_qb, stride_qh, alpha, sm_scale)
    sel_base = SELECTED + req * stride_selb + head * stride_selh
    bt_base = BLOCK_TABLE + req * stride_btb
    t0 = tl.load(GLOBAL_MAXS + hz) - 1.0
    hist = tl.zeros((BINS,), dtype=tl.int32)
    offs_n = tl.arange(0, BLOCK_N)
    for block in range(tl.cdiv(end - start, BLOCK_N)):
        compact = start + block * BLOCK_N + offs_n
        valid = compact < end
        score = _block_scores(q, slope, q_pos, compact, valid,
                              sel_base, bt_base, K, kv_head, offs_d,
                              stride_kblock, stride_ktok, stride_kh,
                              stride_kd, BLOCK_SIZE)
        b = tl.minimum(((score - t0) * BINS).to(tl.int32), BINS - 1)
        included = valid & (b >= 0)
        hist += tl.sum((b[:, None] == bins[None, :]) & included[:, None],
                       axis=0).to(tl.int32)
    tl.store(HIST_SPLIT + base + bins, hist)


@triton.jit
def _paged_stage5a_halley_accumulate(
    Q, K, BLOCK_TABLE, SELECTED, SEL_LENS, SEQ_LENS, SLOPES, TAUS,
    ACC0_SPLIT, ACC1_SPLIT, ACC2_SPLIT,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_btb: tl.constexpr, stride_selb: tl.constexpr,
    stride_selh: tl.constexpr, stride_lb: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr,
    N_H: tl.constexpr, N_KVH: tl.constexpr, H_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MAX_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    coeff_0 = 1 / (alpha - 1)
    coeff_1 = coeff_0 - 1
    coeff_2 = coeff_0 - 2
    req, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_H + head
    out = hz * MAX_SPLITS + split
    start, end, total = _split_bounds(SEL_LENS, req, head, split, stride_lb,
                                      MAX_SPLITS)
    if start >= total:
        tl.store(ACC0_SPLIT + out, 0.0)
        tl.store(ACC1_SPLIT + out, 0.0)
        tl.store(ACC2_SPLIT + out, 0.0)
        return
    kv_head = head // (N_H // N_KVH)
    offs_d = tl.arange(0, H_DIM)
    q, slope, q_pos = _load_query(Q, SLOPES, SEQ_LENS, req, head, offs_d,
                                  stride_qb, stride_qh, alpha, sm_scale)
    sel_base = SELECTED + req * stride_selb + head * stride_selh
    bt_base = BLOCK_TABLE + req * stride_btb
    t = tl.load(TAUS + hz)
    acc0, acc1, acc2 = 0.0, 0.0, 0.0
    offs_n = tl.arange(0, BLOCK_N)
    for block in range(tl.cdiv(end - start, BLOCK_N)):
        compact = start + block * BLOCK_N + offs_n
        score = _block_scores(q, slope, q_pos, compact, compact < end,
                              sel_base, bt_base, K, kv_head, offs_d,
                              stride_kblock, stride_ktok, stride_kh,
                              stride_kd, BLOCK_SIZE)
        mask = score > t
        mask_f = mask.to(tl.float32)
        act = tl.where(mask, score - t, 0.0)
        if alpha == 2.0:
            acc0 += tl.sum(act)
            acc1 += tl.sum(mask_f)
        elif alpha == 1.5:
            acc0 += tl.sum(act * act)
            acc1 += tl.sum(act)
            acc2 += tl.sum(mask_f)
        else:
            log2_act = tl.log2(tl.where(mask, act, 1.0))
            acc0 += tl.sum(tl.where(mask, tl.exp2(log2_act * coeff_0), 0.0))
            acc1 += tl.sum(tl.where(mask, tl.exp2(log2_act * coeff_1), 0.0))
            acc2 += tl.sum(tl.where(mask, tl.exp2(log2_act * coeff_2), 0.0))
    tl.store(ACC0_SPLIT + out, acc0)
    tl.store(ACC1_SPLIT + out, acc1)
    tl.store(ACC2_SPLIT + out, acc2)


@triton.jit
def _paged_stage6a_partial_out(
    Q, K, V, BLOCK_TABLE, SELECTED, SEL_LENS, SEQ_LENS, SLOPES, TAUS,
    PARTIAL_OUT,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kblock: tl.constexpr, stride_ktok: tl.constexpr,
    stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_vblock: tl.constexpr, stride_vtok: tl.constexpr,
    stride_vh: tl.constexpr, stride_vd: tl.constexpr,
    stride_btb: tl.constexpr, stride_selb: tl.constexpr,
    stride_selh: tl.constexpr, stride_lb: tl.constexpr,
    alpha: tl.constexpr, sm_scale: tl.constexpr,
    N_H: tl.constexpr, N_KVH: tl.constexpr, H_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MAX_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    coeff_0 = 1 / (alpha - 1)
    req, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hz = req * N_H + head
    offs_d = tl.arange(0, H_DIM)
    out_base = PARTIAL_OUT + (hz * MAX_SPLITS + split) * H_DIM
    start, end, total = _split_bounds(SEL_LENS, req, head, split, stride_lb,
                                      MAX_SPLITS)
    if start >= total:
        # PARTIAL_OUT is allocated with torch.empty: fully define idle splits.
        tl.store(out_base + offs_d, tl.zeros((H_DIM,), dtype=tl.float32))
        return
    kv_head = head // (N_H // N_KVH)
    q, slope, q_pos = _load_query(Q, SLOPES, SEQ_LENS, req, head, offs_d,
                                  stride_qb, stride_qh, alpha, sm_scale)
    sel_base = SELECTED + req * stride_selb + head * stride_selh
    bt_base = BLOCK_TABLE + req * stride_btb
    t = tl.load(TAUS + hz)
    acc = tl.zeros([H_DIM], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for block in range(tl.cdiv(end - start, BLOCK_N)):
        compact = start + block * BLOCK_N + offs_n
        valid = compact < end
        score = _block_scores(q, slope, q_pos, compact, valid,
                              sel_base, bt_base, K, kv_head, offs_d,
                              stride_kblock, stride_ktok, stride_kh,
                              stride_kd, BLOCK_SIZE)
        mask = score > t
        if tl.sum(mask.to(tl.int32)) > 0:
            rank, token = compact // BLOCK_SIZE, compact % BLOCK_SIZE
            logical = tl.load(sel_base + rank, mask=valid, other=0).to(tl.int32)
            physical = tl.load(bt_base + logical, mask=valid,
                               other=0).to(tl.int64)
            v_rows = (physical * stride_vblock + token * stride_vtok
                      + kv_head * stride_vh)
            v = tl.load(V + v_rows[:, None] + offs_d[None, :] * stride_vd,
                        mask=mask[:, None], other=0.0)
            act = tl.where(mask, score - t, 0.0)
            if alpha == 2.0:
                prob = act
            elif alpha == 1.5:
                prob = act * act
            else:
                prob = tl.where(mask, tl.exp2(tl.log2(tl.where(mask, act, 1.0))
                                              * coeff_0), 0.0)
            acc += tl.sum(v.to(tl.float32) * prob[:, None], axis=0)
    tl.store(out_base + offs_d, acc)


class AdaDecodePagedWorkspace:
    """Persistent, shape-keyed scratch for the generic-alpha decoder."""

    def __init__(self):
        self._cache: dict = {}

    def get(self, batch: int, nheads: int, dim: int, device) -> dict:
        key = (batch, nheads, dim, str(device))
        ws = self._cache.get(key)
        if ws is not None:
            return ws
        splits, bins = ADADECODE_SPLITS, ADADECODE_BINS
        f32 = dict(device=device, dtype=torch.float32)
        ws = {
            "max_vals": torch.empty((batch, nheads, splits), **f32),
            "global_maxs": torch.empty((batch, nheads), **f32),
            "hist_split": torch.empty((batch, nheads, splits, bins),
                                      device=device, dtype=torch.int32),
            "hist_global": torch.empty((batch, nheads, bins), device=device,
                                       dtype=torch.int32),
            "partial_out": torch.empty((batch, nheads, splits, dim), **f32),
            "taus": torch.empty((batch, nheads), **f32),
            "t_los": torch.empty((batch, nheads), **f32),
            "t_his": torch.empty((batch, nheads), **f32),
            "acc0_split": torch.empty((batch, nheads, splits), **f32),
            "acc1_split": torch.empty((batch, nheads, splits), **f32),
            "acc2_split": torch.empty((batch, nheads, splits), **f32),
        }
        self._cache[key] = ws
        return ws


def entmax_decode_selected_pages_generic(
    q: torch.Tensor,               # (B, H_q, D)
    kv_cache: torch.Tensor,        # (2, num_blocks, block_size, H_kv, D)
    block_table: torch.Tensor,     # (B, max_pages) int32
    selected_pages: torch.Tensor,  # (B, H_q, n) int32, logical, -1 pad, tail last
    sel_lens: torch.Tensor,        # (B, H_q) int32
    seq_lens: torch.Tensor,        # (B,) int32
    alpha: float,
    alibi_slopes: torch.Tensor,    # (H_q,) by QUERY head
    niter: int = 10,
    out: torch.Tensor | None = None,
    workspace: AdaDecodePagedWorkspace | None = None,
) -> torch.Tensor:
    """Entmax decode over ``selected_pages`` for any alpha > 1.

    Exact with respect to the selected set. Prefer
    ``entmax_decode_selected_pages`` for alpha in {1.5, 2.0}.
    """
    if q.ndim != 3:
        raise ValueError(f"q must have shape [B, H_q, D], got {tuple(q.shape)}")
    if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
        raise ValueError("kv_cache must be [2, num_blocks, block_size, H_kv, D]")
    batch, nheads, dim = q.shape
    _, _, block_size, n_kv_heads, cache_dim = kv_cache.shape
    if dim != cache_dim or nheads % n_kv_heads:
        raise ValueError("incompatible Q and paged KV head geometry")
    if not alpha > 1.0:
        raise ValueError("generic entmax decode requires alpha > 1")
    if dim & (dim - 1):
        raise ValueError("head dimension must be a power of two")
    if selected_pages.shape[:2] != (batch, nheads):
        raise ValueError("selected_pages must be [B, H_q, n] (by QUERY head)")
    if sel_lens.shape != (batch, nheads):
        raise ValueError("sel_lens must be [B, H_q]")
    if selected_pages.dtype != torch.int32 or sel_lens.dtype != torch.int32:
        raise TypeError("selected_pages and sel_lens must be int32")
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise TypeError("block_table and seq_lens must be int32")
    slopes = alibi_slopes.to(device=q.device, dtype=torch.float32)
    if slopes.shape != (nheads,):
        raise ValueError(f"alibi_slopes must have shape ({nheads},)")
    if out is None:
        out = torch.empty_like(q)
    if out.stride(2) != 1 or out.stride(0) != nheads * out.stride(1):
        # _decode_stage6b addresses OUT as (b*H + h) * stride_oh + d.
        raise ValueError("out must have uniformly strided rows and unit last "
                         "stride")
    if workspace is None:
        workspace = AdaDecodePagedWorkspace()
    ws = workspace.get(batch, nheads, dim, q.device)
    splits, bins = ADADECODE_SPLITS, ADADECODE_BINS
    k_cache, v_cache = kv_cache.unbind(0)
    common = dict(
        stride_qb=q.stride(0), stride_qh=q.stride(1),
        stride_kblock=k_cache.stride(0), stride_ktok=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_btb=block_table.stride(0),
        stride_selb=selected_pages.stride(0),
        stride_selh=selected_pages.stride(1), stride_lb=sel_lens.stride(0),
        alpha=float(alpha), sm_scale=1.0 / math.sqrt(dim),
        N_H=nheads, N_KVH=n_kv_heads, H_DIM=dim, BLOCK_SIZE=block_size,
        MAX_SPLITS=splits, BLOCK_N=ADADECODE_BLOCK_N, num_warps=4,
    )
    inputs = (q, k_cache, block_table, selected_pages, sel_lens, seq_lens,
              slopes)
    grid_bh, grid_bhs = (batch, nheads), (batch, nheads, splits)
    stride_th = ws["taus"].stride(1)

    _paged_stage1_local_max[grid_bhs](*inputs, ws["max_vals"], **common)
    _decode_stage2_reduce_max[grid_bh](
        MAX_VALS=ws["max_vals"], GLOBAL_MAXS=ws["global_maxs"],
        N_H=nheads, MAX_SPLITS=splits)
    _paged_stage3_build_hist[grid_bhs](
        *inputs, ws["global_maxs"], ws["hist_split"], BINS=bins, **common)
    _decode_stage4_reduce_hist[grid_bh](
        HIST_SPLIT=ws["hist_split"], HIST_GLOBAL=ws["hist_global"],
        N_H=nheads, MAX_SPLITS=splits, BINS=bins)
    _decode_stage5_init[grid_bh](
        GLOBAL_MAXS=ws["global_maxs"], HIST_GLOBAL=ws["hist_global"],
        TAUS=ws["taus"], T_LOS=ws["t_los"], T_HIS=ws["t_his"],
        alpha=float(alpha), BINS=bins, N_H=nheads, stride_th=stride_th)
    for _ in range(int(niter)):
        _paged_stage5a_halley_accumulate[grid_bhs](
            *inputs, ws["taus"], ws["acc0_split"], ws["acc1_split"],
            ws["acc2_split"], **common)
        _decode_stage5b_halley_update[grid_bh](
            TAUS=ws["taus"], T_LOS=ws["t_los"], T_HIS=ws["t_his"],
            ACC0_SPLIT=ws["acc0_split"], ACC1_SPLIT=ws["acc1_split"],
            ACC2_SPLIT=ws["acc2_split"], alpha=float(alpha), N_H=nheads,
            MAX_SPLITS=splits, stride_th=stride_th)
    q_, k_, bt_, sel_, lens_, seqs_, slopes_ = inputs
    _paged_stage6a_partial_out[grid_bhs](
        q_, k_, v_cache, bt_, sel_, lens_, seqs_, slopes_, ws["taus"],
        ws["partial_out"],
        stride_vblock=v_cache.stride(0), stride_vtok=v_cache.stride(1),
        stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3), **common)
    _decode_stage6b_reduce_partials[grid_bh](
        PARTIAL_OUT=ws["partial_out"], OUT=out, N_H=nheads, H_DIM=dim,
        MAX_SPLITS=splits, stride_oh=out.stride(1))
    return out
