"""Quest-style page criticality: an upper bound on each page's best score.

For query head h, logical page p and its KV head g(h):

    U[h,p] = s * sum_d max(q[h,d]*kmin[p,g,d], q[h,d]*kmax[p,g,d]) + bmax[h,p]

with s = 1/sqrt(head_dim). The (alpha-1) factor the exact kernels fold into q
(paged_decode_utils.py) is a positive constant, so it cannot change the top-k
ordering — it is deliberately omitted here. The 1/sqrt(d) factor is NOT
optional: ALiBi is defined in the scaled score domain, so dropping it would
under-weight the positional term by sqrt(d) in the ranking.

ALiBi enters BEFORE selection, and must use LOGICAL positions: physical block
ids are arbitrary. For a causal non-negative slope the best position in a page
is its last valid one; for a negative slope it is the first. Slopes are indexed
by QUERY head while the key extrema are indexed by KV head.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _page_criticality(
    Q, K_MIN, K_MAX, BLOCK_TABLE, SEQ_LENS, SLOPES, PAGE_SCORES,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_mblock: tl.constexpr, stride_mh: tl.constexpr,
    stride_btb: tl.constexpr, stride_psb: tl.constexpr,
    stride_psh: tl.constexpr,
    sm_scale: tl.constexpr, N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MAX_PAGES: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    req = tl.program_id(0)
    head = tl.program_id(1)
    page_tile = tl.program_id(2)
    kv_head = head // (N_HEADS // N_KV_HEADS)
    seq_len = tl.load(SEQ_LENS + req).to(tl.int32)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + req * stride_qb + head * stride_qh + offs_d).to(tl.float32)
    q = q * sm_scale
    slope = tl.load(SLOPES + head).to(tl.float32)
    q_pos = seq_len - 1
    # A page is scorable only if it is entirely historical: complete (all
    # BLOCK_SIZE keys present, so its metadata is valid) and not the tail page
    # (which the selector force-includes and must never score from metadata).
    n_full = seq_len // BLOCK_SIZE
    tail = (seq_len - 1) // BLOCK_SIZE
    ps_base = PAGE_SCORES + req * stride_psb + head * stride_psh
    page = page_tile * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page < MAX_PAGES
    scorable = page_mask & (page < tail) & (page < n_full)

    # Resolve logical pages through the request's block table. Masking this
    # load by ``scorable`` is important: unused table entries may be stale and
    # the partial tail's extrema are deliberately not maintained.
    physical = tl.load(BLOCK_TABLE + req * stride_btb + page,
                       mask=scorable, other=0).to(tl.int64)
    m_off = (physical[:, None] * stride_mblock + kv_head * stride_mh
             + offs_d[None, :])

    # max(q*k_min, q*k_max) needs only one endpoint per dimension. Selecting
    # the address before loading halves metadata traffic compared with loading
    # both extrema, while producing the same bound.
    min_ptr = K_MIN + m_off
    max_ptr = K_MAX + m_off
    bound_ptr = tl.where(q[None, :] >= 0.0, max_ptr, min_ptr)
    endpoint = tl.load(bound_ptr, mask=scorable[:, None], other=0.0
                       ).to(tl.float32)
    bound = tl.sum(q[None, :] * endpoint, axis=1)

    # Best reachable ALiBi bias inside the page, by slope sign.
    last = tl.minimum((page + 1) * BLOCK_SIZE - 1, seq_len - 1)
    first = page * BLOCK_SIZE
    best_pos = tl.where(slope >= 0.0, last, first)
    bound += slope * (best_pos - q_pos).to(tl.float32)
    tl.store(ps_base + page, tl.where(scorable, bound, -float("inf")),
             mask=page_mask)


def page_criticality(
    q: torch.Tensor,             # (B, H_q, D)
    k_min: torch.Tensor,         # (num_blocks, H_kv, D)
    k_max: torch.Tensor,
    block_table: torch.Tensor,   # (B, max_pages) int32
    seq_lens: torch.Tensor,      # (B,) int32
    alibi_slopes: torch.Tensor,  # (H_q,) fp32, by QUERY head
    block_size: int,
    out: torch.Tensor,           # (B, H_q, max_pages) fp32
    live_pages: int | None = None,
) -> torch.Tensor:
    """Score logical pages per (request, query head). Non-scorable pages
    (the tail, incomplete pages, and pages past the sequence) get -inf.

    ``live_pages`` (host int, >= ceil(max seq_len / block_size)) limits the
    grid to the live prefix of the block table; columns past it are left
    untouched and must not be read. It is a launch-grid size, not a kernel
    constant, so a growing context never triggers a recompile.
    """
    batch, nheads, dim = q.shape
    n_kv_heads = k_min.shape[1]
    max_pages = block_table.shape[1]
    if live_pages is None:
        live_pages = max_pages
    # Keep roughly the same amount of dot-product work in each program across
    # supported head sizes. Pages are independent, so the third grid dimension
    # exposes enough parallelism even for batch-one decode.
    block_pages = min(32, 1024 // dim)
    grid = (batch, nheads, triton.cdiv(min(live_pages, max_pages), block_pages))
    _page_criticality[grid](
        q, k_min, k_max, block_table, seq_lens, alibi_slopes, out,
        stride_qb=q.stride(0), stride_qh=q.stride(1),
        stride_mblock=k_min.stride(0), stride_mh=k_min.stride(1),
        stride_btb=block_table.stride(0),
        stride_psb=out.stride(0), stride_psh=out.stride(1),
        sm_scale=1.0 / math.sqrt(dim), N_HEADS=nheads,
        N_KV_HEADS=n_kv_heads, HEAD_DIM=dim, BLOCK_SIZE=block_size,
        MAX_PAGES=max_pages, BLOCK_PAGES=block_pages, num_warps=4,
    )
    return out
