import math
import torch
import triton
import triton.language as tl

def _get_autotune_configs():
    configs = []
    for block_pages in [4, 8, 16, 32]:
        for block_dim in [32, 64, 128]:
            for num_warps in [2, 4]:
                configs.append(
                    triton.Config(
                        {'BLOCK_PAGES': block_pages, 'BLOCK_DIM': block_dim},
                        num_warps=num_warps,
                    )
                )
    return configs


# ------------------------------- #
# Triton kernel
# ------------------------------- #

@triton.autotune(configs=_get_autotune_configs(), key=['num_pages', 'head_dim'])
@triton.jit
def _page_criticality_kernel(
    Q,           # [batch, num_heads, 1, head_dim]
    K_min,       # [batch, num_kv_heads, num_pages, head_dim]
    K_max,       # [batch, num_kv_heads, num_pages, head_dim]
    Out,         # [batch, num_heads, num_pages]
    Alibi_Bonus, # [num_heads, num_pages] float32; unused when HAS_ALIBI=False
    # Q strides
    stride_qb, stride_qh, stride_q1, stride_qd,
    # K_min / K_max strides (same layout)
    stride_kb, stride_kh, stride_kp, stride_kd,
    # Out strides
    stride_ob, stride_oh, stride_op,
    # Alibi_Bonus strides
    stride_ah, stride_ap,
    # dims
    num_pages,
    head_dim,
    scale,      # 1/sqrt(head_dim) or 1.0
    # compile-time constants
    GQA_GROUP: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Fused page criticality estimation kernel.

    For each page p and query head h:
        score = sum_d max(q[h,d] * k_min[kv_h, p, d], q[h,d] * k_max[kv_h, p, d]) * scale
                [+ alibi_bonus[h, p]  if HAS_ALIBI]

    Grid: (batch, num_kv_heads, cdiv(num_pages, BLOCK_PAGES))
    GQA group heads are handled by looping inside the kernel so that
    k_min/k_max are loaded once and reused across group members.
    """
    pid_batch = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_page_tile = tl.program_id(2)

    page_start = pid_page_tile * BLOCK_PAGES
    page_offsets = page_start + tl.arange(0, BLOCK_PAGES)  # [BLOCK_PAGES]
    page_mask = page_offsets < num_pages

    kmin_base = K_min + pid_batch * stride_kb + pid_kv_head * stride_kh
    kmax_base = K_max + pid_batch * stride_kb + pid_kv_head * stride_kh

    for g in range(GQA_GROUP):
        q_head_idx = pid_kv_head * GQA_GROUP + g
        q_base = Q + pid_batch * stride_qb + q_head_idx * stride_qh
        acc = tl.zeros([BLOCK_PAGES], dtype=tl.float32)

        for d_start in range(0, head_dim, BLOCK_DIM):
            d_offsets = d_start + tl.arange(0, BLOCK_DIM)
            d_mask = d_offsets < head_dim

            q_ptrs = q_base + d_offsets * stride_qd
            q_chunk = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

            kmin_ptrs = kmin_base + page_offsets[:, None] * stride_kp + d_offsets[None, :] * stride_kd
            kmax_ptrs = kmax_base + page_offsets[:, None] * stride_kp + d_offsets[None, :] * stride_kd
            load_mask = page_mask[:, None] & d_mask[None, :]

            # max(q*k_min, q*k_max) needs only one bound: k_max for a
            # non-negative q component and k_min for a negative one. Select
            # the address before loading to halve metadata traffic while
            # preserving the exact arithmetic used for the chosen product.
            bound_ptrs = tl.where(q_chunk[None, :] >= 0.0, kmax_ptrs, kmin_ptrs)
            bound = tl.load(bound_ptrs, mask=load_mask, other=0.0).to(tl.float32)
            upper = q_chunk[None, :] * bound
            acc += tl.sum(upper, axis=1)

        acc = acc * scale

        if HAS_ALIBI:
            bonus_ptrs = Alibi_Bonus + pid_kv_head * stride_ah + page_offsets * stride_ap
            acc = acc + tl.load(bonus_ptrs, mask=page_mask, other=0.0).to(tl.float32)

        out_base = Out + pid_batch * stride_ob + q_head_idx * stride_oh
        tl.store(out_base + page_offsets * stride_op, acc, mask=page_mask)


def triton_estimate_page_criticality(
    q: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
    apply_scaling: bool = False,
    alibi_slopes: torch.Tensor = None,
    q_pos: int = 0,
    page_size: int = 16,
    seq_len: int = 0,
    exclude_last_page: bool = False,
) -> torch.Tensor:
    """
    Triton-accelerated page criticality estimation.

    Computes upper-bound attention scores per page:
        score[h, p] = sum_d max(q[h,d] * k_min[kv_h, p, d], q[h,d] * k_max[kv_h, p, d])

    When alibi_slopes is provided, a pre-computed [num_heads, num_pages] bonus tensor
    is passed into the kernel and added per-entry:
        bonus[h, p] = alibi_slopes[kv_h] * (min((p+1)*page_size-1, seq_len-1) - q_pos)

    Args:
        q: [batch, num_heads, 1, head_dim]
        k_min: [batch, num_kv_heads, num_pages, head_dim]
        k_max: [batch, num_kv_heads, num_pages, head_dim]
        apply_scaling: multiply by 1/sqrt(head_dim) if True.
        alibi_slopes: [num_kv_heads] float32 raw slopes.
        q_pos: absolute position of the query token (seq_len - 1).
        page_size: tokens per page.
        seq_len: full sequence length (used to clamp page_last_pos).
        exclude_last_page: drop the in-progress last page from scoring.

    Returns:
        page_scores: [batch, num_heads, num_pages]
    """
    batch, num_heads, _, head_dim = q.shape
    _, num_kv_heads, stored_pages, _ = k_min.shape
    num_pages = stored_pages - int(exclude_last_page)
    if num_pages < 0:
        raise ValueError("cannot exclude the last page from an empty cache")
    gqa_group = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim) if apply_scaling else 1.0

    q = q.contiguous()
    k_min = k_min.contiguous()
    k_max = k_max.contiguous()

    if num_pages == 0:
        return torch.empty(batch, num_heads, 0, device=q.device, dtype=q.dtype)

    has_alibi = alibi_slopes is not None
    if has_alibi:
        # Pre-compute [num_kv_heads, num_pages] bonus outside the kernel.
        # bonus[kv_h, p] = slope[kv_h] * (page_last_pos[p] - q_pos)
        page_arange = torch.arange(num_pages, device=q.device, dtype=torch.float32)
        page_last_pos = torch.clamp((page_arange + 1) * page_size - 1, max=seq_len - 1)
        alibi_bonus = alibi_slopes.float().unsqueeze(1) * (page_last_pos - q_pos)
        alibi_bonus = alibi_bonus.contiguous()
    else:
        alibi_bonus = k_min  # dummy pointer, never dereferenced (HAS_ALIBI=False)

    out = torch.empty(batch, num_heads, num_pages, device=q.device, dtype=q.dtype)

    grid = lambda META: (
        batch,
        num_kv_heads,
        triton.cdiv(num_pages, META['BLOCK_PAGES']),
    )

    _page_criticality_kernel[grid](
        q, k_min, k_max, out, alibi_bonus,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_min.stride(0), k_min.stride(1), k_min.stride(2), k_min.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        alibi_bonus.stride(0), alibi_bonus.stride(1),
        num_pages, head_dim, scale,
        GQA_GROUP=gqa_group,
        HAS_ALIBI=has_alibi,
    )

    return out
