"""Top-k page selection: pick the highest-criticality remote pages, then
force-append the tail page.

The budget ``topk_pages`` counts the TOTAL attended pages, tail included:

    remote_k = max(0, min(topk_pages - 1, num_full_historical_pages))
    selected = topk(remote_k) + tail

``selected_pages`` holds LOGICAL page indices (positional meaning is what the
sparse decoder needs for ALiBi); unused slots are padded with -1. The decoder
resolves logical -> block_table[req, logical] -> physical block itself.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _finalize_selection(
    TOPK_IDX, SEQ_LENS, SELECTED, SEL_LENS,
    stride_tb: tl.constexpr, stride_th: tl.constexpr,
    stride_sb: tl.constexpr, stride_sh: tl.constexpr,
    stride_lb: tl.constexpr,
    N_HEADS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    REMOTE_K: tl.constexpr, TOPK_PAGES: tl.constexpr,
    SLOTS_POW2: tl.constexpr,
):
    """Per (request, head): copy the valid top-k picks, append the tail page,
    pad with -1, and record how many tokens the candidate set holds."""
    req = tl.program_id(0)
    head = tl.program_id(1)
    seq_len = tl.load(SEQ_LENS + req).to(tl.int32)
    tail = (seq_len - 1) // BLOCK_SIZE
    n_full = seq_len // BLOCK_SIZE
    # Pages the estimator was allowed to score: complete AND before the tail.
    n_scorable = tl.minimum(n_full, tail)
    n_remote = tl.minimum(REMOTE_K, n_scorable)

    slots = tl.arange(0, SLOTS_POW2)
    picks = tl.load(TOPK_IDX + req * stride_tb + head * stride_th + slots,
                    mask=slots < REMOTE_K, other=0).to(tl.int32)
    # slot < n_remote -> a real pick; slot == n_remote -> the forced tail;
    # beyond -> padding.
    value = tl.where(slots < n_remote, picks,
                     tl.where(slots == n_remote, tail, -1))
    tl.store(SELECTED + req * stride_sb + head * stride_sh + slots, value,
             mask=slots < TOPK_PAGES)

    # Candidate-set size: n_remote full pages plus the tail's real length.
    tail_len = seq_len - tail * BLOCK_SIZE
    tl.store(SEL_LENS + req * stride_lb + head,
             n_remote * BLOCK_SIZE + tail_len)


def select_pages(
    page_scores: torch.Tensor,   # (B, H_q, max_pages) fp32, -inf where unscorable
    seq_lens: torch.Tensor,      # (B,) int32
    topk_pages: int,
    block_size: int,
    topk_vals: torch.Tensor,     # (B, H_q, remote_k) fp32 scratch
    topk_idx: torch.Tensor,      # (B, H_q, remote_k) int64 scratch
    selected: torch.Tensor,      # (B, H_q, topk_pages) int32 out
    sel_lens: torch.Tensor,      # (B, H_q) int32 out
):
    """Fill ``selected`` with logical page indices and ``sel_lens`` with the
    number of candidate tokens per (request, head)."""
    batch, nheads, _ = page_scores.shape
    remote_k = topk_pages - 1
    # torch.topk into persistent buffers; a fused segmented top-k is a later
    # optimisation if profiling justifies it.
    #
    # sorted=True is load-bearing, not cosmetic: _finalize_selection keeps the
    # leading n_remote slots, which is only the best n_remote picks if the
    # results are ranked. With sorted=False torch returns them in arbitrary
    # order and short sequences would keep -inf pages over real ones.
    torch.topk(page_scores, remote_k, dim=-1, sorted=True,
               out=(topk_vals, topk_idx))
    _finalize_selection[(batch, nheads)](
        topk_idx, seq_lens, selected, sel_lens,
        stride_tb=topk_idx.stride(0), stride_th=topk_idx.stride(1),
        stride_sb=selected.stride(0), stride_sh=selected.stride(1),
        stride_lb=sel_lens.stride(0),
        N_HEADS=nheads, BLOCK_SIZE=block_size,
        REMOTE_K=remote_k, TOPK_PAGES=topk_pages,
        SLOTS_POW2=triton.next_power_of_2(topk_pages), num_warps=1,
    )
    return selected, sel_lens
