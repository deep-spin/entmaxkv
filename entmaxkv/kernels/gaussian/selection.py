"""Fused Gaussian threshold selection and page packing.

A page p is selected when its estimated per-page maximum can cross the
(safety-lowered) Gaussian threshold:

    page_upper = mean_p(mu_p + 1.5 sigma_p)
    delta      = max(z_safety * sigma_global,
                     excess * max(tau_hat/(alpha-1) - page_upper, 0))
    tau_floor  = tau_hat - (alpha-1) * delta
    select p  <=>  (alpha-1) * (mu_p + z_page * sigma_p) > tau_floor

Output rows hold ascending logical pages, then the mandatory tail page, then
-1 up to the full buffer width, so both the Gaussian decoder (which reads only
``sel_lens`` tokens) and the fixed-width selected-page decoder can consume
them. Pages are processed in fixed tiles with a running rank offset, so the
live page count is a runtime value and never triggers a recompile.
"""

from statistics import NormalDist

import torch
import triton
import triton.language as tl

SELECT_TILE_PAGES = 1024


@triton.jit(do_not_specialize=["n_pages"])
def _select_and_pack_kernel(
    MU, SIGMA, TAU, SIGMA_GLOBAL, SEQ_LENS, OUT, COUNTS, SEL_LENS,
    n_pages, out_width,
    stride_mb: tl.constexpr, stride_mh: tl.constexpr,
    stride_tb: tl.constexpr, stride_sb: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr,
    block_size: tl.constexpr, alpha_minus_one: tl.constexpr,
    safety_margin_z: tl.constexpr,
    threshold_excess_margin_fraction: tl.constexpr,
    z_page: tl.constexpr, BLOCK_PAGES: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_PAGES)
    seq_len = tl.load(SEQ_LENS + request).to(tl.int32)
    tail = (seq_len - 1) // block_size
    base = MU + request * stride_mb + head * stride_mh
    sigma_base = SIGMA + request * stride_mb + head * stride_mh
    tau = tl.load(TAU + request * stride_tb + head).to(tl.float32)
    sigma_global = tl.load(
        SIGMA_GLOBAL + request * stride_sb + head).to(tl.float32)

    # Pass 1: mean page upper estimate over historical pages.
    upper_sum = 0.0
    for start in range(0, n_pages, BLOCK_PAGES):
        pages = start + offsets
        historical = pages < tail
        mu = tl.load(base + pages, mask=historical, other=0.0).to(tl.float32)
        sigma = tl.load(sigma_base + pages, mask=historical,
                        other=0.0).to(tl.float32)
        upper_sum += tl.sum(tl.where(historical, mu + 1.5 * sigma, 0.0),
                            axis=0)
    page_upper = upper_sum / tl.maximum(tail, 1).to(tl.float32)
    threshold_score = tau / alpha_minus_one
    sigma_delta = safety_margin_z * sigma_global
    excess_delta = threshold_excess_margin_fraction * tl.maximum(
        threshold_score - page_upper, 0.0)
    tau_floor = tau - tl.maximum(sigma_delta, excess_delta) * alpha_minus_one

    # Pass 2: select and pack.
    out_base = OUT + request * stride_ob + head * stride_oh
    count = 0
    for start in range(0, n_pages, BLOCK_PAGES):
        pages = start + offsets
        historical = pages < tail
        mu = tl.load(base + pages, mask=historical, other=0.0).to(tl.float32)
        sigma = tl.load(sigma_base + pages, mask=historical,
                        other=0.0).to(tl.float32)
        selected = historical & (
            alpha_minus_one * (mu + z_page * sigma) > tau_floor)
        selected_i32 = selected.to(tl.int32)
        ranks = count + tl.cumsum(selected_i32, axis=0) - 1
        tl.store(out_base + ranks, pages, mask=selected)
        count += tl.sum(selected_i32, axis=0)
    tl.store(out_base + count, tail)
    total = count + 1
    # Pass 3: pad. Written only past the tail slot, so no address is stored by
    # two different threads (there is no intra-program ordering between them).
    for start in range(0, out_width, BLOCK_PAGES):
        slots = start + offsets
        tl.store(out_base + slots, -1, mask=(slots >= total) & (slots < out_width))
    tl.store(COUNTS + request * stride_tb + head, total)
    tail_tokens = (seq_len - 1) % block_size + 1
    tl.store(SEL_LENS + request * stride_tb + head,
             (total - 1) * block_size + tail_tokens)


def select_and_pack(
    mu_pages: torch.Tensor,      # (B, H, width) fp32
    sigma_pages: torch.Tensor,
    tau_hat: torch.Tensor,       # (B, H) fp32
    sigma_global: torch.Tensor,  # (B, H) fp32
    seq_lens: torch.Tensor,      # (B,) int32
    block_size: int,
    safety_margin_z: float,
    max_quantile: float,
    threshold_excess_margin_fraction: float,
    alpha: float = 1.5,
    out: torch.Tensor | None = None,
    counts: torch.Tensor | None = None,
    sel_lens: torch.Tensor | None = None,
    n_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select and pack ascending logical pages in one Triton launch.

    ``n_pages`` (host int) is the live width; ``out`` is filled across its
    whole width.
    """
    if mu_pages.shape != sigma_pages.shape or mu_pages.ndim != 3:
        raise ValueError("mu_pages and sigma_pages must share [B, H, pages]")
    if mu_pages.stride() != sigma_pages.stride() or mu_pages.stride(2) != 1:
        raise ValueError("mu_pages and sigma_pages must share unit-last strides")
    batch, heads, width = mu_pages.shape
    if n_pages is None:
        n_pages = width
    shape = (batch, heads)
    tau = tau_hat.to(device=mu_pages.device, dtype=torch.float32).contiguous()
    sigma_global = sigma_global.to(
        device=mu_pages.device, dtype=torch.float32).contiguous()
    if out is None:
        out = torch.empty(shape + (width,), device=mu_pages.device,
                          dtype=torch.int32)
    if counts is None:
        counts = torch.empty(shape, device=mu_pages.device, dtype=torch.int32)
    if sel_lens is None:
        sel_lens = torch.empty(shape, device=mu_pages.device,
                               dtype=torch.int32)
    if out.stride(2) != 1 or out.shape[2] <= n_pages - 1:
        raise ValueError("out must be unit-strided and hold every live page")
    per_sample_prob = max_quantile ** (1.0 / block_size)
    z_page = NormalDist().inv_cdf(per_sample_prob)
    _select_and_pack_kernel[(batch, heads)](
        mu_pages, sigma_pages, tau, sigma_global, seq_lens, out, counts,
        sel_lens, n_pages, out.shape[2],
        stride_mb=mu_pages.stride(0), stride_mh=mu_pages.stride(1),
        stride_tb=tau.stride(0), stride_sb=sigma_global.stride(0),
        stride_ob=out.stride(0), stride_oh=out.stride(1),
        block_size=block_size, alpha_minus_one=float(alpha - 1.0),
        safety_margin_z=float(safety_margin_z),
        threshold_excess_margin_fraction=float(
            threshold_excess_margin_fraction), z_page=float(z_page),
        BLOCK_PAGES=SELECT_TILE_PAGES, num_warps=8)
    return out, counts, sel_lens
