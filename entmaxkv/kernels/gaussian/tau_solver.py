"""One-launch bracketed Gaussian tau solvers.

Both solve the distributional entmax constraint ``sum_i n_i E[g_alpha] = 1`` in
the (alpha-1)-scaled tau domain, where ``g_alpha(s) = [(alpha-1)s - tau]_+ **
(1/(alpha-1))``. Closed-form truncated moments cover alpha in {1.5, 2, 4/3};
any other alpha uses the point-mass approximation ``E ~ [mu_y]_+ ** beta`` that
entmaxkv has always used as its generic fallback.

Page and token counts are runtime values, so the solve never recompiles as the
context grows.
"""

import torch
import triton
import triton.language as tl


def _is_closed_form(alpha: float) -> bool:
    return alpha in (1.5, 2.0) or abs(alpha - 4.0 / 3.0) < 1e-12


@triton.jit
def _cdf(x):
    return 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _pdf(x):
    return tl.exp(-0.5 * x * x) * 0.3989422804014327


@triton.jit
def _pow_pos(x, exponent: tl.constexpr):
    """x ** exponent for x >= 0, with 0 ** anything == 0."""
    return tl.where(x > 0.0, tl.exp2(tl.log2(tl.maximum(x, 1.0e-30)) * exponent),
                    0.0)


@triton.jit
def _moment_terms(tau, mean, std, alpha: tl.constexpr):
    """E[g_alpha(S)] and its first two tau-derivatives for S ~ N(mean, std^2)."""
    a: tl.constexpr = alpha - 1.0
    sigma_y = tl.maximum(a * std, 1.0e-10)
    mu_y = a * mean - tau
    t = mu_y / sigma_y
    phi, cdf = _pdf(t), _cdf(t)
    m0 = cdf
    m1 = mu_y * cdf + sigma_y * phi
    if alpha == 1.5:
        m2 = (mu_y * mu_y + sigma_y * sigma_y) * cdf + mu_y * sigma_y * phi
        return m2, -2.0 * m1, 2.0 * m0
    elif alpha == 2.0:
        return m1, -m0, phi / sigma_y
    elif alpha == 1.3333333333333333:
        m2 = (mu_y * mu_y + sigma_y * sigma_y) * cdf + mu_y * sigma_y * phi
        m3 = ((mu_y * mu_y * mu_y + 3.0 * mu_y * sigma_y * sigma_y) * cdf
              + (mu_y * mu_y * sigma_y + 2.0 * sigma_y * sigma_y * sigma_y)
              * phi)
        return m3, -3.0 * m2, 6.0 * m1
    else:
        beta: tl.constexpr = 1.0 / a
        expectation = _pow_pos(mu_y, beta)
        first = -beta * _pow_pos(mu_y, beta - 1.0)
        second = beta * (beta - 1.0) * _pow_pos(mu_y, beta - 2.0)
        return expectation, first, second


@triton.jit
def _candidate(tau, lo, hi, f, df, ddf):
    midpoint = 0.5 * (lo + hi)
    denom = 2.0 * df * df - f * ddf
    denom_safe = tl.where(tl.abs(denom) > 1.0e-18, denom, 1.0)
    newton = f / tl.minimum(df, -1.0e-12)
    halley = 2.0 * f * df / denom_safe
    step = tl.where(tl.abs(denom) > 1.0e-18, halley, newton)
    value = tau - step
    valid = ((value == value) & (tl.abs(value) < 1.0e20)
             & (value > lo) & (value < hi))
    return tl.where(valid, value, midpoint)


@triton.jit
def _single_terms(tau, mean, std, count, alpha: tl.constexpr):
    e, de, dde = _moment_terms(tau, mean, std, alpha)
    return count * e - 1.0, count * de, count * dde


@triton.jit
def _solve_single_gaussian(tau_mean, tau_std, count, alpha: tl.constexpr,
                           iterations: tl.constexpr):
    a: tl.constexpr = alpha - 1.0
    std = tl.maximum(tau_std, 1.0e-6)
    radius = a * std * (tl.sqrt(2.0 * tl.log(count + 1.0)) + 8.0)
    lo, hi = a * tau_mean - radius, a * tau_mean + radius
    tau = 0.5 * (lo + hi)
    for iteration in range(iterations):
        f, df, ddf = _single_terms(tau, tau_mean, std, count, alpha)
        value = _candidate(tau, lo, hi, f, df, ddf)
        f_value, df_value, ddf_value = _single_terms(
            value, tau_mean, std, count, alpha)
        lo = tl.where(f_value > 0.0, value, lo)
        hi = tl.where(f_value <= 0.0, value, hi)
        tau = tl.where(tl.abs(f_value) <= 1.0e-6, value, 0.5 * (lo + hi))
    return tau


@triton.jit
def _single_kernel(MEAN, STD, TOKEN_COUNTS, OUT, total,
                   alpha: tl.constexpr, iterations: tl.constexpr,
                   BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    mean = tl.load(MEAN + offsets, mask=mask, other=0.0).to(tl.float32)
    std = tl.load(STD + offsets, mask=mask, other=1.0).to(tl.float32)
    count = tl.maximum(tl.load(TOKEN_COUNTS + offsets, mask=mask,
                               other=1).to(tl.float32), 1.0)
    tau = _solve_single_gaussian(mean, std, count, alpha, iterations)
    tl.store(OUT + offsets, tau, mask=mask)


@triton.jit
def _mixture_tile_terms(tau, mean, std, active, page_size: tl.constexpr,
                        alpha: tl.constexpr):
    e, de, dde = _moment_terms(tau, mean, std, alpha)
    weight = tl.where(active, page_size + 0.0, 0.0)
    return (tl.sum(weight * e, axis=0), tl.sum(weight * de, axis=0),
            tl.sum(weight * dde, axis=0))


@triton.jit(do_not_specialize=["pages"])
def _mixture_kernel(MU, SIGMA, HIST_PAGES, OUT,
                    ACTIVE_HEADS, HAS_ACTIVE_HEADS: tl.constexpr, pages,
                    heads: tl.constexpr, row_stride: tl.constexpr,
                    page_size: tl.constexpr, alpha: tl.constexpr,
                    iterations: tl.constexpr, BLOCK_PAGES: tl.constexpr):
    row = tl.program_id(0)
    head = row % heads
    if HAS_ACTIVE_HEADS:
        if tl.load(ACTIVE_HEADS + head) == 0:
            tl.store(OUT + row, 0.0)
            return
    a: tl.constexpr = alpha - 1.0
    offsets = tl.arange(0, BLOCK_PAGES)
    historical = tl.load(HIST_PAGES + row).to(tl.int32)
    total = tl.maximum(historical * page_size, 1).to(tl.float32)

    # Keep only BLOCK_PAGES page moments live at once, independently of the
    # context length.
    max_y = -float("inf")
    max_std = 0.0
    for tile_start in tl.range(0, pages, BLOCK_PAGES, num_stages=1,
                               loop_unroll_factor=1):
        page = tile_start + offsets
        active = page < historical
        base = row * row_stride + page
        mean = tl.load(MU + base, mask=page < pages,
                       other=0.0).to(tl.float32)
        std = tl.maximum(tl.load(SIGMA + base, mask=page < pages,
                                 other=1.0).to(tl.float32), 1.0e-6)
        max_y = tl.maximum(
            max_y,
            tl.max(tl.where(active, a * mean, -float("inf")), axis=0))
        max_std = tl.maximum(
            max_std, tl.max(tl.where(active, a * std, 0.0), axis=0))
    max_y = tl.where(historical > 0, max_y, 0.0)
    max_std = tl.maximum(max_std, 1.0e-6)
    radius = max_std * (tl.sqrt(2.0 * tl.log(total + 1.0)) + 8.0)
    lo, hi = max_y - radius, max_y + radius
    tau = 0.5 * (lo + hi)
    for iteration in range(iterations):
        mass, df, ddf = 0.0, 0.0, 0.0
        for tile_start in tl.range(0, pages, BLOCK_PAGES, num_stages=1,
                                   loop_unroll_factor=1):
            page = tile_start + offsets
            active = page < historical
            base = row * row_stride + page
            mean = tl.load(MU + base, mask=page < pages,
                           other=0.0).to(tl.float32)
            std = tl.maximum(
                tl.load(SIGMA + base, mask=page < pages,
                        other=1.0).to(tl.float32), 1.0e-6)
            tile_mass, tile_df, tile_ddf = _mixture_tile_terms(
                tau, mean, std, active, page_size, alpha)
            mass += tile_mass
            df += tile_df
            ddf += tile_ddf
        f = mass - 1.0
        # Update the bracket from the point just evaluated, then propose the
        # next point inside that bracket: one page traversal per iteration.
        lo = tl.where(f > 0.0, tau, lo)
        hi = tl.where(f <= 0.0, tau, hi)
        value = _candidate(tau, lo, hi, f, df, ddf)
        tau = tl.where(tl.abs(f) <= 1.0e-6, tau, value)
    tl.store(OUT + row, tl.where(historical > 0, tau, 0.0))


@triton.jit
def _mixture_init_tiles(
    MU, SIGMA, HIST_PAGES, ACTIVE_HEADS, PARTIAL_MAX_Y, PARTIAL_MAX_STD,
    pages, heads: tl.constexpr, row_stride: tl.constexpr,
    TILE_STRIDE: tl.constexpr, TILE_PAGES: tl.constexpr, alpha: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    head = row % heads
    out = row * TILE_STRIDE + tile
    if tl.load(ACTIVE_HEADS + head) == 0:
        tl.store(PARTIAL_MAX_Y + out, -float("inf"))
        tl.store(PARTIAL_MAX_STD + out, 0.0)
        return
    a: tl.constexpr = alpha - 1.0
    page = tile * TILE_PAGES + tl.arange(0, TILE_PAGES)
    historical = tl.load(HIST_PAGES + row).to(tl.int32)
    active = (page < historical) & (page < pages)
    base = row * row_stride + page
    mean = tl.load(MU + base, mask=page < pages,
                   other=0.0).to(tl.float32)
    std = tl.maximum(tl.load(SIGMA + base, mask=page < pages,
                             other=1.0).to(tl.float32), 1.0e-6)
    tl.store(PARTIAL_MAX_Y + out,
             tl.max(tl.where(active, a * mean, -float("inf")), axis=0))
    tl.store(PARTIAL_MAX_STD + out,
             tl.max(tl.where(active, a * std, 0.0), axis=0))


@triton.jit
def _mixture_init_reduce(
    HIST_PAGES, ACTIVE_HEADS, PARTIAL_MAX_Y, PARTIAL_MAX_STD,
    TAU, TAU_LO, TAU_HI, tiles,
    heads: tl.constexpr, page_size: tl.constexpr, TILE_STRIDE: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % heads
    if tl.load(ACTIVE_HEADS + head) == 0:
        tl.store(TAU + row, 0.0)
        tl.store(TAU_LO + row, 0.0)
        tl.store(TAU_HI + row, 0.0)
        return
    tile = tl.arange(0, BLOCK_TILES)
    mask = tile < tiles
    base = row * TILE_STRIDE + tile
    max_y = tl.max(tl.load(PARTIAL_MAX_Y + base, mask=mask,
                           other=-float("inf")), axis=0)
    max_std = tl.maximum(tl.max(tl.load(
        PARTIAL_MAX_STD + base, mask=mask, other=0.0), axis=0), 1.0e-6)
    historical = tl.load(HIST_PAGES + row).to(tl.int32)
    total = tl.maximum(historical * page_size, 1).to(tl.float32)
    max_y = tl.where(historical > 0, max_y, 0.0)
    radius = max_std * (tl.sqrt(2.0 * tl.log(total + 1.0)) + 8.0)
    lo, hi = max_y - radius, max_y + radius
    tl.store(TAU + row, 0.5 * (lo + hi))
    tl.store(TAU_LO + row, lo)
    tl.store(TAU_HI + row, hi)


@triton.jit
def _mixture_accumulate_tiles(
    MU, SIGMA, HIST_PAGES, ACTIVE_HEADS, TAU,
    PARTIAL_MASS, PARTIAL_DF, PARTIAL_DDF,
    pages, heads: tl.constexpr, row_stride: tl.constexpr,
    page_size: tl.constexpr, TILE_STRIDE: tl.constexpr,
    TILE_PAGES: tl.constexpr, alpha: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    head = row % heads
    out = row * TILE_STRIDE + tile
    if tl.load(ACTIVE_HEADS + head) == 0:
        tl.store(PARTIAL_MASS + out, 0.0)
        tl.store(PARTIAL_DF + out, 0.0)
        tl.store(PARTIAL_DDF + out, 0.0)
        return
    page = tile * TILE_PAGES + tl.arange(0, TILE_PAGES)
    historical = tl.load(HIST_PAGES + row).to(tl.int32)
    active = (page < historical) & (page < pages)
    base = row * row_stride + page
    mean = tl.load(MU + base, mask=page < pages,
                   other=0.0).to(tl.float32)
    std = tl.maximum(tl.load(SIGMA + base, mask=page < pages,
                             other=1.0).to(tl.float32), 1.0e-6)
    mass, df, ddf = _mixture_tile_terms(
        tl.load(TAU + row), mean, std, active, page_size, alpha)
    tl.store(PARTIAL_MASS + out, mass)
    tl.store(PARTIAL_DF + out, df)
    tl.store(PARTIAL_DDF + out, ddf)


@triton.jit
def _mixture_reduce_update(
    ACTIVE_HEADS, TAU, TAU_LO, TAU_HI,
    PARTIAL_MASS, PARTIAL_DF, PARTIAL_DDF, tiles,
    heads: tl.constexpr, TILE_STRIDE: tl.constexpr, BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % heads
    if tl.load(ACTIVE_HEADS + head) == 0:
        return
    tile = tl.arange(0, BLOCK_TILES)
    mask = tile < tiles
    base = row * TILE_STRIDE + tile
    mass = tl.sum(tl.load(PARTIAL_MASS + base, mask=mask, other=0.0), axis=0)
    df = tl.sum(tl.load(PARTIAL_DF + base, mask=mask, other=0.0), axis=0)
    ddf = tl.sum(tl.load(PARTIAL_DDF + base, mask=mask, other=0.0), axis=0)
    tau = tl.load(TAU + row)
    lo, hi = tl.load(TAU_LO + row), tl.load(TAU_HI + row)
    f = mass - 1.0
    lo = tl.where(f > 0.0, tau, lo)
    hi = tl.where(f <= 0.0, tau, hi)
    value = _candidate(tau, lo, hi, f, df, ddf)
    tau = tl.where(tl.abs(f) <= 1.0e-6, tau, value)
    tl.store(TAU + row, tau)
    tl.store(TAU_LO + row, lo)
    tl.store(TAU_HI + row, hi)


def solve_single_gaussian(mean: torch.Tensor, std: torch.Tensor,
                          token_counts: torch.Tensor, alpha: float = 1.5,
                          iterations: int = 40,
                          out: torch.Tensor | None = None):
    """Per-row single-Gaussian tau; every input is a contiguous [B, H]."""
    mean, std = mean.contiguous(), std.contiguous()
    token_counts = token_counts.contiguous()
    if out is None:
        out = torch.empty_like(mean, dtype=torch.float32)
    total = mean.numel()
    block = min(1024, triton.next_power_of_2(total))
    _single_kernel[(triton.cdiv(total, block),)](
        mean, std, token_counts, out, total, alpha=float(alpha),
        iterations=iterations, BLOCK=block, num_warps=4)
    return out


def solve_page_mixture(mu: torch.Tensor, sigma: torch.Tensor,
                       historical_pages: torch.Tensor, page_size: int,
                       alpha: float = 1.5, iterations: int = 20,
                       out: torch.Tensor | None = None,
                       active_heads: torch.Tensor | None = None,
                       workspace: dict | None = None,
                       n_pages: int | None = None):
    """Page-mixture tau over ``mu[..., :n_pages]`` (host int, live width).

    ``mu``/``sigma`` must be [B, H, width] with a unit last stride; the row
    stride is the only shape-derived kernel constant.
    """
    batch, heads, width = mu.shape
    if mu.stride(2) != 1 or sigma.stride() != mu.stride():
        raise ValueError("mu and sigma must share strides with unit last dim")
    if mu.stride(0) != heads * mu.stride(1):
        raise ValueError("mu rows must be uniformly strided")
    row_stride = mu.stride(1)
    pages = width if n_pages is None else int(n_pages)
    historical_pages = historical_pages.to(device=mu.device,
                                           dtype=torch.int32)
    if historical_pages.shape == (batch,):
        historical_pages = historical_pages[:, None].expand(batch, heads)
    if historical_pages.shape != (batch, heads):
        raise ValueError(
            "historical_pages must have shape [batch] or [batch, heads]")
    historical_pages = historical_pages.contiguous()
    if out is None:
        out = torch.empty((batch, heads), device=mu.device,
                          dtype=torch.float32)
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if active_heads is not None:
        active_heads = active_heads.to(device=mu.device, dtype=torch.bool)
        if active_heads.shape != (heads,):
            raise ValueError("active_heads must have shape [heads]")
        active_heads = active_heads.contiguous()
    # Short contexts are faster as one persistent program per row. Long
    # contexts expose the page-tile dimension as independent CTAs so the solve
    # can occupy the whole GPU instead of launching only batch*heads blocks.
    if pages <= 4096:
        # Sized from the buffer width, not the live width, so the tile (a
        # kernel constant) never changes as the context grows.
        block_pages = min(1024, triton.next_power_of_2(width))
        active_ptr = active_heads if active_heads is not None else historical_pages
        _mixture_kernel[(batch * heads,)](
            mu, sigma, historical_pages, out, active_ptr,
            HAS_ACTIVE_HEADS=active_heads is not None, pages=pages,
            heads=heads, row_stride=row_stride, page_size=page_size,
            alpha=float(alpha), iterations=iterations,
            BLOCK_PAGES=block_pages, num_warps=4)
        return out

    if active_heads is None:
        active_heads = torch.ones((heads,), device=mu.device, dtype=torch.bool)
    if workspace is None:
        workspace = {}
    tile_pages = 256
    tiles = triton.cdiv(pages, tile_pages)
    # Partials are sized by the buffer width, not the live width, so neither
    # the buffers nor the kernel constants change as the context grows.
    tile_stride = triton.cdiv(width, tile_pages)
    rows = batch * heads

    def buffer(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = workspace.get(name)
        if (value is None or value.shape != shape or value.device != mu.device
                or value.dtype != torch.float32):
            value = torch.empty(shape, device=mu.device, dtype=torch.float32)
            workspace[name] = value
        return value

    partial_shape = (rows, tile_stride)
    partial_max_y = buffer("partial_max_y", partial_shape)
    partial_max_std = buffer("partial_max_std", partial_shape)
    partial_mass = buffer("partial_mass", partial_shape)
    partial_df = buffer("partial_df", partial_shape)
    partial_ddf = buffer("partial_ddf", partial_shape)
    tau_lo = buffer("tau_lo", (rows,))
    tau_hi = buffer("tau_hi", (rows,))
    tau_flat = out.view(-1)
    block_tiles = triton.next_power_of_2(tile_stride)
    reduce_warps = 4 if block_tiles < 128 else 8

    _mixture_init_tiles[(rows, tiles)](
        mu, sigma, historical_pages, active_heads,
        partial_max_y, partial_max_std, pages,
        heads=heads, row_stride=row_stride, TILE_STRIDE=tile_stride,
        TILE_PAGES=tile_pages, alpha=float(alpha), num_warps=4)
    _mixture_init_reduce[(rows,)](
        historical_pages, active_heads, partial_max_y, partial_max_std,
        tau_flat, tau_lo, tau_hi, tiles,
        heads=heads, page_size=page_size, TILE_STRIDE=tile_stride,
        BLOCK_TILES=block_tiles, num_warps=reduce_warps)
    for _ in range(iterations):
        _mixture_accumulate_tiles[(rows, tiles)](
            mu, sigma, historical_pages, active_heads, tau_flat,
            partial_mass, partial_df, partial_ddf, pages,
            heads=heads, row_stride=row_stride, page_size=page_size,
            TILE_STRIDE=tile_stride, TILE_PAGES=tile_pages,
            alpha=float(alpha), num_warps=4)
        _mixture_reduce_update[(rows,)](
            active_heads, tau_flat, tau_lo, tau_hi,
            partial_mass, partial_df, partial_ddf, tiles,
            heads=heads, TILE_STRIDE=tile_stride, BLOCK_TILES=block_tiles,
            num_warps=reduce_warps)
    return out
