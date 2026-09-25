"""Correctness of the paged cache, selectors and decoders against float64
oracles.

Covers GQA (H_q = 2 H_kv), ALiBi + NoPE heads (mixed zero/nonzero slopes),
and ragged batches. Requires CUDA.
"""

import math

import pytest
import torch

from entmaxkv.attention_gaussian import (
    sparse_attention_decode_gaussian_aware_entmax,
)
from entmaxkv.attention_topk import sparse_attention_decode_paged
from entmaxkv.kernels.gaussian.tau_solver import solve_page_mixture
from entmaxkv.kernels.tau_solver_page_mixture import (
    solve_for_tau_hat_page_gaussian_mixture,
)
from entmaxkv.kv_cache import PagedKVCache
from entmaxkv.selectors import GaussianPageSelector, TopKPageSelector
from entmaxkv.tau_solver import solve_for_tau_hat_single_gaussian

DEVICE = "cuda"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

B, HQ, HKV, D, PS = 2, 4, 2, 32, 16
LENS = [700, 437]
SLOPES = [0.25, 0.0, 0.0625, 0.0]   # ALiBi on heads 0, 2; NoPE on 1, 3


# ---- oracles ---------------------------------------------------------------

def entmax_bisect(scores, alpha, iters=200):
    """float64 entmax by bisection on tau in [max - 1, max]."""
    s = (alpha - 1) * scores.double()
    m = s.max(-1, keepdim=True).values
    lo, hi = m - 1, m.clone()
    for _ in range(iters):
        mid = (lo + hi) / 2
        mass = (s - mid).clamp(min=0).pow(1 / (alpha - 1)).sum(-1, keepdim=True)
        lo = torch.where(mass > 1, mid, lo)
        hi = torch.where(mass > 1, hi, mid)
    return (s - (lo + hi) / 2).clamp(min=0).pow(1 / (alpha - 1))


def oracle(q3, cache, slopes, alpha, keep=None):
    """Dense float64 attention; ``keep[b, h, t]`` restricts the key set."""
    k, v = cache.dense()
    rep = q3.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, 1).double().cpu()
    v = v.repeat_interleave(rep, 1).double().cpu()
    q3, slopes = q3.double().cpu(), slopes.double().cpu()
    out = torch.zeros(q3.shape, dtype=torch.float64)
    for b, n in enumerate(cache.seq_lens_host):
        scores = (k[b, :, :n] @ q3[b][:, :, None])[..., 0] / math.sqrt(D)
        scores = scores + slopes[:, None] * (torch.arange(n) - (n - 1))[None]
        if keep is not None:
            scores = scores.masked_fill(~keep[b][:, :n].cpu(), -float("inf"))
        out[b] = (entmax_bisect(scores, alpha)[:, :, None] * v[b, :, :n]).sum(1)
    return out


def keep_from_selection(pages, sel_lens, cache):
    keep = torch.zeros(B, HQ, cache.max_seq_len, dtype=torch.bool)
    for b in range(B):
        for h in range(HQ):
            n_pages = -(-sel_lens[b, h].item() // PS)
            for p in pages[b, h, :n_pages].tolist():
                keep[b, h, p * PS:(p + 1) * PS] = True
    return keep


# ---- fixtures --------------------------------------------------------------

def make_cache(seed=0, **kwargs):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(B, HKV, max(LENS), D, generator=g)
    v = torch.randn(B, HKV, max(LENS), D, generator=g)
    cache = PagedKVCache(PS, **kwargs)
    cache.initialize(k.to(DEVICE), v.to(DEVICE), seq_lens=LENS)
    return cache


def make_step(seed=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, HQ, 1, D, generator=g).to(DEVICE)
    kn = torch.randn(B, HKV, 1, D, generator=g).to(DEVICE)
    vn = torch.randn(B, HKV, 1, D, generator=g).to(DEVICE)
    return q, kn, vn


def slopes():
    return torch.tensor(SLOPES, device=DEVICE)


# ---- cache -----------------------------------------------------------------

def test_metadata_tracks_appends_and_growth():
    cache = make_cache(max_seq_len=max(LENS) + 1)
    pages_before = cache.max_pages
    g = torch.Generator().manual_seed(5)
    appended = [[] for _ in range(B)]
    for _ in range(40):     # crosses page boundaries and forces a regrow
        kn = torch.randn(B, HKV, 1, D, generator=g).to(DEVICE)
        cache.append(kn, torch.zeros_like(kn))
        for b in range(B):
            appended[b].append(kn[b, :, 0].cpu())
    assert cache.max_pages > pages_before
    assert cache.seq_lens_host == tuple(n + 40 for n in LENS)
    assert cache.seq_lens.tolist() == list(cache.seq_lens_host)

    kd, _ = cache.dense()
    for b, n in enumerate(cache.seq_lens_host):
        tail = torch.stack(appended[b], 1)                  # [H, 40, D]
        assert torch.equal(kd[b, :, n - 40:n].cpu(), tail)
        for p in range(n // PS):                            # complete pages
            blk = cache.block_table[b, p].item()
            page = kd[b, :, p * PS:(p + 1) * PS].transpose(0, 1)
            assert torch.equal(cache.k_min[blk], page.min(0).values)
            assert torch.equal(cache.k_max[blk], page.max(0).values)
            torch.testing.assert_close(cache.k_mean[blk], page.mean(0),
                                       atol=1e-5, rtol=0)
            torch.testing.assert_close(
                cache.k_std[blk], page.std(0, unbiased=False),
                atol=1e-4, rtol=0)


def test_stats_are_opt_in():
    cache = make_cache(stats=("gaussian",))
    assert cache.k_min is None and cache.k_mean is not None


# ---- top-k -----------------------------------------------------------------

def criticality_oracle(q3, cache, b, h, n):
    g = h // (HQ // HKV)
    qs = q3[b, h].double().cpu() / math.sqrt(D)
    slope = SLOPES[h]
    scores = []
    for p in range(min((n - 1) // PS, n // PS)):
        blk = cache.block_table[b, p].item()
        bound = torch.maximum(qs * cache.k_min[blk, g].double().cpu(),
                              qs * cache.k_max[blk, g].double().cpu()).sum()
        best = min((p + 1) * PS - 1, n - 1) if slope >= 0 else p * PS
        scores.append(bound.item() + slope * (best - (n - 1)))
    return scores


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_topk_selection_and_decode(alpha):
    topk = 20   # >= 2 slots per partition
    cache = make_cache()
    q, kn, vn = make_step()
    out = torch.empty_like(q)
    sparse_attention_decode_paged(q, cache, kn, vn, out, topk_pages=topk,
                                  alibi_slopes=slopes(), alpha=alpha,
                                  niter=8)
    selector = cache.workspaces[("topk", topk)]
    ws = next(iter(selector._cache.values()))
    pages, sel_lens = ws["selected"].cpu(), ws["sel_lens"].cpu()
    q3 = q[:, :, 0]
    for b, n in enumerate(cache.seq_lens_host):
        tail = (n - 1) // PS
        for h in range(HQ):
            scores = criticality_oracle(q3, cache, b, h, n)
            best = sorted(range(len(scores)), key=lambda i: -scores[i])
            n_remote = min(topk - 1, len(scores))
            row = pages[b, h].tolist()
            assert sorted(row[:n_remote]) == sorted(best[:n_remote])
            assert row[n_remote] == tail
            assert all(x == -1 for x in row[n_remote + 1:])
            assert sel_lens[b, h] == n_remote * PS + n - tail * PS
    keep = keep_from_selection(pages, sel_lens, cache)
    ref = oracle(q3, cache, slopes(), alpha, keep)
    torch.testing.assert_close(out[:, :, 0].cpu().double(), ref, atol=1e-5,
                               rtol=0)


def test_topk_falls_back_to_exact_when_budget_covers_context():
    cache = make_cache()
    q, kn, vn = make_step()
    out = torch.empty_like(q)
    sparse_attention_decode_paged(q, cache, kn, vn, out, topk_pages=64,
                                  alibi_slopes=slopes(), alpha=1.5)
    assert ("topk", 64) not in cache.workspaces
    ref = oracle(q[:, :, 0], cache, slopes(), 1.5)
    torch.testing.assert_close(out[:, :, 0].cpu().double(), ref, atol=1e-5,
                               rtol=0)


def test_topk_budget_is_tail_inclusive():
    with pytest.raises(ValueError):
        TopKPageSelector(1, PS)


# ---- Gaussian --------------------------------------------------------------

@pytest.mark.parametrize("alpha", [1.5, 2.0, 4 / 3])
def test_gaussian_stats_and_tau(alpha):
    cache = make_cache()
    q3 = make_step()[0][:, :, 0]
    selector = GaussianPageSelector(PS, alpha)
    sel = selector.select(q3, cache.k_mean, cache.k_std, cache.block_table,
                          cache.seq_lens, slopes(), cache.live_pages)
    ws = next(iter(selector._cache.values()))
    for b, n in enumerate(cache.seq_lens_host):
        hist = (n - 1) // PS
        for h in range(HQ):
            g = h // (HQ // HKV)
            blocks = cache.block_table[b, :hist].long()
            qs = q3[b, h].double().cpu() / math.sqrt(D)
            mu = (cache.k_mean[blocks, g].double().cpu() * qs).sum(-1)
            # ALiBi enters the page means BEFORE the global moments.
            mu = mu + SLOPES[h] * (torch.arange(hist) * PS + (PS - 1) / 2
                                   - (n - 1)).double()
            var = (cache.k_std[blocks, g].double().cpu() ** 2 * qs ** 2).sum(-1)
            torch.testing.assert_close(ws["mu"][b, h, :hist].cpu().double(),
                                       mu, atol=1e-4, rtol=0)
            torch.testing.assert_close(ws["sigma"][b, h, :hist].cpu().double(),
                                       var.sqrt(), atol=1e-4, rtol=0)
            mean = mu.mean()
            sigma = (var.mean() + mu.var(unbiased=False)).sqrt()
            assert abs(ws["mean"][b, h].item() - mean.item()) < 1e-4
            assert abs(ws["sigma_global"][b, h].item() - sigma.item()) < 1e-4
            if SLOPES[h] == 0:   # NoPE head: single Gaussian
                ref = solve_for_tau_hat_single_gaussian(
                    mean.view(1, 1, 1), sigma.view(1, 1, 1), hist * PS,
                    alpha).item()
            else:                # ALiBi head: page mixture
                ref = solve_for_tau_hat_page_gaussian_mixture(
                    mu.view(1, -1), var.sqrt().view(1, -1),
                    page_counts=torch.full((1, hist), float(PS),
                                           dtype=torch.float64),
                    alpha=alpha).item()
            assert abs(sel.tau[b, h].item() - ref) < 1e-3 * max(1, abs(ref))


def test_gaussian_selection_rule():
    from statistics import NormalDist
    alpha, z, quantile, excess = 1.5, 0.3, 0.995, 0.1
    cache = make_cache()
    q3 = make_step()[0][:, :, 0]
    selector = GaussianPageSelector(PS, alpha, z, quantile, excess)
    sel = selector.select(q3, cache.k_mean, cache.k_std, cache.block_table,
                          cache.seq_lens, slopes(), cache.live_pages)
    ws = next(iter(selector._cache.values()))
    z_page = NormalDist().inv_cdf(quantile ** (1 / PS))
    a = alpha - 1
    for b, n in enumerate(cache.seq_lens_host):
        tail = (n - 1) // PS
        for h in range(HQ):
            mu = ws["mu"][b, h, :tail].cpu().double()
            sd = ws["sigma"][b, h, :tail].cpu().double()
            tau = sel.tau[b, h].item()
            upper = (mu + 1.5 * sd).mean().item()
            delta = max(z * ws["sigma_global"][b, h].item(),
                        excess * max(tau / a - upper, 0.0))
            floor = tau - delta * a
            expected = [p for p in range(tail)
                        if a * (mu[p] + z_page * sd[p]) > floor]
            row = sel.pages[b, h].tolist()
            count = len(expected) + 1
            assert row[:count] == expected + [tail]
            assert all(x == -1 for x in row[count:])
            assert sel.sel_lens[b, h] == len(expected) * PS + n - tail * PS


def test_tiled_mixture_solver_matches_reference():
    g = torch.Generator().manual_seed(3)
    mu = (torch.randn(2, 3, 5000, generator=g) * 0.5).to(DEVICE)
    sd = (torch.rand(2, 3, 5000, generator=g) * 0.3 + 0.05).to(DEVICE)
    hist = torch.tensor([[5000, 4100, 3000], [4500, 4999, 1]],
                        dtype=torch.int32, device=DEVICE)
    tau = solve_page_mixture(mu, sd, hist, PS, alpha=1.5, iterations=40)
    for b in range(2):
        for h in range(3):
            n = hist[b, h].item()
            ref = solve_for_tau_hat_page_gaussian_mixture(
                mu[b, h, :n].cpu().double().view(1, -1),
                sd[b, h, :n].cpu().double().view(1, -1),
                page_counts=torch.full((1, n), float(PS), dtype=torch.float64),
                alpha=1.5).item()
            assert abs(tau[b, h].item() - ref) < 1e-3


@pytest.mark.parametrize("tau_mode", ["exact", "corrected"])
def test_gaussian_decode_is_exact_on_selection(tau_mode):
    cache = make_cache()
    q, kn, vn = make_step()
    out = torch.empty_like(q)
    sparse_attention_decode_gaussian_aware_entmax(
        q, cache, kn, vn, out, alibi_slopes=slopes(), append_cache=True,
        tau_mode=tau_mode, niter=8)
    selector = cache.workspaces[("gaussian", 1.5, 0.0, 0.995, 0.1)]
    ws = next(iter(selector._cache.values()))
    keep = keep_from_selection(ws["selected"].cpu(), ws["sel_lens"].cpu(),
                               cache)
    ref = oracle(q[:, :, 0], cache, slopes(), 1.5, keep)
    torch.testing.assert_close(out[:, :, 0].cpu().double(), ref, atol=1e-5,
                               rtol=0)


def test_gaussian_fixed_mode_uses_tau_hat():
    cache = make_cache()
    q, kn, vn = make_step()
    out = torch.empty_like(q)
    sparse_attention_decode_gaussian_aware_entmax(
        q, cache, kn, vn, out, alibi_slopes=slopes(), append_cache=True,
        tau_mode="fixed")
    selector = cache.workspaces[("gaussian", 1.5, 0.0, 0.995, 0.1)]
    ws = next(iter(selector._cache.values()))
    decode_ws = cache.workspaces["gaussian_decode"]
    tau_used = next(iter(decode_ws._cache.values()))["tau"]
    torch.testing.assert_close(tau_used, ws["tau"])


# ---- compiled-kernel guards ------------------------------------------------

def _jit_cache_sizes():
    import triton
    from entmaxkv.kernels import (page_criticality, page_topk,
                                  selected_decode)
    from entmaxkv.kernels.gaussian import (decode, page_stats, selection,
                                           tau_solver)
    sizes = {}
    for module in (page_criticality, page_topk, selected_decode, decode,
                   page_stats, selection, tau_solver):
        for name, obj in vars(module).items():
            if isinstance(obj, triton.runtime.JITFunction):
                sizes[f"{module.__name__}.{name}"] = sum(
                    len(c) for c in obj.cache.values())
    return sizes


@pytest.mark.parametrize("method", ["topk", "gaussian"])
def test_no_recompiles_or_syncs_while_decoding(method):
    cache = make_cache(max_seq_len=max(LENS) + 200)
    q, kn, vn = make_step()
    out = torch.empty_like(q)
    alibi = slopes()    # built up front: a host-to-device copy syncs

    def step():
        if method == "topk":
            sparse_attention_decode_paged(q, cache, kn, vn, out, topk_pages=8,
                                          alibi_slopes=alibi)
        else:
            sparse_attention_decode_gaussian_aware_entmax(
                q, cache, kn, vn, out, alibi_slopes=alibi,
                append_cache=True)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    before = _jit_cache_sizes()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(64):     # crosses four page boundaries
            step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert _jit_cache_sizes() == before
