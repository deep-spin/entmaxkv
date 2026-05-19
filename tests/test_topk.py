"""
Benchmark: Quest top-k sparse attention decode (paged kernel).

Run with:
    python tests/test_topk.py
  or via pytest:
    python -m pytest tests/test_topk.py -v -s
"""

import torch
import pytest

from tests.benchmark_utils import (
    reference_attention,
    generate_alibi_slopes,
    compute_errors,
    time_fn,
)


from entmaxkv.kv_cache import QuestKVCache
from entmaxkv.attention_topk import quest_sparse_attention_decode_paged


TOPK_BATCHES = [1, 8]
TOPK_KV_LENS = [1024, 2048, 8192, 16384, 64000]
TOPK_COVERAGES = [0.25, 0.50]
TOPK_ALPHAS = [1.5, 2.0]
TOPK_DTYPES = [torch.float16, torch.float32]
TOPK_DTYPE_IDS = ["fp16", "fp32"]


# ---------------------------------------------------------------------------
# core benchmark
# ---------------------------------------------------------------------------

def run_topk_benchmark(
    *,
    seed: int,
    batch: int,
    kv_heads: int,
    q_heads: int,
    kv_len: int,
    head_dim: int,
    page_size: int,
    coverage: float,
    alpha: float,
    use_alibi: bool,
    dtype: torch.dtype = torch.float32,
    device,
    warmup: int = 20,
    iters: int = 50,
    print_results: bool = True,
) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    alibi = generate_alibi_slopes(q_heads) if use_alibi else None
    page_budget = max(1, int((coverage * kv_len + page_size - 1) // page_size))
    token_budget = page_budget * page_size

    k_full = torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype).contiguous()
    v_full = torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype).contiguous()

    q     = torch.randn(batch, q_heads,  1,      head_dim, device=device, dtype=dtype).contiguous()
    k_new = torch.randn(batch, kv_heads, 1,      head_dim, device=device, dtype=dtype).contiguous()
    v_new = torch.randn(batch, kv_heads, 1,      head_dim, device=device, dtype=dtype).contiguous()

    # Reference: chunked entmax over the full cache (kv_full + k_new appended).
    k_cache_full = torch.cat([k_full, k_new], dim=2)
    v_cache_full = torch.cat([v_full, v_new], dim=2)
    # GQA expansion for the reference
    n_rep = q_heads // kv_heads
    k_ref = k_cache_full.repeat_interleave(n_rep, dim=1)
    v_ref = v_cache_full.repeat_interleave(n_rep, dim=1)

    out_ref = reference_attention(q, k_ref, v_ref, alpha=alpha, alibi_slopes=alibi)

    full_seq_len = kv_len + 1
    total_pages = (full_seq_len + page_size - 1) // page_size
    num_pages_to_select = min(token_budget // page_size, total_pages - 1)
    last_page_start = (total_pages - 1) * page_size
    last_page_size = full_seq_len - last_page_start
    effective_cache_len = num_pages_to_select * page_size + last_page_size
    cache_seqlens = torch.full((batch,), effective_cache_len, device=device, dtype=torch.int32)
    q_seqlens = torch.full((batch,), full_seq_len, device=device, dtype=torch.int32)
    out = torch.zeros_like(q)

    def make_fresh_cache():
        c = QuestKVCache(page_size=page_size)
        c.initialize(k_full, v_full)
        c.append(k_new, v_new)
        return c

    cache = make_fresh_cache()

    def call_decode():
        quest_sparse_attention_decode_paged(
            q=q,
            quest_cache=cache,
            k_new=k_new,
            v_new=v_new,
            out=out,
            token_budget=token_budget,
            cache_seqlens=cache_seqlens,
            q_seqlens=q_seqlens,
            alibi_slopes=alibi,
            alpha=alpha,
            niter=2,
            append_cache=False,
        )

    latency_ms = time_fn(call_decode, warmup=warmup, iters=iters)
    call_decode()
    l2, rel_error, cosine_sim = compute_errors(out_ref, out)

    result = {
        "batch": batch,
        "kv_heads": kv_heads,
        "q_heads": q_heads,
        "kv_len": kv_len,
        "head_dim": head_dim,
        "page_size": page_size,
        "coverage": coverage,
        "page_budget": page_budget,
        "token_budget": token_budget,
        "alpha": alpha,
        "use_alibi": use_alibi,
        "dtype": dtype,
        "l2": l2.item(),
        "rel_error": rel_error.item(),
        "cosine_sim": cosine_sim.item(),
        "latency_ms": latency_ms,
    }

    if print_results:
        dtype_str = "fp16" if dtype == torch.float16 else "fp32"
        tag = (
            f"[topk/paged] batch={batch} kv={kv_heads} q={q_heads} "
            f"len={kv_len} coverage={coverage:.2f} budget={token_budget} "
            f"alpha={alpha} alibi={use_alibi} dtype={dtype_str}"
        )
        print(
            f"{tag}  lat={latency_ms:.2f}ms  "
            f"l2={l2.item():.4f}  rel_err={rel_error.item():.4f}  cosine_sim={cosine_sim.item():.4f}"
        )

    return result


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("dtype", TOPK_DTYPES, ids=TOPK_DTYPE_IDS)
@pytest.mark.parametrize("alpha", TOPK_ALPHAS)
@pytest.mark.parametrize("batch", TOPK_BATCHES)
@pytest.mark.parametrize("kv_len", TOPK_KV_LENS)
@pytest.mark.parametrize("coverage", TOPK_COVERAGES)
def test_topk_paged_basic(batch, kv_len, coverage, alpha, dtype):
    if not torch.cuda.is_available():
        return
    result = run_topk_benchmark(
        seed=1, batch=batch, kv_heads=8, q_heads=8,
        kv_len=kv_len, head_dim=64, page_size=16,
        coverage=coverage, alpha=alpha, use_alibi=False,
        dtype=dtype, device=_device(), print_results=False,
    )
    max_rel_error = (1.0 - coverage) * 2.0
    assert result["rel_error"] < max_rel_error, (
        f"rel_error too large: {result['rel_error']:.4f} >= {max_rel_error:.4f}"
    )
    assert result["latency_ms"] < 500.0


@pytest.mark.parametrize("dtype", TOPK_DTYPES, ids=TOPK_DTYPE_IDS)
@pytest.mark.parametrize("alpha", TOPK_ALPHAS)
@pytest.mark.parametrize("batch", TOPK_BATCHES)
@pytest.mark.parametrize("kv_len", TOPK_KV_LENS)
@pytest.mark.parametrize("coverage", TOPK_COVERAGES)
def test_topk_alibi(batch, kv_len, coverage, alpha, dtype):
    if not torch.cuda.is_available():
        return
    result = run_topk_benchmark(
        seed=2, batch=batch, kv_heads=8, q_heads=8,
        kv_len=kv_len, head_dim=64, page_size=16,
        coverage=coverage, alpha=alpha, use_alibi=True,
        dtype=dtype, device=_device(), print_results=False,
    )
    max_rel_error = (1.0 - coverage) * 2.0
    assert result["rel_error"] < max_rel_error, (
        f"rel_error too large: {result['rel_error']:.4f} >= {max_rel_error:.4f}"
    )



if __name__ == "__main__":
    pytest.main(["-v", "-s", "--color=yes"])
