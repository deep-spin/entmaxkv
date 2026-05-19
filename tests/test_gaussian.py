"""
Benchmark: Gaussian-aware sparse attention decode.

Covers tau_mode in {"exact", "fixed", "corrected"} and optional ALiBi / GQA.

Run with:
    python tests/test_gaussian.py
  or via pytest:
    python -m pytest tests/test_gaussian.py -v -s
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
from entmaxkv.attention_gaussian import quest_sparse_attention_decode_gaussian_aware_entmax


GAUSSIAN_BATCHES = [1, 8]
GAUSSIAN_KV_LENS = [1024, 2048, 8192, 16384, 64000]
GAUSSIAN_DTYPES = [torch.float16, torch.float32]
GAUSSIAN_DTYPE_IDS = ["fp16", "fp32"]


# ---------------------------------------------------------------------------
# core benchmark
# ---------------------------------------------------------------------------

def run_gaussian_benchmark(
    *,
    seed: int,
    batch: int,
    kv_heads: int,
    q_heads: int,
    kv_len: int,
    head_dim: int,
    page_size: int,
    tau_mode: str,
    use_alibi: bool,
    dtype: torch.dtype = torch.float32,
    safety_margin_z: float = 0.0,
    max_quantile: float = 0.995,
    device,
    warmup: int = 3,
    iters: int = 10,
    print_results: bool = True,
) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    alibi = generate_alibi_slopes(q_heads) if use_alibi else None

    q      = torch.randn(batch, kv_heads, 1,      head_dim, device=device, dtype=dtype).contiguous()
    k_full = torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype).contiguous()
    v_full = torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype).contiguous()
    if q_heads != kv_heads:
        q = q.repeat_interleave(q_heads // kv_heads, dim=1)

    k_new = torch.randn(batch, kv_heads, 1, head_dim, device=device, dtype=dtype).contiguous()
    v_new = torch.randn(batch, kv_heads, 1, head_dim, device=device, dtype=dtype).contiguous()

    full_seq_len = kv_len + 1
    q_seqlens = torch.full((batch,), full_seq_len, device=device, dtype=torch.int32)
    out = torch.zeros_like(q)

    def make_fresh_cache():
        c = QuestKVCache(page_size=page_size)
        c.initialize(k_full, v_full)
        c.append(k_new, v_new)
        return c

    cache = make_fresh_cache()

    def call_decode():
        quest_sparse_attention_decode_gaussian_aware_entmax(
            q=q,
            quest_cache=cache,
            k_new=k_new,
            v_new=v_new,
            out=out,
            alpha=1.5,
            safety_margin_z=safety_margin_z,
            max_quantile=max_quantile,
            alibi_slopes=alibi,
            append_cache=False,
            tau_mode=tau_mode,
            q_seqlens=q_seqlens,
        )

    latency_ms = time_fn(call_decode, warmup=warmup, iters=iters)
    call_decode()

    # Reference: chunked entmax over the full cache (kv_full + k_new appended).
    k_cache_full = torch.cat([k_full, k_new], dim=2)
    v_cache_full = torch.cat([v_full, v_new], dim=2)
    n_rep = q_heads // kv_heads
    k_ref = k_cache_full.repeat_interleave(n_rep, dim=1)
    v_ref = v_cache_full.repeat_interleave(n_rep, dim=1)
    out_ref = reference_attention(q, k_ref, v_ref, alpha=1.5, alibi_slopes=alibi)

    l2, rel_error, cosine_sim = compute_errors(out_ref, out)

    result = {
        "tau_mode": tau_mode,
        "batch": batch,
        "kv_heads": kv_heads,
        "q_heads": q_heads,
        "kv_len": kv_len,
        "head_dim": head_dim,
        "page_size": page_size,
        "alpha": 1.5,
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
            f"[gaussian/{tau_mode}] batch={batch} kv={kv_heads} q={q_heads} "
            f"len={kv_len} alpha=1.5 alibi={use_alibi} dtype={dtype_str}"
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


@pytest.mark.parametrize("dtype", GAUSSIAN_DTYPES, ids=GAUSSIAN_DTYPE_IDS)
@pytest.mark.parametrize("batch", GAUSSIAN_BATCHES)
@pytest.mark.parametrize("kv_len", GAUSSIAN_KV_LENS)
def test_gaussian_exact_basic(batch, kv_len, dtype):
    if not torch.cuda.is_available():
        return
    result = run_gaussian_benchmark(
        seed=10, batch=batch, kv_heads=8, q_heads=8,
        kv_len=kv_len, head_dim=64, page_size=16,
        tau_mode="exact", use_alibi=False,
        dtype=dtype, device=_device(), print_results=False,
    )
    assert result["rel_error"] < 0.5, f"rel_error too large: {result['rel_error']:.4f}"
    assert result["latency_ms"] < 500.0


@pytest.mark.parametrize("dtype", GAUSSIAN_DTYPES, ids=GAUSSIAN_DTYPE_IDS)
@pytest.mark.parametrize("batch", GAUSSIAN_BATCHES)
@pytest.mark.parametrize("kv_len", GAUSSIAN_KV_LENS)
def test_gaussian_corrected_basic(batch, kv_len, dtype):
    if not torch.cuda.is_available():
        return
    result = run_gaussian_benchmark(
        seed=12, batch=batch, kv_heads=8, q_heads=8,
        kv_len=kv_len, head_dim=64, page_size=16,
        tau_mode="corrected", use_alibi=False,
        dtype=dtype, device=_device(), print_results=False,
    )
    assert result["rel_error"] < 0.5, f"rel_error too large: {result['rel_error']:.4f}"


@pytest.mark.parametrize("dtype", GAUSSIAN_DTYPES, ids=GAUSSIAN_DTYPE_IDS)
@pytest.mark.parametrize("batch", GAUSSIAN_BATCHES)
@pytest.mark.parametrize("kv_len", GAUSSIAN_KV_LENS)
def test_gaussian_alibi(batch, kv_len, dtype):
    if not torch.cuda.is_available():
        return
    result = run_gaussian_benchmark(
        seed=13, batch=batch, kv_heads=8, q_heads=8,
        kv_len=kv_len, head_dim=64, page_size=16,
        tau_mode="exact", use_alibi=True,
        dtype=dtype, device=_device(), print_results=False,
    )
    assert result["rel_error"] < 0.5, f"rel_error too large: {result['rel_error']:.4f}"


if __name__ == "__main__":
    pytest.main(["-v", "-s", "--color=yes"])
