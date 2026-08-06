"""
Timing script for the entmaxkv decode kernels.

Times the attention entry points directly -- no reference, no error metrics.

    python tests/run_benchmarks.py topk
    python tests/run_benchmarks.py topk --kv-len 32768 --coverage 0.5
    python tests/run_benchmarks.py gaussian
    python tests/run_benchmarks.py gaussian --clamp-tau

By default, sweeps kv_len over 32k/64k/128k and reports a timing table.
Pass --kv-len explicitly to run a single length instead.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entmaxkv.kv_cache import PagedKVCache
from entmaxkv.attention_topk import sparse_attention_decode_paged
from entmaxkv.attention_gaussian import sparse_attention_decode_gaussian_aware_entmax


DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def time_kernel(fn, warmup: int, iters: int) -> dict:
    """Time fn with CUDA events. Returns median/min/p90 in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {
        "median": times[len(times) // 2],
        "min": times[0],
        "p90": times[min(int(0.9 * len(times)), len(times) - 1)],
    }


def build_inputs(args, device, dtype):
    """Allocate q/k_new/v_new and a populated PagedKVCache."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    b, h, hq = args.batch, args.kv_heads, args.q_heads
    n, d = args.kv_len, args.head_dim

    k_full = torch.randn(b, h, n, d, device=device, dtype=dtype).contiguous()
    v_full = torch.randn(b, h, n, d, device=device, dtype=dtype).contiguous()
    q = torch.randn(b, hq, 1, d, device=device, dtype=dtype).contiguous()
    k_new = torch.randn(b, h, 1, d, device=device, dtype=dtype).contiguous()
    v_new = torch.randn(b, h, 1, d, device=device, dtype=dtype).contiguous()

    cache = PagedKVCache(page_size=args.page_size)
    cache.initialize(k_full, v_full)
    cache.append(k_new, v_new)  # appended here; kernels run with append_cache=False

    del k_full, v_full  # initialize() clones these
    torch.cuda.empty_cache()

    slopes = None
    if args.alibi:
        slopes = torch.exp2(-8 * torch.arange(1, hq + 1, device=device) / hq).float()

    return q, k_new, v_new, cache, slopes


def run_topk(args, device, dtype):
    q, k_new, v_new, cache, slopes = build_inputs(args, device, dtype)

    page_size, kv_len = args.page_size, args.kv_len
    page_budget = max(1, int((args.coverage * kv_len + page_size - 1) // page_size))
    token_budget = page_budget * page_size

    full_seq_len = kv_len + 1
    total_pages = (full_seq_len + page_size - 1) // page_size
    n_select = min(token_budget // page_size, total_pages - 1)
    last_page_start = (total_pages - 1) * page_size
    effective_len = n_select * page_size + (full_seq_len - last_page_start)

    cache_seqlens = torch.full((args.batch,), effective_len, device=device, dtype=torch.int32)
    q_seqlens = torch.full((args.batch,), full_seq_len, device=device, dtype=torch.int32)
    out = torch.zeros_like(q)

    def call():
        sparse_attention_decode_paged(
            q=q, kv_cache=cache, k_new=k_new, v_new=v_new, out=out,
            token_budget=token_budget, cache_seqlens=cache_seqlens,
            q_seqlens=q_seqlens, alibi_slopes=slopes, alpha=args.alpha,
            niter=args.niter, append_cache=False,
        )

    print(f"coverage={args.coverage}  token_budget={token_budget}  "
          f"pages={n_select}/{total_pages}")
    call()  # surface errors before timing
    return time_kernel(call, args.warmup, args.iters)


def run_gaussian(args, device, dtype):
    q, k_new, v_new, cache, slopes = build_inputs(args, device, dtype)
    q_seqlens = torch.full((args.batch,), args.kv_len + 1, device=device, dtype=torch.int32)
    out = torch.zeros_like(q)

    def call():
        sparse_attention_decode_gaussian_aware_entmax(
            q=q, kv_cache=cache, k_new=k_new, v_new=v_new, out=out,
            alpha=args.alpha, safety_margin_z=args.safety_margin_z,
            max_quantile=args.max_quantile, alibi_slopes=slopes,
            append_cache=False, tau_mode=args.tau_mode, q_seqlens=q_seqlens,
            clamp_tau=args.clamp_tau,
        )

    print(f"tau_mode={args.tau_mode}  clamp_tau={args.clamp_tau}")
    call()

    sel = cache.last_num_selected_per_head
    if sel is not None:
        total_pages = (args.kv_len + 1 + args.page_size - 1) // args.page_size
        print(f"selected pages: mean={sel.float().mean():.1f} max={sel.max()} "
              f"of {total_pages}")

    return time_kernel(call, args.warmup, args.iters)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kernel", choices=["topk", "gaussian"])
    p.add_argument("--kv-len", type=int, default=None,
                   help="single kv_len; if omitted, sweeps 32k/64k/128k")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--coverage", type=float, default=0.25, help="topk only")
    p.add_argument("--tau-mode", default="corrected",
                   choices=["exact", "fixed", "corrected"], help="gaussian only")
    p.add_argument("--clamp-tau", action="store_true",
                   help="gaussian only; off by default. No effect when tau_mode=exact.")
    p.add_argument("--safety-margin-z", type=float, default=0.0)
    p.add_argument("--max-quantile", type=float, default=0.995)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--niter", type=int, default=2)
    p.add_argument("--alibi", action="store_true")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--q-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- these are Triton kernels and need a GPU.")
        return 1

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]

    if args.clamp_tau and args.tau_mode == "exact":
        print("note: --clamp-tau is ignored when tau_mode=exact")

    kv_lens = [args.kv_len] if args.kv_len is not None else [32 * 1024, 64 * 1024, 128 * 1024]
    run_fn = run_topk if args.kernel == "topk" else run_gaussian

    print(f"{torch.cuda.get_device_name(0)}   {args.kernel}   "
          f"batch={args.batch} {args.dtype} alibi={args.alibi} alpha={args.alpha}")

    results = []
    for kv_len in kv_lens:
        args.kv_len = kv_len
        print(f"\n-- kv_len={kv_len} --")
        t0 = time.perf_counter()
        stats = run_fn(args, device, dtype)
        elapsed = time.perf_counter() - t0
        print(f"median={stats['median']:.3f} ms   min={stats['min']:.3f}   "
              f"p90={stats['p90']:.3f}   ({args.iters} iters, {elapsed:.1f}s)")
        results.append((kv_len, stats))

    if len(results) > 1:
        print(f"\n{'kv_len':>8}  {'median (ms)':>12}  {'min (ms)':>10}  {'p90 (ms)':>10}")
        for kv_len, stats in results:
            print(f"{kv_len:>8}  {stats['median']:>12.3f}  {stats['min']:>10.3f}  {stats['p90']:>10.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
