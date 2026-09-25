# EntmaxKV

Sparse entmax attention for efficient LLM inference with page-based KV cache selection using page metadata.

During autoregressive decoding, attending over the full KV cache grows expensive as context length increases. EntmaxKV reduces memory movement by selecting the most relevant tokens with two selection strategies: **top-k** page scoring and **Gaussian-aware** distributional selection.

## How it works

Tokens in the KV cache are organized into fixed-size pages (default: 16 tokens). The cache uses vLLM's paged layout (`[2, num_blocks, page_size, H_kv, D]` plus a block table), so the kernels here are shared with the [entmax_vllm](../entmax_vllm) serving backend. For each decode step, pages are scored from per-page key statistics (min/max for top-k, mean/std for Gaussian), without touching the cached tokens. Only the selected pages are attended to, and the attention is exact over that selection.

**Two selection strategies:**

- **Top-k** (`attention_topk.py`): Score pages with a Quest-style upper bound on their best score (ALiBi included at each page's best logical position), then keep the top `topk_pages` per query head. The budget counts pages and includes the always-attended tail page.

- **Gaussian-aware** (`attention_gaussian.py`): Model each page's scores as a Gaussian and solve the distributional entmax constraint for the threshold τ̂. Heads with ALiBi use the page-mixture model and NoPE heads use a single Gaussian. Then select every page whose estimated maximum can cross τ̂. The number of selected pages adapts to the score distribution rather than a fixed budget.



## Installation

This project uses [uv](https://docs.astral.sh/uv/) with a `pyproject.toml`/`uv.lock`:

```bash
uv sync
```

Alternatively, install with pip:

```bash
pip install -e .
```

**Requirements**: Python ≥ 3.10, < 3.13, PyTorch 2.6.0, Triton 3.2.0, CUDA GPU (CUDA 12.4 build).

## Quick start

```python
import torch
from entmaxkv import PagedKVCache
from entmaxkv.attention_topk import sparse_attention_decode_paged
from entmaxkv.attention_gaussian import sparse_attention_decode_gaussian_aware_entmax

B, H_q, H_kv, D = 1, 32, 8, 128
device = "cuda"

# Prefill: build the KV cache. Sizing it with max_seq_len avoids regrowing
# (and recompiling) mid-generation.
k_prefill = torch.randn(B, H_kv, 1024, D, device=device, dtype=torch.float16)
v_prefill = torch.randn(B, H_kv, 1024, D, device=device, dtype=torch.float16)

cache = PagedKVCache(page_size=16, max_seq_len=4096)
cache.initialize(k_prefill, v_prefill)   # seq_lens=[...] for ragged batches

# Decode step
q      = torch.randn(B, H_q,  1, D, device=device, dtype=torch.float16)
k_new  = torch.randn(B, H_kv, 1, D, device=device, dtype=torch.float16)
v_new  = torch.randn(B, H_kv, 1, D, device=device, dtype=torch.float16)
out    = torch.empty(B, H_q,  1, D, device=device, dtype=torch.float16)

# Top-k: attend 16 pages per query head (tail page included); appends k_new/v_new
sparse_attention_decode_paged(
    q=q, kv_cache=cache, k_new=k_new, v_new=v_new, out=out,
    topk_pages=16, alpha=1.5,
)

# Gaussian-aware: the number of pages adapts to the score distribution
sparse_attention_decode_gaussian_aware_entmax(
    q=q, kv_cache=cache, k_new=k_new, v_new=v_new, out=out,
    alpha=1.5, tau_mode="corrected", append_cache=True,
)
```


## Repository layout

```
entmaxkv/
├── kv_cache.py                          # PagedKVCache: vLLM block layout + per-page key statistics
├── attention_topk.py                    # Top-k sparse attention
├── attention_gaussian.py                # Gaussian-aware entmax attention
├── selectors.py                         # TopKPageSelector, GaussianPageSelector
├── decoding.py                          # Routing to the decode kernels by (alpha, head_dim)
├── tau_solver.py                        # CPU reference tau solvers (test oracles)
└── kernels/
    ├── paged_decode_utils.py            # Exact decode stages (shared with entmax_vllm)
    ├── decode.py                        # Exact paged decode (alpha 1.5 / 2)
    ├── selected_decode.py               # Decode over selected pages, one K pass (alpha 1.5 / 2)
    ├── adadecode_paged.py               # Decode over selected pages, generic alpha
    ├── adadecode_common.py              # AdaDecode tau-init / Halley / reduction helpers
    ├── adadecode.py                     # Dense AdaDecode orchestrator
    ├── page_metadata.py                 # Per-page k_min / k_max refresh
    ├── page_criticality.py              # Quest-style page upper bound
    ├── page_topk.py                     # Top-k + forced tail page
    ├── tau_solver_page_mixture.py       # Torch page-mixture tau solver (test oracle)
    ├── triton_entmax.py                 # Dense entmax reference implementation
    └── gaussian/
        ├── page_metadata.py             # Per-page k_mean / k_std refresh
        ├── page_stats.py                # Query-conditioned page and global score moments
        ├── tau_solver.py                # Single-Gaussian and page-mixture tau (Triton)
        ├── selection.py                 # Threshold selection + packing
        └── decode.py                    # Gaussian-tau decode with bracketed correction
tests/
├── test_paged_kernels.py                # Kernel correctness vs float64 oracles (pytest, GPU)
├── test_topk.py                         # Top-k accuracy/latency benchmarks (pytest, GPU)
├── test_gaussian.py                     # Gaussian-aware accuracy/latency benchmarks (pytest, GPU)
├── benchmark_utils.py                   # Reference attention, timing, error metrics
└── run_benchmarks.py                    # Standalone CLI for kernel-only timing sweeps
```

## Running tests

`test_topk.py` and `test_gaussian.py` are pytest test suites (parametrized over batch size, kv length, dtype, etc.), so run them with pytest rather than as plain scripts:

```bash
# Kernel correctness against float64 oracles (GQA, ALiBi + NoPE, ragged batches)
uv run pytest tests/test_paged_kernels.py -v

# Top-k sparse attention
uv run pytest tests/test_topk.py -v -s

# Gaussian-aware attention
uv run pytest tests/test_gaussian.py -v -s
```

For raw kernel timing (no reference attention or error metrics), use `run_benchmarks.py`:

```bash
uv run python tests/run_benchmarks.py topk
uv run python tests/run_benchmarks.py gaussian
```


## Efficiency

![Decode-step wall-clock time across context lengths, normalized to Softmax Flash](assets/global_efficiency_optimized.png)


## Citation

```bibtex
@misc{duarte2026entmaxkvsupportawaredecodingentmax,
      title={EntmaxKV: Support-Aware Decoding for Entmax Attention}, 
      author={Gonçalo Duarte and Miguel Couceiro and Marcos V. Treviso},
      year={2026},
      eprint={2605.21649},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.21649}, 
}
```
