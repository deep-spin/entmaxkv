# EntmaxKV

Sparse entmax attention for efficient LLM inference with page-based KV cache compression.

During autoregressive decoding, attending over the full KV cache grows expensive as context length increases. EntmaxKV compresses this cost by selecting only the most relevant pages of cached tokens—using entmax (a sparse generalization of softmax) with two selection strategies: **top-k** page scoring and **Gaussian-aware** distributional selection.

## How it works

Tokens in the KV cache are organized into fixed-size pages (default: 16 tokens). For each decode step, pages are scored by their maximum possible attention contribution, computed from per-page min/max/mean/std key statistics—without touching the actual cached tokens. Only the top pages are attended to.

**Two selection strategies:**

- **Top-k** (`attention_topk.py`): Score pages using a key-range upper bound, select the top-k pages within a token budget. Simple and fast.

- **Gaussian-aware** (`attention_gaussian.py`): Model the distribution of page scores as a Gaussian mixture, solve the distributional entmax constraint analytically to find the attention threshold τ, then select pages with sufficient probability mass above τ. More principled: the number of selected pages adapts to the actual score distribution rather than a fixed budget.


## Installation

```bash
pip install -e .
# optional: scipy for diagnostics
pip install -e ".[diagnostics]"
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.5, Triton ≥ 3.0, CUDA GPU.

## Quick start

```python
import torch
from entmaxkv import QuestKVCache
from entmaxkv.attention_topk import quest_sparse_attention_decode_paged
from entmaxkv.attention_gaussian import quest_sparse_attention_decode_gaussian_aware_entmax

B, H, D = 1, 32, 128
device = "cuda"

# Prefill: build the KV cache
k_prefill = torch.randn(B, H, 1024, D, device=device, dtype=torch.float16)
v_prefill = torch.randn(B, H, 1024, D, device=device, dtype=torch.float16)

cache = QuestKVCache(page_size=16)
cache.initialize(k_prefill, v_prefill)

# Decode step
q      = torch.randn(B, H, 1, D, device=device, dtype=torch.float16)
k_new  = torch.randn(B, H, 1, D, device=device, dtype=torch.float16)
v_new  = torch.randn(B, H, 1, D, device=device, dtype=torch.float16)
out    = torch.empty(B, H, 1, D, device=device, dtype=torch.float16)

cache_seqlens = torch.tensor([1024], device=device, dtype=torch.int32)
q_seqlens     = torch.tensor([1],    device=device, dtype=torch.int32)

# Top-k sparse attention (token_budget controls how many tokens are attended to)
quest_sparse_attention_decode_paged(
    q=q, quest_cache=cache, k_new=k_new, v_new=v_new, out=out,
    token_budget=256, cache_seqlens=cache_seqlens, q_seqlens=q_seqlens,
    alpha=1.5,
)

# Gaussian-aware sparse attention (budget adapts to the score distribution)
quest_sparse_attention_decode_gaussian_aware_entmax(
    q=q, quest_cache=cache, k_new=k_new, v_new=v_new, out=out,
    alpha=1.5, tau_mode="corrected", append_cache=True,
    cache_seqlens=cache_seqlens, q_seqlens=q_seqlens,
)
```

## API

### `QuestKVCache`

Page-based KV cache with incremental statistics.

```python
cache = QuestKVCache(page_size=16)
cache.initialize(k, v)          # called once after prefill
cache.append(k_new, v_new)      # called each decode step (if append_cache=False)
```

Maintains per-page `k_min`, `k_max`, `k_mean`, `k_std` for fast criticality scoring. Supports GQA (grouped-query attention) and ALiBi positional biases.

### `quest_sparse_attention_decode_paged`

Top-k sparse attention decode.

| Argument | Description |
|---|---|
| `q` | Query `[B, H, 1, D]` |
| `quest_cache` | `QuestKVCache` instance |
| `k_new`, `v_new` | Current-step keys/values `[B, H, 1, D]` |
| `out` | Output buffer `[B, H, 1, D]` |
| `token_budget` | Max tokens to attend over |
| `cache_seqlens` | Sequence lengths `[B]` |
| `q_seqlens` | Query lengths `[B]` |
| `alpha` | Entmax alpha (1.5 or 2.0) |
| `alibi_slopes` | Optional ALiBi slopes `[H]` |
| `append_cache` | Whether to update cache with `k_new`/`v_new` |

### `quest_sparse_attention_decode_gaussian_aware_entmax`

Gaussian-aware entmax decode. Same signature as above, replacing `token_budget` with:

| Argument | Description |
|---|---|
| `alpha` | Entmax alpha (1.5 or 2.0) |
| `tau_mode` | `"exact"` \| `"fixed"` \| `"corrected"` |
| `n_newton_steps` | Newton refinement steps when `tau_mode="corrected"` |
| `clamp_tau` | Clamp τ to selected pages (reduces over-selection) |

**`tau_mode` options:**
- `"exact"` — solve for τ per query during decode (most accurate, slower)
- `"fixed"` — use Gaussian τ estimate directly (fastest)
- `"corrected"` — Gaussian estimate + Newton refinement steps (balanced)

## Repository layout

```
entmaxkv/
├── kv_cache.py                          # QuestKVCache: page management & statistics
├── attention_topk.py                    # Top-k sparse attention
├── attention_gaussian.py                # Gaussian-aware entmax attention
├── tau_solver.py                        # Tau solving (CPU): closed-form + iterative
├── gaussian_utils.py                    # Gaussian statistics & threshold clamping
└── kernels/
    ├── adadecode.py                     # Core decode Triton kernels (dense & paged)
    ├── adadecode_paged.py               # Paged decode variant
    ├── adadecode_paged_gaussian_tau.py  # Paged decode with pre-computed tau
    ├── page_criticality.py              # Page scoring kernel
    ├── gaussian_page_stats.py           # Per-page Gaussian statistics
    ├── selection_pack.py                # Page selection & index packing
    ├── tau_clamp.py                     # Tau clamping to selected pages
    ├── tau_solver_gpu.py                # GPU tau solving (alpha=2, 1.5)
    ├── tau_solver_page_mixture.py       # Tau solving for Gaussian mixtures
    ├── tau_mixture_solver_triton.py     # Triton mixture tau solver (ALiBi)
    └── triton_entmax.py                 # Dense entmax reference implementation
tests/
├── test_topk.py                         # Top-k attention benchmarks
├── test_gaussian.py                     # Gaussian-aware attention benchmarks
└── benchmark_utils.py                   # Reference attention, timing, error metrics
```

## Running tests

```bash
# Top-k sparse attention
python tests/test_topk.py

# Gaussian-aware attention
python tests/test_gaussian.py
```

Benchmarks vary batch size, context length (1K–64K tokens), coverage (25%–50%), alpha, and dtype. They report L2 error, relative error, cosine similarity vs. dense reference attention, and per-iteration latency.

