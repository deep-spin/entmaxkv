"""
Benchmark helpers for entmaxkv tests.

Provides:
  - generate_alibi_slopes
  - compute_errors
  - reference_attention  (dense triton_entmax_attention)
  - time_fn
"""

import time
import math

import torch
import torch.nn.functional as F

from entmaxkv.kernels.triton_entmax import triton_entmax


# ---------------------------------------------------------------------------
# data generation
# ---------------------------------------------------------------------------

def generate_alibi_slopes(n_heads: int, dtype=torch.float32) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    slopes = torch.exp2(-8 * torch.arange(1, n_heads + 1, device=device) / n_heads).to(dtype)
    return slopes


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------

def compute_errors(
    reference: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = reference - output
    l2_norm = torch.norm(diff)
    rel_error = l2_norm / torch.norm(reference)
    cosine_sim = F.cosine_similarity(
        output.reshape(1, -1).float(),
        reference.reshape(1, -1).float(),
    )
    return l2_norm, rel_error, cosine_sim


# ---------------------------------------------------------------------------
# reference attention
# ---------------------------------------------------------------------------

def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: float = 1.5,
    alibi_slopes: torch.Tensor = None,
    niter: int = 2,
) -> torch.Tensor:
    """Dense PyTorch attention reference matching adaquest's chunked reference."""
    batch, num_heads, q_len, head_dim = q.shape
    kv_len = k.shape[2]
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    qk = torch.matmul(q_f, k_f.transpose(-1, -2)) / math.sqrt(head_dim)

    if alibi_slopes is not None:
        q_positions = torch.arange(kv_len - q_len, kv_len, device=q.device)
        k_positions = torch.arange(kv_len, device=q.device)
        position_diff = k_positions[None, :] - q_positions[:, None]
        alibi_bias = alibi_slopes.view(1, num_heads, 1, 1) * position_diff.view(1, 1, q_len, kv_len)
        qk = qk + alibi_bias.float()

    if alpha == 1.0:
        p = torch.softmax(qk, dim=-1)
    else:
        p = triton_entmax(qk, alpha=alpha, n_iter=niter, fast_math=True)
    return torch.matmul(p, v_f).to(q.dtype)


# ---------------------------------------------------------------------------
# timing helper
# ---------------------------------------------------------------------------

def time_fn(fn, warmup: int = 20, iters: int = 50) -> float:
    """Return median latency in milliseconds."""
    for _ in range(warmup):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]
