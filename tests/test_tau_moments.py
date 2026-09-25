"""Analytic checks for the CPU Gaussian tau reference moments."""

import math

import torch

from entmaxkv.kernels.tau_solver_page_mixture import (
    _truncated_moments as mixture_moments,
)
from entmaxkv.tau_solver import (
    _evaluate_constraint,
    _evaluate_constraint_scalar,
    _truncated_moments as single_moments,
)


def test_positive_standard_normal_third_moment():
    # E[max(Z, 0)^3] = sqrt(2/pi) for Z ~ N(0, 1).
    zero = torch.tensor(0.0, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    expected = math.sqrt(2.0 / math.pi)

    for moments in (single_moments, mixture_moments):
        assert math.isclose(moments(zero, one)[3].item(), expected,
                            rel_tol=1e-12)

    # In the alpha=4/3 constraint, sigma_Y=(alpha-1)*sigma_g=1.
    constraint = _evaluate_constraint(zero, zero, one * 3.0, 1, 4.0 / 3.0)
    assert math.isclose(constraint.item() + 1.0, expected, rel_tol=1e-12)

    scalar = _evaluate_constraint_scalar(0.0, 0.0, 3.0, 1, 4.0 / 3.0)
    assert math.isclose(scalar + 1.0, expected, rel_tol=1e-12)
