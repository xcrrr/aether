"""Tests for Poincaré ball operations (model.py).

Validates the mathematical invariants that the entire architecture
depends on:  exp/log map round-trips, Möbius addition identities,
ball projection guarantees, and Riemannian gradient rescaling.
"""

from __future__ import annotations

import math

import pytest
import torch

from model import (
    exp_map_zero,
    log_map_zero,
    mobius_add,
    project_to_ball,
    RiemannianRescale,
    _sqrt_c,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=[0.5, 1.0, 2.0], ids=["c=0.5", "c=1.0", "c=2.0"])
def curvature(request) -> float:
    return request.param


@pytest.fixture
def tangent_vectors() -> torch.Tensor:
    """Small tangent vectors at the origin — safe for exp_map."""
    torch.manual_seed(42)
    return torch.randn(4, 16, 64) * 0.1


@pytest.fixture
def ball_points(curvature) -> torch.Tensor:
    """Points inside the Poincaré ball for the given curvature."""
    torch.manual_seed(42)
    x = torch.randn(4, 16, 64) * 0.1
    return project_to_ball(x, curvature)


# ─────────────────────────────────────────────────────────────────────────────
# exp_map / log_map round-trip tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExpLogRoundTrip:
    """exp_map_zero and log_map_zero should be approximate inverses."""

    def test_tangent_to_ball_and_back(self, tangent_vectors, curvature):
        """log_map(exp_map(v)) ≈ v  for small tangent vectors."""
        v = tangent_vectors
        ball = exp_map_zero(v, curvature)
        recovered = log_map_zero(ball, curvature)

        # Float32 tolerance — allow some error near the ball boundary
        torch.testing.assert_close(recovered, v, atol=1e-4, rtol=1e-4)

    def test_ball_to_tangent_and_back(self, ball_points, curvature):
        """exp_map(log_map(x)) ≈ x  for points inside the ball."""
        x = ball_points
        tangent = log_map_zero(x, curvature)
        recovered = exp_map_zero(tangent, curvature)

        torch.testing.assert_close(recovered, x, atol=1e-4, rtol=1e-4)

    def test_zero_tangent_maps_to_origin(self, curvature):
        """exp_map(0) should be near the origin."""
        v = torch.zeros(2, 4, 64)
        ball = exp_map_zero(v, curvature)
        norms = ball.norm(dim=-1)

        assert norms.max().item() < 1e-6, \
            f"exp_map(0) should be near origin, got max norm {norms.max():.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# Möbius addition identity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMobiusAdd:
    """Möbius addition must satisfy algebraic identities."""

    def test_identity_element(self, ball_points, curvature):
        """x ⊕ 0 ≈ x  — zero is the Möbius identity."""
        x = ball_points
        zero = torch.zeros_like(x)
        result = mobius_add(x, zero, curvature)

        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_right_identity(self, ball_points, curvature):
        """0 ⊕ x ≈ x  — zero is also the right identity."""
        x = ball_points
        zero = torch.zeros_like(x)
        result = mobius_add(zero, x, curvature)

        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_inverse_gives_near_origin(self, ball_points, curvature):
        """x ⊕ (-x) ≈ 0  — Möbius inverse property."""
        x = ball_points
        result = mobius_add(x, -x, curvature)
        norms = result.norm(dim=-1)

        # Being near origin — allow numerical tolerance
        assert norms.max().item() < 1e-3, \
            f"x ⊕ (-x) should be near origin, got max norm {norms.max():.4f}"

    def test_result_stays_inside_ball(self, curvature):
        """Möbius addition of two ball points must stay inside the ball."""
        torch.manual_seed(123)
        # Generate two different sets of ball points
        x = project_to_ball(torch.randn(4, 8, 64) * 0.3, curvature)
        y = project_to_ball(torch.randn(4, 8, 64) * 0.3, curvature)
        result = mobius_add(x, y, curvature)

        max_norm = 1.0 / _sqrt_c(curvature)
        norms = result.norm(dim=-1)
        assert norms.max().item() < max_norm, \
            f"Result norm {norms.max():.6f} exceeds ball radius {max_norm:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# project_to_ball
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectToBall:
    """project_to_ball must guarantee outputs lie strictly inside the ball."""

    def test_clamps_large_vectors(self, curvature):
        """Vectors well outside the ball should be projected inside."""
        x = torch.randn(4, 8, 64) * 10.0  # way outside the ball
        projected = project_to_ball(x, curvature)

        max_norm = (1.0 / _sqrt_c(curvature)) - 1e-2  # eps=1e-2 default
        norms = projected.norm(dim=-1)
        assert norms.max().item() <= max_norm + 1e-6

    def test_preserves_small_vectors(self, curvature):
        """Small vectors already inside should be unchanged."""
        x = torch.randn(4, 8, 64) * 0.01  # tiny, well inside
        projected = project_to_ball(x, curvature)

        torch.testing.assert_close(projected, x, atol=1e-5, rtol=1e-5)

    def test_output_dtype_matches_input(self):
        """Dtype should be preserved through projection."""
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(2, 4, 16, dtype=dtype)
            result = project_to_ball(x, 1.0)
            assert result.dtype == dtype, \
                f"Expected {dtype}, got {result.dtype}"


# ─────────────────────────────────────────────────────────────────────────────
# RiemannianRescale
# ─────────────────────────────────────────────────────────────────────────────

class TestRiemannianRescale:
    """Custom autograd function: identity forward, rescaled backward."""

    def test_forward_is_identity(self):
        """Forward pass should return input unchanged."""
        x = torch.randn(2, 4, 16)
        c = torch.tensor(1.0)
        result = RiemannianRescale.apply(x, c)

        torch.testing.assert_close(result, x)

    def test_backward_scales_gradients(self):
        """Backward should rescale by inverse conformal factor squared."""
        x = torch.randn(2, 4, 16, requires_grad=True)
        c = torch.tensor(1.0)

        result = RiemannianRescale.apply(x, c)
        loss = result.sum()
        loss.backward()

        # For points near origin (small norm), the scaling factor
        # (1 - c*||x||^2)^2 / 4 should be close to 1/4
        assert x.grad is not None, "Gradient should flow through RiemannianRescale"
        # Gradient should NOT all be 1.0 (which would mean no rescaling)
        # unless x happens to be exactly zero
        if x.norm() > 1e-6:
            all_ones = torch.ones_like(x.grad)
            assert not torch.allclose(x.grad, all_ones), \
                "RiemannianRescale should modify gradients"


# ─────────────────────────────────────────────────────────────────────────────
# Numerical stability
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalStability:
    """Poincaré ops must not produce NaN on edge-case inputs."""

    def test_exp_map_no_nan_on_large_input(self):
        """Large tangent vectors should not produce NaN."""
        v = torch.randn(2, 4, 32) * 100.0
        result = exp_map_zero(v, 1.0)
        assert not result.isnan().any(), "exp_map_zero produced NaN on large input"

    def test_log_map_no_nan_on_boundary(self):
        """Points near the ball boundary should not produce NaN."""
        # Create points very close to the boundary
        x = torch.randn(2, 4, 32)
        x = x / x.norm(dim=-1, keepdim=True) * 0.98  # near boundary for c=1.0
        result = log_map_zero(x, 1.0)
        assert not result.isnan().any(), "log_map_zero produced NaN near boundary"

    def test_mobius_add_no_nan_near_boundary(self):
        """Möbius add near boundary should not produce NaN."""
        x = torch.randn(2, 4, 32)
        x = x / x.norm(dim=-1, keepdim=True) * 0.95
        y = torch.randn(2, 4, 32)
        y = y / y.norm(dim=-1, keepdim=True) * 0.95
        result = mobius_add(x, y, 1.0)
        assert not result.isnan().any(), "mobius_add produced NaN near boundary"
        assert not result.isinf().any(), "mobius_add produced Inf near boundary"

    def test_bfloat16_safety(self):
        """BF16 inputs should not produce NaN through the full pipeline."""
        v = torch.randn(2, 4, 32, dtype=torch.bfloat16) * 0.1
        ball = exp_map_zero(v, 1.0)
        assert not ball.isnan().any(), "BF16 exp_map produced NaN"

        tangent = log_map_zero(ball, 1.0)
        assert not tangent.isnan().any(), "BF16 log_map produced NaN"

        result = mobius_add(ball, ball * 0.5, 1.0)
        assert not result.isnan().any(), "BF16 mobius_add produced NaN"


# ─────────────────────────────────────────────────────────────────────────────
# Gradient flow
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    """Verify that gradients flow through the geometry operations."""

    def test_exp_map_gradient_flows(self):
        """exp_map_zero should be differentiable."""
        v = (torch.randn(2, 4, 16) * 0.1).requires_grad_(True)  # leaf tensor
        ball = exp_map_zero(v, 1.0)
        ball.sum().backward()
        assert v.grad is not None, "No gradient through exp_map_zero"
        assert not v.grad.isnan().any(), "NaN gradient through exp_map_zero"

    def test_mobius_add_gradient_flows(self):
        """mobius_add should be differentiable w.r.t. both inputs."""
        x = (torch.randn(2, 4, 16) * 0.1).requires_grad_(True)  # leaf tensor
        y = (torch.randn(2, 4, 16) * 0.1).requires_grad_(True)  # leaf tensor

        x_ball = project_to_ball(x, 1.0)
        y_ball = project_to_ball(y, 1.0)
        result = mobius_add(x_ball, y_ball, 1.0)
        result.sum().backward()

        # Check gradients flow to the leaf tensors (x and y)
        assert x.grad is not None, "No gradient through mobius_add (x)"
        assert y.grad is not None, "No gradient through mobius_add (y)"
