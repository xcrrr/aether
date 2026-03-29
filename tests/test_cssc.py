"""Tests for CSSC — Cross-Scale Spatiotemporal Correlation (cssc.py).

Validates causal correctness, attention shapes, scale blending,
and the curvature gate.  Causal leakage is tested explicitly with
a forward-fill probe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from cssc import CSSCAttention, _window_mask, _hyperbolic_decay_bias


# ─────────────────────────────────────────────────────────────────────────────
# Causal pooling — the most critical correctness test
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalPool:
    """_causal_pool must NEVER leak future information."""

    def test_no_future_leakage(self, small_cfg):
        """Pooled segment j must only contain information from segments < j.

        Strategy: Create a sequence where token t has value = t.
        Pool with a stride.  Each pooled slot j should have a mean
        value strictly less than j*stride (i.e., only past tokens).
        """
        cssc = CSSCAttention(small_cfg)
        D = small_cfg.d_model
        stride = small_cfg.cssc_sentence_stride
        T = small_cfg.max_seq_len

        # Use positions as values: token t has constant value = t across all dims
        x = torch.arange(T, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        x = x.expand(1, T, D)  # (1, T, D)

        pooled = CSSCAttention._causal_pool(x, stride)  # (1, T//stride, D)

        # Slot 0 should be zeros (nothing before the first segment)
        assert pooled[0, 0].abs().max().item() < 1e-6, \
            "First pooled slot should be zero (no past context)"

        # Slot j (j>0) should contain the mean of segment j-1
        # Segment j-1 spans positions [(j-1)*stride, j*stride)
        for j in range(1, pooled.shape[1]):
            # The mean value in the pooled slot should be the avg of segment j-1
            # = mean of [(j-1)*stride, ..., j*stride - 1]
            expected_mean = ((j - 1) * stride + j * stride - 1) / 2.0
            actual_mean = pooled[0, j, 0].item()
            assert abs(actual_mean - expected_mean) < 1e-3, \
                f"Slot {j}: expected mean ~{expected_mean:.1f}, got {actual_mean:.1f}"

    def test_single_window_returns_zeros(self, small_cfg):
        """When T <= stride, return zeros (nothing past to attend to)."""
        cssc = CSSCAttention(small_cfg)
        D = small_cfg.d_model
        T = 4
        stride = 32  # larger than T

        x = torch.randn(2, T, D)
        pooled = CSSCAttention._causal_pool(x, stride)

        assert pooled.shape[1] == 1, "Should return exactly 1 slot"
        assert pooled.abs().max().item() < 1e-6, \
            "Single-window pool should be zeros"

    def test_output_shape(self, small_cfg):
        """Pooled output should have T_k = max(1, T // stride) tokens."""
        cssc = CSSCAttention(small_cfg)
        D = small_cfg.d_model
        T = small_cfg.max_seq_len
        stride = small_cfg.cssc_sentence_stride

        x = torch.randn(2, T, D)
        pooled = CSSCAttention._causal_pool(x, stride)

        expected_T_k = max(1, T // stride)
        assert pooled.shape == (2, expected_T_k, D), \
            f"Expected (2, {expected_T_k}, {D}), got {pooled.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# Attention masks
# ─────────────────────────────────────────────────────────────────────────────

class TestAttentionMasks:
    """Window mask and decay bias must enforce causality."""

    def test_window_mask_is_causal(self):
        """Future positions (j > i) must be -inf."""
        T, W = 16, 4
        mask = _window_mask(T, W, device=torch.device("cpu"))

        # Check shape
        assert mask.shape == (1, 1, T, T)

        # Check causality: mask[i, j] should be -inf when j > i
        m = mask[0, 0]
        for i in range(T):
            for j in range(T):
                if j > i:
                    assert m[i, j] == float("-inf"), \
                        f"Future position mask[{i},{j}] should be -inf"

    def test_window_mask_limits_past(self):
        """Positions beyond the window (j < i - window) must be -inf."""
        T, W = 16, 4
        mask = _window_mask(T, W, device=torch.device("cpu"))
        m = mask[0, 0]

        for i in range(T):
            for j in range(T):
                if j <= i and (i - j) >= W:
                    assert m[i, j] == float("-inf"), \
                        f"Out-of-window mask[{i},{j}] should be -inf (dist={i-j})"

    def test_hyperbolic_decay_is_causal(self):
        """Future positions in the decay bias must be -inf."""
        T = 16
        bias = _hyperbolic_decay_bias(T, T, alpha=0.5, device=torch.device("cpu"))
        b = bias[0, 0]

        for i in range(T):
            for j in range(T):
                if j > i:
                    assert b[i, j] == float("-inf"), \
                        f"Future bias[{i},{j}] should be -inf"

    def test_decay_bias_decreases_with_distance(self):
        """Closer positions should have higher (less negative) bias."""
        T = 16
        bias = _hyperbolic_decay_bias(T, T, alpha=0.5, device=torch.device("cpu"))
        b = bias[0, 0]

        # For row i=8, bias should decrease as distance increases
        for j in range(1, 8):
            assert b[8, 8 - j] >= b[8, 8 - j - 1], \
                f"Bias should decrease with distance at row 8"


# ─────────────────────────────────────────────────────────────────────────────
# CSSCAttention module
# ─────────────────────────────────────────────────────────────────────────────

class TestCSSCAttention:
    """Integration tests for the full CSSC module."""

    def test_output_shape(self, small_cfg):
        """Output must be (B, T, D) matching input."""
        cssc = CSSCAttention(small_cfg)
        B, T, D = 2, small_cfg.max_seq_len, small_cfg.d_model
        x = torch.randn(B, T, D)

        out = cssc(x, c=1.0)
        assert out.shape == (B, T, D), \
            f"Expected ({B}, {T}, {D}), got {out.shape}"

    def test_scale_weights_sum_to_one(self, small_cfg):
        """Learned scale weights should always sum to 1 (softmax)."""
        cssc = CSSCAttention(small_cfg)
        weights = cssc.scale_weights
        total = sum(weights)
        assert abs(total - 1.0) < 1e-5, \
            f"Scale weights should sum to 1.0, got {total}"

    def test_context_efficiency_range(self, small_cfg):
        """Context efficiency must be in [0, 1]."""
        cssc = CSSCAttention(small_cfg)
        ce = cssc.context_efficiency
        assert 0.0 <= ce <= 1.0, \
            f"Context efficiency should be in [0, 1], got {ce}"

    def test_zero_init_output_proj(self, small_cfg):
        """out_proj is zero-initialised → CSSC output should start near zero."""
        cssc = CSSCAttention(small_cfg)
        B, T, D = 2, small_cfg.max_seq_len, small_cfg.d_model
        x = torch.randn(B, T, D)

        out = cssc(x, c=1.0)

        # With zero-init out_proj, output should be very small
        assert out.abs().max().item() < 1e-5, \
            f"Zero-init out_proj should produce near-zero output, " \
            f"got max={out.abs().max():.6f}"

    def test_gradient_flows(self, small_cfg):
        """Gradients should flow through the full CSSC path."""
        cssc = CSSCAttention(small_cfg)
        D = small_cfg.d_model
        T = small_cfg.max_seq_len

        x = torch.randn(2, T, D, requires_grad=True)
        out = cssc(x, c=1.0)
        out.sum().backward()

        assert x.grad is not None, "No gradient through CSSC"
        assert not x.grad.isnan().any(), "NaN gradient through CSSC"

    def test_no_nan_output(self, small_cfg):
        """CSSC should not produce NaN on normal inputs."""
        cssc = CSSCAttention(small_cfg)
        D = small_cfg.d_model
        T = small_cfg.max_seq_len

        x = torch.randn(2, T, D)
        out = cssc(x, c=1.0)

        assert not out.isnan().any(), "CSSC produced NaN"
        assert not out.isinf().any(), "CSSC produced Inf"

    def test_curvature_tensor_input(self, small_cfg):
        """CSSC should accept both float and tensor curvature."""
        cssc = CSSCAttention(small_cfg)
        D = small_cfg.d_model
        T = small_cfg.max_seq_len
        x = torch.randn(2, T, D)

        # Float curvature
        out1 = cssc(x, c=1.0)
        assert not out1.isnan().any()

        # Tensor curvature (for learnable curvature)
        c_tensor = torch.tensor(1.0, requires_grad=True)
        out2 = cssc(x, c=c_tensor)
        assert not out2.isnan().any()
