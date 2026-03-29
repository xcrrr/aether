"""Tests for model components (model.py + aether2_model.py).

Covers building blocks (RMSNorm, SwiGLU, SSM), full model forward
passes, weight tying, and the shadow model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from model import (
    RMSNorm,
    SwiGLU,
    PurePyTorchSSM,
    EpisodicMemory,
    ContinuousThoughtTokens,
    exp_map_zero,
    project_to_ball,
)
from aether2_model import Aether2Model, Aether2Block, ShadowModel


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────

class TestRMSNorm:
    """RMSNorm should normalise without mean subtraction."""

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_unit_rms(self):
        """After normalisation, RMS of output should be ~1 (before weight)."""
        norm = RMSNorm(64)
        # Set weight to ones (default) so output RMS ≈ 1
        x = torch.randn(2, 16, 64)
        out = norm(x)
        rms = out.float().pow(2).mean(dim=-1).sqrt()
        # RMS should be close to 1 (the learned weight is all ones at init)
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=0.1, rtol=0.1)

    def test_dtype_preservation(self):
        """Output dtype should match input dtype."""
        norm = RMSNorm(64)
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(2, 4, 64, dtype=dtype)
            out = norm(x)
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU
# ─────────────────────────────────────────────────────────────────────────────

class TestSwiGLU:
    """SwiGLU FFN: output shape and no-NaN checks."""

    def test_output_shape(self):
        ffn = SwiGLU(64, 128)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == (2, 16, 64)

    def test_no_nan(self):
        ffn = SwiGLU(64, 128)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert not out.isnan().any()

    def test_gradient_flows(self):
        ffn = SwiGLU(64, 128)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# PurePyTorchSSM
# ─────────────────────────────────────────────────────────────────────────────

class TestSSM:
    """Pure-PyTorch Mamba SSM — shape and delta output tests."""

    def test_output_shape(self, omega_cfg):
        ssm = PurePyTorchSSM(omega_cfg, d_state_override=8)
        B, T, D = 2, 16, omega_cfg.d_model
        x = torch.randn(B, T, D)
        out = ssm(x)
        assert out.shape == (B, T, D)

    def test_return_delta(self, omega_cfg):
        """When return_delta=True, should return (output, delta_mean)."""
        ssm = PurePyTorchSSM(omega_cfg, d_state_override=8)
        B, T, D = 2, 16, omega_cfg.d_model
        x = torch.randn(B, T, D)
        out, delta = ssm(x, return_delta=True)

        assert out.shape == (B, T, D)
        assert delta.shape == (B, T, 1), \
            f"delta_mean should be (B, T, 1), got {delta.shape}"

    def test_no_nan(self, omega_cfg):
        ssm = PurePyTorchSSM(omega_cfg, d_state_override=8)
        x = torch.randn(2, 16, omega_cfg.d_model)
        out = ssm(x)
        assert not out.isnan().any(), "SSM produced NaN"


# ─────────────────────────────────────────────────────────────────────────────
# EpisodicMemory
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodicMemory:
    """Differentiable key-value memory bank."""

    def test_output_shape(self):
        mem = EpisodicMemory(slots=32, d_model=64, topk=4)
        query = torch.randn(2, 16, 64)
        out = mem(query)
        assert out.shape == (2, 16, 64)

    def test_gate_starts_near_zero(self):
        """Gate is init 0 → tanh(0) = 0 → output should be near zero."""
        mem = EpisodicMemory(slots=32, d_model=64, topk=4)
        query = torch.randn(2, 16, 64)
        out = mem(query)

        assert out.abs().max().item() < 1e-5, \
            "Gate init=0 → tanh(0)=0 → output should be near zero"


# ─────────────────────────────────────────────────────────────────────────────
# ContinuousThoughtTokens
# ─────────────────────────────────────────────────────────────────────────────

class TestContinuousThoughtTokens:
    """Learnable scratchpad embeddings prepended to sequences."""

    def test_prepend_shape(self):
        ctt = ContinuousThoughtTokens(n_tokens=4, d_model=64)
        x = torch.randn(2, 16, 64)
        x_ball = project_to_ball(x, 1.0)
        out = ctt.prepend(x_ball, c=1.0)

        assert out.shape == (2, 20, 64), \
            f"Expected (2, 20, 64), got {out.shape}"

    def test_original_content_preserved(self):
        """Original tokens should be in positions [K:] after prepend."""
        ctt = ContinuousThoughtTokens(n_tokens=4, d_model=64)
        x = torch.randn(2, 16, 64) * 0.01
        x_ball = project_to_ball(x, 1.0)
        out = ctt.prepend(x_ball, c=1.0)

        # The last 16 positions should be the original sequence
        torch.testing.assert_close(out[:, 4:, :], x_ball, atol=1e-6, rtol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Aether2Block
# ─────────────────────────────────────────────────────────────────────────────

class TestAether2Block:
    """Single Aether 2 transformer block tests."""

    def test_output_shape(self, small_cfg):
        """Block should return (output, aux_loss) with correct shapes."""
        block = Aether2Block(small_cfg, layer_idx=0)
        D = small_cfg.d_model
        B, T = 2, small_cfg.max_seq_len

        # Input must be in the Poincaré ball
        x = torch.randn(B, T, D) * 0.1
        x = project_to_ball(exp_map_zero(x, 1.0), 1.0)

        out, aux = block(x)

        assert out.shape == (B, T, D), \
            f"Expected ({B}, {T}, {D}), got {out.shape}"
        assert aux.shape == () or aux.numel() == 1, \
            "aux_loss should be scalar"

    def test_output_stays_in_ball(self, small_cfg):
        """Block output should be inside the Poincaré ball."""
        block = Aether2Block(small_cfg, layer_idx=0, use_cssc=False, use_ggr=False)
        D = small_cfg.d_model
        B, T = 2, small_cfg.max_seq_len

        x = torch.randn(B, T, D) * 0.1
        x = project_to_ball(exp_map_zero(x, 1.0), 1.0)

        out, _ = block(x)
        norms = out.norm(dim=-1)

        assert norms.max().item() < 1.0, \
            f"Block output should be inside the unit ball, max norm = {norms.max():.4f}"

    def test_no_nan(self, small_cfg):
        block = Aether2Block(small_cfg, layer_idx=0)
        D = small_cfg.d_model
        x = torch.randn(2, small_cfg.max_seq_len, D) * 0.1
        x = project_to_ball(exp_map_zero(x, 1.0), 1.0)

        out, aux = block(x)
        assert not out.isnan().any(), "Block produced NaN"


# ─────────────────────────────────────────────────────────────────────────────
# Aether2Model (full model)
# ─────────────────────────────────────────────────────────────────────────────

class TestAether2Model:
    """Full model forward pass shape and weight tying tests."""

    def test_forward_shape(self, small_cfg):
        """Logits should be (B, T + n_thought, V)."""
        model = Aether2Model(small_cfg)
        B, T = 2, small_cfg.max_seq_len
        K = small_cfg.n_thought_tokens
        V = small_cfg.vocab_size

        ids = torch.randint(0, V, (B, T))
        logits, hiddens, aux = model(ids)

        assert logits.shape == (B, T + K, V), \
            f"Expected ({B}, {T + K}, {V}), got {logits.shape}"

    def test_weight_tying(self, small_cfg):
        """embedding.weight and lm_head.weight must be the same tensor."""
        model = Aether2Model(small_cfg)

        assert model.embedding.weight is model.lm_head.weight, \
            "Weight tying broken: embedding.weight is not lm_head.weight"

    def test_no_nan_forward(self, small_cfg):
        """Full forward pass should not produce NaN."""
        model = Aether2Model(small_cfg)
        B, T = 2, small_cfg.max_seq_len
        ids = torch.randint(0, small_cfg.vocab_size, (B, T))

        logits, _, aux = model(ids)
        assert not logits.isnan().any(), "Model produced NaN logits"
        assert not aux.isnan(), "Model produced NaN aux loss"

    def test_backward_no_error(self, small_cfg):
        """Full backward pass should complete without error."""
        model = Aether2Model(small_cfg)
        B, T = 2, small_cfg.max_seq_len
        ids = torch.randint(0, small_cfg.vocab_size, (B, T))

        logits, _, aux = model(ids)
        loss = logits.sum() + aux
        loss.backward()

        # Check at least some parameters got gradients
        n_grads = sum(
            1 for p in model.parameters()
            if p.requires_grad and p.grad is not None
        )
        assert n_grads > 0, "No parameters received gradients"

    def test_parameter_count(self, small_cfg):
        """Parameter count should be positive and reasonable."""
        model = Aether2Model(small_cfg)
        count = model.count_parameters()
        assert count > 0, "Model should have parameters"
        # With tiny config, should be < 10M params
        assert count < 10_000_000, \
            f"Tiny config should have < 10M params, got {count}"


# ─────────────────────────────────────────────────────────────────────────────
# ShadowModel
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowModel:
    """Shadow baseline model for EMA comparison."""

    def test_forward_shape(self, small_cfg):
        """Shadow model should return logits (B, T, V)."""
        shadow = ShadowModel(small_cfg)
        B, T = 2, small_cfg.max_seq_len
        V = small_cfg.vocab_size

        ids = torch.randint(0, V, (B, T))
        logits = shadow(ids)

        assert logits.shape == (B, T, V), \
            f"Expected ({B}, {T}, {V}), got {logits.shape}"

    def test_no_grad(self, small_cfg):
        """Shadow parameters should not require gradients."""
        shadow = ShadowModel(small_cfg)
        for p in shadow.parameters():
            assert not p.requires_grad, \
                "Shadow model parameters should not require grad"

    def test_ema_update(self, small_cfg):
        """EMA update should modify shadow weights towards main model."""
        model = Aether2Model(small_cfg)
        shadow = ShadowModel(small_cfg)

        # Get initial shadow weight
        with torch.no_grad():
            # Find a matching parameter
            s_params = dict(shadow.blocks[0].named_parameters())
            m_params = dict(model.blocks[0].named_parameters())
            for name in s_params:
                if name in m_params and s_params[name].shape == m_params[name].shape:
                    initial = s_params[name].clone()
                    break
            else:
                pytest.skip("No matching param found")

        # Run EMA update
        shadow.ema_update(model.blocks, decay=0.0)  # decay=0 → full copy

        # After decay=0 update, shadow should equal main model
        final = s_params[name]
        expected = m_params[name].to(final.dtype)
        torch.testing.assert_close(final, expected, atol=1e-4, rtol=1e-4)
