"""Tests for GGR — Gated Gradient Routing (ggr.py).

Validates load-balance loss properties, expert routing mechanics,
gradient stability index, and gradient flow through the full
entropy-conditioned sparse MoE path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from ggr import GatedGradientRouter, _ExpertFFN, _SparsityController


# ─────────────────────────────────────────────────────────────────────────────
# Load-balance auxiliary loss
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadBalanceLoss:
    """Switch Transformer load-balance loss must be correctly implemented."""

    def test_uniform_routing_yields_minimum_loss(self, small_cfg):
        """With uniform soft gates and top-k selection, the LB loss equals top_k.

        When all gates are exactly 1/N, topk selects k experts arbitrarily,
        giving f_i = 1 for k experts, f_i = 0 for the rest.
        P_i = 1/N for all. Loss = N × Σ f_i × P_i = N × k × (1/N) = k.
        """
        ggr = GatedGradientRouter(small_cfg)
        N = small_cfg.ggr_n_experts
        k = small_cfg.ggr_top_k
        B, T = 2, 16

        # Perfectly uniform gates: each expert gets exactly 1/N
        uniform_gates = torch.ones(B, T, N) / N

        loss = ggr._load_balance_loss(uniform_gates)

        expected = float(k)
        assert abs(loss.item() - expected) < 1e-4, \
            f"Uniform routing with top-{k} should give loss={expected}, " \
            f"got {loss.item():.6f}"

    def test_collapsed_routing_yields_high_loss(self, small_cfg):
        """When all tokens go to one expert, loss should be > 1.0."""
        ggr = GatedGradientRouter(small_cfg)
        N = small_cfg.ggr_n_experts
        B, T = 2, 16

        # All routing weight on expert 0
        collapsed_gates = torch.zeros(B, T, N)
        collapsed_gates[..., 0] = 1.0

        loss = ggr._load_balance_loss(collapsed_gates)

        # With collapsed routing: f_0 = 1, P_0 = 1, f_i = P_i = 0 for i>0
        # Loss = N × (1 × 1) = N
        expected = float(N)
        assert abs(loss.item() - expected) < 0.5, \
            f"Collapsed routing should give loss≈{expected}, got {loss.item():.4f}"

    def test_loss_is_differentiable(self, small_cfg):
        """The LB loss must produce gradients for the training signal."""
        ggr = GatedGradientRouter(small_cfg)
        B, T, D = 2, 16, small_cfg.d_model

        x = torch.randn(B, T, D, requires_grad=True)
        out, aux_loss = ggr(x)
        aux_loss.backward()

        # Gradients should flow through the router into the input
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in ggr.controller.parameters()
        )
        assert has_grad, "Load-balance loss should produce router gradients"


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Stability Index (GSI)
# ─────────────────────────────────────────────────────────────────────────────

class TestGSI:
    """GSI = 1 - std(loads)/mean(loads).  Range: [0, 1]."""

    def test_balanced_loads_give_high_gsi(self, small_cfg):
        """Perfectly balanced expert loads → GSI = 1.0."""
        ggr = GatedGradientRouter(small_cfg)
        N = small_cfg.ggr_n_experts

        # Manually set balanced loads
        ggr._last_gates = [1.0 / N] * N
        gsi = ggr.gradient_stability_index

        assert abs(gsi - 1.0) < 1e-5, \
            f"Balanced loads should give GSI=1.0, got {gsi:.6f}"

    def test_collapsed_loads_give_low_gsi(self, small_cfg):
        """All load on one expert → GSI near 0."""
        ggr = GatedGradientRouter(small_cfg)
        N = small_cfg.ggr_n_experts

        collapsed = [0.0] * N
        collapsed[0] = 1.0
        ggr._last_gates = collapsed
        gsi = ggr.gradient_stability_index

        assert gsi < 0.3, \
            f"Collapsed loads should give low GSI, got {gsi:.4f}"

    def test_gsi_range(self, small_cfg):
        """GSI must always be in [0, 1]."""
        ggr = GatedGradientRouter(small_cfg)
        B, T, D = 2, 16, small_cfg.d_model

        x = torch.randn(B, T, D)
        ggr(x)  # triggers gate update

        gsi = ggr.gradient_stability_index
        assert 0.0 <= gsi <= 1.0, f"GSI out of range: {gsi}"


# ─────────────────────────────────────────────────────────────────────────────
# Expert routing
# ─────────────────────────────────────────────────────────────────────────────

class TestExpertRouting:
    """Expert routing, gating, and output shape tests."""

    def test_output_shape(self, small_cfg):
        """Forward output must be (B, T, D)."""
        ggr = GatedGradientRouter(small_cfg)
        B, T, D = 2, 16, small_cfg.d_model

        x = torch.randn(B, T, D)
        out, aux = ggr(x)

        assert out.shape == (B, T, D), \
            f"Expected ({B}, {T}, {D}), got {out.shape}"
        assert aux.shape == (), f"aux_loss should be scalar, got {aux.shape}"

    def test_expert_load_sums_to_one(self, small_cfg):
        """Expert load fractions should sum to approximately 1."""
        ggr = GatedGradientRouter(small_cfg)
        B, T, D = 2, 16, small_cfg.d_model

        x = torch.randn(B, T, D)
        ggr(x)

        load = ggr.expert_load
        total = sum(load)
        assert abs(total - 1.0) < 0.05, \
            f"Expert loads should sum to ~1.0, got {total:.4f}"

    def test_no_nan_output(self, small_cfg):
        """GGR should not produce NaN on normal inputs."""
        ggr = GatedGradientRouter(small_cfg)
        B, T, D = 2, 16, small_cfg.d_model

        x = torch.randn(B, T, D)
        out, aux = ggr(x)

        assert not out.isnan().any(), "GGR produced NaN output"
        assert not aux.isnan().any(), "GGR produced NaN aux loss"

    def test_gradient_flows_to_all_experts(self, small_cfg):
        """With soft routing, gradients should reach all expert weights."""
        # Use top_k = N (full soft routing) so all experts get gradients
        small_cfg_copy = Aether2Config(
            d_model=small_cfg.d_model,
            n_layers=small_cfg.n_layers,
            max_seq_len=small_cfg.max_seq_len,
            n_thought_tokens=small_cfg.n_thought_tokens,
            ff_hidden=small_cfg.ff_hidden,
            d_state=small_cfg.d_state,
            d_conv=small_cfg.d_conv,
            expand=small_cfg.expand,
            dt_rank=small_cfg.dt_rank,
            timescale_d_states=small_cfg.timescale_d_states,
            vocab_size=small_cfg.vocab_size,
            n_moe_experts=small_cfg.n_moe_experts,
            moe_layer_stride=small_cfg.moe_layer_stride,
            ggr_n_experts=4,
            ggr_top_k=4,  # full soft routing
            ggr_layer_stride=small_cfg.ggr_layer_stride,
            ggr_entropy_probe_dim=small_cfg.ggr_entropy_probe_dim,
            cssc_n_heads=small_cfg.cssc_n_heads,
            cssc_window_size=small_cfg.cssc_window_size,
            cssc_sentence_stride=small_cfg.cssc_sentence_stride,
            cssc_block_size=small_cfg.cssc_block_size,
            episodic_slots=small_cfg.episodic_slots,
            episodic_topk=small_cfg.episodic_topk,
            use_gradient_checkpointing=False,
            residual_dropout=0.0,
            learnable_curvature=False,
            geometry_gating=False,
            riemannian_correction=False,
            micro_moe_enabled=True,
            cssc_enabled=True,
            rosetta_d_probe=32,
            rosetta_n_layers=1,
            rosetta_n_heads=4,
        )

        ggr = GatedGradientRouter(small_cfg_copy)
        B, T, D = 2, 16, small_cfg_copy.d_model

        x = torch.randn(B, T, D, requires_grad=True)
        out, aux = ggr(x)
        (out.sum() + aux).backward()

        for i, expert in enumerate(ggr.experts):
            has_grad = expert.gate_proj.weight.grad is not None and \
                       expert.gate_proj.weight.grad.abs().sum() > 0
            assert has_grad, \
                f"Expert {i} ({expert.name}) should have gradients"


# ─────────────────────────────────────────────────────────────────────────────
# Sparsity controller
# ─────────────────────────────────────────────────────────────────────────────

class TestSparsityController:
    """Entropy probe and router correctness."""

    def test_entropy_range(self, small_cfg):
        """Normalised entropy should be in [0, 1]."""
        ctrl = _SparsityController(
            small_cfg.d_model,
            small_cfg.ggr_entropy_probe_dim,
            small_cfg.ggr_n_experts,
        )
        x = torch.randn(2, 16, small_cfg.d_model)
        logits, entropy = ctrl(x)

        assert entropy.min() >= -0.01, f"Entropy below 0: {entropy.min():.4f}"
        assert entropy.max() <= 1.01, f"Entropy above 1: {entropy.max():.4f}"

    def test_routing_logits_shape(self, small_cfg):
        """Router should produce (B, T, N) logits."""
        ctrl = _SparsityController(
            small_cfg.d_model,
            small_cfg.ggr_entropy_probe_dim,
            small_cfg.ggr_n_experts,
        )
        x = torch.randn(2, 16, small_cfg.d_model)
        logits, entropy = ctrl(x)

        assert logits.shape == (2, 16, small_cfg.ggr_n_experts)
        assert entropy.shape == (2, 16)


# Need this import for the gradient flow test
from aether2_config import Aether2Config
