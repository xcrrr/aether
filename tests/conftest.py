"""Shared fixtures for the Aether 2 test suite.

All tests run on CPU to be hardware-agnostic.  Configs use small
dimensions so tests finish in seconds, not minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aether2_config import Aether2Config
from aether_config import OmegaConfig


@pytest.fixture
def small_cfg() -> Aether2Config:
    """Tiny Aether2Config for fast CPU tests.

    d_model=64, n_layers=2, seq_len=32 — runs in ~100ms per forward pass.
    """
    return Aether2Config(
        d_model=64,
        n_layers=2,
        max_seq_len=32,
        n_thought_tokens=2,
        ff_hidden=128,           # must be divisible by ggr_n_experts (4)
        d_state=4,
        d_conv=4,
        expand=2,
        dt_rank=8,
        timescale_d_states=(4, 8, 16),
        vocab_size=256,
        n_moe_experts=4,
        moe_layer_stride=4,
        ggr_n_experts=4,
        ggr_top_k=2,
        ggr_layer_stride=2,
        ggr_entropy_probe_dim=16,
        cssc_n_heads=4,
        cssc_window_size=16,
        cssc_sentence_stride=8,
        cssc_block_size=32,
        episodic_slots=16,
        episodic_topk=4,
        use_gradient_checkpointing=False,
        residual_dropout=0.0,
        learnable_curvature=False,  # simpler for unit tests
        geometry_gating=False,
        riemannian_correction=False,
        micro_moe_enabled=True,
        cssc_enabled=True,
        baseline_enabled=True,
        baseline_n_layers=2,
        rosetta_enabled=False,
        fpa_enabled=False,
        # Rosetta probe (even if disabled, config must be valid)
        rosetta_d_probe=32,
        rosetta_n_layers=1,
        rosetta_n_heads=4,
    )


@pytest.fixture
def omega_cfg() -> OmegaConfig:
    """Tiny OmegaConfig for testing model.py components."""
    return OmegaConfig(
        d_model=64,
        n_layers=2,
        max_seq_len=32,
        n_thought_tokens=2,
        ff_hidden=192,    # must be divisible by n_moe_experts (3)
        d_state=4,
        d_conv=4,
        expand=2,
        dt_rank=8,
        timescale_d_states=(4, 8, 16),
        vocab_size=256,
        n_moe_experts=3,
        moe_layer_stride=4,
        episodic_slots=16,
        episodic_topk=4,
        use_gradient_checkpointing=False,
        residual_dropout=0.0,
        learnable_curvature=False,
        geometry_gating=False,
        riemannian_correction=False,
        micro_moe_enabled=False,
        cssc_enabled=False,
        rosetta_d_probe=32,
        rosetta_n_layers=1,
        rosetta_n_heads=4,
    )


@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def B() -> int:
    """Batch size for tests."""
    return 2


@pytest.fixture
def T() -> int:
    """Sequence length for tests."""
    return 32
