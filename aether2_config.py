"""Aether 2 — Global Configuration System.

Extends OmegaConfig with new CSSC v2 (Cross-Scale Spatiotemporal Correlation)
and GGR v2 (Gated Gradient Routing) parameters, plus ablation flags.

CLI flags: --no-cssc, --no-ggr, --no-baseline
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Callable

from aether_config import OmegaConfig


@dataclass
class Aether2Config(OmegaConfig):
    # ── Override parent defaults for 4-expert GGR ─────────────────────────
    # Parent OmegaConfig: ff_hidden=2817 (divisible by 3), n_moe_experts=3
    # Aether2 uses 4 experts → ff_hidden must be divisible by 4
    ff_hidden: int = 2_816           # 2816 = 4×704; replaces parent 2817
    n_moe_experts: int = 4           # keep parent assertion happy

    # ── Picky Learner pre-training override ────────────────────────────────
    # Parent default picky_ce_max=5.0 was designed for fine-tuning.
    # Pre-training starts at CE≈10 (random init); at step 2000 the parent
    # value would reject every batch → training stalls.
    # 12.0 filters only genuinely broken/corrupted batches (CE > 12 is
    # impossible for a healthy sequence at any point in pre-training).
    picky_ce_max: float = 12.0

    # ── Memory-safe training defaults (override parent for 16 GB VRAM) ────
    # micro_batch=4 + grad_accum=16 keeps effective batch=64.
    # VRAM budget with cpu_offload_optimizer=False (default):
    #   model BF16 (696M)  ~1.4 GiB
    #   gradients BF16     ~1.4 GiB
    #   Adam m/v FP32      ~5.6 GiB
    #   activations        ~1.5 GiB  (gradient checkpointing ON)
    #   shadow model       ~0.5 GiB
    #   ─────────────────────────────
    #   peak total         ~10.4 GiB  (well within 16 GiB)
    micro_batch: int = 4             # parent default
    grad_accum_steps: int = 16       # parent default; effective batch stays 64

    # CPU offload for Adam m/v states.
    # OFF by default: the RX 7800 XT has 16 GiB — Adam states (~5.6 GiB) fit
    # comfortably alongside the model.  Enabling this adds PCIe round-trips
    # (~5.6 GiB per optimizer step) and cuts throughput by ~3–5×.
    # Enable only if you hit OOM (e.g. on a GPU with < 12 GiB VRAM).
    cpu_offload_optimizer: bool = False

    # ── SSM scan: larger chunks → fewer Python loop iterations, same VRAM ──
    # Default OmegaConfig chunk_size=64 gives T/64 = 8 Python iterations per SSM.
    # At chunk_size=256, T/256 = ⌈520/256⌉ = 3 iterations — 62% fewer calls.
    # Per-chunk gradient checkpointing is kept (scan_use_chunk_ckpt=True) so
    # VRAM stays bounded: only ONE chunk's (B,chunk,Di,N) intermediates are
    # live at a time during backward.  chunk_size=256 uses 4× more peak activation
    # per chunk (~130 MB vs ~33 MB) but still well within the 16 GiB budget.
    scan_use_chunk_ckpt: bool = True    # keep per-chunk ckpt for VRAM safety
    scan_chunk_size: int = 128          # 5 chunks instead of 8 (default 64); +0.86 GB peak

    # ── CSSC v2 — Cross-Scale Spatiotemporal Correlation ──────────────────
    # Replaces the Mamba-Δ curvature coupling with a full multi-head
    # temporal attention mechanism operating at three distinct scales.
    cssc_n_heads: int = 8
    # Token-level local window (in tokens)
    cssc_window_size: int = 64
    # Sentence-level: group every N tokens into a "sentence" segment
    cssc_sentence_stride: int = 32
    # Block-level: group every N tokens into a "block" segment
    cssc_block_size: int = 256
    # Hyperbolic decay α: weight(t) = 1 / (1 + α · |temporal_distance|)
    cssc_decay_alpha: float = 0.5
    # Learnable blend weights for (token, sentence, block) scales
    cssc_scale_init: tuple[float, float, float] = (0.5, 0.3, 0.2)
    # Dropout on CSSC attention scores
    cssc_attn_dropout: float = 0.05

    # ── GGR v2 — Gated Gradient Routing ───────────────────────────────────
    # Four-expert routing with entropy-based sparsity gating and gradient
    # stability via internal gate normalization.
    ggr_n_experts: int = 4                  # Math / Code / Logic / General
    ggr_expert_names: tuple[str, ...] = ("Math", "Code", "Logic", "General")
    ggr_top_k: int = 2                       # Top-k sparse routing (0 = soft/all)
    # Temperature for soft routing logits
    ggr_temperature: float = 1.0
    # Load-balance auxiliary loss weight (Switch Transformer style)
    ggr_lb_weight: float = 0.01
    # Jitter added to routing logits during training to prevent collapse
    ggr_jitter_eps: float = 0.01
    # Dimension of the complexity probe for entropy computation
    ggr_entropy_probe_dim: int = 64
    # Every N layers gets a GGR router (0 = all layers)
    ggr_layer_stride: int = 2

    # ── Fluid Power Allocation (Phase A: Dynamic Test-Time Compute) ──────────
    # Entropy-conditioned adaptive iteration: uncertain tokens get up to
    # fpa_max_iters extra passes through the full model (including 6× GGR blocks).
    # Confident tokens early-exit.  Set fpa_enabled=True to activate.
    fpa_enabled: bool = False
    # Maximum extra re-routing passes (4 total: 1 base + 3 extra → ~11.2 GiB peak)
    fpa_max_iters: int = 3
    # ACT regulariser weight.  Higher → more early exits; lower → more iterations.
    # Tune: if avg iters stays at max → raise to 0.05; collapses to 1 → lower to 0.005
    fpa_ponder_weight: float = 0.01

    # ── RosettaObserver ────────────────────────────────────────────────────
    # Secondary ~25M-param decoder that runs on detached main-model hidden
    # states.  Stop-gradient: Rosetta's loss never flows into the main model.
    # Purpose: decode what the model encodes in Poincaré space at every step.
    rosetta_enabled: bool = True

    # ── Baseline Shadow Model ──────────────────────────────────────────────
    # Lightweight EMA-updated vanilla transformer running in a shadow buffer
    # to provide real-time DELTA comparisons vs. Aether 2.
    baseline_enabled: bool = True
    # Number of transformer layers in the shadow model
    baseline_n_layers: int = 4
    # Model dimension (intentionally smaller for memory efficiency)
    baseline_d_model: int = 256
    baseline_n_heads: int = 4
    baseline_ff_hidden: int = 512
    # EMA decay for shadow weight updates from the main model's first N layers
    baseline_ema_decay: float = 0.999

    # ── Streaming Dataset ──────────────────────────────────────────────────
    data_path: str = "data/aether_train.jsonl"
    # Pre-allocated circular token buffer size (in tokens, ~2 MB at int16)
    stream_buffer_tokens: int = 2_097_152   # 2M tokens
    stream_prefetch_batches: int = 16       # batches pre-loaded by producer
    stream_num_workers: int = 1

    # ── Device & precision ─────────────────────────────────────────────────
    device: str = "cuda"             # ROCm surfaces as "cuda"
    use_bf16: bool = True            # BF16 mixed precision (preferred on ROCm)

    # ── Logging ────────────────────────────────────────────────────────────
    log_dir: str = "logs"
    log_file: str = "logs/aether_build.log"

    # ── Training overrides ─────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_aether2"

    # ── Derived helpers ────────────────────────────────────────────────────
    @property
    def ggr_expert_hidden(self) -> int:
        """Hidden dim per GGR expert (capacity-neutral)."""
        return self.ff_hidden // self.ggr_n_experts

    @property
    def cssc_head_dim(self) -> int:
        return self.d_model // self.cssc_n_heads

    def __post_init__(self) -> None:
        # Run parent validations
        super().__post_init__()
        assert self.d_model % self.cssc_n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by cssc_n_heads ({self.cssc_n_heads})"
        )
        assert self.ggr_n_experts == len(self.ggr_expert_names), (
            "ggr_n_experts must match len(ggr_expert_names)"
        )
        assert self.ggr_top_k <= self.ggr_n_experts, (
            "ggr_top_k must be ≤ ggr_n_experts"
        )
        assert self.ff_hidden % self.ggr_n_experts == 0, (
            f"ff_hidden ({self.ff_hidden}) must be divisible by ggr_n_experts "
            f"({self.ggr_n_experts}) for capacity-neutral expert routing"
        )
        assert len(self.cssc_scale_init) == 3, "cssc_scale_init must have 3 values"
        assert self.cssc_block_size >= self.cssc_sentence_stride >= 1

    def summary(self) -> str:
        base = super().summary()
        lines = [
            base,
            "─" * 54,
            "  Aether 2 — Extended Configuration",
            "─" * 54,
            f"  CSSC v2        : {'ON' if self.cssc_enabled else 'OFF'}  "
            f"(heads={self.cssc_n_heads}, window={self.cssc_window_size}, "
            f"α={self.cssc_decay_alpha})",
            f"  CSSC scales    : token={self.cssc_window_size}, "
            f"sentence={self.cssc_sentence_stride}, block={self.cssc_block_size}",
            f"  GGR v2         : {'ON' if self.micro_moe_enabled else 'OFF'}  "
            f"(experts={self.ggr_n_experts}, top-k={self.ggr_top_k}, "
            f"stride={self.ggr_layer_stride})",
            f"  GGR experts    : {' | '.join(self.ggr_expert_names)}",
            f"  Baseline       : {'ON' if self.baseline_enabled else 'OFF'}  "
            f"(d={self.baseline_d_model}, L={self.baseline_n_layers})",
            f"  CPU opt offload: {'ON — Adam m/v in CPU RAM' if self.cpu_offload_optimizer else 'OFF'}",
            f"  Fluid Power    : {'ON' if self.fpa_enabled else 'OFF'}  "
            f"(max_iters={self.fpa_max_iters}, ponder_w={self.fpa_ponder_weight})",
            f"  Data path      : {self.data_path}",
            f"  Stream buffer  : {self.stream_buffer_tokens // 1024}K tokens",
            f"  Log file       : {self.log_file}",
            "─" * 54,
        ]
        return "\n".join(lines)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Aether2Config":
        """Build config from parsed CLI args, applying ablation flags."""
        cfg = cls()
        # Ablation flags
        if getattr(args, "no_cssc", False):
            cfg.cssc_enabled = False
        if getattr(args, "no_ggr", False):
            cfg.micro_moe_enabled = False
        if getattr(args, "no_baseline", False):
            cfg.baseline_enabled = False
        if getattr(args, "no_rosetta", False):
            cfg.rosetta_enabled = False
        # CPU offload: default OFF; --cpu-offload enables, --no-cpu-offload disables
        if getattr(args, "cpu_offload", False):
            cfg.cpu_offload_optimizer = True
        if getattr(args, "no_cpu_offload", False):
            cfg.cpu_offload_optimizer = False
        if getattr(args, "fpa", False):
            cfg.fpa_enabled = True
        if getattr(args, "no_bf16", False):
            cfg.use_bf16 = False
        for attr in ("fpa_max_iters", "fpa_ponder_weight"):
            if hasattr(args, attr) and getattr(args, attr) is not None:
                object.__setattr__(cfg, attr, getattr(args, attr))
        # Training overrides
        for attr in ("max_steps", "micro_batch", "grad_accum_steps",
                     "learning_rate", "seed", "data_path", "checkpoint_every",
                     "checkpoint_dir", "log_file", "log_every"):
            if hasattr(args, attr) and getattr(args, attr) is not None:
                object.__setattr__(cfg, attr, getattr(args, attr))
        return cfg

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Register all Aether 2 CLI arguments onto an ArgumentParser."""
        # Ablation
        parser.add_argument("--no-cssc", action="store_true",
                            help="Disable CSSC (run vanilla attention)")
        parser.add_argument("--no-ggr", action="store_true",
                            help="Disable GGR (run standard SwiGLU FFN)")
        parser.add_argument("--no-baseline", action="store_true",
                            help="Disable shadow baseline model")
        parser.add_argument("--no-rosetta", action="store_true",
                            help="Disable RosettaObserver probe")
        parser.add_argument("--cpu-offload", action="store_true",
                            help="Enable CPU optimizer offload (use on GPUs with < 12 GiB VRAM)")
        parser.add_argument("--no-cpu-offload", action="store_true",
                            help="Disable CPU optimizer offload (default; kept for backward compat)")
        parser.add_argument("--fpa", action="store_true",
                            help="Enable Fluid Power Allocation (dynamic test-time compute)")
        parser.add_argument("--fpa-max-iters", type=int, default=None,
                            dest="fpa_max_iters",
                            help="Max extra re-routing passes (default 3 → 4 total)")
        parser.add_argument("--fpa-ponder-weight", type=float, default=None,
                            dest="fpa_ponder_weight",
                            help="ACT regulariser weight (default 0.01)")
        parser.add_argument("--no-bf16", action="store_true",
                    help="Disable BF16 and keep weights/activations in FP32")
        # Training
        parser.add_argument("--max-steps", type=int, default=None)
        parser.add_argument("--micro-batch", type=int, default=None)
        parser.add_argument("--grad-accum", type=int, default=None,
                            dest="grad_accum_steps")
        parser.add_argument("--lr", type=float, default=None,
                            dest="learning_rate")
        parser.add_argument("--checkpoint-every", type=int, default=None,
                            dest="checkpoint_every",
                            help="Save checkpoint every N steps (default 2000)")
        parser.add_argument("--log-every", type=int, default=None,
                    dest="log_every",
                    help="Emit training metrics every N steps (default 50)")
        parser.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Checkpoint output directory (use a Drive path in Colab)")
        parser.add_argument("--log-file", type=str, default=None,
                    help="Log file path (use a Drive path in Colab if desired)")
        parser.add_argument("--data-path", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--dashboard", action="store_true",
                            help="Launch Aether Command Deck (Rich UI)")
        parser.add_argument("--resume", type=str, default=None,
                            metavar="CHECKPOINT_PREFIX")


if __name__ == "__main__":
    cfg = Aether2Config()
    print(cfg.summary())
