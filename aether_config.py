"""Aether Omega — Configuration dataclass.

All hyperparameters for the Aether Omega PoC model.
Optimised for AMD RX 7800 XT (16 GB VRAM, ROCm).

Target: ~445M parameters (main) + ~28M (RosettaObserver) = ~473M combined.
Hardware envelope: ~7.6 GiB VRAM at micro-batch=4, seq=512 (BF16, gradient checkpointing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class OmegaConfig:
    # ------------------------------------------------------------------ #
    # Model architecture                                                   #
    # ------------------------------------------------------------------ #
    vocab_size: int = 32_768        # Full 32k vocabulary
    d_model: int = 1_024            # Hidden dimension
    n_layers: int = 24              # Number of AetherMambaBlocks
    # Mamba SSM inner dimensions
    d_state: int = 16               # SSM latent state size (N) — default for uniform mode
    d_conv: int = 4                 # Local conv width inside SSM
    expand: int = 2                 # d_inner = d_model * expand
    dt_rank: int = 64               # Delta (Δ) projection rank; max(1, d_model//16)
    # Multi-Timescale SSM (Clockwork Mamba): d_state per layer group
    timescale_d_states: tuple[int, ...] = (8, 16, 32)  # fast / medium / slow
    # SwiGLU feed-forward hidden dim.  Must be divisible by n_moe_experts (3)
    # so GGR experts get exact equal capacity.  2817 = 3 × 939 ≈ ceil(d_model*8/3)
    ff_hidden: int = 2_817
    max_seq_len: int = 512

    # Continuous Thought Tokens (private scratchpad prepended to every sequence)
    n_thought_tokens: int = 8       # Learnable tokens; masked from CE loss

    # ------------------------------------------------------------------ #
    # Pillar 1 — Hyperbolic geometry (Poincaré ball)                      #
    # ------------------------------------------------------------------ #
    hyp_curvature: float = 1.0      # c; ball radius = 1/sqrt(c) = 1.0
    # v2 Architectural Upgrades
    riemannian_correction: bool = True   # Gradient rescaling for Poincaré metric
    geometry_gating: bool = True         # Learned Euclidean/Hyperbolic blend per block
    learnable_curvature: bool = True     # Per-block learnable curvature κ
    curvature_init: float = 0.1          # Initial curvature value
    curvature_max: float = 2.0           # Maximum curvature bound
    curvature_warmup_steps: int = 5000   # Steps to ramp curvature scale 0→1

    # ------------------------------------------------------------------ #
    # CSSC — Curvature-Selective State Coupling (novel: world-first)      #
    # Per-token Poincaré curvature driven by the Mamba Δ (delta) gate.   #
    # High-Δ (salient) tokens expand hyperbolic space; low-Δ contracts.  #
    # ------------------------------------------------------------------ #
    cssc_enabled: bool = True        # Couple SSM Δ gate to per-token curvature

    # ------------------------------------------------------------------ #
    # GGR — Geodesic Gravity Routing Micro-MoE (novel: world-first)      #
    # FFN replaced by 3 expert SwiGLUs routed by Poincaré geodesic dist. #
    # Capacity-neutral: each expert uses ff_hidden//n_moe_experts hidden. #
    # ------------------------------------------------------------------ #
    micro_moe_enabled: bool = True   # Use GGR-MoE in every moe_layer_stride-th block
    n_moe_experts: int = 3           # Code / Math / Language domain experts
    moe_layer_stride: int = 4        # GGR-MoE in blocks stride-1, 2*stride-1, … (0-indexed)

    # ------------------------------------------------------------------ #
    # Pillar 2 — Gradient-Based Neurogenesis                              #
    # ------------------------------------------------------------------ #
    delta_dim: int = 64                         # Neurons added per expansion event
    neurogenesis_patience: int = 500            # Steps below variance threshold
    neurogenesis_var_threshold: float = 1e-4    # Grad-variance collapse threshold
    neurogenesis_window: int = 5                # Moving-average window (steps)

    # ------------------------------------------------------------------ #
    # Pillar 3 — Anticipatory Picky Learner                               #
    # ------------------------------------------------------------------ #
    picky_ce_min: float = 0.2       # Drop batch if CE < this (too easy)
    picky_ce_max: float = 5.0       # Drop batch if CE > this (too hard / noisy)
    anticipatory_weight: float = 0.1
    anticipatory_steps: int = 5     # In-sequence horizon (positions, not steps)
    # Iterative Refinement ("Think Twice")
    refinement_enabled: bool = True
    refinement_ce_threshold: float = 3.0  # Run 2nd pass only if CE > this

    # ------------------------------------------------------------------ #
    # Pillar 4 — Rosetta Stone Observer                                   #
    # ------------------------------------------------------------------ #
    rosetta_d_probe: int = 512      # Probe hidden dim
    rosetta_n_layers: int = 4       # Attention layers in probe
    rosetta_n_heads: int = 8        # Attention heads (512 / 8 = 64 head_dim)
    rosetta_weight: float = 0.05    # Rosetta CE loss coefficient

    # ------------------------------------------------------------------ #
    # Bonus — EMA self-distillation                                       #
    # ------------------------------------------------------------------ #
    ema_decay: float = 0.999
    distill_weight: float = 0.1     # KL divergence from EMA teacher

    # ------------------------------------------------------------------ #
    # Bonus — Entropy regularisation                                      #
    # ------------------------------------------------------------------ #
    entropy_weight: float = 0.01    # Penalty to prevent overconfident logits

    # ------------------------------------------------------------------ #
    # Bonus — Persistent episodic memory                                  #
    # ------------------------------------------------------------------ #
    episodic_slots: int = 8_192     # Key/value memory slots
    episodic_topk: int = 8          # Number of memories retrieved per query

    # ------------------------------------------------------------------ #
    # Anti-underfitting / regularisation                                   #
    # ------------------------------------------------------------------ #
    residual_dropout: float = 0.1       # Dropout on tangent-space residuals (0.0 = off)
    gradient_noise_scale: float = 0.01  # Neelakantan et al. 2015 gradient noise
    val_fraction: float = 0.1           # Fraction of data held out for validation
    loss_spike_threshold: float = 3.0   # Skip optimiser step if CE > this × running avg
    tokenizer_path: str = "omega_tokenizer.json"

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    learning_rate: float = 4e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    micro_batch: int = 4
    grad_accum_steps: int = 16      # Effective batch = micro_batch * grad_accum = 64
    max_steps: int = 50_000
    warmup_steps: int = 3_000
    checkpoint_every: int = 2_000
    log_every: int = 50
    eval_every: int = 500           # Neurosymbolic verifier interval
    vram_log_every: int = 100       # VRAM usage logging interval (steps)
    curriculum_warmup: int = 2_000  # Steps before picky thresholds reach full strictness
    seed: int = 42

    # ------------------------------------------------------------------ #
    # Task 2 — Curriculum learning arc (pct-based)                        #
    # ------------------------------------------------------------------ #
    curriculum_enabled:   bool  = True
    curriculum_start_pct: float = 0.0    # Fraction of max_steps before arc activates
    curriculum_end_pct:   float = 0.15   # Fraction of max_steps at which full thresholds hit

    # ------------------------------------------------------------------ #
    # Task 3 — Multi-epoch data cycling                                   #
    # ------------------------------------------------------------------ #
    num_epochs:             int  = 1     # Informational; sampler cycles data endlessly
    shuffle_between_epochs: bool = True  # Reshuffle indices at each epoch wrap
    checkpoint_dir: str = "checkpoints_omega"
    data_path: str = "data/aether_train.jsonl"
    use_8bit_adam: bool = True
    use_gradient_checkpointing: bool = True   # Saves ~6 GiB; essential for OOM safety on 16 GB
    cpu_offload: bool = False                 # Model fits in 16 GB VRAM — no CPU transfer needed

    # ------------------------------------------------------------------ #
    # Derived properties (read-only helpers)                              #
    # ------------------------------------------------------------------ #
    @property
    def d_inner(self) -> int:
        """Inner SSM dimension (d_model * expand)."""
        return self.d_model * self.expand

    @property
    def effective_seq_len(self) -> int:
        """Actual sequence length after prepending thought tokens."""
        return self.max_seq_len + self.n_thought_tokens

    @property
    def ball_radius(self) -> float:
        """Poincaré ball radius = 1 / sqrt(c)."""
        return 1.0 / math.sqrt(self.hyp_curvature)

    def __post_init__(self) -> None:
        assert self.d_model % 2 == 0, "d_model must be even"
        assert self.rosetta_d_probe % self.rosetta_n_heads == 0, (
            "rosetta_d_probe must be divisible by rosetta_n_heads"
        )
        assert self.anticipatory_steps < self.max_seq_len, (
            "anticipatory_steps must be less than max_seq_len"
        )
        assert len(self.timescale_d_states) > 0, "timescale_d_states must be non-empty"
        assert 0.0 <= self.curriculum_start_pct < self.curriculum_end_pct <= 1.0, (
            "curriculum_start_pct must be < curriculum_end_pct, both in [0, 1]"
        )
        assert self.num_epochs >= 1, "num_epochs must be >= 1"
        assert self.n_moe_experts >= 1, "n_moe_experts must be >= 1"
        assert self.moe_layer_stride >= 1, "moe_layer_stride must be >= 1"
        assert self.ff_hidden % self.n_moe_experts == 0, (
            f"ff_hidden ({self.ff_hidden}) must be divisible by n_moe_experts "
            f"({self.n_moe_experts}) so GGR experts receive equal capacity"
        )

    def summary(self) -> str:
        lines = [
            "─" * 54,
            "  Aether Omega — Configuration",
            "─" * 54,
            f"  Architecture   : d_model={self.d_model}, n_layers={self.n_layers}",
            f"  SSM dims       : d_inner={self.d_inner}, d_state={self.d_state}",
            f"  Vocab / seq    : {self.vocab_size} / {self.max_seq_len}",
            f"  Thought tokens : {self.n_thought_tokens}",
            f"  Hyperbolic c   : {self.hyp_curvature}  (radius={self.ball_radius:.4f})",
            f"  v2 Riemannian  : {self.riemannian_correction}",
            f"  v2 Geom-gating : {self.geometry_gating}",
            f"  v2 Learn-κ     : {self.learnable_curvature}  "
            f"(init={self.curvature_init}, max={self.curvature_max}, warmup={self.curvature_warmup_steps})",
            f"  CSSC           : {'ON' if self.cssc_enabled else 'OFF'}  (Δ→curvature coupling, world-first)",
            f"  GGR Micro-MoE  : {'ON' if self.micro_moe_enabled else 'OFF'}  "
            f"({self.n_moe_experts} experts, stride={self.moe_layer_stride}, world-first)",
            f"  Picky CE range : [{self.picky_ce_min}, {self.picky_ce_max}]",
            f"  Anticipatory   : weight={self.anticipatory_weight}, steps={self.anticipatory_steps}",
            f"  Timescale SSM  : {self.timescale_d_states}",
            f"  Refinement     : {'ON' if self.refinement_enabled else 'OFF'} (CE>{self.refinement_ce_threshold})",
            f"  Neurogenesis   : patience={self.neurogenesis_patience}, delta={self.delta_dim}",
            f"  Episodic mem   : {self.episodic_slots} slots, top-k={self.episodic_topk}",
            f"  Grad ckpt      : {self.use_gradient_checkpointing}",
            f"  CPU offload    : {self.cpu_offload}",
            f"  Residual drop  : {self.residual_dropout}",
            f"  Grad noise     : {self.gradient_noise_scale}",
            f"  Val fraction   : {self.val_fraction}",
            f"  Spike thresh   : {self.loss_spike_threshold}×",
            f"  Tokenizer      : {self.tokenizer_path}",
            f"  Eff. batch     : {self.micro_batch * self.grad_accum_steps}",
            f"  Max steps      : {self.max_steps:,}",
            f"  Curriculum     : {'ON' if self.curriculum_enabled else 'OFF'} "
            f"  ({self.curriculum_start_pct:.0%}–{self.curriculum_end_pct:.0%} of steps)",
            f"  Epochs         : {self.num_epochs}  shuffle={self.shuffle_between_epochs}",
            "─" * 54,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = OmegaConfig()
    print(cfg.summary())
