"""Aether Omega — Unified Training Loop.

Integrates all five pillars plus bonus features:

  Pillar 1 — Mamba-Hyperbolic Engine   (model.py)
  Pillar 2 — Gradient-Based Neurogenesis  (NeurogenesisTracker below)
  Pillar 3 — Anticipatory Picky Learner   (PickyBatchSampler + anticipatory loss)
  Pillar 4 — Rosetta Stone Observer       (RosettaObserver + rosetta_loss)
  Bonus    — EMA self-distillation, entropy regularisation, episodic memory,
             continuous thought tokens, neurosymbolic AST verifier

Loss composition:
    L_total = CE_main
            + 0.1  · MSE(h[:, :-5, :], h[:, 5:, :])   (in-seq anticipatory)
            + 0.05 · CE_rosetta                          (Rosetta, detached)
            − 0.01 · H(logits)                           (entropy bonus)
            + 0.1  · KL(logits ‖ ema_logits)            (distillation)

Usage:
    python train.py
    python train.py --max-steps 5 --micro-batch 2 --no-8bit-adam   # smoke test
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from aether_config import OmegaConfig
from dataset import PickyBatchSampler, load_dataset, load_val_dataset, dataset_stats, log_source_coverage
from model import AetherOmegaModel, RosettaObserver, exp_map_zero, log_map_zero


# ──────────────────────────────────────────────────────────────────────────────
# ROCm detection
# ──────────────────────────────────────────────────────────────────────────────

IS_ROCM: bool = getattr(torch.version, "hip", None) is not None
DEVICE:  str  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP: bool = DEVICE == "cuda"

print(f"[init] device={DEVICE}  ROCm={IS_ROCM}  BF16={'yes' if USE_AMP else 'no'}")


def log_vram(tag: str = "") -> None:
    """Print current VRAM usage — helps track memory pressure before OOM."""
    if DEVICE != "cuda":
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    resrv = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[vram] {tag:>12s}  alloc={alloc:.2f} GiB  reserved={resrv:.2f} GiB  "
          f"total={total:.1f} GiB  headroom={total - resrv:.2f} GiB")


def get_curriculum_thresholds(step: int, cfg: OmegaConfig) -> tuple[float, float]:
    """Full curriculum arc: very relaxed at start → tighten linearly to targets.

    When curriculum_enabled=True (default):
        - Before curriculum_start_pct × max_steps : [0.0, 8.0] (keep everything)
        - Linear ramp up to curriculum_end_pct × max_steps: → [picky_ce_min, picky_ce_max]
        - After that: fixed at target thresholds

    Falls back to legacy curriculum_warmup when curriculum_enabled=False.
    """
    if not cfg.curriculum_enabled:
        # Legacy path: use curriculum_warmup
        if step >= cfg.curriculum_warmup:
            return cfg.picky_ce_min, cfg.picky_ce_max
        progress = step / max(cfg.curriculum_warmup, 1)
        ce_min = cfg.picky_ce_min * progress
        ce_max = 8.0 - (8.0 - cfg.picky_ce_max) * progress
        return ce_min, ce_max

    start_step = int(cfg.max_steps * cfg.curriculum_start_pct)
    end_step   = int(cfg.max_steps * cfg.curriculum_end_pct)

    if step < start_step:
        return 0.0, 8.0                          # pre-curriculum: accept everything
    if step >= end_step:
        return cfg.picky_ce_min, cfg.picky_ce_max  # post-curriculum: full strictness

    progress = (step - start_step) / max(end_step - start_step, 1)
    ce_min = cfg.picky_ce_min * progress
    ce_max = 8.0 - (8.0 - cfg.picky_ce_max) * progress
    return ce_min, ce_max


# ──────────────────────────────────────────────────────────────────────────────
# Pillar 2 — Neurogenesis Tracker
# ──────────────────────────────────────────────────────────────────────────────

class NeurogenesisTracker:
    """Monitors gradient variance on FFN layers and expands their hidden dimension
    when variance collapses below threshold for `patience` consecutive steps.

    Target: SwiGLU FFN in each AetherMambaBlock (w_gate, w_up, w_down).
    Expansion: grow the intermediate hidden dim by `delta_dim` neurons.
    This is safe because FFN input/output dims (d_model) stay fixed — only the
    internal hidden dim grows.  Residual connections remain compatible.

    After any expansion the optimizer is flagged for rebuild (handled in train loop).
    """

    def __init__(self, model: AetherOmegaModel, cfg: OmegaConfig) -> None:
        self.cfg   = cfg
        self.model = model
        self.events: list[dict] = []

        # Target: the SwiGLU FFN in each block.
        # GGR-MoE blocks have block.ffn=None — stored as None, skipped in step().
        self._ffns: list[nn.Module | None] = [block.ffn for block in model.blocks]

        # Per-block state
        self._var_history:    list[deque] = [deque(maxlen=cfg.neurogenesis_window)
                                              for _ in self._ffns]
        self._stagnant_steps: list[int]   = [0] * len(self._ffns)
        self.needs_optimizer_rebuild: bool = False

    def step(self, global_step: int) -> None:
        """Call once per training step, after loss.backward() and before optimizer.step()."""
        for i, ffn in enumerate(self._ffns):
            if ffn is None:
                continue                        # GGR-MoE block — neurogenesis not applicable
            # Monitor w_down gradient variance (final projection is most sensitive)
            if ffn.w_down.weight.grad is None:
                continue
            var = ffn.w_down.weight.grad.float().var().item()
            self._var_history[i].append(var)

            if len(self._var_history[i]) < self.cfg.neurogenesis_window:
                continue

            avg_var = sum(self._var_history[i]) / len(self._var_history[i])
            if avg_var < self.cfg.neurogenesis_var_threshold:
                self._stagnant_steps[i] += 1
            else:
                self._stagnant_steps[i] = 0

            if self._stagnant_steps[i] >= self.cfg.neurogenesis_patience:
                self._expand_ffn(i, global_step)
                self._stagnant_steps[i] = 0
                self._var_history[i].clear()

    def _expand_ffn(self, block_idx: int, global_step: int) -> None:
        """Expand SwiGLU hidden dim by delta_dim neurons (v2: SVD-guided init).

        Instead of random Kaiming init, uses the top singular vectors of the
        existing weight matrix to initialise new rows.  This preserves the
        current learned subspace while adding capacity along the most important
        directions.

        w_gate:  (H, D) → (H+d, D)     add rows — SVD-guided
        w_up:    (H, D) → (H+d, D)     add rows — SVD-guided
        w_down:  (D, H) → (D, H+d)     add columns (zero-init: output unchanged)
        """
        ffn = self._ffns[block_idx]
        d   = self.cfg.delta_dim
        dev   = ffn.w_gate.weight.device
        dtype = ffn.w_gate.weight.dtype

        for proj in [ffn.w_gate, ffn.w_up]:
            try:
                # Prefer gradient SVD (directions where model WANTS to learn)
                if proj.weight.grad is not None:
                    M = proj.weight.grad.float()
                else:
                    M = proj.weight.data.float()
                U, S, Vh = torch.linalg.svd(M, full_matrices=False)
                k = min(d, len(S))
                new_rows = Vh[:k] * (S[:k].unsqueeze(1) * 0.01)  # (k, D)
                if k < d:
                    pad = torch.empty(d - k, M.shape[1], device=dev, dtype=torch.float32)
                    nn.init.kaiming_uniform_(pad, a=math.sqrt(5))
                    new_rows = torch.cat([new_rows, pad], dim=0)
                # Small noise to break symmetry
                new_rows = new_rows + torch.randn_like(new_rows) * 0.001
            except Exception:
                # Fallback to Kaiming if SVD fails
                new_rows = torch.empty(d, proj.weight.shape[1], device=dev, dtype=torch.float32)
                nn.init.kaiming_uniform_(new_rows, a=math.sqrt(5))

            proj.weight = nn.Parameter(
                torch.cat([proj.weight.data.float(), new_rows], dim=0).to(dtype)
            )

        # Expand w_down: add d new columns (zero-init — initially a no-op)
        new_cols = torch.zeros(ffn.w_down.out_features, d, device=dev, dtype=dtype)
        ffn.w_down.weight = nn.Parameter(
            torch.cat([ffn.w_down.weight.data, new_cols], dim=1)
        )

        event = {
            "step":       global_step,
            "block_idx":  block_idx,
            "delta_dim":  d,
            "new_ffn_hidden": ffn.w_gate.weight.shape[0],
            "init_method": "svd",
        }
        self.events.append(event)
        self.needs_optimizer_rebuild = True
        print(f"[neurogenesis] step={global_step}  block={block_idx}  "
              f"ffn_hidden→{ffn.w_gate.weight.shape[0]}  (SVD-guided)")


# ──────────────────────────────────────────────────────────────────────────────
# Bonus — Neurosymbolic AST Verifier
# ──────────────────────────────────────────────────────────────────────────────

class SymbolicVerifier:
    """Real generation + decode + AST parse verifier.

    Every eval_every steps: decode 4 prompts from the dataset, generate 128
    new tokens via greedy decoding, decode the output to text with the real
    tokenizer, then try ast.parse().  Reports what fraction is valid Python.
    """

    def __init__(self) -> None:
        self.n_verified = 0
        self.n_passed   = 0
        self._tokenizer = None  # loaded lazily from cfg.tokenizer_path

    def _load_tokenizer(self, tokenizer_path: str):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from tokenizer import OmegaTokenizer
            self._tokenizer = OmegaTokenizer.from_file(tokenizer_path)
        except Exception as e:
            print(f"[verifier] Cannot load tokenizer ({e}) — text decode disabled")
        return self._tokenizer

    def verify_with_model(
        self,
        model: "AetherOmegaModel",
        dataset,
        cfg: "OmegaConfig",
        n_samples: int = 4,
    ) -> float:
        """Generate text, decode, AST-parse. Returns fraction that is valid Python."""
        tok = self._load_tokenizer(cfg.tokenizer_path)
        n_pass = 0
        for _ in range(n_samples):
            try:
                idx    = random.randint(0, len(dataset) - 1)
                sample = dataset[idx]
                prompt = sample["input_ids"][:64].unsqueeze(0).to(DEVICE)
                with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP
                ):
                    gen = _greedy_generate(model, prompt, max_new=128, cfg=cfg)
                gen_ids = gen[0].tolist()
                if tok is not None:
                    text = tok.decode(gen_ids, skip_special=True)
                else:
                    # Fallback: treat integer IDs as text (won't parse as Python,
                    # but at least gives a non-crashing result)
                    text = " ".join(map(str, gen_ids))
                self.n_verified += 1
                try:
                    ast.parse(text)
                    n_pass += 1
                    self.n_passed += 1
                except SyntaxError:
                    pass
            except Exception:
                pass
        return n_pass / max(n_samples, 1)

    def verify(self, text: str) -> float:
        """Legacy single-text verification (used by old eval path)."""
        self.n_verified += 1
        try:
            ast.parse(text)
            self.n_passed += 1
            return 1.0
        except SyntaxError:
            return 0.0

    @property
    def pass_rate(self) -> float:
        return self.n_passed / max(self.n_verified, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Learning rate schedule (cosine with linear warmup)
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: OmegaConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + (cfg.learning_rate - cfg.min_learning_rate) * cosine


def get_curvature_scale(step: int, cfg: OmegaConfig) -> float:
    """Curvature warmup: linear ramp 0→1 over curvature_warmup_steps."""
    if not cfg.learnable_curvature:
        return 1.0
    if step >= cfg.curvature_warmup_steps:
        return 1.0
    return step / max(cfg.curvature_warmup_steps, 1)


# ──────────────────────────────────────────────────────────────────────────────
# CPU Offload Optimizer — FP32 master weights in system RAM
# ──────────────────────────────────────────────────────────────────────────────

class CPUOffloadOptimizer:
    """Keeps FP32 master weights + AdamW states in CPU pinned RAM.

    Flow per step:
      1. Gradients accumulate on GPU in BF16 (normal backward pass).
      2. step() copies gradients GPU→CPU, casts to FP32.
      3. AdamW.step() runs on CPU with full FP32 precision.
      4. Updated FP32 weights are cast back to BF16 and copied to GPU.

    VRAM savings: ~1 GiB (optimizer states + FP32 master weights in system RAM).
    Trade-off: ~50ms extra per step for CPU↔GPU transfer (negligible vs fwd/bwd).

    Incompatible with GradScaler (not needed on ROCm with BF16 anyway).
    """

    def __init__(
        self,
        gpu_param_groups: list[dict],
        lr: float,
        betas: tuple[float, float],
    ) -> None:
        self._gpu_params: list[nn.Parameter] = []  # BF16 on GPU (model's actual params)
        self._cpu_params: list[nn.Parameter] = []  # FP32 on CPU (master copies)
        self._gpu_to_cpu: dict[int, int] = {}       # id(gpu_p) → index

        cpu_param_groups: list[dict] = []

        for group in gpu_param_groups:
            cpu_group: dict = {k: v for k, v in group.items() if k != "params"}
            cpu_group_params: list[nn.Parameter] = []

            for p in group["params"]:
                idx = len(self._gpu_params)
                # FP32 copy in pinned CPU memory
                cpu_p = nn.Parameter(
                    p.data.float().cpu().pin_memory(), requires_grad=True
                )
                self._gpu_params.append(p)
                self._cpu_params.append(cpu_p)
                self._gpu_to_cpu[id(p)] = idx
                cpu_group_params.append(cpu_p)

            cpu_group["params"] = cpu_group_params
            cpu_param_groups.append(cpu_group)

        self._inner = torch.optim.AdamW(cpu_param_groups, lr=lr, betas=betas)
        n_params = len(self._gpu_params)
        cpu_bytes = sum(p.numel() * 4 for p in self._cpu_params)
        print(f"[cpu-offload] {n_params} params, "
              f"FP32 master weights: {cpu_bytes / 1e9:.2f} GiB in pinned RAM")

    # ── Public interface matching torch.optim.Optimizer ──────────────────────

    @property
    def param_groups(self) -> list[dict]:
        return self._inner.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients on GPU parameters (CPU grads are overwritten in step)."""
        for p in self._gpu_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self) -> None:
        """Copy grads GPU→CPU, step AdamW on CPU, sync weights back to GPU."""
        # 1. Transfer gradients GPU → CPU (cast BF16 → FP32)
        for gpu_p, cpu_p in zip(self._gpu_params, self._cpu_params):
            if gpu_p.grad is not None:
                cpu_p.grad = gpu_p.grad.float().cpu()
            else:
                cpu_p.grad = None

        # 2. AdamW step on CPU (FP32 arithmetic — no precision loss)
        self._inner.step()

        # 3. Sync updated weights CPU → GPU (cast FP32 → BF16)
        with torch.no_grad():
            for gpu_p, cpu_p in zip(self._gpu_params, self._cpu_params):
                gpu_p.data.copy_(cpu_p.data.to(gpu_p.dtype), non_blocking=True)

    def state_dict(self) -> dict:
        return self._inner.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._inner.load_state_dict(state_dict)
        # After loading, sync CPU master weights → GPU
        with torch.no_grad():
            for gpu_p, cpu_p in zip(self._gpu_params, self._cpu_params):
                gpu_p.data.copy_(cpu_p.data.to(gpu_p.dtype))


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer factory
# ──────────────────────────────────────────────────────────────────────────────

def make_optimizer(
    model: AetherOmegaModel,
    rosetta: RosettaObserver,
    cfg: OmegaConfig,
    use_8bit: bool,
) -> torch.optim.Optimizer | CPUOffloadOptimizer:
    # v2: collect hyp_proj params → separate group with lr_scale=4.0
    hyp_proj_ids: set[int] = set()
    hyp_proj_params: list[nn.Parameter] = []
    for block in model.blocks:
        for p in block.hyp_proj.parameters():
            if p.requires_grad:
                hyp_proj_ids.add(id(p))
                hyp_proj_params.append(p)

    # v2: collect curvature params → no decay, no lr_scale
    curvature_ids: set[int] = set()
    curvature_params: list[nn.Parameter] = []
    if cfg.learnable_curvature:
        for block in model.blocks:
            if block._curvature_raw is not None and block._curvature_raw.requires_grad:
                curvature_ids.add(id(block._curvature_raw))
                curvature_params.append(block._curvature_raw)

    exclude_ids = hyp_proj_ids | curvature_ids

    # Standard decay / no-decay split (excluding hyp_proj and curvature)
    all_params = list(model.parameters()) + list(rosetta.parameters())
    decay_params    = [p for p in all_params
                       if p.requires_grad and p.dim() >= 2 and id(p) not in exclude_ids]
    no_decay_params = [p for p in all_params
                       if p.requires_grad and p.dim() < 2 and id(p) not in exclude_ids]

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    # v2: hyp_proj with lr_scale=4.0 to compensate RiemannianRescale 1/4 factor
    if hyp_proj_params:
        param_groups.append({"params": hyp_proj_params, "weight_decay": cfg.weight_decay, "lr_scale": 4.0})
    # v2: curvature params — no decay, no lr_scale
    if curvature_params:
        param_groups.append({"params": curvature_params, "weight_decay": 0.0})

    # CPU offload — FP32 master weights in system RAM, BF16 working copies on GPU
    if cfg.cpu_offload and DEVICE == "cuda":
        return CPUOffloadOptimizer(
            param_groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
        )

    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(
                param_groups,
                lr=cfg.learning_rate,
                betas=(cfg.beta1, cfg.beta2),
            )
            print("[optimizer] Using bitsandbytes AdamW8bit")
            return opt
        except Exception as e:
            print(f"[optimizer] bitsandbytes unavailable ({e}); falling back to AdamW")

    opt = torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )
    print("[optimizer] Using torch.optim.AdamW")
    return opt


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:   AetherOmegaModel,
    rosetta: RosettaObserver,
    ema:     AetherOmegaModel,
    opt:     torch.optim.Optimizer,
    step:    int,
    cfg:     OmegaConfig,
    state:   dict,
) -> None:
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prefix = ckpt_dir / f"step_{step:07d}"

    # Save weights in safetensors format.
    # .clone() breaks weight-tying (embedding.weight == lm_head.weight share storage)
    # so safetensors can save them as independent tensors without error.
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()},
              str(prefix) + "_main.safetensors")
    save_file({k: v.contiguous().clone() for k, v in rosetta.state_dict().items()},
              str(prefix) + "_rosetta.safetensors")
    save_file({k: v.contiguous().clone() for k, v in ema.state_dict().items()},
              str(prefix) + "_ema.safetensors")
    # Optimizer (standard torch format — may be large)
    torch.save(opt.state_dict(), str(prefix) + "_optimizer.pt")
    # Training state JSON
    with open(str(prefix) + "_state.json", "w") as f:
        json.dump(state, f, indent=2)
    print(f"[ckpt] Saved step {step} → {ckpt_dir}")


def load_checkpoint(
    model:   AetherOmegaModel,
    rosetta: RosettaObserver,
    ema:     AetherOmegaModel,
    opt:     torch.optim.Optimizer,
    prefix:  str,
) -> dict:
    # strict=False: tolerates optional fields added in future architecture revisions
    info = model.load_state_dict(load_file(prefix + "_main.safetensors"), strict=False)
    if info.missing_keys:
        print(f"[ckpt] model missing keys: {info.missing_keys}")
    if info.unexpected_keys:
        print(f"[ckpt] model unexpected keys: {info.unexpected_keys}")

    rosetta.load_state_dict(load_file(prefix + "_rosetta.safetensors"))

    ema_info = ema.load_state_dict(load_file(prefix + "_ema.safetensors"), strict=False)
    if ema_info.missing_keys:
        print(f"[ckpt] EMA missing keys: {ema_info.missing_keys}")

    try:
        opt.load_state_dict(torch.load(prefix + "_optimizer.pt", map_location=DEVICE,
                                       weights_only=True))
    except Exception as e:
        print(f"[ckpt] Optimizer state incompatible (expected after v2 upgrade): {e}")

    with open(prefix + "_state.json") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: OmegaConfig, resume_prefix: Optional[str] = None) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # ── SIGINT handler for emergency checkpoint ───────────────────────────────
    _emergency_state: dict = {"requested": False}

    def _sigint_handler(signum, frame):
        if _emergency_state["requested"]:
            print("\n[SIGINT] Second interrupt — aborting immediately.")
            sys.exit(1)
        _emergency_state["requested"] = True
        print("\n[SIGINT] Graceful shutdown requested. "
              "Will checkpoint after current step. Press Ctrl+C again to force quit.")

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── Build models ──────────────────────────────────────────────────────────
    dtype = torch.bfloat16 if USE_AMP else torch.float32

    model   = AetherOmegaModel(cfg).to(DEVICE, dtype=dtype)
    rosetta = RosettaObserver(cfg).to(DEVICE, dtype=dtype)

    # EMA teacher — same architecture, weights not tracked by optimizer
    ema = copy.deepcopy(model).to(DEVICE, dtype=dtype)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()  # Always run in eval mode — dropout must be off for a teacher model

    n_main    = model.count_parameters()
    n_rosetta = rosetta.count_parameters()
    print(f"[model] AetherOmega: {n_main / 1e6:.2f}M  Rosetta: {n_rosetta / 1e6:.2f}M  "
          f"Total: {(n_main + n_rosetta) / 1e6:.2f}M")
    log_vram("after model load")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset      = load_dataset(cfg)
    val_dataset  = load_val_dataset(cfg)
    dataset_stats(dataset,     "train")
    dataset_stats(val_dataset, "val")
    log_source_coverage(dataset, n_sample=min(len(dataset), 10_000), label="train")
    picky        = PickyBatchSampler(dataset, model, cfg, device=DEVICE)
    verifier     = SymbolicVerifier()
    neuro_tracker = NeurogenesisTracker(model, cfg)
    # Improvement 2: loss spike detection — rolling window of last 50 CE values
    _ce_history: deque = deque(maxlen=50)

    # ── Optimizer + scaler ────────────────────────────────────────────────────
    use_8bit  = cfg.use_8bit_adam and DEVICE == "cuda"
    optimizer = make_optimizer(model, rosetta, cfg, use_8bit)
    # GradScaler disabled on ROCm (BF16 doesn't need it) and with CPU offload
    use_scaler = USE_AMP and not IS_ROCM and not isinstance(optimizer, CPUOffloadOptimizer)
    scaler     = torch.amp.GradScaler(device="cuda", enabled=use_scaler)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if resume_prefix:
        state_loaded = load_checkpoint(model, rosetta, ema, optimizer, resume_prefix)
        start_step   = state_loaded.get("step", 0)
        print(f"[resume] Resumed from step {start_step}")

    # ── Training state trackers ───────────────────────────────────────────────
    running_ce      = 0.0
    running_ant     = 0.0
    running_rosetta = 0.0
    running_total   = 0.0
    t0              = time.time()
    K = cfg.n_thought_tokens  # thought token count (for label alignment)

    autocast_ctx = torch.amp.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP
    )

    # ── OOM fallback state ────────────────────────────────────────────────────
    oom_reduction_active = False  # True if micro_batch was halved due to OOM

    def _make_state(step_num: int) -> dict:
        return {
            "step":                  step_num,
            "ce_loss":               accum_ce,
            "neurogenesis_events":   neuro_tracker.events,
            "picky_stats":           picky.picky_stats(),
            "symbolic_pass_rate":    verifier.pass_rate,
            "config":                cfg.__dict__,
        }

    # ── Main loop ─────────────────────────────────────────────────────────────
    for step in range(start_step, cfg.max_steps):
        # ── Check SIGINT — save and exit gracefully ─────────────────────
        if _emergency_state["requested"]:
            print(f"[SIGINT] Emergency checkpoint at step {step}...")
            save_checkpoint(model, rosetta, ema, optimizer, step, cfg,
                            _make_state(step))
            print("[SIGINT] Checkpoint saved. Exiting.")
            sys.exit(0)

        # Update learning rate
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_scale", 1.0)

        # v2: curvature warmup scale — set on each block (plain float, no circular ref)
        curv_scale = get_curvature_scale(step, cfg)
        model._curvature_scale = curv_scale
        ema._curvature_scale = curv_scale
        for blk in model.blocks:
            blk._curvature_scale_val = curv_scale
        for blk in ema.blocks:
            blk._curvature_scale_val = curv_scale

        model.train()
        rosetta.train()
        optimizer.zero_grad()

        accum_ce      = 0.0
        accum_ant     = 0.0
        accum_rosetta = 0.0
        accum_total   = 0.0

        # ── Curriculum: adjust picky thresholds over training ──────────────
        cur_ce_min, cur_ce_max = get_curriculum_thresholds(step, cfg)
        picky.cfg.picky_ce_min = cur_ce_min
        picky.cfg.picky_ce_max = cur_ce_max

        # ── Gradient accumulation (with OOM fallback) ──────────────────────
        current_micro_batch = cfg.micro_batch
        step_ok = True

        for micro_step in range(cfg.grad_accum_steps):
            try:
                batch     = picky.get_batch(current_micro_batch)
                # Free picky sampler's eval activations before training forward
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                input_ids    = batch["input_ids"].to(DEVICE, non_blocking=True)
                labels       = batch["labels"].to(DEVICE, non_blocking=True)
                trust_scores = batch["trust_scores"].to(DEVICE, non_blocking=True)

                # Trust weight: [0.8, 1.8] range (trust_scores already in [0,1])
                trust_weight = 0.8 + (trust_scores.float() - 0.5) / 0.5 * 0.4  # (B,)

                with autocast_ctx:
                    # ── Forward pass (capture last layer hidden states) ───
                    logits, hs, feats = model(
                        input_ids,
                        capture_hidden_indices={cfg.n_layers - 1},
                    )

                    # ── Loss 1: Trust-weighted per-sample CE ──────────────
                    B, T, V = logits.shape
                    _per_tok = F.cross_entropy(
                        logits.reshape(B * T, V).float(),
                        labels.reshape(B * T),
                        ignore_index=-100,
                        reduction="none",
                    ).reshape(B, T)
                    _valid = (labels != -100).float()
                    _per_sample = (_per_tok * _valid).sum(1) / _valid.sum(1).clamp(min=1.0)
                    ce_loss = (_per_sample * trust_weight).mean()

                    # ── Think Twice: refine if CE is too high ─────────────
                    if cfg.refinement_enabled and ce_loss.item() > cfg.refinement_ce_threshold:
                        refined_embed = exp_map_zero(
                            model.refinement_proj(feats), cfg.hyp_curvature
                        )
                        logits, hs, feats = model(
                            embed_override=refined_embed,
                            capture_hidden_indices={cfg.n_layers - 1},
                        )
                        B, T, V = logits.shape
                        _per_tok = F.cross_entropy(
                            logits.reshape(B * T, V).float(),
                            labels.reshape(B * T),
                            ignore_index=-100,
                            reduction="none",
                        ).reshape(B, T)
                        _valid = (labels != -100).float()
                        _per_sample = (_per_tok * _valid).sum(1) / _valid.sum(1).clamp(min=1.0)
                        ce_loss = (_per_sample * trust_weight).mean()

                    # ── Loss 2: In-sequence anticipatory MSE (Pillar 3) ───
                    h = log_map_zero(hs[cfg.n_layers - 1], cfg.hyp_curvature)
                    S = cfg.anticipatory_steps
                    ant_loss = torch.tensor(0.0, device=DEVICE)
                    if T > S:
                        h_now    = h[:, :-S, :].float()
                        h_future = h[:, S:,  :].detach().float()
                        ant_loss = cfg.anticipatory_weight * F.mse_loss(h_now, h_future)

                    # ── Loss 3: Rosetta Observer CE (Pillar 4) ────────────
                    rosetta_logits = rosetta(h)
                    rosetta_loss   = cfg.rosetta_weight * F.cross_entropy(
                        rosetta_logits.reshape(B * T, V).float(),
                        labels.reshape(B * T),
                        ignore_index=-100,
                    )

                    # ── Loss 4: Entropy regularisation ────────────────────
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    probs     = log_probs.exp()
                    entropy   = -(probs * log_probs).sum(dim=-1).mean()
                    entropy_loss = -cfg.entropy_weight * entropy

                    # ── Loss 5: EMA self-distillation (KL) ────────────────
                    with torch.no_grad():
                        ema_logits, _, _ = ema(input_ids)
                        ema_probs = F.softmax(ema_logits.float(), dim=-1)
                        del ema_logits
                    # KL per token: sum over vocab dim, then mean over (B, T)
                    # F.kl_div batchmean divides by B only (not B×T), inflating the
                    # loss by T×V on sequence data.  Compute it manually instead.
                    per_tok_kl = (ema_probs * (
                        ema_probs.clamp(min=1e-30).log() - log_probs
                    )).sum(dim=-1)            # (B, T)
                    distill_loss = cfg.distill_weight * per_tok_kl.mean()
                    del ema_probs

                    total_loss = (ce_loss + ant_loss + rosetta_loss +
                                  entropy_loss + distill_loss)
                    total_loss = total_loss / cfg.grad_accum_steps

                # ── Backward ─────────────────────────────────────────────
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                accum_ce      += ce_loss.item()
                accum_ant     += ant_loss.item() if isinstance(ant_loss, torch.Tensor) else ant_loss
                accum_rosetta += rosetta_loss.item()
                accum_total   += total_loss.item()

            except torch.cuda.OutOfMemoryError:
                # ── OOM Fallback Cascade ─────────────────────────────────
                # Level 1: empty cache and retry
                torch.cuda.empty_cache()
                if not oom_reduction_active and current_micro_batch > 1:
                    current_micro_batch = max(1, current_micro_batch // 2)
                    oom_reduction_active = True
                    print(f"[OOM] Reduced micro_batch to {current_micro_batch} at step {step}")
                    log_vram("OOM recovery")
                    continue  # retry this micro_step with smaller batch
                else:
                    # Level 2: skip this step entirely
                    print(f"[OOM] Skipping step {step} — VRAM exhausted even at "
                          f"micro_batch={current_micro_batch}")
                    log_vram("OOM skip")
                    step_ok = False
                    break

        if not step_ok:
            optimizer.zero_grad()
            continue  # skip to next step

        # ── Improvement 2: loss spike detection ──────────────────────────
        if (len(_ce_history) >= 10
                and accum_ce > cfg.loss_spike_threshold * (sum(_ce_history) / len(_ce_history))):
            running_avg = sum(_ce_history) / len(_ce_history)
            print(f"[spike] step={step}  CE={accum_ce:.4f} > "
                  f"{cfg.loss_spike_threshold}×avg={running_avg:.4f} — skipping step")
            optimizer.zero_grad()
            continue
        _ce_history.append(accum_ce)

        # ── Neurogenesis check (Pillar 2) ─────────────────────────────────
        neuro_tracker.step(step)
        if neuro_tracker.needs_optimizer_rebuild:
            optimizer = make_optimizer(model, rosetta, cfg, use_8bit)
            neuro_tracker.needs_optimizer_rebuild = False
            print(f"[neurogenesis] Optimizer rebuilt at step {step}")

        # ── Gradient clip + optimizer step ───────────────────────────────
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(rosetta.parameters()),
            cfg.max_grad_norm,
        )

        # ── Improvement 1: gradient noise injection (Neelakantan et al.) ──
        if cfg.gradient_noise_scale > 0:
            noise_std = cfg.gradient_noise_scale / math.sqrt(1.0 + step)
            all_params = list(model.parameters()) + list(rosetta.parameters())
            for p in all_params:
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * noise_std)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # ── EMA update (no grad) ─────────────────────────────────────────
        with torch.no_grad():
            for ema_p, m_p in zip(ema.parameters(), model.parameters()):
                ema_p.data.mul_(cfg.ema_decay).add_(m_p.data, alpha=1.0 - cfg.ema_decay)

        # ── Accumulate running stats ──────────────────────────────────────
        running_ce      += accum_ce
        running_ant     += accum_ant
        running_rosetta += accum_rosetta
        running_total   += accum_total

        # ── Logging ───────────────────────────────────────────────────────
        if (step + 1) % cfg.log_every == 0 or step == 0:
            elapsed = time.time() - t0
            steps_done = step + 1 - start_step
            ms_per_step = elapsed / steps_done * 1000

            window_size = max(1, min(cfg.log_every, step + 1 - start_step))
            avg_ce      = running_ce      / (window_size * cfg.grad_accum_steps)
            avg_ant     = running_ant     / (window_size * cfg.grad_accum_steps)
            avg_rosetta = running_rosetta / (window_size * cfg.grad_accum_steps)
            avg_total   = running_total   / window_size
            running_ce = running_ant = running_rosetta = running_total = 0.0

            stats = picky.picky_stats()
            oom_note  = "  [!OOM-reduced]" if oom_reduction_active else ""
            # Improvement 5: perplexity (clamp CE to avoid overflow)
            perplexity = math.exp(min(avg_ce, 20.0))
            print(
                f"[{step + 1:>6}/{cfg.max_steps}]  "
                f"lr={lr:.2e}  "
                f"ce={avg_ce:.4f}  ppl={perplexity:.1f}  "
                f"ant={avg_ant:.4f}  ros={avg_rosetta:.4f}  "
                f"total={avg_total:.4f}  kept={stats['kept_pct']:.0f}%  "
                f"{ms_per_step:.0f}ms/step{oom_note}"
            )
            if perplexity > 5000 and step + 1 > 500:
                print(f"[warning] ppl={perplexity:.0f} > 5000 after 500 steps "
                      f"— check init / data / lr")

            # v2: geometry stats
            if cfg.learnable_curvature:
                curvatures = [b.curvature.item() if isinstance(b.curvature, torch.Tensor)
                              else b.curvature for b in model.blocks]
                print(f"  κ: min={min(curvatures):.4f} max={max(curvatures):.4f} "
                      f"mean={sum(curvatures)/len(curvatures):.4f}  "
                      f"warmup={curv_scale:.3f}")
            if cfg.geometry_gating:
                gates_ssm = [torch.sigmoid(b.geom_gate_ssm.bias).item()
                             for b in model.blocks if hasattr(b, 'geom_gate_ssm')]
                gates_ffn = [torch.sigmoid(b.geom_gate_ffn.bias).item()
                             for b in model.blocks if hasattr(b, 'geom_gate_ffn')]
                if gates_ssm:
                    print(f"  gate_ssm: min={min(gates_ssm):.3f} max={max(gates_ssm):.3f} "
                          f"mean={sum(gates_ssm)/len(gates_ssm):.3f}  "
                          f"gate_ffn: min={min(gates_ffn):.3f} max={max(gates_ffn):.3f}")
            # Hyp vs Euc grad norm comparison
            hyp_gnorm = sum(p.grad.float().norm().item()
                            for b in model.blocks for p in b.hyp_proj.parameters()
                            if p.grad is not None) / max(cfg.n_layers, 1)
            euc_gnorm = sum(
                p.grad.float().norm().item()
                for b in model.blocks if b.ffn is not None
                for p in b.ffn.parameters() if p.grad is not None
            ) / max(cfg.n_layers * 3, 1)
            print(f"  grad_norm: hyp_proj={hyp_gnorm:.4f}  ffn={euc_gnorm:.4f}")

        # ── VRAM monitoring ────────────────────────────────────────────
        if (step + 1) % cfg.vram_log_every == 0:
            log_vram(f"step {step + 1}")

        # ── Improvement 3: validation loss + Improvement 4: real AST eval ──
        if (step + 1) % cfg.eval_every == 0:
            _run_eval(model, rosetta, val_dataset, verifier, cfg, step + 1)

        # ── Checkpoint ────────────────────────────────────────────────────
        if (step + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(model, rosetta, ema, optimizer, step + 1, cfg,
                            _make_state(step + 1))

    print("\n[train] Training complete.")
    save_checkpoint(model, rosetta, ema, optimizer, step + 1, cfg,
                    _make_state(step + 1))


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation pass — val loss + AST syntax verifier (Improvement 3 + 4)
# ──────────────────────────────────────────────────────────────────────────────

def _run_eval(
    model:       AetherOmegaModel,
    rosetta:     RosettaObserver,
    val_dataset,
    verifier:    SymbolicVerifier,
    cfg:         OmegaConfig,
    step:        int,
    max_val_batches: int = 32,
) -> None:
    """Compute validation loss and run real AST syntax verification.

    Improvement 3: val CE + ppl, comparison to train loss.
    Improvement 4: decode 4 generated sequences, AST parse.
    """
    model.eval()
    rosetta.eval()
    val_ce_total = 0.0
    val_n_batches = 0

    try:
        n_val = len(val_dataset)
        indices = list(range(n_val))
        random.shuffle(indices)
        batch_ids_list: list[int] = []
        batch_lbl_list: list[int] = []

        for idx in indices[:max_val_batches * cfg.micro_batch]:
            item = val_dataset[idx]
            batch_ids_list.append(item["input_ids"].unsqueeze(0))
            batch_lbl_list.append(item["labels"].unsqueeze(0))
            if len(batch_ids_list) == cfg.micro_batch:
                b_ids = torch.cat(batch_ids_list).to(DEVICE)
                b_lbl = torch.cat(batch_lbl_list).to(DEVICE)
                with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP
                ):
                    logits, _, _ = model(b_ids)
                    B, T, V = logits.shape
                    val_ce = F.cross_entropy(
                        logits.reshape(B * T, V).float(),
                        b_lbl.reshape(B * T),
                        ignore_index=-100,
                    )
                val_ce_total += val_ce.item()
                val_n_batches += 1
                batch_ids_list.clear()
                batch_lbl_list.clear()

        if val_n_batches > 0:
            avg_val_ce = val_ce_total / val_n_batches
            val_ppl = math.exp(min(avg_val_ce, 20.0))
            print(f"[eval]  step={step}  val_ce={avg_val_ce:.4f}  val_ppl={val_ppl:.1f}")

        # Improvement 4: real AST verification (generate + decode + parse)
        ast_pass_rate = verifier.verify_with_model(model, val_dataset, cfg, n_samples=4)
        print(f"[eval]  step={step}  ast_pass_rate={ast_pass_rate:.2%}  "
              f"cumulative_pass={verifier.pass_rate:.2%}  "
              f"n_verified={verifier.n_verified}")

    except Exception as e:
        print(f"[eval] Error during eval: {e}")
    finally:
        model.train()
        rosetta.train()


@torch.no_grad()
def _greedy_generate(
    model:    AetherOmegaModel,
    input_ids: torch.Tensor,
    max_new:  int,
    cfg:      OmegaConfig,
) -> torch.Tensor:
    """Simple greedy decoding — used only for symbolic verification."""
    ids = input_ids
    for _ in range(max_new):
        # Trim to max_seq_len
        ids_trunc = ids[:, -cfg.max_seq_len:]
        logits, _, _ = model(ids_trunc)
        next_tok  = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
    return ids[:, input_ids.size(1):]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aether Omega Training")
    p.add_argument("--max-steps",      type=int,   default=None)
    p.add_argument("--micro-batch",    type=int,   default=None)
    p.add_argument("--grad-accum",     type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--checkpoint-dir", type=str,   default=None)
    p.add_argument("--data-path",      type=str,   default=None)
    p.add_argument("--resume",         type=str,   default=None,
                   help="Path prefix to resume from (without extension)")
    p.add_argument("--no-8bit-adam",   action="store_true",
                   help="Force standard AdamW even if bitsandbytes is available")
    p.add_argument("--seed",           type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = OmegaConfig()

    # CLI overrides
    if args.max_steps      is not None: cfg.max_steps      = args.max_steps
    if args.micro_batch    is not None: cfg.micro_batch    = args.micro_batch
    if args.grad_accum     is not None: cfg.grad_accum_steps = args.grad_accum
    if args.lr             is not None: cfg.learning_rate  = args.lr
    if args.checkpoint_dir is not None: cfg.checkpoint_dir = args.checkpoint_dir
    if args.data_path      is not None: cfg.data_path      = args.data_path
    if args.no_8bit_adam:               cfg.use_8bit_adam  = False
    if args.seed           is not None: cfg.seed           = args.seed

    print(cfg.summary())
    train(cfg, resume_prefix=args.resume)
