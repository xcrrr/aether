"""Aether 2 — Training Loop.

Features:
  • BF16 mixed-precision (ROCm-compatible)
  • Gradient checkpointing (saves ~6 GiB VRAM)
  • Streaming JSONL dataset with circular buffer
  • CSSC + GGR ablation via --no-cssc / --no-ggr flags
  • Baseline shadow model with EMA for real-time delta metrics
  • RosettaObserver: ~25M secondary decoder — latent-space interpretability probe
  • Picky Learner: curriculum-warmed CE-range batch filtering
  • Aether Command Deck dashboard (--dashboard flag)
  • Detailed logging to logs/aether_build.log
  • Safetensors checkpointing

Usage
-----
  python aether2_train.py                         # full training
  python aether2_train.py --no-cssc --no-ggr      # vanilla ablation
  python aether2_train.py --dashboard             # with rich UI
  python aether2_train.py --max-steps 5 --micro-batch 1 --grad-accum 1  # dry run
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as st_load, save_file as st_save

from aether2_config import Aether2Config
from aether2_model import Aether2Model, ShadowModel
from fluid_power import FluidPowerAllocator
from model import RosettaObserver, log_map_zero
from streaming_data import StreamingJSONLDataset

# ─────────────────────────────────────────────────────────────────────────────
# CPU optimizer offload helpers
# ─────────────────────────────────────────────────────────────────────────────

def _offload_optimizer_to_cpu(opt: torch.optim.Optimizer) -> None:
    """Move Adam m/v states from GPU to CPU RAM between optimizer steps.

    Saves ~2× model_params GiB on GPU VRAM. State is restored to GPU before
    each update and offloaded again immediately after, so GPU only holds the
    state during the ~200 ms optimizer step instead of throughout training.
    """
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                state[k] = v.cpu()


def _load_optimizer_to_gpu(opt: torch.optim.Optimizer, device: str) -> None:
    """Restore Adam m/v states from CPU RAM to GPU before the optimizer step."""
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and not v.is_cuda:
                state[k] = v.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: str) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aether2")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Training Metrics Dataclass (shared with dashboard)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingMetrics:
    step: int = 0
    epoch: int = 0
    lr: float = 0.0
    ce_loss: float = float("nan")
    ppl: float = float("nan")
    ggr_aux_loss: float = 0.0
    total_loss: float = float("nan")
    vram_alloc_gib: float = 0.0
    vram_total_gib: float = 0.0
    kept_pct: float = 100.0
    ms_per_step: float = 0.0
    tokens_per_sec: float = 0.0
    grad_norm: float = 0.0
    curvature_mean: float = 1.0
    gate_ssm_mean: float = 0.5
    gate_cssc_mean: float = 0.5
    # GGR per-expert load fractions (length = n_experts)
    ggr_expert_load: list[float] = field(default_factory=lambda: [0.25] * 4)
    ggr_gsi: float = 1.0               # Gradient Stability Index
    cssc_ce: float = 0.5               # Context Efficiency
    # Baseline comparison
    baseline_loss: float = float("nan")
    delta_pct: float = 0.0             # (baseline - aether2) / baseline × 100
    # Fluid Power Allocation (FPA) — only populated when cfg.fpa_enabled
    ponder_cost: float = 0.0
    fpa_avg_iters: float = 1.0         # mean passes per token
    fpa_halt_pct: float = 0.0          # fraction of tokens that early-exited
    # RosettaObserver probe
    rosetta_ce: float = 0.0            # Rosetta cross-entropy (0 if disabled)
    # Picky Learner
    picky_skipped: int = 0             # micro-batches skipped this log interval


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def make_optimizer(
    model: Aether2Model,
    cfg: Aether2Config,
    logger: logging.Logger,
    allocator: "FluidPowerAllocator | None" = None,
    rosetta: "RosettaObserver | None" = None,
) -> torch.optim.Optimizer:
    """AdamW with separate param groups for geometry-sensitive params."""
    decay_params, no_decay_params, geo_params = [], [], []
    geo_keywords = ("hyp_proj", "w_cssc", "curv_gate", "log_scale_weights",
                    "curvature", "geom_gate", "grad_gate")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_geo = any(kw in name for kw in geo_keywords)
        is_no_decay = (p.ndim == 1) or ("norm" in name) or ("bias" in name)
        if is_geo:
            geo_params.append(p)
        elif is_no_decay:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    # FPA EntropyHaltingCriterion.raw_thresh — small 1-D tensors, no weight decay
    if allocator is not None:
        for p in allocator.parameters():
            if p.requires_grad:
                no_decay_params.append(p)

    # RosettaObserver — standard decay/no-decay split, separate from main model
    if rosetta is not None:
        for name, p in rosetta.named_parameters():
            if not p.requires_grad:
                continue
            if (p.ndim == 1) or ("norm" in name) or ("bias" in name):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

    logger.info(
        f"Optimizer groups — decay: {len(decay_params)}, "
        f"no_decay: {len(no_decay_params)}, geo: {len(geo_params)}"
    )

    # lr_scale is stored in each group so set_lr() can reapply the ratio
    # on every step without losing the 4× advantage for geometry params.
    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay, "lr": cfg.learning_rate,       "lr_scale": 1.0},
        {"params": no_decay_params, "weight_decay": 0.0,              "lr": cfg.learning_rate,       "lr_scale": 1.0},
        {"params": geo_params,      "weight_decay": 0.0,              "lr": cfg.learning_rate * 4.0, "lr_scale": 4.0},
    ]

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(param_groups, betas=(cfg.beta1, cfg.beta2))
        logger.info("Using bitsandbytes 8-bit AdamW")
    except ImportError:
        opt = torch.optim.AdamW(param_groups, betas=(cfg.beta1, cfg.beta2))
        logger.info("Using standard AdamW (bitsandbytes not available)")

    return opt


def cosine_lr(step: int, cfg: Aether2Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + (cfg.learning_rate - cfg.min_learning_rate) * cosine


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)


def picky_thresholds(step: int, cfg: Aether2Config) -> tuple[float, float]:
    """Return curriculum-adjusted (ce_min, ce_max) for the Picky Learner.

    During warmup, thresholds are relaxed so that high-CE early-training
    batches are not all rejected.  After curriculum_warmup steps, the
    full [picky_ce_min, picky_ce_max] window is enforced.
    """
    if not cfg.curriculum_enabled:
        return cfg.picky_ce_min, cfg.picky_ce_max
    t = min(1.0, step / max(cfg.curriculum_warmup, 1))
    ce_min = cfg.picky_ce_min * t
    ce_max = cfg.picky_ce_max + (20.0 - cfg.picky_ce_max) * (1.0 - t)
    return ce_min, ce_max


# ─────────────────────────────────────────────────────────────────────────────
# VRAM utilities
# ─────────────────────────────────────────────────────────────────────────────

def vram_stats(device: str) -> tuple[float, float]:
    """Return (allocated_GiB, total_GiB) for the given device."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return alloc, total


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    step: int,
    model: Aether2Model,
    shadow: ShadowModel,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: Aether2Config,
    logger: logging.Logger,
    allocator: "FluidPowerAllocator | None" = None,
    rosetta: "RosettaObserver | None" = None,
) -> None:
    ckpt_dir = Path(cfg.checkpoint_dir) / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Model weights
    st_save(
        {k: v.contiguous().cpu() for k, v in model.state_dict().items()},
        str(ckpt_dir / "model.safetensors"),
    )
    # Shadow weights
    st_save(
        {k: v.contiguous().cpu() for k, v in shadow.state_dict().items()},
        str(ckpt_dir / "shadow.safetensors"),
    )
    # FPA allocator weights (raw_thresh scalars)
    if allocator is not None:
        st_save(
            {k: v.contiguous().cpu() for k, v in allocator.state_dict().items()},
            str(ckpt_dir / "allocator.safetensors"),
        )
    # RosettaObserver weights
    if rosetta is not None:
        st_save(
            {k: v.contiguous().cpu() for k, v in rosetta.state_dict().items()},
            str(ckpt_dir / "rosetta.safetensors"),
        )
    # Optimizer state
    torch.save(opt.state_dict(), ckpt_dir / "optimizer.pt")
    # Scaler state
    torch.save(scaler.state_dict(), ckpt_dir / "scaler.pt")
    # Config
    with open(ckpt_dir / "config.json", "w") as f:
        import dataclasses
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    logger.info(f"Checkpoint saved: {ckpt_dir}")


def load_checkpoint(
    prefix: str,
    model: Aether2Model,
    shadow: ShadowModel,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    logger: logging.Logger,
    allocator: "FluidPowerAllocator | None" = None,
    rosetta: "RosettaObserver | None" = None,
) -> int:
    """Load from checkpoint prefix. Returns resume step."""
    ckpt_dir = Path(prefix)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {prefix}")

    model_w = st_load(str(ckpt_dir / "model.safetensors"))
    model.load_state_dict(model_w, strict=False)
    logger.info(f"Loaded model weights from {ckpt_dir}")

    shadow_f = ckpt_dir / "shadow.safetensors"
    if shadow_f.exists():
        shadow_w = st_load(str(shadow_f))
        shadow.load_state_dict(shadow_w, strict=False)

    alloc_f = ckpt_dir / "allocator.safetensors"
    if allocator is not None and alloc_f.exists():
        alloc_w = st_load(str(alloc_f))
        allocator.load_state_dict(alloc_w, strict=False)
        logger.info("Loaded FPA allocator weights")

    rosetta_f = ckpt_dir / "rosetta.safetensors"
    if rosetta is not None and rosetta_f.exists():
        rosetta_w = st_load(str(rosetta_f))
        rosetta.load_state_dict(rosetta_w, strict=False)
        logger.info("Loaded RosettaObserver weights")

    opt_f = ckpt_dir / "optimizer.pt"
    if opt_f.exists():
        opt.load_state_dict(torch.load(opt_f, map_location="cpu"))

    scaler_f = ckpt_dir / "scaler.pt"
    if scaler_f.exists():
        scaler.load_state_dict(torch.load(scaler_f))

    # Extract step from directory name
    step = int(ckpt_dir.name.replace("step_", ""))
    logger.info(f"Resuming from step {step}")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg: Aether2Config,
    resume_prefix: str | None = None,
    metrics_callback: Callable[[TrainingMetrics], None] | None = None,
    suppress_stdout: bool = False,
) -> None:
    logger = setup_logging(cfg.log_file)

    # When running with --dashboard, suppress stdout so Rich screen mode
    # isn't corrupted by log lines written directly to the terminal.
    if suppress_stdout:
        for h in list(logger.handlers):
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout:
                logger.removeHandler(h)

    logger.info("=" * 60)
    logger.info("Aether 2 — Training Start")
    logger.info("=" * 60)
    logger.info(cfg.summary())

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = cfg.device
    dtype  = torch.bfloat16 if cfg.use_bf16 else torch.float32

    # ── Models ──────────────────────────────────────────────────────────────
    logger.info("Building Aether 2 model…")
    # BF16 weights: halves the ~2.7 GiB FP32 footprint to ~1.35 GiB.
    # All Poincaré ops already cast to float32 internally, so BF16 params
    # are safe. Gradient accumulation and optimizer states are still FP32.
    model  = Aether2Model(cfg).to(device=device, dtype=torch.bfloat16)
    shadow = ShadowModel(cfg).to(device=device, dtype=torch.bfloat16)
    model.print_summary()
    total_p = model.count_parameters()
    logger.info(f"Total parameters: {total_p / 1e6:.1f}M")
    if cfg.cpu_offload_optimizer:
        logger.info("CPU optimizer offload: ON — Adam m/v states will live in CPU RAM")

    # ── RosettaObserver (latent-space interpretability probe) ─────────────────
    rosetta: RosettaObserver | None = None
    if cfg.rosetta_enabled:
        rosetta = RosettaObserver(cfg).to(device=device, dtype=torch.bfloat16)
        r_params = sum(p.numel() for p in rosetta.parameters())
        logger.info(
            f"RosettaObserver: ON  "
            f"({r_params / 1e6:.1f}M params, weight={cfg.rosetta_weight})"
        )

    # ── Fluid Power Allocator (optional) ─────────────────────────────────────
    allocator: FluidPowerAllocator | None = None
    if cfg.fpa_enabled:
        allocator = FluidPowerAllocator(cfg).to(device=device, dtype=torch.bfloat16)
        fpa_params = sum(p.numel() for p in allocator.parameters())
        logger.info(
            f"Fluid Power Allocation: ON  "
            f"(max_iters={cfg.fpa_max_iters}, ponder_w={cfg.fpa_ponder_weight}, "
            f"extra_params={fpa_params})"
        )

    # ── Optimizer & scaler ───────────────────────────────────────────────────
    opt = make_optimizer(model, cfg, logger, allocator=allocator, rosetta=rosetta)
    # BF16 has the same dynamic range as FP32 — loss scaling is not needed.
    # GradScaler is disabled; scaler.scale/unscale/step/update are all no-ops.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    if resume_prefix:
        start_step = load_checkpoint(
            resume_prefix, model, shadow, opt, scaler, logger,
            allocator=allocator, rosetta=rosetta,
        )

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"Starting streaming dataset from: {cfg.data_path}")
    dataset = StreamingJSONLDataset(cfg)
    dataset.start()
    logger.info("Waiting for dataset buffer to fill…")
    if not dataset.wait_for_fill(0.005, timeout=120.0):
        logger.warning("Dataset buffer fill timeout — proceeding with partial fill")

    # ── Baseline (shadow) model loss tracking ────────────────────────────────
    baseline_loss_ema = float("nan")
    aether2_loss_ema  = float("nan")
    ema_alpha = 0.02       # EMA smoothing for display

    # ── Gradient accumulation state ──────────────────────────────────────────
    opt.zero_grad(set_to_none=True)
    accum_ce      = 0.0
    accum_aux     = 0.0
    accum_ponder  = 0.0
    accum_rosetta = 0.0
    accum_skipped = 0
    accum_count   = 0
    grad_norms   = collections.deque(maxlen=50)
    step_times   = collections.deque(maxlen=20)

    # ── Signal handler for graceful exit ─────────────────────────────────────
    _shutdown = {"flag": False}
    def _handler(sig, frame):
        logger.info("Signal received — saving checkpoint and exiting…")
        _shutdown["flag"] = True
    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    logger.info(f"Training from step {start_step} to {cfg.max_steps}")

    t_step_start = time.perf_counter()
    micro_step   = 0

    for step in range(start_step, cfg.max_steps):
        if _shutdown["flag"]:
            break

        # LR schedule
        lr = cosine_lr(step, cfg)
        set_lr(opt, lr)

        # ── Gradient accumulation ─────────────────────────────────────────
        for _ in range(cfg.grad_accum_steps):
            batch = dataset.get_batch_blocking(cfg.micro_batch, device=device)
            input_ids    = batch["input_ids"]    # (B, T)
            labels       = batch["labels"]       # (B, T)
            trust_scores = batch["trust_scores"] # (B,)
            T_l = labels.shape[1]

            with torch.autocast("cuda", dtype=dtype, enabled=cfg.use_bf16):
                if allocator is not None:
                    # FPA path: adaptive compute with entropy-conditioned re-routing
                    logits_aligned, ponder_cost, aux_loss = allocator(model, input_ids)
                    hidden_states_cap: dict = {}
                else:
                    # Capture last hidden state for RosettaObserver if enabled
                    capture_idx = {cfg.n_layers - 1} if rosetta is not None else set()
                    logits, hidden_states_cap, aux_loss = model(
                        input_ids, capture_hidden_indices=capture_idx
                    )
                    logits_aligned = logits[:, -T_l:, :]
                    ponder_cost = torch.tensor(0.0, device=device)

                # CE loss (trust-weighted)
                ce = F.cross_entropy(
                    logits_aligned.reshape(-1, cfg.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(labels.shape)          # (B, T)

                # Normalise trust scores [0,100] → [0,1] for soft sample weighting
                trust_w = (trust_scores / 100.0).clamp(0.0, 1.0)
                trust_w = trust_w.unsqueeze(1).expand_as(ce)
                ce_loss = (ce * trust_w).sum() / (trust_w.sum() + 1e-8)

                # ── RosettaObserver loss ──────────────────────────────────
                # hidden_states_cap[n_layers-1] is in Poincaré ball (detached).
                # Convert to tangent space before passing to Rosetta.
                # Stop-gradient is also enforced inside RosettaObserver.forward().
                if rosetta is not None and (cfg.n_layers - 1) in hidden_states_cap:
                    h_ball   = hidden_states_cap[cfg.n_layers - 1]
                    h_tan    = log_map_zero(h_ball, cfg.hyp_curvature)
                    r_logits = rosetta(h_tan)[:, -T_l:, :]
                    rosetta_ce_loss = F.cross_entropy(
                        r_logits.reshape(-1, cfg.vocab_size),
                        labels.reshape(-1), ignore_index=-100,
                    )
                else:
                    rosetta_ce_loss = torch.tensor(0.0, device=device)

                ggr_aux    = aux_loss * cfg.ggr_lb_weight
                ponder_reg = ponder_cost * cfg.fpa_ponder_weight
                total_loss = (ce_loss + ggr_aux + ponder_reg
                              + rosetta_ce_loss * cfg.rosetta_weight)

                # Gradient accumulation scale
                total_loss = total_loss / cfg.grad_accum_steps

            # ── Picky Learner: skip batch if CE outside curriculum window ──
            ce_lo, ce_hi = picky_thresholds(step, cfg)
            if ce_loss.item() < ce_lo or ce_loss.item() > ce_hi:
                accum_skipped += 1
                continue

            scaler.scale(total_loss).backward()
            accum_ce      += ce_loss.item()
            accum_aux     += ggr_aux.item()
            accum_ponder  += ponder_cost.item()
            accum_rosetta += rosetta_ce_loss.item()
            accum_count   += 1

        # ── All micro-batches skipped by Picky Learner ────────────────────
        if accum_count == 0:
            logger.debug(
                f"Step {step}: all {cfg.grad_accum_steps} micro-batches skipped "
                f"by Picky Learner (CE range [{ce_lo:.2f}, {ce_hi:.2f}]) — no update"
            )
            opt.zero_grad(set_to_none=True)
            accum_skipped = 0
            continue

        # ── Optimizer step ────────────────────────────────────────────────
        # Restore Adam m/v from CPU RAM to GPU just before the update.
        if cfg.cpu_offload_optimizer:
            _load_optimizer_to_gpu(opt, device)

        scaler.unscale_(opt)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        grad_norms.append(grad_norm.item())

        # Loss spike detection
        mean_ce = accum_ce / max(accum_count, 1)
        if not math.isnan(aether2_loss_ema) and mean_ce > aether2_loss_ema * cfg.loss_spike_threshold:
            logger.warning(f"Step {step}: Loss spike {mean_ce:.3f} > "
                           f"{aether2_loss_ema * cfg.loss_spike_threshold:.3f} — skipping step")
            opt.zero_grad(set_to_none=True)
            accum_ce = accum_aux = accum_ponder = accum_rosetta = accum_count = accum_skipped = 0
            if cfg.cpu_offload_optimizer:
                _offload_optimizer_to_cpu(opt)
            continue

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        # Offload Adam m/v back to CPU RAM when CPU offload is active.
        # empty_cache() is only needed here to release the optimizer state tensors
        # that were temporarily pinned on GPU for the update round-trip.
        # When cpu_offload_optimizer is False the Adam states live on GPU full-time
        # and there is nothing to release — calling empty_cache() every step would
        # force a costly ROCm memory defrag sync for no benefit.
        if cfg.cpu_offload_optimizer:
            _offload_optimizer_to_cpu(opt)
            torch.cuda.empty_cache()

        # EMA loss tracking
        if math.isnan(aether2_loss_ema):
            aether2_loss_ema = mean_ce
        else:
            aether2_loss_ema = (1 - ema_alpha) * aether2_loss_ema + ema_alpha * mean_ce

        # ── Shadow model EMA update ───────────────────────────────────────
        if cfg.baseline_enabled:
            shadow.ema_update(model.blocks, cfg.baseline_ema_decay)

        # ── Compute step timing ───────────────────────────────────────────
        t_now   = time.perf_counter()
        ms_step = (t_now - t_step_start) * 1000
        step_times.append(ms_step)
        t_step_start = t_now

        B  = cfg.micro_batch * cfg.grad_accum_steps
        T  = cfg.max_seq_len
        tok_per_sec = B * T / (ms_step / 1000) if ms_step > 0 else 0

        # ── Collect metrics ───────────────────────────────────────────────
        if step % cfg.log_every == 0:
            vram_alloc, vram_total = vram_stats(device)

            # Baseline forward on last batch (shadow model, no grad)
            b_loss_val = float("nan")
            if cfg.baseline_enabled:
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=cfg.use_bf16):
                    s_logits = shadow(input_ids)
                    s_logits_aligned = s_logits[:, -T_l:, :]
                    b_loss_val = F.cross_entropy(
                        s_logits_aligned.reshape(-1, cfg.vocab_size),
                        labels.reshape(-1),
                        ignore_index=-100,
                    ).item()
                if math.isnan(baseline_loss_ema):
                    baseline_loss_ema = b_loss_val
                else:
                    baseline_loss_ema = (1-ema_alpha)*baseline_loss_ema + ema_alpha*b_loss_val

            delta_pct = 0.0
            if not math.isnan(baseline_loss_ema) and not math.isnan(aether2_loss_ema):
                delta_pct = (baseline_loss_ema - aether2_loss_ema) / (baseline_loss_ema + 1e-8) * 100

            # Curvature mean across blocks
            c_mean = float(torch.stack([
                b.curvature.detach() if isinstance(b.curvature, torch.Tensor)
                else torch.tensor(b._c)
                for b in model.blocks
            ]).mean().item())

            # GSI from last GGR block
            gsi = 1.0
            ggr_load = [0.25] * cfg.ggr_n_experts
            cssc_ce  = 0.5
            for blk in reversed(model.blocks):
                if blk.use_ggr and blk.ggr is not None:
                    gsi = blk.ggr.gradient_stability_index
                    ggr_load = blk.ggr.expert_load
                    break
            for blk in model.blocks:
                if blk.use_cssc and blk.cssc is not None:
                    cssc_ce = blk.cssc.context_efficiency
                    break

            mean_ponder  = accum_ponder  / max(accum_count, 1)
            mean_rosetta = accum_rosetta / max(accum_count, 1)
            m = TrainingMetrics(
                step=step,
                epoch=getattr(dataset, "_epoch", 0),
                lr=lr,
                ce_loss=mean_ce,
                ppl=math.exp(min(mean_ce, 20.0)),
                ggr_aux_loss=accum_aux / max(accum_count, 1),
                total_loss=mean_ce + accum_aux / max(accum_count, 1),
                vram_alloc_gib=vram_alloc,
                vram_total_gib=vram_total,
                ms_per_step=ms_step,
                tokens_per_sec=tok_per_sec,
                grad_norm=grad_norms[-1] if grad_norms else 0.0,
                curvature_mean=c_mean,
                ggr_expert_load=ggr_load,
                ggr_gsi=gsi,
                cssc_ce=cssc_ce,
                baseline_loss=baseline_loss_ema,
                delta_pct=delta_pct,
                ponder_cost=mean_ponder,
                fpa_avg_iters=allocator.avg_iters if allocator is not None else 1.0,
                fpa_halt_pct=allocator.halt_pct  if allocator is not None else 0.0,
                rosetta_ce=mean_rosetta,
                picky_skipped=accum_skipped,
            )

            rosetta_str = (
                f" | Rosetta={m.rosetta_ce:.4f}"
                if cfg.rosetta_enabled else ""
            )
            picky_str = (
                f" | skipped={m.picky_skipped}"
                if m.picky_skipped > 0 else ""
            )
            fpa_str = (
                f" | FPA iters={m.fpa_avg_iters:.2f} ponder={m.ponder_cost:.4f}"
                if cfg.fpa_enabled else ""
            )
            logger.info(
                f"Step {step:6d} | LR={lr:.2e} | CE={mean_ce:.4f} "
                f"| PPL={m.ppl:.2f} | GGR-aux={m.ggr_aux_loss:.4f} "
                f"| VRAM={vram_alloc:.2f}/{vram_total:.1f}GiB "
                f"| {tok_per_sec:.0f}tok/s "
                f"| Baseline Δ={delta_pct:+.1f}%"
                f"{rosetta_str}{picky_str}{fpa_str}"
            )

            if metrics_callback is not None:
                metrics_callback(m)

            # Reset accumulators
            accum_ce = accum_aux = accum_ponder = accum_rosetta = accum_count = accum_skipped = 0

        # ── Checkpoint ────────────────────────────────────────────────────
        if step > 0 and step % cfg.checkpoint_every == 0:
            save_checkpoint(step, model, shadow, opt, scaler, cfg, logger,
                            allocator=allocator, rosetta=rosetta)

    # ── Final checkpoint ─────────────────────────────────────────────────────
    logger.info("Training complete. Saving final checkpoint…")
    save_checkpoint(cfg.max_steps, model, shadow, opt, scaler, cfg, logger,
                    allocator=allocator, rosetta=rosetta)
    dataset.stop()
    logger.info("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aether 2 Training Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    Aether2Config.add_cli_args(parser)
    args = parser.parse_args()

    cfg = Aether2Config.from_args(args)

    if args.dashboard:
        try:
            from dashboard import Aether2Dashboard
            with Aether2Dashboard(cfg, max_steps=cfg.max_steps) as dash:
                # suppress_stdout: keeps log lines out of the terminal so the
                # Rich dashboard screen is not corrupted by log output.
                # All logs still go to the log file.
                train(cfg, resume_prefix=args.resume,
                      metrics_callback=dash.update, suppress_stdout=True)
        except ImportError as e:
            print(f"[WARN] Dashboard unavailable ({e}). Running without UI.")
            train(cfg, resume_prefix=args.resume)
    else:
        train(cfg, resume_prefix=args.resume)


if __name__ == "__main__":
    main()
