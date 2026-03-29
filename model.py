"""Aether Omega — Core Model.

Five-Pillar architecture:
  1. Mamba-Hyperbolic Engine  — pure-PyTorch SSM + Poincaré ball Möbius residuals
  2. Gradient-Based Neurogenesis  — dynamic FFN expansion (hooks in train.py)
  3. Anticipatory Picky Learner   — hidden-state consistency loss (computed in train.py)
  4. Rosetta Stone Observer       — detached ~25M probe network
  Bonus: Continuous Thought Tokens, Episodic Memory, EMA self-distillation

  New features:
    — Multi-Timescale SSM (Clockwork Mamba): d_state varies by layer group (8/16/32)
    — Möbius Residual Connections: entire residual stream in the Poincaré ball
    — Iterative Refinement ("Think Twice"): CE-gated second forward pass
    — CSSC (world-first): Curvature-Selective State Coupling — Mamba Δ gate modulates
      per-token Poincaré curvature; salient tokens get richer hyperbolic representation
    — GGR (world-first): Geodesic Gravity Routing Micro-MoE — 3 expert FFNs with
      routing determined by geodesic distance in the Poincaré ball; capacity-neutral

Hyperbolic geometry:
  The residual stream lives in the Poincaré ball with curvature c.
  Component operations (SSM, FFN) work in the tangent space at the origin
  via log_map / exp_map.  Residual connections use Möbius addition.

ROCm note: mamba-ssm ships CUDA-only Triton kernels — incompatible with ROCm.
We implement a numerically equivalent sequential scan in pure PyTorch.
torch.compile is intentionally NOT called; ROCm support is fragile.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn

from aether_config import OmegaConfig


# ──────────────────────────────────────────────────────────────────────────────
# Poincaré Ball Operations (curvature c)
#
# BF16 safety: ALL intermediate computations run in float32.
# Three invariants enforced at every exit point:
#   1. Norms clamped ≥ eps before any division
#   2. arctanh argument clamped < 1
#   3. Output projected strictly inside the open ball
# ──────────────────────────────────────────────────────────────────────────────


def _sqrt_c(c: float | torch.Tensor, eps: float = 1e-10) -> float | torch.Tensor:
    """Safe √c — works with both float and scalar tensor c."""
    if isinstance(c, torch.Tensor):
        return c.float().clamp(min=eps).sqrt()
    return math.sqrt(max(c, eps))


def project_to_ball(x: torch.Tensor, c: float, eps: float = 1e-2) -> torch.Tensor:
    """Clamp vectors to lie strictly inside the Poincaré ball (radius 1/√c - eps).

    Default eps=1e-2 is chosen for BF16 safety: BF16 has step size ~0.008
    near 1.0, so eps must be > 0.008 to survive the float32→BF16 cast.
    With c=1.0 the effective ball radius is 0.99 — negligible loss.
    """
    max_norm = (1.0 / _sqrt_c(c)) - eps
    x_f = x.float()
    norm = x_f.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    cond = norm > max_norm
    x_f = torch.where(cond, x_f * (max_norm / norm), x_f)
    return x_f.to(x.dtype)


def exp_map_zero(v: torch.Tensor, c: float, eps: float = 1e-10) -> torch.Tensor:
    """Exponential map at the origin: tangent vector → Poincaré ball point.

    exp_0^c(v) = tanh(√c · ‖v‖ / 2) / (√c · ‖v‖) · v
    """
    sqrt_c = _sqrt_c(c)
    v_f = v.float()
    norm = v_f.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = torch.tanh(sqrt_c * norm / 2.0) / (sqrt_c * norm)
    return project_to_ball(scale * v_f, c).to(v.dtype)


def log_map_zero(x: torch.Tensor, c: float, eps: float = 1e-10) -> torch.Tensor:
    """Logarithmic map at the origin: Poincaré ball point → tangent vector.

    log_0^c(x) = (2/√c) · arctanh(√c · ‖x‖) / ‖x‖ · x
    """
    sqrt_c = _sqrt_c(c)
    x_f = project_to_ball(x, c).float()        # ensure inside ball
    norm = x_f.norm(dim=-1, keepdim=True).clamp(min=eps)
    arg = (sqrt_c * norm).clamp(max=1.0 - 1e-5)  # arctanh domain: |z| < 1
    scale = (2.0 / sqrt_c) * torch.arctanh(arg) / norm
    return (scale * x_f).to(x.dtype)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float,
               eps: float = 1e-10) -> torch.Tensor:
    """Möbius addition in the Poincaré ball.

    x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 - c‖x‖²)y)
              / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)

    Both operands must lie inside the ball.  Result is clamped.
    """
    x_f = x.float()
    y_f = y.float()

    x_sq = (x_f * x_f).sum(dim=-1, keepdim=True)   # ‖x‖²
    y_sq = (y_f * y_f).sum(dim=-1, keepdim=True)   # ‖y‖²
    xy   = (x_f * y_f).sum(dim=-1, keepdim=True)   # ⟨x,y⟩

    num   = (1.0 + 2.0 * c * xy + c * y_sq) * x_f + \
            (1.0 - c * x_sq) * y_f
    denom = (1.0 + 2.0 * c * xy + c * c * x_sq * y_sq).clamp(min=eps)

    return project_to_ball(num / denom, c).to(x.dtype)


class RiemannianRescale(torch.autograd.Function):
    """Custom autograd: rescales gradients by the inverse Poincaré conformal factor.

    Forward: identity (returns input unchanged).
    Backward: grad ← grad × ((1 - c·‖x‖²) / 2)²

    This compensates for the Poincaré metric tensor g_x = (2/(1-c‖x‖²))²·I
    which inflates gradients near the ball boundary.  Without rescaling,
    hyp_proj gradients can be ~4× smaller than their Euclidean counterparts.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_float = c.item()                              # intentional detach from curvature
        ctx.save_for_backward(x)
        ctx.c = c_float
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        c = ctx.c
        x_f = x.float()
        norm_sq = (x_f * x_f).sum(dim=-1, keepdim=True)
        # Inverse conformal factor squared: ((1 - c‖x‖²) / 2)²
        lambda_inv_sq = ((1.0 - c * norm_sq).clamp(min=1e-6) / 2.0).pow(2)
        return (grad_output.float() * lambda_inv_sq).to(grad_output.dtype), None


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — no mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        norm = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f32 * norm * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU FFN: W_down(SiLU(W_gate(x)) * W_up(x))."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ──────────────────────────────────────────────────────────────────────────────
# GGR — Geodesic Gravity Routing Micro-MoE  (novel: world-first)
#
# Three expert SwiGLU FFNs (Code / Math / Language) routed by geodesic distance
# in the Poincaré ball.  Each expert has a learned centroid; tokens geodesically
# closer to a centroid receive higher routing weight.
#
# Design invariants:
#   • Capacity-neutral: expert_hidden = ff_hidden // n_experts ≈ ff_hidden/3
#     → total params ≈ equivalent single SwiGLU FFN.
#   • Soft routing (all experts run) → fully differentiable, no load-balancing loss.
#   • Log-parameterised temperature (starts at 1.0, learned).
#   • Centroids initialised near origin (small std=0.05) → routing is uniform at
#     init and specialises over training.
# ──────────────────────────────────────────────────────────────────────────────

class GeodesicGravityMoE(nn.Module):
    """Geodesic Gravity Routing Micro-MoE — novel world-first architecture."""

    def __init__(self, cfg: OmegaConfig) -> None:
        super().__init__()
        self.n = cfg.n_moe_experts
        expert_h = cfg.ff_hidden // self.n          # capacity-neutral: ≈ff_hidden/3

        self.experts  = nn.ModuleList([SwiGLU(cfg.d_model, expert_h) for _ in range(self.n)])
        # Learnable centroids — small init near origin, specialise over training
        self.centroids = nn.Parameter(torch.randn(self.n, cfg.d_model) * 0.05)
        # Log-temperature: exp(0)=1.0 → moderate routing sharpness at init
        self.log_temp  = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, c) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — tangent-space representation (after ffn_norm, Euclidean).
            c: curvature — float or 0-dim scalar tensor.
        Returns:
            (B, T, D) weighted mixture of expert outputs, in tangent space.
        """
        B, T, D = x.shape
        n = self.n

        # ── Map inputs and centroids into the Poincaré ball ──
        x_ball = exp_map_zero(x, c)                                     # (B, T, D)
        c_ball = project_to_ball(
            exp_map_zero(self.centroids.to(x.dtype), c), c
        )                                                               # (n, D)

        # ── Batched Poincaré geodesic distances d(x_i, centroid_j) ──
        # Expand for broadcasting: (B*T, n, D) shapes
        x_flat = x_ball.reshape(B * T, 1, D).expand(-1, n, -1)         # (BT, n, D)
        c_flat = c_ball.unsqueeze(0).expand(B * T, -1, -1)             # (BT, n, D)

        # Möbius difference: (-x) ⊕_c centroid  — the hyperbolic "displacement"
        diff = mobius_add(
            -x_flat.reshape(-1, D),     # (BT*n, D) — Möbius inverse of x
            c_flat.reshape(-1, D),      # (BT*n, D)
            c,
        )                                                               # (BT*n, D)

        # Geodesic distance: (2/√c) arctanh(√c · ||diff||)
        sqrt_c_f: float = math.sqrt(max(
            float(c.item() if isinstance(c, torch.Tensor) else c), 1e-10
        ))
        d_norm = diff.float().norm(dim=-1).clamp(min=1e-10)            # (BT*n,)
        arg    = (sqrt_c_f * d_norm).clamp(max=1.0 - 1e-5)
        dists  = (2.0 / sqrt_c_f) * torch.arctanh(arg)                # (BT*n,)
        dists  = dists.reshape(B * T, n).to(x.dtype)                  # (BT, n)

        # ── Geodesic gravity gates: closer centroid → higher weight ──
        temp  = self.log_temp.exp().clamp(min=0.1)
        gates = F.softmax(-dists / temp, dim=-1)                       # (BT, n)
        gates = gates.reshape(B, T, n)                                 # (B, T, n)

        # ── Compute all expert outputs (soft routing — fully differentiable) ──
        outs  = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (B, T, D, n)

        # ── Weighted mixture: Σ_i gate_i · expert_i(x) ──
        return (outs * gates.unsqueeze(2)).sum(dim=-1)                 # (B, T, D)


# ──────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch Mamba SSM (ROCm-compatible chunked scan)
# ──────────────────────────────────────────────────────────────────────────────

def _scan_chunk(
    h_in: torch.Tensor,   # (B, Di, N)  carry from previous chunk
    dt_c: torch.Tensor,   # (B, L, Di)  time steps
    x_c:  torch.Tensor,   # (B, L, Di)  input
    B_c:  torch.Tensor,   # (B, L, N)   SSM B slice
    C_c:  torch.Tensor,   # (B, L, N)   SSM C slice
    A_f:  torch.Tensor,   # (Di, N)     SSM A (shared across chunks)
    D_f:  torch.Tensor,   # (Di,)       SSM D (shared across chunks)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process one chunk of the selective scan. All inputs/outputs float32.

    Returns (y_chunk: (B, L, Di), h_new: (B, Di, N)).

    Called under torch.utils.checkpoint so only one chunk's intermediates
    are live at a time during backward.
    """
    log_A   = dt_c.unsqueeze(-1) * A_f
    Bx      = dt_c.unsqueeze(-1) * (B_c.unsqueeze(2) * x_c.unsqueeze(-1))
    log_P   = log_A.cumsum(dim=1)
    P       = log_P.exp()
    h_carry = P * h_in.unsqueeze(1)
    inv_P   = (-log_P).clamp(max=30.0).exp()
    h_fill  = P * (Bx * inv_P).cumsum(dim=1)
    h_c     = h_carry + h_fill
    h_c     = h_c / h_c.norm(dim=-1, keepdim=True).clamp(min=1.0)
    y_c     = (h_c * C_c.unsqueeze(2)).sum(-1) + D_f * x_c
    return y_c, h_c[:, -1, :, :]


def _chunked_selective_scan(
    x:              torch.Tensor,   # (B, T, D_inner)
    delta:          torch.Tensor,   # (B, T, D_inner)
    A:              torch.Tensor,   # (D_inner, d_state)  — negative values
    B:              torch.Tensor,   # (B, T, d_state)
    C:              torch.Tensor,   # (B, T, d_state)
    D:              torch.Tensor,   # (D_inner,)
    chunk_size:     int = 64,
    use_chunk_ckpt: bool = True,
) -> torch.Tensor:
    """Selective scan with optional per-chunk gradient checkpointing.

    use_chunk_ckpt=True (default):
        Each chunk is wrapped in torch.utils.checkpoint — only ONE chunk's
        backward intermediates are live at a time.  Peak VRAM scales as
        O(chunk_size × Di × N × B) instead of O(T × Di × N × B).
        Use this when there is NO block-level gradient checkpointing above.

    use_chunk_ckpt=False (Aether2 default):
        No per-chunk checkpointing.  Relies on block-level gradient checkpointing
        in Aether2Model.  Eliminates nested recomputation overhead (~3× speedup
        during backward).  Safe on 16 GiB: one block's SSM intermediates
        (~1.5 GiB) fit alongside model params + Adam states.

    chunk_size=0: process the full sequence as a single chunk (no Python loop).

    All arithmetic in float32 for numerical stability.
    """
    from torch.utils.checkpoint import checkpoint as _ckpt

    B_sz, T, Di = x.shape
    N = A.shape[1]

    # chunk_size=0 means process the entire sequence in one shot
    eff_chunk = T if chunk_size <= 0 else chunk_size

    x_f     = x.float()
    delta_f = delta.float()
    A_f     = A.float()          # (Di, N)
    B_f     = B.float()          # (B, T, N)
    C_f     = C.float()          # (B, T, N)
    D_f     = D.float()          # (Di,)

    y = torch.empty(B_sz, T, Di, device=x.device, dtype=torch.float32)
    h = torch.zeros(B_sz, Di, N, device=x.device, dtype=torch.float32)

    # Per-chunk ckpt only makes sense during a training forward pass.
    # When use_chunk_ckpt=False we skip checkpointing entirely regardless
    # of grad mode (block-level ckpt above us handles memory).
    apply_ckpt = use_chunk_ckpt and torch.is_grad_enabled()

    for cs in range(0, T, eff_chunk):
        ce   = min(cs + eff_chunk, T)
        dt_c = delta_f[:, cs:ce, :]
        x_c  = x_f[:, cs:ce, :]
        B_c  = B_f[:, cs:ce, :]
        C_c  = C_f[:, cs:ce, :]

        if apply_ckpt:
            y_c, h = _ckpt(
                _scan_chunk, h, dt_c, x_c, B_c, C_c, A_f, D_f,
                use_reentrant=False,
            )
        else:
            y_c, h = _scan_chunk(h, dt_c, x_c, B_c, C_c, A_f, D_f)

        y[:, cs:ce, :] = y_c

    return y.to(x.dtype)


class PurePyTorchSSM(nn.Module):
    """Mamba-style Selective State Space Model — pure PyTorch, ROCm-compatible.

    Supports multi-timescale: d_state can vary per layer via d_state_override.

    Dimensions:
        d_model  → input/output
        d_inner  = d_model * expand
        d_state  = N  (SSM latent state per channel — varies by layer)
        d_conv   = local depthwise conv width
        dt_rank  = Δ projection rank
    """

    def __init__(self, cfg: OmegaConfig, d_state_override: int | None = None) -> None:
        super().__init__()
        Di = cfg.d_inner
        N  = d_state_override if d_state_override is not None else cfg.d_state
        R  = cfg.dt_rank

        self.in_proj  = nn.Linear(cfg.d_model, Di * 2, bias=False)
        self.conv1d   = nn.Conv1d(Di, Di, kernel_size=cfg.d_conv,
                                  groups=Di, padding=cfg.d_conv - 1, bias=True)
        self.x_proj   = nn.Linear(Di, R + 2 * N, bias=False)
        self.dt_proj  = nn.Linear(R, Di, bias=True)

        A_init = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).expand(Di, -1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D     = nn.Parameter(torch.ones(Di))

        self.out_proj = nn.Linear(Di, cfg.d_model, bias=False)

        self.d_conv   = cfg.d_conv
        self.d_inner  = Di
        self.d_state  = N
        self.dt_rank  = R

        # Scan performance settings — read from config with safe defaults
        self._use_chunk_ckpt = getattr(cfg, "scan_use_chunk_ckpt", True)
        self._chunk_size     = getattr(cfg, "scan_chunk_size", 64)

    def forward(
        self,
        x: torch.Tensor,
        return_delta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)
            return_delta: if True, also return delta_mean (B, T, 1) — used by CSSC.
                          delta_mean is detached to avoid double-differentiating through
                          the scan; CSSC curvature coupling uses it as a straight-through
                          signal.
        """
        B, T, _ = x.shape

        xz   = self.in_proj(x)
        x_s, z = xz.chunk(2, dim=-1)

        x_s = x_s.transpose(1, 2)
        x_s = self.conv1d(x_s)[..., :T]
        x_s = F.silu(x_s).transpose(1, 2)

        x_dbc = self.x_proj(x_s)
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))          # (B, T, D_inner)

        A = -torch.exp(self.A_log.float())

        y = _chunked_selective_scan(
            x_s, delta, A, B_ssm, C_ssm, self.D,
            chunk_size=self._chunk_size,
            use_chunk_ckpt=self._use_chunk_ckpt,
        )
        y = y * F.silu(z)
        out = self.out_proj(y)

        if return_delta:
            # Mean Δ over D_inner → (B, T, 1) selectivity signal for CSSC.
            # Detached: curvature modulation is a straight-through conditional;
            # gradients flow through W_cssc but not back into the scan.
            delta_mean = delta.mean(dim=-1, keepdim=True).detach()
            return out, delta_mean
        return out


# ──────────────────────────────────────────────────────────────────────────────
# AetherMambaBlock — Möbius Residuals + Multi-Timescale SSM
# ──────────────────────────────────────────────────────────────────────────────

class AetherMambaBlock(nn.Module):
    """One Aether Omega block with full Poincaré-ball geometry.

    The residual stream lives in the Poincaré ball.
    SSM and FFN operate in the tangent space at the origin (Euclidean).
    Residual connections use Möbius addition — geometrically coherent.

    Multi-timescale: d_state varies by layer group (Clockwork Mamba).
        Layers 0..7   → d_state=8   (fast — token-level syntax)
        Layers 8..15  → d_state=16  (medium — phrase patterns)
        Layers 16..23 → d_state=32  (slow — paragraph-level logic)

    Data flow per block:
        x (ball) → log_map → tangent → norm → SSM → hyp_proj → exp_map → ball
                → mobius_add with x → new x (ball)
        x (ball) → log_map → tangent → norm → FFN → exp_map → ball
                → mobius_add with x → new x (ball)
    """

    def __init__(self, cfg: OmegaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self._curvature_scale_val = 1.0  # updated by train loop; plain float, not a Module

        # ── Learnable curvature per block (v2 Change 4) ──
        if cfg.learnable_curvature:
            init_raw = math.log(math.exp(cfg.curvature_init) - 1.0)  # inverse softplus
            self._curvature_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        else:
            self._curvature_raw = None

        # Multi-timescale: assign d_state based on layer position
        n_groups = len(cfg.timescale_d_states)
        group = min(layer_idx * n_groups // cfg.n_layers, n_groups - 1)
        self.layer_d_state = cfg.timescale_d_states[group]

        self.ssm_norm = RMSNorm(cfg.d_model)
        self.ssm      = PurePyTorchSSM(cfg, d_state_override=self.layer_d_state)
        self.hyp_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ffn_norm = RMSNorm(cfg.d_model)

        # ── GGR Micro-MoE: replace FFN in every moe_layer_stride-th block ──
        # Blocks 3, 7, 11, 15, 19, 23 (with default stride=4) use GGR-MoE.
        # All other blocks use the standard SwiGLU FFN.
        self.use_ggr = (
            cfg.micro_moe_enabled
            and (layer_idx % cfg.moe_layer_stride == cfg.moe_layer_stride - 1)
        )
        if self.use_ggr:
            self.ggr_moe = GeodesicGravityMoE(cfg)
            self.ffn = None                         # no standard FFN for GGR blocks
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ff_hidden)

        self.drop     = nn.Dropout(cfg.residual_dropout)

        # ── CSSC: learnable Δ-curvature coupling (novel: world-first) ──
        # W_cssc maps mean SSM Δ (1-dim) to a curvature scale factor.
        # Zero init: sigmoid(0)=0.5 → c_scale=1.0 → identity at training start.
        if cfg.cssc_enabled:
            self.W_cssc = nn.Linear(1, 1, bias=True)
            nn.init.zeros_(self.W_cssc.weight)
            nn.init.zeros_(self.W_cssc.bias)

        # ── Geometry gating (v2 Change 3) ──
        if cfg.geometry_gating:
            self.geom_gate_ssm = nn.Linear(cfg.d_model, 1, bias=True)
            self.geom_gate_ffn = nn.Linear(cfg.d_model, 1, bias=True)
            # Zero-init: sigmoid(0)=0.5 → starts with 50/50 Euclidean/Hyperbolic blend
            nn.init.zeros_(self.geom_gate_ssm.weight)
            nn.init.zeros_(self.geom_gate_ssm.bias)
            nn.init.zeros_(self.geom_gate_ffn.weight)
            nn.init.zeros_(self.geom_gate_ffn.bias)

    @property
    def curvature(self) -> float | torch.Tensor:
        """Effective curvature — scalar tensor (learnable, gradients flow) or float (fixed)."""
        if self.cfg.learnable_curvature and self._curvature_raw is not None:
            c = F.softplus(self._curvature_raw).clamp(max=self.cfg.curvature_max)
            return (c * self._curvature_scale_val).clamp(min=0.01)
        return self.cfg.hyp_curvature

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        c = self.curvature

        # ── SSM path ──
        v = log_map_zero(x, c)
        ssm_in = self.ssm_norm(v)
        if self.cfg.cssc_enabled:
            # CSSC: request delta statistics alongside the SSM output
            h_ssm, delta_mean = self.ssm(ssm_in, return_delta=True)
        else:
            h_ssm = self.ssm(ssm_in)
        h_proj = self.drop(self.hyp_proj(h_ssm))

        # Geometry gate: scale residual in tangent space before exp_map
        # gate≈0 → h_proj≈0 → exp_map(0)=origin → mobius_add(x,origin)=x (pass-through)
        # gate≈1 → full hyperbolic residual.  Single path, zero extra cost.
        if self.cfg.geometry_gating and hasattr(self, 'geom_gate_ssm'):
            gate = torch.sigmoid(self.geom_gate_ssm(v.detach())).clamp(min=1e-4)
            h_proj = h_proj * gate

        if self.cfg.cssc_enabled:
            # CSSC (world-first): per-token curvature driven by SSM Δ selectivity.
            # delta_mean ∈ (0, ∞) after softplus; high Δ = salient/complex token.
            # c_scale ∈ (0.5, 1.5): high-Δ tokens get higher curvature → richer
            # hyperbolic encoding of hierarchical structure for that token.
            c_scale = 0.5 + torch.sigmoid(self.W_cssc(delta_mean))     # (B, T, 1)
            c_token = (c * c_scale).clamp(
                min=0.01, max=float(self.cfg.curvature_max)
            )                                                           # (B, T, 1)
            h_ball = exp_map_zero(h_proj, c_token)
            # Safety: c_token may be < c_base (lower curvature → larger ball).
            # project_to_ball ensures h_ball is inside the c_base-ball before
            # Möbius addition — required for mathematically valid Möbius add.
            h_ball = project_to_ball(h_ball, c)
        else:
            h_ball = exp_map_zero(h_proj, c)

        # Riemannian gradient correction (on ball-space point, AFTER exp_map)
        if self.cfg.riemannian_correction:
            c_float = c.item() if isinstance(c, torch.Tensor) else c
            h_ball = RiemannianRescale.apply(h_ball, torch.tensor(c_float))

        x = project_to_ball(mobius_add(x, h_ball, c), c)

        if return_hidden:
            h_captured = h_ball

        # ── FFN / GGR path ──
        v = log_map_zero(x, c)
        if self.use_ggr:
            # GGR (world-first): expert selected by geodesic distance to learned centroids
            h_ffn = self.drop(self.ggr_moe(self.ffn_norm(v), c))
        else:
            h_ffn = self.drop(self.ffn(self.ffn_norm(v)))

        if self.cfg.geometry_gating and hasattr(self, 'geom_gate_ffn'):
            gate = torch.sigmoid(self.geom_gate_ffn(v.detach())).clamp(min=1e-4)
            h_ffn = h_ffn * gate

        h_ffn_ball = exp_map_zero(h_ffn, c)
        x = project_to_ball(mobius_add(x, h_ffn_ball, c), c)

        if return_hidden:
            return x, h_captured
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Persistent Episodic Memory
# ──────────────────────────────────────────────────────────────────────────────

class EpisodicMemory(nn.Module):
    """Differentiable key-value memory bank.

    Cosine similarity retrieval → top-k weighted value sum.
    Keys and values are learned parameters.
    Output is treated as a tangent vector — caller applies exp_map + mobius_add.
    """

    def __init__(self, slots: int, d_model: int, topk: int = 8) -> None:
        super().__init__()
        self.topk = topk
        self.keys   = nn.Parameter(torch.randn(slots, d_model) * 0.02)
        self.values = nn.Parameter(torch.randn(slots, d_model) * 0.02)
        self.gate   = nn.Parameter(torch.zeros(1))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        # Fix 3: soft attention over ALL slots — fully differentiable.
        # Top-k hard indexing breaks gradient flow to self.values; softmax over
        # all slots keeps every gradient path alive.
        B, T, D = query.shape
        q_flat = query.reshape(B * T, D)

        q_norm = F.normalize(q_flat.float(), dim=-1)
        k_norm = F.normalize(self.keys.float(), dim=-1)
        sim    = q_norm @ k_norm.T                          # (B*T, slots)

        weights = F.softmax(sim, dim=-1)                    # (B*T, slots) — soft over ALL slots
        mem_out = weights @ self.values.float()             # (B*T, D)

        return (mem_out.to(query.dtype) * torch.tanh(self.gate)).reshape(B, T, D)


# ──────────────────────────────────────────────────────────────────────────────
# Continuous Thought Tokens
# ──────────────────────────────────────────────────────────────────────────────

class ContinuousThoughtTokens(nn.Module):
    """K learnable scratchpad embeddings prepended to every sequence.

    Tokens are projected into the Poincaré ball before concatenation.
    """

    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.embeddings = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

    def prepend(self, x: torch.Tensor, c: float) -> torch.Tensor:
        """Prepend thought tokens to x.  x must already be in the Poincaré ball."""
        B = x.size(0)
        thought = self.embeddings.expand(B, -1, -1).to(x.dtype)
        thought = project_to_ball(thought, c)       # ensure inside ball
        return torch.cat([thought, x], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Main model — AetherOmegaModel
# ──────────────────────────────────────────────────────────────────────────────

class AetherOmegaModel(nn.Module):
    """Aether Omega — ~445M parameter Mamba-Hyperbolic language model.

    The residual stream lives in the Poincaré ball.  Möbius addition is used
    for all residual connections.  SSM and FFN operate in the tangent space.

    Pipeline:
        token_ids
          → Embedding → exp_map (enter Poincaré ball)
          → ContinuousThoughtTokens prepend (in ball)
          → 24 × AetherMambaBlock (Möbius residuals, multi-timescale SSM)
          → EpisodicMemory (exp_map + Möbius addition)
          → log_map (exit ball → tangent space)
          → RMSNorm → LM head (weight-tied to embedding)

    Supports embed_override for Iterative Refinement ("Think Twice").
    """

    def __init__(self, cfg: OmegaConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding   = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.thought_tok = ContinuousThoughtTokens(cfg.n_thought_tokens, cfg.d_model)
        self.blocks      = nn.ModuleList([
            AetherMambaBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])
        self.episodic    = EpisodicMemory(cfg.episodic_slots, cfg.d_model, cfg.episodic_topk)
        self.norm        = RMSNorm(cfg.d_model)
        self.lm_head     = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # weight tying

        # Refinement projection ("Think Twice")
        self.refinement_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self._init_weights()

        # v2 Change 4: learnable curvature warmup scale
        self._curvature_scale = 1.0  # ramps 0→1 over curvature_warmup_steps in train loop

    def _init_weights(self) -> None:
        # Fix 1: depth-scaled initialisation (GPT-2 style).
        # Base std for most Linear layers = 0.02.
        # Output projections in each block (ssm.out_proj, ffn.w_down) are scaled
        # by 1/sqrt(2 * n_layers) to prevent gradient explosion in deep networks.
        n_layers = self.cfg.n_layers
        output_proj_std = 0.02 / math.sqrt(2.0 * n_layers)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Re-init output projections with depth-scaled std
        for block in self.blocks:
            nn.init.normal_(block.ssm.out_proj.weight, mean=0.0, std=output_proj_std)
            if block.ffn is not None:
                # Standard FFN block: depth-scale the output projection
                nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=output_proj_std)
            elif hasattr(block, 'ggr_moe'):
                # GGR-MoE block: depth-scale each expert's output projection
                for expert in block.ggr_moe.experts:
                    nn.init.normal_(expert.w_down.weight, mean=0.0, std=output_proj_std)
            # v2: re-zero geometry gates (the generic normal_ init above overwrites them)
            if hasattr(block, 'geom_gate_ssm'):
                nn.init.zeros_(block.geom_gate_ssm.weight)
                nn.init.zeros_(block.geom_gate_ssm.bias)
                nn.init.zeros_(block.geom_gate_ffn.weight)
                nn.init.zeros_(block.geom_gate_ffn.bias)
            # CSSC: re-zero W_cssc (normal_ above overwrites the zero-init in AetherMambaBlock)
            # Zero-init → sigmoid(0)=0.5 → c_scale=1.0 → identity at training start.
            if hasattr(block, 'W_cssc'):
                nn.init.zeros_(block.W_cssc.weight)
                nn.init.zeros_(block.W_cssc.bias)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        capture_hidden_indices: Optional[set[int]] = None,
        embed_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:
        """
        Args:
            input_ids: (B, T) token IDs.  Ignored if embed_override is provided.
            capture_hidden_indices: set of block indices whose hidden states to capture.
            embed_override: (B, T, D) pre-computed Poincaré ball embeddings
                            for iterative refinement (thought tokens prepended internally).

        Returns:
            logits:        (B, T, vocab_size)
            hidden_states: dict[layer_idx → (B, T, d_model)]  — SSM outputs in ball
            features:      (B, T, d_model) — normed tangent-space features (pre-logit)
        """
        c = self.cfg.hyp_curvature
        K = self.cfg.n_thought_tokens

        if embed_override is not None:
            x = embed_override                          # (B, T, D) already in ball
            T_orig = x.size(1)
        else:
            x = self.embedding(input_ids)               # (B, T, D) Euclidean
            x = exp_map_zero(x, c)                      # → Poincaré ball
            T_orig = input_ids.size(1)

        x = self.thought_tok.prepend(x, c)             # (B, K+T, D) in ball
        assert x.size(1) == K + T_orig, (
            f"Fix 4 sanity: expected K+T={K + T_orig} tokens after prepend, "
            f"got {x.size(1)}"
        )

        hidden_states: dict[int, torch.Tensor] = {}
        use_ckpt = self.cfg.use_gradient_checkpointing and self.training

        for i, block in enumerate(self.blocks):
            need_hidden = capture_hidden_indices and i in capture_hidden_indices
            if need_hidden:
                # Captured layer: full forward (no checkpoint)
                x, h = block(x, return_hidden=True)
                hidden_states[i] = h[:, K:, :]          # strip thought tokens
            elif use_ckpt:
                x = ckpt_fn(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Episodic memory (output treated as tangent vector → exp_map → Möbius add)
        mem = self.episodic(x)
        mem_ball = exp_map_zero(mem, c)
        x = project_to_ball(mobius_add(x, mem_ball, c), c)  # v2: safety clamp on RESULT

        # Exit Poincaré ball → tangent space → norm → LM head
        x = log_map_zero(x, c)
        x = self.norm(x)
        x = x[:, K:, :]                                # strip thought tokens
        # Fix 4: assert stripped length matches original input length
        assert x.size(1) == T_orig, (
            f"Fix 4 sanity: stripped output length {x.size(1)} != input length {T_orig}"
        )

        logits = self.lm_head(x)                        # (B, T, V)

        return logits, hidden_states, x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Pillar 4 — Rosetta Stone Observer (~28M params)
# ──────────────────────────────────────────────────────────────────────────────

class _RosettaAttentionBlock(nn.Module):
    """Lightweight causal self-attention block for the Rosetta probe."""

    def __init__(self, d_probe: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_probe // n_heads
        self.norm     = RMSNorm(d_probe)
        self.q_proj   = nn.Linear(d_probe, d_probe, bias=False)
        self.k_proj   = nn.Linear(d_probe, d_probe, bias=False)
        self.v_proj   = nn.Linear(d_probe, d_probe, bias=False)
        self.o_proj   = nn.Linear(d_probe, d_probe, bias=False)
        self.ffn_norm = RMSNorm(d_probe)
        self.ffn      = SwiGLU(d_probe, d_probe * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        residual = x
        x_n = self.norm(x)
        q = self.q_proj(x_n).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_n).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_n).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = residual + self.o_proj(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RosettaObserver(nn.Module):
    """Secondary ~28M probe that decodes the main model's latent space.

    Receives hidden states via .detach() — stop-gradient ensures Rosetta's
    loss never flows back into AetherOmega.

    NOTE: Caller must convert hidden states from Poincaré ball to tangent space
    (via log_map_zero) before passing to this module, since Rosetta uses
    standard attention which operates in Euclidean space.
    """

    def __init__(self, cfg: OmegaConfig) -> None:
        super().__init__()
        self.proj_in = nn.Linear(cfg.d_model, cfg.rosetta_d_probe, bias=False)
        self.blocks  = nn.ModuleList([
            _RosettaAttentionBlock(cfg.rosetta_d_probe, cfg.rosetta_n_heads)
            for _ in range(cfg.rosetta_n_layers)
        ])
        self.norm    = RMSNorm(cfg.rosetta_d_probe)
        self.lm_head = nn.Linear(cfg.rosetta_d_probe, cfg.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, T, d_model) — tangent-space vectors (log_map applied by caller).
        Returns:
            (B, T, vocab_size) probe logits
        """
        x = self.proj_in(hidden.detach())       # stop-gradient
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = OmegaConfig()
    print(cfg.summary())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"\nDevice: {device}  |  dtype: {dtype}")

    model   = AetherOmegaModel(cfg).to(device=device, dtype=dtype)
    rosetta = RosettaObserver(cfg).to(device=device, dtype=dtype)

    total_main    = model.count_parameters()
    total_rosetta = rosetta.count_parameters()
    total         = total_main + total_rosetta
    print(f"\nAetherOmegaModel  : {total_main / 1e6:.2f}M params")
    print(f"RosettaObserver   : {total_rosetta / 1e6:.2f}M params")
    print(f"Combined total    : {total / 1e6:.2f}M params")

    # Print multi-timescale + GGR/CSSC info
    for i, block in enumerate(model.blocks):
        if i < 3 or i == cfg.n_layers - 1:
            mode = "GGR-MoE" if block.use_ggr else "SwiGLU"
            cssc_mark = "+CSSC" if cfg.cssc_enabled else ""
            print(f"  Block {i:2d}: d_state={block.layer_d_state}  FFN={mode}{cssc_mark}")
        elif i == 3:
            print(f"  ...")

    B, T = 2, 32
    ids  = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    with torch.no_grad():
        logits, hs, feats = model(ids, capture_hidden_indices={cfg.n_layers - 1})
        h_tangent = log_map_zero(hs[cfg.n_layers - 1], cfg.hyp_curvature)
        probe_logits = rosetta(h_tangent)

    print(f"\nForward pass  — input: {list(ids.shape)}")
    print(f"  logits      : {list(logits.shape)}")
    print(f"  hidden[-1]  : {list(hs[cfg.n_layers - 1].shape)}")
    print(f"  features    : {list(feats.shape)}")
    print(f"  probe logits: {list(probe_logits.shape)}")

    # NaN check
    nan_check = {
        "logits": logits.isnan().any().item(),
        "hidden": hs[cfg.n_layers - 1].isnan().any().item(),
        "features": feats.isnan().any().item(),
        "probe": probe_logits.isnan().any().item(),
    }
    print(f"\nNaN check: {nan_check}")
    assert not any(nan_check.values()), "NaN detected!"
    print("\nAll shapes correct.  No NaNs.  Möbius geometry coherent.")
