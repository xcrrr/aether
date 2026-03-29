"""Aether 2 — Core Model Architecture.

Transformer blocks augmented with:
  • CSSC (Cross-Scale Spatiotemporal Correlation) attention
  • GGR (Gated Gradient Routing) MoE FFN
  • Poincaré ball Möbius residuals (reused from model.py)
  • Gradient checkpointing for ROCm 16 GB VRAM

Architecture per Aether2Block:
  x → log_map → RMSNorm → PurePyTorchSSM → residual
                        → CSSCAttention  → residual  (blended via geom_gate)
                        → [SwiGLU | GGR] → residual → exp_map → project_to_ball

ShadowModel:
  6-layer vanilla Aether2Block (no CSSC, no GGR) updated via EMA of main
  model's first N layers — used as the real-time performance baseline.

ROCm rules: no torch.compile, all Poincaré ops in float32.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn

from aether2_config import Aether2Config
from cssc import CSSCAttention
from ggr import GatedGradientRouter

# Import proven Poincaré primitives + reusable building blocks from model.py
from model import (
    RMSNorm,
    SwiGLU,
    PurePyTorchSSM,
    exp_map_zero,
    log_map_zero,
    mobius_add,
    project_to_ball,
    RiemannianRescale,
    EpisodicMemory,
    ContinuousThoughtTokens,
)


# ─────────────────────────────────────────────────────────────────────────────
# Aether 2 Block
# ─────────────────────────────────────────────────────────────────────────────

class Aether2Block(nn.Module):
    """Single Aether 2 transformer block.

    Two parallel residual streams (SSM + CSSC) fused by a geometry gate,
    followed by an FFN residual (SwiGLU or GGR-MoE).  All residuals live
    in the Poincaré ball; components operate in the tangent space.

    Parameters
    ----------
    cfg : Aether2Config
    layer_idx : int
        Zero-based index. Used to determine d_state tier (Clockwork Mamba)
        and whether this layer uses GGR.
    use_cssc : bool
        Override: set False for ShadowModel or ablation (--no-cssc).
    use_ggr : bool
        Override: set False for ShadowModel or ablation (--no-ggr).
    """

    def __init__(
        self,
        cfg: Aether2Config,
        layer_idx: int,
        use_cssc: bool = True,
        use_ggr: bool = True,
    ) -> None:
        super().__init__()
        D = cfg.d_model
        self.layer_idx = layer_idx

        # ── Curvature (learnable or fixed) ───────────────────────────────
        if cfg.learnable_curvature:
            self.curvature = nn.Parameter(
                torch.tensor(cfg.curvature_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "curvature", torch.tensor(cfg.hyp_curvature, dtype=torch.float32)
            )

        # ── Clockwork Mamba: tier-based d_state ──────────────────────────
        n_tiers = len(cfg.timescale_d_states)
        tier = layer_idx % n_tiers
        d_state = cfg.timescale_d_states[tier]

        # ── SSM path ─────────────────────────────────────────────────────
        self.ssm_norm = RMSNorm(D)
        # PurePyTorchSSM takes the full config + optional d_state_override
        self.ssm = PurePyTorchSSM(cfg, d_state_override=d_state)

        # ── CSSC attention path ───────────────────────────────────────────
        self.use_cssc = use_cssc and cfg.cssc_enabled
        if self.use_cssc:
            self.cssc_norm = RMSNorm(D)
            self.cssc = CSSCAttention(cfg)

        # ── Geometry gating (blend SSM and CSSC outputs) ─────────────────
        if self.use_cssc:
            self.geom_gate = nn.Linear(D, 2, bias=True)
            nn.init.zeros_(self.geom_gate.weight)
            nn.init.constant_(self.geom_gate.bias, 0.0)  # → equal blend at init

        # ── Optional hyperbolic projection after SSM ──────────────────────
        self.hyp_proj = nn.Linear(D, D, bias=False)
        nn.init.eye_(self.hyp_proj.weight)

        # ── CSSC curvature coupling (W_cssc): Δ-mean → c_scale ───────────
        self.w_cssc = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.w_cssc.weight)
        nn.init.zeros_(self.w_cssc.bias)

        # ── FFN path: SwiGLU or GGR ───────────────────────────────────────
        self.use_ggr = (
            use_ggr
            and cfg.micro_moe_enabled
            and (cfg.ggr_layer_stride > 0)
            and (layer_idx % cfg.ggr_layer_stride == cfg.ggr_layer_stride - 1)
        )
        self.ffn_norm = RMSNorm(D)
        if self.use_ggr:
            self.ggr = GatedGradientRouter(cfg)
            self.ffn = None
        else:
            self.ffn = SwiGLU(D, cfg.ff_hidden)
            self.ggr = None

        # ── Riemannian rescale (detached curvature) ───────────────────────
        self.riemannian_rescale = RiemannianRescale

        # ── Residual dropout ──────────────────────────────────────────────
        self.drop = nn.Dropout(p=cfg.residual_dropout) if cfg.residual_dropout > 0 else None

    @property
    def _c(self) -> float:
        """Return curvature as a Python float (safe for Poincaré ops)."""
        c = self.curvature
        if isinstance(c, nn.Parameter) or isinstance(c, torch.Tensor):
            return c.item()
        return float(c)

    def _apply_drop(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x) if self.drop is not None else x

    # ── Block forward (no checkpointing — checkpointing applied in model) ─

    def _forward_inner(
        self,
        x: torch.Tensor,
        c_float: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inner forward. Returns (output, aux_loss).

        Parameters
        ----------
        c_float : pre-computed curvature value as a Python float.
            When supplied by Aether2Model.forward() (which batches all curvature
            reads into a single GPU→CPU sync), this avoids a per-block sync.
            When None (e.g. block.forward() called standalone), falls back to
            c_tensor.item() as before.
        """
        # Derive c_tensor directly from self.curvature — preserves gradient path
        # when learnable_curvature=True so the parameter actually gets updated.
        # c (float) is still needed for Poincaré ops that require Python scalars.
        c_tensor = self.curvature.clamp(min=1e-4).to(device=x.device, dtype=torch.float32)
        c = c_float if c_float is not None else c_tensor.item()

        # Map from ball to tangent space
        h = log_map_zero(x, c)                        # (B, T, D) float → dtype

        # ── SSM residual ─────────────────────────────────────────────────
        h_ssm = self.ssm_norm(h)
        h_ssm_out, delta_mean = self.ssm(h_ssm, return_delta=True)  # delta_mean: (B, T, 1)

        # Per-token curvature modulation via Δ (CSSC coupling)
        # c_tensor participates in graph, giving self.curvature a gradient path
        # via the CSSC curvature gate (see below).
        c_scale = 0.5 + torch.sigmoid(self.w_cssc(delta_mean.detach()))  # (B,T,1)
        # c_tensor is float32; c_scale may be BF16 when model params are BF16 —
        # cast to float32 so the multiplication stays in the correct dtype.
        c_token = c_tensor * c_scale.float()           # (B, T, 1) — c_tensor in graph

        h_ssm_proj = self.hyp_proj(h_ssm_out)         # (B, T, D)

        # Straight-through curvature gate: exactly identity on the forward pass
        # (c_token / c_token ≡ 1), but the backward pass routes a gradient
        # through c_token → c_tensor → self.curvature.  This guarantees that
        # learnable curvature receives gradients even when CSSC is disabled
        # (--no-cssc ablations), in addition to the CSSC curv_gate path below.
        _ct_det = c_token.detach().clamp(min=1e-8)
        h_ssm_proj = h_ssm_proj * (c_token / _ct_det).to(h_ssm_proj.dtype)

        # ── CSSC residual (optional) ──────────────────────────────────────
        if self.use_cssc:
            h_cssc = self.cssc_norm(h)
            # Pass c_tensor (not float c) so CSSCAttention.curv_gate receives a
            # live tensor — gradient flows: self.curvature → c_tensor → curv_gate
            h_cssc_out = self.cssc(h_cssc, c_tensor)   # (B, T, D)

            # Learnable blend gate: sigmoid → (B, T, 2) weights
            gate_logits = self.geom_gate(h.detach())   # (B, T, 2) — detach for compat
            gates = gate_logits.softmax(dim=-1)         # (B, T, 2)
            g_ssm  = gates[..., 0:1]                   # (B, T, 1)
            g_cssc = gates[..., 1:2]                   # (B, T, 1)
            h_combined = g_ssm * h_ssm_proj + g_cssc * h_cssc_out
        else:
            h_combined = h_ssm_proj
            gates = None

        # Riemannian rescale then Möbius residual
        h_combined = RiemannianRescale.apply(h_combined, c_tensor)
        h_combined = self._apply_drop(h_combined)

        # exp_map with per-token curvature
        x_ball = exp_map_zero(h_combined, c)
        x = mobius_add(x, x_ball, c)
        x = project_to_ball(x, c)

        # ── FFN residual ──────────────────────────────────────────────────
        h2 = log_map_zero(x, c)
        h2 = self.ffn_norm(h2)

        aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        if self.use_ggr:
            h2_out, aux_loss = self.ggr(h2, c)
        else:
            h2_out = self.ffn(h2)

        h2_out = RiemannianRescale.apply(h2_out, c_tensor)
        h2_out = self._apply_drop(h2_out)

        x_ffn = exp_map_zero(h2_out, c)
        x = mobius_add(x, x_ffn, c)
        x = project_to_ball(x, c)

        return x, aux_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward with gradient checkpointing support.

        Returns (output, aux_loss).
        Called when the block is run standalone (no Aether2Model batching).
        """
        return self._forward_inner(x)


# ─────────────────────────────────────────────────────────────────────────────
# Aether 2 Model
# ─────────────────────────────────────────────────────────────────────────────

class Aether2Model(nn.Module):
    """Full Aether 2 model.

    Embedding → Continuous Thought Tokens → Poincaré ball → N Aether2Blocks
    → Episodic Memory → RMSNorm → LM Head

    Weight-tied embedding / LM head (same convention as AetherOmegaModel).
    """

    def __init__(self, cfg: Aether2Config) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        V = cfg.vocab_size

        # ── Embedding ─────────────────────────────────────────────────────
        self.embedding = nn.Embedding(V, D)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # ── Continuous Thought Tokens ─────────────────────────────────────
        self.thought_tokens = ContinuousThoughtTokens(cfg.n_thought_tokens, D)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            Aether2Block(cfg, i,
                         use_cssc=cfg.cssc_enabled,
                         use_ggr=cfg.micro_moe_enabled)
            for i in range(cfg.n_layers)
        ])

        # ── Episodic Memory ───────────────────────────────────────────────
        # EpisodicMemory(slots, d_model, topk) — note argument order
        self.episodic = EpisodicMemory(cfg.episodic_slots, D, cfg.episodic_topk)

        # ── Output normalisation ──────────────────────────────────────────
        self.norm_out = RMSNorm(D)

        # ── LM Head (weight-tied) ─────────────────────────────────────────
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.embedding.weight

        # ── Gradient checkpointing flag ───────────────────────────────────
        self.use_ckpt = cfg.use_gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,                  # (B, T)
        capture_hidden_indices: set[int] | None = None,
        embed_override: torch.Tensor | None = None,
        return_final_embedding: bool = False,
    ):
        """Forward pass.

        Parameters
        ----------
        input_ids              : (B, T) token indices
        capture_hidden_indices : set of layer indices to capture hidden states at
        embed_override         : (B, T, D) tangent-space embedding override —
                                 bypasses token lookup; model still applies
                                 exp_map_zero and prepends thought tokens.
                                 Used by FluidPowerAllocator for re-entry passes.
        return_final_embedding : if True, append x_ball (the Poincaré ball
                                 embedding after all blocks + episodic memory,
                                 shape (B, T+n_thought, D)) as a 4th return value.
                                 Existing callers unpacking 3 values are unaffected.

        Returns
        -------
        logits       : (B, T+n_thought, V)  — language model logits
        hidden_states: dict[int, Tensor]    — captured hidden states by layer idx
        aux_loss     : scalar tensor        — sum of GGR load-balance losses
        [x_ball]     : (B, T+n_thought, D) — only when return_final_embedding=True
        """
        cfg = self.cfg
        c = cfg.hyp_curvature

        # ── Embed ─────────────────────────────────────────────────────────
        if embed_override is not None:
            x = embed_override
        else:
            x = self.embedding(input_ids)           # (B, T, D)

        # Enter Poincaré ball
        x = exp_map_zero(x, c)
        x = project_to_ball(x, c)

        # Prepend thought tokens (must be called after entering the ball)
        x = self.thought_tokens.prepend(x, c)       # (B, T+n_thought, D)

        hidden_states: dict[int, torch.Tensor] = {}
        aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        # ── Pre-fetch all block curvatures (one GPU→CPU sync instead of 24) ──
        # Each block's c_tensor.item() forces a host-device sync; doing them all
        # at once via .tolist() costs a single sync for the entire model forward.
        # The float values are passed to _forward_inner so the per-block .item()
        # call is skipped.
        with torch.no_grad():
            _c_all = torch.stack([
                b.curvature.clamp(min=1e-4) for b in self.blocks
            ]).to(dtype=torch.float32)
        c_floats: list[float] = _c_all.tolist()   # one sync here

        # ── Run blocks ────────────────────────────────────────────────────
        for i, block in enumerate(self.blocks):
            c_f = c_floats[i]
            if self.use_ckpt:
                # use_reentrant=False: safer with custom autograd functions.
                # Pass c_f as a non-tensor arg — checkpoint accepts primitives.
                x_new, al = ckpt_fn(block._forward_inner, x, c_f, use_reentrant=False)
            else:
                x_new, al = block._forward_inner(x, c_f)

            x = x_new
            aux_loss = aux_loss + al

            if capture_hidden_indices and i in capture_hidden_indices:
                hidden_states[i] = x.detach()

        # ── Episodic memory ───────────────────────────────────────────────
        # episodic() returns a tangent-space vector; map to ball then Möbius-add.
        # At init gate=0 → mem≈0 → exp_map(0)=0 → mobius_add(x,0)=x (identity).
        mem = self.episodic(x)
        mem_ball = exp_map_zero(mem, c)
        x = project_to_ball(mobius_add(x, mem_ball, c), c)

        # Capture final ball embedding before exiting to logits.
        # Used by FluidPowerAllocator: x_ball is passed back via embed_override
        # (after log_map) on subsequent re-routing passes.
        x_ball = x  # (B, T+n_thought, D) — still in Poincaré ball

        # ── Exit ball → logits ────────────────────────────────────────────
        h = log_map_zero(x, c)
        h = self.norm_out(h)
        logits = self.lm_head(h)                    # (B, T+n_thought, V)

        if return_final_embedding:
            return logits, hidden_states, aux_loss, x_ball
        return logits, hidden_states, aux_loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self) -> None:
        total = self.count_parameters()
        print(f"Aether 2 Model — {total / 1e6:.1f}M parameters")
        ggr_layers = sum(1 for b in self.blocks if b.use_ggr)
        cssc_layers = sum(1 for b in self.blocks if b.use_cssc)
        print(f"  CSSC layers  : {cssc_layers} / {len(self.blocks)}")
        print(f"  GGR layers   : {ggr_layers} / {len(self.blocks)}")


# ─────────────────────────────────────────────────────────────────────────────
# Shadow (Baseline) Model
# ─────────────────────────────────────────────────────────────────────────────

class ShadowModel(nn.Module):
    """Lightweight EMA baseline transformer for real-time comparison.

    Architecture: vanilla Aether2Blocks with both CSSC and GGR disabled.
    Shares d_model with the main model so EMA weight transfer is direct.

    Weights are never optimised; they are updated from main model blocks
    via EMA in the training loop:
        shadow_p ← ema_decay * shadow_p + (1 - ema_decay) * main_p
    """

    def __init__(self, cfg: Aether2Config) -> None:
        super().__init__()
        D = cfg.d_model
        V = cfg.vocab_size
        n = cfg.baseline_n_layers

        self.embedding = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([
            Aether2Block(cfg, i, use_cssc=False, use_ggr=False)
            for i in range(n)
        ])
        self.norm_out = RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.embedding.weight

        self._c = cfg.hyp_curvature

        # Shadow weights are never in the optimiser
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns logits (B, T, V)."""
        c = self._c
        x = self.embedding(input_ids)
        x = exp_map_zero(x, c)
        x = project_to_ball(x, c)
        aux = torch.zeros(1, device=x.device)
        for block in self.blocks:
            x, _ = block(x)
        h = log_map_zero(x, c)
        h = self.norm_out(h)
        return self.lm_head(h)

    @torch.no_grad()
    def ema_update(self, main_blocks: nn.ModuleList, decay: float) -> None:
        """EMA update from the first N layers of the main model.

        Only updates parameters that exist in both shadow and main blocks
        with identical shapes (shadow has no CSSC/GGR params).
        """
        for s_block, m_block in zip(self.blocks, main_blocks):
            s_params = dict(s_block.named_parameters())
            m_params = dict(m_block.named_parameters())
            for name, sp in s_params.items():
                if name in m_params:
                    mp = m_params[name]
                    if sp.shape == mp.shape:
                        sp.data.mul_(decay).add_(mp.data.to(sp.dtype),
                                                  alpha=1.0 - decay)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = Aether2Config(
        n_layers=2,
        max_seq_len=64,
        n_thought_tokens=4,
        episodic_slots=32,
        use_gradient_checkpointing=False,
    )
    print(cfg.summary())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    model = Aether2Model(cfg).to(device)
    model.print_summary()

    shadow = ShadowModel(cfg).to(device)
    print(f"Shadow model: {sum(p.numel() for p in shadow.parameters()) / 1e6:.1f}M params")

    # Forward pass
    ids = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        logits, hiddens, aux = model(ids)
    T_out = cfg.max_seq_len + cfg.n_thought_tokens
    assert logits.shape[1] >= 32, f"Unexpected logits shape: {logits.shape}"
    print(f"\nForward OK: logits={tuple(logits.shape)}, aux_loss={aux.item():.4f}")

    # Shadow forward
    with torch.no_grad():
        s_logits = shadow(ids)
    print(f"Shadow OK:  logits={tuple(s_logits.shape)}")

    # EMA update test
    shadow.ema_update(model.blocks, decay=0.999)
    print("EMA update OK")
    print("\nAll checks passed.")
