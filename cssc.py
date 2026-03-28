"""CSSC — Cross-Scale Spatiotemporal Correlation.

Multi-Head Temporal Attention across three scales:
  • Token-level  (local):     causal sliding window, width = cssc_window_size
  • Sentence-level (mid):     cross-attention from x to pooled sentence segments
  • Block-level  (global):    cross-attention from x to pooled block segments

Hyperbolic temporal decay applied as a log-bias to attention scores:
  decay_bias(Δt) = log(1 / (1 + α · |Δt|)) = -log(1 + α · |Δt|)

Memory-efficient: all attention calls use F.scaled_dot_product_attention,
which dispatches to Flash-Attention or efficient SDPA on ROCm.

ROCm rules:
  - torch.compile is never called.
  - All hyperbolic/curvature ops run in float32; attention itself runs in BF16.
  - "cuda" string is used even on ROCm hardware.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether2_config import Aether2Config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-Mean-Square normalisation (no bias). Matches model.py RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = (x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (x_f * rms).to(x.dtype) * self.weight


def _hyperbolic_decay_bias(
    T_q: int,
    T_k: int,
    alpha: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an additive log-decay bias for attention scores.

    Shape: (1, 1, T_q, T_k) — broadcastable over (B, H, T_q, T_k).
    For position i (query) attending to position j (key):
        bias[i, j] = -log(1 + α · |i - j|)
    Positions where j > i (future) get -inf to enforce causality.
    """
    q_idx = torch.arange(T_q, device=device, dtype=dtype).unsqueeze(1)  # (T_q, 1)
    k_idx = torch.arange(T_k, device=device, dtype=dtype).unsqueeze(0)  # (1, T_k)
    dist = (q_idx - k_idx).abs()                                          # (T_q, T_k)
    bias = -torch.log1p(alpha * dist)                                     # (T_q, T_k)
    # Causal mask: future positions → -inf
    future_mask = k_idx > q_idx                                           # (T_q, T_k)
    bias = bias.masked_fill(future_mask, float("-inf"))
    return bias.unsqueeze(0).unsqueeze(0)                                 # (1,1,T_q,T_k)


def _window_mask(
    T: int,
    window: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive mask that zeroes out attention beyond window tokens.

    Shape: (1, 1, T, T).  Entries outside [i-window, i] → -inf.
    """
    row = torch.arange(T, device=device, dtype=dtype)
    col = torch.arange(T, device=device, dtype=dtype)
    dist = row.unsqueeze(1) - col.unsqueeze(0)          # (T, T), positive = past
    mask = (dist < 0) | (dist >= window)                # True = mask out
    additive = torch.zeros(T, T, device=device, dtype=dtype)
    additive.masked_fill_(mask, float("-inf"))
    return additive.unsqueeze(0).unsqueeze(0)           # (1,1,T,T)


# ─────────────────────────────────────────────────────────────────────────────
# Scale-level attention helpers
# ─────────────────────────────────────────────────────────────────────────────

class _ScaleAttention(nn.Module):
    """Single-scale multi-head attention used within CSSC.

    Keys/values may come from a downsampled representation (sentence / block)
    while queries always come from the full-resolution sequence.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        q_src: torch.Tensor,                # (B, T_q, D) — full-res queries
        kv_src: torch.Tensor,               # (B, T_k, D) — may be downsampled
        attn_bias: torch.Tensor | None,     # (1, 1, T_q, T_k) or None
    ) -> torch.Tensor:                      # (B, T_q, D)
        B, T_q, D = q_src.shape
        T_k = kv_src.shape[1]

        Q = self.q_proj(q_src).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_src).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_src).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, H, T_q/T_k, head_dim)

        # F.scaled_dot_product_attention handles Flash/efficient SDPA dispatch
        if attn_bias is not None:
            # attn_mask must be float for additive bias mode
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_bias.to(Q.dtype),
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,      # causality encoded in bias itself
            )
        else:
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        # (B, H, T_q, head_dim) → (B, T_q, D)
        return out.transpose(1, 2).contiguous().view(B, T_q, D)


# ─────────────────────────────────────────────────────────────────────────────
# CSSC Main Module
# ─────────────────────────────────────────────────────────────────────────────

class CSSCAttention(nn.Module):
    """Cross-Scale Spatiotemporal Correlation attention block.

    Replaces a standard attention layer with three parallel temporal streams:
      1. Token-level  : local sliding-window causal attention (width W)
      2. Sentence-level: cross-attention to average-pooled sentence segments
      3. Block-level   : cross-attention to average-pooled block segments

    All three streams share the same number of heads (cssc_n_heads).
    Their outputs are blended via learned scale weights and projected.

    Curvature modulation (from the main model block) is used to scale the
    final output — high-curvature tokens get amplified CSSC signal.

    Parameters
    ----------
    cfg : Aether2Config
    """

    def __init__(self, cfg: Aether2Config) -> None:
        super().__init__()
        D = cfg.d_model
        H = cfg.cssc_n_heads

        self.d_model = D
        self.n_heads = H
        self.head_dim = D // H
        self.window_size = cfg.cssc_window_size
        self.sentence_stride = cfg.cssc_sentence_stride
        self.block_size = cfg.cssc_block_size
        self.decay_alpha = cfg.cssc_decay_alpha

        drop = cfg.cssc_attn_dropout

        # Three scale-specific attention modules
        self.token_attn    = _ScaleAttention(D, H, dropout=drop)
        self.sentence_attn = _ScaleAttention(D, H, dropout=drop)
        self.block_attn    = _ScaleAttention(D, H, dropout=drop)

        # Learnable log-scale blend weights (token, sentence, block)
        init = torch.tensor(cfg.cssc_scale_init, dtype=torch.float32).log()
        self.log_scale_weights = nn.Parameter(init)

        # Curvature gate: (D,) → scalar, applied per-head
        self.curv_gate = nn.Linear(1, H, bias=True)
        nn.init.zeros_(self.curv_gate.weight)
        nn.init.ones_(self.curv_gate.bias)

        # Layer norms before each scale stream
        self.norm_token    = RMSNorm(D)
        self.norm_sentence = RMSNorm(D)
        self.norm_block    = RMSNorm(D)

        # Output projection — zero-initialised so CSSC starts as identity
        self.out_proj = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # Per-head attention entropy tracking (for visualisation)
        self._last_expert_balance: list[float] = [0.33, 0.33, 0.34]

    # ── Internal pooling ────────────────────────────────────────────────────

    @staticmethod
    def _causal_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
        """Strictly-causal average-pool along T.

        Returns shape (B, max(1, T // stride), D) where pooled segment j
        contains ONLY tokens from segments 0 … j-1 (strictly past).

        Implementation: pool normally then RIGHT-SHIFT by one slot (prepend
        a zero segment, drop the last).  This guarantees that no future token
        within the current segment contaminates the cross-attention keys.

        Worked example (stride=8):
          pool[0] = avg(pos 0-7)  →  shifted: slot 0 = zeros
          pool[1] = avg(pos 8-15) →  shifted: slot 1 = pool[0]  (past ✓)
          Token at pos 11 (seg 1) attends slots 0,1 — both contain only past.
        """
        B, T, D = x.shape
        effective_stride = min(stride, T)
        xp = x.transpose(1, 2)                                # (B, D, T)

        if effective_stride >= T:
            # Whole sequence fits in one window — return zeros (nothing past yet)
            return torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)

        pooled = F.avg_pool1d(xp, kernel_size=effective_stride,
                              stride=effective_stride, ceil_mode=False)
        # pooled: (B, D, T//stride)

        # Causal shift: prepend zero slot, drop the last slot.
        # After shift, slot j holds the content of the PREVIOUS segment (j-1).
        zeros = torch.zeros(B, D, 1, device=x.device, dtype=x.dtype)
        pooled = torch.cat([zeros, pooled[:, :, :-1]], dim=2)

        return pooled.transpose(1, 2)                          # (B, T//stride, D)

    # ── Bias caching ────────────────────────────────────────────────────────

    def _get_token_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return _window_mask(T, self.window_size, device, dtype)

    def _get_sent_bias(
        self, T_q: int, T_k: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Decay bias in sentence-segment space.

        query position i attends to segment j → temporal distance in sentence units.
        """
        q_idx = torch.arange(T_q, device=device, dtype=dtype)       # token positions
        k_idx = torch.arange(T_k, device=device, dtype=dtype)        # segment indices

        # Token i belongs to segment i // stride
        q_seg = (q_idx // self.sentence_stride).unsqueeze(1)         # (T_q, 1)
        k_seg = k_idx.unsqueeze(0)                                    # (1, T_k)
        dist = (q_seg - k_seg).abs().float()
        bias = -torch.log1p(self.decay_alpha * dist)
        # Causal in segment space: future segments are masked
        future = k_seg > q_seg
        bias = bias.masked_fill(future, float("-inf"))
        return bias.unsqueeze(0).unsqueeze(0)                        # (1,1,T_q,T_k)

    def _get_block_bias(
        self, T_q: int, T_k: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Decay bias in block space."""
        q_idx = torch.arange(T_q, device=device, dtype=dtype)
        k_idx = torch.arange(T_k, device=device, dtype=dtype)
        q_blk = (q_idx // self.block_size).unsqueeze(1)
        k_blk = k_idx.unsqueeze(0)
        dist = (q_blk - k_blk).abs().float()
        bias = -torch.log1p(self.decay_alpha * dist)
        future = k_blk > q_blk
        bias = bias.masked_fill(future, float("-inf"))
        return bias.unsqueeze(0).unsqueeze(0)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                    # (B, T, D) in tangent space (BF16/F32)
        c: float | torch.Tensor = 1.0,      # Poincaré curvature scalar
    ) -> torch.Tensor:                      # (B, T, D)
        B, T, D = x.shape
        dev = x.device
        dtype = x.dtype

        # ── Scale 1: Token-level (local sliding window) ──────────────────
        x1 = self.norm_token(x)
        bias_tok = self._get_token_bias(T, dev, torch.float32).to(dtype)
        out1 = self.token_attn(x1, x1, bias_tok)               # (B, T, D)

        # ── Scale 2: Sentence-level (pooled segments) ────────────────────
        x2 = self.norm_sentence(x)
        x2_pooled = self._causal_pool(x2, self.sentence_stride)  # (B, T_s, D)
        T_s = x2_pooled.shape[1]
        if T_s == 0:
            out2 = torch.zeros_like(x)
        else:
            bias_sent = self._get_sent_bias(T, T_s, dev, torch.float32).to(dtype)
            out2 = self.sentence_attn(x2, x2_pooled, bias_sent)  # (B, T, D)

        # ── Scale 3: Block-level (pooled blocks) ─────────────────────────
        x3 = self.norm_block(x)
        x3_pooled = self._causal_pool(x3, self.block_size)       # (B, T_b, D)
        T_b = x3_pooled.shape[1]
        if T_b == 0:
            out3 = torch.zeros_like(x)
        else:
            bias_blk = self._get_block_bias(T, T_b, dev, torch.float32).to(dtype)
            out3 = self.block_attn(x3, x3_pooled, bias_blk)     # (B, T, D)

        # ── Blend with learned scale weights ─────────────────────────────
        w = self.log_scale_weights.softmax(0)                    # (3,) sums to 1
        out = w[0] * out1 + w[1] * out2 + w[2] * out3           # (B, T, D)

        # ── Curvature gate: amplify signal for high-curvature regime ─────
        # When c is a tensor (learnable curvature), use it directly so gradients
        # flow back to self.curvature via curv_gate.  For buffer / float c,
        # construct a fresh tensor as before (no grad needed).
        if isinstance(c, torch.Tensor):
            c_in = c.clamp(min=1e-4).reshape(1, 1).to(device=dev, dtype=torch.float32)
        else:
            c_in = torch.tensor([[float(c)]], device=dev, dtype=torch.float32)
        # curv_gate weights are in the model's dtype (BF16 when model is cast
        # to BF16); c_in is float32 for numerical stability, so cast to match.
        gate = self.curv_gate(c_in.to(self.curv_gate.weight.dtype)).sigmoid()  # (1, n_heads)
        # Reshape to (1, n_heads, 1, 1) then collapse to (1, 1, 1) mean
        curv_scale = gate.mean().to(dtype)
        out = out * curv_scale

        # ── Output projection ─────────────────────────────────────────────
        out = self.out_proj(out)                                  # (B, T, D)

        # Update visualisation state (detached)
        with torch.no_grad():
            sw = w.detach().cpu().tolist()
            self._last_expert_balance = sw   # [token_w, sent_w, block_w]

        return out

    # ── Properties for dashboard ────────────────────────────────────────────

    @property
    def scale_weights(self) -> list[float]:
        """Current scale blend weights (token, sentence, block)."""
        return self.log_scale_weights.detach().softmax(0).cpu().tolist()

    @property
    def context_efficiency(self) -> float:
        """Context Efficiency (CE): fraction of context weight in sentence+block scales."""
        w = self.log_scale_weights.detach().softmax(0)
        return (w[1] + w[2]).item()
