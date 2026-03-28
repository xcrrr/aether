"""GGR — Gated Gradient Routing.

Four-expert MoE FFN with entropy-based sparsity gating and gradient stability.

Architecture:
  1. Complexity probe: project x to small space, compute Shannon entropy H(x).
  2. Entropy-conditioned router: concat(x, H_emb) → logits per expert.
  3. Gate normalisation (LayerNorm on logits): prevents vanishing/exploding grads.
  4. Optional top-k sparse routing (default top-2) with jitter for stability.
  5. Four expert SwiGLU FFNs: Math / Code / Logic / General.
  6. Per-expert gradient gates (learnable per-channel scalars) for fine-grained
     gradient routing.
  7. Soft merge at output; load-balance auxiliary loss returned for the trainer.

ROCm rules: no torch.compile, no Triton kernels. BF16 forward pass with
float32 entropy computation (softmax domain safety).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether2_config import Aether2Config


# ─────────────────────────────────────────────────────────────────────────────
# Expert FFN: SwiGLU variant (capacity-neutral)
# ─────────────────────────────────────────────────────────────────────────────

class _ExpertFFN(nn.Module):
    """Single SwiGLU expert with per-channel gradient gate."""

    def __init__(self, d_model: int, hidden_dim: int, name: str = "Expert") -> None:
        super().__init__()
        self.name = name
        # Gate + up projection (fused SwiGLU)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.norm      = nn.LayerNorm(hidden_dim, elementwise_affine=True)

        # Per-channel gradient gate: initialised to 1 (no suppression)
        # sigmoid(gamma) ∈ (0,1) — allows the expert to selectively gate
        # gradient flow into its hidden dimension.
        self.grad_gate = nn.Parameter(torch.zeros(hidden_dim))  # init → sigmoid ≈ 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) ⊗ up
        g = F.silu(self.gate_proj(x))                    # (B, T, H_expert)
        u = self.up_proj(x)                               # (B, T, H_expert)
        h = g * u                                          # (B, T, H_expert)
        h = self.norm(h)                                   # stable intermediates

        # Apply per-channel gradient gate (in float32 for numerical stability)
        gate = self.grad_gate.sigmoid().to(h.dtype)       # (H_expert,)
        h = h * gate                                       # (B, T, H_expert)

        return self.down_proj(h)                           # (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# Sparsity Controller — entropy-based routing
# ─────────────────────────────────────────────────────────────────────────────

class _SparsityController(nn.Module):
    """Compute Shannon entropy of the input and return routing logits.

    Entropy is computed over a small linear projection (complexity probe),
    then embedded and concatenated with x to condition the router.
    """

    def __init__(self, d_model: int, probe_dim: int, n_experts: int) -> None:
        super().__init__()
        self.probe_dim = probe_dim
        # Complexity probe
        self.probe = nn.Linear(d_model, probe_dim, bias=False)
        # Entropy embedding: scalar → d_model
        self.entropy_embed = nn.Linear(1, d_model, bias=True)
        # Router: (2 × d_model) → n_experts
        self.router = nn.Linear(d_model * 2, n_experts, bias=False)
        # Gate normalisation — essential for deep path gradient stability
        self.gate_norm = nn.LayerNorm(n_experts, elementwise_affine=True)

        nn.init.zeros_(self.router.weight)       # neutral start
        nn.init.zeros_(self.entropy_embed.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (routing_logits, entropy).

        routing_logits: (B, T, n_experts)  — layer-normed, pre-softmax
        entropy:        (B, T)             — normalised Shannon entropy in [0,1]
        """
        # Compute entropy in float32 for numerical safety.
        # Run probe in its native weight dtype (BF16 when model is BF16) then
        # cast output to float32 before softmax/log to avoid precision loss.
        with torch.amp.autocast("cuda", enabled=False):
            probe_out = self.probe(x.to(self.probe.weight.dtype)).float()  # (B, T, probe_dim)
            probs = F.softmax(probe_out, dim=-1)
            H = -(probs * (probs + 1e-8).log()).sum(-1)   # (B, T), raw entropy
            H_norm = H / math.log(self.probe_dim)         # normalise to [0, 1]

        # Embed entropy scalar → d_model
        H_emb = self.entropy_embed(H_norm.unsqueeze(-1).to(x.dtype))  # (B, T, D)

        # Router input: concatenate x with entropy embedding
        router_in = torch.cat([x, H_emb], dim=-1)         # (B, T, 2D)
        logits = self.router(router_in)                    # (B, T, n_experts)

        # Gate normalisation — prevents vanishing/exploding routing gradients
        logits = self.gate_norm(logits)

        return logits, H_norm.detach()


# ─────────────────────────────────────────────────────────────────────────────
# GGR Main Module
# ─────────────────────────────────────────────────────────────────────────────

class GatedGradientRouter(nn.Module):
    """Gated Gradient Routing — 4-expert entropy-conditioned sparse MoE FFN.

    Experts: Math | Code | Logic | General (configurable via Aether2Config).

    The forward pass returns (output, aux_loss) where aux_loss is the
    Switch Transformer load-balance term. Trainers must add this to the
    total loss scaled by cfg.ggr_lb_weight.

    Parameters
    ----------
    cfg : Aether2Config
    """

    def __init__(self, cfg: Aether2Config) -> None:
        super().__init__()
        D = cfg.d_model
        H_e = cfg.ggr_expert_hidden
        N = cfg.ggr_n_experts
        names = cfg.ggr_expert_names

        self.n_experts = N
        self.top_k = cfg.ggr_top_k
        self.temperature = cfg.ggr_temperature
        self.jitter_eps = cfg.ggr_jitter_eps
        self.lb_weight = cfg.ggr_lb_weight
        self.expert_names = list(names)

        # Sparsity controller (entropy-based router)
        self.controller = _SparsityController(D, cfg.ggr_entropy_probe_dim, N)

        # Expert FFNs
        self.experts = nn.ModuleList([
            _ExpertFFN(D, H_e, name=names[i]) for i in range(N)
        ])

        # Output RMSNorm
        self.out_norm = nn.LayerNorm(D, elementwise_affine=True)

        # Visualisation state (updated each forward)
        self._last_gates: list[float] = [1.0 / N] * N
        self._last_entropy: float = 0.5

    # ── Sparse top-k routing ────────────────────────────────────────────────

    def _sparse_route(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply jitter + top-k selection + re-normalisation.

        Returns:
          gates_full : (B, T, N)  — sparse gates (zero for non-selected experts)
          top_indices: (B, T, k)  — selected expert indices
          gates_topk : (B, T, k)  — normalised weights for selected experts
        """
        if self.training and self.jitter_eps > 0:
            noise = torch.empty_like(logits).uniform_(-self.jitter_eps, self.jitter_eps)
            logits = logits + noise

        # Soft gates (used for load-balance loss gradient)
        gates_soft = F.softmax(logits / self.temperature, dim=-1)  # (B, T, N)

        if self.top_k >= self.n_experts or self.top_k <= 0:
            # Full soft routing: no sparsification
            return gates_soft, None, gates_soft

        # Top-k selection
        top_vals, top_idx = torch.topk(gates_soft, self.top_k, dim=-1)  # (B,T,k)
        top_vals_norm = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Reconstruct full sparse gate tensor
        gates_full = torch.zeros_like(gates_soft)
        gates_full.scatter_(-1, top_idx, top_vals_norm)

        return gates_full, top_idx, top_vals_norm

    # ── Load-balance auxiliary loss (Switch Transformer) ────────────────────

    def _load_balance_loss(self, gates_soft: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss penalising unbalanced expert utilisation.

        Loss = N · Σ_i (f_i · P_i)
        f_i = fraction of tokens for which expert i is in the top-k selection
        P_i = mean soft routing probability to expert i (differentiable)

        f_i is measured using the ACTUAL top-k dispatch (not argmax) so that
        an expert used as the 2nd-choice for every token is correctly counted
        as fully loaded, not invisible.
        """
        N = gates_soft.shape[-1]
        top_k = min(self.top_k, N) if self.top_k > 0 else N

        with torch.no_grad():
            if top_k >= N:
                # Full soft routing: every expert serves every token
                f = torch.ones(N, device=gates_soft.device) / N
            else:
                # f_i = fraction of tokens for which expert i is selected
                _, top_idx = gates_soft.topk(top_k, dim=-1)     # (B, T, k)
                # one_hot over last dim: (B, T, k, N) → sum over k → (B, T, N)
                indicator = F.one_hot(top_idx, N).float().sum(dim=-2).clamp(0, 1)
                f = indicator.mean(dim=(0, 1))                   # (N,)

        P = gates_soft.mean(dim=(0, 1))                          # (N,) differentiable
        return N * (f * P).sum()

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                   # (B, T, D) tangent-space input
        _c: float | torch.Tensor = 1.0,    # curvature (unused here, API compat)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        out     : (B, T, D) — merged expert output
        aux_loss: scalar    — load-balance loss (add to total_loss × lb_weight)
        """
        B, T, D = x.shape

        # ── 1. Entropy-based routing ────────────────────────────────────
        logits, entropy = self.controller(x)                      # (B,T,N), (B,T)

        # Soft gates for load-balance gradient (before sparsification)
        gates_soft = F.softmax(logits / self.temperature, dim=-1) # (B, T, N)
        aux_loss = self._load_balance_loss(gates_soft)

        # Sparse gates for forward computation
        gates_full, _, _ = self._sparse_route(logits)            # (B, T, N)

        # ── 2. Expert outputs ────────────────────────────────────────────
        # Stack all expert outputs: (B, T, D, N)
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # (B, T, D, N)

        # ── 3. Soft merge ────────────────────────────────────────────────
        # gates_full: (B, T, N) → unsqueeze → (B, T, 1, N)
        out = (expert_outs * gates_full.unsqueeze(-2)).sum(dim=-1)  # (B, T, D)
        out = self.out_norm(out)

        # ── 4. Update visualisation state ────────────────────────────────
        with torch.no_grad():
            self._last_gates = gates_soft.mean(dim=(0, 1)).detach().cpu().tolist()
            self._last_entropy = entropy.mean().item()

        return out, aux_loss

    # ── Properties for dashboard ────────────────────────────────────────────

    @property
    def expert_load(self) -> list[float]:
        """Current mean gate weight per expert (for real-time visualisation)."""
        return self._last_gates

    @property
    def current_entropy(self) -> float:
        """Current mean normalised input entropy (complexity signal)."""
        return self._last_entropy

    @property
    def gradient_stability_index(self) -> float:
        """Gradient Stability Index (GSI): measures how balanced routing is.

        GSI = 1 - std(loads) / mean(loads)
        1.0 = perfectly balanced; 0.0 = fully collapsed to one expert.
        """
        loads = torch.tensor(self._last_gates, dtype=torch.float32)
        if loads.mean().item() < 1e-8:
            return 0.0
        cv = loads.std() / (loads.mean() + 1e-8)
        return float((1.0 - cv).clamp(0, 1).item())
