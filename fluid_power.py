"""Fluid Power Allocation — Phase A: Dynamic Test-Time Compute.

Entropy-conditioned adaptive iteration for Aether 2 (CSSC v2 + GGR v2).

Tokens whose output distribution has high Shannon entropy (model uncertain)
are re-routed through the full model for up to `fpa_max_iters` additional
passes.  Confident tokens (low entropy) early-exit, saving compute proportional
to their certainty.

Re-entry mechanism
------------------
Aether2Model.forward() now accepts return_final_embedding=True, which returns
x_ball — the Poincaré ball embedding after all blocks + episodic memory.

For each extra pass:
  1. Take x_ball[:, n_thought:, :] (input-token positions; strip thought prefix)
  2. log_map_zero(x_ball_input, c) → tangent space (embed_override expects tangent)
  3. model(input_ids, embed_override=x_tangent, return_final_embedding=True)
  4. Model re-applies exp_map → prepends fresh thought tokens → runs all 24 blocks
  5. The 6 GGR expert blocks see the refined Poincaré representation each pass

Why GGR is the natural re-routing target
-----------------------------------------
GGR v2 computes entropy-based routing logits and normalises through gate_norm.
Re-running GGR on the refined hidden state means: the entropy probe now sees
a "second-opinion" input that has already been processed by all blocks once.
The experts can then converge to a tighter specialisation.

VRAM budget (gradient-checkpointed, micro_batch=4, T=512)
----------------------------------------------------------
  Base pass      : ~7.3 GiB
  Per extra pass : ~1.3 GiB   (carry detached → truncated BPTT per pass)
  3 extra passes : ~11.2 GiB total  →  ~4.8 GiB headroom on 16 GB GPU

Loss contribution
-----------------
  L_total = CE_main
          + ggr_lb_weight × ggr_aux_loss
          + fpa_ponder_weight × ponder_cost

Training dynamics
-----------------
  Steps     0– 2k : thresholds at 0.5×log V ≈ 5.2 nats; all tokens iterate
  Steps  2k–10k   : easy tokens (common words, simple syntax) halt early
  Steps 10k–50k   : stable; math/code ~3-4 passes, prose ~1-2; avg ~2.0
  Tuning:
    avg stays at max  → raise fpa_ponder_weight (0.01 → 0.05)
    avg collapses to 1 → lower (0.01 → 0.005)

References
----------
  Graves (2016) "Adaptive Computation Time for RNNs", arXiv:1603.08983
  Original extension: per-token halting in Poincaré ball geometry + SSM.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether2_config import Aether2Config


# ─────────────────────────────────────────────────────────────────────────────
# EntropyHaltingCriterion
# ─────────────────────────────────────────────────────────────────────────────

class EntropyHaltingCriterion(nn.Module):
    """Learnable per-iteration entropy halt threshold.

    For extra iteration i, token (b, t) halts when:
        H( softmax(logits[b, t]) ) < sigmoid(raw_thresh[i]) × log(V)

    Initialised so each threshold = 0.5 × log(V) ≈ 5.2 nats.
    The thresholds learn jointly with the model: high ponder_weight pushes
    them lower (halt sooner); the task loss pushes them higher (iterate more).

    Parameters
    ----------
    max_iters  : number of extra iterations (= number of thresholds needed)
    vocab_size : vocabulary size V (sets entropy scale log V)
    """

    def __init__(self, max_iters: int, vocab_size: int) -> None:
        super().__init__()
        self.max_H = math.log(vocab_size)
        # sigmoid(0) = 0.5 → initial threshold = 0.5 × log(V) ≈ 5.2 nats
        self.raw_thresh = nn.Parameter(torch.zeros(max_iters))

    @property
    def thresholds(self) -> torch.Tensor:
        """Current effective thresholds in nats, shape (max_iters,)."""
        return torch.sigmoid(self.raw_thresh) * self.max_H

    def threshold_val(self, iter_idx: int) -> torch.Tensor:
        """Scalar threshold for iteration iter_idx — in graph for gradients."""
        return torch.sigmoid(self.raw_thresh[iter_idx]) * self.max_H

    def forward(
        self,
        entropy:  torch.Tensor,   # (B, T) Shannon entropy in nats
        iter_idx: int,             # which extra iteration (0-indexed)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute halt mask and differentiable halt probability.

        Returns
        -------
        halt_mask  : (B, T) bool — True = token halts here.  Detached.
        halt_prob  : (B, T) float — sigmoid soft approximation, in graph.
                     Used for differentiable logit merging and ponder cost.
        """
        t = self.threshold_val(iter_idx)           # scalar, in graph
        # Temperature = 0.2 (1/5) — sharper than default sigmoid
        halt_prob = torch.sigmoid((t - entropy) * 5.0)
        halt_mask = (entropy < t.detach()).detach()
        return halt_mask, halt_prob

    def mean_threshold_nats(self) -> float:
        """Mean threshold across all iterations in nats (diagnostic)."""
        with torch.no_grad():
            return self.thresholds.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# PonderCostRegulariser
# ─────────────────────────────────────────────────────────────────────────────

class PonderCostRegulariser(nn.Module):
    """ACT-style ponder cost — penalises tokens that use too many passes.

    ponder_cost = mean over extra iterations of (expected active fraction)
                = E[extra passes per token] / max_iters   ∈ [0, 1]

    Differentiable via halt_prob, so gradients flow to raw_thresh in
    EntropyHaltingCriterion.
    """

    def forward(self, active_fractions: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        active_fractions : one soft scalar per extra iteration.
            active_fractions[i] = mean of (1-halt_prob) × ~cumulative_halted
            over all (B, T) positions at extra iteration i.

        Returns
        -------
        ponder_cost : scalar in [0, 1]
        """
        if not active_fractions:
            return torch.tensor(0.0)
        return torch.stack(active_fractions).mean()


# ─────────────────────────────────────────────────────────────────────────────
# FluidPowerAllocator
# ─────────────────────────────────────────────────────────────────────────────

class FluidPowerAllocator(nn.Module):
    """Entropy-conditioned adaptive compute allocator for Aether 2.

    Does NOT wrap the model — instead, receives it at forward() time so the
    model and allocator can be managed and checkpointed independently.

    The allocator's own parameters (EntropyHaltingCriterion.raw_thresh) are
    a small set of scalars (fpa_max_iters floats) and must be included in the
    optimizer.  make_optimizer() already picks them up via model.named_parameters()
    as long as allocator is in the same module tree or its params are added to
    a separate param group — see INTEGRATION_GUIDE.py.

    Parameters
    ----------
    cfg : Aether2Config
    """

    def __init__(self, cfg: Aether2Config) -> None:
        super().__init__()
        self.max_iters   = cfg.fpa_max_iters
        self.criterion   = EntropyHaltingCriterion(cfg.fpa_max_iters, cfg.vocab_size)
        self.regulariser = PonderCostRegulariser()
        self.n_thought   = cfg.n_thought_tokens
        self.hyp_c       = cfg.hyp_curvature

        # Visualisation state — updated each forward, read by dashboard
        self._last_avg_iters: float = 1.0
        self._last_halt_pct:  float = 0.0
        self._last_entropy:   float = float("nan")

    # ── Properties for dashboard / logging ───────────────────────────────────

    @property
    def avg_iters(self) -> float:
        """Mean total passes (base + extra) per token, last forward."""
        return self._last_avg_iters

    @property
    def halt_pct(self) -> float:
        """Fraction of (B×T) tokens that halted before max_iters, last forward."""
        return self._last_halt_pct

    @property
    def mean_threshold_nats(self) -> float:
        """Mean entropy halting threshold across all iterations (nats)."""
        return self.criterion.mean_threshold_nats()

    # ── Core helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy in nats per token.  (B, T, V) → (B, T)."""
        log_p = F.log_softmax(logits.float(), dim=-1)
        return -(log_p.exp() * log_p).sum(dim=-1)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        model,                         # Aether2Model with return_final_embedding support
        input_ids: torch.Tensor,       # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adaptive forward pass.

        Returns
        -------
        logits_out  : (B, T, V)      — merged output logits (thought tokens stripped)
        ponder_cost : scalar tensor  — ACT regularisation term
        aux_loss    : scalar tensor  — GGR load-balance losses (all passes summed)
        """
        from model import log_map_zero

        B, T_in = input_ids.shape
        c  = self.hyp_c
        nt = self.n_thought

        # ── Pass 0: base full-model forward ─────────────────────────────
        logits, _hs, aux_loss, x_ball = model(
            input_ids, return_final_embedding=True
        )
        # logits : (B, T_in + nt, V)
        # x_ball : (B, T_in + nt, D)  — Poincaré ball after all blocks + episodic

        logits_out = logits[:, nt:, :]             # (B, T_in, V)
        entropy    = self._entropy(logits_out)     # (B, T_in) — nats

        # Cumulative halt: True = token will not be updated further
        halted = torch.zeros(B, T_in, dtype=torch.bool, device=input_ids.device)

        active_fractions: list[torch.Tensor] = []

        # ── Extra passes ─────────────────────────────────────────────────
        for i in range(self.max_iters):
            halt_mask, halt_prob = self.criterion(entropy, i)   # (B, T_in) each
            halted = halted | halt_mask

            active = ~halted                                     # (B, T_in) bool

            # Differentiable active fraction: (1-halt_prob) for non-halted tokens
            # Gradient flows: active_frac → halt_prob → raw_thresh[i]
            active_frac = ((1.0 - halt_prob) * active.float()).mean()
            active_fractions.append(active_frac)

            if not active.any():
                break   # All tokens halted — skip remaining passes

            # Re-entry: current ball state → tangent → embed_override
            # [:, nt:, :] = input-token positions (drop thought-token prefix)
            # detach carry: truncated BPTT — keeps extra-pass VRAM ≈ 1.3 GiB each
            x_input_ball = x_ball[:, nt:, :].detach()           # (B, T_in, D) in ball
            x_override   = log_map_zero(x_input_ball, c)         # (B, T_in, D) tangent

            # Full model re-entry: exp_map(x_override) → prepend thought tokens
            # → all 24 blocks (SSM + CSSC + 6× GGR) → episodic memory
            logits_new, _hs2, aux_new, x_ball_new = model(
                input_ids,
                embed_override=x_override,
                return_final_embedding=True,
            )
            logits_new_out = logits_new[:, nt:, :]               # (B, T_in, V)
            entropy_new    = self._entropy(logits_new_out)

            # Soft merge:  halt_prob ≈ 1 → keep old (halted), ≈ 0 → use new (active)
            # Gradient flows from logits_out through halt_prob to raw_thresh[i]
            hp = halt_prob.unsqueeze(-1)                          # (B, T_in, 1)
            logits_out = hp * logits_out + (1.0 - hp) * logits_new_out

            # Update entropy for still-active tokens (halted tokens keep old entropy)
            entropy = torch.where(halted, entropy, entropy_new)

            # Update ball embedding for active token positions only
            active_3d = active.unsqueeze(-1)                      # (B, T_in, 1)
            x_input_updated = torch.where(
                active_3d.expand_as(x_ball[:, nt:, :]),
                x_ball_new[:, nt:, :],
                x_ball[:, nt:, :],
            )
            x_ball = torch.cat([x_ball[:, :nt, :], x_input_updated], dim=1)

            aux_loss = aux_loss + aux_new

        # ── Ponder cost ───────────────────────────────────────────────────
        ponder_cost = self.regulariser(active_fractions)

        # ── Update visualisation state ────────────────────────────────────
        with torch.no_grad():
            extra_passes = float(len(active_fractions))
            avg_active = (
                sum(af.item() for af in active_fractions) / max(extra_passes, 1)
            )
            self._last_avg_iters = 1.0 + extra_passes * avg_active
            self._last_halt_pct  = float(halted.float().mean().item())
            self._last_entropy   = float(entropy.mean().item())

        return logits_out, ponder_cost, aux_loss
