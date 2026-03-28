"""
INTEGRATION GUIDE — Phase A: Fluid Power Allocation
=====================================================

This file documents every change made to integrate FluidPowerAllocator into
the Aether 2 training system.  It is NOT executable — it is a reference for
understanding what was added and why.

Files modified
--------------
  fluid_power.py        — REWRITTEN  (was Omega PoC, now Aether 2 production)
  aether2_model.py      — MODIFIED   (return_final_embedding support)
  aether2_config.py     — MODIFIED   (3 new FPA fields + CLI flags)
  aether2_train.py      — MODIFIED   (allocator instantiation + loop integration)

Activation
----------
  FPA is disabled by default (fpa_enabled=False).  Enable with:

    python aether2_train.py --fpa
    python aether2_train.py --fpa --fpa-max-iters 2    # lighter: 3 passes total
    python aether2_train.py --fpa --fpa-ponder-weight 0.05  # force more early exits

  Or programmatically:
    cfg = Aether2Config()
    cfg.fpa_enabled = True
    # cfg.fpa_max_iters = 3       # default
    # cfg.fpa_ponder_weight = 0.01  # default

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  fluid_power.py  — REWRITTEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The original file targeted AetherOmegaModel (Omega architecture).  It was
rewritten to target Aether2Model using the embed_override re-entry mechanism.

Key design change: instead of running just GGR blocks independently, each
extra pass runs the FULL model (all 24 blocks) starting from the current
Poincaré ball state.  This is more correct (SSM + CSSC + GGR all refine)
and uses the existing embed_override infrastructure.

Classes:
  EntropyHaltingCriterion(max_iters, vocab_size)
      nn.Parameter: raw_thresh (max_iters,)   — sigmoid-scaled thresholds
      forward(entropy, iter_idx) → (halt_mask, halt_prob)

  PonderCostRegulariser()
      forward(active_fractions) → scalar ponder_cost ∈ [0, 1]

  FluidPowerAllocator(cfg: Aether2Config)
      nn.Parameters: criterion.raw_thresh — fpa_max_iters floats
      forward(model, input_ids) → (logits_out, ponder_cost, aux_loss)
        logits_out : (B, T, V)  — thought tokens stripped, soft-merged
        ponder_cost: scalar     — ACT cost; multiply by fpa_ponder_weight
        aux_loss   : scalar     — GGR load-balance (summed across all passes)

Re-entry mechanism:
  pass 0  → model(input_ids, return_final_embedding=True)
             → (logits, _, aux, x_ball)
             x_ball: (B, T+n_thought, D) — ball after all blocks + episodic

  pass i  → x_input_ball = x_ball[:, n_thought:, :].detach()
             x_override = log_map_zero(x_input_ball, c)
             → model(input_ids, embed_override=x_override, return_final_embedding=True)
             model re-applies exp_map + prepends fresh thought tokens + all blocks

  Carry is .detach() → truncated BPTT per pass → ~1.3 GiB extra VRAM per pass.

Soft merge (differentiable gate for learning):
  logits_out = halt_prob × logits_old + (1-halt_prob) × logits_new
  halt_prob ≈ 1.0 → halted token, keep old
  halt_prob ≈ 0.0 → active token, use new
  Gradient path: CE → logits_out → halt_prob → raw_thresh ✓


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.  aether2_model.py  — return_final_embedding parameter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Change: Aether2Model.forward() signature extended with one optional parameter.

BEFORE:
    def forward(
        self,
        input_ids: torch.Tensor,
        capture_hidden_indices: set[int] | None = None,
        embed_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:

AFTER:
    def forward(
        self,
        input_ids: torch.Tensor,
        capture_hidden_indices: set[int] | None = None,
        embed_override: torch.Tensor | None = None,
        return_final_embedding: bool = False,      # ← NEW
    ):
        ...
        x_ball = x  # capture before log_map exit (after episodic memory)
        ...
        if return_final_embedding:
            return logits, hidden_states, aux_loss, x_ball
        return logits, hidden_states, aux_loss     # ← unchanged for existing callers

Backward compatibility: all existing callers unpack 3 values and are unaffected.
The 4th return value (x_ball) is only appended when explicitly requested.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3.  aether2_config.py  — FPA config fields
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

New fields added to Aether2Config dataclass (defaults = off):

    fpa_enabled: bool = False       # master switch
    fpa_max_iters: int = 3          # extra passes (3 → 4 total, ~11.2 GiB peak)
    fpa_ponder_weight: float = 0.01 # ACT cost weight in total loss

New CLI flags added to add_cli_args():
    --fpa                    → sets fpa_enabled=True
    --fpa-max-iters INT      → fpa_max_iters
    --fpa-ponder-weight FLOAT → fpa_ponder_weight

summary() now shows:
    Fluid Power    : ON (max_iters=3, ponder_w=0.01)
  or:
    Fluid Power    : OFF (max_iters=3, ponder_w=0.01)

from_args() handles --fpa and numeric overrides.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4.  aether2_train.py  — training loop integration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4a. Import
    from fluid_power import FluidPowerAllocator

4b. Allocator instantiation (after model is built):
    allocator = None
    if cfg.fpa_enabled:
        allocator = FluidPowerAllocator(cfg).to(device=device, dtype=torch.bfloat16)

4c. Optimizer: allocator params added to no_decay group
    make_optimizer(model, cfg, logger, allocator=allocator)
    → internally: no_decay_params += list(allocator.parameters())
    This ensures raw_thresh gets gradient updates via AdamW.

4d. TrainingMetrics: 3 new fields
    ponder_cost: float = 0.0
    fpa_avg_iters: float = 1.0
    fpa_halt_pct: float = 0.0

4e. Accumulator: accum_ponder added alongside accum_ce / accum_aux

4f. Micro-batch forward (the core change):

    BEFORE:
        logits, _, aux_loss = model(input_ids)
        T_l = labels.shape[1]
        logits_aligned = logits[:, -T_l:, :]
        ...
        total_loss = ce_loss + ggr_aux

    AFTER:
        if allocator is not None:
            logits_aligned, ponder_cost, aux_loss = allocator(model, input_ids)
            # logits_aligned already (B, T, V) — thought tokens stripped by FPA
        else:
            logits, _, aux_loss = model(input_ids)
            T_l = labels.shape[1]
            logits_aligned = logits[:, -T_l:, :]
            ponder_cost = torch.tensor(0.0, device=device)
        ...
        ponder_reg = ponder_cost * cfg.fpa_ponder_weight
        total_loss = ce_loss + ggr_aux + ponder_reg

4g. Logging:
    Step XXXXXX | ... | FPA iters=2.34 ponder=0.0012    (only when fpa_enabled)

4h. Checkpoint save/load: allocator state_dict saved as allocator.safetensors
    load_checkpoint(..., allocator=allocator) restores it if file exists.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VRAM breakdown (fpa_enabled, micro_batch=4, T=512, grad-ckpt ON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Base training (no FPA):        ~7.3 GiB
  + 1 extra pass (fpa_max=1):    ~8.6 GiB
  + 2 extra passes (fpa_max=2):  ~9.9 GiB
  + 3 extra passes (fpa_max=3):  ~11.2 GiB  ← default
  Headroom on 16 GB RX 7800 XT:  ~4.8 GiB


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected training dynamics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Steps     0– 2k   All tokens use max passes.  thresholds ≈ 5.2 nats (init).
                     fpa_avg_iters ≈ 4.0,  fpa_halt_pct ≈ 0%

  Steps  2k–10k     Common tokens (the, is, (, )) start early-exiting.
                     fpa_avg_iters drops toward 2.5
                     fpa_halt_pct rises toward 40–60%

  Steps 10k–50k     Stable specialisation.
                     Math / code tokens: 3–4 passes
                     English prose:      1–2 passes
                     fpa_avg_iters ≈ 2.0,  fpa_halt_pct ≈ 70%

  Tuning guide:
    fpa_avg_iters stays at max throughout → ponder_weight too low → raise it
    fpa_avg_iters collapses to 1 immediately → ponder_weight too high → lower it


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Publication context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Target: "Entropy-Conditioned Adaptive Compute in Hyperbolic State Space Models"
  First paper combining:
    • ACT-style dynamic compute (Graves 2016)
    • Poincaré ball geometry (Ganeane et al. 2018)
    • SSM-based backbone (Mamba, Gu & Dao 2023)
    • Per-token halting with full model re-entry via curved space embedding carry
"""
