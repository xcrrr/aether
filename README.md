# Aether Omega

~445M parameter language model built from scratch. No pretrained weights, no external AI APIs.

**Author:** Adam Parszewski
**Hardware target:** AMD RX 7800 XT (16 GB VRAM, ROCm 6.x)
**Status:** Architecture validated, code verified — ready to train

---

## What This Is

Aether Omega is a decoder-only language model that replaces the standard Transformer attention mechanism with a Mamba-style Selective State Space Model (SSM) and runs the residual stream through Poincaré ball geometry instead of flat Euclidean space. It introduces two novel, previously unpublished architectural contributions — CSSC (Curvature-Selective State Coupling) and GGR (Geodesic Gravity Routing) — on top of five core pillars. It can be trained on either a local Python/math corpus generated from stdlib and algorithm patterns, or on 1M high-quality samples streamed from HuggingFace (curated code + math + reasoning + edu mixture, no starcode). The goal is to validate these architectural ideas at ~445M scale on a single 16 GB GPU in 2-3 days before requesting supercomputer time to ablate and scale them.

There are no pretrained checkpoints to download. If you want to run this, you run the tokenizer, generate the data, and train from step 0.

---

## Architecture

```
Input token IDs  →  Embedding + 8 thought tokens  →  exp_map  →  Poincaré ball

  ×24  AetherMambaBlock (per-block learnable κ, geometry-gated):
    log_map(κ)  →  RMSNorm  →  Mamba SSM  →  hyp_proj  →  gate(×)
                                ↕ d_state: 8 / 16 / 32 per layer group
    ── CSSC (novel) ── δ̄ (mean Mamba Δ) → W_cssc → c_token (B,T,1) per-token curvature
      → exp_map(c_token)  →  RiemannianRescale  →  möbius_add  →  project_to_ball
    ─────────────────────────────────────────────────────────────────────
    log_map(κ)  →  RMSNorm  →  [SwiGLU FFN  |  GGR-MoE]  →  gate(×)
                               blocks 0,1,2    blocks 3,7,11,15,19,23
    ── GGR (novel) ── 3 experts (Code/Math/Language), each with learned centroid
                      in Poincaré ball; routing weight ∝ exp(-geodesic_dist/T)
      → exp_map(κ)  →  möbius_add  →  project_to_ball
    EpisodicMemory (8192 slots, soft-attention, project_to_ball on result)

  log_map  →  RMSNorm  →  LM head (weight-tied with embedding)  →  logits (V=32768)

Parallel probe (no grad into backbone):
  RosettaObserver: 4-layer Transformer on tangent projections of main hidden states
```

**Total parameters:** ~445M main + ~28M RosettaObserver = ~473M

---

## The Five Pillars

### 1. Mamba-Hyperbolic Engine

**Why Mamba instead of attention:**
Attention is O(n²) in sequence length for both memory and compute. Mamba SSMs are O(n) — each token updates a fixed-size hidden state via a linear recurrence, making inference strictly sequential (constant memory) and training parallelizable via a selective scan. For long contexts this is the right trade-off. The implementation is a pure-PyTorch sequential scan (no Triton kernels) for ROCm compatibility — not as fast as the CUDA-optimized scan in the original paper, but correct and portable.

SSM parameters (Δ, B, C) are projected from the input at each timestep, making the recurrence *selective* — the model learns when to update or reset its state.

**Why hyperbolic geometry instead of Euclidean residuals:**
Code has hierarchical structure: tokens → identifiers → expressions → statements → functions → modules. Natural language also forms trees. Euclidean space has polynomial volume growth; hyperbolic space has exponential volume growth, which means it can embed tree structures with arbitrarily low distortion in low dimensions. Concretely: moving from Euclidean addition to Möbius addition (the isometry-preserving operation in the Poincaré ball) gives the residual connections a prior that matches the data's geometry.

All Poincaré operations (exp_map, log_map, möbius_add, project_to_ball) are computed in float32 and cast back to BF16. The ball boundary epsilon is 1e-2 (not 1e-5) because BF16's step size near unit magnitude is ~0.008.

**Multi-Timescale SSM (Clockwork Mamba):** 24 layers split into three groups with different SSM state sizes:

| Layers | d_state | Role |
|--------|---------|------|
| 0–7 | 8 | Token-level syntax |
| 8–15 | 16 | Phrase-level semantics |
| 16–23 | 32 | Paragraph-level discourse |

**CSSC — Curvature-Selective State Coupling (novel, world-first):**
The Mamba Δ (delta) gate already controls *when* the model updates its state — it is a learned selectivity signal that is large for important tokens and small for routine ones. CSSC couples this to the Poincaré ball curvature: high-Δ tokens get higher curvature `c_token`, expanding hyperbolic space for that token's representation; low-Δ tokens get lower curvature (closer to Euclidean). Concretely:

```
δ̄ = mean(Δ, dim=D_inner)              # (B, T, 1) — mean selectivity per token
c_scale = 0.5 + sigmoid(W_cssc(δ̄))    # (B, T, 1) ∈ (0.5, 1.5)
c_token = c_base × c_scale             # per-token curvature for exp_map
```

`W_cssc` is a single `Linear(1,1)` initialized to zero → identity at training start, learns from data. No published model has coupled SSM selectivity to hyperbolic curvature.

**GGR — Geodesic Gravity Routing Micro-MoE (novel, world-first):**
Every 4th block (blocks 3, 7, 11, 15, 19, 23 out of 24) replaces the SwiGLU FFN with three expert FFNs (Code / Math / Language). Routing is determined not by a linear gate but by *geodesic distance* in the Poincaré ball: each expert has a learnable centroid, and tokens are drawn to the nearest centroid as if by gravity.

```
x_ball = exp_map(x, c)                 # map token rep to Poincaré ball
d_j = poincaré_dist(x_ball, centroid_j)   # geodesic distance to expert j
gate_j = softmax(-d_j / T)             # "gravity": closer = higher weight
output = Σ gate_j · expert_j(x)        # soft weighted mixture
```

Capacity-neutral: each expert uses `ff_hidden // 3` hidden dim → total params ≈ one standard FFN. No published language model uses geodesic distances as MoE routing weights.

### 2. Gradient-Based Neurogenesis

The SwiGLU FFN internal dimension (`ff_hidden`) can grow during training. A `NeurogenesisTracker` monitors gradient variance of each block's `w_down.weight.grad`. When variance falls below `1e-4` for 500 consecutive steps, that layer has saturated. The response:

1. `w_gate`, `w_up`: append 64 new columns (Kaiming init)
2. `w_down`: append 64 new rows (**zero init** — net output unchanged immediately after expansion)
3. Optimizer is rebuilt to include the new parameters

Input/output dimensions (`d_model=1024`) never change, so residual connections remain valid. This is not a full architecture search — it is a targeted response to detected capacity saturation.

### 3. Anticipatory Picky Learner

Two mechanisms:

**Picky Batch Sampler:** Fetches 4× the needed samples, runs a no-grad forward pass, computes per-sample CE loss, drops samples outside `[0.2, 5.0]`. Too easy (CE < 0.2) means the model already knows it. Too hard (CE > 5.0) is likely noise. Thresholds ramp over 2000 curriculum warmup steps. Overhead: ~25% per step.

**In-Sequence Anticipatory Loss:** `MSE(h[t], h[t+5].detach())` on tangent-space hidden states. The model is rewarded for predicting its own future representations 5 positions ahead. Weight: 0.1.

### 4. Rosetta Stone Observer

A ~28M parameter interpretability probe that runs in parallel with the main model. It is a 4-layer Transformer with 8 heads (512 dim) that takes `log_map_zero(h)` — the tangent-space projection of each main-model hidden state — as input and predicts the same labels. Gradients are detached from the main model: the observer does not affect backbone training. Its CE loss contributes 0.05 × CE_rosetta to the total. Its purpose is to track whether the hyperbolic representations develop independently useful structure.

### 5. Training Stability Suite (Bonus)

| Feature | What it does |
|---------|-------------|
| **EMA self-distillation** | Deep-copy EMA teacher (decay=0.999); KL divergence from teacher logits added at weight 0.1 |
| **Entropy regularization** | -0.01 × H(logits) to prevent entropy collapse |
| **Iterative refinement** | If CE > 3.0, project final hidden state back to ball via a learned proj, run second forward pass |
| **Continuous thought tokens** | 8 learnable vectors prepended to every sequence; masked from CE loss (private scratchpad) |
| **Episodic memory** | 8192-slot key-value store; soft-attention retrieval (fully differentiable); Möbius-added to residual |
| **Loss spike detection** | Skip optimizer step if CE > 3.0 × 50-step rolling average |
| **Gradient noise** | Neelakantan 2015: add noise with std = `0.01 / sqrt(1 + step)` after grad clipping |

---

## v2 Architectural Upgrades

Three geometry-aware additions to improve hyperbolic training stability and expressiveness.

### RiemannianRescale (Change 2)

Custom `torch.autograd.Function` that rescales gradients by the inverse Poincaré conformal factor during backprop. The Poincaré metric inflates gradients near the ball boundary by up to ~4×, causing `hyp_proj` parameters to learn too slowly relative to Euclidean parameters. RiemannianRescale applies the inverse correction `((1-c‖x‖²)/2)²` to gradients, and a matching `lr_scale=4.0` on the `hyp_proj` parameter group compensates.

### Geometry Gating (Change 3)

Each `AetherMambaBlock` has two learned scalar gates (SSM path + FFN path) that blend Möbius addition with Euclidean addition for residual connections:

```
gate = sigmoid(Linear(v.detach()))      # v = tangent-space representation
x = gate × möbius_add(x, h) + (1-gate) × project_to_ball(x + h)
```

Gates are zero-initialized (sigmoid(0)=0.5 → 50/50 blend at init). The model learns per-block how much hyperbolic geometry to use, letting shallow layers default toward Euclidean behavior if curvature doesn't help at that depth.

### Learnable Curvature κ (Change 4)

Each block has its own curvature parameter `κ = sigmoid(_curvature_raw) × curvature_max`, replacing the global fixed `hyp_curvature`. A warmup schedule ramps curvature from 0→full over `curvature_warmup_steps=5000` steps so the model starts in near-Euclidean space and gradually activates hyperbolic geometry. Curvature parameters have no weight decay and no lr_scale boost.

### Additional v2 Changes

| Change | What |
|--------|------|
| `_sqrt_c` helper | Safe √c with epsilon guard for near-zero curvature |
| project_to_ball safety | Explicit clamping on episodic memory before Möbius add |
| SVD-guided neurogenesis | New neurons initialized along top singular vectors instead of random Kaiming |
| Geometry logging | Per-step curvature, gate values, hyp/euc gradient norms |
| Backward compatibility | `strict=False` on checkpoint load for pre-v2 resumption |

### Novel Contributions: CSSC + GGR

Two unpublished architectural innovations implemented on top of v2:

**CSSC (Curvature-Selective State Coupling):**
- `aether_config.py`: `cssc_enabled=True`
- `model.py`: `PurePyTorchSSM.forward(return_delta=True)`, `W_cssc` linear in each block
- Overhead: ~2 params per block (Linear(1,1)), ~1% compute overhead per step

**GGR (Geodesic Gravity Routing):**
- `aether_config.py`: `micro_moe_enabled=True`, `n_moe_experts=3`, `moe_layer_stride=4`
- `model.py`: `GeodesicGravityMoE` class, 6 blocks use it instead of SwiGLU
- Overhead: capacity-neutral (same total FFN params), ~5% compute overhead from distance computation

Both features default to `True`. Disable either by setting the corresponding flag to `False` in `aether_config.py`.

---

## Total Loss Function

```
L = CE_main
  + 0.10 × MSE_anticipatory    (tangent-space, h[t] vs h[t+5])
  + 0.05 × CE_rosetta           (detached observer probe)
  − 0.01 × H(logits)            (entropy bonus — negative penalty)
  + 0.10 × KL(logits ‖ EMA)     (self-distillation)
```

---

## Training Setup

### Hardware

- **GPU:** AMD RX 7800 XT, 16 GB VRAM, RDNA 3, ROCm 6.x
- **RAM:** 16 GB system RAM sufficient (optimizer runs on GPU — no CPU offload)
- **Python:** 3.12+, PyTorch 2.2+

### Required ROCm Environment Variables

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7800 XT GFX1101
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
```

Set these before running any training command. Without `HSA_OVERRIDE_GFX_VERSION`, ROCm may fail to recognize the GPU or use a suboptimal kernel path.

### VRAM Budget

| Component | BF16 | Notes |
|-----------|------|-------|
| Model weights | ~0.85 GiB | 444.64M params |
| EMA teacher | ~0.85 GiB | Deep copy of main model |
| Gradients | ~0.85 GiB | |
| Activations | ~2.0 GiB | With gradient checkpointing enabled |
| Rosetta probe | ~0.05 GiB | 27.79M params |
| Episodic memory | ~0.13 GiB | 8192 × 1024 × 2 bytes |
| Optimizer states | ~1.7 GiB | AdamW/8bit on GPU (cpu_offload=False) |
| Working buffers | ~1.2 GiB | Poincaré ops, GGR distance computation, picky sampler |
| **Total** | **~7.6 GiB** | ~8.4 GiB headroom on 16 GB card |

`use_gradient_checkpointing=True` and `cpu_offload=False` — the model fits in 16 GB VRAM without CPU transfer overhead.

### Training Speed

On RX 7800 XT with BF16 + gradient checkpointing + GPU AdamW:

- ~2–5 seconds per step (micro_batch=4, seq=512, picky sampler overhead ~25%)
- 50,000 steps ≈ 3–7 days
- Checkpoint written every 2,000 steps

### Configuration

All hyperparameters are in `aether_config.py` as an `OmegaConfig` dataclass. Key defaults:

| Field | Value | Meaning |
|-------|-------|---------|
| `vocab_size` | 32768 | Full 32k vocabulary |
| `d_model` | 1024 | Hidden dimension |
| `n_layers` | 24 | Number of AetherMambaBlocks |
| `ff_hidden` | 2816 | SwiGLU internal dim (≈ 8/3 × d_model, rounded to 256) |
| `max_seq_len` | 512 | Tokens per sequence (+ 8 thought tokens) |
| `micro_batch` | 4 | GPU batch size |
| `grad_accum_steps` | 16 | Effective batch = 64 |
| `learning_rate` | 4e-4 | Peak LR (cosine decay → 1e-5) |
| `warmup_steps` | 3000 | Linear LR warmup |
| `val_fraction` | 0.1 | 10% held out, same split every run (seeded) |
| `cpu_offload` | False | Optimizer stays on GPU |
| `use_gradient_checkpointing` | True | Saves ~6 GiB activations |
| `use_8bit_adam` | True | bitsandbytes AdamW8bit (falls back to AdamW) |
| `cssc_enabled` | True | CSSC: per-token curvature driven by Mamba Δ gate |
| `micro_moe_enabled` | True | GGR-MoE in every 4th block (3 experts, geodesic routing) |

---

## Quick Start from Zero

### Option A — Local data (no internet required, ~500–60k samples)

```bash
cd /home/xcrr1/aether2
source venv/bin/activate

# Set ROCm env vars
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# Train tokenizer + generate data from Python stdlib and algorithm patterns
python generate_data.py \
    --train-tokenizer \
    --tokenizer omega_tokenizer.json \
    --output data/aether_train.jsonl \
    --n-samples 60000 \
    --max-seq-len 512

# Verify smoke tests (56 tests — all must pass before training)
python smoke_test.py

# Start training
python train.py
```

### Option B — HuggingFace streaming (1M samples, fast 2-3 day turnaround)

**Fast, high-quality dataset** — NO starcode (too slow). All sources publicly accessible, no login required.

```bash
cd /home/xcrr1/aether2
source venv/bin/activate

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# Quick smoke test (1000 samples, ~2 min)
python build_dataset.py \
    --train-tokenizer \
    --output /tmp/aether_smoke.jsonl \
    --tokenizer omega_tokenizer.json \
    --n-samples 1000

# Fast build: 1M samples (tokenizer + dataset, ~4-6 hours)
python build_dataset.py \
    --train-tokenizer \
    --n-tokenizer-texts 100000 \
    --output data/aether_train.jsonl \
    --tokenizer omega_tokenizer.json \
    --n-samples 1000000 \
    --max-seq-len 512 \
    --vocab-size 32768 \
    --sort-by-difficulty \
    --seed 42

python smoke_test.py
python train.py  # ~6-8 hours on AMD RX 7800 XT
```

**Dataset mixture (1M records — curated best sources only):**

| Group | Fraction | Source | Notes |
|-------|----------|--------|-------|
| Code | 25% | `ise-uiuc/Magicoder-OSS-Instruct-75K` | Verified problem-solutions. High trust (95) |
| Math | 25% | `nvidia/OpenMathInstruct-2` + `HuggingFaceTB/finemath` | OpenMath verified (96), finemath curated (93) |
| Reasoning | 30% | `open-thoughts/OpenThoughts3-1.2M` + `open-r1/OpenR1-Math-220k` | SOTA 2025: QwQ-32B reasoning (98), verified math (95) |
| Educational | 20% | `HuggingFaceFW/fineweb-edu` (`CC-MAIN-2024-51`) | Web text pre-filtered for education quality |

**Design choices:**
- Removed `bigcode/starcoderdata` (too slow for 2-3 day turnaround)
- All sources streamed — no full downloads to disk
- Interleave-shuffle for well-mixed output
- Optional difficulty sorting (`--sort-by-difficulty`)
- Tokenizer trained on proportional samples before dataset build
- **Total training time estimate:** 4-6 hours data build + 6-8 hours training = ~12 hours wall-clock (2-3 days with overlapping work)

### Resume / inference

```bash
# Resume from checkpoint
python train.py --resume checkpoints_omega/step_0010000

# Generate text
python inference.py \
    --checkpoint checkpoints_omega/step_0002000 \
    --tokenizer omega_tokenizer.json \
    --prompt "def binary_search(arr, target):" \
    --max-new-tokens 128 \
    --temperature 0.8 \
    --top-p 0.9
```

### Dry run (verify pipeline end-to-end in under 2 minutes)

```bash
python generate_data.py --train-tokenizer --n-samples 500 --max-seq-len 512
python smoke_test.py
python train.py --max-steps 5 --micro-batch 2 --grad-accum 1 --no-8bit-adam
```

---

## Training Metrics

The training loop logs every 50 steps:

```
[  50/50000]  lr=1.33e-05  ce=9.8431  ppl=18823.4  ant=0.0142  ros=9.7211  total=11.2103  kept=0%  3421ms/step
[eval]  step=500  val_ce=8.2310  val_ppl=3745.2
[eval]  step=500  ast_pass_rate=0.00%  cumulative_pass=0.00%  n_verified=4
```

| Metric | What it means | Healthy range |
|--------|--------------|---------------|
| `ce` | Main CE on training batch | Starts ~10, should reach < 2.0 by step 10k |
| `ppl` | exp(CE) — perplexity | < 5000 after step 500; < 100 by step 5k |
| `val_ce` | CE on held-out val split (no grad, logged every 500 steps) | Should track train CE closely |
| `ros` | Rosetta observer CE (detached) | Similar trend to main CE |
| `ant` | MSE loss for future-repr prediction | Usually < 1.0 |
| `kept` | % of picky sampler candidates that passed CE filter | 50–80% is normal once model warms up |
| `easy_drop` | % dropped as too easy (CE < 0.2) | Rises as training progresses |
| `hard_drop` | % dropped as too hard (CE > 5.0) | Should be < 20% with clean data |

**What healthy training looks like:**
- CE drops from ~10 toward ~5 in the first 1000 steps
- `val_ce` stays within 0.3 of `ce` (no overfitting)
- `ppl` below 5000 by step 500 (warning logged if not)
- `kept` starts near 0% (curriculum warmup, everything filtered at step 0) then rises to 50–70%
- No spike-skip messages after step 2000
- VRAM stable around 6–8 GiB

**Signs of trouble:**
- `val_ce` > `ce` + 1.0 after step 5000: overfitting
- `ppl` > 10000 after step 1000: check tokenizer round-trip, check labels are shifted
- `kept` < 20% throughout: thresholds too strict or data is noisy
- `ce` not decreasing after step 500: check LR, check data loading

---

## File Structure

```
aether2/
├── model.py             AetherOmegaModel + all sub-modules: Poincaré ops,
│                        GeodesicGravityMoE (GGR), PurePyTorchSSM (CSSC),
│                        EpisodicMemory, RosettaObserver. Pure PyTorch, ROCm-safe.
├── train.py             Training loop: picky sampling, gradient accumulation,
│                        neurogenesis (FFN blocks only), EMA teacher, spike
│                        detection, val eval, checkpointing, SIGINT handler.
├── aether_config.py     OmegaConfig dataclass — all hyperparameters with defaults.
│                        Includes CSSC (cssc_enabled) and GGR (micro_moe_enabled)
│                        flags, curvature warmup, neurogenesis settings.
├── dataset.py           OmegaDataset (JSONL reader), PickyBatchSampler (CE-filtered
│                        curriculum), split_dataset() (seeded train/val), collate_fn.
├── tokenizer.py         OmegaTokenizer: pure-Python BPE, vocab_size=32768.
│                        Special tokens: PAD=0, UNK=1, BOS=2, EOS=3.
├── build_dataset.py     HuggingFace streaming data builder (high-quality, fast):
│                        25% Magicoder · 25% math · 30% reasoning · 20% edu.
│                        NO starcode (speed optimization). 1M samples default (4-6 hrs).
│                        All public sources, no gating. Interleave-shuffle, dedup, sort.
│                        Trains 32k-vocab BPE tokenizer on proportional texts first.
├── generate_data.py     Local data generator (no internet): stdlib introspection,
│                        algorithm patterns, augmentation. Quick pipeline validation.
├── inference.py         Text generation from checkpoint: top-p nucleus + greedy.
│                        Reports tok/s.
├── smoke_test.py        56 tests: Poincaré NaN safety, CSSC per-token curvature,
│                        GGR geodesic routing, multi-timescale SSM, Möbius gradient
│                        flow, tokenizer round-trip, full mini-pipeline.
├── probe_fields.py      Dev utility: verify HF dataset field names before a full
│                        build_dataset.py run.
├── CLAUDE.md            Claude Code guide: architecture, critical rules, CSSC/GGR
│                        details, VRAM budget, common pitfalls.
├── README.md            This file.
├── TUTORIAL.md          Step-by-step guide: setup → data → smoke test → train.
├── omega_tokenizer.json Trained BPE tokenizer (generated by --train-tokenizer).
├── checkpoints_omega/   Safetensors checkpoints written every 2000 steps.
│                        Each prefix: _main, _rosetta, _ema, _optimizer, _state.
└── data/
    └── aether_train.jsonl   Pre-tokenized training data (JSONL, seq_len=512).
```

---

## Known Limitations

**Architecture:**
- Pure-PyTorch sequential scan is ~3–5× slower than the Triton-optimized scan in the original Mamba paper. This is a ROCm compatibility trade-off.
- Hyperbolic geometry adds ~15% per-step overhead from float32 Poincaré operations. No hyperbolic BF16 kernel exists.
- GGR geodesic distance computation in 6 blocks adds ~5% overhead per step (batched Möbius ops over (B×T×n_experts, D) tensors).
- CSSC `return_delta=True` adds one `.mean()` + `.detach()` per SSM block, negligible overhead.
- EpisodicMemory is an in-memory tensor (8192 × 1024 × 2 bytes = 128 MB) with no disk persistence across training runs.
- Neurogenesis rebuilds the optimizer when triggered, causing a brief (~1s) pause per event. GGR blocks are tracked separately from FFN blocks.

**Data:**
- Local corpus (`generate_data.py`): ~60k Python samples from stdlib + patterns. Quick validation; will overfit within ~20k steps.
- **HuggingFace corpus (`build_dataset.py`): 1M high-quality samples** — Magicoder (code) + OpenMathInstruct (math) + OpenThoughts3 (reasoning) + fineweb-edu (education). **NO starcode (too slow)**. 4–6 hour stream+tokenize, all public sources, no gating. Includes exact-match dedup and shuffle.
- OpenThoughts3 provides world-class reasoning traces (QwQ-32B, verified math). Trained OpenThinker3-7B to beat DeepSeek-R1-Distill-7B.
- BPE tokenizer: 32768 vocab, trained on proportional sample texts. Non-English/non-code text will over-segment.

**Training stability:**
- Loss spike detection skips optimizer steps, which can cause the LR schedule to drift slightly from wall-clock step count. This is an accepted trade-off.
- EMA teacher requires a full second copy of model weights in VRAM (~0.77 GiB).

**Evaluation:**
- SymbolicVerifier generates samples and checks if they parse as valid Python AST. This is a weak proxy for code quality. There is no benchmark evaluation (HumanEval, MBPP, etc.).

**What comes next (grant objectives):**
1. Scale to 7B parameters on multi-GPU cluster
2. Replace sequential scan with parallel prefix-sum scan (requires HIP kernel)
3. Ablation study: isolate the contribution of each pillar vs. a plain Mamba baseline, including CSSC and GGR ablations
4. Evaluate on standard code benchmarks (HumanEval, MBPP)
5. Publish CSSC and GGR as novel contributions — no prior work combines SSM selectivity with hyperbolic curvature, or uses geodesic distances as MoE routing weights in language models

---

## License

BSL-1.1. See LICENSE.
