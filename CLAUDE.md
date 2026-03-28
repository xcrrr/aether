# Aether 2 — Claude Code Guide

## Project Identity

**Aether 2** is a 696M parameter language model built from scratch by Adam Parszewski.
No pretrained weights. No external AI APIs. Every component is original.
Built for an AGH University supercomputer grant.

- **Root:** `/home/xcrr1/aether2/`
- **Hardware:** AMD RX 7800 XT, 16 GB VRAM, ROCm 6.x
- **Python:** 3.12+, PyTorch 2.2+

## Architecture at a Glance

```
Aether2Model (696M params)
  Embedding (32768 × 1024) + ContinuousThoughtTokens (8 learnable)
  → exp_map → Poincaré ball (learnable curvature κ per block)

  ×24 Aether2Block:
    ├── SSM path:  log_map → RMSNorm → PurePyTorchSSM → hyp_proj → geom_gate
    │   └── CSSC v2: multi-head temporal attention at 3 scales (token / sentence / block)
    │       → hyperbolic decay bias → curvature gate → möbius_add → project_to_ball
    └── FFN path:  log_map → RMSNorm → [SwiGLU | GGR-MoE] → geom_gate
        └── GGR v2 (every 2nd block): 4 expert SwiGLUs (Math/Code/Logic/General)
            entropy probe → LayerNorm(logits) → top-2 sparse gates
            → Σ soft-merge → exp_map → möbius_add → project_to_ball

  EpisodicMemory (8192 slots, Möbius add)
  log_map → RMSNorm → LM head (weight-tied) → logits (V=32768)

ShadowModel (EMA baseline, ~4-layer vanilla transformer)
  Tracks main model weights via EMA; used for real-time Δ% comparison only.
```

## Key Files

| File | Purpose |
|------|---------|
| `aether2_train.py` | Main training loop — BF16, grad accum, FPA, baseline delta, checkpointing |
| `aether2_model.py` | `Aether2Block`, `Aether2Model`, `ShadowModel` |
| `aether2_config.py` | `Aether2Config` — all hyperparameters + CLI flags |
| `cssc.py` | CSSC v2 — Cross-Scale Spatiotemporal Correlation attention |
| `ggr.py` | GGR v2 — Gated Gradient Routing 4-expert MoE |
| `fluid_power.py` | Fluid Power Allocation — entropy-adaptive test-time compute |
| `streaming_data.py` | Circular-buffer streaming JSONL dataset |
| `dashboard.py` | Rich terminal live-training UI |
| `model.py` | Foundation layer — Poincaré ops, PurePyTorchSSM, RMSNorm, SwiGLU |
| `aether_config.py` | `OmegaConfig` — parent dataclass extended by `Aether2Config` |
| `build_dataset.py` | HuggingFace streaming dataset builder (10M samples, 4 groups) |
| `utils/tokenizer.py` | `OmegaTokenizer` — pure-Python BPE, vocab_size=32768 |
| `INTEGRATION_GUIDE.py` | Reference doc for Fluid Power Allocation integration |

## Novel Architectural Contributions

### CSSC v2 — Cross-Scale Spatiotemporal Correlation
- Multi-head temporal attention simultaneously at **token** (W=64), **sentence** (stride=32), and **block** (stride=256) scales
- Hyperbolic decay bias: `−log(1 + α·|Δt|)` enforces long-range consistency
- Three-scale outputs blended via learned softmax weights + curvature gate
- Uses `F.scaled_dot_product_attention` — ROCm-compatible, no Triton

### GGR v2 — Gated Gradient Routing
- 4 expert SwiGLUs: **Math · Code · Logic · General**, each `ff_hidden/4` hidden — capacity-neutral
- Entropy probe (Shannon H) conditions routing logits → high-entropy inputs trigger broader routing
- `LayerNorm` on routing logits prevents vanishing/exploding gradients through routing paths
- Top-2 sparse gates + load-balance aux loss (Switch Transformer style, weight=0.01)

### Fluid Power Allocation (FPA)
- Entropy-conditioned adaptive iteration at test time: uncertain tokens get up to 3 extra full-model passes
- Confident tokens early-exit via `EntropyHaltingCriterion`
- ACT regulariser (`ponder_weight=0.01`) controls early-exit rate
- Disabled by default; enable with `--fpa`

## Critical Rules

1. **NEVER `torch.compile`** — broken on ROCm
2. **NEVER `mamba_ssm`** — CUDA-only Triton kernels, incompatible with ROCm
3. **ALL Poincaré ops in float32** — BF16 `arctanh` overflows near boundary; cast back to BF16 after
4. **`project_to_ball` eps=1e-2** — BF16 step size ~0.008 needs larger margin than float32
5. **`arctanh` arg clamped < 1.0 - 1e-5** — domain boundary safety
6. **ROCm device string is `"cuda"`** — PyTorch maps HIP automatically; never use `"hip"`
7. **`v.detach()` as gate input** — essential for gradient checkpointing compatibility
8. **`strict=False` on checkpoint load** — backward compat with older checkpoints

## Config: Key Defaults

```python
d_model = 1024      n_layers = 24       ff_hidden = 2816
vocab_size = 32768  max_seq_len = 512   n_thought_tokens = 8
micro_batch = 4     grad_accum = 16     max_steps = 50_000
lr = 4e-4 → 1e-5   warmup = 3000       checkpoint_every = 2000
cssc_enabled = True    ggr_n_experts = 4    ggr_top_k = 2
fpa_enabled = False    cpu_offload_optimizer = False
```

## Environment

```bash
cd /home/xcrr1/aether2
source venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
```

## Development Workflow

```bash
# Verify setup — dry run (5 steps, minimal config)
python aether2_train.py --max-steps 5 --micro-batch 1 --grad-accum 1 --no-baseline

# Self-test foundation layer (Poincaré ops, SSM)
python model.py

# Full training (recommended)
nice -n 10 python aether2_train.py --checkpoint-every 500

# Full training with Rich dashboard
nice -n 10 python aether2_train.py --checkpoint-every 500 --dashboard

# Resume from checkpoint
python aether2_train.py --checkpoint-every 500 \
    --resume checkpoints_aether2/step_00002000

# Ablation — vanilla transformer (no CSSC, no GGR)
python aether2_train.py --no-cssc --no-ggr

# Enable Fluid Power Allocation (adaptive test-time compute)
python aether2_train.py --fpa

# If GPU < 12 GiB VRAM: enable CPU optimizer offload
python aether2_train.py --cpu-offload
```

## VRAM Budget (BF16, micro_batch=4, seq=512, gradient checkpointing ON)

`cpu_offload_optimizer=False` (default — Adam states stay on GPU):

| Component | GiB |
|-----------|-----|
| Model weights BF16 (696M) | ~1.4 |
| Gradients BF16 | ~1.4 |
| Adam m/v states FP32 | ~5.6 |
| Activations (grad ckpt) | ~1.5 |
| Shadow baseline model | ~0.5 |
| **Peak total** | **~10.4** |

Use `--cpu-offload` only on GPUs with < 12 GiB VRAM. It saves ~5.6 GiB on-device
but adds ~3–5× overhead (PCIe round-trip every optimizer step).

## Common Pitfalls

1. **Imports:** use `from aether2_model import Aether2Model` — not `from model import AetherOmegaModel`
2. **Curvature:** `block.curvature` is a `torch.Tensor` when `learnable_curvature=True`, `float` otherwise — handle both
3. **GGR blocks:** `block.ffn` is `None` for GGR blocks — always check before accessing
4. **CSSC + gradient checkpointing:** works because `delta_mean` is `.detach()`-ed before returning
5. **Checkpoints:** safetensors format only — never `torch.save` / `torch.load`
6. **`empty_cache()`:** do NOT call in the hot training loop — causes costly ROCm memory defrag per step

## Data Sources (`build_dataset.py`, 10M samples)

| Group | % | Main sources |
|-------|---|-------------|
| Code | 30% | starcoderdata (Python/JS), Magicoder-75K |
| Math | 25% | finemath-4plus, OpenMathInstruct-2 |
| Reasoning | 25% | OpenThoughts3-1.2M, OpenR1-Math-220k |
| Educational | 20% | fineweb-edu CC-MAIN-2024-51 |
