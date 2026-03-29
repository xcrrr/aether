# Aether 2

A next-generation language model architecture built from scratch on AMD ROCm.
No pretrained weights. No external AI APIs. Every component is original.

**Hardware target:** AMD RX 7800 XT · 16 GB VRAM · ROCm 6.x
**Python:** 3.12+ · PyTorch 2.10+ROCm · BF16 mixed precision

---

## Architecture

Aether 2 replaces the Omega SSM stack with two novel mechanisms layered on
top of the proven Poincaré ball residual stream from Aether Omega.

### CSSC — Cross-Scale Spatiotemporal Correlation

Multi-head temporal attention operating simultaneously at three scales:

| Scale | Mechanism | Receptive Field |
|-------|-----------|-----------------|
| Token-level | Causal sliding-window attention | W = 64 tokens |
| Sentence-level | Cross-attention to avg-pooled segments | stride = 32 |
| Block-level | Cross-attention to avg-pooled blocks | stride = 256 |

**Hyperbolic temporal decay** is applied as an additive log-bias to every
attention score matrix, enforcing long-range logical consistency:

```
decay_bias(Δt) = −log(1 + α · |Δt|)     α = 0.5 (default)
```

The three scale outputs are blended via learned softmax weights and modulated
by a curvature gate inherited from the Poincaré geometry layer.

All attention calls use `F.scaled_dot_product_attention` — dispatches to the
ROCm-compatible efficient SDPA path with no Triton kernels required.

### GGR — Gated Gradient Routing

Entropy-conditioned sparse MoE FFN with four expert paths:
**Math** · **Code** · **Logic** · **General**

```
Input x (B, T, D)
  │
  ├─▶ Complexity probe  ─▶ Shannon entropy H(x)
  │                            │
  ├─▶ Router (x ‖ H_emb) ─▶ LayerNorm(logits) ─▶ Top-2 sparse gates
  │
  ├─▶ Math FFN    (SwiGLU, D/4 hidden) ─▶ × gate₀
  ├─▶ Code FFN    (SwiGLU, D/4 hidden) ─▶ × gate₁
  ├─▶ Logic FFN   (SwiGLU, D/4 hidden) ─▶ × gate₂
  └─▶ General FFN (SwiGLU, D/4 hidden) ─▶ × gate₃
                       └──── Σ soft-merge ──▶ Output
```

Key properties:

- **Entropy-based sparsity**: input complexity (Shannon entropy of a linear
  probe) conditions routing logits — high-entropy inputs trigger broader routing.
- **Gradient stability**: `LayerNorm` on routing logits prevents
  vanishing/exploding gradients through deep routing paths.
- **Per-channel gradient gates**: each expert owns a learnable `(D/4,)` sigmoid
  vector that fine-tunes gradient flow into expert hidden dimensions.
- **Load-balance auxiliary loss**: Switch Transformer formulation prevents
  expert collapse; weight controlled by `ggr_lb_weight` (default `0.01`).

### Baseline Shadow Model

A 4-layer vanilla `Aether2Block` (CSSC and GGR disabled) runs alongside the
main model, EMA-updated from the first N layers. Used exclusively for real-time
Δ% comparison in the dashboard — adds zero optimiser state overhead.

---

## Repository Structure

```
aether2/
│
├── Core Production Modules
│   ├── aether2_config.py     Global config + CLI ablation flags
│   ├── cssc.py               Cross-Scale Spatiotemporal Correlation attention
│   ├── ggr.py                Gated Gradient Routing — 4-expert entropy MoE
│   ├── aether2_model.py      Aether2Block · Aether2Model · ShadowModel
│   ├── streaming_data.py     mmap circular-buffer streaming dataset
│   ├── aether2_train.py      Training loop: BF16, grad ckpt, baseline deltas
│   └── dashboard.py          Aether Command Deck — Rich live terminal UI
│
├── Foundational Layer  (direct imports by production modules)
│   ├── model.py              Poincaré ops · PurePyTorchSSM · SwiGLU · RMSNorm
│   └── aether_config.py      OmegaConfig parent dataclass
│
├── Data Pipeline
│   ├── build_dataset.py      HuggingFace 10M-sample streaming dataset builder
│   ├── data/
│   │   └── aether_train.jsonl   3.71 GB pre-tokenized training corpus
│   └── omega_tokenizer.json  Custom 32k BPE tokenizer
│
├── Utils  (standalone tools, not on the critical path)
│   └── utils/tokenizer.py      BPE tokenizer implementation
│
└── Runtime
    ├── venv/                 Python virtual environment
    ├── requirements.txt      Package list
    ├── logs/                 → logs/aether_build.log
    └── checkpoints_aether2/ Safetensors checkpoints (model + shadow + opt)
```

---

## Quick Start

```bash
cd aether2
source venv/bin/activate

# ROCm environment (required for AMD GPU)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
```

---

## Running the System

### Full Training — Aether Command Deck

Launches the live Rich terminal dashboard with the CSSC Horizon arc heatmap,
GGR Engine flow diagram, and real-time Δ% vs the shadow baseline:

```bash
python aether2_train.py --dashboard
```

### Ablation — Vanilla Transformer Baseline

Disables both CSSC and GGR. Runs a plain SwiGLU transformer to generate the
ground-truth comparison floor for the Δ% metric:

```bash
python aether2_train.py --no-cssc --no-ggr --dashboard
```

### Single-Component Ablations

```bash
# CSSC only (standard FFN, no expert routing)
python aether2_train.py --no-ggr --dashboard

# GGR only (windowed attention, no multi-scale temporal)
python aether2_train.py --no-cssc --dashboard
```

### Resume Training

```bash
python aether2_train.py --dashboard \
    --resume checkpoints_aether2/step_00002000
```

### Dry Run (no VRAM pressure)

```bash
python aether2_train.py \
    --max-steps 5 --micro-batch 1 --grad-accum 1 --no-baseline
```

### Dashboard Preview (no training)

```bash
python dashboard.py
```

### Inference from Checkpoint

```bash
python utils/inference.py \
    --checkpoint checkpoints_aether2/step_00010000 \
    --tokenizer omega_tokenizer.json \
    --prompt "def quicksort(arr):" \
    --max-new-tokens 256 \
    --temperature 0.8
```

---

## Configuration Reference

All hyperparameters live in `Aether2Config` (`aether2_config.py`).
Key defaults tuned for the RX 7800 XT 16 GB target:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 1024 | Hidden dimension |
| `n_layers` | 24 | Transformer depth |
| `max_seq_len` | 512 | Context window (tokens) |
| `cssc_n_heads` | 8 | CSSC attention heads |
| `cssc_window_size` | 64 | Token-level window width |
| `cssc_sentence_stride` | 32 | Sentence-level pool stride |
| `cssc_block_size` | 256 | Block-level pool stride |
| `cssc_decay_alpha` | 0.5 | Hyperbolic decay rate α |
| `ggr_n_experts` | 4 | Expert count |
| `ggr_top_k` | 2 | Sparse routing top-k |
| `ggr_lb_weight` | 0.01 | Load-balance aux loss weight |
| `micro_batch` | 4 | Per-step micro-batch size |
| `grad_accum_steps` | 16 | Effective batch = 64 sequences |
| `learning_rate` | 4e-4 | Peak LR (cosine decay → 1e-5) |
| `warmup_steps` | 3000 | Linear LR warmup |
| `max_steps` | 50 000 | Total optimizer steps |
| `cpu_offload_optimizer` | `False` | CPU offload for Adam states (OFF on 16 GiB) |
| `use_gradient_checkpointing` | `True` | Recompute activations — saves ~4 GiB VRAM |
| `use_bf16` | `True` | BF16 mixed precision |
| `stream_buffer_tokens` | 2M | Circular dataset buffer size |

Override any field by passing keyword arguments to `Aether2Config(field=value)`,
or via the CLI flags documented in `Aether2Config.add_cli_args`.

---

## VRAM Budget (RX 7800 XT 16 GiB, default config)

| Component | GiB |
|-----------|-----|
| Model weights BF16 (696M) | ~1.4 |
| Gradients BF16 | ~1.4 |
| Adam m/v states FP32 | ~5.6 |
| Activations (grad checkpointing ON) | ~1.5 |
| Shadow baseline model | ~0.5 |
| **Peak total** | **~10.4** |

The 5.6 GiB margin keeps ROCm's allocator from fragmenting under the
`PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512` setting.

If you are on a GPU with less than 12 GiB VRAM, pass `--cpu-offload` to move
Adam states to CPU RAM. This saves ~5.6 GiB on GPU at the cost of ~3–5×
lower throughput due to PCIe round-trips every optimizer step.

---

## Performance Notes

**Why the default changed (cpu_offload_optimizer = False)**

An earlier default had CPU optimizer offload enabled. On a 16 GiB card this
created two hidden bottlenecks on every optimizer step:

1. **PCIe round-trips** — ~5.6 GiB of Adam m/v tensors were moved CPU → GPU
   before the update and GPU → CPU after. At ~10 GB/s effective PCIe bandwidth
   that alone accounts for ~0.5–1 s of overhead per step.
2. **`torch.cuda.empty_cache()` on every step** — the call was needed to release
   the temporarily-pinned GPU state tensors, but on ROCm it triggers a full
   memory defragmentation + GPU sync (~200–500 ms per call).

With `cpu_offload_optimizer = False` (default) both overheads disappear and
throughput rises from ~425 tok/s to ~2 000–3 500 tok/s, cutting the full
50 000-step run from ~44 days to ~3–5 days on the same hardware.

**Expected training time at 50 000 steps**

| Config | ~tok/s | ~days to 50k steps |
|--------|--------|--------------------|
| cpu_offload ON (old default) | 425 | 44 |
| cpu_offload OFF (new default) | 2 000–3 500 | 3–5 |

---

## Metrics Glossary

| Metric | Definition |
|--------|-----------|
| **CE** | Cross-entropy loss — primary training objective |
| **PPL** | Perplexity = `exp(CE)` |
| **GSI** | Gradient Stability Index = `1 − std(expert_loads) / mean(expert_loads)` |
| **CE (context eff.)** | Fraction of CSSC blend weight on sentence + block scales |
| **Δ vs Baseline** | `(shadow_CE − aether2_CE) / shadow_CE × 100` — positive = Aether 2 is better |

---

## ROCm Constraints

| Rule | Reason |
|------|--------|
| Device string is always `"cuda"` | PyTorch maps HIP automatically |
| `torch.compile` is never called | ROCm support is fragile |
| `mamba_ssm` is not used | CUDA-only Triton kernels |
| All Poincaré ops cast to `float32` | BF16 `arctanh` overflows near boundary |
| Checkpoints use `safetensors` only | Never `torch.save`/`torch.load` |

---

## Author

**Adam Parszewski** — All architectural components are original research. No pretrained weights.
