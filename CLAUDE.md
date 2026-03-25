# Aether Omega — Claude Code Guide

## Project Identity

**Aether Omega** is a ~445M parameter Mamba-Hyperbolic language model built from scratch by Adam Parszewski.
No pretrained weights. No external AI APIs. Every component is original. Built for an AGH University supercomputer grant.

- **Root:** `/home/xcrr1/aether2/`
- **Hardware:** AMD RX 7800 XT, 16 GB VRAM, ROCm 6.x
- **Python:** 3.12+, PyTorch 2.2+

## Architecture at a Glance

```
AetherOmegaModel (~445M params)
  Embedding (32768 × 1024) + ContinuousThoughtTokens (8 learnable)
  → exp_map → Poincaré ball (curvature c)

  ×24 AetherMambaBlock:
    ├── SSM path: log_map → RMSNorm → PurePyTorchSSM → hyp_proj → geom_gate
    │   └── CSSC: δ̄ (mean Δ) → W_cssc → c_token (B,T,1) per-token curvature
    │       → exp_map(c_token) → RiemannianRescale → möbius_add → project_to_ball
    └── FFN path: log_map → RMSNorm → [SwiGLU | GGR-MoE] → geom_gate
        └── GGR (every 4th block): 3 expert SwiGLUs + Poincaré centroids
            routing ∝ exp(-geodesic_dist_to_centroid / T)
            → exp_map → möbius_add → project_to_ball

  EpisodicMemory (8192 slots, soft-attention, Möbius add)
  log_map → RMSNorm → LM head (weight-tied with embedding) → logits (V=32768)

RosettaObserver (~28M params, detached probe)
  4-layer Transformer on log_map(main_hidden_states)
```

## Key Files

| File | Purpose |
|------|---------|
| `model.py` | All model modules: Poincaré ops, GeodesicGravityMoE, PurePyTorchSSM, AetherMambaBlock, AetherOmegaModel, RosettaObserver |
| `aether_config.py` | `OmegaConfig` dataclass — all hyperparameters, includes CSSC/GGR flags |
| `train.py` | Training loop: picky sampler, grad accum, EMA, neurogenesis, spike detection |
| `tokenizer.py` | `OmegaTokenizer`: pure-Python BPE, vocab_size=32768, PAD=0/UNK=1/BOS=2/EOS=3 |
| `dataset.py` | `OmegaDataset`, `PickyBatchSampler`, `collate_fn` |
| `generate_data.py` | Local data from Python stdlib (no internet) |
| `build_dataset.py` | HuggingFace streaming 10M dataset (9 sources, 4 groups) |
| `smoke_test.py` | 56 tests — all must pass before training |
| `inference.py` | Text generation from checkpoint |

## Novel Architectural Contributions

### CSSC — Curvature-Selective State Coupling (world-first)
- Config: `cssc_enabled: bool = True`
- The Mamba Δ gate (selectivity signal) modulates per-token Poincaré curvature
- `c_scale = 0.5 + sigmoid(W_cssc(delta_mean))` ∈ (0.5, 1.5)
- `c_token = c_base × c_scale` → used in `exp_map_zero(h_proj, c_token)`
- `W_cssc` zero-init → identity at start, learns from data
- Zero extra inference overhead beyond one Linear(1,1) per block

### GGR — Geodesic Gravity Routing Micro-MoE (world-first)
- Config: `micro_moe_enabled=True`, `n_moe_experts=3`, `moe_layer_stride=4`
- Blocks 3, 7, 11, 15, 19, 23 use `GeodesicGravityMoE` instead of SwiGLU FFN
- 3 expert SwiGLUs (Code/Math/Language), each `ff_hidden//3` hidden dim — capacity-neutral
- Routing by Poincaré geodesic distance to learned centroids (3 × d_model params)
- Soft routing (all experts run) — fully differentiable, no load-balancing loss needed

## Critical Rules

1. **NEVER `torch.compile`** — broken on ROCm
2. **NEVER `mamba_ssm`** — CUDA-only Triton kernels, incompatible with ROCm
3. **ALL Poincaré ops in float32** — BF16 arctanh overflows; cast back to BF16 after
4. **`project_to_ball` eps=1e-2** — BF16 step size near 1.0 is ~0.008, needs larger margin
5. **`arctanh` arg clamped < 1.0 - 1e-5** — domain boundary safety
6. **ROCm device string is `"cuda"`** — PyTorch maps it automatically; never use `"hip"`
7. **`c.item()` in RiemannianRescale** — intentional detach from learnable curvature
8. **`v.detach()` as gate input** — essential for gradient checkpointing compatibility
9. **`strict=False` on checkpoint load** — backward compat with older checkpoints
10. **NEVER weaken smoke_test.py** — fix code, not the test

## Config: Key Defaults

```python
d_model = 1024      n_layers = 24       ff_hidden = 2816
d_state varies: [8, 16, 32] by layer group (Clockwork Mamba)
vocab_size = 32768  max_seq_len = 512   n_thought_tokens = 8
micro_batch = 4     grad_accum = 16     max_steps = 50_000
lr = 4e-4 → 1e-5   warmup = 3000       checkpoint_every = 2000
cssc_enabled = True   micro_moe_enabled = True
n_moe_experts = 3     moe_layer_stride = 4
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
# Smoke tests (56 tests — run before any code change validation)
python smoke_test.py

# Self-test model architecture
python model.py

# Dry run (5 training steps)
python train.py --max-steps 5 --micro-batch 2 --grad-accum 1 --no-8bit-adam

# Full training
python train.py

# Resume
python train.py --resume checkpoints_omega/step_0010000
```

## VRAM Budget (BF16, micro_batch=4, seq=512, gradient checkpointing ON)

| Component | GiB |
|-----------|-----|
| Model weights (444.64M) | ~0.85 |
| EMA teacher copy | ~0.85 |
| Gradients | ~0.85 |
| Activations (grad ckpt) | ~2.0 |
| Rosetta (27.79M) | ~0.05 |
| Episodic memory | ~0.13 |
| Optimizer (8-bit Adam) | ~1.7 |
| Buffers (Poincaré, GGR) | ~1.2 |
| **Total** | **~7.6** |

## Common Pitfalls

1. **Import:** `from model import AetherOmegaModel` — not relative imports
2. **curvature:** `block.curvature` returns tensor when `learnable_curvature=True`, float when False — handle both
3. **GGR blocks:** `block.ffn` is `None` for GGR blocks — always check `block.ffn is not None` before accessing
4. **CSSC + gradient checkpointing:** works because delta_mean is `.detach()`-ed before returning
5. **Neurogenesis:** monitors only non-GGR blocks (FFN blocks). GGR experts are not expanded by neurogenesis.
6. **Checkpoints:** safetensors format only — never `torch.save`/`torch.load`

## Data Sources (build_dataset.py, 10M samples)

| Group | % | Main sources |
|-------|---|-------------|
| Code | 30% | starcoderdata (Python/JS), Magicoder-75K |
| Math | 25% | finemath-4plus, OpenMathInstruct-2 |
| Reasoning | 25% | OpenThoughts3-1.2M, OpenR1-Math-220k |
| Educational | 20% | fineweb-edu CC-MAIN-2024-51 |
