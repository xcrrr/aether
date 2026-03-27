# Aether Omega — Quick Start (2-3 Days)

## One-Command Setup

```bash
cd /home/xcrr1/aether2
source venv/bin/activate

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
```

## Pipeline (Sequential, ~12 hours wall-clock)

### 1. Build Dataset + Tokenizer (4-6 hours)
```bash
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
```

**What this does:**
- Trains BPE tokenizer on 100K samples from HuggingFace
- Streams 1M high-quality samples (Magicoder + OpenMathInstruct + OpenThoughts + fineweb-edu)
- **NO starcode** (optimization for 2-3 day turnaround)
- Outputs: `data/aether_train.jsonl` (2-3 GB) + `omega_tokenizer.json`

### 2. Run Smoke Tests (5 minutes)
```bash
python smoke_test.py
```

Expected: **56/56 PASSED**
- Validates Poincaré geometry NaN-safety
- Verifies CSSC (per-token curvature coupling)
- Verifies GGR (geodesic routing)
- Gradient flow through all paths

### 3. Train (6-8 hours on AMD RX 7800 XT)
```bash
python train.py
```

**Default config:**
- micro_batch = 4
- grad_accum = 16 → effective_batch = 64
- max_steps = 50,000
- Learning rate: 4e-4 → 1e-5 (cosine annealing)
- Checkpoint every 2,000 steps → `checkpoints_omega/`

**During training, logs every 50 steps:**
```
[step 50] loss=10.23 | val_loss=10.18 | lr=4.0e-4 | dt=0.85s | tps=502 tok/s
```

---

## Fast Dry-Run (2 minutes, verify end-to-end)

```bash
python generate_data.py --train-tokenizer --n-samples 500 --max-seq-len 512
python smoke_test.py
python train.py --max-steps 5 --micro-batch 2 --grad-accum 1 --no-8bit-adam
```

---

## Resume Training

```bash
python train.py --resume checkpoints_omega/step_0010000
```

---

## Generate Text

```bash
python inference.py \
    --checkpoint checkpoints_omega/step_0025000 \
    --tokenizer omega_tokenizer.json \
    --prompt "def binary_search(arr, target):" \
    --max-new-tokens 128 \
    --temperature 0.8 \
    --top-p 0.9
```

---

## Dataset Breakdown

| Group | % | Source | Trust |
|-------|---|--------|-------|
| Code | 25% | Magicoder-75K | 95 |
| Math | 25% | OpenMathInstruct-2 + finemath | 96/93 |
| Reasoning | 30% | OpenThoughts3 + OpenR1-Math | 98/95 |
| Educational | 20% | fineweb-edu | 88 |

**Total:** ~1M samples = 512M tokens → 50k steps at batch 64

---

## Expected Results

After 50k steps (~8 hours):
- **Loss:** ~3.5-4.0 (untrained model starts ~10.5)
- **Validation loss:** Similar trajectory
- **Checkpoints:** 25 saved (every 2k steps)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `RuntimeError: CUDA out of memory` | Reduce `micro_batch` to 2, adjust `grad_accum` to 32 |
| `starcoderdata` taking too long | Already removed from `build_dataset.py` (use Magicoder only) |
| Tokenizer training slow | Reduce `--n-tokenizer-texts` to 50k |
| Network timeout during HF stream | Re-run `build_dataset.py` — partial checkpoints are saved |

---

## What's Next (After Training)

1. Analyze loss curves (compare CSSC/GGR contributions)
2. Run inference on code/math benchmarks
3. Scale to 7B on supercomputer (AGH grant)
4. Ablate CSSC and GGR individually
5. Publish novel contributions

---

**Made for:** Adam Parszewski @ AGH University
**Timeline:** 2-3 days total (dataset + training)
**Target:** Validate CSSC + GGR at ~445M scale before supercomputer request
