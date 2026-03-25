# Aether Omega — Complete Tutorial

Step-by-step guide from zero to trained model. Covers environment setup, data generation, smoke testing, training, inference, and troubleshooting.

**Hardware assumed:** AMD GPU with ROCm (tested on RX 7800 XT, 16 GB VRAM). Works on CPU too (much slower).

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Option A: Local Data (No Internet)](#2-option-a-local-data-no-internet)
3. [Option B: HuggingFace 10M Dataset](#3-option-b-huggingface-10m-dataset)
4. [Smoke Tests](#4-smoke-tests)
5. [Training](#5-training)
6. [Monitoring Training](#6-monitoring-training)
7. [Inference / Text Generation](#7-inference--text-generation)
8. [Resuming from Checkpoint](#8-resuming-from-checkpoint)
9. [Full Pipeline Dry Run (2 Minutes)](#9-full-pipeline-dry-run-2-minutes)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Environment Setup

```bash
cd /home/xcrr1/aether2
source venv/bin/activate

# ROCm environment (required for AMD GPUs)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Verify GPU is visible
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
```

**Expected output:** `GPU: AMD Radeon RX 7800 XT, VRAM: 16.0 GB`

If no GPU, the code falls back to CPU with float32 (training will be very slow but functional).

---

## 2. Option A: Local Data (No Internet)

Generates training data from Python stdlib introspection and algorithmic patterns. Good for validating the pipeline. Model will overfit after ~20k steps.

### 2a. Generate data + train tokenizer

```bash
python generate_data.py \
    --train-tokenizer \
    --tokenizer omega_tokenizer.json \
    --output data/aether_train.jsonl \
    --n-samples 60000 \
    --max-seq-len 512
```

**What this does:**
1. Extracts source code from 50 Python stdlib modules
2. Generates builtin usage patterns and algorithmic implementations
3. Augments with variable-rename and style variants
4. Trains a 32K-vocab BPE tokenizer on the corpus
5. Tokenizes everything into 512-token windows
6. Writes JSONL with `input_ids`, `labels`, `trust_score`, `source`

**Expected output:** `data/aether_train.jsonl` (~60K records) + `omega_tokenizer.json`

### 2b. Check the data

```bash
# Line count
wc -l data/aether_train.jsonl

# Verify format
head -1 data/aether_train.jsonl | python -c "
import json, sys
d = json.loads(sys.stdin.readline())
assert len(d['input_ids']) == 512
assert max(d['input_ids']) < 32768
print(f'OK: {len(d[\"input_ids\"])} tokens, max_id={max(d[\"input_ids\"])}, source={d[\"source\"]}')
"

# Source distribution
python -c "
import json
from collections import Counter
c = Counter()
with open('data/aether_train.jsonl') as f:
    for line in f:
        c[json.loads(line)['source']] += 1
for k, v in sorted(c.items()):
    print(f'  {k}: {v:,} ({v/sum(c.values())*100:.1f}%)')
print(f'  TOTAL: {sum(c.values()):,}')
"
```

---

## 3. Option B: HuggingFace 10M Dataset

Streams from 9 public datasets — no login or approval gates required. This is the recommended path for real training.

### 3a. Verify all sources are accessible

```bash
python build_dataset.py --verify-only
```

**Expected:** All 8 unique datasets show a checkmark. If any fail, check your internet connection and the error message.

### 3b. Quick smoke test (1000 samples)

```bash
python build_dataset.py \
    --train-tokenizer \
    --n-samples 1000 \
    --output /tmp/aether_smoke.jsonl \
    --tokenizer /tmp/omega_tok_smoke.json \
    --vocab-size 32768 \
    --max-seq-len 512 \
    --seed 42
```

**Time:** ~30-40 minutes (BPE training with pure-Python tokenizer is the bottleneck for 32K vocab). The dataset generation itself takes ~5 minutes.

**What to check:**
- All 4 source tags present: `code`, `math`, `reasoning`, `educational`
- Proportions: code 30%, math 25%, reasoning 25%, educational 20%
- Format validation passes
- `max(input_ids) < 32768`

### 3c. Full 10M dataset build

```bash
# If you already have a tokenizer from step 3b, skip --train-tokenizer:
python build_dataset.py \
    --train-tokenizer \
    --n-tokenizer-texts 500000 \
    --output data/aether_train.jsonl \
    --tokenizer omega_tokenizer.json \
    --n-samples 10000000 \
    --max-seq-len 512 \
    --vocab-size 32768 \
    --sort-by-difficulty \
    --seed 42
```

**Time estimate:**
| Phase | Duration |
|-------|----------|
| Tokenizer training (500K texts, 32K vocab) | 2-6 hours |
| Dataset generation (streaming from 9 sources) | 6-12 hours |
| Deduplication + shuffle | ~10 minutes |
| Difficulty sort | ~10 minutes |
| **Total** | **~8-18 hours** |

**Tip:** Run in `tmux` or `screen` so it survives a disconnected terminal:
```bash
tmux new -s aether
# ... run the build command ...
# Ctrl+B, D to detach. tmux attach -t aether to resume.
```

### 3d. Verify the dataset

```bash
# Total records
wc -l data/aether_train.jsonl

# Format check
head -1 data/aether_train.jsonl | python -c "
import json, sys
d = json.loads(sys.stdin.readline())
assert len(d['input_ids']) == 512
assert max(d['input_ids']) < 32768
print('Format OK')
"

# Source distribution
python -c "
import json
from collections import Counter
c = Counter()
with open('data/aether_train.jsonl') as f:
    for line in f:
        c[json.loads(line)['source']] += 1
for k, v in sorted(c.items()):
    print(f'  {k}: {v:,} ({v/sum(c.values())*100:.1f}%)')
print(f'  TOTAL: {sum(c.values()):,}')
"
```

**Expected distribution (10M):**
```
  code:        3,000,000 (30.0%)
  educational: 2,000,000 (20.0%)
  math:        2,500,000 (25.0%)
  reasoning:   2,500,000 (25.0%)
  TOTAL:      10,000,000
```

---

## 4. Smoke Tests

Run the 56 architectural smoke tests before training. These validate NaN safety, Poincaré ball geometry, CSSC curvature coupling, GGR geodesic routing, SSM gradients, tokenizer round-trip, and a full mini-pipeline.

```bash
python smoke_test.py
```

**Expected:** All 56 tests show `PASS` and `SUMMARY: PASSED: 56 / FAILED: 0`. Any `FAIL` means a NaN or Inf was detected — do not train until resolved.

**Key test groups:**
| Group | What it validates |
|-------|------------------|
| Poincaré ball ops | exp_map / log_map / möbius_add NaN-free in BF16 |
| Multi-timescale SSM | d_state 8/16/32 across layer groups produce finite outputs |
| Möbius residual | Hyperbolic addition stays inside the ball |
| CSSC | Per-token curvature from Δ gate is finite, c_token ∈ (0.01, 2.0) |
| GGR Micro-MoE | Geodesic routing gates sum to 1.0, expert outputs finite |
| Geometry gating | Euclidean/hyperbolic blend gates work correctly |
| RiemannianRescale | Gradient rescaling backward pass is finite |
| Iterative refinement | Second forward pass ("think twice") is stable |
| Rosetta observer | Detached probe produces valid logits |
| Full forward+backward | Complete model with all pillars, gradient flow confirmed |
| Tokenizer round-trip | encode(decode(x)) == x for sample text |
| Mini pipeline | data load → model → loss → backward → step |

---

## 5. Training

### 5a. Start training (full run)

```bash
python train.py
```

This uses all defaults from `aether_config.py`:
- 50,000 steps, micro_batch=4, grad_accum=16 (effective batch=64)
- BF16 with gradient checkpointing
- Cosine LR schedule (4e-4 → 1e-5)
- Checkpoint every 2,000 steps

### 5b. Training with custom settings

```bash
# Fewer steps (quick experiment)
python train.py --max-steps 5000

# Larger batch, lower LR
python train.py --micro-batch 8 --grad-accum 8 --lr 2e-4

# No 8-bit Adam (if bitsandbytes is broken)
python train.py --no-8bit-adam

# Custom data path
python train.py --data-path /path/to/custom_data.jsonl

# Custom checkpoint directory
python train.py --checkpoint-dir my_checkpoints
```

### 5c. What happens during training

The training loop integrates all five architectural pillars plus the two novel contributions:

1. **Picky Batch Sampler** fetches 4x candidates, scores them, keeps samples with CE in [0.2, 5.0]
2. **Forward pass** runs through 24 Mamba-Hyperbolic blocks with Poincaré geometry
   - **CSSC** (every block): Mamba Δ gate modulates per-token curvature — salient tokens get richer hyperbolic space
   - **GGR-MoE** (blocks 3, 7, 11, 15, 19, 23): geodesic distance to 3 Poincaré centroids determines expert routing
3. **Anticipatory loss** rewards predicting future hidden states (MSE, 5 positions ahead)
4. **Rosetta Observer** runs a detached probe for interpretability
5. **EMA Teacher** provides self-distillation targets (KL divergence)
6. **Entropy bonus** prevents overconfident logits
7. **Neurogenesis** monitors gradient variance, expands SwiGLU FFN layers (non-GGR blocks) when saturated
8. **Loss spike detection** skips optimizer steps if loss spikes > 3x rolling average

---

## 6. Monitoring Training

### Log output

Every 50 steps (configurable via `log_every`):
```
[  50/50000]  lr=1.33e-05  ce=9.8431  ppl=18823.4  ant=0.0142  ros=9.7211  total=11.2103  kept=0%  3421ms/step
```

Every 500 steps (configurable via `eval_every`):
```
[eval]  step=500  val_ce=8.2310  val_ppl=3745.2
[eval]  step=500  ast_pass_rate=0.00%  cumulative_pass=0.00%  n_verified=4
```

### Key metrics to watch

| Metric | What it means | Healthy sign |
|--------|--------------|-------------|
| `ce` | Cross-entropy loss | Drops from ~10 → ~5 in first 1000 steps |
| `ppl` | Perplexity (exp(CE)) | < 5000 by step 500 |
| `val_ce` | Validation CE (every 500 steps) | Within 0.3 of train CE |
| `ros` | Rosetta observer CE | Similar trend to main CE |
| `ant` | Anticipatory MSE loss | Usually < 1.0 |
| `kept` | % of picky sampler candidates kept | Rises to 50-80% as model learns |
| `ms/step` | Time per step | 2000-5000ms on RX 7800 XT |

### Warning signs

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `val_ce` > `ce` + 1.0 after step 5k | Overfitting | More data, more dropout |
| `ppl` > 10000 after step 1000 | Broken data pipeline | Check tokenizer round-trip, check labels are shifted |
| `kept` < 20% throughout | Picky thresholds too strict | Adjust `picky_ce_min`/`picky_ce_max` |
| `ce` not decreasing after step 500 | Learning rate issue | Check LR, check data loading |
| OOM crash | VRAM exceeded | Enable gradient checkpointing, reduce micro_batch |

### VRAM monitoring

VRAM is logged every 100 steps:
```
[vram]     step_100  alloc=5.21 GiB  reserved=6.50 GiB  total=16.0 GiB  headroom=9.50 GiB
```

Target: < 10 GiB allocated. If headroom drops below 2 GiB, reduce `micro_batch`.

### Checkpoints

Saved every 2,000 steps to `checkpoints_omega/`:
```
checkpoints_omega/
  step_0002000_main.safetensors
  step_0002000_rosetta.safetensors
  step_0002000_ema.safetensors
  step_0002000_optimizer.safetensors
  step_0002000_state.json          # step count, LR, loss history
```

---

## 7. Inference / Text Generation

After training (or from any checkpoint):

```bash
# Top-p sampling (creative)
python inference.py \
    --checkpoint checkpoints_omega/step_0010000 \
    --tokenizer omega_tokenizer.json \
    --prompt "def binary_search(arr, target):" \
    --max-new-tokens 256 \
    --temperature 0.8 \
    --top-p 0.9

# Greedy decoding (deterministic)
python inference.py \
    --checkpoint checkpoints_omega/step_0010000 \
    --prompt "def fibonacci(n):" \
    --greedy

# Longer generation
python inference.py \
    --checkpoint checkpoints_omega/step_0050000 \
    --prompt "Problem: Find the sum of all prime numbers below 1000\n\nSolution:\n" \
    --max-new-tokens 512 \
    --temperature 0.7
```

**Note:** The `--checkpoint` path is the prefix without file extension. The script loads `_main.safetensors` automatically.

**Output includes tokens/second** — useful for benchmarking inference speed.

---

## 8. Resuming from Checkpoint

Training can be interrupted (Ctrl+C sends SIGINT, which is caught gracefully) and resumed:

```bash
# Resume from step 10000
python train.py --resume checkpoints_omega/step_0010000

# Resume with different hyperparameters
python train.py --resume checkpoints_omega/step_0010000 --lr 1e-4 --max-steps 100000
```

Resume restores: model weights, optimizer state, EMA teacher, Rosetta observer, step counter, LR schedule position.

---

## 9. Full Pipeline Dry Run (2 Minutes)

Test everything end-to-end in under 2 minutes:

```bash
cd /home/xcrr1/aether2
source venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# Step 1: Generate minimal data + tokenizer (~30s)
python generate_data.py \
    --train-tokenizer \
    --n-samples 500 \
    --max-seq-len 512

# Step 2: Run smoke tests (~15s)
python smoke_test.py

# Step 3: Train for 5 steps (~30s)
python train.py \
    --max-steps 5 \
    --micro-batch 2 \
    --grad-accum 1 \
    --no-8bit-adam

# Step 4: Generate text from the (untrained) checkpoint
python inference.py \
    --checkpoint checkpoints_omega/step_0000005 \
    --prompt "def hello():" \
    --max-new-tokens 64 \
    --greedy
```

**Expected:** All steps complete without errors. The generated text will be garbage (5 training steps), but the pipeline is validated.

---

## 10. Troubleshooting

### ROCm / GPU issues

**"No GPU detected"**
```bash
# Check ROCm sees the GPU
rocm-smi
# Should show your AMD GPU with temperature, utilization, etc.

# Check PyTorch sees it
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

**"HIP error" or wrong GFX version**
```bash
# Set the correct GFX version for your GPU
# RX 7800 XT = GFX1101 → override to 11.0.0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### OOM (Out of Memory)

```bash
# Reduce batch size
python train.py --micro-batch 2 --grad-accum 32

# Gradient checkpointing is ON by default. If somehow disabled:
# Edit aether_config.py: use_gradient_checkpointing = True
```

### Tokenizer issues

```bash
# Test tokenizer round-trip
python tokenizer.py --load omega_tokenizer.json --test "def hello(x: int) -> str: return str(x)"
```

If round-trip fails, retrain the tokenizer with more data.

### Dataset issues

```bash
# Verify HuggingFace source access
python build_dataset.py --verify-only

# Check a specific record
python -c "
import json
with open('data/aether_train.jsonl') as f:
    d = json.loads(f.readline())
print(f'input_ids[:10] = {d[\"input_ids\"][:10]}')
print(f'labels[:10]    = {d[\"labels\"][:10]}')
print(f'source         = {d[\"source\"]}')
print(f'trust_score    = {d[\"trust_score\"]}')
print(f'len            = {len(d[\"input_ids\"])}')
print(f'max_id         = {max(d[\"input_ids\"])}')
# Labels should be input_ids shifted left by 1:
assert d['labels'][0] == d['input_ids'][1], 'Labels not shifted!'
print('Label shift OK')
"
```

### Training loss not decreasing

1. Check the data is valid (run the dataset verification above)
2. Check the tokenizer matches (`vocab_size` in tokenizer == 32768)
3. Try a higher learning rate: `--lr 1e-3`
4. Try without picky sampling: set `picky_ce_min=0.0, picky_ce_max=99.0` in `aether_config.py`

### "terminate called without an active exception" / "Aborted (core dumped)"

This is a known ROCm/PyTorch cleanup issue on AMD GPUs. It happens on Python exit, **not during training**. It is harmless — your data/checkpoints are fine. Ignore it.

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate env | `source venv/bin/activate` |
| Set ROCm vars | `export HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| Local data (60K) | `python generate_data.py --train-tokenizer --n-samples 60000` |
| HF data verify | `python build_dataset.py --verify-only` |
| HF data smoke (1K) | `python build_dataset.py --train-tokenizer --n-samples 1000 --output /tmp/smoke.jsonl --tokenizer /tmp/tok.json` |
| HF data full (10M) | `python build_dataset.py --train-tokenizer --n-tokenizer-texts 500000 --n-samples 10000000 --sort-by-difficulty` |
| Smoke tests | `python smoke_test.py` |
| Train (default) | `python train.py` |
| Train (quick test) | `python train.py --max-steps 5 --micro-batch 2 --grad-accum 1 --no-8bit-adam` |
| Resume training | `python train.py --resume checkpoints_omega/step_0010000` |
| Generate text | `python inference.py --checkpoint checkpoints_omega/step_0010000 --prompt "def foo():"` |
| Test tokenizer | `python tokenizer.py --load omega_tokenizer.json --test "hello world"` |
