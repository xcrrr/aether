#!/usr/bin/env python3
"""Aether Omega — Inference / text generation.

Loads a saved checkpoint and generates text from a prompt.

CLI:
  python inference.py \\
      --checkpoint checkpoints_omega/step_0002000 \\
      --tokenizer omega_tokenizer.json \\
      --prompt "def binary_search(arr, target):" \\
      --max-new-tokens 256 \\
      --temperature 0.8 \\
      --top-p 0.9

  python inference.py --checkpoint checkpoints_omega/step_0002000 \\
      --prompt "def " --greedy
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from aether_config import OmegaConfig
from model import AetherOmegaModel


# ── Device setup ─────────────────────────────────────────────────────────────

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


# ── Sampling ─────────────────────────────────────────────────────────────────

def _top_p_sample(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    """Nucleus (top-p) sampling from a (vocab_size,) logit vector."""
    logits = logits.float()
    if temperature != 1.0:
        logits = logits / temperature

    probs   = F.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = probs.sort(descending=True)
    cum_probs = sorted_probs.cumsum(dim=0)

    # Keep smallest set whose cumulative prob ≥ top_p
    cutoff  = (cum_probs - sorted_probs) < top_p
    sorted_probs[~cutoff] = 0.0
    sorted_probs.div_(sorted_probs.sum())

    next_token = sorted_ids[torch.multinomial(sorted_probs, num_samples=1)]
    return next_token.item()


def _greedy_sample(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


# ── Generation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model:          AetherOmegaModel,
    prompt_ids:     torch.Tensor,
    max_new_tokens: int,
    cfg:            OmegaConfig,
    greedy:         bool = False,
    temperature:    float = 0.8,
    top_p:          float = 0.9,
) -> torch.Tensor:
    """Autoregressive generation.

    Args:
        model:          AetherOmegaModel in eval mode.
        prompt_ids:     (1, T) prompt token IDs.
        max_new_tokens: Number of new tokens to generate.
        cfg:            OmegaConfig (for max_seq_len, eos_id).
        greedy:         If True, use argmax (deterministic).
        temperature:    Sampling temperature (ignored when greedy=True).
        top_p:          Nucleus sampling probability mass.

    Returns:
        (1, T + max_new_tokens) tensor with prompt + generated tokens.
    """
    ids = prompt_ids.clone()
    eos_id = 3  # matches tokenizer

    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
        for _ in range(max_new_tokens):
            # Trim to model's max seq len
            ids_trunc = ids[:, -cfg.max_seq_len:]
            logits, _, _ = model(ids_trunc)         # (1, T', V)
            last_logits  = logits[0, -1, :]         # (V,)

            if greedy:
                next_tok = _greedy_sample(last_logits)
            else:
                next_tok = _top_p_sample(last_logits, top_p, temperature)

            ids = torch.cat([ids, torch.tensor([[next_tok]], device=DEVICE)], dim=1)
            if next_tok == eos_id:
                break

    return ids


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aether Omega — text generation from a checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint",      type=str, required=True,
                   help="Checkpoint path prefix (without extension), "
                        "e.g. checkpoints_omega/step_0002000")
    p.add_argument("--tokenizer",       type=str, default="omega_tokenizer.json")
    p.add_argument("--prompt",          type=str, default="def ",
                   help="Text prompt for generation")
    p.add_argument("--max-new-tokens",  type=int, default=256)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--top-p",           type=float, default=0.9)
    p.add_argument("--greedy",          action="store_true",
                   help="Greedy (argmax) decoding — deterministic output")
    return p.parse_args()


def main() -> None:
    args = _parse()

    # ── Check checkpoint ──────────────────────────────────────────────────────
    weights_path = Path(args.checkpoint + "_main.safetensors")
    if not weights_path.exists():
        print(
            f"\n[error] Checkpoint not found: {weights_path}\n"
            "\nTo create a checkpoint, run:\n"
            "  python generate_data.py --train-tokenizer --n-samples 60000\n"
            "  python train.py --max-steps 2000\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    from tokenizer import OmegaTokenizer
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        print(f"[error] Tokenizer not found: {tok_path}", file=sys.stderr)
        sys.exit(1)
    tok = OmegaTokenizer.from_file(tok_path)
    print(f"[inf] Tokenizer loaded: vocab_size={tok.vocab_size}")

    # ── Build model ───────────────────────────────────────────────────────────
    cfg   = OmegaConfig()
    dtype = torch.bfloat16 if USE_AMP else torch.float32
    model = AetherOmegaModel(cfg).to(DEVICE, dtype=dtype)
    model.load_state_dict(load_file(str(weights_path), device=DEVICE))
    model.eval()
    n_params = model.count_parameters()
    print(f"[inf] Model loaded: {n_params / 1e6:.1f}M params  device={DEVICE}  dtype={dtype}")

    # ── Encode prompt ─────────────────────────────────────────────────────────
    prompt_ids_list = tok.encode(args.prompt, add_bos=True, add_eos=False)
    prompt_tensor   = torch.tensor([prompt_ids_list], dtype=torch.long, device=DEVICE)
    print(f"\n[inf] Prompt ({len(prompt_ids_list)} tokens): {args.prompt!r}")
    print(f"[inf] Generating {args.max_new_tokens} tokens  "
          f"{'(greedy)' if args.greedy else f'(temp={args.temperature}, top_p={args.top_p})'}")
    print("-" * 60)

    # ── Generate ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    output_ids = generate(
        model,
        prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        cfg=cfg,
        greedy=args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    elapsed = time.perf_counter() - t0

    # ── Decode and print ──────────────────────────────────────────────────────
    new_tokens = output_ids[0, len(prompt_ids_list):].tolist()
    full_ids   = output_ids[0].tolist()
    text       = tok.decode(full_ids, skip_special=True)
    new_text   = tok.decode(new_tokens, skip_special=True)

    print(text)
    print("-" * 60)
    n_new = len(new_tokens)
    tok_per_sec = n_new / max(elapsed, 1e-9)
    print(f"\n[inf] Generated {n_new} tokens in {elapsed:.2f}s  ({tok_per_sec:.1f} tok/s)")


if __name__ == "__main__":
    main()
