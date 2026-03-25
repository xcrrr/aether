#!/usr/bin/env python3
"""Build Aether Omega pre-training dataset — world-class 10M-sample data blend.

Dataset mixture (10M samples default):
  30% code        — bigcode/starcoderdata (Python 2.9M + JS 25K) + Magicoder 75K
  25% math        — HuggingFaceTB/finemath 2M + nvidia/OpenMathInstruct-2 500K
  25% reasoning   — OpenThoughts3 1.2M + OpenR1-Math 220K + OpenMathInstruct-2 1.08M
  20% educational — HuggingFaceFW/fineweb-edu 2M

All sources publicly accessible without gating.

Output format (JSONL, one record per line):
  {"input_ids": [int x 512], "labels": [int x 512], "trust_score": float, "source": str}

  input_ids: BPE-tokenized, padded/truncated to exactly max_seq_len (512)
  labels:    input_ids shifted left by 1 (next-token prediction), padding = -100
  All token IDs < vocab_size (32768).
  Minimum 16 non-PAD tokens per sample.

Usage:
  python build_dataset.py \\
      --train-tokenizer \\
      --output data/aether_train.jsonl \\
      --tokenizer omega_tokenizer.json \\
      --n-samples 10000000 \\
      --max-seq-len 512 \\
      --vocab-size 32768 \\
      --sort-by-difficulty \\
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# ── Constants ─────────────────────────────────────────────────────────────────

PAD_ID       = 0
BOS_ID       = 2
EOS_ID       = 3
IGNORE_INDEX = -100
MIN_TOKENS   = 16
BASE_N       = 10_000_000

# ── Group mix (must sum to 1.0) ──────────────────────────────────────────────

GROUP_FRACTIONS: dict[str, float] = {
    "code":        0.30,
    "math":        0.25,
    "reasoning":   0.25,
    "educational": 0.20,
}
assert abs(sum(GROUP_FRACTIONS.values()) - 1.0) < 1e-9


# ── Sub-source definitions ───────────────────────────────────────────────────
#
# Within each group: capped sources are processed first, the fill source
# (cap=0) last.  Fill sources are massive corpora that won't run out.

@dataclass(frozen=True)
class SubSource:
    """One HuggingFace dataset slice contributing to the final mix."""
    name:        str
    group:       str               # code | math | reasoning | educational
    trust_score: float             # 0-100, written to JSONL
    hf_repo:     str               # HuggingFace repo ID
    hf_config:   str | None = None
    hf_data_dir: str | None = None
    fmt:         str = "text"      # text | magicoder | problem_solution | math_instruct | conversations
    cap:         int = 0           # >0 = max records at BASE_N; 0 = fill to group target
    skip_first:  int = 0           # skip N HF records (OpenMathInstruct-2 split)
    edu_filter:  float = 0.0       # require score >= this (fineweb-edu)


SUBSOURCES: dict[str, list[SubSource]] = {
    "code": [
        SubSource("magicoder",      "code", 95.0,
                  "ise-uiuc/Magicoder-OSS-Instruct-75K",
                  fmt="magicoder", cap=75_000),
        SubSource("starcoder_js",   "code", 88.0,
                  "bigcode/starcoderdata",
                  hf_data_dir="javascript", fmt="text", cap=25_000),
        SubSource("starcoder_py",   "code", 90.0,
                  "bigcode/starcoderdata",
                  hf_data_dir="python", fmt="text"),                 # fill
    ],
    "math": [
        SubSource("openmath_math",  "math", 96.0,
                  "nvidia/OpenMathInstruct-2",
                  fmt="math_instruct", cap=500_000),
        SubSource("finemath",       "math", 93.0,
                  "HuggingFaceTB/finemath",
                  hf_config="finemath-4plus", fmt="text"),           # fill
    ],
    "reasoning": [
        SubSource("openthoughts3",  "reasoning", 98.0,
                  "open-thoughts/OpenThoughts3-1.2M",
                  fmt="conversations", cap=1_200_000),
        SubSource("openr1_math",    "reasoning", 95.0,
                  "open-r1/OpenR1-Math-220k",
                  fmt="problem_solution", cap=220_000),
        SubSource("openmath_reason","reasoning", 95.0,
                  "nvidia/OpenMathInstruct-2",
                  fmt="math_instruct", skip_first=500_000),          # fill
    ],
    "educational": [
        SubSource("fineweb_edu",    "educational", 88.0,
                  "HuggingFaceFW/fineweb-edu",
                  hf_config="CC-MAIN-2024-51", fmt="text"),          # fill (already edu-filtered)
    ],
}

ALL_SUBSOURCES: list[SubSource] = [s for grp in SUBSOURCES.values() for s in grp]
_BY_NAME: dict[str, SubSource] = {s.name: s for s in ALL_SUBSOURCES}

# Representative source per group for tokenizer training (no skip, large corpus)
_TOK_SOURCES: dict[str, str] = {
    "code":        "starcoder_py",
    "math":        "finemath",
    "reasoning":   "openthoughts3",
    "educational": "fineweb_edu",
}


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_text(item: dict, fmt: str) -> str:
    """Return training text for one dataset record based on format type."""

    if fmt == "text":
        # starcoderdata uses "content", finemath/fineweb-edu use "text"
        return (item.get("text") or item.get("content") or "").strip()

    if fmt == "magicoder":
        problem  = (item.get("problem")  or "").strip()
        solution = (item.get("solution") or "").strip()
        if not problem or not solution:
            return ""
        lang   = (item.get("lang") or "").strip()
        header = f"# {lang}\n" if lang else ""
        return f"{header}Problem: {problem}\n\nSolution:\n{solution}"

    if fmt == "problem_solution":
        # OpenR1-Math-220k: problem + generated_solution (or solution)
        problem  = (item.get("problem") or item.get("question") or "").strip()
        solution = (item.get("solution") or item.get("generated_solution") or "").strip()
        if not problem or not solution:
            return ""
        return f"Problem: {problem}\n\nSolution:\n{solution}"

    if fmt == "math_instruct":
        # nvidia/OpenMathInstruct-2: problem + generated_solution
        problem  = (item.get("problem") or item.get("question") or "").strip()
        solution = (item.get("generated_solution") or item.get("solution") or "").strip()
        if not problem or not solution:
            return ""
        return f"Problem: {problem}\n\nSolution:\n{solution}"

    if fmt == "conversations":
        # OpenThoughts3: conversations list with "from"/"value" or "role"/"content"
        convos = item.get("conversations") or item.get("messages") or []
        parts: list[str] = []
        for turn in convos:
            c = (turn.get("value") or turn.get("content") or "").strip()
            if c:
                parts.append(c)
        return "\n\n".join(parts)

    return ""


# ── HuggingFace dataset streaming ────────────────────────────────────────────

def _load_hf_stream(src: SubSource):
    """Return a fresh streaming IterableDataset for the given sub-source."""
    from datasets import load_dataset

    kw: dict = {"split": "train", "streaming": True}
    if src.hf_config:
        return load_dataset(src.hf_repo, src.hf_config, **kw)
    if src.hf_data_dir:
        return load_dataset(src.hf_repo, data_dir=src.hf_data_dir, **kw)
    return load_dataset(src.hf_repo, **kw)


def iter_texts(src: SubSource, max_texts: int, skip: int = 0) -> Iterator[str]:
    """Stream pre-filtered texts from a sub-source, up to max_texts."""
    ds = _load_hf_stream(src)

    if skip > 0:
        try:
            ds = ds.skip(skip)
        except AttributeError:
            # Fallback for older `datasets` versions without .skip()
            _it = iter(ds)
            for _ in range(skip):
                try:
                    next(_it)
                except StopIteration:
                    return
            ds = _it

    yielded = 0
    for item in ds:
        # Score filter (fineweb-edu)
        if src.edu_filter > 0 and float(item.get("score") or 0.0) < src.edu_filter:
            continue

        text = extract_text(item, src.fmt)
        if len(text) < 50:
            continue

        yield text
        yielded += 1
        if yielded >= max_texts:
            break


# ── Source verification ──────────────────────────────────────────────────────

def verify_sources() -> bool:
    """Test every unique HF dataset is accessible. Returns True if all OK."""
    print("\n[verify] Testing all data sources ...\n")
    ok = True
    seen: set[tuple] = set()

    for src in ALL_SUBSOURCES:
        key = (src.hf_repo, src.hf_config, src.hf_data_dir)
        if key in seen:
            print(f"  \u2713 {src.name:<25} (same stream as earlier source)")
            continue
        seen.add(key)

        try:
            ds = _load_hf_stream(src)
            sample = next(iter(ds))
            fields = list(sample.keys())
            text   = extract_text(sample, src.fmt)
            print(f"  \u2713 {src.name:<25} fields={fields[:6]}  text_len={len(text)}")
        except Exception as exc:
            print(f"  \u2717 {src.name:<25} FAILED: {exc}", file=sys.stderr)
            ok = False

    if ok:
        print("\n  All sources accessible.\n")
    else:
        print("\n  Some sources FAILED. See errors above.\n", file=sys.stderr)
    return ok


# ── Tokenizer ────────────────────────────────────────────────────────────────

def train_tokenizer(output_path: str, vocab_size: int, n_texts: int = 500_000) -> None:
    """Collect texts proportionally from all groups and train BPE tokenizer."""
    from tokenizer import OmegaTokenizer

    print(f"\n[tokenizer] Collecting {n_texts:,} texts for BPE training ...\n")
    all_texts: list[str] = []

    for group, frac in GROUP_FRACTIONS.items():
        src    = _BY_NAME[_TOK_SOURCES[group]]
        target = int(n_texts * frac)
        print(f"  {src.name:<25} target={target:,}", end="", flush=True)
        n = 0
        gen = iter_texts(src, target * 3)
        try:
            for text in gen:
                all_texts.append(text[:4096])
                n += 1
                if n >= target:
                    break
        except Exception as exc:
            print(f"\n    WARNING: {src.name}: {exc}", file=sys.stderr)
        finally:
            gen.close()   # release HF streaming connection
        print(f"  \u2192 {n:,}")

    if not all_texts:
        sys.exit("[tokenizer] ERROR: no texts collected.")

    print(f"\n[tokenizer] Training BPE on {len(all_texts):,} texts  "
          f"vocab_size={vocab_size} ...")
    tok = OmegaTokenizer()
    tok.train(all_texts, vocab_size=vocab_size)
    tok.save(output_path)
    print(f"[tokenizer] Saved \u2192 {output_path}\n")


def load_tokenizer(path: str):
    """Load a trained OmegaTokenizer from disk."""
    from tokenizer import OmegaTokenizer
    tok = OmegaTokenizer.from_file(path)
    print(f"[tokenizer] Loaded '{path}'  vocab_size={tok.vocab_size}")
    return tok


# ── Record builder ───────────────────────────────────────────────────────────

def text_to_windows(
    text:        str,
    tokenizer,
    group:       str,
    trust_score: float,
    vocab_size:  int,
    seq_len:     int,
) -> list[tuple[dict, tuple[int, ...]]]:
    """Tokenize text and split into non-overlapping fixed-length windows.

    Returns list of (record_dict, first_64_ids) pairs.
    first_64_ids is used for exact-match deduplication.

    Window layout (seq_len = 512):
      input_ids = [BOS, tok_1, ..., tok_N, PAD, ...PAD]   len = seq_len
      labels    = [tok_1, ..., tok_N, EOS, -100, ...-100]  len = seq_len
    """
    text = text.strip()
    if not text:
        return []
    if len(text) > 65_536:
        text = text[:65_536]

    all_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    if not all_ids:
        return []

    window  = seq_len - 2   # content slots (BOS + content + EOS)
    results: list[tuple[dict, tuple[int, ...]]] = []

    for start in range(0, len(all_ids), window):
        chunk  = all_ids[start : start + window]
        n_real = len(chunk) + 1   # BOS + chunk
        if n_real < MIN_TOKENS:
            continue

        full = [BOS_ID] + chunk + [EOS_ID]
        inp  = full[:-1]
        lbl  = full[1:]

        pad_len   = seq_len - len(inp)
        input_ids = inp + [PAD_ID]       * pad_len
        labels    = lbl + [IGNORE_INDEX] * pad_len

        assert len(input_ids) == seq_len
        assert len(labels)    == seq_len

        # Validate: all real token IDs within vocab range
        if any(x >= vocab_size or x < 0 for x in input_ids if x != PAD_ID):
            continue
        if any(x >= vocab_size for x in labels if x != IGNORE_INDEX):
            continue

        record = {
            "input_ids":   input_ids,
            "labels":      labels,
            "trust_score": trust_score,
            "source":      group,
        }
        fp = tuple(input_ids[:64])
        results.append((record, fp))

    return results


# ── Per-source temp-file writer ──────────────────────────────────────────────

def generate_source(
    src:        SubSource,
    target:     int,
    tokenizer,
    vocab_size: int,
    seq_len:    int,
    dedup:      set[int],
    tmp_dir:    str,
    skip:       int = 0,
) -> tuple[str, int, int]:
    """Stream from src, tokenize, dedup, write temp JSONL.

    Returns (temp_path, records_written, tokens_counted).
    """
    path    = os.path.join(tmp_dir, f"{src.name}.jsonl")
    written = 0
    tokens  = 0
    t0      = time.time()

    # Overfetch ratio: text sources produce many windows per input text
    if src.fmt == "text":
        max_texts = target * 6
    elif src.fmt == "conversations":
        max_texts = target * 3
    else:
        max_texts = target * 3

    gen = iter_texts(src, max_texts, skip=skip)
    try:
        with open(path, "w", encoding="utf-8") as f:
            for text in gen:
                if written >= target:
                    break

                windows = text_to_windows(
                    text, tokenizer, src.group,
                    src.trust_score, vocab_size, seq_len,
                )

                for rec, fp in windows:
                    if written >= target:
                        break

                    # Dedup: exact match on first 64 tokens via hash
                    h = hash(fp)
                    if h in dedup:
                        continue
                    dedup.add(h)

                    f.write(json.dumps(rec) + "\n")
                    written += 1
                    tokens  += sum(1 for x in rec["input_ids"] if x != PAD_ID)

                # Progress every 100K
                if written > 0 and written % 100_000 == 0:
                    el = time.time() - t0
                    print(f"    [{src.name}] {written:>10,}/{target:,}  "
                          f"rate={written / el:,.0f}/s")
    finally:
        gen.close()   # release HF streaming connection

    el = time.time() - t0
    print(f"  \u2713 {src.name:<25} {written:>10,}/{target:,} records  "
          f"{tokens:>12,} tok  {el:.1f}s")
    return path, written, tokens


# ── Interleave-shuffle ───────────────────────────────────────────────────────

def interleave_shuffle(
    tmp_files:   list[str],
    output_path: str,
    seed:        int,
) -> int:
    """Combine temp files into a globally shuffled output.

    Algorithm: build an array of source indices (one per record), shuffle it,
    then read sequentially from each source file following the shuffled order.
    Memory: O(total_records) for the index array (~40 MB per 10M records).

    Returns total records written.
    """
    # Count lines in each temp file
    counts: list[int] = []
    for p in tmp_files:
        n = 0
        with open(p) as f:
            for _ in f:
                n += 1
        counts.append(n)

    total = sum(counts)
    if total == 0:
        return 0

    print(f"\n[shuffle] Interleaving {total:,} records from "
          f"{len(tmp_files)} temp files ...")

    # Build and shuffle source-selection index
    rng = random.Random(seed)
    idx: list[int] = []
    for i, c in enumerate(counts):
        idx.extend([i] * c)
    rng.shuffle(idx)

    # Open all temp files for reading
    handles = [open(p, "r", encoding="utf-8") for p in tmp_files]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    try:
        with open(output_path, "w", encoding="utf-8") as out:
            for si in idx:
                line = handles[si].readline()
                if not line:
                    continue
                out.write(line)
                written += 1
                if written % 2_000_000 == 0:
                    print(f"  [shuffle] {written:>10,}/{total:,}")
    finally:
        for h in handles:
            h.close()

    print(f"[shuffle] Done: {written:,} records \u2192 {output_path}")
    return written


# ── Sort by difficulty ───────────────────────────────────────────────────────

def sort_by_difficulty(path: str) -> None:
    """Sort JSONL ascending by difficulty = unique_tokens / seq_len.

    Uses byte-offset indexing so only the index lives in memory, not all
    record data.  ~720 MB for 10M records.
    """
    print(f"\n[sort] Indexing records by difficulty ...")
    t0 = time.time()

    # First pass: collect (difficulty, byte_offset, line_length)
    entries: list[tuple[float, int, int]] = []
    with open(path, "rb") as f:
        while True:
            off  = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            rec  = json.loads(line)
            real = [x for x in rec["input_ids"] if x != PAD_ID]
            diff = len(set(real)) / len(rec["input_ids"]) if rec["input_ids"] else 0.0
            entries.append((diff, off, len(line)))

    print(f"[sort] Sorting {len(entries):,} records ...")
    entries.sort(key=lambda x: x[0])

    # Second pass: write sorted output using byte offsets
    tmp = path + ".sorted"
    with open(path, "rb") as fin, open(tmp, "wb") as fout:
        for _, off, ln in entries:
            fin.seek(off)
            fout.write(fin.read(ln))

    os.replace(tmp, path)
    print(f"[sort] Done: {len(entries):,} records sorted in "
          f"{time.time() - t0:.1f}s")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt(s: float) -> str:
    """Format seconds into a human-readable duration."""
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s / 60:.1f}min"
    return f"{s / 3600:.1f}h"


# ── Main build pipeline ─────────────────────────────────────────────────────

def build(
    output_path:    str,
    tokenizer_path: str,
    n_samples:      int,
    vocab_size:     int,
    seq_len:        int,
    seed:           int,
    do_sort:        bool = False,
) -> None:
    """Full pipeline: verify -> generate -> dedup -> shuffle -> sort -> stats."""
    t_start = time.time()
    scale   = n_samples / BASE_N

    # ── Check existing output ─────────────────────────────────────────────
    out = Path(output_path)
    if out.exists():
        existing = sum(1 for _ in out.open())
        if existing >= n_samples:
            print(f"[build] Output already has {existing:,} lines >= "
                  f"target {n_samples:,}. Skipping build.")
            return
        print(f"[build] Output has {existing:,} lines < target. Rebuilding.")

    # ── Verify all sources ────────────────────────────────────────────────
    if not verify_sources():
        sys.exit("[ERROR] Some sources are inaccessible. Fix before building.")

    # ── Load tokenizer ────────────────────────────────────────────────────
    tok = load_tokenizer(tokenizer_path)
    assert tok.vocab_size == vocab_size, (
        f"Tokenizer vocab_size={tok.vocab_size} != requested {vocab_size}"
    )

    # ── Compute per-group targets ─────────────────────────────────────────
    group_targets: dict[str, int] = {
        g: int(n_samples * f) for g, f in GROUP_FRACTIONS.items()
    }
    # Assign rounding remainder to educational
    group_targets["educational"] += n_samples - sum(group_targets.values())

    print(f"\n[build] Target: {n_samples:,} records across "
          f"{len(ALL_SUBSOURCES)} sub-sources\n")
    print(f"  {'Group':<15} {'Target':>10}  {'%':>6}")
    print(f"  {'-' * 15} {'-' * 10}  {'-' * 6}")
    for g, t in group_targets.items():
        print(f"  {g:<15} {t:>10,}  {100 * t / n_samples:>5.1f}%")

    # ── Generate per-source temp files ────────────────────────────────────
    dedup:      set[int]                    = set()
    tmp_dir                                 = tempfile.mkdtemp(prefix="aether_ds_")
    tmp_files:  list[str]                   = []
    stats:      dict[str, tuple[int, int]]  = {}   # name -> (records, tokens)
    total_rec = total_tok = 0

    for group, sources in SUBSOURCES.items():
        gtarget = group_targets[group]
        gdone   = 0

        print(f"\n{'─' * 60}")
        print(f"  {group.upper()} (target: {gtarget:,})")
        print(f"{'─' * 60}")

        for src in sources:
            if gdone >= gtarget:
                break

            remaining = gtarget - gdone

            # Capped sources: scale cap proportionally; fill sources: take remainder
            if src.cap > 0:
                src_target = min(max(1, int(src.cap * scale)), remaining)
            else:
                src_target = remaining

            if src_target <= 0:
                continue

            # Scale skip proportionally for small builds (avoids 500K skip for --n-samples 1000)
            effective_skip = int(src.skip_first * scale) if src.skip_first else 0

            path, written, tokens = generate_source(
                src, src_target, tok, vocab_size, seq_len,
                dedup, tmp_dir, skip=effective_skip,
            )

            tmp_files.append(path)
            stats[src.name] = (written, tokens)
            gdone     += written
            total_rec += written
            total_tok += tokens

    # ── Interleave-shuffle ────────────────────────────────────────────────
    final_n = interleave_shuffle(tmp_files, output_path, seed)

    # ── Sort by difficulty ────────────────────────────────────────────────
    if do_sort and final_n > 0:
        sort_by_difficulty(output_path)

    # ── Cleanup temp files ────────────────────────────────────────────────
    for p in tmp_files:
        try:
            os.unlink(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # ── Validate output ───────────────────────────────────────────────────
    if out.exists() and final_n > 0:
        with out.open() as f:
            first = json.loads(f.readline())
        assert "input_ids" in first and "labels" in first
        assert "trust_score" in first and "source" in first
        assert len(first["input_ids"]) == seq_len, (
            f"input_ids len={len(first['input_ids'])} != seq_len={seq_len}")
        assert len(first["labels"]) == seq_len
        assert max(first["input_ids"]) < vocab_size, (
            f"max(input_ids)={max(first['input_ids'])} >= vocab_size={vocab_size}")
        assert first["source"] in {"code", "math", "reasoning", "educational"}
        print(f"\n[validate] Output format OK \u2713")

    # ── Final statistics ──────────────────────────────────────────────────
    elapsed = time.time() - t_start

    print(f"\n{'=' * 72}")
    print(f"  Build complete in {_fmt(elapsed)}")
    print(f"{'=' * 72}")
    print(f"  {'Source':<25} {'Records':>10} {'Tokens':>14} {'Trust':>6}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 14} {'-' * 6}")
    for src in ALL_SUBSOURCES:
        if src.name in stats:
            r, t = stats[src.name]
            print(f"  {src.name:<25} {r:>10,} {t:>14,} {src.trust_score:>5.0f}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 14}")
    print(f"  {'TOTAL':<25} {total_rec:>10,} {total_tok:>14,}")

    # Group breakdown
    print(f"\n  {'Group':<15} {'Records':>10} {'%':>7}")
    print(f"  {'-' * 15} {'-' * 10} {'-' * 7}")
    gc: dict[str, int] = {}
    for src in ALL_SUBSOURCES:
        if src.name in stats:
            gc[src.group] = gc.get(src.group, 0) + stats[src.name][0]
    for g in GROUP_FRACTIONS:
        n   = gc.get(g, 0)
        pct = 100 * n / total_rec if total_rec else 0
        print(f"  {g:<15} {n:>10,} {pct:>6.1f}%")

    sz = out.stat().st_size / 1024 ** 3 if out.exists() else 0
    print(f"\n  Output       : {output_path}")
    print(f"  File size    : {sz:.2f} GB")
    print(f"  Dedup keys   : {len(dedup):,}")
    print(f"  Avg tok/rec  : {total_tok / total_rec:.1f}" if total_rec else "")
    if do_sort:
        print(f"  Sorted       : by difficulty ascending (curriculum)")

    # Source tags check
    source_tags: set[str] = set()
    with out.open() as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            source_tags.add(json.loads(line).get("source", ""))
    required = {"code", "math", "reasoning", "educational"}
    missing  = required - source_tags
    if missing:
        print(f"\n  [WARNING] Missing source tags: {missing}", file=sys.stderr)
    else:
        print(f"  Source tags  : {sorted(source_tags)} \u2713")

    print(f"{'=' * 72}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build Aether Omega pre-training dataset (10M samples)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", default="data/aether_train.jsonl",
                   help="Output JSONL path (default: data/aether_train.jsonl)")
    p.add_argument("--tokenizer", default="omega_tokenizer.json",
                   help="Tokenizer JSON path (default: omega_tokenizer.json)")
    p.add_argument("--n-samples", type=int, default=10_000_000,
                   help="Total records to write (default: 10,000,000)")
    p.add_argument("--vocab-size", type=int, default=32_768,
                   help="Vocabulary size (default: 32768)")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="Sequence length in tokens (default: 512)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--train-tokenizer", action="store_true",
                   help="Train BPE tokenizer from proportional texts before building")
    p.add_argument("--n-tokenizer-texts", type=int, default=500_000,
                   help="Texts for tokenizer training (default: 500,000)")
    p.add_argument("--sort-by-difficulty", action="store_true",
                   help="Sort output by ascending difficulty (unique/total tokens)")
    p.add_argument("--verify-only", action="store_true",
                   help="Only verify source accessibility, then exit")
    args = p.parse_args()

    # ── Verify-only mode ──────────────────────────────────────────────────
    if args.verify_only:
        sys.exit(0 if verify_sources() else 1)

    # ── Train tokenizer ───────────────────────────────────────────────────
    if args.train_tokenizer:
        n_tok = args.n_tokenizer_texts
        # Auto-scale tokenizer training for small builds (fast smoke tests)
        if args.n_samples < 100_000:
            n_tok = min(n_tok, max(5_000, args.n_samples * 5))
            print(f"[auto] Scaled tokenizer texts to {n_tok:,} for small build")
        train_tokenizer(args.tokenizer, args.vocab_size, n_tok)

    # ── Build dataset ─────────────────────────────────────────────────────
    if args.n_samples > 0:
        build(
            output_path=args.output,
            tokenizer_path=args.tokenizer,
            n_samples=args.n_samples,
            vocab_size=args.vocab_size,
            seq_len=args.max_seq_len,
            seed=args.seed,
            do_sort=args.sort_by_difficulty,
        )
    else:
        print("[build] n_samples=0, skipping dataset build.")


if __name__ == "__main__":
    main()
