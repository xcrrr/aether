#!/usr/bin/env python3
"""Create a small, schema-preserving JSONL subset for Colab prototype runs.

The output keeps the exact record layout expected by StreamingJSONLDataset:
  {"input_ids": [...], "labels": [...], "trust_score": float, "source": str}

Examples
--------
  python prepare_colab_subset.py \
      --input data/aether_train.jsonl \
      --output data/aether_train_colab.jsonl \
      --max-records 2048 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small random subset of the training JSONL for Colab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="data/aether_train.jsonl",
                        help="Source JSONL path")
    parser.add_argument("--output", default="data/aether_train_colab.jsonl",
                        help="Destination JSONL path")
    parser.add_argument("--max-records", type=int, default=2048,
                        help="Number of records to keep in the subset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reservoir sampling")
    return parser.parse_args()


def _validate_record(record: dict, expected_len: int | None) -> int:
    required = {"input_ids", "labels", "trust_score", "source"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")

    input_ids = record["input_ids"]
    labels = record["labels"]
    if not isinstance(input_ids, list) or not isinstance(labels, list):
        raise ValueError("input_ids and labels must be lists")
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have matching lengths")

    seq_len = len(input_ids)
    if expected_len is not None and seq_len != expected_len:
        raise ValueError(
            f"Sequence length mismatch: expected {expected_len}, got {seq_len}"
        )

    if not isinstance(record["source"], str):
        raise ValueError("source must be a string")

    float(record["trust_score"])
    return seq_len


def reservoir_sample(input_path: Path, max_records: int, seed: int) -> tuple[list[dict], int, int, Counter[str]]:
    rng = random.Random(seed)
    sample: list[dict] = []
    total = 0
    seq_len: int | None = None
    by_source: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                seq_len = _validate_record(record, seq_len)
            except Exception as exc:
                raise ValueError(f"Invalid record at line {line_no}: {exc}") from exc

            total += 1
            if len(sample) < max_records:
                sample.append(record)
            else:
                idx = rng.randint(0, total - 1)
                if idx < max_records:
                    sample[idx] = record

    for record in sample:
        by_source[record["source"]] += 1

    return sample, total, seq_len or 0, by_source


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.max_records <= 0:
        raise ValueError("--max-records must be > 0")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    sample, total, seq_len, by_source = reservoir_sample(
        input_path=input_path,
        max_records=args.max_records,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in sample:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(sample):,} / {total:,} records to {output_path}")
    print(f"Sequence length: {seq_len}")
    if by_source:
        print("Source mix:")
        for source, count in sorted(by_source.items()):
            print(f"  {source:<16} {count:>6}")


if __name__ == "__main__":
    main()