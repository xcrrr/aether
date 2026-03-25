"""Aether Omega — Dataset & Picky Batch Sampler.

Pillar 3 — Anticipatory Picky Learner (data side):
    PickyBatchSampler runs a no_grad() forward pass on 4× candidates and drops
    batches where per-sample CE loss < 0.2 (too easy) or > 5.0 (too hard/noisy).

Data format (data/aether_train.jsonl):
    Each line is JSON with fields:
        input_ids   : list[int]  — token IDs (vocab=16000)
        labels      : list[int]  — typically same as input_ids shifted by 1
        trust_score : float      — optional quality weight

If a sample's sequence is shorter than max_seq_len it is padded with 0s;
labels at padding positions are set to -100 (ignored by cross_entropy).
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from aether_config import OmegaConfig


# ──────────────────────────────────────────────────────────────────────────────
# Base dataset
# ──────────────────────────────────────────────────────────────────────────────

class OmegaDataset(Dataset):
    """Map-style dataset wrapping the pre-tokenized JSONL training file."""

    def __init__(self, path: str | Path, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len
        self.records: list[dict] = []
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        ids = rec["input_ids"]
        lbl = rec.get("labels", ids)

        # Truncate to max_seq_len
        ids = ids[:self.max_seq_len]
        lbl = lbl[:self.max_seq_len]

        T = len(ids)
        pad_len = self.max_seq_len - T

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels    = torch.tensor(lbl, dtype=torch.long)

        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            labels    = F.pad(labels,    (0, pad_len), value=-100)

        # Normalize trust_score to [0, 1] (stored as 0–100 in JSONL)
        trust = float(rec.get("trust_score", 90.0)) / 100.0
        trust_score = torch.tensor(trust, dtype=torch.float32)

        return {"input_ids": input_ids, "labels": labels, "trust_score": trust_score}


_PIN = torch.cuda.is_available()


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    ids   = torch.stack([b["input_ids"]   for b in batch])
    lbl   = torch.stack([b["labels"]      for b in batch])
    trust = torch.stack([b["trust_score"] for b in batch])
    if _PIN:
        ids   = ids.pin_memory()
        lbl   = lbl.pin_memory()
        trust = trust.pin_memory()
    return {"input_ids": ids, "labels": lbl, "trust_scores": trust}


def split_dataset(cfg: OmegaConfig) -> tuple["OmegaDataset", "OmegaDataset"]:
    """Load data and split deterministically into train / val sets.

    The split is seeded with *cfg.seed* so the same val set is always used
    across runs and resumes (resumption safety).

    Returns:
        (train_dataset, val_dataset)
    """
    full = OmegaDataset(cfg.data_path, cfg.max_seq_len)
    n_total = len(full)
    n_val   = max(1, int(n_total * cfg.val_fraction))
    n_train = n_total - n_val

    # Deterministic shuffle → first n_train = train, rest = val
    rng     = random.Random(cfg.seed)
    indices = list(range(n_total))
    rng.shuffle(indices)

    train_ds = _SubsetDataset(full, indices[:n_train])
    val_ds   = _SubsetDataset(full, indices[n_train:])
    return train_ds, val_ds


class _SubsetDataset(Dataset):
    """Lightweight subset wrapper that re-uses an existing OmegaDataset."""

    def __init__(self, base: "OmegaDataset", indices: list[int]) -> None:
        self._base    = base
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._base[self._indices[idx]]


def load_dataset(cfg: OmegaConfig) -> "OmegaDataset | _SubsetDataset":
    """Return only the *train* split (val fraction excluded)."""
    train_ds, _ = split_dataset(cfg)
    return train_ds


def load_val_dataset(cfg: OmegaConfig) -> "_SubsetDataset":
    """Return only the *val* split (deterministic, seeded)."""
    _, val_ds = split_dataset(cfg)
    return val_ds


def dataset_stats(dataset: Dataset, label: str = "dataset") -> None:
    """Print basic statistics about a tokenized dataset."""
    n = len(dataset)
    if n == 0:
        print(f"[stats] {label}: empty")
        return
    lengths: list[int] = []
    for i in range(min(n, 5000)):   # sample up to 5k for speed
        item = dataset[i]
        ids  = item["input_ids"]
        # Count non-PAD tokens
        non_pad = int((ids != 0).sum().item())
        lengths.append(non_pad)
    mean_len = sum(lengths) / len(lengths)
    var_len  = sum((l - mean_len) ** 2 for l in lengths) / max(len(lengths) - 1, 1)
    std_len  = math.sqrt(var_len)
    total_tok = sum(lengths)
    print(
        f"[stats] {label}: n={n:,}  "
        f"len_mean={mean_len:.0f}  len_std={std_len:.0f}  "
        f"total_tokens≈{total_tok * n / len(lengths):,.0f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Picky Batch Sampler (Pillar 3)
# ──────────────────────────────────────────────────────────────────────────────

class PickyBatchSampler:
    """Curriculum-learning batch sampler with CE-based difficulty filtering.

    Algorithm:
        1. Fetch `oversample_factor × batch_size` candidate samples.
        2. Run model.eval() + no_grad() forward pass on all candidates.
        3. Compute per-sample mean CE loss.
        4. Keep samples where picky_ce_min ≤ CE ≤ picky_ce_max.
        5. If enough kept → return first `batch_size`; else pad with easiest kept.
        6. model.train() restored before returning.

    Overhead: ~25% per training step (one 4× oversample eval forward pass).
    This is acceptable for a PoC; production would use async prefetch workers.
    """

    def __init__(
        self,
        dataset: OmegaDataset,
        model: torch.nn.Module,
        cfg: OmegaConfig,
        device: str | torch.device,
        oversample_factor: int = 4,
    ) -> None:
        self.dataset           = dataset
        self.model             = model
        self.cfg               = cfg
        self.device            = device
        self.oversample_factor = oversample_factor
        self._indices          = list(range(len(dataset)))
        self._epoch            = 1

        # Initial shuffle — always randomise on startup
        random.shuffle(self._indices)
        self._pos = 0
        print(f"[epoch] Epoch 1 begin  (samples={len(self._indices):,})")

        # Statistics for logging
        self.total_candidates    = 0
        self.total_kept          = 0
        self.total_dropped_easy  = 0
        self.total_dropped_hard  = 0

    def _shuffle(self) -> None:
        """Called at each dataset wrap-around (new epoch)."""
        self._epoch += 1
        if self.cfg.shuffle_between_epochs:
            random.shuffle(self._indices)
        self._pos = 0
        print(
            f"[epoch] Epoch {self._epoch} begin  "
            f"(samples={len(self._indices):,}  "
            f"shuffle={self.cfg.shuffle_between_epochs})"
        )

    def _fetch_candidates(self, n: int) -> dict[str, torch.Tensor]:
        """Pull n samples from the dataset, reshuffling if needed."""
        indices: list[int] = []
        while len(indices) < n:
            remaining = n - len(indices)
            available = len(self._indices) - self._pos
            if available <= 0:
                self._shuffle()
                available = len(self._indices)
            take = min(remaining, available)
            indices.extend(self._indices[self._pos:self._pos + take])
            self._pos += take

        batch = [self.dataset[i] for i in indices]
        return collate_fn(batch)

    @torch.no_grad()
    def _compute_per_sample_ce(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample mean CE loss (BF16 autocast, no grad)."""
        self.model.eval()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(str(self.device) != "cpu")):
            logits, _, _ = self.model(input_ids)   # (B, T, V)
        # Per-token CE, ignore -100 padding
        B, T, V = logits.shape
        per_token = F.cross_entropy(
            logits.reshape(B * T, V).float(),
            labels.reshape(B * T),
            ignore_index=-100,
            reduction="none",
        ).reshape(B, T)                         # (B, T)
        # Mean over non-ignored tokens
        valid_mask = (labels != -100).float()
        valid_count = valid_mask.sum(dim=1).clamp(min=1.0)
        per_sample_ce = (per_token * valid_mask).sum(dim=1) / valid_count  # (B,)
        return per_sample_ce

    def get_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Return a filtered batch of exactly `batch_size` samples."""
        n_candidates = batch_size * self.oversample_factor
        candidates   = self._fetch_candidates(n_candidates)

        input_ids    = candidates["input_ids"].to(self.device)
        labels       = candidates["labels"].to(self.device)
        trust_scores = candidates["trust_scores"].to(self.device)

        per_sample_ce = self._compute_per_sample_ce(input_ids, labels)  # (B_cand,)
        self.model.train()

        # Filter
        keep_mask  = (per_sample_ce >= self.cfg.picky_ce_min) & \
                     (per_sample_ce <= self.cfg.picky_ce_max)
        drop_easy  = (per_sample_ce < self.cfg.picky_ce_min).sum().item()
        drop_hard  = (per_sample_ce > self.cfg.picky_ce_max).sum().item()

        self.total_candidates    += n_candidates
        self.total_dropped_easy  += int(drop_easy)
        self.total_dropped_hard  += int(drop_hard)

        kept_ids   = input_ids[keep_mask]
        kept_lbl   = labels[keep_mask]
        kept_trust = trust_scores[keep_mask]
        n_kept     = kept_ids.size(0)
        self.total_kept += n_kept

        if n_kept >= batch_size:
            return {
                "input_ids":    kept_ids[:batch_size],
                "labels":       kept_lbl[:batch_size],
                "trust_scores": kept_trust[:batch_size],
            }

        # Not enough kept — pad by repeating available (rare edge case)
        if n_kept == 0:
            # All candidates were filtered; fall back to raw candidates
            return {
                "input_ids":    input_ids[:batch_size],
                "labels":       labels[:batch_size],
                "trust_scores": trust_scores[:batch_size],
            }

        repeats      = math.ceil(batch_size / n_kept)
        padded_ids   = kept_ids.repeat(repeats, 1)[:batch_size]
        padded_lbl   = kept_lbl.repeat(repeats, 1)[:batch_size]
        padded_trust = kept_trust.repeat(repeats)[:batch_size]
        return {
            "input_ids":    padded_ids,
            "labels":       padded_lbl,
            "trust_scores": padded_trust,
        }

    def picky_stats(self) -> dict:
        total = max(self.total_candidates, 1)
        return {
            "total_candidates":   self.total_candidates,
            "kept_pct":           100.0 * self.total_kept / total,
            "dropped_easy_pct":   100.0 * self.total_dropped_easy / total,
            "dropped_hard_pct":   100.0 * self.total_dropped_hard / total,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Source coverage logging (startup diagnostic)
# ──────────────────────────────────────────────────────────────────────────────

def log_source_coverage(
    dataset,
    n_sample: int = 10_000,
    label: str = "dataset",
) -> None:
    """Print the source-tag distribution of up to n_sample records."""
    from collections import Counter

    counts: Counter = Counter()
    n = min(len(dataset), n_sample)

    for i in range(n):
        # Access the raw record without tensor conversion
        if hasattr(dataset, "records"):
            rec = dataset.records[i]
        elif hasattr(dataset, "_base") and hasattr(dataset._base, "records"):
            rec = dataset._base.records[dataset._indices[i]]
        elif hasattr(dataset, "_dataset") and hasattr(dataset._dataset, "records"):
            # torch.utils.data.Subset
            rec = dataset._dataset.records[dataset._indices[i]]
        else:
            continue
        counts[rec.get("source", "unknown")] += 1

    total = sum(counts.values())
    if total == 0:
        return
    print(f"[coverage] {label} source distribution (sampled {total:,}):")
    for src, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / total
        print(f"  {src:<24}  {cnt:>8,}  ({pct:5.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# Standard DataLoader (for non-picky evaluation passes)
# ──────────────────────────────────────────────────────────────────────────────

def make_eval_loader(
    cfg: OmegaConfig,
    n_samples: int = 256,
    pin_memory: bool = False,
) -> DataLoader:
    """Small fixed-size DataLoader for evaluation (no picky filtering)."""
    dataset = load_dataset(cfg)
    # Take first n_samples for eval
    indices = list(range(min(n_samples, len(dataset))))
    subset  = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=cfg.micro_batch,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    cfg = OmegaConfig(micro_batch=2, max_seq_len=64)
    ds  = load_dataset(cfg)
    print(f"Dataset size: {len(ds)} samples")
    sample = ds[0]
    print(f"  input_ids shape : {sample['input_ids'].shape}")
    print(f"  labels shape    : {sample['labels'].shape}")
    print(f"  label range     : {sample['labels'].min().item()} .. {sample['labels'].max().item()}")
    print(f"  trust_score     : {sample['trust_score'].item():.4f}")

    # Picky sampler smoke-test (CPU, tiny model)
    from model import AetherOmegaModel
    model = AetherOmegaModel(cfg)
    sampler = PickyBatchSampler(ds, model, cfg, device="cpu", oversample_factor=2)
    batch = sampler.get_batch(2)
    print(f"\nPicky batch: input_ids={batch['input_ids'].shape}  labels={batch['labels'].shape}")
    print(f"Picky stats: {sampler.picky_stats()}")
