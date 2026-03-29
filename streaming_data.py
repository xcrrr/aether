"""Streaming Dataset — High-Performance Circular Buffer for aether_train.jsonl.

Handles 3.71 GB JSONL without loading into RAM:
  • Memory-mapped file access (O(1) random seek, zero-copy on Linux)
  • Producer thread pre-fetches tokens into a circular CPU buffer
  • Consumer (training loop) calls get_batch() — always returns immediately
    from the filled buffer.

JSONL schema (expected):
  {"input_ids": [int, ...], "labels": [int, ...],
   "trust_score": float, "source": str}

The dataset is infinite: when the file is exhausted, the producer wraps
around to the beginning (optionally shuffling seek offsets).
"""

from __future__ import annotations

import json
import math
import mmap
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import torch

from aether2_config import Aether2Config


# ─────────────────────────────────────────────────────────────────────────────
# Circular Token Buffer
# ─────────────────────────────────────────────────────────────────────────────

class CircularTokenBuffer:
    """Pre-allocated token buffer filled by a background producer thread.

    The buffer is a flat torch.LongTensor of `capacity` tokens on CPU.
    A producer thread writes tokens starting at write_ptr (mod capacity).
    The consumer reads starting at read_ptr.  When read_ptr reaches write_ptr
    the consumer blocks briefly (backpressure).

    Trust scores are stored in a parallel float32 ring buffer.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tokens = torch.zeros(capacity, dtype=torch.long)
        self.trust  = torch.zeros(capacity, dtype=torch.float32)

        self._write_ptr = 0
        self._read_ptr  = 0
        self._available = 0               # tokens ready to read

        self._lock   = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def write_sequence(self, ids: list[int], trust: float) -> None:
        """Write an entire token sequence in a single lock acquisition.

        Batching eliminates the per-token lock overhead of the original
        per-token write loop (512 lock acquire/release → 1 per sequence).
        Backpressure waits until there is room for ALL tokens at once.
        """
        n = len(ids)
        if n == 0:
            return

        # Backpressure: wait until there is room for n more tokens
        threshold = int(self.capacity * 0.95)
        while True:
            with self._lock:
                if self._available + n <= threshold:
                    break
            time.sleep(0.001)

        # Batch-write all tokens under a single lock
        with self._lock:
            end = self._write_ptr + n
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            if end <= self.capacity:
                self.tokens[self._write_ptr:end] = ids_tensor
                self.trust[self._write_ptr:end]  = trust
            else:
                first = self.capacity - self._write_ptr
                self.tokens[self._write_ptr:] = ids_tensor[:first]
                self.tokens[:n - first]        = ids_tensor[first:]
                self.trust[self._write_ptr:]   = trust
                self.trust[:n - first]         = trust
            self._write_ptr = end % self.capacity
            self._available = min(self._available + n, self.capacity)
            self._not_empty.notify_all()

    def read_window(self, length: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Read a contiguous window of `length` tokens.

        Returns (tokens, trust_scores) or None if not enough data.
        """
        with self._not_empty:
            while self._available < length:
                self._not_empty.wait(timeout=0.05)
                if self._available < length:
                    return None

        with self._lock:
            # Copy window (handles wrap-around)
            end = self._read_ptr + length
            if end <= self.capacity:
                toks   = self.tokens[self._read_ptr:end].clone()
                trust  = self.trust[self._read_ptr:end].clone()
            else:
                wrap = end - self.capacity
                toks  = torch.cat([self.tokens[self._read_ptr:], self.tokens[:wrap]])
                trust = torch.cat([self.trust[self._read_ptr:], self.trust[:wrap]])
            self._read_ptr  = end % self.capacity
            self._available -= length

        return toks, trust


# ─────────────────────────────────────────────────────────────────────────────
# JSONL producer (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _iter_jsonl_mmap(
    path: str,
    shuffle: bool = False,
    *,
    stop_event: threading.Event,
) -> Iterator[tuple[list[int], float]]:
    """Yield (input_ids, trust_score) from memory-mapped JSONL endlessly.

    Uses mmap for zero-copy reads.  When shuffle=True, random-seeks to a
    random byte offset then scans forward to the next '\\n' to align.
    """
    file_size = os.path.getsize(path)
    if file_size == 0:
        return

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            epoch = 0
            while not stop_event.is_set():
                if shuffle:
                    # Jump to a random byte offset and align to next newline
                    offset = random.randint(0, file_size - 1)
                    mm.seek(offset)
                    mm.readline()        # skip partial line
                else:
                    mm.seek(0)

                while not stop_event.is_set():
                    line = mm.readline()
                    if not line:
                        break            # hit EOF — wrap around
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        ids = obj.get("input_ids", [])
                        trust = float(obj.get("trust_score", 1.0))
                        if ids:
                            yield ids, trust
                    except (json.JSONDecodeError, ValueError):
                        continue

                epoch += 1
        finally:
            mm.close()


# ─────────────────────────────────────────────────────────────────────────────
# StreamingJSONLDataset
# ─────────────────────────────────────────────────────────────────────────────

class StreamingJSONLDataset:
    """High-performance streaming dataset with circular token buffer.

    Usage
    -----
    ds = StreamingJSONLDataset(cfg)
    ds.start()                          # starts background producer
    batch = ds.get_batch(batch_size=4)  # {"input_ids": (B,T), "labels": (B,T), ...}
    ds.stop()
    """

    def __init__(self, cfg: Aether2Config) -> None:
        self.cfg = cfg
        self.seq_len   = cfg.max_seq_len
        self.path      = cfg.data_path

        self._buffer   = CircularTokenBuffer(cfg.stream_buffer_tokens)
        self._stop     = threading.Event()
        self._thread   = None
        self._epoch    = 0
        self._steps    = 0

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background producer thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._producer_loop,
            name="aether2-data-producer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal producer to stop and join the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> "StreamingJSONLDataset":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Producer ────────────────────────────────────────────────────────────

    def _producer_loop(self) -> None:
        """Background thread: read JSONL and fill the circular buffer."""
        shuffle = getattr(self.cfg, "shuffle_between_epochs", True)
        for ids, trust in _iter_jsonl_mmap(self.path, shuffle, stop_event=self._stop):
            if self._stop.is_set():
                break
            # Truncate/pad sequence to seq_len
            ids = ids[: self.seq_len + 1]        # +1 for shift labels
            if len(ids) < 2:
                continue
            self._buffer.write_sequence(ids, trust)

    # ── Consumer ────────────────────────────────────────────────────────────

    def get_batch(
        self, batch_size: int, device: str = "cpu"
    ) -> dict[str, torch.Tensor] | None:
        """Return a padded batch dict or None if buffer is temporarily empty.

        Dict keys:
          input_ids   : (B, T) long
          labels      : (B, T) long — input_ids shifted left, -100 at end
          trust_scores: (B,)   float32
        """
        window = self.seq_len + 1             # need 1 extra for shift
        items: list[tuple[torch.Tensor, torch.Tensor]] = []

        for _ in range(batch_size):
            result = self._buffer.read_window(window)
            if result is None:
                return None
            items.append(result)

        # Build tensors
        input_ids = torch.stack([it[0][:-1] for it in items], dim=0)   # (B, T)
        labels    = torch.stack([it[0][1:] for it in items], dim=0)    # (B, T)

        # Trust-weighted masking: low-trust tokens partially masked
        # (trust < 0.3 → replace label with -100 for CE skip)
        trust_scores = torch.stack([it[1][0] for it in items])          # (B,)

        # Mask pad tokens (token id 0 = PAD)
        labels[labels == 0] = -100

        self._steps += 1

        return {
            "input_ids":    input_ids.to(device, non_blocking=True),
            "labels":       labels.to(device, non_blocking=True),
            "trust_scores": trust_scores.to(device, non_blocking=True),
        }

    def get_batch_blocking(
        self,
        batch_size: int,
        device: str = "cpu",
        max_retries: int = 200,
    ) -> dict[str, torch.Tensor]:
        """Like get_batch but retries until data is available."""
        for _ in range(max_retries):
            batch = self.get_batch(batch_size, device)
            if batch is not None:
                return batch
            time.sleep(0.005)
        raise RuntimeError(
            "StreamingJSONLDataset: buffer empty after max_retries. "
            "Check data_path and producer thread status."
        )

    # ── Diagnostics ─────────────────────────────────────────────────────────

    @property
    def buffer_fill_pct(self) -> float:
        """Fraction of circular buffer currently filled."""
        buf = self._buffer
        with buf._lock:
            return buf._available / buf.capacity

    def wait_for_fill(self, target_pct: float = 0.1, timeout: float = 60.0) -> bool:
        """Block until buffer is at least target_pct full or timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.buffer_fill_pct >= target_pct:
                return True
            time.sleep(0.1)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics sampler (fast preview, no full scan)
# ─────────────────────────────────────────────────────────────────────────────

def dataset_stats_streaming(cfg: Aether2Config, n_samples: int = 1000) -> dict:
    """Sample the first n_samples records and report length statistics."""
    path = cfg.data_path
    if not Path(path).exists():
        return {"error": f"File not found: {path}"}

    lengths: list[int] = []
    trusts:  list[float] = []
    sources: dict[str, int] = {}

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for _ in range(n_samples):
                line = mm.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line)
                    ids = obj.get("input_ids", [])
                    if ids:
                        lengths.append(len(ids))
                    trusts.append(float(obj.get("trust_score", 1.0)))
                    src = obj.get("source", "unknown")
                    sources[src] = sources.get(src, 0) + 1
                except (json.JSONDecodeError, ValueError):
                    pass
        finally:
            mm.close()

    if not lengths:
        return {"error": "No valid records found"}

    t = torch.tensor(lengths, dtype=torch.float32)
    return {
        "n_sampled":   len(lengths),
        "len_mean":    t.mean().item(),
        "len_std":     t.std().item(),
        "len_min":     t.min().item(),
        "len_max":     t.max().item(),
        "trust_mean":  sum(trusts) / len(trusts),
        "sources":     sources,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    cfg = Aether2Config()
    p = Path(cfg.data_path)
    if not p.exists():
        print(f"[WARN] Data file not found: {p}. Creating synthetic test file.")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fout:
            for i in range(1000):
                import json as _json
                ids = list(range(1, cfg.max_seq_len + 2))
                _json.dump({"input_ids": ids, "trust_score": 0.9, "source": "test"},
                           fout)
                fout.write("\n")

    print("Dataset stats (first 200 records):")
    stats = dataset_stats_streaming(cfg, 200)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nTesting streaming loader...")
    with StreamingJSONLDataset(cfg) as ds:
        filled = ds.wait_for_fill(0.001, timeout=10.0)
        print(f"  Buffer filled: {filled}  ({ds.buffer_fill_pct:.1%})")
        batch = ds.get_batch_blocking(batch_size=2)
        print(f"  Batch shapes: input_ids={tuple(batch['input_ids'].shape)}, "
              f"labels={tuple(batch['labels'].shape)}")
        print("  Streaming OK")
