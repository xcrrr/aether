"""Tests for streaming data pipeline (streaming_data.py).

Tests the CircularTokenBuffer and batch construction independently
of the JSONL producer thread (no file I/O required).
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import torch
import pytest

from streaming_data import CircularTokenBuffer, StreamingJSONLDataset


# ─────────────────────────────────────────────────────────────────────────────
# CircularTokenBuffer
# ─────────────────────────────────────────────────────────────────────────────

class TestCircularTokenBuffer:
    """Thread-safe circular token buffer tests."""

    def test_write_then_read(self):
        """Basic write/read cycle should return the same tokens."""
        buf = CircularTokenBuffer(capacity=100)
        buf.write_sequence([10, 20, 30], trust=0.9)

        result = buf.read_window(3)
        assert result is not None

        tokens, trust = result
        assert tokens.tolist() == [10, 20, 30]
        # Trust should be 0.9 for all positions
        assert (trust == 0.9).all()

    def test_multiple_writes(self):
        """Multiple write_sequence calls should concatenate in the buffer."""
        buf = CircularTokenBuffer(capacity=100)
        buf.write_sequence([1, 2, 3], trust=0.8)
        buf.write_sequence([4, 5, 6], trust=0.9)

        result = buf.read_window(6)
        assert result is not None
        tokens, _ = result
        assert tokens.tolist() == [1, 2, 3, 4, 5, 6]

    def test_wrap_around(self):
        """Reads should work correctly across the buffer boundary."""
        buf = CircularTokenBuffer(capacity=8)

        # Fill most of the buffer
        buf.write_sequence([1, 2, 3, 4, 5, 6], trust=1.0)
        buf.read_window(6)   # consume, advancing read_ptr to 6

        # Write across the wrap boundary (positions 6, 7, 0, 1)
        buf.write_sequence([7, 8, 9, 10], trust=1.0)
        result = buf.read_window(4)

        assert result is not None
        tokens, _ = result
        assert tokens.tolist() == [7, 8, 9, 10]

    def test_read_returns_none_when_empty(self):
        """Reading from empty buffer should return None (not block forever)."""
        buf = CircularTokenBuffer(capacity=100)
        result = buf.read_window(10)
        assert result is None, "Empty buffer should return None"

    def test_fill_percentage(self):
        """Buffer fill tracking should be correct."""
        buf = CircularTokenBuffer(capacity=100)
        assert buf._available == 0

        buf.write_sequence(list(range(50)), trust=1.0)
        assert buf._available == 50


# ─────────────────────────────────────────────────────────────────────────────
# StreamingJSONLDataset (with synthetic file)
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamingDataset:
    """Integration tests using a small synthetic JSONL file."""

    @pytest.fixture
    def synthetic_jsonl(self, small_cfg) -> str:
        """Create a temporary JSONL file with valid records."""
        seq_len = small_cfg.max_seq_len
        records = []
        for i in range(100):
            # +1 for the shift-label offset
            ids = list(range(1, seq_len + 2))
            records.append({
                "input_ids": ids,
                "trust_score": 0.9,
                "source": "test",
            })

        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "test_data.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        return str(path)

    def test_batch_keys(self, small_cfg, synthetic_jsonl):
        """Batch dict should have input_ids, labels, trust_scores."""
        small_cfg.data_path = synthetic_jsonl
        small_cfg.stream_buffer_tokens = 10_000

        ds = StreamingJSONLDataset(small_cfg)
        ds.start()
        try:
            filled = ds.wait_for_fill(0.001, timeout=10.0)
            assert filled, "Dataset should fill within timeout"

            batch = ds.get_batch_blocking(batch_size=2, max_retries=50)
            assert "input_ids" in batch
            assert "labels" in batch
            assert "trust_scores" in batch
        finally:
            ds.stop()

    def test_batch_shapes(self, small_cfg, synthetic_jsonl):
        """Batch tensors should have correct shapes."""
        small_cfg.data_path = synthetic_jsonl
        small_cfg.stream_buffer_tokens = 10_000

        ds = StreamingJSONLDataset(small_cfg)
        ds.start()
        try:
            ds.wait_for_fill(0.001, timeout=10.0)
            batch = ds.get_batch_blocking(batch_size=4, max_retries=50)

            B = 4
            T = small_cfg.max_seq_len
            assert batch["input_ids"].shape == (B, T), \
                f"input_ids shape: expected ({B}, {T}), got {batch['input_ids'].shape}"
            assert batch["labels"].shape == (B, T), \
                f"labels shape: expected ({B}, {T}), got {batch['labels'].shape}"
            assert batch["trust_scores"].shape == (B,), \
                f"trust_scores shape: expected ({B},), got {batch['trust_scores'].shape}"
        finally:
            ds.stop()

    def test_labels_are_shifted(self, small_cfg, synthetic_jsonl):
        """Labels should be input_ids shifted left by 1 position."""
        small_cfg.data_path = synthetic_jsonl
        small_cfg.stream_buffer_tokens = 10_000

        ds = StreamingJSONLDataset(small_cfg)
        ds.start()
        try:
            ds.wait_for_fill(0.001, timeout=10.0)
            batch = ds.get_batch_blocking(batch_size=2, max_retries=50)

            input_ids = batch["input_ids"]
            labels = batch["labels"]

            # For non-padding, non-masked positions, labels[t] should equal
            # the token that follows input_ids[t] in the original sequence.
            # Since our test data is sequential (1, 2, 3, ...), labels
            # should be one ahead of input_ids where not masked.
            for b in range(2):
                for t in range(labels.shape[1]):
                    if labels[b, t].item() != -100 and labels[b, t].item() != 0:
                        # Label should be input_ids + 1 in our sequential data
                        assert labels[b, t].item() == input_ids[b, t].item() + 1, \
                            f"Labels should be shifted: label={labels[b, t]}, " \
                            f"input={input_ids[b, t]}"
                        break  # just check the first valid position
        finally:
            ds.stop()

    def test_context_manager(self, small_cfg, synthetic_jsonl):
        """Context manager should start/stop cleanly."""
        small_cfg.data_path = synthetic_jsonl
        small_cfg.stream_buffer_tokens = 10_000

        with StreamingJSONLDataset(small_cfg) as ds:
            ds.wait_for_fill(0.001, timeout=10.0)
            batch = ds.get_batch_blocking(batch_size=1, max_retries=50)
            assert batch is not None

        # After exiting, thread should be stopped
        assert ds._stop.is_set(), "Stop event should be set after exit"
