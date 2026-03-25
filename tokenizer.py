#!/usr/bin/env python3
"""Omega Tokenizer — Byte Pair Encoding from scratch.

vocab_size = 16 000 (matches OmegaConfig.vocab_size exactly)
Special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3

Pre-tokenization splits text into "words" at whitespace and symbol
boundaries. BPE merges are learned within word boundaries only.
Unicode identifiers are handled natively (character-level, not byte-level).
All encoding is deterministic — same input always produces same output.

CLI:
  python tokenizer.py --train corpus.txt --output omega_tokenizer.json --vocab-size 16000
  python tokenizer.py --load omega_tokenizer.json --test "def hello(x: int) -> str: return str(x)"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ── Special tokens ────────────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"    # id=0  padding
UNK_TOKEN = "<UNK>"    # id=1  unknown
BOS_TOKEN = "<BOS>"    # id=2  beginning-of-sequence
EOS_TOKEN = "<EOS>"    # id=3  end-of-sequence

SPECIAL_TOKENS: list[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS:    dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}


# ── Pre-tokenization ──────────────────────────────────────────────────────────
#
# Words are: identifiers/numbers | runs of spaces/tabs | single newlines |
#            multi-char operators | single punctuation chars.
# This preserves indentation structure (crucial for Python).
#
_WORD_RE = re.compile(
    r"""(?x)
    [A-Za-z_]\w*                       # identifier / keyword
    | 0[xX][0-9a-fA-F]+               # hex integer
    | 0[bB][01]+                       # binary integer
    | 0[oO][0-7]+                      # octal integer
    | \d+(?:\.\d*)?(?:[eE][+-]?\d+)?  # decimal / float
    | ->|::|\.\.\.|\*\*|//             # multi-char operators (part 1)
    | <<=|>>=|<<|>>                    # shift operators
    | [+\-*/%&|^]=|\*\*=|//=          # augmented assignment
    | ==|!=|<=|>=|:=                   # comparison / walrus
    | [ \t]+                           # spaces/tabs (indentation unit)
    | \r?\n                            # newline
    | .                                # any other single character
    """
)


def _pretokenize(text: str) -> list[str]:
    """Split *text* into pre-token words suitable for BPE training/encoding."""
    return [m.group(0) for m in _WORD_RE.finditer(text)]


# ── BPE core helpers ──────────────────────────────────────────────────────────

def _count_pairs(
    vocab: dict[tuple[str, ...], int],
) -> Counter[tuple[str, str]]:
    """Count adjacent pair frequencies across all words, weighted by frequency."""
    counts: Counter[tuple[str, str]] = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def _merge_pair(
    vocab: dict[tuple[str, ...], int],
    pair: tuple[str, str],
) -> dict[tuple[str, ...], int]:
    """Apply one BPE merge to the entire vocabulary."""
    merged = pair[0] + pair[1]
    new_vocab: dict[tuple[str, ...], int] = {}
    for word, freq in vocab.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


# ── OmegaTokenizer ────────────────────────────────────────────────────────────

class OmegaTokenizer:
    """BPE tokenizer for AetherOmega.

    Vocabulary layout:
        ID 0 → <PAD>   padding
        ID 1 → <UNK>   unknown character / out-of-vocab token
        ID 2 → <BOS>   beginning-of-sequence
        ID 3 → <EOS>   end-of-sequence
        ID 4+ → learned BPE tokens (single characters then merges)

    All methods are deterministic after training.  No internal randomness.
    """

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._merges: list[tuple[str, str]] = []  # ordered merge list
        self._vocab_size: int = 0
        self._trained: bool = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, texts: list[str], vocab_size: int = 16_000) -> None:
        """Train BPE on *texts*.

        Args:
            texts:      List of training strings (Python source code etc.).
            vocab_size: Target vocabulary size.  Must be ≥ 4 (special tokens).
        """
        if vocab_size < 4:
            raise ValueError(f"vocab_size must be ≥ 4; got {vocab_size}")

        print(f"[tokenizer] Training BPE on {len(texts)} texts  target_vocab={vocab_size}")

        # ── Step 1: word frequency count ──────────────────────────────────────
        word_counter: Counter[str] = Counter()
        for text in texts:
            word_counter.update(_pretokenize(text))

        # ── Step 2: build initial character vocabulary ─────────────────────────
        char_vocab: set[str] = set()
        for word in word_counter:
            char_vocab.update(word)

        vocab_list: list[str] = list(SPECIAL_TOKENS)
        for ch in sorted(char_vocab):
            if ch not in SPECIAL_IDS:
                vocab_list.append(ch)

        # ── Step 3: initialise BPE word representations ───────────────────────
        bpe_vocab: dict[tuple[str, ...], int] = {}
        for word, freq in word_counter.items():
            key = tuple(word)
            bpe_vocab[key] = bpe_vocab.get(key, 0) + freq

        # ── Step 4: BPE merge loop ─────────────────────────────────────────────
        merges: list[tuple[str, str]] = []
        n_merges = vocab_size - len(vocab_list)

        if n_merges <= 0:
            print(f"[tokenizer] Warning: char-level vocab ({len(vocab_list)}) "
                  f"already ≥ target ({vocab_size}); no merges performed.")
        else:
            for merge_idx in range(n_merges):
                pair_counts = _count_pairs(bpe_vocab)
                if not pair_counts:
                    print(f"[tokenizer] No more pairs; stopping at {merge_idx} merges.")
                    break

                # Best pair: most frequent; tie-break: lexicographic for determinism
                best = max(pair_counts, key=lambda p: (pair_counts[p], p))
                if pair_counts[best] < 2:
                    print(f"[tokenizer] All pairs have freq < 2; stopping at {merge_idx}.")
                    break

                merged_tok = best[0] + best[1]
                merges.append(best)
                vocab_list.append(merged_tok)
                bpe_vocab = _merge_pair(bpe_vocab, best)

                if (merge_idx + 1) % 1000 == 0:
                    print(f"[tokenizer]   {merge_idx + 1:>6} merges  vocab={len(vocab_list)}")

        # ── Step 5: build bidirectional mappings ──────────────────────────────
        self._token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
        self._id_to_token = dict(enumerate(vocab_list))
        self._merges = merges
        self._vocab_size = len(vocab_list)
        self._trained = True

        print(f"[tokenizer] Done: vocab_size={self._vocab_size}  merges={len(merges)}")

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _apply_bpe(self, word: str) -> list[str]:
        """Apply all BPE merges to a single pre-token word."""
        if not word:
            return []
        tokens: list[str] = list(word)
        if len(tokens) == 1:
            return tokens
        for pair in self._merges:
            if len(tokens) < 2:
                break
            merged = pair[0] + pair[1]
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == pair[0]
                        and tokens[i + 1] == pair[1]):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode *text* to a list of token IDs.

        Args:
            text:    Input string.
            add_bos: Prepend BOS token (id=2).
            add_eos: Append EOS token (id=3).

        Returns:
            Deterministic list of integer token IDs.
        """
        if not self._trained:
            raise RuntimeError(
                "Tokenizer not trained. Call .train() or .load() first."
            )
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for word in _pretokenize(text):
            for tok in self._apply_bpe(word):
                ids.append(self._token_to_id.get(tok, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: list[int],
        skip_special: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        Args:
            ids:           List of token IDs.
            skip_special:  If True, omit PAD/UNK/BOS/EOS from output.

        Returns:
            Decoded text string.
        """
        skip = {self.pad_id, self.unk_id, self.bos_id, self.eos_id} if skip_special else set()
        parts: list[str] = []
        for id_ in ids:
            if id_ in skip:
                continue
            parts.append(self._id_to_token.get(id_, ""))
        return "".join(parts)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save tokenizer to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab":      self._token_to_id,
            "merges":     [list(m) for m in self._merges],
            "vocab_size": self._vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[tokenizer] Saved → {path}  (vocab_size={self._vocab_size})")

    def load(self, path: str | Path) -> None:
        """Load tokenizer from a JSON file written by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._token_to_id = {k: int(v) for k, v in data["vocab"].items()}
        self._id_to_token = {int(v): k for k, v in data["vocab"].items()}
        self._merges      = [tuple(m) for m in data["merges"]]
        self._vocab_size  = data["vocab_size"]
        self._trained     = True

    @classmethod
    def from_file(cls, path: str | Path) -> "OmegaTokenizer":
        """Create and load tokenizer in a single call."""
        tok = cls()
        tok.load(path)
        return tok


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Omega BPE Tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tokenizer.py --train corpus.txt --output omega_tokenizer.json\n"
            "  python tokenizer.py --load omega_tokenizer.json "
            "--test 'def hello(): pass'\n"
        ),
    )
    p.add_argument("--train",       type=str, default=None,
                   help="Path to corpus text file for training")
    p.add_argument("--output",      type=str, default="omega_tokenizer.json",
                   help="Output path for trained tokenizer (default: omega_tokenizer.json)")
    p.add_argument("--vocab-size",  type=int, default=16_000,
                   help="Target vocabulary size (default: 16000)")
    p.add_argument("--load",        type=str, default=None,
                   help="Load existing tokenizer from file")
    p.add_argument("--test",        type=str, default=None,
                   help="Test string to encode and decode")
    args = p.parse_args()

    if args.train:
        corpus_path = Path(args.train)
        if not corpus_path.exists():
            print(f"[error] Corpus file not found: {corpus_path}", file=sys.stderr)
            sys.exit(1)
        corpus_text = corpus_path.read_text(encoding="utf-8")
        # Split into non-empty lines as training units
        texts = [ln for ln in corpus_text.splitlines() if ln.strip()]
        print(f"[tokenizer] Loaded corpus: {len(texts)} lines, "
              f"{len(corpus_text):,} chars")
        tok = OmegaTokenizer()
        tok.train(texts, vocab_size=args.vocab_size)
        tok.save(args.output)

        if args.test:
            _run_test(tok, args.test)
        return

    if args.load:
        tok = OmegaTokenizer.from_file(args.load)
        print(f"[tokenizer] Loaded: vocab_size={tok.vocab_size}")
        if args.test:
            _run_test(tok, args.test)
        return

    p.print_help()


def _run_test(tok: OmegaTokenizer, text: str) -> None:
    ids     = tok.encode(text)
    decoded = tok.decode(ids)
    rt_ok   = decoded == text
    print(f"\nTest encode : {text!r}")
    print(f"  Token count : {len(ids)}")
    print(f"  Token IDs   : {ids[:32]}{'...' if len(ids) > 32 else ''}")
    print(f"  Tokens      : {[tok._id_to_token.get(i, '?') for i in ids[:32]]}"
          f"{'...' if len(ids) > 32 else ''}")
    print(f"  Decoded     : {decoded!r}")
    print(f"  Round-trip  : {'OK' if rt_ok else 'MISMATCH'}")
    if not rt_ok:
        # Show first difference
        for i, (a, b) in enumerate(zip(text, decoded)):
            if a != b:
                print(f"  First diff at char {i}: {a!r} vs {b!r}")
                break


if __name__ == "__main__":
    _cli()
