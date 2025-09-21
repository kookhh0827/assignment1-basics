from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import regex as re
from tests.common import gpt2_bytes_to_unicode



def gpt2_unicode_to_bytes() -> dict[str, int]:
    mapping = gpt2_bytes_to_unicode()
    return {v: k for k, v in mapping.items()}


class Tokenizer:
    """
    Byte-level BPE tokenizer with special-token support.

    - vocab: dict[int, bytes] mapping token id -> token bytes
    - merges: list[tuple[bytes, bytes]] merge rules in creation order
    - special_tokens: optional list[str]; if provided, ensure they are present in vocab
    """

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.id_to_bytes: dict[int, bytes] = dict(vocab)
        self.bytes_to_id: dict[bytes, int] = {b: i for i, b in self.id_to_bytes.items()}

        # Ensure user-provided special tokens are in the vocabulary
        if special_tokens:
            next_id = max(self.id_to_bytes.keys()) + 1 if self.id_to_bytes else 0
            for tok in special_tokens:
                tok_b = tok.encode("utf-8")
                if tok_b not in self.bytes_to_id:
                    self.id_to_bytes[next_id] = tok_b
                    self.bytes_to_id[tok_b] = next_id
                    next_id += 1

        # Merge ranks: earlier merges have higher priority (lower rank value)
        self.merges: list[tuple[bytes, bytes]] = merges
        self.ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}

        # Pre-tokenization pattern (GPT-2)
        self._pretoken_pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Special token splitting regex: capture specials so we can keep them
        self.special_tokens: list[str] = special_tokens or []
        if self.special_tokens:
            # Sort by length desc to ensure longer overlapping specials match first
            specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(t) for t in specials_sorted]
            self._special_split_re = re.compile("(" + "|".join(escaped) + ")")
        else:
            # Regex that never matches
            self._special_split_re = re.compile(r"(?!x)x")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | bytes | Path,
        merges_filepath: str | bytes | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Load a tokenizer from files. Supports GPT-2-style reference files:
        - vocab JSON mapping string token -> index (like train-bpe-reference-vocab.json)
        - merges TXT with lines: "token1 token2" using GPT-2 byte->unicode representation
        """
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)

        # Load vocab
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        # Convert to id->bytes using GPT-2 mapping
        uni2byte = gpt2_unicode_to_bytes()
        id_to_bytes: dict[int, bytes] = {}
        for token_str, idx in vocab_json.items():
            token_bytes = bytes([uni2byte[ch] for ch in token_str])
            id_to_bytes[int(idx)] = token_bytes

        # Load merges
        merges: list[tuple[bytes, bytes]] = []
        with merges_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                a_str, b_str = line.split(" ")
                a_b = bytes([uni2byte[ch] for ch in a_str])
                b_b = bytes([uni2byte[ch] for ch in b_str])
                merges.append((a_b, b_b))

        return cls(id_to_bytes, merges, special_tokens=special_tokens)

    def _encode_segment_to_ids(self, segment: str) -> list[int]:
        """Encode a non-special segment: pretokenize then apply BPE merges per pretoken."""
        ids: list[int] = []
        for m in self._pretoken_pat.finditer(segment):
            token = m.group(0)
            sym_list = [bytes([b]) for b in token.encode("utf-8")]
            # Greedy merging per ranks
            if len(sym_list) >= 2 and self.ranks:
                while True:
                    best_index: int | None = None
                    best_rank: int | None = None
                    for i in range(len(sym_list) - 1):
                        pair = (sym_list[i], sym_list[i + 1])
                        r = self.ranks.get(pair)
                        if r is not None and (best_rank is None or r < best_rank):
                            best_rank = r
                            best_index = i
                    if best_index is None:
                        break
                    i = best_index
                    merged = sym_list[i] + sym_list[i + 1]
                    sym_list = sym_list[:i] + [merged] + sym_list[i + 2:]
            for sym in sym_list:
                ids.append(self.bytes_to_id[sym])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode input text into a sequence of token IDs, respecting special tokens."""
        if not text:
            return []
        out: list[int] = []
        parts = self._special_split_re.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                out.extend(self._encode_segment_to_ids(part))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily yield token IDs from an iterable of input strings.
        Ensures we don't split tokens across chunk boundaries by retaining
        the trailing non-special segment as buffer between chunks.
        """
        buffer = ""
        for chunk in iterable:
            if not chunk:
                continue
            buffer += chunk

            parts = self._special_split_re.split(buffer)
            if not parts:
                continue

            # Keep the trailing non-special text in buffer; process the rest now
            trailing = parts[-1]
            to_process = parts[:-1]

            for part in to_process:
                if not part:
                    continue
                if part in self.special_tokens:
                    yield self.bytes_to_id[part.encode("utf-8")]
                else:
                    for tid in self._encode_segment_to_ids(part):
                        yield tid

            buffer = trailing

        # Flush remaining buffer at end
        if buffer:
            parts = self._special_split_re.split(buffer)
            for part in parts:
                if not part:
                    continue
                if part in self.special_tokens:
                    yield self.bytes_to_id[part.encode("utf-8")]
                else:
                    for tid in self._encode_segment_to_ids(part):
                        yield tid

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        if not ids:
            return ""
        buf = bytearray()
        for i in ids:
            buf.extend(self.id_to_bytes[i])
        # Replace malformed sequences with U+FFFD
        return buf.decode("utf-8", errors="replace")


__all__ = ["Tokenizer"]


