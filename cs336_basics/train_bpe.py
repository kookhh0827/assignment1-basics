import os
import regex as re
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently, aligning
    boundaries to the next occurrence of any special token so we never split
    a special token across chunks.

    May return fewer than desired_num_chunks + 1 boundaries if multiple
    boundaries collapse to the same location.
    """
    assert all(isinstance(t, (bytes, bytearray)) for t in split_special_tokens), (
        "Must represent special tokens as bytestrings"
    )

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = max(1, file_size // desired_num_chunks)
    # Start with uniformly spaced guesses; final boundary is end-of-file
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    mini_chunk_size = 4096

    if not split_special_tokens:
        # No alignment needed if there are no special tokens to respect
        return sorted(set(boundaries))

    for bi in range(1, len(boundaries) - 1):
        initial_pos = boundaries[bi]
        file.seek(initial_pos)
        current_seek = initial_pos
        while True:
            buf = file.read(mini_chunk_size)
            if buf == b"":
                boundaries[bi] = file_size
                break

            # Find earliest next occurrence of any special token within buf
            found_positions = []
            for tok in split_special_tokens:
                pos = buf.find(tok)
                if pos != -1:
                    found_positions.append(current_seek + pos)

            if found_positions:
                boundaries[bi] = min(found_positions)
                break

            current_seek += mini_chunk_size

    return sorted(set(boundaries))


def _compile_special_split_regex(special_tokens: list[str]) -> re.Pattern[str]:
    """
    Build a regex that matches any special token. We escape each token so
    metacharacters like '|' are treated literally.
    """
    if not special_tokens:
        # Regex that never matches
        return re.compile(r"(?!x)x")
    escaped = [re.escape(tok) for tok in special_tokens]
    return re.compile("|".join(escaped))


def _split_text_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split on any occurrence of the provided special tokens and return only
    the non-empty text segments (special tokens are removed).
    """
    split_re = _compile_special_split_regex(special_tokens)
    parts = split_re.split(text)
    return [p for p in parts if p and p not in special_tokens]


def _split_chunk_worker(args: tuple[bytes, list[str]]) -> list[str]:
    chunk_bytes, special_tokens = args
    text = chunk_bytes.decode("utf-8", errors="ignore")
    return _split_text_on_special_tokens(text, special_tokens)


def _pretokenize_count_worker(args: tuple[list[str], str]) -> dict[tuple[bytes, ...], int]:
    """
    Worker that takes a list of text segments and a regex pattern string,
    tokenizes with finditer, and returns counts as a dict mapping
    tuple[bytes,...] to int.
    """
    segments, pattern_str = args
    pat = re.compile(pattern_str)
    from collections import Counter
    local_counts: Counter[tuple[bytes, ...]] = Counter()
    for segment in segments:
        for m in pat.finditer(segment):
            token = m.group(0)
            b = token.encode('utf-8')
            token_tuple = tuple(bytes([x]) for x in b)
            local_counts[token_tuple] += 1
    return dict(local_counts)


def split_chunks_on_special_tokens(
    input_path: str,
    boundaries: list[int],
    special_tokens: list[str],
    num_workers: int | None = None,
) -> list[list[str]]:
    """
    Read each chunk (start,end) defined by boundaries and split on special tokens.
    If num_workers > 1, process chunks in parallel using a process pool.
    Returns a nested list of segments: one list per original chunk, preserving order.
    """
    if not boundaries or len(boundaries) < 2:
        return []

    # Gather chunk byte buffers first to avoid interleaved file seeks across processes
    chunk_buffers: list[bytes] = []
    with open(input_path, "rb") as fbin:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            fbin.seek(start)
            chunk_buffers.append(fbin.read(end - start))

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1))

    if num_workers > 1 and len(chunk_buffers) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            results: list[list[str]] = list(
                ex.map(_split_chunk_worker, [(buf, special_tokens) for buf in chunk_buffers])
            )
    else:
        results = [_split_chunk_worker((buf, special_tokens)) for buf in chunk_buffers]

    return results


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    '''
    Train a BPE tokenizer on the input data.
    '''
    # Choose a reasonable number of chunks based on available CPUs
    num_processes = max(1, os.cpu_count() or 1)

    # 1) ===== Compute chunk boundaries aligned to special tokens =====
    special_tokens_bytes = [t.encode("utf-8") for t in special_tokens]
    with open(input_path, "rb") as fbin:
        boundaries = find_chunk_boundaries(fbin, num_processes, special_tokens_bytes)

    # Split each chunk on special tokens so no merges cross boundaries
    segments_per_chunk = split_chunks_on_special_tokens(
        input_path=input_path,
        boundaries=boundaries,
        special_tokens=special_tokens,
        num_workers=num_processes,
    )

    # Compile pre-tokenization pattern string (compiled in workers)
    pretoken_pattern_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 2) ===== Parallel pre-tokenization counting per chunk =====
    from collections import Counter
    token_counts: Counter[tuple[bytes, ...]] = Counter()
    if num_processes > 1 and len(segments_per_chunk) > 1:
        with ProcessPoolExecutor(max_workers=num_processes) as ex:
            partials = list(
                ex.map(
                    _pretokenize_count_worker,
                    [(chunk_segments, pretoken_pattern_str) for chunk_segments in segments_per_chunk],
                )
            )
        for d in partials:
            token_counts.update(d)
    else:
        # Fallback to serial counting
        pat = re.compile(pretoken_pattern_str)
        for chunk_segments in segments_per_chunk:
            for segment in chunk_segments:
                for m in pat.finditer(segment):
                    token = m.group(0)
                    b = token.encode('utf-8')
                    token_tuple = tuple(bytes([x]) for x in b)
                    token_counts[token_tuple] += 1

    # 3) ===== BPE merge loop =====
    def compute_pair_counts(tokens_counter: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
        pair_counts_local: dict[tuple[bytes, bytes], int] = {}
        for token_tuple, freq in tokens_counter.items():
            if freq <= 0 or len(token_tuple) < 2:
                continue
            # accumulate adjacent pairs
            prev = token_tuple[0]
            for cur in token_tuple[1:]:
                pair = (prev, cur)
                pair_counts_local[pair] = pair_counts_local.get(pair, 0) + freq
                prev = cur
        return pair_counts_local

    def apply_merge_to_token(token_tuple: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        a, b = pair
        merged: list[bytes] = []
        i = 0
        n = len(token_tuple)
        while i < n:
            if i + 1 < n and token_tuple[i] == a and token_tuple[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(token_tuple[i])
                i += 1
        return tuple(merged)

    def apply_merge(tokens_counter: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
        new_counter: dict[tuple[bytes, ...], int] = {}
        for token_tuple, freq in tokens_counter.items():
            if freq <= 0:
                continue
            new_tuple = apply_merge_to_token(token_tuple, pair)
            new_counter[new_tuple] = new_counter.get(new_tuple, 0) + freq
        return new_counter

    merges: list[tuple[bytes, bytes]] = []
    max_merges = max(0, vocab_size - 256 - len(special_tokens))
    for _ in range(max_merges):
        pair_counts = compute_pair_counts(token_counts)
        if not pair_counts:
            break
        # pick best by (count, lexicographic pair)
        best_pair, best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        if best_count <= 0:
            break
        merges.append(best_pair)
        token_counts = apply_merge(token_counts, best_pair)

    # 4) ===== Build vocabulary =====
    # Validate capacity: special tokens + 256 byte tokens must fit
    if vocab_size < len(special_tokens) + 256:
        raise ValueError(
            f"vocab_size={vocab_size} too small for {len(special_tokens)} special tokens + 256 byte tokens"
        )
    vocab: dict[int, bytes] = {}
    
    # Special tokens first, in provided order
    for special_index, token_str in enumerate(special_tokens):
        vocab[special_index] = token_str.encode("utf-8")
    
    # Raw byte tokens next (0..255)
    base_index = len(special_tokens)
    for byte_value in range(256):
        vocab[base_index + byte_value] = bytes([byte_value])
    
    # Learned merge tokens until we reach vocab_size
    merges_capacity = vocab_size - (base_index + 256)
    num_merges_to_take = min(merges_capacity, len(merges))
    for merge_index in range(num_merges_to_take):
        a, b = merges[merge_index]
        vocab[base_index + 256 + merge_index] = a + b

    return vocab, merges

if __name__ == "__main__":
    train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])