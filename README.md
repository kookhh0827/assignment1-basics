# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Implementation Overview (`cs336_basics/`)

- **`train_bpe.py`**: BPE training utilities and pipeline
  - Splits the file into CPU-aligned chunks and aligns boundaries to the next special token (`find_chunk_boundaries`).
  - Splits each chunk by special tokens, then counts pre-tokenized segments in parallel.
  - Greedy adjacent-pair merging loop to produce `merges` (ties broken lexicographically).
  - Final vocab order: special tokens → 256 raw byte tokens → learned merge tokens.
  - Returns `(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]])`.

- **`tokenizer.py`**: Byte-level BPE tokenizer
  - Uses the GPT‑2 pre-tokenization regex, then applies BPE merges.
  - **Special-token aware**: splits input on provided special tokens and preserves their IDs.
  - `Tokenizer.from_files(...)` loads GPT‑2 style `vocab.json`/`merges.txt` (with byte↔unicode mapping).
  - `encode`/`encode_iterable` handle streaming inputs without breaking specials; `decode` restores UTF‑8.

- **`nn_modules.py`**: Core neural modules
  - `Linear`: no bias, `trunc_normal_` init (σ²=2/(d_in+d_out), ±3σ), `einops.einsum` matmul.
  - `Embedding`: `trunc_normal_` init; returns embeddings via index lookup.
  - `RMSNorm`: normalize by RMS then apply learned scale.
  - `SiLU`: x·sigmoid(x).
  - `SwiGLU`: `W2(SiLU(W1x) ⊙ W3x)` form.
  - `RotaryPositionalEmbedding` (RoPE): caches cos/sin with `register_buffer(persistent=False)`, expands on demand, rotates even/odd dims.
  - `softmax`: numerically stable (subtract max).
  - `scaled_dot_product_attention`: mask uses True=allowed/False=masked; efficient with `einops`.
  - `MultiheadSelfAttention`: QKV projections → head split → optional RoPE → causal mask → output projection.
  - `TransformerBlock`: pre-norm residual (Attention, FFN).
  - `TransformerLM`: token embeddings → L blocks → final `RMSNorm` → `Linear` logits.

- **`optimizer.py`**: `AdamW`
  - Standard Adam moments with bias correction and decoupled weight decay (`-lr*wd` applied directly to params). No sparse grads.

- **`scheduler.py`**: cosine scheduler with linear warmup
  - Implements `cosine_annealing_with_linear_warmup(t, eta_max, eta_min, Tw, Tc)`.

- **`loss.py`**: cross entropy
  - Treats last dim as vocab; stable `logsumexp` formulation; returns mean loss.

- **`clipping.py`**: gradient clipping
  - Computes global L2 norm across all parameter grads and scales by `max_l2_norm`.

- **`dataloader.py`**: batch sampling
  - Samples contiguous windows from a 1D token array to produce `(inputs, targets)` on the requested device.

- **`checkpoint.py`**: checkpoint I/O
  - `save_checkpoint`/`load_checkpoint` for model/optimizer states and iteration.


