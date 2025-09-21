from torch import nn
import torch
import math
# einops
import einops

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        # Linear weights: N μ = 0, σ2 = 2 / (d_in + d_out) truncated at [−3σ, 3σ].
        # use torch.nn.init.trunc_normal_
        # we don't need bias
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_out, d_in, device=device, dtype=dtype), mean=0, std=math.sqrt(2 / (d_in + d_out)), a=-3*math.sqrt(2 / (d_in + d_out)), b=3*math.sqrt(2 / (d_in + d_out))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.weight, x, "... d_out d_in, ... d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(vocab_size, d_model, device=device, dtype=dtype), mean=0, std=1, a=-3, b=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight: vocab_size, d_model, x: batch_size, sequence_length
        # lookup the embedding for each token in x
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.w1 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w2 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU: x * sigmoid(x)
        # SwiGLU: W2(SiLU(W1x) ⊙ W3x),
        # w1: d_ff, d_model, w2: d_model, d_ff, w3: d_ff, d_model, x: batch_size, sequence_length, d_model
        return self.w2(self.w1(x) * torch.sigmoid(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        # cache uses self.register_buffer(persistent=False)
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # Initialize caches for cos and sin of rotation frequencies
        self.register_buffer(
            "cos_cached",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.empty(0),
            persistent=False,
        )
        self._maybe_build_cache(max_seq_len, device=device, dtype=torch.float32)

    def _maybe_build_cache(self, needed_len: int, device=None, dtype=None) -> None:
        # Build or extend cache to at least needed_len
        if self.cos_cached.numel() != 0 and self.cos_cached.shape[0] >= needed_len:
            return
        seq_len = max(needed_len, self.max_seq_len)
        half_dim = self.d_k // 2
        assert self.d_k % 2 == 0, "RoPE requires even d_k"

        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.d_k, 2, device=device, dtype=dtype) / self.d_k)
        )  # (half_dim,)
        positions = torch.arange(seq_len, device=device, dtype=dtype)  # (seq_len,)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)  # (seq_len, half_dim)
        cos_cached = torch.cos(freqs)
        sin_cached = torch.sin(freqs)

        # Register/assign buffers (keep persistent=False)
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: ..., sequence_length, d_k
        # token_positions: ..., sequence_length
        assert x.shape[-1] == self.d_k, "Input last dim must equal d_k"
        seq_len = x.shape[-2]
        self._maybe_build_cache(seq_len, device=x.device, dtype=x.dtype)

        if token_positions is None:
            # Default to [0, 1, ..., seq_len-1]
            positions = torch.arange(seq_len, device=x.device)
        else:
            positions = token_positions

        # Gather cos/sin for positions; shapes: (..., seq_len, half_dim)
        cos = self.cos_cached.to(device=x.device, dtype=x.dtype)[positions]
        sin = self.sin_cached.to(device=x.device, dtype=x.dtype)[positions]

        # Split last dim into pairs and apply rotation
        x_even = x[..., :, 0::2]
        x_odd = x[..., :, 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # Interleave even/odd back into original shape
        x_out = torch.empty_like(x)
        x_out[..., :, 0::2] = x_rot_even
        x_out[..., :, 1::2] = x_rot_odd
        return x_out

# softmax
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # numerically stable softmax: subtract max then exponentiate
    max_x = torch.amax(x, dim=dim, keepdim=True)
    shifted = x - max_x
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)


# scaled_dot_product_attention
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # Q: ..., queries, d_k
    # K: ..., keys, d_k
    # V: ..., values, d_v
    # mask: ..., queries, keys
    # return: ..., queries, d_v
    d_k = Q.shape[-1]
    scores = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        # True means allowed; False positions are masked (set to -inf)
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    out = einops.einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return out


# multihead_self_attention_with_rope
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int=None, theta: float=None, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # RoPE operates on head dimension d_head
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: ..., sequence_length, d_model
        # Project to Q, K, V in one go per tensor
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split heads: (..., seq, d_model) -> (..., head, seq, d_head)
        q = einops.rearrange(q, "... seq (head d) -> ... head seq d", head=self.num_heads)
        k = einops.rearrange(k, "... seq (head d) -> ... head seq d", head=self.num_heads)
        v = einops.rearrange(v, "... seq (head d) -> ... head seq d", head=self.num_heads)

        # Apply RoPE to Q and K only; head is treated as batch dimension so same rotation per head
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Build causal mask: disallow attending to future positions
        seq_len = x.shape[-2]
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        causal = causal.view(*(1 for _ in range(q.dim() - 2)), seq_len, seq_len)
        allowed = ~causal

        # Scaled dot-product attention per head
        attn_out = scaled_dot_product_attention(q, k, v, mask=allowed)  # (..., head, seq, d_head)

        # Merge heads back: (..., seq, d_model)
        out = einops.rearrange(attn_out, "... head seq d -> ... seq (head d)")
        return self.o_proj(out)

# transformer_block
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int=None, theta: float=None, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: ..., sequence_length, d_model
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# transformer_lm
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(indices)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits