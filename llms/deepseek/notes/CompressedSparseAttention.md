# Compressed Sparse Attention (CSA)

## The Problem with Standard Attention

In a standard transformer, every token attends to every other token in the sequence. The attention matrix is of size `T x T` where `T` is the sequence length. This means:

- Compute scales as **O(T²)**
- KV cache scales as **O(T)**

For short sequences (512, 2048 tokens) this is manageable. For long sequences — 100K, 1M tokens — it becomes computationally infeasible. A 1M token sequence would require a 1M x 1M attention matrix. That is one trillion attention score computations per layer, per forward pass.

This is the long-context problem. CSA is one approach to solving it.

---

## The Core Idea

CSA divides the context into two regions for each query token at position `t`:

1. **Local window** — the most recent `w` tokens before `t`. These tokens receive exact, full-precision attention. They are the most immediately relevant to the current token.

2. **Old context** — everything before the local window. Instead of attending to each of these tokens individually, they are compressed into a smaller set of summary tokens using a learned compression function. The query attends to these summary tokens rather than the originals.

This gives you the best of both worlds: precise attention where it matters most (recent context), and approximate but cheap attention over the broader history.

---

## Why This Works

The intuition is that tokens far back in the sequence are less likely to be immediately relevant to the current token than tokens nearby. A sentence being written now is more likely to depend on the previous paragraph than on something said ten thousand tokens ago. The local window captures fine-grained recent dependencies. The compressed old context captures coarse-grained global context.

The compression is learned, not hand-crafted. The compressor — typically a linear projection — learns during training which information across a group of tokens is worth preserving in the summary representation. It does not randomly drop tokens; it distills them.

---

## Formal Definition

Let `x` be a sequence of `T` tokens, each of dimension `D`. For a query token at position `t`:

**Local window tokens:**

```
x_local = x[t - w : t + 1]     shape: [w, D]
```

**Old tokens (outside the window):**

```
x_old = x[0 : t - w]           shape: [t - w, D]
```

**Compression of old tokens:**

Group `x_old` into chunks of size `r` (compress ratio), then project each chunk:

```
x_old reshaped: [floor((t-w)/r), r*D]
x_compressed = Linear(r*D → D)(x_old reshaped)    shape: [floor((t-w)/r), D]
```

**Effective KV set for token t:**

```
K_eff = concat(K(x_compressed), K(x_local))
V_eff = concat(V(x_compressed), V(x_local))
```

**Attention:**

```
Attention(Q(x_t), K_eff, V_eff)
```

The number of tokens attended to is:

```
w + floor((t - w) / r)
```

For a 1M token sequence with `w=512` and `r=8`, the effective context size per token is approximately `512 + 124,937 = 125,449` instead of `1,000,000`. That is an 8x reduction in the old context alone. Combined with not needing to store all KV pairs for old tokens, the memory savings are substantial.

---

## Comparison to Standard Attention

| Property | Standard Attention | CSA |
|---|---|---|
| Compute per token | O(T) | O(w + T/r) |
| KV cache size | O(T) | O(w + T/r) |
| Recent context | Exact | Exact |
| Old context | Exact | Approximate (compressed) |
| Information loss | None | Yes, in old context |
| Suitable for long context | No | Yes |

---

## How Compression Works in Detail

The compressor is a linear layer:

```
compressor = nn.Linear(D * r, D)
```

For a group of `r` consecutive old tokens with embeddings `[e_1, e_2, ..., e_r]`, each of dimension `D`:

1. Concatenate them: `[e_1 || e_2 || ... || e_r]` → shape `[r * D]`
2. Project: `compressor([e_1 || ... || e_r])` → shape `[D]`

The result is a single vector of dimension `D` that summarizes `r` tokens. This vector gets its own K and V projections and participates in attention just like a real token.

The key point is that the compressor is trained end-to-end with the rest of the model. It learns to preserve the information that downstream attention actually uses. It is not a heuristic — it is a learned function.

---

## Implementation

Below is a complete non-production implementation in PyTorch for understanding purposes.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CompressedSparseAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        local_window: int = 8,
        compress_ratio: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.local_window = local_window
        self.compress_ratio = compress_ratio

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Learned compression: r tokens → 1 summary token
        self.compressor = nn.Linear(d_model * compress_ratio, d_model)

    def compress_old_tokens(self, old_tokens: torch.Tensor):
        """
        old_tokens: [B, T_old, D]
        returns:    [B, T_old // r, D]
        """
        B, T_old, D = old_tokens.shape
        trim = T_old - (T_old % self.compress_ratio)
        old_tokens = old_tokens[:, :trim, :]

        if trim == 0:
            return None

        # Group r tokens together, project to single summary
        grouped = old_tokens.reshape(
            B,
            trim // self.compress_ratio,
            self.compress_ratio * D
        )
        return self.compressor(grouped)    # [B, trim//r, D]

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.reshape(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def attention(self, q, k, v) -> torch.Tensor:
        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        outputs = []

        for t in range(T):
            q_t = Q[:, t:t+1, :]

            # Local window: exact attention
            local_start = max(0, t - self.local_window + 1)
            k_local = K[:, local_start:t+1, :]
            v_local = V[:, local_start:t+1, :]

            # Old tokens: compressed attention
            if local_start > 0:
                old_tokens = x[:, :local_start, :]
                compressed = self.compress_old_tokens(old_tokens)
            else:
                compressed = None

            if compressed is not None:
                k_comp = self.W_k(compressed)
                v_comp = self.W_v(compressed)
                k_all = torch.cat([k_comp, k_local], dim=1)
                v_all = torch.cat([v_comp, v_local], dim=1)
            else:
                k_all = k_local
                v_all = v_local

            q_h = self.split_heads(q_t)
            k_h = self.split_heads(k_all)
            v_h = self.split_heads(v_all)

            out_h = self.attention(q_h, k_h, v_h)
            out = out_h.permute(0, 2, 1, 3).reshape(B, 1, D)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        return self.W_o(out)
```

---

## Walking Through a Concrete Example

Assume:
- Sequence length `T = 32`
- Local window `w = 8`
- Compress ratio `r = 4`

For token at position `t = 24`:

```
Local window tokens: positions 17 to 24  → 8 tokens, exact attention
Old tokens: positions 0 to 16           → 17 tokens
After trim to divisible by 4: 16 tokens → 16 / 4 = 4 summary tokens
Total KV size: 8 + 4 = 12
Standard attention KV size: 25
```

For token at position `t = 31`:

```
Local window tokens: positions 24 to 31 → 8 tokens, exact attention
Old tokens: positions 0 to 23          → 24 tokens → 6 summary tokens
Total KV size: 8 + 6 = 14
Standard attention KV size: 32
```

As `T` grows, the savings compound. At `T = 1,000,000` with `w = 512` and `r = 8`:

```
Standard attention KV size per token: 1,000,000
CSA KV size per token:                512 + (999,488 / 8) = 125,448
Reduction:                            ~8x in KV size
```

---

## Limitations

**Information loss in old context.** The compression is lossy. If a critical piece of information from far back in the sequence needs precise retrieval, CSA may fail where standard attention would not. The compressor learns to preserve what is statistically useful, not necessarily what is specifically needed for a given query.

**Compression is uniform.** Every group of `r` tokens gets the same compression treatment regardless of content. More sophisticated variants (like learned routing or importance-based selection) could improve this, but add complexity.

**Token order within groups.** The reshape-and-project compressor sees `r` tokens concatenated in order. It does not have positional information about which token within the group contributed what. More sophisticated compressors use attention within the group before projecting.

**Not a drop-in replacement.** The local window assumption breaks for tasks where distant tokens are heavily referenced. Retrieval-heavy tasks, long-range coreference resolution, or tasks where the answer depends on the first few tokens of a very long document may suffer.

---

## How CSA Fits into a Hybrid Architecture

In practice, models like DeepSeek-V4-Pro do not use CSA for every layer. They use a **hybrid** strategy:

- Some layers use CSA (cheap, good for local patterns and compressed global context)
- Some layers use standard full attention (expensive but exact, used sparingly)
- Some layers use HCA (Heavily Compressed Attention — even more aggressive compression than CSA)

The intuition is that not every layer needs full global attention. Early layers can build local representations cheaply. A few carefully placed full-attention layers can then integrate global context. This combination gives near-full-attention quality at a fraction of the cost.

---

## Summary

CSA is a structured approximation to full attention that exploits the locality assumption — nearby tokens matter more than distant ones. It achieves this by:

1. Attending exactly to a local window of recent tokens
2. Attending approximately to a learned compression of older tokens
3. Keeping the KV cache proportional to `w + T/r` instead of `T`

The compression is learned end-to-end, which means the model decides what to preserve. The tradeoff is approximate recall of old context in exchange for dramatically reduced compute and memory at long sequence lengths.