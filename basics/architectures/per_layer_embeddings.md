# Per-Layer Embeddings (PLE)

Per-layer embeddings are an extra token-specific embedding path used in some Gemma 4 models. The basic idea is simple: each transformer block gets its own small learned embedding slice, so the model gains extra capacity without making the repeated attention + feed-forward stack much wider.

## Why This Matters

- **Capacity without full compute scaling:** The main transformer blocks stay closer to the smaller effective model size.
- **Embedding-style parameters are cheaper:** Extra parameters live in embedding tables and small projection layers rather than in every attention and MLP block.
- **Useful for edge-sized models:** This is most appealing when you want a stronger model under a tight latency or memory budget.

## How The Path Works

1. Token IDs are mapped through a per-layer embedding lookup.
2. The normal token embeddings are projected into the same packed PLE space.
3. The two contributions are combined and reshaped so each transformer layer receives its own slice.
4. Inside the block, the standard attention and feed-forward path runs first.
5. The layer-specific PLE slice is gated by the hidden state, projected back to model width, normalized, and added as an extra residual update.

So PLE is not replacing attention or the FFN. It is an additional residual path that injects token-specific, layer-specific information.

## Distinction From Similar Ideas

- **Not KV sharing:** KV sharing reduces inference cache growth. PLE does not target the cache; it adds representational capacity.
- **Not MoE:** MoE sparsely routes tokens through experts. PLE keeps the dense transformer path and adds a compact per-layer embedding residual.
- **Not a wider backbone:** A larger dense model increases compute everywhere. PLE concentrates extra parameters in cheaper embedding-style components.

## Gemma 4 Interpretation

Gemma 4 E2B and E4B use the idea in a way that makes the parameter accounting more nuanced:

- **E** stands for **effective** parameters.
- The smaller number is closer to the main transformer-stack compute.
- The larger total includes extra embedding parameters.

That means the headline model size is not the full story. The useful mental model is: the dense computation path behaves more like the smaller model, while the extra embeddings provide additional capacity at a lower compute cost.

| Model | Effective params | Total params |
|---|---:|---:|
| Gemma 4 E2B | 2.3B | 5.1B |
| Gemma 4 E4B | 4.5B | 8B |

## Practical Takeaway

PLE is best thought of as a capacity add-on. It gives each layer its own small token-specific memory, which can improve expressiveness without paying the full cost of making the core transformer uniformly larger.

For a compact model, that tradeoff can be attractive. For larger models, the same idea is less obviously compelling because the backbone already has more capacity and sparse-expert methods may be a better fit.