# Hyper-Connections and mHC

---

## The Problem with Residual Connections

Every transformer uses this:

```
x_out = x + f(x)
```

The `+ x` is a fixed skip connection. It exists to prevent vanishing gradients — nothing more. The model has zero control over how information flows between layers. Every layer gets the same fixed on-ramp and off-ramp regardless of what the data needs.

---

## What Hyper-Connections Change

ByteDance asked: what if the skip connection was learned instead of fixed?

Instead of one stream of hidden states flowing through the network, HC runs **N parallel streams**. Think of N parallel highways instead of one. Two learned things control the routing:

**Matrix A** of shape `[N, N]` — controls how streams mix together before going into `f`. If `A[i][j] = 0.8`, stream `i` gets 80% of its input from stream `j`. If `A` is close to the identity matrix, it behaves exactly like a standard residual. If off-diagonal values grow large, information starts skipping or repeating across layers — the model learns depth-level routing.

**Vector B** of shape `[N]` — controls how `f`'s output gets distributed back to each stream after the layer. `B[k] = 0` means stream `k` is bypassed entirely. `B[k]` large means stream `k` gets a strong update from this layer.

The result is that the model can learn to skip layers, repeat layers, or pull information from multiple depths simultaneously. ByteDance showed this converges 1.8x faster and gains +6 points on ARC-Challenge over standard residuals.

---

## Why HC Breaks at Scale

Each layer applies `A` to the streams:

```
x_out = A @ x_in
```

Stack 32 layers and this becomes:

```
x_final = A_32 @ A_31 @ ... @ A_1 @ x_0
```

Matrices multiply together. If any matrix has values slightly above 1 in some direction, those values compound exponentially across 32 layers. DeepSeek tried HC on a 27B model and measured signal gains exceeding **3000x** — the signal became so large that gradients exploded and training diverged completely.

The root cause is that `A` is unconstrained. Nothing stops it from taking large values, and once it does, stacking many of them together is catastrophic.

---

## The Fix — Birkhoff Polytope

DeepSeek's fix: constrain `A` to be a **doubly stochastic matrix** — a matrix where every row sums to 1 and every column sums to 1.

```
0.5   0.3   0.2     → row sum = 1.0
0.2   0.5   0.3     → row sum = 1.0
0.3   0.2   0.5     → row sum = 1.0
↓     ↓     ↓
1.0   1.0   1.0   (column sums)
```

Why does this prevent explosion? Row sum = 1 means the output of `A @ x` is a weighted average of the input streams. Weighted averages cannot exceed the maximum input — amplification is mathematically impossible. Column sum = 1 means each input stream contributes exactly one unit of total weight across all outputs — nothing is lost or double-counted.

Together: information is **routed and conserved**, never amplified. No matter how many layers you stack, signal magnitude stays bounded. The set of all doubly stochastic matrices is called the Birkhoff Polytope. mHC = HC with `A` constrained to live on this polytope.

---

## Sinkhorn-Knopp — How to Actually Enforce It

You cannot just hope `A` stays doubly stochastic during training. After every gradient update, you need to actively project `A` back onto the Birkhoff Polytope.

The Sinkhorn-Knopp algorithm (1967) does this in two alternating steps:

1. Divide each row by its row sum → rows now sum to 1
2. Divide each column by its column sum → columns now sum to 1

Repeat ~20 times. That is the entire algorithm. Each step fixes one constraint and slightly breaks the other, but the breakage shrinks each iteration until both constraints are satisfied simultaneously.

```python
def sinkhorn_knopp(M, n_iters=20):
    M = torch.exp(M)                                    # make values positive
    for _ in range(n_iters):
        M = M / M.sum(dim=1, keepdim=True)              # normalize rows
        M = M / M.sum(dim=0, keepdim=True)              # normalize columns
    return M
```

In mHC, this runs every forward pass on the raw parameter `A` before it touches any data. The parameter can drift anywhere during training — Sinkhorn-Knopp snaps it back to the manifold before use.

---

## mHC vs HC — The One Line Difference

```python
# HC — unconstrained A
x_mixed = torch.einsum('ij, btjd -> btid', self.A, x_streams)

# mHC — constrained A
A_constrained = sinkhorn_knopp(self.A)          # ← this one line
x_mixed = torch.einsum('ij, btjd -> btid', A_constrained, x_streams)
```

Same architecture. Same number of parameters. Same expressiveness. One projection call added. DeepSeek reports ~6-7% training overhead — negligible for a model that would otherwise diverge at scale.

---

## Why This Matters

The same pattern appears everywhere in modern LLM architecture:

- **LoRA** — weight updates during fine-tuning are unconstrained and expensive → constrain `ΔW` to be low rank
- **CSA** — attending to all T tokens is unconstrained and expensive → constrain to local window + compressed summaries
- **mHC** — mixing matrix `A` is unconstrained and causes explosion → constrain to Birkhoff Polytope

Find what is unconstrained. Add the right mathematical constraint. Efficiency and stability follow.