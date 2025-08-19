```markdown
# 🧭 ALiBi (Attention with Linear Biases)

## What is ALiBi?

ALiBi is a method for encoding positional information in Transformers.  
Instead of giving tokens explicit positional embeddings (like sinusoidal or RoPE),  
it adds a **linear bias** to the attention scores based on how far apart two tokens are.

👉 This makes models trained on shorter sequences (e.g., 2k tokens) **generalize naturally** to much longer ones (100k+).

---

## Intuition

- Nearby tokens are usually more relevant (e.g., “today” relates to “yesterday” more than “10 years ago”).
- ALiBi enforces this by **penalizing distant tokens linearly** in the attention mechanism.
- Embeddings stay the same — only the **attention score matrix** is adjusted.

---

## Example

### Step 1: Normal Attention Scores (Q·Kᵀ)

Let’s say we have 3 tokens:

- T1 (pos=1), T2 (pos=2), T3 (pos=3)

Attention scores before ALiBi:
```

```
  T1    T2    T3
```

T1 1.0 2.0 3.0
T2 2.0 1.5 2.5
T3 3.0 2.5 1.2

```

---

### Step 2: Compute ALiBi Penalties
Let slope = **-0.2**.
Penalty = slope × distance.

```

```
  T1    T2    T3
```

T1 0.0 -0.2 -0.4
T2 -0.2 0.0 -0.2
T3 -0.4 -0.2 0.0

```

---

### Step 3: Add Penalties to Scores
```

```
  T1    T2    T3
```

T1 1.0 1.8 2.6
T2 1.8 1.5 2.3
T3 2.6 2.3 1.2

```

---

## ✅ Key Takeaways
- **No embeddings needed** for position, just a bias on attention.
- **Linear penalty** means far tokens aren’t ignored, but are less important.
- **Scales gracefully** → models can handle much longer contexts than they were trained on.

👉 Used in models like **LLaMA** and many long-context LLMs.
```
