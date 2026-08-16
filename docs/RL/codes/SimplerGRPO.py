"""
Load three models. Live X (trains), snapshot Y (frozen copy), reference Z (frozen forever).

Generate 8 completions from Y. Save log π_old per token -> [8, 500].

Score them. Rewards -> group-normalized advantages -> [8].

Forward pass completions through X, grads ON  -> log π_θ  [8, 500].
Forward pass completions through Z, no_grad    -> log π_ref [8, 500].

ratio = exp(log π_θ - log π_old)     # how far X moved from Y
KL    = divergence(X || Z)           # how far X moved from Z

Objective per token = clipped(ratio × advantage) - β × KL
  advantage: push good tokens up, bad ones down
  clip:      cap the step size vs Y
  KL:        stay anchored to Z

loss = -mean over all tokens, all generations
backward, step. Then Y <- X. Z never changes.
"""