# What I Learned — Dynamic Programming for RL (from the transcript)

## Big Picture

- We’re in the **reinforcement learning (RL)** part of the course, focused on computing **value functions** and ultimately an **optimal policy**.
- Today’s tool: **Dynamic Programming (DP)** — a family of _planning_ algorithms that solve RL problems **when the environment model is known** (transition probabilities and rewards).

---

## Key Terms

- **Prediction problem (policy evaluation):** Given a policy $\pi$, estimate its state-value function $V^\pi(s)$.
- **Control problem (policy improvement):** Improve a policy using value estimates, aiming for an optimal policy $\pi^*$.

---

## The Bellman View

- **Bellman expectation equation** (for a given policy $\pi$):

  $$
  V^\pi(s) \;=\; \sum_a \pi(a\!\mid\!s)\sum_{s',r} p(s',r\!\mid\!s,a)\,\big[r+\gamma V^\pi(s')\big]
  $$

  It says the value of a state equals expected immediate reward plus **discounted value of the next state**.

- Writing this for **every state** gives **$n$ equations in $n$ unknowns** (hard to solve directly for large $n$).

---

## Dynamic Programming Methods

### 1) Iterative Policy Evaluation (Prediction)

Compute $V^\pi$ by fixed-point iteration:

$$
V_{k+1}(s)\leftarrow \sum_a \pi(a\!\mid\!s)\sum_{s',r} p(s',r\!\mid\!s,a)\,[r+\gamma V_k(s')]
$$

- Start with $V_0(s)=0$ (or anything), loop until values **converge**.
- Requires a **full state sweep** each iteration.

**Pseudocode**

```text
initialize V(s) := 0 for all s
repeat
  Δ := 0
  for each state s:
    v_old := V(s)
    V(s) := Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
    Δ := max(Δ, |V(s) - v_old|)
until Δ < θ   // small threshold
```

### 2) Policy Improvement (Control)

Given $V^\pi$, greedily update the policy:
s

$$
\pi_{\text{new}}(s) \in \arg\max_a \sum_{s',r} p(s',r\!\mid\!s,a)\,[r+\gamma V^\pi(s')]
$$

- If the greedy action equals the current policy’s action **for every state**, the policy is **stable**.

### 3) Policy Iteration

Alternating **evaluation ↔ improvement**:

1. **Evaluate** current policy $\pi$ to get $V^\pi$.
2. **Improve** $\pi$ greedily using $V^\pi$.
3. Repeat until the policy stops changing → **$\pi^*, V^*$**.

**High-level**

```text
initialize π arbitrarily
repeat
  V^π := PolicyEvaluation(π)
  π_new := GreedyPolicy(V^π)
until π_new == π
return π_new, V^π
```

### 4) Value Iteration (One-Step Lookahead)

Combine evaluation and improvement in a single update:

$$
V_{k+1}(s)\leftarrow \max_a \sum_{s',r} p(s',r\!\mid\!s,a)\,[r+\gamma V_k(s')]
$$

- After convergence, derive a greedy policy w\.r.t. $V$.

---

## Example Highlight — Car Rental (Practical DP)

- **State:** $(\#\text{cars at loc1}, \#\text{cars at loc2})$ with caps (e.g., ≤20 each).
- **Action:** Move up to 5 cars overnight between locations (positive one way, negative the other).
- **Reward:** $+10$ per rental; $-2$ per car moved.
- **Dynamics:** Daily **rental requests and returns** at each location follow **Poisson** distributions.
- **Goal:** Find for every state how many cars to move nightly to **maximize expected revenue**.
- **Method:** **Policy iteration**:

  - Evaluate current policy’s values.
  - Improve by choosing the action with the highest expected return.

- **Outcome:** A **state→action map** (policy heatmap) that moves cars from the fuller location toward the needier one, with movement magnitude decreasing as locations balance.

---

## Strengths & Limitations of DP

**Strengths**

- Provably convergent to $V^\pi$ / $V^*$ and $\pi^*$ under standard assumptions.
- Builds precise intuition for value functions, policies, and backups.

**Limitations**

- **Requires a known model** $p(s',r\!\mid\!s,a)$ → not a learning-from-experience method.
- **Full state sweeps** and big tables → computationally heavy for large state spaces.

> These limitations motivate **Monte Carlo** and **Temporal-Difference (TD)** methods next: they learn from **experience** and can avoid full model knowledge.

---

## Mental Model: Backup Diagrams

- Visual “look-ahead” from a state through possible **actions → next states**.
- Bellman updates are **expectations over these backups**; policy improvement is **argmax over action backups**.

---

## How I’d Apply This (Forward-Looking)

- Use **policy iteration/value iteration** as a **planning baseline** in small/known-model tasks (simulators, games, ops planning).
- For larger or partially known environments, move to:

  - **Model-free** (MC/TD) with function approximation.
  - **Approximate DP / planning** with learned models (Dyna-style) or offline value estimation.

- For **reasoning LLM agents**, treat tool-use or thought states as MDP states and:

  - Start with DP in simplified simulators to shape a reward/value scaffold.
  - Transition to **TD-style** value learning from real trajectories where the model is unknown.

---

## Quick Checklist to Implement

- [ ] Define MDP: states, actions, rewards, transitions (if known), $\gamma$.
- [ ] Choose method: **Policy Iteration** (evaluation + improvement) or **Value Iteration**.
- [ ] Implement **backups** per Bellman equations.
- [ ] Convergence criteria: small $\Delta$ on values or policy stability.
- [ ] Inspect resulting **policy map**; test in environment/simulator.

---

## TL;DR

Dynamic Programming in RL = **Bellman-based planning** using a known model:

1. **Evaluate** a policy by iteratively backing up values,
2. **Improve** the policy by acting greedily w\.r.t. those values,
3. Repeat → **optimal policy and value function**.
   Great for intuition and small modeled problems; next up: **Monte Carlo & TD** to overcome the “needs-a-model + full sweep” limitations.
