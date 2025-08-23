# Reinforcement Learning — Classical Foundations

This lecture built directly on the **four basic elements of RL** (policy, rewards, values, and model) and introduced the mathematical foundation of how agents *learn to act optimally*. We’ll walk through each concept and then the algorithms that implement them.

---

## 1. Recap of the Four Elements of RL

1. **Policy (π):**

   * The “behavior” of the agent.
   * Given a state, the policy decides which action to take.
   * Without a policy, the agent just acts randomly (like in our Lunar Lander example).

2. **Reward (R):**

   * The immediate feedback from the environment after each action.
   * Represents *short-term desirability*.

3. **Value (V):**

   * Long-term desirability.
   * Instead of just looking at one reward, we sum up all **future discounted rewards** the agent can collect starting from a state.

4. **Model (of the environment):**

   * Describes how the world works (state transitions + rewards).
   * In practice, many RL settings are **model-free**, meaning we don’t know the exact model.

---

## 2. Markov Property

* A state is **Markovian** if the future depends **only on the present state**, not on the full history.
* Example: Chess board configuration = Markovian (no need to know the previous moves).
* RL usually assumes **Markov Decision Processes (MDPs)**.

---

## 3. Value Functions

### 3.1 State Value Function (**Vπ(s)**)

* Measures how *good* it is to be in a state **s**, if the agent follows a policy π afterward.

* Formally:

  $$
  V^\pi(s) = \mathbb{E}[G_t \mid S_t = s]
  $$

  where $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$

* **γ (gamma):** Discount factor (0 < γ < 1) → prioritizes immediate rewards over distant ones.

---

### 3.2 Action Value Function (**Qπ(s, a)**)

* Measures how good it is to take action **a** in state **s**, and then follow policy π.

* Formally:

  $$
  Q^\pi(s, a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]
  $$

* Relationship with Vπ(s):

  * If π always chooses action a in state s → then Qπ(s, a) = Vπ(s).

---

## 4. Bellman Equations

### 4.1 Bellman Expectation Equation

* Richard Bellman (1950s) showed that value functions satisfy a **recursive relation**:

$$
V^\pi(s) = \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1})]
$$

* “The value of a state = expected immediate reward + expected discounted value of next state.”

---

### 4.2 Bellman Optimality Equation

* For the **optimal policy**, the best value is achieved by taking the action with maximum Q-value:

$$
V^*(s) = \max_a \Big( R(s,a) + \gamma V^*(s') \Big)
$$

This tells us how to pick the **best action** in each state.

---

## 5. Dynamic Programming (DP)

* Early method for solving RL problems (when the model of environment is **known**).
* Two key steps:

  1. **Policy Evaluation** → Estimate Vπ(s) for a fixed policy.
  2. **Policy Improvement** → Update the policy by choosing actions with the highest Q-value.
* Repeat evaluation + improvement until convergence = **Policy Iteration**.

👉 Limitation: Requires **knowing the model** (transition probabilities), which is impractical in most real-world tasks.

---

## 6. Monte Carlo (MC) Methods

* **Model-free**: Learn from *experience (episodes)* instead of knowing the full environment.

* Idea:

  * Run many episodes.
  * For each state/action, average the **returns** observed after visiting it.
  * Update value/Q estimates with these averages.

* **Monte Carlo Prediction:** Estimates Vπ(s).

* **Monte Carlo Control:** Uses Q(s, a) and an **ε-greedy policy** to improve action choices.

👉 Drawback: Updates happen **only at the end of an episode**.

---

## 7. Temporal Difference (TD) Methods

* Combine the best of both worlds:

  * Like MC: model-free.
  * Like DP: update **incrementally at each step**, not just at the end.

* **TD Update Rule:**

  $$
  V(s) \leftarrow V(s) + \alpha \Big[ R + \gamma V(s') - V(s) \Big]
  $$

* Key Algorithms:

  1. **SARSA (On-policy):**
     Updates Q(s, a) using the *actual action* taken under the ε-greedy policy.

     $$
     Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ R + \gamma Q(s',a') - Q(s,a) \Big]
     $$

  2. **Q-Learning (Off-policy):**
     Uses the **max over next actions**, regardless of the policy’s exploration.

     $$
     Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ R + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]
     $$

👉 Q-learning is the algorithm DeepMind used to train agents to play Atari games.

---

## 8. Exploration vs. Exploitation

* If you always exploit (pick best action) → you might miss better paths.
* If you always explore (random actions) → you won’t learn efficiently.
* **ε-greedy policy** balances this:

  * With probability (1−ε), exploit.
  * With probability (ε), explore randomly.
* Over time, ε is **decayed** (more exploration early, more exploitation later).

---

## 9. Practical Example: Lunar Lander

* Using OpenAI Gymnasium:

  * **Monte Carlo Agent:** Updates Q-table after each episode.
  * **TD Agent (Q-learning):** Updates Q-values after every step.

* Observations:

  * Monte Carlo → slower, updates only at episode end.
  * TD (Q-learning) → more efficient, learns faster, produces smoother landings.

* Visualization with **TensorBoard** helps track rewards improving over episodes.

---

# Summary of the “Three Horsemen” of Classical RL

1. **Dynamic Programming (DP):**

   * Needs full model (impractical in real life).
   * Good for theory.

2. **Monte Carlo (MC):**

   * Model-free, learns from episodes.
   * Updates only at episode end → slow.

3. **Temporal Difference (TD):**

   * Model-free, updates each step.
   * Efficient, practical.
   * Core of modern RL algorithms (SARSA, Q-learning, and beyond).

---

✅ By now you should have:

* A clear sense of what **value functions (V and Q)** mean.
* Why **Bellman equations** are central.
* How **DP, MC, and TD** connect historically.
* Why **Q-learning** became the backbone of modern RL.
