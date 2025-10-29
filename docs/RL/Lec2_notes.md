## 1. The Concept of Value

Value is one of the four foundational elements of reinforcement learning, alongside policy, rewards, and the environment model. It provides a critical measure for decision-making by considering the future impact of a state or action.

### Intuition: Long-Term Desirability

The primary distinction is that **Value is fundamentally different from immediate rewards**. While rewards concern only short-term desirability, **Value focuses on the long-term desirability of the state**.

Instead of just looking at single rewards, the Value calculation involves **adding all the rewards together** until the end of the episode to evaluate the resulting **return ($G_t$)** received by the agent. Later rewards are discounted by a factor ($\gamma$) because **immediate rewards are more valuable** than those received later in the trajectory.

### Types of Value Functions

The sources identify two essential value functions, both aiming to estimate the expected return:

#### A. State Value Function ($V_{\pi}(s)$)

The state value function is formally defined as the **expected return** the agent receives, starting in a specific state ($s$) and following a particular policy ($\pi$) thereafter until the end of the episode.

- **Notation:** It is consistently denoted by the symbol $V_{\pi}(s)$ across RL literature.
- **Estimation:** To calculate the value of a state, the agent may run multiple episodes (e.g., 10 episodes), determine the return for each episode ($G_t$), sum them up, and then divide by the total number of episodes to calculate the **mean (expected value)** of the returns.

#### B. Action Value Function ($Q(s, a)$)

The action value function quantifies the value associated with a specific action taken in a specific state.

- **Notation:** It is denoted by the famous and frequently repeated symbol $Q(s, a)$.
- **Definition:** It is formally defined as the **expected return starting in a state, taking an action, and then following a policy thereafter**.
- **Utility:** The value of states and actions (the Q values) are highly useful for implementation because they provide direct information: an agent can look at the Q table and see which action has the highest Q value for a given state. An agent's policy can be formed by telling it to **always choose the action with the maximum Q value**.
- **Relation to $V(s)$:** The relationship between $Q(s, a)$ and $V(s)$ depends entirely on the policy chosen by the agent. If the policy dictates choosing action $A_1$, then $Q(s, A_1)$ will exactly match $V(s)$.

### Value Estimation via Bellman Equations

Estimating the value function is often referred to as the **prediction problem** in RL [23, 15:57]. This process was significantly simplified by Richard Bellman's key finding:

1.  **The Recursive Nature (Bellman Equation):** Bellman stated that the value of being in a state ($V(s)$) can be expressed recursively in terms of the value of the next state ($V(s')$).
    - **Intuition:** The value of a state is equal to the expected immediate reward ($R$) plus the **discounted expected value of the next state** ($\gamma \cdot V(s')$). This recursive relationship is powerful and is at the heart of algorithms used in complex applications.
2.  **The Optimal Choice (Bellman Optimality Equation):** This equation extends the basic Bellman equation to define the best possible value by selecting the action that maximizes the expected return.
    - The optimal action is chosen by selecting the **maximum** of the immediate reward plus the discounted value of the next state: $\max_a (R + \gamma \cdot V(s'))$ [41, 28:28].
    - This is equivalent to finding the **maximum of the action value functions** for all possible actions: $\max (Q(s, a))$. Solving for this maximum allows the agent to confidently choose the optimal action for every state.

---

## 2. The Role of the Model

The Model of the environment is defined by the necessary information to determine the dynamics of the system, primarily requiring knowledge of the **transition probabilities** from one state ($S$) to the next state ($S'$) [77, 78, 55:49].

RL methods are categorized based on whether they use or require a model: Model-Based or Model-Free.

### A. Model-Based Methods (Dynamic Programming)

Model-based methods, such as Dynamic Programming (DP), **require a complete model of the environment**.

- **Process:** DP methods use the iterative application of the Bellman equations to estimate the optimal value functions. This process involves repeating two steps—**policy evaluation** (estimating $V_{\pi}(s)$ for a given policy) and **policy improvement** (changing the policy to maximize Q values).
- **Practical Challenge:** Dynamic Programming is currently **not used in practice**. This is because requiring a complete model—knowing the exact probability of transitioning from $S$ to $S'$—is often **impossible to obtain in most cases**. For example, when training an agent to play chess, it is extremely hard to acquire the probability of transitioning between specific board configurations [78, 56:16]. Furthermore, DP often requires solving hundreds of simultaneous equations, which is computationally expensive [55:08, 55:14, 55:20].

### B. Model-Free Methods

Model-free methods are preferred because they **do not require a model of the environment**. They learn directly through **raw experience**.

#### 1. Monte Carlo (MC) Methods

MC methods learn by **simulating a large number of episodes** and collecting raw experience.

- **Process:** The agent calculates the return (sum of discounted rewards) for all visited states and actions. The final estimate of the action value function ($Q(s, a)$) is the **average of all the returns** received for that state-action pair across all episodes [85, 135:59].
- **Intuition:** MC is like updating your knowledge only after the _entire_ episode is complete. You must wait to receive all future rewards before updating your value function estimate.
- **Action Selection:** MC uses the estimated $Q(s, a)$ values to determine the policy, typically employing an **epsilon greedy policy** to balance exploitation (choosing the action with maximum $Q$ value) and exploration (taking random actions). Exploration is necessary to ensure the agent encounters all states and actions, preventing it from missing an optimal path.

#### 2. Temporal Difference (TD) Methods

Temporal Difference methods are considered the **"best of both worlds"**. They combine learning from raw experience (like MC) with updating estimates based on other learned estimates (like DP). TD methods are highly useful in practice due to their efficiency.

- **Efficiency Intuition:** Unlike MC, which waits for the entire episode to complete, TD makes **incremental updates after each step**. This is similar to how humans think and make updates—incrementally, rather than waiting until the final outcome.
- **Mechanism:** TD solves the challenge of waiting for the final return by **approximating the return** using the immediate reward plus the discounted estimated value of the next state. This uses the structure of the Bellman equation to define the update target (the "TD target").
- **Algorithms:** Q-learning and SARSA are prominent examples of TD methods. These algorithms efficiently update the action value function ($Q(s, a)$). Q-learning, for instance, uses a maximum operation derived from the Bellman optimality equation when calculating the update target.
