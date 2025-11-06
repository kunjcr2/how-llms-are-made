### Policy Gradient Methods

•	Objective: Directly learn a policy (probability distribution over actions given a state) without explicitly calculating Q-values. This is more scalable for complex problems than tabular methods or Q-value approximation with neural networks (0:40, 5:15, 6:16).
•	Parameterized Policy: The policy is represented by a function with parameters (theta), often like weights in a neural network (7:33, 9:35).
•	Softmax Function: Used to convert numerical preferences for actions into probabilities between 0 and 1 (10:30).

### Policy Gradient Theorem

•	Goal: Maximize a performance measure J(theta), which is typically the value function of the initial state (V_pi(s0)), representing the expected cumulative rewards (13:13, 16:21).
•	Gradient Ascent: To maximize J(theta), we use gradient ascent: `thetat+1 = thetat + alpha * grad(J(theta))` (14:10).
•	The Challenge: Calculating `grad(J(theta))` is difficult because the policy's performance depends on states visited and actions taken, which are stochastic (15:47).
•	The Breakthrough (1999): The Policy Gradient Theorem provided a formula for `grad(J(theta))` (18:03).
•	Log-Derivative Trick: A key mathematical trick used in the derivation: `grad(P(tau)) = P(tau) * grad(log P(tau))` (23:14).
•	Core Formula: `grad(J(theta)) = E[sum{t=0 to T} grad(log pi(at|s_t, theta)) * G(tau)]` (29:38).
•	This means: sample trajectories, sum the gradients of the log-probabilities of actions taken, and multiply by the total return (G) of the trajectory.
•	Intuition: If an action leads to a positive return (G is positive), increase its probability; if it leads to a negative return (G is negative), decrease its probability (33:57, 34:49).

### REINFORCE Algorithm

•	First Policy Gradient Algorithm: Directly uses the derived formula to update policy parameters (37:12).
•	Policy Update Rule: `thetat+1 = thetat + alpha * E[sum{t=0 to T} grad(log pi(at|s_t, theta)) * G(tau)]` (36:36).
•	Problem: High Variance. The return `G(tau)` for each trajectory can vary wildly, leading to unstable gradient estimates (38:28, 40:21).

### REINFORCE with Baseline

•	Solution to Variance: Subtract a baseline `B(s)` from the return `G(tau)` (41:50).
•	Modified Formula: `grad(J(theta)) = E[sum{t=0 to T} grad(log pi(at|s_t, theta)) * (G(tau) - B(s))]` (42:12).
•	Benefits: Reduces variance without introducing bias (as `B(s)` doesn't depend on actions) (42:49).
•	Common Baseline: The state-value function `V_pi(s)` is often used as the baseline (45:41).
•	Actor-Critic Formulation: When the baseline is `V_pi(s)`, the method is often called actor-critic. The "actor" (policy) takes actions, and the "critic" (value function) evaluates the actions (46:33).

### Advantage Function

•	Definition: `A(s,a) = Qpi(s,a) - Vpi(s)` (1:15:04).
•	Intuition: Measures how much better or worse a specific action `a` in state `s` is compared to the average expected return from that state (1:16:15).
•	Positive advantage: Action is better than average, increase its probability.
•	Negative advantage: Action is worse than average, decrease its probability.
•	Estimation (Temporal Difference Error): Advantage can be estimated using the immediate reward and the value function of the next state: `A(s,a) = Rt + gamma * Vpi(St+1) - Vpi(S_t)` (1:22:30).
•	N-step Backups and GAE:
•	One-step backup: Low variance, high bias.
•	Monte Carlo (N-step) backup: High variance, low bias (1:27:03).
•	Generalized Advantage Estimation (GAE): Uses a `lambda` parameter to balance bias and variance by combining different n-step returns (1:25:00). `lambda=0` is one-step TD, `lambda=1` is Monte Carlo (1:26:19).