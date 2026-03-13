"""
Temporal Difference (TD) Methods for Reinforcement Learning

TD methods combine Monte Carlo (learn from experience) and Dynamic Programming
(bootstrap from estimates). Unlike Monte Carlo, updates happen after every step —
no need to wait for the episode to end.

TD Update Rule:
    V(S) ← V(S) + α [ R + γ·V(S') − V(S) ]
          - R + γ·V(S')   : TD Target  (immediate reward + discounted next-state estimate)
          - TD Error       : difference between TD target and current estimate

Two algorithms implemented here:
    1. TD(0) Prediction  – estimates V(s) for a fixed policy
    2. SARSA (TD Control) – learns an optimal policy via Q(s, a) updates, S A R S' A'

SARSA Update rule:
    Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
"""

import numpy as np


# ============================================================================
# GRID WORLD SIMULATOR  (same environment as MonteCarlo.py)
# ============================================================================

def step(state, action, grid_size=4):
    """
    Take one step in a 4x4 grid world.
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: -1 per step, +10 at goal (state 15), episode ends at goal.
    Returns: (next_state, reward, done)
    """
    row, col = divmod(state, grid_size)

    if action == 0:    row = max(row - 1, 0)               # up
    elif action == 1:  col = min(col + 1, grid_size - 1)   # right
    elif action == 2:  row = min(row + 1, grid_size - 1)   # down
    elif action == 3:  col = max(col - 1, 0)               # left

    next_state = row * grid_size + col
    done = (next_state == grid_size * grid_size - 1)       # goal = state 15
    reward = 10 if done else -1
    return next_state, reward, done


def epsilon_greedy(Q, state, epsilon):
    """Pick action epsilon-greedily from Q[state]."""
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])   # explore
    return int(np.argmax(Q[state]))            # exploit


# ============================================================================
# TD(0) PREDICTION  –  estimate V(s) under a random policy
# ============================================================================

def td0_prediction(num_episodes=500, alpha=0.1, gamma=0.9):
    """
    TD(0): estimate the state-value function V(s) for a random policy.

    Update after every single step (online):
        V(S) ← V(S) + α [ R + γ·V(S') − V(S) ]

    Uses a random (uniform) policy to keep the focus on the TD update rule.
    """
    V = np.zeros(16)   # state-value table

    for ep in range(num_episodes):
        state = 0      # always start at top-left

        while True:
            # Random because we are just evaluating the value functions
            action = np.random.randint(4)                              # random policy
            next_state, reward, done = step(state, action)

            # TD target = immediate reward + discounted next-state value
            td_target = reward + gamma * V[next_state] * (not done)
            td_error  = td_target - V[state]
            V[state] += alpha * td_error                               # TD(0) update

            state = next_state
            if done:
                break

    return V


# ============================================================================
# SARSA (TD CONTROL)  –  on-policy Q-learning
# ============================================================================

def sarsa_update(Q, state, action, reward, next_state, next_action, alpha, gamma, done):
    """
    One SARSA update step.

    SARSA update rule:
        Q(S, A) ← Q(S, A) + α [ R + γ·Q(S', A') − Q(S, A) ]

    'On-policy': the next action A' is sampled from the same policy being
    improved (epsilon-greedy), so we learn Q-values for the policy we're
    actually following.

    Args:
        Q:           Q-table (states × actions)
        state:       current state S
        action:      action taken A
        reward:      reward received R
        next_state:  next state S'
        next_action: next action A' (already chosen by same policy)
        alpha:       learning rate
        gamma:       discount factor
        done:        whether the episode ended
    """
    td_target = reward + gamma * Q[next_state, next_action] * (not done)
    td_error  = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    return Q


def sarsa_example(num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2, grid_size=4):
    """
    4x4 grid: Start at (0,0), Goal at (3,3).
    Actions: 0=up, 1=right, 2=down, 3=left.
    Reward: -1 per step, +10 at goal.

    Runs SARSA for num_episodes:
      - Choose A from Q using epsilon-greedy
      - Take A, observe R, S'
      - Choose A' from Q using epsilon-greedy
      - Update Q(S, A) with the SARSA rule
      - S ← S', A ← A'
    """
    GRID_SIZE = grid_size
    Q = np.zeros((GRID_SIZE*GRID_SIZE, 4))
    episode_lengths = []

    for ep in range(num_episodes):
        state  = 0
        action = epsilon_greedy(Q, state, epsilon)
        steps  = 0

        while True:
            next_state, reward, done = step(state, action, GRID_SIZE)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            Q = sarsa_update(Q, state, action, reward, next_state, next_action,
                             alpha, gamma, done)

            state  = next_state
            action = next_action
            steps += 1

            if done:
                break

        episode_lengths.append(steps)

        if (ep + 1) % 100 == 0:
            avg_len = np.mean(episode_lengths[-100:])
            print(f"Episode {ep+1:4d} | avg steps (last 100): {avg_len:.1f}")

    print("\n--- Training complete ---")
    print(f"Q(state=0,  right): {Q[0,  1]:.3f}  (go right from start)")
    print(f"Q(state=14, right): {Q[14, 1]:.3f}  (one step from goal)")
    print(f"\nBest action per state (0=up, 1=right, 2=down, 3=left):")
    action_names = ["up", "right", "down", "left"]
    for s in range(Q.shape[0]):
        row, col = divmod(s, GRID_SIZE)
        best = int(np.argmax(Q[s]))
        print(f"  state ({row},{col}): {action_names[best]}")

    return Q


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # print("=== TD(0) Prediction (random policy) ===")
    # V = td0_prediction(num_episodes=1000, alpha=0.1, gamma=0.9)
    # print("State values (0=top-left, 15=goal):")
    # print(V.reshape(4, 4).round(2))

    print("\n=== SARSA Control ===")
    sarsa_example(num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.2, grid_size=4)
