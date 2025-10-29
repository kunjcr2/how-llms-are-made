"""
Reinforcement Learning: Q-Learning vs Monte Carlo

An agent learns optimal actions by trial and error:
- Q(s,a): Expected cumulative reward for action a in state s
- Goal: Learn Q-values to find best policy (action selection strategy)

Two approaches:
1. Monte Carlo: Wait for episode to finish, learn from actual outcomes
2. Q-Learning: Learn immediately from each step using estimates
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon, num_actions):
    """
    Balance exploration (try new things) vs exploitation (use best known action).
    
    With probability epsilon: random action (explore)
    Otherwise: action with max Q-value (exploit)
    """
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(Q[state])


def monte_carlo_update(episode, Q, gamma=0.99):
    """
    Update Q-values from complete episode using actual returns.
    
    How it works:
    1. Episode finishes (e.g., game ends)
    2. Calculate return G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
       (gamma discounts future rewards: γ=0.9 means future worth 90% of now)
    3. Update Q(s,a) = average of all returns seen from that (s,a)
    
    Pros: Simple, unbiased (uses real outcomes)
    Cons: Must wait for complete episodes
    
    Args:
        episode: [(state, action, reward), ...] from one complete episode
        Q: Q-table to update
        gamma: Discount factor (0 to 1, typically 0.9-0.99)
    """
    returns_sum = {}  # Accumulated returns for each (state, action)
    visit_count = {}  # How many times we've seen each (state, action)
    G = 0  # Return (cumulative discounted reward)
    
    # Work backwards through episode to calculate returns
    for state, action, reward in reversed(episode):
        G = reward + gamma * G  # Add current reward to discounted future
        
        sa_pair = (state, action)
        if sa_pair not in visit_count:
            returns_sum[sa_pair] = 0
            visit_count[sa_pair] = 0
        
        returns_sum[sa_pair] += G
        visit_count[sa_pair] += 1
        
        # Q-value = average return from this state-action pair
        Q[state, action] = returns_sum[sa_pair] / visit_count[sa_pair]
    
    return Q


def q_learning_update(state, action, reward, next_state, Q, 
                      alpha=0.1, gamma=0.99, done=False):
    """
    Update Q-value immediately using Temporal Difference learning.
    
    Core idea: Q(s,a) should equal reward + discounted best future Q-value
    Update rule: Q(s,a) ← Q(s,a) + α * [TD_target - Q(s,a)]
                                         └── TD_error ──┘
    
    Where TD_target = r + γ * max_a' Q(s', a')  if not done
                    = r                          if done
    
    How it works:
    1. Take action, observe reward r and next state s'
    2. Estimate total return: r + γ*max(Q(s',a'))
    3. Update current Q(s,a) toward this estimate
    4. Alpha controls learning speed (0.1 = take 10% step toward target)
    
    Pros: Learns immediately, works with continuing tasks, fast
    Cons: Uses estimates (bootstrapping), can be unstable
    
    Args:
        state, action: What you did
        reward: Immediate reward received
        next_state: Where you ended up
        Q: Q-table to update
        alpha: Learning rate (0 to 1, typically 0.01-0.5)
        gamma: Discount factor
        done: True if episode ended (no next state)
    """
    if done:
        td_target = reward  # No future rewards
    else:
        # Best possible future value from next_state
        td_target = reward + gamma * np.max(Q[next_state])
    
    # How wrong was our Q-value?
    td_error = td_target - Q[state, action]
    
    # Move Q-value toward target
    Q[state, action] += alpha * td_error
    
    return Q


# ============================================================================
# SIMPLE GRIDWORLD EXAMPLE
# ============================================================================

def gridworld_example():
    """
    4x4 grid: Start at (0,0), Goal at (3,3)
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: -1 per step, +10 at goal
    """
    # States 0-15 (flattened 4x4 grid), 4 actions
    Q = np.zeros((16, 4))
    
    # Monte Carlo: One complete episode - [(box, action, reward)]
    episode = [(0,1,-1), (1,1,-1), (2,2,-1), (6,2,-1), (10,1,-1), (11,2,-1), (15,0,10)]
    Q_mc = monte_carlo_update(episode, Q.copy(), gamma=0.9)
    print("Monte Carlo learned Q(0,1):", Q_mc[0,1])
    
    # Q-Learning: Step by step
    Q_ql = Q.copy()
    for s, a, r, s_next, done in [(0,1,-1,1,False), (1,1,-1,2,False), (15,0,10,15,True)]:
        Q_ql = q_learning_update(s, a, r, s_next, Q_ql, alpha=0.1, gamma=0.9, done=done)
    print("Q-Learning learned Q(0,1):", Q_ql[0,1])


if __name__ == "__main__":
    gridworld_example()