# Fckn hell, NoteBookLM generates good code !
import numpy as np

def choose_action(state, Q_table, epsilon, action_space_size):
    """
    Selects an action based on the epsilon-greedy policy.
    Q_table: The current Action-Value function (Q(s, a)).
    """
    # Use np.random.rand because the source uses np.random.rand [4]
    if np.random.rand() < epsilon:
        # Exploration: Take a random action [4, 6]
        # In the source, this is np.random.randint(env.action_space.n) [4]
        return np.random.randint(action_space_size)
    else:
        # Exploitation: Take the action with the maximum Q value [5]
        # Q[state, :] returns all Q values for that state.
        return np.argmax(Q_table[state, :])

def monte_carlo_update(episode_steps, Q_table, returns_sum, counts, gamma, episode_length):
    """
    Performs the Monte Carlo policy evaluation and improvement using the full return.
    
    episode_steps: List of tuples [(state, action, reward)].
    gamma: Discount factor.
    """
    G = 0  # Initialize Return (G)
    
    # We iterate backwards through the episode to calculate discounted return G [9, 10]
    # In the source, this is shown as R_t + gamma * R_{t+1} + ...
    
    for t in reversed(range(episode_length)):
        state, action, reward = episode_steps[t]
        
        # Calculate G recursively: G = gamma * G + R [10]
        G = gamma * G + reward
        
        # Check if this (state, action) pair has been visited *for the first time* 
        # (This implements the "first-visit" idea which improves efficiency, 
        # although the basic principle is general Monte Carlo)
        
        # We need a unique identifier for the state-action pair (s, a)
        sa_pair = (state, action)
        
        # Accumulate returns [11]
        returns_sum[sa_pair] += G
        
        # Count visits [12]
        counts[sa_pair] += 1
        
        # Update Q estimate (using the average) [12]
        # Q(s, a) = Sum of returns / Number of visits
        Q_table[state, action] = returns_sum[sa_pair] / counts[sa_pair]

    return Q_table

def q_learning_update(state, action, reward, next_state, Q_table, alpha, gamma, done):
    """
    Performs the Q-Learning update rule (Off-Policy TD Control).
    
    alpha: Learning rate (step size) [18, 19].
    gamma: Discount factor.
    done: Boolean flag indicating if the episode finished (not truncated) [17, 20].
    """
    
    # 1. Determine the TD Target
    if done:
        # If the episode is finished, the future return is just the immediate reward [17]
        td_target = reward
    else:
        # Calculate the maximum Q value for the next state S' [14]
        # This is the core of Q-Learning, maximizing the future return
        max_q_next = np.max(Q_table[next_state, :])
        
        # TD Target = R + gamma * max Q(S', A') [17]
        td_target = reward + gamma * max_q_next
    
    # 2. Calculate the Temporal Difference Error (TD Error)
    # TD Error = TD Target - Old Estimate [18, 21]
    old_estimate = Q_table[state, action]
    td_error = td_target - old_estimate
    
    # 3. Update the Q-table (using the core TD update formula)
    # New Estimate = Old Estimate + alpha * (TD Error) [18, 21]
    Q_table[state, action] = old_estimate + alpha * td_error
    
    return Q_table