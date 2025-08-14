Few small concepts. **Greedy Policy** is where we choose the action with highest value. **Epsilon greedy policy** is where we choose action with highest value but for small amount of steps, we take another value which is not highest.

- **Process**: What happens is Agent is given a state whihc based on the policy and value function of next steps to be taken, makes an actioon. Which is then passed through the environment and environment sends back a new state and a reward.

- **Policy** is denoted by PI.

- **Expected Return**: The goal is not about maximizing the reward, its about maximizing the RETURN which is addition of all the rewards so far. It is called as _Expected Returns_. Denoted by **G_t**.

$G_t$ = $R_{t+1}$ + $R_{t+2}$ + ... + $R_T$

- **Episodic tasks** are those where a game or a goal can be achieved or not. Like Tic Tac Toe - we can loose or win. If the game ends, episode ends. **BUT** there are certain tasks where episodes doesnt make sense like Martian rover on Mars, they are called **Continuos Tasks**.

- **Discounting**: Rewards that are gained earlier are mre valuable compared to those which are given after a long time. We do `R_1 + gamma*R_2 + gamma*R3 + ...` instead of `R_1 + R_2 + R_3`; etc.

- **Discounted expected Return**:

$G_t$ = $R_{t+1}$ + r\*$R_{t+2}$ + $r^2R_{t+2}$ + ...

- Gamma is a discount rate.

### **Finite Markov Decision Processes**:

So this is the property where we Assume or sometimes even prove that to get the action of next state, we ONLY need the previous state and not ALL the previous state.

Conffusin, right ? In chess, we dont need to know what happened in apst, our current state is enough to do actions on the state while being in a middle of a conversation with someone, we need to know past conversation.
