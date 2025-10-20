import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

eps = 10

for ep in range(eps):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample() # Policy function (a random)
        state, reward, terminated, truncated, info = env.step(action)
        
        done = truncated or terminated
        total_reward += reward

    print(f'Episode {ep+1} finished with Reward - {total_reward}')

env.close()