import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

env.reset()

for step in range(200):
    env.render()
    env.step(env.action_space.sample()) # we are just passing in sample actions but once we have a policy function everything would work fine !

env.close()