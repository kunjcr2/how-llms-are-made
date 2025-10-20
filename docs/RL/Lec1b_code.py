import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human") # rendering the environment
env.reset() # this gets the first observation of the environment
EPS = 200

# Running till EPS time
for ep in range(EPS):
    env.render() # this renders the updated environment to us !
    env.step(env.action_space.sample()) # we are passing a rendom action sample to the .step()

    # .step() does the actiono n the previous state and gets updated step and reward in return !

env.close()