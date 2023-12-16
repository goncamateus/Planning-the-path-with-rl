import gymnasium as gym
import envs

env = gym.make("Obstacle-v1", render_mode="human")
env.reset()
d = t = False
while not(d or t):
    s, r, d, t, i = env.step(env.action_space.sample())
    print(i)
