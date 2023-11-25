import gymnasium as gym
import envs

env = gym.make("Trajectory-v0", render_mode="human")
env.reset()
s, r, d, t, i = env.step(env.action_space.sample())
print(i)
input()