import gymnasium as gym
import envs

env = gym.make("Obstacle-v0", render_mode="human")
env.reset()
env.step(env.action_space.sample())
input()