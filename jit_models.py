import os
import envs
import gymnasium as gym
import numpy as np
import torch

from methods.sac import GaussianPolicy

from utils.experiment import strtobool


def load_model(model_path):
    model = torch.load(model_path, map_location="cpu")
    return model


def jit_model(env_id, caps):
    env = gym.make(env_id, render_mode="human")
    path = env_id + "-caps" if caps else env_id
    state_dict = load_model(f"trained_models/{path}/actor.pt")
    num_inputs = np.array(env.observation_space.shape).prod()
    num_actions = np.array(env.action_space.shape).prod()
    actor = GaussianPolicy(
        num_inputs,
        num_actions,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
        epsilon=1e-6,
        action_space=env.action_space,
    )
    print(f"Action scale: {actor.action_scale}")
    print(f"Action bias: {actor.action_bias}")
    actor.load_state_dict(state_dict)
    actor.eval()
    actor.to("cpu")
    obs, _ = env.reset()
    traced_script_module = torch.jit.trace(actor, torch.Tensor(obs.reshape(1, -1)))
    traced_script_module.save(f"trained_models/{path}/actor_jit.pt")


if __name__ == "__main__":
    ids = [
        "Baseline-v0",
        "Baseline-v1",
        "Enhanced-v0",
        "Enhanced-v1",
        "Obstacle-v0",
        "Obstacle-v1",
    ]
    for gym_id in ids:
        jit_model(env_id=gym_id, caps=False)
        jit_model(env_id=gym_id, caps=True)
