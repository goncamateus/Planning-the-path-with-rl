import argparse
import envs
import gymnasium as gym
import numpy as np
import torch
import pandas as pd

from pyvirtualdisplay import Display

from methods.sac import GaussianPolicy

from utils.experiment import strtobool


def load_model(model_path):
    model = torch.load(model_path)
    return model


def main(env_id, caps):
    env = gym.make(env_id, render_mode="rgb_array")
    path = env_id + "-caps" if caps else env_id
    state_dict = load_model(f"models/{path}/actor.pt")
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
    actor.load_state_dict(state_dict)
    actor.eval()
    actor.to("cpu")
    logs = []
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        trunc = False
        while not (done or trunc):
            state = torch.Tensor(obs.reshape(1, -1))
            action = actor.get_action(state)[0]
            obs, reward, done, trunc, info = env.step(action)
        log = {
            "Episode Length": info["reward_steps"],
            "Episode Length (seconds)": info["reward_steps"] * 0.025,
            "Cumulative Pairwise Action Distance": info["reward_action_var"],
        }
        logs.append(log)
    env.close()
    df = pd.DataFrame(logs)
    df.to_csv(f"results_analysis/{path}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym-id", type=str, default="Baseline-v0")
    parser.add_argument(
        "--caps",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
    )
    args = parser.parse_args()
    main(env_id=args.gym_id, caps=args.caps)
