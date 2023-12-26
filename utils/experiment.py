import argparse
import os
import random
import time

from yaml import safe_load

import gymnasium as gym
import numpy as np
import torch
import wandb

import envs


def strtobool(value: str) -> bool:
    value = value.lower()
    if value in ("y", "yes", "on", "1", "true", "t"):
        return True
    return False


def make_env(args, idx, run_name):
    def thunk():
        if args.capture_video and idx == 0:
            env = gym.make(args.gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % args.video_freq == 0,
            )
        else:
            env = gym.make(args.gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(args.seed)
        return env

    return thunk


def base_hyperparams():
    hyper_params = {
        "seed": 0,
        "total_timesteps": 1000000,
        "torch_deterministic": True,
        "cuda": True,
        "capture_video": False,
        "video_freq": 50,
        "track": False,
        "num_envs": 1,
        "buffer_size": int(1e6),
        "gamma": 0.99,
        "tau": 1 - 5e-3,
        "batch_size": 256,
        "exploration_noise": 0.1,
        "learning_starts": 5e3,
        "policy_lr": 3e-4,
        "q_lr": 1e-3,
        "policy_frequency": 2,
        "target_network_frequency": 1,
        "noise_clip": 0.5,
        "alpha": 0.2,
        "epsilon": 1e-6,
        "autotune": True,
        "reward_scaling": 1.0,
        "lambda_temporal": 0,
        "lambda_spacial": 0,
        "caps_epsilon": 1,
    }
    return hyper_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Baseline")
    parser.add_argument("--setup", type=str, default="Pure")
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)",
    )
    parser.add_argument(
        "--video-freq",
        type=int,
        default=50,
        help="Frequency of saving videos, in episodes",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Log on wandb",
    )
    args = parser.parse_args()
    args.env = args.env.lower().title()
    args.setup = args.setup.lower().upper()
    return args


def get_experiment(arguments):
    with open("experiments.yml", "r") as f:
        params = safe_load(f)
    experiment = base_hyperparams()
    experiment.update(vars(arguments))
    experiment.update(params[arguments.env][arguments.setup])
    experiment = argparse.Namespace(**experiment)
    return experiment


def setup_run(exp_name, params):
    project = "PathPlanning"
    if params.seed == 0:
        params.seed = int(time.time())
    params.method = "sac"
    wandb_run = wandb.init(
        project=project,
        name=exp_name,
        entity="goncamateus",
        config=vars(params),
        monitor_gym=True,
        mode=None if params.track else "disabled",
        save_code=True,
    )
    artifact = wandb.Artifact("model", type="model")
    print(vars(params))

    # TRY NOT TO MODIFY: seeding
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic

    return wandb_run, artifact
