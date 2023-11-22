import random

import gymnasium as gym
import numpy as np
import pygame

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render import COLORS
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree

from envs.navigation import *

# The idea is to create an environment where the agent gives a trajectory to a certain target
# The trajectory is defined by a list of points bounded by the field
# The robot has to go from one point to the next one according to the navigation algorithm
# The robot has to follow the trajectory in order
# The episode has only one action which is trajectory
# The reward will measure if each point of the trajectory is progressive to the target
# The reward will measure if the trajectory is continuous


class TrajectoryEnv(SSLBaseEnv):
    def __init__(
        self,
        field_type: int,
        n_robots_blue: int,
        n_robots_yellow: int,
        time_step: float,
        render_mode=None,
    ):
        super().__init__(
            field_type, n_robots_blue, n_robots_yellow, time_step, render_mode
        )

        self._trajectory = []
        self._trajectory_index = 0
        self._trajectory_size = 40
        self.reward_info = {
            "reward_dist": 0,
            "reward_action_var": 0,
            "reward_continuity": 0,
            "reward_objective": 0,
            "reward_total": 0,
        }

    def _calculate_reward_dist(self, trajectory: np.ndarray, target: np.ndarray):
        distances = np.linalg.norm(trajectory - target, axis=1)
        my_arr = np.zeros(len(distances - 1))
        for i in range(len(distances) - 1):
            transition = distances[i] - distances[i + 1]
            if transition < -0.05:
                if i == 0:
                    my_arr[i] = 1
                else:
                    my_arr[i] = my_arr[i - 1] + 1
            if transition >= 0.05:
                if i == 0:
                    my_arr[i] = -1
                else:
                    if my_arr[i - 1] >= 0:
                        my_arr[i] = -1
                    else:
                        my_arr[i] = my_arr[i - 1] - 1

            if transition >= -0.05 and transition < 0.05:
                if i == 0:
                    my_arr[i] = 0
                else:
                    if my_arr[i - 1] <= 0:
                        my_arr[i] = my_arr[i - 1]
                    else:
                        my_arr[i] = 0
        return np.sum(my_arr) / np.arange(self._trajectory_size - 1).sum()

    def _calculate_reward_continuity(self, trajectory: np.ndarray):
        vectors = trajectory[1:] - trajectory[:-1]
        normed_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        pairwise_dot = np.einsum("ij,ij->i", normed_vectors[:-1], normed_vectors[1:])
        dot_in_range = (pairwise_dot > 0).astype(int)
        return np.sum(dot_in_range) / np.arange(self._trajectory_size - 1).sum()

    def _calculate_reward_action_var(self, trajectory: np.ndarray):
        distances = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
        return np.sum(distances)