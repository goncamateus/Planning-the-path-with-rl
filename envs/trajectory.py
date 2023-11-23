import random

import gymnasium as gym
import numpy as np
import pygame

from envs.navigation import *
from gymnasium.utils.colorize import colorize
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render import COLORS
from rsoccer_gym.Utils import KDTree
from envs.enhanced import DIST_TOLERANCE, SSLPathPlanningEnv


class TrajectoryEnv(SSLPathPlanningEnv):
    def __init__(
        self,
        field_type=2,
        n_robots_yellow=0,
        render_mode=None,
    ):
        super().__init__(field_type, 1, n_robots_yellow, render_mode)
        self._trajectory_size = 40
        self._target = np.array([0, 0])
        self._trajectory = []
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._trajectory_size, 2), dtype=np.float32
        )
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
            if transition < -DIST_TOLERANCE:
                if i == 0:
                    my_arr[i] = 1
                else:
                    my_arr[i] = my_arr[i - 1] + 1
            if transition >= DIST_TOLERANCE:
                if i == 0:
                    my_arr[i] = -1
                else:
                    if my_arr[i - 1] >= 0:
                        my_arr[i] = -1
                    else:
                        my_arr[i] = my_arr[i - 1] - 1

            if transition >= -DIST_TOLERANCE and transition < DIST_TOLERANCE:
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

    def _calculate_reward_objective(self, trajectory: np.ndarray, target: np.ndarray):
        dist_to_target = np.linalg.norm(trajectory[-1] - target)
        reward = -dist_to_target if dist_to_target > DIST_TOLERANCE else 10
        print(colorize("GOAL!", "green", bold=True, highlight=True))
        return reward

    def _calculate_action_var(self, trajectory: np.ndarray):
        distances = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
        return np.sum(distances)

    def _calculate_reward_and_done(self):
        reward_dist = self._calculate_reward_dist(self._trajectory, self._target)
        reward_continuity = self._calculate_reward_continuity(self._trajectory)
        reward_objective = self._calculate_reward_objective(
            self._trajectory, self._target
        )
        action_var = self._calculate_action_var(self._trajectory)
        self.reward_info["reward_dist"] += reward_dist
        self.reward_info["reward_continuity"] += reward_continuity
        self.reward_info["reward_objective"] += reward_objective
        self.reward_info["reward_action_var"] += action_var

        reward = reward_dist + reward_continuity + reward_objective
        self.reward_info["reward_total"] += reward

        return reward, True

    def step(self, actions: np.ndarray):
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y
        self._trajectory = actions.copy()
        self._trajectory[:, 0] *= field_half_length
        self._trajectory[:, 1] *= field_half_width
        reward, done = self._calculate_reward_and_done()

        # This part is a plus, it allows to see the robot moving
        for action in actions:
            robot = self.frame.robots_blue[0]
            robot_pos = np.array([robot.x, robot.y])
            robot_to_action = np.linalg.norm(robot_pos - action)
            while robot_to_action < DIST_TOLERANCE:
                commands = self._get_commands(action)
                self.rsim.send_commands(commands)
                self.sent_commands = commands

                self.last_frame = self.frame
                self.frame = self.rsim.get_frame()
                if self.render_mode == "human":
                    self.render()

                robot = self.frame.robots_blue[0]
                robot_pos = np.array([robot.x, robot.y])
                robot_to_action = np.linalg.norm(robot_pos - action)

        observation = self._frame_to_observations()

        return observation, reward, done, False, self.reward_info

    def _render(self):
        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
                int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
            )

        super()._render()
        for action in self._trajectory:
            pos_x, pos_y = pos_transform(action[0], action[1])
            pos = Point2D(x=pos_x, y=pos_y)
            self.draw_target(
                self.window_surface,
                pos_transform,
                pos,
                self.target_angle,
                self.action_color,
            )

    def _get_initial_positions_frame(self):
        pos_frame = super()._get_initial_positions_frame()
        self.reward_info = {
            "reward_dist": 0,
            "reward_action_var": 0,
            "reward_continuity": 0,
            "reward_objective": 0,
            "reward_total": 0,
        }
        self._trajectory = []
        self._target = np.array([self.target_point.x, self.target_point.y])
        return pos_frame
