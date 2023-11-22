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
