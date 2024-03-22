import random

import gymnasium as gym
import numpy as np
import pygame

from gymnasium.utils.colorize import colorize
from rsoccer_gym.Entities import Ball
from rsoccer_gym.Entities import Frame
from rsoccer_gym.Entities import Robot
from rsoccer_gym.Render import COLORS
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree

from envs.navigation import abs_smallest_angle_diff
from envs.navigation import dist_to
from envs.navigation import go_to_point_new
from envs.navigation import GoToPointEntryNew
from envs.navigation import MAX_VELOCITY
from envs.navigation import Point2D


ANGLE_TOLERANCE: float = np.deg2rad(5)  # 5 degrees
SPEED_MIN_TOLERANCE: float = 0.05  # m/s == 5 cm/s
SPEED_MAX_TOLERANCE: float = 0.3  # m/s == 30 cm/s
DIST_TOLERANCE: float = 0.05  # m == 5 cm


class SSLPathPlanningEnv(SSLBaseEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(
        self,
        field_type=1,
        n_robots_yellow=0,
        repeat_action=1,
        render_mode=None,
    ):
        super().__init__(
            field_type=field_type,
            n_robots_blue=1,
            n_robots_yellow=n_robots_yellow,
            time_step=0.025,
            render_mode=render_mode,
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32  # hyp tg.
        )

        n_obs = 6 + 7 * self.n_robots_blue
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        self.target_point: Point2D = Point2D(0, 0)
        self.target_angle: float = 0.0
        self.target_velocity: Point2D = Point2D(0, 0)

        self.actual_action = None
        self.last_action = None
        self.last_dist_reward = 0
        self.last_angle_reward = 0
        self.last_speed_reward = 0

        self.reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_action_var": 0,
            "reward_objective": 0,
            "reward_total": 0,
            "reward_steps": 0,
        }
        self.all_actions = []
        self.repeat_action = repeat_action
        self.action_color = COLORS["PINK"]
        self.robot_path = []
        FPS = 120
        if self.repeat_action > 1:
            FPS = np.ceil((1200 // self.repeat_action) / 10)
        self.metadata["render_fps"] = FPS

        print("Environment initialized")

    def _frame_to_observations(self):
        observation = list()

        observation.append(self.norm_pos(self.target_point.x))
        observation.append(self.norm_pos(self.target_point.y))
        observation.append(np.sin(self.target_angle))
        observation.append(np.cos(self.target_angle))
        observation.append(self.norm_v(self.target_velocity.x))
        observation.append(self.norm_v(self.target_velocity.y))

        observation.append(self.norm_pos(self.frame.robots_blue[0].x))
        observation.append(self.norm_pos(self.frame.robots_blue[0].y))
        observation.append(np.sin(np.deg2rad(self.frame.robots_blue[0].theta)))
        observation.append(np.cos(np.deg2rad(self.frame.robots_blue[0].theta)))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_x))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_y))
        observation.append(self.norm_w(self.frame.robots_blue[0].v_theta))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, action):
        robot = self.frame.robots_blue[0]
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y

        target_x = action[0] * field_half_length
        target_y = action[1] * field_half_width
        target_angle = np.arctan2(action[2], action[3])
        target_vel_x = 0
        target_vel_y = 0

        self.all_actions.append((target_x, target_y))

        entry: GoToPointEntryNew = GoToPointEntryNew()
        entry.target = Point2D(target_x, target_y)
        entry.target_angle = target_angle
        entry.target_velocity = Point2D(target_vel_x, target_vel_y)
        angle = np.deg2rad(robot.theta)
        position = Point2D(x=robot.x, y=robot.y)
        vel = Point2D(x=robot.v_x, y=robot.v_y)
        in_distance = dist_to(entry.target, self.target_point) < DIST_TOLERANCE
        in_angle = (
            abs_smallest_angle_diff(entry.target_angle, self.target_angle)
            < ANGLE_TOLERANCE
        )

        self.action_color = COLORS["PINK"]
        if in_distance and in_angle:
            self.action_color = COLORS["ORANGE"]
        elif in_distance:
            self.action_color = COLORS["BLUE"]
        elif in_angle:
            self.action_color = COLORS["GREEN"]

        result = go_to_point_new(
            agent_position=position, agent_vel=vel, agent_angle=angle, entry=entry
        )
        return [
            Robot(
                yellow=False,
                id=0,
                v_x=result.velocity.x,
                v_y=result.velocity.y,
                v_theta=result.angular_velocity,
            )
        ]

    def is_v_in_range(self, current, target) -> bool:
        return abs(current - target) <= self.SPEED_TOLERANCE

    def step(self, action):
        self.actual_action = action
        for _ in range(self.repeat_action):
            self.steps += 1
            # Join agent action with environment actions
            commands = self._get_commands(action)
            # Send command to simulator
            self.rsim.send_commands(commands)
            self.sent_commands = commands

            # Get Frame from simulator
            self.last_frame = self.frame
            self.frame = self.rsim.get_frame()
            # Calculate environment observation, reward and done condition
            observation = self._frame_to_observations()
            reward, done = self._calculate_reward_and_done()
            if done:
                break

        self.robot_path.append(
            (self.frame.robots_blue[0].x, self.frame.robots_blue[0].y)
        )
        self.last_action = action
        if self.render_mode == "human":
            self.render()
        return observation, reward, done, False, self.reward_info

    def _dist_reward(self):
        action_target_x = self.actual_action[0] * self.field.length / 2
        action_target_y = self.actual_action[1] * self.field.width / 2
        action = Point2D(x=action_target_x, y=action_target_y)
        target = self.target_point
        actual_dist = dist_to(action, target)
        reward = -actual_dist if actual_dist > DIST_TOLERANCE else 10
        return reward, actual_dist

    def _angle_reward(self):
        action_angle = np.arctan2(self.actual_action[2], self.actual_action[3])
        target = self.target_angle
        angle_diff = abs_smallest_angle_diff(action_angle, target)
        angle_reward = -angle_diff / np.pi if angle_diff > ANGLE_TOLERANCE else 1
        return angle_reward, angle_diff

    def _speed_reward(self):
        action_speed_x = self.actual_action[4] * MAX_VELOCITY
        action_speed_y = self.actual_action[5] * MAX_VELOCITY
        action_speed = Point2D(x=action_speed_x, y=action_speed_y)
        target = self.target_velocity
        vel_error = dist_to(action_speed, target)
        reward = -vel_error if vel_error > self.SPEED_TOLERANCE else 0.1
        speed_reward = reward - self.last_speed_reward
        self.last_speed_reward = reward
        return speed_reward

    def _calculate_reward_and_done(self):
        done = False
        reward = 0
        dist_reward, distance = self._dist_reward()
        angle_reward, angle_diff = self._angle_reward()
        robot_dist = np.linalg.norm(
            np.array(
                [
                    self.frame.robots_blue[0].x - self.target_point.x,
                    self.frame.robots_blue[0].y - self.target_point.y,
                ]
            )
        )
        robot_vel_error = np.linalg.norm(
            np.array(
                [
                    self.frame.robots_blue[0].v_x - self.target_velocity.x,
                    self.frame.robots_blue[0].v_y - self.target_velocity.y,
                ]
            )
        )

        if (
            distance < DIST_TOLERANCE
            and angle_diff < ANGLE_TOLERANCE
            and robot_dist < DIST_TOLERANCE
            and robot_vel_error < self.SPEED_TOLERANCE
        ):
            done = True
            reward = 1000
            self.reward_info["reward_objective"] += reward
            print(colorize("GOAL!", "green", bold=True, highlight=True))
        else:
            reward = dist_reward + angle_reward

        if done or self.steps >= 1200:
            # pairwise distance between all actions
            action_var = np.linalg.norm(
                np.array(self.all_actions[1:]) - np.array(self.all_actions[:-1])
            )
            self.reward_info["reward_action_var"] = action_var

        self.reward_info["reward_dist"] += dist_reward
        self.reward_info["reward_angle"] += angle_reward
        self.reward_info["reward_total"] += reward
        self.reward_info["reward_steps"] = self.steps

        return reward, done

    def _get_initial_positions_frame(self):
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def get_random_x():
            return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

        def get_random_y():
            return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

        def get_random_theta():
            return random.uniform(0, 360)

        def get_random_speed():
            return random.uniform(0, self.max_v)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=get_random_x(), y=get_random_y())

        self.target_point = Point2D(x=get_random_x(), y=get_random_y())
        self.target_angle = np.deg2rad(get_random_theta())
        random_speed: float = 0
        random_velocity_direction: float = np.deg2rad(get_random_theta())

        self.target_velocity = Point2D(
            x=random_speed * np.cos(random_velocity_direction),
            y=random_speed * np.sin(random_velocity_direction),
        )

        # Adjust speed tolerance according to target velocity
        target_speed_norm = np.sqrt(
            self.target_velocity.x**2 + self.target_velocity.y**2
        )
        self.SPEED_TOLERANCE = (
            SPEED_MIN_TOLERANCE
            + (SPEED_MAX_TOLERANCE - SPEED_MIN_TOLERANCE)
            * target_speed_norm
            / self.max_v
        )

        min_gen_dist = 0.2

        places = KDTree()
        places.insert((self.target_point.x, self.target_point.y))
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (get_random_x(), get_random_y())

            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(
                id=i, yellow=False, x=pos[0], y=pos[1], theta=get_random_theta()
            )

        for i in range(self.n_robots_yellow):
            pos = (get_random_x(), get_random_y())
            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(
                id=i, yellow=True, x=pos[0], y=pos[1], theta=get_random_theta()
            )
        self.last_action = None
        self.last_dist_reward = 0
        self.last_angle_reward = 0
        self.last_speed_reward = 0
        self.all_actions = []
        self.reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_action_var": 0,
            "reward_objective": 0,
            "reward_total": 0,
            "reward_steps": 0,
        }
        self.robot_path = [(pos_frame.robots_blue[0].x, pos_frame.robots_blue[0].y)] * 2
        return pos_frame

    def draw_target(self, screen, transformer, point, angle, color):
        x, y = transformer(point.x, point.y)
        size = 0.09 * self.field_renderer.scale
        pygame.draw.circle(screen, color, (x, y), size, 5)
        pygame.draw.line(
            screen,
            COLORS["BLACK"],
            (x, y),
            (
                x + size * np.cos(angle),
                y + size * np.sin(angle),
            ),
            2,
        )

    def _render(self):
        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
                int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
            )

        super()._render()
        self.draw_target(
            self.window_surface,
            pos_transform,
            self.target_point,
            self.target_angle,
            COLORS["ORANGE"],
        )
        # Draw action
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y
        if self.actual_action is not None:
            pos_x = self.actual_action[0] * field_half_length
            pos_y = self.actual_action[1] * field_half_width
            target_angle = np.arctan2(self.actual_action[2], self.actual_action[3])
            pos = Point2D(x=pos_x, y=pos_y)
            self.draw_target(
                self.window_surface,
                pos_transform,
                pos,
                target_angle,
                self.action_color,
            )
        # Draw path
        if len(self.all_actions) > 1:

            my_path = [pos_transform(*p) for p in self.all_actions[:-1]]
            for point in my_path:
                pygame.draw.circle(self.window_surface, COLORS["RED"], point, 3)
        my_path = [pos_transform(*p) for p in self.robot_path]
        pygame.draw.lines(self.window_surface, COLORS["RED"], False, my_path, 1)


class TestEnv(SSLPathPlanningEnv):

    def _get_initial_positions_frame(self):
        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=-4, y=-2)

        self.target_point = Point2D(x=2, y=0)
        self.target_angle = np.deg2rad(0)
        self.target_velocity = Point2D(x=0, y=0)

        # Adjust speed tolerance according to target velocity
        target_speed_norm = np.sqrt(
            self.target_velocity.x**2 + self.target_velocity.y**2
        )
        self.SPEED_TOLERANCE = (
            SPEED_MIN_TOLERANCE
            + (SPEED_MAX_TOLERANCE - SPEED_MIN_TOLERANCE)
            * target_speed_norm
            / self.max_v
        )

        places = KDTree()
        places.insert((self.target_point.x, self.target_point.y))
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (2, -2)
            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(
                id=i, yellow=False, x=pos[0], y=pos[1], theta=270
            )

        for i in range(self.n_robots_yellow):
            pos = (2, -1)
            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(
                id=i, yellow=True, x=pos[0], y=pos[1], theta=270
            )
        self.last_action = None
        self.last_dist_reward = 0
        self.last_angle_reward = 0
        self.last_speed_reward = 0
        self.all_actions = []
        self.reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_action_var": 0,
            "reward_objective": 0,
            "reward_total": 0,
            "reward_steps": 0,
        }
        self.robot_path = [(pos_frame.robots_blue[0].x, pos_frame.robots_blue[0].y)] * 2
        return pos_frame
