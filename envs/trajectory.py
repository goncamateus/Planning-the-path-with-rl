import gymnasium as gym
import numpy as np
import pygame

from gymnasium.utils.colorize import colorize

from envs.enhanced import COLORS
from envs.enhanced import DIST_TOLERANCE
from envs.enhanced import SSLPathPlanningEnv


class TrajectoryEnv(SSLPathPlanningEnv):
    FPS = 120

    def __init__(
        self,
        field_type=2,
        n_robots_yellow=0,
        render_mode=None,
    ):
        super().__init__(field_type, n_robots_yellow, 1, render_mode)
        self._trajectory_size = 5
        self._target = np.array([0, 0])
        self._trajectory = []
        self._trajectory_idx = 0
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._trajectory_size * 2,), dtype=np.float32
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
            transition = distances[i + 1] - distances[i]
            if transition > 0:
                if i == 0:
                    my_arr[i] = 1
                else:
                    my_arr[i] = my_arr[i - 1] + 1
            if transition <= 0:
                if i == 0:
                    my_arr[i] = -1
                else:
                    if my_arr[i - 1] >= 0:
                        my_arr[i] = -1
                    else:
                        my_arr[i] = my_arr[i - 1] - 1
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
        if reward == 10:
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
        actions = actions.reshape(-1, 2)
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y
        self._trajectory = actions.copy()
        self._trajectory[:, 0] *= field_half_length
        self._trajectory[:, 1] *= field_half_width

        # This part is a plus, it allows to see the robot moving
        if self.render_mode == "human":
            for i, action in enumerate(actions):
                self._trajectory_idx = i
                robot = self.frame.robots_blue[0]
                robot_pos = np.array([robot.x, robot.y])
                robot_to_action = np.linalg.norm(robot_pos - self._trajectory[i])
                while robot_to_action > DIST_TOLERANCE:
                    with_angle = np.array([action[0], action[1], 0, 0])
                    commands = self._get_commands(with_angle)
                    self.rsim.send_commands(commands)
                    self.sent_commands = commands

                    self.last_frame = self.frame
                    self.frame = self.rsim.get_frame()

                    robot = self.frame.robots_blue[0]
                    robot_pos = np.array([robot.x, robot.y])
                    robot_to_action = np.linalg.norm(robot_pos - self._trajectory[i])
                    self.render()
        if self.render_mode != "human":
            self.frame.robots_blue[0].x = self._trajectory[-1][0]
            self.frame.robots_blue[0].y = self._trajectory[-1][1]
        reward, done = self._calculate_reward_and_done()
        observation = self._frame_to_observations()

        return observation, reward, done, False, self.reward_info

    def draw_arrow(
        self,
        surface: pygame.Surface,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color,
        body_width: int = 1,
        head_width: int = 10,
        head_height: int = 10,
    ):
        # Draw an arrow between start and end with the arrow head at the end.

        # Args:
        #     surface (pygame.Surface): The surface to draw on
        #     start (pygame.Vector2): Start position
        #     end (pygame.Vector2): End position
        #     color (pygame.Color): Color of the arrow
        #     body_width (int, optional): Defaults to 2.
        #     head_width (int, optional): Defaults to 4.
        #     head_height (float, optional): Defaults to 2.

        arrow = start - end
        angle = arrow.angle_to(pygame.Vector2(0, -1))
        body_length = arrow.length() - head_height

        # Create the triangle head around the origin
        head_verts = [
            pygame.Vector2(0, head_height / 2),  # Center
            pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
            pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
        ]
        # Rotate and translate the head into place
        translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(
            -angle
        )
        for i in range(len(head_verts)):
            head_verts[i].rotate_ip(-angle)
            head_verts[i] += translation
            head_verts[i] += start

        pygame.draw.polygon(surface, COLORS["BLACK"], head_verts)

        # Stop weird shapes when the arrow is shorter than arrow head
        if arrow.length() >= head_height:
            # Calculate the body rect, rotate and translate into place
            body_verts = [
                pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
                pygame.Vector2(body_width / 2, body_length / 2),  # Topright
                pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
                pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
            ]
            translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
            for i in range(len(body_verts)):
                body_verts[i].rotate_ip(-angle)
                body_verts[i] += translation
                body_verts[i] += start

            pygame.draw.polygon(surface, color, body_verts)

    def _render(self):
        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
                int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
            )

        super()._render()
        pos_x, pos_y = pos_transform(self._trajectory[0][0], self._trajectory[0][1])
        pygame.draw.circle(
            self.window_surface,
            COLORS["RED"],
            pygame.Vector2(pos_x, pos_y),
            self.field.rbt_radius * self.field_renderer.scale,
        )
        trajetos = self._trajectory
        for i, action in enumerate(trajetos):
            pos_x, pos_y = pos_transform(action[0], action[1])
            pygame.draw.circle(
                self.window_surface,
                self.action_color,
                pygame.Vector2(pos_x, pos_y),
                5,
                width=2,
            )
            if i < len(trajetos) - 1:
                next_action = trajetos[i + 1]
                next_pos_x, next_pos_y = pos_transform(next_action[0], next_action[1])
                self.draw_arrow(
                    self.window_surface,
                    pygame.Vector2(pos_x, pos_y),
                    pygame.Vector2(next_pos_x, next_pos_y),
                    COLORS["BLACK"],
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
        self._trajectory = [[pos_frame.robots_blue[0].x, pos_frame.robots_blue[0].y]]
        self._target = np.array([self.target_point.x, self.target_point.y])
        return pos_frame
