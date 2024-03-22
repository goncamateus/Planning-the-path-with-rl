import gymnasium
import numpy as np

from rsoccer_gym.Entities import Robot, Frame, Ball

from envs.enhanced import (
    SSLPathPlanningEnv,
    SPEED_MIN_TOLERANCE,
    SPEED_MAX_TOLERANCE,
    Point2D,
    KDTree,
)


class ObstacleEnv(SSLPathPlanningEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(
        self,
        n_robots_yellow=1,
        repeat_action=16,
        render_mode=None,
    ):
        super().__init__(
            n_robots_yellow=n_robots_yellow,
            field_type=2,
            render_mode=render_mode,
            repeat_action=repeat_action,
        )
        self.do_random_walk = np.random.randint(2)
        n_obs = 6 + 7 * self.n_robots_blue + 5 * self.n_robots_yellow
        self.observation_space = gymnasium.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

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

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

        return np.array(observation, dtype=np.float32)

    def _calculate_reward_and_done(self):
        reward, done = super()._calculate_reward_and_done()
        reward += self._obstacle_reward()
        if self._check_collision():
            done = True
            reward = -1000
        if done:
            self.do_random_walk = np.random.randint(2)
        return reward, done

    def _check_collision(self):
        for i in range(len(self.frame.robots_yellow)):
            obstacle_pos = np.array(
                [
                    self.frame.robots_yellow[i].x,
                    self.frame.robots_yellow[i].y,
                ]
            )
            agent_pos = np.array(
                (
                    self.frame.robots_blue[0].x,
                    self.frame.robots_blue[0].y,
                )
            )
            dist = np.linalg.norm(agent_pos - obstacle_pos)
            if dist < 0.2:
                return True
        return False

    def _obstacle_reward(self):
        reward = 0
        agent_pos = np.array(
            (
                self.frame.robots_blue[0].x,
                self.frame.robots_blue[0].y,
            )
        )
        for i in range(len(self.frame.robots_yellow)):
            obstacle_pos = np.array(
                [
                    self.frame.robots_yellow[i].x,
                    self.frame.robots_yellow[i].y,
                ]
            )
            dist = np.linalg.norm(agent_pos - obstacle_pos)
            std = 1
            exponential = np.exp((-0.5) * (dist / std) ** 2)
            gaussian = exponential / (std * np.sqrt(2 * np.pi))
            reward -= gaussian
        return reward

    def _get_commands(self, action):
        commands = super()._get_commands(action)
        if self.do_random_walk:
            yellow_commands = []
            for i in range(self.n_robots_yellow):
                yellow_commands.append(
                    Robot(
                        yellow=True,
                        id=self.frame.robots_yellow[i].id,
                        v_x=np.random.uniform(-2.5, 2.5),
                        v_y=np.random.uniform(-2.5, 2.5),
                        v_theta=np.random.uniform(-np.pi, np.pi),
                    )
                )
            commands = commands + yellow_commands
        return commands

    def _get_initial_positions_frame(self):
        pos_frame = super()._get_initial_positions_frame()
        # Put the obstacle between the agent and the target
        agent_pos = np.array(
            (
                pos_frame.robots_blue[0].x,
                pos_frame.robots_blue[0].y,
            )
        )
        target_pos = np.array(
            (
                self.target_point.x,
                self.target_point.y,
            )
        )
        obstacle_pos = agent_pos + (target_pos - agent_pos) / 2
        pos_frame.robots_yellow[0].x = obstacle_pos[0]
        pos_frame.robots_yellow[0].y = obstacle_pos[1]
        return pos_frame


class TestObstacleEnv(ObstacleEnv):

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
        # Put the obstacle between the agent and the target
        agent_pos = np.array(
            (
                pos_frame.robots_blue[0].x,
                pos_frame.robots_blue[0].y,
            )
        )
        target_pos = np.array(
            (
                self.target_point.x,
                self.target_point.y,
            )
        )
        obstacle_pos = agent_pos + (target_pos - agent_pos) / 2
        pos_frame.robots_yellow[0].x = obstacle_pos[0]
        pos_frame.robots_yellow[0].y = obstacle_pos[1]
        return pos_frame
