import numpy as np
from envs.enhanced import SSLPathPlanningEnv


class SSLPathPlanningMediumEnv(SSLPathPlanningEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(
        self,
        n_robots_yellow=1,
        render_mode=None,
    ):
        super().__init__(
            n_robots_yellow=n_robots_yellow,
            field_type=2,
            repeat_action=16,
            render_mode=render_mode,
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

        return np.array(observation, dtype=np.float32)

    def _calculate_reward_and_done(self):
        reward, done = super()._calculate_reward_and_done()
        reward += self._obstacle_reward()
        if self._check_collision():
            done = True
            reward = -1000
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