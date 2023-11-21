from gymnasium.envs.registration import register

register(
    id="PathPlanning-v0",
    entry_point="envs.baseline:SSLPathPlanningBaseLineEnv",
    kwargs={"field_type": 2, "n_robots_yellow": 0},
    max_episode_steps=1200,
)

register(
    id="PathPlanning-v1",
    entry_point="envs.enhanced:SSLPathPlanningEnv",
    kwargs={"field_type": 2, "n_robots_yellow": 0, "repeat_action": 1},
    max_episode_steps=1200,
)

register(
    id="PathPlanning-v2",
    entry_point="envs.enhanced:SSLPathPlanningEnv",
    kwargs={"field_type": 2, "n_robots_yellow": 0, "repeat_action": 16},
    max_episode_steps=75,
)

register(
    id="PathPlanningMedium-v0",
    entry_point="envs.enhanced:SSLPathPlanningMediumEnv",
    kwargs={"n_robots_yellow": 1},
    max_episode_steps=75,
)