"""Utility functions for path planning"""

from dataclasses import dataclass
from collections import namedtuple
from typing import Final, Optional
import math
import numpy as np

PROP_VELOCITY_MIN_FACTOR: Final[float] = 0.1
MAX_VELOCITY: Final[float] = 2.2
ANGLE_EPSILON: Final[float] = 0.1
ANGLE_KP: Final[float] = 2
ROTATE_IN_POINT_MIN_VEL_FACTOR: Final[float] = 0.18
ROTATE_IN_POINT_APPROACH_KP: Final[float] = 2
ROTATE_IN_POINT_MAX_VELOCITY: Final[float] = 1.8
ROTATE_IN_POINT_ANGLE_KP: Final[float] = 5
MIN_DIST_TO_PROP_VELOCITY: Final[float] = 720

ADJUST_ANGLE_MIN_DIST: Final[float] = 50

M_TO_MM: Final[float] = 1000.0

# added variables (needs adjusting):
MAX_ACCEL: Final[float] = 1
MAX_DV: Final[float] = 0.1
EPS: Final[float] = 10e-4
MIN_ANGLE_TO_ROTATE: Final[float] = np.deg2rad(2.5)


Point2D = namedtuple("Point2D", ["x", "y"])
RobotMove = namedtuple("RobotMove", ["velocity", "angular_velocity"])


def dist_to(p_1: Point2D, p_2: Point2D) -> float:
    """Returns the distance between two points"""
    return ((p_1.x - p_2.x) ** 2 + (p_1.y - p_2.y) ** 2) ** 0.5


def length(point: Point2D) -> float:
    """Returns the length of a vector"""
    return (point.x**2 + point.y**2) ** 0.5


def normalize(point: Point2D) -> float:
    """Returns the normalized vector"""
    return Point2D(point.x / length(point), point.y / length(point))


def pt_angle(point: Point2D) -> float:
    """Returns the angle of a vector"""
    return math.atan2(point.y, point.x)


def math_map(
    value: float, in_min: float, in_max: float, out_min: float, out_max: float
) -> float:
    """Maps a value from one range to another"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def math_bound(value: float, min_val: float, max_val: float) -> float:
    """Bounds a value between a min and max value"""
    return min(max_val, max(min_val, value))


def math_modularize(value: float, mod: float) -> float:
    """Make a value modular between 0 and mod"""
    if not -mod <= value <= mod:
        value = math.fmod(value, mod)

    if value < 0:
        value += mod

    return value


def smallest_angle_diff(angle_a: float, angle_b: float) -> float:
    """Returns the smallest angle difference between two angles"""
    angle: float = math_modularize(angle_b - angle_a, 2 * math.pi)
    if angle >= math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle


def abs_smallest_angle_diff(angle_a: float, angle_b: float) -> float:
    """Returns the absolute smallest angle difference between two angles"""
    return abs(smallest_angle_diff(angle_a, angle_b))


def from_polar(radius: float, theta: float) -> Point2D:
    """Returns a point from polar coordinates"""
    return Point2D(radius * math.cos(theta), radius * math.sin(theta))


def cross(p_1: Point2D, p_2: Point2D) -> float:
    """Returns the cross product of two points"""
    return p_1.x * p_2.y - p_1.y * p_2.x


def dot(p_1: Point2D, p_2: Point2D) -> float:
    """Returns the dot product of two points"""
    return p_1.x * p_2.x + p_1.y * p_2.y


def angle_between(p_1: Point2D, p_2: Point2D) -> float:
    """Returns the angle between two points"""
    return math.atan2(cross(p_1, p_2), dot(p_1, p_2))


def rotate_to_local(p_global: Point2D, robot_angle: float) -> Point2D:
    """Returns the point p_global expressed in robot's local frame"""
    local_x = np.cos(robot_angle) * p_global.x + np.sin(robot_angle) * p_global.y
    local_y = -np.sin(robot_angle) * p_global.x + np.cos(robot_angle) * p_global.y
    return Point2D(local_x, local_y)


@dataclass()
class GoToPointEntryNew:
    """Go to point entry"""

    target: Point2D = Point2D(0.0, 0.0)
    target_angle: float = 0.0
    target_velocity: Point2D = Point2D(0.0, 0.0)

    max_velocity: Optional[float] = None
    k_p: Optional[float] = None
    max_accel: Optional[float] = None


def rotate_vector(vector: Point2D, angle_radians: float) -> Point2D:
    # Convert to numpy
    vector = np.array([vector.x, vector.y])

    # Compute rotation
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )
    rotated_vector = np.dot(rotation_matrix, vector)

    # Convert back to Point2D
    rotated_vector = Point2D(rotated_vector[0], rotated_vector[1])

    return rotated_vector


def go_to_point_new(
    agent_position: Point2D,
    agent_vel: Point2D,
    agent_angle: float,
    entry: GoToPointEntryNew,
) -> RobotMove:
    """Returns the robot move"""
    # If Player send max speed, this max speed has to be respected
    # Ohterwise, use the max speed received in the parameter
    max_velocity: float = entry.max_velocity if entry.max_velocity else MAX_VELOCITY

    # Rotate axis and express vectors in terms of the target velocity's direction
    if np.linalg.norm(entry.target_velocity)<=EPS:
        robot_to_target = Point2D(entry.target.x - agent_position.x, entry.target.y - agent_position.y)
        target_velocity_angle = np.arctan2(robot_to_target.y, robot_to_target.x)
    else:
        target_velocity_angle = np.arctan2(entry.target_velocity.y, entry.target_velocity.x)

    entry.target_velocity = rotate_vector(entry.target_velocity, -target_velocity_angle)
    agent_vel = rotate_vector(agent_vel, -target_velocity_angle)
    entry.target = rotate_vector(entry.target, -target_velocity_angle)
    agent_position = rotate_vector(agent_position, -target_velocity_angle)
    robot_to_target = Point2D(
        entry.target.x - agent_position.x, entry.target.y - agent_position.y
    )

    # CALCULATING VX
    vx_desired = entry.target_velocity.x
    vx_current = agent_vel.x
    x_desired = entry.target.x
    robot_to_target_x = robot_to_target.x

    # Checks whether the robot must deaccelerate to reach the next desired state
    robot_has_passed_the_target = robot_to_target.x * vx_desired < 0

    # Checks if the robot is moving away from target
    robot_is_moving_away_from_target = robot_to_target.x * vx_current < 0

    # Robot should adopt 0-point as target
    robot_should_go_to_0_point = (
        robot_has_passed_the_target or robot_is_moving_away_from_target
    )

    # Change target to 0-point if needed
    if robot_should_go_to_0_point:
        zero_point_x = x_desired - np.sign(vx_desired) * vx_desired**2 / (
            2 * MAX_ACCEL
        )
        robot_to_target_x = zero_point_x - agent_position.x
        vx_desired = 0

    # Checks distance to start changing to target velocity
    x_dist_to_target = abs(robot_to_target_x)
    min_dist_to_deaccel = abs(vx_current**2 - vx_desired**2) / (2 * MAX_ACCEL)
    robot_in_start_deaccel_dist = x_dist_to_target < min_dist_to_deaccel

    # Checks if distance to start changing to target velocity was reached
    if robot_in_start_deaccel_dist:
        vx = vx_current + np.sign(vx_desired - vx_current) * MAX_DV

    # Else, move towards target
    else:
        vx = vx_current + np.sign(robot_to_target_x) * MAX_DV

    # Checks if the robot is already on maximum velocity
    if abs(vx) > max_velocity:
        vx = np.sign(vx) * max_velocity

    # CALCULATING VY
    vy_desired = entry.target_velocity.y
    vy_current = agent_vel.y
    y_desired = entry.target.y
    robot_to_target_y = robot_to_target.y

    # Checks whether the robot must deaccelerate to reach the next desired state
    robot_has_passed_the_target = robot_to_target.y * vy_desired < 0

    # Checks if the robot is moving away from target
    robot_is_moving_away_from_target = robot_to_target.y * vy_current < 0

    # Robot should adopt 0-point as target
    robot_should_go_to_0_point = (
        robot_has_passed_the_target or robot_is_moving_away_from_target
    )

    # Change target to 0-point if needed
    if robot_should_go_to_0_point:
        zero_point_y = y_desired - np.sign(vy_desired) * vy_desired**2 / (
            2 * MAX_ACCEL
        )
        robot_to_target_y = zero_point_y - agent_position.y
        vy_desired = 0

    # Checks distance to start changing to target velocity
    y_dist_to_target = abs(robot_to_target_y)
    min_dist_to_deaccel = abs(vy_current**2 - vy_desired**2) / (2 * MAX_ACCEL)
    robot_in_start_deaccel_dist = y_dist_to_target < min_dist_to_deaccel

    # Checks if distance to start changing to target velocity was reached
    if robot_in_start_deaccel_dist:
        vy = vy_current + np.sign(vy_desired - vy_current) * MAX_DV

    # Else, move towards target
    else:
        vy = vy_current + np.sign(robot_to_target_y) * MAX_DV

    # Checks if the robot is already on maximum velocity
    if abs(vy) > max_velocity:
        vy = np.sign(vy) * max_velocity

    v_desired = Point2D(vx, vy)

    # Rotate Vx e Vy back to global axis
    v_desired = rotate_vector(v_desired, target_velocity_angle)

    # Rotate to robot's local frame
    v_desired_local = rotate_to_local(v_desired, agent_angle)

    # CALCULATING VW
    k_p: float = entry.k_p if entry.k_p else ANGLE_KP
    angle_to_rotate = smallest_angle_diff(agent_angle, entry.target_angle)
    if abs(angle_to_rotate) < MIN_ANGLE_TO_ROTATE:
        vw = 0
    else:
        vw = k_p * angle_to_rotate

    return RobotMove(v_desired_local, vw)
