"""The regression that motivated rebot_gripette.follower.

Upstream, get_observation() reports raw motor degrees while send_action()
multiplies by joint_directions. Feeding an observation back as an action - what
teleop recording, dataset replay and policy rollout all do - therefore mirrors
every joint with direction -1 and scales the gripper by 6. These tests drive
the real send_action()/get_observation() against fake motors and assert the
round-trip is the identity.
"""
import math

import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from lerobot_robot_seeed_b601 import SeeedB601RSFollower

from rebot_gripette import ReBotFollower

# A pose in LOGICAL space (the space send_action accepts and, after the fix,
# get_observation reports). Chosen so every joint stays inside joint_limits
# after the direction multiply, so nothing is clipped and the round-trip is
# testing the sign convention and not the clamp.
LOGICAL_POSE = {
    "shoulder_pan": 10.0,    # direction  1.0
    "shoulder_lift": 20.0,   # direction  1.0
    "elbow_flex": -30.0,     # direction -1.0  -> motor  30
    "wrist_flex": 15.0,      # direction -1.0  -> motor -15
    "wrist_yaw": -5.0,       # direction -1.0  -> motor   5
    "wrist_roll": 40.0,      # direction  1.0
    "gripper": 40.0,         # direction  6.0  -> motor 240
}


def _round_trip(follower, logical_pose):
    """Send `logical_pose`, park the fake motors where it aimed, read back.

    send_action() returns the MOTOR-space goal it committed to (after direction
    and joint-limit clipping), which is exactly what a perfectly-tracking motor
    would report. Writing it into the fake state simulates that.
    """
    action = {f"{j}.pos": v for j, v in logical_pose.items()}
    committed = follower.send_action(action)
    for key, motor_degrees in committed.items():
        follower.motors[key.removesuffix(".pos")].set_motor_degrees(motor_degrees)
    return committed, follower.get_observation()


def test_action_observation_round_trip_is_identity(follower):
    """obs == action for every joint, including direction -1 and the gripper."""
    _, obs = _round_trip(follower, LOGICAL_POSE)
    for joint, logical in LOGICAL_POSE.items():
        assert obs[f"{joint}.pos"] == pytest.approx(logical, abs=1e-9), joint


@pytest.mark.parametrize("joint", ["elbow_flex", "wrist_flex", "wrist_yaw", "gripper"])
def test_unfixed_observation_would_have_been_wrong(follower, joint):
    """Guards the fix itself: the base class's observation really is mirrored.

    If this ever stops failing-to-match, upstream has changed its convention
    and the override in follower.py has become a double correction.
    """
    _round_trip(follower, LOGICAL_POSE)
    # Unbound call: the upstream implementation, on the same fake motor state.
    raw = SeeedB601RSFollower.get_observation(follower)
    direction = follower.config.joint_directions[joint]
    assert direction != 1.0
    assert raw[f"{joint}.pos"] == pytest.approx(LOGICAL_POSE[joint] * direction, abs=1e-9)
    assert raw[f"{joint}.pos"] != pytest.approx(LOGICAL_POSE[joint], abs=1e-6)


def test_elbow_flex_commands_the_motor_the_right_way(follower):
    """A negative logical elbow_flex must reach the motor as positive degrees."""
    committed, _ = _round_trip(follower, LOGICAL_POSE)
    assert committed["elbow_flex.pos"] == pytest.approx(30.0)
    kind, pos_rad, *_ = follower.motors["elbow_flex"].sent[-1]
    assert kind == "mit"
    assert pos_rad == pytest.approx(math.radians(30.0))


def test_gripper_scale_of_six_round_trips(follower):
    """The gripper's direction is a 6x gear scale, not a sign - both ways."""
    committed, obs = _round_trip(follower, {**LOGICAL_POSE, "gripper": 40.0})
    assert committed["gripper.pos"] == pytest.approx(240.0)
    assert obs["gripper.pos"] == pytest.approx(40.0)


def test_velocity_is_converted_but_torque_is_not(follower):
    """pos and vel carry the sign convention; torque is left in motor space."""
    motor = follower.motors["elbow_flex"]
    motor.state.vel = math.radians(12.0)
    motor.state.torq = 3.3
    obs = follower.get_observation()
    # direction -1.0: velocity flips, torque does not (see module docstring).
    assert obs["elbow_flex.vel"] == pytest.approx(-12.0)
    assert obs["elbow_flex.torque"] == pytest.approx(3.3)


def test_hold_action_returns_only_positions(follower):
    follower.motors["wrist_roll"].set_motor_degrees(40.0)
    held = follower.hold_action()
    assert set(held) == {f"{j}.pos" for j in follower.motor_names}
    assert held["wrist_roll.pos"] == pytest.approx(40.0)


def test_zero_direction_raises_instead_of_commanding_zero(follower):
    """The base class's joint_directions.get(name, 0.0) default is a trap."""
    follower.config.joint_directions["wrist_roll"] = 0.0
    with pytest.raises(ValueError, match="command the joint to zero"):
        follower.get_observation()


def test_name_is_not_overridden():
    """Overriding `name` moves calibration_fpath and makes connect() re-zero.

    Robot.__init__ derives calibration_dir from `name`; a different name points
    at a directory with no calibration file, is_calibrated returns False, and
    connect() calls calibrate() -> set_zero_position() on all seven motors.
    """
    assert "name" not in ReBotFollower.__dict__
    assert ReBotFollower.name == SeeedB601RSFollower.name
