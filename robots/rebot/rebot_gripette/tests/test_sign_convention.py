"""One joint space, pinned and enforced.

Upstream ships a joint_directions table that mirrors elbow_flex, wrist_flex
and wrist_yaw and scales the gripper by 6, and then applies it on one side
only: send_action multiplies by it, get_observation returns raw motor degrees.
That mapping exists to pair their follower with their leader arm. This stack
has no leader and does not use the reBot's rack gripper, so
ReBotFollowerConfig pins every entry to 1.0 and motor space is the only space.

These tests drive the real send_action()/get_observation() against fake motors
and assert nothing is transformed anywhere - and that the arm refuses to
connect if the vendor mapping ever comes back.
"""
import math

import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from lerobot_robot_seeed_b601 import (SeeedB601RSFollower,  # noqa: E402
                                      SeeedB601RSFollowerConfig)

from rebot_gripette import ReBotFollower  # noqa: E402

# A pose in MOTOR space, which is now the only space. Every joint is inside
# joint_limits so nothing is clipped and the round-trip tests the convention
# rather than the clamp.
MOTOR_POSE = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 20.0,
    "elbow_flex": 30.0,      # vendor direction -1.0: the regression joint
    "wrist_flex": -15.0,     # vendor direction -1.0
    "wrist_yaw": 5.0,        # vendor direction -1.0
    "wrist_roll": 40.0,
    "gripper": 40.0,         # vendor direction 6.0 - a gear ratio, not a sign
}
VENDOR_MAPPED = ("elbow_flex", "wrist_flex", "wrist_yaw", "gripper")
ARM_JOINTS = [j for j in MOTOR_POSE if j != "gripper"]


def _round_trip(follower, pose):
    """Send `pose`, park the fake motors where it aimed, read back.

    send_action() returns the motor-space goal it committed to, which is what a
    perfectly-tracking motor would report. Writing it into the fake state
    simulates that.
    """
    committed = follower.send_action({f"{j}.pos": v for j, v in pose.items()})
    for key, motor_degrees in committed.items():
        follower.motors[key.removesuffix(".pos")].set_motor_degrees(motor_degrees)
    return committed, follower.get_observation()


# --------------------------------------------------------------------------- #
# The config
# --------------------------------------------------------------------------- #

def test_every_motor_has_an_explicit_unit_direction(follower):
    """Unit, and present. A joint MISSING from the dict is the dangerous case:
    send_action reads it with .get(name, 0.0) and commands it to zero, which is
    a hard stop for shoulder_lift and elbow_flex."""
    directions = follower.config.joint_directions
    for name in follower.config.motor_can_ids:
        assert directions[name] == 1.0, name


def test_the_vendor_config_really_does_transform(follower):
    """Guards the premise: if upstream ever ships the identity itself, the pin
    and the guard in follower.py are dead weight and should go."""
    vendor = SeeedB601RSFollowerConfig(id="_", port="can0").joint_directions
    assert {n: vendor[n] for n in VENDOR_MAPPED} != {n: 1.0 for n in VENDOR_MAPPED}


# --------------------------------------------------------------------------- #
# Nothing is transformed
# --------------------------------------------------------------------------- #

def test_action_observation_round_trip_is_identity(follower):
    """The six arm joints only - the gripper is stripped from actions, so it
    never reaches the motor and has nothing to round-trip."""
    _, obs = _round_trip(follower, MOTOR_POSE)
    for joint in ARM_JOINTS:
        assert obs[f"{joint}.pos"] == pytest.approx(MOTOR_POSE[joint], abs=1e-9), joint


def test_the_observation_is_the_raw_motor_angle(follower):
    """Our get_observation must now agree with the upstream one exactly: the
    override that used to divide by direction is gone, so if these ever differ
    something has reintroduced a transform."""
    _round_trip(follower, MOTOR_POSE)
    raw = SeeedB601RSFollower.get_observation(follower)
    obs = follower.get_observation()
    for joint in MOTOR_POSE:
        assert obs[f"{joint}.pos"] == pytest.approx(raw[f"{joint}.pos"], abs=1e-12)


@pytest.mark.parametrize("joint", ["elbow_flex", "wrist_flex", "wrist_yaw"])
def test_the_vendor_mapped_joints_reach_the_motor_untouched(follower, joint):
    """The specific regression: +30 deg of elbow_flex must arrive as +30, not
    -30, which would be past its lower stop at zero."""
    committed, _ = _round_trip(follower, MOTOR_POSE)
    assert committed[f"{joint}.pos"] == pytest.approx(MOTOR_POSE[joint])
    kind, pos_rad, *_ = follower.motors[joint].sent[-1]
    assert kind == "mit"
    assert pos_rad == pytest.approx(math.radians(MOTOR_POSE[joint]))


def test_the_gripper_is_observed_in_the_same_space(follower):
    """Vendor direction 6.0 is a gear ratio, so 40 deg of motor travel would
    have been reported as 6.67. Motor 7 is never commanded (see below) but it
    is still on the bus and still read, and its reading must not be rescaled."""
    follower.motors["gripper"].set_motor_degrees(40.0)
    assert follower.get_observation()["gripper.pos"] == pytest.approx(40.0)


def test_velocity_and_torque_are_motor_space(follower):
    motor = follower.motors["elbow_flex"]
    motor.state.vel = math.radians(12.0)
    motor.state.torq = 3.3
    obs = follower.get_observation()
    assert obs["elbow_flex.vel"] == pytest.approx(12.0)
    assert obs["elbow_flex.torque"] == pytest.approx(3.3)


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #

def test_connect_refuses_a_restored_vendor_mapping(follower):
    """A plugin upgrade that put -1.0 back must stop the run, not mirror three
    joints quietly. The check runs before the bus is opened."""
    follower.config.joint_directions["elbow_flex"] = -1.0
    with pytest.raises(ValueError, match="identity"):
        follower.connect()


def test_connect_refuses_a_missing_direction(follower):
    follower.config.joint_directions.pop("wrist_roll")
    with pytest.raises(ValueError, match="wrist_roll"):
        follower.connect()


def test_the_missing_direction_error_says_why_zero_is_dangerous(follower):
    """Or someone 'fixes' it by giving the base class its 0.0 default back."""
    follower.config.joint_directions.pop("shoulder_lift")
    with pytest.raises(ValueError, match="hard stop"):
        follower.check_joint_directions()


# --------------------------------------------------------------------------- #
# Unchanged by this
# --------------------------------------------------------------------------- #

def test_hold_action_returns_only_arm_positions(follower):
    follower.motors["wrist_roll"].set_motor_degrees(40.0)
    held = follower.hold_action()
    assert set(held) == {f"{j}.pos" for j in ARM_JOINTS}
    assert held["wrist_roll.pos"] == pytest.approx(40.0)


def test_name_is_not_overridden():
    """Overriding `name` moves calibration_fpath and makes connect() re-zero.

    Robot.__init__ derives calibration_dir from `name`; a different name points
    at a directory with no calibration file, is_calibrated returns False, and
    connect() calls calibrate() -> set_zero_position() on all seven motors.
    """
    assert "name" not in ReBotFollower.__dict__
    assert ReBotFollower.name == SeeedB601RSFollower.name


# --------------------------------------------------------------------------- #
# Motor 7 is read-only, and safe_zero can never run
# --------------------------------------------------------------------------- #

def test_the_gripper_is_stripped_from_every_action(follower):
    """The reBot's rack gripper is real hardware on this arm and the Gripette
    replaces it. Enforced in send_action rather than at each call site: before
    this, 'we never command the gripper' held only because every caller
    happened to build an arm-only action."""
    committed = follower.send_action({f"{j}.pos": v for j, v in MOTOR_POSE.items()})
    assert "gripper.pos" not in committed
    assert follower.motors["gripper"].sent == []
    # The six arm joints still go through untouched.
    for joint in ARM_JOINTS:
        assert committed[f"{joint}.pos"] == pytest.approx(MOTOR_POSE[joint])


def test_a_gripper_only_action_commands_nothing(follower):
    follower.send_action({"gripper.pos": 120.0})
    assert follower.motors["gripper"].sent == []


def test_hold_action_excludes_the_gripper(follower):
    follower.motors["gripper"].set_motor_degrees(40.0)
    assert "gripper.pos" not in follower.hold_action()


def test_safe_zero_is_suppressed_from_construction(follower):
    """Not from the shutdown path: the dangerous case is a connect that fails,
    where Robot.__del__ performs the disconnect and no call site gets a say."""
    assert follower._skip_safe_zero_on_disconnect is True


def test_the_gripper_zero_probe_is_a_no_op(follower):
    """The upstream probe DRIVES motor 7 closed with torque, then refuses the
    whole connect unless it is near a home position that means nothing here."""
    follower.detect_gripper_zero()
    assert follower.motors["gripper"].sent == []


def test_connect_refuses_without_a_calibration(follower, tmp_path):
    """is_calibrated False plus calibrate=True makes the base class re-zero all
    seven motors. A typo in --id is enough to reach that."""
    follower.calibration.clear()
    with pytest.raises(RuntimeError, match="re-zero"):
        follower.connect()
