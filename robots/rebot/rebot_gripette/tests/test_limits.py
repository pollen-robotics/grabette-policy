"""The RobStride safety clamps: the table itself, and how it is applied.

These values were sized from a measured torque envelope (see limits.py) and are
hardware-validated. The table test is a tripwire: changing a number here means
changing the arm's protection, so it has to be a deliberate edit in two places.
"""
import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from rebot_gripette import TORQUE_LIMITS, VEL_MAX_LIMITS, apply_limits
from rebot_gripette.limits import LIMIT_TORQUE, VEL_MAX

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]

EXPECTED_TORQUE = {
    "shoulder_pan": 12.0,
    "shoulder_lift": 24.0,
    "elbow_flex": 16.0,
    "wrist_flex": 6.0,
    "wrist_yaw": 6.0,
    "wrist_roll": 6.0,
    "gripper": 5.0,
}

EXPECTED_VEL_MAX = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 0.4,
    "elbow_flex": 0.4,
    "wrist_flex": 1.0,
    "wrist_yaw": 1.0,
    "wrist_roll": 1.0,
    "gripper": 1.0,
}


def test_every_motor_has_a_torque_clamp():
    assert sorted(TORQUE_LIMITS) == sorted(JOINTS)
    assert TORQUE_LIMITS == EXPECTED_TORQUE


def test_every_motor_has_a_vel_max():
    assert sorted(VEL_MAX_LIMITS) == sorted(JOINTS)
    assert VEL_MAX_LIMITS == EXPECTED_VEL_MAX


def test_apply_limits_writes_and_verifies_both_registers(follower):
    applied = apply_limits(follower.motors)
    for joint in JOINTS:
        motor = follower.motors[joint]
        assert motor.params[LIMIT_TORQUE] == TORQUE_LIMITS[joint]
        assert motor.params[VEL_MAX] == VEL_MAX_LIMITS[joint]
        assert applied[joint] == {"limit_torque": TORQUE_LIMITS[joint],
                                  "vel_max": VEL_MAX_LIMITS[joint]}


def test_apply_limits_never_persists_to_flash(follower):
    """FakeMotor.store_parameters() raises: 0x7000 is runtime-only.

    Type-22 SAVE_PARAMETERS covers only the 0x20xx range, so persisting these
    would be a false promise. They must be re-applied on EVERY connect.
    """
    apply_limits(follower.motors)   # would raise AssertionError if it persisted


def test_apply_limits_is_strict_about_a_missing_motor(follower):
    del follower.motors["gripper"]
    with pytest.raises(RuntimeError, match="gripper: no such motor"):
        apply_limits(follower.motors)


def test_apply_limits_detects_a_readback_mismatch(follower):
    """A motor that silently refuses the write must not pass as protected."""
    motor = follower.motors["shoulder_lift"]
    motor.robstride_write_param_f32 = lambda param_id, value: None
    motor.params[LIMIT_TORQUE] = 1.0
    with pytest.raises(RuntimeError, match="shoulder_lift: limit_torque readback"):
        apply_limits(follower.motors)


def test_vel_max_can_be_skipped(follower):
    applied = apply_limits(follower.motors, apply_vel_max=False)
    assert all("vel_max" not in v for v in applied.values())
    assert VEL_MAX not in follower.motors["shoulder_pan"].params
