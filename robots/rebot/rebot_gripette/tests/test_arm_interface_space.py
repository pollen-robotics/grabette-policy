"""The ArmInterface must speak the model's space, and must not invent a pose.

TWO FAILURES, both of which hid behind passing tests.

1. SPACE. The exported URDF is MOTOR space: 22_mirror.py drove the model from
   raw mechPos reads with every sign at +1 and confirmed joint-by-joint
   agreement. The follower reports LOGICAL space (motor / joint_directions),
   which is mirrored on elbow_flex, wrist_flex and wrist_yaw. Feeding logical
   angles to the kinematics therefore computes on a mirrored pose. It hid
   because reads AND writes were both logical, so the round trip was the
   identity and holds held - only the model disagreed, and no test compared
   the model to the arm.

2. SILENCE. The plugin substitutes 0.0 deg for a motor with no feedback frame
   rather than reporting the gap (seeed_b601_follower.py). Zero is the LOWER
   HARD STOP for shoulder_lift and elbow_flex, so one dropped frame would tell
   the servicer the arm is folded onto its stops and let it interpolate away
   from there.
"""
import math

import numpy as np
import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from rebot_gripette.arm_interface import ReBotArmInterface  # noqa: E402
from rebot_gripette.spec import ARM_JOINT_NAMES  # noqa: E402

# Motor-space degrees. elbow_flex, wrist_flex and wrist_yaw have direction
# -1.0, so these are the joints where the two spaces disagree; the others are
# controls that must come back unchanged either way.
MOTOR_POSE = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 60.0,
    "elbow_flex": 60.0,
    "wrist_flex": 20.0,
    "wrist_yaw": 15.0,
    "wrist_roll": 30.0,
}
MIRRORED = ("elbow_flex", "wrist_flex", "wrist_yaw")


@pytest.fixture
def iface(follower):
    """Caching disabled: these tests change the motor state between reads."""
    for name, deg in MOTOR_POSE.items():
        follower.motors[name].set_motor_degrees(deg)
    return ReBotArmInterface(follower, state_cache_ttl_s=0.0)


# --------------------------------------------------------------------------- #
# Space
# --------------------------------------------------------------------------- #

def test_get_positions_reports_motor_space(iface):
    got = np.rad2deg(iface.get_positions())
    expected = [MOTOR_POSE[n] for n in ARM_JOINT_NAMES]
    assert got == pytest.approx(expected, abs=1e-9)


def test_the_mirrored_joints_keep_their_motor_sign(iface, follower):
    """The specific regression: direction -1.0 must not reach the model.

    A positive motor-space elbow_flex is away from its stop at zero. If the
    logical value leaked through, this would come back negative.
    """
    got = dict(zip(ARM_JOINT_NAMES, np.rad2deg(iface.get_positions())))
    for name in MIRRORED:
        assert follower.config.joint_directions[name] == -1.0
        assert got[name] > 0, f"{name} came back mirrored: {got[name]:.1f} deg"
        assert got[name] == pytest.approx(MOTOR_POSE[name])


def test_send_command_puts_motor_degrees_on_the_wire(iface, follower):
    """What reaches the motor must be the motor-space target, unmirrored."""
    target = np.deg2rad([MOTOR_POSE[n] for n in ARM_JOINT_NAMES])
    iface.send_command_rad(target)
    for name in ARM_JOINT_NAMES:
        kind, pos_rad, *_ = follower.motors[name].sent[-1]
        assert kind == "mit"
        assert math.degrees(pos_rad) == pytest.approx(MOTOR_POSE[name], abs=1e-6)


def test_read_then_send_commands_no_motion(iface, follower):
    """The interface's own round trip: what it reads, sent back, is a no-op."""
    iface.send_command_rad(iface.get_positions())
    for name in ARM_JOINT_NAMES:
        _, pos_rad, *_ = follower.motors[name].sent[-1]
        assert math.degrees(pos_rad) == pytest.approx(MOTOR_POSE[name], abs=1e-6)


def test_the_gripper_is_never_commanded(iface, follower):
    """Motor 7 is the stock rack gripper the Gripette replaces."""
    iface.send_command_rad(iface.get_positions())
    assert follower.motors["gripper"].sent == []


def test_fk_matches_the_model_fed_raw_motor_angles(iface):
    """The check that was missing: compare the interface against the MODEL.

    Every test that existed compared the interface against itself, which is
    exactly why a mirrored pose round-tripped cleanly.
    """
    pytest.importorskip("placo", reason="needs the [kinematics] extra")
    pytest.importorskip("rebot_gripette_model")
    from gripette_core.kinematics import CONTROL_FRAME, Kinematics
    from rebot_gripette.spec import REBOT_SPEC

    kin = Kinematics(REBOT_SPEC)
    from_iface = kin.forward(iface.get_positions(), frame=CONTROL_FRAME)
    from_motor = kin.forward(
        np.deg2rad([MOTOR_POSE[n] for n in ARM_JOINT_NAMES]), frame=CONTROL_FRAME)
    assert np.allclose(from_iface, from_motor, atol=1e-9)


def test_a_logical_pose_would_have_been_somewhere_else_entirely(iface):
    """Guards the premise: the two spaces really do differ at this pose.

    Without this, a future change making both spaces identical (every
    direction +1) would make the tests above pass vacuously.
    """
    pytest.importorskip("placo", reason="needs the [kinematics] extra")
    pytest.importorskip("rebot_gripette_model")
    from gripette_core.kinematics import CONTROL_FRAME, Kinematics
    from rebot_gripette.spec import REBOT_SPEC

    kin = Kinematics(REBOT_SPEC)
    motor = np.deg2rad([MOTOR_POSE[n] for n in ARM_JOINT_NAMES])
    logical = np.deg2rad([MOTOR_POSE[n] * (-1 if n in MIRRORED else 1)
                          for n in ARM_JOINT_NAMES])
    gap_mm = float(np.linalg.norm(
        kin.forward(motor, frame=CONTROL_FRAME)[:3, 3]
        - kin.forward(logical, frame=CONTROL_FRAME)[:3, 3]) * 1000.0)
    assert gap_mm > 100.0, (
        f"the two spaces differ by only {gap_mm:.1f} mm at this pose, so the "
        "tests above no longer discriminate - pick a pose that does")


# --------------------------------------------------------------------------- #
# Silence
# --------------------------------------------------------------------------- #

def test_a_silent_motor_raises_instead_of_reading_as_zero(iface, follower):
    follower.motors["elbow_flex"].set_silent()
    with pytest.raises(RuntimeError, match="elbow_flex"):
        iface.get_positions()


def test_the_error_says_why_zero_is_dangerous(iface, follower):
    """The message has to carry the reason, or someone 'fixes' it by defaulting."""
    follower.motors["shoulder_lift"].set_silent()
    with pytest.raises(RuntimeError, match="hard stop"):
        iface.get_positions()


def test_every_arm_joint_is_checked(iface, follower):
    for name in ARM_JOINT_NAMES:
        follower.motors[name].set_silent()
        with pytest.raises(RuntimeError, match=name):
            iface.get_positions()
        follower.motors[name].set_silent(False)
    # All six restored: the interface works again.
    assert len(iface.get_positions()) == len(ARM_JOINT_NAMES)


def test_a_silent_gripper_does_not_block_the_arm(iface, follower):
    """Motor 7 is not an arm joint and is never commanded; its silence is not
    a reason to refuse to report the arm's pose."""
    follower.motors["gripper"].set_silent()
    assert len(iface.get_positions()) == len(ARM_JOINT_NAMES)


def test_a_failed_read_is_not_cached(follower):
    """A stale cache after a failure would re-serve a pose the arm has left."""
    for name, deg in MOTOR_POSE.items():
        follower.motors[name].set_motor_degrees(deg)
    iface = ReBotArmInterface(follower, state_cache_ttl_s=60.0)

    follower.motors["wrist_roll"].set_silent()
    with pytest.raises(RuntimeError):
        iface.get_positions()

    follower.motors["wrist_roll"].set_silent(False)
    got = dict(zip(ARM_JOINT_NAMES, np.rad2deg(iface.get_positions())))
    assert got["wrist_roll"] == pytest.approx(MOTOR_POSE["wrist_roll"])
