"""The ArmInterface must speak the model's space, and must not invent a pose.

SPACE. There is one joint space: motor degrees, which are the URDF's angles.
The export was validated against raw mechPos (22_mirror.py) and confirmed on
hardware (examples/mirror_arm.py), and ReBotFollowerConfig pins
joint_directions to the identity so the follower neither mirrors nor scales
anything. These tests check the interface really does hand the kinematics the
arm's own angles - by comparing against the MODEL, which is the check that was
missing when a mirrored pose still round-tripped cleanly through every test
that compared the interface only against itself.

SILENCE. The plugin substitutes 0.0 deg for a motor with no feedback frame
rather than reporting the gap (seeed_b601_follower.py). Zero is the LOWER HARD
STOP for shoulder_lift and elbow_flex, so one dropped frame would tell the
servicer the arm is folded onto its stops and let it interpolate away from
there.
"""
import math

import numpy as np
import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from rebot_gripette.arm_interface import ReBotArmInterface  # noqa: E402
from rebot_gripette.spec import ARM_JOINT_NAMES  # noqa: E402

# Motor-space degrees. elbow_flex, wrist_flex and wrist_yaw are the joints the
# vendor mapping would have mirrored, so they are the ones a regression shows
# up on first; the others are controls that come back unchanged either way.
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


def test_the_vendor_mapped_joints_keep_their_motor_sign(iface, follower):
    """The specific regression: nothing may mirror these on the way to the model.

    A positive motor-space elbow_flex is away from its stop at zero. If a
    direction of -1.0 reached here, this would come back negative.
    """
    got = dict(zip(ARM_JOINT_NAMES, np.rad2deg(iface.get_positions())))
    for name in MIRRORED:
        assert follower.config.joint_directions[name] == 1.0
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


def test_a_mirrored_pose_would_have_been_somewhere_else_entirely(iface):
    """Guards the premise: this pose can actually tell a mirror from a non-mirror.

    The test above compares the interface's FK against the model's FK. That
    only means something if a mirrored pose would land somewhere visibly
    different - at a pose near zero on all three joints it would not, and the
    comparison would pass vacuously. This asserts the pose discriminates.
    """
    pytest.importorskip("placo", reason="needs the [kinematics] extra")
    pytest.importorskip("rebot_gripette_model")
    from gripette_core.kinematics import CONTROL_FRAME, Kinematics
    from rebot_gripette.spec import REBOT_SPEC

    kin = Kinematics(REBOT_SPEC)
    motor = np.deg2rad([MOTOR_POSE[n] for n in ARM_JOINT_NAMES])
    mirrored = np.deg2rad([MOTOR_POSE[n] * (-1 if n in MIRRORED else 1)
                          for n in ARM_JOINT_NAMES])
    gap_mm = float(np.linalg.norm(
        kin.forward(motor, frame=CONTROL_FRAME)[:3, 3]
        - kin.forward(mirrored, frame=CONTROL_FRAME)[:3, 3]) * 1000.0)
    assert gap_mm > 100.0, (
        f"a mirrored pose is only {gap_mm:.1f} mm away at this pose, so the "
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
