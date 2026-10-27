"""Stopping driving the arm must not mean dropping it.

disconnect() disables every motor (disable_torque_on_disconnect defaults
True). With the arm extended that is a free fall - it dropped the arm from 77
degrees on 2026-10-27. damped_release() is the answer for the cases where a
controlled ramp is not available: zero stiffness, non-zero damping, so the arm
sinks instead of falling and the disable lands on an arm already at rest.

What damping does NOT do is hold the arm up, and no test here should imply it
does.
"""
import math

import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from rebot_gripette.follower import RELEASE_KD  # noqa: E402


def _mit_sends(motor):
    return [s for s in motor.sent if s[0] == "mit"]


def test_release_commands_zero_stiffness_and_real_damping(follower):
    """kp=0 is what makes it a release: no position target to fight."""
    follower.damped_release(max_s=0.5)
    for name in follower.arm_motor_names():
        sends = _mit_sends(follower.motors[name])
        assert sends, f"{name} was never commanded"
        _, pos, vel, kp, kd, tau = sends[-1]
        assert kp == 0.0, f"{name} still has stiffness: kp={kp}"
        assert kd == RELEASE_KD[name] > 0.0
        assert (pos, vel, tau) == (0.0, 0.0, 0.0)


def test_release_never_touches_the_gripper(follower):
    """Motor 7 is read-only here, release included."""
    follower.damped_release(max_s=0.5)
    assert follower.motors["gripper"].sent == []


def test_release_returns_once_the_arm_is_still(follower):
    """The fakes report zero velocity, so it should settle almost at once
    rather than burning the whole timeout."""
    import time
    t0 = time.time()
    assert follower.damped_release(max_s=5.0) is True
    assert time.time() - t0 < 2.0


def test_release_reports_failure_if_the_arm_keeps_moving(follower):
    """A joint still moving at the end is not a settled arm, and saying so is
    the difference between 'released' and 'about to be dropped'."""
    follower.motors["elbow_flex"].state.vel = math.radians(30.0)
    assert follower.damped_release(max_s=0.6) is False


def test_a_silent_motor_does_not_read_as_still(follower):
    """get_state() None must not be mistaken for zero velocity - that would
    report a settled arm while it is in fact falling unobserved."""
    for name in follower.arm_motor_names():
        follower.motors[name].set_silent()
    assert follower.damped_release(max_s=0.6) is False
