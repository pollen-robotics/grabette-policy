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


# --------------------------------------------------------------------------- #
# on_sample: the hook examples/measure_release.py records the descent through
# --------------------------------------------------------------------------- #

def test_on_sample_sees_every_arm_joint_and_no_gripper(follower):
    """The recorder measures the shipped loop, so it must be handed the same
    joints that loop drives - all six, and motor 7 never."""
    seen = []
    follower.damped_release(max_s=0.5, on_sample=lambda t, s: seen.append((t, s)))
    assert seen, "on_sample was never called"
    for _, states in seen:
        assert set(states) == set(follower.arm_motor_names())
        assert "gripper" not in states


def test_on_sample_timestamps_advance(follower):
    """t is elapsed seconds from the start of the release; a recorder that got
    a constant would produce a flat descent curve and a wrong kd."""
    seen = []
    follower.motors["elbow_flex"].state.vel = math.radians(30.0)
    follower.damped_release(max_s=0.6, on_sample=lambda t, s: seen.append(t))
    assert len(seen) > 1
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert seen[-1] > seen[0]


def test_on_sample_is_told_about_a_silent_motor(follower):
    """None has to reach the recorder rather than being filtered out, or a
    joint that went quiet mid-descent silently leaves the record."""
    seen = []
    follower.motors["wrist_roll"].set_silent()
    follower.damped_release(max_s=0.5, on_sample=lambda t, s: seen.append(s))
    assert seen
    assert all(s["wrist_roll"] is None for s in seen)


def test_a_broken_recorder_does_not_pass_as_a_release(follower):
    """Swallowing the exception would leave the caller believing the arm was
    damped while the loop had stopped commanding damping."""
    def boom(t, states):
        raise RuntimeError("recorder blew up")

    with pytest.raises(RuntimeError):
        follower.damped_release(max_s=0.5, on_sample=boom)


def test_release_without_a_recorder_is_unchanged(follower):
    """on_sample defaults off: the server and the teardown call this without
    one and must keep behaving exactly as before."""
    assert follower.damped_release(max_s=5.0) is True
