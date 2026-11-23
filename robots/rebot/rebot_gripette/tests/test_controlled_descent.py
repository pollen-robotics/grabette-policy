"""Stopping driving the arm must not mean dropping it.

disconnect() disables every motor (disable_torque_on_disconnect defaults
True). With the arm extended that is a free fall - it dropped the arm from 77
degrees on 2026-10-27. controlled_descent() is the answer for the cases where
the planned ramp is not available: lower the two gravity-loaded joints to
their stops under POSITION control, holding every other joint where it is, and
disable at the bottom.

This replaced damped_release(), which tried to do the same job with MIT kp=0
and a per-joint kd. Measured on the arm on 2026-10-28 from the 15/25 home, it
was indistinguishable from cutting torque. The arithmetic agrees: a damped
joint sinks at the speed where kd*w balances gravity, so 10 deg/s against the
~7 N.m this arm carries at the shoulder needs kd ~= 40, four times the largest
value the vendor plugin ever sends. Damping cannot hold this arm up; stiffness
can. No test here should pass with kp=0.

The fakes do not simulate motion - get_state() replays whatever the test set.
So everything below asserts on what was COMMANDED, which is the part this
method is responsible for anyway.
"""
import math

import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from rebot_gripette.follower import DESCENT_FLOOR  # noqa: E402


def _commanded_deg(motor):
    """Every position this motor was commanded to, in degrees, in order."""
    return [math.degrees(s[1]) for s in motor.sent if s[0] == "mit"]


def _raise(follower, **joints_deg):
    """Place joints off their stops, as if the arm were up."""
    for name, deg in joints_deg.items():
        follower.motors[name].set_motor_degrees(deg)


def test_the_gravity_loaded_joints_are_commanded_to_their_stops(follower):
    """Zero is the lower hard stop for both, and where the arm rests at
    power-on. A disable there costs nothing."""
    _raise(follower, shoulder_lift=20.0, elbow_flex=30.0)
    follower.controlled_descent(deg_per_s=400.0)
    for name, floor in DESCENT_FLOOR.items():
        cmds = _commanded_deg(follower.motors[name])
        assert cmds, f"{name} was never commanded"
        assert cmds[-1] == pytest.approx(floor, abs=0.5), (
            f"{name} ended commanded to {cmds[-1]:.2f}, not its floor {floor}")


def test_every_other_joint_is_held_not_abandoned(follower):
    """The joints gravity does not load are commanded to stay put. Leaving
    them uncommanded is not the same thing - they would be dragged by the two
    that are moving, and the gripper would sweep instead of descending."""
    _raise(follower, shoulder_lift=20.0, elbow_flex=30.0,
           shoulder_pan=-40.0, wrist_roll=-70.0)
    follower.controlled_descent(deg_per_s=400.0)
    for name in follower.arm_motor_names():
        if name in DESCENT_FLOOR:
            continue
        cmds = _commanded_deg(follower.motors[name])
        assert cmds, f"{name} was never commanded"
        assert max(cmds) - min(cmds) < 0.1, (
            f"{name} should have been held, but its target moved "
            f"{max(cmds) - min(cmds):.2f} deg")


def test_the_descent_keeps_stiffness(follower):
    """The whole reason this replaced the damped release. A descent sent with
    kp=0 is a drop, however nicely it is interpolated."""
    _raise(follower, shoulder_lift=20.0, elbow_flex=30.0)
    follower.controlled_descent(deg_per_s=400.0)
    for name in follower.arm_motor_names():
        sends = [s for s in follower.motors[name].sent if s[0] == "mit"]
        assert sends, f"{name} was never commanded"
        assert 0.0 not in {s[3] for s in sends}, \
            f"{name} was commanded with kp=0 during a descent"


def test_the_descent_never_touches_the_gripper(follower):
    """Motor 7 is read-only here too."""
    _raise(follower, shoulder_lift=20.0, elbow_flex=30.0)
    follower.controlled_descent(deg_per_s=400.0)
    assert follower.motors["gripper"].sent == []


def test_the_target_only_moves_down(follower):
    """A joint being lowered must be lowered - a target that wandered back up
    would be fighting gravity on the way to the floor."""
    _raise(follower, elbow_flex=40.0)
    follower.controlled_descent(deg_per_s=200.0)
    cmds = _commanded_deg(follower.motors["elbow_flex"])
    assert len(cmds) > 2
    assert all(b <= a + 1e-6 for a, b in zip(cmds, cmds[1:])), \
        "the elbow target went back up during the descent"


def test_the_descent_starts_where_the_arm_is(follower):
    """Stepping the target would jolt the arm at the one moment it is least
    able to take it. The first command must be the measured pose."""
    _raise(follower, elbow_flex=40.0)
    follower.controlled_descent(deg_per_s=200.0)
    assert _commanded_deg(follower.motors["elbow_flex"])[0] == pytest.approx(40.0, abs=0.5)


def test_the_descent_is_rate_bounded_not_time_bounded(follower):
    """Starting higher means a longer descent, not a faster one. A fixed
    duration would make the speed depend on how far the arm happened to be
    from the floor."""
    import time

    _raise(follower, shoulder_lift=5.0)
    t0 = time.time()
    follower.controlled_descent(deg_per_s=50.0)
    short = time.time() - t0

    _raise(follower, shoulder_lift=60.0)
    t0 = time.time()
    follower.controlled_descent(deg_per_s=50.0)
    far = time.time() - t0

    assert far > short


def test_a_descent_from_the_floor_still_terminates(follower):
    """The arm is often already down - the teardown calls this unconditionally
    on the abort path. A zero-length descent must not divide by zero or spin."""
    for name in DESCENT_FLOOR:
        follower.motors[name].set_motor_degrees(0.0)
    assert follower.controlled_descent(deg_per_s=10.0) is True


# --------------------------------------------------------------------------- #
# on_sample: the hook examples/verify_descent.py records the descent through
# --------------------------------------------------------------------------- #

def test_on_sample_sees_every_arm_joint_and_no_gripper(follower):
    """The recorder scores the shipped loop, so it must be handed the same
    joints that loop drives - all six, and motor 7 never."""
    seen = []
    _raise(follower, shoulder_lift=20.0)
    follower.controlled_descent(deg_per_s=200.0,
                                on_sample=lambda t, s: seen.append((t, s)))
    assert seen, "on_sample was never called"
    for _, states in seen:
        assert set(states) == set(follower.arm_motor_names())
        assert "gripper" not in states


def test_on_sample_timestamps_advance(follower):
    """t is elapsed seconds from the start of the descent; a recorder handed a
    constant would report a flat ramp and hide a runaway."""
    seen = []
    _raise(follower, shoulder_lift=40.0)
    follower.controlled_descent(deg_per_s=200.0,
                                on_sample=lambda t, s: seen.append(t))
    assert len(seen) > 1
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert seen[-1] > seen[0]


def test_a_broken_recorder_does_not_pass_as_a_descent(follower):
    """Swallowing the exception would leave the caller believing the arm was
    put down while the loop had in fact stopped part-way."""
    def boom(t, states):
        raise RuntimeError("recorder blew up")

    _raise(follower, shoulder_lift=20.0)
    with pytest.raises(RuntimeError):
        follower.controlled_descent(deg_per_s=200.0, on_sample=boom)


def test_a_descent_without_a_recorder_is_unchanged(follower):
    """on_sample defaults off: the server and the teardown call this without
    one."""
    _raise(follower, shoulder_lift=20.0)
    assert follower.controlled_descent(deg_per_s=400.0) is True
