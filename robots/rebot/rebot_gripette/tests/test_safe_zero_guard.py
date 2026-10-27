"""Shutting the server down must not move the arm.

The upstream plugin's `disconnect()` calls `safe_zero()`, which interpolates
all six arm joints to MOTOR ZERO. Stage 2 of that sweep is CAN ids 2 and 3 —
shoulder_lift and elbow_flex — and on this arm zero is the LOWER HARD STOP for
exactly those two (test_spec_and_home.py pins that fact down). So a plain
disconnect lowers the arm onto its stops.

Parking the arm is a reasonable thing to ask for. Doing it because someone hit
Ctrl-C is not: an interrupt is the one moment a motion must never start. The
server therefore calls `skip_safe_zero_on_disconnect()` first.

That behaviour used to happen anyway, by accident — torque was disabled first,
disabled RobStride motors emit no feedback, safe_zero's read raised, and
disconnect() swallowed the exception. These tests exist so the accident becomes
a decision, and so a plugin update that changes any part of the chain fails
here instead of silently moving the arm.
"""
import inspect
import pathlib

import pytest

import rebot_gripette


# --------------------------------------------------------------------------- #
# The premise: what upstream actually does
# --------------------------------------------------------------------------- #

def test_disconnect_still_consults_the_skip_flag():
    """If upstream stops honouring it, our suppression silently stops working."""
    pytest.importorskip("lerobot_robot_seeed_b601")
    from lerobot_robot_seeed_b601.seeed_b601_follower import SeeedB601FollowerBase

    src = inspect.getsource(SeeedB601FollowerBase.disconnect)
    assert "safe_zero" in src, (
        "disconnect() no longer calls safe_zero — the hazard may be gone, but "
        "re-read it before deleting the suppression")
    assert "_skip_safe_zero_on_disconnect" in src, (
        "disconnect() calls safe_zero but no longer honours the skip flag. "
        "Ctrl-C would drive the arm to zero, i.e. onto two hard stops.")


def test_safe_zero_sweeps_the_two_hard_stop_joints_last():
    """Documents WHICH joints the sweep moves, since that is the whole hazard.

    Stage 2 is CAN ids 2 and 3 = shoulder_lift and elbow_flex, the two joints
    whose zero IS their lower limit.
    """
    pytest.importorskip("lerobot_robot_seeed_b601")
    from lerobot_robot_seeed_b601.seeed_b601_follower import SeeedB601FollowerBase

    src = inspect.getsource(SeeedB601FollowerBase.safe_zero)
    assert "(2, 3)" in src, (
        "safe_zero's staging changed; re-check which joints it drives to zero "
        "and whether zero is still a hard stop for them")


# --------------------------------------------------------------------------- #
# Our suppression
# --------------------------------------------------------------------------- #

def test_the_flag_is_already_set_at_construction(follower):
    """It used to start False and be set on the shutdown path. That left the
    worst case uncovered: when connect() raises, Robot.__del__ performs the
    disconnect and no call site is reached. Suppression now predates any
    possible failure."""
    assert follower._skip_safe_zero_on_disconnect is True


def test_skip_still_sets_the_flag_if_something_cleared_it(follower):
    """Upstream's connect() clears the flag after its internal calibrate()
    call. ReBotFollower.connect refuses that path outright - it raises rather
    than let the base class re-zero the motors - so the clearing is currently
    unreachable, but the setter must keep working rather than be decoration."""
    follower._skip_safe_zero_on_disconnect = False
    follower.skip_safe_zero_on_disconnect()
    assert follower._skip_safe_zero_on_disconnect is True


def test_skip_commands_nothing(follower):
    """Suppressing a motion must not itself be a motion."""
    follower.skip_safe_zero_on_disconnect()
    for name, motor in follower.motors.items():
        assert motor.sent == [], f"{name} was commanded while suppressing safe_zero"


def test_a_renamed_upstream_attribute_fails_loudly(follower):
    """Silence here would mean the arm moves on Ctrl-C with nothing to show."""
    del follower._skip_safe_zero_on_disconnect
    with pytest.raises(AttributeError, match="two hard stops"):
        follower.skip_safe_zero_on_disconnect()


# --------------------------------------------------------------------------- #
# The server wires it up
# --------------------------------------------------------------------------- #

def _server_source():
    """Read as text: importing the server pulls in grpc, placo and the model.

    Neither a signal handler nor a failed port bind can be exercised without
    hardware, so the ordering is asserted on the source. Crude, but the
    alternative is no test at all on the two paths where getting it wrong moves
    the arm.
    """
    return pathlib.Path(rebot_gripette.__file__).with_name(
        "grpc_server_real.py").read_text()


def test_every_disconnect_is_preceded_by_a_suppression():
    """EVERY path, not just Ctrl-C.

    The second one is the single-instance guard: by the time it runs, connect()
    has already happened on a bus another server is driving with torque live —
    so safe_zero would really move the arm, rather than failing on an
    unreadable position the way it does when the motors are disabled.
    """
    src = _server_source()
    disconnects = [i for i in range(len(src))
                   if src.startswith("robot.disconnect()", i)]
    skips = [i for i in range(len(src))
             if src.startswith("robot.skip_safe_zero_on_disconnect()", i)]
    assert disconnects, "the server no longer disconnects"
    assert len(skips) == len(disconnects), (
        f"{len(disconnects)} disconnect call(s) but {len(skips)} "
        f"suppression(s) — one of them drives the arm to zero")
    for d in disconnects:
        assert any(s < d for s in skips), (
            "a disconnect() has no suppression before it; safe_zero would "
            "already have run and the arm would already have moved")
