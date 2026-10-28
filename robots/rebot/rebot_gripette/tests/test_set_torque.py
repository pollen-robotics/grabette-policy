"""Enabling and disabling torque on the real arm.

Both directions were broken and neither test nor simulator noticed: the code
called `bus.enable_torque()` / `bus.disable_torque()`, which do not exist on a
motorbridge Controller. The first use raised AttributeError - on the shutdown
path, which meant the server failed to disable torque at all, and SetTorque
would have failed the same way. Found by starting the server against the arm.
"""
import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from rebot_gripette.arm_interface import ReBotArmInterface  # noqa: E402


def _torque_calls(motor):
    return [s[0] for s in motor.sent if s[0] in ("enable", "disable")]


def test_disabling_torque_reaches_every_arm_motor(follower):
    iface = ReBotArmInterface(follower)
    iface.set_torque(False)
    for name in follower.arm_motor_names():
        assert _torque_calls(follower.motors[name]) == ["disable"], name


def test_enabling_torque_reaches_every_arm_motor(follower):
    iface = ReBotArmInterface(follower)
    iface.set_torque(True)
    for name in follower.arm_motor_names():
        assert "enable" in _torque_calls(follower.motors[name]), name


def test_torque_never_touches_the_gripper(follower):
    """Motor 7 is the reBot rack gripper: read-only here. `bus.enable_all()`
    would have energized it, which is why this goes per-motor over the six."""
    iface = ReBotArmInterface(follower)
    iface.set_torque(True)
    iface.set_torque(False)
    assert _torque_calls(follower.motors["gripper"]) == []


def test_enabling_reapplies_the_torque_limits(follower):
    """The RobStride 0x7000 range is runtime-only and does not survive a
    disable/enable cycle. An arm enabled without this runs at the firmware
    default ceiling."""
    iface = ReBotArmInterface(follower)
    iface.set_torque(True)
    # limits.apply_limits writes the clamp registers on every arm motor.
    for name in follower.arm_motor_names():
        assert follower.motors[name].params, f"{name} got no limit writes"


def test_the_methods_called_exist_on_the_real_driver():
    """The bug this file exists for: the names were plausible and absent. Pin
    them against the actual motorbridge Motor, not against a fake that would
    happily accept anything."""
    motorbridge = pytest.importorskip("motorbridge")

    assert hasattr(motorbridge.Motor, "enable")
    assert hasattr(motorbridge.Motor, "disable")
    assert not hasattr(motorbridge.Controller, "enable_torque")
    assert not hasattr(motorbridge.Controller, "disable_torque")


def test_the_handles_are_looked_up_where_they_live():
    """The third bug of the same shape in one file: a plausible attribute that
    does not exist. `bus.motors` was always None, so the limits were never
    re-applied and the warning said so into a log nobody was reading."""
    import inspect

    from rebot_gripette import arm_interface

    src = inspect.getsource(arm_interface.ReBotArmInterface._reapply_limits)
    # Code only: the comment above it names the wrong attribute deliberately,
    # to say what the bug was.
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert 'getattr(self._robot, "motors"' in code
    assert "self._robot.bus" not in code
