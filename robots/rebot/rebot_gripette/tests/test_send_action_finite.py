"""ReBotFollower.send_action refuses non-finite goals (RA-1, driver side).

The upstream joint-limit clip is `max(lo, min(hi, position))`, and
`min(hi, nan)` returns `hi`: a NaN goal is turned into the joint's UPPER limit
and commanded. Nothing non-finite may reach the bus.
"""
import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from rebot_gripette.spec import ARM_JOINT_NAMES  # noqa: E402


def _commands(follower):
    return {n: [s for s in m.sent if s[0] in ("mit", "pos_vel", "force_pos")]
            for n, m in follower.motors.items()}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_send_action_rejects_non_finite(follower, bad):
    action = {f"{n}.pos": 30.0 for n in ARM_JOINT_NAMES}
    action["elbow_flex.pos"] = bad
    with pytest.raises(ValueError, match="non-finite"):
        follower.send_action(action)
    assert all(not c for c in _commands(follower).values()), _commands(follower)


def test_send_action_still_sends_finite_goals(follower):
    """The guard must not block a normal command."""
    follower.send_action({f"{n}.pos": 30.0 for n in ARM_JOINT_NAMES})
    assert any(_commands(follower).values())
