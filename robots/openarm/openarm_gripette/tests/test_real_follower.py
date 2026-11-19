"""OpenArm7Follower.send_action safety (real driver code on a fake bus)."""

import pytest


# RA-1 (driver side): the joint-limit clip maps NaN to the upper limit, so the
# driver must refuse non-finite goals outright instead of moving the arm.
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_send_action_rejects_non_finite(follower, bad):
    action = {f"joint_{i}.pos": 0.0 for i in range(1, 8)}
    action["joint_4.pos"] = 90.0
    action["joint_2.pos"] = bad
    with pytest.raises(ValueError):
        follower.send_action(action)
    assert follower.bus.mit_writes == []
