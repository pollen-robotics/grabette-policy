"""Reset safety on the real-arm servicer (fake arm, no hardware)."""

import numpy as np
import pytest
from openarm_gripette_simu.proto import arm_pb2


# RA-1: a NaN/inf joint target must be refused before any motor write. The
# driver's clip `max(lo, min(hi, nan))` returns `hi`, so NaN used to ramp every
# joint to its UPPER limit while Reset reported success.
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_reset_rejects_non_finite_joints(make_servicer, fake_arm, fast_reset, bad):
    s = make_servicer(fake_arm)
    q = [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0]
    q[1] = bad
    resp = s.Reset(arm_pb2.ResetRequest(joint_positions=q), None)
    assert not resp.success
    assert fake_arm.sent == []
