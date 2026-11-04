"""Reset safety on the real-arm servicer (fake arm, no hardware)."""

import threading
import time

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


# RA-16: a Reset with 1-6 or 8+ joints silently went HOME instead of erroring
# (a malformed client request moved the arm somewhere it didn't ask for).
@pytest.mark.parametrize("n", [1, 6, 8])
def test_reset_wrong_length_errors(make_servicer, fake_arm, fast_reset, n):
    s = make_servicer(fake_arm)
    resp = s.Reset(arm_pb2.ResetRequest(joint_positions=[0.0] * n), None)
    assert not resp.success
    assert fake_arm.sent == []


def test_reset_empty_goes_home(make_servicer, fake_arm, fast_reset):
    from openarm_gripette import grpc_server_real as srv

    fake_arm.q = fake_arm.q + 0.1
    s = make_servicer(fake_arm)
    resp = s.Reset(arm_pb2.ResetRequest(), None)
    assert resp.success
    np.testing.assert_allclose(fake_arm.sent[-1], srv.HOME_JOINTS_RAD)
    assert s._interp_enabled


# RA-3: a Reset that fails mid-ramp (CAN error) used to re-enable the
# interpolator unconditionally in its `finally:` while the targets were only
# re-synced on success — so the interpolator drove the arm BACK to the stale
# pre-Reset target, and a latched IK-jump watchdog was silently cleared.
@pytest.mark.parametrize("latched_before", [False, True])
def test_failed_reset_keeps_motion_frozen(make_servicer, fake_arm, fast_reset, latched_before):
    s = make_servicer(fake_arm)
    stale = fake_arm.q.copy()
    stale[0] = 0.3
    fake_arm.q = stale.copy()
    s._latest_target_joints = stale.copy()
    s._current_cmd_joints = stale.copy()
    if latched_before:
        s._interp_enabled = False  # IK-jump watchdog latch

    goal = stale.copy()
    goal[0] = -0.5
    fake_arm.fail_after = 3  # ramp dies after 3 writes
    resp = s.Reset(arm_pb2.ResetRequest(joint_positions=goal.tolist()), None)
    assert not resp.success
    assert not s._interp_enabled

    # The interpolator must not resume streaming toward the stale target.
    fake_arm.fail_after = None
    q_stopped = fake_arm.q.copy()
    n = len(fake_arm.sent)
    time.sleep(0.15)
    assert len(fake_arm.sent) == n
    np.testing.assert_allclose(fake_arm.q, q_stopped)


# RA-2: SetTorque(False) is the client-side e-stop (abort_torque_off, 2 s
# timeout). Reset used to hold _cmd_lock through its whole 3 s ramp, so the
# torque-off blocked until the ramp finished and the client timed out.
def test_torque_off_preempts_reset(make_servicer, fake_arm, monkeypatch):
    from openarm_gripette import grpc_server_real as srv

    monkeypatch.setattr(srv, "RESET_DURATION_S", 2.0)
    s = make_servicer(fake_arm)
    result = {}
    th = threading.Thread(
        target=lambda: result.setdefault("r", s.Reset(arm_pb2.ResetRequest(), None))
    )
    th.start()
    time.sleep(0.2)
    assert th.is_alive() and len(fake_arm.sent) > 0  # ramp in progress

    t0 = time.monotonic()
    resp = s.SetTorque(arm_pb2.SetTorqueRequest(enable=False), None)
    latency = time.monotonic() - t0
    n_at_off = len(fake_arm.sent)
    th.join(timeout=3.0)

    assert resp.success
    assert latency < 0.15, f"torque-off waited {latency:.2f}s for the Reset ramp"
    assert fake_arm.torque_calls == [False]
    assert len(fake_arm.sent) <= n_at_off + 1  # ramp stopped
    assert not result["r"].success  # the aborted Reset reports failure
    assert not s._interp_enabled  # and does not re-arm the interpolator
