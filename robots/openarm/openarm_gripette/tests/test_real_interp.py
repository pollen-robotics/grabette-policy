"""Setpoint-interpolator robustness on the real-arm servicer (fake arm)."""

import time


# RA-12: get_positions() and the arithmetic sat outside the try, so a single
# CAN read error killed the interpolator thread silently while
# SendCartesianDelta kept returning success (and the arm stopped following).
def test_interp_survives_read_error(make_servicer, fake_arm):
    s = make_servicer(fake_arm)
    fake_arm.read_errors = 1  # the interpolator's first read fails once
    target = fake_arm.q.copy()
    target[0] += 0.05
    s._latest_target_joints = target  # _current_cmd_joints is None -> it reads
    time.sleep(0.15)
    assert s._interp_thread.is_alive()
    assert len(fake_arm.sent) > 0  # and it recovered: still streaming commands
