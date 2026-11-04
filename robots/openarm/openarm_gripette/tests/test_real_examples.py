"""Hardware-free tests for the real-arm example scripts (fake CAN / fake motors)."""

import importlib.util
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import can
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCanBus:
    """python-can Bus stand-in answering Damiao REFRESH queries.

    status: {motor_id: status nibble} put in the high 4 bits of reply byte 0
    (0x0 disabled, 0x1 enabled, >= 0x8 fault). Motors absent from it are silent.
    """

    def __init__(self, status):
        self.status = status
        self.sent = []
        self._rx = deque()

    def send(self, msg):
        self.sent.append(msg)
        if msg.arbitration_id == 0x7FF and msg.data[2] == 0xCC:
            motor_id = msg.data[0]
            if motor_id in self.status:
                data = bytes([(self.status[motor_id] << 4) | motor_id, 0x80, 0, 0x80, 0, 0, 30, 30])
                self._rx.append(can.Message(arbitration_id=0x10 + motor_id, data=data, is_extended_id=False))

    def recv(self, timeout=None):
        return self._rx.popleft() if self._rx else None

    def shutdown(self):
        pass


# RA-9: the e-stop script counted ANY refresh reply as "disabled", so a motor
# that missed every disable frame but still answers REFRESH (status 0x1 =
# enabled) was reported as torque-free.
def test_disable_script_reports_still_enabled_motor(monkeypatch, capsys):
    mod = _load_example("disable_all_motors_can")
    status = {i: 0x0 for i in range(1, 8)}
    status[4] = 0x1  # joint_4 missed the disable and is still holding torque
    fake = FakeCanBus(status)
    monkeypatch.setattr(mod, "can", SimpleNamespace(interface=SimpleNamespace(Bus=lambda **kw: fake),
                                                    Message=can.Message))
    monkeypatch.setattr(mod, "time", SimpleNamespace(sleep=lambda s: None, monotonic=time.monotonic))
    monkeypatch.setattr(sys, "argv", ["disable_all_motors_can", "--can_port", "fake"])
    mod.main()
    out = capsys.readouterr().out
    assert "torque-free" not in out
    assert "joint_4" in out and "ENABLED" in out


def test_disable_script_all_disabled_is_reported_ok(monkeypatch, capsys):
    mod = _load_example("disable_all_motors_can")
    fake = FakeCanBus({i: 0x0 for i in range(1, 8)})
    monkeypatch.setattr(mod, "can", SimpleNamespace(interface=SimpleNamespace(Bus=lambda **kw: fake),
                                                    Message=can.Message))
    monkeypatch.setattr(mod, "time", SimpleNamespace(sleep=lambda s: None, monotonic=time.monotonic))
    monkeypatch.setattr(sys, "argv", ["disable_all_motors_can", "--can_port", "fake"])
    mod.main()
    out = capsys.readouterr().out
    assert "torque-free" in out
    assert "WARNING" not in out


# RA-15: the per-step cap was only checked at the NOMINAL dt, but each delta is
# target(t) - prev, so after an RPC stall one delta covered the whole stall.
def test_sinusoid_step_clamped_after_stall(monkeypatch):
    import numpy as np

    monkeypatch.syspath_prepend(str(EXAMPLES))  # for `from _torque_guard import ...`
    mod = _load_example("cartesian_sinusoid")
    cap_m = mod.MAX_PER_STEP_MM / 1000.0
    amp, omega, dt = 0.02, 2 * np.pi * 0.25, 0.1  # script defaults: 3.1 mm/step nominal
    times = list(np.arange(0.0, 1.0, dt)) + list(np.arange(2.0, 4.0, dt))  # 1 s stall
    prev, deltas = 0.0, []
    for t in times:
        d, prev = mod.clamp_step(amp * np.sin(omega * t), prev, cap_m)
        deltas.append(d)
    assert max(abs(d) for d in deltas) <= cap_m + 1e-12
    # and the commanded offset catches back up with the reference
    assert abs(prev - amp * np.sin(omega * times[-1])) < 1e-9


class _StuckMotor:
    """openarm_can motor stand-in that never reports a mechanical stop
    (stale comms / dead feedback: frozen position, zero torque)."""

    def get_position(self):
        return 0.0

    def get_velocity(self):
        return 0.0

    def get_torque(self):
        return 0.0


class _FakeComp:
    def __init__(self):
        self.targets = []
        self._motors = [_StuckMotor() for _ in range(7)]

    def get_motors(self):
        return self._motors

    def mit_control_one(self, idx, param):
        self.targets.append(param[2])  # commanded q (rad)
        # Test-side backstop so an uncapped loop FAILS instead of hanging.
        assert len(self.targets) < 100_000, "bump_to_limit wound up without a travel cap"


# RA-11 (travel cap only): bump_to_limit had no travel/timeout cap — with stale
# feedback the stop condition never fires and q_target winds up forever, so
# when comms return the motor sees a target far past its mechanical stop.
@pytest.mark.parametrize("joint", ["J1", "J4", "J7"])
def test_bump_to_limit_aborts_after_max_travel(monkeypatch, joint):
    import numpy as np

    mod = _load_example("calibrate_arm_no_gripper")
    monkeypatch.setattr(mod, "oa", SimpleNamespace(MITParam=lambda *p: p))
    monkeypatch.setattr(mod, "time", SimpleNamespace(sleep=lambda s: None))
    comp = _FakeComp()
    jid = mod.JointID[joint]
    with pytest.raises(RuntimeError):
        mod.bump_to_limit(SimpleNamespace(recv_all=lambda: None), comp, jid, step_deg=-0.2)
    lo, hi = mod.mech_lim[jid]
    # Never commanded further than the joint's full mechanical range (+ margin).
    assert max(abs(q) for q in comp.targets) <= (hi - lo) + np.deg2rad(10.0) + 1e-9
