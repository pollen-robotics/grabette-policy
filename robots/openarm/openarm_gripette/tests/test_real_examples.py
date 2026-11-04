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
