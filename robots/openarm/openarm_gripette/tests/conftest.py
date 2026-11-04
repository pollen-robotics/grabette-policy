"""Hardware-free fixtures for the real-arm driver and gRPC servicer tests.

Nothing here opens CAN, serial, cameras or sockets:
  - `FakeArm` stands in for `grpc_server_real.ArmInterface` (radians, sim
    joint order) with perfect tracking: measured == last command.
  - `FakeBus` stands in for LeRobot's `DamiaoMotorsBus` underneath a real
    `OpenArm7Follower`, so the driver's own clip / safety code runs for real.
  - `Kinematics` is the real Placo model (pure computation on the URDF).
"""

import time

import numpy as np
import pytest

from openarm_gripette import OpenArm7Follower, OpenArm7FollowerConfig
from openarm_gripette import grpc_server_real as srv
from openarm_gripette.config_openarm7_follower import RIGHT_DEFAULT_JOINTS_LIMITS
from openarm_gripette_simu import Kinematics


class FakeArm:
    """ArmInterface stand-in. Records every motor write.

    fail_after: number of further successful writes before send_command_rad
        raises (simulated CAN error); None = never fail.
    read_errors: number of upcoming get_positions() calls that raise.
    """

    def __init__(self, q):
        self.q = np.array(q, dtype=float)
        self.sent: list[np.ndarray] = []
        self.torque_calls: list[bool] = []
        self.fail_after: int | None = None
        self.read_errors = 0

    def get_positions(self) -> np.ndarray:
        if self.read_errors > 0:
            self.read_errors -= 1
            raise OSError("simulated CAN read failure")
        return self.q.copy()

    def send_command_rad(self, q):
        if self.fail_after is not None:
            if self.fail_after <= 0:
                raise OSError("simulated CAN write failure")
            self.fail_after -= 1
        self.sent.append(np.array(q, dtype=float))
        self.q = np.array(q, dtype=float)

    def set_torque(self, enable: bool):
        self.torque_calls.append(bool(enable))


class FakeBus:
    """DamiaoMotorsBus stand-in: positions (deg) jump to the MIT target sent."""

    is_connected = True

    def __init__(self):
        self.pos = {f"joint_{i}": 0.0 for i in range(1, 8)}
        self.pos["joint_4"] = 90.0
        self.motors = {k: None for k in self.pos}
        self.mit_writes: list[dict] = []

    def sync_read(self, data_name, motors=None):
        return dict(self.pos)

    def sync_read_all_states(self, motors=None, *, num_retry=0):
        return {k: {"position": v, "velocity": 0.0, "torque": 0.0} for k, v in self.pos.items()}

    def _mit_control_batch(self, cmds):
        self.mit_writes.append(dict(cmds))
        for k, (_kp, _kd, q, _dq, _tau) in cmds.items():
            self.pos[k] = q

    def enable_torque(self, motors=None, num_retry=0):
        pass

    def disable_torque(self, motors=None, num_retry=0):
        pass


@pytest.fixture(scope="session")
def kin():
    # Same weights as the real server's CLI defaults.
    return Kinematics(position_weight=100.0, orientation_weight=10.0)


@pytest.fixture
def fake_arm():
    return FakeArm(srv.HOME_JOINTS_RAD)


@pytest.fixture
def make_servicer(kin):
    """Factory: ArmServicer on a FakeArm. Interp threads are stopped on teardown."""
    made = []

    def _make(arm, **kw):
        s = srv.ArmServicer(arm, kin, time.monotonic(), **kw)
        made.append(s)
        return s

    yield _make
    for s in made:
        s.stop()


@pytest.fixture
def fast_reset(monkeypatch):
    """Shorten the Reset ramp so tests don't sleep 3 s (10 steps at 50 Hz)."""
    monkeypatch.setattr(srv, "RESET_DURATION_S", 0.2)


@pytest.fixture
def follower():
    """Real OpenArm7Follower wired to a FakeBus (bypasses DamiaoMotorsBus init)."""
    rob = OpenArm7Follower.__new__(OpenArm7Follower)
    cfg = OpenArm7FollowerConfig(port="fake", side="right", max_relative_target=8.0, cameras={})
    cfg.joint_limits = dict(RIGHT_DEFAULT_JOINTS_LIMITS)
    rob.config = cfg
    rob.bus = FakeBus()
    rob.cameras = {}
    rob.id = "fake"
    return rob
