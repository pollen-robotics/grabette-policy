"""Hardware-free fixtures for the real-arm driver tests.

Nothing here opens CAN, serial, cameras or sockets:
  - `FakeBus` stands in for LeRobot's `DamiaoMotorsBus` underneath a real
    `OpenArm7Follower`, so the driver's own clip / safety code runs for real.

The servicer-level safety tests (Reset, SetTorque, IK-jump watchdog,
interpolator) live with the shared servicer in
`libs/gripette-core/tests/test_arm_service.py`, run against both the OpenArm
and the reBot.
"""

import pytest

from openarm_gripette import OpenArm7Follower, OpenArm7FollowerConfig
from openarm_gripette.config_openarm7_follower import RIGHT_DEFAULT_JOINTS_LIMITS


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
