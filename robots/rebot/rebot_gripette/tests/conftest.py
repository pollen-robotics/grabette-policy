"""Hardware-free doubles for the reBot follower's CAN layer.

Nothing in this directory opens a CAN bus, constructs a motorbridge Controller
or energizes a motor. `FakeMotor` records what the follower *would* have put on
the wire and replays a state the test chose.
"""
import math

import pytest

# NOTE: deliberately NO module-level importorskip here. A skip in conftest.py
# skips the ENTIRE directory, which would silently disable test_spec_and_home.py
# — tests that need only placo and the model, and are the ones worth running in
# a CI lane that installs no lerobot. Modules needing the plugin skip themselves.

def _follower_classes():
    """Import the follower lazily, so importing this conftest cannot fail."""
    pytest.importorskip(
        "lerobot_robot_seeed_b601",
        reason="lerobot-robot-seeed-b601 (upstream plugin + motorbridge) is not installed",
    )
    from rebot_gripette import ReBotFollower, ReBotFollowerConfig
    return ReBotFollower, ReBotFollowerConfig



class FakeState:
    """What motorbridge's `motor.get_state()` returns. Angles in RADIANS."""

    def __init__(self, pos=0.0, vel=0.0, torq=0.0, t_mos=30.0):
        self.pos = pos
        self.vel = vel
        self.torq = torq
        self.t_mos = t_mos


class FakeMotor:
    """A motorbridge RobStride handle that writes to memory instead of CAN."""

    def __init__(self):
        self.state = FakeState()
        self.sent = []          # (kind, *args) of every command "sent"
        self.params = {}        # robstride_write_param_f32 register writes
        self._silenced = None   # state stashed while silent, so it can return

    # --- feedback path -------------------------------------------------
    def request_feedback(self):
        pass

    def get_state(self):
        return self.state

    def set_motor_degrees(self, degrees):
        """Place the motor at `degrees` of MOTOR-space travel."""
        self.state.pos = math.radians(degrees)

    def set_silent(self, silent=True):
        """Simulate a dropped feedback frame: get_state() returns None.

        Worth a real affordance rather than assigning `state = None` in a
        test, because it is reversible: the plugin substitutes 0.0 deg for a
        silent motor, and a test that cannot un-silence one cannot check that
        the arm recovers afterwards.
        """
        if silent:
            if self.state is not None:
                self._silenced = self.state
            self.state = None
        else:
            self.state = self._silenced or FakeState()

    # --- command path --------------------------------------------------
    def send_mit(self, pos, vel, kp, kd, tau):
        self.sent.append(("mit", pos, vel, kp, kd, tau))

    def send_pos_vel(self, pos, vel):
        self.sent.append(("pos_vel", pos, vel))

    def send_force_pos(self, pos, vel, ratio):
        self.sent.append(("force_pos", pos, vel, ratio))

    # --- parameter path (rebot_gripette.limits) ------------------------
    def robstride_write_param_f32(self, param_id, value):
        self.params[param_id] = value

    def robstride_get_param_f32(self, param_id, timeout_ms):
        return self.params[param_id]

    def store_parameters(self):
        # See limits.py: type-22 SAVE_PARAMETERS does not cover the 0x7000
        # range, so calling it would imply a persistence we do not get.
        raise AssertionError("store_parameters() must never be called on 0x7000 registers")


class FakeBus:
    """Stands in for MotorBridgeController. Owns no socket."""

    def poll_feedback_once(self):
        pass


@pytest.fixture
def follower(tmp_path):
    """A connected-looking ReBotFollower whose bus and motors are fakes.

    connect() is deliberately NOT called: it would open a CAN socket and, with
    an empty calibration dir, walk into calibrate() and re-zero the motors.
    Instead the two attributes connect() would have set are injected directly.
    `calibration_dir=tmp_path` keeps the real calibration file untouched.
    """
    ReBotFollower, ReBotFollowerConfig = _follower_classes()
    config = ReBotFollowerConfig(id="pytest", port="can0", calibration_dir=tmp_path)
    robot = ReBotFollower(config)
    robot.bus = FakeBus()
    robot.motors = {name: FakeMotor() for name in robot.motor_names}
    yield robot
    # is_connected is derived from `bus`; clearing it stops Robot.__del__ from
    # calling disconnect() -> safe_zero() during garbage collection.
    robot.bus = None
