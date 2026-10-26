"""Calibration ranges: logical space, and INT - never float.

MotorCalibration.range_min/range_max are typed `int` and draccus refuses to
decode a float into one. A float in the file makes it unloadable, which makes
is_calibrated False, which makes connect() re-zero the arm. That chain is the
reason these tests exist.
"""
import json

import pytest

pytest.importorskip("lerobot_robot_seeed_b601")

from lerobot.motors import MotorCalibration

from rebot_gripette import (
    CalibratedRangesMixin,
    ReBotFollowerConfig,
    fix_calibration_file,
    logical_ranges,
)

# limits / direction, widened outward. The two worked examples in the module
# docstring (elbow_flex and gripper) are included verbatim.
EXPECTED_RANGES = {
    "shoulder_pan": (-145, 145),
    "shoulder_lift": (0, 170),
    "elbow_flex": (-200, 0),     # motor 0..200,  direction -1.0
    "wrist_flex": (-90, 80),
    "wrist_yaw": (-90, 90),
    "wrist_roll": (-130, 130),
    "gripper": (0, 45),          # motor 0..270,  direction 6.0
}


@pytest.fixture
def config(tmp_path):
    return ReBotFollowerConfig(id="pytest", port="can0", calibration_dir=tmp_path)


def test_ranges_are_in_logical_space(config):
    assert logical_ranges(config) == EXPECTED_RANGES


def test_range_values_are_int_not_float(config):
    """draccus rejects a float here - see the module docstring."""
    for joint, (lo, hi) in logical_ranges(config).items():
        assert type(lo) is int, f"{joint} range_min is {type(lo).__name__}"
        assert type(hi) is int, f"{joint} range_max is {type(hi).__name__}"


def test_ranges_are_widened_never_narrowed(config):
    """floor/ceil, so a reachable position can never fall outside the range."""
    for joint, (lo, hi) in logical_ranges(config).items():
        motor_lo, motor_hi = config.joint_limits[joint]
        direction = config.joint_directions[joint]
        a, b = motor_lo / direction, motor_hi / direction
        assert lo <= min(a, b)
        assert hi >= max(a, b)


def test_zero_direction_raises(config):
    config.joint_directions["wrist_roll"] = 0.0
    with pytest.raises(ValueError, match="joint_directions"):
        logical_ranges(config)


def test_fix_calibration_file_rewrites_ranges_and_keeps_the_zero(tmp_path, config):
    """Correcting ranges must not disturb homing_offset - the zero lives on the
    motors, not in this file."""
    path = tmp_path / "rebot1.json"
    original = {
        joint: {"id": i + 1, "drive_mode": 0, "homing_offset": 0,
                "range_min": -90, "range_max": 90}
        for i, joint in enumerate(EXPECTED_RANGES)
    }
    path.write_text(json.dumps(original, indent=4))

    fix_calibration_file(path, config=config, backup=False)
    written = json.loads(path.read_text())

    for joint, (lo, hi) in EXPECTED_RANGES.items():
        assert written[joint]["range_min"] == lo
        assert written[joint]["range_max"] == hi
        assert written[joint]["homing_offset"] == 0
        assert written[joint]["drive_mode"] == 0


def test_fix_calibration_file_backs_up_once(tmp_path, config):
    path = tmp_path / "rebot1.json"
    path.write_text(json.dumps({"elbow_flex": {"range_min": -90, "range_max": 90}}))
    fix_calibration_file(path, config=config)
    backup = path.with_suffix(".json.orig")
    assert json.loads(backup.read_text())["elbow_flex"]["range_min"] == -90


def test_mixin_corrects_ranges_after_the_base_procedure(config):
    """CalibratedRangesMixin runs the base calibrate() unchanged, then fixes the
    +/-90 placeholders the base wrote, then saves."""

    class _BaseStub:
        """Stands in for the plugin's calibrate(): writes +/-90 everywhere."""

        def __init__(self, cfg):
            self.config = cfg
            self.calibration = {}
            self.saved = 0

        def calibrate(self):
            self.calibration = {
                joint: MotorCalibration(id=i + 1, drive_mode=0, homing_offset=0,
                                        range_min=-90, range_max=90)
                for i, joint in enumerate(EXPECTED_RANGES)
            }

        def _save_calibration(self):
            self.saved += 1

    class _Calibrated(CalibratedRangesMixin, _BaseStub):
        pass

    robot = _Calibrated(config)
    robot.calibrate()

    assert robot.saved == 1
    for joint, (lo, hi) in EXPECTED_RANGES.items():
        entry = robot.calibration[joint]
        assert (entry.range_min, entry.range_max) == (lo, hi)
        assert type(entry.range_min) is int
        assert entry.homing_offset == 0
