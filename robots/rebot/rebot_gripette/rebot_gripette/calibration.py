"""Write calibration ranges that reflect the arm's real travel.

THE BUG
    SeeedB601FollowerBase.calibrate() writes, for every joint regardless of its
    actual travel:

        MotorCalibration(range_min=-90, range_max=90, homing_offset=0)

    The plugin itself never reads those values back - send_action clips against
    config.joint_limits instead - so this is inert for direct control. But in
    standard LeRobot, range_min/range_max are what normalised observation and
    action spaces are scaled against. A policy trained or run through that
    machinery would see elbow_flex mis-scaled by more than 2x.

WHICH SPACE THE RANGES BELONG IN
    The space observations and actions actually use, which for this package is
    motor space: ReBotFollowerConfig pins joint_directions to the identity, so
    config.joint_limits ARE the recorded ranges.

    The division by direction below is therefore a no-op with our config, and
    is kept only because this module also runs against the vendor config, for
    which it is not - there, elbow_flex -1.0 maps motor 0..200 to -200..0 and
    gripper 6.0 maps motor 0..270 to 0..45. Passing the wrong config writes a
    file in the wrong space, which is why the default here is ours.

WHAT IS DELIBERATELY LEFT ALONE
    homing_offset and drive_mode stay 0. This plugin zeroes via
    set_zero_position() on the motors themselves and never applies a homing
    offset, so a non-zero value here would be an unused number that later reads
    as meaningful.
"""
import json
import math
import shutil
from pathlib import Path

from lerobot_robot_seeed_b601 import SeeedB601RSFollower

from rebot_gripette.follower import ReBotFollowerConfig

# The URDF models 0..180 for elbow_flex while the plugin config permits 0..200.
# Those extra 20 degrees are outside the dynamics model and any IK built on it.
URDF_LIMITS = {"shoulder_pan": (-160, 160), "shoulder_lift": (0, 180),
               "elbow_flex": (0, 180), "wrist_flex": (-90, 90),
               "wrist_yaw": (-90, 90), "wrist_roll": (-180, 180)}


def calibration_ranges(config):
    """Per-joint (min, max) in the space observations and actions use.

    Returns INTEGERS: MotorCalibration.range_min/range_max are typed int, and
    draccus refuses to decode a float into one - writing -145.0 makes the file
    unloadable, which in turn makes is_calibrated False, which makes connect()
    re-zero the arm. Widened outward (floor/ceil) rather than rounded, so the
    recorded range can never exclude a reachable position.
    """
    out = {}
    for name, (lo, hi) in config.joint_limits.items():
        d = config.joint_directions.get(name)
        if not d:
            raise ValueError(f"joint_directions[{name!r}] is {d!r}")
        a, b = lo / d, hi / d
        out[name] = (int(math.floor(min(a, b))), int(math.ceil(max(a, b))))
    return out


def fix_calibration_file(path, config=None, backup=True):
    """Rewrite ranges in an existing calibration file. No hardware, no re-zeroing.

    The zero itself lives on the motors, not in this file, so correcting ranges
    does not disturb the calibration you already performed.
    """
    path = Path(path)
    config = config or ReBotFollowerConfig(id="_", port="can0")
    data = json.loads(path.read_text())
    ranges = calibration_ranges(config)

    if backup:
        bak = path.with_suffix(".json.orig")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"backup: {bak}")

    print(f"{'joint':<15}{'was':>16}{'now':>20}{'motor space':>16}")
    for name, entry in data.items():
        if name not in ranges:
            continue
        lo, hi = ranges[name]
        mlo, mhi = config.joint_limits[name]
        was = f"{entry.get('range_min')}..{entry.get('range_max')}"
        entry["range_min"] = lo
        entry["range_max"] = hi
        print(f"{name:<15}{was:>16}{f'{lo:g}..{hi:g}':>20}{f'{mlo:g}..{mhi:g}':>16}")

    path.write_text(json.dumps(data, indent=4) + "\n")
    print(f"\nwritten: {path}")

    for name, (lo, hi) in config.joint_limits.items():
        u = URDF_LIMITS.get(name)
        if u and (lo < u[0] or hi > u[1]):
            print(f"WARNING  {name}: config permits {lo:g}..{hi:g} but the URDF "
                  f"models only {u[0]:g}..{u[1]:g} - outside the dynamics model")
    return data


class CalibratedRangesMixin:
    """Makes calibrate() record real ranges instead of +/-90 for everything.

    Runs the base procedure unchanged - including the manual zero-pose prompt
    and set_zero_position() - then corrects the ranges before saving.

    Mix in ahead of the follower, so this calibrate() wraps the plugin's:

        class MyFollower(CalibratedRangesMixin, ReBotFollower): ...
    """

    def calibrate(self) -> None:
        super().calibrate()
        if not self.calibration:
            return
        for name, (lo, hi) in calibration_ranges(self.config).items():
            if name in self.calibration:
                self.calibration[name].range_min = lo
                self.calibration[name].range_max = hi
        self._save_calibration()
        print("Calibration ranges corrected to real joint travel.")


if __name__ == "__main__":
    cal = (Path.home() / ".cache/huggingface/lerobot/calibration/robots"
           / SeeedB601RSFollower.name / "rebot1.json")
    if not cal.exists():
        raise SystemExit(f"No calibration file at {cal}")
    fix_calibration_file(cal)
