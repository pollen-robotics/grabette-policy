"""rebot_gripette — real-hardware driver for the reBot B601-RS follower.

Wraps LeRobot's upstream `SeeedB601RSFollower` (the `lerobot-robot-seeed-b601`
plugin) and corrects two of its defects, each documented in the module that
fixes it:

- `follower.py`    — get_observation() reported raw motor degrees while
                     send_action() applies `joint_directions`, so feeding an
                     observation back as an action mirrored every joint with
                     direction -1 and scaled the gripper by 6.
- `calibration.py` — calibrate() wrote +/-90 as the range of every joint,
                     which mis-scales normalised obs/action spaces.

`limits.py` is not a fix but a safety requirement: the RobStride 0x7000
register range is runtime-only, so the torque and vel_max clamps must be
re-applied on EVERY connect.

Imports are LAZY (PEP 562). Importing this package must not drag in the
plugin or motorbridge: `spec.py` — the ArmSpec, the joint order and the home
pose — is pure data plus a model path, and the sim, the IK feasibility filter
and the model tests all want it on machines that will never see the arm. Same
reason `gripette_core` is lazy.

Importing `rebot_gripette.follower` registers `seeed_b601_rs_follower_fixed`
as a LeRobot `--robot.type` choice (via `@RobotConfig.register_subclass`).
Two consequences, both load-bearing:

- `make_device_from_device_class` resolves `ReBotFollower` with getattr() on
  THIS module, which `__getattr__` below satisfies.
- Registration happens at import time, and laziness means merely importing
  the package no longer performs it. A CLI script must touch the follower
  (`import rebot_gripette.follower`, or any follower attribute) before
  draccus parses its arguments.

The distribution is named `rebot-gripette`, not `lerobot_robot_*`, so LeRobot
does not auto-import it either (see
`lerobot.utils.import_utils.register_third_party_plugins`).
"""

import importlib

# Public name -> submodule that defines it.
_EXPORTS = {
    "ARM_JOINT_NAMES": ".spec",
    "REBOT_SPEC": ".spec",
    "HOME_JOINTS_RAD": ".spec",
    "ReBotArmInterface": ".arm_interface",
    "ReBotFollower": ".follower",
    "ReBotFollowerConfig": ".follower",
    "APPLY_VEL_MAX": ".limits",
    "TORQUE_LIMITS": ".limits",
    "VEL_MAX_LIMITS": ".limits",
    "apply_limits": ".limits",
    "CalibratedRangesMixin": ".calibration",
    "URDF_LIMITS": ".calibration",
    "fix_calibration_file": ".calibration",
    "calibration_ranges": ".calibration",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
