"""Apply RobStride safety limits at startup, because flash cannot be trusted.

Background: the RS manual (S4.1.9) states that single-parameter writes are
"lost in power failure", and that type-22 SAVE_PARAMETERS only persists
registers whose function code starts with 0x20. Every register motorbridge
exposes is in the 0x7000 RUNTIME range, so store_parameters() is a no-op for
them. Empirically RS00 retains 0x7000 writes across a power cycle and RS06 does
not - undocumented behaviour that should not be relied on in either direction.

Therefore: treat these limits as software state, applied on every connect. That
works identically across motor families and survives motor swaps and firmware
reflashes, which flash-resident config does not.

Usage with the LeRobot plugin - AFTER connect(), BEFORE any send_action():

    from rebot_gripette import ReBotFollower, ReBotFollowerConfig, apply_limits

    robot = ReBotFollower(ReBotFollowerConfig(id="rebot1", port="can0"))
    robot.connect()
    apply_limits(robot.motors)          # raises if any joint is unprotected

Standalone (opens its own bus, energizes nothing):

    python -m rebot_gripette.limits
"""
import logging

logger = logging.getLogger(__name__)

LIMIT_TORQUE = 0x700B
VEL_MAX = 0x7024
TOLERANCE = 0.01

# Firmware torque clamp, N.m. Sized in 06_torque_envelope.py from the worst-case
# static gravity torque over the full joint ranges. Gravity-dominated joints use
# a multiple of worst case; gravity-free joints (pan/yaw/roll compute to ~0) are
# sized from motor rating instead, since friction, payload and acceleration -
# none of which that sweep models - are what actually bound them.
TORQUE_LIMITS = {
    "shoulder_pan":  12.0,   # gravity 0.00  | rating-based
    "shoulder_lift": 24.0,   # gravity 15.45 | 1.55x worst case
    "elbow_flex":    16.0,   # gravity  7.15 | 2.2x  worst case
    "wrist_flex":     6.0,   # gravity  2.01 | 3x    worst case
    "wrist_yaw":      6.0,   # gravity  0.85 | rating-based
    "wrist_roll":     6.0,   # gravity  0.00 | rating-based
    "gripper":        5.0,   # above the plugin's 3.5 software cap
}

# The Studio template's vel_max values, which REVERT to 10.0 on the three RS06
# joints at every power cycle - a 25x increase in the PP-mode speed limit on the
# two gravity-loaded joints, happening silently. Re-applied here for the same
# reason as the torque clamps. Only affects native PP/CSP modes; the plugin
# drives arm joints in MIT, which ignores it. Set APPLY_VEL_MAX = False to skip.
APPLY_VEL_MAX = True
VEL_MAX_LIMITS = {
    "shoulder_pan":  1.0,
    "shoulder_lift": 0.4,
    "elbow_flex":    0.4,
    "wrist_flex":    1.0,
    "wrist_yaw":     1.0,
    "wrist_roll":    1.0,
    "gripper":       1.0,
}


def _write_verified(motor, param_id, value):
    """Write one f32 register and read it back. Returns the readback value."""
    motor.robstride_write_param_f32(param_id, value)
    return motor.robstride_get_param_f32(param_id, 1000)
    # Deliberately no store_parameters(): per the manual, type 22 does not cover
    # the 0x7000 range, so calling it here would imply a persistence we do not
    # get. These limits live in software and are re-applied every connect.


def apply_limits(motors, strict=True, apply_vel_max=APPLY_VEL_MAX):
    """Apply torque (and optionally vel_max) limits to a dict of motor handles.

    motors: {joint_name: motorbridge Motor}, e.g. robot.motors
    strict: raise if any joint could not be verified. Default True on purpose -
            a process that cannot protect the arm should refuse to run rather
            than silently proceed with an unprotected joint.

    Returns {joint_name: {"limit_torque": v, "vel_max": v}} of verified values.
    """
    applied, failed = {}, []

    for name, target in TORQUE_LIMITS.items():
        motor = motors.get(name)
        if motor is None:
            failed.append(f"{name}: no such motor")
            continue
        try:
            got = _write_verified(motor, LIMIT_TORQUE, target)
            if abs(got - target) > TOLERANCE:
                failed.append(f"{name}: limit_torque readback {got:.3f} != {target}")
            else:
                applied.setdefault(name, {})["limit_torque"] = got
        except Exception as exc:
            failed.append(f"{name}: limit_torque {type(exc).__name__}: {exc}")

    if apply_vel_max:
        for name, target in VEL_MAX_LIMITS.items():
            motor = motors.get(name)
            if motor is None:
                continue
            try:
                got = _write_verified(motor, VEL_MAX, target)
                if abs(got - target) > TOLERANCE:
                    failed.append(f"{name}: vel_max readback {got:.3f} != {target}")
                else:
                    applied.setdefault(name, {})["vel_max"] = got
            except Exception as exc:
                failed.append(f"{name}: vel_max {type(exc).__name__}: {exc}")

    if failed:
        msg = "Safety limits NOT fully applied:\n  " + "\n  ".join(failed)
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    return applied

# NOTE: the standalone "apply and read back" diagnostic that used to live here
# as a __main__ block is now examples/read_torque_limits.py. A library module
# must not open the CAN bus just because someone ran the file.
