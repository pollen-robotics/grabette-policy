"""Shared machinery for the staged bring-up examples. Opens nothing on import.

Every constant here was established on the real arm in the rebot-lerobot
experiments (scripts 11-14), not chosen for this file. Provenance is noted per
value, because a number whose origin is forgotten is a number someone will
"tidy up" later.

Imports are deliberately function-local: `read_torque_limits.py` wants the
motor table without dragging in lerobot, torch and the plugin.

THE DISCIPLINE, and why each part of it exists:

  * Preflight reads position from DISABLED motors, via parameter reads. A
    RobStride emits no feedback frames until enabled, so get_state() returns
    None beforehand; a script that reads the start pose with get_state() gets
    nothing and, worse, a caller that treats None as 0.0 commands the arm to
    zero. Refuse to energize an arm we cannot fully read.

  * The first commanded pose is the MEASURED pose. Scripts 11/12 ramped kp
    from 0 to target over ENGAGE_S so the arm was never handed a step change.
    Through the LeRobot plugin the gains are not ours to ramp - the equivalent
    is to start the trajectory exactly where the arm already is, so the first
    action commands no motion at all.

  * All six joints are held, not just the one being moved. With only one joint
    powered the unpowered links swing off their stops in ways the geometry
    does not let us predict (11's safety note).

  * Every leg is checked against an abort envelope: position deviation,
    torque against the per-joint clamp, and MOSFET temperature.
"""
import math
import time

# (joint, CAN id, RobStride model). Joints 1-3 are RS-06, 4-7 RS-00.
# Motor 7 is the stock rack gripper that the Gripette replaces: it is on the
# bus and gets clamped, but nothing here ever commands it.
MOTORS = [("shoulder_pan", 1, "rs-06"), ("shoulder_lift", 2, "rs-06"),
          ("elbow_flex", 3, "rs-06"), ("wrist_flex", 4, "rs-00"),
          ("wrist_yaw", 5, "rs-00"), ("wrist_roll", 6, "rs-00"),
          ("gripper", 7, "rs-00")]

ARM_MOTORS = [m for m in MOTORS if m[0] != "gripper"]

# RobStride runtime registers.
MECH_POS = 0x7019     # mechanical angle, readable while DISABLED
VBUS = 0x701C         # bus voltage, a sag here means the supply is the problem

# Control rate used throughout the bring-up experiments (11, 12, 13).
RATE_HZ = 50.0

# Abort envelope.
#   ABORT_DEG: worst per-joint tracking deviation tolerated. 11 used 5.0 for a
#     5 deg shoulder move, 13 used 8.0 for a coordinated two-joint leg. 5.0 is
#     the tight end; raise it for a deliberately large move, do not raise it to
#     make a failing move pass.
#   TORQUE_ABORT_FRAC / TEMP_ABORT_C: from 14's compliant runs.
ABORT_DEG = 5.0
TORQUE_ABORT_FRAC = 0.80
TEMP_ABORT_C = 60.0

# Steady-state tracking droop under gravity is already 1-2 deg, so the plugin's
# rate limiter has to sit above that or it clamps continuously during normal
# motion. 13 settled on 5.0: still bounds how far a frozen control loop can
# leave the arm from where it was.
MAX_RELATIVE_TARGET = 5.0

# Trajectory speed. 11 moved 5 deg in 4 s, 12 moved 12 deg in 6 s, 13 moved 12
# deg in 5 s - i.e. 1.2 to 2.4 deg/s. Durations are derived from this and the
# largest joint delta, so a bigger move takes longer rather than faster.
DEG_PER_S = 2.0
MIN_LEG_S = 2.0



def urdf_limits():
    """{joint: (lower_deg, upper_deg)} from the exported model's URDF.

    Re-exported from rebot_gripette.spec so the examples and the package read
    the SAME limits. They were separate copies until a third copy - a hardcoded
    table in calibration.py - was found to be wrong on five of six joints.
    """
    from rebot_gripette.spec import urdf_limits as _limits
    return _limits()


def leg_duration(max_delta_deg, deg_per_s=DEG_PER_S):
    """Seconds for a leg whose largest joint delta is `max_delta_deg`."""
    return max(MIN_LEG_S, abs(max_delta_deg) / deg_per_s)


def ease(f):
    """Cosine ease, zero velocity at both ends (13's `profile`)."""
    return 0.5 * (1.0 - math.cos(math.pi * min(max(f, 0.0), 1.0)))


# --------------------------------------------------------------------------- #
# Preflight, on disabled motors
# --------------------------------------------------------------------------- #

def cold_read_positions(handles, motors=ARM_MOTORS):
    """Motor-space degrees per joint, read while the motors are DISABLED.

    Parameter reads (type 17) are answered whether or not the motor is
    enabled; feedback frames (type 2) are not sent until it is. Everything
    done before enable() must come from here.

    Returns {joint: degrees}; joints that did not answer are absent, which the
    caller must treat as fatal rather than defaulting.
    """
    out = {}
    for name, _, _ in motors:
        try:
            out[name] = math.degrees(
                handles[name].robstride_get_param_f32(MECH_POS, 1000))
        except Exception as exc:
            print(f"  {name}: mechPos read FAILED - {type(exc).__name__}: {exc}")
    return out


def require_full_read(positions, motors=ARM_MOTORS):
    """Abort unless every joint reported. Nothing has been energized yet."""
    missing = [n for n, _, _ in motors if n not in positions]
    if missing:
        raise SystemExit(
            f"No position from {missing} - nothing was enabled. Check power "
            f"and CAN wiring before retrying.")
    return positions


# --------------------------------------------------------------------------- #
# The follower
# --------------------------------------------------------------------------- #

def connect_follower(can, device_id, gravity_compensation=False):
    """Connect, refusing to re-zero the arm, and apply the volatile clamps.

    gravity_compensation defaults OFF, unlike the plugin. The exported model's
    link inertias still put each link's COM at its motor, so the predicted
    gravity torque has the right magnitude at the calibration pose and the
    WRONG angle dependence everywhere else. That is fine for kinematics and
    position control and not fine as a feedforward term. Turn it on only once
    the COMs are real.
    """
    from rebot_gripette.follower import ReBotFollower, ReBotFollowerConfig
    from rebot_gripette.limits import apply_limits

    config = ReBotFollowerConfig(
        id=device_id,
        port=can,
        can_adapter="socketcan",
        gravity_compensation=gravity_compensation,
        max_relative_target=MAX_RELATIVE_TARGET,
    )
    robot = ReBotFollower(config)

    # connect(calibrate=True) silently runs calibrate() when is_calibrated is
    # False, and calibrate() calls set_zero_position() on every motor. Refuse
    # to connect unless the existing calibration actually loaded, so a wrong
    # path or a renamed class can never turn a bring-up run into a re-zeroing
    # of the arm.
    print(f"calibration: {robot.calibration_fpath}")
    if not robot.is_calibrated:
        raise SystemExit(
            "Calibration did not load - connect() would re-zero all seven "
            "motors. Refusing. Check that the path above exists and matches "
            "Robot.name.")
    print(f"loaded, {len(robot.calibration)} joints")

    robot.connect()

    # The 0x7000 range is volatile, so the clamps are software state applied on
    # every connect. strict=True: an arm with an unprotected joint must not run.
    applied = apply_limits(robot.motors)
    print(f"torque clamps applied to {len(applied)} joints")
    return robot


def controlled_descent(robot, **kwargs):
    """Put the arm down under position control. See ReBotFollower.

    Lives on the follower, not here: the gRPC server's signal handler needs the
    same behaviour, and examples/ is not importable from the package.
    """
    print("  [descent] lowering shoulder_lift and elbow_flex to their stops "
          "under position control; the other joints are held")
    done = robot.controlled_descent(**kwargs)
    print("  [descent] at the floor" if done else
          "  [descent] DID NOT COMPLETE - disabling anyway")
    return done


def disconnect_follower(robot, descend=True):
    """Teardown that does not move the arm, and does not drop it either.

    Two separate hazards, and both have bitten this arm:

      * disconnect() calls safe_zero(), which drives all six joints to motor
        zero - the lower hard stop for shoulder_lift and elbow_flex. Suppressed
        from ReBotFollower.__init__ now, re-asserted here.
      * disconnect() then disables every motor, because
        disable_torque_on_disconnect defaults True. With the arm extended that
        is a drop. On 2026-10-27 it dropped the arm from 77 degrees.

    `descend` lowers the arm to its stops first, so the disable happens with
    the arm already down. Pass descend=False only when it is already there.
    """
    if descend:
        controlled_descent(robot)
    robot.skip_safe_zero_on_disconnect()
    robot.disconnect()
    print("disconnected; the arm was put down, not dropped")


# --------------------------------------------------------------------------- #
# Motor-space helpers
# --------------------------------------------------------------------------- #
# Everything here is in MOTOR degrees, which is also the URDF's space and the
# space the follower reads and writes - ReBotFollowerConfig pins
# joint_directions to the identity, so the vendor's leader-arm mapping is not
# in play and nothing is converted anywhere. See follower.py.

def motor_pose(robot):
    """Current pose in MOTOR degrees, arm joints only."""
    obs = robot.get_observation()
    return {n: obs[f"{n}.pos"] for n, _, _ in ARM_MOTORS}


def send_motor_pose(robot, targets_motor):
    """Command a MOTOR-space pose. The gripper motor is never included."""
    robot.send_action({f"{n}.pos": v for n, v in targets_motor.items()})


def read_param(robot, joint, param_id):
    """One register, NaN if it did not answer. Diagnostics only."""
    try:
        return robot.motors[joint].robstride_get_param_f32(param_id, 1000)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Moving
# --------------------------------------------------------------------------- #

def _envelope_violation(robot, targets_motor, now_motor, abort_deg):
    """Return a reason string if the arm is outside the envelope, else None."""
    dev = {n: abs(now_motor[n] - targets_motor[n]) for n in targets_motor}
    worst_joint = max(dev, key=dev.get)
    if dev[worst_joint] > abort_deg:
        return (f"{worst_joint} deviated {dev[worst_joint]:.2f} deg from its "
                f"target (limit {abort_deg})")

    from rebot_gripette.limits import TORQUE_LIMITS
    obs = robot.get_observation()
    for n, _, _ in ARM_MOTORS:
        torque = obs.get(f"{n}.torque")
        if torque is not None and abs(torque) > TORQUE_ABORT_FRAC * TORQUE_LIMITS[n]:
            return (f"{n} torque {torque:.2f} N.m exceeded "
                    f"{TORQUE_ABORT_FRAC:.0%} of its {TORQUE_LIMITS[n]} N.m clamp")
        state = robot.motors[n].get_state()
        if state is not None and state.t_mos > TEMP_ABORT_C:
            return f"{n} MOSFET at {state.t_mos:.1f} C (limit {TEMP_ABORT_C})"
    return None


def run_leg(robot, start_motor, end_motor, duration_s, label,
            abort_deg=ABORT_DEG, sample_s=1.0):
    """Interpolate start -> end with a cosine ease. True if it completed.

    Returns False rather than raising on an abort, so the caller can still run
    its teardown and leave the arm held rather than dropped.
    """
    moving = [n for n in end_motor if abs(end_motor[n] - start_motor[n]) > 1e-9]
    t0, next_sample = time.time(), 0.0
    while (t := time.time() - t0) < duration_s:
        s = ease(min(t / duration_s, 1.0))
        targets = {n: start_motor[n] + (end_motor[n] - start_motor[n]) * s
                   for n in end_motor}
        send_motor_pose(robot, targets)

        if t >= next_sample:
            now = motor_pose(robot)
            reason = _envelope_violation(robot, targets, now, abort_deg)
            # Full joint names: shoulder_pan and shoulder_lift both abbreviate
            # to "shoulder", which is ambiguous exactly when someone is reading
            # this line closely.
            shown = "  ".join(f"{n}={now[n]:+7.2f}"
                              for n in (moving or list(end_motor)[:2]))
            print(f"  {label} t={t:5.2f}  {shown}")
            if reason:
                print(f"  !! ABORT: {reason}")
                return False
            next_sample += sample_s
        time.sleep(1.0 / RATE_HZ)
    return True


def path_segments(waypoints_motor):
    """[(from, to, largest joint delta)] for consecutive waypoints.

    Distance along a traced path is measured as the largest joint delta per
    segment - the same currency leg_duration prices a single leg in - so
    deg_per_s means the same thing whether a motion is one leg or forty.
    """
    return [(a, b, max(abs(b[n] - a[n]) for n in b))
            for a, b in zip(waypoints_motor, waypoints_motor[1:])]


def pose_along(segs, d, last):
    """Interpolate a segment list at distance `d` from its start.

    A zero-length segment is SKIPPED, not landed on. Returning from it would
    swallow the rest of the distance and freeze the path at that waypoint -
    and a square whose last waypoint returns to its first produces exactly
    such a segment.
    """
    for a, b, seg in segs:
        if seg <= 1e-9:
            continue
        if d <= seg:
            return {n: a[n] + (b[n] - a[n]) * (d / seg) for n in b}
        d -= seg
    return dict(last)


def run_path(robot, waypoints_motor, deg_per_s=DEG_PER_S, label="path",
             abort_deg=ABORT_DEG, sample_s=1.0, ease_s=1.0):
    """Walk a whole waypoint list as ONE continuous motion. True if completed.

    run_leg per waypoint is the wrong tool for a traced path. Its cosine ease
    goes to zero velocity at each end, and MIN_LEG_S floors every leg at 2 s
    however small it is - so a 41-waypoint square whose segments are 0.5-1.3
    deg took 82 s and stopped 41 times. That is not a slow trace, it is a
    stutter, and it is the one thing that makes a square hard to judge by eye.

    Here the waypoints are strung onto a single timeline. Distance along it is
    measured in the same currency leg_duration uses - the largest joint delta
    of each segment - so `deg_per_s` keeps meaning what it means everywhere
    else. The ease is applied ONCE at each end of the whole path, over `ease_s`,
    so the arm accelerates in, traces at a constant rate, and settles out.

    The abort envelope is unchanged and checked against the interpolated
    target, exactly as run_leg does.
    """
    segs = path_segments(waypoints_motor)
    total = sum(d for _, _, d in segs)
    if total <= 1e-9:
        return True

    # Ease in and out over a fixed TIME at each end, with a constant-rate
    # middle: a cosine over the whole path would make the middle faster than
    # deg_per_s to compensate for the slow ends.
    cruise_s = total / deg_per_s
    duration = cruise_s + ease_s          # the two half-eases cost ease_s
    print(f"  {label}: {len(waypoints_motor)} waypoints, {total:.1f} deg of "
          f"joint travel at {deg_per_s:g} deg/s -> {duration:.1f} s")

    def distance_at(t):
        """Distance along the path at time t, with eased ends."""
        if t <= ease_s:
            return deg_per_s * ease_s * ease(t / ease_s) / 2.0
        if t >= duration - ease_s:
            r = (duration - t) / ease_s
            return total - deg_per_s * ease_s * ease(r) / 2.0
        return deg_per_s * (t - ease_s / 2.0)

    t0, next_sample = time.time(), 0.0
    while (t := time.time() - t0) < duration:
        targets = pose_along(segs, min(distance_at(t), total),
                             waypoints_motor[-1])
        send_motor_pose(robot, targets)

        if t >= next_sample:
            now = motor_pose(robot)
            reason = _envelope_violation(robot, targets, now, abort_deg)
            shown = "  ".join(f"{n}={now[n]:+7.2f}" for n, _, _ in ARM_MOTORS)
            print(f"  {label} t={t:6.2f}  {shown}")
            if reason:
                print(f"  !! ABORT: {reason}")
                return False
            next_sample += sample_s
        time.sleep(1.0 / RATE_HZ)

    send_motor_pose(robot, dict(waypoints_motor[-1]))
    return True


def hold(robot, targets_motor, duration_s, label="hold", abort_deg=ABORT_DEG):
    """Hold one fixed target. The target is NOT re-read from the arm.

    Re-reading it every cycle is a free-running integrator, not a hold: drift
    moves the joint, the target follows, and nothing opposes it (~1.5 deg/s
    measured). See ReBotFollower.hold_action.
    """
    return run_leg(robot, targets_motor, targets_motor, duration_s, label,
                   abort_deg=abort_deg)


def print_pose(title, pose_motor, limits=None):
    """One row per joint, with URDF limits alongside when supplied."""
    print(f"\n{title}")
    for name, _, _ in ARM_MOTORS:
        row = f"  {name:<15}{pose_motor[name]:+9.3f} deg"
        if limits and name in limits:
            lo, hi = limits[name]
            margin = min(pose_motor[name] - lo, hi - pose_motor[name])
            flag = "  <-- ON A STOP" if margin < 1.0 else ""
            row += f"   limits [{lo:7.1f},{hi:7.1f}]  margin {margin:6.1f}{flag}"
        print(row)
