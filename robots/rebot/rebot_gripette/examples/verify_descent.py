#!/usr/bin/env python3
"""Rung 2.6. Check that the controlled descent actually puts the arm down.

    uv run python examples/verify_descent.py --can can0 --id rebot1 \
        --home_joints 0 15 25 -10 0 0

WHAT THIS REPLACED
    This script started life as measure_release.py: it measured the kd that
    damped_release() would need, by letting the arm sink with kp=0 and solving
    tau = kd*w at terminal velocity. Run on 2026-10-28 from the 15/25 home with
    the shipped RELEASE_KD, the sink was indistinguishable from cutting torque,
    which matches the arithmetic - sinking at 10 deg/s against the ~7 N.m this
    arm carries at the shoulder needs kd ~= 40, four times the largest value
    the vendor plugin ever sends. Damping cannot hold this arm. The release was
    replaced by ReBotFollower.controlled_descent, which uses position control,
    and this script now checks that instead.

WHAT IT CHECKS
    The descent is a position ramp, so unlike the sink it has a commanded path
    to be compared against. Per joint:

      * the joints that were supposed to move reached the floor,
      * the joints that were supposed to be HELD stayed put - this is the half
        damping could never do, and the half that keeps the gripper from
        sweeping the workspace on the way down,
      * peak speed stayed near the commanded rate rather than running away,
      * torque stayed inside the clamps.

    None of that needs a model, a tape measure or a judgement call: the
    commanded trajectory is known, so every number here is an error against it.

WHERE TO RUN IT
    From the home pose, 15 and 25 deg off the stops. Short, and the descent is
    the same motion the teardown would make from there anyway.
"""
import argparse
import math
import sys

import numpy as np

from _bringup import (ARM_MOTORS, MOTORS, cold_read_positions, connect_follower,
                      disconnect_follower, hold, leg_duration, motor_pose,
                      print_pose, require_full_read, run_leg, urdf_limits)

# A held joint that drifts more than this during the descent is not being held
# well enough to call it held - the point of holding them is that the gripper
# goes straight down rather than sweeping.
HOLD_TOLERANCE_DEG = 3.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--id", default="rebot1", help="LeRobot device id (picks the calibration)")
    ap.add_argument("--home_joints", type=float, nargs=6, required=True, metavar="DEG",
                    help="REQUIRED. The pose to descend FROM, in ArmSpec order.")
    ap.add_argument("--descent_deg_per_s", type=float, default=None,
                    help="descent rate; the follower's default if unset")
    ap.add_argument("--deg_per_s", type=float, default=2.0,
                    help="speed of the ramp UP to the descent pose")
    ap.add_argument("--settle_s", type=float, default=2.5)
    args = ap.parse_args()

    from rebot_gripette.follower import DESCENT_DEG_PER_S, DESCENT_FLOOR
    from rebot_gripette.spec import ARM_JOINT_NAMES, check_home_pose

    try:
        check_home_pose(np.deg2rad(args.home_joints))
    except ValueError as e:
        print(f"--home_joints rejected: {e}")
        sys.exit(2)

    home = dict(zip(ARM_JOINT_NAMES, args.home_joints))
    rate = args.descent_deg_per_s or DESCENT_DEG_PER_S

    print(f"Descent pose: {[round(d, 1) for d in args.home_joints]} deg")
    print(f"Descending {sorted(DESCENT_FLOOR)} to {DESCENT_FLOOR} at "
          f"{rate} deg/s; every other joint is HELD where it is.")
    print("This script WILL move the arm there and then put it down. "
          "Hand on the power.\n")

    from motorbridge import Controller

    with Controller(args.can) as ctrl:
        handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in MOTORS}
        start = require_full_read(cold_read_positions(handles))

    limits = urdf_limits()
    print_pose("measured start pose (motor deg)", start, limits)

    max_delta = max(abs(home[n] - start[n]) for n, _, _ in ARM_MOTORS)
    duration = leg_duration(max_delta, args.deg_per_s)
    print(f"\nlargest joint move {max_delta:.1f} deg -> {duration:.1f} s leg")

    robot = connect_follower(args.can, args.id)
    samples, before, after = [], None, None
    try:
        held = motor_pose(robot)
        print("\n[engage] holding the measured pose - the arm must NOT move")
        if not hold(robot, held, 1.5, label="engage"):
            return

        print(f"\n[home] ramping to the descent pose over {duration:.1f} s")
        if not run_leg(robot, held, home, duration, "home"):
            print("the ramp aborted - putting the arm down without measuring")
            robot.controlled_descent()
            return
        if not hold(robot, home, args.settle_s, label="settle"):
            robot.controlled_descent()
            return

        before = motor_pose(robot)
        kwargs = {"deg_per_s": rate} if args.descent_deg_per_s else {}
        print(f"\n[descent] lowering at {rate} deg/s")
        robot.controlled_descent(
            on_sample=lambda t, states: samples.append((t, states)), **kwargs)
        after = motor_pose(robot)
    finally:
        # The descent has just run, so the arm is at the floor.
        disconnect_follower(robot, descend=False)

    if before is not None and after is not None:
        _report(samples, before, after, rate)


def descent_errors(series, started_at, floor, rate_deg_s):
    """What one joint did, against what it was told to do.

    `series` is [(t, MotorState)] for that joint, `started_at` its position in
    degrees when the descent began, and `floor` its commanded end position -
    which for a HELD joint is where it started, so the same arithmetic scores
    both roles. Pure, so the numbers can be tested without an arm.
    """
    ended_at = math.degrees(series[-1][1].pos)
    peak = max((s for _, s in series), key=lambda s: abs(s.vel))
    return {
        "travel_deg": abs(ended_at - started_at),
        "peak_deg_s": abs(math.degrees(peak.vel)),
        "torque_nm": abs(peak.torq),
        # How far past the commanded rate the joint ran. Position control
        # should track the ramp, so well over 1.0 means the joint is falling
        # faster than it is being lowered.
        "overspeed": abs(math.degrees(peak.vel)) / rate_deg_s,
        "residual_deg": abs(ended_at - floor),
    }


def _report(samples, before, after, rate_deg_s):
    """Did the moving joints arrive, and did the held joints hold?"""
    from rebot_gripette.follower import DESCENT_FLOOR

    if not samples:
        print("\nno samples recorded - nothing to report")
        return

    print(f"\n{len(samples)} samples over {samples[-1][0]:.1f} s\n")
    print("DESCENT, against the commanded ramp")
    print(f"  {'joint':<15}{'role':>7}{'travel':>9}{'peak':>9}"
          f"{'x rate':>8}{'torque':>10}{'residual':>11}")

    worst_hold = 0.0
    for name, _, _ in ARM_MOTORS:
        series = [(t, s[name]) for t, s in samples if s.get(name) is not None]
        if not series:
            print(f"  {name:<15}{'no state returned during the descent':>50}")
            continue

        moving = name in DESCENT_FLOOR
        floor = DESCENT_FLOOR[name] if moving else before[name]
        d = descent_errors(series, before[name], floor, rate_deg_s)
        if not moving:
            worst_hold = max(worst_hold, d["travel_deg"])
        print(f"  {name:<15}{'lower' if moving else 'hold':>7}"
              f"{d['travel_deg']:>8.1f}d{d['peak_deg_s']:>8.1f}"
              f"{d['overspeed']:>8.2f}{d['torque_nm']:>9.2f}N"
              f"{d['residual_deg']:>10.2f}d")

    print("\n  travel/residual in deg, peak in deg/s, torque in N.m at the "
          "peak sample.")
    print("  'x rate' is peak speed over the commanded rate: near 1 means the "
          "joint was lowered, well above 1 means it fell.")

    if worst_hold > HOLD_TOLERANCE_DEG:
        print(f"\n  HELD JOINTS DRIFTED {worst_hold:.1f} deg - more than "
              f"{HOLD_TOLERANCE_DEG} deg. The descent is supposed to go "
              f"straight down; this one swept.")
    else:
        print(f"\n  held joints stayed within {worst_hold:.2f} deg - the arm "
              f"went down rather than across.")

    print("\nfinal pose vs where the descent started:")
    for name, _, _ in ARM_MOTORS:
        print(f"  {name:<15}{before[name]:>9.2f} -> {after[name]:>8.2f} deg")


if __name__ == "__main__":
    main()
