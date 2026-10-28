#!/usr/bin/env python3
"""Rung 2.6. Measure the damping that the damped release actually needs.

    uv run python examples/measure_release.py --can can0 --id rebot1 \
        --home_joints 0 15 25 -10 0 0

RELEASE_KD shipped unmeasured: the kd halves of gains that were tuned WITH a
non-zero kp, which is a different problem. This script puts a number on them.

HOW ONE RUN IS ENOUGH
    A damped release is MIT with kp=0, so the only torque is -kd*w. The arm
    accelerates until that balances gravity and then sinks at a roughly
    constant speed, where

        tau_gravity = kd * w

    Both sides are observable: MotorState carries vel and torq. So the kd that
    would produce a chosen descent speed follows from a single release at any
    kd that keeps the arm on the bus:

        kd_needed = kd_used * w_peak / w_target        (from the speed)
        kd_needed = tau_peak / w_target                (from the torque)

    They are independent routes to the same number. Printing both is the check:
    if they disagree badly, the sink never reached terminal velocity and the
    estimate is not trustworthy.

WHY max_s IS RAISED FOR THIS RUN
    In normal use a release that has not settled within RELEASE_MAX_S is
    disabled anyway - _bringup.damped_release prints "STILL MOVING - disabling
    anyway" and the teardown continues. That makes TOO MUCH damping unsafe too:
    the arm is cut loose part-way down instead of at the bottom. Measuring with
    the production timeout would therefore truncate the descent curve. --max_s
    defaults to 30 s here so the whole sink is recorded; the value RELEASE_MAX_S
    should take is an OUTPUT of this run, not an input.

WHERE TO RUN IT
    From the home pose, 15 and 25 deg off the stops. Both failure modes are
    small there: too little damping arrives at a stop with little energy, too
    much is disabled barely above it. Do not raise the pose to get a longer
    descent - a longer descent is exactly the thing being avoided.

WHAT IT DOES NOT DO
    It does not write RELEASE_KD. It prints what the measurement implies and
    leaves the edit to a human, because the number it prints is a safety
    constant and one run on one pose is thin evidence for it.
"""
import argparse
import math
import sys

import numpy as np

from _bringup import (ARM_MOTORS, MOTORS, cold_read_positions, connect_follower,
                      disconnect_follower, hold, leg_duration, motor_pose,
                      print_pose, require_full_read, run_leg, urdf_limits)

# A joint that barely moves during the sink is not loaded by gravity at this
# pose, so its terminal velocity says nothing about its damping. Below this
# much travel the estimate is suppressed rather than reported as a small kd.
MIN_TRAVEL_DEG = 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--id", default="rebot1", help="LeRobot device id (picks the calibration)")
    ap.add_argument("--home_joints", type=float, nargs=6, required=True, metavar="DEG",
                    help="REQUIRED. The pose to release FROM, in ArmSpec order.")
    ap.add_argument("--target_deg_s", type=float, default=10.0,
                    help="descent speed the recommended kd is computed for")
    ap.add_argument("--kd_scale", type=float, default=1.0,
                    help="multiplies RELEASE_KD for this run. 1.0 measures the "
                         "shipped table, which is a result either way.")
    ap.add_argument("--max_s", type=float, default=30.0,
                    help="release timeout for the MEASUREMENT - long on purpose, "
                         "so the descent is not truncated. See the docstring.")
    ap.add_argument("--deg_per_s", type=float, default=2.0,
                    help="speed of the ramp UP to the release pose")
    ap.add_argument("--settle_s", type=float, default=2.5)
    args = ap.parse_args()

    from rebot_gripette.follower import RELEASE_KD
    from rebot_gripette.spec import ARM_JOINT_NAMES, check_home_pose

    try:
        check_home_pose(np.deg2rad(args.home_joints))
    except ValueError as e:
        print(f"--home_joints rejected: {e}")
        sys.exit(2)

    home = dict(zip(ARM_JOINT_NAMES, args.home_joints))
    kd = {n: v * args.kd_scale for n, v in RELEASE_KD.items()}

    print(f"Release pose: {[round(d, 1) for d in args.home_joints]} deg")
    print(f"kd for this run (RELEASE_KD x {args.kd_scale}): "
          f"{ {n: round(v, 3) for n, v in kd.items()} }")
    print("This script WILL move the arm there and then let it sink. "
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
    samples = []
    try:
        held = motor_pose(robot)
        print("\n[engage] holding the measured pose - the arm must NOT move")
        if not hold(robot, held, 1.5, label="engage"):
            return

        print(f"\n[home] ramping to the release pose over {duration:.1f} s")
        if not run_leg(robot, held, home, duration, "home"):
            print("the ramp aborted - releasing without measuring")
            robot.damped_release(kd=kd, max_s=args.max_s)
            return
        if not hold(robot, home, args.settle_s, label="settle"):
            robot.damped_release(kd=kd, max_s=args.max_s)
            return

        print(f"\n[release] kp=0, kd as above, up to {args.max_s:.0f} s. "
              f"The arm should sink to its stops.")
        settled = robot.damped_release(
            kd=kd, max_s=args.max_s,
            on_sample=lambda t, states: samples.append((t, states)))
        print("  [release] at rest" if settled else
              f"  [release] STILL MOVING after {args.max_s:.0f} s")
    finally:
        # The arm has just been released, so it is at or near its stops; a
        # second release would only re-energise it to do nothing.
        disconnect_follower(robot, release=False)

    _report(samples, kd, args.target_deg_s)


def descent_summary(series, kd_used, target_deg_s):
    """What one joint's sink did, and the kd that would make it target speed.

    `series` is [(t, MotorState)] for that joint, already stripped of the ticks
    where it returned nothing. Pure arithmetic, kept out of the printing so the
    numbers can be tested - they are the basis for a safety constant.

    Returns a dict. kd_from_speed / kd_from_torque are None when the joint
    barely moved: gravity does not load every joint at every pose, and a joint
    that did not sink has no terminal velocity to extrapolate from. Reporting a
    near-zero kd there would be the dangerous kind of wrong.
    """
    w_target = math.radians(target_deg_s)
    travel = abs(math.degrees(series[-1][1].pos - series[0][1].pos))
    peak = max((s for _, s in series), key=lambda s: abs(s.vel))
    w_peak, tau_peak = abs(peak.vel), abs(peak.torq)

    loaded = travel >= MIN_TRAVEL_DEG
    return {
        "travel_deg": travel,
        "peak_deg_s": math.degrees(w_peak),
        "torque_nm": tau_peak,
        "kd_used": kd_used,
        # tau = kd*w at terminal velocity, so kd scales inversely with speed.
        "kd_from_speed": kd_used * w_peak / w_target if loaded else None,
        "kd_from_torque": tau_peak / w_target if loaded else None,
    }


def _report(samples, kd_used, target_deg_s):
    """Per joint: what the sink did, and the kd that would make it target speed."""
    if not samples:
        print("\nno samples recorded - nothing to report")
        return

    names = [n for n, _, _ in ARM_MOTORS]

    print(f"\n{len(samples)} samples over {samples[-1][0]:.1f} s\n")
    print(f"DESCENT, and the kd for {target_deg_s:.0f} deg/s")
    print(f"  {'joint':<15}{'travel':>8}{'peak':>9}{'torque':>9}"
          f"{'kd used':>9}{'kd (speed)':>12}{'kd (torque)':>13}")

    for n in names:
        series = [(t, s[n]) for t, s in samples if s.get(n) is not None]
        if not series:
            print(f"  {n:<15}{'no state returned during the release':>50}")
            continue

        d = descent_summary(series, kd_used[n], target_deg_s)
        line = (f"  {n:<15}{d['travel_deg']:>7.1f}d{d['peak_deg_s']:>8.1f}"
                f"{d['torque_nm']:>8.2f}N{d['kd_used']:>9.2f}")
        if d["kd_from_speed"] is None:
            print(f"{line}{'  not loaded at this pose':>25}")
            continue
        print(f"{line}{d['kd_from_speed']:>12.2f}{d['kd_from_torque']:>13.2f}")

    print("\n  travel in deg, peak speed in deg/s, torque in N.m at the peak "
          "sample, kd in N.m/(rad/s).")
    print("  The two kd columns are independent estimates of the same number "
          "(speed: kd_used*w/w_target, torque: tau/w_target).")
    print("  They agree when the sink reached terminal velocity. A large "
          "disagreement means it did not, and neither number is usable.")

    # The timeout RELEASE_MAX_S should take, as an output of this run.
    t_last_moving = [t for t, s in samples
                     if any(abs(math.degrees(v.vel)) > 2.0
                            for v in s.values() if v is not None)]
    if t_last_moving:
        print(f"\n  the arm was still moving at t={t_last_moving[-1]:.1f} s; "
              f"RELEASE_MAX_S needs to exceed that with margin "
              f"(currently 8.0)")


if __name__ == "__main__":
    main()
