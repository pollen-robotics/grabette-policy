#!/usr/bin/env python3
"""Rung 3. One absolute joint pose, reached and held. Then optionally back.

    uv run python examples/set_arm_pose.py --can can0 --joints 10 65 55 0 0 0

Rung 2 proved the arm can be ramped to one specific pose from wherever it
starts. This rung is the general case: any pose, from any pose, which is what
every rung above it is built out of. Run it only after rung 2 was clean.

Targets are given in MOTOR space, which is also the model's space if the URDF
convention question in read_arm_state.py resolved that way. Positive
shoulder_lift and elbow_flex move AWAY from their stops at zero.

Safety is the same as rung 2 and the reasoning is there: validated before the
bus is opened, first command is the measured pose, cosine ease at ~2 deg/s, and
an abort envelope on deviation, torque and temperature. --return_home brings
the arm back to where it started, which is the honest way to end a test: a
rung that leaves the arm somewhere new changes the start pose of the next one.

The pose is checked with the same `check_home_pose` guard the server uses - it
is a joint-limit clearance check, not a statement that the pose is sensible.
It knows nothing about your table, the pedestal, the Gripette's envelope or
self-collision.
"""
import argparse
import sys

import numpy as np

from _bringup import (ARM_MOTORS, MOTORS, cold_read_positions, connect_follower,
                      disconnect_follower, hold, leg_duration, motor_pose,
                      print_pose, require_full_read, run_leg, urdf_limits)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--id", default="rebot", help="LeRobot device id (picks the calibration)")
    ap.add_argument("--joints", type=float, nargs=6, required=True, metavar="DEG",
                    help="target pose, six angles in degrees, ArmSpec order: "
                         "shoulder_pan shoulder_lift elbow_flex wrist_flex "
                         "wrist_yaw wrist_roll")
    ap.add_argument("--return_home", action="store_true",
                    help="go back to the measured start pose before finishing")
    ap.add_argument("--deg_per_s", type=float, default=2.0)
    ap.add_argument("--dwell_s", type=float, default=3.0)
    args = ap.parse_args()

    from rebot_gripette.spec import ARM_JOINT_NAMES, check_home_pose

    try:
        check_home_pose(np.deg2rad(args.joints))
    except ValueError as e:
        print(f"--joints rejected: {e}")
        sys.exit(2)

    target = dict(zip(ARM_JOINT_NAMES, args.joints))
    print(f"Target pose: {[round(d, 1) for d in args.joints]} deg")
    print("This script WILL move the arm. Hand on the power.\n")

    from motorbridge import Controller

    with Controller(args.can) as ctrl:
        handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in MOTORS}
        start = require_full_read(cold_read_positions(handles))

    limits = urdf_limits()
    print_pose("measured start pose (motor deg)", start, limits)

    robot = connect_follower(args.can, args.id)
    ok = False
    try:
        held = motor_pose(robot)
        print("\n[engage] holding the measured pose - the arm must NOT move")
        if not hold(robot, held, 1.5, label="engage"):
            return

        out_s = leg_duration(max(abs(target[n] - held[n]) for n, _, _ in ARM_MOTORS),
                             args.deg_per_s)
        print(f"\n[out] ramping to the target over {out_s:.1f} s")
        ok = run_leg(robot, held, target, out_s, "out ")
        if ok:
            ok = hold(robot, target, args.dwell_s, label="dwell")

        if ok:
            reached = motor_pose(robot)
            print_pose("reached (motor deg)", reached, limits)
            worst = max(abs(reached[n] - target[n]) for n, _, _ in ARM_MOTORS)
            print(f"\nworst residual vs commanded: {worst:.2f} deg")

        if ok and args.return_home:
            back_s = leg_duration(
                max(abs(held[n] - target[n]) for n, _, _ in ARM_MOTORS),
                args.deg_per_s)
            print(f"\n[back] returning to the start pose over {back_s:.1f} s")
            ok = run_leg(robot, target, held, back_s, "back")
            if ok:
                end = motor_pose(robot)
                print("\nreturn accuracy vs start (motor deg):")
                for name, _, _ in ARM_MOTORS:
                    print(f"  {name:<15}{end[name] - held[name]:+8.3f}")
    finally:
        disconnect_follower(robot)
        print("ABORTED - the arm stopped where it was." if not ok else "Done.")


if __name__ == "__main__":
    main()
