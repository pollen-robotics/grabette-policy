#!/usr/bin/env python3
"""Rung 1.5. Connect, hold where the arm already is, disconnect. NO TRAJECTORY.

Exists because rung 2's first attempt never reached its ramp: connect() raised
inside the upstream gripper probe, the destructor ran disconnect(), and
safe_zero swept the rack gripper wide open. That was a failure of the CONNECT
path, and rung 2 tests it only by also moving the arm 75 degrees.

    uv run python examples/check_connect.py --can can0 --id rebot1

This is the smallest step that exercises what broke:

  1. Cold-read all seven motors while everything is disabled.
  2. Connect - which now checks joint_directions, refuses without a loaded
     calibration, and does NOT run the gripper probe.
  3. Compare the follower's view against the cold read.
  4. Command the MEASURED pose and hold it. The arm is wherever it is, so a
     correct hold commands no motion at all; the abort envelope is live.
  5. Disconnect without safe_zero.
  6. Re-read motor 7 and compare. Nothing here commands the gripper, so an
     unchanged reading is the empirical half of that guarantee - the tests
     prove our code does not send to it, this proves nothing else did either.

There is no --home_joints and no ramp. If this rung is not clean, rung 2 has
no business running.
"""
import argparse

from _bringup import (ARM_MOTORS, MECH_POS, MOTORS, cold_read_positions,
                      connect_follower, disconnect_follower, hold, motor_pose,
                      print_pose, require_full_read, urdf_limits)

# A joint on a hard stop can disagree with the follower's first feedback frame
# by a little; more than this means the two disagree about where the arm is.
AGREEMENT_DEG = 2.0

# The gripper is not commanded, but it is unpowered and mounted, so it can sag.
GRIPPER_DRIFT_DEG = 2.0


def read_gripper(handles):
    """Motor 7 in degrees, or None if it did not answer."""
    import math
    try:
        return math.degrees(
            handles["gripper"].robstride_get_param_f32(MECH_POS, 1000))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--id", default="rebot1",
                    help="LeRobot device id (picks the calibration file)")
    ap.add_argument("--hold_s", type=float, default=3.0)
    args = ap.parse_args()

    from motorbridge import Controller

    # Preflight on a bus of our own, before the follower takes it over.
    with Controller(args.can) as ctrl:
        handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in MOTORS}
        start = require_full_read(cold_read_positions(handles))
        gripper_before = read_gripper(handles)

    limits = urdf_limits()
    print_pose("measured start pose (motor deg)", start, limits)
    print(f"\ngripper (motor 7, never commanded): "
          f"{'no answer' if gripper_before is None else f'{gripper_before:+.2f} deg'}")

    print("\nConnecting. This energizes the arm; it will hold where it is.")
    robot = connect_follower(args.can, args.id)
    ok = False
    try:
        held = motor_pose(robot)
        print_pose("pose as the follower sees it (motor deg)", held)
        drift = max(abs(held[n] - start[n]) for n, _, _ in ARM_MOTORS)
        print(f"\nagreement with the cold read: {drift:.3f} deg", end="  ")
        if drift > AGREEMENT_DEG:
            print(f"LARGE (> {AGREEMENT_DEG}) - the follower and the raw bus "
                  f"disagree about\n  where the arm is. Stop; do not run rung 2.")
            return
        print("ok")

        print(f"\n[hold] commanding the measured pose for {args.hold_s:.1f} s - "
              f"the arm must NOT move")
        ok = hold(robot, held, args.hold_s, label="hold")
    finally:
        disconnect_follower(robot)

    # The gripper check has to happen on a bus of our own again, after the
    # follower has released the motors.
    with Controller(args.can) as ctrl:
        handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in MOTORS}
        gripper_after = read_gripper(handles)

    print(f"\ngripper after: "
          f"{'no answer' if gripper_after is None else f'{gripper_after:+.2f} deg'}")
    if gripper_before is not None and gripper_after is not None:
        moved = abs(gripper_after - gripper_before)
        print(f"gripper moved {moved:.2f} deg during the whole connect/hold/"
              f"disconnect cycle", end="  ")
        print("ok" if moved <= GRIPPER_DRIFT_DEG else
              f"<-- UNEXPECTED (> {GRIPPER_DRIFT_DEG}); something commanded "
              f"motor 7")

    print("\nConnect path is clean; rung 2 can run." if ok else
          "\nABORTED during the hold - the arm stopped where it was. "
          "Do not run rung 2.")


if __name__ == "__main__":
    main()
