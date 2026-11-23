#!/usr/bin/env python3
"""Rung 2. The ramp to the home pose - the FIRST COMMANDED MOTION of the arm.

    uv run python examples/reset_arm.py --can can0 \
        --home_joints 0 60 60 0 0 0        # <- YOUR experiment's pose

--home_joints is required and has no default, exactly as on the server. The
home pose is the startup position for an EXPERIMENT, not a property of the
robot: the OpenArm's is its simulator's START_JOINTS, the pose its dataset was
recorded from. The reBot has no such experiment yet, so any value shipped here
would be a guess that this script then drives the arm into.

The pose is checked against the model before the bus is opened - joint count,
and every joint at least 5 deg clear of a hard stop. Those are NECESSARY
conditions only. Nothing here knows about your table, the pedestal, the
Gripette's envelope or self-collision. Watch it with a hand on the power.

WHAT IT DOES, in order:
  1. Cold-read the six joints with the motors disabled. Abort if any is silent.
  2. Connect, refusing to proceed if the calibration did not load (connect()
     would otherwise re-zero every motor).
  3. Apply the volatile torque clamps.
  4. Command the MEASURED pose first, so the arm is never handed a step change
     on enable. Hold it briefly and confirm nothing moved.
  5. Interpolate to the home pose with a cosine ease at ~2 deg/s, aborting on
     deviation, torque or temperature.
  6. Ramp back DOWN to the measured start pose and only then release.

Step 6 is not optional politeness. disconnect() disables every motor, so
ending a run with the arm extended means dropping it - which is exactly what
happened on 2026-10-27, from 77 degrees. The default ending therefore returns
the arm to where it started, on its stops, where a disable costs nothing.

--stay_held skips the descent and leaves the arm up with torque on, for a
server about to take over. Anything that goes wrong instead lowers the two
gravity-loaded joints to their stops under position control, holding the rest
where they are. See ReBotFollower.controlled_descent - and in particular why
it is not the damped release this used to do.

Step 4 is the equivalent of the kp ramp used in the raw-CAN bring-up scripts:
the plugin owns the gains, so the way to avoid a jolt is to start the
trajectory exactly where the arm already is.
"""
import argparse
import sys

import numpy as np

from _bringup import (ARM_MOTORS, MOTORS, cold_read_positions, connect_follower,
                      controlled_descent, disconnect_follower, hold,
                      leg_duration, motor_pose, print_pose, require_full_read,
                      run_leg, urdf_limits)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--id", default="rebot", help="LeRobot device id (picks the calibration)")
    ap.add_argument("--home_joints", type=float, nargs=6, required=True, metavar="DEG",
                    help="REQUIRED. Six angles in degrees, in ArmSpec order: "
                         "shoulder_pan shoulder_lift elbow_flex wrist_flex "
                         "wrist_yaw wrist_roll.")
    ap.add_argument("--deg_per_s", type=float, default=2.0,
                    help="trajectory speed; the duration follows from the move size")
    ap.add_argument("--settle_s", type=float, default=2.5)
    ap.add_argument("--stay_held", action="store_true",
                    help="leave the arm AT the home pose with torque on, for a "
                         "server about to take over. Without this the arm is "
                         "ramped back down to where it started.")
    args = ap.parse_args()

    from rebot_gripette.spec import ARM_JOINT_NAMES, check_home_pose

    # Validate BEFORE the bus is opened, so a bad pose fails with the arm
    # untouched rather than half-way through a connect.
    try:
        check_home_pose(np.deg2rad(args.home_joints))
    except ValueError as e:
        print(f"--home_joints rejected: {e}")
        sys.exit(2)

    home = dict(zip(ARM_JOINT_NAMES, args.home_joints))
    print(f"Home pose (the experiment's startup position): "
          f"{[round(d, 1) for d in args.home_joints]} deg")
    print("This script WILL move the arm there. Hand on the power.\n")

    from motorbridge import Controller

    # Preflight on a bus of our own, before the follower takes it over: the
    # follower's observation path needs enabled motors, and we want the start
    # pose while everything is still dead.
    with Controller(args.can) as ctrl:
        handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in MOTORS}
        start = require_full_read(cold_read_positions(handles))

    limits = urdf_limits()
    print_pose("measured start pose (motor deg)", start, limits)

    max_delta = max(abs(home[n] - start[n]) for n, _, _ in ARM_MOTORS)
    duration = leg_duration(max_delta, args.deg_per_s)
    print(f"\nlargest joint move {max_delta:.1f} deg -> {duration:.1f} s leg")

    robot = connect_follower(args.can, args.id)
    ok = False
    try:
        # Step 4: the first thing commanded is where the arm already is.
        held = motor_pose(robot)
        print_pose("pose as the follower sees it, after enable (motor deg)", held)
        drift = max(abs(held[n] - start[n]) for n, _, _ in ARM_MOTORS)
        print(f"\nagreement with the cold read: {drift:.3f} deg "
              f"({'fine' if drift < 2.0 else 'LARGE - check calibration before continuing'})")

        print("\n[engage] holding the measured pose - the arm must NOT move")
        if not hold(robot, held, 1.5, label="engage"):
            return

        print(f"\n[home] ramping to the home pose over {duration:.1f} s")
        ok = run_leg(robot, held, home, duration, "home")
        if ok:
            ok = hold(robot, home, args.settle_s, label="settle")

        if ok:
            reached = motor_pose(robot)
            print_pose("reached (motor deg)", reached, limits)
            worst = max(abs(reached[n] - home[n]) for n, _, _ in ARM_MOTORS)
            print(f"\nworst residual vs commanded home: {worst:.2f} deg")
            print("Steady-state droop of 1-2 deg under gravity is expected "
                  "with gravity compensation off.")
    finally:
        _teardown(robot, held, duration, ok, args.stay_held)
        print("ABORTED - the arm was put down, not dropped." if not ok
              else "Done.")


def _teardown(robot, start_motor, duration, ok, stay_held):
    """End in a state that does not depend on anyone catching the arm.

    Three cases, and the difference matters because disconnect() disables every
    motor:

      stay_held  the arm stays up, torque on, for a server taking over. The
                 motors are left energized deliberately; nothing here is
                 watching them afterwards.
      ok         ramp back down to where the arm started, on its stops, and
                 only then disable. A disable there costs nothing.
      aborted    lower the two gravity-loaded joints to their stops, holding
                 the others, and disable at the bottom.

    The aborted case used to do a damped release instead, on the reasoning
    that a ramp cannot be trusted once something is already wrong. That was
    the wrong trade: the envelope aborts on deviation, torque or temperature,
    and position control still works in all three, while the release was
    measured to be indistinguishable from cutting torque. The case where the
    descent genuinely cannot run is a motor that has stopped answering - and
    damping did nothing there either.
    """
    if stay_held:
        print("\n--stay_held: leaving the arm at the home pose with TORQUE ON.")
        print("Nothing is watching it now. Hand over to a server or power down.")
        robot.config.disable_torque_on_disconnect = False
        robot.skip_safe_zero_on_disconnect()
        robot.disconnect()
        return

    if ok:
        print(f"\n[return] ramping back to the start pose over {duration:.1f} s")
        now = motor_pose(robot)
        if run_leg(robot, now, start_motor, duration, "return"):
            disconnect_follower(robot, descend=False)   # already on its stops
            return
        print("  the return leg aborted - lowering from wherever it stopped")

    controlled_descent(robot)
    disconnect_follower(robot, descend=False)           # just put down


if __name__ == "__main__":
    main()
