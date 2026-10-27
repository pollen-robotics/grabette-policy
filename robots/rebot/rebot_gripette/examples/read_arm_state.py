#!/usr/bin/env python3
"""Rung 1 of the bring-up ladder. Reads the arm. Commands NOTHING.

No motor is enabled, no torque is applied, no position is sent. Position comes
from parameter reads (RobStride type 17), which are answered by a DISABLED
motor - unlike feedback frames, which are not sent until enable(). So this is
the one rung that cannot move the arm even if it is wrong.

What it is for:

  1. Confirm the bus. Six motors answering is the precondition for everything
     above this rung.
  2. Record the STARTING POSE - where the arm physically is at power-on. On
     this arm that is resting on its hard mechanical stops, and the printout
     marks which joints are on one. This is an observation about the robot; it
     is NOT the home pose (see spec.py).
  3. Report the control-frame position the kinematics compute from it, as a
     sanity check against a tape measure.

    uv run python examples/read_arm_state.py --can can0

ONE JOINT SPACE
    Motor degrees are the URDF's angles. Nothing converts, mirrors or rescales
    anywhere: ReBotFollowerConfig pins the vendor's joint_directions mapping to
    the identity (follower.py explains why it exists and why it is not wanted
    here). Rung 0, examples/mirror_arm.py, is what confirms that against the
    hardware - drive the model from the real arm by hand and watch it track.
"""
import argparse

import numpy as np

from _bringup import (ARM_MOTORS, MOTORS, VBUS, cold_read_positions,
                      require_full_read, urdf_limits)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--no_kinematics", action="store_true",
                    help="skip the FK section (no placo / model needed)")
    args = ap.parse_args()

    from motorbridge import Controller

    with Controller(args.can) as ctrl:
        handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in MOTORS}

        print(f"Reading {len(ARM_MOTORS)} arm joints on {args.can}. "
              f"Nothing is enabled; the arm cannot move.\n")
        pos = require_full_read(cold_read_positions(handles))

        # ---- starting pose, against the model's limits ---------------------
        limits = urdf_limits()

        print("STARTING POSE - motor degrees, as reported by mechPos")
        print(f"  {'joint':<15}{'position':>10}{'urdf limits':>22}{'margin':>9}")
        on_a_stop = []
        for name, _, _ in ARM_MOTORS:
            lo, hi = limits[name]
            margin = min(pos[name] - lo, hi - pos[name])
            if margin < 1.0:
                on_a_stop.append(name)
            print(f"  {name:<15}{pos[name]:>10.3f}"
                  f"{f'[{lo:.0f}, {hi:.0f}]':>22}{margin:>9.1f}"
                  f"{'  <-- ON A STOP' if margin < 1.0 else ''}")

        if on_a_stop:
            print(f"\n  Resting on a hard stop: {', '.join(on_a_stop)}. Expected "
                  f"at power-on; this is the starting pose, not a fault.")
            print("  A joint on a stop can read a few hundredths PAST its URDF "
                  "limit.\n  That is where the limit was defined, not a fault "
                  "either - but a\n  solver fed this pose must clamp.")

        # ---- supply ---------------------------------------------------------
        print("\nBUS VOLTAGE (a sag under load points at the supply, not the arm)")
        for name, _, _ in ARM_MOTORS:
            try:
                v = handles[name].robstride_get_param_f32(VBUS, 1000)
                print(f"  {name:<15}{v:>8.2f} V")
            except Exception as exc:
                print(f"  {name:<15}{'read failed':>8} - {type(exc).__name__}")

        if args.no_kinematics:
            return

        # ---- where the model says the gripper is ---------------------------
        from gripette_core.kinematics import CONTROL_FRAME, Kinematics
        from rebot_gripette.spec import ARM_JOINT_NAMES, REBOT_SPEC

        kin = Kinematics(REBOT_SPEC)
        q = np.deg2rad([pos[n] for n in ARM_JOINT_NAMES])
        t = kin.forward(q, frame=CONTROL_FRAME)[:3, 3] * 1000.0

        print("\nCONTROL FRAME, computed from the pose above")
        print(f"  x={t[0]:+8.1f}  y={t[1]:+8.1f}  z={t[2]:+8.1f} mm, from the "
              f"base frame")
        print("  Worth measuring once against the real arm. A gross "
              "disagreement means\n  the model and the robot have parted "
              "company - run rung 0 before anything\n  that moves.")


if __name__ == "__main__":
    main()
