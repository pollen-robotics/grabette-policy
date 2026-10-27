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
  3. Settle the sign convention empirically, which is the part worth your
     attention - see below.

    uv run python examples/read_arm_state.py --can can0

THE SIGN CONVENTION THIS RUNG CONFIRMS
    Two spaces are in play. MOTOR space is what the motor reports. LOGICAL
    space is motor divided by `joint_directions`, which is -1.0 for elbow_flex,
    wrist_flex and wrist_yaw - so those three are mirrored between the two.

    The exported URDF is MOTOR space: 22_mirror.py drove the model from raw
    mechPos with every sign at +1 and confirmed joint-by-joint agreement. The
    arm interface converts the follower's logical output back to motor space
    before the kinematics see it, in both directions.

    That is settled from source. This rung confirms it against a ruler, which
    source cannot do: move the arm by hand to a pose you can measure, run this,
    and check the MOTOR row matches reality. If the LOGICAL row matches
    instead, something has regressed - stop before any rung that moves.
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

        print("STARTING POSE - motor space, as reported by mechPos")
        print(f"  {'joint':<15}{'motor':>10}{'logical':>10}"
              f"{'urdf limits':>22}{'margin':>9}")
        on_a_stop = []
        for name, _, _ in ARM_MOTORS:
            lo, hi = limits[name]
            margin = min(pos[name] - lo, hi - pos[name])
            if margin < 1.0:
                on_a_stop.append(name)
            # joint_directions lives on the plugin config; read it without
            # connecting anything.
            print(f"  {name:<15}{pos[name]:>10.3f}{_logical(name, pos[name]):>10.3f}"
                  f"{f'[{lo:.0f}, {hi:.0f}]':>22}{margin:>9.1f}"
                  f"{'  <-- ON A STOP' if margin < 1.0 else ''}")

        if on_a_stop:
            print(f"\n  Resting on a hard stop: {', '.join(on_a_stop)}. Expected "
                  f"at power-on; this is the starting pose, not a fault.")

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

        # ---- the sign-convention question ----------------------------------
        from gripette_core.kinematics import CONTROL_FRAME, Kinematics
        from rebot_gripette.spec import ARM_JOINT_NAMES, REBOT_SPEC

        kin = Kinematics(REBOT_SPEC)
        q_motor = np.deg2rad([pos[n] for n in ARM_JOINT_NAMES])
        q_logical = np.deg2rad([_logical(n, pos[n]) for n in ARM_JOINT_NAMES])

        print("\nCONTROL FRAME POSITION under each reading of the URDF")
        print("  Measure the real gripper against the base with a ruler. The MOTOR")
        print("  row is the one that should match; if LOGICAL matches instead,")
        print("  something regressed - stop before any rung that moves the arm.")
        for label, q in (("URDF == motor space   (expected - what the code does)", q_motor),
                         ("URDF == logical space (the old bug; mirrored on 3 joints)", q_logical)):
            t = kin.forward(q, frame=CONTROL_FRAME)[:3, 3] * 1000.0
            print(f"  {label:<52} x={t[0]:+8.1f}  y={t[1]:+8.1f}  z={t[2]:+8.1f} mm")

        delta = float(np.linalg.norm(
            kin.forward(q_motor, frame=CONTROL_FRAME)[:3, 3]
            - kin.forward(q_logical, frame=CONTROL_FRAME)[:3, 3]) * 1000.0)
        if delta < 1.0:
            print(f"\n  The two agree to {delta:.2f} mm at this pose, so this pose")
            print("  cannot distinguish them. Move the mirrored joints (elbow_flex,")
            print("  wrist_flex, wrist_yaw) well off zero by hand and re-run.")
        else:
            print(f"\n  They disagree by {delta:.1f} mm here, so this pose DOES")
            print("  distinguish them. Measure and record which one is right.")


_DIRECTIONS = None


def _logical(joint, motor_degrees):
    """motor -> logical, without connecting anything: read the config default."""
    global _DIRECTIONS
    if _DIRECTIONS is None:
        from lerobot_robot_seeed_b601 import SeeedB601RSFollowerConfig
        _DIRECTIONS = SeeedB601RSFollowerConfig(id="_", port="can0").joint_directions
    return motor_degrees / _DIRECTIONS[joint]


if __name__ == "__main__":
    main()
