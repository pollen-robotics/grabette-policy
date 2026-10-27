#!/usr/bin/env python3
"""Rungs 4 and 5. Trace a square with the control frame, via IK.

    uv run python examples/cartesian_square.py --can can0 --size_cm 2    # rung 4
    uv run python examples/cartesian_square.py --can can0 --size_cm 10   # rung 5

This is the first rung where the KINEMATICS are in the loop rather than just
the motors: every waypoint is an IK solve against the exported model. A clean
square means the URDF, the joint signs, the calibration and the solver all
agree with the hardware. The OpenArm README is blunt about why it matters - a
policy run on an arm that cannot trace a square produces garbage silently, and
the garbage looks like a bad policy rather than a bad arm.

DRY RUN FIRST, and it needs no hardware:

    uv run python examples/cartesian_square.py --dry_run --size_cm 10 \
        --from_joints 0 60 60 0 0 0

That solves the whole square offline from a given pose and reports the IK
residual and the largest joint step per segment. If the dry run cannot close
the square, the arm certainly cannot, and nothing had to move to find out.

SIX DOF HAS NO NULL SPACE
    The OpenArm is redundant: it can trade orientation against position and
    slide along its null space to avoid a branch flip. This arm cannot -
    position plus orientation is exactly determined. So a waypoint that is
    geometrically reachable can still be unreachable along a given path, and
    the failure mode is a large joint step between two neighbouring waypoints
    (a branch flip) rather than a solver error. --max_joint_step_deg catches
    that BEFORE anything is commanded; it defaults to the same 15 deg the
    server's IK-jump watchdog uses.

The square is traced in the plane normal to --axis, which defaults to x (the
arm's reach direction), so the square is drawn in y-z - a vertical square in
front of the arm. Orientation is held at whatever the start pose gives.
"""
import argparse
import sys

import numpy as np

from _bringup import (ARM_MOTORS, MOTORS, cold_read_positions, connect_follower,
                      disconnect_follower, hold, leg_duration, motor_pose,
                      print_pose, require_full_read, run_leg, urdf_limits)

AXES = {"x": 0, "y": 1, "z": 2}


def square_waypoints(start_xyz, axis, size_m, steps_per_side):
    """Corners of a square in the plane normal to `axis`, starting at a corner.

    The first waypoint is the start position itself, so the first segment
    commands no Cartesian motion and the arm eases into the trajectory.
    """
    u, v = [i for i in range(3) if i != axis]
    corners = [(0.0, 0.0), (size_m, 0.0), (size_m, size_m), (0.0, size_m), (0.0, 0.0)]
    pts = []
    for (u0, v0), (u1, v1) in zip(corners, corners[1:]):
        for k in range(steps_per_side):
            f = k / steps_per_side
            p = np.array(start_xyz, dtype=float)
            p[u] += u0 + (u1 - u0) * f
            p[v] += v0 + (v1 - v0) * f
            pts.append(p)
    pts.append(np.array(start_xyz, dtype=float))
    return pts


def solve_square(kin, frame, q_start_rad, axis, size_m, steps_per_side,
                 max_joint_step_deg):
    """Solve every waypoint offline. Returns (poses_rad, report, problems)."""
    start_T = kin.forward(q_start_rad, frame=frame)
    waypoints = square_waypoints(start_T[:3, 3], axis, size_m, steps_per_side)

    poses, report, problems = [], [], []
    q = np.array(q_start_rad, dtype=float)
    for i, p in enumerate(waypoints):
        target = start_T.copy()          # orientation held at the start pose
        target[:3, 3] = p
        q_next = kin.inverse(target, current_joint_positions=q, frame=frame)
        reached = kin.forward(q_next, frame=frame)
        err_mm = float(np.linalg.norm(reached[:3, 3] - p) * 1000.0)
        step_deg = float(np.max(np.abs(np.rad2deg(q_next - q))))

        if err_mm > 1.0:
            problems.append(f"waypoint {i}: IK residual {err_mm:.2f} mm - "
                            f"outside the reachable envelope along this path")
        if step_deg > max_joint_step_deg:
            problems.append(f"waypoint {i}: joint step {step_deg:.1f} deg "
                            f"(limit {max_joint_step_deg}) - almost certainly a "
                            f"branch flip; 6 DoF has no null space to escape it")

        report.append((i, p, err_mm, step_deg))
        poses.append(q_next)
        q = q_next
    return poses, report, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--id", default="rebot", help="LeRobot device id (picks the calibration)")
    ap.add_argument("--size_cm", type=float, required=True,
                    help="side length. Rung 4 is 2, rung 5 is 10. Do not skip rung 4.")
    ap.add_argument("--axis", choices=sorted(AXES), default="x",
                    help="normal of the square's plane (default x: a vertical "
                         "square in front of the arm)")
    ap.add_argument("--steps_per_side", type=int, default=10)
    ap.add_argument("--max_joint_step_deg", type=float, default=15.0,
                    help="matches the server's IK-jump watchdog")
    ap.add_argument("--deg_per_s", type=float, default=2.0)
    ap.add_argument("--dry_run", action="store_true",
                    help="solve offline and report; touches no hardware")
    ap.add_argument("--from_joints", type=float, nargs=6, metavar="DEG",
                    help="start pose for --dry_run (required with it)")
    args = ap.parse_args()

    from gripette_core.kinematics import CONTROL_FRAME, Kinematics
    from rebot_gripette.spec import ARM_JOINT_NAMES, REBOT_SPEC

    size_m = args.size_cm / 100.0
    kin = Kinematics(REBOT_SPEC)

    # ---- where the square starts from ------------------------------------
    if args.dry_run:
        if args.from_joints is None:
            print("--dry_run needs --from_joints (the pose to solve from)")
            sys.exit(2)
        start_motor = dict(zip(ARM_JOINT_NAMES, args.from_joints))
    else:
        from motorbridge import Controller
        with Controller(args.can) as ctrl:
            handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in MOTORS}
            start_motor = require_full_read(cold_read_positions(handles))
        print_pose("measured start pose (motor deg)", start_motor, urdf_limits())

    q_start = np.deg2rad([start_motor[n] for n in ARM_JOINT_NAMES])

    # ---- solve the whole path before commanding any of it -----------------
    poses, report, problems = solve_square(
        kin, CONTROL_FRAME, q_start, AXES[args.axis], size_m,
        args.steps_per_side, args.max_joint_step_deg)

    print(f"\nSolved {len(poses)} waypoints for a {args.size_cm:g} cm square "
          f"in the plane normal to {args.axis}")
    print(f"  {'wp':>3}{'x':>9}{'y':>9}{'z':>9}{'IK err':>9}{'max step':>10}")
    for i, p, err_mm, step_deg in report:
        flag = "  <--" if (err_mm > 1.0 or step_deg > args.max_joint_step_deg) else ""
        print(f"  {i:>3}{p[0]*1000:>9.1f}{p[1]*1000:>9.1f}{p[2]*1000:>9.1f}"
              f"{err_mm:>9.2f}{step_deg:>10.1f}{flag}")

    if problems:
        print(f"\n{len(problems)} problem(s) - NOTHING WAS COMMANDED:")
        for p in problems:
            print(f"  * {p}")
        print("\nMove the start pose into a better part of the envelope and "
              "re-solve. Do not raise --max_joint_step_deg to make this pass.")
        sys.exit(1)

    worst_err = max(r[2] for r in report)
    print(f"\nclean: worst IK residual {worst_err:.2f} mm, "
          f"worst joint step {max(r[3] for r in report):.1f} deg")

    if args.dry_run:
        print("\nDry run only - no hardware was touched.")
        return

    # ---- trace it ---------------------------------------------------------
    print("\nThis script WILL move the arm through the square above. "
          "Hand on the power.\n")
    robot = connect_follower(args.can, args.id)
    ok = False
    try:
        held = motor_pose(robot)
        print("[engage] holding the measured pose - the arm must NOT move")
        if not hold(robot, held, 1.5, label="engage"):
            return

        current = held
        ok = True
        for i, q in enumerate(poses):
            target = dict(zip(ARM_JOINT_NAMES, np.rad2deg(q)))
            delta = max(abs(target[n] - current[n]) for n, _, _ in ARM_MOTORS)
            seg_s = leg_duration(delta, args.deg_per_s)
            ok = run_leg(robot, current, target, seg_s, f"wp{i:02d}",
                         sample_s=max(seg_s, 0.5))
            if not ok:
                break
            current = target

        if ok:
            end = motor_pose(robot)
            print("\nclosure error vs the start pose (motor deg):")
            for name, _, _ in ARM_MOTORS:
                print(f"  {name:<15}{end[name] - held[name]:+8.3f}")
            print("\nIf the square was not visually clean, STOP here. "
                  "A policy on an arm that cannot trace a square fails silently.")
    finally:
        disconnect_follower(robot)
        print("ABORTED - the arm stopped where it was." if not ok else "Done.")


if __name__ == "__main__":
    main()
