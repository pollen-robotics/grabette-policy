#!/usr/bin/env python3
"""Apply and read back the per-joint torque / velocity limits. NO MOTION.

Moved out of rebot_gripette/limits.py, where it lived as a `__main__` block.
A library module that opens `Controller("can0")` when run is a trap: anything
that executes the file — a stray `python limits.py`, a doc tool, an IDE "run
file" button — touches the bus. Diagnostics that talk to hardware belong in
examples/, like the OpenArm's.

This writes the limits and reads them back. It does NOT enable torque and does
NOT command any position, so the arm cannot move. It is the right first thing
to run after a power cycle, because those registers are VOLATILE: the
RobStride 0x7000 range is runtime-only and `store_parameters()` does not
persist it (only 0x2000 survives). Every connect must re-apply them.

    uv run python examples/read_torque_limits.py --can can0
"""
import argparse

from rebot_gripette.limits import apply_limits

# (joint, CAN id, RobStride model). Joints 1-3 are RS-06, 4-7 RS-00.
MOTORS = [("shoulder_pan", 1, "rs-06"), ("shoulder_lift", 2, "rs-06"),
          ("elbow_flex", 3, "rs-06"), ("wrist_flex", 4, "rs-00"),
          ("wrist_yaw", 5, "rs-00"), ("wrist_roll", 6, "rs-00"),
          ("gripper", 7, "rs-00")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--strict", action="store_true",
                    help="raise if any limit fails to read back as written")
    args = ap.parse_args()

    from motorbridge import Controller

    with Controller(args.can) as ctrl:
        handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in MOTORS}
        result = apply_limits(handles, strict=args.strict)

        print(f"{'joint':<15}{'limit_torque':>14}{'vel_max':>10}")
        for name, _, _ in MOTORS:
            vals = result.get(name, {})
            t = vals.get("limit_torque")
            v = vals.get("vel_max")
            print(f"{name:<15}"
                  f"{(f'{t:.2f}' if t is not None else 'FAILED'):>14}"
                  f"{(f'{v:.2f}' if v is not None else '-'):>10}")
        print("\nNot persistent by design - re-apply on every connect.")


if __name__ == "__main__":
    main()
