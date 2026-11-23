#!/usr/bin/env python3
"""Record a pose a human is holding the arm in. Commands NOTHING.

Same read path as rung 1 - parameter reads on disabled motors - so no motor is
enabled, no torque is applied and nothing is sent. The arm is limp and someone
is physically holding it.

    uv run python examples/capture_pose.py --can can0 --hold_s 4

WHY A SAMPLE AND NOT A SNAPSHOT
    A hand-held arm is never still. One read records the tremor as much as the
    pose, and worse, gives no indication of how much tremor there was. This
    samples for `--hold_s` and reports the MEDIAN per joint (robust to the odd
    drifted sample) alongside the peak-to-peak spread, which is the honest
    measure of how precisely the pose was captured.

WHAT THE RESULT IS FOR
    A seed, not a home pose. The printed angles go into the simulator to be
    refined into something defensible - clear of the stops, well conditioned
    for IK, able to trace a square - and it is that refined pose that becomes
    --home_joints. A pose held by hand is a statement of intent about where
    the arm should start, not a specification.

    Stop clearance is checked here anyway, because a seed already touching a
    stop is not worth refining: the refinement has nowhere to go on that joint.
"""
import argparse
import statistics
import time

from _bringup import ARM_MOTORS, MOTORS, cold_read_positions, urdf_limits

# spec.check_home_pose demands 5 deg; a seed that is already tighter than that
# leaves the sim no room on that joint, so warn at the same threshold.
CLEARANCE_DEG = 5.0


def summarise(samples):
    """[{joint: deg}, ...] -> {joint: (median, peak_to_peak, n)}.

    Pure, so the arithmetic is tested without an arm to hold.
    """
    out = {}
    for name in samples[0]:
        values = [s[name] for s in samples if name in s]
        out[name] = (statistics.median(values),
                     max(values) - min(values),
                     len(values))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--hold_s", type=float, default=4.0,
                    help="seconds to sample while the pose is held")
    ap.add_argument("--countdown_s", type=float, default=3.0,
                    help="seconds before sampling starts, to get into position")
    args = ap.parse_args()

    from motorbridge import Controller

    with Controller(args.can) as ctrl:
        handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in MOTORS}

        print(f"Nothing is enabled; the arm is limp and cannot move itself.\n"
              f"Hold it in the pose you want. Sampling for {args.hold_s:.0f} s.")
        for remaining in range(int(args.countdown_s), 0, -1):
            print(f"  {remaining}...", flush=True)
            time.sleep(1.0)
        print("  sampling", flush=True)

        samples = []
        deadline = time.time() + args.hold_s
        while time.time() < deadline:
            pos = cold_read_positions(handles)
            if len(pos) == len(ARM_MOTORS):     # partial reads are not a pose
                samples.append(pos)
        if not samples:
            raise SystemExit("No complete read in the sampling window - check "
                             "power and CAN wiring. Nothing was enabled.")

        stats = summarise(samples)
        limits = urdf_limits()

        print(f"\nHELD POSE - median of {len(samples)} samples, motor degrees")
        print(f"  {'joint':<15}{'median':>10}{'wobble':>9}{'urdf limits':>20}"
              f"{'margin':>9}")
        tight = []
        for name, _, _ in ARM_MOTORS:
            median, spread, _ = stats[name]
            lo, hi = limits[name]
            margin = min(median - lo, hi - median)
            if margin < CLEARANCE_DEG:
                tight.append(name)
            print(f"  {name:<15}{median:>10.2f}{spread:>9.2f}"
                  f"{f'[{lo:.0f}, {hi:.0f}]':>20}{margin:>9.1f}"
                  f"{'  <-- TOO CLOSE' if margin < CLEARANCE_DEG else ''}")

        worst = max(stats.values(), key=lambda v: v[1])[1]
        print(f"\n  Worst wobble {worst:.2f} deg over the window. Treat the "
              f"medians as good to\n  about that much - a seed, not a "
              f"specification.")

        if tight:
            print(f"\n  WITHIN {CLEARANCE_DEG:.0f} deg OF A STOP: "
                  f"{', '.join(tight)}. spec.check_home_pose would reject this "
                  f"pose,\n  and refining it leaves no room on those joints. "
                  f"Re-hold the arm further\n  from the stop and run again.")

        print("\n  as --home_joints (degrees, seed only - refine in sim first):")
        print("    " + " ".join(f"{stats[n][0]:.1f}" for n, _, _ in ARM_MOTORS))


if __name__ == "__main__":
    main()
