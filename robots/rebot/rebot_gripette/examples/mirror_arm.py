#!/usr/bin/env python3
"""Rung 0 of the bring-up ladder. Live mirror: real arm -> MuJoCo. NO TORQUE.

Nothing is enabled, nothing is commanded. Position comes from parameter reads
(RobStride type 17), which a DISABLED motor answers - the same read rung 1
uses. There is no enable() and no send_mit() anywhere in this file, so it
cannot move the arm; the arm is dead weight resting on its stops and you move
it with your hands.

    uv run python examples/mirror_arm.py --can can0

Move a joint and watch the model follow. A disagreement in DIRECTION,
MAGNITUDE or ZERO shows up immediately and visually, which is stronger than
any single-pose numeric check: it catches a mirrored export that is
self-consistent in the numbers and still wrong. This is how the exported model
was validated in the first place (rebot-lerobot's 22_mirror.py).

WHAT TO LOOK FOR
    --space motor  (default) drives the model from the raw motor angle, which
                   is what this package commands and what the URDF models. The
                   model should track the real arm joint for joint.
    --space vendor drives it through the upstream plugin's joint_directions
                   mapping instead - the leader-arm table this package pins to
                   the identity (see follower.py). elbow_flex, wrist_flex and
                   wrist_yaw are -1.0 there, so those three should visibly run
                   BACKWARDS against the real arm.

    Both are predictions. If they hold, the model is confirmed against the
    hardware across the whole range rather than at one pose. If `motor` is the
    one that runs backwards, stop - the kinematics are mirrored and no rung
    that commands a pose is safe.

    There are deliberately no --signs/--offsets knobs. A joint that disagrees
    is a finding about the model, not something to correct at the viewer.

THE FLOATING BASE
    The export has an axis_1_freejoint and scene.xml does not weld it, so
    stepping the physics would topple the arm. This calls mj_forward instead -
    kinematics only, no dynamics - and pins the free joint to identity every
    frame. For a kinematic mirror that is exactly right.

The Gripette fingers (proximal, distal) stay at zero: motor 7 is the reBot's
rack gripper, which the Gripette replaces, and the Feetech servos are not on
the CAN bus.
"""
import argparse
import math
import time

from _bringup import ARM_MOTORS, MECH_POS, require_full_read

# Parameter reads are a request/response round trip per joint, so the rate is
# bounded by the bus, not by the viewer. 30 Hz is what 22_mirror.py sustained
# and is well above what the eye needs to judge direction.
RATE_HZ = 30.0

# Short, unlike the 1000 ms preflight read: in a loop a dropped frame must cost
# one stale sample, not a visible stall.
READ_TIMEOUT_MS = 200


def to_space(pos_motor, space, directions):
    """Motor degrees -> whichever reading the model is being driven from.

    `vendor` divides by the upstream joint_directions exactly as the plugin's
    get_observation/send_action pair would have, which mirrors three joints.
    Pure, so the prediction in the docstring above is checked offline by the
    test suite rather than only by eye at the hardware.
    """
    if space == "motor":
        return dict(pos_motor)
    return {n: v / directions[n] for n, v in pos_motor.items()}


def read_positions(handles, timeout_ms=READ_TIMEOUT_MS):
    """{joint: motor degrees} for whichever joints answered this cycle.

    A joint that did not answer is ABSENT rather than 0.0 - zero is the lower
    hard stop for shoulder_lift and elbow_flex, so a defaulted read would draw
    the arm folded onto its stops. The caller keeps the previous sample.
    """
    out = {}
    for name, _, _ in ARM_MOTORS:
        try:
            out[name] = math.degrees(
                handles[name].robstride_get_param_f32(MECH_POS, timeout_ms))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0", help="SocketCAN interface")
    ap.add_argument("--space", choices=("motor", "vendor"), default="motor",
                    help="which reading of the angles drives the model")
    ap.add_argument("--rate_hz", type=float, default=RATE_HZ)
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer
    from motorbridge import Controller
    from rebot_gripette_model import REBOT_B601RS_SCENE

    model = mujoco.MjModel.from_xml_path(str(REBOT_B601RS_SCENE))
    data = mujoco.MjData(model)

    # Map by joint name. The model's hinges are named exactly as the motors
    # are, and a rename should stop the script rather than silently shift the
    # mapping by one.
    qadr = {}
    for name, _, _ in ARM_MOTORS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"model has no joint named {name!r} - the export "
                             f"and the motor table have diverged")
        qadr[name] = model.jnt_qposadr[jid]

    free = [i for i in range(model.njnt)
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]

    directions = None
    if args.space == "vendor":
        from lerobot_robot_seeed_b601 import SeeedB601RSFollowerConfig
        directions = SeeedB601RSFollowerConfig(id="_", port=args.can).joint_directions

    print(f"scene  {REBOT_B601RS_SCENE}")
    print(f"space  {args.space}" + ("  <-- elbow_flex, wrist_flex and wrist_yaw "
                                    "should run BACKWARDS" if args.space == "vendor" else ""))
    print(f"free joint: {'yes - pinned each frame' if free else 'no'}")

    with Controller(args.can) as ctrl:
        handles = {n: ctrl.add_robstride_motor(c, 0xFD, m) for n, c, m in ARM_MOTORS}

        # Refuse to start on a partial read: a missing joint would sit at
        # whatever the model's default is and look like a real pose.
        pos = require_full_read(read_positions(handles, timeout_ms=1000))
        print("\nNothing is enabled and nothing is commanded; the arm is limp "
              "and resting on its stops.")
        print("Move it by hand and watch the model. Ctrl-C or close the viewer "
              "to stop.\n")

        drops = 0
        dt = 1.0 / args.rate_hz
        with mujoco.viewer.launch_passive(model, data) as viewer:
            last_print = 0.0
            while viewer.is_running():
                t0 = time.perf_counter()

                fresh = read_positions(handles)
                drops += len(ARM_MOTORS) - len(fresh)
                pos.update(fresh)          # a silent joint holds its last pose

                shown = to_space(pos, args.space, directions)
                for f in free:             # pin the base; nothing may drift
                    a = model.jnt_qposadr[f]
                    data.qpos[a:a + 3] = 0.0
                    data.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
                for name, adr in qadr.items():
                    data.qpos[adr] = math.radians(shown[name])

                mujoco.mj_forward(model, data)   # kinematics only, no dynamics
                viewer.sync()

                if time.perf_counter() - last_print > 0.5:
                    row = "  ".join(f"{n.split('_')[0]}={pos[n]:+7.2f}"
                                    for n, _, _ in ARM_MOTORS)
                    print(f"\r motor deg  {row}   dropped {drops}",
                          end="", flush=True)
                    last_print = time.perf_counter()

                slack = dt - (time.perf_counter() - t0)
                if slack > 0:
                    time.sleep(slack)

    print("\nstopped; nothing was ever enabled")


if __name__ == "__main__":
    main()
