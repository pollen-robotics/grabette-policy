"""ArmService gRPC server for the real reBot B601-RS.

Exposes the SAME ArmService as the OpenArm and as the simulator, because the
service implementation is literally the same class — `gripette_core.
arm_service.ArmServicer`. This module supplies only the reBot half: the
ArmSpec, the CAN-side ArmInterface and the CLI.

    uv run python -m rebot_gripette.grpc_server_real --can can0 --arm_port 50052

Then point the eval / teleop client at `--arm_addr <robot-ip>:50052`, exactly
as for the OpenArm.

SAFETY, READ BEFORE RUNNING
    The server does NOT enable torque on startup; it comes up limp, and the
    client must call SetTorque(enable=True). Enabling re-applies the per-joint
    torque limits, because the RobStride 0x7000 range is volatile.

    CTRL-C DOES NOT MOVE THE ARM, AND DOES NOT DROP IT. Two separate
    hazards live in the upstream disconnect():

      safe_zero()  drives all six joints to motor zero - the LOWER HARD STOP
                   for shoulder_lift and elbow_flex on this arm. An interrupt
                   must never start a motion, so it is suppressed (from
                   ReBotFollower.__init__, so even a failed connect is covered).
      torque off   disable_torque_on_disconnect defaults True, so the motors
                   are cut while the arm is wherever the policy left it. That
                   is a free fall.

    shutdown() therefore PUTS THE ARM DOWN before disabling: shoulder_lift and
    elbow_flex are lowered to their stops under position control, the other
    four joints held where they are, so the disable lands on an arm already
    resting. This was a damped release (MIT kp=0) until 2026-10-28, when the
    damping was measured on the arm and found indistinguishable from cutting
    torque - see ReBotFollower.controlled_descent for the arithmetic. Park the
    arm deliberately if you want it somewhere specific; do not get there by
    killing the server.

    THE HOME POSE HAS NO DEFAULT, on purpose. Two different things get
    called "home":

      starting pose  where the robot physically is at power-on - on this arm,
                     resting on its hard mechanical stops. An observation.
      home pose      the startup position for an EXPERIMENT. `Reset` ramps the
                     arm to it, so it is the first commanded motion of a run.

    The second is a property of the experiment, not of the robot: the
    OpenArm's is its simulator's START_JOINTS, the pose its dataset was
    recorded from. The reBot has no such experiment yet, so there is nothing
    to inherit and nothing to guess - a pose picked merely for being "inside
    the limits" is still a guess, and Reset would drive the arm into it.

    So --home_joints is REQUIRED. Supply six angles in degrees, in the order
    shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_yaw wrist_roll.
    They are checked against the model (count, and >=5 deg clear of every
    hard stop) - necessary conditions only. Nothing here knows about your
    table, the pedestal, the Gripette's envelope or self-collision. Watch the
    first Reset with a hand on the power.
"""

import argparse
import logging
import signal
import sys
import time
from concurrent import futures

import grpc
import numpy as np
from gripette_core.arm_service import ArmServicer
from gripette_core.kinematics import Kinematics
from gripette_core.proto import arm_pb2_grpc

from rebot_gripette.arm_interface import ReBotArmInterface
from rebot_gripette.follower import ReBotFollower, ReBotFollowerConfig
from rebot_gripette.spec import ARM_JOINT_NAMES, REBOT_SPEC, check_home_pose

logger = logging.getLogger(__name__)



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--can", default="can0", help="SocketCAN interface")
    p.add_argument(
        "--home_joints", type=float, nargs=6, required=True, metavar="DEG",
        help="REQUIRED. The experiment's startup pose, six angles in degrees, "
             "in the order " + " ".join(ARM_JOINT_NAMES) + ". Reset ramps the "
             "arm to this, so it is the first commanded motion. There is no "
             "default: it belongs to the experiment, not the robot.")
    p.add_argument("--arm_port", type=int, default=50052)
    p.add_argument("--id", default="rebot", help="LeRobot device id (picks the calibration)")

    # Solver weights. The OpenArm defaults are 100:1, and its eval server uses
    # orientation_weight=10. This arm is 6-DoF and has NO null space, so it
    # cannot trade orientation against position the way the redundant OpenArm
    # can; expect to need tuning here rather than inheriting blindly.
    p.add_argument("--ik_position_weight", type=float, default=100.0)
    p.add_argument("--ik_orientation_weight", type=float, default=10.0)

    p.add_argument("--interp_hz", type=float, default=50.0)
    p.add_argument("--interp_alpha", type=float, default=0.3)
    p.add_argument("--max_ik_jump_deg", type=float, default=15.0)
    p.add_argument("--max_ik_jump_violations", type=int, default=2)
    p.add_argument("--max_target_lead_mm", type=float, default=50.0)
    p.add_argument("--clearance_margin_mm", type=float, default=20.0,
                   help="keep the guarded links this far from the bench. 0 "
                        "disables the gate - see --no_clearance.")
    p.add_argument("--no_clearance", action="store_true",
                   help="run WITHOUT the bench clearance gate. The IK-jump "
                        "watchdog still catches branch flips, but nothing "
                        "then stops a smooth IK solution driving into the "
                        "bench. Intended for a bench-less rig, not as a way "
                        "to silence a gate that is firing.")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _build_clearance_gate(args, arm_iface):
    """The bench clearance gate, or None if it is off or unavailable.

    The IK-jump watchdog catches the SOLVER flipping branch. It cannot catch a
    smooth run of well-conditioned solutions that walks into the bench the arm
    is bolted to, because nothing about those solutions is ill-conditioned.
    This is the guard for that, and it is the same `joint_gate` hook the
    simulator uses - one implementation, measured against the same declared
    links (rebot_gripette.obstacles).

    MuJoCo is an OPTIONAL dependency of this package: a machine that only
    drives the arm need not carry the simulation stack. Without it the gate
    cannot be built, and the server says so loudly rather than coming up
    silently unguarded - a missing safety gate must never look like a working
    one.
    """
    if args.no_clearance:
        logger.warning(
            "--no_clearance: the bench clearance gate is OFF. The IK-jump "
            "watchdog still catches branch flips, but NOTHING stops a smooth "
            "IK solution driving the arm into the bench.")
        return None

    try:
        from gripette_sim.clearance import GeomClearance
        from gripette_sim.simulation import _load_model
    except ImportError as e:
        logger.error(
            f"Bench clearance gate UNAVAILABLE ({e}). The arm will run with "
            f"the IK-jump watchdog only - nothing stops a smooth IK solution "
            f"driving into the bench. Install the clearance extra "
            f"(`uv sync --package rebot-gripette --extra clearance`), or pass "
            f"--no_clearance to accept this deliberately.")
        return None

    from rebot_gripette.obstacles import GUARDED_BODIES, OBSTACLE_GEOMS
    from rebot_gripette_model import REBOT_B601RS_BENCH

    model = _load_model(REBOT_B601RS_BENCH)
    clearance = GeomClearance(
        model, OBSTACLE_GEOMS, GUARDED_BODIES, list(REBOT_SPEC.joint_names),
        margin=args.clearance_margin_mm / 1000.0)
    if not clearance.enabled:
        logger.error(
            "Bench clearance gate found no geometry to guard - running "
            "UNGUARDED. Check rebot_gripette.obstacles against the model.")
        return None

    logger.info(
        f"Bench clearance gate armed: {args.clearance_margin_mm:.0f} mm from "
        f"the bench, guarding {len(GUARDED_BODIES)} links. A command that "
        f"would breach it is REFUSED and the arm holds its last good pose; "
        f"moves that increase clearance are always allowed.")
    # `current_joints` is what lets an arm already inside the margin retreat.
    # Without it every escape command is also inside the margin and the arm is
    # deadlocked until the server is killed.
    return clearance.as_joint_gate(current_joints=arm_iface.get_positions)


def main():
    args = parse_args()
    # force=True because importing placo/mujoco configures the root logger
    # first, which makes a plain basicConfig a silent no-op: INFO vanishes and
    # warnings come out in the default format. That hid the line saying
    # whether the clearance gate armed, which is the one line worth reading at
    # startup.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    start_time = time.monotonic()

    # Validate the home pose BEFORE opening the CAN bus, so a bad value fails
    # with the arm untouched rather than half-way through a connect.
    try:
        home_joints = check_home_pose(np.deg2rad(args.home_joints))
    except ValueError as e:
        logger.error(f"--home_joints rejected: {e}")
        sys.exit(2)
    logger.info(
        f"Home pose (experiment startup position): "
        f"{[round(d, 1) for d in args.home_joints]} deg. Reset will RAMP the "
        f"arm to this - make sure it is clear of the table and of itself.")

    logger.info(f"Connecting to the reBot on {args.can} (id={args.id})")
    robot = ReBotFollower(ReBotFollowerConfig(port=args.can, id=args.id))
    robot.connect()
    logger.info("Connected. Torque is NOT enabled; call SetTorque to arm the robot.")

    arm_iface = ReBotArmInterface(robot)

    logger.info(
        f"Loading kinematics (placo, URDF from rebot_gripette_model), "
        f"weights pos={args.ik_position_weight} orient={args.ik_orientation_weight}")
    kin = Kinematics(
        REBOT_SPEC,
        position_weight=args.ik_position_weight,
        orientation_weight=args.ik_orientation_weight,
    )

    joint_gate = _build_clearance_gate(args, arm_iface)

    servicer = ArmServicer(
        arm_iface,
        kin,
        home_joints,
        start_time,
        joint_gate=joint_gate,
        interp_hz=args.interp_hz,
        interp_alpha=args.interp_alpha,
        max_ik_jump_deg=args.max_ik_jump_deg,
        max_ik_jump_violations=args.max_ik_jump_violations,
        max_target_lead_mm=args.max_target_lead_mm,
    )

    # SINGLE-INSTANCE GUARD. gRPC enables SO_REUSEPORT by default on Linux, so
    # a second server binds the same port with no error, and both instances
    # then drive the same CAN bus from their own interpolators. On the OpenArm
    # that made the arm shake violently. Disable reuseport AND check the
    # return value (add_insecure_port returns 0 on failure).
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                         options=[("grpc.so_reuseport", 0)])
    arm_pb2_grpc.add_ArmServiceServicer_to_server(servicer, server)
    if server.add_insecure_port(f"0.0.0.0:{args.arm_port}") == 0:
        logger.error(
            f"Could not bind port {args.arm_port} — another server is almost "
            f"certainly already running. Refusing to start a second one.")
        servicer.stop()
        # Worse here than on Ctrl-C. connect() has already run, on a bus the
        # FIRST server is actively driving with torque enabled — so feedback is
        # live and safe_zero would genuinely succeed in hauling the arm to zero
        # while the other process commands it. The "disabled motors can't be
        # read, so safe_zero raises" accident does not save us in this path.
        robot.skip_safe_zero_on_disconnect()
        robot.disconnect()
        sys.exit(1)

    def shutdown(signum, _frame):
        logger.warning(f"Signal {signum} — stopping, putting the arm down.")
        servicer.stop()
        # An interrupt must not start a motion. Without this, disconnect()
        # calls safe_zero() and drives the arm to zero — two hard stops on this
        # arm. Parking is a thing a human asks for, not a side effect of Ctrl-C.
        robot.skip_safe_zero_on_disconnect()
        try:
            # ...and it must not drop the arm either. Cutting torque outright
            # is a free fall from wherever the policy left it. This lowers the
            # two gravity-loaded joints to their stops under position control
            # first — a few seconds — so the disable below happens with the arm
            # already down. Damping was tried here and measured useless on
            # 2026-10-28; see ReBotFollower.controlled_descent.
            robot.controlled_descent()
        except Exception:
            logger.exception("Controlled descent failed; disabling torque anyway.")
        try:
            arm_iface.set_torque(False)
        finally:
            robot.disconnect()
        server.stop(0)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.start()
    logger.info(f"ArmService listening on 0.0.0.0:{args.arm_port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
