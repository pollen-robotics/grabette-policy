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
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
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

    servicer = ArmServicer(
        arm_iface,
        kin,
        home_joints,
        start_time,
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
        robot.disconnect()
        sys.exit(1)

    def shutdown(signum, _frame):
        logger.warning(f"Signal {signum} — stopping, disabling torque.")
        servicer.stop()
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
