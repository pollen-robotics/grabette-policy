"""ArmService gRPC server for the real reBot B601-RS.

Exposes the SAME ArmService as the OpenArm and as the simulator, because the
service implementation is literally the same class — `gripette_core.
arm_service.ArmServicer`. This module supplies only the reBot half: the
ArmSpec, the home pose, the CAN-side ArmInterface and the CLI.

    uv run python -m rebot_gripette.grpc_server_real --can can0 --arm_port 50052

Then point the eval / teleop client at `--arm_addr <robot-ip>:50052`, exactly
as for the OpenArm.

SAFETY, READ BEFORE RUNNING
    The server does NOT enable torque on startup; it comes up limp, and the
    client must call SetTorque(enable=True). Enabling re-applies the per-joint
    torque limits, because the RobStride 0x7000 range is volatile.

    Home is not all zeros. shoulder_lift is 0..170 deg and elbow_flex 0..200
    deg, so zero sits exactly on two hard stops; a reset to zero would push
    the arm into them and leave Placo's joint-limit constraint with no room on
    one side. HOME_JOINTS_RAD below is inside the envelope.
"""

import argparse
import logging
import signal
import sys
import time
from concurrent import futures

import grpc
from gripette_core.arm_service import ArmServicer
from gripette_core.kinematics import Kinematics
from gripette_core.proto import arm_pb2_grpc

from rebot_gripette.arm_interface import ReBotArmInterface
from rebot_gripette.follower import ReBotFollower, ReBotFollowerConfig
from rebot_gripette.spec import HOME_JOINTS_RAD, REBOT_SPEC

logger = logging.getLogger(__name__)



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--can", default="can0", help="SocketCAN interface")
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
        HOME_JOINTS_RAD,
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
