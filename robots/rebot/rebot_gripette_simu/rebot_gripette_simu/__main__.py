"""Launch the reBot simulation gRPC server.

    uv run python -m rebot_gripette_simu --home_joints 0 15 25 -10 0 0
    uv run python -m rebot_gripette_simu --home_joints 0 15 25 -10 0 0 --headless

--home_joints is REQUIRED, for the same reason it is required on the real
server (rebot_gripette.grpc_server_real): "home" is the startup position for an
EXPERIMENT, not a property of the robot, and `Reset` drives the arm to it. The
reBot has no experiment yet, so any default shipped here would be a guess that
the simulator then trains a policy around - and the value someone later copies
onto the hardware.

Degrees on the command line, in ArmSpec order:

    shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_yaw wrist_roll

checked by `spec.check_home_pose` (joint count, and >=5 deg clear of every hard
stop) before anything is built.
"""

import argparse
import logging
import sys

import numpy as np

from .server import ARM_PORT, GRIPPER_PORT, SimulationServer


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home_joints", type=float, nargs=6, required=True,
                   metavar="DEG",
                   help="REQUIRED. Six angles in degrees, in ArmSpec order.")
    p.add_argument("--scene", type=str, default=None,
                   help="scene XML; defaults to scenes/table.xml")
    p.add_argument("--gripper_port", type=int, default=GRIPPER_PORT)
    p.add_argument("--arm_port", type=int, default=ARM_PORT)
    p.add_argument("--headless", action="store_true", help="run without a viewer")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    from rebot_gripette.spec import check_home_pose

    home_rad = np.deg2rad(args.home_joints)
    try:
        check_home_pose(home_rad)
    except ValueError as e:
        print(f"--home_joints rejected: {e}")
        sys.exit(2)

    server = SimulationServer(home_joints_rad=home_rad, scene_xml=args.scene)
    server.run(gripper_port=args.gripper_port, arm_port=args.arm_port,
               headless=args.headless)


if __name__ == "__main__":
    main()
