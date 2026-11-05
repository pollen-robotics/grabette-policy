# ruff: noqa: N802  # gRPC servicer methods use PascalCase (protobuf convention)
"""gRPC ArmService server for a real OpenArm (7-DOF, no gripper on CAN).

This server exposes the same ArmService API as `openarm_gripette_simu`, but drives
a real Pollen OpenArm via CAN. It uses the Placo `Kinematics` class and URDF from
`openarm_gripette_model` for bit-for-bit FK/IK compatibility with the simulator.

The **Gripette has its own gRPC service** (same API as `gripper.proto`) running
independently. The eval client connects to two separate endpoints:

  - ArmService     → this server (arm control via CAN)
  - GripperService → the Gripette's own service (camera + gripper motors)

So this script implements ONLY the ArmService.

Behavioral notes:
  - SendCartesianDelta: accumulates on an internal target, runs IK, sends to CAN.
    Identical semantics to the simulator.
  - Reset: interpolates smoothly to the home pose (can't teleport a real arm).
    Cube randomization is a no-op — returns dummy cube coords for API compat.
  - GetSuccessStatus: always returns goal_reached=False (no cube tracking here).

Prerequisites: CAN bus setup + firmware-zero calibration — see README.md.

Usage (on the CAN-connected machine):
  uv run python -m openarm_gripette.grpc_server_real \\
      --can_port can0 --side right --arm_port 50052

Then, from the inference machine, point the eval / teleop client at
`--arm_addr <robot-ip>:50052` (see policies/DiffusionPolicy/README.md).
"""

import argparse
import logging
import threading
import time
from concurrent import futures

import grpc
import numpy as np

# The ArmService implementation is shared with every other Gripette-equipped
# arm, so sim and real and OpenArm and reBot cannot drift apart behaviourally.
# This module supplies only the OpenArm-specific half: the ArmInterface over
# LeRobot/Damiao, the home pose, and the CLI.
from gripette_core.arm_service import ArmServicer

# Kinematics + proto stubs from the simulator package (ensures bit-for-bit match)
from openarm_gripette_simu import Kinematics
from openarm_gripette_simu.kinematics import ARM_JOINT_NAMES as KIN_ARM_JOINT_NAMES
from openarm_gripette_simu.proto import arm_pb2_grpc

from openarm_gripette import OpenArm7Follower, OpenArm7FollowerConfig

logger = logging.getLogger(__name__)

# Default safe home pose (matches simulator START_JOINTS).
HOME_JOINTS_RAD = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])

# Reset interpolation: move from current to home over this many seconds.
RESET_DURATION_S = 3.0
RESET_HZ = 50

# LeRobot's OpenArm driver uses degrees; simulator/policy use radians.
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Arm wrapper: unit conversion + joint-name mapping
# ---------------------------------------------------------------------------


class ArmInterface:
    """OpenArm's implementation of `gripette_core.arm_service.ArmInterface`.

    Adapter between the simulator's joint API (radians, r_arm_* names) and
    LeRobot's OpenArm7Follower (degrees, joint_1..joint_7 names).

    Caches the most recent joint read with a short TTL so that GetArmState and
    SendCartesianDelta called within the same control cycle share a single CAN
    refresh. Without this, every inference step triggers two full refreshes
    (one for client-side state, one for IK), doubling bus load and packet drops.
    """

    def __init__(
        self,
        robot: OpenArm7Follower,
        arm_joint_map: dict[str, str],
        state_cache_ttl_s: float = 0.02,
    ):
        self._robot = robot
        self._arm_joint_map = arm_joint_map  # sim_name -> lerobot_name
        self._lerobot_arm_names = [arm_joint_map[n] for n in KIN_ARM_JOINT_NAMES]
        self._lock = threading.Lock()
        self._state_cache_ttl = state_cache_ttl_s
        self._cached_positions_rad: np.ndarray | None = None
        self._cached_positions_ts: float = 0.0

    def get_positions(self) -> np.ndarray:
        """Read arm joint positions in radians, in simulator order (r_arm_pitch, ...).

        Returns a cached value if the last refresh is within state_cache_ttl_s.
        """
        now = time.monotonic()
        if (
            self._cached_positions_rad is not None
            and (now - self._cached_positions_ts) < self._state_cache_ttl
        ):
            return self._cached_positions_rad.copy()
        with self._lock:
            # Double-checked: another thread may have refreshed while we waited.
            now = time.monotonic()
            if (
                self._cached_positions_rad is not None
                and (now - self._cached_positions_ts) < self._state_cache_ttl
            ):
                return self._cached_positions_rad.copy()
            obs = self._robot.get_observation()
            positions_deg = np.array(
                [obs[f"{name}.pos"] for name in self._lerobot_arm_names], dtype=np.float64
            )
            self._cached_positions_rad = positions_deg * DEG_TO_RAD
            self._cached_positions_ts = time.monotonic()
            return self._cached_positions_rad.copy()

    def send_command_rad(self, joint_angles_rad: np.ndarray):
        """Send arm joint commands (radians, in simulator order)."""
        joint_angles_deg = joint_angles_rad * RAD_TO_DEG
        action = {
            f"{self._lerobot_arm_names[i]}.pos": float(joint_angles_deg[i])
            for i in range(len(joint_angles_rad))
        }
        with self._lock:
            self._robot.send_action(action)

    def set_torque(self, enable: bool):
        """Enable/disable torque on all arm motors (Damiao enable/disable).

        Disable = motors freewheel; the arm falls under gravity.

        The Damiao enable/disable frames are fire-and-forget in LeRobot: a
        motor that misses the frame stays in its previous state and the driver
        only logs it at DEBUG level (observed in practice: joint_7 kept torque
        after a server shutdown). Send TWO full passes with a pause so a
        single lost frame can't silently leave a motor powered.
        """
        with self._lock:
            for _ in range(2):
                if enable:
                    self._robot.bus.enable_torque(num_retry=2)
                else:
                    self._robot.bus.disable_torque(num_retry=2)
                time.sleep(0.05)


# ---------------------------------------------------------------------------
# CLI + server setup
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="gRPC ArmService for real OpenArm (7-DOF, no gripper)")
    p.add_argument("--can_port", type=str, default="can0", help="CAN interface name")
    p.add_argument("--side", type=str, default="right", choices=["left", "right"])
    p.add_argument(
        "--ik_orientation_weight",
        type=float,
        default=10.0,
        help="Placo frame-task orientation weight. Default 10.0 on real (sim "
        "uses 1.0). 100:1 position-priority leaks rotation into wrist_yaw "
        "when translating image-right or image-down at the standard home pose "
        "— observable in cartesian_square as a visible camera yaw on edges "
        "perpendicular to the optical axis. Bumping to 10 keeps orientation "
        "held at the cost of <1 mm position error per step. Try 50+ only if "
        "10 still shows yaw drift; trades more position accuracy.",
    )
    p.add_argument(
        "--ik_position_weight",
        type=float,
        default=100.0,
        help="Placo frame-task position weight. Default 100.0 (same as sim).",
    )
    p.add_argument(
        "--max_relative_target",
        type=float,
        default=8.0,
        help="Max per-step joint motion in degrees (safety limit). Each command "
        "then does an extra CAN sync_read to check current position — costs one "
        "full bus transaction. Pass a large value (e.g. 180) to effectively "
        "disable the clamp while still paying the read cost, or remove the "
        "`max_relative_target` kwarg in the config to skip it entirely.",
    )
    p.add_argument("--arm_port", type=int, default=50052, help="gRPC listen port")
    p.add_argument(
        "--kp_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to all MIT position_kp values at startup. "
        "Useful for taming a stiff controller without editing the config: "
        "0.5 halves all kp (softer tracking, smoother under sparse commands), "
        "2.0 doubles them (stiffer).",
    )
    p.add_argument(
        "--kd_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to all MIT position_kd values at startup. "
        "Typically scale kd ~ sqrt(kp_scale) to preserve damping ratio; in "
        "practice leave at 1.0 first and tune from there.",
    )
    p.add_argument(
        "--interp_hz",
        type=float,
        default=50.0,
        help="Joint-space setpoint interpolator rate (Hz). The interpolator "
        "thread sends MIT commands at this rate, filling in the time between "
        "sparse Cartesian delta RPCs with a smooth joint trajectory.",
    )
    p.add_argument(
        "--interp_alpha",
        type=float,
        default=0.3,
        help="Exponential approach rate per interp tick: next = cur + alpha * (target - cur). "
        "Smaller = smoother + more lag (e-folding time = 1 / (alpha * interp_hz)). "
        "Typical range 0.1 (heavy smoothing, ~100ms lag at 50Hz) to 0.5 (light smoothing, ~40ms lag).",
    )
    p.add_argument(
        "--max_ik_jump_deg",
        type=float,
        default=15.0,
        help="IK-jump watchdog: max per-joint change between two consecutive "
        "Placo IK solutions, in degrees. If any joint exceeds this in one step, "
        "the Cartesian delta is REJECTED (integrator rolled back). This is the "
        "specific guard against singularity-driven branch flips that the OOD "
        "Cartesian watchdog and per-step motor clamp can't catch. "
        "Set <= 0 to disable. Typical: 10–20°. Default: 15.",
    )
    p.add_argument(
        "--max_ik_jump_violations",
        type=int,
        default=2,
        help="After N consecutive IK-jump rejections, disable the interpolator "
        "and re-sync the integrator to the current FK pose. Forces the operator "
        "to home before further motion. Default: 2.",
    )
    p.add_argument(
        "--max_target_lead_mm",
        type=float,
        default=80.0,
        help="Contact guard (TRIP-ONLY): reject any delta that would put the "
        "integrator TARGET more than this far (mm) ahead of the MEASURED pose "
        "AND increase the lead. Bounds press force under contact without ever "
        "modifying the reference (a continuous pull-back corrupts the "
        "trajectory). Lead-reducing deltas (retreats) always pass. Costs one "
        "extra bus read per Cartesian command. <= 0 disables. Default: 80 "
        "(above normal full-speed tracking lag).",
    )
    p.add_argument(
        "--arm_joint_map",
        type=str,
        nargs="+",
        default=[
            "r_arm_pitch=joint_1",
            "r_arm_roll=joint_2",
            "r_arm_yaw=joint_3",
            "r_elbow=joint_4",
            "r_wrist_yaw=joint_5",
            "r_wrist_roll=joint_6",
            "r_wrist_pitch=joint_7",
        ],
        help="Map simulator joint names to LeRobot motor names (format: sim_name=lerobot_name)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    arm_joint_map = dict(item.split("=") for item in args.arm_joint_map)
    logger.info(f"Arm joint map (sim → lerobot): {arm_joint_map}")

    for sim_name in KIN_ARM_JOINT_NAMES:
        if sim_name not in arm_joint_map:
            raise ValueError(f"Missing mapping for simulator joint '{sim_name}' in --arm_joint_map")

    # ---- Robot setup (arm only, no gripper on CAN) ----
    logger.info(f"Connecting to OpenArm on {args.can_port}, side={args.side}")
    robot_config = OpenArm7FollowerConfig(
        port=args.can_port,
        side=args.side,
        can_interface="socketcan",
        max_relative_target=args.max_relative_target,
        cameras={},  # no cameras on this driver — camera comes via Gripette's gRPC
    )
    if args.kp_scale != 1.0:
        robot_config.position_kp = [v * args.kp_scale for v in robot_config.position_kp]
        logger.info(
            f"Scaled MIT position_kp by {args.kp_scale}: {[round(v, 2) for v in robot_config.position_kp]}"
        )
    if args.kd_scale != 1.0:
        robot_config.position_kd = [v * args.kd_scale for v in robot_config.position_kd]
        logger.info(
            f"Scaled MIT position_kd by {args.kd_scale}: {[round(v, 2) for v in robot_config.position_kd]}"
        )
    robot = OpenArm7Follower(robot_config)
    # calibrate=False: the server never modifies calibration. If the firmware
    # zeros or the calibration file need updating, run `lerobot-calibrate`
    # explicitly before starting the server.
    robot.connect(calibrate=False)
    logger.info("Robot connected (7-DOF arm, gripper is external)")

    arm_iface = ArmInterface(robot, arm_joint_map)

    # ---- Kinematics (same URDF as simulator, but with stricter orientation
    #      weight on real to keep the wrist locked through optical-axis
    #      translation; see --ik_orientation_weight in CLI help) ----
    logger.info(
        f"Loading kinematics (placo + URDF from openarm_gripette_model), "
        f"frame-task weights pos={args.ik_position_weight} orient={args.ik_orientation_weight}"
    )
    kin = Kinematics(
        position_weight=args.ik_position_weight,
        orientation_weight=args.ik_orientation_weight,
    )
    logger.info("Kinematics loaded")

    # ---- gRPC server ----
    # SINGLE-INSTANCE GUARD. gRPC enables SO_REUSEPORT by default on Linux, so a
    # SECOND server binds the same port WITHOUT any error — and two instances of
    # this process each run their own interpolator/integrator while writing to
    # the same CAN bus and motors. They fight, and the arm shakes violently
    # (observed on the real robot 2026-07-28: two servers on :50052, 17:02-17:04).
    # Disabling so_reuseport makes the second bind FAIL, and we check the return
    # value (add_insecure_port returns 0 on failure) instead of ignoring it.
    start_time = time.monotonic()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                         options=[("grpc.so_reuseport", 0)])
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
    arm_pb2_grpc.add_ArmServiceServicer_to_server(servicer, server)
    # grpcio signals a failed bind either by returning 0 (older) or by raising
    # RuntimeError (current) — handle both so the refusal is always explained.
    try:
        _bound = server.add_insecure_port(f"[::]:{args.arm_port}")
    except RuntimeError:
        _bound = 0
    if _bound == 0:
        raise SystemExit(
            f"REFUSING TO START: port {args.arm_port} is already in use — another "
            f"arm server is almost certainly running and driving the motors.\n"
            f"Two servers on one arm each hold their own target and fight, which "
            f"shakes the arm violently.\n"
            f"Check with:  ss -tlnp | grep {args.arm_port}   and   "
            f"pgrep -af grpc_server_real\n"
            f"Stop the existing one before starting this."
        )
    server.start()
    logger.info(f"ArmService listening on port {args.arm_port}")
    logger.info("Press Ctrl+C to stop")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.stop(grace=2.0)
        servicer.stop()
        # Belt and braces: explicit double-pass torque disable BEFORE
        # disconnect. robot.disconnect() also disables, but with single
        # fire-and-forget frames per motor — one lost frame leaves that motor
        # powered and holding (seen on joint_7/wrist_pitch). The arm FALLS.
        try:
            arm_iface.set_torque(False)
            logger.info("Torque disabled on all motors (double-pass).")
        except Exception as e:
            logger.error(f"Explicit torque disable failed: {e} — check the arm, some motors may still be powered!")
        robot.disconnect()
        logger.info("Robot disconnected. Goodbye.")


if __name__ == "__main__":
    main()
