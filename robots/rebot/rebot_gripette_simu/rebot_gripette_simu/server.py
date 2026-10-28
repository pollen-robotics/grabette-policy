"""The reBot simulation server: physics, camera, and the two gRPC services.

Wiring, mostly. The services themselves are shared:

    GripperService  gripette_sim.gripper_service   - same gripper, every arm
    ArmService      gripette_core.arm_service      - same service the REAL
                                                     reBot serves

The second one is the load-bearing claim of this package. `grpc_server_real.py`
builds an `ArmServicer` over a CAN-backed `ArmInterface`; this builds the same
class over a MuJoCo-backed one. The Cartesian integrator, the setpoint
interpolator, the IK-jump watchdog and the reset ramp are therefore not
"equivalent" between sim and hardware - they are the same object. A policy that
works here and fails on the arm fails for reasons about the robot.

THREADING
    Physics, camera rendering and viewer sync all run on the MAIN thread; the
    gRPC servers run in background threads and read a cached camera frame.
    Rendering off the main thread fights the viewer for the GL context.
"""

import logging
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import mujoco
import mujoco.viewer
import numpy as np
from gripette_core.arm_service import ArmServicer
from gripette_core.kinematics import CONTROL_FRAME, Kinematics
from gripette_core.proto import arm_pb2_grpc, gripper_pb2_grpc
from gripette_sim.clearance import GeomClearance
from gripette_sim.gripper_service import GripperServicer
from rebot_gripette.spec import REBOT_SPEC

from rebot_gripette.obstacles import (GUARDED_BODIES, OBSTACLE_GEOMS,
                                      TABLE_MARGIN)
from .sim_arm_interface import SimArmInterface
from .simulation import Simulation

logger = logging.getLogger(__name__)

GRIPPER_PORT = 50051
ARM_PORT = 50052
VIEWER_FPS = 60
CAMERA_FPS = 30                 # matches the real Grabette stream
CAMERA_OUT_SIZE = (960, 720)    # (width, height), as streamed to a policy

# The scene shipped with this package. Not the model's bare scene.xml: that one
# has no table and nothing to reach for.
DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "scenes" / "table.xml"


class SimulationServer:
    """MuJoCo simulation of the reBot, with the Gripette and arm gRPC services."""

    def __init__(self, home_joints_rad, scene_xml=None, orientation_weight=10.0):
        self._sim = Simulation(scene_xml or DEFAULT_SCENE)
        self._kin = Kinematics(REBOT_SPEC, orientation_weight=orientation_weight)
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._home_joints = np.asarray(home_joints_rad, dtype=float)
        self._running = False

        self._arm_iface = SimArmInterface(self._sim, list(REBOT_SPEC.joint_names))

        # Start AT the home pose. The real arm has to ramp to it from wherever
        # it powered on; the sim can simply be placed there, and should be, so
        # the first observation a policy sees is the one it was trained from.
        self._sim.reset_arm(self._home_joints)
        logger.info("Arm initialized to home %s deg",
                    np.round(np.rad2deg(self._home_joints), 1).tolist())

        # Gripper OPEN. Without this the position actuator's default ctrl plus
        # gravity drag the fingers to the closed bound, and the sim boots fully
        # CLOSED - out of distribution for any policy whose demos all start
        # open, which makes it predict lift/hold instead of approach/close.
        self._sim.set_joint_commands(np.zeros(2), joint_names=["proximal", "distal"])
        self._sim.reset_joints(np.zeros(2), joint_names=["proximal", "distal"])
        logger.info("Gripper initialized to OPEN (proximal=0, distal=0)")

        # Clearance gate against the bench. The IK-jump watchdog catches the
        # solver flipping branch; this catches a solution that is perfectly
        # smooth and goes straight into the table.
        self._clearance = GeomClearance(
            self._sim.model, OBSTACLE_GEOMS, GUARDED_BODIES,
            list(REBOT_SPEC.joint_names), margin=TABLE_MARGIN)

        self._camera_frame = self._sim.render_camera(out_size=CAMERA_OUT_SIZE)

    def get_camera_frame(self):
        """The most recent rendered frame. Read by the gripper service thread."""
        return self._camera_frame

    def _start_grpc(self, gripper_port, arm_port):
        self._gripper_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        gripper_pb2_grpc.add_GripperServiceServicer_to_server(
            GripperServicer(self._sim, self, self._lock, self._start_time),
            self._gripper_server)
        self._gripper_server.add_insecure_port(f"[::]:{gripper_port}")
        self._gripper_server.start()
        logger.info("GripperService on :%d", gripper_port)

        self._arm_servicer = ArmServicer(
            arm=self._arm_iface,
            kin=self._kin,
            home_joints=self._home_joints,
            start_time=self._start_time,
            # `current_joints` lets the gate allow moves that RETREAT from the
            # table even when they start inside the margin; without it an arm
            # that got in there would have no command left that could get out.
            joint_gate=self._clearance.as_joint_gate(
                current_joints=self._arm_iface.get_positions),
        )
        self._arm_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        arm_pb2_grpc.add_ArmServiceServicer_to_server(
            self._arm_servicer, self._arm_server)
        self._arm_server.add_insecure_port(f"[::]:{arm_port}")
        self._arm_server.start()
        logger.info("ArmService on :%d (control frame %s)", arm_port, CONTROL_FRAME)

    def run(self, gripper_port=GRIPPER_PORT, arm_port=ARM_PORT, headless=False):
        """Run the simulation. Blocking; call from the main thread."""
        self._start_grpc(gripper_port, arm_port)

        dt = self._sim.model.opt.timestep
        viewer_interval = max(1, int(1.0 / (VIEWER_FPS * dt)))
        camera_interval = max(1, int(1.0 / (CAMERA_FPS * dt)))

        viewer = None
        if not headless:
            viewer = mujoco.viewer.launch_passive(
                self._sim.model, self._sim.data,
                show_left_ui=False, show_right_ui=False)

        logger.info("Physics loop running. Ctrl+C to stop.")
        self._running = True
        t_wall = time.perf_counter()
        step = 0
        try:
            while self._running and (viewer is None or viewer.is_running()):
                with self._lock:
                    self._sim.step()
                    if step % camera_interval == 0:
                        self._camera_frame = self._sim.render_camera(
                            out_size=CAMERA_OUT_SIZE)
                step += 1

                if viewer is not None and step % viewer_interval == 0:
                    viewer.sync()

                t_target = t_wall + step * dt
                t_now = time.perf_counter()
                if t_target > t_now:
                    time.sleep(t_target - t_now)
        except KeyboardInterrupt:
            pass
        finally:
            # No descent on the way out, unlike the real server: nothing here
            # can fall on anything. The arm is left where it is.
            self._running = False
            self._arm_servicer.stop()
            self._gripper_server.stop(grace=2)
            self._arm_server.stop(grace=2)
            if viewer is not None:
                viewer.close()
            logger.info("Server stopped.")
