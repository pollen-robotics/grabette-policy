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
from gripette_sim.lift_tracker import LiftTracker
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

# The body an episode is scored on. Named `red_cube` in both arms' grasp
# scenes - here it is a drink can - because that is the name the dataset and
# the OpenArm's servicer already use, and renaming it would only make the two
# arms' scenes disagree for no gain.
PICK_BODY = "red_cube"

# The scene shipped with this package. Not the model's bare scene.xml: that one
# has no table and nothing to reach for.
DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "scenes" / "table.xml"


class SimulationServer:
    """MuJoCo simulation of the reBot, with the Gripette and arm gRPC services."""

    def __init__(self, home_joints_rad, scene_xml=None, orientation_weight=10.0,
                 clearance_margin_m=TABLE_MARGIN, clearance_gate=True):
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
        #
        # The margin is settable because the default is a REACHING figure, and
        # a grasp is not a reach: picking an object off the bench puts the
        # fingertips within millimetres of the surface by definition. At the
        # 20 mm default a grasp policy had 176 of 200 commands vetoed, the
        # closest at -6.7 mm - a genuine penetration, not a misreading. Lower
        # it for a pick task, with the obvious cost: the gate is what stops a
        # bad solution driving into the table, so a smaller number buys the
        # policy room and spends the guard's authority. Same knob the real
        # server exposes as --clearance_margin_mm.
        #
        # THE MARGIN IS ALSO THE QUERY'S CUTOFF, AND THAT MATTERS. `clearance`
        # passes it to mj_geomDistance as distmax, so anything further away
        # costs nothing to evaluate and comes back clamped. Keep it that way.
        # MuJoCo 3.9's convex-distance solver returns a hard 0.0 for some
        # mesh/box pairs when it is asked for a LONG separation - reproducible
        # in a model holding nothing but one mesh and one box 375 mm apart,
        # where it reports 0.0 while filling a 455 mm witness segment, and
        # unaffected by ccd_iterations, ccd_tolerance or the collision mask.
        # Asked with distmax <= the margin it never enters that path: checked
        # over 3000 random configurations against a closed-form hull-vertex-
        # to-box reference, this gate gave 0 false passes and 1 false block,
        # agreeing to a median of 0.02 mm. Raise distmax above the margin and
        # that guarantee is gone.
        # THE GATE CAN BE TURNED OFF HERE, AND THE REAL SERVER GETS NO SUCH
        # FLAG. Not because the query is wrong - it is sound (see above) - but
        # because this gate cannot express the task. GUARDED_BODIES is
        # all-or-nothing per body under one global margin, and a pick says two
        # different things at once:
        #
        #     the FINGERTIPS are supposed to reach the bench;
        #     the ARM LINKS are not.
        #
        # Measured, driving test_pick_can_200_smooth_trim-best at the eval
        # start pose. The start itself is clear - no guarded geom within 20 mm
        # - but descending ~8 deg to close on the can puts gripette_r_distal
        # 5.8 mm under the bench top, which is the grasp, not a fault:
        #
        #     margin   0 mm : 236 of 250 commands refused, 0.0 mm displacement
        #     margin -15 mm : 238 of 250 refused; the policy then wants -17 to
        #                     -21 mm, so the number was never the problem
        #
        # The stall is what makes it permanent. The arm is caught in the first
        # three steps by the target-lead cap, and a frozen arm returns a frozen
        # observation, so the policy re-predicts the same descent forever. With
        # the gate off the same lead-cap rejections happen, the arm moves
        # anyway, the view changes, and it recovers to a lift.
        #
        # The fix is a margin PER GUARDED BODY - fingers near 0, links at
        # 20 mm - in obstacles.py and clearance.py, which are shared with the
        # real server. Until then: hardware keeps the gate unconditionally,
        # because over-blocking is the safe way to be wrong on a real arm, and
        # sim gets a switch, because there is nothing here to protect and an
        # experiment that cannot move is worth less than one with no guard.
        self._clearance = GeomClearance(
            self._sim.model, OBSTACLE_GEOMS, GUARDED_BODIES,
            list(REBOT_SPEC.joint_names), margin=clearance_margin_m)
        self._clearance_gate = clearance_gate
        if not clearance_gate:
            logger.warning("CLEARANCE GATE DISABLED - nothing stops a command "
                           "driving a link into the bench. Simulation only.")

        # Eyes on the object, so an episode can be SCORED. Without this the
        # shared ArmServicer gives the real arm's honest "I cannot see a cube"
        # and every simulated episode reports 0/1 whatever happened - which is
        # how a run that closed the gripper on empty air was once read as a
        # lift. Disabled by itself in a scene with nothing to pick up.
        self._lift = LiftTracker(self._sim, PICK_BODY, lock=self._lock)

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
            joint_gate=(self._clearance.as_joint_gate(
                current_joints=self._arm_iface.get_positions)
                if self._clearance_gate else None),
            object_tracker=self._lift if self._lift.enabled else None,
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
