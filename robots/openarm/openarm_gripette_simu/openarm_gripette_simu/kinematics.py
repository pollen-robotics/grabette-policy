"""Placo-based kinematics for the OpenArm right arm.

Provides forward and inverse kinematics targeting the 'camera' frame.
Only computes desired joint positions — no physics, no simulation.
"""

import numpy as np
import placo
from openarm_gripette_model import OPENARM_RIGHT_DIR

# The 7 arm joints in MuJoCo ordering (excludes gripper and mimic joints)
ARM_JOINT_NAMES = [
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_elbow",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]

# Passive joint coupled to r_wrist_roll: mimic = MIMIC_RATIO * roll (URDF <mimic>,
# MuJoCo <equality>). Not an arm joint: never exposed in the 7-joint vectors.
ROLL_JOINT = "r_wrist_roll"
MIMIC_JOINT = "r_wrist_roll_mimic"
MIMIC_RATIO = -1.0

# Hard roll limit used by IK (rad). The real driver clips joint_6 (= r_wrist_roll)
# to +/-40 deg (RIGHT_DEFAULT_JOINTS_LIMITS in openarm_gripette's
# config_openarm7_follower.py), tighter than the URDF's +/-44.9 deg. IK must stay
# inside the DRIVER range, else the driver silently clips the command and the arm
# is not where IK thinks it is. 3 deg margin so IK never parks on the stop.
ROLL_IK_LIMIT = np.radians(40.0 - 3.0)

# Frame names
CAMERA_FRAME = "camera"
GRIPPER_FRAME = "gripper_center"
OAKL_FRAME = "oak_l"   # SLAM / control frame the grasp trajectory commands

# THE control frame: the single frame the whole pipeline (recorded data action,
# IK feasibility filter, arm Cartesian-delta control, server reset, replay) must
# agree on. The policy commands this frame; IK targets it. Change it HERE only.
# Must be oak_l for the current free-floating dataset (the scene is rooted at
# oak_l, so the recorded mocap pose IS the oak_l pose — see collect_grasp_dataset
# and the assert there). Switching to CAMERA_FRAME additionally requires the data
# recording + IK targets to apply the oak_l->camera transform.
CONTROL_FRAME = OAKL_FRAME

# Note on the arm-eval "singularity explosion": solver-level mitigations (extra dq
# damping, a posture task, velocity limits) were investigated and did NOT cleanly
# help — the explosion is an IK branch-flip when the integrator target marches past
# the reachable workspace, which solver-cost tuning can't fix at acceptable
# accuracy. It's handled target-side by the eval gate's singularity guard
# (arm_servicer), not in the solver.


class Kinematics:
    """Placo wrapper for FK/IK on the OpenArm right arm."""

    def __init__(self, model_dir=None, position_weight: float = 100.0,
                 orientation_weight: float = 1.0):
        """
        Args:
            model_dir: URDF dir (defaults to OPENARM_RIGHT_DIR).
            position_weight, orientation_weight: Placo frame-task weights
                (default 100:1). The eval server uses orientation_weight=10 to
                lock rotation more strictly near the home wrist-roll singularity:
                the ~16cm oak_l→finger lever amplifies oak_l rotation error into a
                grasp-point miss. The default 1 keeps position-only priority.
        """
        model_dir = str(model_dir) if model_dir else str(OPENARM_RIGHT_DIR)
        self.robot = placo.RobotWrapper(model_dir)

        # Hard roll limit = driver range minus margin (see ROLL_IK_LIMIT).
        self.robot.set_joint_limits(ROLL_JOINT, -ROLL_IK_LIMIT, ROLL_IK_LIMIT)

        # Set up the IK solver
        self.solver = self.robot.make_solver()
        self.solver.mask_fbase(True)  # base is fixed
        self.solver.enable_joint_limits(True)
        self.solver.dt = 0.01

        # Mask the gripper joints so IK only moves the arm.
        self.solver.mask_dof("proximal")
        self.solver.mask_dof("distal")

        # Wrist roll coupling. In the URDF, r_wrist_roll's own child is a side
        # branch; the joint that actually carries the wrist pitch and the whole
        # gripper/camera chain is r_wrist_roll_mimic (<mimic multiplier="-1"/>,
        # which placo does not apply by itself). MuJoCo ties them with an
        # equality constraint, mimic = -roll. Placo has to model the same
        # coupling, else FK is blind to r_wrist_roll and IK never uses it:
        #   - forward()/inverse() set mimic = -roll whenever they set the joints
        #     (_set_arm_joints);
        #   - IK keeps it via a hard gear task, dq_mimic = -1 * dq_roll, so the
        #     solver moves both together as one DoF.
        self._gear_task = self.solver.add_gear_task()
        self._gear_task.set_gear(MIMIC_JOINT, ROLL_JOINT, MIMIC_RATIO)
        self._gear_task.configure("wrist_roll_mimic", "hard", 1.0)

        # Small regularization keeps the QP full-rank (min-velocity solution).
        self.solver.add_regularization_task(1e-4)

        # Frame task on camera, with position weighted 100x higher than
        # orientation. Equal weighting (the previous default) caused the QP
        # to compromise — landing 30-200 mm off the position to better
        # match an orientation we may not actually reach. With position
        # priority, the solver nails position first and accepts whatever
        # orientation is achievable. Placo's FrameTask.configure takes
        # (name, type, position_weight, orientation_weight).
        self.robot.update_kinematics()
        T_cam = self.robot.get_T_world_frame(CAMERA_FRAME)
        self._frame_task = self.solver.add_frame_task(CAMERA_FRAME, T_cam)
        self._frame_task.configure(CAMERA_FRAME, "soft", position_weight, orientation_weight)

        # Fixed offsets to the camera frame (the solver always tasks the camera),
        # so targets given in other frames can be converted to a camera target.
        T_grip = self.robot.get_T_world_frame(GRIPPER_FRAME)
        self._T_grip_to_cam = np.linalg.inv(T_grip) @ T_cam
        T_oakl = self.robot.get_T_world_frame(OAKL_FRAME)
        self._T_oakl_to_cam = np.linalg.inv(T_oakl) @ T_cam

    def forward(self, joint_positions: np.ndarray, frame: str = CONTROL_FRAME) -> np.ndarray:
        """Compute a frame's pose from arm joint positions.

        Args:
            joint_positions: 7-element array of arm joint angles (rad).
            frame: frame name to compute FK for (default: CONTROL_FRAME).

        Returns:
            4x4 homogeneous transform (world -> frame).
        """
        self._set_arm_joints(joint_positions)
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(frame).copy()

    def inverse(
        self,
        target_pose: np.ndarray,
        current_joint_positions: np.ndarray | None = None,
        n_iter: int = 500,
        frame: str = CONTROL_FRAME,
    ) -> np.ndarray:
        """Solve IK for a target pose of the given frame.

        Args:
            target_pose: 4x4 homogeneous transform (world -> frame).
            current_joint_positions: optional 7-element starting config.
                If None, uses the robot's current state.
            n_iter: number of solver iterations.
            frame: which frame to target ('camera' or 'gripper_center').
                   Gripper targets are converted to camera targets internally.

        Returns:
            7-element array of arm joint angles (rad).
        """
        if current_joint_positions is not None:
            self._set_arm_joints(current_joint_positions)
            self.robot.update_kinematics()

        # Convert a target given in another frame to the camera target the
        # solver tasks, using the fixed offset for that frame.
        if frame == GRIPPER_FRAME:
            cam_target = target_pose @ self._T_grip_to_cam
        elif frame == OAKL_FRAME:
            cam_target = target_pose @ self._T_oakl_to_cam
        else:
            cam_target = target_pose

        self._frame_task.T_world_frame = cam_target

        for _ in range(n_iter):
            self.solver.solve(True)
            self.robot.update_kinematics()

        return self.get_arm_joints()

    def _set_arm_joints(self, joint_positions) -> None:
        """Write the 7 arm joints (ARM_JOINT_NAMES order) and the coupled mimic."""
        for i, name in enumerate(ARM_JOINT_NAMES):
            self.robot.set_joint(name, joint_positions[i])
        self.robot.set_joint(MIMIC_JOINT, MIMIC_RATIO * self.robot.get_joint(ROLL_JOINT))

    def get_arm_joints(self) -> np.ndarray:
        """Read the current arm joint positions from the Placo model."""
        return np.array([self.robot.get_joint(name) for name in ARM_JOINT_NAMES])
