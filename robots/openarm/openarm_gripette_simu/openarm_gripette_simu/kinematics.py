"""OpenArm right arm kinematics — the OpenArm binding of `gripette_core`.

The solver itself now lives in `gripette_core.kinematics`, shared with the
other Gripette-equipped arms. This module supplies the OpenArm's `ArmSpec` and
keeps the historical import surface (`Kinematics`, `ARM_JOINT_NAMES`, the frame
constants) working unchanged, so no caller had to move.
"""

import numpy as np
from gripette_core.kinematics import (  # noqa: F401  (re-exported)
    CAMERA_FRAME,
    CONTROL_FRAME,
    GRIPPER_FRAME,
    OAKL_FRAME,
    ArmSpec,
)
from gripette_core.kinematics import Kinematics as _Kinematics
from openarm_gripette_model import OPENARM_RIGHT_DIR

# The 7 arm joints in MuJoCo ordering (excludes gripper and mimic joints).
# A list, not the spec's tuple: callers index and iterate it, and one of them
# (examples/joint_control.py) imports it directly.
ARM_JOINT_NAMES = [
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_elbow",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]

# Hard roll limit used by IK (rad). The real driver clips joint_6 (= r_wrist_roll)
# to +/-40 deg (RIGHT_DEFAULT_JOINTS_LIMITS in openarm_gripette's
# config_openarm7_follower.py), tighter than the URDF's +/-44.9 deg. IK must stay
# inside the DRIVER range, else the driver silently clips the command and the arm
# is not where IK thinks it is. 3 deg margin so IK never parks on the stop.
ROLL_IK_LIMIT = float(np.radians(40.0 - 3.0))

OPENARM_SPEC = ArmSpec(
    model_dir=str(OPENARM_RIGHT_DIR),
    joint_names=tuple(ARM_JOINT_NAMES),
    # Gripper fingers: present in the URDF, but the IK must not drive them or
    # it would "reach" by closing the hand.
    masked_dofs=("proximal", "distal"),
    # In the URDF, r_wrist_roll's own child is a side branch; the joint that
    # carries the wrist pitch and the whole gripper/camera chain is
    # r_wrist_roll_mimic (<mimic multiplier="-1"/>). MuJoCo ties them with an
    # equality, mimic = -roll. Masking the mimic instead (the old setup) made
    # FK blind to r_wrist_roll and IK never moved it (SIM-1).
    mimic_joints=(("r_wrist_roll_mimic", "r_wrist_roll", -1.0),),
    joint_limits=(("r_wrist_roll", -ROLL_IK_LIMIT, ROLL_IK_LIMIT),),
)


class Kinematics(_Kinematics):
    """`gripette_core.Kinematics` pre-bound to the OpenArm right arm.

    Signature is unchanged from before the core split, so `Kinematics()`,
    `Kinematics(orientation_weight=10.0)` and `Kinematics(model_dir=...)` all
    behave exactly as they did.
    """

    def __init__(self, model_dir=None, position_weight: float = 100.0,
                 orientation_weight: float = 1.0):
        super().__init__(
            OPENARM_SPEC,
            model_dir=model_dir,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )
