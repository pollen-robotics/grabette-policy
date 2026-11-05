"""OpenArm right arm kinematics — the OpenArm binding of `gripette_core`.

The solver itself now lives in `gripette_core.kinematics`, shared with the
other Gripette-equipped arms. This module supplies the OpenArm's `ArmSpec` and
keeps the historical import surface (`Kinematics`, `ARM_JOINT_NAMES`, the frame
constants) working unchanged, so no caller had to move.
"""

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

OPENARM_SPEC = ArmSpec(
    model_dir=str(OPENARM_RIGHT_DIR),
    joint_names=tuple(ARM_JOINT_NAMES),
    # Gripper fingers and the wrist-roll mimic: present in the URDF, but the
    # IK must not drive them or it would "reach" by closing the hand.
    masked_dofs=("proximal", "distal", "r_wrist_roll_mimic"),
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
