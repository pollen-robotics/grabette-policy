"""The reBot's ArmSpec, joint order and home pose.

Separate from follower.py and arm_interface.py on purpose: this is pure data
plus a path into the model package, and it must stay importable WITHOUT the
`lerobot-robot-seeed-b601` plugin or any CAN library. The sim, the IK
feasibility filter and the model tests all want these numbers on machines that
will never touch the arm.

Needs placo only through `gripette_core`, which the model path is fed to.
"""

import numpy as np
from gripette_core.kinematics import ArmSpec
from rebot_gripette_model import REBOT_B601RS_DIR

# The six arm joints in URDF order. Verified against the real arm with a live
# mirror (read the motors, drive the model, watch for a joint going the wrong
# way): URDF space == motor space, so a command moves model and robot the same
# way by the same amount. Two joints were inverted in an earlier export and
# fixed in CAD, so this is a property to re-check after any re-export rather
# than an invariant of the exporter.
ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)

REBOT_SPEC = ArmSpec(
    model_dir=str(REBOT_B601RS_DIR),
    joint_names=ARM_JOINT_NAMES,
    # The Gripette's two finger joints are in the URDF. Masking them stops the
    # IK "reaching" a target by closing the hand.
    masked_dofs=("proximal", "distal"),
)

# SAFE HOME — and emphatically NOT all zeros.
#
# shoulder_lift is 0..170 deg and elbow_flex 0..200 deg: both START at zero.
# The OpenArm homes at [0,0,0,1.57,0,0,0], and the naive translation of that
# to this arm would sit exactly on two hard stops — every Reset would drive
# the arm into them, and Placo's joint-limit constraint would have no room on
# one side of either joint. tests/test_spec_and_home.py pins this down.
HOME_JOINTS_RAD = np.deg2rad([0.0, 60.0, 60.0, 0.0, 0.0, 0.0])
