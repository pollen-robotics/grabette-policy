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

# --------------------------------------------------------------------------- #
# Poses. Two DIFFERENT things, routinely confused:
#
#   STARTING POSE  the robot's physical initial state — where it actually is
#                  when you power it on. On this arm that is resting on its
#                  hard mechanical stops. It is an observation, not a choice,
#                  and nothing here commands it.
#
#   HOME POSE      the startup position FOR AN EXPERIMENT. `Reset` ramps the
#                  arm to it, so it is the first commanded motion of a run.
#                  It is a property of the experiment, not of the robot: on
#                  the OpenArm it is literally the simulator's START_JOINTS,
#                  i.e. the pose its dataset was recorded from and its policy
#                  expects to begin at.
#
# There is NO reBot home pose yet, because there is no reBot experiment yet —
# the sim scene that would define it is still to come. A value invented from
# "somewhere inside the joint limits" would be a guess that Reset then drives
# the arm into, and a guess that looks authoritative because it is a named
# constant. So there is deliberately no default here.
#
# When the reBot sim scene exists, this must be set to ITS start joints, the
# same way the OpenArm's is, and the two must be kept equal.
#
# Callers pass a home pose explicitly (see grpc_server_real.py --home_joints).
# `check_home_pose` below enforces the constraints that are knowable from the
# model alone; it cannot tell you the pose is right for the experiment.
HOME_JOINTS_RAD = None

# Minimum distance a commanded home pose must keep from any joint limit.
HOME_LIMIT_MARGIN_DEG = 5.0


def urdf_limits():
    """{joint: (lower_deg, upper_deg)} read from the exported model's URDF.

    The single source for this arm's travel. It is read from the XML every time
    rather than transcribed into a table, because a transcribed table is a
    second copy that silently stops matching the model it claims to describe -
    which is exactly what calibration.URDF_LIMITS did: it was wrong on five of
    six joints, and wrong in the direction that keeps its warning quiet.

    Read straight from the XML rather than through placo, so callers that want
    limits do not have to load a kinematics solver.
    """
    import math
    import xml.etree.ElementTree as ET

    from rebot_gripette_model import REBOT_B601RS_URDF

    root = ET.parse(REBOT_B601RS_URDF).getroot()
    out = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (math.degrees(float(lim.get("lower"))),
                                  math.degrees(float(lim.get("upper"))))
    return out


def check_home_pose(joints_rad, margin_deg: float = HOME_LIMIT_MARGIN_DEG):
    """Raise unless `joints_rad` is a usable home pose for this arm.

    NECESSARY conditions only. Passing means the pose is reachable and not
    jammed against a stop; it does NOT mean the pose is the right place to
    start your experiment, and it knows nothing about the table, the pedestal,
    the Gripette's envelope or self-collision.

    Checks:
      * the right number of joints,
      * every joint at least `margin_deg` inside its URDF limit — zero is the
        LOWER limit for both shoulder_lift and elbow_flex, so an all-zeros
        pose is literally on two hard stops and is rejected here.
    """
    q = np.asarray(joints_rad, dtype=float)
    if q.shape != (len(ARM_JOINT_NAMES),):
        raise ValueError(
            f"home pose has {q.size} joints, expected {len(ARM_JOINT_NAMES)} "
            f"({', '.join(ARM_JOINT_NAMES)})")

    limits = urdf_limits()

    problems = []
    for name, deg in zip(ARM_JOINT_NAMES, np.rad2deg(q)):
        lo, hi = limits[name]
        margin = min(deg - lo, hi - deg)
        if margin < margin_deg:
            problems.append(
                f"{name}={deg:.1f} deg is {margin:.1f} deg from its limit "
                f"[{lo:.0f}, {hi:.0f}]")
    if problems:
        raise ValueError(
            "home pose is too close to a hard stop: " + "; ".join(problems))
    return q
