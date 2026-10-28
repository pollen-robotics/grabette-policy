"""The reBot's binding of the shared MuJoCo wrapper.

Two class attributes. Everything else - the XML surgery that makes MuJoCo
resolve the robot include's `meshdir`, the `gripette_cam` injected at the
calibrated camera site, the fisheye render - is `gripette_sim.simulation`,
shared with the OpenArm.

No mimic joints: the reBot's six joints are independent. The OpenArm's wrist
roll drives one, which is why that is a declared mapping upstream rather than
a hardcoded special case.
"""

from gripette_sim import ArmSimulation
from rebot_gripette.spec import ARM_JOINT_NAMES
from rebot_gripette_model import REBOT_B601RS_SCENE

# All actuated joints in MuJoCo ordering, arm first then the gripper's two.
ACTUATOR_NAMES = [*ARM_JOINT_NAMES, "proximal", "distal"]


class Simulation(ArmSimulation):
    """MuJoCo simulation of the reBot B601-RS + Gripette."""

    ARM_JOINT_NAMES = list(ARM_JOINT_NAMES)
    DEFAULT_SCENE = REBOT_B601RS_SCENE
