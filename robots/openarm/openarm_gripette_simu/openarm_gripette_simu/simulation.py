"""OpenArm's binding of the shared MuJoCo wrapper.

The wrapper itself is `gripette_sim.simulation`. Everything that was OpenArm
about it was the two values bound below; the XML surgery, the camera injection
and the fisheye render are the same work for any arm from the same export
toolchain, so they live in one place now.
"""

from gripette_sim import ArmSimulation
from gripette_sim.simulation import GRIPETTE_CAM, _load_model  # noqa: F401
from openarm_gripette_model import OPENARM_RIGHT_SCENE

from .kinematics import ARM_JOINT_NAMES

# All actuated joints in MuJoCo ordering. Kept at module scope because callers
# and tests import it from here.
ACTUATOR_NAMES = [*ARM_JOINT_NAMES, "proximal", "distal"]


class Simulation(ArmSimulation):
    """MuJoCo simulation of the OpenArm + Gripette."""

    ARM_JOINT_NAMES = ARM_JOINT_NAMES
    DEFAULT_SCENE = OPENARM_RIGHT_SCENE
    # The OpenArm's wrist roll drives a mimic joint that follows it negated.
    MIMIC_JOINTS = {"r_wrist_roll": "r_wrist_roll_mimic"}
