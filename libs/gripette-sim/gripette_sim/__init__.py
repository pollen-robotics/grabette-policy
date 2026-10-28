"""Shared MuJoCo pieces for Gripette-equipped arms.

The split from `gripette-core` is deliberate and load-bearing: core is
imported by the real-hardware drivers and therefore refuses mujoco and opencv,
which all three modules here need. See the README.

An arm binds its two values once:

    class MySimulation(ArmSimulation):
        ARM_JOINT_NAMES = [...]
        DEFAULT_SCENE = MY_SCENE
"""

from gripette_sim.camera import FisheyeCamera
from gripette_sim.gripper_service import GripperServicer
from gripette_sim.simulation import GRIPETTE_CAM, GRIPPER_JOINT_NAMES, Simulation

__all__ = [
    "ArmSimulation",
    "FisheyeCamera",
    "GRIPETTE_CAM",
    "GRIPPER_JOINT_NAMES",
    "GripperServicer",
    "Simulation",
]


class ArmSimulation(Simulation):
    """A `Simulation` with one arm's joint names and default scene bound.

    Exists so each arm's binding is two class attributes rather than a
    forwarded constructor. Subclasses keep `Simulation(scene_xml)`'s original
    one-positional-argument signature, which is how every existing call site
    spells it.
    """

    ARM_JOINT_NAMES: list[str] = []
    DEFAULT_SCENE = None
    MIMIC_JOINTS: dict[str, str] = {}

    def __init__(self, scene_xml=None):
        if not self.ARM_JOINT_NAMES:
            raise TypeError(
                f"{type(self).__name__} must set ARM_JOINT_NAMES; without it "
                "the actuator list is just the gripper's two joints and the "
                "arm would silently never be commanded.")
        super().__init__(
            scene_xml,
            arm_joint_names=self.ARM_JOINT_NAMES,
            default_scene=self.DEFAULT_SCENE,
            mimic_joints=self.MIMIC_JOINTS,
        )
