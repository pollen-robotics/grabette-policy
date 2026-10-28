"""The reBot simulation, in the parts that need no GL context or bus.

The claim this package exists to make is that the simulated arm and the real
one are driven by the SAME service, not by two implementations that agree
today. Most of these tests are about keeping that true.
"""
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("placo", reason="needs gripette-core[kinematics]")


# --------------------------------------------------------------------------- #
# The shared service really is shared
# --------------------------------------------------------------------------- #

def test_sim_and_real_serve_the_same_arm_servicer_class():
    """Not 'equivalent behaviour' — the same object. If these ever diverge,
    a policy can pass in sim and fail on the arm for reasons that are about
    the control code rather than the robot."""
    from gripette_core.arm_service import ArmServicer
    from rebot_gripette_simu import server as sim_server

    assert sim_server.ArmServicer is ArmServicer

    import rebot_gripette.grpc_server_real as real
    assert real.ArmServicer is ArmServicer


def test_the_gripper_service_is_the_openarms():
    """Same gripper hardware, same calibration, one implementation."""
    from gripette_sim.gripper_service import GripperServicer as Core
    from openarm_gripette_simu.gripper_servicer import GripperServicer as OpenArm
    from rebot_gripette_simu.server import GripperServicer as ReBot

    assert ReBot is Core is OpenArm


def test_the_camera_is_the_openarms():
    """The fisheye calibration is a property of the Gripette. Two copies would
    mean a calibration fix landing on one arm only."""
    from gripette_sim.camera import FisheyeCamera as Core
    from openarm_gripette_simu.camera import FisheyeCamera as OpenArm

    assert OpenArm is Core


# --------------------------------------------------------------------------- #
# The arm binding
# --------------------------------------------------------------------------- #

def test_the_simulation_binds_the_rebots_six_joints():
    from rebot_gripette.spec import ARM_JOINT_NAMES
    from rebot_gripette_simu.simulation import ACTUATOR_NAMES, Simulation

    assert Simulation.ARM_JOINT_NAMES == list(ARM_JOINT_NAMES)
    assert len(Simulation.ARM_JOINT_NAMES) == 6
    # Arm first, then the gripper's two — the order the actuator list assumes.
    assert ACTUATOR_NAMES[-2:] == ["proximal", "distal"]


def test_the_rebot_declares_no_mimic_joints():
    """The OpenArm's wrist roll drives one; the reBot's six are independent.
    Declared per arm rather than hardcoded, which is what lets both share the
    wrapper."""
    from openarm_gripette_simu.simulation import Simulation as OpenArm
    from rebot_gripette_simu.simulation import Simulation as ReBot

    assert ReBot.MIMIC_JOINTS == {}
    assert OpenArm.MIMIC_JOINTS == {"r_wrist_roll": "r_wrist_roll_mimic"}


def test_a_simulation_without_joint_names_refuses_to_build():
    """Silently empty ARM_JOINT_NAMES would leave the actuator list as the
    gripper's two joints and the arm would simply never be commanded."""
    from gripette_sim import ArmSimulation

    class Nameless(ArmSimulation):
        DEFAULT_SCENE = "unused.xml"

    with pytest.raises(TypeError, match="ARM_JOINT_NAMES"):
        Nameless()


def test_a_simulation_with_no_scene_at_all_says_so():
    from gripette_sim import ArmSimulation

    class Sceneless(ArmSimulation):
        ARM_JOINT_NAMES = ["a"]

    with pytest.raises(ValueError, match="needs a scene"):
        Sceneless()


# --------------------------------------------------------------------------- #
# The sim-side ArmInterface
# --------------------------------------------------------------------------- #

class _FakeSim:
    """Records what the interface would have done to a simulation."""

    def __init__(self, n=6):
        self.commands = []
        self.enabled = None
        self._pos = np.zeros(n)

    def get_joint_positions(self, names):
        return self._pos[:len(names)].copy()

    def set_joint_commands(self, q, names):
        self.commands.append(np.asarray(q, dtype=float).copy())

    def set_arm_actuators_enabled(self, enable, names):
        self.enabled = enable


def test_a_command_of_the_wrong_width_is_refused():
    """Six joints, not seven. A silently truncated or zero-padded command is
    an arm moving somewhere nobody asked for."""
    from rebot_gripette_simu.sim_arm_interface import SimArmInterface

    iface = SimArmInterface(_FakeSim(), ["a", "b", "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="6 joint angles"):
        iface.send_command_rad(np.zeros(7))


def test_commands_are_dropped_while_torque_is_off():
    """Otherwise a command silently re-engages an arm the caller believes is
    limp."""
    from rebot_gripette_simu.sim_arm_interface import SimArmInterface

    sim = _FakeSim()
    iface = SimArmInterface(sim, list("abcdef"))
    iface.set_torque(False)
    sim.commands.clear()
    iface.send_command_rad(np.ones(6))
    assert sim.commands == []


def test_re_enabling_torque_holds_the_pose_the_arm_is_in():
    """Not the stale target. A hand-placed arm must not be snapped back."""
    from rebot_gripette_simu.sim_arm_interface import SimArmInterface

    sim = _FakeSim()
    iface = SimArmInterface(sim, list("abcdef"))
    iface.set_torque(False)
    sim._pos = np.arange(6, dtype=float)      # someone moved it
    sim.commands.clear()
    iface.set_torque(True)
    assert sim.commands, "re-enabling torque commanded nothing"
    assert np.allclose(sim.commands[-1], np.arange(6))


def test_disabling_torque_disables_the_actuators():
    """In sim as on hardware, torque off means the arm falls. A simulation
    that kept holding position would be the one place a policy could learn
    that releasing torque is harmless."""
    from rebot_gripette_simu.sim_arm_interface import SimArmInterface

    sim = _FakeSim()
    iface = SimArmInterface(sim, list("abcdef"))
    iface.set_torque(False)
    assert sim.enabled is False
    iface.set_torque(True)
    assert sim.enabled is True


# --------------------------------------------------------------------------- #
# The home pose is required, here as on the real server
# --------------------------------------------------------------------------- #

def test_the_cli_requires_a_home_pose():
    """'Home' is a property of the experiment, not the robot. A default here
    becomes the number someone copies onto the hardware."""
    src = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "rebot_gripette_simu" / "__main__.py").read_text()
    assert '"--home_joints"' in src and "required=True" in src
    assert "check_home_pose" in src


def test_the_scene_puts_the_table_where_this_arm_reaches():
    """The reBot reaches in -x. A table copied from the OpenArm's scene would
    be behind it, and its 0.40 m top would put this arm's gripper under the
    surface at the home pose."""
    from rebot_gripette_simu.server import DEFAULT_SCENE

    xml = DEFAULT_SCENE.read_text()
    assert 'name="table"' in xml
    assert 'pos="-0.35 0 0"' in xml, "the table must be at negative x"
    # offscreen buffer must cover the camera calibration resolution
    assert 'offwidth="1296"' in xml and 'offheight="972"' in xml
