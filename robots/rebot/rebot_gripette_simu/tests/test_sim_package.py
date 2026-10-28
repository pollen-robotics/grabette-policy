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


def test_the_arm_is_bolted_to_the_table_not_standing_beside_it():
    """The reBot mounts ON the bench. The table is therefore built DOWNWARD
    from z=0 with the arm's base left at the world origin - not the arm raised
    onto a surface, which would offset Placo's FK, the Cartesian integrator
    and every recorded pose by the table height."""
    from rebot_gripette_simu.server import DEFAULT_SCENE

    xml = DEFAULT_SCENE.read_text()
    assert 'name="tabletop"' in xml
    # Top face at z=0: centre -0.015 with half-thickness 0.015.
    assert 'size="0.40 0.35 0.015" pos="-0.30 0 -0.015"' in xml
    assert 'pos="0 0 -0.73"' in xml, "the floor must be below the table"
    # offscreen buffer must cover the camera calibration resolution
    assert 'offwidth="1296"' in xml and 'offheight="972"' in xml


def test_the_table_is_under_the_arms_reach():
    """This arm reaches in -x, so a table copied from the OpenArm's scene
    would be behind it."""
    from rebot_gripette_simu.server import DEFAULT_SCENE

    xml = DEFAULT_SCENE.read_text()
    assert 'pos="-0.30 0 -0.015"' in xml, "the tabletop must be at negative x"


def test_the_simulated_base_cannot_move():
    """A free base makes FK quietly wrong. Checked here as well as in the
    model package, because this is the scene that actually gets simulated."""
    import mujoco

    from rebot_gripette_simu.server import DEFAULT_SCENE
    from gripette_sim.simulation import _load_model

    model = _load_model(DEFAULT_SCENE)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "axis_1")
    assert model.body_dofnum[base] == 0, (
        "the robot's root body has degrees of freedom - it is not bolted down")


# --------------------------------------------------------------------------- #
# The clearance gate
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def clearance():
    from gripette_sim.clearance import GeomClearance
    from gripette_sim.simulation import _load_model
    from rebot_gripette.spec import REBOT_SPEC
    from rebot_gripette_simu.obstacles import (GUARDED_BODIES, OBSTACLE_GEOMS,
                                               TABLE_MARGIN)
    from rebot_gripette_simu.server import DEFAULT_SCENE

    model = _load_model(DEFAULT_SCENE)
    return GeomClearance(model, OBSTACLE_GEOMS, GUARDED_BODIES,
                         list(REBOT_SPEC.joint_names), margin=TABLE_MARGIN)


def test_the_gate_is_actually_armed(clearance):
    """A gate that silently found no geoms would pass every other test here
    while guarding nothing."""
    assert clearance.enabled
    assert clearance.obstacle_gids and clearance.arm_gids


def test_a_pose_driving_into_the_table_is_blocked(clearance):
    """The failure this exists to prevent: an IK solution that is perfectly
    smooth - no branch flip for the watchdog to catch - and goes through the
    bench."""
    gate = clearance.as_joint_gate()
    into_table = np.deg2rad([0, 60, 20, -60, 0, 0])
    assert clearance.clearance(into_table) < 0, "expected penetration"
    assert gate(into_table) is not None


def test_the_resting_pose_is_allowed(clearance):
    """The arm sits folded on its lower stops at power-on, bolted to this very
    table. A gate that fires there would fire forever and block the first
    command of every run - which is why the base links are not guarded."""
    gate = clearance.as_joint_gate()
    assert gate(np.deg2rad([0, 0, 0, 0, 0, 0])) is None


def test_the_home_pose_is_allowed(clearance):
    gate = clearance.as_joint_gate()
    assert gate(np.deg2rad([0, 15, 25, -10, 0, 0])) is None


def test_retreating_from_inside_the_margin_is_always_allowed(clearance):
    """Without this an arm that got inside the margin is deadlocked: every
    command that could free it is also a command inside the margin, so the
    gate refuses them all and the only way out is to kill the server."""
    stuck = np.deg2rad([0, 90, 10, -80, 0, 0])
    assert 0 < clearance.clearance(stuck) < clearance.margin, "fixture pose moved"

    gate = clearance.as_joint_gate(current_joints=lambda: stuck)
    retreat = np.deg2rad([0, 90, 40, -80, 0, 0])      # elbow up, away from it
    assert clearance.clearance(retreat) > clearance.clearance(stuck)
    assert gate(retreat) is None

    deeper = np.deg2rad([0, 90, 10, -85, 0, 0])
    assert gate(deeper) is not None, "going deeper must still be refused"


def test_the_inner_links_are_deliberately_unguarded():
    """They are in permanent contact: the arm is BOLTED to this surface. Named
    explicitly so adding them back is a decision, not an accident."""
    from rebot_gripette_simu.obstacles import GUARDED_BODIES

    for base_link in ("axis_1", "axis_2", "axis_3"):
        assert base_link not in GUARDED_BODIES


def test_the_servicer_takes_the_gate_and_rolls_back_on_a_veto():
    """The gate must reject with the same integrator rollback as the IK-jump
    watchdog. A veto that left the delta in the integrator would have the
    target march into the table while every command was being refused."""
    import inspect

    from gripette_core import arm_service

    assert "joint_gate" in inspect.signature(arm_service.ArmServicer).parameters
    src = inspect.getsource(arm_service.ArmServicer.SendCartesianDelta)
    gate_at = src.index("self._joint_gate(target_joints)")
    rollback = src.index("self._target_pos -= delta_pos_world", gate_at)
    assert rollback > gate_at
