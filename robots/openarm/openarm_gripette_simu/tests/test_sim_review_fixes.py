"""Regression tests for the sim server fixes from the 2026-11 code review (SIM-*).

They build the real table_grasp.xml scene in-process (no gRPC server, no
network ports) and call the servicer / server methods directly. Poses keep the
wrist roll at 0 (the roll FK fix lives on another branch).
"""

import threading
from pathlib import Path

import mujoco
import numpy as np
import pytest

from openarm_gripette_simu import Kinematics, Simulation
from openarm_gripette_simu.arm_servicer import START_JOINTS, ArmServicer
from openarm_gripette_simu.proto import arm_pb2

SCENE = Path(__file__).resolve().parents[1] / "scenes" / "table_grasp.xml"

# Gripper centre lowered to table height over the legacy cube region
# (IK'd once from START_JOINTS; wrist roll 0). Blocks ~half of that region.
POSE_OVER_CUBE_REGION = [0.276, 0.006, -0.004, 0.934, 0.006, 0.0, 0.36]


@pytest.fixture(scope="module")
def sim():
    return Simulation(SCENE)


@pytest.fixture
def servicer(sim):
    """Legacy-path servicer (no server back-reference), seeded."""
    s = ArmServicer(sim, Kinematics(), threading.Lock(), 0.0)
    s._rng = np.random.default_rng(0)
    return s


def arm_cube_contacts(sim):
    """Contacts between the cube BODY (any of its geoms) and anything that is
    not the table or the world (floor)."""
    m, d = sim.model, sim.data
    cube = m.body("red_cube").id
    n = 0
    for i in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        if cube in (b1, b2):
            other = b2 if b1 == cube else b1
            name = m.body(other).name
            if "table" not in name and name != "world":
                n += 1
    return n


def reset(servicer, joints=()):
    return servicer.Reset(arm_pb2.ResetRequest(joint_positions=list(joints)), None)


# ── SIM-2: legacy Reset validates the cube against the NEW arm pose ────────

def test_legacy_reset_does_not_spawn_the_cube_inside_the_arm(servicer, sim):
    bad = 0
    for _ in range(20):
        assert reset(servicer, START_JOINTS).success          # arm somewhere else...
        assert reset(servicer, POSE_OVER_CUBE_REGION).success  # ...then teleported here
        bad += arm_cube_contacts(sim) > 0
    assert bad == 0, f"{bad}/20 resets left the cube touching the arm"


def test_legacy_reset_gives_up_instead_of_hanging(servicer, monkeypatch):
    """A pose that blocks the whole cube region must fail the Reset, not spin
    the rejection loop forever (with the sim lock held)."""
    monkeypatch.setattr(servicer, "_cube_contacts_robot", lambda: True)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", reset(servicer, START_JOINTS)),
                         daemon=True)
    t.start()
    t.join(timeout=20.0)
    assert not t.is_alive(), "Reset is stuck in the cube rejection loop"
    assert not out["r"].success


# ── SIM-3: legacy cube sampling uses THIS scene's table ────────────────────

def _tabletop(sim):
    """(x_min, x_max, y_min, y_max, top_z) of the loaded scene's tabletop."""
    g = sim.model.geom("tabletop").id
    c, s = sim.data.geom_xpos[g], sim.model.geom_size[g]
    return c[0] - s[0], c[0] + s[0], c[1] - s[1], c[1] + s[1], c[2] + s[2]


def test_legacy_reset_keeps_the_cube_on_this_table(servicer, sim):
    mujoco.mj_forward(sim.model, sim.data)
    x0, x1, y0, y1, top = _tabletop(sim)
    half_h = float(sim.model.geom_size[sim.model.geom("red_cube").id][1])   # cylinder
    for _ in range(60):
        r = reset(servicer, START_JOINTS)
        assert r.success
        assert x0 < r.cube_x < x1 and y0 < r.cube_y < y1, (r.cube_x, r.cube_y)
        assert abs(r.cube_z - (top + half_h)) < 1e-3, r.cube_z


# ── SimulationServer (in-process; gRPC is never started) ───────────────────

@pytest.fixture(scope="module")
def server():
    from openarm_gripette_simu.server import SimulationServer

    srv = SimulationServer(scene_xml=SCENE)
    srv._rng = np.random.default_rng(0)
    return srv


def _place_cube(sim, xyz):
    m, d = sim.model, sim.data
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    a = m.jnt_qposadr[j]
    d.qpos[a:a + 3] = xyz
    d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)


# ── SIM-4: arm-cube penetration counts as a colliding start ────────────────

def test_arm_cube_penetration_is_a_collision(server):
    from openarm_gripette_simu.kinematics import GRIPPER_FRAME

    sim = server._sim
    cube_bid = sim.model.body("red_cube").id
    _place_cube(sim, (0.44, -0.25, 0.41))                 # on the table, out of the way
    assert not server._arm_config_collides(START_JOINTS, cube_bid)   # baseline: free
    grip = server._kin.forward(START_JOINTS, frame=GRIPPER_FRAME)[:3, 3]
    _place_cube(sim, grip)                                # cube inside the jaws
    assert server._arm_config_collides(START_JOINTS, cube_bid)
    _place_cube(sim, (0.44, -0.142, 0.41))                # put it back for later tests
