"""Placo kinematics must see the wrist roll (SIM-1).

In the URDF, `r_wrist_roll_mimic` (mimic of `r_wrist_roll`, multiplier -1) is
the joint that carries the wrist pitch and the gripper/camera chain;
`r_wrist_roll`'s own child is a side branch. MuJoCo ties the two with an
equality constraint (mimic = -roll). These tests check placo FK/IK against
MuJoCo directly (robot.xml, no server modules):
    uv run pytest robots/openarm/openarm_gripette_simu/tests/test_kinematics_wrist_roll.py
"""
import mujoco
import numpy as np
import pytest
from openarm_gripette_model import OPENARM_RIGHT_XML

from openarm_gripette_simu.kinematics import (
    ARM_JOINT_NAMES,
    CONTROL_FRAME,
    ROLL_IK_LIMIT,
    Kinematics,
)

ROLL_IDX = ARM_JOINT_NAMES.index("r_wrist_roll")


def _rot_err_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    c = (np.trace(A @ B.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(str(OPENARM_RIGHT_XML))
    return model, mujoco.MjData(model)


def _joint_ranges(model):
    """(lo, hi) arrays of the 7 arm joints' MuJoCo ranges."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    rng = np.array([model.jnt_range[i] for i in ids])
    return rng[:, 0], rng[:, 1]


def _random_configs(model, n, seed, shrink=0.9):
    """n random arm configs inside the (shrunk) joint limits, roll always != 0."""
    lo, hi = _joint_ranges(model)
    mid, half = (lo + hi) / 2.0, shrink * (hi - lo) / 2.0
    rng = np.random.default_rng(seed)
    qs = rng.uniform(mid - half, mid + half, size=(n, 7))
    # Keep the roll clearly away from 0 so a roll-blind FK can't pass.
    small = np.abs(qs[:, ROLL_IDX]) < 0.2
    qs[small, ROLL_IDX] = np.sign(qs[small, ROLL_IDX] + 1e-9) * 0.4
    return qs


def _mujoco_site_pose(model, data, q, site):
    """World pose of a MuJoCo site for arm config q (mimic set per the equality)."""
    data.qpos[:] = 0.0
    for name, val in zip(ARM_JOINT_NAMES, q):
        data.joint(name).qpos[0] = val
    data.joint("r_wrist_roll_mimic").qpos[0] = -q[ROLL_IDX]
    mujoco.mj_kinematics(model, data)
    T = np.eye(4)
    T[:3, :3] = data.site(site).xmat.reshape(3, 3)
    T[:3, 3] = data.site(site).xpos
    return T


def test_fk_matches_mujoco(kin, mj):
    model, data = mj
    for q in _random_configs(model, 20, seed=0):
        T_placo = kin.forward(q, frame=CONTROL_FRAME)
        T_mj = _mujoco_site_pose(model, data, q, CONTROL_FRAME)
        pos_err = np.linalg.norm(T_placo[:3, 3] - T_mj[:3, 3])
        rot_err = _rot_err_deg(T_placo[:3, :3], T_mj[:3, :3])
        assert pos_err < 0.5e-3, f"q={q}: pos err {pos_err * 1e3:.2f} mm"
        assert rot_err < 0.1, f"q={q}: rot err {rot_err:.3f} deg"


def test_fk_depends_on_wrist_roll(kin):
    q = np.array([0.3, 0.1, 0.0, 1.2, 0.0, 0.0, 0.3])
    T0 = kin.forward(q, frame=CONTROL_FRAME)
    q[ROLL_IDX] = 0.5
    T1 = kin.forward(q, frame=CONTROL_FRAME)
    # A 0.5 rad roll moves oak_l ~2 cm and rotates it by ~0.5 rad.
    assert np.linalg.norm(T1[:3, 3] - T0[:3, 3]) > 5e-3
    assert _rot_err_deg(T1[:3, :3], T0[:3, :3]) > 20.0


def test_ik_round_trip_with_roll(kin, mj):
    # Target AND result are judged in MuJoCo (the ground truth), so a roll-blind
    # placo model can't pass by being self-consistent.
    model, data = mj
    rng = np.random.default_rng(1)
    for q_star in _random_configs(model, 10, seed=2, shrink=0.7):
        T_star = _mujoco_site_pose(model, data, q_star, CONTROL_FRAME)
        seed = q_star + rng.uniform(-0.1, 0.1, 7)
        seed[ROLL_IDX] = 0.0   # roll must actually be solved for, not inherited
        q = kin.inverse(T_star, current_joint_positions=seed, frame=CONTROL_FRAME)
        assert q.shape == (7,)
        T = _mujoco_site_pose(model, data, q, CONTROL_FRAME)
        pos_err = np.linalg.norm(T[:3, 3] - T_star[:3, 3])
        rot_err = _rot_err_deg(T[:3, :3], T_star[:3, :3])
        assert pos_err < 1e-3, f"q*={q_star}: pos err {pos_err * 1e3:.2f} mm"
        assert rot_err < 0.5, f"q*={q_star}: rot err {rot_err:.3f} deg"


def test_ik_roll_stays_inside_driver_limit(kin, mj):
    """The real driver clips joint_6 to +/-40 deg. A target that needs MORE roll
    than that must not make IK command past ROLL_IK_LIMIT (it uses the other
    joints instead)."""
    model, data = mj
    assert np.degrees(ROLL_IK_LIMIT) <= 40.0
    q_star = np.array([0.3, 0.1, 0.0, 1.2, 0.0, 0.0, 0.3])
    q_star[ROLL_IDX] = 0.78            # URDF-legal, beyond the driver range
    T_star = _mujoco_site_pose(model, data, q_star, CONTROL_FRAME)
    q = kin.inverse(T_star, current_joint_positions=q_star, frame=CONTROL_FRAME)
    assert abs(q[ROLL_IDX]) <= ROLL_IK_LIMIT + 1e-6, np.degrees(q[ROLL_IDX])


@pytest.mark.parametrize("source", ["placo", "mujoco"])
def test_wrist_roll_matches_real_arm(kin, mj, source):
    """+r_wrist_roll must turn oak_l the way +joint_6 turns the REAL arm.

    Checked on the real arm 2026-11-05 (.claude/roll_sign_check.py): the
    Onshape export had the roll reversed; robot.urdf / robot.xml were hand-
    edited (roll + mimic axes flipped). Convention: the real motors are the
    reference and the model follows. At home, +roll then rotates oak_l about
    its own -y axis. A re-export that drops the hand edit fails here."""
    model, data = mj
    home = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])
    rolled = home.copy()
    rolled[ROLL_IDX] = 0.1
    if source == "placo":
        T0, T1 = (kin.forward(q, frame=CONTROL_FRAME) for q in (home, rolled))
    else:
        T0, T1 = (_mujoco_site_pose(model, data, q, CONTROL_FRAME) for q in (home, rolled))
    R = T0[:3, :3].T @ T1[:3, :3]                       # rotation in the oak_l frame
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= np.linalg.norm(axis)
    assert axis[1] < -0.95, f"+roll rotates oak_l about {axis.round(2)} (expected ~[0, -1, 0])"


def test_ik_keeps_mimic_coupled(kin):
    """After IK, the placo mimic joint equals -roll (what MuJoCo/the real arm do)."""
    T = kin.forward(np.array([0.2, 0.0, 0.1, 1.0, 0.1, 0.4, 0.2]), frame=CONTROL_FRAME)
    q = kin.inverse(T, current_joint_positions=np.array([0.2, 0.0, 0.1, 1.0, 0.1, 0.3, 0.2]),
                    frame=CONTROL_FRAME)
    assert kin.robot.get_joint("r_wrist_roll_mimic") == pytest.approx(-q[ROLL_IDX], abs=1e-6)
