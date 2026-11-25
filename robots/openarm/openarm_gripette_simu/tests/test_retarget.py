"""Joint-space retargeting of handheld (Grabette) demos onto the OpenArm.

    uv run pytest robots/openarm/openarm_gripette_simu/tests/test_retarget.py

The central test is a synthetic round trip: a known joint trajectory is turned
into what a Grabette recording would contain (oak_l poses in an arbitrary SLAM
world, re-anchored at frame 0, plus the table plane seen at frame 0), and the
retargeting must recover the placement and an IK chain that reproduces it.
"""
from collections import Counter

import numpy as np
import pytest

from openarm_gripette_simu.kinematics import CONTROL_FRAME, Kinematics
from openarm_gripette_simu.retarget import (
    ChainResult,
    Limits,
    Placement,
    Retargeter,
    fit_table_plane,
    joint_state_action,
    load_retarget_model,
    placement_from_start_pose,
    relative_poses,
    repair_joint_spikes,
    rotvec_to_matrix,
    select_delta,
    world_poses,
)

TABLE_Z = 0.35
# A comfortable start config (the derived red-cube start pose).
Q_START = np.array([-0.259, 1.017, 0.244, 2.018, 1.062, 0.0, -0.654])


def _rot_err_deg(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def _matrix_to_rotvec(R):
    """Inverse of rotvec_to_matrix for test data (angle < pi)."""
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return w * ang


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


@pytest.fixture(scope="module")
def retargeter(kin):
    return Retargeter(kin, load_retarget_model(), Limits(), home_seeds=[Q_START])


@pytest.fixture(scope="module")
def demo(kin):
    """A smooth joint trajectory, its oak_l poses in the robot frame, and the Grabette-style recording.

    Returns dict(q, T, pose8, n, h0, placement): pose8 is the 8D action as recorded (oak_l pose in an
    arbitrary SLAM world, gripper zeros), n/h0 the frame-0 table plane in the camera frame."""
    t = np.linspace(0, 1, 60)[:, None]
    q = Q_START + 0.25 * np.sin(np.pi * t) * np.array([0.3, -0.4, 0.2, -0.5, 0.3, 0.2, 0.4])
    T = np.stack([kin.forward(qi, frame=CONTROL_FRAME) for qi in q])
    # Arbitrary SLAM world: the recording only knows poses up to a rigid transform.
    W = np.eye(4)
    W[:3, :3] = rotvec_to_matrix(np.array([[0.3, -1.1, 0.7]]))[0]
    W[:3, 3] = [1.0, -2.0, 0.5]
    Ts = W @ T
    pose8 = np.zeros((len(q), 8))
    pose8[:, :3] = Ts[:, :3, 3]
    pose8[:, 3:6] = [_matrix_to_rotvec(R) for R in Ts[:, :3, :3]]
    R0 = T[0, :3, :3]
    n = R0.T @ np.array([0, 0, 1.0])              # table normal (up) seen from the frame-0 camera
    h0 = T[0, 2, 3] - TABLE_Z                    # frame-0 camera height above the table
    return dict(q=q, T=T, pose8=pose8, n=n, h0=h0, placement=placement_from_start_pose(T[0], TABLE_Z))


# ---------------------------------------------------------------- geometry

def test_rotvec_to_matrix_matches_rodrigues():
    rv = np.array([[0.0, 0.0, np.pi / 2], [0.0, 0.0, 0.0]])
    R = rotvec_to_matrix(rv)
    np.testing.assert_allclose(R[0], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(R[1], np.eye(3), atol=1e-12)


def test_relative_poses_are_anchored_at_frame_zero(demo):
    t_rel, R_rel = relative_poses(demo["pose8"])
    np.testing.assert_allclose(t_rel[0], 0, atol=1e-12)
    np.testing.assert_allclose(R_rel[0], np.eye(3), atol=1e-9)
    # Relative motion is independent of the SLAM world: compare with the robot-frame truth.
    T0inv = np.linalg.inv(demo["T"][0])
    rel_true = T0inv @ demo["T"]
    np.testing.assert_allclose(t_rel, rel_true[:, :3, 3], atol=1e-9)
    np.testing.assert_allclose(R_rel, rel_true[:, :3, :3], atol=1e-9)


def test_fit_table_plane_recovers_normal_and_height():
    rng = np.random.default_rng(0)
    n_true = np.array([0.05, -0.85, -0.3])
    n_true /= np.linalg.norm(n_true)
    h_true = 0.18
    # Points on the plane {p : n.p = -h}, i.e. the camera (origin) is h above it on the +n side.
    a = np.cross(n_true, [1, 0, 0]); a /= np.linalg.norm(a)
    b = np.cross(n_true, a)
    uv = rng.uniform(-0.3, 0.3, size=(3000, 2))
    P = -h_true * n_true + uv[:, :1] * a + uv[:, 1:] * b + rng.normal(0, 0.002, (3000, 1)) * n_true
    outliers = rng.uniform(-0.5, 0.5, size=(800, 3)) + [0, 0, 0.6]
    n, h, frac = fit_table_plane(np.vstack([P, outliers]), rng)
    assert np.degrees(np.arccos(np.clip(n @ n_true, -1, 1))) < 0.5
    assert abs(h - h_true) < 0.002
    assert 0.7 < frac < 0.85


def test_world_poses_reproduce_the_robot_frame_trajectory(demo):
    t_rel, R_rel = relative_poses(demo["pose8"])
    Tw = world_poses(t_rel, R_rel, demo["n"], demo["h0"], demo["placement"])
    np.testing.assert_allclose(Tw[:, :3, 3], demo["T"][:, :3, 3], atol=1e-9)
    np.testing.assert_allclose(Tw[:, :3, :3], demo["T"][:, :3, :3], atol=1e-9)


def test_placement_shift_moves_and_turns_about_the_anchor(demo):
    t_rel, R_rel = relative_poses(demo["pose8"])
    p = demo["placement"]
    moved = world_poses(t_rel, R_rel, demo["n"], demo["h0"], p.shifted(0.05, -0.02, 0.0))
    base = world_poses(t_rel, R_rel, demo["n"], demo["h0"], p)
    np.testing.assert_allclose(moved[:, :3, 3] - base[:, :3, 3], np.tile([0.05, -0.02, 0], (len(base), 1)), atol=1e-12)
    turned = world_poses(t_rel, R_rel, demo["n"], demo["h0"], p.shifted(0, 0, 90.0))
    np.testing.assert_allclose(turned[0, :3, 3], base[0, :3, 3], atol=1e-12)   # anchor stays put
    np.testing.assert_allclose(turned[:, 2, 3], base[:, 2, 3], atol=1e-12)     # heights unchanged


# ---------------------------------------------------------------- feasibility

def test_chain_retargets_a_reachable_demo(retargeter, demo):
    t_rel, R_rel = relative_poses(demo["pose8"])
    Ts = world_poses(t_rel, R_rel, demo["n"], demo["h0"], demo["placement"])
    r = retargeter.chain(Ts, TABLE_Z)
    assert r.ok, r.reason
    assert r.max_pos_err < 0.002 and r.max_rot_err_deg < 1.0
    assert r.q.shape == (len(Ts), 7)
    # FK of the recovered joints reproduces the demo (joints may differ by redundancy, poses must not).
    assert retargeter.max_fk_error(r.q, Ts) < 0.002


def test_chain_rejects_an_unreachable_placement(retargeter, demo):
    t_rel, R_rel = relative_poses(demo["pose8"])
    far = demo["placement"].shifted(1.5, 0.0, 0.0)
    r = retargeter.chain(world_poses(t_rel, R_rel, demo["n"], demo["h0"], far), TABLE_Z)
    assert not r.ok
    assert r.fail_frame == 0 and r.reason.startswith("home")


def test_chain_rejects_links_below_the_table(retargeter, demo):
    """Same demo, but the table declared 40 cm higher: the arm would be inside it."""
    t_rel, R_rel = relative_poses(demo["pose8"])
    Ts = world_poses(t_rel, R_rel, demo["n"], demo["h0"], demo["placement"])
    r = retargeter.chain(Ts, TABLE_Z + 0.40)
    assert not r.ok and "table" in r.reason


# ---------------------------------------------------------------- per-episode correction choice

def _res(ok, margin=0.3):
    return ChainResult(ok=ok, fail_frame=None if ok else 3, reason=None if ok else "pos 20mm",
                       q=np.zeros((1, 7)), max_pos_err=0, max_rot_err_deg=0, max_step=0,
                       min_joint_margin=margin, min_pedestal=0.02, min_link_height=0.1)


BOUNDS = (0.05, 15.0)


def test_select_delta_keeps_zero_when_it_is_feasible_with_margin():
    cands = [((0, 0, 0), _res(True, 0.15)), ((0.025, 0, 0), _res(True, 0.4))]
    assert select_delta(cands, margin_target=0.1, bounds=BOUNDS)[0] == (0, 0, 0)


def test_select_delta_ignores_numerical_noise_at_a_joint_limit():
    """IK clamps a joint exactly at its limit: the margin comes out as -1e-12, not 0.
    That is not a reason to correct the placement."""
    cands = [((0, 0, 0), _res(True, -1e-12)), ((0.025, 0, 0), _res(True, 0.3))]
    assert select_delta(cands, margin_target=0.0, bounds=BOUNDS)[0] == (0, 0, 0)


def test_select_delta_takes_the_smallest_feasible_correction():
    cands = [((0, 0, 0), _res(False)), ((0.05, 0, 0), _res(True)), ((0.025, 0, 0), _res(True)),
             ((0, 0, 7.5), _res(True))]
    # |(0.025,0,0)| = 0.5 bound-units, |(0,0,7.5)| = 0.5 too: tie -> larger margin wins; equal -> first.
    assert select_delta(cands, margin_target=0.1, bounds=BOUNDS)[0] == (0.025, 0, 0)


def test_select_delta_prefers_adequate_margin_over_size():
    cands = [((0, 0, 0), _res(True, 0.02)), ((0.05, 0, 0), _res(True, 0.2))]
    assert select_delta(cands, margin_target=0.1, bounds=BOUNDS)[0] == (0.05, 0, 0)


def test_select_delta_returns_none_and_reasons_when_nothing_is_feasible():
    cands = [((0, 0, 0), _res(False)), ((0.05, 0, 0), _res(False))]
    delta, res, reasons = select_delta(cands, margin_target=0.1, bounds=BOUNDS)
    assert delta is None and res is None and reasons == Counter({"pos": 2})


# ---------------------------------------------------------------- post-processing

def test_repair_joint_spikes_fixes_isolated_glitches_only():
    q = np.zeros((10, 7))
    q[4, 2] = 0.8                  # isolated one-frame glitch
    q[7:, 0] = 0.8                 # sustained jump: a real reconfiguration, must be left alone
    fixed = repair_joint_spikes(q, thr=0.3)
    assert fixed == 1
    assert q[4, 2] == 0.0 and q[8, 0] == 0.8


def test_joint_state_action_next_joints_and_copied_gripper():
    q = np.arange(3 * 7, dtype=float).reshape(3, 7)
    g_state = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]])
    g_action = np.array([[0.1, 1.0], [0.2, 1.0], [0.3, 0.4]])   # projected: commanded close != next state
    state, action = joint_state_action(q, g_state, g_action)
    assert state.shape == action.shape == (3, 9)
    np.testing.assert_allclose(action[:, :7], q[[1, 2, 2]])        # next joints, last repeated
    np.testing.assert_allclose(state[:, :7], q)
    np.testing.assert_allclose(state[:, 7:], g_state)
    np.testing.assert_allclose(action[:, 7:], g_action)            # gripper copied, not derived
