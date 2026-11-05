"""Regression tests for convert_dataset review findings (TR-8, TR-16). Synthetic poses only."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert_dataset as cd  # noqa: E402
from rotation import rotation_6d_to_rotation_matrix_numpy, rotvec_to_rotation_6d  # noqa: E402


def _line_poses(n=100, glitch_deg=0.0, at=50):
    """2 mm/frame along x, fixed orientation, optional one-frame yaw glitch."""
    pos = np.stack([0.002 * np.arange(n), np.zeros(n), np.zeros(n)], axis=1)
    rv = np.zeros((n, 3))
    rv[at, 2] = np.radians(glitch_deg)
    return np.concatenate([pos, rotvec_to_rotation_6d(rv), np.full((n, 2), 0.3)],
                          axis=1).astype(np.float32)


def _rot_deg(actions):
    R = rotation_6d_to_rotation_matrix_numpy(actions[:, 3:9])
    return np.degrees(Rotation.from_matrix(R).magnitude())


# ── TR-8: pose smoothing must not smear a rotation glitch ──────────────

def test_rotation_glitch_is_not_smeared_under_the_despike_threshold():
    poses, ep = _line_poses(glitch_deg=20.0), np.zeros(100, int)
    ang = _rot_deg(cd.compute_delta_actions(cd.smooth_poses_11d(poses, ep, 9), ep))
    clean = np.r_[ang[:49], ang[51:-1]]
    assert clean.max() < 0.5, f"glitch leaked into neighbours: {clean.max():.2f} deg"
    assert (ang[49:51] > 5.0).all(), "glitch deltas must stay above the despike cap"


def test_smoothing_unchanged_without_glitches():
    poses, ep = _line_poses(), np.zeros(100, int)
    out = cd.smooth_poses_11d(poses, ep, 9)
    np.testing.assert_allclose(out[:, :3], poses[:, :3], atol=1e-6)


# ── TR-16: relative state from the despiked deltas ─────────────────────

def test_relative_state_ignores_a_despiked_relocalization_jump():
    poses, ep = _line_poses(), np.zeros(100, int)
    poses[50:, 0] += 0.2                                 # 200 mm SLAM re-acquisition
    deltas = cd.compute_delta_actions(poses, ep)
    deltas[49, :9] = cd.HOLD_DELTA                       # what despike does to it
    s = cd.compute_relative_to_start_state(cd.integrate_delta_poses(deltas, poses, ep), ep)
    # 99 steps of 2 mm minus the one held step
    assert s[-1, 0] == pytest.approx(0.002 * 98, abs=1e-4)


def test_integrated_poses_reproduce_clean_poses():
    poses = _line_poses()
    rv = np.zeros((100, 3))
    rv[:, 2] = np.radians(np.arange(100) * 0.5)
    poses[:, 3:9] = rotvec_to_rotation_6d(rv)
    ep = np.r_[np.zeros(60, int), np.ones(40, int)]
    out = cd.integrate_delta_poses(cd.compute_delta_actions(poses, ep), poses, ep)
    np.testing.assert_allclose(out, poses, atol=1e-4)
