"""Tests for convert_dataset.compute_delta_actions: hold rows are a valid rotation.

Run from policies/DiffusionPolicy:
    uv run pytest tests/test_convert_dataset.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_dataset import HOLD_DELTA, compute_delta_actions  # noqa: E402
from rotation import rotation_6d_to_rotation_matrix_numpy, rotvec_to_rotation_6d  # noqa: E402


def _poses(n: int) -> np.ndarray:
    """(n, 11) absolute poses: steady 2 mm/frame along x, 1°/frame about z."""
    pos = np.stack([0.002 * np.arange(n), np.zeros(n), np.zeros(n)], axis=1)
    rotvec = np.stack([np.zeros(n), np.zeros(n), np.radians(np.arange(n))], axis=1)
    gripper = np.full((n, 2), 0.3)
    return np.concatenate([pos, rotvec_to_rotation_6d(rotvec), gripper], axis=1).astype(np.float32)


def test_episode_boundaries_are_identity_holds():
    episodes = np.array([0] * 5 + [1] * 4)
    actions = compute_delta_actions(_poses(9), episodes)

    for last in (4, 8):  # last frame of each episode
        np.testing.assert_array_equal(actions[last, :9], HOLD_DELTA)
        R = rotation_6d_to_rotation_matrix_numpy(actions[last, 3:9])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


def test_hold_delta_is_a_valid_rotation():
    rows = HOLD_DELTA[3:9].reshape(2, 3)
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_interior_deltas_unchanged():
    episodes = np.zeros(6, dtype=int)
    actions = compute_delta_actions(_poses(6), episodes)

    for i in range(5):
        np.testing.assert_allclose(np.linalg.norm(actions[i, :3]), 0.002, atol=1e-6)
        R = rotation_6d_to_rotation_matrix_numpy(actions[i, 3:9])
        np.testing.assert_allclose(Rotation.from_matrix(R).magnitude(), np.radians(1.0), atol=1e-5)
    np.testing.assert_allclose(actions[:, 9:], 0.3)
