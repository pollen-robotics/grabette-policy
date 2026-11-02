"""Tests for delta_6d_to_matrix: a degenerate 6D delta decodes to no rotation.

Policies trained on hold frames written as all zeros predict a near-zero 6D,
which plain Gram-Schmidt turns into an arbitrary rotation. Needs numpy only:
    uv run pytest robots/openarm/openarm_gripette_simu/tests/test_delta_6d_guard.py
"""
import numpy as np
import pytest

from openarm_gripette_simu.rotation import (
    delta_6d_to_matrix,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
)


def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0],
                     [np.sin(a), np.cos(a), 0.0],
                     [0.0, 0.0, 1.0]])


def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


@pytest.mark.parametrize("r6d", [
    np.zeros(6),                                      # what the dataset's hold rows hold
    np.array([0.003, -0.002, 0.001, 0.0, 0.002, -0.001]),  # a typical hold prediction
    np.array([0.99, 0.01, 0.0, 0.002, 0.001, 0.0]),   # only the second row collapsed
])
def test_degenerate_delta_is_identity(r6d):
    R, degenerate = delta_6d_to_matrix(r6d)
    assert degenerate
    np.testing.assert_array_equal(R, np.eye(3))


def test_unguarded_decode_of_hold_noise_is_a_large_rotation():
    # Documents the failure the guard exists for.
    noise = np.array([0.003, -0.002, 0.001, 0.0, 0.002, -0.001])
    assert _angle_deg(rotation_6d_to_matrix(noise)) > 30.0


@pytest.mark.parametrize("deg", [0.0, 1.0, 5.0, 30.0])
def test_valid_delta_is_unchanged(deg):
    r6d = rotation_matrix_to_6d(_rot_z(deg))
    R, degenerate = delta_6d_to_matrix(r6d)
    assert not degenerate
    np.testing.assert_allclose(R, _rot_z(deg), atol=1e-9)


def test_slightly_short_rows_still_decode():
    # A healthy but imperfect prediction (rows ~0.6 long) is renormalized, not dropped.
    r6d = 0.6 * rotation_matrix_to_6d(_rot_z(2.0))
    R, degenerate = delta_6d_to_matrix(r6d)
    assert not degenerate
    np.testing.assert_allclose(R, _rot_z(2.0), atol=1e-9)
