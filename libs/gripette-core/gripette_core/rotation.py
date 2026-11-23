"""6D continuous rotation representation (Zhou et al., CVPR 2019).

Encodes orientation as the first two columns of the rotation matrix (6 values).
No singularities, smooth gradients for neural networks.
"""

import numpy as np


def rotation_matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to 6D representation.

    Args:
        matrix: shape (3, 3) or (N, 3, 3).

    Returns:
        6D vector(s), shape (6,) or (N, 6).
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to a 3x3 rotation matrix.

    Uses Gram-Schmidt orthogonalization to recover the third column.

    Args:
        rot_6d: shape (6,) or (N, 6).

    Returns:
        Rotation matrix, shape (3, 3) or (N, 3, 3).
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-2)


# Both rows of a valid 6D rotation have unit length. Diffusion checkpoints
# trained on convert_dataset.py output predict a near-ZERO 6D on "hold" chunks
# (the converter writes hold frames as all zeros, not as the identity), and
# Gram-Schmidt then normalizes that noise into an arbitrary rotation — up to
# 150° in one step, which turns the wrist and trips the IK-jump watchdog.
# Measured on red-cube-blue-box-30fps_diffusion-best: degenerate predictions
# have a row norm <= 0.39, healthy ones >= 0.51.
MIN_DELTA_ROW_NORM = 0.5


def delta_6d_to_matrix(rot_6d: np.ndarray) -> tuple[np.ndarray, bool]:
    """Decode a 6D rotation DELTA, mapping a degenerate one to the identity.

    A delta with either row shorter than MIN_DELTA_ROW_NORM is not a rotation —
    it is a policy saying "hold" — so it decodes to no rotation instead of to
    whatever direction its noise points in.

    Args:
        rot_6d: shape (6,).

    Returns:
        (rotation matrix (3, 3), True if the delta was degenerate).
    """
    rot_6d = np.asarray(rot_6d, dtype=np.float64)
    row_norms = np.linalg.norm(rot_6d.reshape(2, 3), axis=-1)
    if row_norms.min() < MIN_DELTA_ROW_NORM:
        return np.eye(3), True
    return rotation_6d_to_matrix(rot_6d), False
