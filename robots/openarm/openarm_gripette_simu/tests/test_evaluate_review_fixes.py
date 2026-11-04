"""Regression tests for evaluate.py fixes from the 2026-11 code review (EV-*).

Like test_chunk_relative_eval.py, these load the REAL examples/evaluate.py and
drive its functions / run_episode() against small fakes, so a fix is checked in
the code that ships rather than in a mirror of it.

Needs the `eval` extra (evaluate.py imports lerobot/torch).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_EVALUATE = Path(__file__).resolve().parents[1] / "examples" / "evaluate.py"

torch = pytest.importorskip("torch")
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

from openarm_gripette_simu.rotation import delta_6d_to_matrix  # noqa: E402


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("_evaluate", _EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _server_angle_deg(dr6):
    """Rotation the arm server would actually apply for this 6D delta."""
    R, _ = delta_6d_to_matrix(dr6)
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


# ── EV-1: --clamp_rot_deg must not turn a "hold" into a rotation ──────────

@pytest.mark.parametrize("dr6", [
    np.array([0.003, -0.002, 0.001, 0.0, 0.002, -0.001]),   # typical hold noise
    np.random.default_rng(0).normal(scale=0.08, size=6),    # reviewer's repro
])
def test_clamp_keeps_a_degenerate_hold_a_hold(ev, dr6):
    assert _server_angle_deg(dr6) == 0.0          # the server reads it as "hold"
    _dp, out, _was = ev.clamp_delta(np.zeros(3), dr6.copy(), None, np.radians(2.0))
    assert _server_angle_deg(out) == 0.0, "clamp re-encoded a hold as a real rotation"


def test_clamp_of_an_exactly_zero_6d_does_not_raise(ev):
    # Raising here kills the async executor thread (run_episode_async then spins).
    dp, out, was = ev.clamp_delta(np.zeros(3), np.zeros(6), 0.005, np.radians(2.0))
    assert _server_angle_deg(out) == 0.0 and not was


def test_clamp_still_caps_a_real_rotation(ev):
    from openarm_gripette_simu.rotation import rotation_matrix_to_6d
    dr6 = rotation_matrix_to_6d(Rotation.from_rotvec([0, 0, np.radians(10)]).as_matrix())
    _dp, out, was = ev.clamp_delta(np.zeros(3), dr6, None, np.radians(2.0))
    assert was and abs(_server_angle_deg(out) - 2.0) < 1e-6
