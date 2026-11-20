"""Regression tests for the data-collection / replay example scripts, from the
2026-11 code review (EV-8 .. EV-13).

The scripts live in examples/ (not a package), so each is loaded by path like
evaluate.py is in the other tests. No physics episodes are run here: the fixes
are tested on the pure helpers the scripts call.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name):
    """Import examples/<name>.py (it resolves its sibling imports itself, or
    via the examples dir put on sys.path for the duration of the import).

    sys.path is restored afterwards (in full: several scripts also insert
    their own dir at import): other suites in the same pytest run (the
    reBot's) have example scripts with the SAME bare names, e.g.
    cartesian_square, and a leftover sys.path[0] entry made their
    `import cartesian_square` load the OpenArm one."""
    saved = list(sys.path)
    sys.path.insert(0, str(_EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(f"_ex_{name}", _EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


# ── EV-8: --initial-joints must put the CONTROL frame (oak_l) at home ──────

def test_sample_init_joints_puts_the_control_frame_at_home():
    from openarm_gripette_simu.kinematics import CONTROL_FRAME, Kinematics

    mod = _load("sample_init_joints")
    q, wp = mod.sample_init_joints(seed=0)
    T = Kinematics().forward(q, frame=CONTROL_FRAME)
    err_mm = 1000 * np.linalg.norm(T[:3, 3] - wp.home_xyz)
    # Same convergence bar the sim server's reset uses (20 mm); the camera-site
    # recipe this replaces left oak_l ~40 mm off because BODY_TO_CAMERA is the
    # identity while the URDF's camera site sits 40 mm / 20 deg from oak_l.
    assert err_mm < 20.0, f"oak_l {err_mm:.1f} mm from the sampled home"


# ── EV-10: replay --arm must place the cube at the REAL grasp ──────────────

def _episode(prox_segments):
    """Synthetic 8-D raw episode: x advances 1 cm per frame (so each frame's
    pose is distinct) and proximal follows the given (value, n_frames)
    segments. Raw data closes proximal NEGATIVE."""
    prox = np.concatenate([np.full(n, v, float) for v, n in prox_segments])
    acts = np.zeros((len(prox), 8))
    acts[:, 0] = 0.01 * np.arange(len(prox))
    acts[:, 6] = prox
    return acts


def test_cube_frame_of_a_release_episode_is_the_final_grasp():
    """Release episodes START closed (more closed than the grasp), then open,
    approach, close on the cube and lift."""
    mod = _load("replay_dataset")
    acts = _episode([(-1.2, 10), (0.0, 30), (-0.8, 20)])
    assert mod.pick_cube_frame(acts) == 40


def test_cube_frame_of_a_recover_episode_is_the_second_grasp():
    """Recover episodes close once on a deliberate miss (here MORE closed: the
    jaws met nothing), reopen, reposition, and close for real."""
    mod = _load("replay_dataset")
    acts = _episode([(0.0, 10), (-1.0, 8), (0.0, 12), (-0.7, 15)])
    assert mod.pick_cube_frame(acts) == 30


def test_cube_frame_of_a_normal_episode_is_unchanged():
    mod = _load("replay_dataset")
    acts = _episode([(0.0, 25), (-0.9, 20)])
    assert mod.pick_cube_frame(acts) == 25


# ── EV-13: the IK-filter summary is per PLANNED episode, not per request ───

def test_ik_drop_rate_is_bounded_by_the_attempts():
    """--episodes 100 needing 130 plans (physics failures force extra ones)
    with 40 IK drops: 40/130 dropped, not 40/100 — and never above 100%."""
    mod = _load("collect_grasp_dataset")
    n_planned, avg_accepted, drop_rate = mod.ik_filter_summary(
        n_plans=130, n_ik_dropped=40, ik_attempts_total=40 * 200 + 90 * 3,
        max_ik_attempts=200)
    assert n_planned == 90
    assert avg_accepted == pytest.approx(3.0)
    assert drop_rate == pytest.approx(100.0 * 40 / 130)
    _n, _a, worst = mod.ik_filter_summary(n_plans=150, n_ik_dropped=150,
                                          ik_attempts_total=150 * 200, max_ik_attempts=200)
    assert worst == pytest.approx(100.0)
