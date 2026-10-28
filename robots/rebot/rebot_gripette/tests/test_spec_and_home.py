"""The reBot's ArmSpec, and the guard on whatever home pose a caller supplies.

TERMINOLOGY, because conflating these is how an arm gets driven into a stop:

  starting pose  where the robot physically is at power-on — on this arm,
                 resting on its hard mechanical stops. An observation about
                 the robot. Nothing commands it.

  home pose      the startup position for an EXPERIMENT. `Reset` ramps the arm
                 to it, so it is the first commanded motion of a run. It is a
                 property of the experiment, not of the robot: the OpenArm's
                 is its simulator's START_JOINTS, the pose its dataset was
                 recorded from.

There is no reBot home pose constant to test, deliberately — the reBot has no
experiment yet, so any value would be a guess wearing the costume of a named
constant. What IS testable is `check_home_pose`, the guard applied to whatever
the operator passes to --home_joints.

Note what the guard CANNOT do: it knows the URDF and nothing else. It cannot
tell you a pose is right for your experiment, and it knows nothing about the
table, the pedestal, the Gripette's envelope or self-collision.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("placo", reason="needs the [kinematics] extra")
pytest.importorskip("rebot_gripette_model")

from gripette_core.kinematics import CONTROL_FRAME, Kinematics  # noqa: E402
from rebot_gripette.spec import (  # noqa: E402
    ARM_JOINT_NAMES,
    HOME_JOINTS_RAD,
    REBOT_SPEC,
    check_home_pose,
)
from rebot_gripette_model import REBOT_B601RS_URDF  # noqa: E402


def _urdf_limits():
    root = ET.parse(REBOT_B601RS_URDF).getroot()
    out = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (math.degrees(float(lim.get("lower"))),
                                  math.degrees(float(lim.get("upper"))))
    return out


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #

def test_spec_has_six_joints_in_urdf_order():
    assert len(REBOT_SPEC.joint_names) == 6
    limits = _urdf_limits()
    for n in REBOT_SPEC.joint_names:
        assert n in limits, f"{n} is not a joint in the URDF"


def test_the_gripette_fingers_are_masked():
    """Unmasked, the IK can 'reach' a target by closing the hand."""
    assert set(REBOT_SPEC.masked_dofs) == {"proximal", "distal"}


# --------------------------------------------------------------------------- #
# Joint limits: one source, and it is the model
# --------------------------------------------------------------------------- #

def test_published_limits_are_the_models_limits():
    """spec.urdf_limits() is what calibration, the examples and check_home_pose
    all read. Compared here against an independent parse of the same file, so
    the check does not just re-run the implementation."""
    from rebot_gripette.spec import urdf_limits

    published = urdf_limits()
    for name, (lo, hi) in _urdf_limits().items():
        assert published[name] == (pytest.approx(lo), pytest.approx(hi))


def test_every_arm_joint_has_a_limit():
    """A joint missing from the table reads as unconstrained - check_home_pose
    would KeyError, and the calibration warning would silently skip it."""
    from rebot_gripette.spec import urdf_limits

    limits = urdf_limits()
    for name in ARM_JOINT_NAMES:
        assert name in limits, f"{name} has no URDF limit"


def test_the_limits_are_not_transcribed_anywhere():
    """calibration.py used to carry its own copy, wrong on five of six joints
    and wrong in the direction that kept its warning quiet. A transcribed table
    is a second source that stops matching the model without saying so."""
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parents[1] / "rebot_gripette"
    src = (pkg / "calibration.py").read_text()
    assert "URDF_LIMITS = {" not in src


def test_the_real_limits_are_what_the_stale_table_claimed_they_were_not():
    """Pins the four joints the old table got wrong, so a regression to those
    values fails here rather than on the arm."""
    from rebot_gripette.spec import urdf_limits

    limits = urdf_limits()
    assert limits["shoulder_pan"][1] == pytest.approx(145.0, abs=1.0)
    assert limits["shoulder_lift"][1] == pytest.approx(170.0, abs=1.0)
    assert limits["elbow_flex"][1] == pytest.approx(200.0, abs=1.0)
    assert limits["wrist_roll"][1] == pytest.approx(130.0, abs=1.0)


# --------------------------------------------------------------------------- #
# There must be no invented default
# --------------------------------------------------------------------------- #

def test_there_is_no_default_home_pose():
    """A default here would be a guess that Reset drives the arm into.

    When the reBot sim scene exists, set this to ITS start joints — the way
    the OpenArm's home is its simulator's START_JOINTS — and change this test
    to assert the two agree. Do not fill it in from 'somewhere in the middle
    of the joint limits'.
    """
    assert HOME_JOINTS_RAD is None, (
        "a home pose was given a default. It belongs to the experiment, not "
        "the robot: tie it to the sim scene's start joints, or leave it unset "
        "so the operator must choose.")


# --------------------------------------------------------------------------- #
# The guard on a supplied pose — necessary conditions only
# --------------------------------------------------------------------------- #

def test_all_zeros_is_rejected():
    """Zero is the LOWER limit for shoulder_lift and elbow_flex.

    This is the specific trap of translating the OpenArm's all-zeros-ish home
    across: on this arm it is two hard stops.
    """
    with pytest.raises(ValueError, match="hard stop"):
        check_home_pose(np.zeros(6))


def test_zero_really_is_the_lower_limit_for_those_two_joints():
    """Guards the PREMISE of the test above, so a re-export can't silence it."""
    limits = _urdf_limits()
    for n in ("shoulder_lift", "elbow_flex"):
        assert abs(limits[n][0]) < 1e-6, (
            f"{n}'s lower limit is no longer 0; the all-zeros hazard has "
            "changed and the surrounding reasoning needs revisiting")


def test_wrong_joint_count_is_rejected():
    with pytest.raises(ValueError, match="expected 6"):
        check_home_pose(np.zeros(7))


def test_a_pose_against_the_upper_stop_is_rejected():
    limits = _urdf_limits()
    q = np.zeros(6)
    q[ARM_JOINT_NAMES.index("elbow_flex")] = math.radians(limits["elbow_flex"][1])
    q[ARM_JOINT_NAMES.index("shoulder_lift")] = math.radians(60.0)
    with pytest.raises(ValueError, match="hard stop"):
        check_home_pose(q)


def test_a_pose_clear_of_every_stop_is_accepted():
    """An ARBITRARY in-envelope pose: accepted by the guard, which is the
    point — acceptance means 'not jammed', never 'correct for the experiment'."""
    q = np.deg2rad([0.0, 60.0, 60.0, 0.0, 0.0, 0.0])
    out = check_home_pose(q)
    assert np.allclose(out, q)


def test_the_margin_is_configurable_and_actually_applied():
    q = np.deg2rad([0.0, 3.0, 60.0, 0.0, 0.0, 0.0])   # 3 deg from the stop
    check_home_pose(q, margin_deg=1.0)                 # fine at 1 deg
    with pytest.raises(ValueError, match="hard stop"):
        check_home_pose(q, margin_deg=5.0)


# --------------------------------------------------------------------------- #
# Kinematics sanity on an accepted pose
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def kin():
    return Kinematics(REBOT_SPEC)


@pytest.fixture(scope="module")
def sample_pose():
    """Any pose the guard accepts. NOT a recommended home pose."""
    return np.deg2rad([0.0, 60.0, 60.0, 0.0, 0.0, 0.0])


def test_fk_is_finite_and_away_from_the_base(kin, sample_pose):
    T = kin.forward(sample_pose, frame=CONTROL_FRAME)
    assert np.all(np.isfinite(T))
    assert np.linalg.norm(T[:3, 3]) > 0.05, "control frame is at the base origin"


def test_ik_round_trips(kin, sample_pose):
    """If IK cannot return to its own FK pose, nothing downstream works."""
    target = kin.forward(sample_pose, frame=CONTROL_FRAME)
    sol = kin.inverse(target, current_joint_positions=sample_pose,
                      frame=CONTROL_FRAME)
    reached = kin.forward(sol, frame=CONTROL_FRAME)
    err_mm = float(np.linalg.norm(reached[:3, 3] - target[:3, 3]) * 1000)
    assert err_mm < 1.0, f"IK round-trip error {err_mm:.2f} mm"


def test_ik_solutions_stay_within_joint_limits(kin, sample_pose):
    """Placo is told to respect limits; confirm it does on this 6-DoF arm."""
    limits = _urdf_limits()
    target = kin.forward(sample_pose, frame=CONTROL_FRAME)
    for axis in range(3):
        t = target.copy()
        t[axis, 3] += 0.02
        sol = np.rad2deg(kin.inverse(t, current_joint_positions=sample_pose,
                                     frame=CONTROL_FRAME))
        for name, q in zip(REBOT_SPEC.joint_names, sol):
            lo, hi = limits[name]
            assert lo - 1.0 <= q <= hi + 1.0, (
                f"IK put {name} at {q:.1f} deg, outside [{lo:.0f}, {hi:.0f}]")
