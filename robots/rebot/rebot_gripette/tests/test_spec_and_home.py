"""The reBot's ArmSpec and home pose must be usable before anyone powers the arm.

The home pose is the one thing here that CANNOT be inherited from the OpenArm.
shoulder_lift is 0..170 deg and elbow_flex 0..200 deg, so both START at zero:
the OpenArm's all-zeros home would sit exactly on two hard stops, pushing the
arm into them on every Reset and leaving Placo's joint-limit constraint with
no room on one side. These tests pin that down on a model, which is the only
place it can be caught before the arm is energised.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("placo", reason="needs the [kinematics] extra")
pytest.importorskip("rebot_gripette_model")

from gripette_core.kinematics import CONTROL_FRAME, Kinematics  # noqa: E402
from rebot_gripette_model import REBOT_B601RS_URDF  # noqa: E402

# Imported lazily inside tests: grpc_server_real pulls grpc, which the base
# test environment need not have.


def _spec_and_home():
    from rebot_gripette.spec import HOME_JOINTS_RAD, REBOT_SPEC
    return REBOT_SPEC, HOME_JOINTS_RAD


def _urdf_limits():
    root = ET.parse(REBOT_B601RS_URDF).getroot()
    out = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (math.degrees(float(lim.get("lower"))),
                                  math.degrees(float(lim.get("upper"))))
    return out


def test_spec_has_six_joints_in_urdf_order():
    spec, _ = _spec_and_home()
    assert len(spec.joint_names) == 6
    limits = _urdf_limits()
    for n in spec.joint_names:
        assert n in limits, f"{n} is not a joint in the URDF"


def test_the_gripette_fingers_are_masked():
    """Unmasked, the IK can 'reach' a target by closing the hand."""
    spec, _ = _spec_and_home()
    assert set(spec.masked_dofs) == {"proximal", "distal"}


def test_home_is_not_on_a_hard_stop():
    """The whole reason this arm needs its own home pose."""
    spec, home = _spec_and_home()
    limits = _urdf_limits()
    assert len(home) == len(spec.joint_names)
    for name, q in zip(spec.joint_names, np.rad2deg(home)):
        lo, hi = limits[name]
        margin = min(q - lo, hi - q)
        assert margin > 5.0, (
            f"home has {name} at {q:.1f} deg, only {margin:.1f} deg from a "
            f"limit of [{lo:.0f}, {hi:.0f}] — too close to a hard stop")


def test_all_zeros_would_have_been_on_a_stop():
    """Guards the reasoning, so nobody 'simplifies' home back to zeros."""
    limits = _urdf_limits()
    on_stop = [n for n in ("shoulder_lift", "elbow_flex")
               if abs(limits[n][0]) < 1e-6]
    assert on_stop == ["shoulder_lift", "elbow_flex"], (
        "the premise changed: zero is no longer the lower limit for these "
        "joints, so the custom home pose may no longer be needed")


def test_fk_at_home_is_finite_and_in_front_of_the_base():
    spec, home = _spec_and_home()
    kin = Kinematics(spec)
    T = kin.forward(home, frame=CONTROL_FRAME)
    assert np.all(np.isfinite(T))
    assert np.linalg.norm(T[:3, 3]) > 0.05, "control frame is at the base origin"


def test_ik_round_trips_at_home():
    """If IK cannot return to its own FK pose, nothing downstream will work."""
    spec, home = _spec_and_home()
    kin = Kinematics(spec)
    target = kin.forward(home, frame=CONTROL_FRAME)
    sol = kin.inverse(target, current_joint_positions=home, frame=CONTROL_FRAME)
    reached = kin.forward(sol, frame=CONTROL_FRAME)
    err_mm = float(np.linalg.norm(reached[:3, 3] - target[:3, 3]) * 1000)
    assert err_mm < 1.0, f"IK round-trip error {err_mm:.2f} mm at home"


def test_ik_solutions_stay_within_joint_limits():
    """Placo is told to respect limits; confirm it actually does on this arm."""
    spec, home = _spec_and_home()
    kin = Kinematics(spec)
    limits = _urdf_limits()
    target = kin.forward(home, frame=CONTROL_FRAME)
    # Nudge 2 cm along each axis in turn and check every solution is legal.
    for axis in range(3):
        t = target.copy()
        t[axis, 3] += 0.02
        sol = np.rad2deg(kin.inverse(t, current_joint_positions=home,
                                     frame=CONTROL_FRAME))
        for name, q in zip(spec.joint_names, sol):
            lo, hi = limits[name]
            assert lo - 1.0 <= q <= hi + 1.0, (
                f"IK put {name} at {q:.1f} deg, outside [{lo:.0f}, {hi:.0f}]")
