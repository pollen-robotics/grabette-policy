"""robot.urdf and robot.xml must describe the same robot.

They are produced by two separate onshape-to-robot passes and then patched by
postprocess_rebot.py. Placo reads the URDF, MuJoCo reads the MJCF. When they
disagree the IK solves for one robot and the sim steps another, and nothing
raises - which is exactly what happened: the first import had a 5.35 kg arm in
the URDF and a 3.40 kg arm in the MJCF, because the postprocess patched only
the URDF.

Cheap, pure-XML checks. No mujoco/placo import, so this runs anywhere.
"""

import math
import xml.etree.ElementTree as ET

import pytest

from rebot_gripette_model import (
    REBOT_B601RS_ASSETS,
    REBOT_B601RS_SCENE,
    REBOT_B601RS_URDF,
    REBOT_B601RS_XML,
)

# The physical arm, verified on hardware. Same numbers the LeRobot plugin uses.
PHYSICAL_LIMITS_DEG = {
    "shoulder_pan": (-145, 145),
    "shoulder_lift": (0, 170),
    "elbow_flex": (0, 200),
    "wrist_flex": (-80, 90),
    "wrist_yaw": (-90, 90),
    "wrist_roll": (-130, 130),
}


@pytest.fixture(scope="module")
def urdf_masses():
    root = ET.parse(REBOT_B601RS_URDF).getroot()
    out = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            out[link.get("name")] = float(inertial.find("mass").get("value"))
    return out


@pytest.fixture(scope="module")
def mjcf_masses():
    root = ET.parse(REBOT_B601RS_XML).getroot()
    out = {}
    for body in root.iter("body"):
        inertial = body.find("inertial")
        if inertial is not None and inertial.get("mass") is not None:
            out[body.get("name")] = float(inertial.get("mass"))
    return out


def test_total_mass_agrees(urdf_masses, mjcf_masses):
    # URDF additionally carries the 7 Gripette frames as ~1e-9 kg markers,
    # which MJCF represents as sites rather than bodies; they round away.
    u, m = sum(urdf_masses.values()), sum(mjcf_masses.values())
    assert u == pytest.approx(m, abs=1e-4), (
        f"URDF {u:.4f} kg vs MJCF {m:.4f} kg - run postprocess_rebot.py --apply"
    )


def test_every_shared_link_agrees_on_mass(urdf_masses, mjcf_masses):
    disagree = {
        n: (urdf_masses[n], mjcf_masses[n])
        for n in set(urdf_masses) & set(mjcf_masses)
        if abs(urdf_masses[n] - mjcf_masses[n]) > 1e-6
    }
    assert not disagree, f"mass mismatch: {disagree}"


def test_mass_is_plausible(urdf_masses):
    """Guards the failure mode where the export lands with no materials."""
    total = sum(urdf_masses.values())
    assert 4.5 < total < 8.0, f"{total:.2f} kg is not a plausible 6-DoF arm"


def test_joint_limits_match_the_physical_arm():
    """A CAD mate range wider than the arm will drive it into a hard stop."""
    root = ET.parse(REBOT_B601RS_URDF).getroot()
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in PHYSICAL_LIMITS_DEG:
            continue
        limit = joint.find("limit")
        assert limit is not None, f"{name} has no <limit>"
        lo = math.degrees(float(limit.get("lower")))
        hi = math.degrees(float(limit.get("upper")))
        exp_lo, exp_hi = PHYSICAL_LIMITS_DEG[name]
        assert lo == pytest.approx(exp_lo, abs=1.0), f"{name} lower {lo}"
        assert hi == pytest.approx(exp_hi, abs=1.0), f"{name} upper {hi}"


def test_all_six_arm_joints_are_present():
    root = ET.parse(REBOT_B601RS_URDF).getroot()
    names = {j.get("name") for j in root.findall("joint")}
    assert set(PHYSICAL_LIMITS_DEG) <= names
    assert {"proximal", "distal"} <= names, "Gripette finger joints missing"


def test_the_gripette_control_frames_exist():
    """oak_l is THE control frame the whole pipeline targets."""
    root = ET.parse(REBOT_B601RS_URDF).getroot()
    names = {link.get("name") for link in root.findall("link")}
    for frame in ("camera", "oak_l", "oak_r", "gripper_center",
                  "gripper_base", "thumb_tip", "finger_tip"):
        assert frame in names, f"missing Gripette frame {frame}"


def test_every_referenced_mesh_is_on_disk():
    root = ET.parse(REBOT_B601RS_URDF).getroot()
    refs = {m.get("filename") for m in root.iter("mesh") if m.get("filename")}
    assert refs, "no meshes referenced at all"
    missing = [r for r in refs
               if not (REBOT_B601RS_ASSETS.parent
                       / r.replace("package://", "")).exists()]
    assert not missing, f"{len(missing)} missing, e.g. {missing[:3]}"


def test_meshes_are_real_files_not_lfs_pointers():
    """A clone without `git lfs pull` leaves 130-byte text stubs behind.

    They parse as neither STL nor anything else, and MuJoCo's error for it is
    obscure, so fail here with a sentence that says what to do.
    """
    stls = sorted(REBOT_B601RS_ASSETS.glob("*.stl"))
    assert len(stls) == 121, f"expected 121 meshes, found {len(stls)}"
    stubs = [p.name for p in stls
             if p.stat().st_size < 1024
             and p.read_bytes().startswith(b"version https://git-lfs")]
    assert not stubs, (
        f"{len(stubs)} mesh(es) are unfetched git-LFS pointers "
        f"(e.g. {stubs[:2]}). Run `git lfs pull`."
    )


def test_scene_includes_the_robot():
    text = REBOT_B601RS_SCENE.read_text()
    assert 'include file="robot.xml"' in text


def test_the_base_is_bolted_down():
    """The raw Onshape export gives the root body a <freejoint>, which makes
    the whole robot a FREE BODY: it rests on whatever is beneath it by
    contact, visibly wobbles, and - the part that matters - its base can
    translate and rotate.

    Placo's FK, the ArmServicer's Cartesian integrator and every pose in a
    recorded dataset are all expressed from a base at the world origin. A base
    that drifts makes all of them quietly wrong rather than noisily broken.
    The OpenArm export has no freejoint anywhere; postprocess_rebot.py strips
    this one, and this test is what catches a re-export putting it back.
    """
    from rebot_gripette_model import REBOT_B601RS_XML

    root = ET.parse(REBOT_B601RS_XML).getroot()
    free = [b.get("name") for b in root.iter("body")
            if b.find("freejoint") is not None]
    assert not free, (
        f"{free} still has a <freejoint> - the arm is not bolted down. "
        f"Re-run postprocess_rebot.py --apply after the export.")
