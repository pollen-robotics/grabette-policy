"""postprocess_mujoco.py on scratch copies of the generated model (no hardware).

RA-8: the script inserted a hard-coded `gripette_cam` pose that is ~15 cm away
from the calibrated `camera` site, and that camera then overrides the
site-derived one that simulation.py injects. The camera must be derived from
the site (same pos; site quat rotated 180 deg about x, the MuJoCo camera
convention used by simulation.py).
"""

import importlib.util
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[2] / "openarm_gripette_model"
MODEL_DIR = MODEL_ROOT / "openarm_gripette_model" / "openarm_right"


def _load_postprocess(tmp_path):
    for name in ("robot.xml", "scene.xml"):
        shutil.copy(MODEL_DIR / name, tmp_path / name)
    spec = importlib.util.spec_from_file_location("postprocess_mujoco", MODEL_ROOT / "postprocess_mujoco.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    pp.ROBOT_XML = tmp_path / "robot.xml"
    pp.SCENE_XML = tmp_path / "scene.xml"
    return pp


def _floats(s):
    return np.array([float(v) for v in s.split()])


def test_postprocess_camera_matches_site(tmp_path):
    pp = _load_postprocess(tmp_path)
    pp.main()
    root = ET.parse(tmp_path / "robot.xml").getroot()

    parent_of = {child: parent for parent in root.iter() for child in parent}
    site = next(e for e in root.iter("site") if e.get("name") == "camera")
    cams = [e for e in root.iter("camera") if e.get("name") == "gripette_cam"]
    assert len(cams) == 1
    cam = cams[0]

    assert parent_of[cam] is parent_of[site]  # same body frame
    np.testing.assert_allclose(_floats(cam.get("pos")), _floats(site.get("pos")), atol=1e-6)
    expected_q = np.zeros(4)
    mujoco.mju_mulQuat(expected_q, _floats(site.get("quat")), np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(_floats(cam.get("quat")), expected_q, atol=1e-6)
    assert cam.get("fovy") == "130"

    # Idempotent: a second run leaves robot.xml unchanged.
    before = (tmp_path / "robot.xml").read_text()
    pp.main()
    assert (tmp_path / "robot.xml").read_text() == before
