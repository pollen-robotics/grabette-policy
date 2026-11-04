#!/usr/bin/env python3
"""Post-process onshape-to-robot MuJoCo output for the OpenArm Gripette model.

Run this after every `onshape-to-robot-mujoco openarm_right` generation.

Patches to robot.xml:
  1. Remove meshdir="assets" from <compiler> and prefix mesh filenames with "assets/"
     → makes robot.xml includable from scene files in other directories
  2. Set contype="2" conaffinity="1" on the collision geom class
     → prevents self-collision between adjacent robot links
  3. Increase armature (0.005 → 0.1) and actuator gains (kp 50 → 500)
     → stiffer position tracking in simulation
  4. Add the Gripette camera element next to the camera site, pose derived
     from the site → MuJoCo camera for offscreen rendering (with 180° pitch
     correction, fovy=130)

Patches to scene.xml:
  5. Add offwidth="1296" offheight="972" to <global> so offscreen rendering works

Usage:
    cd openarm_gripette_model
    uv run onshape-to-robot-mujoco openarm_right
    python postprocess_mujoco.py
"""

import re
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "openarm_gripette_model" / "openarm_right"
ROBOT_XML = MODEL_DIR / "robot.xml"
SCENE_XML = MODEL_DIR / "scene.xml"

def camera_element(site_el: str) -> str:
    """Build the gripette_cam element from the calibrated `camera` site.

    The pose is DERIVED from the site on every generation (a hard-coded pose
    goes stale when the CAD changes — it once sat ~15 cm from the site).
    Same convention as simulation.py: same position as the site; the site's
    optical axis is +z while MuJoCo cameras look along -z, so the camera quat
    is site_quat * (0, 1, 0, 0), i.e. a 180° rotation about the site x-axis.
    """
    pos_m = re.search(r'pos="([^"]*)"', site_el)
    quat_m = re.search(r'quat="([^"]*)"', site_el)
    pos = pos_m.group(1) if pos_m else "0 0 0"
    w, x, y, z = (float(v) for v in quat_m.group(1).split()) if quat_m else (1.0, 0.0, 0.0, 0.0)
    # Hamilton product (w, x, y, z) * (0, 1, 0, 0) = (-x, w, z, -y).
    cq = (-x, w, z, -y)
    return (
        '                    <!-- Gripette camera (180° pitch correction from site frame to MuJoCo camera convention) -->\n'
        '                    <!-- fovy=130° for wide pinhole render, fisheye distortion applied in post -->\n'
        f'                    <camera name="gripette_cam" pos="{pos}"'
        f' quat="{cq[0]:.7g} {cq[1]:.7g} {cq[2]:.7g} {cq[3]:.7g}" fovy="130"/>'
    )


def patch_robot_xml():
    text = ROBOT_XML.read_text()
    original = text

    # 1. Remove meshdir from compiler, prefix mesh files
    text = re.sub(r'(<compiler\b[^>]*)\s+meshdir="assets"([^>]*>)', r'\1\2', text)
    text = re.sub(r'<mesh file="(?!assets/)([^"]+)"', r'<mesh file="assets/\1"', text)

    # 2. Collision class filtering
    text = re.sub(
        r'(<default class="collision">\s*<geom group="3")(\s*/>)',
        r'\1 contype="2" conaffinity="1"\2',
        text,
    )

    # 3. Stiffer actuators
    text = re.sub(
        r'<joint frictionloss="0\.1" armature="0\.005"/>',
        '<joint frictionloss="0.1" armature="0.1"/>',
        text,
    )
    text = re.sub(r'<position kp="50"', '<position kp="500"', text)

    # 4. Gripette camera
    if "gripette_cam" not in text:
        pat = r'(<!-- Frame camera -->\n\s*(<site group="3" name="camera"[^/]*/>)\n)'
        m = re.search(pat, text)
        if m:
            text = text[:m.end()] + camera_element(m.group(2)) + "\n" + text[m.end():]
        else:
            print("WARNING: camera site not found — skipping gripette_cam insertion")

    if text != original:
        ROBOT_XML.write_text(text)
        print(f"Patched {ROBOT_XML.name}")
    else:
        print(f"No changes in {ROBOT_XML.name}")


def patch_scene_xml():
    if not SCENE_XML.exists():
        print(f"{SCENE_XML} not found — skipping")
        return
    text = SCENE_XML.read_text()
    original = text

    # 5. Add offwidth/offheight to <global> tag (for offscreen rendering)
    if "offwidth" not in text:
        text = re.sub(
            r'(<global\b[^/]*?)(/?>)',
            r'\1 offwidth="1296" offheight="972"\2',
            text,
            count=1,
        )

    if text != original:
        SCENE_XML.write_text(text)
        print(f"Patched {SCENE_XML.name}")
    else:
        print(f"No changes in {SCENE_XML.name}")


def main():
    patch_robot_xml()
    patch_scene_xml()


if __name__ == "__main__":
    main()
