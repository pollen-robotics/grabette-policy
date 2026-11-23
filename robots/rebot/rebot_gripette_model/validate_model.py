#!/usr/bin/env python3
"""Validate an onshape-to-robot export of the reBot + Gripette.

Re-runnable after every export, because the OpenArm model README warns a
re-export "can silently regress collision classes, gains, axes, and joint
ranges" - so this checks the things that regress, not just that files exist.

Checks, in rough order of how badly each breaks the pipeline:
  1. URDF loads, topology is 6 arm joints + gripette proximal/distal
  2. Gripette frames present - oak_l especially, it is THE control frame
  3. Masses are real (no zeros, no duplicated placeholders)
  4. Link/mesh names are ASCII (Placo/MuJoCo/ROS choke otherwise)
  5. Joint limits match the physical arm, not CAD mate ranges
  6. Every referenced mesh file exists
  7. MuJoCo compiles the MJCF, and Placo loads the URDF, if installed

Exits non-zero if any BLOCKER fails.
"""
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# What the LeRobot plugin config says the real arm does, in degrees.
PLUGIN_LIMITS = {
    1: ("shoulder_pan", -145, 145),
    2: ("shoulder_lift", 0, 170),
    3: ("elbow_flex", 0, 200),
    4: ("wrist_flex", -80, 90),
    5: ("wrist_yaw", -90, 90),
    6: ("wrist_roll", -130, 130),
}
GRIPETTE_FRAMES = ["camera", "oak_l", "oak_r", "thumb_tip",
                   "gripper_center", "gripper_base", "finger_tip"]

problems, warnings = [], []


def blocker(msg):
    problems.append(msg)
    print(f"  BLOCKER  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  warning  {msg}")


def ok(msg):
    print(f"  ok       {msg}")


ap = argparse.ArgumentParser()
ap.add_argument("model_dir", type=Path)
args = ap.parse_args()
D = args.model_dir

print(f"Validating {D}\n")

# ---------------------------------------------------------------- files
print("[files]")
urdf_p, mjcf_p = D / "robot.urdf", D / "robot.xml"
for f in ("robot.urdf", "robot.xml", "scene.xml"):
    (ok if (D / f).exists() else blocker)(f"{f} {'present' if (D/f).exists() else 'MISSING'}")
if not urdf_p.exists():
    sys.exit("cannot continue without robot.urdf")

root = ET.parse(urdf_p).getroot()
links = {l.get("name"): l for l in root.findall("link")}
joints = root.findall("joint")
actuated = [j for j in joints if j.get("type") in ("revolute", "continuous", "prismatic")]

# ---------------------------------------------------------------- topology
print("\n[topology]")
print(f"  {len(links)} links, {len(joints)} joints "
      f"({len(actuated)} actuated)")
chain = []
kids = {}
for j in joints:
    kids.setdefault(j.find("parent").get("link"), []).append(j)
child_of = {j.find("child").get("link") for j in joints}
roots = [n for n in links if n not in child_of]
if len(roots) != 1:
    blocker(f"expected exactly 1 root link, found {len(roots)}: {roots}")
else:
    ok(f"single root: {roots[0]}")

if len(actuated) == 8:
    ok("8 actuated joints (6 arm + gripette proximal/distal)")
else:
    warn(f"{len(actuated)} actuated joints, expected 8")

# ---------------------------------------------------------------- frames
print("\n[gripette frames]")
# In URDF, frames come through as massless links on fixed joints.
names = set(links)
for f in GRIPETTE_FRAMES:
    hit = [n for n in names if f == n or n.endswith("_" + f) or f in n]
    if hit:
        ok(f"{f:<15} -> {hit[0]}")
    elif f == "oak_l":
        blocker("oak_l MISSING - it is the control frame the IK targets")
    else:
        warn(f"{f} missing")

# ---------------------------------------------------------------- masses
print("\n[masses]")
masses = {}
for name, l in links.items():
    i = l.find("inertial")
    if i is None:
        continue
    m = float(i.find("mass").get("value"))
    masses[name] = m
total = sum(masses.values())
# onshape-to-robot writes an EPSILON mass (~1e-9), not literal zero, so a
# "<= 0" test silently passes. Frames legitimately have epsilon mass and no
# geometry; a link WITH geometry and epsilon mass is a missing CAD material.
EPS = 1e-4
def has_geom(n):
    return bool(links[n].findall("visual") or links[n].findall("collision"))

massless_real = [n for n, m in masses.items() if m < EPS and has_geom(n)]
massless_frames = [n for n, m in masses.items() if m < EPS and not has_geom(n)]
seen = {}
for n, m in masses.items():
    if m >= EPS:
        seen.setdefault(round(m, 6), []).append(n)
dupes = {m: ns for m, ns in seen.items() if len(ns) > 1}

print(f"  total {total:.4f} kg over {len(masses)} links with inertials")
if massless_frames:
    ok(f"{len(massless_frames)} massless frame markers (expected)")
if massless_real:
    blocker(f"link(s) with geometry but no mass: {massless_real} "
            "- missing material assignment in CAD")
else:
    ok("every link with geometry has mass")
if dupes:
    for m, ns in dupes.items():
        warn(f"{len(ns)} links share mass {m} kg exactly ({ns}) "
             "- looks like a placeholder, not a measurement")
else:
    ok("no duplicated placeholder masses")
if total < 4.5:
    warn(f"total {total:.2f} kg is well under the old reBot URDF's ~6.0 kg "
         "- masses likely under-assigned")

# ---------------------------------------------------------------- names
print("\n[names]")
def nonascii(s):
    return any(ord(c) > 127 for c in s)

bad_links = [n for n in links if nonascii(n)]
bad_meshes = sorted({m.get("filename") for m in root.iter("mesh")
                     if m.get("filename") and nonascii(m.get("filename"))})
if bad_links:
    # Downgraded from blocker: MuJoCo compiles and Placo loads these fine, so
    # the claim that it "breaks" them was wrong. Still worth fixing - it is an
    # unnamed CAD body, the same root cause as the missing masses, and it will
    # bite in ROS or anything that assumes ASCII identifiers.
    warn(f"non-ASCII link name(s): {bad_links} - unnamed CAD body; loaders "
         "cope but fix it at source")
else:
    ok("all link names ASCII")
if bad_meshes:
    warn(f"{len(bad_meshes)} mesh filename(s) non-ASCII, e.g. {bad_meshes[0]}")
else:
    ok("all mesh filenames ASCII")

# ---------------------------------------------------------------- limits
print("\n[joint limits vs the physical arm]")
# Walk the kinematic chain from the root. Document order is NOT chain order -
# using it matched 'distal' against 'shoulder_pan'.
by_parent = {}
for j in joints:
    by_parent.setdefault(j.find("parent").get("link"), []).append(j)
rev = []
def descend(link):
    for j in by_parent.get(link, []):
        if j.get("type") in ("revolute", "continuous"):
            rev.append(j)
        descend(j.find("child").get("link"))
descend(roots[0] if roots else next(iter(links)))
print(f"  chain order: {' -> '.join(j.get('name') for j in rev)}")

for idx, (plug_name, lo_exp, hi_exp) in PLUGIN_LIMITS.items():
    if idx > len(rev):
        break
    j = rev[idx - 1]
    lim = j.find("limit")
    if lim is None:
        warn(f"{j.get('name')}: no <limit>")
        continue
    lo = math.degrees(float(lim.get("lower")))
    hi = math.degrees(float(lim.get("upper")))
    span_exp = hi_exp - lo_exp
    tag = ""
    if hi - lo > span_exp * 1.25:
        tag = f"  <-- {(hi-lo)/span_exp:.1f}x wider than the arm allows"
        warn(f"{j.get('name')} [{lo:+.0f},{hi:+.0f}] vs {plug_name} "
             f"[{lo_exp:+.0f},{hi_exp:+.0f}]{tag}")
    elif (lo < 0) != (lo_exp < 0) and abs(lo) > 5 and abs(lo_exp) > 5:
        warn(f"{j.get('name')} [{lo:+.0f},{hi:+.0f}] vs {plug_name} "
             f"[{lo_exp:+.0f},{hi_exp:+.0f}]  <-- sign flipped?")
    else:
        ok(f"{j.get('name'):<16}[{lo:+.0f},{hi:+.0f}] ~ {plug_name} "
           f"[{lo_exp:+.0f},{hi_exp:+.0f}]")

# ---------------------------------------------------------------- meshes
print("\n[mesh files]")
missing = []
refs = {m.get("filename") for m in root.iter("mesh") if m.get("filename")}
for r in refs:
    p = (D / r.replace("package://", "")).resolve()
    if not p.exists():
        missing.append(r)
if missing:
    blocker(f"{len(missing)} referenced mesh(es) not on disk, "
            f"e.g. {missing[:2]}")
else:
    ok(f"all {len(refs)} referenced meshes present")

# ---------------------------------------------------------------- loaders
print("\n[loaders]")
try:
    import mujoco
    if mjcf_p.exists():
        try:
            mm = mujoco.MjModel.from_xml_path(str(mjcf_p))
            ok(f"MuJoCo compiled robot.xml: {mm.nq} dof, {mm.nbody} bodies, "
               f"{mm.ngeom} geoms")
            if mm.ngeom > 200:
                warn(f"{mm.ngeom} geoms - collision is probably 1:1 with "
                     "visual; postprocess_mujoco.py should thin it")
        except Exception as e:
            blocker(f"MuJoCo failed to compile robot.xml: "
                    f"{str(e).splitlines()[0]}")
except ImportError:
    print("  skip     mujoco not installed")

try:
    import placo
    try:
        rw = placo.RobotWrapper(str(D))
        ok(f"Placo loaded the URDF: {rw.model.nq} nq")
    except Exception as e:
        blocker(f"Placo failed to load: {str(e).splitlines()[0]}")
except ImportError:
    print("  skip     placo not installed (needed by the Grabette kinematics)")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 60)
print(f"{len(problems)} blocker(s), {len(warnings)} warning(s)")
if problems:
    for p in problems:
        print(f"  BLOCKER  {p}")
sys.exit(1 if problems else 0)
