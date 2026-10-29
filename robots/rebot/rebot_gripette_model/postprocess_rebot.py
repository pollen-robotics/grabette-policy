#!/usr/bin/env python3
"""Re-apply patches the raw Onshape export does not produce. Idempotent.

Same role as OpenArm's postprocess_mujoco.py, and for the same reason: their
model README warns a re-export "can silently regress collision classes, gains,
axes, and joint ranges". Run this after EVERY export.

Patches masses in BOTH robot.urdf and robot.xml. That "both" is the whole
point: Placo reads the URDF and MuJoCo reads the MJCF, they come from two
separate onshape-to-robot passes, and an earlier version of this script
patched the URDF only. The result was a 5.35 kg arm in the IK and a 3.40 kg
arm in the sim, with nothing to complain about it.

The camera and collision classes are separate and not done yet.

ACTUATOR GAINS
    The export ships kp=50 on every position actuator - a generic default, not
    a measured property of anything. The OpenArm's model uses 500, with 120 on
    the two gripper DOFs, and runs the same policies through the same arm
    service.

    kp=50 does not merely make the arm soft, it breaks the control loop: the
    simulated arm settles 55 mm from the pose it is commanded to hold, while
    arm_service rejects any command whose target runs more than 50 mm ahead of
    the measured pose (it reads that as contact). The guard therefore fires on
    physics alone, before a policy has asked for anything, and episodes die at
    random steps.

    THE TARGET HERE IS TRACKING, NOT FIDELITY. The real arm droops ~2.5 deg at
    the elbow and kp=150 would reproduce that nicely - but a simulator whose
    job is to test a policy should put the arm where the policy asked, so that
    the executed trajectory IS the commanded one. Measured across a sweep
    (static control-frame error at the folded and grasp poses; overshoot and
    settling on a 20 deg elbow step):

        kp=50    55 / 48 mm    the shipped value, over the cap at rest
        kp=250   10 / 16 mm
        kp=500    5 /  9 mm    no overshoot, 0.06 s settle
        kp=750    3 /  6 mm    better, but nearer the edge
        kp=1000               DIVERGES at this armature

    500 it is: a 6-10x margin under the cap, no overshoot, and the value the
    sibling arm already runs. armature stays at the export's 0.005 - raising it
    to the OpenArm's 0.1 only slowed convergence without helping.

    The gripper keeps its own kp=120, as on the OpenArm. Letting the fingers
    inherit 500 would quadruple a gain picked for squeezing a soft object.

MASSES
    The Seeed STEP import carries no materials, so onshape-to-robot gave mass
    only to parts that came from the vendor motor CAD:

        axis_1  0.621163 = 1x RS06        axis_4/5/6  0.310818 = 1x RS00 each
        axis_3  1.242330 = 2x RS06        axis_2      0 until set by hand

    i.e. all motors correct, all structure missing - about 2 kg of it. The
    structural estimates here come from the OLD reBot URDF
    (vectorBH6/reBotArm_control_py) matched by chain position.

    THE RULE IS max(current, old), NOT overwrite. Applied per file against
    that file's own current mass, which is what makes it safe to run on both:
    an already-patched URDF is a fixed point (max(1.252, 1.252) = 1.252) while
    the raw MJCF converges to the same target (max(0.311, 1.252) = 1.252). So
    the two files agree afterwards without needing a third source of truth,
    and re-running changes nothing. The old URDF assigns motors
    to a different link than this export does, so two of its values are
    LIGHTER than the motor we know is present (link5 0.2012 kg against an RS00
    at 0.3108). Overwriting there would delete known mass. Current values are a
    hard lower bound.

    Inertia is scaled by the same ratio as the mass. That assumes the added
    structure has the same spatial distribution as the motor, which is wrong in
    detail but at least dimensionally consistent - leaving the tensor untouched
    while quadrupling the mass would not be.

    !! CENTRE OF MASS IS NOT CORRECTED. It stays where the export put it, at
    the motor rather than the link centroid. Gravity torque MAGNITUDE will be
    roughly right; its ANGLE DEPENDENCE will not. That is exactly the defect
    measured on the old URDF (model slope -0.048 N.m/deg against a true
    +0.016 on elbow_flex). Do not trust this model for gravity compensation.
    Computing mass, COM and inertia together from the STL volumes is the real
    fix.
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Old reBot URDF masses, matched by chain position. Structural estimates only.
OLD_BY_CHAIN = [1.1774, 0.4655, 1.5520, 1.2520, 0.4600, 0.2012, 0.1000]

# Actuator gains - see ACTUATOR GAINS above. The arm's lives on the model's
# default class; the gripper overrides it per actuator.
MJCF_CLASS = "rebot_gripette"
ARM_KP = 500
GRIPPER_KP = 120
GRIPPER_ACTUATORS = ("proximal", "distal")

ap = argparse.ArgumentParser()
ap.add_argument("model_dir", type=Path)
ap.add_argument("--apply", action="store_true",
                help="write the file; default is a dry run")
ap.add_argument("--keep", nargs="*", default=["axis_2"],
                help="links whose mass was set by hand - never overwrite. "
                     "axis_2 by default: it has no motor, so its value is "
                     "pure structure, whereas the old URDF's link1 likely "
                     "bundles a motor in.")
args = ap.parse_args()

urdf = args.model_dir / "robot.urdf"
tree = ET.parse(urdf)
root = tree.getroot()
links = {l.get("name"): l for l in root.findall("link")}
joints = root.findall("joint")

by_parent = {}
for j in joints:
    by_parent.setdefault(j.find("parent").get("link"), []).append(j)
children = {j.find("child").get("link") for j in joints}
chain_root = next(n for n in links if n not in children)

# Walk the main serial chain: at each step follow the child that itself has
# children, so we stay on the arm rather than branching into frame markers.
chain = [chain_root]
node = chain_root
while True:
    kids = [j.find("child").get("link") for j in by_parent.get(node, [])]
    nxt = [k for k in kids if by_parent.get(k)]
    if not nxt:
        break
    node = nxt[0]
    chain.append(node)

print(f"chain ({len(chain)}): {' -> '.join(chain)}\n")
print(f"{'link':<42}{'current':>10}{'old':>9}{'new':>10}  note")

total_before = total_after = 0.0
changes = []
for i, name in enumerate(chain):
    l = links[name]
    inert = l.find("inertial")
    if inert is None:
        continue
    mnode = inert.find("mass")
    cur = float(mnode.get("value"))
    total_before += cur
    old = OLD_BY_CHAIN[i] if i < len(OLD_BY_CHAIN) else None

    if name in args.keep:
        new, note = cur, "kept: set by hand"
    elif old is None:
        new, note = cur, "beyond the old chain, kept"
    elif old > cur:
        new, note = old, f"structure added (+{old-cur:.3f} kg)"
    else:
        new, note = cur, "kept: old is BELOW the motor mass"

    total_after += new
    print(f"{name:<42}{cur:>10.5f}{(old if old else 0):>9.4f}{new:>10.5f}  {note}")
    if abs(new - cur) > 1e-9:
        changes.append((name, cur, new))

# Links off the main chain (gripette fingers) are already correct.
for name, l in links.items():
    if name in chain:
        continue
    inert = l.find("inertial")
    if inert is None:
        continue
    m = float(inert.find("mass").get("value"))
    total_before += m
    total_after += m

print(f"\ntotal {total_before:.4f} -> {total_after:.4f} kg "
      f"(+{total_after-total_before:.4f})")

if not changes:
    print("URDF: nothing to change - already patched")

# ------------------------------------------------------------------ MJCF
# Same target masses, applied to the MuJoCo file. Bodies are NESTED here, so
# the chain is the nesting order rather than a parent/child table, but the
# rule and the target are identical - see the note on max(current, old) above.
mjcf_p = args.model_dir / "robot.xml"
if not mjcf_p.exists():
    print(f"\nno {mjcf_p} - skipping MJCF")
    raise SystemExit(0)

mtree = ET.parse(mjcf_p)
mroot = mtree.getroot()

# Walk down the body nesting, following the child that itself has children,
# so we stay on the arm rather than branching into the gripette fingers.
mchain = []
node = mroot.find("worldbody")
while True:
    kids = node.findall("body")
    if not kids:
        break
    nxt = [k for k in kids if k.findall("body")] or kids
    node = nxt[0]
    mchain.append(node)

# ---- base weld: the root body must not be free ----
_wb = mroot.find("worldbody")
_free = []
for body in _wb.findall("body"):
    fj = body.find("freejoint")
    if fj is not None:
        _free.append((body, fj))
if _free:
    for body, _ in _free:
        print(f"\nbase weld: {body.get('name')} has a <freejoint> - the robot "
              f"is a free body and its base can move. Removing.")
else:
    print("\nbase weld: root body already fixed - nothing to change")

# ---- actuator gains: the export's kp=50 breaks the control loop ----
# See ACTUATOR GAINS above. The arm's gain lives on the model's default class;
# the gripper's two DOFs override it per actuator, as on the OpenArm.
_gain_changes = []
_cls = next((c for c in mroot.iter("default") if c.get("class") == MJCF_CLASS), None)
if _cls is None:
    print(f"\ngains: no <default class='{MJCF_CLASS}'> - cannot set the arm gain")
else:
    _pos = _cls.find("position")
    if _pos is None:
        print(f"\ngains: class '{MJCF_CLASS}' has no <position> default")
    elif _pos.get("kp") != str(ARM_KP):
        _gain_changes.append((_pos, "kp", _pos.get("kp"), str(ARM_KP)))

for _act in mroot.iter("position"):
    if _act.get("name") in GRIPPER_ACTUATORS and _act.get("kp") != str(GRIPPER_KP):
        _gain_changes.append((_act, "kp", _act.get("kp"), str(GRIPPER_KP)))

if _gain_changes:
    for el, attr, cur, new in _gain_changes:
        what = el.get("name") or f"class {MJCF_CLASS} default"
        print(f"gains: {what} {attr} {cur} -> {new}")
else:
    print("gains: already set - nothing to change")

print(f"\nMJCF chain ({len(mchain)}): "
      f"{' -> '.join(b.get('name') for b in mchain)}")
print(f"{'body':<42}{'current':>10}{'old':>9}{'new':>10}  note")

m_before = m_after = 0.0
m_changes = []
for i, body in enumerate(mchain):
    inert = body.find("inertial")
    if inert is None:
        continue
    name = body.get("name")
    cur = float(inert.get("mass"))
    m_before += cur
    old = OLD_BY_CHAIN[i] if i < len(OLD_BY_CHAIN) else None
    if name in args.keep:
        new, note = cur, "kept: set by hand"
    elif old is None:
        new, note = cur, "beyond the old chain, kept"
    elif old > cur:
        new, note = old, f"structure added (+{old - cur:.3f} kg)"
    else:
        new, note = cur, "kept: old is BELOW the motor mass"
    m_after += new
    print(f"{name:<42}{cur:>10.5f}{(old if old else 0):>9.4f}{new:>10.5f}  {note}")
    if abs(new - cur) > 1e-9:
        m_changes.append((body, cur, new))

# Bodies off the main chain (gripette fingers) are already correct.
_chain_ids = {id(b) for b in mchain}
for body in mroot.iter("body"):
    if id(body) in _chain_ids:
        continue
    inert = body.find("inertial")
    if inert is not None and inert.get("mass") is not None:
        m = float(inert.get("mass"))
        m_before += m
        m_after += m

print(f"\ntotal {m_before:.4f} -> {m_after:.4f} kg (+{m_after - m_before:.4f})")

if not m_changes:
    print("MJCF: no mass change - already patched")

if not (changes or m_changes or _free or _gain_changes):
    print("\nBoth files already patched. Nothing to do.")
    raise SystemExit(0)

if not args.apply:
    print(f"\nDRY RUN ({len(changes)} URDF + "
          f"{len(m_changes) + len(_free) + len(_gain_changes)} "
          f"MJCF change(s)). "
          "Re-run with --apply to write.")
    raise SystemExit(0)

# ---- writes, both files, only past this point ----
if changes:
    backup = urdf.with_suffix(".urdf.orig")
    if not backup.exists():
        shutil.copy2(urdf, backup)
        print(f"backup -> {backup}")
    for name, cur, new in changes:
        inert = links[name].find("inertial")
        inert.find("mass").set("value", f"{new:.6f}")
        ratio = new / cur if cur > 0 else 1.0
        it = inert.find("inertia")
        if it is not None and cur > 0:
            for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                if it.get(k) is not None:
                    it.set(k, f"{float(it.get(k)) * ratio:.9g}")
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    print(f"written {urdf}")

if m_changes or _free or _gain_changes:
    mbackup = mjcf_p.with_suffix(".xml.orig")
    if not mbackup.exists():
        shutil.copy2(mjcf_p, mbackup)
        print(f"backup -> {mbackup}")
    for body, cur, new in m_changes:
        inert = body.find("inertial")
        inert.set("mass", f"{new:.6f}")
        ratio = new / cur if cur > 0 else 1.0
        # MuJoCo accepts either fullinertia (ixx iyy izz ixy ixz iyz) or
        # diaginertia (ixx iyy izz). Scale whichever is present, by the same
        # ratio as the URDF, for the same reason: leaving the tensor alone
        # while quadrupling the mass would not even be dimensionally sane.
        for attr in ("fullinertia", "diaginertia"):
            v = inert.get(attr)
            if v:
                inert.set(attr, " ".join(f"{float(x) * ratio:.9g}"
                                         for x in v.split()))
    for body, fj in _free:
        body.remove(fj)
        print(f"removed <freejoint> from {body.get('name')}")
    for el, attr, cur, new in _gain_changes:
        el.set(attr, new)
        print(f"set {attr}={new} on {el.get('name') or 'the class default'}")
    mtree.write(mjcf_p, encoding="utf-8", xml_declaration=True)
    print(f"written {mjcf_p}")

# The two files must now agree, or the whole exercise was pointless.
u_total = sum(float(l.find("inertial").find("mass").get("value"))
              for l in links.values() if l.find("inertial") is not None)
if abs(u_total - m_after) > 1e-4:
    print(f"\nWARNING: URDF total {u_total:.4f} kg != MJCF total "
          f"{m_after:.4f} kg - they still disagree")
else:
    print(f"\nURDF and MJCF agree: {u_total:.4f} kg")
