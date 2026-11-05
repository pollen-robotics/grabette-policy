# rebot-gripette-model

Robot model for the **reBot B601-RS + Gripette**: URDF (Placo/IK), MuJoCo XML
(simulation), and mesh assets. Mirrors `openarm_gripette_model` so anything
that takes a model directory — notably `gripette_core.ArmSpec` — can be
pointed at either arm.

```python
from rebot_gripette_model import REBOT_B601RS_DIR, REBOT_B601RS_URDF
```

## Contents

| Path | What |
|---|---|
| `rebot_b601rs/robot.urdf` | URDF. What Placo loads for FK/IK. |
| `rebot_b601rs/robot.xml` | MJCF. What MuJoCo loads. |
| `rebot_b601rs/scene.xml` | MJCF scene, `<include>`s `robot.xml`. |
| `rebot_b601rs/config.json` | onshape-to-robot config — the re-export recipe. |
| `rebot_b601rs/assets/*.stl` | 121 meshes, 67 MB, **stored via git LFS**. |

The Onshape export also emits a `.part` sidecar per mesh. Nothing references
them, so they are not vendored.

## Joints

6 arm joints + the Gripette's `proximal`/`distal`. Limits below are the
**physical** arm's, verified on hardware, and they match the LeRobot plugin.

| # | Joint | Limit (deg) | Motor |
|---|---|---|---|
| 1 | `shoulder_pan` | −145 … +145 | RS-06 |
| 2 | `shoulder_lift` | 0 … +170 | RS-06 |
| 3 | `elbow_flex` | 0 … +200 | RS-06 |
| 4 | `wrist_flex` | −80 … +90 | RS-00 |
| 5 | `wrist_yaw` | −90 … +90 | RS-00 |
| 6 | `wrist_roll` | −130 … +130 | RS-00 |

Direction and scale were verified against the real arm with a live mirror
(read the motors, drive the model, watch for a joint going the wrong way).
**URDF space == motor space**: a command moves both the same way by the same
amount. Two joints (`elbow_flex`, `wrist_flex`) were inverted in an earlier
export and fixed in CAD, so this is a property worth re-checking after any
re-export, not an invariant of the exporter.

## After EVERY re-export, run the postprocess

```bash
python postprocess_rebot.py rebot_gripette_model/rebot_b601rs            # dry run
python postprocess_rebot.py rebot_gripette_model/rebot_b601rs --apply
```

The raw export does not carry masses: the Seeed STEP import has no materials,
so onshape-to-robot assigned mass only to parts from the vendor motor CAD —
all motors correct, ~2 kg of structure missing. The script fills that in from
the old reBot URDF, matched by chain position, under a `max(current, old)`
rule so it can never *delete* known motor mass.

**It patches both `robot.urdf` and `robot.xml`.** That matters: they come from
two separate export passes, Placo reads one and MuJoCo the other. An earlier
version patched the URDF only, which left a 5.35 kg arm in the IK and a
3.40 kg arm in the sim — silently. The script now ends by asserting the two
totals agree, and re-running it is a no-op.

Verify with `validate_model.py`, which checks the things a re-export is known
to regress (topology, Gripette frames, masses, joint limits, mesh presence)
rather than merely that files exist.

## Known limitations

- **Centre of mass is still at the motor**, not the link centroid. Gravity
  torque *magnitude* is roughly right; its *angle dependence* is not — on the
  old URDF the model slope was −0.048 N·m/deg against a measured +0.016 on
  `elbow_flex`, i.e. inverted. Fine for kinematics and for position control,
  where the position gain absorbs it. **Do not use this model for gravity
  compensation or torque feedforward.** The real fix is computing mass, COM
  and inertia together from the STL volumes.
- **Inertia is scaled by the mass ratio**, which assumes the added structure
  is distributed like the motor. Wrong in detail, but dimensionally consistent
  — unlike leaving the tensor alone while quadrupling the mass.
- **`wrist_flex`'s real limit is unmeasured.** −80…+90 is the bare arm's; the
  Gripette changes the envelope and whatever it fouls now sets the true limit.
- **No MuJoCo camera yet** (`ncam == 0`), though all seven Gripette sites are
  present. The camera is added in the sim package, where it can be verified by
  render-diff against the OpenArm sim — a policy conditions on what the camera
  sees, so "looks about right" is not good enough.
- **Collision geometry is 1:1 with visual** — 370 geoms, every visual mesh
  duplicated as a collision mesh. MuJoCo will compile and run it, but contact
  will be needlessly slow and finger/table contact unnecessarily fiddly.
  Thinning it (primitive collision shapes, or convex hulls) belongs with the
  sim package, alongside the camera.
- **Some link and mesh names are non-ASCII** (`实体1`), from unnamed bodies in
  the vendor STEP. MuJoCo and Placo both load them fine; fix at the CAD source
  if ROS or another ASCII-assuming toolchain ever enters the picture.
