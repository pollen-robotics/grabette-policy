# gripette-core

Shared core for **Gripette-equipped robot arms**: the gRPC contract, Placo
FK/IK, the 6D rotation helpers, and the IK feasibility filter.

No MuJoCo, no OpenCV, and placo only behind the `[kinematics]` extra. The
real-hardware drivers import this on robots where the simulation stack is
neither wanted nor installable — that constraint is the reason the package
exists, so please keep it true.

## Why

One trained policy should drive either arm. That works only if both arms
present the *same* interface: the same proto, the same control frame, the same
FK/IK conventions. Putting those in one place is what makes "same interface" a
fact rather than an aspiration.

The Gripette is the same physical gripper on every arm, so the frame names
(`camera`, `gripper_center`, `oak_l`) are constants here, not per-arm
configuration. `CONTROL_FRAME` is `oak_l` and is THE frame the whole pipeline
agrees on.

## Binding an arm

Everything arm-specific goes in an `ArmSpec`:

```python
from gripette_core import ArmSpec, Kinematics

SPEC = ArmSpec(
    model_dir="/path/to/urdf/dir",
    joint_names=("j1", "j2", ...),       # MuJoCo order; this IS the wire order
    masked_dofs=("proximal", "distal"),  # DOFs IK must not drive
)
kin = Kinematics(SPEC)
```

Arm packages subclass to pre-bind their spec, so callers keep a zero-argument
constructor — see `openarm_gripette_simu.kinematics.Kinematics`.

`masked_dofs` matters: without it the solver can "reach" a target by closing
the fingers.

## Install shapes

| Install | Gets you | For |
|---|---|---|
| `gripette-core` | proto stubs + rotation helpers | clients: `teleop_bridge`, eval |
| `gripette-core[kinematics]` | the above + placo FK/IK | sim packages, arm drivers |

The `[kinematics]` extra pins `cmeel-urdfdom==4.0.1` / `cmeel-tinyxml2==10.0.0`
as real dependencies. The workspace root carries the same pins as
`[tool.uv] constraint-dependencies`, but that is uv-only *and* root-only — it
does not travel in wheel metadata, so without these an install outside this
workspace resolves cmeel-urdfdom 6.x and `import placo` fails on a missing
`.so.4`.

## Layout

| Path | What |
|---|---|
| `gripette_core/kinematics.py` | `ArmSpec`, `Kinematics`, frame constants |
| `gripette_core/rotation.py` | 6D continuous rotation (Zhou et al., CVPR 2019) |
| `gripette_core/ik_feasibility.py` | reachability filter for cross-embodiment data |
| `gripette_core/proto/` | generated stubs — the only copy in this repo |
| `proto/` | `.proto` sources |

## Protos

`gripette_core/proto/` holds the generated stubs. `openarm_gripette_simu.proto`
re-exports them rather than shipping a second copy, which keeps both import
paths resolving to one module object.

`gripper.proto` must stay field-for-field identical to the gripper's own copy
in the `grabette` repo — the eval client talks to both the sim server and the
real device over the wire. The **`proto-sync` CI job** enforces that: it fetches
`packages/gripette/proto/gripper.proto` from grabette at the commit this
workspace pins and diffs it against `proto/gripper.proto` here. Bump the pin
and this file together.

Do not rely on protobuf to catch a divergence at runtime. It keys its
descriptor pool by `.proto` filename but **deduplicates byte-identical**
descriptors, so two copies coexist silently and only raise `duplicate file
name` once they already differ — in production, not in CI.

Regenerate from `libs/gripette-core/` with
`uv run --extra dev python generate_proto.py`. It covers `gripper.proto` only;
`arm.proto`'s stubs are committed and not rebuilt by that script.
