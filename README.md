# grabette-policy

Policy learning for [GRABETTE](https://github.com/pollen-robotics/grabette):
training recipes, the action representations and diagnostics they use,
and the two Gripette-equipped robots they are evaluated and deployed on.

Datasets come from the GRABETTE recording rig
([grabette](https://github.com/pollen-robotics/grabette)) and are turned into
LeRobot datasets by the SLAM pipeline
([grabette-slam](https://github.com/pollen-robotics/grabette-slam)).

| Path | What |
|---|---|
| [`libs/grabette-dataset-tools`](libs/grabette-dataset-tools) | Grasp projection, chunk-relative stats, raw-capture normalization, publish gate |
| [`libs/grabette-chunkrel`](libs/grabette-chunkrel) | Chunk-relative action representation + its LeRobot processor steps |
| [`libs/grabette-attention`](libs/grabette-attention) | Offline attention maps and view-ablation diagnostics |
| [`policies/DiffusionPolicy`](policies/DiffusionPolicy) | Diffusion Policy data prep and training (own `uv.lock`) |
| [`policies/Pi05`](policies/Pi05) | Pi0.5 fine-tuning (own `uv.lock`) |
| [`libs/gripette-core`](libs/gripette-core) | Shared across both arms and sim/real: proto, kinematics, rotation, the `ArmService` implementation. Imports no mujoco — the hardware drivers depend on it |
| [`libs/gripette-sim`](libs/gripette-sim) | Shared MuJoCo pieces: simulation wrapper, fisheye camera, gripper service, obstacle clearance, lift tracking |
| [`policies/DiffusionPolicy`](policies/DiffusionPolicy) | Diffusion Policy data prep and training (own `uv.lock`) |
| [`policies/Pi05`](policies/Pi05) | Pi0.5 fine-tuning (own `uv.lock`) |
| [`robots/openarm`](robots/openarm) | OpenArm-Gripette (7-DoF, torso-mounted): model, MuJoCo sim, real-arm driver, teleop bridge |
| [`robots/rebot`](robots/rebot) | reBot-Gripette (6-DoF, bench-mounted): model, MuJoCo sim, real-arm driver |

Both arms serve the **same** `ArmService` from `gripette-core`, in simulation
and on hardware, so one eval script drives any of the four combinations via
`--arm_addr` / `--gripper_addr`.

## Development

Requires [uv](https://docs.astral.sh/uv/). `libs/*` and `robots/*/*` form one
uv workspace with a single lock; DiffusionPolicy and Pi05 are standalone
projects with their own lock.

```bash
uv sync --all-packages                       # the workspace
cd policies/DiffusionPolicy && uv sync       # a standalone policy project
```

Both arms' meshes are in Git LFS (OpenArm 105 STLs, reBot 121): install
[git-lfs](https://git-lfs.com) before cloning.

### The grabette pin

The gripper's own package, `gripette`, comes from the grabette repo at a pinned
commit (`[tool.uv.sources]` in the root `pyproject.toml`). The sim's
`gripper.proto` must match gripette's; CI checks it against the same commit.
To bump: update the `rev`, run `uv lock`, and let CI's proto-sync job confirm.
