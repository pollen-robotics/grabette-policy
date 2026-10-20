# grabette-policy

Policy learning for [GRABETTE](https://github.com/pollen-robotics/grabette):
training integrations, the action representations and diagnostics they use,
and the OpenArm-Gripette robot they are evaluated and deployed on.

Datasets come from the GRABETTE recording rig
([grabette](https://github.com/pollen-robotics/grabette)) and are turned into
LeRobot datasets by the SLAM pipeline
([grabette-slam](https://github.com/pollen-robotics/grabette-slam)).

| Path | What |
|---|---|
| [`packages/grabette-dataset-tools`](packages/grabette-dataset-tools) | Grasp projection, chunk-relative stats, raw-capture normalization, publish gate |
| [`packages/grabette-chunkrel`](packages/grabette-chunkrel) | Chunk-relative action representation + its LeRobot processor steps |
| [`packages/grabette-attention`](packages/grabette-attention) | Offline attention maps and view-ablation diagnostics |
| [`integrations/DiffusionPolicy`](integrations/DiffusionPolicy) | Diffusion Policy data prep and training (own `uv.lock`) |
| [`integrations/Pi05`](integrations/Pi05) | Pi0.5 fine-tuning (own `uv.lock`) |
| [`integrations/openarm`](integrations/openarm) | OpenArm-Gripette: model, MuJoCo sim, real-arm driver, teleop bridge |

## Development

Requires [uv](https://docs.astral.sh/uv/). `packages/*` and
`integrations/openarm/*` form one uv workspace with a single lock; DiffusionPolicy
and Pi05 are standalone projects with their own lock.

```bash
uv sync --all-packages                       # the workspace
cd integrations/DiffusionPolicy && uv sync   # a standalone integration
```

The OpenArm meshes are in Git LFS: install [git-lfs](https://git-lfs.com) before cloning.

### The grabette pin

The gripper's own package, `gripette`, comes from the grabette repo at a pinned
commit (`[tool.uv.sources]` in the root `pyproject.toml`). The sim's
`gripper.proto` must match gripette's; CI checks it against the same commit.
To bump: update the `rev`, run `uv lock`, and let CI's proto-sync job confirm.
