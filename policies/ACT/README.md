# ACT — chunk-relative actions

Trains [ACT](https://arxiv.org/abs/2304.13705) on Grabette demonstrations with
chunk-relative actions: every action of a chunk is an offset from the chunk's
first pose, expressed in that pose's frame
([`libs/grabette-chunkrel`](../../libs/grabette-chunkrel)). Stock lerobot
`0.6.0`, no fork. Standalone uv project (own `uv.lock`):

```bash
cd policies/ACT && uv sync
```

## Recipe

**1. Dataset.** A raw Grabette dataset, *not* the output of `convert_dataset.py`:

- `action` is 8-D absolute `[x, y, z, ax, ay, az, proximal, distal]` (rotvec);
- `observation.state` is the 2-D gripper pair (`action[6:8]`). Datasets older
  than `generate_dataset.py` emitting it must have it added first.

Write the chunk-relative action stats for the chunk size you will train with:

```bash
uv run --project ../../libs/grabette-dataset-tools python -c "from pathlib import Path; from grabette_dataset_tools.chunk_relative_stats import write_relative_action_stats as w; w(Path('<dataset root>'), chunk_size=50)"
```

**2. Train.** All `lerobot-train` flags work:

```bash
uv run python train.py --policy.type=act --policy.chunk_size=50 \
    --policy.n_action_steps=50 --dataset.repo_id=<dataset> --dataset.root=<dataset root> ...
```

On HF Jobs, add
`--with "git+https://github.com/pollen-robotics/grabette-policy@develop#subdirectory=libs/grabette-chunkrel"`.
Without a system FFmpeg (macOS, HF Jobs images) add `--dataset.video_backend=pyav`.

**3. Evaluate** with the robot client, in sync mode (no `--async_exec`):

```bash
uv run --project ../../robots/openarm/openarm_gripette_simu --extra eval \
    python ../../robots/openarm/openarm_gripette_simu/examples/evaluate.py \
    --checkpoint <checkpoint>/pretrained_model --chunk_relative auto
```

## What `train.py` refuses

- Stats that are not chunk-relative, or were computed for another `chunk_size`.
- An action that is not 8-D, or a dataset without `observation.state`.
- `--policy.temporal_ensemble_coeff`: it averages chunks with different reference poses.
- Resuming/fine-tuning from a checkpoint that was not trained with this launcher.

Use `n_action_steps` ≥ 40: short horizons replay per-step noise as jitter.

## Tests

```bash
uv run pytest tests
```
