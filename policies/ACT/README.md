# ACT — chunk-relative actions

Trains [ACT](https://arxiv.org/abs/2304.13705) on Grabette demonstrations with
chunk-relative actions: every action of a chunk is an offset from the chunk's
first pose, expressed in that pose's frame
([`libs/grabette-chunkrel`](../../libs/grabette-chunkrel)). Stock lerobot
`0.6.1`, no fork. Standalone uv project (own `uv.lock`):

```bash
cd policies/ACT && uv sync
```

## Recipe

**1. Dataset.** A raw Grabette dataset, *not* the output of `convert_dataset.py`:

- `action` is 8-D absolute `[x, y, z, ax, ay, az, proximal, distal]` (rotvec);
- `observation.state` is the 2-D gripper pair (`action[6:8]`). Datasets older
  than `generate_dataset.py` emitting it must have it added first.

**One-shot prep:** `run_pipeline.sh` cleans, normalises, optionally projects
the gripper, resizes the images to 648x486 (the Gripette's stream size: ACT
has no internal resize; `evaluate.py` also rescales live frames to the
checkpoint's size), decode-checks the videos and writes the stats below, then
prints the train command:

```bash
./run_pipeline.sh simheo/red-cube-blue-box --camera right_cam0 --strip-prefix right_
```

`--grasp-projection` re-expresses the gripper as `(strategy, closure)` so the
policy commands a full close. If the robot then **closes too early** (operators who
close while still approaching), add `--close-at 0.8 --close-stop-speed 0.03` to
fire the full close when the hand reaches the object; see
[When the full close fires](../../libs/grabette-dataset-tools/docs/grasp_projection.md#when-the-full-close-fires).
`./run_pipeline.sh --help` lists every option.

The stats' rotation cap follows `fps` in `meta/info.json`, so it must be the
real rate. By hand, write the chunk-relative action stats for the chunk size you will train with:

```bash
uv run --project ../../libs/grabette-dataset-tools python -c "from pathlib import Path; from grabette_dataset_tools.chunk_relative_stats import write_relative_action_stats as w; w(Path('<dataset root>'), chunk_size=50)"
```

**2. Train.** All `lerobot-train` flags work except `--job.target` (use `hf jobs uv run`, below):

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
- Resuming/fine-tuning from a checkpoint that was not trained with this launcher
  (a stock ACT checkpoint cannot be fine-tuned this way). The stats and config
  checks also run when resuming or fine-tuning.

Use `n_action_steps` ≥ 40: short horizons replay per-step noise as jitter.

## Tests

```bash
uv run pytest tests
```
