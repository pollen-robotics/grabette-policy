# grabette-dataset-tools

Training-side transforms for GRABETTE LeRobot datasets. They run on a dataset
**after** it has been built from raw recordings by the SLAM pipeline
([grabette-slam](https://github.com/pollen-robotics/grabette-slam)), and
before training.

```bash
uv sync --package grabette-dataset-tools
uv run --package grabette-dataset-tools --extra test pytest libs/grabette-dataset-tools/tests
```

## Grasp projection

Re-expresses the gripper's two joint angles as `(strategy, closure)`, so a
policy can command a full close and let the object stop the fingers. The
design, and how to record for it, are in [`docs/grasp_projection.md`](docs/grasp_projection.md)
and [`docs/grasp_projection_recording_procedure.md`](docs/grasp_projection_recording_procedure.md).

```bash
uv run python -m grabette_dataset_tools.grasp_projection_convert --help
```

`grabette_dataset_tools.grasp_projection` reads the joint limits from
`gripette.config` (the gripper's own package, in the
[grabette](https://github.com/pollen-robotics/grabette) repo), so the converter
and the eval loop agree on what a full close means.

## Normalize a raw capture

Raw captures recorded with a camera prefix (e.g. `observation.images.right_cam0`)
and no state column do not fit the training scripts. `normalize_raw` renames the
camera feature everywhere it appears (info.json, video dir, episode parquet
columns, stats.json), optionally strips the prefix from every feature name,
and adds the gripper-only `observation.state` the training scripts expect:

```bash
uv run python -m grabette_dataset_tools.normalize_raw \
    --src_root /path/to/raw --dst_root /path/to/normalized --repo_id <user>/<dataset> \
    --camera right_cam0 --strip_prefix right_ --task "pick the sugar cube"
```

## Publish gate

Before pushing a dataset, `check_publish` (a library function, no CLI yet)
verifies info.json, the task string (natural language, not a slug), the
gripper channel naming and unit range, the grasp-projection sidecar, the
stats file and its representation provenance, the dataset card (viewer link),
the LeRobot tag, and the fps (`deep=True` decodes the videos):

```python
from grabette_dataset_tools.checks.publish import check_publish
status = check_publish("<user>/<dataset>")   # Hub id or local path -> dict of findings
```

## Relative action stats (optional chunk-relative mode)

Chunk-relative training (`GRABETTE_CHUNK_RELATIVE=1`, see
[`../grabette-chunkrel/docs/relative_actions_lerobot_native.md`](../grabette-chunkrel/docs/relative_actions_lerobot_native.md))
normalizes actions with statistics of the *relative* representation, not of the
stored deltas. `write_relative_action_stats` computes them and stamps the stats
file with a provenance marker; the training guard refuses a stats/representation
mismatch. Deltas (the default) do not need this step.

```python
from grabette_dataset_tools.chunk_relative_stats import write_relative_action_stats
write_relative_action_stats("/path/to/dataset", chunk_size=50)
```
