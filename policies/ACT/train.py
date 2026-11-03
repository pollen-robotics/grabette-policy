# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["lerobot[training]==0.6.0"]
# ///
"""ACT launcher with chunk-relative actions: stock `lerobot-train` plus the
grabette-chunkrel steps.

Stock ACT predicts absolute actions and its processor pipeline has no
relative-action slot. This wrapper inserts `ChunkRelativeActionsStep` before
the normaliser and its inverse after the unnormaliser
(`grabette_chunkrel.chunk_relative_processor.insert_chunk_relative`), so ACT
learns offsets from each chunk's first pose, composed into that pose's frame.
See `libs/grabette-chunkrel/README.md`.

Before building the pipeline it refuses, rather than silently training wrong:
  - dataset action stats that are not chunk-relative, or were computed for
    another chunk_size (`write_relative_action_stats`);
  - a non-8-D action (the 11-D per-step-delta datasets from convert_dataset.py);
  - temporal ensembling (it averages chunks with different reference poses).

All lerobot-train flags work unchanged, except --job.target (refused: the remote
job would run stock lerobot-train; use `hf jobs uv run` as below).

Usage (full recipe in README.md):
  uv run python train.py --policy.type=act --policy.chunk_size=50 \\
      --policy.n_action_steps=50 --dataset.repo_id=<raw 8-D dataset> ...

HF Jobs uploads this one file, so the chunkrel package must be added with:
  --with "git+https://github.com/pollen-robotics/grabette-policy@develop#subdirectory=libs/grabette-chunkrel"
"""

import lerobot.scripts.lerobot_train as lerobot_train
from lerobot.policies import factory as policy_factory

_original = policy_factory.make_pre_post_processors

# Marker key that `write_relative_action_stats` leaves in meta/stats.json.
_RELATIVE_MARKER = "action_relative_meta"
# x y z + rotvec (3) + 2 gripper dims.
_ACTION_DIM = 8


def _as_int(value):
    """Unwrap a stats leaf (a 1-element numpy array after lerobot's stats
    loading) to a plain int."""
    flat = getattr(value, "flat", None)
    return int(next(iter(flat))) if flat is not None else int(value)


def check_relative_stats(dataset_stats, chunk_size):
    """Refuse absolute stats, or chunk-relative stats computed for another
    chunk_size: both normalise with the wrong scale and the loss won't show it."""
    meta = (dataset_stats or {}).get(_RELATIVE_MARKER)
    if meta is None:
        raise RuntimeError(
            f"the dataset's action stats are ABSOLUTE (no '{_RELATIVE_MARKER}' in "
            "meta/stats.json). Run grabette_dataset_tools.chunk_relative_stats."
            f"write_relative_action_stats(root, chunk_size={chunk_size}) first."
        )
    stats_chunk = _as_int(meta["chunk_size"])
    if stats_chunk != chunk_size:
        raise RuntimeError(
            f"action stats were computed for chunk_size={stats_chunk} but the policy "
            f"uses chunk_size={chunk_size}. Relative-action stats scale with the "
            "horizon; recompute them for this chunk size."
        )


def check_act_config(policy_cfg):
    """Refuse configs this launcher cannot train correctly."""
    if policy_cfg.type != "act":
        raise RuntimeError(f"this launcher trains act only, got --policy.type={policy_cfg.type}")
    if policy_cfg.temporal_ensemble_coeff is not None:
        raise RuntimeError(
            "temporal ensembling is not supported with chunk-relative actions: it "
            "averages chunks expressed relative to different reference poses."
        )
    if "observation.state" not in policy_cfg.input_features:
        raise RuntimeError(
            "ACT needs observation.state (the 2-D gripper pair); this dataset has "
            "none. It predates generate_dataset.py emitting it: add it as action[6:8]."
        )
    shape = tuple(policy_cfg.output_features["action"].shape)
    if shape != (_ACTION_DIM,):
        raise RuntimeError(
            f"expected an {_ACTION_DIM}-D action (x y z, rotvec, 2 gripper dims), got "
            f"{shape}. The 11-D per-step-delta datasets from convert_dataset.py are "
            "not chunk-relative; train on the raw dataset."
        )


def _fresh_pipeline(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
    # Imported here so a missing package fails with a clear ImportError (see
    # the HF Jobs note above). The import also registers the steps, which a
    # saved pipeline needs to load.
    from grabette_chunkrel.chunk_relative_processor import (
        ChunkRelativeActionsStep,
        insert_chunk_relative,
    )

    check_act_config(policy_cfg)
    # On resume/fine-tune lerobot replaces the saved normaliser stats with the
    # current dataset's (normalizer_processor override), so check those.
    override = (kwargs.get("preprocessor_overrides") or {}).get("normalizer_processor", {})
    check_relative_stats(kwargs.get("dataset_stats") or override.get("stats"), policy_cfg.chunk_size)

    if pretrained_path is not None:
        # Resume or fine-tune: the saved pipeline already carries the steps.
        pre, post = _original(policy_cfg, pretrained_path=pretrained_path,
                              pretrained_revision=pretrained_revision, **kwargs)
        if not any(isinstance(s, ChunkRelativeActionsStep) for s in pre.steps):
            raise RuntimeError(
                f"the processor pipeline at {pretrained_path} has no chunk-relative "
                "steps: it was not trained with this launcher."
            )
        return pre, post

    pre, post = _original(policy_cfg, **kwargs)
    return insert_chunk_relative(pre, post)


def _refuse_remote(cfg):
    # --job.target makes lerobot-train submit a pod that runs PLAIN
    # lerobot-train, without this wrapper or grabette-chunkrel: it would train
    # on absolute actions normalised with relative stats, silently.
    raise RuntimeError(
        "--job.target is not supported: the remote job would run stock lerobot-train "
        "without the chunk-relative steps. Use `hf jobs uv run ... --with "
        "<grabette-chunkrel> train.py` instead (see README.md)."
    )


# Patch both the factory and the reference lerobot_train imported.
policy_factory.make_pre_post_processors = _fresh_pipeline
lerobot_train.make_pre_post_processors = _fresh_pipeline
lerobot_train.submit_to_hf = _refuse_remote

if __name__ == "__main__":
    lerobot_train.main()
