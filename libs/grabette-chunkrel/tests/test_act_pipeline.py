"""Integration: our steps inserted into the REAL ACT processor pipeline.

Stock ACT has no relative-action slot to replace (unlike pi05), so
`insert_chunk_relative` adds one. What would fail silently if it went wrong:

  1. POSITION. The forward step must run before the normaliser (the stats
     describe relative actions) and the inverse after the unnormaliser.
  2. ROUND TRIP. pre then post must give back the absolute poses.
  3. DOUBLE INSTALL. Resuming a run must not stack a second pair of steps.
"""

import numpy as np
import pytest
import torch

from grabette_chunkrel.chunk_relative import to_chunk_relative

pytest.importorskip("lerobot", reason="needs lerobot")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402

from grabette_chunkrel.chunk_relative_processor import insert_chunk_relative  # noqa: E402

K, B, ADIM, SDIM = 50, 2, 8, 2
IMG = "observation.images.cam0"


def absolute_batch(seed=0):
    r = np.random.default_rng(seed)
    pos = np.cumsum(r.normal(scale=0.01, size=(B, K, 3)), axis=1) + np.array([3.0, -1.0, 0.5])
    rot = np.cumsum(r.normal(scale=0.02, size=(B, K, 3)), axis=1) + np.array([0.4, -2.0, 1.1])
    grip = np.abs(r.normal(scale=0.2, size=(B, K, 2)))
    return np.concatenate([pos, rot, grip], axis=2)


def stats_of(a):
    return {"mean": torch.tensor(a.mean(0), dtype=torch.float32),
            "std": torch.tensor(a.std(0) + 1e-8, dtype=torch.float32),
            "min": torch.tensor(a.min(0), dtype=torch.float32),
            "max": torch.tensor(a.max(0), dtype=torch.float32)}


def act_pipelines():
    cfg = ACTConfig(
        device="cpu",
        chunk_size=K,
        n_action_steps=K,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(SDIM,)),
            IMG: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ADIM,))},
    )
    rel = np.concatenate([to_chunk_relative(c)[0] for c in absolute_batch()])
    stats = {
        "action": stats_of(rel),
        "observation.state": stats_of(np.random.default_rng(1).random((16, SDIM))),
        IMG: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)},
    }
    return make_act_pre_post_processors(cfg, stats)


def batch(absolute):
    return {
        "observation.state": torch.full((B, SDIM), 0.5),
        IMG: torch.rand(B, 3, 96, 96),
        "action": torch.tensor(absolute, dtype=torch.float32),
    }


def names(pipe):
    return [type(s).__name__ for s in pipe.steps]


def test_insert_places_steps_around_normaliser():
    pre, post = act_pipelines()
    pre2, post2 = insert_chunk_relative(pre, post)
    assert names(pre2) == [
        "RenameObservationsProcessorStep", "AddBatchDimensionProcessorStep",
        "ChunkRelativeActionsStep", "DeviceProcessorStep", "NormalizerProcessorStep"]
    assert names(post2) == [
        "UnnormalizerProcessorStep", "AbsoluteFromChunkRelativeStep", "DeviceProcessorStep"]
    assert post2.steps[1].relative_step is pre2.steps[2]
    assert len(pre.steps) == 4 and len(post.steps) == 2, "input pipelines must be untouched"


def test_roundtrip_through_act_pipeline():
    pre, post = insert_chunk_relative(*act_pipelines())
    absolute = absolute_batch()
    normalised = pre(batch(absolute))["action"]
    recovered = post(normalised)
    torch.testing.assert_close(recovered, torch.tensor(absolute, dtype=torch.float32),
                               atol=1e-5, rtol=0)


def test_normalised_relative_actions_in_range():
    """The normaliser sees relative actions: with relative stats they land
    around 0 with unit spread. Absolute poses would sit tens of std away."""
    pre, post = insert_chunk_relative(*act_pipelines())
    pose = pre(batch(absolute_batch()))["action"][..., :6]
    assert pose.mean().abs() < 0.5
    assert 0.5 < pose.std() < 2.0


def test_insert_refuses_double_install():
    pre2, post2 = insert_chunk_relative(*act_pipelines())
    with pytest.raises(ValueError, match="already"):
        insert_chunk_relative(pre2, post2)
