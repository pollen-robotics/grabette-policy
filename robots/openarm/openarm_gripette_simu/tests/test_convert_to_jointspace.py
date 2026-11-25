"""Dataset-side helpers of examples/convert_to_jointspace.py (needs the `dataset` extra).

    uv run --extra dataset pytest robots/openarm/openarm_gripette_simu/tests/test_convert_to_jointspace.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

_SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "convert_to_jointspace.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("convert_to_jointspace", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_match_episodes_maps_cleaned_to_original_indices(cli):
    rng = np.random.default_rng(0)
    orig = [rng.normal(size=(n, 8)) for n in (5, 7, 5, 6)]
    cleaned = [orig[0], orig[2], orig[3]]               # episode 1 was rejected by the cleaning step
    assert cli.match_episodes(cleaned, orig) == [0, 2, 3]


def test_match_episodes_refuses_an_unmatched_episode(cli):
    orig = [np.zeros((5, 8)), np.ones((5, 8))]
    with pytest.raises(ValueError, match="episode 0"):
        cli.match_episodes([np.full((5, 8), 2.0)], orig)


def _fake_dataset(root: Path):
    """Two episodes (3 and 2 frames) split over two parquet files, 11D action / 2D state."""
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    rows = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for fi, sl in enumerate([slice(0, 2), slice(2, 5)]):
        part = rows[sl]
        pq.write_table(pa.table({
            "episode_index": pa.array([e for e, _ in part], pa.int64()),
            "frame_index": pa.array([f for _, f in part], pa.int64()),
            "action": pa.array([[0.0] * 11 for _ in part], pa.list_(pa.float32())),
            "observation.state": pa.array([[0.0] * 2 for _ in part], pa.list_(pa.float32())),
        }), root / "data" / "chunk-000" / f"file-{fi:03d}.parquet")
    info = {"features": {"action": {"dtype": "float32", "shape": [11], "names": ["a"] * 11},
                         "observation.state": {"dtype": "float32", "shape": [2], "names": ["strategy", "closure"]}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))


def test_drop_undeclared_episode_stats(cli, tmp_path):
    """Stats columns of a removed feature (e.g. a camera dropped by remove_feature) must go, others stay."""
    (tmp_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"features": {"observation.images.cam0": {}, "action": {}}}))
    p = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    pq.write_table(pa.table({"episode_index": [0], "stats/action/min": [[0.0]],
                             "stats/observation.images.cam0/min": [[[[0.0]]] * 3],
                             "stats/observation.images.right_cam1/min": [[[[0.0]]] * 3]}), p)
    assert cli.drop_undeclared_episode_stats(tmp_path) == ["observation.images.right_cam1"]
    assert pq.read_schema(p).names == ["episode_index", "stats/action/min", "stats/observation.images.cam0/min"]


def test_rewrite_joint_columns_by_episode_and_frame(cli, tmp_path):
    _fake_dataset(tmp_path)
    state = {0: np.arange(27, dtype=np.float32).reshape(3, 9), 1: 100 + np.arange(18, dtype=np.float32).reshape(2, 9)}
    action = {e: v + 0.5 for e, v in state.items()}
    cli.rewrite_joint_columns(tmp_path, state, action, gripper_names=["strategy", "closure"])
    t = pa.concat_tables([pq.read_table(p) for p in sorted((tmp_path / "data").rglob("*.parquet"))]).to_pydict()
    for e, f, s, a in zip(t["episode_index"], t["frame_index"], t["observation.state"], t["action"]):
        np.testing.assert_allclose(s, state[e][f])
        np.testing.assert_allclose(a, action[e][f])
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    for key in ("action", "observation.state"):
        assert info["features"][key]["shape"] == [9]
        assert info["features"][key]["names"][:7] == [f"q{i}" for i in range(7)]
        assert info["features"][key]["names"][7:] == ["strategy", "closure"]
