"""Frame sources and frame selection.

The dump_obs reader is tested against files written the way evaluate.py writes
them (cv2.imwrite, so BGR on disk). The dataset source is exercised in the
GPU integration test; here we test the selection logic that both share.
"""
import json
import sys
import types

import numpy as np
import pytest

from grabette_attention.sources import DatasetSource, DumpObsSource, select_frames

cv2 = pytest.importorskip("cv2")


def write_dump(tmp_path, n=3):
    for i in range(n):
        # A frame that is unambiguously RED in RGB terms.
        rgb = np.zeros((8, 12, 3), np.uint8)
        rgb[:, :, 0] = 255
        cv2.imwrite(str(tmp_path / f"obs_{i:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    lines = [json.dumps({"step": i, "state": [0.1 * i, 0.2]}) for i in range(n)]
    (tmp_path / "state.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_dump_obs_frames_come_back_as_rgb(tmp_path):
    write_dump(tmp_path)
    source = DumpObsSource(tmp_path, task="pick the sugar cube", camera_key="observation.images.cam0")
    frames = list(source.frames())
    assert len(frames) == 3
    first = frames[0].images["observation.images.cam0"]
    # Red in RGB means channel 0 is hot. If the BGR conversion were skipped,
    # channel 2 would be hot instead.
    assert first[0, 0, 0] == 255
    assert first[0, 0, 2] == 0


def test_dump_obs_pairs_each_frame_with_its_state(tmp_path):
    write_dump(tmp_path)
    source = DumpObsSource(tmp_path, task="t", camera_key="cam0")
    frames = list(source.frames())
    assert frames[2].state[0] == pytest.approx(0.2)
    assert frames[1].frame == 1


def test_grasp_selection_finds_the_first_closure_crossing():
    gripper = np.array([0.0, 0.0, 0.1, 0.6, 0.9, 0.9, 0.2], np.float32)
    indices, note = select_frames(gripper, n_frames=7, mode="grasp")
    assert indices == [3]
    assert "closure" in note


def test_grasp_selection_can_return_a_window_around_the_grasp():
    gripper = np.array([0.0, 0.0, 0.1, 0.6, 0.9], np.float32)
    indices, _ = select_frames(gripper, n_frames=5, mode="grasp", count=3)
    assert indices == [2, 3, 4]


def test_grasp_selection_falls_back_to_a_stride_and_says_so():
    indices, note = select_frames(None, n_frames=10, mode="grasp", count=3)
    assert len(indices) == 3
    assert "stride" in note and "no gripper" in note


def test_a_never_closing_episode_falls_back_to_a_stride():
    gripper = np.zeros(10, np.float32)
    indices, note = select_frames(gripper, n_frames=10, mode="grasp", count=2)
    assert len(indices) == 2
    assert "never" in note


def test_stride_mode_spreads_frames_over_the_episode():
    indices, _ = select_frames(None, n_frames=10, mode="stride", count=5)
    assert indices == [0, 2, 4, 6, 8]


def test_explicit_indices_are_returned_untouched():
    indices, note = select_frames(None, n_frames=10, mode=[2, 7])
    assert indices == [2, 7]
    assert "explicit" in note


def test_an_out_of_range_explicit_index_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        select_frames(None, n_frames=5, mode=[9])


def test_stride_note_reports_actual_count_not_requested():
    indices, note = select_frames(None, n_frames=3, mode="stride", count=5)
    # Should return only 3 frames (all available), not 5
    assert len(indices) == 3
    assert indices == [0, 1, 2]
    # Note should report actual count sampled, not requested count
    assert "3" in note and "5" not in note


class _FakeLeRobotDataset:
    """One camera, `n` frames; item["task"] is set (or omitted) per test.

    Mirrors the one real fact this exercises: `dataset[idx]["task"]` already
    carries the episode's own task string (`DatasetReader.get_item` sets it
    from `meta.tasks`). `DatasetSource` must prefer an explicitly supplied
    `task` over it, and fall back to it only when no `task` was supplied.

    `actions` / `action_names` model the parquet action column
    (`hf_dataset["action"]`) and `meta/info.json`'s channel names. `getitem_calls`
    counts full item reads, each of which decodes video in the real dataset.
    """

    def __init__(self, item_task, repo_id="user/d", root=None, episodes=None,
                 actions=None, action_names=None):
        self._item_task = item_task
        self._actions = (
            np.zeros((1, 3), np.float32) if actions is None
            else np.asarray(actions, np.float32)
        )
        action = {"dtype": "float32", "shape": [self._actions.shape[1]]}
        if action_names is not None:
            action["names"] = list(action_names)
        self.features = {"action": action}
        self.hf_dataset = {"action": [row for row in self._actions]}
        self.getitem_calls = 0

    def __len__(self):
        return len(self._actions)

    def __getitem__(self, idx):
        self.getitem_calls += 1
        item = {
            "cam0": np.zeros((3, 4, 4), np.float32),
            "observation.state": np.zeros(2, np.float32),
            "action": self._actions[idx],
        }
        if self._item_task is not None:
            item["task"] = self._item_task
        return item


def _install_fake_lerobot_dataset(monkeypatch, item_task, **kwargs):
    made = []

    def factory(*a, **k):
        made.append(_FakeLeRobotDataset(item_task, *a, **k, **kwargs))
        return made[-1]

    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = factory
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)
    return made


# A 6-frame episode whose gripper closes at frame 4 on the last channel.
_CLOSING = [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.9], [0, 0, 0.9]]


def test_grasp_selection_reads_the_action_column_without_decoding_every_frame(monkeypatch):
    made = _install_fake_lerobot_dataset(
        monkeypatch, item_task="t", actions=_CLOSING,
        action_names=["x", "strategy", "closure"],
    )
    source = DatasetSource("user/d", episodes=[0], camera_keys=["cam0"], task=None)
    frames = list(source.frames())
    assert [f.frame for f in frames] == [4]
    assert "closure at frame 4" in source.notes[0]
    # Only the ONE selected frame is read in full; the track came from the
    # action column, not from len(dataset) video decodes.
    assert made[0].getitem_calls == 1


def test_a_raw_angle_dataset_is_not_read_as_closure(monkeypatch):
    # Raw 8D/11D datasets end in the distal finger ANGLE, not a [0, 1]
    # closure. Thresholding it at 0.5 picks an arbitrary frame; fall back to
    # a stride and say why instead.
    _install_fake_lerobot_dataset(
        monkeypatch, item_task="t", actions=_CLOSING,
        action_names=["x", "proximal", "distal"],
    )
    source = DatasetSource("user/d", episodes=[0], camera_keys=["cam0"], task=None)
    list(source.frames())
    assert source.notes[0].startswith("no closure channel")
    assert "distal" in source.notes[0] and "stride" in source.notes[0]


def test_an_explicitly_supplied_task_overrides_the_datasets_own(monkeypatch):
    _install_fake_lerobot_dataset(monkeypatch, item_task="test_pick_mustard_200")
    source = DatasetSource(
        "user/d", episodes=[0], camera_keys=["cam0"],
        task="pick up the mustard bottle",
        selection="stride", count=1,
    )
    obs = next(iter(source.frames()))
    assert obs.task == "pick up the mustard bottle"
    assert source.task_source[0] == "override"


def test_the_datasets_own_task_is_used_when_none_was_supplied(monkeypatch):
    _install_fake_lerobot_dataset(monkeypatch, item_task="pick the sugar cube")
    source = DatasetSource(
        "user/d", episodes=[0], camera_keys=["cam0"], task=None,
        selection="stride", count=1,
    )
    obs = next(iter(source.frames()))
    assert obs.task == "pick the sugar cube"
    assert source.task_source[0] == "dataset"


def test_neither_a_supplied_nor_a_dataset_task_raises(monkeypatch):
    _install_fake_lerobot_dataset(monkeypatch, item_task=None)
    source = DatasetSource(
        "user/d", episodes=[0], camera_keys=["cam0"], task=None,
        selection="stride", count=1,
    )
    with pytest.raises(ValueError, match="no --task supplied"):
        next(iter(source.frames()))
