"""Tests for rollout.py's RTC engine: chunk hand-off between the inference
thread and the control loop.

Needs the `eval` extra (rollout.py imports evaluate.py and lerobot/torch).
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
rollout = pytest.importorskip("rollout")

CHUNK, DT, INFER_S = 10, 0.01, 0.05


class FakeFlowPolicy:
    """Chunk k is [k, 0..CHUNK-1, 0...]: column 0 tells which chunk an action
    came from, column 1 its index inside that chunk."""

    def __init__(self):
        self.config = SimpleNamespace(type="pi05", chunk_size=CHUNK)
        self.calls = []

    def init_rtc_processor(self):
        pass

    def reset(self):
        pass

    def predict_action_chunk(self, batch, inference_delay, prev_chunk_left_over):
        self.calls.append({"delay": inference_delay, "left_over": prev_chunk_left_over})
        time.sleep(INFER_S)
        chunk = torch.zeros(1, CHUNK, 11)
        chunk[0, :, 0] = len(self.calls) - 1
        chunk[0, :, 1] = torch.arange(CHUNK)
        return chunk


def test_rtc_engine_drops_actions_executed_during_inference():
    policy = FakeFlowPolicy()
    engine = rollout.RTCEngine(policy, lambda b: b, lambda t: t, rollout.RTCConfig(execution_horizon=5),
                               fps=1 / DT, queue_threshold=CHUNK)
    engine.start()
    engine.notify_observation({})
    executed = []
    try:
        for _ in range(60):
            if (a := engine.get_action()) is not None:
                executed.append((int(a[0]), int(a[1])))
            time.sleep(DT)
    finally:
        engine.stop()

    assert executed[0] == (0, 0)
    assert len(policy.calls) >= 3
    # From the second chunk on, guidance gets the queued tail and a delay
    # measured from the previous inference (INFER_S / DT ticks).
    assert policy.calls[1]["left_over"] is not None
    assert policy.calls[1]["delay"] >= 1
    # Each chunk starts executing right after the actions consumed during its
    # own inference, never from its head again.
    for k in range(1, len(engine.dropped)):
        first = next(((c, i) for c, i in executed if c == k), None)
        if first is not None:
            assert first == (k, engine.dropped[k]) and engine.dropped[k] > 0


def test_rtc_engine_refuses_non_flow_policies():
    act = SimpleNamespace(config=SimpleNamespace(type="act"))
    with pytest.raises(SystemExit, match="flow-matching"):
        rollout.RTCEngine(act, None, None, rollout.RTCConfig(), fps=30, queue_threshold=25)
