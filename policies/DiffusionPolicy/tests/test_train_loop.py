"""Tests for train.py's training loop, driven with fakes.

No dataset, no model, no GPU: the dataset metadata, the dataset, the policy, the
processors and the DataLoader are all replaced, so `main()` runs its real loop
(optimizer, resume bookkeeping, checkpoint cadence, validation) in milliseconds.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train  # noqa: E402

B, N_OBS, HORIZON = 2, 2, 16


class Log:
    """What the fakes observed during one main() run."""

    def __init__(self):
        self.lrs: list[float] = []         # lr at every optimizer.step()
        self.grad_norms: list[float] = []  # grad norm at every optimizer.step()
        self.saves: list[str] = []         # directory name of every policy save
        self.train_forwards = 0


LOG = Log()


class FakePolicy(torch.nn.Module):
    def __init__(self, *_a, **_k):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(4))

    def forward(self, batch):
        if self.training:
            LOG.train_forwards += 1
        # A large gradient (norm 2000), so clipping at 10 is observable.
        return (1e3 * self.w).sum() + 0.0 * torch.rand(()), None

    def save_pretrained(self, d):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        LOG.saves.append(d.name)

    @classmethod
    def from_pretrained(cls, _path):
        return cls()


class Ident:
    def __call__(self, batch):
        return batch

    def save_pretrained(self, _d):
        pass


class FakeMeta:
    fps = 50
    stats = {}
    features = {
        "action": {"dtype": "float32", "shape": [11], "names": None},
        "observation.state": {"dtype": "float32", "shape": [2], "names": None},
        "observation.images.cam0": {"dtype": "video", "shape": [96, 96, 3],
                                    "names": ["height", "width", "channels"]},
    }
    episodes = {"episode_index": list(range(10))}

    def __init__(self, *_a, **_k):
        pass


class FakeDataset:
    def __init__(self, *_a, **_k):
        pass

    def __len__(self):
        return 100


class FakeLoader:
    """Yields identical synthetic batches; ignores workers and shuffling."""

    def __init__(self, *_a, **_k):
        pass

    def __iter__(self):
        for _ in range(4):
            yield {
                "observation.state": torch.zeros(B, N_OBS, 2),
                "observation.images.cam0": torch.zeros(B, N_OBS, 3, 96, 96),
                "action": torch.zeros(B, HORIZON, 11),
            }


@pytest.fixture
def run_main(monkeypatch, tmp_path):
    """Run train.main() with the given extra CLI args; returns the Log."""
    import lerobot.optim.optimizers as optimizers

    monkeypatch.setattr(train, "LeRobotDatasetMetadata", FakeMeta)
    monkeypatch.setattr(train, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(train, "DiffusionPolicy", FakePolicy)
    monkeypatch.setattr(train, "make_pre_post_processors", lambda *a, **k: (Ident(), Ident()))
    monkeypatch.setattr(train.torch.utils.data, "DataLoader", FakeLoader)

    # Observe every optimizer step: lr and the gradient norm actually applied.
    orig_build = optimizers.AdamConfig.build

    def build(self, params):
        opt = orig_build(self, params)

        def pre_hook(o, *_):
            LOG.lrs.append(o.param_groups[0]["lr"])
            g = [p.grad for grp in o.param_groups for p in grp["params"] if p.grad is not None]
            LOG.grad_norms.append(float(torch.linalg.vector_norm(torch.stack(
                [torch.linalg.vector_norm(x) for x in g]))))

        opt.register_step_pre_hook(pre_hook)
        return opt

    monkeypatch.setattr(optimizers.AdamConfig, "build", build)

    def run(*extra):
        global LOG
        LOG = Log()
        argv = ["train.py", "--device", "cpu", "--output_dir", str(tmp_path / "out"),
                "--eval_freq", "1000000", "--log_freq", "1000000", *extra]
        monkeypatch.setattr(sys, "argv", argv)
        train.main()
        return LOG

    return run, tmp_path / "out"


# ── TR-10: resume continues at the NEXT step ────────────────────────────

def test_resume_starts_after_the_saved_step(run_main):
    run, out = run_main
    run("--training_steps", "6", "--save_freq", "3")
    ckpt = out / "checkpoint_000003"
    assert ckpt.is_dir()
    mtime = (ckpt / "config.json").stat().st_mtime_ns

    log = run("--training_steps", "6", "--save_freq", "3", "--resume_from", str(ckpt))
    # checkpoint_000003 holds steps 0..3, so resuming must run steps 4 and 5 only.
    assert log.train_forwards == 2, f"resume ran {log.train_forwards} steps, expected 2"
    assert "checkpoint_000003" not in log.saves, "resume rewrote the checkpoint it resumed from"
    assert (ckpt / "config.json").stat().st_mtime_ns == mtime


def test_final_save_records_the_next_step(run_main):
    """The final save records the step count reached, i.e. the next step,
    the same convention as the periodic checkpoints."""
    run, out = run_main
    run("--training_steps", "4", "--save_freq", "1000")
    state = torch.load(out / "train_state.pt")
    assert state["step"] == 4


# ── TR-2: the configured LR schedule and grad clipping ──────────────────

@pytest.mark.xfail(strict=True, reason=(
    "TR-2 NEEDS DECISION: train.py never builds cfg.get_scheduler_preset() and "
    "ignores grad_clip_norm, so LR is a constant 3e-4 although DiffusionConfig "
    "says cosine + warmup 2000. Past certified runs used the constant LR; "
    "enabling the schedule changes the recipe."))
def test_lr_follows_warmup_then_cosine_and_grads_are_clipped(run_main):
    run, _ = run_main
    log = run("--training_steps", "30")
    # warmup 2000 > 30 steps, so LR must still be rising at the end
    assert log.lrs[0] < log.lrs[-1]
    assert max(log.grad_norms) <= 10.0 + 1e-6


def test_lr_is_currently_constant(run_main):
    """Pins today's behaviour (constant LR, unclipped) so a change is deliberate."""
    run, _ = run_main
    log = run("--training_steps", "5")
    assert len(set(log.lrs)) == 1 and log.lrs[0] == pytest.approx(3e-4)


# ── TR-14: validation loss is reproducible ──────────────────────────────

class NoisyPolicy(torch.nn.Module):
    """Loss depends on the global RNG, like diffusion noise and timesteps."""

    def forward(self, _batch):
        return torch.rand(()), None


def test_val_loss_is_deterministic():
    loader = [{"x": torch.zeros(1)}] * 5
    pol = NoisyPolicy()
    a = train.compute_val_loss(pol, Ident(), loader, torch.device("cpu"))
    b = train.compute_val_loss(pol, Ident(), loader, torch.device("cpu"))
    assert a == b


def test_val_loss_does_not_consume_the_training_rng():
    loader = [{"x": torch.zeros(1)}] * 5
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    train.compute_val_loss(NoisyPolicy(), Ident(), loader, torch.device("cpu"))
    assert torch.equal(torch.rand(3), expected)
