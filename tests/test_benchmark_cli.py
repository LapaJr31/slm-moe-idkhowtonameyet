import sys

import pytest

import moe_slm


@pytest.mark.parametrize("batch,accum", [(16, 4), (32, 2), (64, 1), (96, 1), (128, 1), (256, 1)])
def test_benchmark_cli_batch_sizes(tmp_path, monkeypatch, batch, accum):
    import mlflow

    (tmp_path / "dataset").mkdir()
    tokenizer = tmp_path / "tokenizers" / "smol-data-1b-50k"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").touch()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.pt").touch()
    calls = []

    class Trainer:
        BATCH_SIZE = 64
        GRAD_ACCUM = 1

        def __init__(self, dataset, model, run_type):
            assert (self.BATCH_SIZE, self.GRAD_ACCUM) == (batch, accum)
            assert run_type == "benchmark_batch"
            self.train_dl = object()

        def load_checkpoint(self, path, *, benchmark):
            calls.append(("load", path.name, benchmark))

        def run_epoch(self, loader, *, benchmark_steps):
            assert loader is self.train_dl
            calls.append(("run", benchmark_steps))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["moe_slm", "--benchmark-batch-size", str(batch)])
    monkeypatch.setattr(moe_slm, "preTrain", Trainer)
    monkeypatch.setattr(moe_slm, "MoESLM", lambda **kwargs: object())
    monkeypatch.setattr(moe_slm, "load_from_disk", lambda path: object())
    monkeypatch.setattr(mlflow, "end_run", lambda: calls.append(("end",)))
    moe_slm.main()
    assert calls == [("load", "latest.pt", True), ("run", 15), ("end",)]


@pytest.mark.parametrize("batch", [0, -1, 48])
def test_invalid_benchmark_batch_rejected(monkeypatch, batch):
    monkeypatch.setattr(sys, "argv", ["moe_slm", "--benchmark-batch-size", str(batch)])
    with pytest.raises(SystemExit, match="2"):
        moe_slm.main()
