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


@pytest.mark.parametrize("arguments", [
    ["--offload-host", "laptop"],
    ["--offload-dir", "/home/user/checkpoints"],
    ["--offload-host", "laptop", "--offload-dir", "relative"],
    ["--offload-host", "laptop", "--offload-dir", "/tmp/out", "--benchmark-batch-size", "64"],
])
def test_invalid_offload_arguments_rejected(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["moe_slm", *arguments])
    with pytest.raises(SystemExit, match="2"):
        moe_slm.main()


@pytest.mark.parametrize("fail_training", [False, True])
def test_offload_worker_closes_on_training_exit(tmp_path, monkeypatch, fail_training):
    import mlflow
    import moe_slm.checkpoint_offload

    (tmp_path / "dataset").mkdir()
    tokenizer = tmp_path / "tokenizers" / "smol-data-1b-50k"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").touch()
    calls = []

    class Trainer:
        CHECKPOINT_DIR = tmp_path / "checkpoints"

        def __init__(self, dataset, model, run_type):
            pass

        def train(self):
            calls.append(("train", self.offloader))
            if fail_training:
                raise RuntimeError("training interrupted")

    class Offloader:
        def __init__(self, directory, host, destination):
            calls.append(("start", directory, host, destination))

        def close(self):
            calls.append(("close",))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["moe_slm", "--offload-host", "laptop", "--offload-dir", "/home/user/ckpts"])
    monkeypatch.setattr(moe_slm, "preTrain", Trainer)
    monkeypatch.setattr(moe_slm, "MoESLM", lambda **kwargs: object())
    monkeypatch.setattr(moe_slm, "load_from_disk", lambda path: object())
    monkeypatch.setattr(moe_slm.checkpoint_offload, "CheckpointOffloader", Offloader)
    monkeypatch.setattr(mlflow, "end_run", lambda: calls.append(("end",)))
    if fail_training:
        with pytest.raises(RuntimeError, match="training interrupted"):
            moe_slm.main()
    else:
        moe_slm.main()
    assert calls[0] == ("start", Trainer.CHECKPOINT_DIR, "laptop", "/home/user/ckpts")
    assert isinstance(calls[1][1], Offloader)
    assert calls[-2:] == [("close",), ("end",)]
