import os
import shlex
import threading
import time
from pathlib import Path

import pytest
import torch

from moe_slm.checkpoint_offload import CheckpointOffloader
from moe_slm.train import preTrain


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("timed out waiting for offload worker")


def install_fake_ssh(tmp_path, monkeypatch, body):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    ssh = bindir / "ssh"
    ssh.write_text("#!/bin/sh\n" + body + "\n")
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


def replace_checkpoint(directory, name, content):
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory / "write.tmp"
    temp.write_bytes(content)
    path = directory / f"{name}.pt"
    os.replace(temp, path)
    return path


def test_ssh_transfer_publishes_complete_files_with_quoted_paths(tmp_path, monkeypatch):
    install_fake_ssh(tmp_path, monkeypatch, 'for last do :; done\nexec /bin/sh -c "$last"')
    source = tmp_path / "source"
    destination = tmp_path / "remote's folder"
    destination.mkdir()
    (destination / "latest.pt").write_bytes(b"old")
    offloader = CheckpointOffloader(source, "test-host", str(destination))
    try:
        offloader.enqueue(replace_checkpoint(source, "latest", b"new latest"))
        offloader.enqueue(replace_checkpoint(source, "best", b"best result"))
        wait_for(lambda: (destination / "latest.pt").read_bytes() == b"new latest")
        wait_for(lambda: (destination / "best.pt").exists())
    finally:
        offloader.close()
    assert (source / "latest.pt").read_bytes() == b"new latest"
    assert (source / "best.pt").read_bytes() == b"best result"
    assert (destination / "best.pt").read_bytes() == b"best result"
    assert not list(destination.glob("*.part"))


def test_interrupted_upload_preserves_destination_and_queues_retry(tmp_path, monkeypatch):
    marker = tmp_path / "started"
    install_fake_ssh(tmp_path, monkeypatch,
        f'touch {shlex.quote(str(marker))}\nfor last do :; done\nhead -c 5 | /bin/sh -c "$last"')
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "latest.pt").write_bytes(b"previous complete checkpoint")
    offloader = CheckpointOffloader(source, "test-host", str(destination))
    offloader.enqueue(replace_checkpoint(source, "latest", b"new complete checkpoint"))
    wait_for(marker.exists)
    wait_for(lambda: (source / ".offload" / "latest.pending").exists())
    offloader.close(timeout=0.1)
    assert (destination / "latest.pt").read_bytes() == b"previous complete checkpoint"
    assert not list(destination.glob("*.part"))
    assert (source / ".offload" / "latest.pending").read_bytes() == b"new complete checkpoint"


def test_slow_failure_coalesces_snapshots_and_recovers_on_restart(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    transferred = []

    def fail(self, path, name):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("offline")

    monkeypatch.setattr(CheckpointOffloader, "_transfer", fail)
    source = tmp_path / "source"
    offloader = CheckpointOffloader(source, "host", "/destination")
    offloader.enqueue(replace_checkpoint(source, "latest", b"first"))
    assert entered.wait(5)
    offloader.enqueue(replace_checkpoint(source, "latest", b"second"))
    offloader.enqueue(replace_checkpoint(source, "latest", b"third"))
    offloader.enqueue(replace_checkpoint(source, "best", b"best"))
    release.set()
    offloader.close(timeout=1)
    queue = source / ".offload"
    assert (queue / "latest.pending").read_bytes() == b"third"
    assert (queue / "best.pending").read_bytes() == b"best"
    assert (source / "latest.pt").read_bytes() == b"third"

    def succeed(self, path, name):
        transferred.append((name, path.read_bytes()))

    monkeypatch.setattr(CheckpointOffloader, "_transfer", succeed)
    recovered = CheckpointOffloader(source, "host", "/destination")
    wait_for(lambda: len(transferred) == 2)
    recovered.close()
    assert transferred == [("latest", b"third"), ("best", b"best")]
    assert not list(queue.glob("*.pending"))


def test_transient_failure_retries_while_training_continues(tmp_path, monkeypatch):
    attempts = []

    def transfer(self, path, name):
        attempts.append(path.read_bytes())
        if len(attempts) == 1:
            raise OSError("temporary SSH failure")

    monkeypatch.setattr(CheckpointOffloader, "_transfer", transfer)
    monkeypatch.setattr(CheckpointOffloader, "RETRY_INITIAL_SECONDS", 0.02)
    offloader = CheckpointOffloader(tmp_path, "host", "/destination")
    offloader.enqueue(replace_checkpoint(tmp_path, "latest", b"checkpoint"))
    wait_for(lambda: len(attempts) >= 2)
    offloader.close()
    assert attempts == [b"checkpoint", b"checkpoint"]
    assert not (tmp_path / ".offload" / "latest.pending").exists()


def test_recovery_prefers_new_over_pending_and_inflight(tmp_path, monkeypatch):
    queue = tmp_path / ".offload"
    queue.mkdir()
    for state, content in [("inflight", b"old"), ("pending", b"middle"), ("new", b"new")]:
        (queue / f"latest.{state}").write_bytes(content)
    received = []
    monkeypatch.setattr(CheckpointOffloader, "_transfer", lambda self, path, name: received.append(path.read_bytes()))
    offloader = CheckpointOffloader(tmp_path, "host", "/destination")
    wait_for(lambda: received)
    offloader.close()
    assert received == [b"new"]
    assert sorted(p.name for p in queue.iterdir()) == ["lock"]


def test_existing_checkpoint_is_queued_when_starting(tmp_path, monkeypatch):
    (tmp_path / "latest.pt").write_bytes(b"saved before worker started")
    received = []
    monkeypatch.setattr(CheckpointOffloader, "_transfer", lambda self, path, name: received.append(path.read_bytes()))
    offloader = CheckpointOffloader(tmp_path, "host", "/destination")
    wait_for(lambda: received)
    offloader.close()
    assert received == [b"saved before worker started"]


def test_startup_keeps_pending_snapshot_newer_than_current_file(tmp_path, monkeypatch):
    current = tmp_path / "latest.pt"
    current.write_bytes(b"older")
    queue = tmp_path / ".offload"
    queue.mkdir()
    pending = queue / "latest.pending"
    pending.write_bytes(b"newer")
    os.utime(current, ns=(1_000_000_000, 1_000_000_000))
    os.utime(pending, ns=(2_000_000_000, 2_000_000_000))
    received = []
    monkeypatch.setattr(CheckpointOffloader, "_transfer", lambda self, path, name: received.append(path.read_bytes()))
    offloader = CheckpointOffloader(tmp_path, "host", "/destination")
    wait_for(lambda: received)
    offloader.close()
    assert received == [b"newer"]


def test_shutdown_terminates_stalled_ssh_and_keeps_snapshot(tmp_path, monkeypatch):
    marker = tmp_path / "started"
    install_fake_ssh(tmp_path, monkeypatch, f'touch {shlex.quote(str(marker))}\nexec sleep 30')
    source = tmp_path / "source"
    offloader = CheckpointOffloader(source, "host", "/destination")
    offloader.enqueue(replace_checkpoint(source, "latest", b"checkpoint"))
    wait_for(marker.exists)
    start = time.monotonic()
    offloader.close(timeout=0.05)
    assert time.monotonic() - start < 3
    assert (source / ".offload" / "latest.pending").read_bytes() == b"checkpoint"


def test_offloader_prevents_two_workers_for_same_queue(tmp_path):
    offloader = CheckpointOffloader(tmp_path, "host", "/destination")
    try:
        with pytest.raises(RuntimeError, match="Another checkpoint offloader"):
            CheckpointOffloader(tmp_path, "host", "/destination")
    finally:
        offloader.close()


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_save_enqueues_only_after_atomic_save(tmp_path, capsys, fail):
    trainer = preTrain.__new__(preTrain)
    trainer.CHECKPOINT_DIR = tmp_path
    trainer.device = "cpu"
    trainer.total_tokens = 7
    trainer.best_val_loss = 1.0
    trainer.start_epoch = 0
    trainer.model = torch.nn.Linear(2, 2)
    trainer.adamw = torch.optim.AdamW(trainer.model.parameters())
    trainer.muon = torch.optim.AdamW(trainer.model.parameters())
    trainer.schedule_adamw = torch.optim.lr_scheduler.LambdaLR(trainer.adamw, lambda _: 1)
    trainer.schedule_muon = torch.optim.lr_scheduler.LambdaLR(trainer.muon, lambda _: 1)
    queued = []

    class Offloader:
        def enqueue(self, path):
            queued.append((path.name, torch.load(path, weights_only=False)["tokens_seen"]))
            if fail:
                raise OSError("hard links unavailable")

    trainer.offloader = Offloader()
    path = trainer.save_checkpoint("latest")
    assert path.exists()
    assert queued == [("latest.pt", 7)]
    if fail:
        assert "local checkpoint remains available" in capsys.readouterr().out


def test_inflight_copy_does_not_starve_pending_best(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    completed = []

    def transfer(self, path, name):
        if not completed:
            entered.set()
            assert release.wait(5)
        completed.append((name, path.read_bytes()))

    monkeypatch.setattr(CheckpointOffloader, "_transfer", transfer)
    source = tmp_path / "source"
    offloader = CheckpointOffloader(source, "host", "/destination")
    offloader.enqueue(replace_checkpoint(source, "latest", b"old"))
    assert entered.wait(5)
    offloader.enqueue(replace_checkpoint(source, "latest", b"new"))
    offloader.enqueue(replace_checkpoint(source, "best", b"best"))
    release.set()
    wait_for(lambda: len(completed) == 3)
    offloader.close()
    assert completed == [("latest", b"old"), ("best", b"best"), ("latest", b"new")]
