"""Best-effort asynchronous SSH offloading of completed checkpoints."""

import fcntl
import os
import shlex
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path, PurePosixPath


class CheckpointOffloader:
    """Keep stable hard-linked snapshots until they have been sent over SSH."""

    NAMES = ("latest", "best")
    RETRY_INITIAL_SECONDS = 5.0
    RETRY_MAX_SECONDS = 60.0

    def __init__(self, checkpoint_dir, host, remote_dir, *, transfer_timeout=1800):
        if not host or host.startswith("-") or not remote_dir.startswith("/"):
            raise ValueError("Offload host must be an SSH destination and remote directory must be absolute")
        self.host = host
        self.remote_dir = PurePosixPath(remote_dir)
        self.transfer_timeout = transfer_timeout
        self.directory = Path(checkpoint_dir) / ".offload"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.directory / "lock").open("a")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise RuntimeError(f"Another checkpoint offloader owns {self.directory}") from None
        self._condition = threading.Condition()
        self._closing = False
        self._force_stop = False
        self._process = None
        self._next_name = 0
        self._recover()
        # Recover a save interrupted between the atomic rename and enqueue.
        for name in self.NAMES:
            checkpoint = Path(checkpoint_dir) / f"{name}.pt"
            if checkpoint.exists():
                try:
                    pending = self._path(name, "pending")
                    if pending.exists() and pending.stat().st_mtime_ns > checkpoint.stat().st_mtime_ns:
                        continue
                    self.enqueue(checkpoint)
                except OSError as exc:
                    print(f"[offload] could not queue existing {checkpoint.name}: {exc}", flush=True)
        self._thread = threading.Thread(target=self._run, name="checkpoint-offload", daemon=True)
        self._thread.start()

    def _path(self, name, state):
        return self.directory / f"{name}.{state}"

    def _recover(self):
        # A new snapshot always supersedes a pending one, which supersedes an
        # in-flight one. All three are complete, atomically created hard links.
        for name in self.NAMES:
            new, pending, inflight = (self._path(name, state) for state in ("new", "pending", "inflight"))
            if new.exists():
                os.replace(new, pending)
            if inflight.exists():
                if pending.exists():
                    inflight.unlink()
                else:
                    os.replace(inflight, pending)

    def enqueue(self, checkpoint):
        checkpoint = Path(checkpoint)
        name = checkpoint.stem
        if name not in self.NAMES or checkpoint.name != f"{name}.pt":
            return
        with self._condition:
            if self._closing:
                raise RuntimeError("Checkpoint offloader is closing")
            new, pending = self._path(name, "new"), self._path(name, "pending")
            new.unlink(missing_ok=True)
            os.link(checkpoint, new)
            os.replace(new, pending)
            self._condition.notify()

    def _transfer(self, snapshot, name):
        destination = self.remote_dir / f"{name}.pt"
        temporary = self.remote_dir / f".{name}.{uuid.uuid4().hex}.part"
        # Verify the byte count before publishing: a disconnected SSH stream
        # must not turn a truncated temporary file into the final checkpoint.
        command = (
            f"umask 077; mkdir -p -- {shlex.quote(str(self.remote_dir))} || exit 1; "
            f"tmp={shlex.quote(str(temporary))}; trap 'rm -f -- \"$tmp\"' EXIT; "
            f"cat > \"$tmp\" && [ \"$(wc -c < \"$tmp\")\" -eq {snapshot.stat().st_size} ] "
            f"&& mv -f -- \"$tmp\" {shlex.quote(str(destination))}"
        )
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=2", self.host, command]
        with snapshot.open("rb") as source, tempfile.TemporaryFile() as stderr:
            with self._condition:
                if self._force_stop:
                    raise RuntimeError("shutdown deadline reached")
                process = subprocess.Popen(argv, stdin=source, stdout=subprocess.DEVNULL, stderr=stderr)
                self._process = process
            try:
                try:
                    result = process.wait(timeout=self.transfer_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("SSH transfer timed out") from None
                if result:
                    stderr.seek(0)
                    detail = stderr.read(4096).decode(errors="replace").strip()
                    raise RuntimeError(f"SSH exited {result}: {detail}")
            finally:
                with self._condition:
                    self._process = None

    def _run(self):
        delay = self.RETRY_INITIAL_SECONDS
        while True:
            with self._condition:
                pending = self._next_pending()
                while pending is None and not self._closing:
                    self._condition.wait()
                    pending = self._next_pending()
                if self._force_stop or pending is None:
                    return
                self._next_name = (self.NAMES.index(pending) + 1) % len(self.NAMES)
                inflight = self._path(pending, "inflight")
                os.replace(self._path(pending, "pending"), inflight)

            try:
                self._transfer(inflight, pending)
            except (OSError, RuntimeError) as exc:
                print(f"[offload] {pending}.pt failed: {exc}; checkpoint remains queued", flush=True)
                with self._condition:
                    new_pending = self._path(pending, "pending")
                    if new_pending.exists():
                        inflight.unlink(missing_ok=True)
                    else:
                        os.replace(inflight, new_pending)
                    if self._closing:
                        return
                    self._condition.wait(timeout=delay)
                delay = min(delay * 2, self.RETRY_MAX_SECONDS)
            else:
                with self._condition:
                    inflight.unlink(missing_ok=True)
                print(f"[offload] transferred {pending}.pt to {self.host}:{self.remote_dir}", flush=True)
                delay = self.RETRY_INITIAL_SECONDS

    def _next_pending(self):
        names = self.NAMES[self._next_name:] + self.NAMES[:self._next_name]
        return next((name for name in names if self._path(name, "pending").exists()), None)

    def close(self, timeout=10.0):
        """Wait briefly for transfers; retain unsent snapshots for next run."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            with self._condition:
                self._force_stop = True
                if self._process is not None:
                    self._process.kill()
                self._condition.notify_all()
            self._thread.join(2.0)
        if self._thread.is_alive():
            print("[offload] worker did not stop; pending snapshots will be retried on restart", flush=True)
            return
        unsent = [name for name in self.NAMES if any(self._path(name, state).exists() for state in ("new", "pending", "inflight"))]
        if unsent:
            print(f"[offload] unsent checkpoints retained for restart: {', '.join(unsent)}", flush=True)
        self._lock_file.close()
