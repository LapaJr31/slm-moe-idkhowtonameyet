import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "prelaunch.sh"


@pytest.fixture
def prelaunch_environment(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()

    def executable(name, text):
        path = bindir / name
        path.write_text(text)
        path.chmod(0o755)

    executable("git", "#!/bin/sh\nexit 1\n")
    executable("curl", "#!/bin/sh\nexit 0\n")
    executable("uv", """#!/bin/sh
if [ "$1" = run ] && [ "$2" = pytest ]; then
    printf '1 passed\\n'
    exit 0
fi
if [ "$1" = run ] && [ "$2" = python ]; then
    shift 2
    exec "$TEST_PYTHON" "$@"
fi
exit 1
""")
    executable("nvidia-smi", """#!/bin/sh
case "$1" in
  *memory.total*) printf 'GPU, 100 MiB, 0 MiB, 30\\n' ;;
  *memory.used*) printf '0\\n' ;;
esac
""")
    executable("ssh", """#!/bin/sh
if [ "${TEST_SSH_FAIL:-0}" = 1 ]; then
    printf 'authentication failed\\n' >&2
    exit 255
fi
if [ "${TEST_SSH_WARNING:-0}" = 1 ]; then
    printf 'SSH warning on stderr\\n' >&2
fi
for last do :; done
exec /bin/sh -c "$last"
""")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "dataset_info.json").write_text(
        '{"splits":{"train":{"num_examples":100}}}'
    )
    tokenizer = tmp_path / "tokenizers" / "smol-data-1b-50k"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").touch()
    source = tmp_path / "src" / "moe_slm"
    source.mkdir(parents=True)
    (source / "train.py").write_text("    BATCH_SIZE = 64\n")
    env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "TEST_PYTHON": sys.executable,
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "PRELAUNCH_TEST_LOG": str(tmp_path / "pytest.log"),
    }

    def run(*arguments, **env_overrides):
        return subprocess.run(
            [str(SCRIPT), *arguments], cwd=tmp_path, env={**env, **env_overrides},
            text=True, capture_output=True, timeout=20,
        )

    return run


def test_no_offload_configuration_is_optional(prelaunch_environment):
    result = prelaunch_environment()
    assert result.returncode == 0
    assert "offloading not configured" in result.stdout
    assert "GO — preflight passed" in result.stdout


def test_existing_destination_is_checked_without_shell_injection(prelaunch_environment, tmp_path):
    sentinel = tmp_path / "injected"
    destination = tmp_path / f"laptop's $(touch {sentinel.name}) directory"
    destination.mkdir()
    result = prelaunch_environment(
        "--offload-host", "lapka@laptop", "--offload-dir", str(destination),
        TEST_SSH_WARNING="1",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "destination exists and appears writable" in result.stdout
    assert not sentinel.exists()
    assert list(destination.iterdir()) == []


def test_missing_destination_can_be_created(prelaunch_environment, tmp_path):
    destination = tmp_path / "new" / "checkpoints"
    result = prelaunch_environment("--offload-host=laptop", f"--offload-dir={destination}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "destination absent, existing parent appears writable" in result.stdout
    assert not destination.exists()


def test_failed_ssh_makes_prelaunch_fail(prelaunch_environment, tmp_path):
    result = prelaunch_environment(
        "--offload-host", "laptop", "--offload-dir", str(tmp_path), TEST_SSH_FAIL="1"
    )
    assert result.returncode == 1
    assert "non-interactive SSH to laptop failed: authentication failed" in result.stdout
    assert "NO-GO" in result.stdout


def test_destination_must_be_a_directory(prelaunch_environment, tmp_path):
    destination = tmp_path / "file"
    destination.write_text("existing file")
    result = prelaunch_environment("--offload-host", "laptop", "--offload-dir", str(destination))
    assert result.returncode == 1
    assert "exists but is not a directory" in result.stdout
    assert destination.read_text() == "existing file"


def test_dangling_destination_symlink_is_rejected(prelaunch_environment, tmp_path):
    destination = tmp_path / "dangling"
    destination.symlink_to(tmp_path / "missing")
    result = prelaunch_environment("--offload-host", "laptop", "--offload-dir", str(destination))
    assert result.returncode == 1
    assert "exists but is not a directory" in result.stdout


@pytest.mark.parametrize("args", [
    ["--offload-host", "laptop"],
    ["--offload-dir", "/tmp/checkpoints"],
    ["--offload-host", "laptop", "--offload-dir", "relative"],
    ["--offload-host", "-invalid", "--offload-dir", "/tmp/checkpoints"],
    ["--offload-host"],
])
def test_invalid_arguments_are_rejected(prelaunch_environment, args):
    result = prelaunch_environment(*args)
    assert result.returncode == 2
