#!/bin/sh
# Pre-launch checklist for moe_slm training runs.
# Run on the training box BEFORE a long job:  ./prelaunch.sh
# Exit code 0 = go, 1 = fix something first.

fail=0
say() { printf '\n[%s] %s\n' "$1" "$2"; }
ok()  { printf '  OK   %s\n' "$1"; }
bad() { printf '  FAIL %s\n' "$1"; fail=1; }
warn(){ printf '  WARN %s\n' "$1"; }

offload_host=""
offload_dir=""
host_set=0
dir_set=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --offload-host|--offload-dir)
            if [ "$#" -lt 2 ]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            option=$1
            shift
            if [ "$option" = "--offload-host" ]; then
                offload_host=$1
                host_set=1
            else
                offload_dir=$1
                dir_set=1
            fi
            ;;
        --offload-host=*) offload_host=${1#*=}; host_set=1 ;;
        --offload-dir=*) offload_dir=${1#*=}; dir_set=1 ;;
        -h|--help)
            printf 'Usage: %s [--offload-host SSH_HOST --offload-dir ABSOLUTE_PATH]\n' "$0"
            exit 0
            ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done
if [ "$host_set" != "$dir_set" ]; then
    printf '%s\n' '--offload-host and --offload-dir must be provided together' >&2
    exit 2
fi
if [ "$host_set" -eq 1 ]; then
    case "$offload_host" in
        ''|-*) printf '%s\n' 'Invalid SSH offload host' >&2; exit 2 ;;
    esac
    case "$offload_dir" in
        /*) ;;
        *) printf '%s\n' 'Offload directory must be an absolute path' >&2; exit 2 ;;
    esac
fi

say "1. code freshness"
if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
    if [ -n "$(git status --porcelain -- src/ tests/ 2>/dev/null)" ]; then
        warn "uncommitted changes in src/ or tests/ — local edits may be shadowing the repo"
        git status --porcelain -- src/ tests/ | head -5
    else
        ok "working tree clean"
    fi
    git fetch -q 2>/dev/null
    behind=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo "?")
    if [ "$behind" = "0" ]; then ok "up to date with remote"
    elif [ "$behind" = "?" ]; then warn "no upstream configured — can't check freshness"
    else bad "$behind commits behind remote — git pull first"; fi
else
    warn "not a git checkout — can't verify freshness"
fi

say "2. unit tests"
test_log=${PRELAUNCH_TEST_LOG:-/tmp/prelaunch_tests.log}
if uv run pytest tests/ -m "not slow" -q >"$test_log" 2>&1; then
    ok "$(grep -oE '[0-9]+ passed[^,]*' "$test_log" | tail -1) (uv run pytest)"
else
    bad "tests failing — see $test_log"
    grep -E "FAILED|ERROR" "$test_log" | head -5
fi

say "3. mlflow connectivity"
uri="${MLFLOW_TRACKING_URI:-}"
if [ -z "$uri" ]; then
    bad "MLFLOW_TRACKING_URI not set (the recurring mistake)"
else
    ok "MLFLOW_TRACKING_URI=$uri"
    if curl -sf -o /dev/null --max-time 10 "$uri/api/2.0/mlflow/experiments/search?max_results=1"; then
        ok "server reachable"
    else
        bad "server not reachable at $uri"
    fi
fi

say "4. dataset / tokenizer artifacts"
if [ -f dataset/dataset_info.json ]; then
    rows=$(uv run python -c "import json; print(json.load(open('dataset/dataset_info.json'))['splits']['train']['num_examples'])" 2>/dev/null || echo "?")
    ok "dataset present: $rows rows ($rows x $SEQLEN tokens)"
else
    bad "dataset/ missing"
fi
if [ -f tokenizers/smol-data-1b-50k/tokenizer.json ]; then
    ok "tokenizer present"
else
    warn "tokenizer missing — will be trained (slow) unless intended"
fi

say "5. checkpoint compatibility"
if [ -f checkpoints/latest.pt ]; then
    if uv run python - <<'EOF' >/dev/null 2>&1
import torch, os
ckpt = torch.load("checkpoints/latest.pt", map_location="cpu", weights_only=False)
os.environ.setdefault("MOE_SLM_CKPT_PARAMS", str(sum(v.numel() for v in ckpt["model"].values())))
EOF
    then ok "latest.pt loads"
    else bad "checkpoints/latest.pt exists but failed to load (likely stale/param mismatch) — mv it aside"; fi
else
    ok "no stale checkpoint"
fi

say "6. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader | while IFS=, read -r name total used temp; do
        printf '  %s | %s total, %s used, %s\n' "$name" "$total" "$used" "$temp"
    done
    if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000 {exit 1}'; then
        ok "GPUs idle"
    else
        warn "GPU memory already in use — another process running?"
    fi
else
    warn "nvidia-smi not found"
fi

say "7. training constants (verify against your intent)"
grep -E "^\s+(BATCH_SIZE|GRAD_ACCUM|LR_MUON|LR_ADAMW|AUX_COEF|Z_COEF|SEQLEN|CE_CHUNK) =" src/moe_slm/train.py || warn "train.py constants not found"
echo "  -> effective batch = BATCH_SIZE * GRAD_ACCUM; LR must be scaled to match"

say "8. checkpoint offloading"
if [ "$host_set" -eq 0 ]; then
    warn "offloading not configured — pass --offload-host and --offload-dir to check it"
elif ! command -v ssh >/dev/null 2>&1; then
    bad "ssh is not installed; checkpoint offloading cannot connect"
elif ! quoted_dir=$(uv run python -c 'import shlex, sys; print(shlex.quote(sys.argv[1]))' "$offload_dir"); then
    bad "could not prepare the offload destination check"
else
    # Match the offloader's SSH settings. Check permissions without creating files.
    offload_error=$(mktemp "${TMPDIR:-/tmp}/moe-slm-ssh.XXXXXX") || exit 1
    if offload_status=$(ssh -o BatchMode=yes -o ConnectTimeout=10 \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=2 "$offload_host" "
        path=$quoted_dir
        if [ -d \"\$path\" ]; then
            if [ -w \"\$path\" ] && [ -x \"\$path\" ]; then printf 'writable'; else printf 'unwritable'; fi
        elif [ -e \"\$path\" ] || [ -L \"\$path\" ]; then
            printf 'not-directory'
        else
            parent=\$path
            while [ ! -e \"\$parent\" ] && [ ! -L \"\$parent\" ] && [ \"\$parent\" != / ]; do parent=\$(dirname \"\$parent\"); done
            if [ -d \"\$parent\" ] && [ -w \"\$parent\" ] && [ -x \"\$parent\" ]; then
                printf 'creatable'
            else
                printf 'unwritable'
            fi
        fi" 2>"$offload_error"); then
        case "$offload_status" in
            writable) ok "SSH connected; destination exists and appears writable: $offload_dir" ;;
            creatable) ok "SSH connected; destination absent, existing parent appears writable: $offload_dir" ;;
            not-directory) bad "offload destination exists but is not a directory: $offload_dir" ;;
            unwritable) bad "offload destination or its existing parent is not writable: $offload_dir" ;;
            *) bad "unexpected offload check response from $offload_host: $offload_status" ;;
        esac
    else
        bad "non-interactive SSH to $offload_host failed: $(cat "$offload_error")"
    fi
    rm -f "$offload_error"
fi

printf '\n%s\n' "======================================"
if [ "$fail" -eq 0 ]; then printf 'GO — preflight passed\n'; exit 0
else printf 'NO-GO — fix the FAIL items above\n'; exit 1; fi
