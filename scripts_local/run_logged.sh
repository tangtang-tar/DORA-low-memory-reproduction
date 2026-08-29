#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: $0 RUN_NAME COMMAND [ARG ...]" >&2
    echo "Example: $0 phase_c2 python scripts_local/phase_c2_run.py" >&2
}

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

run_name=$1
shift

if [[ ! "$run_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dora_root=${DORA_ROOT:-$(dirname -- "$script_dir")}
timestamp=$(date '+%Y%m%d_%H%M%S')
run_dir="$dora_root/logs/${timestamp}_${run_name}"
mkdir -p "$run_dir"

command_file="$run_dir/command.txt"
environment_file="$run_dir/environment.txt"
output_file="$run_dir/output.log"
status_file="$run_dir/status.txt"

printf '%q ' "$@" >"$command_file"
printf '\n' >>"$command_file"

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "working_directory=$(pwd)"
    echo "dora_root=$dora_root"
    echo "python=$(command -v python || true)"
    python --version 2>&1 || true
    echo "git_src_commit=$(git -C "$dora_root/src" rev-parse HEAD 2>/dev/null || true)"
    echo "git_src_status_begin"
    git -C "$dora_root/src" status --short 2>/dev/null || true
    echo "git_src_status_end"
    echo "conda_environment=${CONDA_PREFIX:-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
    echo "selected_packages_begin"
    python -m pip freeze 2>/dev/null | grep -Ei '^(torch|transformers|datasets|vllm|accelerate|bitsandbytes|flagembedding|numpy|scipy)==' || true
    echo "selected_packages_end"
} >"$environment_file"

started_at=$(date +%s)
echo "[$(date --iso-8601=seconds)] START $run_name" | tee "$output_file"
printf 'Command: ' | tee -a "$output_file"
cat "$command_file" | tee -a "$output_file"

set +e
"$@" > >(tee -a "$output_file") 2> >(tee -a "$output_file" >&2)
exit_code=$?
set -e

finished_at=$(date +%s)
{
    echo "run_name=$run_name"
    echo "started_epoch=$started_at"
    echo "finished_epoch=$finished_at"
    echo "duration_seconds=$((finished_at - started_at))"
    echo "exit_code=$exit_code"
} >"$status_file"

echo "[$(date --iso-8601=seconds)] END $run_name exit_code=$exit_code" | tee -a "$output_file"
exit "$exit_code"
