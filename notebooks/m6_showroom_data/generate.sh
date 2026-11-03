#!/usr/bin/env bash
set -euo pipefail
[[ $# -le 2 ]] || { printf 'Usage: bash notebooks/m6_showroom_data/generate.sh [dataset-name] [brief-path]\n' >&2; exit 2; }
[[ -f notebooks/m6_showroom_data/gen.md ]] || { printf 'Run from the project root.\n' >&2; exit 1; }
dataset=${1:-showroom-en}
brief=${2:-notebooks/m6_showroom_data/gen.md}
[[ "$dataset" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || { printf 'Invalid dataset name.\n' >&2; exit 2; }
[[ -f "$brief" ]] || { printf 'Brief not found: %s\n' "$brief" >&2; exit 2; }
brief=$(realpath "$brief") # Resolve before uv changes the working directory to notebooks/.
stamp=$(date -u +%Y%m%dT%H%M%S%6NZ)
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true; exit 130' INT TERM
generate() {
  local model=$1 variant=$2 directory=$3
  "${OPENCODE_BIN:-opencode}" run --agent build --model "opencode/$model" --variant "$variant" \
    --file "$directory/gen.md" --format json \
    "Follow gen.md. Output directory: $directory. Record-ID prefix: ${directory##*/}. Generate 500 records." \
    < /dev/null > "$directory/generation.log" 2>&1
}
for job in 'gemini-4-argon high' 'glm-5.3 max'; do
  read -r model variant <<< "$job"
  directory=$(uv --directory notebooks run --locked --offline python m6_showroom_data/workspace.py raw \
    --dataset "../data/$dataset" --model "opencode/$model" --variant "$variant" \
    --brief "$brief" --timestamp "$stamp" --job "$((${#pids[@]} + 1))")
  printf 'Starting %s (%s): %s\n' "$model" "$variant" "$directory"
  generate "$model" "$variant" "$directory" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf 'Finished (status %s). Attempts and logs: data/%s/raw/\n' "$status" "$dataset"
exit "$status"
