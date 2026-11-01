#!/usr/bin/env bash
set -euo pipefail
[[ $# == 0 ]] || { printf 'Usage: bash notebooks/m6_showroom_data/generate.sh\n' >&2; exit 2; }
[[ -f notebooks/m6_showroom_data/gen.md ]] || { printf 'Run from the project root.\n' >&2; exit 1; }
stamp=$(date -u +%Y%m%dT%H%M%S%6NZ)
logs="tmp/showroom-generation/${stamp}_$$"
mkdir -p "$logs" data/showroom/raw
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true; exit 130' INT TERM
for job in 'gemini-4-argon high' 'glm-5.3 max'; do
  read -r model variant <<< "$job"
  seed=$((RANDOM * 10 + ${#pids[@]})) # Different last digit for each job.
  printf 'Starting %s (%s), seed=%s\n' "$model" "$variant" "$seed"
  "${OPENCODE_BIN:-opencode}" run --agent build --model "opencode/$model" --variant "$variant" \
    --file notebooks/m6_showroom_data/gen.md --format json \
    "Follow gen.md. Use UTC timestamp $stamp and batch seed $seed." \
    < /dev/null > "$logs/${model}_s${seed}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf 'Finished (status %s). Logs: %s\n' "$status" "$logs"
exit "$status"
