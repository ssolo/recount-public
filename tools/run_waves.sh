#!/bin/bash
# run_waves.sh — run recount fits in sequential waves
#
# Reads a "waves file" where each line is a comma-separated list of
# "<dataset> <kind>" pairs forming one wave. Runs all pairs in a wave
# concurrently with $THREADS threads each, blocks on the wave with `wait`,
# then advances to the next wave.
#
# Less CPU-efficient than pool_fits.sh (idle cores while waiting for a wave
# to fully complete) but easier to reason about for short benchmark sweeps.
#
# Usage:
#     tools/run_waves.sh <threads> <waves_file> [tag]
#
# Example waves file (tools/queues/all_six_sota_map.waves):
#     williams sota_ml, williams map_sigma1, dpann80 sota_ml, dpann80 map_sigma1
#     proteo75 sota_ml, proteo75 map_sigma1, eury114 sota_ml, eury114 map_sigma1
#     ed194 sota_ml, ed194 map_sigma1, arc269 sota_ml, arc269 map_sigma1
set -euo pipefail

THREADS="${1:-4}"
WAVES_FILE="${2:?waves file required}"
TAG="${3:-waves}"

cd "$(dirname "$0")/.."

if [ ! -f "$WAVES_FILE" ]; then
  echo "Waves file not found: $WAVES_FILE" >&2
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p validation/logs validation/outputs/sota_ml validation/outputs/map_sigma1 validation/outputs/csuros_match
export RECOUNT_NUM_THREADS=$THREADS

MASTER_LOG="validation/logs/waves_${TAG}_${TS}.log"
exec >>"$MASTER_LOG" 2>&1

log() { echo "[$(date +%H:%M)] $*"; }

launch_one() {
  local ds=$1 kind=$2
  local log_path="validation/logs/${ds}_${kind}_${TAG}_${TS}.log"
  case "$kind" in
    sota_ml)
      PYTHONPATH=. nice -n 5 python3 -u validation/ml.py --dataset $ds \
        --num-starts 3 --polish-sigmas 0.05,0.10 --num-polish 2 \
        --cycle-iters 200 --max-cycles 25 --optimizer native_bfgs \
        --out-dir validation/outputs/sota_ml --seed 2025 \
        --num-threads $THREADS > "$log_path" 2>&1 &
      ;;
    map_sigma1)
      PYTHONPATH=. nice -n 5 python3 -u validation/map.py --dataset $ds \
        --sigma 1.0 --num-starts 3 --cycle-iters 200 --max-cycles 25 \
        --optimizer native_bfgs --out-dir validation/outputs/map_sigma1 \
        --seed 2025 --num-threads $THREADS > "$log_path" 2>&1 &
      ;;
    csuros_match)
      PYTHONPATH=. nice -n 5 python3 -u validation/ml.py --dataset $ds \
        --num-starts 1 --cycle-iters 200 --max-cycles 40 \
        --optimizer native_bfgs --out-dir validation/outputs/csuros_match \
        --seed 2025 --num-threads $THREADS > "$log_path" 2>&1 &
      ;;
    *)
      log "  ! unknown kind: $kind" ;;
  esac
}

WAVE_N=0
while IFS= read -r line; do
  line="${line%%#*}"
  line="$(echo "$line" | xargs)"
  [ -z "$line" ] && continue
  WAVE_N=$((WAVE_N + 1))
  log "=== Wave $WAVE_N === $line"
  IFS=',' read -ra pairs <<< "$line"
  for pair in "${pairs[@]}"; do
    pair="$(echo "$pair" | xargs)"
    [ -z "$pair" ] && continue
    launch_one $pair
    log "  ▶ launched $pair"
  done
  log "  waiting for wave $WAVE_N to finish..."
  wait
  log "  wave $WAVE_N done"
done < "$WAVES_FILE"

log "=== All waves complete ==="
