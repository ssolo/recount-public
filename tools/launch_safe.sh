#!/bin/bash
# launch_safe.sh — single-fit launcher with pre-flight idempotency checks.
#
# Use for solo-mode launches (1 fit, 16 threads, full M4 Max). Refuses to
# launch if the same fit is already running, or if a fresh summary
# (mtime > 2026-05-18 16:30 JST cutoff) already exists on disk. Override
# either check with LAUNCH_FORCE=1.
#
# Usage:
#   tools/launch_safe.sh <dataset> <kind> [threads]
#
# Examples:
#   tools/launch_safe.sh arc269 csuros_match           # 16 threads (default)
#   tools/launch_safe.sh arc269 csuros_match 16
#   LAUNCH_FORCE=1 tools/launch_safe.sh arc269 sota_ml_1start
#
# Kinds:  csuros_match | sota_ml | sota_ml_1start | map_sigma1 |
#         map_sigma1_15start | mixture_K2
#
# Reuses the kind dispatch from tools/pool_fits.sh (sources its case
# block via subshell with active_count / launch_one). Output writes to
# the standard validation/outputs/<kind-specific-subdir>/ tree.
set -u
cd "$(dirname "$0")/.."

DS="${1:?usage: launch_safe.sh <dataset> <kind> [threads]}"
KIND="${2:?usage: launch_safe.sh <dataset> <kind> [threads]}"
THREADS="${3:-16}"

mkdir -p validation/logs

# --- shared kind→outdir mapping (must stay in sync with tools/pool_fits.sh) ---
out_dir_for_kind() {
  case "$1" in
    csuros_match)           echo "validation/outputs/csuros_match" ;;
    sota_ml|sota_ml_1start) echo "validation/outputs/sota_ml" ;;
    map_sigma1)             echo "validation/outputs/map_sigma1" ;;
    map_sigma1_15start)     echo "validation/outputs/profile_likelihood_arc269" ;;
    mixture_K2)             echo "validation/outputs/mixture_K2_bounded" ;;
    *)                      echo "" ;;
  esac
}
summary_path_for() {
  local ds=$1 kind=$2 outdir
  outdir=$(out_dir_for_kind "$kind")
  [ -z "$outdir" ] && { echo ""; return; }
  case "$kind" in
    map_sigma1|map_sigma1_15start) echo "${outdir}/${ds}_sigma1.0_summary.json" ;;
    mixture_K2)                    echo "${outdir}/${ds}_K2_summary.json" ;;
    *)                             echo "${outdir}/${ds}_summary.json" ;;
  esac
}

OUTDIR=$(out_dir_for_kind "$KIND")
if [ -z "$OUTDIR" ]; then
  echo "ERROR: unknown kind '$KIND'." >&2
  echo "Valid kinds: csuros_match | sota_ml | sota_ml_1start | map_sigma1 | map_sigma1_15start | mixture_K2" >&2
  exit 2
fi
mkdir -p "$OUTDIR"

# --- pre-flight check 1: already in flight? ---
if [ "${LAUNCH_FORCE:-0}" != "1" ]; then
  if ps -ax -o command= 2>/dev/null \
       | grep -E "validation/(ml|map|mixture_ml)\.py.*--dataset[= ]$DS .*--out-dir[= ]$OUTDIR" \
       | grep -v grep > /dev/null; then
    echo "REFUSE: $DS $KIND is already running (out-dir $OUTDIR)" >&2
    echo "        Set LAUNCH_FORCE=1 to override." >&2
    exit 3
  fi
fi

# --- pre-flight check 2: fresh summary on disk? ---
if [ "${LAUNCH_FORCE:-0}" != "1" ]; then
  SUMPATH=$(summary_path_for "$DS" "$KIND")
  if [ -n "$SUMPATH" ] && [ -f "$SUMPATH" ]; then
    CUTOFF=$(date -j -f '%Y-%m-%d %H:%M' '2026-05-18 16:30' +%s 2>/dev/null || echo 0)
    MT=$(stat -f %m "$SUMPATH" 2>/dev/null || stat -c %Y "$SUMPATH" 2>/dev/null || echo 0)
    if [ "$MT" -gt "$CUTOFF" ]; then
      echo "REFUSE: fresh summary already exists at $SUMPATH" >&2
      echo "        (mtime=$(date -r $MT '+%Y-%m-%d %H:%M') > cutoff $(date -r $CUTOFF '+%Y-%m-%d %H:%M'))" >&2
      echo "        Set LAUNCH_FORCE=1 to override, or delete the summary first." >&2
      exit 4
    fi
  fi
fi

# --- pre-flight check 3: don't oversubscribe ---
ACTIVE=$(ps -ax | grep -E 'validation/(ml|map|mixture_ml)\.py' | grep -v grep | wc -l | tr -d ' ')
if [ "$ACTIVE" -gt 0 ] && [ "${LAUNCH_FORCE:-0}" != "1" ]; then
  echo "WARN: $ACTIVE other fit(s) already running — launching anyway, but at $THREADS threads"
  echo "      this may oversubscribe your CPU. Set LAUNCH_FORCE=0 (default) to skip the warning"
  echo "      or kill the others with: pkill -f 'validation/(ml|map|mixture_ml)\\.py'"
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="validation/logs/${DS}_${KIND}_solo_${TS}.log"

echo "▶ launching $DS $KIND on $THREADS threads"
echo "  out-dir: $OUTDIR"
echo "  log:     $LOG"

case "$KIND" in
  csuros_match)
    PYTHONPATH=. nohup python3 -u validation/ml.py --dataset $DS \
      --num-starts 1 --cycle-iters 200 --max-cycles 40 \
      --optimizer native_bfgs --out-dir "$OUTDIR" \
      --seed 2025 --num-threads $THREADS > "$LOG" 2>&1 &
    ;;
  sota_ml)
    PYTHONPATH=. nohup python3 -u validation/ml.py --dataset $DS \
      --num-starts 3 --polish-sigmas 0.05,0.10 --num-polish 2 \
      --cycle-iters 200 --max-cycles 25 --optimizer native_bfgs \
      --out-dir "$OUTDIR" --seed 2025 \
      --num-threads $THREADS > "$LOG" 2>&1 &
    ;;
  sota_ml_1start)
    PYTHONPATH=. nohup python3 -u validation/ml.py --dataset $DS \
      --num-starts 1 --polish-sigmas 0.05,0.10 --num-polish 2 \
      --cycle-iters 200 --max-cycles 25 --optimizer native_bfgs \
      --out-dir "$OUTDIR" --seed 2025 \
      --num-threads $THREADS > "$LOG" 2>&1 &
    ;;
  map_sigma1)
    PYTHONPATH=. nohup python3 -u validation/map.py --dataset $DS \
      --sigma 1.0 --num-starts 3 --cycle-iters 200 --max-cycles 25 \
      --optimizer native_bfgs --out-dir "$OUTDIR" \
      --seed 2025 --num-threads $THREADS > "$LOG" 2>&1 &
    ;;
  map_sigma1_15start)
    PYTHONPATH=. nohup python3 -u validation/map.py --dataset $DS \
      --sigma 1.0 --num-starts 15 --cycle-iters 200 --max-cycles 25 \
      --optimizer native_bfgs --out-dir "$OUTDIR" \
      --seed 2025 --num-threads $THREADS > "$LOG" 2>&1 &
    ;;
  mixture_K2)
    PYTHONPATH=. nohup python3 -u validation/mixture_ml.py --dataset $DS \
      --K 2 --num-starts 3 --cycle-iters 200 --max-cycles 25 \
      --optimizer native_bfgs --bounded \
      --out-dir "$OUTDIR" --seed 2025 \
      --num-threads $THREADS > "$LOG" 2>&1 &
    ;;
esac
disown
PID=$!
echo "  PID:     $PID"
echo "  tail with: tail -f $LOG"
