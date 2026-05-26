#!/bin/bash
# pool_fits.sh — keep a fixed number of recount ML/MAP fits running in parallel
#
# Reads a queue file (one "dataset kind" per line; e.g. "dpann80 sota_ml" or
# "ed194 map_sigma1") and maintains exactly $SLOTS fits active at any time,
# each pinned to $THREADS threads. Submits the next queue item the moment a
# slot frees. Sized for a 16-core Apple Silicon Mac:
#
#     SLOTS=4 × THREADS=4  → no oversubscription, full CPU all the time.
#
# Usage:
#     tools/pool_fits.sh <slots> <threads> <queue_file> [tag]
#
# Example:
#     tools/pool_fits.sh 4 4 tools/queues/all_six_sota_map.txt allC
#
# Queue file format (one entry per line):
#     <dataset> <kind>
# where <kind> is "sota_ml" or "map_sigma1" or "csuros_match".
# Blank lines and # comments are ignored.
#
# Each fit's log is written to validation/logs/<dataset>_<kind>_<tag>_<TS>.log
# A master log goes to validation/logs/pool_<tag>_<TS>.log
set -euo pipefail

SLOTS="${1:-4}"
THREADS="${2:-4}"
QUEUE_FILE="${3:?queue file required}"
TAG="${4:-pool}"

cd "$(dirname "$0")/.."  # cd to repo root (assumes tools/ is at top level)

if [ ! -f "$QUEUE_FILE" ]; then
  echo "Queue file not found: $QUEUE_FILE" >&2
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p validation/logs validation/outputs/sota_ml validation/outputs/map_sigma1 validation/outputs/csuros_match
export RECOUNT_NUM_THREADS=$THREADS

MASTER_LOG="validation/logs/pool_${TAG}_${TS}.log"
exec >>"$MASTER_LOG" 2>&1

log() { echo "[$(date +%H:%M)] $*"; }

# Load queue into a bash array (skip blank / # lines)
QUEUE=()
while IFS= read -r line; do
  line="${line%%#*}"          # strip inline comments
  line="$(echo "$line" | xargs)"   # trim whitespace
  [ -z "$line" ] && continue
  QUEUE+=("$line")
done < "$QUEUE_FILE"

log "Pool scheduler starting — slots=$SLOTS threads=$THREADS queue_size=${#QUEUE[@]}"
log "Queue: ${QUEUE[*]}"

# Map (dataset, kind) → (out-dir, summary-suffix). Used by both the
# pre-launch idempotency guard and any external checks.
out_dir_for_kind() {
  case "$1" in
    csuros_match)         echo "validation/outputs/csuros_match" ;;
    sota_ml|sota_ml_1start) echo "validation/outputs/sota_ml" ;;
    map_sigma1)           echo "validation/outputs/map_sigma1" ;;
    map_sigma1_15start)   echo "validation/outputs/profile_likelihood_arc269" ;;
    mixture_K2)           echo "validation/outputs/mixture_K2_bounded" ;;
    *)                    echo "" ;;
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

# Pre-launch guard: skip if the same fit is already in flight OR if a
# fresh summary already exists on disk (mtime > post-port cutoff). Set
# POOL_FORCE=1 to bypass both checks.
POST_PORT_CUTOFF=$(date -j -f '%Y-%m-%d %H:%M' '2026-05-18 16:30' +%s 2>/dev/null || echo 0)
should_skip() {
  local ds=$1 kind=$2
  [ "${POOL_FORCE:-0}" = "1" ] && return 1
  local outdir sumpath
  outdir=$(out_dir_for_kind "$kind")
  [ -z "$outdir" ] && return 1   # unknown kind — let launch_one report it

  # 1) Already running for this dataset + out-dir combination?
  if ps -ax -o command= 2>/dev/null \
       | grep -E "validation/(ml|map|mixture_ml)\.py.*--dataset[= ]$ds .*--out-dir[= ]$outdir" \
       | grep -v grep > /dev/null; then
    log "  ! skip $ds $kind: same fit already in flight (out-dir $outdir)"
    return 0
  fi

  # 2) Fresh summary on disk (mtime > 2026-05-18 16:30 cutoff)?
  sumpath=$(summary_path_for "$ds" "$kind")
  if [ -n "$sumpath" ] && [ -f "$sumpath" ]; then
    local mt
    mt=$(stat -f %m "$sumpath" 2>/dev/null || stat -c %Y "$sumpath" 2>/dev/null || echo 0)
    if [ "$mt" -gt "$POST_PORT_CUTOFF" ]; then
      log "  ✓ skip $ds $kind: fresh summary at $sumpath"
      return 0
    fi
  fi
  return 1
}

launch_one() {
  local ds=$1 kind=$2
  if should_skip "$ds" "$kind"; then
    return 0
  fi
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
    sota_ml_1start)
      # 1 random start + 2-round polish — for arc269 where 3 starts would
      # take 60+ hours and a single deep start is the right tradeoff.
      PYTHONPATH=. nice -n 5 python3 -u validation/ml.py --dataset $ds \
        --num-starts 1 --polish-sigmas 0.05,0.10 --num-polish 2 \
        --cycle-iters 200 --max-cycles 25 --optimizer native_bfgs \
        --out-dir validation/outputs/sota_ml --seed 2025 \
        --num-threads $THREADS > "$log_path" 2>&1 &
      ;;
    map_sigma1_15start)
      # 15-start global MAP σ=1 — used for the profile-likelihood column
      # (`validation/outputs/profile_likelihood_arc269/`). Goes deeper than
      # the standard 3-start MAP σ=1; tightens the bootstrap CI on arc269.
      PYTHONPATH=. nice -n 5 python3 -u validation/map.py --dataset $ds \
        --sigma 1.0 --num-starts 15 --cycle-iters 200 --max-cycles 25 \
        --optimizer native_bfgs \
        --out-dir validation/outputs/profile_likelihood_arc269 \
        --seed 2025 --num-threads $THREADS > "$log_path" 2>&1 &
      ;;
    mixture_K2)
      # K=2 LogisticShift mixture, bounded (logit dup constraint).
      # Requires validation/mixture_ml.py to have been migrated from
      # LOG_RATE_CLIP to logit — see PROGRESS.md for status.
      PYTHONPATH=. nice -n 5 python3 -u validation/mixture_ml.py --dataset $ds \
        --K 2 --num-starts 3 --cycle-iters 200 --max-cycles 25 \
        --optimizer native_bfgs --bounded \
        --out-dir validation/outputs/mixture_K2_bounded --seed 2025 \
        --num-threads $THREADS > "$log_path" 2>&1 &
      ;;
    *)
      log "  ! unknown kind: $kind" ; return 1 ;;
  esac
  log "  ▶ launched $ds $kind (log: $log_path)"
}

active_count() {
  ps aux | grep -E 'validation/(ml|map|mixture_ml)\.py' | grep -v grep | wc -l | tr -d ' '
}

top_up() {
  local active need
  active=$(active_count)
  need=$((SLOTS - active))
  while [ $need -gt 0 ] && [ ${#QUEUE[@]} -gt 0 ]; do
    local pair="${QUEUE[0]}"
    QUEUE=("${QUEUE[@]:1}")
    launch_one $pair || true
    need=$((need - 1))
  done
}

# Initial fill
top_up

# Loop: keep filling until queue empty AND no fits running
while [ ${#QUEUE[@]} -gt 0 ] || [ "$(active_count)" -gt 0 ]; do
  sleep 60
  top_up
done

log "Pool done — all fits complete"
