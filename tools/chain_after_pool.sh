#!/bin/bash
# chain_after_pool.sh — autonomous chain that fires when the current 4-slot
# pool drains.
#
# Pipeline:
#   Phase A: wait until no validation/ml.py or validation/map.py is running
#   Phase B: launch cold ML (csuros_match) pool for the 5 datasets that
#            don't have a fresh-on-disk cold ML fit (williams already in
#            flight separately, so it is excluded from the queue)
#   Phase C: wait for that pool to drain
#   Phase D: stale-data audit + cleanup + README cross-check + git push
#
# Designed to be launched with nohup + disown so it survives Claude session
# death:
#
#     cd <your local clone of recount>
#     nohup tools/chain_after_pool.sh > validation/logs/chain_after_pool.log 2>&1 &
#     disown
set -u
cd "$(dirname "$0")/.."

LOG=validation/logs/chain_after_pool.log
mkdir -p validation/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

count_active() {
  ps -ax | grep -E 'validation/(ml|map|mixture_ml)\.py' | grep -v grep | wc -l | tr -d ' '
}

# Phase A: wait for current pool to fully drain.
log "=== Phase A: waiting for current pool to drain ==="
log "  active fits at start: $(count_active)"
while [ "$(count_active)" -gt 0 ]; do
  sleep 180
done
log "  all fits drained — moving to Phase B"

# Phase B: launch cold ML pool for the 5 remaining datasets.
log "=== Phase B: launching cold ML pool ==="
nohup tools/pool_fits.sh 4 4 tools/queues/cold_ml_remaining.txt coldML \
  > /dev/null 2>&1 &
COLD_POOL_PID=$!
disown
log "  cold ML pool launched (bash PID $COLD_POOL_PID)"

# Give it 30 s to spawn its children before counting actives.
sleep 30

# Phase C: wait for cold ML to drain.
log "=== Phase C: waiting for cold ML pool to drain ==="
log "  active fits at start: $(count_active)"
while [ "$(count_active)" -gt 0 ]; do
  sleep 180
done
log "  cold ML pool drained — moving to Phase D"

# Phase D: stale-data audit + cleanup + README cross-check + push.
log "=== Phase D: audit + cleanup + README cross-check ==="
PYTHONPATH=. python3 tools/audit_and_cleanup.py 2>&1 | tee -a "$LOG" || true

log "=== chain complete ==="
