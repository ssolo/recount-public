#!/bin/bash
# queue_full_subset_run.sh — single canonical BFGS recipe applied to every
# dataset (dpann80, proteo75, eury114, ed194, williams, coleman, arc269)
# across 4 fit modes, matching the canonical optimizer setup:
#
#   native_bfgs   --cycle-iters 100   --max-cycles 12   --seed 2025
#
# Each fit mode writes to the canonical subdir the plot scripts already
# read from, so subclade_ancestor_comparison.py picks up the results
# automatically.
#
# Fit modes:
#   1. cold ML BOUNDED      → bounded_csuros/        ml.py  --num-starts 1 --polish-sigmas ""
#   2. Brownian-extend ML   → brownian_extend_<ds>/  brownian_extend_ml.py (warm-starts from mode 4)
#   3. MAP σ=1 cold         → map_sigma1/            map.py --sigma 1.0 --num-starts 1
#   4. MAP Brownian σ=1     → brownian_<ds>/         map.py --prior brownian --sigma-brownian-* 1.0
#
# Dependency: mode 2 (Brownian-extend) reads the Brownian rates produced
# by mode 4. The default MODES order "1 3 4 2" guarantees mode 4 runs
# before mode 2 for each dataset.
#
# Skip rule: if {OUT_DIR}/{DS}_{tag}_summary.json already exists and was
# produced with --cycle-iters 100 --max-cycles 12 (or init_mode
# "brownian_extend" for mode 2), the run is skipped.
#
# Usage:
#   bash validation/queue_full_subset_run.sh              # all 6 datasets, all 4 modes
#   bash validation/queue_full_subset_run.sh dpann80      # one dataset, all 4 modes
#   MODES="1 3" bash validation/queue_full_subset_run.sh  # just cold ML + MAP cold
#
# Threads: NUM_THREADS env var (default 8). Coleman + ed194 benefit from 16.
# Log: validation/logs/queue_full_subset_<timestamp>.log + per-job logs.

set -u

cd "$(git rev-parse --show-toplevel)" 2>/dev/null || cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

NUM_THREADS="${NUM_THREADS:-8}"
# Order: 1 (cold ML), 3 (MAP cold), 4 (MAP Brownian), then 2 (Brownian-extend
# ML) which depends on mode 4's output.
MODES="${MODES:-1 3 4 2}"

# Argv selects datasets; default = all 7 in size order (smallest first;
# arc269 last because it dominates per-fit wall and the skip-logic will
# typically only need to run its missing modes).
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(williams dpann80 proteo75 eury114 ed194 coleman arc269)
fi

TS=$(date +%Y%m%d_%H%M%S)
QLOG="validation/logs/queue_full_subset_${TS}.log"
mkdir -p validation/logs
echo "[$(date)] queue_full_subset starting  datasets=${DATASETS[*]}  modes=$MODES  threads=$NUM_THREADS" | tee "$QLOG"

# ---------- helpers --------------------------------------------------------
run() {
    local TAG=$1; shift
    local LOG="validation/logs/${TAG}_${TS}.log"
    echo "[$(date)] >>> ${TAG}  (log: $LOG)" | tee -a "$QLOG"
    if PYTHONPATH="$(pwd)" python3 -u "$@" > "$LOG" 2>&1; then
        echo "[$(date)] <<< ${TAG} OK" | tee -a "$QLOG"
    else
        echo "[$(date)] <<< ${TAG} FAILED rc=$?" | tee -a "$QLOG"
    fi
}

already_done() {
    # already_done <dataset> <out-dir> <tag-glob>
    local DS=$1 OUT=$2 TAG=$3
    local SUM
    for SUM in "$OUT"/${DS}*summary.json; do
        [ -f "$SUM" ] || continue
        python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
fs = d.get('fit_settings', {})
ok = fs.get('cycle_iters') == 100 and fs.get('max_cycles') == 12
sys.exit(0 if ok else 1)
" "$SUM" 2>/dev/null && return 0
    done
    return 1
}

# ---------- fit-mode dispatcher -------------------------------------------
fit_one() {
    local DS=$1 MODE=$2

    case "$MODE" in
        1) # cold ML BOUNDED
            local OUT="validation/outputs/bounded_csuros"
            if already_done "$DS" "$OUT" "ml"; then
                echo "[$(date)] SKIP ${DS}_mode1: bounded_csuros/${DS} already exists with canonical settings" | tee -a "$QLOG"
                return
            fi
            mkdir -p "$OUT"
            run "${DS}_cold_ml" \
                validation/ml.py --dataset "$DS" \
                --num-starts 1 --polish-sigmas "" \
                --cycle-iters 100 --max-cycles 12 --seed 2025 \
                --optimizer native_bfgs --num-threads "$NUM_THREADS" \
                --out-dir "$OUT"
            ;;
        2) # Brownian-extend ML — depends on mode 4 (Brownian MAP) having run first
            local OUT="validation/outputs/brownian_extend_${DS}"
            local SUM="$OUT/${DS}_summary.json"
            if [ -f "$SUM" ] && python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get('fit_settings', {}).get('init_mode') == 'brownian_extend' else 1)
" "$SUM" 2>/dev/null; then
                echo "[$(date)] SKIP ${DS}_mode2: brownian_extend_${DS} already exists" | tee -a "$QLOG"
                return
            fi
            local BROWN_RATES="validation/outputs/brownian_${DS}/${DS}_sigma1.0_final_rates.npz"
            if [ ! -f "$BROWN_RATES" ]; then
                echo "[$(date)] SKIP ${DS}_mode2: prereq missing $BROWN_RATES (run mode 4 first)" | tee -a "$QLOG"
                return
            fi
            mkdir -p "$OUT"
            run "${DS}_brown_extend_ml" \
                validation/brownian_extend_ml.py --dataset "$DS" \
                --cycle-iters 100 --max-cycles 5 --seed 2025 \
                --num-threads "$NUM_THREADS" \
                --out-dir validation/outputs
            ;;
        3) # MAP σ=1 cold
            local OUT="validation/outputs/map_sigma1"
            if already_done "$DS" "$OUT" "sigma1.0"; then
                echo "[$(date)] SKIP ${DS}_mode3: map_sigma1/${DS} already exists with canonical settings" | tee -a "$QLOG"
                return
            fi
            mkdir -p "$OUT"
            run "${DS}_MAP_sigma1" \
                validation/map.py --dataset "$DS" --sigma 1.0 \
                --num-starts 1 --cycle-iters 100 --max-cycles 12 --seed 2025 \
                --optimizer native_bfgs --num-threads "$NUM_THREADS" \
                --out-dir "$OUT"
            ;;
        4) # MAP Brownian σ=1
            local OUT="validation/outputs/brownian_${DS}"
            if already_done "$DS" "$OUT" "sigma1.0"; then
                echo "[$(date)] SKIP ${DS}_mode4: brownian_${DS} already exists with canonical settings" | tee -a "$QLOG"
                return
            fi
            mkdir -p "$OUT"
            run "${DS}_MAP_brownian" \
                validation/map.py --dataset "$DS" \
                --sigma 1.0 --prior brownian \
                --sigma-brownian-gain 1.0 --sigma-brownian-dup 1.0 --sigma-brownian-length 1.0 \
                --num-starts 1 --cycle-iters 100 --max-cycles 12 --seed 2025 \
                --optimizer native_bfgs --num-threads "$NUM_THREADS" \
                --out-dir "$OUT"
            ;;
        *)
            echo "Unknown mode $MODE" >&2
            ;;
    esac
}

# ---------- main loop ------------------------------------------------------
for DS in "${DATASETS[@]}"; do
    echo "[$(date)] === dataset: $DS ===" | tee -a "$QLOG"
    for M in $MODES; do
        fit_one "$DS" "$M"
    done
done

# ---------- arc269 sensitivity check: Ωmin=4 (subset-style) ----------------
# Same Brownian + extend pipeline as the canonical modes 4 + 2, but with
# min_copies_override=4 to make arc269 comparable to the four subsets
# (which all use Ωmin=4). Lands in brownian_arc269_omin4/ +
# brownian_extend_arc269_omin4/.
arc269_omin4_check() {
    local OUT4="validation/outputs/brownian_arc269_omin4"
    local OUT2="validation/outputs/brownian_extend_arc269_omin4"
    local RATES="$OUT4/arc269_sigma1.0_final_rates.npz"

    # mode 4: MAP Brownian σ=1, Ωmin=4
    if [ -f "$OUT4/arc269_sigma1.0_summary.json" ]; then
        echo "[$(date)] SKIP arc269_omin4_brownian: $OUT4 already exists" | tee -a "$QLOG"
    else
        mkdir -p "$OUT4"
        run "arc269_omin4_brownian" \
            validation/map.py --dataset arc269 \
            --min-copies-override 4 \
            --sigma 1.0 --prior brownian \
            --sigma-brownian-gain 1.0 --sigma-brownian-dup 1.0 --sigma-brownian-length 1.0 \
            --num-starts 1 --cycle-iters 100 --max-cycles 12 --seed 2025 \
            --optimizer native_bfgs --num-threads "$NUM_THREADS" \
            --out-dir "$OUT4"
    fi

    # mode 2: Brownian-extend ML, Ωmin=4
    if [ -f "$OUT2/arc269_summary.json" ]; then
        echo "[$(date)] SKIP arc269_omin4_brown_extend: $OUT2 already exists" | tee -a "$QLOG"
    elif [ ! -f "$RATES" ]; then
        echo "[$(date)] SKIP arc269_omin4_brown_extend: prereq $RATES missing (mode 4 failed?)" | tee -a "$QLOG"
    else
        mkdir -p "$OUT2"
        run "arc269_omin4_brown_extend" \
            validation/brownian_extend_ml.py --dataset arc269 \
            --min-copies-override 4 \
            --rates-from "$RATES" \
            --out-subdir "brownian_extend_arc269_omin4" \
            --cycle-iters 100 --max-cycles 5 --seed 2025 \
            --num-threads "$NUM_THREADS" \
            --out-dir validation/outputs
    fi
}
arc269_omin4_check

# ---------- post: per-family tables + plots --------------------------------
echo "[$(date)] regenerating per-family posterior tables..." | tee -a "$QLOG"
for OUT in validation/outputs/bounded_csuros \
           validation/outputs/map_sigma1 \
           validation/outputs/brownian_* \
           validation/outputs/brownian_extend_* \
           validation/outputs/brownian_arc269_omin4 \
           validation/outputs/brownian_extend_arc269_omin4; do
    [ -d "$OUT" ] || continue
    python3 -u validation/per_family_presence.py --all --fit-dir "$OUT" \
        --num-threads "$NUM_THREADS" >> "$QLOG" 2>&1 || \
        echo "  per_family_presence failed for $OUT" | tee -a "$QLOG"
done

echo "[$(date)] regenerating plots..." | tee -a "$QLOG"
python3 -u validation/subclade_ancestor_comparison.py --variant bounded >> "$QLOG" 2>&1 || true

echo "[$(date)] QUEUE COMPLETE." | tee -a "$QLOG"
