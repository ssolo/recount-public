"""Example 3 — The corrected LL depends on Ωmin: Ωmin=2 vs Ωmin=4.

Csurös 2026 (Materials and Methods) states that the phylogenetic-
reconciliation datasets are filtered to families with ≥4 total copies,
so the sampling-bias correction (paper Eq 4) must condition at
Ωmin = 4. Count fits these datasets at Ωmin = 4 — its bundled rate
models are Ωmin=4 optima: the gradient of the corrected log-likelihood
vanishes at Ωmin=4 and is large (hundreds of nats per unit) at Ωmin=2.

Demonstrates why the conditioning matters: at Count's stored rates,
evaluate the corrected log-likelihood ln L*(Ξ) = Σ_f ln L(Ξ_f)
- F·ln(1-L(0)) for each min4-filtered table under the WRONG
conditioning (Ωmin=2) and the CORRECT one (Ωmin=4, matching the ≥4
filter). They differ by hundreds to thousands of nats — the size of
the error from scoring an Ωmin=4-filtered table at Ωmin=2.

Prerequisites:
  - librecount.dylib built
  - docs/csuros_data/arc269/ present (bundled in this repo;
    override with the RECOUNT_CSUROS_DATA env var to point at an
    external copy of Csurös' bundle)
  - validation/Williams2017.countxml.gz present

Usage:
  PYTHONPATH=. python3 examples/03_focal_paper_LL_gap_at_omega_4.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from recount.io.countxml import load_countxml
from recount.native_backend import corrected_log_likelihood_native


# Csurös data bundle is mirrored under docs/csuros_data/ in the repo.
# Override with RECOUNT_CSUROS_DATA env var for an external copy.
import os
_DATA_ROOT = Path(os.environ.get("RECOUNT_CSUROS_DATA", "docs/csuros_data"))
ARC269 = _DATA_ROOT / "arc269"
WILLIAMS = Path("validation/Williams2017.countxml.gz")


def list_focal_tables() -> Iterable[tuple]:
    """Yield (path, session_id, table_name) for every min4-filtered table."""
    files = [
        WILLIAMS,
        ARC269 / "datasets-sims-ED194-E114-D80-P75.countxml.gz",
        ARC269 / "cryptic-e114-min4.countxml.gz",
        ARC269 / "cryptic-t94-min4.countxml.gz",
    ]
    for path in files:
        if not path.exists():
            continue
        try:
            sessions = load_countxml(str(path))
        except Exception as e:
            print(f"# skipping {path.name} ({e})")
            continue
        for sid, sess in sessions.items():
            if sess.rates is None or not np.all(np.isfinite(sess.rates.gain[:-1])):
                continue
            for tname, tbl in sess.tables.items():
                if "min4" not in tname:
                    continue
                if tbl.profiles.shape[0] == 0:
                    continue
                yield path, sid, tname, sess, tbl


def main() -> int:
    print("# Focal paper datasets: corrected ln L* at Ωmin=2 (wrong conditioning)"
          " vs Ωmin=4 (correct — the Ωmin Count fit at)")
    print()
    print(f"# {'dataset':>30}  {'session':>22}  {'table':>40}"
          f"  {'F':>6}  {'LL (Ωmin=2)':>14}  {'LL (Ωmin=4)':>14}  {'ΔLL':>10}")
    rows = []
    for path, sid, tname, sess, tbl in list_focal_tables():
        profiles = tbl.profiles.astype(np.int32)
        F = profiles.shape[0]
        g, l, d, t = (
            sess.rates.gain, sess.rates.loss, sess.rates.dup, sess.rates.length
        )
        try:
            ll2 = corrected_log_likelihood_native(
                sess.tree, g, l, d, t, profiles, min_copies=2)
            ll4 = corrected_log_likelihood_native(
                sess.tree, g, l, d, t, profiles, min_copies=4)
        except Exception as e:
            print(f"# {path.name}/{sid}/{tname}: {type(e).__name__}: {e}")
            continue
        dll = ll4 - ll2
        ds_name = path.stem.replace(".countxml", "")
        print(f"  {ds_name[:30]:>30}  {sid[:22]:>22}  {tname[:40]:>40}"
              f"  {F:>6}  {ll2:>14.2f}  {ll4:>14.2f}  {dll:>+10.2f}")
        rows.append((ds_name, sid, tname, F, ll2, ll4, dll))

    print()
    print(f"# Summary: {len(rows)} focal datasets evaluated.")
    if rows:
        deltas = [r[6] for r in rows]
        print(f"# ΔLL (Ωmin=4 − Ωmin=2):  min={min(deltas):+.1f}  "
              f"median={float(np.median(deltas)):+.1f}  max={max(deltas):+.1f}")
        biggest = max(rows, key=lambda r: r[6])
        print(f"# Largest Ωmin=2-vs-4 gap: {biggest[0]} / {biggest[1]} / {biggest[2]}: "
              f"{biggest[6]:+.1f} nats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
