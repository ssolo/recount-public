"""refresh_readme_tables.py — rewrite README.md tables from fresh saved
fits. Idempotent.

Each table is identified by its preceding header text + its column-header
line, so the same row prefix (e.g. "| dpann80 |") doesn't accidentally
match the wrong table.

Cells without a fresh fit (mtime < 2026-05-18 16:30 JST, the post-port
cutoff) get the literal "pending".

Run by tools/hourly_checkup.sh.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
OUT = REPO / "validation" / "outputs"

CUTOFF = time.mktime(time.strptime("2026-05-18 16:30", "%Y-%m-%d %H:%M"))

# (label_in_table, internal_name); arc269 LACA has a distinct label.
ARC269_ROWS = [
    ("dpann80 (D80, 80 leaves)",    "dpann80"),
    ("proteo75 (P75, 75 leaves)",   "proteo75"),
    ("eury114 (E114, 114 leaves)",  "eury114"),
    ("ed194 (ED194, 194 leaves)",   "ed194"),
    ("arc269 LACA (269 leaves)",    "arc269"),
]
LL_ROWS = [
    ("dpann80",  "dpann80"),
    ("proteo75", "proteo75"),
    ("eury114",  "eury114"),
    ("ed194",    "ed194"),
    ("arc269",   "arc269"),
]


def _load_summary(kind: str, ds: str) -> Optional[dict]:
    suffix = "_sigma1.0" if kind == "map_sigma1" else ""
    p = OUT / kind / f"{ds}{suffix}_summary.json"
    if not p.exists() or p.stat().st_mtime < CUTOFF:
        return None
    with open(p) as f:
        return json.load(f)


def _load_rates(kind: str, ds: str):
    suffix = "_sigma1.0" if kind == "map_sigma1" else ""
    p = OUT / kind / f"{ds}{suffix}_final_rates.npz"
    if not p.exists() or p.stat().st_mtime < CUTOFF:
        return None
    with np.load(p) as f:
        return float(f["dup"].max()), float(f["gain"].max())


def _fmt_ll_delta(d: dict | None) -> str:
    if not d:
        return "pending"
    ll = (d.get("final") or {}).get("ll")
    cbs = (d.get("csuros_baseline") or {}).get("ll")
    if not isinstance(ll, float):
        return "pending"
    if isinstance(cbs, float):
        return f"{ll:,.0f} ({ll - cbs:+.0f})"
    return f"{ll:,.0f}"


def _fmt_fm(d: dict | None) -> str:
    if not d: return "pending"
    v = (d.get("final") or {}).get("root_families_corr")
    return f"{v:,.0f}" if isinstance(v, float) else "pending"


def _fmt_cp(d: dict | None) -> str:
    if not d: return "pending"
    v = (d.get("final") or {}).get("root_copies_corr")
    return f"{v:,.0f}" if isinstance(v, float) else "pending"


def _fmt_dup_gain(t) -> str:
    if t is None:
        return "pending"
    md, mg = t
    if mg >= 100:    mg_s = f"{mg:.0f}"
    elif mg >= 1:    mg_s = f"{mg:.2f}"
    else:            mg_s = f"{mg:.3f}"
    return f"{md:.3f} / {mg_s}"


def _rewrite_table(text: str, anchor_marker: str,
                   header_re: str, rows: list[tuple[str, list[str]]]) -> str:
    """Find the table whose header (column-header line) matches `header_re`,
    in the block following `anchor_marker` (literal substring). For each
    (label, cells) in `rows`, replace that row in the table.

    The label is matched as: line starts with `| <label> |` (whitespace-
    tolerant). Cells are joined as `| c1 | c2 | ... | cN |`.
    """
    idx = text.find(anchor_marker)
    if idx < 0:
        return text
    # Find the header line containing `header_re` after the anchor
    tail = text[idx:]
    m = re.search(header_re, tail, re.MULTILINE)
    if not m:
        return text
    header_line_end = idx + tail.find('\n', m.end()) + 1
    # Find the end of the table: first blank line after the header
    end_idx = text.find('\n\n', header_line_end)
    if end_idx < 0:
        end_idx = len(text)
    table_body = text[header_line_end:end_idx]
    # Process row by row
    new_body_lines = []
    for line in table_body.splitlines(keepends=True):
        replaced = False
        for label, cells in rows:
            row_re = re.compile(rf"^\|\s*{re.escape(label)}\s*\|")
            if row_re.match(line.rstrip("\n")):
                trailing_nl = "\n" if line.endswith("\n") else ""
                new_body_lines.append(
                    f"| {label} | " + " | ".join(cells) + " |" + trailing_nl
                )
                replaced = True
                break
        if not replaced:
            new_body_lines.append(line)
    new_body = "".join(new_body_lines)
    return text[:header_line_end] + new_body + text[end_idx:]


def main():
    text = README.read_text()
    orig = text

    csuros_fm = {"dpann80":"1,210","proteo75":"2,096","eury114":"1,453",
                 "ed194":"1,419","arc269":"*— not fit —*"}
    csuros_cp = {"dpann80":"1,396","proteo75":"3,381","eury114":"1,736",
                 "ed194":"1,692","arc269":"*— not fit —*"}
    csuros_ll = {"dpann80":"-94,147","proteo75":"-154,159","eury114":"-274,643",
                 "ed194":"-395,601","arc269":"*— not fit —*"}
    csuros_cpf = {"dpann80":"1.15","proteo75":"1.61","eury114":"1.20",
                  "ed194":"1.19","arc269":"—"}
    csuros_max_dup = {"dpann80":"1.0000","proteo75":"1.0000","eury114":"1.0000",
                      "ed194":"1.0000","arc269":"*— not fit —*"}
    csuros_max_gain = {"dpann80":"4.9×10¹³","proteo75":"5.2×10¹⁴",
                       "eury114":"5.8×10¹²","ed194":"1.3×10¹⁴","arc269":"—"}

    # ---- Root families ----
    rows = []
    for label, ds in ARC269_ROWS:
        rows.append((label, [
            csuros_fm[ds],
            _fmt_fm(_load_summary("csuros_match", ds)),
            _fmt_fm(_load_summary("sota_ml", ds)),
            _fmt_fm(_load_summary("map_sigma1", ds)),
        ]))
    text = _rewrite_table(
        text,
        anchor_marker="Root *families*",
        header_re=r"^\|\s*Dataset\s*\|\s*Csurös ML\s*\|\s*Bounded cold ML\s*\|",
        rows=rows,
    )

    # ---- Root copies ----
    rows = []
    for label, ds in ARC269_ROWS:
        m = _load_summary("map_sigma1", ds)
        map_cpf = "—"
        if m:
            f_fm = (m.get("final") or {}).get("root_families_corr")
            f_cp = (m.get("final") or {}).get("root_copies_corr")
            if isinstance(f_fm, float) and isinstance(f_cp, float) and f_fm > 0:
                map_cpf = f"{f_cp/f_fm:.2f}"
        rows.append((label.split(' (')[0], [   # short label in copies table
            csuros_cp[ds],
            _fmt_cp(_load_summary("csuros_match", ds)),
            _fmt_cp(_load_summary("sota_ml", ds)),
            _fmt_cp(m),
            csuros_cpf[ds], map_cpf,
        ]))
    # short labels in the copies table differ — adjust for arc269 LACA
    rows = [
        (lbl.replace("arc269", "arc269 LACA") if lbl == "arc269" else lbl, cells)
        for (lbl, cells) in rows
    ]
    text = _rewrite_table(
        text,
        anchor_marker="Root *copies*",
        header_re=r"^\|\s*Dataset\s*\|\s*Csurös ML\s*\|\s*Bounded cold ML\s*\|.*Csurös cp/fm",
        rows=rows,
    )

    # ---- LL diagnostic ----
    rows = []
    for label, ds in LL_ROWS:
        rows.append((label, [
            csuros_ll[ds],
            _fmt_ll_delta(_load_summary("csuros_match", ds)),
            _fmt_ll_delta(_load_summary("sota_ml", ds)),
            _fmt_ll_delta(_load_summary("map_sigma1", ds)),
        ]))
    text = _rewrite_table(
        text,
        anchor_marker="**LL diagnostic**",
        header_re=r"^\|\s*Dataset\s*\|\s*Csurös LL\s*\|\s*Cold ML LL",
        rows=rows,
    )

    # ---- Basin diagnostics ----
    rows = []
    for label, ds in LL_ROWS:
        rows.append((label, [
            csuros_max_dup[ds],
            csuros_max_gain[ds],
            _fmt_dup_gain(_load_rates("csuros_match", ds)),
            _fmt_dup_gain(_load_rates("sota_ml", ds)),
            _fmt_dup_gain(_load_rates("map_sigma1", ds)),
        ]))
    text = _rewrite_table(
        text,
        anchor_marker="**Basin diagnostics**",
        header_re=r"^\|\s*Dataset\s*\|\s*Csurös max dup\s*\|",
        rows=rows,
    )

    if text != orig:
        README.write_text(text)
        print(f"README tables refreshed at {time.strftime('%H:%M:%S')}")
    else:
        print(f"No README table changes at {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
