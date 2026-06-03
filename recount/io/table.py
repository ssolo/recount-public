"""Profile table reader: TSV with `family_name leaf1 leaf2 ... leafN`.

Used by ``recount analyze``. File format:

    Family<TAB>leaf_1<TAB>leaf_2<TAB>...<TAB>leaf_N
    fam001<TAB>0<TAB>1<TAB>0<TAB>...
    fam002<TAB>2<TAB>0<TAB>1<TAB>...

The first column header may be anything (typically ``Family`` or
``family``); the remaining headers must match leaf names in the tree.

Returns ``(family_names, profiles)`` where ``profiles`` is a
``(num_families, num_tree_leaves)`` int32 array with columns reordered
to match the tree's leaf order (missing taxa default to 0).
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_profile_table(
    path, tree_leaf_names: List[str],
) -> Tuple[List[str], np.ndarray]:
    """Read a TSV count table. ``path`` may be ``.tsv``, ``.txt``,
    ``.csv``, or ``.gz`` of any of those (heuristic: gzipped if filename
    ends with ``.gz``)."""
    import csv
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", newline="") as fh:
        # Auto-detect delimiter from a sniff of the first non-comment row
        sample = ""
        peek_lines = []
        for ln in fh:
            if ln.strip() and not ln.startswith("#"):
                peek_lines.append(ln)
                sample += ln
                if len(peek_lines) >= 2:
                    break
        delim = "\t" if "\t" in sample.split("\n")[0] else ","
        # Re-open and parse via csv (handles quoted commas in annotation columns)
        fh.seek(0)
        reader = csv.reader(
            (ln for ln in fh if ln.strip() and not ln.startswith("#")),
            delimiter=delim,
        )
        header_cols = next(reader)
        file_taxa = header_cols[1:]

        # Map tree leaf → file column index (or None if missing)
        col_for_leaf = {}
        for name in tree_leaf_names:
            col_for_leaf[name] = file_taxa.index(name) if name in file_taxa else None

        fams: List[str] = []
        n_leaves = len(tree_leaf_names)
        rows = []
        for cols in reader:
            fams.append(cols[0])
            row = np.zeros(n_leaves, dtype=np.int32)
            for i, name in enumerate(tree_leaf_names):
                j = col_for_leaf[name]
                if j is None:
                    continue
                val = cols[1 + j].strip()
                if val == "" or val == "?":
                    row[i] = 0  # absent / ambiguous → 0
                else:
                    row[i] = int(val)
            rows.append(row)
    profiles = np.stack(rows, axis=0)
    return fams, profiles
