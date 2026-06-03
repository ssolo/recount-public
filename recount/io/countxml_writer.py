"""CountXML writer — emits a session XML compatible with Csurös' Java Count.

Format produced (mirrors the structure of files read by
``recount.io.countxml.load_countxml``):

    <?xml version="1.0" encoding="utf-8" ?>
    <CountML version="recount-native" date="...">
    <session id="..." type="recount.analyze">
    <tree id="..." name="..." parent="...">
    <![CDATA[ <newick>; ]]>
    <model id="..." name="..." parent="...">
    <![CDATA[
    <one row per node: length<TAB>loss<TAB>1.0<TAB>gain<TAB>// params<TAB>p<TAB>q<TAB>kappa<TAB>// N[Ti/leafname ...]>
    |variation   common  1   1   linear  // .gainpar loss
    |variation   LogisticShift   1.0  1.0  1.0  // LogisticShift#1[0.0,0.0; p=1.0/logp=0.0]
    |root        NegativeBinomial   <κ_root>   <q_root>
    ]]>
    </model>
    </tree>
    <table id="..." name="..." parent="..." isbinary="false">
    <![CDATA[
    Family<TAB>leaf_1<TAB>leaf_2<TAB>...
    <one row per family>
    ]]>
    </table>
    </session>
    </CountML>

The fitted rates are written in the `length loss dup gain` order with
Count-style trailing comments. This file round-trips through
``load_countxml`` for the tree+model+table, though the human-readable
comments after `// params` and `// N[...]` are not strictly required by
the reader.
"""
from __future__ import annotations

import datetime
import gzip
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from recount.rates import GLDRates
from recount.tree import Tree


def write_countxml(
    out_path,
    tree: Tree,
    rates: GLDRates,
    family_names: List[str],
    profiles: np.ndarray,
    internal_names: Optional[List[str]] = None,
    session_id: str = "recount-analyze",
    table_name: str = "table.txt",
) -> None:
    """Write a Count-compatible XML session.

    ``profiles`` shape (F, num_leaves), dtype int.
    ``internal_names`` optional list of names per node (leaves + internals);
        if omitted, internal nodes are named ``W<idx>`` Count-style.
    ``out_path`` may end with ``.gz`` for gzip output.
    """
    p = Path(out_path)
    leaf_names = list(tree.leaf_names) if tree.leaf_names else [f"leaf{i}" for i in range(tree.num_leaves)]
    int_names: List[str] = [""] * tree.num_nodes
    if internal_names:
        for i, nm in enumerate(internal_names):
            int_names[i] = nm
    # Fill in any blanks
    for v in range(tree.num_nodes):
        if int_names[v]:
            continue
        if tree.is_leaf[v]:
            int_names[v] = leaf_names[v] if v < len(leaf_names) else f"leaf{v}"
        else:
            int_names[v] = f"W{v}"

    newick = _write_newick(tree, int_names, rates.length)

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8" ?>')
    lines.append(f'<CountML version="recount-native" date="{datetime.datetime.now().isoformat(timespec="seconds")}">')
    lines.append(f'<session id="{session_id}" type="recount.analyze">')
    lines.append(f'<tree id="{session_id}.T0" name="tree.nwk" parent="{session_id}">')
    lines.append("<![CDATA[")
    lines.append(newick)
    lines.append("]]>")
    lines.append(f'<model id="{session_id}.T0.R0" name="rates.txt" parent="{session_id}.T0">')
    lines.append("<![CDATA[")
    # Rate rows: length  dup  loss  gain(scaled) // params loss dup gain // N[...]
    # Column order + the gain scaling must match load_countxml._parse_model_cdata.
    root = tree.root
    for v in range(tree.num_nodes):
        length = rates.length[v]
        loss   = rates.loss[v]
        dup    = rates.dup[v]
        gain   = rates.gain[v]
        # Length: keep +inf at root
        length_s = "Infinity" if math.isinf(length) else f"{length:.16g}"
        kind_tag = "T" if tree.is_leaf[v] else "U"
        # Build a Count-style trailing comment so the file is human-readable
        comment = f"// N[{kind_tag}{v}/{int_names[v]} len {length_s} prnt {(int_names[int(tree.parent[v])] + '/' + str(tree.parent[v])) if tree.parent[v] >= 0 else '-'}]"
        # The 4 numeric columns are  length  dup  loss  gain_scaled  — the
        # exact order load_countxml._parse_model_cdata reads back (t, λ, μ,
        # γ).  The gain column carries Count's printRates scaling that the
        # loader undoes: κ·λ for Pólya (dup>0), γ·μ/p_raw for Poisson.  Write
        # it scaled so the recount→CountXML→recount round-trip is exact.
        if dup > 0.0:
            gain_col = gain * dup
        elif loss > 0.0:
            from recount.rates import rate_to_p as _rate_to_p
            p_raw, _ = _rate_to_p(loss, dup, length)
            gain_col = gain * loss / p_raw if p_raw > 0.0 else gain
        else:
            gain_col = gain
        lines.append(f"{length_s}\t{dup:.16g}\t{loss:.16g}\t{gain_col:.16g}"
                     f"\t// params\t{loss:.16g}\t{dup:.16g}\t{gain:.16g}\t{comment}")
    # Variation footer (single category, no shift — base GLD)
    lines.append("|variation\tcommon\t1\t1\tlinear\t// .gainpar loss")
    lines.append("|variation\tLogisticShift\t1.0\t1.0\t1.0\t// LogisticShift#1[0.0,0.0; p=1.0/logp=0.0]")
    # Root parameters
    if rates.dup[root] > 0:
        lines.append(f"|root\tNegativeBinomial\t{rates.gain[root]:.16g}\t{rates.dup[root]:.16g}")
    else:
        lines.append(f"|root\tPoisson\t{rates.gain[root]:.16g}")
    lines.append("]]>")
    lines.append("</model>")
    lines.append("</tree>")
    lines.append(f'<table id="{session_id}.D1" name="{table_name}" parent="{session_id}" isbinary="false">')
    lines.append("<![CDATA[")
    lines.append("Family\t" + "\t".join(leaf_names))
    for f_idx, fam in enumerate(family_names):
        row = profiles[f_idx]
        lines.append(fam + "\t" + "\t".join(str(int(c)) for c in row))
    lines.append("]]>")
    lines.append("</table>")
    lines.append("</session>")
    lines.append("</CountML>")

    text = "\n".join(lines) + "\n"
    if p.suffix == ".gz":
        with gzip.open(p, "wt") as fh:
            fh.write(text)
    else:
        with open(p, "w") as fh:
            fh.write(text)


def _write_newick(tree: Tree, names: List[str], lengths: np.ndarray) -> str:
    """Emit Newick from a Tree + names + branch lengths."""
    root = tree.root
    # Build child lists
    parent = np.asarray(tree.parent)
    children = [[] for _ in range(tree.num_nodes)]
    for v in range(tree.num_nodes):
        p = int(parent[v])
        if p >= 0:
            children[p].append(v)

    def emit(v):
        if tree.is_leaf[v]:
            return f"{names[v]}:{_fmt_len(lengths[v])}"
        inner = ",".join(emit(c) for c in children[v])
        if v == root:
            return f"({inner}){names[v]}"
        return f"({inner}){names[v]}:{_fmt_len(lengths[v])}"

    return emit(root) + ";"


def _fmt_len(x: float) -> str:
    if math.isinf(x):
        return "1.0"  # Newick can't carry +inf; use 1.0 as a placeholder
    return f"{x:.6g}"
