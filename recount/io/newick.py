"""Standalone Newick tree reader for `recount analyze`.

Wraps the existing parser in countxml.py to handle a bare Newick file
(`.tre`, `.nwk`, `.newick`, or `-` for stdin). Returns a tuple of
``(Tree, branch_lengths)`` where the lengths array is indexed by node
index (with `+inf` at the root).

Branch lengths in the Newick are kept as-is and are used as the
**initial** GLD edge length per node; the ML fit re-estimates them
unless ``fix_root_length`` keeps the root at +inf (the Williams /
arc269 convention).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from recount.io.countxml import (
    _parse_newick,
    _assign_indices,
    _build_arrays_from_newick,
)
from recount.tree import Tree


def read_newick(path) -> Tuple[Tree, np.ndarray, list]:
    """Read a Newick file. Returns (Tree, lengths, internal_names).

    ``lengths[v]`` = branch length above node v (parent edge);
    ``lengths[root] = +inf`` by convention.
    """
    if str(path) == "-":
        import sys
        text = sys.stdin.read()
    else:
        text = Path(path).read_text()
    root = _parse_newick(text)
    _assign_indices(root)
    parent, leaf_names, lengths = _build_arrays_from_newick(root)
    tree = Tree(parent=parent, leaf_names=leaf_names)
    # Internal node names (W60, W116 in the Williams convention) — pulled
    # from the Newick by walking back through the parsed tree.
    internal_names = [""] * tree.num_nodes
    _walk_collect_names(root, internal_names)
    return tree, lengths, internal_names


def _walk_collect_names(node, names_out):
    """Populate names_out[v] = node label from the parsed Newick."""
    if node.name:
        names_out[node.index] = node.name
    for c in node.children:
        _walk_collect_names(c, names_out)
