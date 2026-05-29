"""Rooted ordered tree, indexed leaves-first with parent index > child index.

This indexing convention (from count.ds.IndexedTree in the Java sources) makes
the bottom-up recursion `for v in range(N): ...` a valid post-order traversal,
and the top-down `for v in range(N-1, -1, -1): ...` a valid pre-order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class Tree:
    """Rooted tree on N nodes.

    Indexing:
        * Nodes 0..N-1, leaves first (so the first num_leaves indices are leaves).
        * For every non-root node v, parent[v] > v.
        * parent[root] == -1.

    Parameters
    ----------
    parent
        Length-N integer array; parent[v] = index of v's parent, or -1 for the root.
    leaf_names
        Optional list of leaf names (used only for human-readable output).
    """
    parent: np.ndarray
    leaf_names: Sequence[str] = field(default_factory=list)
    children: List[np.ndarray] = field(init=False)
    is_leaf: np.ndarray = field(init=False)
    num_nodes: int = field(init=False)
    num_leaves: int = field(init=False)
    root: int = field(init=False)

    def __post_init__(self) -> None:
        self.parent = np.asarray(self.parent, dtype=np.int64)
        n = len(self.parent)
        self.num_nodes = n
        kids: List[List[int]] = [[] for _ in range(n)]
        for v in range(n):
            p = int(self.parent[v])
            if p >= 0:
                kids[p].append(v)
        self.children = [np.array(c, dtype=np.int64) for c in kids]
        self.is_leaf = np.array([len(c) == 0 for c in kids], dtype=bool)
        self.num_leaves = int(self.is_leaf.sum())
        roots = np.where(self.parent == -1)[0]
        if len(roots) != 1:
            raise ValueError(f"need exactly one root; got {len(roots)}")
        self.root = int(roots[0])
        for v in range(n):
            p = int(self.parent[v])
            if p >= 0 and p <= v:
                raise ValueError(
                    f"node {v} has parent {p} ≤ v — tree must be indexed with "
                    "parent index greater than child index (leaves first)"
                )
