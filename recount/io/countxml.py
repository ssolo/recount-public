"""Loader for Count's ``.countxml`` session format (and gzipped variants).

A countxml file is a custom XML wrapper containing one or more sessions, each
holding trees, rate models, and profile tables. The schema is documented in
``count.io.CountXML`` in the Java source; the elements we care about are:

  * ``<tree>``   — CDATA holding a Newick string
  * ``<model>``  — CDATA with one line per node, columns ``t λ μ γ`` (then
                    a ``// ...`` comment); node order is leaves-first DFS,
                    matching the layout this package assumes.
  * ``<table>``  — CDATA with a tab-delimited family-by-leaf profile.

The Newick node indexing is reproduced here exactly: leaves get indices in
left-to-right Newick order (0..L-1), and internal nodes get post-order
indices (L..N-1) — so the root is N-1.
"""
from __future__ import annotations

import gzip
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from recount.rates import GLDRates
from recount.tree import Tree


# ----------------------------------------------------------------------------
# Newick parser
# ----------------------------------------------------------------------------

_NEWICK_TOKEN_RE = re.compile(r"([(),:;])")


class _NewickNode:
    __slots__ = ("name", "length", "children", "index")

    def __init__(self) -> None:
        self.name: Optional[str] = None
        self.length: float = 1.0
        self.children: List["_NewickNode"] = []
        self.index: int = -1


def _tokenize_newick(s: str) -> List[str]:
    """Split a Newick string into tokens, dropping whitespace.

    Punctuation `()`, `,`, `:`, `;` are returned as standalone tokens;
    everything else becomes a label or numeric token.  Single-quoted
    labels (e.g. ``'unclassified Candidatus Nanohaloarchaea'``) come
    through as one token even when they contain spaces.
    """
    out: List[str] = []
    buf: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "'":
            # Quoted label — read up to the next unescaped quote.  Newick
            # uses '' as an escaped single-quote inside a quoted label.
            if buf:
                out.append("".join(buf))
                buf = []
            j = i + 1
            label = []
            while j < len(s):
                if s[j] == "'":
                    if j + 1 < len(s) and s[j + 1] == "'":
                        label.append("'")
                        j += 2
                        continue
                    break
                label.append(s[j])
                j += 1
            out.append("".join(label))
            i = j + 1
            continue
        if c.isspace():
            if buf:
                out.append("".join(buf))
                buf = []
            i += 1
            continue
        if c in "(),:;":
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(c)
            i += 1
            continue
        if c == "[":
            # Newick comment; skip to closing ]
            depth = 1
            i += 1
            while i < len(s) and depth > 0:
                if s[i] == "[":
                    depth += 1
                elif s[i] == "]":
                    depth -= 1
                i += 1
            continue
        buf.append(c)
        i += 1
    if buf:
        out.append("".join(buf))
    return out


def _parse_newick(newick: str) -> _NewickNode:
    """Recursive-descent Newick parser. Returns the root node."""
    tokens = _tokenize_newick(newick)
    pos = [0]

    def peek() -> str:
        return tokens[pos[0]] if pos[0] < len(tokens) else ""

    def consume() -> str:
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def parse_subtree() -> _NewickNode:
        node = _NewickNode()
        if peek() == "(":
            consume()
            while True:
                node.children.append(parse_subtree())
                if peek() == ",":
                    consume()
                    continue
                if peek() == ")":
                    consume()
                    break
                raise ValueError(f"unexpected token in Newick: {peek()!r}")
        # optional name
        if peek() not in (",", ")", ":", ";", ""):
            node.name = consume()
        # optional length
        if peek() == ":":
            consume()
            node.length = float(consume())
        return node

    root = parse_subtree()
    # gobble trailing semicolon if present
    if peek() == ";":
        consume()
    return root


def _assign_indices(root: _NewickNode) -> Tuple[int, int]:
    """Assign indices to a Newick tree:

        * Leaves: 0 .. L-1 in left-to-right traversal order.
        * Internal nodes: L .. N-1 in post-order.

    Returns ``(num_leaves, num_internal)``.
    """
    # Pass 1: number leaves in left-to-right order.
    leaf_counter = [0]

    def number_leaves(n: _NewickNode) -> None:
        if not n.children:
            n.index = leaf_counter[0]
            leaf_counter[0] += 1
            return
        for c in n.children:
            number_leaves(c)

    number_leaves(root)
    num_leaves = leaf_counter[0]

    # Pass 2: number internal nodes in post-order, starting from num_leaves.
    counter = [num_leaves]

    def number_internal(n: _NewickNode) -> None:
        if not n.children:
            return
        for c in n.children:
            number_internal(c)
        n.index = counter[0]
        counter[0] += 1

    number_internal(root)
    return num_leaves, counter[0] - num_leaves


def _build_arrays_from_newick(root: _NewickNode) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Walk the indexed Newick tree and produce (parent_array, leaf_names, edge_lengths)."""
    num_leaves, num_internal = _assign_indices(root)
    n = num_leaves + num_internal
    parent = np.full(n, -1, dtype=np.int64)
    leaf_names: List[str] = [""] * num_leaves
    edge_length = np.zeros(n)

    def walk(node: _NewickNode, parent_idx: int) -> None:
        v = node.index
        if parent_idx >= 0:
            parent[v] = parent_idx
            edge_length[v] = node.length
        else:
            edge_length[v] = np.inf  # root edge length is conventionally +inf
        if not node.children:
            leaf_names[v] = node.name or f"leaf{v}"
        for c in node.children:
            walk(c, v)

    walk(root, -1)
    return parent, leaf_names, edge_length


# ----------------------------------------------------------------------------
# rate-model and profile-table parsers
# ----------------------------------------------------------------------------


_FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|Infinity|NaN")

# Each rate row's comment includes the node identifier in either of the forms
# "N[T<idx>/..." for leaves or "N[U<idx>/..." for internal nodes. The index in
# that ID is the TRUE node index (Count uses separate counters for leaves and
# internals, so the row order in the file does NOT equal node-index order).
_NODE_ID_RE = re.compile(r"N\[(T|U)(\d+)(?:\s|/)")


def _parse_model_cdata(text: str, num_nodes: int) -> np.ndarray:
    """Parse the model CDATA: one row per node, columns ``t λ μ γ_scaled``.

    Each row's 4th column is **scaled**: Count's ``RateVariationParser.printRates``
    writes ``γ·μ`` when ``λ==0`` (Poisson) and ``κ·λ`` when ``λ>0`` (Pólya).
    On load we undo this scaling. Additionally, Count's
    ``TreeWithRates.getGainParameter`` returns ``γ·p_raw`` in the Poisson
    case (i.e. the Poisson intensity actually used in the recursion is the
    stored γ times the edge's loss probability), so we apply that extra
    factor here. The Pólya case is left as the bare κ.

    Rows may have trailing ``// N[T<idx>/...]`` comments holding the node
    identifier; the root row is prefixed with ``#ROOTRATES``. Trailing
    meta-lines (``|variation``, ``|root``, etc.) are skipped. Returns an
    array of shape (num_nodes, 4) indexed by **node index** (so row v has
    the rates for tree node v).
    """
    out = np.full((num_nodes, 4), np.nan, dtype=np.float64)
    seen = np.zeros(num_nodes, dtype=bool)

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("|"):
            continue

        # Pull the node index out of the comment if present.
        m = _NODE_ID_RE.search(raw_line)

        body = stripped
        if body.startswith("#ROOTRATES"):
            body = body[len("#ROOTRATES"):]
        if "//" in body:
            body = body.split("//", 1)[0]
        parts = body.split()
        if len(parts) < 4:
            continue

        def to_float(s: str) -> float:
            if s == "Infinity":
                return float("inf")
            if s == "-Infinity":
                return float("-inf")
            return float(s)
        try:
            t, lam, mu, gamma = (
                to_float(parts[0]), to_float(parts[1]),
                to_float(parts[2]), to_float(parts[3]),
            )
        except ValueError:
            continue

        if m is None:
            raise ValueError(
                "rate row is missing its node id comment (expected 'N[T<idx>/' or "
                f"'N[U<idx>/'); raw line: {raw_line!r}"
            )
        idx = int(m.group(2))
        if idx < 0 or idx >= num_nodes:
            raise ValueError(f"node index {idx} out of range (num_nodes={num_nodes})")
        if seen[idx]:
            raise ValueError(f"duplicate rate row for node {idx}")

        # Undo Count's legacy scaling of the gain column. printRates writes
        # grate·μ when λ==0 and grate·λ otherwise; here we divide back out.
        if lam == 0.0 and mu != 0.0:
            gamma = gamma / mu
            # For Poisson edges Count's getGainParameter additionally returns
            # γ·p_raw (the Poisson intensity used in the recursion).  Apply
            # that factor now using rate_to_p so the stored ``gain`` is the
            # intensity ``r`` directly.  See Count.model.TreeWithRates.
            from recount.rates import rate_to_p as _rate_to_p_internal
            p_raw, _ = _rate_to_p_internal(mu, lam, t)
            gamma = gamma * p_raw
        elif lam != 0.0:
            gamma = gamma / lam

        out[idx] = (t, lam, mu, gamma)
        seen[idx] = True

    missing = np.where(~seen)[0]
    if missing.size:
        raise ValueError(f"missing rate rows for nodes: {missing.tolist()}")
    return out


def _parse_table_cdata(
    text: str, leaf_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Parse the table CDATA: tab-delimited rows ``Family <leaf_counts...>``.

    Returns ``(profiles, family_names)`` where ``profiles`` is a
    (num_families, num_leaves) int64 array of counts, with columns reordered
    to match ``leaf_names`` (the tree's leaf order).
    """
    # Drop blank lines and trailing meta-lines (Count writes e.g. "#NFAM\t5378\tfamilies").
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        raise ValueError("<table> contained no rows")
    header = lines[0].split("\t")
    # header[0] is "Family"; the remaining columns are leaf taxa.
    file_taxa = header[1:]

    # Map each tree-leaf to its column index in the file. Missing taxa are
    # treated as "0 copies" for every family.
    col_for_taxon: Dict[str, Optional[int]] = {
        name: (file_taxa.index(name) if name in file_taxa else None)
        for name in leaf_names
    }

    family_names: List[str] = []
    rows: List[np.ndarray] = []
    for line in lines[1:]:
        cols = line.split("\t")
        family_names.append(cols[0])
        profile_row = np.zeros(len(leaf_names), dtype=np.int64)
        for i, name in enumerate(leaf_names):
            j = col_for_taxon[name]
            if j is None:
                continue
            val = cols[1 + j].strip()
            if not val:
                profile_row[i] = 0
            else:
                profile_row[i] = int(val)
        rows.append(profile_row)

    return np.asarray(rows, dtype=np.int64), family_names


# ----------------------------------------------------------------------------
# top-level loader
# ----------------------------------------------------------------------------


@dataclass
class CountSession:
    """One session extracted from a countxml file."""
    tree: Tree
    rates: Optional[GLDRates]  # None if the session has no <model>
    tables: Dict[str, "ProfileTable"]


@dataclass
class ProfileTable:
    """A profile table with family names and the count matrix."""
    name: str
    profiles: np.ndarray  # shape (num_families, num_leaves)
    family_names: List[str]


def _open_maybe_gz(path):
    """Open a path as text, transparently decompressing .gz."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="ascii", errors="replace")
    return open(path, "rt", encoding="ascii", errors="replace")


def _load_xml_bytes(path) -> bytes:
    """Read raw bytes from path (gz or plain).

    Count's writer emits files that declare ``encoding="us-ascii"`` even
    though the gene annotations contain non-ASCII bytes (UTF-8 replacement
    characters from upstream IDs).  We strip the encoding declaration so
    Python's parser doesn't reject the file.
    """
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            data = f.read()
    else:
        with open(path, "rb") as f:
            data = f.read()
    # Drop the XML declaration line entirely so ET infers UTF-8 (its default).
    # Replace any subsequent stray non-UTF-8 bytes with U+FFFD via the
    # decode/encode round-trip.
    if data.startswith(b"<?xml"):
        eol = data.find(b"?>")
        if eol >= 0:
            data = data[eol + 2:]
    return data.decode("utf-8", errors="replace").encode("utf-8")


def load_countxml(path) -> Dict[str, CountSession]:
    """Load every session in a countxml(.gz) file.

    Returns a dict mapping session-id → CountSession. Each session has:
      * ``tree``  — recount.Tree
      * ``rates`` — recount.GLDRates (per the first <model> in the session)
      * ``tables`` — dict of table-name → ProfileTable for every <table> in
                    the session

    For a typical Count session file with one tree, one rate model and one
    or more tables, ``next(iter(load_countxml(path).values()))`` gives the
    object you want.
    """
    data = _load_xml_bytes(path)
    root_el = ET.fromstring(data)

    sessions: Dict[str, CountSession] = {}
    for sess_el in root_el.iter("session"):
        sess_id = sess_el.get("id", "")
        # Each session has zero or one tree (we take the first), one model
        # under that tree, and zero or more tables.
        tree_el = sess_el.find("tree")
        if tree_el is None or tree_el.text is None:
            continue
        nwk_root = _parse_newick(tree_el.text)
        parent, leaf_names, edge_length = _build_arrays_from_newick(nwk_root)
        tree = Tree(parent=parent, leaf_names=leaf_names)

        # Rate model — optional. Count sessions can carry a tree + tables but
        # no model (no rates have been fit yet).
        model_el = tree_el.find("model")
        rates: Optional[GLDRates] = None
        if model_el is not None and model_el.text is not None:
            rate_arr = _parse_model_cdata(model_el.text, tree.num_nodes)
            # The root row sometimes has a numeric length (e.g. 1.0); the
            # convention everywhere else in recount is +inf for the root.
            rate_arr[tree.root, 0] = np.inf
            rates = GLDRates(
                tree=tree,
                length=rate_arr[:, 0],
                dup=rate_arr[:, 1],
                loss=rate_arr[:, 2],
                gain=rate_arr[:, 3],
            )

        tables: Dict[str, ProfileTable] = {}
        for tbl_el in sess_el.iter("table"):
            if tbl_el.text is None:
                continue
            tbl_name = tbl_el.get("name") or tbl_el.get("id") or f"table{len(tables)}"
            profs, fams = _parse_table_cdata(tbl_el.text, leaf_names)
            tables[tbl_name] = ProfileTable(name=tbl_name, profiles=profs, family_names=fams)

        sessions[sess_id] = CountSession(tree=tree, rates=rates, tables=tables)

    if not sessions:
        raise ValueError(f"no sessions found in {path}")
    return sessions
