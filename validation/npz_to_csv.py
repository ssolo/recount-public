"""Dump every array in an NPZ file to a CSV (one CSV per array).

Output naming: <stem>__<arrayname>.csv next to the input (or in --out-dir).
For 2-D arrays of shape (R, C), the CSV has R rows × C columns. For 1-D
arrays, one value per row. Scalars are written as a single-row CSV with
the value.

Defaults are tuned for the per-family presence tables in
validation/outputs/*/<ds>_per_family_FxN.npz, but the script accepts any
NPZ.

Usage:
    PYTHONPATH=. python3 validation/npz_to_csv.py \\
        validation/outputs_no_dup_cap/map_sigma1/dpann80_per_family_FxN.npz

    # all five datasets, with binary presence (threshold 0.5) only:
    for f in validation/outputs_no_dup_cap/map_sigma1/*_per_family_FxN.npz; do
        PYTHONPATH=. python3 validation/npz_to_csv.py \\
            "$f" --arrays present --threshold 0.5 --out-dir /tmp/per_family_csv
    done

NOTE: these CSVs are LOCAL ONLY by convention — they're typically much
bigger than the source NPZ (arc269 present.csv is ~190 MB binary, ~700 MB
as float). Don't commit them.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _dump_array(name: str, arr: np.ndarray, out_path: Path,
                threshold: float | None, fmt: str) -> None:
    """Write one named array as a CSV."""
    if threshold is not None and arr.dtype.kind in "fc":
        # Threshold float arrays to 0/1 int (intended for the `present` matrix)
        arr = (arr >= threshold).astype(np.int8)
        print(f"  {name}: thresholded at {threshold} -> 0/1 int8")
    if arr.ndim == 0:
        # Scalar
        with open(out_path, "w", newline="") as fh:
            fh.write(f"value\n{arr.item()}\n")
        print(f"  {name}: scalar -> {out_path}")
        return
    if arr.ndim == 1:
        with open(out_path, "w", newline="") as fh:
            fh.write(f"value\n")
            for v in arr:
                fh.write(f"{v}\n" if arr.dtype.kind != "f" else f"{v:{fmt}}\n")
        print(f"  {name}: 1-D, length {arr.shape[0]} -> {out_path}")
        return
    if arr.ndim == 2:
        R, C = arr.shape
        header = ["row"] + [f"col_{c}" for c in range(C)]
        is_float = arr.dtype.kind == "f"
        with open(out_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            if is_float:
                for r in range(R):
                    row = [r] + [f"{x:{fmt}}" for x in arr[r]]
                    w.writerow(row)
            else:
                for r in range(R):
                    w.writerow([r, *arr[r].tolist()])
        print(f"  {name}: {R}×{C} -> {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
        return
    # ndim >= 3 — flatten the leading dims
    print(f"  {name}: ndim={arr.ndim} not supported; skipping")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", type=Path, help="path to .npz file")
    ap.add_argument("--arrays", nargs="*", default=None,
                    help="only dump these named arrays (default: all)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="if set, threshold float arrays to 0/1 int8 at this cut-off")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write CSVs here (default: same dir as the .npz)")
    ap.add_argument("--fmt", default=".6g",
                    help="format spec for float values (default '.6g')")
    args = ap.parse_args(argv)

    if not args.npz.exists():
        print(f"ERROR: {args.npz} not found", file=sys.stderr)
        return 1

    out_dir = args.out_dir if args.out_dir is not None else args.npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.npz.stem

    print(f"Loading {args.npz}")
    d = np.load(args.npz, allow_pickle=False)
    names = d.files if args.arrays is None else args.arrays
    print(f"  arrays in file: {d.files}")
    print(f"  will dump: {list(names)}")

    for name in names:
        if name not in d.files:
            print(f"WARNING: {name} not in NPZ; skipping")
            continue
        arr = np.asarray(d[name])
        out_path = out_dir / f"{stem}__{name}.csv"
        _dump_array(name, arr, out_path, args.threshold, args.fmt)
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
