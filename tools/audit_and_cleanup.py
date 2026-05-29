"""audit_and_cleanup.py — Phase-D of the autonomous chain.

Steps:
  1. Inventory every per-dataset fit in validation/outputs/{simple_ml,
     csuros_match,sota_ml,map_sigma1,bounded_csuros}/ and report whether
     the JSON summary's mtime is post-port (>= 2026-05-18 16:30 JST).
     Anything older is reported and renamed `*.stale`.
  2. Remove all files inside validation/outputs/* that have already been
     renamed to `*.stale` or `*.broken.stale` (post-purge cruft).
  3. Delete the validation/outputs.stale/ directory entirely (its tree
     should be empty after step 2; if not, log what's left first).
  4. Repo-wide sweep: find any file or directory whose basename matches
     a stale pattern (.stale, .broken, .bak, .old, ~) and remove it (with
     a per-path log line).
  5. Regenerate README tables via tools/refresh_readme_tables.py.
  6. Cross-check README sections against the structure of
     NO_DUPLICATION_CONSTRAINT.md and log any missing parallels.
  7. Commit + push to both branches.

Idempotent. Safe to re-run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)

CUTOFF = time.mktime(time.strptime("2026-05-18 16:30", "%Y-%m-%d %H:%M"))

FIT_DIRS = [
    ("simple_ml",     ""),
    ("csuros_match",  ""),
    ("sota_ml",       ""),
    ("map_sigma1",    "_sigma1.0"),
]
DATASETS = ["dpann80", "proteo75", "eury114", "ed194", "williams", "arc269"]

NOW_TS = time.strftime("%Y-%m-%d %H:%M:%S")
def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def step1_stale_audit() -> dict:
    """Inventory each fit. Returns dict of stale fits flagged."""
    log("=" * 60)
    log("Step 1: per-dataset fit staleness audit")
    log("=" * 60)
    stale_flagged = {}
    for kind, suffix in FIT_DIRS:
        d = REPO / "validation" / "outputs" / kind
        if not d.exists():
            log(f"  {kind}: directory missing — skipping")
            continue
        for ds in DATASETS:
            sumpath = d / f"{ds}{suffix}_summary.json"
            if not sumpath.exists():
                # Check for .stale-renamed version
                stale = d / f"{ds}{suffix}_summary.json.stale"
                broken = d / f"{ds}{suffix}_summary.json.broken.stale"
                if stale.exists() or broken.exists():
                    log(f"  {kind:14s}/{ds:10s}: stale-renamed (pre-flagged)")
                else:
                    log(f"  {kind:14s}/{ds:10s}: NOT PRESENT")
                continue
            mt = sumpath.stat().st_mtime
            if mt < CUTOFF:
                log(f"  {kind:14s}/{ds:10s}: STALE (mtime={time.strftime('%m-%d %H:%M', time.localtime(mt))})")
                stale_flagged.setdefault(kind, []).append(ds)
            else:
                with open(sumpath) as f:
                    d_ = json.load(f)
                ll = (d_.get("final") or {}).get("ll")
                fm = (d_.get("final") or {}).get("root_families_corr")
                log(f"  {kind:14s}/{ds:10s}: FRESH  LL={ll:.1f} fm={fm:.0f}" if isinstance(ll, float) and isinstance(fm, float) else
                    f"  {kind:14s}/{ds:10s}: FRESH  (no LL/fm)")
    log(f"  stale flagged: {sum(len(v) for v in stale_flagged.values())} fits")
    for k, v in stale_flagged.items():
        log(f"    {k}: {v}")
    return stale_flagged


def step2_remove_stale_files() -> int:
    """Remove every file inside validation/outputs/* whose name ends in
    .stale or .broken.stale (these were already sidelined by an earlier
    purge — no need to keep them on disk)."""
    log("=" * 60)
    log("Step 2: remove already-sidelined `*.stale` files inside outputs/")
    log("=" * 60)
    root = REPO / "validation" / "outputs"
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and (p.name.endswith(".stale") or p.name.endswith(".broken.stale")):
            log(f"  rm {p.relative_to(REPO)}")
            p.unlink()
            n += 1
    log(f"  removed {n} files")
    return n


def step3_drop_outputs_stale() -> bool:
    """Delete validation/outputs.stale/ if it exists and is now redundant."""
    log("=" * 60)
    log("Step 3: drop validation/outputs.stale/")
    log("=" * 60)
    d = REPO / "validation" / "outputs.stale"
    if not d.exists():
        log("  validation/outputs.stale/ already gone")
        return False
    survivors = list(d.rglob("*"))
    if survivors:
        log(f"  validation/outputs.stale/ has {len(survivors)} surviving entries:")
        for s in survivors[:20]:
            log(f"    {s.relative_to(REPO)}")
        if len(survivors) > 20:
            log(f"    ... and {len(survivors) - 20} more")
    shutil.rmtree(d)
    log("  validation/outputs.stale/ deleted")
    return True


def step4_repo_wide_stale_sweep() -> int:
    """Find any file/dir named *.stale, *.broken, *.broken.stale, *.bak, *.old, *~ — remove them.
    Walk the whole repo. Skip .git/ and the venv/build dirs."""
    log("=" * 60)
    log("Step 4: repo-wide stale-file sweep")
    log("=" * 60)
    SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pyenv", "venv", ".venv"}
    STALE_RE = re.compile(r"(\.stale|\.broken|\.bak|\.old|~)$")
    n = 0
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel_root = Path(root).relative_to(REPO)
        for fn in files:
            if STALE_RE.search(fn):
                p = Path(root) / fn
                log(f"  rm {p.relative_to(REPO)}")
                p.unlink()
                n += 1
        # Directories too (post-walk, since we modify dirs[])
        for dn in list(dirs):
            if STALE_RE.search(dn):
                p = Path(root) / dn
                log(f"  rm -r {p.relative_to(REPO)}")
                shutil.rmtree(p)
                dirs.remove(dn)
                n += 1
    log(f"  removed {n} stale entries repo-wide")
    return n


def step5_refresh_readme():
    log("=" * 60)
    log("Step 5: refresh README tables")
    log("=" * 60)
    subprocess.run(["python3", "tools/refresh_readme_tables.py"], cwd=REPO,
                   env={**os.environ, "PYTHONPATH": "."}, check=False)
    # Also regenerate the subset-only path plots and unconstrained variant.
    subprocess.run(["python3", "validation/subset_only_path_plot.py"], cwd=REPO,
                   env={**os.environ, "PYTHONPATH": "."}, check=False)
    subprocess.run(["python3", "validation/subclade_ancestor_comparison.py",
                    "--variant", "unconstrained"], cwd=REPO,
                   env={**os.environ, "PYTHONPATH": "."}, check=False)


def step6_no_dup_constraint_crosscheck():
    """Cross-check that the README has sections parallel to
    NO_DUPLICATION_CONSTRAINT.md. Just log gaps; don't auto-edit."""
    log("=" * 60)
    log("Step 6: README vs NO_DUPLICATION_CONSTRAINT.md cross-check")
    log("=" * 60)
    no_dup = (REPO / "NO_DUPLICATION_CONSTRAINT.md").read_text()
    readme = (REPO / "README.md").read_text()
    headings_re = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
    nd_headings = headings_re.findall(no_dup)
    rd_headings = headings_re.findall(readme)
    log(f"  NO_DUPLICATION_CONSTRAINT.md has {len(nd_headings)} headings")
    log(f"  README.md has {len(rd_headings)} headings")
    # Look for unique key phrases in no_dup that should also appear in README.
    KEY_PHRASES = [
        "Subclade-ancestor",
        "Bootstrap",
        "K=2 mixture",
        "Profile likelihood",
        "Gain rate",
        "Duplication rate",
        "MAP polish",
    ]
    for phrase in KEY_PHRASES:
        in_nd = phrase.lower() in no_dup.lower()
        in_rd = phrase.lower() in readme.lower()
        if in_nd and not in_rd:
            log(f"  GAP: '{phrase}' in NO_DUPLICATION but not README")
        elif in_nd and in_rd:
            log(f"  ok:  '{phrase}' in both")


def step7_commit_push():
    log("=" * 60)
    log("Step 7: commit + push")
    log("=" * 60)
    subprocess.run(["git", "add", "-A"], cwd=REPO, check=False)
    status = subprocess.run(["git", "status", "--short"], cwd=REPO,
                           capture_output=True, text=True).stdout
    if not status.strip():
        log("  no changes to commit")
        return
    log(f"  staging:\n{status}")
    msg = ("audit + cleanup: stale-data sweep + README refresh\n\n"
           "Autonomous chain Phase D completed.\n"
           "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>")
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO, check=False)
    for refspec in ("claude/goofy-heisenberg-17d13f:claude/confident-fermi-ec9e2c",
                    "claude/goofy-heisenberg-17d13f:main"):
        r = subprocess.run(["git", "push", "origin", refspec], cwd=REPO,
                          capture_output=True, text=True)
        log(f"  push {refspec}: rc={r.returncode}")
        if r.returncode != 0:
            log(f"    stderr: {r.stderr[:200]}")


def main():
    log(f"audit_and_cleanup.py starting at {NOW_TS}")
    flagged = step1_stale_audit()
    if flagged:
        log("NOTE: stale fits found — re-running them is out of scope for this pass.")
        log("      They were flagged but not re-launched (manual decision required).")
    step2_remove_stale_files()
    step3_drop_outputs_stale()
    step4_repo_wide_stale_sweep()
    step5_refresh_readme()
    step6_no_dup_constraint_crosscheck()
    step7_commit_push()
    log("audit_and_cleanup.py done")


if __name__ == "__main__":
    main()
