"""Worked example for README section 1 — copy numbers cannot separate
transfer from inheritance.

Five genomes, species tree (((A,B),C),(D,E)).  One gene family is
present in A (1 copy), B (2), C (1), D (1), and absent in E.  The
unrooted gene tree of the five copies, ((a,d),(b1,b2),c), pairs gene a
(species A) with gene d (species D): gene d is a transfer from the A
lineage.  Reconciliation, reading that gene tree, recovers the true
history — the family originates on the branch leading to the ABC
ancestor, one copy is transferred from the A lineage to D, and one
copy duplicates on the terminal branch to B; the root (the ABCDE
ancestor), the DE ancestor and E are empty of the family.

GLD reads only the copy-number profile [A=1,B=2,C=1,D=1,E=0].  This
script computes GLD's posterior P(>= 1 copy) at every node along the
two root-ward trajectories

    ABCDE(root) -> DE  -> D
    ABCDE(root) -> ABC -> AB -> A

and produces two trajectory plots — one per fixed global loss rate,
one curve per global gain rate:

  validation/outputs/transfer_traj_loss1.png  + .pdf
  validation/outputs/transfer_traj_loss2.png  + .pdf

The numeric table is also written to transfer_identifiability.txt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from recount import Tree
from recount.native_backend import per_family_posteriors_native
from validation import _pubstyle as ps   # sets the Agg backend on import

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUT = Path("validation/outputs")

# species tree (((A,B),C),(D,E)): 0=A 1=B 2=C 3=D 4=E 5=AB 6=ABC 7=DE 8=root
TREE = Tree(parent=np.array([5, 5, 6, 7, 7, 6, 8, 8, -1]),
            leaf_names=["A", "B", "C", "D", "E"])
NN = TREE.num_nodes
ROOT = TREE.root
PROFILE = np.array([[1, 2, 1, 1, 0]], dtype=np.int32)   # A=1 B=2 C=1 D=1 E=0

NODE_LABEL = {0: "A", 3: "D", 5: "AB", 6: "ABC", 7: "DE", 8: "ABCDE\n(root)"}
TRAJ_D = [8, 7, 3]          # root -> DE -> D       (root-to-leaf, x left→right)
TRAJ_A = [8, 6, 5, 0]       # root -> ABC -> AB -> A
TRAJ_TITLE = {"D": r"D $\rightarrow$ DE $\rightarrow$ root trajectory",
              "A": r"A $\rightarrow$ AB $\rightarrow$ ABC $\rightarrow$ root trajectory"}
# true history: present at A, AB, ABC, D; empty elsewhere
TRUE_PRESENT = {0: 1, 3: 1, 5: 1, 6: 1, 7: 0, 8: 0}

DUP = 0.20                  # global duplication rate, held fixed
GAINS = [0.03, 0.1, 0.3, 1.0, 3.0]
LOSSES = [1.0, 2.0]
# one Okabe-Ito colour per global gain rate (shared publication palette)
GAIN_COLORS = [ps.BLUE, ps.SKYBLUE, ps.GREEN, ps.ORANGE, ps.VERMILION]


def _rates(gain: float, loss: float, dup: float):
    g = np.full(NN, gain, dtype=float)
    l = np.full(NN, loss, dtype=float)
    d = np.full(NN, dup, dtype=float)
    t = np.full(NN, 1.0, dtype=float)
    t[ROOT] = np.inf
    return g, l, d, t


def p_present(gain: float, loss: float, dup: float) -> np.ndarray:
    """Posterior P(>= 1 copy) at every node."""
    _, present = per_family_posteriors_native(
        TREE, *_rates(gain, loss, dup), PROFILE)
    return present[0]


# ===================== trajectory plots (loss 1, loss 2) ===============
def trajectory_plot(loss: float, stem: Path):
    fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_2COL, 3.3))
    pres = {g: p_present(g, loss, DUP) for g in GAINS}

    for ax, traj, key, letter in [(axes[0], TRAJ_D, "D", "A"),
                                  (axes[1], TRAJ_A, "A", "B")]:
        x = np.arange(len(traj))
        # true history — black dotted reference
        ax.plot(x, [TRUE_PRESENT[v] for v in traj], color=ps.C_DATA,
                marker="D", markersize=5.5, lw=1.3, linestyle=":",
                markerfacecolor=ps.C_DATA, zorder=4)
        # GLD posterior, one curve per global gain rate
        for gi, g in enumerate(GAINS):
            ax.plot(x, [pres[g][v] for v in traj], color=GAIN_COLORS[gi],
                    marker="o", markersize=5, lw=1.7, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.axhline(0.0, color=ps.LIGHTGREY, lw=0.7, zorder=1)
        ax.axhline(1.0, color=ps.LIGHTGREY, lw=0.7, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([NODE_LABEL[v] for v in traj])
        ax.set_xlim(-0.28, len(traj) - 0.72)
        ax.set_ylim(-0.07, 1.12)
        ax.set_xlabel(r"node   (root $\rightarrow$ leaf)")
        ax.set_ylabel("GLD posterior  P(family present)")
        ax.set_title(TRAJ_TITLE[key], pad=6)
        ps.grid(ax, axis="y")
        ps.panel_label(ax, letter, dx=-0.165, dy=1.10)

    handles = [mlines.Line2D([], [], color=GAIN_COLORS[gi], marker="o",
                             lw=1.7, markersize=5,
                             label=f"gain rate = {g:g}")
               for gi, g in enumerate(GAINS)]
    handles.append(mlines.Line2D([], [], color=ps.C_DATA, marker="D", lw=1.3,
                                 linestyle=":", markersize=5.5,
                                 label="true history"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.015))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.275,
                        top=0.875, wspace=0.30)
    ps.save_fig(fig, stem)
    plt.close(fig)
    print(f"  wrote {stem.name}.png + .pdf")


# ===================== numeric table (text record) ====================
def write_table():
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("Transfer-identifiability worked example — GLD P(>= 1 copy) per node")
    emit("species tree (((A,B),C),(D,E));  family profile A=1 B=2 C=1 D=1 E=0")
    emit(f"global rates, unit branch lengths, duplication rate fixed at {DUP}.")
    emit("True history: present at A, AB, ABC, D;  the root,")
    emit("the DE ancestor and E are empty.")
    for loss in LOSSES:
        pres = {g: p_present(g, loss, DUP) for g in GAINS}
        emit("")
        emit(f"=== loss rate = {loss:g}   (columns = global gain rate) ===")
        emit("  " + f"{'gain rate →':<16}" + "".join(f"{g:>8g}" for g in GAINS))
        for traj, name in [(TRAJ_A, "A -> AB -> ABC -> root"),
                           (TRAJ_D, "D -> DE -> root")]:
            emit(f"  trajectory  {name} :")
            for v in traj:
                lbl = NODE_LABEL[v].replace("\n", " ")
                emit(f"    {lbl:<14}"
                     + "".join(f"{pres[g][v]:>8.3f}" for g in GAINS))
    (OUT / "transfer_identifiability.txt").write_text("\n".join(lines) + "\n")
    print(f"\n  wrote {OUT / 'transfer_identifiability.txt'}")


def main():
    write_table()
    print()
    ps.apply()                          # install shared rcParams — once
    trajectory_plot(1.0, OUT / "transfer_traj_loss1")
    trajectory_plot(2.0, OUT / "transfer_traj_loss2")
    print("\nDONE.")


if __name__ == "__main__":
    main()
