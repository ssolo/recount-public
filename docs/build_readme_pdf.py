#!/usr/bin/env python3
"""Build a LaTeX / PDF rendering of the project README.

README.md (repo root) is the single source of truth.  This script
regenerates docs/README.tex and docs/README.pdf from it, so the LaTeX
version never has to be hand-maintained: after editing README.md, run

    python3 docs/build_readme_pdf.py

What it does
------------
1. Pre-processes the Markdown so pandoc can render it cleanly:
     * every `<img src="...png">` HTML tag is rewritten to a Markdown
       image that points at the PDF version of the plot when one
       exists (`validation/outputs/foo.png` -> `foo.pdf`), so the
       LaTeX build embeds vector figures;
     * `<details>` / `<summary>` blocks are flattened (a PDF has no
       collapsible sections — the content becomes always-visible);
     * image paths are made relative to docs/ (where README.tex lives).
2. Runs `pandoc` (Markdown -> LaTeX) to write docs/README.tex.
3. Runs `xelatex` twice to write docs/README.pdf (xelatex is used so
   the Unicode in the README — Greek letters, arrows, ≥, ≈, … —
   renders without escaping).

Requirements: pandoc and xelatex on PATH.  If xelatex is missing the
script still writes README.tex and reports the PDF step as skipped.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repo root
DOCS = ROOT / "docs"
README = ROOT / "README.md"
TEX = DOCS / "README.tex"


def preprocess(md: str) -> str:
    """Rewrite the Markdown into a pandoc-friendly form."""
    # --- <details> / <summary> : flatten (no collapsibles in a PDF) ----
    md = re.sub(r"</?details>[ \t]*\n?", "", md)
    md = re.sub(r"<summary>(.*?)</summary>",
                lambda m: "**" + m.group(1).strip() + "**\n", md, flags=re.S)

    # --- <img ...> HTML tags -> Markdown images, PDF figure preferred --
    def _img(m: re.Match) -> str:
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(0)))
        src = attrs.get("src", "")
        alt = attrs.get("alt", "").replace("[", "(").replace("]", ")")
        p = ROOT / src
        pdf = p.with_suffix(".pdf")
        use = pdf if pdf.exists() else p
        rel = "../" + str(use.relative_to(ROOT))           # README.tex is in docs/
        return f"![{alt}]({rel}){{width=92%}}"

    md = re.sub(r"<img\b[^>]*?/?>", _img, md)

    # the lone Markdown image lives under docs/figures/ -> docs-relative
    md = md.replace("](docs/figures/", "](figures/")
    return md


def fix_longtables(tex: str) -> str:
    """Convert pandoc's longtable environments to centred, width-limited
    tabulars, so the README's wide multi-column tables never overflow the
    text block.  `adjustbox` with `max width` shrinks an over-wide table
    and leaves one that already fits at its natural size.  Every README
    table is short (well under a page), so dropping longtable's
    page-breaking costs nothing."""
    def conv(m: "re.Match[str]") -> str:
        hm = re.search(r"\\toprule\\noalign\{\}\n(.*?)\\midrule\\noalign\{\}",
                        m.group("pre"), re.S)
        header = hm.group(1) if hm else ""
        return ("\\begin{center}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n"
                "\\begin{tabular}{" + m.group("cs") + "}\n"
                "\\toprule\\noalign{}\n" + header + "\\midrule\\noalign{}\n"
                + m.group("body") + "\\bottomrule\\noalign{}\n"
                "\\end{tabular}\n\\end{adjustbox}\n\\end{center}")
    return re.sub(
        r"\\begin\{longtable\}\[\]\{(?P<cs>.*?)\}\n"
        r"(?P<pre>.*?)\\endlastfoot\n"
        r"(?P<body>.*?)\\end\{longtable\}",
        conv, tex, flags=re.S)


def main() -> int:
    if shutil.which("pandoc") is None:
        sys.exit("error: pandoc not found on PATH")

    md = preprocess(README.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(md)
        tmp_md = fh.name

    subprocess.run(
        # commonmark_x: CommonMark with the GitHub-flavoured extensions
        # (pipe tables, …) plus `attributes`, so `{width=...}` on an
        # image is honoured. (`gfm` alone drops image attributes.)
        ["pandoc", tmp_md, "--standalone", "--from=commonmark_x",
         "--to=latex",
         "--metadata", "title=recount — reproduction and analysis of the "
         "Csűrös 2026 GLD framework",
         "--variable", "geometry:margin=0.8in",
         "--variable", "colorlinks=true",
         "--variable", "fontsize=10pt",
         # adjustbox shrinks the README's wide multi-column tables to
         # fit the text block (see fix_longtables below).
         "--variable", "header-includes=\\usepackage{adjustbox}",
         "--toc", "--toc-depth=2",
         "-o", str(TEX)],
        check=True)
    TEX.write_text(fix_longtables(TEX.read_text(encoding="utf-8")),
                   encoding="utf-8")
    print(f"  wrote {TEX.relative_to(ROOT)}")

    if shutil.which("xelatex") is None:
        print("  xelatex not found — README.pdf NOT built (README.tex is ready)")
        return 0
    for _ in range(2):                       # twice: resolve the ToC / refs
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             "README.tex"],
            cwd=DOCS, capture_output=True, text=True)
    if r.returncode != 0:
        print("  xelatex reported errors; README.tex is written, "
              "see docs/README.log", file=sys.stderr)
        return 1
    for ext in (".aux", ".log", ".out", ".toc"):
        (DOCS / f"README{ext}").unlink(missing_ok=True)
    print(f"  wrote {(DOCS / 'README.pdf').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
