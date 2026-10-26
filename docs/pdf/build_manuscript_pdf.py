#!/usr/bin/env python
"""Build docs/pdf/PAPER_DRAFT.pdf from docs/PAPER_DRAFT.md.

The manuscript markdown is left untouched. This script assembles a copy with
the six figures (vector PDFs from results/figures/) inserted at the end of the
subsection in which each is first cited, each on a fresh page with its full
legend taken verbatim from results/figures/<fig>_caption.md, converts it with
pandoc to a LaTeX body, wraps it in a fixed preamble and compiles with tectonic.

Usage:
    python docs/pdf/build_manuscript_pdf.py [--out docs/pdf/PAPER_DRAFT.pdf]

Requirements: pandoc (>= 3) and tectonic on PATH, or set PANDOC / TECTONIC.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Two manuscript profiles share the build. `draft` is the NMI-oriented draft with
# its figures in results/figures/; `v2` is the BA-aligned rewrite with the
# figures of tmp_analysis/figs_v2/ in results/figures_v2/.
PROFILES = {
    "draft": dict(
        draft=REPO / "docs" / "PAPER_DRAFT.md",
        figdir=REPO / "results" / "figures",
        out=REPO / "docs" / "pdf" / "PAPER_DRAFT.pdf",
        cite=r"\*\*{key}",          # figures are cited in bold in the draft
        figures=[("Figure 1", "fig1_convergence", "Figure 1"),
                 ("Figure 2", "fig2_ventral_gradient", "Figure 2"),
                 ("Figure 3", "fig3_readout_sweep", "Figure 3"),
                 ("Figure 4", "fig4_ood_degradation", "Figure 4"),
                 ("Figure 5", "fig5_human_correspondence", "Figure 5")],
        extended=[("Extended Data Fig", "figE1_routes_and_accuracy", "Extended Data Figure 1")],
    ),
    "v2": dict(
        draft=REPO / "docs" / "PAPER_v2.md",
        figdir=REPO / "results" / "figures_v2",
        out=REPO / "docs" / "pdf" / "PAPER_v2.pdf",
        cite=r"\(Fig\. {key}",      # cited as (Fig. 1a) etc.
        figures=[("1", "fig1_model", "Figure 1"),
                 ("2", "fig2_streams", "Figure 2"),
                 ("3", "fig3_cortex", "Figure 3"),
                 ("4", "fig4_readout", "Figure 4"),
                 ("5", "fig5_dissociations", "Figure 5"),
                 ("6", "fig6_generality", "Figure 6")],
        extended=[("E1", "figE1_ood_curves", "Extended Data Figure 1"),
                  ("E2", "figE2_dimensionality", "Extended Data Figure 2"),
                  ("E3", "figE3_cortex_controls", "Extended Data Figure 3"),
                  ("E4", "figE4_ext_gradient", "Extended Data Figure 4"),
                  ("E5", "figE5_readout_w", "Extended Data Figure 5"),
                  ("E6", "figE6_zoo", "Extended Data Figure 6")],
    ),
}
# v3 is the dvd300 rewrite: same figure set and citation style as v2, drawn on the model in results/figures_v3/.
PROFILES["v3"] = dict(PROFILES["v2"], draft=REPO / "docs" / "PAPER_v3.md", figdir=REPO / "results" / "figures_v3",
                      out=REPO / "docs" / "pdf" / "PAPER_v3.pdf")
DRAFT = PROFILES["draft"]["draft"]
FIGDIR = PROFILES["draft"]["figdir"]
FIGURES = PROFILES["draft"]["figures"]
EXTENDED = PROFILES["draft"]["extended"]
CITE = PROFILES["draft"]["cite"]

PANDOC = os.environ.get("PANDOC", "pandoc")
TECTONIC = os.environ.get("TECTONIC", "tectonic")

PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm,headheight=14pt]{geometry}
\usepackage{fontspec}
\setmainfont{texgyrepagella-regular.otf}[BoldFont=texgyrepagella-bold.otf,
  ItalicFont=texgyrepagella-italic.otf,BoldItalicFont=texgyrepagella-bolditalic.otf]
\setsansfont{texgyreheros-regular.otf}[BoldFont=texgyreheros-bold.otf,
  ItalicFont=texgyreheros-italic.otf,BoldItalicFont=texgyreheros-bolditalic.otf]
\setmonofont{DejaVuSansMono.ttf}[Scale=MatchLowercase,BoldFont=DejaVuSansMono-Bold.ttf,
  ItalicFont=DejaVuSansMono-Oblique.ttf,BoldItalicFont=DejaVuSansMono-BoldOblique.ttf]
\PassOptionsToPackage{hyphens}{url}
\usepackage{amsmath,amssymb}
\usepackage{newunicodechar}
% glyphs Pagella lacks, mapped font-independently
\newunicodechar{₁}{\textsubscript{1}}
\newunicodechar{₂}{\textsubscript{2}}
\newunicodechar{ₚ}{\textsubscript{p}}
\newunicodechar{ₘ}{\textsubscript{m}}
\newunicodechar{⁻}{\textsuperscript{-}}
\newunicodechar{⁴}{\textsuperscript{4}}
\newunicodechar{⁶}{\textsuperscript{6}}
\newunicodechar{⁸}{\textsuperscript{8}}
\newunicodechar{ᵃ}{\textsuperscript{a}}
\newunicodechar{ᵇ}{\textsuperscript{b}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{✱}{\ensuremath{\ast}}
\usepackage{graphicx}
\usepackage{longtable,booktabs,array,calc}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\usepackage{caption}
% Pandoc wraps a caption-less longtable in \def\LTcaptype{none}. With the caption
% package loaded, ltcaption still calls \refstepcounter{\LTcaptype}, so the counter
% has to exist. Nothing ever typesets it.
\newcounter{none}
\usepackage{microtype}
\usepackage{xcolor}
\definecolor{linkblue}{RGB}{0,72,140}
\usepackage[hidelinks]{hyperref}
\hypersetup{colorlinks=true,linkcolor=linkblue,urlcolor=linkblue,citecolor=linkblue,
  pdftitle={__TITLE__},pdfauthor={}}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\itshape __SHORT__}
\fancyhead[R]{\small\thepage}
\fancyfoot[L]{\footnotesize Manuscript draft \textperiodcentered{} __STAMP__}
\renewcommand{\headrulewidth}{0.3pt}
\setcounter{secnumdepth}{-1}
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries\sffamily}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries\sffamily}{}{0pt}{}
\titleformat{\subsubsection}{\normalsize\bfseries\sffamily}{}{0pt}{}
\titlespacing*{\section}{0pt}{20pt plus 4pt}{8pt}
\titlespacing*{\subsection}{0pt}{14pt plus 3pt}{5pt}
\titlespacing*{\subsubsection}{0pt}{10pt plus 2pt}{3pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 1pt}
\linespread{1.06}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcommand{\passthrough}[1]{#1}
\urlstyle{same}
\renewcommand{\UrlBreaks}{\do\/\do\-\do\.\do\_\do\?\do\=\do\&\do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j\do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t\do\u\do\v\do\w\do\x\do\y\do\z\do\A\do\B\do\C\do\D\do\E\do\F\do\G\do\H\do\I\do\J\do\K\do\L\do\M\do\N\do\O\do\P\do\Q\do\R\do\S\do\T\do\U\do\V\do\W\do\X\do\Y\do\Z\do\0\do\1\do\2\do\3\do\4\do\5\do\6\do\7\do\8\do\9}
\emergencystretch=2em
\hbadness=10000
\begin{document}
\begin{center}
{\LARGE\bfseries\sffamily __TITLE__\par}
\vspace{10pt}
{\small Manuscript draft, compiled __STAMP__ from \texttt{__SOURCE__} at commit \texttt{__COMMIT__}.\par}
\end{center}
\vspace{6pt}
"""

POSTAMBLE = "\n\\end{document}\n"


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, text=True, **kw)


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def legend_markdown(stem: str, display: str) -> str:
    """Caption file -> legend markdown. Drops the '# Figure N — caption' heading
    and horizontal rules; folds the leading bold title into 'Figure N | Title'."""
    text = (FIGDIR / f"{stem}_caption.md").read_text(encoding="utf8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("# ") and ln.strip() != "---"]
    paras = re.split(r"\n\s*\n", "\n".join(lines).strip())
    title = paras[0].strip()
    m = re.match(r"^\*\*(.+)\*\*$", title, flags=re.S)
    title_txt = m.group(1).strip() if m else title
    title_txt = " ".join(title_txt.split())
    if not title_txt.endswith("."):
        title_txt += "."
    head = f"**{display} | {title_txt}**"
    body = "\n\n".join(p.strip() for p in paras[1:])
    return head + "\n\n" + body


def figure_block(stem: str, display: str, newpage: bool = True) -> str:
    pdf = FIGDIR / f"{stem}.pdf"
    if not pdf.exists():
        pdf = FIGDIR / f"{stem}.png"
    legend = legend_markdown(stem, display)
    return (
        ("\n\n\\clearpage\n\n" if newpage else "\n\n")
        + "\\begin{center}\\includegraphics[width=\\linewidth,height=0.6\\textheight,"
        f"keepaspectratio]{{{pdf.as_posix()}}}\\end{{center}}\n\n"
        "\\begingroup\\small\\setlength{\\parskip}{4pt}\n\n"
        f"{legend}\n\n"
        "\\endgroup\n\n"
    )


_TT = re.compile(r"\\texttt\{((?:[^{}\\]|\\.|\{[^{}]*\})*)\}")


def breakable_code(tex: str) -> str:
    """Let long inline code (file paths, identifiers, hostnames) break after
    '/', '.', '_', ',' and '{' instead of running into the margin."""
    def fix(m):
        s = m.group(1)
        s = re.sub(r"(/|\\_|\.|,|-|\\\{)", r"\1\\allowbreak{}", s)
        return "\\texttt{" + s + "}"
    return _TT.sub(fix, tex)


def strip_status_block(lines: list[str]) -> list[str]:
    """Remove internal '**Status:' / '**Positioning' preamble paragraphs that
    precede the Abstract heading, if present."""
    out, i = [], 0
    abstract_seen = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## Abstract"):
            abstract_seen = True
        if not abstract_seen and re.match(r"^\*\*(Status|Positioning)", ln):
            while i < len(lines) and lines[i].strip() != "":
                i += 1
            continue
        if not abstract_seen and ln.strip() == "---":
            i += 1
            continue
        out.append(ln)
        i += 1
    return out


def assemble(md: str) -> tuple[str, str]:
    lines = md.splitlines()
    title = next(ln[2:].strip() for ln in lines if ln.startswith("# "))
    lines = [ln for ln in lines if not ln.startswith("# ")]
    lines = strip_status_block(lines)

    def is_heading(ln: str) -> bool:
        return bool(re.match(r"^#{2,4} ", ln))

    # Figures are placed by their first citation inside the Results section, so
    # that the Introduction's forward references do not pull every figure to
    # the end of the Introduction.
    start = next((i for i, ln in enumerate(lines) if re.match(r"^## 2\. Results", ln)), 0)

    # Figures are placed in numerical order: figure N goes after the paragraph
    # holding its first citation that follows figure N-1's citation, so the
    # compiled PDF never shows Fig. 4 before Fig. 2 (a §2.1 forward reference to
    # Fig. 4c used to pull it ahead). A figure whose next citation lies beyond
    # the Results section is placed at the end of Results.
    results_end = next((i for i, ln in enumerate(lines) if re.match(r"^## 3\. Discussion", ln)), len(lines))
    insertions: dict[int, list[str]] = {}
    cursor = start
    for key, stem, display in FIGURES:
        pat = re.compile(CITE.format(key=re.escape(key)))
        first = next((i for i, ln in enumerate(lines) if i >= cursor and pat.search(ln)), None)
        if first is None:
            print(f"warning: {key} not cited after the previous figure; placing at end of Results", file=sys.stderr)
            first = results_end
        if first >= results_end:
            nxt = results_end
        else:
            nxt = next((j for j in range(first + 1, len(lines)) if is_heading(lines[j])), len(lines))
        insertions.setdefault(nxt, []).append(figure_block(stem, display))
        cursor = first

    out = []
    for i, ln in enumerate(lines):
        if i in insertions:
            out.extend(insertions[i])
        out.append(ln)
    if len(lines) in insertions:
        out.extend(insertions[len(lines)])

    out.append("\n\n\\clearpage\n\n## Extended Data\n")
    for _key, stem, display in EXTENDED:
        out.append(figure_block(stem, display, newpage=False))
    return title, "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="draft")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", default=None, help="keep intermediates here")
    args = ap.parse_args()

    global DRAFT, FIGDIR, FIGURES, EXTENDED, CITE
    prof = PROFILES[args.profile]
    DRAFT, FIGDIR, FIGURES, EXTENDED, CITE = prof["draft"], prof["figdir"], prof["figures"], prof["extended"], prof["cite"]
    if args.out is None:
        args.out = str(prof["out"])

    md = DRAFT.read_text(encoding="utf8")
    title, assembled = assemble(md)
    commit = git_head()
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="mpk_pdf_"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "assembled.md").write_text(assembled, encoding="utf8")

    run([PANDOC, str(work / "assembled.md"),
         "-f", "markdown+smart+autolink_bare_uris+raw_tex",
         "-t", "latex", "--wrap=none",
         "-o", str(work / "body.tex")])
    body = (work / "body.tex").read_text(encoding="utf8")
    body = breakable_code(body)

    def esc(s: str) -> str:
        return s.replace("_", r"\_").replace("&", r"\&")

    short = esc(title if len(title) < 90 else title.split(":")[0])
    source = esc(str(DRAFT.relative_to(REPO)))
    tex = (PREAMBLE.replace("__TITLE__", esc(title)).replace("__SHORT__", short).replace("__STAMP__", stamp)
           .replace("__SOURCE__", source)
           .replace("__COMMIT__", commit)
           + body + POSTAMBLE)
    (work / "manuscript.tex").write_text(tex, encoding="utf8")

    run([TECTONIC, "-X", "compile", "--keep-logs", "--keep-intermediates",
         str(work / "manuscript.tex")], cwd=work)

    log = (work / "manuscript.log").read_text(encoding="utf8", errors="replace")
    missing = sorted(set(re.findall(r"Missing character: There is no (\S+)", log)))
    if missing:
        print("WARNING missing glyphs:", " ".join(missing), file=sys.stderr)
    over = len(re.findall(r"Overfull \\hbox", log))
    print(f"overfull hboxes: {over}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / "manuscript.pdf", out)
    print(f"wrote {out}  (intermediates in {work})")


if __name__ == "__main__":
    main()
