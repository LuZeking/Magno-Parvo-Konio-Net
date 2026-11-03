"""Stage B of docs/PLAN_SUPP_MODEL_2026-11-03.md: insert the model-of-record rows into docs/PAPER_v3.md's
Supplementary Tables S1-S10 (from results/supp_v3/supp_tables_model.md, verbatim) and make the section-4 text
edits (header, preface, Notes S1, S2, S4, S5). Every replacement asserts its old text occurs exactly once and the
new text not at all; the file is written only if every edit applies (all or nothing). Idempotent: a second run
finds the old texts gone and refuses.

Usage: python tmp_analysis/figs_v3/patch_supp_model.py [--md results/supp_v3/supp_tables_model.md] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PAPER = os.path.join(ROOT, "docs", "PAPER_v3.md")

S1_MODEL = """**For the model of record.** The main text makes none of the following claims; each item is drawn from its own statements.

- That the two-stream split is necessary for the dissociation, or that it buys a shape-bias advantage over one stream: no single-trunk network was trained on the dvd300 schedule, so the single-stream rows of Supplementary Table S4 have no counterpart for the model.
- Any dorsal/ventral or anatomical magno/parvo mapping of the two trunks: the streams are named by their input, and the atlas rows of Supplementary Table S10 are read as sign and ordering only.
- That the cortical division separates task load or the carrier from texture bias: in every network the stream carrying extrastriate cortex also carries the task, and in the model it is the chromatic one too (Results 2.3); no capacity-matched or task-load-matched control was trained.
- That predicted cortex is texture-biased on the cue-conflict images, or that the human readout was measured: the cortex-to-decision test returned its registered failure reading with P-C1, P-C2 and P-C4 not met (Results 2.5), and its instrument does not exceed its input floor beyond hV4.
- That the model's shape bias is human-like: its rate lies below the observers', its error consistency stays below a third of the human–human ceiling and does not separate it from the control (Results 2.4), and the trajectory buys no out-of-distribution robustness (Supplementary Table S5).
- That the chromatic stream predicts colour-selective cortex: it was not tested on the model (the koniocellular control of Supplementary Note S2 is the BA front end's).
- That the resolution-tracking result is established beyond three seeds against three: its exact Mann–Whitney p sits at the floor, and its predicted-cortex counterpart points the other way (Results 2.5).

**The BA network's list (PAPER_v2).**

"""

PREFACE_OLD = ("> **Supplementary material in this draft.** Supplementary Notes S1–S5 and Tables S1–S10 below are PAPER_v2's and describe the BA network "
               "(`dkl_bcs`); the main text cites them for the BA network only. Their rows for the model of record are being regenerated ⟦PENDING⟧.")
PREFACE_NEW = ("> **Supplementary material in this draft.** Each Supplementary Table gives the model of record's rows first (model, control and channel swap, "
               "three seeds each, and the untrained network on their shared front end where the table has such a column), generated from stored outputs by "
               "`tmp_analysis/figs_v3/supp_tables_model.py` (record `results/supp_v3/`; plan `docs/PLAN_SUPP_MODEL_2026-11-03.md`), and then PAPER_v2's rows "
               "for the BA network (`dkl_bcs`), which the main text cites as the comparison. Supplementary Notes S2 and S4 are the BA network's and say so; "
               "Notes S1, S3 and S5 cover both.")

EDITS = [
    ("header: pending sentence", "Still marked ⟦PENDING⟧: the Supplementary rows for the model. ", ""),
    ("header: 2026-11-03 entry",
     "and Results 2.5 quotes the forced estimator's 0.648 beside the cortical rates, whose denominator it shares.",
     "and Results 2.5 quotes the forced estimator's 0.648 beside the cortical rates, whose denominator it shares. **2026-11-03:** the Supplementary Tables carry "
     "the model of record's rows above PAPER_v2's BA rows, generated from stored outputs under `docs/PLAN_SUPP_MODEL_2026-11-03.md` (record `results/supp_v3/`, "
     "job 26764879); Note S1 lists the claims the model does not make, Note S5 the plan, and Notes S2 and S4 are labelled as the BA network's."),
    ("preface", PREFACE_OLD, PREFACE_NEW),
    ("Note S1: model list", "## Supplementary Note S1 — claims deliberately not made\n\n- That the two-stream split is necessary",
     "## Supplementary Note S1 — claims deliberately not made\n\n" + S1_MODEL + "- That the two-stream split is necessary"),
    ("Note S2: label", "The thesis omitted the koniocellular pathway for simplicity. Because L−M fails",
     "This control belongs to the BA front end of PAPER_v2 and was not run on the model of record. The thesis omitted the koniocellular pathway for simplicity. Because L−M fails"),
    ("Note S4: heading", "## Supplementary Note S4 — passages condensed in the main text",
     "## Supplementary Note S4 — PAPER_v2's passages on the BA network, condensed in its main text"),
    ("Table S4 caption: parameter breakdown (results/supp_v3/supp_tables_model.json, tables.S4.arms.*.parameters.by_module)",
     "parameters are counted from the stored state dict with BatchNorm running statistics excluded.",
     "parameters are counted from the stored state dict with BatchNorm running statistics excluded (two trunks of 11,170,240, the joint classifier of 57,456 "
     "and the two per-stream heads of 57,456 each, identical across the three arms)."),
    ("Note S5: plan", "its predictions P-C1–P-C5, decision rules, floors and nulls are fixed there.",
     "its predictions P-C1–P-C5, decision rules, floors and nulls are fixed there. The Supplementary rows for the model of record were generated under "
     "`docs/PLAN_SUPP_MODEL_2026-11-03.md`, which fixed their sources, statistics and a consistency gate against the main text before the job ran "
     "(`results/supp_v3/EXECUTION_LEDGER.md`)."),
]


def blocks(md: str) -> dict[str, tuple[str, list[str]]]:
    """{'S3': (caption sentence, table lines)} from the generator's markdown."""
    out = {}
    parts = re.split(r"(?m)^(?=\*\*Supplementary Table S\d+ \|)", md)
    for part in parts:
        m = re.match(r"\*\*Supplementary Table (S\d+) \| .*? — model of record\.\*\* (.*)", part.split("\n", 1)[0])
        if not m:
            continue
        body = part.split("\n", 1)[1].strip("\n").split("\n")
        out[m.group(1)] = (m.group(2).strip(), body)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(ROOT, "results", "supp_v3", "supp_tables_model.md"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    text = open(PAPER, encoding="utf8").read()
    B = blocks(open(args.md, encoding="utf8").read())
    assert sorted(B, key=lambda k: int(k[1:])) == [f"S{i}" for i in range(1, 11)], sorted(B)
    new = text
    for k in [f"S{i}" for i in range(1, 11)]:
        cap, body = B[k]
        pat = re.compile(rf"(?m)^(\*\*Supplementary Table {k} \| [^\n]*\n)\n(?=\|)")
        assert len(pat.findall(new)) == 1, (k, len(pat.findall(new)))
        ins = f"*Model of record.* {cap}\n\n" + "\n".join(body) + "\n\n*BA network (PAPER_v2).*\n\n"
        assert ins not in new
        new = pat.sub(lambda m: m.group(1) + "\n" + ins, new, count=1)
    for name, old, rep in EDITS:
        assert new.count(old) == 1, (name, new.count(old))
        assert rep == "" or rep not in new, (name, "new text already present")
        new = new.replace(old, rep)
    assert new.count("⟦PENDING⟧") == 1, new.count("⟦PENDING⟧")   # the 2026-10-26 changelog sentence, resolved on 10-28, keeps its marker
    n_rows = sum(1 for k in B for l in B[k][1] if l.startswith("|") and not l.startswith("|---"))
    print(f"edits: {len(EDITS)} text edits, 10 table blocks, {n_rows} table lines inserted; PAPER {len(text.splitlines())} -> {len(new.splitlines())} lines")
    if args.dry_run:
        return 0
    open(PAPER, "w", encoding="utf8").write(new)
    print("written", PAPER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
