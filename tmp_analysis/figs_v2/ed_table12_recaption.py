"""Rewrite Extended Data Table 2's caption from the committed aggregate, leaving its rows alone.

  PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/ed_table12_recaption.py

readout_w_cortex.py writes both the caption and the rows, but it needs the per-subject npz files on
/scratch. The caption is a pure function of results/figures_v2/readout_w_cortex.json, so it can be
refreshed here; the numbers are identical either way.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from readout_w_cortex import caption12  # noqa: E402

MD = "results/figures_v2/ed_tables.md"
HEAD = "**Extended Data Table 2 |"


def main():
    out = json.load(open("results/figures_v2/readout_w_cortex.json"))
    md = open(MD, encoding="utf8").read()
    i = md.index(HEAD)
    j = md.index("\n\n", i)                       # the caption is one paragraph
    md = md[:i] + caption12(out) + md[j:]
    open(MD, "w", encoding="utf8").write(md)
    print(f"recaptioned Extended Data Table 2 in {MD}")


if __name__ == "__main__":
    main()
