"""Forced two-way shape/texture estimator, from the stored per-image logits.

The paper's primary cue-conflict statistic follows Geirhos et al.: restrict the
logits to the 16 categories, take the argmax, and score shape / texture over the
images decided as one of the two. Between 58% and 77% of images are decided as a
third category and drop out of the denominator. The forced two-way estimator
keeps every image: for each image compare the logit of the shape category with
the logit of the texture category and score which is larger. Both estimators are
reported; the forced one is the referees' request (NMI review v2, major 5).

Writes results/figures_v2/forced_choice.json and appends Supplementary Table S9
to results/figures_v2/ed_tables.md.

Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/forced_choice.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from src.analysis.behavior.cue_conflict import load_class_index

CI = load_class_index()


def forced(logits, shapes, textures):
    ls = logits[np.arange(len(shapes)), [CI[c] for c in shapes]]
    lt = logits[np.arange(len(textures)), [CI[c] for c in textures]]
    return float(np.mean(ls > lt))


def restricted(logits, shapes, textures):
    from src.analysis.behavior.cue_conflict import shape_bias
    return shape_bias(logits, list(shapes), list(textures), CI)


def arm(pattern, cond):
    out = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if cond not in z.files:
            continue
        r = restricted(z[cond], z["shapes"], z["textures"])
        out.append(dict(file=os.path.basename(f), forced=forced(z[cond], z["shapes"], z["textures"]),
                        restricted=r["shape_bias"], decided=r["n_shape"] + r["n_texture"]))
    return out


ROWS = [
    ("diet, magno stream alone (best ckpt)", "results/cue_conflict_bestep/dkl_bcs_seed?_ep*_logits.npz", "magno_only"),
    ("diet, both streams w = 0.5 (best ckpt)", "results/cue_conflict_bestep/dkl_bcs_seed?_ep*_logits.npz", "full"),
    ("diet, parvo stream alone (best ckpt)", "results/cue_conflict_bestep/dkl_bcs_seed?_ep*_logits.npz", "parvo_only"),
    ("diet, magno stream alone (final ckpt)", "results/cue_conflict/dkl_bcs_seed?_logits.npz", "magno_only"),
    ("diet, both streams w = 0.5 (final ckpt)", "results/cue_conflict/dkl_bcs_seed?_logits.npz", "full"),
    ("diet, parvo stream alone (final ckpt)", "results/cue_conflict/dkl_bcs_seed?_logits.npz", "parvo_only"),
    ("no diet, both streams", "results/cue_conflict/dkl_base_seed?_logits.npz", "full"),
    ("no diet, magno stream alone", "results/cue_conflict/dkl_base_seed?_logits.npz", "magno_only"),
    ("no diet, parvo stream alone", "results/cue_conflict/dkl_base_seed?_logits.npz", "parvo_only"),
    ("swap, both streams", "results/cue_conflict/dkl_bcs_swap_seed?_logits.npz", "full"),
    ("swap, parvo stream alone (L−M + transform)", "results/cue_conflict/dkl_bcs_swap_seed?_logits.npz", "parvo_only"),
    ("parvo_highsf, both streams", "results/cue_conflict/dkl_bcs_parvo_highsf_seed?_logits.npz", "full"),
    ("parvo_highsf, stream 2 alone (L−M + luminance)", "results/cue_conflict/dkl_bcs_parvo_highsf_seed?_logits.npz", "parvo_only"),
    ("single stream, ResNet-34 (matched)", "results/cue_conflict_single34_bestep/*_logits.npz", "full"),
    ("single stream, ResNet-18 (half)", "results/cue_conflict_single_bestep/*_logits.npz", "full"),
    ("no diet, accuracy-matched (epochs 10, 10, 11), magno alone", "results/cue_conflict_matched/*_logits.npz", "magno_only"),
    ("thesis checkpoint (progressive schedule), both streams", "results/cue_conflict/thesis_dvd_trans_logits.npz", "full"),
    ("untrained, diet front end, magno alone", "results/cue_conflict/untrained_blur_cs_logits.npz", "magno_only"),
    ("untrained, no-diet front end, magno alone", "results/cue_conflict/untrained_noblur_logits.npz", "magno_only"),
]


def main():
    res = {}
    lines = ["**Supplementary Table S9 | Cue-conflict shape bias under the forced two-way estimator.** The primary estimator (Methods 4.3) restricts the logits to the 16 categories, takes the argmax, and scores shape against texture over the images decided as one of the two, so images decided as a third category drop out of the denominator (“decided”). The forced estimator keeps all 1,200 images and scores, per image, whether the logit of the shape category exceeds the logit of the texture category. Mean over seeds [range]. Source: the stored per-image logits in `results/cue_conflict*/`.", "",
             "| model / readout | restricted (primary) | decided / 1,200 | forced two-way | seeds |", "|---|---|---|---|---|"]
    for lab, pat, cond in ROWS:
        rows = arm(pat, cond)
        if not rows:
            print("  missing", lab); continue
        fc = np.array([r["forced"] for r in rows]); rs = np.array([r["restricted"] for r in rows]); de = np.array([r["decided"] for r in rows])
        res[lab] = dict(forced=fc.tolist(), restricted=rs.tolist(), decided=de.tolist())
        rng = lambda v: f"{v.mean():.3f}" + (f" [{v.min():.3f}, {v.max():.3f}]" if len(v) > 1 else "")
        lines.append(f"| {lab} | {rng(rs)} | {de.mean():.0f} | {rng(fc)} | {len(rows)} |")
        print(f"  {lab:55s} restricted {rs.mean():.3f}  forced {fc.mean():.3f}  decided {de.mean():.0f}  n={len(rows)}")
    # the 2x2 on the forced estimator
    def cell(lab):
        return np.mean(res[lab]["forced"])
    two = {"chromatic, no transform (no diet, parvo)": cell("no diet, parvo stream alone"),
           "chromatic, transform (swap, parvo)": cell("swap, parvo stream alone (L−M + transform)"),
           "achromatic, no transform (parvo_highsf, stream 2)": cell("parvo_highsf, stream 2 alone (L−M + luminance)"),
           "achromatic, transform (diet, magno; final ckpt)": cell("diet, magno stream alone (final ckpt)")}
    res["2x2_forced"] = two
    print("  2x2 forced:", {k: round(v, 3) for k, v in two.items()})
    lines.append("")
    json.dump(res, open("results/figures_v2/forced_choice.json", "w"), indent=1)
    # Upsert Table 10 in place: Tables 11-16 in the same file belong to other generators
    import re as _re
    md = open("results/figures_v2/ed_tables.md", encoding="utf8").read()
    block = "\n".join(lines).strip() + "\n"
    pat = _re.compile(r"\*\*Supplementary Table S9 \|.*?(?=\n\*\*Extended Data Table|\Z)", _re.S)
    if pat.search(md):
        md = pat.sub(lambda m: block, md, count=1)
    else:
        md = md.rstrip() + "\n\n" + block
    open("results/figures_v2/ed_tables.md", "w", encoding="utf8").write(md)
    print("upserted Supplementary Table S9")


if __name__ == "__main__":
    main()
