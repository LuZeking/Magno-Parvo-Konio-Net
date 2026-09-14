"""Extended Data Tables 13 (behaviour) and 14 (cortex) for the generality test.

Reads results/figures_v2/generality_behaviour.json, the *_gain_accuracy.json and
*_refit_sum.json files, and results/figures_v2/generality_cortex.json; replaces or
inserts the two tables in results/figures_v2/ed_tables.md before the END marker.
Usage: python tmp_analysis/figs_v2/ed_tables_generality.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

RUN = os.environ.get("GEN_RUNID", "2026-09-14_gen")
BEH_DIR = f"/scratch/byrong/mpk/generality/behaviour/{RUN}"
ORDER = ["concat", "single34", "single18", "rgb_tsn", "vit_b_16", "resnet152", "convnext_base", "convnext_small", "resnet50", "vgg16"]
LABEL = {"concat": "two-stream, diet, streams concatenated (validation)", "single34": "single ResNet-34, diet",
         "single18": "single ResNet-18, diet", "rgb_tsn": "single ResNet-18, no diet (`rgb_tsn`)",
         "vit_b_16": "ImageNet ViT-B/16", "resnet152": "ImageNet ResNet-152", "convnext_base": "ImageNet ConvNeXt-B",
         "convnext_small": "ImageNet ConvNeXt-S", "resnet50": "ImageNet ResNet-50", "vgg16": "ImageNet VGG-16"}
REGIONS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "dorsal"]
FAM_LABEL = {"two_stream_diet": "two-stream, diet (magno − parvo)", "single34": "single ResNet-34, diet",
             "rgb_tsn": "single ResNet-18, no diet", "resnet50": "ImageNet ResNet-50"}


def fmt(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def table13(beh):
    fam = {f["key"]: f for f in beh["families"]}
    rows = ["| network | units | trained rate | w = 0 | w = 1 | span (held-out) [range] | ρ(w, rate) | forced-estimator span | shape-leaning units | w within 5 pts of trained top-1 | refit span linear / two-layer | seeds |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in ORDER:
        if key not in fam:
            continue
        f = fam[key]
        tags = [r["tag"] for r in f["seeds"]]
        acc = []
        for t in tags:
            p = os.path.join(BEH_DIR, f"{t}_gain_accuracy.json")
            if os.path.exists(p):
                z = json.load(open(p)); acc.append((min(z["w_within_5pts"]), max(z["w_within_5pts"]), z["trained_top1"]))
        acc_s = f"{np.mean([a[0] for a in acc]):.2f}–{np.mean([a[1] for a in acc]):.2f} (top-1 {np.mean([a[2] for a in acc]):.3f})" if acc else "—"
        ref = [r["refit"] for r in f["seeds"] if "refit" in r]
        if ref:
            lin = np.mean([r["linear"]["span"] for r in ref if "linear" in r]); mlp = np.mean([r["mlp"]["span"] for r in ref if "mlp" in r])
            ref_s = f"{lin:.2f} / {mlp:.2f}"
        else:
            ref_s = "—"
        rows.append(f"| {LABEL[key]} | {f['seeds'][0]['n_units']} | {f['trained']:.3f} | {f['rate_w0']:.3f} | {f['rate_w1']:.3f} | "
                    f"{f['span']:.3f} [{f['span_range'][0]:.2f}, {f['span_range'][1]:.2f}] | {f['rho']:.2f} | {f['forced_span']:.3f} | "
                    f"{f['frac_shape_leaning']:.2f}" + (f" (S from stream 1: {f['frac_S_from_stream1']:.2f})" if f.get("frac_S_from_stream1") is not None else "")
                    + f" | {acc_s} | {ref_s} | {len(f['seeds'])} |")
    cap = ("**Extended Data Table 13 | The readout span without a stream split: fixed-head gain sweep on ten frozen networks.** "
           "Each unit of a network's penultimate layer is scored by the fraction of its variance between shape categories minus that between texture categories on the 1,200 cue-conflict images (one-way η²); the median split gives a shape-leaning half S and a texture-leaning half T. "
           "logits(w) = head(g(w) ⊙ y) with g = 2w on S and 2(1 − w) on T, the network's own head untouched; w = 0.5 is the trained network. "
           "The split is defined on one stratified half of the images and the rate scored on the other (10 random splits, both directions; mean, with the range over splits and seeds in brackets). "
           "Rates by the paper's primary estimator (112-class, decided images) for our networks and the Geirhos 16-class probability aggregation for the ImageNet networks; “forced-estimator span” is the same sweep on the forced two-way estimator. "
           "“shape-leaning units” is the fraction of units with η²_shape > η²_texture on all images; for the two-stream concatenation the fraction of S that belongs to stream 1 is given in parentheses (P3). "
           "“w within 5 pts” is the w-range over which top-1 on held-out natural images (MiniEcoset validation; a stratified 10-per-class ILSVRC-2012 validation subset for the ImageNet networks) stays within 5 points of the trained value. "
           "“refit span” is the lossy-sum protocol z(w) = w·y_S + (1 − w)·y_T with a linear and a two-layer head refitted at every w (Methods 4.3 budget), our networks only. "
           "Pre-specification: `docs/PLAN_GENERALITY_2026-09-14.md`. Source: `/scratch/byrong/mpk/generality/behaviour/" + RUN + "/`.")
    return cap + "\n\n" + "\n".join(rows)


def table14(ctx):
    rows = ["| network | statistic | " + " | ".join(REGIONS) + " | hV4 − V1 (n neg, p) | extrastriate − V1 (n neg, p) | ρ(rank, index) V1–hV4 |",
            "|---|---|" + "---|" * len(REGIONS) + "---|---|---|"]
    for fam in ("two_stream_diet", "single34", "rgb_tsn", "resnet50"):
        if fam not in ctx["families"]:
            continue
        fo = ctx["families"][fam]
        st = fo.get("stats", {})
        def stat_s(k):
            v = st.get(k)
            return f"{v['mean']:+.3f} ({v['n_neg']}/{v['n']}, {fmt(v['p'], 4)})" if v else "—"
        rho = st.get("rho_rank_V1_hV4", {}).get("mean")
        rows.append(f"| {FAM_LABEL[fam]} | index | " + " | ".join(f"{np.nanmean(fo['index'][r]):+.2f}" for r in REGIONS)
                    + f" | {stat_s('hV4_minus_V1')} | {stat_s('extrastriate_minus_V1')} | {fmt(rho, 2, True)} |")
        rows.append(f"| | U_shape | " + " | ".join(f"{np.nanmean(fo['uS'][r]):.3f}" for r in REGIONS) + " | | | |")
        rows.append(f"| | U_texture | " + " | ".join(f"{np.nanmean(fo['uT'][r]):.3f}" for r in REGIONS) + " | | | |")
        if fo.get("w_star"):
            wm = st.get("wstar_median", {})
            wst = st.get("wstar_extrastriate_minus_V1")
            rows.append(f"| | w*_cortex (median) | " + " | ".join(fmt(wm.get(r), 2) for r in REGIONS)
                        + f" | | Δw* extrastriate − V1 {wst['mean']:+.2f} ({wst['n_neg']}/{wst['n']} neg, {fmt(wst['p'], 4)}) | |" if wst else " | | | |")
    cap = ("**Extended Data Table 14 | Cortex without a stream split: the shape- and texture-leaning halves as encoding bands.** "
           "For each single-trunk network the penultimate units were split as in Extended Data Table 13 (all 1,200 images) and the two halves entered as two bands of a banded ridge model of every `nsdgeneral` voxel (same split, folds and alpha grid as Methods 4.5); "
           "U_shape and U_texture are the noise-ceiling-normalised unique variances and the index is (U_shape − U_texture)/(U_shape + U_texture), median over reliable voxels (noise ceiling > 0.1, ≥ 20 voxels), mean over eight subjects (seeds averaged per subject). "
           "Regions: V1–hV4 from `prf-visualrois`; ventral (VO1/2, PHC1/2), lateral (LO1/2, TO1/2) and dorsal (V3A/B, IPS0–2) from Kastner2015. "
           "The two-stream diet row applies the same computation to the magno (shape-biased) and parvo (texture-biased) bands of Fig. 3. "
           "w*_cortex is the mixing weight of the lossy sum z(w) = w·y_S + (1 − w)·y_T that maximises an ROI's median test R² (Extended Data Fig. 5 protocol). Exact paired Wilcoxon over subjects. "
           "Source: `/scratch/byrong/mpk/generality/cortex/" + RUN + "/`.")
    return cap + "\n\n" + "\n".join(rows)


def upsert(md, title_prefix, block):
    pat = re.compile(r"\*\*" + re.escape(title_prefix) + r".*?(?=\n\*\*Extended Data Table|\n<!-- ED_TABLES_END -->)", re.S)
    if pat.search(md):
        return pat.sub(lambda m: block + "\n", md, count=1)
    return md.replace("<!-- ED_TABLES_END -->", block + "\n\n<!-- ED_TABLES_END -->")


def main():
    beh = json.load(open("results/figures_v2/generality_behaviour.json"))
    md_path = "results/figures_v2/ed_tables.md"
    md = open(md_path, encoding="utf8").read()
    md = upsert(md, "Extended Data Table 13 |", table13(beh))
    if os.path.exists("results/figures_v2/generality_cortex.json"):
        ctx = json.load(open("results/figures_v2/generality_cortex.json"))
        if ctx["families"]:
            md = upsert(md, "Extended Data Table 14 |", table14(ctx))
    open(md_path, "w", encoding="utf8").write(md)
    print("updated", md_path)


if __name__ == "__main__":
    main()
