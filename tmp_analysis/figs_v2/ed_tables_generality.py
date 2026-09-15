"""Extended Data Tables 13 (subspace behaviour), 14 (unit-split behaviour) and 15 (cortex) for the generality test.

Reads only the committed results/figures_v2/{subspace_behaviour,generality_behaviour,generality_cortex,subspace_cortex}.json;
replaces or inserts the three tables in results/figures_v2/ed_tables.md (then run insert_ed_tables.py).
Usage: python tmp_analysis/figs_v2/ed_tables_generality.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

ORDER = ["concat", "single34", "single18", "rgb_tsn", "vit_b_16", "resnet152", "convnext_base", "convnext_small", "resnet50", "vgg16"]
LABEL = {"concat": "two-stream, diet, streams concatenated (validation)", "single34": "single ResNet-34, diet",
         "single18": "single ResNet-18, diet", "rgb_tsn": "single ResNet-18, no diet (`rgb_tsn`)",
         "vit_b_16": "ImageNet ViT-B/16", "resnet152": "ImageNet ResNet-152", "convnext_base": "ImageNet ConvNeXt-B",
         "convnext_small": "ImageNet ConvNeXt-S", "resnet50": "ImageNet ResNet-50", "vgg16": "ImageNet VGG-16"}
REGIONS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "dorsal"]
FAM_LABEL = {"two_stream_diet": "two-stream, diet (magno − parvo)", "single34": "single ResNet-34, diet",
             "rgb_tsn": "single ResNet-18, no diet", "resnet50": "ImageNet ResNet-50",
             "vit_b_16": "ImageNet ViT-B/16", "convnext_base": "ImageNet ConvNeXt-B"}


def fmt(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def table13(sub):
    """Subspace rule (PLAN_SUBSPACE D1-D3)."""
    fam = {f["key"]: f for f in sub["families"]}
    rows = ["| network | D | trained rate | w = 0 | w = 1 | span (held-out) [range] | ρ(w, rate) | gated span (w-range; largest change of top-1) | forced-estimator span | refit span linear / two-layer (largest shortfall of top-1) | variance in S / T | order sensitivity, S first / T first | median-split gated span | seeds |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in ORDER:
        if key not in fam:
            continue
        f = fam[key]
        seeds = f["seeds"]
        acc = [s["gain_accuracy"] for s in seeds if s.get("gain_accuracy")]
        if acc and f.get("gated_span") is not None:
            dtop = max(max(abs(np.array(a["top1"]) - a["trained_top1"])) for a in acc) * 100
            gate_s = f"{f['gated_span']:.2f} ({f['gate'][0]:.2f}–{f['gate'][1]:.2f}; {dtop:.1f} pts)"
        elif key == "concat":
            e_full = [s["descriptives"].get("energy_S_on_stream1_full") for s in seeds]
            e_held = [s["descriptives"].get("energy_S_on_stream1_heldout") for s in seeds]
            if all(v is not None for v in e_full):
                gate_s = f"— (shape basis on stream 1: {min(e_full):.2f}–{max(e_full):.2f}"
                if all(v is not None for v in e_held):
                    gate_s += f"; held-out bases {min(e_held):.2f}–{max(e_held):.2f}"
                gate_s += ")"
            else:
                gate_s = "—"
        else:
            gate_s = "—"
        ref = f.get("refit")
        if ref:
            short = max(-min(ref["linear"]["min_val_minus_ceiling"], ref["mlp"]["min_val_minus_ceiling"]), 0) * 100
            ref_s = f"{ref['linear']['span']:.2f} / {ref['mlp']['span']:.2f} ({short:.1f} pts)"
        else:
            ref_s = "—"
        vS = np.mean([s["descriptives"]["var_frac_S"] for s in seeds]); vT = np.mean([s["descriptives"]["var_frac_T"] for s in seeds])
        sf = np.mean([s["sensitivity"]["S_first"]["span"] for s in seeds]); tf = np.mean([s["sensitivity"]["T_first"]["span"] for s in seeds])
        unit = f.get("unit_gated_span")
        rows.append(f"| {LABEL[key]} | {seeds[0]['n_units']} | {f['trained']:.3f} | {f['rate_w0']:.3f} | {f['rate_w1']:.3f} | "
                    f"{f['span']:.3f} [{f['span_range'][0]:.2f}, {f['span_range'][1]:.2f}] | {f['rho']:.2f} | {gate_s} | {f['forced_span']:.3f} | "
                    f"{ref_s} | {vS:.2f} / {vT:.2f} | {sf:.2f} / {tf:.2f} | {fmt(unit)} | {len(seeds)} |")
    cap = ("**Extended Data Table 13 | The readout span without a stream split: one gain on a network's own shape- and texture-predictive subspaces, head fixed.** "
           "For each frozen network with logits = W·y + b, the 15 discriminant directions of a 16-class shape LDA and of a 16-class texture LDA on the penultimate features y (PCA to 256 dimensions, Ledoit–Wolf shrinkage; fitted on one stratified half of the 1,200 cue-conflict images) are made mutually orthogonal by symmetric (Löwdin) orthogonalisation, giving orthonormal bases B_S and B_T (15 directions each) and the residual B_R. "
           "logits(w) = W·(P_R y + 2w·P_S y + 2(1 − w)·P_T y) + b with the network's own head untouched; w = 0.5 is the trained network. "
           "The rate is scored on the other half of the images (10 random splits, both directions; mean, with the range over splits and seeds in brackets), by the paper's primary estimator (112-class, decided images) for our networks and the Geirhos 16-class probability aggregation for the ImageNet networks; “forced-estimator span” is the same sweep on the forced two-way estimator. "
           "“gated span” is max − min of the seed-mean held-out curve over the w at which top-1 of the gained features (bases fitted on all 1,200 images) on held-out natural images (MiniEcoset validation for our networks; a stratified 10-per-class ILSVRC-2012 validation subset for the ImageNet networks) stays within 5 points of the trained value, followed by that w-range and the largest change of top-1 anywhere in the sweep. "
           "“refit span” is the block-form lossy sum z(w) = [w·s + (1 − w)·t, B_Rᵀy], s = B_Sᵀy and t = B_Tᵀy paired by discriminant rank, with a linear and a two-layer head refitted at each w (Methods 4.3 budget), our networks only; in parentheses the largest shortfall of validation accuracy, over w, from a head refitted on the unmixed features. "
           "“variance in S / T” is the fraction of the feature variance on the fitting half lying in each subspace; “order sensitivity” replaces the symmetric orthogonalisation by sequential Gram–Schmidt with the shape basis or the texture basis first. "
           "“median-split gated span” is the gated span of the pre-specified unit-level rule of Extended Data Table 14 on the same networks and images. "
           "For the two-stream concatenation, which has no held-out natural-image features, the fraction of the shape basis's energy lying on stream-1 units (all-image bases; held-out bases in parentheses; range over seeds) replaces the gate. "
           "Pre-specification: `docs/PLAN_SUBSPACE_2026-09-15.md` (predictions P-A, P-A′, P-B, P-D). Source: `results/figures_v2/subspace_behaviour.json`, aggregated by `tmp_analysis/figs_v2/subspace_behaviour.py` from the outputs of `src/analysis/generality/subspace_factor.py`, `gain_accuracy.py` and `refit_sum.py` (`slurm/subspace_{cc,refit}.sh`); the per-network files are outside the repository.")
    return cap + "\n\n" + "\n".join(rows)


def table14(beh):
    """Unit-split rule (PLAN_GENERALITY D1-D3), formerly Extended Data Table 13."""
    fam = {f["key"]: f for f in beh["families"]}
    rows = ["| network | units | trained rate | w = 0 | w = 1 | span (held-out) [range] | ρ(w, rate) | forced-estimator span | shape-leaning units | w within 5 pts of trained top-1 | refit span linear / two-layer | span, S = {d > 0} | span, top decile | seeds |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in ORDER:
        if key not in fam:
            continue
        f = fam[key]
        acc = [(min(r["gain_accuracy"]["w_within_5pts"]), max(r["gain_accuracy"]["w_within_5pts"]), r["gain_accuracy"]["trained_top1"])
               for r in f["seeds"] if r.get("gain_accuracy")]

        def rule_s(rule):
            e = f.get("rules", {}).get(rule)
            if not e:
                return "—"
            if e.get("undefined"):
                return "undefined (empty half)"
            out = f"{e['span']:.2f} [{e['span_min']:.2f}, {e['span_max']:.2f}] (ρ ≥ {e['rho_min']:.2f}"
            if e.get("n_S") is not None:
                out += f"; |S| = {e['n_S']:.0f}"
            if "w5_lo" in e:
                out += f"; gate {e['w5_lo']:.2f}–{e['w5_hi']:.2f}"
            return out + ")"
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
                    + f" | {acc_s} | {ref_s} | {rule_s('positive')} | {rule_s('decile')} | {len(f['seeds'])} |")
    cap = ("**Extended Data Table 14 | The readout span without a stream split under the pre-specified unit-level rule: fixed-head gain sweep on the shape- and texture-leaning halves of the units.** "
           "The special case of Extended Data Table 13 in which the two subspaces are spanned by disjoint sets of units. Each unit of a network's penultimate layer is scored by the fraction of its variance between shape categories minus that between texture categories on the 1,200 cue-conflict images (one-way η²); the median split gives a shape-leaning half S and a texture-leaning half T. "
           "logits(w) = head(g(w) ⊙ y) with g = 2w on S and 2(1 − w) on T, the network's own head untouched; w = 0.5 is the trained network. "
           "The split is defined on one stratified half of the images and the rate scored on the other (10 random splits, both directions, the same splits as Extended Data Table 13; mean, with the range over splits and seeds in brackets). "
           "Rates by the paper's primary estimator (112-class, decided images) for our networks and the Geirhos 16-class probability aggregation for the ImageNet networks; “forced-estimator span” is the same sweep on the forced two-way estimator. "
           "“shape-leaning units” is the fraction of units with η²_shape > η²_texture on all images; for the two-stream concatenation the fraction of S that belongs to stream 1 is given in parentheses (P3). "
           "“w within 5 pts” is the w-range over which top-1 on held-out natural images (MiniEcoset validation; a stratified 10-per-class ILSVRC-2012 validation subset for the ImageNet networks) stays within 5 points of the trained value. "
           "“refit span” is the lossy-sum protocol z(w) = w·y_S + (1 − w)·y_T with a linear and a two-layer head refitted at every w (Methods 4.3 budget), our networks only. "
           "The last two columns are a sensitivity analysis added after the pre-specified analysis, at a referee's request: the same held-out sweep with S defined as the absolute shape-leaning set (units with d_j > 0) or as the top decile of units by d_j, and T as the remainder; given are the held-out span with its range over splits and seeds, the minimum Spearman ρ, the mean size of S and the w-range within 5 points of the trained top-1 (no gate exists for the two-stream concatenation, which has no held-out natural-image features). "
           "Pre-specification: `docs/PLAN_GENERALITY_2026-09-14.md`. Source: `results/figures_v2/generality_behaviour.json`, aggregated by `tmp_analysis/figs_v2/generality_behaviour.py` from the per-network outputs of `src/analysis/generality/` (`slurm/generality_{cc,mini,inval,rules}.sh`); the per-network files are outside the repository.")
    return cap + "\n\n" + "\n".join(rows)


def cortex_rows(ctx, fams, suffix=""):
    rows = []
    for fam in fams:
        if fam not in ctx["families"]:
            continue
        fo = ctx["families"][fam]
        st = fo.get("stats", {})

        def stat_s(k):
            v = st.get(k)
            return f"{v['mean']:+.3f} ({v['n_neg']}/{v['n']}, {fmt(v['p'], 4)})" if v else "—"
        rho = st.get("rho_rank_V1_hV4", {}).get("mean")
        rows.append(f"| {FAM_LABEL[fam]}{suffix} | index | " + " | ".join(f"{np.nanmean(fo['index'][r]):+.2f}" for r in REGIONS)
                    + f" | {stat_s('hV4_minus_V1')} | {stat_s('extrastriate_minus_V1')} | {fmt(rho, 2, True)} |")
        rows.append(f"| | U_shape | " + " | ".join(f"{np.nanmean(fo['uS'][r]):.3f}" for r in REGIONS) + " | | | |")
        rows.append(f"| | U_texture | " + " | ".join(f"{np.nanmean(fo['uT'][r]):.3f}" for r in REGIONS) + " | | | |")
        if fo.get("shared"):
            rows.append(f"| | shared | " + " | ".join(f"{np.nanmean(fo['shared'][r]):.3f}" for r in REGIONS) + " | | | |")
            rows.append(f"| | r² (full model, nc-normalised) | " + " | ".join(f"{np.nanmean(fo['r2'][r]):.3f}" for r in REGIONS) + " | | | |")
        if fo.get("w_star"):
            wm = st.get("wstar_median", {})
            wst = st.get("wstar_extrastriate_minus_V1")
            rows.append(f"| | w*_cortex (median) | " + " | ".join(fmt(wm.get(r), 2) for r in REGIONS)
                        + (f" | | Δw* extrastriate − V1 {wst['mean']:+.2f} ({wst['n_neg']}/{wst['n']} neg, {fmt(wst['p'], 4)}) | |" if wst else " | | | |"))
    return rows


def table15(ctx, sub_ctx=None):
    """Cortex: unit-split bands (PLAN_GENERALITY D4) and, when available, the subspace bands (PLAN_SUBSPACE D4)."""
    rows = ["| network | statistic | " + " | ".join(REGIONS) + " | hV4 − V1 (n neg, p) | extrastriate − V1 (n neg, p) | ρ(rank, index) V1–hV4 |",
            "|---|---|" + "---|" * len(REGIONS) + "---|---|---|"]
    if sub_ctx and sub_ctx.get("families"):
        rows.append("| **subspace bands S, T, R** | | " + " | " * len(REGIONS) + " | | |")
        rows += cortex_rows(sub_ctx, ["rgb_tsn", "single34", "resnet50", "vit_b_16", "convnext_base"])
        rows.append("| **unit halves S, T** | | " + " | " * len(REGIONS) + " | | |")
    rows += cortex_rows(ctx, ["two_stream_diet", "single34", "rgb_tsn", "resnet50"])
    sub_txt = ""
    if sub_ctx and sub_ctx.get("families"):
        sub_txt = ("Upper block (subspace bands): the three bands are the shape-predictive subspace S = Y·B_S (15 dimensions), the texture-predictive subspace T = Y·B_T (15) and the residual R = Y·B_R of Extended Data Table 13 (bases fitted on all 1,200 images), with U_S = r²_{STR} − r²_{TR} and U_T = r²_{STR} − r²_{SR} from the three-band model and the two two-band models (random search, 30 and 20 samples); “shared” is r²_{STR} − U_S − U_T and “r²” the three-band model's noise-ceiling-normalised r². "
                   "w*_cortex there is the weight of the 15-dimensional lossy sum z(w) = w·s + (1 − w)·t alone. "
                   "Lower block (unit halves): ")
    cap = ("**Extended Data Table 15 | Cortex without a stream split: the shape- and texture-predictive components as encoding bands.** "
           + sub_txt +
           "for each single-trunk network the penultimate units were split as in Extended Data Table 14 (all 1,200 images) and the two halves entered as two bands of a banded ridge model of every `nsdgeneral` voxel (same split, folds and alpha grid as Methods 4.5); "
           "U_shape and U_texture are the noise-ceiling-normalised unique variances and the index is (U_shape − U_texture)/(U_shape + U_texture), median over reliable voxels (noise ceiling > 0.1, ≥ 20 voxels), mean over eight subjects (seeds averaged per subject). "
           "Regions: V1–hV4 from `prf-visualrois`; ventral (VO1/2, PHC1/2), lateral (LO1/2, TO1/2) and dorsal (V3A/B, IPS0–2) from Kastner2015. "
           "The two-stream diet row applies the same computation to the magno (shape-biased) and parvo (texture-biased) bands of Fig. 3. "
           "“shared” is the cross-validated variance the bands account for jointly (the full model's r² minus the unique variances) and “r²” the full model's noise-ceiling-normalised r², medians over the same voxels. "
           "w*_cortex is the mixing weight of the lossy sum z(w) = w·y_S + (1 − w)·y_T that maximises an ROI's median test R² (Extended Data Fig. 5 protocol). Exact paired Wilcoxon over subjects. "
           "Source: `results/figures_v2/generality_cortex.json`" + (" and `results/figures_v2/subspace_cortex.json`" if sub_txt else "") +
           ", aggregated by `tmp_analysis/figs_v2/generality_cortex.py` from the per-subject fits of `src/analysis/generality/cortex_bands.py` (`slurm/generality_{nsd,cortex}.sh`" + (", `slurm/subspace_{nsd,cortex}.sh`" if sub_txt else "") +
           "), which are deposited alongside the encoding outputs of Fig. 3 (Code availability).")
    return cap + "\n\n" + "\n".join(rows)


def upsert(md, title_prefix, block):
    """Replace the table whose caption starts with title_prefix (up to the next table or EOF), else append."""
    pat = re.compile(r"\*\*" + re.escape(title_prefix) + r".*?(?=\n\*\*Extended Data Table|\Z)", re.S)
    if pat.search(md):
        return pat.sub(lambda m: block + "\n", md, count=1)
    return md.rstrip("\n") + "\n\n" + block + "\n"


def main():
    beh = json.load(open("results/figures_v2/generality_behaviour.json"))
    sub = json.load(open("results/figures_v2/subspace_behaviour.json"))
    md_path = "results/figures_v2/ed_tables.md"
    md = open(md_path, encoding="utf8").read()
    # one-time renumbering of the two older tables (13 -> 14 unit split, 14 -> 15 cortex): drop them and re-emit
    for old in ("Extended Data Table 14 | Cortex without a stream split: the shape- and texture-leaning halves",
                "Extended Data Table 13 | The readout span without a stream split: fixed-head gain sweep on ten frozen"):
        md = re.sub(r"\*\*" + re.escape(old) + r".*?(?=\n\*\*Extended Data Table|\Z)", "", md, flags=re.S).rstrip("\n") + "\n"
    md = upsert(md, "Extended Data Table 13 |", table13(sub))
    md = upsert(md, "Extended Data Table 14 |", table14(beh))
    if os.path.exists("results/figures_v2/generality_cortex.json"):
        ctx = json.load(open("results/figures_v2/generality_cortex.json"))
        sub_ctx = json.load(open("results/figures_v2/subspace_cortex.json")) if os.path.exists("results/figures_v2/subspace_cortex.json") else None
        if ctx["families"]:
            md = upsert(md, "Extended Data Table 15 |", table15(ctx, sub_ctx))
    open(md_path, "w", encoding="utf8").write(md)
    print("updated", md_path)


if __name__ == "__main__":
    main()
