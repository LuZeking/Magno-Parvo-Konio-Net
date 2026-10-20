"""Extended Data tables for PAPER_v2.md, generated from the committed result
files so that every number in the tables is traceable to an artefact.

Writes results/figures_v2/ed_tables.md; docs/PAPER_v2.md includes it between
the markers <!-- ED_TABLES_START --> and <!-- ED_TABLES_END --> (see
tmp_analysis/figs_v2/insert_ed_tables.py).

Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/ed_tables.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
from scipy import stats as sps

from src.analysis.encoding.stats import load_results

NC_MIN, MIN_VOX = 0.1, 20
VIS = [1, 2, 3, 4, 5, 6, 7]
HIER = np.array([1, 1, 2, 2, 3, 3, 4])
RUNS = "/scratch/byrong/VST_data/mpk_runs"
OUT = "results/figures_v2/ed_tables.md"
D = np.load("results/figures/fig1_data.npz", allow_pickle=True)


def profile(out_dir, model, ratio=True, drop=()):
    res = load_results(out_dir, model)
    subs = [s for s in sorted(res) if s not in drop]
    out = np.full((len(subs), 7), np.nan)
    for i, s in enumerate(subs):
        z = res[s]
        roi, nc, um, up = z["roi_prf-visualrois"], z["noise_ceiling"], z["unique_magno"], z["unique_parvo"]
        for j, lab in enumerate(VIS):
            v = (roi == lab) & (nc > NC_MIN) & ((um + up) > 1e-3)
            if v.sum() >= MIN_VOX:
                r = (um[v] - up[v]) / (um[v] + up[v]) if ratio else um[v] - up[v]
                out[i, j] = np.median(r[np.isfinite(r)])
    return out


def grad(p):
    return p[:, 6] - p[:, :2].mean(1)


def wil(a, b=None):
    d = a if b is None else a - b
    return sps.wilcoxon(d).pvalue


def f(x, nd=3, sign=True):
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "−")


L = []
P = L.append

# ------------------------------------------------------------ T1 gradient per seed and statistic
P("**Extended Data Table 1 | The cortical gradient hV4 − mean(V1v, V1d), per seed and statistic.** “normalised” is the per-voxel index (U_m − U_p)/(U_m + U_p), “raw” the per-voxel difference U_m − U_p, each summarised by its median over reliable voxels per ROI. Mean over 8 subjects of the per-subject contrast; “sep.” is the number of subjects in which the diet arm is more negative than the no-diet arm, and for the rows with no paired comparison the number in which the contrast is negative; p is the exact Wilcoxon, paired for the diet against no-diet rows and one-sample otherwise (floor .0078). The n = 7 column drops subject 01 (Results 2.3). The swap and parvo_highsf arms are degenerate by construction, in every seed: the swapped network leaves the parvo band a median unique variance 0.005–0.010 of the magno band's, and the parvo + high-SF network leaves the magno band 0.003–0.005 of the parvo band's, so their gradients describe one surviving band and are not comparable with the two-band arms. Sources: `results/encoding_cc/` (diet, no diet, thesis) and `results/encoding/` (untrained, swap, parvo_highsf).")
P("")
P("| statistic | seed | diet | no diet (n = 8) | no diet (n = 7) | sep. | p |")
P("|---|---|---|---|---|---|---|")
for ratio, name in [(True, "normalised"), (False, "raw")]:
    for sd in range(3):
        gd = grad(profile("results/encoding_cc", f"dkl_bcs_seed{sd}_trained", ratio))
        gn = grad(profile("results/encoding_cc", f"dkl_base_seed{sd}_trained", ratio))
        gn7 = grad(profile("results/encoding_cc", f"dkl_base_seed{sd}_trained", ratio, drop=(1,)))
        P(f"| {name if sd == 0 else ''} | {sd} | {f(gd.mean())} | {f(gn.mean())} | {f(gn7.mean())} | {(gd < gn).sum()}/8 | {wil(gd, gn):.4f} |")
gu = grad(profile("results/encoding", "untrained_seed0", True)); gun = grad(profile("results/encoding", "untrained_seed0_noblur", True))
P(f"| untrained, diet front end | — | {f(gu.mean())} | | | {(gu < 0).sum()}/8 < 0 | {wil(gu):.4f} |")
P(f"| untrained, no-diet front end | — | | {f(gun.mean())} | | {(gun < 0).sum()}/8 < 0 | {wil(gun):.4f} |")
for arm, lab, deg in [("dkl_bcs_swap", "swap", "U_p ≈ 0"), ("dkl_bcs_parvo_highsf", "parvo_highsf", "U_m ≈ 0")]:
    for sd in range(3):
        gc = grad(profile("results/encoding", f"{arm}_seed{sd}_trained", True))
        name = f"{lab} (degenerate: {deg})" if sd == 0 else ""
        P(f"| {name} | {sd} | {f(gc.mean())} | | | {(gc < 0).sum()}/8 < 0 | {wil(gc):.4f} |")
gt = grad(profile("results/encoding_cc", "thesis_dvd_trans", True))
P(f"| thesis checkpoint (progressive schedule) | — | {f(gt.mean())} | | | | |")
P("")

# ------------------------------------------------------------ T2 monotonicity and exact null
null = json.load(open("results/hierarchy_permutation_null.json"))
P("**Supplementary Table S1 | Monotonicity of the profile and the exact permutation null.** Spearman ρ between hierarchical rank [1,1,2,2,3,3,4] and stream preference, mean over subjects; the null enumerates all 630 distinct labellings of the seven ROIs (one-sided in the pre-stated direction, floor 1/630 = .0016). “per-voxel index” is the manuscript's statistic (Fig. 3c; Results 2.3): the median over reliable voxels of (U_m − U_p)/(U_m + U_p) per ROI, as in Extended Data Table 1; “ratio of medians” is median(U_m − U_p)/median(U_m + U_p) per ROI, the statistic of an earlier draft, kept for comparison. Source: `results/hierarchy_permutation_null.json` (which labels the two `raw` and `norm`).")
P("")
P("| arm | seed | statistic | observed mean ρ | exact p | most extreme of 630 | null 2.5th pct |")
P("|---|---|---|---|---|---|---|")
STAT_LABEL = {"raw": "per-voxel index", "norm": "ratio of medians"}   # see the caption; the source file's labels are misleading
for e in sorted(null, key=lambda e: (e["arm"] != "diet", e["model"], e["stat"] != "raw")):
    sd = re.search(r"seed(\d)", e["model"]).group(1)
    P(f"| {e['arm']} | {sd} | {STAT_LABEL[e['stat']]} | {e['observed_mean_rho']:+.3f} | {e['p_exact_one_sided']:.4f} | {'yes' if e['is_most_extreme'] else 'no'} | {e['null_q025']:+.3f} |")
P("")

# ------------------------------------------------------------ T3 in-silico cue conflict per seed
def cc_streams(model):
    mg, pv = [], []
    for fpath in sorted(glob.glob(f"results/encoding_cc/{model}_subj*.npz")):
        z = np.load(fpath)
        ok = z["noise_ceiling"] > NC_MIN
        med = lambda v: np.median(v[ok & np.isfinite(v)])
        mg.append(med(z["cc_magno_vsb"])); pv.append(med(z["cc_parvo_vsb"]))
    return np.array(mg), np.array(pv)

P("**Supplementary Table S2 | In-silico cue conflict through the fitted encoding models, per seed.** Median over reliable voxels of the per-voxel predicted shape-bias index for each stream, mean over 8 subjects; the contrast is magno − parvo, paired within subject against the no-diet arm of the same seed. Source: `results/encoding_cc/`.")
P("")
P("| seed | arm | magno stream | parvo stream | magno − parvo | diet − no diet | sep. | p |")
P("|---|---|---|---|---|---|---|---|")
for sd in range(3):
    md, pd_ = cc_streams(f"dkl_bcs_seed{sd}_trained"); mn, pn = cc_streams(f"dkl_base_seed{sd}_trained")
    dd, dn = md - pd_, mn - pn
    P(f"| {sd} | diet | {f(md.mean())} | {f(pd_.mean())} | {f(dd.mean())} | {f((dd - dn).mean())} | {(dd > dn).sum()}/8 | {wil(dd, dn):.4f} |")
    P(f"| {sd} | no diet | {f(mn.mean())} | {f(pn.mean())} | {f(dn.mean())} | | | |")
P("")

# ------------------------------------------------------------ T4 the 2x2
P("**Supplementary Table S3 | The carrier × transform 2×2 on the probe and on behaviour.** Mean over training seeds with the seed range in brackets; “decided” is the mean number of the 1,200 cue-conflict images the readout classifies as the shape or the texture category (the shape-bias denominator). Cells are read out on stream 2 in three cases and on the diet's magno stream in the fourth. Source: `results/figures/fig1_data.npz` (from `results/shape_texture/`, `results/cue_conflict/`). “probe bias, exemplars held out” is the exemplar-balanced probe of Methods 4.3, scored on object and texture exemplars the probe was not trained on (three exemplars per category for both labels, leave one exemplar out, PCA inside the folds, mean of 20 draws; `docs/PLAN_EXEMPLAR_2026-09-21.md`, `results/figures_v2/exemplar_streams.json`); “probe bias” is the original stratified-fold probe, whose test images share exemplars with its training folds.")
P("")
P("| carrier | transform | condition, stream | probe bias | probe bias, exemplars held out | cue-conflict shape bias | decided / 1,200 | seeds |")
P("|---|---|---|---|---|---|---|---|")
EX2 = json.load(open("results/figures_v2/exemplar_streams.json"))["two_by_two"]["Q1"]["cells"]
cells = [("chromatic (L−M)", "none", "no diet, parvo", "chromatic", "none"),
         ("chromatic (L−M)", "yes", "swap, parvo", "chromatic", "transform"),
         ("achromatic (luminance)", "none", "parvo_highsf, stream 2", "achromatic", "none"),
         ("achromatic (luminance)", "yes", "diet, magno", "achromatic", "transform")]
for carrier, tr, cond, ck, tk in cells:
    pr, be, de = D[f"a__{ck}__{tk}__probe"], D[f"a__{ck}__{tk}__beh"], D[f"a__{ck}__{tk}__decided"]
    q1 = np.array(EX2[f"{ck}_{tk}"])
    P(f"| {carrier} | {tr} | {cond} | {f(pr.mean())} [{np.ptp(pr):.3f}] | {f(q1.mean())} [{np.ptp(q1):.3f}] | {be.mean():.3f} [{np.ptp(be):.3f}] | {de.mean():.0f} | {len(pr)} |")
P("")

# ------------------------------------------------------------ T5 single-stream controls and ladder
def cc(pattern, cond):
    rows = []
    for fp in sorted(glob.glob(pattern)):
        if not fp.endswith(".json"):
            continue
        d = json.load(open(fp)); c = d["conditions"][cond]
        ep = re.search(r"_ep(\d+)", os.path.basename(fp)); seed = int(re.search(r"seed(\d)", fp).group(1))
        rows.append((seed, int(ep.group(1)) if ep else None, c["shape_bias"], c["n_shape"] + c["n_texture"]))
    return rows

def acc(run, ep):
    h = json.load(open(f"{RUNS}/{run}/history.json")); v = [e["val_top1"] for e in h]
    return v[ep - 1] if ep else v[-1]

params = {"two-stream (2 × ResNet-18)": 22397936, "single stream, ResNet-34 (matched)": 21338992, "single stream, ResNet-18 (half)": 11230832}
P("**Supplementary Table S4 | Single-stream controls trained with the identical diet, and the readout ladder.** Best saved checkpoint per seed, except the no-diet row, which is scored at the final checkpoint because its accuracy does not degrade (Fig. 2d; Methods 4.7); mean [range] over three seeds. Accuracy is validation top-1 at the scored checkpoint. Sources: `results/cue_conflict_bestep/`, `results/cue_conflict_single34_bestep/`, `results/cue_conflict_single_bestep/`, run histories, `results/param_matched/`.")
P("")
P("| model / readout | parameters | shape bias | decided / 1,200 | accuracy |")
P("|---|---|---|---|---|")
for lab, pat, cond, run, key in [
        ("diet, magno stream alone", "results/cue_conflict_bestep/dkl_bcs_seed?_ep*.json", "magno_only", "dkl_bcs_seed{}", "two-stream (2 × ResNet-18)"),
        ("diet, both streams (w = 0.5)", "results/cue_conflict_bestep/dkl_bcs_seed?_ep*.json", "full", "dkl_bcs_seed{}", "two-stream (2 × ResNet-18)"),
        ("diet, parvo stream alone", "results/cue_conflict_bestep/dkl_bcs_seed?_ep*.json", "parvo_only", "dkl_bcs_seed{}", "two-stream (2 × ResNet-18)"),
        ("single stream, ResNet-34 (matched)", "results/cue_conflict_single34_bestep/*.json", "full", "dkl_bcs_single34_seed{}", "single stream, ResNet-34 (matched)"),
        ("single stream, ResNet-18 (half)", "results/cue_conflict_single_bestep/*.json", "full", "dkl_bcs_single_seed{}", "single stream, ResNet-18 (half)"),
        ("no diet, both streams", "results/cue_conflict/dkl_base_seed?.json", "full", "dkl_base_seed{}", "two-stream (2 × ResNet-18)")]:
    rows = cc(pat, cond)
    sb = np.array([r[2] for r in rows]); de = np.array([r[3] for r in rows]); ac = np.array([acc(run.format(r[0]), r[1]) for r in rows])
    P(f"| {lab} | {params[key]:,} | {sb.mean():.3f} [{sb.min():.3f}, {sb.max():.3f}] | {de.mean():.0f} | {ac.mean():.3f} [{ac.min():.3f}, {ac.max():.3f}] |")
P("")

# ------------------------------------------------------------ T6 OOD
FAM = ["uniform-noise", "high-pass", "false-colour", "rotation", "phase-scrambling", "contrast", "low-pass"]
def auc(prefix):
    runs = [json.load(open(fp)) for fp in sorted(glob.glob(f"results/ood_bestep/{prefix}*.json"))]
    return {fam: np.array([(np.array(r["families"][fam]["accuracy"]) / r["families"][fam]["accuracy"][0]).mean() for r in runs]) for fam in FAM}
A = {k: auc(p) for k, p in [("diet", "dkl_bcs_seed"), ("no diet", "dkl_base_seed"), ("swap", "dkl_bcs_swap_seed"), ("parvo_highsf", "dkl_bcs_parvo_highsf_seed"), ("thesis schedule", "thesis_dvd_trans")]}
P("**Supplementary Table S5 | Out-of-distribution robustness: relative-accuracy AUC per corruption family and condition.** Mean over three seeds [range]; the thesis's schedule-trained checkpoint is a single run. “sep.” marks complete seed separation between the diet and no-diet arms (the exact 3 v 3 floor, p = 0.100). Families marked ‡ were declared trained-in before the run. Source: `results/ood_bestep/`.")
P("")
P("| family | diet | no diet | swap | parvo_highsf | thesis schedule | diet − no diet | sep. |")
P("|---|---|---|---|---|---|---|---|")
for fam in FAM:
    d, n = A["diet"][fam], A["no diet"][fam]
    sep = (d.max() < n.min()) or (d.min() > n.max())
    cell = lambda v: f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]" if len(v) > 1 else f"{v.mean():.3f}"
    P(f"| {fam}{' ‡' if fam in ('contrast', 'low-pass') else ''} | {cell(d)} | {cell(n)} | {cell(A['swap'][fam])} | {cell(A['parvo_highsf'][fam])} | {cell(A['thesis schedule'][fam])} | {f(d.mean() - n.mean())} | {'yes' if sep else 'no'} |")
P("")

# ------------------------------------------------------------ T7 error consistency
P("**Supplementary Table S6 | Error consistency (Cohen's κ) with the ten human observers.** The trained readout (w = 0.5), the magno stream alone (w = 1; the κ = 0.0082, 3.5% of Results 2.5 is the mean of the three final-checkpoint runs) and the best readout over the 21 values of w, per run; κ as a percentage of the human–human ceiling (κ = 0.234, 45 pairs). Published CNN values are from Geirhos et al. (2020). Source: `results/sweep_error_consistency/`.")
P("")
P("| run | shape bias at w = 0.5 | κ at w = 0.5 | % ceiling | κ at w = 1 (magno alone) | % ceiling | rate at w = 1 | best κ over w | % ceiling | w at best | rate at best |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for fp in sorted(glob.glob("results/sweep_error_consistency/dkl_*_kappa_sweep.json")):   # BA runs; fs2_* sweeps live beside them
    r = json.load(open(fp)); ceil = r["human_human_ceiling"]
    sw = sorted(r["sweep"], key=lambda q: q["w"]); mid = [q for q in sw if abs(q["w"] - 0.5) < 1e-6][0]
    best = max(sw, key=lambda q: q["kappa_mean"]); top = [q for q in sw if abs(q["w"] - 1.0) < 1e-6][0]
    P(f"| {r['model_name']} | {mid['shape_bias']:.3f} | {mid['kappa_mean']:+.4f} | {mid['kappa_pct_ceiling']:.1f} | {top['kappa_mean']:+.4f} | {top['kappa_pct_ceiling']:.1f} | {top['shape_bias']:.3f} | {best['kappa_mean']:+.4f} | {best['kappa_pct_ceiling']:.1f} | {best['w']:.2f} | {best['shape_bias']:.3f} |")
for name, sb, k in [("AlexNet (published)", 0.429, 0.032), ("GoogLeNet (published)", 0.312, 0.025), ("VGG-16 (published)", 0.172, 0.017)]:
    P(f"| {name} | {sb:.3f} | {k:+.4f} | {100 * k / 0.2335:.1f} | | | | | | | |")
P("")

# ------------------------------------------------------------ T8 estimator floor and residual variants (stored analyses)
P("**Supplementary Table S7 | The estimator's own bias, and the residual-misfit variants.** Synthetic voxels were generated from each arm's fitted 1,000-component feature spaces with the measured per-ROI noise ceilings imposed and the true magno − parvo difference set to zero everywhere; the recovered contrast is the instrument's signed floor. “Displacement” is the real contrast minus the floor, paired per subject. The six variants differ in how the floor is paired (group or per-subject) and whether the residual misfit of the recovery simulation is also removed (before or after rescaling by the recovered slope). Sources: `results/vp_recovery_realnc/diet_floor_paired_2026-09-07.txt`, `floor_pairing_2026-09-07.txt`, `prediction_scored_2026-09-07.txt`.")
P("")
P("| quantity | diet | no diet |")
P("|---|---|---|")
P("| floor, normalised contrast (true = 0) | +0.0375 (8/8 > 0, p = .0078) | +0.0511 (8/8 > 0, p = .0078) |")
P("| floor, raw contrast | −0.0021 (p = .15) | +0.0009 (p = .84) |")
P("| real contrast, normalised (seed 0) | −0.8335 | −0.0785 |")
P("| displacement real − floor | −0.871 (8/8, p = .0078) | −0.130 (8/8, p = .0078); n = 7: −0.103 (7/7, p = .0156) |")
P("| recovered slope of estimate vs truth | 1.210 | 1.281 |")
P("| variant: group floor, minimal | −0.720 (8/8, p = .0078) | −0.101 (8/8, p = .0078) |")
P("| variant: per-subject floor, minimal | −0.720 (8/8, p = .0078) | −0.101 (8/8, p = .0078) |")
P("| variant: group floor + residual before scaling | −0.626 (8/8, p = .0078) | −0.034 (6/8, p = .055) |")
P("| variant: per-subject floor + residual before scaling | −0.626 (8/8, p = .0078) | −0.034 (5/8, p = .31) |")
P("| variant: group floor + residual after scaling | −0.607 (8/8, p = .0078) | −0.015 (4/8, p = 1.00) |")
P("| variant: per-subject floor + residual after scaling | −0.607 (8/8, p = .0078) | −0.015 (5/8, p = .84) |")
P("| per-ROI normalised floor, V1v … hV4 (no-diet geometry) | | +0.025, +0.025, +0.026, +0.025, +0.027, +0.026, +0.026 (ρ vs rank +0.54, p = .22) |")
P("")

# ------------------------------------------------------------ T9 pRF regressions (stored analyses)
P("**Supplementary Table S8 | Stream preference regressed on pRF eccentricity and pRF size within subject.** Per-voxel stream-preference index regressed on log pRF eccentricity and log pRF size (NSD pRF estimates, voxels with pRF R² > 5%, noise ceiling > 0.1); “ecc | size” is the partial slope for eccentricity controlling for size and vice versa; mean over 8 subjects, with the number of subjects on the stated side and the exact Wilcoxon p. Collinearity between the two predictors is modest (mean VIF 1.66). The arm interaction is the paired difference of partial slopes, seed-averaged within arm. Sources: `results/prf_continuous/dissociation_2026-09-08.txt`, `interaction_2026-09-08.txt`, `input_null_2026-09-08.txt`. The paired trained − untrained rows compare each subject's partial slope in the trained seed-0 network with the untrained network on the diet front end; training with the diet changes neither statistic significantly and the two disagree in sign, while training without the diet removes the dissociation on the raw statistic.")
P("")
P("| model | statistic | ecc alone | size alone | ecc \\| size | size \\| ecc |")
P("|---|---|---|---|---|---|")
P("| diet, seed 0 | normalised | +0.066 (0/8 neg, .0078) | −0.013 (5/8 neg, .31) | +0.119 (0/8 neg, .0078) | −0.086 (8/8 neg, .0078) |")
P("| diet, seed 1 | normalised | +0.065 (0/8, .0078) | −0.029 (7/8, .039) | +0.132 (0/8, .0078) | −0.110 (8/8, .0078) |")
P("| diet, seed 2 | normalised | +0.064 (0/8, .0078) | −0.012 (5/8, .25) | +0.115 (0/8, .0078) | −0.083 (8/8, .0078) |")
P("| no diet, seed 0 | normalised | +0.018 (4/8, .64) | +0.033 (1/8, .039) | −0.004 (4/8, 1.00) | +0.035 (0/8, .0078) |")
P("| no diet, seed 1 | normalised | +0.004 (4/8, .95) | +0.015 (4/8, .38) | −0.009 (6/8, .46) | +0.021 (2/8, .11) |")
P("| no diet, seed 2 | normalised | −0.007 (5/8, .38) | +0.012 (3/8, .31) | −0.022 (6/8, .25) | +0.024 (1/8, .016) |")
P("| diet, seed 0 | raw | +0.103 (1/8, .039) | −0.158 (8/8, .0078) | +0.325 (0/8, .0078) | −0.356 (8/8, .0078) |")
P("| no diet, seed 0 | raw | +0.130 (1/8, .023) | +0.181 (0/8, .0078) | +0.028 (3/8, .84) | +0.163 (0/8, .0078) |")
P("| Gabor bank, diet front end (input null) | normalised | | | +0.035 (6/8 pos, .15) | +0.002 (4/8 neg, .95) |")
P("| untrained network, diet front end | normalised | | | +0.083 (7/8 pos, .016) | −0.038 (6/8 neg, .039) |")
P("| untrained network, undieted front end | normalised | | | +0.089 (7/8 pos, .023) | −0.038 (8/8 neg, .0078) |")
P("| trained diet − untrained (diet front end), paired | normalised | | | +0.036 (6/8 pos, .11) | |")
P("| trained diet − untrained (diet front end), paired | raw | | | −0.077 (2/8 pos, .11) | |")
P("| trained no diet − untrained (diet front end), paired | normalised | | | −0.087 (2/8 pos, .15) | |")
P("| trained no diet − untrained (diet front end), paired | raw | | | −0.374 (0/8 pos, .0078) | |")
P("| arm interaction, diet − no diet (seed-averaged) | normalised | | | +0.134 (8/8, .0078) | −0.119 (0/8 pos, .0078) |")
P("| arm interaction, diet − no diet (seed-averaged) | raw | | | +0.305 (8/8, .0078) | −0.502 (0/8 pos, .0078) |")
P("")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
# Upsert, never rewrite: other generators own Tables 10-16 in the same file
# (forced_choice.py, ed_table11.py, readout_w_cortex.py, ed_tables_generality.py, ed_table16_zoo.py).
generated = "\n".join(L).strip() + "\n"
blocks = [b for b in re.split(r"(?=\n\*\*Extended Data Table \d+ \|)", "\n" + generated) if b.strip()]
existing = open(OUT, encoding="utf8").read() if os.path.exists(OUT) else ""
for block in blocks:
    block = block.strip("\n")
    title = re.match(r"\*\*Extended Data Table \d+ \|", block).group(0)
    pat = re.compile(r"\*\*" + re.escape(title[2:]) + r".*?(?=\n\*\*Extended Data Table|\Z)", re.S)
    if pat.search(existing):
        existing = pat.sub(lambda m: block + "\n", existing, count=1)
    else:
        existing = existing.rstrip("\n") + "\n\n" + block + "\n"
open(OUT, "w", encoding="utf8").write(existing)
print(f"upserted {len(blocks)} tables into {OUT} ({existing.count(chr(10))} lines)")
