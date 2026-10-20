"""Extend the stream-preference gradient past hV4.

The manuscript's cortical claim (Results 2.3) is read on seven early ROIs
(prf-visualrois: V1v..hV4). The stored banded-ridge fits cover every voxel of
nsdgeneral, so the same per-voxel index (Um - Up)/(Um + Up), median over
reliable voxels, can be read on the other NSD atlases without refitting:

  * Kastner2015 (Wang et al. 2015)  : V1..hV4, VO1/2, PHC1/2, LO1/2, TO1/2,
                                     V3A/B, IPS0-5 -- ventral, lateral, dorsal
  * streams (NSD)                    : early, mid-{ventral,lateral,parietal},
                                     ventral, lateral, parietal
  * fLoc                             : faces, places, bodies, words

Conventions are those of fig3_cortex.py / visualrois.py: nc > 0.1,
Um + Up > 1e-3, >= 20 voxels per ROI per subject; seeds are averaged at the
subject level; paired tests are exact Wilcoxon over the eight subjects.

Writes results/figures_v2/ext_gradient.{json,md} (--set v2, the BA arms) or
results/figures_v3/ext_gradient.{json,md} (--set v3: the model of record dvd300, its
control, the untrained network on their test input and the BA arms).
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/ext_gradient.py [--set v3]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats as sps

from src.analysis.encoding import nsd_data
from src.analysis.encoding.stats import load_results

NC_MIN, MIN_VOX = 0.1, 20

KASTNER = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4",
           8: "VO1", 9: "VO2", 10: "PHC1", 11: "PHC2", 12: "TO2", 13: "TO1",
           14: "LO2", 15: "LO1", 16: "V3B", 17: "V3A", 18: "IPS0", 19: "IPS1",
           20: "IPS2", 21: "IPS3", 22: "IPS4", 23: "IPS5", 24: "SPL1", 25: "FEF"}
STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal",
           5: "ventral", 6: "lateral", 7: "parietal"}
FLOC = {"floc-faces": {1: "OFA", 2: "FFA-1", 3: "FFA-2", 4: "mTL-faces", 5: "aTL-faces"},
        "floc-places": {1: "OPA", 2: "PPA", 3: "RSC"},
        "floc-bodies": {1: "EBA", 2: "FBA-1", 3: "FBA-2", 4: "mTL-bodies"},
        "floc-words": {1: "OWFA", 2: "VWFA-1", 3: "VWFA-2", 4: "mfs-words", 5: "mTL-words"}}

ARMS = {
    "untrained": ("results/encoding", ["untrained_seed0"]),
    "no diet": ("results/encoding_cc", [f"dkl_base_seed{s}_trained" for s in range(3)]),
    "diet": ("results/encoding_cc", [f"dkl_bcs_seed{s}_trained" for s in range(3)]),
    "swap": ("results/encoding", [f"dkl_bcs_swap_seed{s}_trained" for s in range(3)]),
    "parvo_highsf": ("results/encoding", [f"dkl_bcs_parvo_highsf_seed{s}_trained" for s in range(3)]),
    "thesis": ("results/encoding_cc", ["thesis_dvd_trans"]),
}
# PAPER_v3: the BA fits live in the mpk-rewrite worktree, the fs2 fits on /scratch
BA_ENC = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-rewrite/results/encoding_cc"
FS_ENC = "/scratch/byrong/mpk/encoding/results/2026-10-16_fs2"
ARMS_V3 = {
    "untrained": (FS_ENC, ["untrained_dvd300_seed0"]),
    "BA no diet": (BA_ENC, [f"dkl_base_seed{s}_trained" for s in range(3)]),
    "control": (FS_ENC, [f"fs2_dvdfix300_avg_w050_ls_seed{s}" for s in range(3)]),   # seeds with fits
    "BA": (BA_ENC, [f"dkl_bcs_seed{s}_trained" for s in range(3)]),
    "model": (FS_ENC, [f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)]),
}
SETS = {"v2": (ARMS, ("diet", "no diet"), "results/figures_v2"),
        "v3": (ARMS_V3, ("model", "control"), "results/figures_v3")}

# regional groupings for the contrasts
VENTRAL = ["VO1", "VO2", "PHC1", "PHC2"]
LATERAL = ["LO1", "LO2", "TO1", "TO2"]
DORSAL = ["V3A", "V3B", "IPS0", "IPS1", "IPS2", "IPS3"]
EARLY = ["V1v", "V1d"]

_label_cache: dict[tuple[int, str], np.ndarray] = {}


def labels(subject: int, atlas: str, n_vox: int) -> np.ndarray:
    key = (subject, atlas)
    if key not in _label_cache:
        mask = nsd_data.load_mask(subject, "nsdgeneral")
        if int(mask.sum()) != n_vox:
            raise RuntimeError(f"subj{subject}: mask {mask.sum()} != stored {n_vox}")
        _label_cache[key] = nsd_data.load_roi_labels(subject, atlas, mask)
    return _label_cache[key]


def roi_index(z, roi, lab):
    um, up, nc = z["unique_magno"], z["unique_parvo"], z["noise_ceiling"]
    v = (roi == lab) & (nc > NC_MIN) & ((um + up) > 1e-3)
    if v.sum() < MIN_VOX:
        return np.nan, int(v.sum()), np.nan
    r = (um[v] - up[v]) / (um[v] + up[v])
    r2 = z["r2_joint"][v]
    return float(np.median(r[np.isfinite(r)])), int(v.sum()), float(np.median(r2))


def fitted(out_dir, models):
    """The models with stored fits (a control seed still training has none); refuses an arm with none."""
    have = [m for m in models if glob.glob(os.path.join(out_dir, f"{m}_subj*.npz"))]
    if not have:
        raise SystemExit(f"no encoding fits for {models} in {out_dir}")
    return have


def profile(arm, arms=ARMS):
    out_dir, models = arms[arm]
    per_model = []
    for m in fitted(out_dir, models):
        res = load_results(out_dir, m)
        rows = {}
        for s in sorted(res):
            z = res[s]
            n = len(z["unique_magno"])
            atl = {"Kastner2015": (labels(s, "Kastner2015", n), KASTNER),
                   "streams": (z["roi_streams"], STREAMS)}
            for a, names in FLOC.items():
                atl[a] = (labels(s, a, n), names)
            for a, (roi, names) in atl.items():
                for lab, name in names.items():
                    idx, nv, r2 = roi_index(z, roi, lab)
                    rows[(s, a, name)] = (idx, nv, r2)
        per_model.append(rows)
    # seed-average at the subject level
    keys = sorted(set().union(*[r.keys() for r in per_model]))
    avg = {}
    for k in keys:
        vals = [r[k][0] for r in per_model if k in r and np.isfinite(r[k][0])]
        nvs = [r[k][1] for r in per_model if k in r]
        r2s = [r[k][2] for r in per_model if k in r and np.isfinite(r[k][2])]
        avg[k] = (float(np.mean(vals)) if vals else np.nan, int(np.mean(nvs)) if nvs else 0,
                  float(np.mean(r2s)) if r2s else np.nan, len(vals))
    return avg


def table(avg, atlas, names):
    subs = sorted({k[0] for k in avg})
    out = {}
    for name in names.values():
        v = np.array([avg.get((s, atlas, name), (np.nan,))[0] for s in subs])
        nv = np.array([avg.get((s, atlas, name), (np.nan, 0))[1] for s in subs])
        out[name] = dict(per_subject=v.tolist(), mean=float(np.nanmean(v)) if np.isfinite(v).any() else None,
                         n_subjects=int(np.isfinite(v).sum()), median_vox=float(np.median(nv)))
    return out


def group_mean(avg, s, names):
    v = [avg.get((s, "Kastner2015", n), (np.nan,))[0] for n in names]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def contrasts(avg):
    subs = sorted({k[0] for k in avg})
    c = {}
    c["hV4 - early"] = [avg.get((s, "Kastner2015", "hV4"), (np.nan,))[0] - group_mean(avg, s, EARLY) for s in subs]
    c["ventral(VO,PHC) - early"] = [group_mean(avg, s, VENTRAL) - group_mean(avg, s, EARLY) for s in subs]
    c["lateral(LO,TO) - early"] = [group_mean(avg, s, LATERAL) - group_mean(avg, s, EARLY) for s in subs]
    c["dorsal(V3AB,IPS0-3) - early"] = [group_mean(avg, s, DORSAL) - group_mean(avg, s, EARLY) for s in subs]
    c["dorsal - ventral"] = [group_mean(avg, s, DORSAL) - group_mean(avg, s, VENTRAL) for s in subs]
    c["lateral - ventral"] = [group_mean(avg, s, LATERAL) - group_mean(avg, s, VENTRAL) for s in subs]
    c["LO - hV4"] = [group_mean(avg, s, ["LO1", "LO2"]) - avg.get((s, "Kastner2015", "hV4"), (np.nan,))[0] for s in subs]
    c["streams: lateral - ventral"] = [avg.get((s, "streams", "lateral"), (np.nan,))[0] - avg.get((s, "streams", "ventral"), (np.nan,))[0] for s in subs]
    c["streams: parietal - ventral"] = [avg.get((s, "streams", "parietal"), (np.nan,))[0] - avg.get((s, "streams", "ventral"), (np.nan,))[0] for s in subs]
    c["streams: ventral - early"] = [avg.get((s, "streams", "ventral"), (np.nan,))[0] - avg.get((s, "streams", "early"), (np.nan,))[0] for s in subs]
    return {k: np.array(v) for k, v in c.items()}


def wilcoxon(x):
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.all(x == 0):
        return None
    return float(sps.wilcoxon(x, alternative="two-sided", method="exact").pvalue)


def f(x, d=3):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{d}f}".replace("-", "−")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="v2")
    arms, (A, B), out = SETS[ap.parse_args().set]
    os.makedirs(out, exist_ok=True)
    profs = {arm: profile(arm, arms) for arm in arms}
    res = {"convention": dict(nc_min=NC_MIN, min_vox=MIN_VOX, index="(Um-Up)/(Um+Up) median over reliable voxels; seeds averaged per subject")}
    md = ["# Stream preference beyond hV4 (per-voxel index, median over reliable voxels; mean over 8 subjects, seeds averaged per subject)", ""]
    for atlas, names in [("Kastner2015", KASTNER), ("streams", STREAMS)] + list(FLOC.items()):
        res[atlas] = {arm: table(profs[arm], atlas, names) for arm in arms}
        md += [f"## {atlas}", "", "| ROI | " + " | ".join(arms) + f" | median voxels ({A}) |", "|---|" + "---|" * (len(arms) + 1)]
        for name in names.values():
            cells = []
            for arm in arms:
                t = res[atlas][arm][name]
                cells.append("—" if t["mean"] is None else f"{f(t['mean'])} (n={t['n_subjects']})")
            md.append(f"| {name} | " + " | ".join(cells) + f" | {res[atlas][A][name]['median_vox']:.0f} |")
        md.append("")
    # contrasts
    con = {arm: contrasts(profs[arm]) for arm in arms}
    res["contrasts"] = {}
    md += ["## Regional contrasts (per-subject, mean ± sd over 8 subjects; p = exact Wilcoxon against 0; last column = paired " + f"{A} − {B})", "",
           "| contrast | " + " | ".join(arms) + f" | {A} − {B} (paired) |", "|---|" + "---|" * (len(arms) + 1)]
    for k in con[A]:
        cells = []
        res["contrasts"][k] = {}
        for arm in arms:
            x = con[arm][k]
            p = wilcoxon(x)
            res["contrasts"][k][arm] = dict(per_subject=x.tolist(), mean=float(np.nanmean(x)), sd=float(np.nanstd(x, ddof=1)), p_exact=p,
                                            n_neg=int((x < 0).sum()), n=int(np.isfinite(x).sum()))
            cells.append(f"{f(np.nanmean(x))} ± {np.nanstd(x, ddof=1):.3f}" + (f", p={p:.4f}" if p is not None else "") + f" ({int((x<0).sum())}/{int(np.isfinite(x).sum())} neg)")
        d = con[A][k] - con[B][k]
        p = wilcoxon(d)
        res["contrasts"][k][f"paired {A} - {B}"] = dict(per_subject=d.tolist(), mean=float(np.nanmean(d)), p_exact=p)
        cells.append(f"{f(np.nanmean(d))}" + (f", p={p:.4f}" if p is not None else "") + f" ({int((d<0).sum())}/{int(np.isfinite(d).sum())} neg)")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md.append("")
    # components: is the saturation a parvo rise or a magno collapse?
    md += ["## Components on Kastner2015 (median over reliable voxels of the noise-ceiling-normalised unique variances Um, Up and of joint r²/NC; mean over subjects, seeds averaged)", "",
           f"| ROI | Um {A} | Up {A} | Um {B} | Up {B} | Um untrained | Up untrained | r²/NC {A} | r²/NC {B} |", "|---|---|---|---|---|---|---|---|---|"]
    comp = {}
    for arm in (A, B, "untrained"):
        out_dir, models = arms[arm]
        acc = {}
        for m in fitted(out_dir, models):
            rr = load_results(out_dir, m)
            for s in sorted(rr):
                z = rr[s]
                roi = labels(s, "Kastner2015", len(z["unique_magno"]))
                um, up, nc = z["unique_magno"], z["unique_parvo"], z["noise_ceiling"]
                r2n = z["r2_joint_nc"] if "r2_joint_nc" in z else z["r2_joint"] / np.maximum(nc, 1e-6)
                for lab, name in KASTNER.items():
                    v = (roi == lab) & (nc > NC_MIN) & ((um + up) > 1e-3)
                    if v.sum() >= MIN_VOX:
                        acc.setdefault(name, []).append((np.median(um[v]), np.median(up[v]), np.median(r2n[v])))
        comp[arm] = {k: np.mean(np.array(v), 0).tolist() for k, v in acc.items()}
    res["components"] = comp
    for name in KASTNER.values():
        if name not in comp[A]:
            continue
        d, n, u = comp[A][name], comp[B][name], comp["untrained"].get(name, [np.nan] * 3)
        md.append(f"| {name} | {d[0]:.4f} | {d[1]:.4f} | {n[0]:.4f} | {n[1]:.4f} | {u[0]:.4f} | {u[1]:.4f} | {d[2]:.3f} | {n[2]:.3f} |")
    md.append("")
    json.dump(res, open(f"{out}/ext_gradient.json", "w"), indent=1)
    open(f"{out}/ext_gradient.md", "w", encoding="utf8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
