"""Cortex results of PAPER_v2 (Results 2.3, Extended Data Figs 4-5) for the
Stage-2 fusion TSNs, beside the paper's own arms.

Arms
  BA diet / BA no diet  dkl_bcs / dkl_base, 3 seeds each (the published fits;
                        recomputing them here checks this script against the
                        paper's numbers)
  A                     fs2_bcs_avg_w050_ls, 3 seeds (thesis diet, Stage-2 recipe)
  dvd300                fs2_dvd300_avg_w050_ls, 3 seeds (DVD trajectory 0 -> 300 mo)
  control               fs2_dvdfix300_avg_w050_ls, seed 0 (300-mo diet from step 0:
                        dvd300's test input without the trajectory)

Statistics, as in tmp_analysis/figs_v2/{fig3_cortex,ext_gradient,readout_w_cortex}.py
and tmp_analysis/fig1_data.py (helpers reused where importable):
  gradient  hV4 - mean(V1v, V1d) of the per-ROI median (Um - Up)/(Um + Up)
            (nc > 0.1, Um + Up > 1e-3, >= 20 voxels)
  rho       Spearman(hierarchical rank [1,1,2,2,3,3,4], profile), per subject
  cc        median cc_magno_vsb - median cc_parvo_vsb over voxels with nc > 0.1
  r2/NC     median r2_joint_nc over voxels with nc > 0.1
  atlas     ext_gradient contrasts on Kastner2015 (region - V1), and the median
            noise-ceiling-normalised Um, Up over reliable voxels beyond V3
  w*        readout_w: w*(V1), mean w* over ventral/lateral/dorsal, their difference
Per-seed values are means over subjects; arm comparisons pair subjects on the
seed-mean (exact Wilcoxon, 8 subjects -> floor p = .0078).

Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/fs2_cortex_summary.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats as sps

import ext_gradient as EG          # tmp_analysis/figs_v2
import readout_w_cortex as RW      # tmp_analysis/figs_v2 (the summary, not the fit)
from src.analysis.encoding.stats import load_results

NC_MIN, MIN_VOX = 0.1, 20
VIS = [1, 2, 3, 4, 5, 6, 7]
HIER = np.array([1, 1, 2, 2, 3, 3, 4])
BA_ENC = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-rewrite/results/encoding_cc"
BA_RW = "/scratch/byrong/mpk/readout_w/results/2026-09-13_w21_last"
FS_ENC = "/scratch/byrong/mpk/encoding/results/2026-10-16_fs2"
FS_RW = "/scratch/byrong/mpk/readout_w/results/2026-10-16_fs2_w21"
ARMS = {
    "BA diet": (BA_ENC, BA_RW, [f"dkl_bcs_seed{s}_trained" for s in range(3)]),
    "BA no diet": (BA_ENC, BA_RW, [f"dkl_base_seed{s}_trained" for s in range(3)]),
    "A": (FS_ENC, FS_RW, [f"fs2_bcs_avg_w050_ls_seed{s}" for s in range(3)]),
    "dvd300": (FS_ENC, FS_RW, [f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)]),
    "control": (FS_ENC, FS_RW, ["fs2_dvdfix300_avg_w050_ls_seed0"]),
}
PAIRS = [("dvd300", "A"), ("dvd300", "control"), ("dvd300", "BA diet")]
SUBJ = list(range(1, 9))
OUT_MD = "docs/slides_2026-10-12_ba_vs_fusion/figures/fs2_cortex_summary.md"


def profile(z):
    roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
    um, up = z["unique_magno"], z["unique_parvo"]
    prof = np.full(len(VIS), np.nan)
    for j, lab in enumerate(VIS):
        v = (roi == lab) & (nc > NC_MIN) & ((um + up) > 1e-3)
        if v.sum() >= MIN_VOX:
            r = (um[v] - up[v]) / (um[v] + up[v])
            prof[j] = np.median(r[np.isfinite(r)])
    return prof


def med(v, ok):
    v = v[ok & np.isfinite(v)]
    return float(np.median(v)) if v.size >= MIN_VOX else np.nan


def beyond_v3(z, s):
    """Median Um, Up (nc-normalised) over reliable voxels in Kastner2015 labels
    beyond V3 (VO1 ... IPS2; labels 8-20)."""
    n = len(z["unique_magno"])
    k = EG.labels(s, "Kastner2015", n)
    ok = (z["noise_ceiling"] > NC_MIN) & np.isin(k, np.arange(8, 21))
    return med(z["unique_magno"], ok), med(z["unique_parvo"], ok)


def encoding_stats(enc_dir, models):
    """seeds x subjects arrays of every per-subject statistic (NaN if missing)."""
    keys = ("grad", "rho", "cc", "r2nc", "um_ext", "up_ext")
    out = {k: np.full((len(models), len(SUBJ)), np.nan) for k in keys}
    out["prof"] = np.full((len(models), len(SUBJ), len(VIS)), np.nan)
    for i, m in enumerate(models):
        for s, z in load_results(enc_dir, m).items():
            j = SUBJ.index(s)
            p = profile(z)
            out["prof"][i, j] = p
            out["grad"][i, j] = p[6] - p[:2].mean()
            out["rho"][i, j] = sps.spearmanr(HIER, p)[0] if np.isfinite(p).all() else np.nan
            ok = z["noise_ceiling"] > NC_MIN
            if "cc_magno_vsb" in z:
                out["cc"][i, j] = med(z["cc_magno_vsb"], ok) - med(z["cc_parvo_vsb"], ok)
            out["r2nc"][i, j] = med(z["r2_joint_nc"], ok)
            out["um_ext"][i, j], out["up_ext"][i, j] = beyond_v3(z, s)
    return out


def readout_stats(rw_dir, models):
    """Per subject: seed-mean curve per ROI -> w*(V1), w*(extrastriate)."""
    import glob
    per = {}
    for m in models:
        for f in sorted(glob.glob(os.path.join(rw_dir, f"{m}_subj*.npz"))):
            z = np.load(f, allow_pickle=True)
            grid, res = RW.curves(z)
            for roi, d in res.items():
                per.setdefault(int(z["subject"]), {}).setdefault(roi, []).append(d["curve"])
    if not per:
        return None
    ext_rois = ("ventral (VO, PHC)", "lateral (LO, TO)", "dorsal (V3A/B, IPS0–2)")
    v1, ext, subs = [], [], []
    for s in sorted(per):
        A = per[s]
        if not all(r in A for r in ("V1",) + ext_rois):
            continue
        ws = lambda r: float(grid[int(np.argmax(np.mean(A[r], 0)))])
        subs.append(s); v1.append(ws("V1")); ext.append(np.mean([ws(r) for r in ext_rois]))
    v1, ext = np.array(v1), np.array(ext)
    d = ext - v1
    p = float(sps.wilcoxon(d, method="exact").pvalue) if len(d) >= 5 and np.any(d != 0) else None
    return dict(subjects=subs, w_v1=v1.tolist(), w_ext=ext.tolist(), delta=d.tolist(), p_exact=p,
                n_seeds=len(models))


def wil(x):
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.all(x == 0):
        return None
    return float(sps.wilcoxon(x, method="exact").pvalue)


def fmt(x, d=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}".replace("-", "−")


def main():
    E, R, C = {}, {}, {}
    for arm, (enc, rw, models) in ARMS.items():
        E[arm] = encoding_stats(enc, models)
        R[arm] = readout_stats(rw, models)
        EG.ARMS[arm] = (enc, models)
        try:
            C[arm] = EG.contrasts(EG.profile(arm))
        except Exception as e:          # atlas relabel needs all subjects' npz
            print(f"[atlas] {arm}: {e}")
            C[arm] = None

    md = ["# Cortex results for the Stage-2 fusion TSNs (PAPER_v2 Results 2.3, ED Figs 4-5)", "",
          "Generated by `tmp_analysis/fs2_cortex_summary.py`. Per seed: mean over subjects "
          "(n = subjects with a fit). Arm columns `seed-mean`: mean over subjects of the "
          "per-subject seed mean.", "",
          "## Banded encoding, V1v → hV4 (Fig. 3)", "",
          "| arm | n subj | gradient hV4 − V1 per seed | seed-mean | ρ(rank) per seed | cc magno − parvo per seed | r²/NC per seed |",
          "|---|---|---|---|---|---|---|"]
    js = {"arms": {}}
    for arm, e in E.items():
        n = int(np.isfinite(e["grad"]).sum(1).max())
        if n == 0:
            md.append(f"| {arm} | 0 | — | — | — | — | — |"); continue
        g = np.nanmean(e["grad"], 1); r = np.nanmean(e["rho"], 1)
        c = np.nanmean(e["cc"], 1); q = np.nanmean(e["r2nc"], 1)
        sm = np.nanmean(np.nanmean(e["grad"], 0))
        md.append(f"| {arm} | {n} | {' / '.join(fmt(x) for x in g)} | {fmt(sm)} | "
                  f"{' / '.join(fmt(x) for x in r)} | {' / '.join(fmt(x, 2) for x in c)} | "
                  f"{' / '.join(f'{x:.3f}' for x in q)} |")
        prof = np.nanmean(np.nanmean(e["prof"], 0), 0)
        js["arms"][arm] = dict(n_subjects=n, grad_per_seed=g.tolist(), grad_seedmean_per_subject=np.nanmean(e["grad"], 0).tolist(),
                               rho_per_seed=r.tolist(), cc_per_seed=c.tolist(), r2nc_per_seed=q.tolist(),
                               profile=prof.tolist(), um_ext=np.nanmean(e["um_ext"]).item(), up_ext=np.nanmean(e["up_ext"]).item())
    md += ["", "Profile V1v, V1d, V2v, V2d, V3v, V3d, hV4 (seed-mean, subject-mean):", ""]
    for arm, d in js["arms"].items():
        md.append(f"- {arm}: " + " ".join(fmt(x) for x in d["profile"]))

    md += ["", "## Paired arm contrasts (per subject, seed-mean; exact Wilcoxon)", "",
           "| contrast | Δ gradient | n neg | p | Δ cc | p | Δ r²/NC | n pos | p |", "|---|---|---|---|---|---|---|---|---|"]
    js["pairs"] = {}
    for a, b in PAIRS:
        if a not in js["arms"] or b not in js["arms"]:
            continue
        dg = np.nanmean(E[a]["grad"], 0) - np.nanmean(E[b]["grad"], 0)
        dc = np.nanmean(E[a]["cc"], 0) - np.nanmean(E[b]["cc"], 0)
        dq = np.nanmean(E[a]["r2nc"], 0) - np.nanmean(E[b]["r2nc"], 0)
        js["pairs"][f"{a} - {b}"] = dict(grad=dg.tolist(), cc=dc.tolist(), r2nc=dq.tolist())
        md.append(f"| {a} − {b} | {fmt(np.nanmean(dg))} | {int((dg < 0).sum())}/{int(np.isfinite(dg).sum())} | {wil(dg)} | "
                  f"{fmt(np.nanmean(dc), 2)} | {wil(dc)} | {fmt(np.nanmean(dq))} | {int((dq > 0).sum())}/{int(np.isfinite(dq).sum())} | {wil(dq)} |")

    md += ["", "## Beyond hV4 (ED Fig. 4): region − V1 on Kastner2015, and unique variance beyond V3", "",
           "| arm | hV4 − V1 | ventral − V1 | lateral − V1 | dorsal − V1 | Um beyond V3 | Up beyond V3 |", "|---|---|---|---|---|---|---|"]
    keys = ["hV4 - early", "ventral(VO,PHC) - early", "lateral(LO,TO) - early", "dorsal(V3AB,IPS0-3) - early"]
    for arm in ARMS:
        if C[arm] is None or arm not in js["arms"]:
            continue
        cells = []
        for k in keys:
            x = C[arm][k]
            cells.append(f"{fmt(np.nanmean(x))} ({int((x < 0).sum())}/{int(np.isfinite(x).sum())} neg, p={wil(x)})")
        js["arms"][arm]["atlas"] = {k: C[arm][k].tolist() for k in keys}
        md.append(f"| {arm} | " + " | ".join(cells) + f" | {js['arms'][arm]['um_ext']:.4f} | {js['arms'][arm]['up_ext']:.4f} |")

    md += ["", "## Cortex-optimal readout weight (ED Fig. 5 / Table 2)", "",
           "| arm | n subj | w*(V1) | w*(extrastriate) | Δ ext − V1 | n neg | p |", "|---|---|---|---|---|---|---|"]
    js["readout_w"] = R
    for arm, r in R.items():
        if r is None:
            md.append(f"| {arm} | 0 | — | — | — | — | — |"); continue
        d = np.array(r["delta"])
        md.append(f"| {arm} | {len(d)} | {np.mean(r['w_v1']):.3f} | {np.mean(r['w_ext']):.3f} | {fmt(d.mean())} | "
                  f"{int((d < 0).sum())}/{len(d)} | {r['p_exact']} |")

    text = "\n".join(md) + "\n"
    print(text)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w").write(text)
    os.makedirs(os.path.join(FS_ENC, "summary"), exist_ok=True)
    json.dump(js, open(os.path.join(FS_ENC, "summary", "fs2_cortex_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
