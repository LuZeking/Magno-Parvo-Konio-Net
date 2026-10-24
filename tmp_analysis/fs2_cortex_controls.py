"""Extended Data Fig. 3 and Supplementary Note S3 on the model of record: the cortical gradient
against its nuisance axes (eccentricity, pRF size) and the estimator's own bias.

The same analyses PAPER_v2 ran on the BA network (figs_v2/figE3_cortex_controls.py at 7673c18,
prf_dissociation.py, prf_interaction.py, prf_untrained_pair.py, vp_recovery_realnc.py), on the
model (fs2_dvd300_avg_w050_ls, seeds 0-2), its control (fs2_dvdfix300_avg_w050_ls, seeds 0-2),
the channel swap (fs2_dvd300swap_avg_w050_ls, seeds 0-2) and the untrained network on their
shared front end (untrained_dvd300_seed0). Conventions unchanged:

  * voxels       noise ceiling > 0.1, U_m + U_p > 1e-3; an ROI cell needs >= 20 voxels
  * statistics   normalised (U_m - U_p)/(U_m + U_p) and raw U_m - U_p, both reported
  * gradient     hV4 - mean(V1v, V1d) of the per-ROI medians, per subject; trained arms are
                 seed-averaged per subject before any test
  * strata       NSD prf-eccrois bands 1-5; pRF-size tertiles set per subject on the pooled
                 V1 + hV4 voxels with pRF R2 > 5%
  * regression   z(per-voxel index) ~ z(log(ecc + 0.5)) + z(log size), within subject, voxels with
                 pRF R2 > 5% and finite ecc/size, size > 0 (>= 200 voxels); partial betas
  * tests        exact two-sided Wilcoxon over the 8 subjects (floor p = .0078)

The estimator-bias block reads vp_recovery_realnc.py's outputs for seed 0 of the model and the
control (slurm/vp_recovery_fs2.sh) and is skipped, and listed as skipped, until all 16 exist.

Usage: PYTHONPATH=.:tmp_analysis python tmp_analysis/fs2_cortex_controls.py [--out results/figures_v3/cortex_controls.json]
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

ENC = os.environ.get("FS2_ENCODING_DIR", "/scratch/byrong/mpk/encoding/results/2026-10-16_fs2")
SIM = os.environ.get("FS2_VPREC_DIR", "/scratch/byrong/mpk/encoding/results/2026-10-24_vprec_fs2")
NC_MIN, DENOM_MIN, MIN_VOX, R2_MIN, MIN_REG = 0.1, 1e-3, 20, 5.0, 200
ROI = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
BANDS = ["0-0.5", "0.5-1", "1-2", "2-4", ">4"]
TERTILES = ["small", "medium", "large"]
ARMS = {"model": [f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)],
        "control": [f"fs2_dvdfix300_avg_w050_ls_seed{s}" for s in range(3)],
        "channel swap": [f"fs2_dvd300swap_avg_w050_ls_seed{s}" for s in range(3)],
        "untrained": ["untrained_dvd300_seed0"]}
SUBJECTS = list(range(1, 9))

_prf = {}


def prf(s):
    """(eccentricity, size, R2) in the nsdgeneral voxel order of the encoding outputs."""
    if s not in _prf:
        mask = nsd_data.load_mask(s, "nsdgeneral")
        _prf[s] = tuple(nsd_data.load_prf(s, k, mask) for k in ("eccentricity", "size", "R2"))
    return _prf[s]


def zs(v):
    return (v - v.mean()) / v.std()


def index(z, raw):
    um, up, nc = z["unique_magno"], z["unique_parvo"], z["noise_ceiling"]
    ok = (nc > NC_MIN) & ((um + up) > DENOM_MIN) & np.isfinite(um) & np.isfinite(up)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (um - up) if raw else (um - up) / (um + up)
    return val, ok


def med(val, sel):
    return float(np.median(val[sel])) if sel.sum() >= MIN_VOX else np.nan


def grad(prof):
    """hV4 - mean(V1v, V1d); a missing V1 half leaves the other (as figE3 at 7673c18)."""
    v1 = prof[:2][np.isfinite(prof[:2])]
    return float(prof[6] - v1.mean()) if v1.size and np.isfinite(prof[6]) else np.nan


def per_model(model):
    """Per subject: profiles, stratum gradients and partial slopes, on both statistics."""
    res = load_results(ENC, model)
    assert sorted(res) == SUBJECTS, (model, sorted(res))
    out = {}
    for s in SUBJECTS:
        z = res[s]; roi = z["roi_prf-visualrois"]; band = z["roi_prf-eccrois"]
        ecc, size, r2 = prf(s)
        assert len(ecc) == len(roi), (model, s, len(ecc), len(roi))
        row = {}
        for stat in ("norm", "raw"):
            val, ok = index(z, stat == "raw")
            prof = np.array([med(val, ok & (roi == k)) for k in range(1, 8)])
            # eccentricity bands
            g_ecc = [grad(np.array([med(val, ok & (band == b) & (roi == k)) for k in range(1, 8)]))
                     for b in range(1, 6)]
            # pRF-size tertiles on pooled V1 + hV4
            good = ok & (r2 > R2_MIN) & np.isfinite(size) & (size > 0)
            q = np.quantile(size[good & np.isin(roi, [1, 2, 7])], [1 / 3, 2 / 3])
            strata = [good & (size <= q[0]), good & (size > q[0]) & (size <= q[1]), good & (size > q[1])]
            g_size = [grad(np.array([med(val, ok & st & (roi == k)) for k in range(1, 8)])) for st in strata]
            # continuous partial slopes
            sel = ok & (r2 > R2_MIN) & np.isfinite(ecc) & np.isfinite(size) & (size > 0)
            if sel.sum() >= MIN_REG:
                le, ls = zs(np.log(ecc[sel] + 0.5)), zs(np.log(size[sel]))
                A = np.column_stack([np.ones(int(sel.sum())), le, ls])
                coef, *_ = np.linalg.lstsq(A, zs(val[sel]), rcond=None)
                r = float(np.corrcoef(le, ls)[0, 1])
                reg = {"ecc_partial": float(coef[1]), "size_partial": float(coef[2]),
                       "n_vox": int(sel.sum()), "r_logecc_logsize": r, "vif": 1.0 / (1.0 - r * r)}
            else:
                reg = None
            row[stat] = {"profile": prof.tolist(), "gradient": grad(prof), "ecc_bands": g_ecc,
                         "size_tertiles": g_size, "size_cuts": q.tolist(), "regression": reg}
        out[s] = row
    return out


def seed_mean(per_seed, stat, key, sub=None):
    """subjects x ... array of a quantity, averaged over the arm's seeds."""
    def get(d, s):
        v = d[s][stat][key]
        return v[sub] if sub is not None else v
    return np.nanmean(np.stack([np.array([get(d, s) for s in SUBJECTS], dtype=float) for d in per_seed]), 0)


def wil(x):
    """(mean, n < 0, n, exact two-sided Wilcoxon p) over finite entries."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    p = float(sps.wilcoxon(x, method="exact").pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan
    return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
            "n_neg": int((x < 0).sum()), "n_pos": int((x > 0).sum()), "n": int(len(x)), "p": p}


def sim_block():
    """The estimator's own bias, seed 0 of the model and the control (vp_recovery_realnc.py)."""
    arms = {"model": "fs2_dvd300_avg_w050_ls_seed0", "control": "fs2_dvdfix300_avg_w050_ls_seed0"}
    files = {a: sorted(glob.glob(f"{SIM}/vp_recovery_{m}_subj*.json")) for a, m in arms.items()}
    if any(len(f) != 8 for f in files.values()):
        return {"skipped": f"needs 8 files per arm in {SIM}; found " + ", ".join(f"{a} {len(f)}" for a, f in files.items())}

    def contrasts(rows, prefix):
        um = np.array([r[f"{prefix}unique_m"] for r in rows]); up = np.array([r[f"{prefix}unique_p"] for r in rows])
        ratio = (um - up) / (um + up)
        return grad(ratio), grad(um - up), ratio

    out = {}
    for a, m in arms.items():
        real = {}
        for s in SUBJECTS:
            z = np.load(f"{ENC}/{m}_subj{s:02d}.npz", allow_pickle=True)
            val, ok = index(z, False); roi = z["roi_prf-visualrois"]
            real[s] = grad(np.array([med(val, ok & (roi == k)) for k in range(1, 8)]))
        B, A, per_roi = {}, {}, []
        for f in files[a]:
            d = json.load(open(f)); s = int(d["subject"])
            nb, rb, ratio_b = contrasts(d["conditions"]["B_manufacture"]["rows"], "est_")
            B[s] = (nb, rb); per_roi.append(ratio_b)
            ea, _, _ = contrasts(d["conditions"]["A_recovery"]["rows"], "est_")
            ta, _, _ = contrasts(d["conditions"]["A_recovery"]["rows"], "true_")
            A[s] = (ea, ta, d["conditions"]["A_recovery"]["rho_est_vs_rank"])
        floor_n = np.array([B[s][0] for s in SUBJECTS]); floor_r = np.array([B[s][1] for s in SUBJECTS])
        disp = np.array([real[s] - B[s][0] for s in SUBJECTS])
        out[a] = {"run": m,
                  "floor_norm": {"per_subject": floor_n.tolist(), **wil(floor_n)},
                  "floor_raw": {"per_subject": floor_r.tolist(), **wil(floor_r)},
                  "floor_per_roi_norm_mean": np.mean(per_roi, 0).tolist(),
                  "real_norm_seed0": [real[s] for s in SUBJECTS],
                  "displacement_norm": {"per_subject": disp.tolist(), **wil(disp)},
                  "recovery_A_est_over_true": float(np.mean([A[s][0] / A[s][1] for s in SUBJECTS])),
                  "recovery_A_rho_est": float(np.mean([A[s][2] for s in SUBJECTS]))}
    dm = np.array(out["model"]["displacement_norm"]["per_subject"]); dc = np.array(out["control"]["displacement_norm"]["per_subject"])
    out["model_minus_control_displacement"] = wil(dm - dc)
    out["displacement_ratio_model_over_control"] = float(dm.mean() / dc.mean())
    out["floor_geometries_mannwhitney_p"] = float(sps.mannwhitneyu(out["model"]["floor_norm"]["per_subject"],
                                                                   out["control"]["floor_norm"]["per_subject"]).pvalue)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures_v3/cortex_controls.json")
    args = ap.parse_args()

    per = {a: [per_model(m) for m in ms] for a, ms in ARMS.items()}
    J = {"encoding_dir": ENC, "arms": ARMS, "subjects": SUBJECTS, "rois": ROI, "bands": BANDS,
         "per_subject": {a: {stat: {"profile": seed_mean(per[a], stat, "profile").tolist(),
                                    "gradient": seed_mean(per[a], stat, "gradient").tolist(),
                                    "ecc_bands": seed_mean(per[a], stat, "ecc_bands").tolist(),
                                    "size_tertiles": seed_mean(per[a], stat, "size_tertiles").tolist()}
                             for stat in ("norm", "raw")} for a in ARMS},
         "per_seed_regression": {m: {stat: {s: d[s][stat]["regression"] for s in SUBJECTS} for stat in ("norm", "raw")}
                                 for a, ms in ARMS.items() for m, d in zip(ms, per[a])}}

    T = {}
    for stat in ("norm", "raw"):
        T[stat] = {}
        for a in ARMS:
            ps = J["per_subject"][a][stat]
            T[stat][a] = {"gradient": wil(ps["gradient"]),
                          "ecc_bands": [wil(np.array(ps["ecc_bands"])[:, b]) for b in range(5)],
                          "size_tertiles": [wil(np.array(ps["size_tertiles"])[:, b]) for b in range(3)],
                          "ecc_partial": wil([np.mean([per[a][i][s][stat]["regression"]["ecc_partial"] for i in range(len(per[a]))]) for s in SUBJECTS]),
                          "size_partial": wil([np.mean([per[a][i][s][stat]["regression"]["size_partial"] for i in range(len(per[a]))]) for s in SUBJECTS])}
        # paired contrasts: model against each comparison arm, per stratum and per partial slope
        for other in ("control", "channel swap", "untrained"):
            pm, po = J["per_subject"]["model"][stat], J["per_subject"][other][stat]
            key = f"model_minus_{other.replace(' ', '_')}"
            T[stat][key] = {
                "ecc_bands": [wil(np.array(pm["ecc_bands"])[:, b] - np.array(po["ecc_bands"])[:, b]) for b in range(5)],
                "size_tertiles": [wil(np.array(pm["size_tertiles"])[:, b] - np.array(po["size_tertiles"])[:, b]) for b in range(3)]}
            for term in ("ecc_partial", "size_partial"):
                mm = [np.mean([per["model"][i][s][stat]["regression"][term] for i in range(3)]) for s in SUBJECTS]
                oo = [np.mean([per[other][i][s][stat]["regression"][term] for i in range(len(per[other]))]) for s in SUBJECTS]
                T[stat][key][term] = wil(np.array(mm) - np.array(oo))
    J["tests"] = T
    J["vif_mean_model_seed0"] = float(np.mean([per["model"][0][s]["norm"]["regression"]["vif"] for s in SUBJECTS]))
    J["estimator_bias"] = sim_block()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(J, open(args.out, "w"), indent=1, allow_nan=True)
    print(f"saved {args.out}")

    # human-readable summary
    f = lambda w: f"{w['mean']:+.3f} [{w['min']:+.3f}, {w['max']:+.3f}] neg {w['n_neg']}/{w['n']} p={w['p']:.4f}"
    for stat in ("norm", "raw"):
        print(f"\n=== statistic: {stat} ===")
        for a in ARMS:
            t = T[stat][a]
            print(f"  {a:13s} gradient {f(t['gradient'])}")
            print("    ecc bands   " + " | ".join(f"{BANDS[b]}: {w['mean']:+.3f} ({w['n_neg']}/{w['n']})" for b, w in enumerate(t["ecc_bands"])))
            print("    size terts  " + " | ".join(f"{TERTILES[b]}: {w['mean']:+.3f} ({w['n_neg']}/{w['n']})" for b, w in enumerate(t["size_tertiles"])))
            print(f"    ecc|size  {f(t['ecc_partial'])}")
            print(f"    size|ecc  {f(t['size_partial'])}")
        for other in ("control", "channel_swap", "untrained"):
            t = T[stat][f"model_minus_{other}"]
            print(f"  model - {other}:")
            print("    ecc bands   " + " | ".join(f"{BANDS[b]}: {w['mean']:+.3f} ({w['n_neg']}/{w['n']} p={w['p']:.4f})" for b, w in enumerate(t["ecc_bands"])))
            print("    size terts  " + " | ".join(f"{TERTILES[b]}: {w['mean']:+.3f} ({w['n_neg']}/{w['n']} p={w['p']:.4f})" for b, w in enumerate(t["size_tertiles"])))
            print(f"    ecc|size  {f(t['ecc_partial'])}")
            print(f"    size|ecc  {f(t['size_partial'])}")
    print(f"\nmean VIF (model seed 0) {J['vif_mean_model_seed0']:.2f}")
    eb = J["estimator_bias"]
    if "skipped" in eb:
        print(f"\nestimator bias: SKIPPED -- {eb['skipped']}")
    else:
        for a in ("model", "control"):
            e = eb[a]
            print(f"\nestimator bias, {a} ({e['run']}):")
            print(f"  floor normalised {f(e['floor_norm'])}")
            print(f"  floor raw        {f(e['floor_raw'])}")
            print("  floor per ROI (norm, subject mean) " + " ".join(f"{v:+.3f}" for v in e["floor_per_roi_norm_mean"]))
            print(f"  displacement (real seed0 - own floor) {f(e['displacement_norm'])}")
            print(f"  recovery A: est/true contrast {e['recovery_A_est_over_true']:.3f}, rho(est) {e['recovery_A_rho_est']:+.3f}")
        print(f"  model - control displacement {f(eb['model_minus_control_displacement'])}")
        print(f"  displacement ratio model/control {eb['displacement_ratio_model_over_control']:.2f}")
        print(f"  floor geometries Mann-Whitney p {eb['floor_geometries_mannwhitney_p']:.3f}")


if __name__ == "__main__":
    main()
