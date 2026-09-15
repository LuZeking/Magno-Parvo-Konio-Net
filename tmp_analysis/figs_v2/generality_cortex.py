"""Aggregate the generality cortex results (src/analysis/generality/cortex_bands.py).

  PYTHONPATH=. python tmp_analysis/figs_v2/generality_cortex.py [--run-id 2026-09-14_gen]

Per (network, subject, region): median over reliable voxels of the index
(U_S - U_T)/(U_S + U_T) between the shape-leaning and texture-leaning halves,
the two unique variances, and, from the *_w.npz files, the cortex-optimal w
of the lossy sum. Regions: V1, V2, V3, hV4 (prf-visualrois) and the ventral,
lateral and dorsal groups of the Kastner2015 atlas, as in Extended Data Fig. 5.
The two-stream diet model (results/encoding_cc, magno = shape-leaning stream)
is aggregated the same way as the reference profile.
Writes results/figures_v2/generality_cortex.json and prints the stats block.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats as sps

NC_MIN, MIN_VOX = 0.1, 20
VIS = {"V1": [1, 2], "V2": [3, 4], "V3": [5, 6], "hV4": [7]}
KAST = {"ventral": [8, 9, 10, 11], "lateral": [12, 13, 14, 15], "dorsal": [16, 17, 18, 19, 20]}
REGIONS = list(VIS) + list(KAST)
FAMILIES = {"rgb_tsn": ["run_rgb_tsn_seed0", "run_rgb_tsn_seed1", "run_rgb_tsn_seed2"],
            "single34": ["run_dkl_bcs_single34_seed0", "run_dkl_bcs_single34_seed1", "run_dkl_bcs_single34_seed2"],
            "resnet50": ["tv_resnet50"],
            # PLAN_SUBSPACE D4 adds two ImageNet networks (only present under --run-id 2026-09-15_sub)
            "vit_b_16": ["tv_vit_b_16"], "convnext_base": ["tv_convnext_base"]}


def region_masks(vis, kas, nc, extra_ok=None):
    ok = nc > NC_MIN
    if extra_ok is not None:
        ok &= extra_ok
    out = {}
    for name in REGIONS:
        roi, labs = (vis, VIS[name]) if name in VIS else (kas, KAST[name])
        out[name] = ok & np.isin(roi, labs)
    return out


def index_stats(uS, uT, m, shared=None, r2=None):
    v = m & np.isfinite(uS) & np.isfinite(uT) & ((uS + uT) > 1e-3)
    if v.sum() < MIN_VOX:
        return dict(index=np.nan, uS=np.nan, uT=np.nan, shared=np.nan, r2=np.nan, n=int(v.sum()))
    r = (uS[v] - uT[v]) / (uS[v] + uT[v])
    out = dict(index=float(np.median(r)), uS=float(np.median(uS[v])), uT=float(np.median(uT[v])), n=int(v.sum()))
    out["shared"] = float(np.nanmedian(shared[v])) if shared is not None else np.nan
    out["r2"] = float(np.nanmedian(r2[v])) if r2 is not None else np.nan
    return out


def kastner_labels(subject, mask_shape_ref):
    from src.analysis.encoding import nsd_data
    mask = nsd_data.load_mask(subject, "nsdgeneral")
    return nsd_data.load_roi_labels(subject, "Kastner2015", mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="2026-09-14_gen")
    ap.add_argument("--out", default="results/figures_v2/generality_cortex.json")
    args = ap.parse_args()
    root = f"/scratch/byrong/mpk/generality/cortex/{args.run_id}"
    out = {"run_id": args.run_id, "regions": REGIONS, "families": {}}

    # ---- single trunks
    for fam, tags in FAMILIES.items():
        per_subject = {}   # subject -> list over seeds of dict(region -> stats)
        per_subject_w = {}
        for tag in tags:
            for f in sorted(glob.glob(os.path.join(root, f"{tag}_subj??.npz"))):
                z = np.load(f, allow_pickle=True)
                s = int(z["subject"])
                masks = region_masks(z["roi_prf-visualrois"], z["roi_Kastner2015"], z["noise_ceiling"])
                st = {r: index_stats(z["unique_S"], z["unique_T"], masks[r], z["shared"], z["r2_joint_nc"]) for r in REGIONS}
                st["_r2_joint_nc_median"] = float(np.nanmedian(z["r2_joint_nc"][z["noise_ceiling"] > NC_MIN]))
                st["_r2_S_median"] = float(np.nanmedian(z["r2_S"][z["noise_ceiling"] > NC_MIN]))
                st["_r2_T_median"] = float(np.nanmedian(z["r2_T"][z["noise_ceiling"] > NC_MIN]))
                per_subject.setdefault(s, []).append(st)
                fw = f[:-4] + "_w.npz"
                if os.path.exists(fw):
                    zw = np.load(fw, allow_pickle=True)
                    grid, r2w = zw["grid"], zw["r2_w"]
                    okw = (zw["noise_ceiling"] > NC_MIN) & np.isfinite(r2w).all(0)
                    ws = {}
                    for r in REGIONS:
                        v = masks[r] & okw
                        if v.sum() >= MIN_VOX:
                            cur = np.median(r2w[:, v], axis=1)
                            ws[r] = dict(w_star=float(grid[int(np.argmax(cur))]), curve=cur.tolist(),
                                         rel_at_half=float(cur[len(cur) // 2] / cur.max()) if cur.max() > 0 else np.nan)
                    per_subject_w.setdefault(s, []).append(ws)
        if not per_subject:
            continue
        subjects = sorted(per_subject)
        fam_out = dict(subjects=subjects, n_seeds={s: len(v) for s, v in per_subject.items()}, index={}, uS={}, uT={}, shared={}, r2={}, w_star={})
        for r in REGIONS:
            fam_out["index"][r] = [float(np.nanmean([st[r]["index"] for st in per_subject[s]])) for s in subjects]
            fam_out["uS"][r] = [float(np.nanmean([st[r]["uS"] for st in per_subject[s]])) for s in subjects]
            fam_out["uT"][r] = [float(np.nanmean([st[r]["uT"] for st in per_subject[s]])) for s in subjects]
            fam_out["shared"][r] = [float(np.nanmean([st[r]["shared"] for st in per_subject[s]])) for s in subjects]
            fam_out["r2"][r] = [float(np.nanmean([st[r]["r2"] for st in per_subject[s]])) for s in subjects]
            if per_subject_w:
                fam_out["w_star"][r] = [float(np.nanmean([ws[r]["w_star"] for ws in per_subject_w[s] if r in ws])) if s in per_subject_w else np.nan for s in subjects]
        fam_out["r2_joint_nc"] = [float(np.nanmean([st["_r2_joint_nc_median"] for st in per_subject[s]])) for s in subjects]
        # stats
        idx = fam_out["index"]
        v1 = np.array(idx["V1"]); hv4 = np.array(idx["hV4"])
        ext = np.nanmean([idx["ventral"], idx["lateral"], idx["dorsal"]], 0)
        def wtest(a, b):
            d = np.array(a) - np.array(b); d = d[np.isfinite(d)]
            if len(d) < 5 or np.all(d == 0):
                return dict(mean=float(np.nanmean(d)) if len(d) else np.nan, n_neg=int((d < 0).sum()), n=int(len(d)), p=np.nan)
            return dict(mean=float(d.mean()), n_neg=int((d < 0).sum()), n=int(len(d)), p=float(sps.wilcoxon(d, alternative="two-sided", method="exact").pvalue))
        fam_out["stats"] = dict(hV4_minus_V1=wtest(hv4, v1), extrastriate_minus_V1=wtest(ext, v1),
                                dorsal_minus_ventral=wtest(idx["dorsal"], idx["ventral"]))
        rank = np.array([1, 2, 3, 4])
        rhos = [sps.spearmanr(rank, [idx[r][i] for r in ("V1", "V2", "V3", "hV4")]).correlation for i in range(len(subjects))]
        fam_out["stats"]["rho_rank_V1_hV4"] = dict(mean=float(np.nanmean(rhos)), values=[float(x) for x in rhos])
        if fam_out["w_star"]:
            wv1 = np.array(fam_out["w_star"]["V1"]); wext = np.nanmean([fam_out["w_star"][r] for r in KAST], 0)
            fam_out["stats"]["wstar_extrastriate_minus_V1"] = wtest(wext, wv1)
            fam_out["stats"]["wstar_median"] = {r: float(np.nanmedian(fam_out["w_star"][r])) for r in REGIONS}
        out["families"][fam] = fam_out

    # ---- two-stream diet reference (magno = shape-leaning stream 1)
    ref = {}
    for seed in range(3):
        for f in sorted(glob.glob(f"results/encoding_cc/dkl_bcs_seed{seed}_trained_subj??.npz")):
            z = np.load(f, allow_pickle=True)
            s = int(re.search(r"subj(\d\d)", f).group(1))
            try:
                kas = kastner_labels(s, None)
            except Exception as e:  # noqa
                print("no Kastner labels for", s, e); continue
            masks = region_masks(z["roi_prf-visualrois"], kas, z["noise_ceiling"])
            ref.setdefault(s, []).append({r: index_stats(z["unique_magno"], z["unique_parvo"], masks[r], z["shared"], z["r2_joint_nc"]) for r in REGIONS})
    if ref:
        subjects = sorted(ref)
        out["families"]["two_stream_diet"] = dict(subjects=subjects, index={r: [float(np.nanmean([st[r]["index"] for st in ref[s]])) for s in subjects] for r in REGIONS},
                                                   uS={r: [float(np.nanmean([st[r]["uS"] for st in ref[s]])) for s in subjects] for r in REGIONS},
                                                   uT={r: [float(np.nanmean([st[r]["uT"] for st in ref[s]])) for s in subjects] for r in REGIONS},
                                                   shared={r: [float(np.nanmean([st[r]["shared"] for st in ref[s]])) for s in subjects] for r in REGIONS},
                                                   r2={r: [float(np.nanmean([st[r]["r2"] for st in ref[s]])) for s in subjects] for r in REGIONS})

    for fam, fo in out["families"].items():
        print(f"\n== {fam}  (n subjects {len(fo['subjects'])})")
        print("   region   index(mean)   U_S     U_T   shared    r2" + ("     w*(median)" if fo.get("w_star") else ""))
        for r in REGIONS:
            line = f"   {r:8s} {np.nanmean(fo['index'][r]):+7.3f}   {np.nanmean(fo['uS'][r]):6.3f}  {np.nanmean(fo['uT'][r]):6.3f}  {np.nanmean(fo.get('shared', {}).get(r, [np.nan])):6.3f}  {np.nanmean(fo.get('r2', {}).get(r, [np.nan])):6.3f}"
            if fo.get("w_star") and r in fo["w_star"]:
                line += f"      {np.nanmedian(fo['w_star'][r]):.2f}"
            print(line)
        if "stats" in fo:
            for k, v in fo["stats"].items():
                print("   ", k, v)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
