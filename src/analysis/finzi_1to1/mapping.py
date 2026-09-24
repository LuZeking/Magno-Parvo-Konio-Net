"""Fixed-quota 1-to-1 unit-to-voxel assignment (plan §4-§8).

Modes, per network x subject:
  main  TSN: group A = stream 1 (magno), B = stream 2 (parvo), position-balanced
        quota, D draws per hemisphere; plus an unforced variant (each stream alone
        can cover every voxel; descriptive). TDANN: Finzi's own design -- all
        responsive units, rectangular assignment -- for the descriptive
        reproduction of §8. Both write per-unit dorsal-minus-ventral and
        lateral-minus-ventral mean train correlations (`*_prefs.npz`) for the
        LOSO controls.
  pc    LOSO known-positive (§7): groups = top / bottom quarter of each grid
        position's units ranked by the other seven subjects' preference; plus the
        negative control (random position-balanced split).

    python -m src.analysis.finzi_1to1.mapping --network diet_s0 --subject 1 --mode main
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.analysis.finzi_1to1 import OUT_ROOT, STREAM_LABELS, betas_cache_path, features_path

LAB = {v: k for k, v in STREAM_LABELS.items()}  # name -> label
N_NC_BINS = 5
MIN_BIN_VOXELS = 20


def results_dir() -> str:
    return os.path.join(OUT_ROOT, "results")


def out_stem(network: str, subject: int) -> str:
    return os.path.join(results_dir(), f"{network}__subj{subject:02d}")


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x -= x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    x /= sd
    return x


def load(network: str, subject: int) -> dict:
    b = np.load(betas_cache_path(subject))
    f = np.load(features_path(network, subject))
    for k in ("train_image_ids", "test_image_ids"):
        if not np.array_equal(b[k], f[k]):
            raise RuntimeError(f"{network} subj{subject}: {k} differ between betas and features")
    ftr = f["train_feats"].astype(np.float32)
    fte = f["test_feats"].astype(np.float32)
    responsive = (ftr.std(0) > 0) & (fte.std(0) > 0)
    d = dict(
        n_units_all=ftr.shape[1],
        unit_id=np.flatnonzero(responsive),
        unit_stream=f["unit_stream"][responsive],
        unit_pos=f["unit_pos"][responsive],
        hemi=b["hemi"], label=b["label"], nc=b["nc"],
        provenance=str(f["provenance"]),
    )
    ztr_f = zscore(ftr[:, responsive])
    zte_f = zscore(fte[:, responsive])
    del ftr, fte
    ztr_b = zscore(b["train_betas"])
    zte_b = zscore(b["test_betas"])
    d["C_train"], d["C_test"], d["vox"] = {}, {}, {}
    for h in (0, 1):
        idx = np.flatnonzero(d["hemi"] == h)
        d["vox"][h] = idx
        d["C_train"][h] = (ztr_f.T @ ztr_b[:, idx]) / ztr_f.shape[0]
        d["C_test"][h] = (zte_f.T @ zte_b[:, idx]) / zte_f.shape[0]
    return d


PREF_STATS = ("p90", "meanr")  # p90 = amendment 1 (used by the gate); meanr = as registered


def unit_prefs(d: dict) -> dict[str, np.ndarray]:
    """Per statistic: (2 contrasts [DV, LV], 2 hemis, n_units_all) train-r differences.

    `meanr` (as registered) is a unit's mean train r with a stream's voxels. It
    cancels when voxel loadings on a unit's features have mixed signs, which the
    synthetic check showed can null a planted preference; `p90` (the 90th
    percentile of the unit's r over that stream's voxels) tracks the voxels a
    unit can actually win in the assignment (plan amendment 1).
    """
    out = {k: np.full((2, 2, d["n_units_all"]), np.nan, dtype=np.float32) for k in PREF_STATS}
    for h in (0, 1):
        lab = d["label"][d["vox"][h]]
        C = d["C_train"][h]
        for k in PREF_STATS:
            f = (lambda c: np.percentile(c, 90, axis=1)) if k == "p90" else (lambda c: c.mean(1))
            m = {n: f(C[:, lab == LAB[n]]) for n in ("ventral", "lateral", "dorsal")}
            out[k][0, h, d["unit_id"]] = m["dorsal"] - m["ventral"]
            out[k][1, h, d["unit_id"]] = m["lateral"] - m["ventral"]
    return out


def assign(C_tr: np.ndarray, C_te: np.ndarray, rows_a: np.ndarray, rows_b: np.ndarray):
    """Hungarian assignment; returns per-voxel group (1 = A, 0 = B, -1 none) and test r."""
    rows = np.r_[rows_a, rows_b]
    is_a = np.r_[np.ones(len(rows_a), np.int8), np.zeros(len(rows_b), np.int8)]
    r, c = linear_sum_assignment(1.0 - C_tr[rows].astype(np.float64))
    n_vox = C_tr.shape[1]
    grp = np.full(n_vox, -1, np.int8)
    grp[c] = is_a[r]
    r_test = np.full(n_vox, np.nan)
    r_test[c] = C_te[rows[r], c]
    return grp, r_test


def strat_delta(a: np.ndarray, lab: np.ndarray, nc: np.ndarray, s1: str, s2: str):
    """NC-quintile-stratified share_A(s1) - share_A(s2) over assigned voxels."""
    edges = np.quantile(nc, np.linspace(0, 1, N_NC_BINS + 1))
    bins = np.clip(np.searchsorted(edges, nc, side="right") - 1, 0, N_NC_BINS - 1)
    diffs = []
    for b in range(N_NC_BINS):
        m1 = (bins == b) & (lab == LAB[s1])
        m2 = (bins == b) & (lab == LAB[s2])
        if m1.sum() >= MIN_BIN_VOXELS and m2.sum() >= MIN_BIN_VOXELS:
            diffs.append(a[m1].mean() - a[m2].mean())
    return (float(np.mean(diffs)) if diffs else float("nan")), len(diffs)


def summarize(grp: np.ndarray, r_test: np.ndarray, lab: np.ndarray, nc: np.ndarray) -> dict:
    asg = grp >= 0
    a, lab_a, nc_a = grp[asg].astype(np.float64), lab[asg], nc[asg]
    rn = r_test / np.sqrt(nc / 100.0)
    out = {"n_assigned": int(asg.sum()), "n_voxels": int(len(grp))}
    for name, l in LAB.items():
        m = asg & (lab == l)
        out[f"share_{name}"] = float(grp[m].mean()) if m.any() else float("nan")
        out[f"n_{name}"] = int(m.sum())
        out[f"fr_{name}"] = float(np.nanmean(rn[m])) if m.any() else float("nan")
        for g, gname in ((1, "A"), (0, "B")):
            mm = m & (grp == g)
            out[f"fr_{name}_{gname}"] = float(np.nanmean(rn[mm])) if mm.any() else float("nan")
    out["share_all"] = float(a.mean())
    out["dDV_raw"] = out["share_dorsal"] - out["share_ventral"]
    out["dLV_raw"] = out["share_lateral"] - out["share_ventral"]
    out["dDV_strat"], out["nbins_DV"] = strat_delta(a, lab_a, nc_a, "dorsal", "ventral")
    out["dLV_strat"], out["nbins_LV"] = strat_delta(a, lab_a, nc_a, "lateral", "ventral")
    return out


def pools_by_pos(pos: np.ndarray, rows: np.ndarray) -> dict[int, np.ndarray]:
    return {p: rows[pos[rows] == p] for p in np.unique(pos)}


def quota(pools_a: dict, pools_b: dict, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    ra, rb = [], []
    for p in sorted(pools_a):
        pa, pb = pools_a[p], pools_b.get(p, np.array([], int))
        k = min(n, len(pa), len(pb))
        if k:
            ra.append(rng.choice(pa, k, replace=False))
            rb.append(rng.choice(pb, k, replace=False))
    return np.concatenate(ra), np.concatenate(rb)


def run_main(d: dict, network: str, draws: int, draws_unforced: int) -> dict:
    res = {"draws": [], "unforced": [], "finzi_all_units": []}
    rows = np.arange(len(d["unit_id"]))
    n_pos = len(np.unique(d["unit_pos"]))
    is_tdann = network.startswith("tdann")
    for h in (0, 1):
        lab, nc = d["label"][d["vox"][h]], d["nc"][d["vox"][h]]
        C_tr, C_te = d["C_train"][h], d["C_test"][h]
        n_vox = C_tr.shape[1]
        if is_tdann:
            t = time.time()
            grp, r_test = assign(C_tr, C_te, rows, rows[:0])
            s = summarize(grp, r_test, lab, nc)
            s.update(hemi=h, seconds=round(time.time() - t, 1))
            res["finzi_all_units"].append(s)
            print(f"[main] {network} h{h} all-units fr v/l/d = "
                  f"{s['fr_ventral']:.3f}/{s['fr_lateral']:.3f}/{s['fr_dorsal']:.3f}", flush=True)
            continue
        pa = pools_by_pos(d["unit_pos"], rows[d["unit_stream"] == 0])
        pb = pools_by_pos(d["unit_pos"], rows[d["unit_stream"] == 1])
        for kind, n_per, nd in (("draws", -(-n_vox // (2 * n_pos)), draws),
                                ("unforced", -(-n_vox // n_pos), draws_unforced)):
            for i in range(nd):
                rng = np.random.default_rng([h, i, 0 if kind == "draws" else 1])
                t = time.time()
                ra, rb = quota(pa, pb, n_per, rng)
                grp, r_test = assign(C_tr, C_te, ra, rb)
                s = summarize(grp, r_test, lab, nc)
                s.update(hemi=h, draw=i, n_units=int(len(ra) + len(rb)),
                         n_per_pos=int(n_per), seconds=round(time.time() - t, 1))
                res[kind].append(s)
            last = res[kind][-1]
            print(f"[main] {network} h{h} {kind} dDV_strat={last['dDV_strat']:+.3f} "
                  f"share_all={last['share_all']:.3f} ({last['seconds']}s/draw)", flush=True)
    return res


def run_pc(d: dict, network: str, subject: int, draws: int) -> dict:
    others = [s for s in range(1, 9) if s != subject]
    pos = d["unit_pos"]
    n_pos = len(np.unique(pos))
    groups = {}
    for stat, field in (("p90", "pc"), ("meanr", "pc_meanr")):
        prefs = [np.load(out_stem(network, s) + "_prefs.npz")[stat][0] for s in others]
        dbar = np.nanmean(np.concatenate(prefs, 0), axis=0)  # mean over subjects x hemis
        score = dbar[d["unit_id"]]
        rows = np.arange(len(d["unit_id"]))[np.isfinite(score)]
        top, bot = {}, {}
        for p in np.unique(pos):
            rp = rows[pos[rows] == p]
            rp = rp[np.argsort(score[rp])]
            q = max(1, len(rp) // 4)
            bot[p], top[p] = rp[:q], rp[-q:]
        groups[field] = (top, bot)
    allp = pools_by_pos(pos, np.arange(len(d["unit_id"])))  # negative control: all units
    res = {"pc": [], "pc_meanr": [], "neg": [], "n_other_subjects": len(others)}
    for h in (0, 1):
        lab, nc = d["label"][d["vox"][h]], d["nc"][d["vox"][h]]
        C_tr, C_te = d["C_train"][h], d["C_test"][h]
        n_per = -(-C_tr.shape[1] // (2 * n_pos))
        for i in range(draws):
            for k, (field, (top, bot)) in enumerate(groups.items()):
                rng = np.random.default_rng([h, i, 2, k])
                ra, rb = quota(top, bot, n_per, rng)
                s = summarize(*assign(C_tr, C_te, ra, rb), lab, nc)
                s.update(hemi=h, draw=i, n_units=int(len(ra) + len(rb)))
                res[field].append(s)
            rng = np.random.default_rng([h, i, 3])
            half_a, half_b = {}, {}
            for p, rp in allp.items():
                perm = rng.permutation(rp)
                half_a[p], half_b[p] = perm[: len(rp) // 2], perm[len(rp) // 2:]
            ra, rb = quota(half_a, half_b, n_per, rng)
            s = summarize(*assign(C_tr, C_te, ra, rb), lab, nc)
            s.update(hemi=h, draw=i, n_units=int(len(ra) + len(rb)))
            res["neg"].append(s)
        print(f"[pc] {network} subj{subject:02d} h{h} "
              f"pc dDV_strat={np.mean([x['dDV_strat'] for x in res['pc'] if x['hemi'] == h]):+.3f} "
              f"neg={np.mean([x['dDV_strat'] for x in res['neg'] if x['hemi'] == h]):+.3f}",
              flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--mode", choices=("main", "pc"), default="main")
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--draws-unforced", type=int, default=3)
    args = p.parse_args()
    t0 = time.time()
    stem = out_stem(args.network, args.subject)
    out = f"{stem}__{args.mode}.json"
    if os.path.exists(out):
        print(f"[mapping] exists: {out}", flush=True)
        return
    d = load(args.network, args.subject)
    print(f"[mapping] {args.network} subj{args.subject:02d} units={len(d['unit_id'])}/"
          f"{d['n_units_all']} voxels lh={len(d['vox'][0])} rh={len(d['vox'][1])} "
          f"loaded {time.time() - t0:.0f}s", flush=True)
    if args.mode == "main":
        np.savez(stem + "_prefs.tmp.npz", **unit_prefs(d))
        os.replace(stem + "_prefs.tmp.npz", stem + "_prefs.npz")
        res = run_main(d, args.network, args.draws, args.draws_unforced)
    else:
        res = run_pc(d, args.network, args.subject, max(1, args.draws // 2))
    res.update(network=args.network, subject=args.subject, mode=args.mode,
               n_units_responsive=int(len(d["unit_id"])), n_units_all=int(d["n_units_all"]),
               provenance=d["provenance"], seconds=round(time.time() - t0, 1))
    with open(out + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out + ".tmp", out)
    print(f"[mapping] wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
