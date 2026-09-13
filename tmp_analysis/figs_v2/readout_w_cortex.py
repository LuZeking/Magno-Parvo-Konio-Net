"""Aggregate the cortex-optimal readout weight (src/analysis/encoding/readout_w_cortex.py)
and set it against the behaviour-optimal weight (results/fit_w_observers) on the same axis.

Per (model, subject, ROI): the median over reliable voxels (nc > 0.1, ≥ 20 voxels) of
test R² at each of 21 mixing weights w; w*_cortex = argmax. Seeds are averaged per
subject; arms are diet (dkl_bcs) and no diet (dkl_base), final checkpoints.
Behaviour: w*_trial per human observer (trial-fit, diet final checkpoints, 3 seeds) and
the shape-bias-rate sweep over w (results/fusion_sweep).

Writes results/figures_v2/readout_w_cortex.json, appends Extended Data Table 12 to
results/figures_v2/ed_tables.md and draws results/figures_v2/figE5_readout_w.{pdf,png}.
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/readout_w_cortex.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

import style as S

RES_DIR = os.environ.get("READOUT_W_DIR", "/scratch/byrong/mpk/readout_w/results/2026-09-13_w21_last")
NC_MIN, MIN_VOX = 0.1, 20
VIS = {"V1": [1, 2], "V2": [3, 4], "V3": [5, 6], "hV4": [7]}
KAST = {"ventral (VO, PHC)": [8, 9, 10, 11], "lateral (LO, TO)": [12, 13, 14, 15], "dorsal (V3A/B, IPS0–2)": [16, 17, 18, 19, 20]}
ROI_ORDER = list(VIS) + list(KAST)
ARM_OF = {"dkl_bcs": "diet", "dkl_base": "no diet"}


def load_all():
    out = {}
    for f in sorted(glob.glob(os.path.join(RES_DIR, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        m = str(z["model_name"]); s = int(z["subject"])
        arm = ARM_OF["dkl_bcs" if m.startswith("dkl_bcs") else "dkl_base"]
        seed = int(re.search(r"seed(\d)", m).group(1))
        out[(arm, seed, s)] = z
    return out


def curves(z):
    """median R² over reliable voxels per w, per ROI; plus per-voxel argmax stats."""
    nc = z["noise_ceiling"]; r2 = z["r2"]; grid = z["grid"]
    ok = (nc > NC_MIN) & np.isfinite(r2).all(0)
    vis, kas = z["roi_prf-visualrois"], z["roi_Kastner2015"]
    res = {}
    for name, labs in list(VIS.items()) + list(KAST.items()):
        roi = vis if name in VIS else kas
        v = ok & np.isin(roi, labs)
        if v.sum() < MIN_VOX:
            continue
        cur = np.median(r2[:, v], axis=1)
        cc = float(np.median(z["r2_concat"][v]))
        vox_w = grid[np.argmax(r2[:, v], axis=0)]
        res[name] = dict(curve=cur, w_star=float(grid[int(np.argmax(cur))]), concat=cc,
                         frac_parvo=float(np.mean(vox_w <= 0.25)), frac_magno=float(np.mean(vox_w >= 0.75)),
                         vox_w_median=float(np.median(vox_w)), n_vox=int(v.sum()))
    return grid, res


def main():
    data = load_all()
    if not data:
        raise SystemExit(f"no results in {RES_DIR}")
    grid = None
    per = {}  # (arm, subject) -> roi -> list over seeds of dicts
    for (arm, seed, s), z in data.items():
        grid, res = curves(z)
        for roi, d in res.items():
            per.setdefault((arm, s), {}).setdefault(roi, []).append(d)
    arms = sorted({k[0] for k in per})
    subs = sorted({k[1] for k in per})
    agg = {arm: {} for arm in arms}
    for arm in arms:
        for roi in ROI_ORDER:
            rows = []
            for s in subs:
                lst = per.get((arm, s), {}).get(roi)
                if not lst:
                    continue
                cur = np.mean([d["curve"] for d in lst], 0)
                rows.append(dict(subject=s, curve=cur, w_star=float(grid[int(np.argmax(cur))]),
                                 w_star_seeds=[d["w_star"] for d in lst], concat=float(np.mean([d["concat"] for d in lst])),
                                 frac_parvo=float(np.mean([d["frac_parvo"] for d in lst])), frac_magno=float(np.mean([d["frac_magno"] for d in lst])),
                                 vox_w_median=float(np.mean([d["vox_w_median"] for d in lst])), n_seeds=len(lst)))
            if rows:
                agg[arm][roi] = rows

    # behaviour: trial-fit w per observer (diet, final ckpt, 3 seeds) and the rate sweep
    beh_w = {}
    for f in sorted(glob.glob("results/fit_w_observers/dkl_bcs_seed?_final_wfit.json")):
        d = json.load(open(f))
        for obs, o in d["observers"].items():
            beh_w.setdefault(obs, []).append(o["trial_fit"]["w_hat"])
    beh_w = {k: float(np.mean(v)) for k, v in beh_w.items()}
    sweeps = []
    for f in sorted(glob.glob("results/fusion_sweep/dkl_bcs_seed?_final_wsweep.json")):
        sw = json.load(open(f))["sweep"]
        sweeps.append(([p["w"] for p in sw], [p["shape_bias"] for p in sw]))
    sw_w = np.array(sweeps[0][0]); sw_rate = np.mean([r for _, r in sweeps], 0)
    sweeps_nd = []
    for f in sorted(glob.glob("results/fusion_sweep/dkl_base_seed?_final_wsweep.json")):
        sw = json.load(open(f))["sweep"]
        sweeps_nd.append([p["shape_bias"] for p in sw])
    rate_at = lambda w: float(np.interp(w, sw_w, sw_rate))

    # ---------- JSON + table ----------
    out = {"grid": grid.tolist(), "behaviour_w_trial_per_observer": beh_w,
           "behaviour_w_trial_median": float(np.median(list(beh_w.values()))), "arms": {}}
    lines = ["**Extended Data Table 12 | The readout weight cortex prefers against the one behaviour prefers.** For each mixing weight w the classifier's own input z(w) = w·y₁ + (1 − w)·y₂ (y₁ magno stream output, y₂ parvo stream output, 512-d each) is the design of a ridge encoding model of every nsdgeneral voxel (same split, folds and alpha grid as the banded model; Methods 4.5). w*_cortex is the w that maximises the median test R² over reliable voxels of the ROI (mean over seeds per subject; median [range] over 8 subjects). \"R² at w = 0.5 / max\" is the median R² at the paper's readout weight relative to the best w; \"concat\" is the median R² of the 1,024-d concatenation, the ceiling for any mixture. \"Voxels magno / parvo\" is the fraction of the ROI's voxels whose own best w is ≥ 0.75 / ≤ 0.25. The behaviour-optimal weight (trial fit to each of 10 human observers' cue-conflict choices, diet final checkpoints, 3 seeds) is given in the last row for comparison, with the cue-conflict shape-bias rate the diet network produces at each weight.", "",
             "| arm | ROI | w*_cortex median [range] | R² at w = 0.5 / max | max R² (median) | concat R² | voxels best w ≥ 0.75 | voxels best w ≤ 0.25 | rate at w*_cortex | n |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    i05 = int(np.argmin(np.abs(grid - 0.5)))
    for arm in arms:
        out["arms"][arm] = {}
        for roi in ROI_ORDER:
            rows = agg[arm].get(roi)
            if not rows:
                continue
            ws = np.array([r["w_star"] for r in rows]); cur = np.array([r["curve"] for r in rows])
            ratio = cur[:, i05] / cur.max(1)
            out["arms"][arm][roi] = dict(w_star_per_subject=ws.tolist(), w_star_median=float(np.median(ws)),
                                          curve_mean=cur.mean(0).tolist(), curve_per_subject=cur.tolist(),
                                          r2_at_half_over_max=ratio.tolist(), max_r2=cur.max(1).tolist(),
                                          concat=[r["concat"] for r in rows], frac_magno=[r["frac_magno"] for r in rows],
                                          frac_parvo=[r["frac_parvo"] for r in rows], subjects=[r["subject"] for r in rows])
            lines.append(f"| {arm} | {roi} | {np.median(ws):.2f} [{ws.min():.2f}, {ws.max():.2f}] | {np.median(ratio):.3f} | {np.median(cur.max(1)):.4f} | "
                         f"{np.median([r['concat'] for r in rows]):.4f} | {np.mean([r['frac_magno'] for r in rows]):.2f} | {np.mean([r['frac_parvo'] for r in rows]):.2f} | "
                         f"{rate_at(np.median(ws)) if arm == 'diet' else float('nan'):.2f} | {len(rows)} |")
    bw = np.array(list(beh_w.values()))
    lines.append(f"| behaviour (diet) | 10 human observers, trial fit | {np.median(bw):.2f} [{bw.min():.2f}, {bw.max():.2f}] | — | — | — | — | — | {rate_at(np.median(bw)):.2f} | 10 |")
    lines.append("")
    json.dump(out, open("results/figures_v2/readout_w_cortex.json", "w"), indent=1)
    md = open("results/figures_v2/ed_tables.md", encoding="utf8").read()
    if "Extended Data Table 12" in md:
        md = md[:md.index("**Extended Data Table 12")]
    open("results/figures_v2/ed_tables.md", "w", encoding="utf8").write(md.rstrip() + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------- figure ----------
    fig = plt.figure(figsize=(S.FULL, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.25], wspace=0.42, left=0.07, right=0.99, top=0.86, bottom=0.2)
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # (a) R² curves, diet arm
    show = ["V1", "hV4", "ventral (VO, PHC)", "dorsal (V3A/B, IPS0–2)"]
    cols = {"V1": "#B0B0B0", "hV4": "#5A5A5A", "ventral (VO, PHC)": S.DIET, "dorsal (V3A/B, IPS0–2)": "#009E73"}
    for roi in show:
        rows = agg["diet"].get(roi)
        if not rows:
            continue
        cur = np.array([r["curve"] for r in rows])
        curn = cur / cur.max(1, keepdims=True)
        ax_a.plot(grid, curn.mean(0), "-", color=cols[roi], lw=1.6, label=roi.split(" (")[0])
        ax_a.fill_between(grid, curn.mean(0) - curn.std(0, ddof=1) / np.sqrt(len(rows)), curn.mean(0) + curn.std(0, ddof=1) / np.sqrt(len(rows)), color=cols[roi], alpha=0.15, lw=0)
    ax_a.set_xlabel("readout weight w (0 = parvo only, 1 = magno only)"); ax_a.set_ylabel("median R² / best over w")
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1.05)
    ax_a.legend(loc="lower left", ncol=1)
    S.letter(ax_a, "a", title="diet arm: cortex prefers a parvo-heavy readout")
    # (b) w* per ROI per subject, both arms
    y = 0
    yt, yl = [], []
    for roi in ROI_ORDER:
        for arm, c, dy in [("no diet", S.NODIET, 0.18), ("diet", S.DIET, -0.18)]:
            rows = agg.get(arm, {}).get(roi)
            if not rows:
                continue
            ws = np.array([r["w_star"] for r in rows])
            ax_b.plot(ws + np.random.default_rng(0).uniform(-0.012, 0.012, len(ws)), np.full(len(ws), y + dy), "o", color=c, ms=3.2, mec="white", mew=0.5, alpha=0.9, zorder=4)
            ax_b.plot([np.median(ws)] * 2, [y + dy - 0.14, y + dy + 0.14], "-", color=c, lw=2, zorder=5)
        yt.append(y); yl.append(roi.split(" (")[0]); y -= 1
    if beh_w:
        ax_b.plot(list(beh_w.values()), np.full(len(beh_w), y), "o", color=S.HUMAN, ms=3.2, mec="white", mew=0.5, zorder=4)
        ax_b.plot([np.median(list(beh_w.values()))] * 2, [y - 0.14, y + 0.14], "-", color=S.HUMAN, lw=2, zorder=5)
        yt.append(y); yl.append("behaviour\n(10 observers)")
    ax_b.axhline(y + 0.5, color=S.RULE, lw=0.7)
    S.refline(ax_b, x=0.5)
    ax_b.set_yticks(yt); ax_b.set_yticklabels(yl, fontsize=6.6)
    ax_b.set_xlim(-0.03, 1.03); ax_b.set_xlabel("optimal readout weight w*")
    ax_b.plot([], [], "o", color=S.NODIET, ms=3, label="no diet"); ax_b.plot([], [], "o", color=S.DIET, ms=3, label="diet")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3)
    S.letter(ax_b, "b", title="the two optima sit at opposite ends")
    # (c) rate curve with both optima
    ax_c.plot(sw_w, sw_rate, "-", color=S.DIET, lw=1.8, label="diet, shape-bias rate")
    if sweeps_nd:
        ax_c.plot(sw_w, np.mean(sweeps_nd, 0), "-", color=S.NODIET, lw=1.4, label="no diet")
    S.refline(ax_c, y=0.5)
    ext = [r["w_star"] for roi in KAST for r in agg["diet"].get(roi, [])]
    early = [r["w_star"] for roi in ("V1",) for r in agg["diet"].get(roi, [])]
    for vals, c, lab, yy in [(ext, S.DIET, "cortex-optimal w*\n(extrastriate, diet)", 0.30), (early, "#B0B0B0", "cortex-optimal w*\n(V1, diet)", 0.20), (list(beh_w.values()), S.HUMAN, "behaviour-optimal w*\n(10 observers)", 0.10)]:
        if not vals:
            continue
        m = float(np.median(vals))
        ax_c.plot(vals, np.full(len(vals), yy), "|", color=c, ms=7, mew=1.1, zorder=4)
        ax_c.plot([m], [rate_at(m)], "o", color=c, ms=5, mec="white", mew=0.7, zorder=6)
        ax_c.plot([m, m], [yy, rate_at(m)], ":", color=c, lw=0.8, zorder=3)
        ax_c.text(min(m + 0.03, 0.72), yy, lab, fontsize=6.2, color=c, va="center", ha="left")
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0.05, 1.0)
    ax_c.set_xlabel("readout weight w"); ax_c.set_ylabel("cue-conflict shape bias")
    ax_c.legend(loc="upper left", ncol=1)
    S.letter(ax_c, "c", title="same axis: what cortex wants, what behaviour wants")
    S.save(fig, "figE5_readout_w")


if __name__ == "__main__":
    main()
