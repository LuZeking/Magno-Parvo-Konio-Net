"""Extended Data Figure 3 — the cortical gradient against its nuisance axes.

  (a) the raw statistic (median over voxels of U_m − U_p) per ROI, beside the
      normalised index of Fig. 3a, for the untrained network and the two trained
      arms (3 seeds each).
  (b) the gradient hV4 − mean(V1v, V1d) computed within each of NSD's five
      eccentricity bands (`prf-eccrois`): V1 and hV4 voxels compared only with
      voxels of the same eccentricity band. One point per subject, per band.
  (c) the same gradient within pRF-size tertiles, tertile boundaries set per
      subject on the pooled V1 + hV4 voxels using NSD's pRF size estimates
      (voxels with pRF R² > 5%).

Both arms, both statistics; ≥ 20 voxels per ROI cell or the cell is dropped.
Sources: results/encoding_cc/ (trained arms), results/encoding/ (untrained), NSD pRF maps.
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/figE3_cortex_controls.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

import style as S
from src.analysis.encoding.stats import load_results
from src.analysis.encoding import nsd_data

NC_MIN, MIN_VOX, R2_MIN = 0.1, 20, 5.0
VIS = [1, 2, 3, 4, 5, 6, 7]
ROI = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
BANDS = ["0–0.5°", "0.5–1°", "1–2°", "2–4°", "> 4°"]
ARMS = {"untrained": ("results/encoding", ["untrained_seed0"], S.UNTRAINED),
        "no diet": ("results/encoding_cc", [f"dkl_base_seed{s}_trained" for s in range(3)], S.NODIET),
        "diet": ("results/encoding_cc", [f"dkl_bcs_seed{s}_trained" for s in range(3)], S.DIET)}

_prf_cache = {}


def prf_size(subject):
    if subject not in _prf_cache:
        mask = nsd_data.load_mask(subject, "nsdgeneral")
        size = nsd_data.load_prf(subject, "size", mask); r2 = nsd_data.load_prf(subject, "R2", mask)
        _prf_cache[subject] = (size, r2)
    return _prf_cache[subject]


def index(z, raw):
    um, up = z["unique_magno"], z["unique_parvo"]
    ok = (z["noise_ceiling"] > NC_MIN) & ((um + up) > 1e-3) & np.isfinite(um) & np.isfinite(up)
    val = (um - up) if raw else (um - up) / (um + up)
    return val, ok


def roi_median(val, sel):
    return np.median(val[sel]) if sel.sum() >= MIN_VOX else np.nan


def profiles(out_dir, model, raw):
    res = load_results(out_dir, model)
    prof = np.full((len(res), 7), np.nan)
    for i, s in enumerate(sorted(res)):
        val, ok = index(res[s], raw)
        roi = res[s]["roi_prf-visualrois"]
        for j, lab in enumerate(VIS):
            prof[i, j] = roi_median(val, ok & (roi == lab))
    return prof


def grad_by_strata(out_dir, model, raw, strat):
    """strat = 'ecc' (5 bands) or 'size' (3 tertiles). Returns subjects × strata gradient."""
    res = load_results(out_dir, model)
    n_str = 5 if strat == "ecc" else 3
    out = np.full((len(res), n_str), np.nan)
    for i, s in enumerate(sorted(res)):
        z = res[s]; val, ok = index(z, raw); roi = z["roi_prf-visualrois"]
        if strat == "ecc":
            lab = z["roi_prf-eccrois"]; strata = [(lab == b) for b in range(1, 6)]
        else:
            size, r2 = prf_size(s)
            good = ok & (r2 > R2_MIN) & np.isfinite(size) & (size > 0)
            pool = good & np.isin(roi, [1, 2, 7])
            q = np.quantile(size[pool], [1 / 3, 2 / 3])
            strata = [good & (size <= q[0]), good & (size > q[0]) & (size <= q[1]), good & (size > q[1])]
        for k, st in enumerate(strata):
            v1 = np.nanmean([roi_median(val, ok & st & (roi == 1)), roi_median(val, ok & st & (roi == 2))])
            hv4 = roi_median(val, ok & st & (roi == 7))
            out[i, k] = hv4 - v1
    return out


def main():
    fig = plt.figure(figsize=(S.FULL, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32, left=0.08, right=0.985, top=0.92, bottom=0.09)

    # (a) raw and normalised profiles side by side
    for j, (raw, ylab, ttl) in enumerate([(False, "normalised index\nmedian of (Um − Up)/(Um + Up)", "Normalised index (as Fig. 3a)"),
                                          (True, "raw statistic\nmedian of Um − Up (units of r² / NC)", "Raw statistic")]):
        ax = fig.add_subplot(gs[0, j]); S.refline(ax, y=0)
        for k, (d, ms_, c) in ARMS.items():
            prof = np.stack([profiles(d, m, raw) for m in ms_]).mean(0)
            mu, se = prof.mean(0), prof.std(0, ddof=1) / np.sqrt(prof.shape[0])
            ax.fill_between(range(7), mu - se, mu + se, color=c, alpha=0.18, lw=0)
            ax.plot(range(7), mu, "-o", color=c, lw=1.8, ms=3, mec="white", mew=0.5,
                    ls=(0, (3, 2)) if k == "untrained" else "-", label=k + (" (3 seeds)" if k != "untrained" else " (diet front end)"))
            print(f"  (a) {'raw' if raw else 'norm':4s} {k:10s} " + " ".join(f"{v:+.3f}" for v in mu))
        ax.set_xticks(range(7)); ax.set_xticklabels(ROI); ax.set_ylabel(ylab)
        if j == 0:
            ax.legend(loc="lower left", fontsize=6.2, handlelength=1.6)
        S.letter(ax, "ab"[j], dx=-0.07, dy=0.03, title=ttl)

    # (b) gradient within eccentricity bands, (c) within pRF-size tertiles
    for j, (strat, labels, xlab, ttl) in enumerate([("ecc", BANDS, "pRF eccentricity band (NSD prf-eccrois)", "Gradient within eccentricity bands"),
                                                    ("size", ["small", "medium", "large"], "pRF-size tertile (per subject, V1 + hV4 voxels)", "Gradient within pRF-size tertiles")]):
        ax = fig.add_subplot(gs[1, j]); S.refline(ax, y=0)
        for a_i, k in enumerate(["no diet", "diet"]):
            d, ms_, c = ARMS[k]
            g = np.nanmean(np.stack([grad_by_strata(d, m, False, strat) for m in ms_]), 0)   # seeds × subj × strata → subj × strata
            for b in range(g.shape[1]):
                x = b + (a_i - 0.5) * 0.28
                v = g[:, b]; v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                ax.plot(np.full(len(v), x) + np.linspace(-0.05, 0.05, len(v)), v, "o", color=c, ms=3, mec="white", mew=0.5, alpha=0.9, zorder=3)
                ax.plot([x - 0.11, x + 0.11], [v.mean()] * 2, "-", color=c, lw=2.0, zorder=4)
                print(f"  ({'c' if j else 'b'}) {k:8s} {labels[b]:8s} n={len(v)} mean {v.mean():+.3f} range {v.min():+.3f}..{v.max():+.3f} neg {(v < 0).sum()}/{len(v)}")
            gd = g
            if k == "diet":
                gn = np.nanmean(np.stack([grad_by_strata(ARMS['no diet'][0], m, False, strat) for m in ARMS['no diet'][1]]), 0)
                for b in range(g.shape[1]):
                    both = np.isfinite(gd[:, b]) & np.isfinite(gn[:, b])
                    if both.sum() >= 5:
                        p = sps.wilcoxon(gd[both, b] - gn[both, b]).pvalue
                        ax.text(b, -1.20, f"{(gd[both, b] < gn[both, b]).sum()}/{both.sum()}\np = {p:.3f}", ha="center", va="bottom", fontsize=5.4, color=S.MUTED)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6.4)
        ax.set_xlim(-0.6, len(labels) - 0.4); ax.set_ylim(-1.22, 0.45)
        ax.set_xlabel(xlab); ax.set_ylabel("gradient hV4 − mean(V1v, V1d)\nwithin stratum (normalised)")
        h = [plt.Line2D([], [], color=S.DIET, lw=2, label="diet"), plt.Line2D([], [], color=S.NODIET, lw=2, label="no diet")]
        ax.legend(handles=h, loc="upper right", fontsize=6.2, handlelength=1.3)
        S.note(ax, "labels: subjects with diet < no diet; exact Wilcoxon p", x=0.02, y=0.97, va="top", size=5.6)
        S.letter(ax, "cd"[j], dx=-0.07, dy=0.03, title=ttl)

    S.save(fig, "figE3_cortex_controls")


if __name__ == "__main__":
    main()
