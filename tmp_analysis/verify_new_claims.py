#!/usr/bin/env python
"""Re-derive every number the draft states for the artefact floor (E2.1) and the
progressive-transition OOD arm (E6.2), straight from the result JSONs.

Exits non-zero if any draft number drifts from what the files say. Run it after
touching either section, and after the diet-arm half of the recovery analysis
lands (job 26442320) -- the E2.1 numbers are currently the no-diet geometry only.

    python tmp_analysis/verify_new_claims.py
"""
import glob
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import spearmanr, wilcoxon

FAILS: list[str] = []


def chk(name: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'BAD'} {name:38s} got {got:+.4f}  draft {want:+.4f}")
    if not ok:
        FAILS.append(name)


def hv4_minus_v1(v: np.ndarray) -> float:
    """§2.2's statistic: hV4 minus the mean of V1v and V1d.

    visualrois.py:72-75. NOT the value at hV4 and NOT a median over the ROI
    profile -- the floor was first published as the latter (+0.126) and had to
    be corrected to +0.051. All four of §2.2's published numbers reproduce
    under this reading and under no other.
    """
    return float(v[6] - v[:2].mean())


def stats_from(rows: list[dict], prefix: str) -> tuple[float, float, float]:
    """(ratio-index contrast, difference contrast, monotonicity rho).

    Both statistics run on nc-divided unique variances, matching production:
    variance_partition returns unique_magno = unique_m / nc, and visualrois.py
    reads that. "normalised" in §2.2 means normalised by (um+up), NOT by nc.
    """
    um = np.array([r[f"{prefix}unique_m"] for r in rows])
    up = np.array([r[f"{prefix}unique_p"] for r in rows])
    ratio = (um - up) / (um + up)
    with warnings.catch_warnings():
        # condition B's TRUTH is constant by construction (solved to zero), so
        # its rho is undefined. Only the estimated rho is ever read.
        warnings.simplefilter("ignore")
        rho = float(spearmanr(np.arange(len(ratio)), ratio).statistic)
    return hv4_minus_v1(ratio), hv4_minus_v1(um - up), rho


# Floors are geometry-specific, so BOTH arms are pinned. (arm, condition):
# (ratio contrast, diff contrast, rho, n positive on ratio)
FLOORS = {
    ("dkl_base", "B_manufacture"): (0.051, 0.0009, 0.68, 8),
    ("dkl_base", "A_recovery"): (1.700, 0.499, 1.000, 8),
    ("dkl_bcs", "B_manufacture"): (0.038, -0.0021, 0.80, 8),
    ("dkl_bcs", "A_recovery"): (1.476, 0.480, 1.000, 8),
}
# est/true on each geometry: the ratio index inflates, the difference compresses.
# Opposite directions on BOTH geometries is why "-0.862 is a floor" is withdrawn.
FIDELITY = {"dkl_base": (1.255, 0.713), "dkl_bcs": (1.151, 0.686)}


def check_recovery() -> None:
    for arm in ("dkl_base", "dkl_bcs"):
        _check_recovery_arm(arm)
    # The two geometries must stay indistinguishable, else §2.2's claim that the
    # floor is a property of the instrument rather than the fitted model fails.
    from scipy.stats import mannwhitneyu
    a = _ratio_contrasts("dkl_base", "B_manufacture")
    b = _ratio_contrasts("dkl_bcs", "B_manufacture")
    pv = mannwhitneyu(a, b).pvalue
    print("[floor geometry comparison]")
    chk("no-diet floor", a.mean(), 0.051, 0.001)
    chk("diet floor", b.mean(), 0.038, 0.001)
    print(f"  {'OK ' if pv > 0.05 else 'BAD'} not distinguishable at 8v8: "
          f"p={pv:.3f} (draft .33)")
    if pv <= 0.05:
        FAILS.append("floor geometries distinguishable")


def _files(arm: str) -> list[str]:
    files = sorted(glob.glob(
        f"results/vp_recovery_realnc/vp_recovery_{arm}_seed0_trained_subj*.json"))
    assert len(files) == 8, f"expected 8 {arm} subjects, found {len(files)}"
    return files


def _ratio_contrasts(arm: str, cond: str) -> np.ndarray:
    return np.array([stats_from(json.load(open(f))["conditions"][cond]["rows"],
                                "est_")[0] for f in _files(arm)])


def _check_recovery_arm(arm: str) -> None:
    files = _files(arm)
    want = [(c, *FLOORS[(arm, c)]) for c in ("B_manufacture", "A_recovery")]
    for cond, wr, wd, wrho, wpos in want:
        vals = [stats_from(json.load(open(f))["conditions"][cond]["rows"], "est_")
                for f in files]
        RAT, DIF, RHO = (np.array(x) for x in zip(*vals))
        print(f"[{arm} {cond}]")
        chk("normalised (ratio) contrast", RAT.mean(), wr, 0.001)
        chk("raw (difference) contrast", DIF.mean(), wd, 0.001)
        chk("monotonicity rho median", float(np.median(RHO)), wrho, 0.005)
        pos = int((RAT > 0).sum())
        print(f"  {'OK ' if pos == wpos else 'BAD'} ratio sign {pos}/8 positive "
              f"(draft {wpos}/8), wilcoxon p={wilcoxon(RAT).pvalue:.4f}")
        if pos != wpos:
            FAILS.append(f"{cond} sign")
        if cond == "B_manufacture":
            lo, hi = {"dkl_base": (0.007, 0.094),
                      "dkl_bcs": (0.014, 0.065)}[arm]
            chk("B ratio range low", RAT.min(), lo, 0.001)
            chk("B ratio range high", RAT.max(), hi, 0.001)
            # the whole point: the truth really is solved to zero
            tru = [stats_from(json.load(open(f))["conditions"][cond]["rows"], "true_")
                   for f in files]
            mx = max(abs(t[0]) for t in tru)
            print(f"  {'OK ' if mx == 0 else 'BAD'} true contrast is exactly zero "
                  f"(max |true| = {mx:.2e})")
            if mx != 0:
                FAILS.append("B truth nonzero")
            p = wilcoxon(DIF).pvalue
            wp, wn = {"dkl_base": (0.84, 4), "dkl_bcs": (0.15, 2)}[arm]
            ok = p > 0.05 and int((DIF > 0).sum()) == wn
            print(f"  {'OK ' if ok else 'BAD'} raw floor n.s. p={p:.4f} "
                  f"(draft {wp}), {int((DIF > 0).sum())}/8 positive (draft {wn}/8)")
            if not ok:
                FAILS.append(f"{arm} raw floor")
        if cond == "A_recovery":
            # magnitude fidelity: the two statistics are biased OPPOSITE ways,
            # which is why "-0.862 is a floor" was withdrawn.
            tru = [stats_from(json.load(open(f))["conditions"][cond]["rows"], "true_")
                   for f in files]
            TR = np.array([t[0] for t in tru]); TD = np.array([t[1] for t in tru])
            wi, wc = FIDELITY[arm]
            chk("ratio inflation est/true", RAT.mean() / TR.mean(), wi, 0.01)
            chk("diff compression est/true", DIF.mean() / TD.mean(), wc, 0.01)

    # The measured profile the simulation is conditioned on. Its DECLINE is the
    # whole point: the first run used an ascending ramp and inverted the sign.
    nc = [0.408, 0.413, 0.392, 0.351, 0.352, 0.320, 0.322]
    r = spearmanr(np.arange(7), nc)
    print("[NC profile]")
    chk("rho", r.statistic, -0.893, 0.002)
    print(f"  OK  p={r.pvalue:.4f} (draft states .007)")


# §2.2's table as the draft now states it, from results/encoding_cc/.
# (statistic, seed): (diet, no-diet n=8, no-diet n=7)
TABLE_2_2 = {
    ("normalised", 0): (-0.834, -0.079, -0.054),
    ("normalised", 1): (-0.785, -0.096, -0.085),
    ("normalised", 2): (-0.745, -0.049, -0.031),
    ("raw", 0): (-0.166, -0.027, -0.021),
    ("raw", 1): (-0.145, -0.031, -0.030),
    ("raw", 2): (-0.153, -0.023, -0.021),
}

# Declared source. results/encoding/ holds the diet arm at seed 0 only; where
# the two overlap they are numerically identical. The superseded table
# (-0.862/-0.802/-0.781, -0.176/-0.151/-0.160, dz -11.70) does not regenerate
# under any configuration searched -- see Extended Data E2.2.
ENCODING_DIR = "results/encoding_cc"


def roi_profile(out_dir: str, tag: str, ratio: bool,
                drop: tuple[int, ...] = ()) -> tuple[list[int], np.ndarray]:
    """visualrois.py convention: nc>0.1, (um+up)>1e-3, >=20 vox, per-ROI median.

    `ratio` picks §2.2's statistic: True = normalised (the per-voxel ratio
    index), False = raw (the difference). Both run on the nc-divided arrays.
    """
    from src.analysis.encoding.stats import load_results
    res = load_results(out_dir, tag)
    subs = [s for s in sorted(res) if s not in drop]
    out = np.full((len(subs), 7), np.nan)
    for i, s in enumerate(subs):
        z = res[s]
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        for j, lab in enumerate([1, 2, 3, 4, 5, 6, 7]):
            v = (roi == lab) & (nc > 0.1) & ((um + up) > 1e-3)
            if v.sum() >= 20:
                r = (um[v] - up[v]) / (um[v] + up[v]) if ratio else um[v] - up[v]
                out[i, j] = np.median(r[np.isfinite(r)])
    return subs, out


def check_production_statistic() -> None:
    """§2.2's twelve stated values, plus the n=7 subj01 sensitivity.

    Locks BOTH the statistic (hV4 - V1) and the source directory. If either
    drifts, this fails rather than silently reporting a different quantity --
    which is exactly what went unnoticed in the superseded table.
    """
    print(f"[production §2.2 table, {ENCODING_DIR}]")
    for (stat, seed), (wd, wn8, wn7) in TABLE_2_2.items():
        ratio = stat == "normalised"
        _, pd_ = roi_profile(ENCODING_DIR, f"dkl_bcs_seed{seed}_trained", ratio)
        _, pn8 = roi_profile(ENCODING_DIR, f"dkl_base_seed{seed}_trained", ratio)
        _, pn7 = roi_profile(ENCODING_DIR, f"dkl_base_seed{seed}_trained",
                             ratio, drop=(1,))
        cd = np.array([hv4_minus_v1(r) for r in pd_])
        c8 = np.array([hv4_minus_v1(r) for r in pn8])
        c7 = np.array([hv4_minus_v1(r) for r in pn7])
        chk(f"{stat[:4]} seed{seed} diet", float(np.nanmean(cd)), wd, 0.001)
        chk(f"{stat[:4]} seed{seed} nodiet n=8", float(np.nanmean(c8)), wn8, 0.001)
        chk(f"{stat[:4]} seed{seed} nodiet n=7", float(np.nanmean(c7)), wn7, 0.001)
        ok = np.isfinite(cd) & np.isfinite(c8)
        p8 = wilcoxon(cd[ok] - c8[ok]).pvalue
        sep = int((cd[ok] < c8[ok]).sum())
        good = abs(p8 - 0.0078125) < 1e-6 and sep == ok.sum()
        print(f"  {'OK ' if good else 'BAD'} {stat[:4]} seed{seed} p={p8:.4f} "
              f"separation {sep}/{ok.sum()} (draft .0078, 8/8)")
        if not good:
            FAILS.append(f"{stat} seed{seed} p/sep")

    # The floor-to-effect ratios §2.2 quotes as 16x and 1.5x.
    _, pd_ = roi_profile(ENCODING_DIR, "dkl_bcs_seed0_trained", True)
    _, pn_ = roi_profile(ENCODING_DIR, "dkl_base_seed0_trained", True)
    diet = abs(float(np.nanmean([hv4_minus_v1(r) for r in pd_])))
    nod = abs(float(np.nanmean([hv4_minus_v1(r) for r in pn_])))
    check_displacement()


# §2.2's statistic for both arms: real contrast minus the SAME subject's own
# zero-truth floor. A ratio of magnitudes would be wrong -- the floor is a
# signed positive bias, not a noise level, so it cannot manufacture a negative
# observation and its magnitude is not a tolerance. (draft: displacement, sign)
DISPLACEMENT = {
    ("dkl_bcs", ()): (-0.871, 8),
    ("dkl_base", ()): (-0.130, 8),
    ("dkl_base", (1,)): (-0.103, 7),
}


def check_displacement() -> None:
    print("  [paired displacement: real - own zero-truth floor]")
    for (arm, drop), (want, wn) in DISPLACEMENT.items():
        tag = f"{arm}_seed0_trained"
        subs_r, pr = roi_profile(ENCODING_DIR, tag, True, drop=drop)
        real = {s: hv4_minus_v1(pr[i]) for i, s in enumerate(subs_r)}
        floor = {}
        for f in _files(arm):
            d = json.load(open(f))
            floor[d["subject"]] = stats_from(
                d["conditions"]["B_manufacture"]["rows"], "est_")[0]
        subs = [s for s in sorted(set(real) & set(floor)) if s not in drop]
        disp = np.array([real[s] - floor[s] for s in subs])
        label = f"{arm}{' n=7' if drop else ''}"
        chk(f"{label} displacement", float(disp.mean()), want, 0.001)
        neg, p = int((disp < 0).sum()), wilcoxon(disp).pvalue
        ok = neg == wn == len(disp) and p < 0.02
        print(f"  {'OK ' if ok else 'BAD'} {label} sign {neg}/{len(disp)} "
              f"negative (draft {wn}), p={p:.4f}")
        if not ok:
            FAILS.append(f"{label} displacement sign")
    # The between-arm factor §2.2 quotes.
    def d(arm, drop=()):
        tag = f"{arm}_seed0_trained"
        subs_r, pr = roi_profile(ENCODING_DIR, tag, True, drop=drop)
        real = {s: hv4_minus_v1(pr[i]) for i, s in enumerate(subs_r)}
        fl = {}
        for f in _files(arm):
            j = json.load(open(f))
            fl[j["subject"]] = stats_from(
                j["conditions"]["B_manufacture"]["rows"], "est_")[0]
        ss = [s for s in sorted(set(real) & set(fl)) if s not in drop]
        return float(np.mean([real[s] - fl[s] for s in ss]))
    chk("between-arm factor (draft 6.7x)", d("dkl_bcs") / d("dkl_base"), 6.7, 0.1)
    check_slope_model()


# Recovered true contrast under each treatment of the multiplicative term.
# The factor is what the claim rests on, and it barely moves; the magnitudes do.
SLOPE_TREATMENTS = {
    "additive only": (-0.130, -0.871, 6.7),
    "per-ROI, bias removed": (-0.101, -0.720, 7.1),
}


def check_slope_model() -> None:
    """The contrast floor must equal b(hV4) - b(V1) exactly, and the slope must
    be fitted after removing each ROI's OWN bias -- the simulated bias rises
    across ROIs, so a single pooled intercept is misspecified. A free intercept
    on the bias-removed fit should return to zero; it does for the diet arm and
    not quite for no-diet, which is why the no-diet magnitude is called
    approximate in §2.2.
    """
    print("  [bias model]")
    out = {}
    for arm, obs in (("dkl_base", -0.0785), ("dkl_bcs", -0.8335)):
        A = [_ab(f, "A_recovery") for f in _files(arm)]
        B = [_ab(f, "B_manufacture") for f in _files(arm)]
        b_roi = np.mean([x[1] for x in B], axis=0)
        floor = float(hv4_minus_v1(b_roi))
        measured = float(np.mean([hv4_minus_v1(x[1]) for x in B]))
        chk(f"{arm} floor == b(hV4)-b(V1)", floor, measured, 1e-9)
        T = np.concatenate([x[0] for x in A])
        E = np.concatenate([x[1] for x in A]) - np.tile(b_roi, len(A))
        a = float(np.sum(E * T) / np.sum(T * T))
        slope, icept = (float(v) for v in np.polyfit(T, E, 1))
        out[arm] = (obs - floor) / a
        want_a = {"dkl_base": 1.281, "dkl_bcs": 1.210}[arm]
        want_i = {"dkl_base": -0.043, "dkl_bcs": -0.002}[arm]
        chk(f"{arm} slope (bias removed)", a, want_a, 0.005)
        chk(f"{arm} residual intercept", icept, want_i, 0.005)
    for name, (wn, wd, wf) in SLOPE_TREATMENTS.items():
        if name == "per-ROI, bias removed":
            chk("no-diet true contrast", out["dkl_base"], wn, 0.001)
            chk("diet true contrast", out["dkl_bcs"], wd, 0.001)
            chk("factor, bias-removed slope", out["dkl_bcs"] / out["dkl_base"],
                wf, 0.05)


def _ab(path: str, cond: str) -> tuple[np.ndarray, np.ndarray]:
    """(true, estimated) per-ROI ratio index for one subject."""
    r = json.load(open(path))["conditions"][cond]["rows"]
    tm = np.array([x["true_unique_m"] for x in r])
    tp = np.array([x["true_unique_p"] for x in r])
    em = np.array([x["est_unique_m"] for x in r])
    ep = np.array([x["est_unique_p"] for x in r])
    return (tm - tp) / (tm + tp), (em - ep) / (em + ep)

    # subj01's DISPLACEMENT is the stable quantity; the ratio is not. Expressed
    # as a ratio the anomaly looks confined to the no-diet arm (4.65x vs 1.23x)
    # and we wrongly inferred a wrong-front-end fit from that. The baselines
    # differ ~15-fold, so the ratio is meaningless across arms: the absolute
    # shift is -0.195 vs -0.189, identical to within 0.006. Pinning both makes
    # the discrepancy visible rather than inferable.
    check_subj01_displacement()


# (arm prefix, seed): subj01 minus the sibling median, normalised statistic.
# Ratios for the same six fits span 1.15x to 5.73x; these span -0.091 to -0.205.
SUBJ01_DISPLACEMENT = {
    ("dkl_base", 0): -0.1952, ("dkl_base", 1): -0.0907, ("dkl_base", 2): -0.1441,
    ("dkl_bcs", 0): -0.1894, ("dkl_bcs", 1): -0.2053, ("dkl_bcs", 2): -0.1154,
}


def check_subj01_displacement() -> None:
    print("  [subj01 displacement, stable across arms]")
    for (pre, seed), want in SUBJ01_DISPLACEMENT.items():
        subs, pr = roi_profile(ENCODING_DIR, f"{pre}_seed{seed}_trained", True)
        c = {s: hv4_minus_v1(pr[i]) for i, s in enumerate(subs)}
        others = [v for k, v in c.items() if k != 1]
        chk(f"{pre} seed{seed} displacement",
            c[1] - float(np.median(others)), want, 0.001)
    # The claim that overturned the front-end inference: same shift, both arms.
    def disp(pre: str) -> float:
        subs, pr = roi_profile(ENCODING_DIR, f"{pre}_seed0_trained", True)
        c = {s: hv4_minus_v1(pr[i]) for i, s in enumerate(subs)}
        return c[1] - float(np.median([v for k, v in c.items() if k != 1]))
    gap = abs(disp("dkl_base") - disp("dkl_bcs"))
    print(f"  {'OK ' if gap < 0.01 else 'BAD'} seed-0 arms agree to {gap:.4f} "
          f"(draft says within 0.006)")
    if gap >= 0.01:
        FAILS.append("subj01 arm agreement")


# (dvd_trans, diet mean, diet min, diet max, no-diet mean) as the draft states them
OOD_DRAFT = {
    "uniform-noise":    (0.640, 0.634, 0.619, 0.647, 0.729),
    "high-pass":        (0.680, 0.678, 0.647, 0.696, 0.827),
    "rotation":         (0.572, 0.574, 0.567, 0.582, 0.615),
    "false-colour":     (0.839, 0.854, 0.845, 0.860, 0.938),
    "phase-scrambling": (0.362, 0.382, 0.365, 0.391, 0.373),
    "low-pass":         (0.436, 0.444, 0.434, 0.464, 0.356),
    "contrast":         (0.489, 0.481, 0.460, 0.493, 0.549),
}


def check_ood() -> None:
    d = "results/ood_bestep"

    def rel_auc(files: list[str], fam: str) -> np.ndarray:
        """Relative accuracy AUC, matching ood_verdict.curves()."""
        acc = np.array([json.load(open(f))["families"][fam]["accuracy"]
                        for f in files], float)
        return (acc / acc[:, :1]).mean(1)

    diet = sorted(glob.glob(f"{d}/dkl_bcs_seed*.json"))
    nod = sorted(glob.glob(f"{d}/dkl_base_seed*.json"))
    trans = [f"{d}/thesis_dvd_trans.json"]
    print("[OOD relative AUC]")
    inside = closer = 0
    for fam, (wt, wd, wlo, whi, wnd) in OOD_DRAFT.items():
        t = rel_auc(trans, fam)[0]
        dd, nn = rel_auc(diet, fam), rel_auc(nod, fam)
        chk(f"{fam} trans", t, wt, 0.001)
        chk(f"{fam} diet mean", dd.mean(), wd, 0.001)
        chk(f"{fam} diet lo", dd.min(), wlo, 0.001)
        chk(f"{fam} diet hi", dd.max(), whi, 0.001)
        chk(f"{fam} nodiet mean", nn.mean(), wnd, 0.001)
        inside += dd.min() <= t <= dd.max()
        closer += abs(t - dd.mean()) < abs(t - nn.mean())
    print(f"  inside diet seed range: {inside}/7 (draft says 5)")
    print(f"  closer to diet than no-diet: {closer}/7 (draft says 6)")
    if inside != 5:
        FAILS.append("inside count")
    if closer != 6:
        FAILS.append("closer count")
    for lab, files, want in [("trans", trans, 0.5682), ("diet", diet, 0.5530),
                             ("nodiet", nod, 0.6951)]:
        c = np.mean([json.load(open(f))["families"]["uniform-noise"]["accuracy"][0]
                     for f in files])
        chk(f"clean acc {lab}", float(c), want, 0.0005)


def main() -> int:
    check_recovery()
    check_production_statistic()
    check_ood()
    print("\nFAILURES:", FAILS if FAILS else "none")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
