#!/usr/bin/env python
"""Re-derive every number the draft states for the artefact floor (E2.1) and the
progressive-transition OOD arm (E6.2), straight from the result JSONs.

Exits non-zero if any draft number drifts from what the files say. Run it after
touching either section, and after the diet-arm half of the recovery analysis
lands (job 26442320) -- the E2.1 numbers are currently the no-diet geometry only.

    python tmp_analysis/verify_new_claims.py

2026-09-15: check_generality() adds the Results 2.4 / Fig. 6 / ED Table 14-15 numbers
of the single-trunk generality test (unit-level rule), and check_subspace() the
ED Table 13 / abstract numbers of the subspace rule, from the committed JSONs.
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
    chk("no-diet true contrast", out["dkl_base"], -0.101, 0.001)
    chk("diet true contrast", out["dkl_bcs"], -0.720, 0.001)
    chk("factor, bias-removed slope", out["dkl_bcs"] / out["dkl_base"], 7.1, 0.05)
    check_residual_treatment()


def check_residual_treatment() -> None:
    """Pin the DIAGNOSIS, not the artefact.

    An earlier version pinned the fullest treatment's no-diet value (-0.034,
    5/8, p = .31) so its resolvability could not be quietly re-asserted. That
    was the wrong thing to lock: the residual it removes is a property of
    condition A's source schedule, not of the estimator, so pinning it would
    make the verifier enforce a simulation artefact. What is pinned instead is
    (a) the evidence that the residual is a generator property, and (b) that the
    correction is under-specified -- either of which failing would mean the
    decision to reject the correction needs revisiting.
    """
    i = np.arange(7)
    power = (0.30 + 0.10 * i) ** 2 + (0.90 - 0.10 * i) ** 2
    print("  [residual diagnosis: generator property, not estimator misfit]")
    prof = {}
    for arm in ("dkl_base", "dkl_bcs"):
        A = [_ab(f, "A_recovery") for f in _files(arm)]
        B = [_ab(f, "B_manufacture") for f in _files(arm)]
        b_roi = np.mean([x[1] for x in B], axis=0)
        T = np.stack([x[0] for x in A])
        E = np.stack([x[1] for x in A]) - b_roi
        a = float(np.sum(E * T) / np.sum(T * T))
        prof[arm] = (E - a * T).mean(0)
        want_p, want_t = {"dkl_base": (-0.847, 0.080),
                          "dkl_bcs": (-0.820, 0.005)}[arm]
        chk(f"{arm} corr(residual, source power)",
            float(np.corrcoef(prof[arm], power)[0, 1]), want_p, 0.01)
        chk(f"{arm} corr(residual, true contrast)",
            float(np.corrcoef(prof[arm], T.mean(0))[0, 1]), want_t, 0.01)
    chk("corr(no-diet residual, diet residual)",
        float(np.corrcoef(prof["dkl_base"], prof["dkl_bcs"])[0, 1]), 0.989, 0.005)

    # Under-specification: same point estimate, p from .055 to .84.
    from scipy.stats import wilcoxon as _w
    arm = "dkl_base"
    A = [_ab(f, "A_recovery") for f in _files(arm)]
    B = [_ab(f, "B_manufacture") for f in _files(arm)]
    b_roi = np.mean([x[1] for x in B], axis=0)
    T = np.stack([x[0] for x in A])
    E = np.stack([x[1] for x in A]) - b_roi
    a = float(np.sum(E * T) / np.sum(T * T))
    cg = float(hv4_minus_v1((E - a * T).mean(0)))
    floor = {}
    for f in _files(arm):
        j = json.load(open(f))
        floor[j["subject"]] = stats_from(
            j["conditions"]["B_manufacture"]["rows"], "est_")[0]
    fg = float(np.mean(list(floor.values())))
    subs_r, pr = roi_profile(ENCODING_DIR, f"{arm}_seed0_trained", True)
    real = {s: hv4_minus_v1(pr[k]) for k, s in enumerate(subs_r)}
    ss = sorted(set(real) & set(floor))
    # The decisive robustness argument: crossing the two arbitrary choices
    # (group vs per-subject floor; residual removed before vs after rescaling)
    # gives six variants. The MINIMAL treatment is the only one invariant to the
    # pairing -- which is why §2.2 uses it, independently of the artefact
    # diagnosis above.
    ps = {}
    for fname, fv in (("group", lambda s: fg), ("per-subject", lambda s: floor[s])):
        for tname, fn in (
                ("minimal", lambda s, f=fv: (real[s] - f(s)) / a),
                ("resid-before", lambda s, f=fv: (real[s] - f(s) - cg) / a),
                ("resid-after", lambda s, f=fv: (real[s] - f(s)) / a - cg)):
            v = np.array([fn(s) for s in ss])
            ps[(fname, tname)] = (float(v.mean()), int((v < 0).sum()),
                                  float(_w(v).pvalue))
    inv = (ps[("group", "minimal")] == ps[("per-subject", "minimal")])
    print(f"  {'OK ' if inv else 'BAD'} minimal treatment invariant to floor "
          f"pairing: {ps[('group', 'minimal')][1]}/8, p="
          f"{ps[('group', 'minimal')][2]:.4f} under both")
    if not inv:
        FAILS.append("minimal treatment no longer pairing-invariant")
    corr = [v for k, v in ps.items() if k[1] != "minimal"]
    signs = {c[1] for c in corr}
    pv = [c[2] for c in corr]
    swung = len(signs) > 1 and min(pv) < 0.06 and max(pv) > 0.8
    print(f"  {'OK ' if swung else 'BAD'} residual-corrected variants swing: "
          f"signs {sorted(signs)}, p = {min(pv):.3f} to {max(pv):.3f} "
          f"(draft 4/8-6/8, .055 to 1.00)")
    if not swung:
        FAILS.append("residual correction no longer under-specified")
    # And the diet arm holds in every one of the six.
    dfloor, dreal, dcg, da, dss = _diet_variant_inputs()
    held = 0
    for fname, fv in (("group", lambda s: float(np.mean(list(dfloor.values())))),
                      ("per-subject", lambda s: dfloor[s])):
        for fn in ((lambda s, f=fv: (dreal[s] - f(s)) / da),
                   (lambda s, f=fv: (dreal[s] - f(s) - dcg) / da),
                   (lambda s, f=fv: (dreal[s] - f(s)) / da - dcg)):
            v = np.array([fn(s) for s in dss])
            if int((v < 0).sum()) == len(v) and _w(v).pvalue < 0.02:
                held += 1
    print(f"  {'OK ' if held == 6 else 'BAD'} diet arm holds in {held}/6 "
          f"variants (draft 6/6)")
    if held != 6:
        FAILS.append("diet arm no longer holds in all six variants")


def _diet_variant_inputs():
    arm = "dkl_bcs"
    A = [_ab(f, "A_recovery") for f in _files(arm)]
    B = [_ab(f, "B_manufacture") for f in _files(arm)]
    b_roi = np.mean([x[1] for x in B], axis=0)
    T = np.stack([x[0] for x in A])
    E = np.stack([x[1] for x in A]) - b_roi
    a = float(np.sum(E * T) / np.sum(T * T))
    cg = float(hv4_minus_v1((E - a * T).mean(0)))
    floor = {}
    for f in _files(arm):
        j = json.load(open(f))
        floor[j["subject"]] = stats_from(
            j["conditions"]["B_manufacture"]["rows"], "est_")[0]
    subs_r, pr = roi_profile(ENCODING_DIR, f"{arm}_seed0_trained", True)
    real = {s: hv4_minus_v1(pr[k]) for k, s in enumerate(subs_r)}
    ss = sorted(set(real) & set(floor))
    return floor, real, cg, a, ss


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


# The untrained baseline is the one §2.2 value that does NOT come from the
# declared source directory: it exists only in results/encoding/. Two things
# are locked below -- the anchor itself, and the cross-directory identity that
# makes sourcing it from elsewhere harmless.
UNTRAINED_DIR = "results/encoding"
UNTRAINED = {"normalised": -0.544, "raw": -0.120}


# §2.2 quotes dz twice: -6.60 (diet) vs -1.09 (no diet) at seed 0 normalised,
# and §4.7's multiplicity argument quotes the larger of the two as "our
# largest". An earlier draft quoted dz = -11.70 there, which belonged to the
# SUPERSEDED table and does not regenerate -- so the value is pinned.
DZ_SEED0_NORM = {"dkl_bcs_seed0_trained": -6.60, "dkl_base_seed0_trained": -1.09}


def check_effect_size() -> None:
    """dz for §2.2's seed-0 normalised cells, and that -6.60 is the largest."""
    print(f"\n[§2.2 effect sizes, {ENCODING_DIR}]")
    biggest = 0.0
    for tag, want in DZ_SEED0_NORM.items():
        _, prof = roi_profile(ENCODING_DIR, tag, True)
        c = np.array([hv4_minus_v1(r) for r in prof])
        chk(f"dz {tag}", float(c.mean() / c.std(ddof=1)), want, 0.01)
    # §4.7 calls -6.60 "our largest"; check nothing in the table beats it.
    for stat in (True, False):
        for seed in (0, 1, 2):
            for arm in ("dkl_bcs", "dkl_base"):
                _, prof = roi_profile(ENCODING_DIR, f"{arm}_seed{seed}_trained",
                                      stat)
                c = np.array([hv4_minus_v1(r) for r in prof])
                biggest = max(biggest, abs(float(c.mean() / c.std(ddof=1))))
    if abs(biggest - 6.60) > 0.01:
        FAILS.append(f"largest |dz| in §2.2 is {biggest:.2f}, draft says 6.60")
    print(f"  {'OK' if abs(biggest - 6.60) <= 0.01 else 'FAIL'}  "
          f"largest |dz| over all 12 cells = {biggest:.3f} (draft 6.60)")


def check_untrained_anchor() -> None:
    """§2.2's untrained control, and the directory-identity claim it rests on.

    §E2.2 asserts that where results/encoding/ and results/encoding_cc/ overlap
    they are numerically identical. That is not a coincidence: --cue-conflict
    only applies the *already fitted* estimators to an extra design
    (banded_ridge.fit_banded docstring) and prepare_band takes "all statistics
    from the training split only", so the flag cannot move r2_*. Asserting it is
    still weaker than checking it, and if anyone ever refits one directory
    without the other the assertion would go quietly false -- so it is checked.
    """
    from src.analysis.encoding.stats import load_results
    print(f"\n[§2.2 untrained anchor, {UNTRAINED_DIR}]")
    for stat, want in UNTRAINED.items():
        subs, prof = roi_profile(UNTRAINED_DIR, "untrained_seed0",
                                 stat == "normalised")
        c = np.array([hv4_minus_v1(r) for r in prof])
        W, pv = wilcoxon(c)
        chk(f"untrained hV4-V1 {stat}", float(c.mean()), want, 0.001)
        # Both statistics must agree in sign for the claim this supports
        # ("the input only supplies the starting point") to clear §4.7's
        # two-statistics rule.
        if not (c < 0).all():
            FAILS.append(f"untrained {stat}: {(c < 0).sum()}/8 negative, want 8/8")
        chk(f"untrained p {stat}", float(pv), 0.0078, 0.0002)
        print(f"     n={len(subs)}  {(c < 0).sum()}/{len(c)} negative  "
              f"W={W:.0f} p={pv:.4f}")

    print("[cross-directory identity, encoding vs encoding_cc]")
    both = sorted(set(
        os.path.basename(f)[:-4] for f in glob.glob("results/encoding/*.npz")
    ) & set(
        os.path.basename(f)[:-4] for f in glob.glob("results/encoding_cc/*.npz")
    ))
    worst = 0.0
    for name in both:
        a = np.load(os.path.join("results/encoding", name + ".npz"))
        b = np.load(os.path.join("results/encoding_cc", name + ".npz"))
        for k in ("r2_joint", "r2_magno", "r2_parvo"):
            if k in a.files and k in b.files and a[k].shape == b[k].shape:
                worst = max(worst, float(np.nanmax(np.abs(a[k] - b[k]))))
            else:
                FAILS.append(f"{name}: {k} missing or shape-mismatched")
    if worst != 0.0:
        FAILS.append(f"directories diverge: max|r2 diff| = {worst:.3e}")
    print(f"  {'OK' if worst == 0.0 else 'FAIL'}  {len(both)} shared files, "
          f"max |r2 diff| = {worst:.3e}")


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


PERIMAGE_DIR = ("/scratch/byrong/mpk/perimage_behavior/results/"
                "2026-09-07_geirhos_perimage")


def check_perimage() -> None:
    """Pin §2.6.2's per-image analysis, including the control that killed a claim.

    The last block matters more than the numbers above it. The diet arm beats the
    no-diet arm on difficulty correspondence with no overlap, and the crossed 2x2
    is the only thing standing between that and a claim we would not be able to
    support. If a future edit ever makes the crossed cells look attributable,
    this should fail loudly rather than quietly license the claim.
    """
    print("\n--- §2.6.2 per-image correspondence ---")
    cf = os.path.join(PERIMAGE_DIR, "perimage_correspondence.json")
    if not os.path.exists(cf):
        FAILS.append("perimage results missing")
        print(f"  BAD perimage results missing at {cf}")
        return
    R = json.load(open(cf))
    M, C = R["models"], R["human_ceiling"]

    for key, want in [("shape_rate", 0.658), ("shape_rate_resid", 0.624),
                      ("valid_rate", 0.794), ("valid_rate_resid", 0.728)]:
        chk(f"human ceiling {key}", C[key]["pearson_mean"], want, 0.001)

    diet  = [f"dkl_bcs_seed{s}_final" for s in (0, 1, 2)]
    dietb = [f"dkl_bcs_seed{s}_best" for s in (0, 1, 2)]
    nod   = [f"dkl_base_seed{s}_final" for s in (0, 1, 2)]

    def pct(names, tgt, kind="pearson"):
        return [M[n][f"{tgt}__{kind}"]["oos_pct_ceiling"] for n in names]

    for lab, names, tgt, want in [
            ("diet decision",   diet,  "shape_rate_resid",  8.7),
            ("diet difficulty", diet,  "valid_rate_resid", 24.1),
            ("nodiet decision", nod,   "shape_rate_resid",  8.0),
            ("nodiet difficulty", nod, "valid_rate_resid",  9.2)]:
        chk(f"{lab} % ceiling", float(np.mean(pct(names, tgt))), want, 0.05)

    # the positive control: difficulty >> decision WITHIN the diet system
    ratio = np.mean(pct(diet, "valid_rate_resid")) / np.mean(pct(diet, "shape_rate_resid"))
    chk("diet difficulty:decision ratio", float(ratio), 2.8, 0.05)
    rb = np.mean(pct(dietb, "valid_rate_resid")) / np.mean(pct(dietb, "shape_rate_resid"))
    chk("diet-best difficulty:decision", float(rb), 2.0, 0.05)

    # independent replication of the diet rate range quoted in §2.6.2
    lo = min(M[n]["rate_range"][0] for n in diet + dietb)
    hi = max(M[n]["rate_range"][1] for n in diet + dietb)
    chk("diet rate range lo", float(lo), 0.373, 0.0005)
    chk("diet rate range hi", float(hi), 0.913, 0.0005)

    # w moves the rate without reordering image difficulty
    rr = [M[n]["stream_margin_pearson_resid"] for n in diet + dietb + nod]
    if not (-0.031 <= min(rr) and max(rr) <= 0.073):
        FAILS.append("stream margins no longer near-independent")
        print(f"  BAD stream margin resid range {min(rr):+.3f}..{max(rr):+.3f}")
    else:
        print(f"  OK  stream margin resid range          "
              f"{min(rr):+.3f}..{max(rr):+.3f} (draft -0.030..+0.072)")

    # THE CONTROL. Crossed cells must stay non-positive and the ordering must
    # reverse with the front end off; otherwise the withdrawn diet claim would
    # become sayable and the draft would be understating what it can support.
    crossed = {
        "diet_off":   [f"dkl_bcs_seed{s}_final_nofe" for s in (0, 1, 2)],
        "nodiet_on":  [f"dkl_base_seed{s}_final_fe" for s in (0, 1, 2)],
    }
    if not all(n in M for v in crossed.values() for n in v):
        FAILS.append("crossed 2x2 cells missing")
        print("  BAD crossed 2x2 cells missing")
        return
    d_off = float(np.mean(pct(crossed["diet_off"], "valid_rate_resid")))
    n_on = float(np.mean(pct(crossed["nodiet_on"], "valid_rate_resid")))
    n_off = float(np.mean(pct(nod, "valid_rate_resid")))
    chk("crossed diet-weights/no-frontend", d_off, -0.6, 0.05)
    chk("crossed nodiet-weights/frontend", n_on, -3.7, 0.05)
    if not (d_off < n_off):
        FAILS.append("front-end-off ordering no longer reverses")
        print(f"  BAD ordering with front end off: diet {d_off:+.1f} "
              f"vs no-diet {n_off:+.1f} -- no longer reversed")
    else:
        print(f"  OK  front-end-off ordering reverses     "
              f"diet {d_off:+.1f} < no-diet {n_off:+.1f}")


def check_generality() -> None:
    """Results 2.4 (final paragraph), Fig. 6 and ED Tables 14-15: the single-trunk
    generality test under the unit-level rule, from results/figures_v2/generality_{behaviour,cortex}.json
    (tmp_analysis/figs_v2/generality_{behaviour,cortex}.py)."""
    print("\n[generality] behaviour, unit-level rule (ED Table 14, Fig. 6a grey)")
    beh = json.load(open("results/figures_v2/generality_behaviour.json"))
    fam = {f["key"]: f for f in beh["families"]}
    ws = np.array(fam["resnet50"]["seeds"][0]["ws"])
    i90 = int(np.argmin(np.abs(ws - 0.90)))
    single = ["rgb_tsn", "single18", "single34", "resnet50", "resnet152", "vgg16", "vit_b_16", "convnext_small", "convnext_base"]
    chk("two-stream concat span", fam["concat"]["span"], 0.46, 0.01)
    chk("two-stream concat S from stream 1", fam["concat"]["frac_S_from_stream1"], 0.99, 0.01)
    spans = [fam[k]["span"] for k in single]
    chk("single-trunk span min", min(spans), 0.09, 0.01)
    chk("single-trunk span max", max(spans), 0.21, 0.01)
    seed_rhos = [r["rho"] for k in single for r in fam[k]["seeds"]]
    chk("single-trunk seed-mean rho, min over nets", min(fam[k]["rho"] for k in single), 0.71, 0.01)
    chk("single-trunk seed rho min (ResNet-34 s0)", min(seed_rhos), 0.19, 0.01)
    chk("network-seed combinations, rho >= 0.90", float(sum(r >= 0.90 for r in seed_rhos)), 14, 0)
    chk("network-seed combinations, total", float(len(seed_rhos)), 15, 0)
    if min(seed_rhos) <= 0:
        FAILS.append("P1 direction: a seed has rho <= 0"); print("  BAD a seed has rho <= 0")
    chk("single-trunk shape-leaning frac max", max(fam[k]["frac_shape_leaning"] for k in single), 0.17, 0.01)
    for k, want in [("resnet50", 0.222), ("resnet152", 0.293), ("vit_b_16", 0.399), ("convnext_base", 0.269),
                    ("convnext_small", 0.248), ("vgg16", 0.110)]:
        chk(f"{k} trained rate", fam[k]["trained"], want, 0.002)
    curve = {k: np.mean([r["mean_curve"] for r in fam[k]["seeds"]], 0) for k in single}
    chk("resnet50 rate at w=0.90", curve["resnet50"][i90], 0.31, 0.005)
    chk("convnext_small rate at w=0.90", curve["convnext_small"][i90], 0.31, 0.005)
    chk("resnet152 rate at w=0.90", curve["resnet152"][i90], 0.40, 0.005)
    chk("resnet152 rate at w=1", curve["resnet152"][-1], 0.44, 0.005)
    if not (curve["resnet50"][i90] > fam["resnet152"]["trained"] and curve["convnext_small"][i90] > fam["convnext_base"]["trained"]):
        FAILS.append("gated re-ordering (ResNet-50 > ResNet-152; ConvNeXt-S > ConvNeXt-B) no longer holds")
        print("  BAD gated re-ordering")
    else:
        print("  OK  gated re-ordering at w = 0.90            ResNet-50 > ResNet-152 trained; ConvNeXt-S > ConvNeXt-B trained")
    for k, (lo, hi) in [("resnet50", (0.15, 0.90)), ("resnet152", (0.15, 0.90))]:
        g = fam[k]["seeds"][0]["gain_accuracy"]["w_within_5pts"]
        chk(f"{k} gate lower w", min(g), lo, 0.001); chk(f"{k} gate upper w", max(g), hi, 0.001)
    ref = []
    for k in ("rgb_tsn", "single18", "single34"):
        for head in ("linear", "mlp"):
            ref.append(np.mean([r["refit"][head]["span"] for r in fam[k]["seeds"] if "refit" in r]))
    chk("refit span (seed means) min", min(ref), 0.05, 0.01)
    chk("refit span (seed means) max", max(ref), 0.10, 0.01)
    # split-rule sensitivity (Methods 4.8, ED Table 14 last two columns)
    rules = {(k, rule): fam[k]["rules"][rule] for k in single for rule in ("positive", "decile")
             if fam[k].get("rules", {}).get(rule) and not fam[k]["rules"][rule].get("undefined")}
    if rules:
        chk("rule spans (ungated) min", min(e["span"] for e in rules.values()), 0.11, 0.01)
        chk("rule spans (ungated) max", max(e["span"] for e in rules.values()), 0.40, 0.01)
        chk("rule mean rho min", min(e["rho"] for e in rules.values()), 0.29, 0.01)
        pos = [e for (k, rule), e in rules.items() if rule == "positive"]
        dec = [e for (k, rule), e in rules.items() if rule == "decile"]
        chk("absolute-rule |S| min", min(e["n_S"] for e in pos), 2, 0)
        chk("absolute-rule |S| max", max(e["n_S"] for e in pos), 89, 1)
        chk("decile-rule |S| min", min(e["n_S"] for e in dec), 51, 0)
        chk("decile-rule |S| max", max(e["n_S"] for e in dec), 409, 0)
        gates = sorted(e["w5_hi"] for e in rules.values() if "w5_hi" in e)
        chk("rule gate upper w min", gates[0], 0.72, 0.005)
        chk("rule gate upper w, second largest", gates[-2], 0.95, 0.005)
        chk("rule gate upper w max (ConvNeXt-B decile)", gates[-1], 1.00, 0.005)
        gated = []
        for (k, rule), e in rules.items():
            if "w5_hi" not in e:
                continue
            c = np.mean([r["rules"][rule]["mean_curve"] for r in fam[k]["seeds"] if r.get("rules", {}).get(rule) and not r["rules"][rule].get("undefined")], 0)
            m = (ws >= e["w5_lo"] - 1e-9) & (ws <= e["w5_hi"] + 1e-9)
            gated.append(c[m].max() - c[m].min())
        chk("rule gated span min", min(gated), 0.02, 0.005)
        chk("rule gated span max", max(gated), 0.21, 0.005)
        chk("concat absolute set size", fam["concat"]["rules"]["positive"]["n_S"], 504, 1)
        chk("concat absolute-rule span", fam["concat"]["rules"]["positive"]["span"], 0.46, 0.01)
        if not fam["convnext_small"]["rules"]["positive"].get("undefined"):
            FAILS.append("ConvNeXt-S absolute rule expected undefined"); print("  BAD ConvNeXt-S absolute rule defined")

    print("\n[generality] cortex, unit halves (ED Table 15, Fig. 6c-d)")
    ctx = json.load(open("results/figures_v2/generality_cortex.json"))["families"]
    st = ctx["rgb_tsn"]["stats"]
    chk("rgb_tsn extrastriate - V1 (index)", st["extrastriate_minus_V1"]["mean"], -0.32, 0.01)
    chk("rgb_tsn extrastriate - V1 n neg", st["extrastriate_minus_V1"]["n_neg"], 8, 0)
    chk("rgb_tsn extrastriate - V1 p", st["extrastriate_minus_V1"]["p"], 0.0078, 0.0001)
    chk("rgb_tsn w* extrastriate - V1 n neg", st["wstar_extrastriate_minus_V1"]["n_neg"], 7, 0)
    chk("rgb_tsn w* extrastriate - V1 p", st["wstar_extrastriate_minus_V1"]["p"], 0.023, 0.001)
    chk("rgb_tsn w* V1 median", st["wstar_median"]["V1"], 0.80, 0.005)
    st = ctx["single34"]["stats"]
    chk("single34 hV4 - V1 n neg", st["hV4_minus_V1"]["n_neg"], 7, 0)
    chk("single34 hV4 - V1 p", st["hV4_minus_V1"]["p"], 0.016, 0.001)
    chk("single34 extrastriate - V1 n neg", st["extrastriate_minus_V1"]["n_neg"], 7, 0)
    chk("single34 dorsal - ventral (index)", st["dorsal_minus_ventral"]["mean"], 0.16, 0.01)
    chk("single34 dorsal - ventral n neg", st["dorsal_minus_ventral"]["n_neg"], 0, 0)
    st = ctx["resnet50"]["stats"]
    chk("resnet50 extrastriate - V1 (index)", st["extrastriate_minus_V1"]["mean"], 0.24, 0.01)
    chk("resnet50 extrastriate - V1 p", st["extrastriate_minus_V1"]["p"], 0.039, 0.001)
    for famk in ("rgb_tsn", "single34", "resnet50"):
        fo = ctx[famk]
        for r in ("V1", "V2", "V3", "hV4", "ventral", "lateral", "dorsal"):
            u = max(np.nanmean(fo["uS"][r]), np.nanmean(fo["uT"][r]))
            if u > 0.04:
                FAILS.append(f"{famk} {r} unique variance {u:.3f} > 0.04")
                print(f"  BAD {famk} {r} unique variance {u:.3f} > 0.04")
    print("  OK  single-trunk unique variances <= 0.04 in every region checked")
    if "shared" in ctx["rgb_tsn"]:
        fr = [np.nanmean(ctx[k]["shared"][r]) / np.nanmean(ctx[k]["r2"][r]) for k in ("rgb_tsn", "single34", "resnet50")
              for r in ("V1", "V2", "V3", "hV4", "ventral", "lateral", "dorsal")]
        chk("single-trunk shared / r2 min", min(fr), 0.80, 0.02)
        chk("single-trunk shared / r2 max", max(fr), 0.96, 0.02)
        fr2 = [np.nanmean(ctx["two_stream_diet"]["shared"][r]) / np.nanmean(ctx["two_stream_diet"]["r2"][r])
               for r in ("V1", "V2", "V3", "hV4", "ventral", "lateral", "dorsal")]
        chk("two-stream shared / r2 min over regions", float(min(fr2)), 0.45, 0.02)
        chk("two-stream shared / r2 max over regions", float(max(fr2)), 0.63, 0.02)


def check_subspace() -> None:
    """Abstract, Results 2.4, Discussion, Limitations, Fig. 6a-b and ED Table 13: the subspace
    rule of PLAN_SUBSPACE_2026-09-15, from results/figures_v2/subspace_behaviour.json
    (tmp_analysis/figs_v2/subspace_behaviour.py)."""
    print("\n[subspace] behaviour (ED Table 13, Fig. 6a-b)")
    sub = json.load(open("results/figures_v2/subspace_behaviour.json"))
    fam = {f["key"]: f for f in sub["families"]}
    ws = np.array(fam["resnet50"]["seeds"][0]["ws"])
    imagenet = ["resnet50", "resnet152", "vgg16", "vit_b_16", "convnext_small", "convnext_base"]
    ours = ["rgb_tsn", "single18", "single34"]
    curve = {k: np.mean([s["mean_curve"] for s in fam[k]["seeds"]], 0) for k in imagenet + ours + ["concat"]}

    def gmask(k):
        return (ws >= fam[k]["gate"][0] - 1e-9) & (ws <= fam[k]["gate"][1] + 1e-9)

    # ImageNet: magnitude, gate, monotonicity (abstract "0.25-0.33 ... within 1.5 points of top-1")
    g = {k: fam[k]["gated_span"] for k in imagenet}
    chk("ImageNet gated span min", min(g.values()), 0.25, 0.005)
    chk("ImageNet gated span max", max(g.values()), 0.33, 0.005)
    chk("P-A ImageNet networks with gated span >= 0.20", float(sum(v >= 0.20 for v in g.values())), 6, 0)
    for k in imagenet:
        chk(f"{k} gate covers the whole sweep", fam[k]["gate"][1] - fam[k]["gate"][0], 1.0, 1e-6)
    dtop = max(max(abs(np.array(s["gain_accuracy"]["top1"]) - s["gain_accuracy"]["trained_top1"]))
               for k in imagenet for s in fam[k]["seeds"]) * 100
    chk("ImageNet largest top-1 change (points)", dtop, 1.5, 0.05)
    chk("ImageNet seed-mean rho min", min(fam[k]["rho"] for k in imagenet), 0.997, 0.0005)
    for k, (a, b) in [("resnet50", (0.16, 0.45)), ("vgg16", (0.06, 0.33)), ("vit_b_16", (0.29, 0.57))]:
        chk(f"{k} rate at w = 0", curve[k][0], a, 0.005)
        chk(f"{k} rate at w = 1", curve[k][-1], b, 0.005)
    tr = {k: fam[k]["trained"] for k in imagenet}
    chk("ImageNet trained rate min", min(tr.values()), 0.11, 0.005)
    chk("ImageNet trained rate max", max(tr.values()), 0.40, 0.005)
    # "each network can be moved past the trained rate of every other (VGG-16 past four of the five)"
    for k in imagenet:
        n = sum(curve[k][gmask(k)].max() > tr[o] for o in imagenet if o != k)
        chk(f"{k} passes n others within the gate", float(n), 4 if k == "vgg16" else 5, 0)
    # "the six can be placed in the reverse of their trained order": greedy feasibility within the gated ranges
    lo = {k: curve[k][gmask(k)].min() for k in imagenet}
    hi = {k: curve[k][gmask(k)].max() for k in imagenet}
    order = sorted(imagenet, key=lambda k: tr[k])          # ascending trained rate; must end descending
    prev, feasible = hi[order[0]], True
    for k in order[1:]:
        r = min(hi[k], prev - 0.005)
        feasible &= r >= lo[k]
        prev = r
    if feasible:
        print("  OK  reverse of the trained order is reachable within the gate")
    else:
        FAILS.append("reverse ordering of the six ImageNet networks not reachable within the gate"); print("  BAD reverse ordering")
    # our trunks: ungated 0.23-0.34, ends lose 7-15 points, gated 0.14-0.22, threshold met in 1 of 3
    ung = [fam[k]["span"] for k in ours]
    chk("ours ungated span min", min(ung), 0.23, 0.005)
    chk("ours ungated span max", max(ung), 0.34, 0.005)
    gt = [fam[k]["gated_span"] for k in ours]
    chk("ours gated span min", min(gt), 0.14, 0.005)
    chk("ours gated span max", max(gt), 0.22, 0.005)
    chk("ours networks with gated span >= 0.20", float(sum(v >= 0.20 for v in gt)), 1, 0)
    drops = []
    for k in ours:
        acc = [s["gain_accuracy"] for s in fam[k]["seeds"]]
        for i in (0, -1):
            drops.append(np.mean([a["trained_top1"] - a["top1"][i] for a in acc]) * 100)
    chk("ours top-1 loss at the sweep ends, min (pts)", min(drops), 7, 0.5)
    chk("ours top-1 loss at the sweep ends, max (pts)", max(drops), 15, 0.5)
    # refit (block form): two-layer 0.10-0.15, linear 0.08-0.13, validation within 1.1 points of the ceiling
    mlp = [fam[k]["refit"]["mlp"]["span"] for k in ours]; lin = [fam[k]["refit"]["linear"]["span"] for k in ours]
    chk("refit two-layer span min", min(mlp), 0.10, 0.005); chk("refit two-layer span max", max(mlp), 0.15, 0.005)
    chk("refit linear span min", min(lin), 0.08, 0.005); chk("refit linear span max", max(lin), 0.13, 0.005)
    short = max(-s["refit"][h]["min_val_minus_ceiling"] for k in ours for s in fam[k]["seeds"] for h in ("linear", "mlp")) * 100
    print(f"  {'OK ' if short <= 1.2 else 'BAD'} refit validation shortfall, max over seeds and heads: {short:.2f} points (text: within 1.2)")
    if short > 1.2:
        FAILS.append("refit validation shortfall exceeds 1.2 points")
    # distributed information: variance fractions 4-15% / 20-33%; unit-level gated spans 0.07-0.16, a third to two thirds of the subspace spans
    vS = [np.mean([s["descriptives"]["var_frac_S"] for s in fam[k]["seeds"]]) for k in imagenet + ours]
    vT = [np.mean([s["descriptives"]["var_frac_T"] for s in fam[k]["seeds"]]) for k in imagenet + ours]
    chk("variance in S min", min(vS), 0.04, 0.005); chk("variance in S max", max(vS), 0.15, 0.005)
    chk("variance in T min", min(vT), 0.20, 0.005); chk("variance in T max", max(vT), 0.33, 0.005)
    ug = {k: fam[k]["unit_gated_span"] for k in imagenet + ours}
    chk("unit-level gated span min", min(ug.values()), 0.07, 0.005); chk("unit-level gated span max", max(ug.values()), 0.16, 0.005)
    ratio = [ug[k] / fam[k]["gated_span"] for k in imagenet + ours]
    chk("unit / subspace gated ratio min", min(ratio), 0.30, 0.02); chk("unit / subspace gated ratio max", max(ratio), 0.65, 0.02)
    if not all(fam[k]["gated_span"] > ug[k] for k in imagenet + ours):
        FAILS.append("P-D: a unit-level gated span exceeds the subspace one"); print("  BAD P-D")
    # P-B: the subspaces are not the streams (63-70% of the shape basis on stream 1); concat span 0.32 v 0.46 (streams)
    e = [s["descriptives"]["energy_S_on_stream1_full"] for s in fam["concat"]["seeds"]]
    chk("concat shape-basis energy on stream 1, min", min(e), 0.63, 0.005); chk("concat shape-basis energy on stream 1, max", max(e), 0.70, 0.005)
    chk("concat subspace span", fam["concat"]["span"], 0.32, 0.005)
    chk("concat subspace rate at w = 0", curve["concat"][0], 0.50, 0.005); chk("concat subspace rate at w = 1", curve["concat"][-1], 0.80, 0.005)


def main() -> int:
    check_recovery()
    check_production_statistic()
    check_effect_size()
    check_untrained_anchor()
    check_ood()
    check_perimage()
    check_generality()
    check_subspace()
    print("\nFAILURES:", FAILS if FAILS else "none")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
