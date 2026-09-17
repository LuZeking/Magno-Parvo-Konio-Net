"""§65's verdict: does S-(L+M) predict colour-biased cortex where L-M failed?

Written and committed BEFORE job 26422215 finished, so the grading cannot
drift to fit the numbers. Thresholds come from §65.6; this script only
evaluates them.

Order is deliberate and enforced:

  1. THE GATE (§65.4). `r2_magno` must be identical between each konio arm
     and the published parvo fit, because it is a RidgeCV fit on the magno
     band ALONE and the magno feature matrix is bit-identical across arms
     (verified by tmp_analysis/smoke_konio.py). If the gate fails, the script
     refuses to interpret anything else -- §9.1's lesson is that a pipeline
     difference reading as a result is this project's worst failure mode.
  2. The 25-ROI Kastner profile of Delta = U_k - U_p, per subject.
  3. The pre-declared contrasts: hV4, VO pooled, and the colour-vs-rest
     specificity test that replaces the ecc residualisation §65.5 showed is
     not computable at VO.
  4. A verdict coded against §65.6's A/B/C/D.

Both statistics (§8): every contrast is reported NC-normalised (the units of
§7.1's reference table, hence of §65.6's thresholds) AND raw, and a claim is
only licensed where the two agree.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import stats as sps

from src.analysis.encoding import nsd_data
from src.analysis.encoding.stats import load_results

OUT_DIR = "results/encoding"
# Seed: `python tmp_analysis/konio_verdict.py [SEED]`. Seed 0 is the published pair (both arms);
# seeds 1-2 (W2c, slurm/encode_konio_seeds.sh) have the matched arm only, each against the
# untrained L-M network at the same seed, so the magno gate below still applies within a seed.
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
REF = "untrained_seed0" if SEED == 0 else f"untrained_seed{SEED}"
ARMS = ([("matched (PRIMARY)", "untrained_konio_matched"),
         ("raw (sensitivity)", "untrained_konio_raw")] if SEED == 0 else
        [("matched (PRIMARY)", f"untrained_konio_matched_seed{SEED}")])

KASTNER = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d",
           7: "hV4", 8: "VO1", 9: "VO2", 10: "PHC1", 11: "PHC2",
           12: "TO2", 13: "TO1", 14: "LO2", 15: "LO1", 16: "V3B",
           17: "V3A", 18: "IPS0", 19: "IPS1", 20: "IPS2", 21: "IPS3",
           22: "IPS4", 23: "IPS5", 24: "SPL1", 25: "FEF"}
NAMES = [KASTNER[k] for k in sorted(KASTNER)]
COLOUR = ["hV4", "VO1", "VO2"]          # the pre-declared colour-biased set
VO = ["VO1", "VO2"]
NC_MIN, MIN_VOX = 0.1, 20

# §65.6, in NC-normalised units to match §7.1's reference table.
C_TOL = 0.005            # |Delta| below this in hV4 AND VO -> outcome C
P_A = 0.0156             # outcome A needs p <= this (n=8 floor is .0078)
SUBJ_A = 7               # ...and >= this many subjects with the sign

# §7.1 reference values, for orientation only -- never used as a threshold.
REF_71 = {"VO pooled": +0.0044, "hV4": -0.012, "VO2": +0.011,
          "VO1": -0.002, "V1d": +0.086, "V1v": +0.083}


def p(msg=""):
    print(msg, flush=True)


def need(name):
    r = load_results(OUT_DIR, name)
    if not r:
        p(f"MISSING: no results for {name!r} in {OUT_DIR}/ -- "
          f"job 26422215 has not produced this cell yet.")
    return r


def wilcox(x, label, indent="  "):
    """Paired one-sample Wilcoxon on a per-subject vector."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        p(f"{indent}{label:<34} n={len(x)} too few subjects")
        return None
    W, pv = sps.wilcoxon(x)
    dz = x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.inf
    p(f"{indent}{label:<34} mean={x.mean():+.5f}  median={np.median(x):+.5f}"
      f"  p={pv:.4f}  {int((x > 0).sum())}/{len(x)} pos  dz={dz:+.2f}")
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p=float(pv), npos=int((x > 0).sum()), n=len(x))


ref = need(REF)
if not ref:
    sys.exit("cannot run without the reference cell")
subs = sorted(ref)
p(f"reference cell {REF!r}: subjects {subs}")

# Kastner labels are loaded once per subject; the encoding fits all used the
# same deterministic nsdgeneral mask, so the labels index straight onto the
# stored per-voxel arrays.
labels, ncs = {}, {}
for s in subs:
    mask = nsd_data.load_mask(s, "nsdgeneral")
    kl = nsd_data.load_roi_labels(s, "Kastner2015", mask)
    if kl.size != ref[s]["noise_ceiling"].size:
        sys.exit(f"subj{s}: Kastner {kl.size} != npz "
                 f"{ref[s]['noise_ceiling'].size} -- mask mismatch")
    labels[s] = kl
    ncs[s] = ref[s]["noise_ceiling"]

for arm_label, arm_name in ARMS:
    p()
    p("#" * 78)
    p(f"# ARM: {arm_label}   [{arm_name}]")
    p("#" * 78)
    arm = need(arm_name)
    if not arm:
        continue
    shared = [s for s in subs if s in arm]
    if len(shared) < len(subs):
        p(f"NOTE: only {len(shared)} of {len(subs)} subjects present "
          f"({shared}); every contrast below is at that n.")
    if not shared:
        continue

    # ---------------------------------------------------------------- gate
    p()
    p("=" * 74)
    p("1. THE GATE (§65.4): r2_magno must be unchanged by swapping the")
    p("   chromatic band. Above 1e-3 the arm is INVALID.")
    p("=" * 74)
    worst = 0.0
    for s in shared:
        d = np.abs(arm[s]["r2_magno"] - ref[s]["r2_magno"])
        d = d[np.isfinite(d)]
        w = float(d.max()) if d.size else np.nan
        worst = max(worst, w)
        p(f"  subj{s:02d}  max|dr2_magno| = {w:.3e}   "
          f"mean = {d.mean():.3e}   n_vox = {d.size}")
    p(f"\n  WORST ACROSS SUBJECTS: {worst:.3e}")
    if worst > 1e-3:
        p("  *** GATE FAILED (>1e-3). Something that was supposed to be held")
        p("  *** fixed differed. Per §65.4 nothing else in this arm may be")
        p("  *** interpreted. Skipping.")
        continue
    if worst > 1e-4:
        p("  GATE MARGINAL (1e-4..1e-3): above the pre-registered 1e-4 but")
        p("  below the invalidating 1e-3. Treat every number below as")
        p("  provisional and find the cause before claiming anything.")
    else:
        p("  GATE PASSED: the magno side is identical, so Delta below is")
        p("  exactly r2_joint(konio) - r2_joint(parvo) with no magno-side")
        p("  estimation error.")

    # ------------------------------------------------------ per-ROI Delta
    # NC-normalised (primary, matches §7.1's units) and raw, per §8.
    for stat_label, k_u in (("NC-normalised", ""), ("raw", "_raw")):
        # The pre-noise-ceiling arrays were renamed *_raw -> *_pre_nc on 2026-09-07
        # (src/analysis/encoding/banded_ridge.py); stored fits carry whichever name
        # was current when they were written, so resolve against the file.
        def _key(stem, z):
            for suf in ((k_u, "_pre_nc") if k_u == "_raw" else (k_u,)):
                if f"{stem}{suf}" in z:
                    return f"{stem}{suf}"
            raise KeyError(f"{stem}{k_u} (and _pre_nc) missing")
        km, kp = _key("unique_magno", arm[shared[0]]), _key("unique_parvo", arm[shared[0]])
        km_r, kp_r = _key("unique_magno", ref[shared[0]]), _key("unique_parvo", ref[shared[0]])
        tab = np.full((len(shared), len(NAMES)), np.nan)
        nvox = np.zeros((len(shared), len(NAMES)), int)
        for i, s in enumerate(shared):
            # U_k - U_p. r2_magno cancels (gated above), so this equals
            # r2_joint(konio) - r2_joint(parvo), NC-scaled where asked.
            delta = arm[s][kp] - ref[s][kp_r]
            ok = (ncs[s] > NC_MIN) & np.isfinite(delta)
            for j, lab in enumerate(sorted(KASTNER)):
                v = ok & (labels[s] == lab)
                nvox[i, j] = v.sum()
                if v.sum() >= MIN_VOX:
                    tab[i, j] = np.median(delta[v])

        p()
        p("=" * 74)
        p(f"2/3. Delta = median(U_konio - U_parvo), {stat_label}, "
          f"n={len(shared)}, NC>{NC_MIN}")
        p("=" * 74)

        def pool(labs):
            idx = [NAMES.index(n) for n in labs]
            return np.nanmean(tab[:, idx], axis=1)

        p("  per-ROI, sorted by mean Delta (positive = konio explains more):")
        order = np.argsort(-np.nanmean(tab, axis=0))
        for j in order:
            m = np.nanmean(tab[:, j])
            if not np.isfinite(m):
                continue
            npos = int(np.nansum(tab[:, j] > 0))
            nn = int(np.sum(np.isfinite(tab[:, j])))
            star = "  <-- colour set" if NAMES[j] in COLOUR else ""
            p(f"    {NAMES[j]:<6} {m:+.5f}   {npos}/{nn} subj   "
              f"med n_vox={int(np.median(nvox[:, j]))}{star}")

        p()
        p("  pre-declared contrasts:")
        r_hv4 = wilcox(tab[:, NAMES.index("hV4")], "hV4", "    ")
        r_vo = wilcox(pool(VO), "VO (VO1,VO2)", "    ")
        r_col = wilcox(pool(COLOUR), "colour set (hV4,VO1,VO2)", "    ")

        # Specificity: the colour set against the remaining ROIs. This is
        # what §65.5 substitutes for the ecc residualisation that cannot be
        # computed at VO, and it is the instrument that killed hMT+ (§16.1).
        other_idx = [j for j, n in enumerate(NAMES) if n not in COLOUR]
        other = np.nanmedian(tab[:, other_idx], axis=1)
        n_other = int(np.median(np.sum(np.isfinite(tab[:, other_idx]), 1)))
        p()
        p(f"  specificity (§65.5's substitute for ecc residualisation; "
          f"~{n_other} other ROIs):")
        r_spec = wilcox(pool(COLOUR) - other,
                        "colour set - other ROIs", "    ")
        p(f"    mean Delta over other ROIs = {np.nanmean(other):+.5f}")

        if stat_label == "NC-normalised":
            primary = dict(hv4=r_hv4, vo=r_vo, col=r_col, spec=r_spec)
        else:
            secondary = dict(hv4=r_hv4, vo=r_vo, col=r_col, spec=r_spec)

    # -------------------------------------------------------------- verdict
    p()
    p("=" * 74)
    p("4. VERDICT against §65.6 (NC-normalised primary; agreement with raw")
    p("   is required before any claim)")
    p("=" * 74)
    for nm in ("hv4", "vo", "col", "spec"):
        a, b = primary.get(nm), secondary.get(nm)
        if a and b:
            agree = np.sign(a["mean"]) == np.sign(b["mean"])
            p(f"  {nm:<5} NC {a['mean']:+.5f} (p={a['p']:.4f})   "
              f"raw {b['mean']:+.5f} (p={b['p']:.4f})   "
              f"signs {'AGREE' if agree else 'DISAGREE'}")

    hv4, vo, spec = primary["hv4"], primary["vo"], primary["spec"]
    if not (hv4 and vo and spec):
        p("  cannot grade: a pre-declared contrast is missing")
        continue

    pos = (hv4["mean"] > 0 and vo["mean"] > 0
           and hv4["npos"] >= SUBJ_A and vo["npos"] >= SUBJ_A
           and hv4["p"] <= P_A and vo["p"] <= P_A)
    specific = spec["mean"] > 0 and spec["p"] <= 0.05

    if pos and specific:
        p("  -> OUTCOME A: KONIO RESCUES THE COLOUR PREDICTION.")
        p("     Positive in hV4 and VO at the pre-declared strength AND")
        p("     concentrated in colour-biased cortex. §65.7's rule now")
        p("     binds: a 3-seed untrained replication is REQUIRED before")
        p("     this is claimed.")
    elif pos and not specific:
        p("  -> OUTCOME B: KONIO WINS GLOBALLY, WHICH IS NOT A COLOUR")
        p("     RESULT. The general-factor problem that killed hMT+ (§16.1),")
        p("     recurring on a new channel. Licenses 'S-(L+M) is a better")
        p("     chromatic regressor overall' and NOTHING about VO.")
    elif (abs(hv4["mean"]) < C_TOL and abs(vo["mean"]) < C_TOL
          and hv4["p"] > 0.05 and vo["p"] > 0.05):
        p("  -> OUTCOME C: NO DIFFERENCE. The colour null is not about which")
        p("     chromatic axis. Both opponent axes were tested and neither")
        p("     predicts colour-biased cortex. One paragraph, not a panel.")
    elif hv4["mean"] < 0 and vo["mean"] < 0:
        p("  -> OUTCOME D: KONIO IS WORSE. Decisive against B3 -- and more")
        p("     so in the raw cell, where konio loses while carrying 2x the")
        p("     contrast (§65.3). Retire the konio channel and say so.")
    else:
        p("  -> UNCLASSIFIED: the result falls between the pre-declared")
        p("     outcomes (e.g. hV4 and VO disagree in sign, or the effect is")
        p("     above the C tolerance but below A's strength). Report the")
        p("     numbers and the fact that no pre-declared outcome fits.")
        p("     Do NOT relabel it as whichever outcome it is nearest.")

p()
p("done")
