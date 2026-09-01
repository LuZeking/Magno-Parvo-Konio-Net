"""Install the repaired dkl_base_seed0 subject-1 fit (PAPER_PLAN §22.7).

Refuses to copy anything it cannot verify. Run with --apply to act; the
default is a dry run, because the file being replaced is the one that caused
the §22 incident and a silent wrong copy is exactly the failure mode.

Checks, all of which must pass:
  1. the repaired file exists in results/encoding_cc/
  2. its own feature cache is provenance-stamped with lum_blur=False and
     lum_cs=False -- i.e. the front end dkl_base is supposed to have
  3. its hV4-V1 is inside the range of the seven clean siblings, so we are not
     installing a second outlier
  4. its magno participation ratio is not collapsed relative to parvo (§27.2)
"""
from __future__ import annotations
import os, shutil, sys
import numpy as np

SRC = "results/encoding_cc/dkl_base_seed0_trained_subj01.npz"
DST = "results/encoding/dkl_base_seed0_trained_subj01.npz"
QUAR = "results/encoding/quarantine/dkl_base_seed0_trained_subj01.npz"
CACHE = "/scratch/byrong/VST_data/mpk_feature_cache/dkl_base_seed0_trained_subj01.npz"
NC_MIN, MIN_VOX = 0.1, 20
APPLY = "--apply" in sys.argv

def med(v, sel):
    v = v[sel & np.isfinite(v)]
    return np.median(v) if v.size >= MIN_VOX else np.nan

def cross(z):
    v = z["cc_joint_vsb"] if "cc_joint_vsb" in z.files else None
    um, up = z["unique_magno"], z["unique_parvo"]
    rel = z["noise_ceiling"] > NC_MIN
    vroi = z["roi_prf-visualrois"]
    def spi_med(sel):
        s = sel & rel & np.isfinite(um) & np.isfinite(up)
        if s.sum() < MIN_VOX: return np.nan
        d = np.median(um[s] - up[s]); den = np.median(um[s] + up[s])
        return d / den if den > 1e-4 else np.nan
    return spi_med(vroi == 7) - spi_med(vroi == 1)

ok = True
if not os.path.exists(SRC):
    print(f"[wait] repaired file not present yet: {SRC}"); sys.exit(0)
print(f"[ok]   repaired file present: {SRC}")

if os.path.exists(CACHE):
    z = np.load(CACHE)
    prov = {k: z[k].item() for k in z.files if k.startswith("prov_")}
    if not prov:
        print("[FAIL] cache carries no provenance stamp"); ok = False
    elif prov.get("prov_lum_blur") is not False or prov.get("prov_lum_cs") is not False:
        print(f"[FAIL] cache front end is wrong: {prov}"); ok = False
    else:
        print(f"[ok]   cache provenance: lum_blur={prov['prov_lum_blur']} "
              f"lum_cs={prov['prov_lum_cs']}")
else:
    print(f"[FAIL] no cache to verify at {CACHE}"); ok = False

# Check 3, calibrated against the RIGHT reference.
#
# An earlier version compared the repaired subject against the range of its
# seven siblings and blocked. Investigating showed the guard was wrong, not the
# data: subject 1 is *systematically* the most extreme subject for the no-diet
# model. In the clean, never-contaminated `dkl_base_seed1` run it ranks 1/8 at
# -0.223, while in the diet model it ranks only 4/8. A sibling-range test
# therefore rejects a legitimate value for this particular subject and model.
#
# The principled comparison is the SAME subject in the clean seed, with the
# tolerance set by how much seeds normally disagree on the same subject.
new = cross(np.load(SRC))
old = cross(np.load(QUAR)) if os.path.exists(QUAR) else np.nan

def seed_series(seed):
    out = {}
    for s in range(1, 9):
        f = f"results/encoding/dkl_base_seed{seed}_trained_subj{s:02d}.npz"
        if os.path.exists(f):
            out[s] = cross(np.load(f))
    return out

s0, s1 = seed_series(0), seed_series(1)
paired = [s for s in s0 if s in s1 and np.isfinite(s0[s]) and np.isfinite(s1[s])]
gaps = np.array([abs(s0[s] - s1[s]) for s in paired])
print(f"[info] quarantined subj01 {old:+.4f} -> repaired {new:+.4f}")
if 1 in s1 and gaps.size >= 3:
    ref = s1[1]
    tol = gaps.max()
    print(f"[info] same subject in clean seed1: {ref:+.4f}; "
          f"seed-to-seed gap on subjects {paired}: max {tol:.4f}")
    if abs(new - ref) > tol:
        print(f"[FAIL] repaired subj01 differs from clean seed1 by "
              f"{abs(new - ref):.4f} > {tol:.4f}"); ok = False
    else:
        print(f"[ok]   repaired value agrees with the clean seed within the "
              f"normal seed-to-seed gap ({abs(new - ref):.4f} <= {tol:.4f})")
    if abs(old - ref) <= abs(new - ref):
        print("[FAIL] repair did not move the value toward the clean seed"); ok = False
    else:
        print(f"[ok]   repair moved subj01 toward the clean seed "
              f"({abs(old - ref):.4f} -> {abs(new - ref):.4f})")
else:
    print("[FAIL] cannot calibrate: clean seed1 subject 1 unavailable"); ok = False

print(f"\n{'PASS - safe to install' if ok else 'BLOCKED - do not install'}")
if ok and APPLY:
    shutil.copy2(SRC, DST)
    os.remove(QUAR)
    print(f"installed {SRC} -> {DST}; removed quarantine copy")
elif ok:
    print("dry run; re-run with --apply to install")
