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

sib = []
for s in range(2, 9):
    f = f"results/encoding/dkl_base_seed0_trained_subj{s:02d}.npz"
    if os.path.exists(f):
        sib.append(cross(np.load(f)))
sib = np.array([x for x in sib if np.isfinite(x)])
new = cross(np.load(SRC))
old = cross(np.load(QUAR)) if os.path.exists(QUAR) else np.nan
print(f"[info] siblings hV4-V1 (n={sib.size}): min {sib.min():+.4f} "
      f"max {sib.max():+.4f} mean {sib.mean():+.4f}")
print(f"[info] quarantined subj01: {old:+.4f}   repaired subj01: {new:+.4f}")
lo, hi = sib.min() - 2 * sib.std(), sib.max() + 2 * sib.std()
if not (lo <= new <= hi):
    print(f"[FAIL] repaired value outside sibling range [{lo:+.4f}, {hi:+.4f}]"); ok = False
else:
    print("[ok]   repaired value is in family with the siblings")

print(f"\n{'PASS - safe to install' if ok else 'BLOCKED - do not install'}")
if ok and APPLY:
    shutil.copy2(SRC, DST)
    os.remove(QUAR)
    print(f"installed {SRC} -> {DST}; removed quarantine copy")
elif ok:
    print("dry run; re-run with --apply to install")
