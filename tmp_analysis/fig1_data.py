"""Extract everything the four-panel Figure 1 needs into one cache.

PAPER_PLAN §46.3, as amended by §47.4 and §49. The figure is a CONVERGENCE
figure: four instruments that agree on the diet-vs-no-diet contrast. It is not
four measurements of one construct -- §47.4 showed shape bias and effective
dimensionality factorise differently over identical inputs, so the panels are
laid out as separate instruments and the caption must say so.

  (a) §25  the input 2x2 -- behaviour and probe side by side
  (b) §43.1 in-silico cue conflict -> BOLD, per-subject magno-parvo VSB
  (c) §49  the set point -- front end vs trained, three conditions
  (d) §43.3 the cost -- median r2/NC, diet vs no diet

Panel (d) is not optional (§46.3). A four-panel figure whose panels all favour
the diet, omitting the one measure that does not, would be cherry-picking.

Writes results/figures/fig1_data.npz
"""
from __future__ import annotations
import glob, json, os
import numpy as np

from src.analysis.encoding.stats import load_results

NC_MIN, MIN_VOX = 0.1, 20
out = {}

# ---------------------------------------------------------------- panel (a)
# The 2x2 on INPUT factors. Each cell is the stream that receives the named
# carrier: for three cells that is the parvo stream, for the diet cell it is
# magno. Seeds are the replication unit here (not subjects) -- these are model
# properties measured on the Geirhos set, with no brain data involved.
#
# parvo_highsf seed 0 is EXCLUDED: its cue-conflict run decided only 139/150
# images when the convergence guard ran, so its shape-bias denominator is not
# comparable. Seeds 1 and 2 agree to 0.012.
#
# swap seed 2 IS included, and §25's table (n=2) is superseded. It was excluded
# there by the same guard while still training (§20.2, "96/150"); it was re-run
# 2026-08-31 22:50 after converging and now decides 182/1200 images -- MORE than
# seeds 0 and 1 (159, 155). Including it moves the cell by +0.011 (behaviour)
# and +0.010 (probe) and changes no claim. See §50.
#
# The decided-image count is carried through to the figure because it varies
# 4x across cells (155-595) and it bounds how precisely each behavioural point
# can be read: a stream that answers "shape or texture" for only ~13% of the
# set returns something near a coin flip by construction (§18.5, §20.3).
CELLS = {                       # (carrier, transform) -> (models, which stream)
    ("chromatic", "none"):      ([f"dkl_base_seed{s}" for s in (0, 1, 2)], "parvo"),
    ("chromatic", "transform"): ([f"dkl_bcs_swap_seed{s}" for s in (0, 1, 2)], "parvo"),
    ("achromatic", "none"):     ([f"dkl_bcs_parvo_highsf_seed{s}" for s in (1, 2)], "parvo"),
    ("achromatic", "transform"):([f"dkl_bcs_seed{s}" for s in (0, 1, 2)], "magno"),
}
for (carrier, trans), (models, stream) in CELLS.items():
    beh, prb, dec = [], [], []
    for m in models:
        fb = f"results/cue_conflict/{m}.json"
        fp = f"results/shape_texture/{m}_trained.json"
        if os.path.exists(fb):
            c = json.load(open(fb))["conditions"][f"{stream}_only"]
            beh.append(c["shape_bias"])
            dec.append(c["n_shape"] + c["n_texture"])
        else:
            print(f"  MISSING {fb}")
        if os.path.exists(fp):
            prb.append(json.load(open(fp))["streams"][stream]["bias"])
        else:
            print(f"  MISSING {fp}")
    k = f"a__{carrier}__{trans}"
    out[k + "__beh"] = np.array(beh)
    out[k + "__probe"] = np.array(prb)
    out[k + "__decided"] = np.array(dec)
    print(f"(a) {carrier:>10s} x {trans:<9s} [{stream:5s}] "
          f"behaviour {np.mean(beh):.3f} (n={len(beh)})   probe {np.mean(prb):+.3f}"
          f"   decided {int(np.mean(dec))}/1200")

# ---------------------------------------------------------------- panel (b)
# In-silico cue conflict through the fitted encoding models. Per subject, the
# median VSB of each stream over reliable voxels; the contrast is magno-parvo.
def cc_streams(model):
    subs, mg, pv = [], [], []
    for f in sorted(glob.glob(f"results/encoding_cc/{model}_subj*.npz")):
        z = np.load(f)
        ok = (z["noise_ceiling"] > NC_MIN)
        def med(v):
            v = v[ok & np.isfinite(v)]
            return np.median(v) if v.size >= MIN_VOX else np.nan
        subs.append(int(f.rsplit("subj", 1)[1][:2]))
        mg.append(med(z["cc_magno_vsb"])); pv.append(med(z["cc_parvo_vsb"]))
    return np.array(subs), np.array(mg), np.array(pv)

for key, model in [("diet", "dkl_bcs_seed0_trained"), ("nodiet", "dkl_base_seed0_trained")]:
    s, mg, pv = cc_streams(model)
    out[f"b__{key}__subs"], out[f"b__{key}__magno"], out[f"b__{key}__parvo"] = s, mg, pv
    print(f"(b) {key:6s} n={len(s)}  magno {mg.mean():+.3f}  parvo {pv.mean():+.3f}  "
          f"magno-parvo {np.mean(mg - pv):+.3f}")

# ---------------------------------------------------------------- panel (c)
# The set point. Read from the PR store built by pr_retention_highsf.py so the
# eigendecompositions are not repeated; each entry is (magno PR, parvo PR).
STORE = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "pr_store.npz")
_pr = dict(np.load(STORE))
def ratio(model):
    d = {int(k.partition("|subj")[2]): v[0] / v[1]
         for k, v in _pr.items() if k.partition("|subj")[0] == model}
    return np.array([d[s] for s in sorted(d)])

# Four arms, three SEEDS each. §58 replicated this panel's instrument across
# training seeds and found the trained LEVEL is not reproducible (the diet-on-lum
# arm spans 1.603-2.507; diet-on-rg spans 1.995-4.383). Earlier versions of this
# panel labelled one seed's level per arm, which asserted a set point the text
# has since withdrawn. The panel now plots every seed and claims only the SIDE
# of parity -- which does replicate, 12/12 seeds, 8/8 subjects each.
ARMS = [("diet_normal",  "untrained_seed0",        "dkl_bcs_seed{}_trained"),
        ("diet_highsf",  "untrained_parvo_highsf", "dkl_bcs_parvo_highsf_seed{}_trained"),
        ("diet_swap",    "untrained_swap",         "dkl_bcs_swap_seed{}_trained"),
        ("nodiet",       "untrained_seed0_noblur", "dkl_base_seed{}_trained")]
for key, front, trained in ARMS:
    f = ratio(front)
    out[f"c__{key}__front"] = f
    seed_means = []
    for sd in (0, 1, 2):
        t = ratio(trained.format(sd))
        out[f"c__{key}__trained{sd}"] = t
        seed_means.append(t.mean())
        # the claim, per seed: how many subjects land on the diet side of parity
        out[f"c__{key}__above{sd}"] = np.array([(t > 1.0).sum(), len(t)])
    out[f"c__{key}__seedmeans"] = np.array(seed_means)
    print(f"(c) {key:12s} front {f.mean():.3f} -> trained seeds "
          f"{np.array2string(np.array(seed_means), precision=3)}  "
          f"range {np.ptp(seed_means):.3f}")

# ---------------------------------------------------------------- panel (d)
# The cost. Median r2_joint/NC over reliable voxels, per subject.
for key, model in [("diet", "dkl_bcs_seed0_trained"), ("nodiet", "dkl_base_seed0_trained")]:
    R = load_results("results/encoding", model)
    subs = sorted(R)
    v = []
    for s in subs:
        z = R[s]
        ok = (z["noise_ceiling"] > NC_MIN) & np.isfinite(z["r2_joint_nc"])
        v.append(np.median(z["r2_joint_nc"][ok]))
    out[f"d__{key}__subs"], out[f"d__{key}__r2nc"] = np.array(subs), np.array(v)
    print(f"(d) {key:6s} n={len(subs)}  median r2/NC {np.mean(v):.3f}")

os.makedirs("results/figures", exist_ok=True)
np.savez("results/figures/fig1_data.npz", **out)
print("\nwrote results/figures/fig1_data.npz")
