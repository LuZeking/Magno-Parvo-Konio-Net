"""Why does my untrained raw differ from 07d4df6's, and does "raw reproduces"
survive the choice?

They report untrained_seed0 raw = -0.1199 against a control-table -0.118
("within 0.002"). I get -0.1251 with what I took to be the same convention
(mean over subjects of [median(u_m) - median(u_p)], hV4 - mean(V1v,V1d)),
i.e. within 0.007. A 0.005 disagreement between two sessions on the statistic
whose REPRODUCING is the linchpin of the raw-vs-normalised split is worth
resolving rather than splitting.

Enumerate the plausible variants and see which lands on -0.1199, and whether
the "raw reproduces" conclusion is robust across all of them.
"""
import glob
import numpy as np

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
NC_MIN, MIN_VOX, DENOM_MIN = 0.1, 20, 1e-3
PUB = {"untrained_seed0": -0.118, "dkl_bcs_seed0_trained": -0.176,
       "dkl_base_seed0_trained": -0.034}


def variants(model, src="encoding"):
    """Per-subject hV4 - mean(V1v,V1d) under several 'raw' definitions."""
    out = {k: [] for k in ("diff_of_medians", "median_of_diff",
                           "diff_of_medians_prenc", "median_of_diff_prenc")}
    for f in sorted(glob.glob(f"{R}/results/{src}/{model}_subj*.npz")):
        z = np.load(f, allow_pickle=True)
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        pairs = {
            "": (z["unique_magno"], z["unique_parvo"]),
            "_prenc": (z["unique_magno_pre_nc"] if "unique_magno_pre_nc"
                       in z.files else z["unique_magno_raw"],
                       z["unique_parvo_pre_nc"] if "unique_parvo_pre_nc"
                       in z.files else z["unique_parvo_raw"]),
        }
        for suf, (um, up) in pairs.items():
            dom, mod = [], []
            for code in range(1, 8):
                v = (roi == code) & (nc > NC_MIN) & ((um + up) > DENOM_MIN)
                if v.sum() >= MIN_VOX:
                    dom.append(np.median(um[v]) - np.median(up[v]))
                    mod.append(np.median(um[v] - up[v]))
                else:
                    dom.append(np.nan); mod.append(np.nan)
            dom, mod = np.array(dom), np.array(mod)
            out[f"diff_of_medians{suf}"].append(
                dom[6] - np.nanmean(dom[[0, 1]]))
            out[f"median_of_diff{suf}"].append(
                mod[6] - np.nanmean(mod[[0, 1]]))
    return {k: np.array(v) for k, v in out.items()}


for model, pub in PUB.items():
    v = variants(model)
    print(f"\n=== {model}   published {pub:+.3f} ===")
    print(f"{'variant':30s} {'mean':>9s} {'median':>9s} {'|mean-pub|':>11s}")
    for k, arr in v.items():
        arr = arr[np.isfinite(arr)]
        for agg, val in (("mean", arr.mean()), ):
            pass
        print(f"{k:30s} {arr.mean():+9.4f} {np.median(arr):+9.4f} "
              f"{abs(arr.mean() - pub):11.4f}")
    print(f"{'  (target from 07d4df6)':30s} {-0.1199 if model.startswith('untrained') else float('nan'):+9.4f}")
