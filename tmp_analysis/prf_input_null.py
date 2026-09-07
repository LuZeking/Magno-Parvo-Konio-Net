"""Is the ecc/size dissociation a NETWORK result or an input-level one?

The untrained pair (§100) shows ecc|size ~ +0.40/+0.47 raw in BOTH front ends,
larger than the trained diet arm's +0.300. So the dissociation exists before
any training. That raises the question §7.2 item 1 was written for: does the
network contribute anything, or is this carried by the transformed input?

`gabor_blur_cs` and `gabor_noblur` are the input-level nulls already fitted at
n=8 -- a Gabor-energy representation of each stream's own transformed input,
with no network at all. If they show the same ecc|size > 0, size|ecc < 0
pattern, then §98's dissociation is a property of the INPUT and the panel is a
front-end validation rather than a model result. That is the control that
"rescued the EEG track" (§7.2 item 1) and it applies here unchanged.

Every model here lives in results/encoding/ (§91).
"""
import glob
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/"
                   ".claude/worktrees/mpk-dev")

from src.analysis.encoding import nsd_data  # noqa: E402

SRC = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
       "mpk-dev/results/encoding")
CC = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/"
      "mpk-dev/results/encoding_cc")
_c = {}


def z(v):
    return (v - v.mean()) / v.std()


def prf(s):
    if s not in _c:
        m = nsd_data.load_mask(s, "nsdgeneral")
        _c[s] = (nsd_data.load_prf(s, "eccentricity", m),
                 nsd_data.load_prf(s, "size", m),
                 nsd_data.load_prf(s, "R2", m))
    return _c[s]


def betas(model, src, stat="raw", r2_min=5.0):
    out = {}
    for f in sorted(glob.glob(f"{src}/{model}_subj*.npz")):
        s = int(os.path.basename(f).split("subj")[-1].split(".")[0])
        d = np.load(f, allow_pickle=True)
        um, up, nc = d["unique_magno"], d["unique_parvo"], d["noise_ceiling"]
        ecc, size, r2 = prf(s)
        ok = ((nc > 0.1) & ((um + up) > 1e-3) & (r2 > r2_min)
              & np.isfinite(ecc) & np.isfinite(size) & (size > 0))
        if ok.sum() < 200:
            continue
        le, ls = z(np.log(ecc[ok] + 0.5)), z(np.log(size[ok]))
        y = z(((um[ok] - up[ok]) / (um[ok] + up[ok])) if stat == "norm"
              else (um[ok] - up[ok]))
        n = int(ok.sum())
        m2, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), le, ls]), y,
                                 rcond=None)
        out[s] = (m2[1], m2[2])
    return out


MODELS = [
    ("gabor_blur_cs", SRC, "INPUT NULL, diet front end"),
    ("gabor_noblur", SRC, "INPUT NULL, no-blur front end"),
    ("untrained_seed0", SRC, "untrained, diet front end"),
    ("untrained_seed0_noblur", SRC, "untrained, no-blur front end"),
    ("dkl_bcs_seed0_trained", CC, "TRAINED diet"),
    ("dkl_base_seed0_trained", CC, "TRAINED no diet"),
]

for stat in ("raw", "norm"):
    print(f"\n{'=' * 78}\nstatistic = {stat}\n{'=' * 78}")
    print(f"{'model':26s} {'n':>2s} {'ecc|size':>22s} {'size|ecc':>22s}")
    store = {}
    for model, src, label in MODELS:
        b = betas(model, src, stat)
        if not b:
            print(f"{model:26s}  -- not found")
            continue
        store[model] = b
        subs = sorted(b)
        e = np.array([b[s][0] for s in subs])
        z_ = np.array([b[s][1] for s in subs])
        pe, pz = sps.wilcoxon(e).pvalue, sps.wilcoxon(z_).pvalue
        print(f"{model:26s} {len(subs):2d} "
              f"{f'{e.mean():+.3f} {int((e>0).sum())}/{len(e)} p={pe:.4f}':>22s} "
              f"{f'{z_.mean():+.3f} {int((z_<0).sum())}/{len(z_)}neg p={pz:.4f}':>22s}")
        print(f"{'':26s}    ({label})")

    # the decisive contrast: does the network add anything over its own input?
    print(f"\n  {'contrast (paired, ecc|size)':44s} {'diff':>8s} {'pos':>5s} "
          f"{'p':>8s}")
    for a, b_, lab in (
            ("untrained_seed0", "gabor_blur_cs",
             "untrained diet  -  input null diet"),
            ("dkl_bcs_seed0_trained", "gabor_blur_cs",
             "TRAINED diet  -  input null diet"),
            ("dkl_base_seed0_trained", "gabor_blur_cs",
             "TRAINED no-diet  -  input null diet"),
            ("dkl_bcs_seed0_trained", "untrained_seed0",
             "TRAINED diet  -  untrained diet"),
            ("dkl_base_seed0_trained", "untrained_seed0",
             "TRAINED no-diet  -  untrained diet")):
        if a not in store or b_ not in store:
            continue
        subs = sorted(set(store[a]) & set(store[b_]))
        d = np.array([store[a][s][0] - store[b_][s][0] for s in subs])
        print(f"  {lab:44s} {d.mean():+8.3f} {int((d > 0).sum()):3d}/{len(d)} "
              f"{sps.wilcoxon(d).pvalue:8.4f}")

print("\nIf the input null already carries ecc|size > 0, the dissociation is a")
print("front-end property and §98 is a front-end validation, not a model")
print("result. If the network adds to it, that difference is the model result.")
