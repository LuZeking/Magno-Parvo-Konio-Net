"""The ranking's test-retest floor by resampling unit (docs/PLAN_EXEMPLAR_2026-09-21.md, addendum C; post hoc).

  PYTHONPATH=. python tmp_analysis/figs_v2/exemplar_floors.py

For the 38 ImageNet networks, the trained ranking is re-estimated twice on independent resamples, and the
fraction of the 703 pairs whose order differs is the floor. Three resampling units: images (the floor
already reported, 8.9%), object-exemplar clusters (160) and texture-exemplar clusters (48), a cluster
carrying all of its images. A pair is resolved under a unit when the 95% interval of its trained-rate
difference over the draws excludes zero. The observed reversal (trained rate against the rate at the shape
end of the gate, E1) is given over all pairs and over the resolved pairs, for the gate on the all-image
bases (Methods 4.8) and, when results/figures_v2/exemplar_gate.json exists, the gate on the held-out bases.

Reads the per-image trained decisions of array A (<tag>_exemplar.npz); writes
results/figures_v2/exemplar_floors.json.
"""
import json
import os

import numpy as np

EX_DIR = os.environ.get("EXEMPLAR_DIR", "/scratch/byrong/mpk/exemplar/results/2026-09-21_exemplar")
N_DRAWS = 2000
OUT = "results/figures_v2/exemplar_floors.json"


def main():
    z = json.load(open("results/figures_v2/exemplar_zoo.json"))
    im = [r for r in z["networks"] if r["group"] == "imagenet"]
    tags = [r["tag"] for r in im]
    Hs, Ht = [], []
    for t in tags:
        d = np.load(os.path.join(EX_DIR, f"{t}_exemplar.npz"))
        Hs.append(d["hit_shape"].astype(float)); Ht.append(d["hit_texture"].astype(float))
        obj, tex = d["object_exemplar"], d["texture_exemplar"]
    Hs, Ht = np.array(Hs), np.array(Ht)
    n_img = Hs.shape[1]
    tr = Hs.sum(1) / (Hs.sum(1) + Ht.sum(1))
    assert np.allclose(tr, [r["trained"] for r in im]), "trained rates do not match exemplar_zoo.json"
    ii, jj = np.triu_indices(len(tags), 1)
    rev = lambda a, b: (a[ii] - a[jj]) * (b[ii] - b[jj]) < 0

    shape_ends = {"allimage_gate": np.array([r["spans"]["E1"]["shape_end"] for r in im])}
    if os.path.exists("results/figures_v2/exemplar_gate.json"):
        g = {n["tag"]: n for n in json.load(open("results/figures_v2/exemplar_gate.json"))["networks"]}
        shape_ends["heldout_gate"] = np.array([g[t]["heldout"]["shape_end"] for t in tags])

    units = {"image": np.arange(n_img)}
    for name, lab in (("object_exemplar", obj), ("texture_exemplar", tex)):
        _, units[name] = np.unique(lab, return_inverse=True)
    rng = np.random.default_rng(0)
    out = dict(plan="docs/PLAN_EXEMPLAR_2026-09-21.md addendum C (post hoc)", n_networks=len(tags),
               n_pairs=int(len(ii)), n_draws=N_DRAWS, source=EX_DIR, units={})
    for name, u in units.items():
        k = int(u.max()) + 1
        R, flips = [], []
        for _ in range(N_DRAWS):
            c1 = np.bincount(rng.integers(0, k, k), minlength=k)[u].astype(float)
            c2 = np.bincount(rng.integers(0, k, k), minlength=k)[u].astype(float)
            r1 = (Hs @ c1) / (Hs @ c1 + Ht @ c1); r2 = (Hs @ c2) / (Hs @ c2 + Ht @ c2)
            R.append(r1); flips.append(rev(r1, r2))
        R = np.array(R); flips = np.array(flips)
        D = R[:, ii] - R[:, jj]
        res = (np.percentile(D, 2.5, 0) > 0) | (np.percentile(D, 97.5, 0) < 0)
        f = flips.mean(1)
        row = dict(n_clusters=k, floor_all=float(f.mean()), floor_all_ci=[float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))],
                   n_resolved=int(res.sum()), floor_resolved=float(flips[:, res].mean()) if res.any() else None,
                   observed={})
        for key, se in shape_ends.items():
            o = rev(tr, se)
            row["observed"][key] = dict(all=float(o.mean()), n_all=int(o.sum()),
                                        resolved=float(o[res].mean()) if res.any() else None, n_resolved_reversed=int(o[res].sum()))
        out["units"][name] = row
        print(f"{name:17s} ({k:4d}) floor all {row['floor_all']:.3f} [{row['floor_all_ci'][0]:.3f}, {row['floor_all_ci'][1]:.3f}]; "
              f"resolved {row['n_resolved']}/{len(ii)}, floor there {row['floor_resolved']:.4f}; observed "
              + "; ".join(f"{k2}: all {v['all']:.3f}, resolved {v['resolved']:.3f} ({v['n_resolved_reversed']}/{row['n_resolved']})"
                          for k2, v in row["observed"].items()))
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
