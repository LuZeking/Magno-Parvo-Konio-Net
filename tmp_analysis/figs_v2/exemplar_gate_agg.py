"""Aggregate array C (docs/PLAN_EXEMPLAR_2026-09-21.md, addendum C; post hoc): the accuracy gate on the
held-out bases.

  PYTHONPATH=. python tmp_analysis/figs_v2/exemplar_gate_agg.py

Gates (Methods 4.8 rule: w-range whose top-1 stays within 5 points of the trained top-1):
  allimage  top-1 of the bases fitted on all 1,200 images (the manuscript's gate until now);
  heldout   mean top-1 over the 30 E1 fold bases (primary under addendum C);
  strict    every one of the 30 fold bases within 5 points.
Per network: the gate, the worst drop of top-1, and the E1 gated span and shape end under each gate.
Over the 38 ImageNet networks: gated-span range/median, number >= 0.20, reversal against the trained
ranking and Kendall tau. Writes results/figures_v2/exemplar_gate.json.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import kendalltau

GATE_DIR = os.environ.get("GATE_DIR", "/scratch/byrong/mpk/exemplar/results/2026-09-23_gate")
OUT = "results/figures_v2/exemplar_gate.json"
BUDGET = 0.05


def gated(mc, ws, ok):
    lo, hi = ws[ok].min(), ws[ok].max()
    m = (ws >= lo - 1e-9) & (ws <= hi + 1e-9)
    c = mc[m]
    return dict(w_range=[float(lo), float(hi)], gated_span=float(c.max() - c.min()), shape_end=float(c[-1]),
                texture_end=float(c[0]))


def main():
    z = json.load(open("results/figures_v2/exemplar_zoo.json"))
    by = {r["tag"]: r for r in z["networks"]}
    rows = []
    for f in sorted(glob.glob(os.path.join(GATE_DIR, "*_gate.json"))):
        d = json.load(open(f)); t = d["tag"]
        ws = np.array(d["ws"]); mc = np.array(d["mean_curve"]); F = np.array(d["fold_top1"])
        ref = d["trained_top1"]
        A = np.array(d["allimage_top1"])
        masks = dict(allimage=A >= ref - BUDGET, heldout=F.mean(0) >= ref - BUDGET, strict=F.min(0) >= ref - BUDGET)
        row = dict(tag=t, label=by[t]["label"], group=by[t]["group"], trained=by[t]["trained"],
                   trained_top1=ref, repro_max_abs_diff=d["repro_max_abs_diff"], n_fold_bases=int(len(F)),
                   worst_drop_allimage=float(ref - A.min()), worst_drop_heldout=float(ref - F.mean(0).min()),
                   worst_drop_single_basis=float(ref - F.min()),
                   n_bases_leaving=int((F.min(1) < ref - BUDGET).sum()),
                   fold_top1_mean=F.mean(0).tolist(), allimage_top1=A.tolist())
        for k, m in masks.items():
            row[k] = gated(mc, ws, m)
        assert abs(row["allimage"]["gated_span"] - by[t]["spans"]["E1"]["gated"]) < 1e-9, t
        rows.append(row)

    im = [r for r in rows if r["group"] == "imagenet"]
    tr = np.array([r["trained"] for r in im]); ii, jj = np.triu_indices(len(im), 1)
    summary = dict(n_networks=len(rows), n_imagenet=len(im),
                   repro_max_abs_diff=max(r["repro_max_abs_diff"] for r in rows),
                   worst_drop_allimage_max=max(r["worst_drop_allimage"] for r in im),
                   worst_drop_heldout_max=max(r["worst_drop_heldout"] for r in im),
                   worst_drop_heldout_median=float(np.median([r["worst_drop_heldout"] for r in im])),
                   worst_drop_allimage_median=float(np.median([r["worst_drop_allimage"] for r in im])),
                   ratio_heldout_to_allimage_median=float(np.median([r["worst_drop_heldout"] / max(r["worst_drop_allimage"], 1e-9) for r in im])),
                   narrowed_heldout=[r["tag"] for r in rows if r["heldout"]["w_range"] != [0.0, 1.0]],
                   narrowed_strict=[r["tag"] for r in rows if r["strict"]["w_range"] != [0.0, 1.0]],
                   narrowed_allimage=[r["tag"] for r in rows if r["allimage"]["w_range"] != [0.0, 1.0]])
    for k in ("allimage", "heldout", "strict"):
        g = np.array([r[k]["gated_span"] for r in im]); se = np.array([r[k]["shape_end"] for r in im])
        rev = ((tr[ii] - tr[jj]) * (se[ii] - se[jj]) < 0)
        summary[k] = dict(gated_min=float(g.min()), gated_max=float(g.max()), gated_median=float(np.median(g)),
                          n_ge_0_20=int((g >= 0.20).sum()), frac_reversed=float(rev.mean()), n_reversed=int(rev.sum()),
                          tau=float(kendalltau(tr, se).statistic))
    json.dump(dict(plan="docs/PLAN_EXEMPLAR_2026-09-21.md addendum C (post hoc)", source=GATE_DIR,
                   summary=summary, networks=rows), open(OUT, "w"), indent=1)
    s = summary
    print(f"{len(rows)} networks; reproduction max |d| {s['repro_max_abs_diff']:.2e}")
    print(f"worst top-1 drop, 38 ImageNet: all-image max {100*s['worst_drop_allimage_max']:.2f} (median {100*s['worst_drop_allimage_median']:.2f}); "
          f"held-out max {100*s['worst_drop_heldout_max']:.2f} (median {100*s['worst_drop_heldout_median']:.2f}); ratio median {s['ratio_heldout_to_allimage_median']:.2f}")
    print(f"gate narrowed: all-image {s['narrowed_allimage']}; held-out {s['narrowed_heldout']}; strict {s['narrowed_strict']}")
    for k in ("allimage", "heldout", "strict"):
        v = s[k]
        print(f"  {k:9s} gated span {v['gated_min']:.3f}-{v['gated_max']:.3f} (median {v['gated_median']:.3f}), >=0.20 in {v['n_ge_0_20']}; "
              f"reversal {v['frac_reversed']:.3f} ({v['n_reversed']}/703), tau {v['tau']:.3f}")
    for r in rows:
        if r["heldout"]["w_range"] != [0.0, 1.0] or r["group"] != "imagenet":
            print(f"   {r['tag']:24s} gate all-image {r['allimage']['w_range']} held-out {r['heldout']['w_range']} strict {r['strict']['w_range']}; "
                  f"span {r['allimage']['gated_span']:.3f} / {r['heldout']['gated_span']:.3f} / {r['strict']['gated_span']:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
