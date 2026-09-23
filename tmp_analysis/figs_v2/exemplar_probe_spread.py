"""Spread of the held-out probe over preprocessing variants (docs/PLAN_EXEMPLAR_2026-09-21.md, addendum C,
item 3; post hoc, no new computation).

  python tmp_analysis/figs_v2/exemplar_probe_spread.py

Reads results/figures_v2/exemplar_zoo.json. For each probe variant over the 38 ImageNet networks: median, range
and count below zero of the bias; Kendall tau of its ranking against the trained-rate ranking; Spearman rho
against the primary Q1; the Stylized-ImageNet ResNet-50's bias and whether it exceeds every ImageNet network.
Writes results/figures_v2/exemplar_probe_spread.json.
"""
import json

import numpy as np
from scipy.stats import kendalltau, spearmanr

VARIANTS = ("Q1", "Q2", "S64", "S256n", "Q1b", "Q0")     # Q1 primary; Q0 exemplar-shared


def main():
    z = json.load(open("results/figures_v2/exemplar_zoo.json"))
    im = [r for r in z["networks"] if r["group"] == "imagenet"]
    sin = next(r for r in z["networks"] if r["tag"] == "hub_resnet50_sin")
    tr = np.array([r["trained"] for r in im]); q1 = np.array([r["probes"]["Q1"]["bias"] for r in im])
    out = dict(plan="docs/PLAN_EXEMPLAR_2026-09-21.md addendum C (post hoc)", n_imagenet=len(im), variants={})
    for k in VARIANTS:
        v = np.array([r["probes"][k]["bias"] for r in im])
        out["variants"][k] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()), n_below_0=int((v < 0).sum()),
                                  tau_trained=float(kendalltau(tr, v).statistic), rho_Q1=float(spearmanr(q1, v).correlation),
                                  sin=float(sin["probes"][k]["bias"]), sin_above_all=bool(sin["probes"][k]["bias"] > v.max()))
        o = out["variants"][k]
        print(f"{k:6s} median {o['median']:+.3f} [{o['min']:+.3f}, {o['max']:+.3f}] below 0 {o['n_below_0']}/38; "
              f"tau(trained) {o['tau_trained']:.2f}; rho(Q1) {o['rho_Q1']:.2f}; SIN {o['sin']:+.3f} above all {o['sin_above_all']}")
    held = [k for k in VARIANTS if k != "Q0"]
    out["heldout_tau_range"] = [min(out["variants"][k]["tau_trained"] for k in held), max(out["variants"][k]["tau_trained"] for k in held)]
    json.dump(out, open("results/figures_v2/exemplar_probe_spread.json", "w"), indent=1)


if __name__ == "__main__":
    main()
