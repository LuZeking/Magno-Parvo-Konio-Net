"""How to report shape bias, in one command, for any torchvision ImageNet network.

  python -m src.analysis.generality.report --model tv:resnet50
  python -m src.analysis.generality.report --model tv:convnext_base --work-dir /tmp/mpk_report

Runs the paper's pipeline on a frozen network (Methods 4.3, 4.8; docs/PLAN_ZOO_2026-09-15.md)
and prints the three numbers a shape-bias claim should carry:

  1. the cue-conflict shape-bias rate at the trained readout (Geirhos 16-class rule);
  2. the span of rates the same frozen features reach under one gain on the network's own
     shape-predictive subspace, restricted to readouts within 5 points of the trained top-1
     on a stratified 10-per-class ILSVRC-2012 validation subset (the unit-level median split
     is printed beside it);
  3. the probe-based representational bias: shape minus texture accuracy of a linear probe.

Each stage writes its file under --work-dir/<features|behaviour> and is skipped when the
file exists, so re-running is free. Needs the cue-conflict set (src/analysis/behavior/
cue_conflict.py) and, for the gate, ILSVRC-2012 validation images at $IMAGENET_VAL
(default /scratch/byrong/imagenet/val). CPU is enough: ~20 min for a ResNet-50.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

WORK = "/scratch/byrong/mpk/generality/report"


def run(mod: str, *args: str) -> None:
    cmd = [sys.executable, "-m", f"src.analysis.generality.{mod}", *args]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def gated(curve, ws, gate):
    m = (ws >= min(gate) - 1e-9) & (ws <= max(gate) + 1e-9)
    c = np.asarray(curve)[m]
    return float(c.max() - c.min()), float(c[0]), float(c[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="tv:<arch>[:<weights>] | hub:<name> | run:<name> | concat:<name>")
    p.add_argument("--work-dir", default=WORK)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--per-class", type=int, default=10, help="ILSVRC validation images per class for the gate")
    p.add_argument("--no-gate", action="store_true", help="skip the accuracy gate (no ImageNet images available)")
    args = p.parse_args()

    tag = args.model.replace(":", "_")
    feat, beh = os.path.join(args.work_dir, "features"), os.path.join(args.work_dir, "behaviour")
    cc, val = os.path.join(feat, f"{tag}_cue_conflict.npz"), os.path.join(feat, f"{tag}_imagenet_val.npz")
    if not os.path.exists(cc):
        run("extract", "--model", args.model, "--set", "cue_conflict", "--out-dir", feat)
    scorer = str(np.load(cc, allow_pickle=True)["scorer"])
    gate_ok = not args.no_gate and scorer == "imagenet16"
    if gate_ok and not os.path.exists(val):
        run("extract", "--model", args.model, "--set", "imagenet_val", "--out-dir", feat, "--per-class", str(args.per_class))
    if not os.path.exists(os.path.join(beh, f"{tag}_subspace.npz")):
        run("subspace_factor", "--features", cc, "--out-dir", beh, "--n-splits", str(args.n_splits))
    if not os.path.exists(os.path.join(beh, f"{tag}_unit_factor.npz")):
        run("unit_factor", "--features", cc, "--out-dir", beh, "--n-splits", str(args.n_splits))
    if gate_ok:
        if not os.path.exists(os.path.join(beh, f"{tag}_gain_accuracy_subspace.json")):
            run("gain_accuracy", "--factor", os.path.join(beh, f"{tag}_subspace.npz"), "--val", val, "--out-dir", beh, "--out-suffix", "_subspace")
        if not os.path.exists(os.path.join(beh, f"{tag}_gain_accuracy.json")):
            run("gain_accuracy", "--factor", os.path.join(beh, f"{tag}_unit_factor.npz"), "--val", val, "--out-dir", beh)
    if not os.path.exists(os.path.join(beh, f"{tag}_probe_bias.json")):
        run("probe_bias", "--features", cc, "--out-dir", beh)

    z = json.load(open(os.path.join(beh, f"{tag}_subspace.json")))["summary"]
    u = json.load(open(os.path.join(beh, f"{tag}_unit_factor.json")))["summary"]
    pb = json.load(open(os.path.join(beh, f"{tag}_probe_bias.json")))
    ws = np.array(z["ws"])
    tr = z["trained"]
    print("\n" + "=" * 78)
    print(f"{args.model}  ({str(np.load(cc, allow_pickle=True)['checkpoint'])}); D = {z['n_units']} penultimate units")
    print("-" * 78)
    print(f"1. shape-bias rate at the trained readout        {tr['shape_bias']:.3f}   "
          f"(forced two-way {tr['forced_two_way']:.3f}; {tr['n_shape'] + tr['n_texture']}/{z['n_images']} images decided)")
    if gate_ok:
        g = json.load(open(os.path.join(beh, f"{tag}_gain_accuracy_subspace.json")))
        span, lo, hi = gated(z["primary"]["mean_curve"], ws, g["w_within_5pts"])
        top1 = np.array(g["top1"]); m = np.isin(ws, g["w_within_5pts"])
        print(f"2. reachable span within 5 points of top-1       {span:.3f}   "
              f"(rate {lo:.3f} -> {hi:.3f} for w in [{min(g['w_within_5pts']):.2f}, {max(g['w_within_5pts']):.2f}]; "
              f"top-1 {g['trained_top1']:.3f}, min {top1[m].min():.3f}; ungated {z['primary']['span_mean']:.3f}, rho {z['primary']['rho_mean']:.2f})")
        if not u.get("undefined"):
            ug = json.load(open(os.path.join(beh, f"{tag}_gain_accuracy.json")))
            uspan = gated(u["primary"]["mean_curve"], ws, ug["w_within_5pts"])[0]
            print(f"   under the unit-level median split               {uspan:.3f}   (ungated {u['primary']['span_mean']:.3f})")
    else:
        print(f"2. reachable span (ungated; no accuracy gate)     {z['primary']['span_mean']:.3f}   (rho {z['primary']['rho_mean']:.2f})")
        if not u.get("undefined"):
            print(f"   under the unit-level median split               {u['primary']['span_mean']:.3f}")
    print(f"3. probe-based representational bias             {pb['probe_bias']:+.3f}   "
          f"(shape {pb['shape_acc']:.3f}, texture {pb['texture_acc']:.3f}; chance {pb['chance']:.3f})")
    print("=" * 78)
    print("Rule, gate and estimators: Methods 4.3 and 4.8; docs/PLAN_SUBSPACE_2026-09-15.md; docs/PLAN_ZOO_2026-09-15.md.")


if __name__ == "__main__":
    main()
