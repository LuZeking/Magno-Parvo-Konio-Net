"""How to report shape bias, in one command, for any torchvision ImageNet network.

  python -m src.analysis.generality.report --model tv:resnet50
  python -m src.analysis.generality.report --model tv:convnext_base --work-dir /tmp/mpk_report

Runs the paper's pipeline on a frozen network (Methods 4.3, 4.8; docs/PLAN_ZOO_2026-09-15.md)
and prints the three numbers a shape-bias claim should carry:

  1. the cue-conflict shape-bias rate at the trained readout (Geirhos 16-class rule);
  2. the span of rates the same frozen features reach under a reciprocal gain on the network's
     own shape- and texture-predictive subspaces, restricted to readouts within 5 points of the
     trained top-1 on a stratified 10-per-class ILSVRC-2012 validation subset, scored on object and
     texture exemplars the subspaces were not fitted on (docs/PLAN_EXEMPLAR_2026-09-21.md; the
     exemplar-shared span and the unit-level median split are printed beside it);
  3. the probe-based representational bias: shape minus texture accuracy of an exemplar-balanced
     linear probe scored on held-out exemplars, with its 5-95% interval over exemplar draws (the
     exemplar-shared probe beside it).

None of the three is a model-quality or human-likeness score, and the rate (1) can be moved by
the readout without changing the features: report all three, never the rate alone.

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
    if not os.path.exists(os.path.join(beh, f"{tag}_exemplar.json")):
        run("exemplar", "--features", cc, "--behaviour-dir", beh, "--out-dir", beh)

    z = json.load(open(os.path.join(beh, f"{tag}_subspace.json")))["summary"]
    u = json.load(open(os.path.join(beh, f"{tag}_unit_factor.json")))["summary"]
    pb = json.load(open(os.path.join(beh, f"{tag}_probe_bias.json")))
    ex = json.load(open(os.path.join(beh, f"{tag}_exemplar.json")))
    e1, q1 = ex["spans"]["E1"], ex["probes"]["Q1"]
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
        print(f"2. span within 5 pts of top-1, exemplars held out {e1['primary']['gated_span']:.3f}   "
              f"(rate {e1['primary']['texture_end']:.3f} -> {e1['primary']['shape_end']:.3f}; forced two-way {e1['forced']['gated_span']:.3f}; "
              f"range over repetitions {e1['primary']['span_min']:.3f}-{e1['primary']['span_max']:.3f})")
        print(f"   exemplars shared (the published estimator)     {span:.3f}   "
              f"(rate {lo:.3f} -> {hi:.3f} for w in [{min(g['w_within_5pts']):.2f}, {max(g['w_within_5pts']):.2f}]; "
              f"top-1 {g['trained_top1']:.3f}, min {top1[m].min():.3f}; ungated {z['primary']['span_mean']:.3f}, rho {z['primary']['rho_mean']:.2f})")
        if not u.get("undefined"):
            ug = json.load(open(os.path.join(beh, f"{tag}_gain_accuracy.json")))
            uspan = gated(u["primary"]["mean_curve"], ws, ug["w_within_5pts"])[0]
            print(f"   under the unit-level median split               {uspan:.3f}   (ungated {u['primary']['span_mean']:.3f})")
    else:
        print(f"2. UNGATED span, exemplars held out              {e1['primary']['span_mean']:.3f}   (no accuracy gate was applied, so this is not "
              f"comparable with the paper's gated spans; exemplars shared {z['primary']['span_mean']:.3f})")
        if not u.get("undefined"):
            print(f"   under the unit-level median split               {u['primary']['span_mean']:.3f}")
    print(f"3. probe bias, exemplars held out                {q1['bias']:+.3f}   "
          f"(5-95% over draws {q1['bias_p05']:+.3f} to {q1['bias_p95']:+.3f}; shape {q1['shape_acc']:.3f}, texture {q1['texture_acc']:.3f}; "
          f"chance {pb['chance']:.3f})")
    print(f"   exemplars shared (the published estimator)     {pb['probe_bias']:+.3f}   "
          f"(shape {pb['shape_acc']:.3f}, texture {pb['texture_acc']:.3f})")
    print("=" * 78)
    print("Rule, gate and estimators: Methods 4.3 and 4.8; docs/PLAN_SUBSPACE_2026-09-15.md; docs/PLAN_ZOO_2026-09-15.md;")
    print("docs/PLAN_EXEMPLAR_2026-09-21.md. Not a model-quality or human-likeness score; report all three numbers.")


if __name__ == "__main__":
    main()
