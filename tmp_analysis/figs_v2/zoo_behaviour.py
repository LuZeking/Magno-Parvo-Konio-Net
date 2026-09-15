"""Aggregate the model-zoo run (docs/PLAN_ZOO_2026-09-15.md) and evaluate P-Z1-P-Z4.

  PYTHONPATH=. python tmp_analysis/figs_v2/zoo_behaviour.py [--run-id 2026-09-15_zoo]

Reads, per network, <tag>_subspace.json, <tag>_gain_accuracy_subspace.json (gate),
<tag>_unit_factor.json + <tag>_gain_accuracy.json (unit rule) and <tag>_probe_bias.json from
/scratch/byrong/mpk/generality/behaviour/<run-id>/, writes results/figures_v2/zoo_behaviour.json
and prints one row per network plus the ranking statistics. Networks whose files are missing
are listed and skipped, so the script can run on a partial array.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
from scipy.stats import kendalltau, spearmanr

# spec, label, family (conv | attention), group (imagenet | recipe | sin)
ZOO = [
    ("tv:alexnet", "AlexNet", "conv", "imagenet"),
    ("tv:vgg11_bn", "VGG-11-BN", "conv", "imagenet"),
    ("tv:vgg16", "VGG-16", "conv", "imagenet"),
    ("tv:vgg16_bn", "VGG-16-BN", "conv", "imagenet"),
    ("tv:vgg19_bn", "VGG-19-BN", "conv", "imagenet"),
    ("tv:googlenet", "GoogLeNet", "conv", "imagenet"),
    ("tv:inception_v3", "Inception-v3", "conv", "imagenet"),
    ("tv:resnet18", "ResNet-18", "conv", "imagenet"),
    ("tv:resnet34", "ResNet-34", "conv", "imagenet"),
    ("tv:resnet50", "ResNet-50", "conv", "imagenet"),
    ("tv:resnet101", "ResNet-101", "conv", "imagenet"),
    ("tv:resnet152", "ResNet-152", "conv", "imagenet"),
    ("tv:resnext50_32x4d", "ResNeXt-50 32x4d", "conv", "imagenet"),
    ("tv:resnext101_32x8d", "ResNeXt-101 32x8d", "conv", "imagenet"),
    ("tv:wide_resnet50_2", "Wide-ResNet-50-2", "conv", "imagenet"),
    ("tv:wide_resnet101_2", "Wide-ResNet-101-2", "conv", "imagenet"),
    ("tv:densenet121", "DenseNet-121", "conv", "imagenet"),
    ("tv:densenet201", "DenseNet-201", "conv", "imagenet"),
    ("tv:mobilenet_v2", "MobileNet-V2", "conv", "imagenet"),
    ("tv:mobilenet_v3_large", "MobileNet-V3-L", "conv", "imagenet"),
    ("tv:shufflenet_v2_x1_0", "ShuffleNet-V2 x1.0", "conv", "imagenet"),
    ("tv:mnasnet1_0", "MNASNet-1.0", "conv", "imagenet"),
    ("tv:efficientnet_b0", "EfficientNet-B0", "conv", "imagenet"),
    ("tv:efficientnet_b3", "EfficientNet-B3", "conv", "imagenet"),
    ("tv:efficientnet_v2_s", "EfficientNet-V2-S", "conv", "imagenet"),
    ("tv:regnet_y_8gf", "RegNet-Y-8GF", "conv", "imagenet"),
    ("tv:regnet_y_16gf", "RegNet-Y-16GF", "conv", "imagenet"),
    ("tv:convnext_tiny", "ConvNeXt-T", "conv", "imagenet"),
    ("tv:convnext_small", "ConvNeXt-S", "conv", "imagenet"),
    ("tv:convnext_base", "ConvNeXt-B", "conv", "imagenet"),
    ("tv:convnext_large", "ConvNeXt-L", "conv", "imagenet"),
    ("tv:vit_b_32", "ViT-B/32", "attention", "imagenet"),
    ("tv:vit_b_16", "ViT-B/16", "attention", "imagenet"),
    ("tv:vit_l_16", "ViT-L/16", "attention", "imagenet"),
    ("tv:swin_t", "Swin-T", "attention", "imagenet"),
    ("tv:swin_s", "Swin-S", "attention", "imagenet"),
    ("tv:swin_b", "Swin-B", "attention", "imagenet"),
    ("tv:maxvit_t", "MaxViT-T", "attention", "imagenet"),
    ("tv:resnet50:IMAGENET1K_V2", "ResNet-50 (V2 recipe)", "conv", "recipe"),
    ("hub:resnet50_sin", "ResNet-50 SIN", "conv", "sin"),
    ("hub:resnet50_sin_in", "ResNet-50 SIN+IN", "conv", "sin"),
    ("hub:resnet50_sin_in_ft", "ResNet-50 SIN+IN, ft IN", "conv", "sin"),
]
FIG6 = {"tv:resnet50", "tv:resnet152", "tv:vgg16", "tv:vit_b_16", "tv:convnext_small", "tv:convnext_base"}


def gated(curve, ws, gate):
    m = (ws >= min(gate) - 1e-9) & (ws <= max(gate) + 1e-9)
    c = np.asarray(curve)[m]
    return float(c.max() - c.min()), float(c[-1]), float(c[0])   # span, shape end (largest w), texture end (smallest w)


def load_network(root, spec):
    tag = spec.replace(":", "_")
    f_sub = os.path.join(root, f"{tag}_subspace.json")
    f_gate = os.path.join(root, f"{tag}_gain_accuracy_subspace.json")
    if not (os.path.exists(f_sub) and os.path.exists(f_gate)):
        return None
    z = json.load(open(f_sub))["summary"]
    g = json.load(open(f_gate))
    ws = np.array(z["ws"])
    span_g, shape_end, texture_end = gated(z["primary"]["mean_curve"], ws, g["w_within_5pts"])
    top1 = np.array(g["top1"])
    in_gate = np.array([w in g["w_within_5pts"] for w in z["ws"]])
    r = dict(tag=tag, n_units=z["n_units"], trained=z["trained"]["shape_bias"], trained_forced=z["trained"]["forced_two_way"],
             decided=z["trained"]["n_shape"] + z["trained"]["n_texture"], ws=z["ws"], mean_curve=z["primary"]["mean_curve"],
             span=z["primary"]["span_mean"], span_range=[z["primary"]["span_min"], z["primary"]["span_max"]],
             rho=z["primary"]["rho_mean"], rho_min=z["primary"]["rho_min"], rate_w0=z["primary"]["rate_w0"], rate_w1=z["primary"]["rate_w1"],
             var_frac_S=z["descriptives"]["var_frac_S"], var_frac_T=z["descriptives"]["var_frac_T"],
             trained_top1=g["trained_top1"], top1=g["top1"], gate=[min(g["w_within_5pts"]), max(g["w_within_5pts"])],
             whole_sweep_in_gate=bool(in_gate.all()), max_abs_dtop1_in_gate=float(np.abs(top1[in_gate] - g["trained_top1"]).max()),
             gated_span=span_g, shape_end=shape_end, texture_end=texture_end)
    f_unit, f_ugate = os.path.join(root, f"{tag}_unit_factor.json"), os.path.join(root, f"{tag}_gain_accuracy.json")
    if os.path.exists(f_unit) and os.path.exists(f_ugate):
        u = json.load(open(f_unit))["summary"]
        ug = json.load(open(f_ugate))
        if not u.get("undefined"):
            r["unit_span"] = u["primary"]["span_mean"]
            r["unit_gated_span"] = gated(u["primary"]["mean_curve"], np.array(u["ws"]), ug["w_within_5pts"])[0]
            r["unit_gate"] = [min(ug["w_within_5pts"]), max(ug["w_within_5pts"])]
    f_probe = os.path.join(root, f"{tag}_probe_bias.json")
    if os.path.exists(f_probe):
        pb = json.load(open(f_probe))
        r["probe"] = dict(shape_acc=pb["shape_acc"], texture_acc=pb["texture_acc"], bias=pb["probe_bias"], chance=pb["chance"])
    return r


def reversals(a, b):
    """Pairs whose order differs between rankings by a and by b (ties count as not reversed)."""
    n_rev, n_pairs = 0, 0
    for i, j in itertools.combinations(range(len(a)), 2):
        n_pairs += 1
        if (a[i] - a[j]) * (b[i] - b[j]) < 0:
            n_rev += 1
    return n_rev, n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="2026-09-15_zoo")
    ap.add_argument("--out", default="results/figures_v2/zoo_behaviour.json")
    args = ap.parse_args()
    root = f"/scratch/byrong/mpk/generality/behaviour/{args.run_id}"
    rows, missing = [], []
    for spec, label, family, group in ZOO:
        r = load_network(root, spec)
        if r is None:
            missing.append(spec)
            continue
        r.update(spec=spec, label=label, family=family, group=group, fig6=spec in FIG6)
        rows.append(r)
    print(f"{'network':26s} {'D':>5s} {'top1':>5s} {'trained':>7s} {'gate':>10s} {'tex-end':>7s} {'shp-end':>7s} {'gated':>6s} {'ρ':>5s} {'unit-g':>6s} {'probe':>6s} {'all-in':>6s}")
    for r in sorted(rows, key=lambda x: (x["group"], -x["trained"])):
        ug = "%.3f" % r["unit_gated_span"] if "unit_gated_span" in r else "—"
        pb = "%+.3f" % r["probe"]["bias"] if "probe" in r else "—"
        print(f"{r['label']:26s} {r['n_units']:5d} {r['trained_top1']:5.3f} {r['trained']:7.3f} {r['gate'][0]:4.2f}–{r['gate'][1]:4.2f} "
              f"{r['texture_end']:7.3f} {r['shape_end']:7.3f} {r['gated_span']:6.3f} {r['rho']:5.2f} {ug:>6s} {pb:>6s} {'yes' if r['whole_sweep_in_gate'] else 'no':>6s}")
    if missing:
        print(f"\nmissing ({len(missing)}): " + ", ".join(missing))

    im = [r for r in rows if r["group"] == "imagenet"]
    ev, stats = {}, {}
    if len(im) >= 3:
        tr = np.array([r["trained"] for r in im]); se = np.array([r["shape_end"] for r in im]); te = np.array([r["texture_end"] for r in im])
        gs = np.array([r["gated_span"] for r in im]); t1 = np.array([r["trained_top1"] for r in im])
        n_rev, n_pairs = reversals(tr, se)
        stats = dict(n=len(im), tau_trained_vs_shape_end=float(kendalltau(tr, se).statistic), tau_trained_vs_texture_end=float(kendalltau(tr, te).statistic),
                     reversals_trained_vs_shape_end=n_rev, n_pairs=n_pairs, frac_reversed=n_rev / n_pairs,
                     frac_gated_ge_0_20=float(np.mean(gs >= 0.20)), frac_whole_sweep_in_gate=float(np.mean([r["whole_sweep_in_gate"] for r in im])),
                     gated_span_min=float(gs.min()), gated_span_median=float(np.median(gs)), gated_span_max=float(gs.max()),
                     rho_span_vs_trained=float(spearmanr(gs, tr).statistic), rho_span_vs_top1=float(spearmanr(gs, t1).statistic),
                     mean_gated_span_conv=float(np.mean([r["gated_span"] for r in im if r["family"] == "conv"])),
                     mean_gated_span_attention=float(np.mean([r["gated_span"] for r in im if r["family"] == "attention"])) if any(r["family"] == "attention" for r in im) else None)
        pr = [r for r in im if "probe" in r]
        if len(pr) >= 3:
            stats["tau_trained_vs_probe_bias"] = float(kendalltau([r["trained"] for r in pr], [r["probe"]["bias"] for r in pr]).statistic)
            stats["rho_trained_vs_probe_bias"] = float(spearmanr([r["trained"] for r in pr], [r["probe"]["bias"] for r in pr]).statistic)
            stats["n_probe"] = len(pr)
        complete = len(im) == 38
        ev["P-Z1_ranking_not_invariant"] = dict(tau=stats["tau_trained_vs_shape_end"], frac_reversed=stats["frac_reversed"], n=len(im), complete=complete,
                                                met=bool(stats["tau_trained_vs_shape_end"] < 0.7 and stats["frac_reversed"] >= 0.25))
        ev["P-Z2_magnitude_across_zoo"] = dict(frac_gated_ge_0_20=stats["frac_gated_ge_0_20"], frac_whole_sweep_in_gate=stats["frac_whole_sweep_in_gate"], n=len(im), complete=complete,
                                              met=bool(stats["frac_gated_ge_0_20"] >= 0.75 and stats["frac_whole_sweep_in_gate"] >= 0.5))
        if "tau_trained_vs_probe_bias" in stats:
            ev["P-Z4_rate_vs_probe_dissociate"] = dict(tau=stats["tau_trained_vs_probe_bias"], n=stats["n_probe"], complete=stats["n_probe"] == 38,
                                                      met=bool(stats["tau_trained_vs_probe_bias"] < 0.7))
    sin = next((r for r in rows if r["spec"] == "hub:resnet50_sin"), None)
    if sin is not None and im:
        ev["P-Z3_SIN_control"] = dict(
            a_trained_is_max=dict(sin=sin["trained"], max_other=float(max(r["trained"] for r in rows if r["spec"] != "hub:resnet50_sin")),
                                  met=bool(sin["trained"] >= max(r["trained"] for r in rows if r["spec"] != "hub:resnet50_sin"))),
            b_gated_span_ge_0_20_same_sign=dict(gated_span=sin["gated_span"], rho=sin["rho"], met=bool(sin["gated_span"] >= 0.20 and sin["rho"] > 0)),
            c_probe_bias_exceeds_all_imagenet=(dict(sin=sin["probe"]["bias"], max_imagenet=float(max(r["probe"]["bias"] for r in im if "probe" in r)),
                                                    met=bool(sin["probe"]["bias"] > max(r["probe"]["bias"] for r in im if "probe" in r)))
                                               if "probe" in sin and any("probe" in r for r in im) else None),
            complete=len(im) == 38)
    out = dict(run_id=args.run_id, plan="docs/PLAN_ZOO_2026-09-15.md", networks=rows, missing=missing, ranking=stats, evaluation=ev)
    print("\nRanking statistics over ImageNet-trained networks:")
    for k, v in stats.items():
        print(f"  {k:36s} {v if not isinstance(v, float) else round(v, 3)}")
    print("\nPre-registered predictions:")
    for k, v in ev.items():
        print(f"  {k:34s} " + json.dumps(v, default=lambda o: round(float(o), 3))[:400])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
