"""Aggregate the generality behaviour results (unit factorisation + gain sweep + refit).

  python tmp_analysis/figs_v2/generality_behaviour.py [--run-id 2026-09-14_gen]

Reads /scratch/byrong/mpk/generality/behaviour/<run-id>/*_unit_factor.json,
*_refit_sum.json, *_gain_accuracy.json and the split-rule sensitivity files
*_unit_factor_{positive,decile}.json (+ *_gain_accuracy_{positive,decile}.json),
prints one row per network and writes results/figures_v2/generality_behaviour.json,
the single committed source of Fig. 6a-b and Extended Data Table 13.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

FAMILIES = [
    ("rgb_tsn", "single ResNet-18, RGB, no diet (rgb_tsn)", ["run_rgb_tsn_seed0", "run_rgb_tsn_seed1", "run_rgb_tsn_seed2"]),
    ("single18", "single ResNet-18, diet (dkl_bcs_single)", ["run_dkl_bcs_single_seed0", "run_dkl_bcs_single_seed1", "run_dkl_bcs_single_seed2"]),
    ("single34", "single ResNet-34, diet (dkl_bcs_single34)", ["run_dkl_bcs_single34_seed0", "run_dkl_bcs_single34_seed1", "run_dkl_bcs_single34_seed2"]),
    ("concat", "two-stream diet, streams concatenated (validation)", ["concat_dkl_bcs_seed0", "concat_dkl_bcs_seed1", "concat_dkl_bcs_seed2"]),
    ("resnet50", "ImageNet ResNet-50", ["tv_resnet50"]),
    ("resnet152", "ImageNet ResNet-152", ["tv_resnet152"]),
    ("vgg16", "ImageNet VGG-16", ["tv_vgg16"]),
    ("vit_b_16", "ImageNet ViT-B/16", ["tv_vit_b_16"]),
    ("convnext_small", "ImageNet ConvNeXt-S", ["tv_convnext_small"]),
    ("convnext_base", "ImageNet ConvNeXt-B", ["tv_convnext_base"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="2026-09-14_gen")
    ap.add_argument("--out", default="results/figures_v2/generality_behaviour.json")
    args = ap.parse_args()
    root = f"/scratch/byrong/mpk/generality/behaviour/{args.run_id}"
    out = {"run_id": args.run_id, "families": []}
    print(f"{'network':52s} {'trained':>8s} {'w=0':>6s} {'w=1':>6s} {'span':>14s} {'rho':>5s} {'forced span':>11s} {'S-lean':>6s} {'S from s1':>9s}")
    for key, label, tags in FAMILIES:
        rows = []
        for tag in tags:
            f = os.path.join(root, f"{tag}_unit_factor.json")
            if not os.path.exists(f):
                continue
            z = json.load(open(f))["summary"]
            r = dict(tag=tag, trained=z["trained"]["shape_bias"], trained_forced=z["trained"]["forced_two_way"],
                     decided=z["trained"]["n_shape"] + z["trained"]["n_texture"], n_units=z["n_units"],
                     ws=z["ws"], mean_curve=z["primary"]["mean_curve"], min_curve=z["primary"]["min_curve"],
                     max_curve=z["primary"]["max_curve"], forced_curve=z["forced"]["mean_curve"],
                     span=z["primary"]["span_mean"], span_min=z["primary"]["span_min"], span_max=z["primary"]["span_max"],
                     rho=z["primary"]["rho_mean"], rho_min=z["primary"]["rho_min"],
                     forced_span=z["forced"]["span_mean"], rate_w0=z["primary"]["rate_w0"], rate_w1=z["primary"]["rate_w1"],
                     rate_w05=z["primary"]["rate_w05"], frac_shape_leaning=z["descriptives"]["frac_units_shape_leaning"],
                     frac_S_from_stream1=z["descriptives"].get("frac_S_from_stream1_heldout_mean"),
                     frac_S_from_stream1_min=z["descriptives"].get("frac_S_from_stream1_heldout_min"),
                     decided_curve=z["decided_mean_curve"])
            fr = os.path.join(root, f"{tag}_refit_sum.json")
            if os.path.exists(fr):
                rr = json.load(open(fr))
                ref = {}
                for head in ("linear", "mlp"):
                    sub = [x for x in rr if x["head"] == head and x["w"] is not None]
                    if not sub:
                        continue
                    ws = sorted({x["w"] for x in sub})
                    curve = [np.mean([x["shape_bias"] for x in sub if x["w"] == w]) for w in ws]
                    acc = [np.mean([x["val_top1"] for x in sub if x["w"] == w]) for w in ws]
                    full = [x for x in rr if x["head"] == head and x["w"] is None]
                    best = max(acc)
                    ok = [c for c, a in zip(curve, acc) if a >= best - 0.05]
                    ref[head] = dict(ws=ws, curve=[float(c) for c in curve], val=[float(a) for a in acc],
                                     span=float(max(curve) - min(curve)), span_acc5=float(max(ok) - min(ok)) if ok else None,
                                     rho=float(np.corrcoef(ws, curve)[0, 1]),
                                     full_val=float(full[0]["val_top1"]) if full else None,
                                     full_sb=float(full[0]["shape_bias"]) if full else None)
                r["refit"] = ref
            fa = os.path.join(root, f"{tag}_gain_accuracy.json")
            if os.path.exists(fa):
                za = json.load(open(fa))
                r["gain_accuracy"] = dict(set=za["set"], n=za["n"], trained_top1=za["trained_top1"], top1=za["top1"],
                                          w_within_5pts=za["w_within_5pts"])
            rules = {}
            for rule in ("positive", "decile"):
                fu = os.path.join(root, f"{tag}_unit_factor_{rule}.json")
                if not os.path.exists(fu):
                    continue
                zu = json.load(open(fu))["summary"]
                if zu.get("undefined"):
                    rules[rule] = dict(undefined=True, frac_shape_leaning=zu["frac_units_shape_leaning"])
                    continue
                entry = dict(span=zu["primary"]["span_mean"], span_min=zu["primary"]["span_min"], span_max=zu["primary"]["span_max"],
                             rho=zu["primary"]["rho_mean"], rho_min=zu["primary"]["rho_min"], n_S=zu.get("n_S_full"),
                             n_heldout_curves=zu.get("n_heldout_curves"), rate_w0=zu["primary"]["rate_w0"],
                             rate_w1=zu["primary"]["rate_w1"], mean_curve=zu["primary"]["mean_curve"],
                             forced_span=zu["forced"]["span_mean"])
                fga = os.path.join(root, f"{tag}_gain_accuracy_{rule}.json")
                if os.path.exists(fga):
                    zg = json.load(open(fga))
                    entry["w_within_5pts"] = zg["w_within_5pts"]; entry["trained_top1"] = zg["trained_top1"]; entry["top1"] = zg["top1"]
                rules[rule] = entry
            if rules:
                r["rules"] = rules
            rows.append(r)
        if not rows:
            continue
        agg = dict(key=key, label=label, seeds=rows,
                   trained=float(np.mean([r["trained"] for r in rows])),
                   span=float(np.mean([r["span"] for r in rows])),
                   span_range=[float(min(r["span_min"] for r in rows)), float(max(r["span_max"] for r in rows))],
                   rho=float(np.mean([r["rho"] for r in rows])), rho_min=float(min(r["rho_min"] for r in rows)),
                   forced_span=float(np.mean([r["forced_span"] for r in rows])),
                   rate_w0=float(np.mean([r["rate_w0"] for r in rows])), rate_w1=float(np.mean([r["rate_w1"] for r in rows])),
                   frac_shape_leaning=float(np.mean([r["frac_shape_leaning"] for r in rows])))
        rules_agg = {}
        for rule in ("positive", "decile"):
            present = [r["rules"][rule] for r in rows if r.get("rules", {}).get(rule)]
            defined = [d for d in present if not d.get("undefined")]
            if not present:
                continue
            if not defined:
                rules_agg[rule] = dict(undefined=True, n_seeds=len(present))
                continue
            e = dict(span=float(np.mean([d["span"] for d in defined])), span_min=float(min(d["span_min"] for d in defined)),
                     span_max=float(max(d["span_max"] for d in defined)), rho=float(np.mean([d["rho"] for d in defined])),
                     rho_min=float(min(d["rho_min"] for d in defined)), n_defined=len(defined), n_seeds=len(present),
                     rate_w0=float(np.mean([d["rate_w0"] for d in defined])), rate_w1=float(np.mean([d["rate_w1"] for d in defined])))
            ns = [d["n_S"] for d in defined if d.get("n_S") is not None]
            if ns:
                e["n_S"] = float(np.mean(ns))
            w5 = [d["w_within_5pts"] for d in defined if d.get("w_within_5pts")]
            if w5:
                e["w5_lo"] = float(np.mean([min(x) for x in w5])); e["w5_hi"] = float(np.mean([max(x) for x in w5]))
            rules_agg[rule] = e
        if rules_agg:
            agg["rules"] = rules_agg
        s1 = [r["frac_S_from_stream1"] for r in rows if r.get("frac_S_from_stream1") is not None]
        if s1:
            agg["frac_S_from_stream1"] = float(np.mean(s1))
            agg["frac_S_from_stream1_min"] = float(min(r["frac_S_from_stream1_min"] for r in rows))
        out["families"].append(agg)
        print(f"{label:52s} {agg['trained']:8.3f} {agg['rate_w0']:6.3f} {agg['rate_w1']:6.3f} "
              f"{agg['span']:6.3f} [{agg['span_range'][0]:.2f},{agg['span_range'][1]:.2f}] {agg['rho']:5.2f} "
              f"{agg['forced_span']:11.3f} {agg['frac_shape_leaning']:6.2f} "
              f"{(agg.get('frac_S_from_stream1') or float('nan')):9.2f}  n={len(rows)}")
        for rule, e in agg.get("rules", {}).items():
            print(f"    rule {rule:8s}: " + ("undefined" if e.get("undefined") else
                  f"span {e['span']:.3f} [{e['span_min']:.2f},{e['span_max']:.2f}] rho_min {e['rho_min']:.2f} |S| {e.get('n_S', float('nan')):.0f}"
                  + (f" gate {e['w5_lo']:.2f}-{e['w5_hi']:.2f}" if "w5_lo" in e else "")))
        for r in rows:
            if "refit" in r:
                for h, v in r["refit"].items():
                    print(f"    refit {r['tag']} {h:6s}: span {v['span']:.3f} (acc-5 {v['span_acc5']}), rho {v['rho']:.2f}, "
                          f"full-y val {v['full_val']:.3f} sb {v['full_sb']:.3f}; val range {min(v['val']):.3f}-{max(v['val']):.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
