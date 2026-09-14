"""Aggregate the generality behaviour results (unit factorisation + gain sweep + refit).

  python tmp_analysis/figs_v2/generality_behaviour.py [--run-id 2026-09-14_gen]

Reads /scratch/byrong/mpk/generality/behaviour/<run-id>/*_unit_factor.json and
*_refit_sum.json, prints one row per network and writes
results/figures_v2/generality_behaviour.json for the figure and the ED table.
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
        s1 = [r["frac_S_from_stream1"] for r in rows if r.get("frac_S_from_stream1") is not None]
        if s1:
            agg["frac_S_from_stream1"] = float(np.mean(s1))
            agg["frac_S_from_stream1_min"] = float(min(r["frac_S_from_stream1_min"] for r in rows))
        out["families"].append(agg)
        print(f"{label:52s} {agg['trained']:8.3f} {agg['rate_w0']:6.3f} {agg['rate_w1']:6.3f} "
              f"{agg['span']:6.3f} [{agg['span_range'][0]:.2f},{agg['span_range'][1]:.2f}] {agg['rho']:5.2f} "
              f"{agg['forced_span']:11.3f} {agg['frac_shape_leaning']:6.2f} "
              f"{(agg.get('frac_S_from_stream1') or float('nan')):9.2f}  n={len(rows)}")
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
