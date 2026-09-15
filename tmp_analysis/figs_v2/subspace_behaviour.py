"""Aggregate the subspace readout results (PLAN_SUBSPACE D1-D3) and evaluate P-A, P-A', P-B, P-D.

  PYTHONPATH=. python tmp_analysis/figs_v2/subspace_behaviour.py [--run-id 2026-09-14_gen]

Reads /scratch/byrong/mpk/generality/behaviour/<run-id>/<tag>_subspace.json,
<tag>_gain_accuracy_subspace.json and <tag>_refit_sum_subspace.json, and the committed
results/figures_v2/generality_behaviour.json (unit-split baseline, for P-D). Writes
results/figures_v2/subspace_behaviour.json and prints one row per network with the
pre-registered thresholds applied.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from tmp_analysis.figs_v2.generality_behaviour import FAMILIES  # noqa: E402  (same family definitions)

IMAGENET = ["resnet50", "resnet152", "vgg16", "vit_b_16", "convnext_small", "convnext_base"]
OURS = ["rgb_tsn", "single18", "single34"]


def gated_span(curve, ws, gate):
    """max - min of a curve over the w within the accuracy gate (list of w values)."""
    if not gate:
        return None
    m = (ws >= min(gate) - 1e-9) & (ws <= max(gate) + 1e-9)
    c = np.asarray(curve)[m]
    return float(c.max() - c.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="2026-09-14_gen")
    ap.add_argument("--out", default="results/figures_v2/subspace_behaviour.json")
    args = ap.parse_args()
    root = f"/scratch/byrong/mpk/generality/behaviour/{args.run_id}"
    base = {f["key"]: f for f in json.load(open("results/figures_v2/generality_behaviour.json"))["families"]}
    out = {"run_id": args.run_id, "plan": "docs/PLAN_SUBSPACE_2026-09-15.md", "families": []}
    print(f"{'network':46s} {'trained':>7s} {'w=0':>6s} {'w=1':>6s} {'span':>6s} {'rho':>5s} {'gate':>10s} {'gated':>6s} {'unit-gated':>10s} {'S->s1':>6s}")
    for key, label, tags in FAMILIES:
        rows = []
        for tag in tags:
            f = os.path.join(root, f"{tag}_subspace.json")
            if not os.path.exists(f):
                continue
            z = json.load(open(f))["summary"]
            ws = np.array(z["ws"])
            r = dict(tag=tag, trained=z["trained"]["shape_bias"], n_units=z["n_units"], ws=z["ws"],
                     mean_curve=z["primary"]["mean_curve"], min_curve=z["primary"]["min_curve"], max_curve=z["primary"]["max_curve"],
                     forced_curve=z["forced"]["mean_curve"], span=z["primary"]["span_mean"], span_min=z["primary"]["span_min"],
                     span_max=z["primary"]["span_max"], rho=z["primary"]["rho_mean"], rho_min=z["primary"]["rho_min"],
                     forced_span=z["forced"]["span_mean"], rate_w0=z["primary"]["rate_w0"], rate_w1=z["primary"]["rate_w1"],
                     descriptives=z["descriptives"],
                     sensitivity={m: dict(span=z["sensitivity"][m]["primary"]["span_mean"], rho=z["sensitivity"][m]["primary"]["rho_mean"])
                                  for m in z["sensitivity"]})
            fa = os.path.join(root, f"{tag}_gain_accuracy_subspace.json")
            if os.path.exists(fa):
                za = json.load(open(fa))
                r["gain_accuracy"] = dict(set=za["set"], n=za["n"], trained_top1=za["trained_top1"], top1=za["top1"], w_within_5pts=za["w_within_5pts"])
                r["gated_span"] = gated_span(r["mean_curve"], ws, za["w_within_5pts"])
                r["gated_forced_span"] = gated_span(r["forced_curve"], ws, za["w_within_5pts"])
            fr = os.path.join(root, f"{tag}_refit_sum_subspace.json")
            if os.path.exists(fr):
                rr = json.load(open(fr))
                ref = {}
                for head in ("linear", "mlp"):
                    sub = [x for x in rr if x["head"] == head and x["w"] is not None]
                    if not sub:
                        continue
                    wsr = sorted({x["w"] for x in sub})
                    curve = [float(np.mean([x["shape_bias"] for x in sub if x["w"] == w])) for w in wsr]
                    acc = [float(np.mean([x["val_top1"] for x in sub if x["w"] == w])) for w in wsr]
                    full = [x for x in rr if x["head"] == head and x["w"] is None]
                    ceiling = float(full[0]["val_top1"]) if full else None
                    ref[head] = dict(ws=wsr, curve=curve, val=acc, span=float(max(curve) - min(curve)),
                                     rho=float(np.corrcoef(wsr, curve)[0, 1]), full_val=ceiling,
                                     full_sb=float(full[0]["shape_bias"]) if full else None,
                                     min_val_minus_ceiling=float(min(acc) - ceiling) if ceiling is not None else None)
                r["refit"] = ref
            rows.append(r)
        if not rows:
            continue
        agg = dict(key=key, label=label, seeds=rows,
                   trained=float(np.mean([r["trained"] for r in rows])), span=float(np.mean([r["span"] for r in rows])),
                   span_range=[float(min(r["span_min"] for r in rows)), float(max(r["span_max"] for r in rows))],
                   rho=float(np.mean([r["rho"] for r in rows])), rho_min=float(min(r["rho_min"] for r in rows)),
                   forced_span=float(np.mean([r["forced_span"] for r in rows])),
                   rate_w0=float(np.mean([r["rate_w0"] for r in rows])), rate_w1=float(np.mean([r["rate_w1"] for r in rows])))
        g = [r["gated_span"] for r in rows if r.get("gated_span") is not None]
        if g:
            agg["gated_span"] = float(np.mean(g))
            agg["gate"] = [float(np.mean([min(r["gain_accuracy"]["w_within_5pts"]) for r in rows if "gain_accuracy" in r])),
                           float(np.mean([max(r["gain_accuracy"]["w_within_5pts"]) for r in rows if "gain_accuracy" in r]))]
            # unit-split gated span on the same gate definition, from the committed baseline
            if key in base:
                ug = []
                for rb in base[key]["seeds"]:
                    if rb.get("gain_accuracy"):
                        ug.append(gated_span(rb["mean_curve"], np.array(rb["ws"]), rb["gain_accuracy"]["w_within_5pts"]))
                if ug:
                    agg["unit_gated_span"] = float(np.mean(ug))
        e1 = [r["descriptives"].get("energy_S_on_stream1_full") for r in rows if r["descriptives"].get("energy_S_on_stream1_full") is not None]
        if e1:
            agg["energy_S_on_stream1"] = float(np.mean(e1)); agg["energy_S_on_stream1_min"] = float(min(e1))
            agg["energy_T_on_stream2"] = float(np.mean([r["descriptives"]["energy_T_on_stream2_full"] for r in rows]))
        if any("refit" in r for r in rows):
            agg["refit"] = {h: dict(span=float(np.mean([r["refit"][h]["span"] for r in rows if h in r.get("refit", {})])),
                                    rho=float(np.mean([r["refit"][h]["rho"] for r in rows if h in r.get("refit", {})])),
                                    min_val_minus_ceiling=float(np.mean([r["refit"][h]["min_val_minus_ceiling"] for r in rows if h in r.get("refit", {})])))
                            for h in ("linear", "mlp") if any(h in r.get("refit", {}) for r in rows)}
        out["families"].append(agg)
        gate_s = "%.2f-%.2f" % tuple(agg["gate"]) if "gate" in agg else "—"
        gated_s = "%.3f" % agg["gated_span"] if "gated_span" in agg else "—"
        ugated_s = "%.3f" % agg["unit_gated_span"] if "unit_gated_span" in agg else "—"
        e1_s = "%.2f" % agg["energy_S_on_stream1"] if "energy_S_on_stream1" in agg else "—"
        print(f"{label:46s} {agg['trained']:7.3f} {agg['rate_w0']:6.3f} {agg['rate_w1']:6.3f} {agg['span']:6.3f} {agg['rho']:5.2f} "
              f"{gate_s:>10s} {gated_s:>6s} {ugated_s:>10s} {e1_s:>6s}  n={len(rows)}")

    fam = {f["key"]: f for f in out["families"]}
    ev = {}
    im_g = {k: fam[k].get("gated_span") for k in IMAGENET if k in fam}
    ours_g = {k: fam[k].get("gated_span") for k in OURS if k in fam}
    if all(v is not None for v in im_g.values()) and len(im_g) == 6:
        ev["P-A_imagenet_gated_ge_0.20"] = dict(n=int(sum(v >= 0.20 for v in im_g.values())), values=im_g, met=bool(sum(v >= 0.20 for v in im_g.values()) >= 4))
    if all(v is not None for v in ours_g.values()) and len(ours_g) == 3:
        ev["P-A_ours_gated_ge_0.20"] = dict(values=ours_g, met=bool(all(v >= 0.20 for v in ours_g.values())))
    if fam:
        rho_single = {k: fam[k]["rho"] for k in IMAGENET + OURS if k in fam}
        ev["P-A_rho_ge_0.90_all"] = dict(values=rho_single, met=bool(all(v >= 0.90 for v in rho_single.values())))
    if "concat" in fam and "energy_S_on_stream1_min" in fam["concat"]:
        c = fam["concat"]
        ev["P-B"] = dict(energy_S_on_stream1_min=c["energy_S_on_stream1_min"], energy_T_on_stream2=c["energy_T_on_stream2"], span=c["span"],
                         met=bool(c["energy_S_on_stream1_min"] >= 0.90 and c["energy_T_on_stream2"] >= 0.90 and c["span"] >= 0.40))
    pd = {k: (fam[k].get("gated_span"), fam[k].get("unit_gated_span")) for k in IMAGENET + OURS if k in fam}
    if pd and all(a is not None and b is not None for a, b in pd.values()):
        ev["P-D_subspace_gt_unit_gated"] = dict(values=pd, met=bool(all(a > b for a, b in pd.values())))
    rf = {k: fam[k]["refit"] for k in OURS if k in fam and "refit" in fam[k]}
    if len(rf) == 3 and all("mlp" in v and "linear" in v for v in rf.values()):
        ev["P-A'_refit"] = dict(values=rf, met=bool(all(v["mlp"]["span"] >= 0.15 and v["mlp"]["rho"] >= 0.90 and abs(v["mlp"]["span"] - v["linear"]["span"]) <= 0.05
                                                      and v["mlp"]["min_val_minus_ceiling"] >= -0.05 for v in rf.values())))
    out["evaluation"] = ev
    print("\nPre-registered predictions:")
    for k, v in ev.items():
        print(f"  {k:32s} met={v['met']}  " + json.dumps({kk: vv for kk, vv in v.items() if kk != 'met'}, default=lambda o: round(float(o), 3))[:300])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
