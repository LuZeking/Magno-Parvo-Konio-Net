"""PLAN_SPAN outcomes: what sets the span — location and leverage of one evidence distribution.

  PYTHONPATH=. python tmp_analysis/figs_v2/span_model.py [--run-id 2026-09-15_zoo]

Reads the per-network files written by `src/analysis/generality/span_model.py` and the committed
`zoo_behaviour.json` (reported gated span, probe, labels, groups), evaluates predictions S-1 to
S-7 of `docs/PLAN_SPAN_2026-09-18.md`, reports the zoo on the forced estimator (review m5), and
writes `results/figures_v2/span_model.json` plus a markdown table.

The zoo is a fixed set of networks, so across-network statistics are descriptive: Spearman rho
with a percentile bootstrap over networks, no p-values.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from kappa_rate import kendall, spearman  # noqa: E402

BEH = "/scratch/byrong/mpk/generality/behaviour"
OUT_JSON = "results/figures_v2/span_model.json"
OUT_MD = "results/figures_v2/span_model.md"
SIN = "hub_resnet50_sin"


def rho_ci(a, b, n=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        r = spearman(a[i], b[i])
        if np.isfinite(r):
            draws.append(r)
    return dict(rho=spearman(a, b), ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="2026-09-15_zoo")
    args = p.parse_args()
    zoo = json.load(open("results/figures_v2/zoo_behaviour.json"))
    nets = []
    for n in zoo["networks"]:
        f = os.path.join(BEH, args.run_id, f"{n['tag']}_span_model.json")
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
        m = json.load(open(f))
        rs = np.array([r["span"] for r in m["random"]])
        nets.append(dict(
            tag=n["tag"], label=n["label"], group=n["group"],
            T_p=n["gated_span"], T_f=m["T_f"], m0_span=m["m0_span"], m1_span=m["m1_span"],
            m0_max_abs_err=m["m0_max_abs_err"], m1_max_abs_err=m["m1_max_abs_err"],
            forced_trained=m["forced_trained"], rate_trained=n["trained"], Phi_z=m["Phi_z"], z=m["z"], lam=m["lam"],
            mean_eS=m["mean_eS"], mean_eT=m["mean_eT"], texture_term=-m["mean_eT"], mean_g=m["mean_g"], sd=m["sd"],
            corr_delta_g=m["corr_delta_g"],
            H_cc=m["H_cc"]["H"], H_cc_S=m["H_cc"]["S"], H_cc_T=m["H_cc"]["T"], H_val=m["H_val"]["H"],
            var_frac=n["var_frac_S"] + n["var_frac_T"],
            rand_mean_signed=float(rs.mean()), rand_max_abs=float(np.abs(rs).max()),
            rand_median_abs=float(np.median(np.abs(rs))),
            rand_gate_min_width=float(min(r["gate"][1] - r["gate"][0] for r in m["random"])),
            rand_forced_max_abs=float(max(abs(r["forced_span"]) for r in m["random"])),
            span_insample=float(m["rate_curve"][int(round(m["gate"][1] * 20))] - m["rate_curve"][int(round(m["gate"][0] * 20))]),
            forced_shape_end=m["forced_curve"][int(round(m["gate"][1] * 20))],
            rate_shape_end_insample=m["rate_curve"][int(round(m["gate"][1] * 20))],
            rate_trained_insample=m["rate_curve"][10],
            probe=n.get("probe"), top1=n["trained_top1"], D=n["n_units"],
            exact_end_margins=m["exact_end_margins"], top1_check=[m["top1_trained_val"], m["top1_trained_stored"]],
        ))
    im = [n for n in nets if n["group"] == "imagenet"]
    col = lambda k, s=im: [n[k] for n in s]  # noqa: E731
    sin = next(n for n in nets if n["tag"] == SIN)

    P = {}
    med_err = float(np.median(col("m0_max_abs_err")))
    r = spearman(col("m0_span"), col("T_f"))
    P["S1"] = dict(median_max_abs_err=med_err, rho_m0_Tf=r, met=bool(med_err <= 0.05 and r >= 0.9))
    a, b_, c = spearman(col("m1_span"), col("T_f")), spearman(col("m1_span"), col("T_p")), spearman(col("Phi_z"), col("forced_trained"))
    P["S2"] = dict(rho_m1_Tf=a, rho_m1_Tp=b_, rho_Phiz_forced=c, met=bool(a >= 0.8 and b_ >= 0.7 and c >= 0.9))
    a, b_ = rho_ci(col("lam"), col("T_p")), rho_ci(col("z"), col("T_p"))
    P["S3"] = dict(lam_Tp=a, z_Tp=b_, met=bool(a["rho"] >= 0.7 and abs(b_["rho"]) <= 0.3))
    a = rho_ci(col("lam"), col("z"))
    P["S4"] = dict(lam_z=a, met=bool(abs(a["rho"]) <= 0.3))
    lam_min, tex_min = min(col("lam")), min(col("texture_term"))
    P["S5a"] = dict(sin_lam=sin["lam"], imagenet_lam_min=lam_min, imagenet_lam_range=[lam_min, max(col("lam"))],
                    met=bool(sin["lam"] < lam_min))
    P["S5b"] = dict(sin_texture_term=sin["texture_term"], imagenet_texture_term_min=tex_min,
                    imagenet_texture_term_range=[tex_min, max(col("texture_term"))], met=bool(sin["texture_term"] < tex_min))
    a, b_, c = rho_ci(col("H_cc"), col("lam")), rho_ci(col("H_cc"), col("T_p")), rho_ci(col("var_frac"), col("T_p"))
    P["S6"] = dict(H_lam=a, H_Tp=b_, varfrac_Tp=c, gain_over_varfrac=b_["rho"] - c["rho"],
                   met=bool(a["rho"] >= 0.5 and b_["rho"] >= 0.5 and b_["rho"] - c["rho"] >= 0.10))
    med_signed = float(np.median(np.abs(col("rand_mean_signed"))))
    n_exceed = int(sum(n["T_p"] > n["rand_max_abs"] for n in im))
    n_exceed_ins = int(sum(n["span_insample"] > n["rand_max_abs"] for n in im))
    P["S7"] = dict(median_abs_mean_signed=med_signed, n_Tp_exceeds_max_random=n_exceed,
                   n_insample_exceeds_max_random=n_exceed_ins, n=len(im),
                   random_max_abs_range=[min(col("rand_max_abs")), max(col("rand_max_abs"))],
                   random_median_abs_median=float(np.median(col("rand_median_abs"))),
                   all42_n_exceeds=int(sum(n["T_p"] > n["rand_max_abs"] for n in nets)),
                   met=bool(med_signed <= 0.03 and n_exceed >= 36))

    # review m5: the zoo on the forced estimator, and the in-sample primary estimator
    tf, rf, df, npair = kendall(col("forced_trained"), col("forced_shape_end"))
    ti, ri, di, _ = kendall(col("rate_trained_insample"), col("rate_shape_end_insample"))
    forced = dict(forced_trained_range=[min(col("forced_trained")), max(col("forced_trained"))],
                  rho_forced_vs_primary_trained=spearman(col("forced_trained"), col("rate_trained")),
                  tau_forced=tf, reversals_forced=df, frac_forced=rf,
                  tau_insample_primary=ti, reversals_insample_primary=di, frac_insample_primary=ri, n_pairs=npair,
                  tau_heldout_primary=zoo["ranking"]["tau_trained_vs_shape_end"],
                  reversals_heldout_primary=zoo["ranking"]["reversals_trained_vs_shape_end"])

    rep = {k: rho_ci(col("lam"), [n["probe"][k2] for n in im]) for k, k2 in
           (("probe_shape", "shape_acc"), ("probe_texture", "texture_acc"), ("probe_bias", "bias"))}
    rep["top1"] = rho_ci(col("lam"), col("top1"))
    rep["D"] = rho_ci(col("lam"), col("D"))
    rep["H_val_lam"] = rho_ci(col("H_val"), col("lam"))
    rep["H_val_Tp"] = rho_ci(col("H_val"), col("T_p"))
    rep["Tp_Tf"] = spearman(col("T_p"), col("T_f"))
    rep["span_rate"] = spearman(col("T_p"), col("rate_trained"))
    rep["lam_mean_g"] = spearman(col("lam"), col("mean_g"))
    rep["lam_sd"] = spearman(col("lam"), [1 / n["sd"] for n in im])
    rep["m1_max_abs_err_median"] = float(np.median(col("m1_max_abs_err")))

    failed = [k for k, v in P.items() if not v["met"]]
    out = dict(plan="docs/PLAN_SPAN_2026-09-18.md", run_id=args.run_id, n_imagenet=len(im),
               predictions=P, failed=failed, forced_estimator=forced, reported=rep, networks=nets)
    json.dump(out, open(OUT_JSON, "w"), indent=1)

    md = ["| network | trained forced rate | Φ(z) | z | λ | shape term | texture term | H | T_f | M0 | M1 | T_p | random span: max |.| |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in sorted(nets, key=lambda n: (n["group"] != "imagenet", -n["lam"])):
        md.append(f"| {n['label']}{'' if n['group'] == 'imagenet' else ' *'} | {n['forced_trained']:.3f} | {n['Phi_z']:.3f} | "
                  f"{n['z']:+.2f} | {n['lam']:.3f} | {n['mean_eS']:.2f} | {n['texture_term']:.2f} | {n['H_cc']:.3f} | "
                  f"{n['T_f']:.3f} | {n['m0_span']:.3f} | {n['m1_span']:.3f} | {n['T_p']:.3f} | {n['rand_max_abs']:.3f} |")
    open(OUT_MD, "w").write("\n".join(md) + "\n")
    for k, v in P.items():
        print(k, "MET" if v["met"] else "FAILED", {kk: vv for kk, vv in v.items() if kk != "met"})
    print("forced estimator (m5):", forced)
    print("reported:", rep)
    print("failed:", failed)


if __name__ == "__main__":
    main()
