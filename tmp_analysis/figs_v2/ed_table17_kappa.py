"""Supplementary Table S13: error consistency at a matched cue-conflict rate (PLAN_KAPPA).

  PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/ed_table17_kappa.py

Reads results/figures_v2/kappa_rate.json (tmp_analysis/figs_v2/kappa_rate.py) and upserts the
table into results/figures_v2/ed_tables.md; insert_ed_tables.py then places it in the paper.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ed_tables_generality import upsert  # noqa: E402


def n(x, d=1):
    return f"{x:.{d}f}".replace("-", "−")


def table17(K):
    P, B = K["post_hoc"], K["post_hoc"]["bound"]
    pr = K["predictions"]
    rows = ["| network | w at the matched rate | rate | κ, % of human–human ceiling | κ at the trained readout | κ / κ_max, % of the human-pair value | probe bias | trained rate |",
            "|---|---|---|---|---|---|---|---|"]
    for r in K["matched"]:
        norm = r.get("kappa_norm_pct")
        star = " \\*" if r["group"] != "imagenet" else ""
        rows.append(f"| {r['label']}{star} | {r['w']:.2f} | {r['rate']:.3f} | {n(r['kappa_pct'])} | "
                    f"{n(r['kappa_trained_pct'])} | {'—' if norm is None else n(norm)} | "
                    f"{n(r['probe_bias'], 2)} | {r['trained_rate']:.3f} |")
    cap = ("**Supplementary Table S13 | Error consistency with the ten observers at a matched cue-conflict rate.** "
           "Every frozen network of Extended Data Table 4 under the same gain (Methods 4.8), swept over "
           "w ∈ {0, 0.05, …, 1}; for each network the row is the **gated** readout whose rate is closest to "
           f"{K['target_rate']:.2f} (tolerance ± {K['tol']:.2f}; networks whose gate does not reach it are absent, "
           f"{len(pr['K2']['excluded'])} of 38: {', '.join(pr['K2']['excluded'])}). κ is Cohen's κ on trial-by-trial "
           "agreement over the images both parties decided, mean over the ten observers, as a percentage of the "
           "human–human ceiling κ = 0.234 (Methods 4.3; Supplementary Table S6). “κ at the trained readout” is the "
           "same quantity at w = 0.5, the number Extended Data Table 4 reports and the form in which the field "
           "reports error consistency. “κ / κ_max” normalises by the largest κ attainable at the two marginals, "
           "κ_max = (1 − |p_model − p_human| − p_e)/(1 − p_e), expressed as a percentage of the same ratio for human "
           f"pairs ({100 * B['ceiling_over_max']:.0f}% of their own bound); values above 100% mean a network attains a "
           "larger share of its much smaller attainable agreement than two observers do of theirs, not that it is more "
           "human-like than a human. "
           f"Over the {pr['K4b']['n_readouts']} gated readouts of the 38 ImageNet-trained networks, raw κ is a "
           f"function of the rate alone (one quadratic in the rate, R² = {P['curve']['r2']:.2f}, RMSE "
           f"{P['curve']['rmse']:.1f} points of ceiling; per-network offsets {n(P['curve']['offset_range'][0])} to "
           f"{n(P['curve']['offset_range'][1])} points, unrelated to the probe bias, ρ = "
           f"{n(P['curve']['rho_offset_probe'], 2)}), the bound itself running from {P['bound']['kappa_max_range'][0]:.3f} "
           f"to {P['bound']['kappa_max_range'][1]:.3f}. Normalised by the bound, the rank dependence on the rate "
           f"disappears (pooled rank R² {B['r2_rate_norm']:.3f} against {B['r2_rate_raw']:.3f}; within-network median "
           f"Spearman ρ {n(B['within_rho_norm_median'], 2)} against {n(B['within_rho_raw_median'], 2)}). Ranking the "
           f"{P['n_networks']} networks by κ at their trained readouts against by κ here gives Kendall τ = "
           f"{P['tau_kappa_trained_vs_matched']:.2f} with {P['n_pairs_reversing']} of {P['n_pairs']} pairs "
           f"({100 * P['kappa_pairs_reversing']:.0f}%) reversing. The Stylized-ImageNet ResNet-50 cannot be gained "
           f"below {min(b['rate'] for b in P['high_rate'] if b['group'] != 'imagenet'):.2f} inside its gate; at a common "
           f"rate of {P['high_rate_target']:.2f} it is "
           + "; ".join(f"{b['label']} {n(b['norm'], 0)}%" + (f" [{b['norm_ci'][0]:.0f}, {b['norm_ci'][1]:.0f}]" if b.get("norm_ci") else "")
                       for b in B["high_rate"])
           + " on the normalised scale (95% percentile bootstrap over the ten observers). "
           "Pre-specification: `docs/PLAN_KAPPA_2026-09-17.md` (predictions K-1 to K-6: four failed and a fifth could not "
           "be tested as specified; the "
           "normalisation by κ_max is post hoc). Source: `results/figures_v2/kappa_rate.json`, aggregated by "
           "`tmp_analysis/figs_v2/kappa_rate.py` from the per-network sweeps of "
           "`src/analysis/generality/kappa_sweep.py` (`slurm/zoo_kappa.sh`); the per-network files are outside the "
           "repository. `\\*` not ImageNet-trained.")
    return cap + "\n\n" + "\n".join(rows)


def main():
    K = json.load(open("results/figures_v2/kappa_rate.json"))
    md_path = "results/figures_v2/ed_tables.md"
    md = open(md_path, encoding="utf8").read()
    md = upsert(md, "Supplementary Table S13 |", table17(K))
    open(md_path, "w", encoding="utf8").write(md)
    print("updated", md_path, f"({len(K['matched'])} rows)")


if __name__ == "__main__":
    main()
