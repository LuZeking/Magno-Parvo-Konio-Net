"""Extended Data Table 16: the readout span across the model zoo (PLAN_ZOO).

  PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/ed_table16_zoo.py

Reads results/figures_v2/zoo_behaviour.json (tmp_analysis/figs_v2/zoo_behaviour.py) and upserts
the table into results/figures_v2/ed_tables.md; insert_ed_tables.py then places it in the paper.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ed_tables_generality import fmt, upsert  # noqa: E402

GROUP_TITLE = {"imagenet": "**ImageNet-trained (torchvision IMAGENET1K_V1)**",
               "recipe": "**same architecture, different training recipe**",
               "sin": "**Stylized-ImageNet controls (Geirhos et al., 2019)**"}


def table16(zoo):
    rows = ["| network | D | top-1 (subset) | trained rate (images decided) | gate: w-range (largest Δtop-1) | rate at texture end / shape end | gated span, subspace rule | ρ(w, rate) | gated span, unit rule | probe: shape / texture accuracy (bias) | κ with observers, % of ceiling: trained (texture end → shape end) | whole sweep in gate |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    nets = zoo["networks"]
    for group in ("imagenet", "recipe", "sin"):
        sub = sorted([n for n in nets if n["group"] == group], key=lambda n: -n["trained"])
        if not sub:
            continue
        rows.append(f"| {GROUP_TITLE[group]} | | | | | | | | | | | |")
        for n in sub:
            pr = n.get("probe")
            probe_s = f"{pr['shape_acc']:.2f} / {pr['texture_acc']:.2f} ({pr['bias']:+.2f})" if pr else "—"
            kp = n.get("kappa")
            kappa_s = (f"{kp['trained_pct']:.1f} ({kp['texture_end_pct']:.1f} → {kp['shape_end_pct']:.1f})"
                       if kp and kp.get("texture_end_pct") is not None else (f"{kp['trained_pct']:.1f}" if kp else "—"))
            rows.append(f"| {n['label']} | {n['n_units']} | {n['trained_top1']:.3f} | {n['trained']:.3f} ({n['decided']}) | "
                        f"{n['gate'][0]:.2f}–{n['gate'][1]:.2f} ({n['max_abs_dtop1_in_gate'] * 100:.1f} pts) | "
                        f"{n['texture_end']:.3f} / {n['shape_end']:.3f} | {n['gated_span']:.3f} | {n['rho']:.2f} | "
                        f"{fmt(n.get('unit_gated_span'), 3)} | {probe_s} | {kappa_s} | {'yes' if n['whole_sweep_in_gate'] else 'no'} |")
    st = zoo.get("ranking", {})
    n_im = st.get("n", 0)
    stat_s = ""
    if st:
        stat_s = (f"Over the {n_im} ImageNet-trained networks: Kendall τ between the ranking by trained rate and by the shape-end rate {st['tau_trained_vs_shape_end']:.2f} "
                  f"(by the texture-end rate {st['tau_trained_vs_texture_end']:.2f}); {st['reversals_trained_vs_shape_end']} of {st['n_pairs']} pairs ({100 * st['frac_reversed']:.0f}%) reverse order; "
                  f"gated span {st['gated_span_min']:.2f}–{st['gated_span_max']:.2f} (median {st['gated_span_median']:.2f}), ≥ 0.20 in {100 * st['frac_gated_ge_0_20']:.0f}% and the whole sweep inside the gate in {100 * st['frac_whole_sweep_in_gate']:.0f}%; "
                  f"Spearman ρ between gated span and trained rate {st['rho_span_vs_trained']:+.2f}, and trained top-1 {st['rho_span_vs_top1']:+.2f}"
                  + (f"; error consistency at the trained readout {st['kappa_trained_pct_min']:.1f}–{st['kappa_trained_pct_max']:.1f}% of the ceiling (median {st['kappa_trained_pct_median']:.1f}%), "
                     f"at most {st['kappa_best_in_gate_pct_max']:.1f}% anywhere inside a gate, changing by a median of {st['kappa_shape_minus_texture_end_pct_median']:+.1f} points from the texture end to the shape end; ρ(κ, trained rate) {st['rho_kappa_vs_trained_rate']:+.2f}" if "kappa_trained_pct_min" in st else "")
                  + (f"; Kendall τ between trained rate and probe bias {st['tau_trained_vs_probe_bias']:.2f}" if "tau_trained_vs_probe_bias" in st else "") + ". ")
    cap = ("**Extended Data Table 16 | The readout span across the field's model zoo.** "
           "Frozen ImageNet networks under the subspace rule of Extended Data Table 13 (PCA to 256 dimensions, 16-class shape and texture LDA, 15 directions each, symmetric orthogonalisation; bases fitted on one stratified half of the 1,200 cue-conflict images and the rate scored on the other, 10 splits both directions), each network's own head untouched, the Geirhos 16-class probability aggregation throughout; “images decided” is the number of the 1,200 assigned to their shape or their texture category at the trained readout, the denominator of the rate. "
           "“gate” is the w-range over which top-1 on a stratified 10-per-class ILSVRC-2012 validation subset (10,000 images; bases fitted on all cue-conflict images) stays within 5 points of the trained value, with the largest change of top-1 inside it; “texture end / shape end” are the held-out rates at the smallest and the largest gated w, and “gated span” is their difference, under the subspace rule and under the pre-specified unit-level median split of Extended Data Table 14. "
           "“probe” is the linear probe of Methods 4.3 (PCA to 256 dimensions, multinomial logistic regression, 5-fold stratified cross-validation) predicting the shape class and, separately, the texture class from the same features; bias is shape minus texture accuracy (chance 0.0625). "
           "Preprocessing is Resize 256 / CenterCrop 224 except for Inception-v3 (342/299), EfficientNet-B3 (320/300) and EfficientNet-V2-S (384/384), which were not trained at 224. "
           "“κ with observers” is error consistency (Cohen's κ on trial-by-trial agreement, Results 2.5; Extended Data Table 7) with each of the ten human observers, mean over observers, as a percentage of the human–human ceiling κ = 0.234, at the trained readout and, in parentheses, at the texture end and the shape end of the gate under the same gain; computed under this pipeline's preprocessing, so the VGG-16 value differs from the published response file's 7.3% (Extended Data Table 7) as its trained rate does. "
           + stat_s +
           "The three Stylized-ImageNet networks are the published checkpoints of Geirhos et al. (2019), all ResNet-50: trained on Stylized-ImageNet (SIN), on SIN + IN, and on SIN + IN then fine-tuned on IN. "
           "Pre-specification: `docs/PLAN_ZOO_2026-09-15.md` (predictions P-Z1–P-Z4). Source: `results/figures_v2/zoo_behaviour.json`, aggregated by `tmp_analysis/figs_v2/zoo_behaviour.py` from the outputs of `slurm/zoo_cc.sh`; the per-network files are outside the repository.")
    return cap + "\n\n" + "\n".join(rows)


def main():
    zoo = json.load(open("results/figures_v2/zoo_behaviour.json"))
    md_path = "results/figures_v2/ed_tables.md"
    md = open(md_path, encoding="utf8").read()
    md = upsert(md, "Extended Data Table 16 |", table16(zoo))
    open(md_path, "w", encoding="utf8").write(md)
    print("updated", md_path, f"({len(zoo['networks'])} networks; missing {len(zoo['missing'])})")


if __name__ == "__main__":
    main()
