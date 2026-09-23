"""Supplementary Table S14: what the gain acts on — the span decomposed (PLAN_SPAN).

  PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/ed_table18_span.py

Reads results/figures_v2/span_model.json (tmp_analysis/figs_v2/span_model.py) and upserts the
table into results/figures_v2/ed_tables.md; insert_ed_tables.py then places it in the paper.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ed_table16_zoo import GROUP_TITLE  # noqa: E402
from ed_tables_generality import upsert  # noqa: E402


def n(x, d=2):
    return f"{x:.{d}f}".replace("-", "−")


def ordinal(k):
    """1 -> 'smallest', 9 -> '9th smallest'."""
    return "smallest" if k == 1 else f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')} smallest"


def table18(S):
    P, F, H = S["predictions"], S["forced_estimator"], S["post_hoc"]
    rows = ["| network | trained rate: primary / forced | span: reported / in sample / forced | forced span, M0 / M1 | z / λ | shape term / texture term (texture share) | random-direction span, absolute: median / largest |",
            "|---|---|---|---|---|---|---|"]
    nets = S["networks"]
    for group in ("imagenet", "recipe", "sin"):
        sub = sorted([x for x in nets if x["group"] == group], key=lambda x: -x["texture_share"])
        rows.append(f"| {GROUP_TITLE[group]} | | | | | | |")
        for x in sub:
            rows.append(f"| {x['label']} | {x['rate_trained']:.3f} / {x['forced_trained']:.3f} | "
                        f"{x['T_p']:.3f} / {x['span_insample']:.3f} / {x['T_f']:.3f} | {x['m0_span']:.3f} / {x['m1_span']:.3f} | "
                        f"{n(x['z'])} / {x['lam']:.2f} | {x['mean_eS']:.2f} / {x['texture_term']:.2f} ({x['texture_share']:.2f}) | "
                        f"{x['rand_median_abs']:.3f} / {x['rand_max_abs']:.3f} |")
    s5 = H["sin"]
    ez = json.load(open("results/figures_v2/exemplar_zoo.json"))
    im_ex = [r for r in ez["networks"] if r["group"] == "imagenet"]
    sin_e1 = next(r for r in ez["networks"] if r["tag"] == "hub_resnet50_sin")["spans"]["E1"]["gated"]
    EX = dict(rho_q1=ez["stats"]["rho_texture_share_Q1"], tau_E1=ez["stats"]["ranking"]["tau_trained_vs_E1_shape_end"],
              frac_E1=ez["stats"]["ranking"]["E1_frac_reversed"],
              sin_rank_E1=1 + sum(r["spans"]["E1"]["gated"] < sin_e1 for r in im_ex))
    cap = ("**Supplementary Table S14 | What the gain acts on: the span decomposed.** "
           "Every frozen network of Extended Data Table 4, at its trained readout. For each cue-conflict image the Geirhos rule "
           "gives a margin Δ = log p_shape − log p_texture (p_c the mean softmax probability over category c's ImageNet classes; "
           "the forced two-way estimator calls the image “shape” iff Δ > 0). The gain of Methods 4.8 adds (2w − 1)·W(P_S − P_T)y "
           "to the logits exactly, so to first order it changes Δ by (2w − 1)(e_S − e_T), with e_S and e_T the evidence the head "
           "receives through the shape and the texture subspace (u·W P_S y and u·W P_T y, u the gradient of Δ with respect to the "
           "logits). “Shape term” is the mean of e_S over the 1,200 images (evidence added at the shape end), “texture term” the "
           "mean of −e_T (texture evidence removed there), both in units of the margin; texture share = texture term / (shape term "
           "+ texture term). M0 is the first-order forced-choice span from the per-image margins, M1 the two-number span "
           "Φ(z + λ) − Φ(z − λ) with z = mean Δ / sd Δ and λ = mean(e_S − e_T) / sd Δ. “Reported” is the held-out span of Extended "
           "Data Table 16 (primary estimator, on halves that share exemplars); “in sample” is the same estimator and “forced” the forced two-way estimator, both with "
           "the bases fitted on all 1,200 images. Random-direction span: ten draws (seeds 0–9) of two mutually orthogonal random "
           "15-dimensional subspaces inside each network's 256-dimensional PCA space of the cue-conflict features, same gain, "
           "estimator (in sample) and gate; every random sweep stays inside its gate. "
           f"Pre-specification: `docs/PLAN_SPAN_2026-09-18.md`, predictions S-1 to S-7, written after Extended Data Table 4 and "
           "before this analysis; the whole analysis is post hoc with respect to the manuscript's first three plans. Met: S-1 "
           f"(M0 reproduces the forced-choice curve, median largest error {P['S1']['median_max_abs_err']:.3f}; ρ(M0, forced span) "
           f"= {P['S1']['rho_m0_Tf']:.2f}), S-5b (the Stylized-ImageNet texture term {P['S5b']['sin_texture_term']:.2f} is below "
           f"every ImageNet network's, {P['S5b']['imagenet_texture_term_range'][0]:.2f}–{P['S5b']['imagenet_texture_term_range'][1]:.2f}) "
           f"and S-7 (median absolute mean signed random span {P['S7']['median_abs_mean_signed']:.3f}; the reported span exceeds the "
           f"largest random span in {P['S7']['n_Tp_exceeds_max_random']} of {P['S7']['n']} ImageNet networks and "
           f"{P['S7']['all42_n_exceeds']} of 42). Failed: S-2 (M1 ranks the forced spans at ρ = {P['S2']['rho_m1_Tf']:.2f} but "
           f"the reported spans at {P['S2']['rho_m1_Tp']:.2f}, against 0.7), S-3 (ρ(λ, reported span) = {P['S3']['lam_Tp']['rho']:.2f}, "
           f"against 0.7), S-4 (ρ(λ, z) = {n(P['S4']['lam_z']['rho'])}, against |ρ| ≤ 0.3), S-5a (the Stylized-ImageNet λ "
           f"{P['S5a']['sin_lam']:.2f} lies inside the ImageNet range {P['S5a']['imagenet_lam_range'][0]:.2f}–"
           f"{P['S5a']['imagenet_lam_range'][1]:.2f}) and S-6 (a decision-free head-weighted subspace share ranks the reported "
           f"spans at ρ = {P['S6']['H_Tp']['rho']:.2f}, against 0.5, no better than the unweighted variance share, "
           f"{P['S6']['varfrac_Tp']['rho']:.2f}). "
           "Post hoc: the reported and the forced spans rank the 38 ImageNet networks at ρ = "
           f"{S['reported']['Tp_Tf']:.2f} (in-sample primary {H['rho_Tp_insample_primary']:.2f} with the reported and "
           f"{H['rho_insample_primary_Tf']:.2f} with the forced); the forced span rises with the trained forced rate (ρ = "
           f"{H['rho_Tf_forced_trained']['rho']:.2f} [{H['rho_Tf_forced_trained']['ci'][0]:.2f}, {H['rho_Tf_forced_trained']['ci'][1]:.2f}]) "
           f"where the reported one does not ({n(S['reported']['span_rate'])}); among the {s5['of']} networks (the 38 and itself) the "
           f"Stylized-ImageNet span is the {ordinal(s5['rank_T_p'])} reported, the {ordinal(s5['rank_insample'])} "
           f"in sample and the {ordinal(s5['rank_T_f'])} forced, and the {ordinal(EX['sin_rank_E1'])} with exemplars held out. The texture term "
           f"exceeds the shape term in {H['n_texture_dominant']} of {H['n']} ImageNet networks (texture share median "
           f"{H['texture_share_median']:.2f}, range {H['texture_share_range'][0]:.2f}–{H['texture_share_range'][1]:.2f}; the "
           f"exception is {', '.join(H['texture_share_exception'])}), and the texture share tracks the probe bias of Extended Data "
           f"Table 16 (ρ = {n(H['rho_texture_share_probe_bias']['rho'])} [{n(H['rho_texture_share_probe_bias']['ci'][0])}, "
           f"{n(H['rho_texture_share_probe_bias']['ci'][1])}]; {n(H['rho_texture_share_probe_bias_all42'])} over all 42) more "
           f"closely than the trained rate does ({n(H['rho_rate_probe_bias']['rho'])} [{n(H['rho_rate_probe_bias']['ci'][0])}, "
           f"{n(H['rho_rate_probe_bias']['ci'][1])}]; difference in |ρ| {H['abs_rho_difference']['point']:.2f} "
           f"[{H['abs_rho_difference']['ci'][0]:.2f}, {H['abs_rho_difference']['ci'][1]:.2f}]); against the exemplar-balanced probe "
           f"scored on held-out exemplars (Extended Data Table 4) it is {n(EX['rho_q1']['rho'])} [{n(EX['rho_q1']['ci'][0])}, "
           f"{n(EX['rho_q1']['ci'][1])}], no closer than the trained rate. Intervals are 95% percentile "
           "bootstraps over networks (10,000 resamples), descriptive of this fixed zoo. On the forced estimator the ranking by "
           f"trained rate and by the shape-end rate agree at Kendall τ = {F['tau_forced']:.2f} with {F['reversals_forced']} of "
           f"{F['n_pairs']} pairs ({100 * F['frac_forced']:.0f}%) reversing (in-sample primary τ = {F['tau_insample_primary']:.2f}, "
           f"{100 * F['frac_insample_primary']:.0f}%; held-out primary {F['tau_heldout_primary']:.2f}, "
           f"{100 * F['reversals_heldout_primary'] / F['n_pairs']:.0f}% with exemplars shared and {EX['tau_E1']:.2f}, "
           f"{100 * EX['frac_E1']:.1f}% with exemplars held out). Source: `results/figures_v2/span_model.json`, aggregated "
           "by `tmp_analysis/figs_v2/span_model.py` from `src/analysis/generality/span_model.py` (`slurm/zoo_span.sh`); the "
           "per-network files are outside the repository.")
    return cap + "\n\n" + "\n".join(rows)


def main():
    S = json.load(open("results/figures_v2/span_model.json"))
    md_path = "results/figures_v2/ed_tables.md"
    md = open(md_path, encoding="utf8").read()
    md = upsert(md, "Supplementary Table S14 |", table18(S))
    open(md_path, "w", encoding="utf8").write(md)
    print("updated", md_path, f"({len(S['networks'])} rows)")


if __name__ == "__main__":
    main()
