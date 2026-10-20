# PAPER_v2 → v3: the developmental-diet TSN as model of record (plan, 2026-10-19)

User decision 2026-10-19: the main TSN is `fs2_dvd300_avg_w050_ls` (seeds 0–2). It is the
BA network at test time (two ResNet-18, averaged layer-4, one head, one cross-entropy), trained
with the Stage-2 recipe (AdamW 1e-3, wd 0.05, cosine, EMA 0.9995, label smoothing 0.1,
50 epochs). Its magno diet follows the DVD curves at one age that ramps 0 → 300 months over
training; tests use 300 months. Controls:

- `fs2_dvdfix300` — the 300-month diet from the first step, so the same test input without the
  order. Seed 0 is done; seeds 1–2 are training (26671132).
- `fs2_dvd300swap` — the ramp applied to L−M instead. Seeds 0–2 are queued (26671793).

The paper's BA numbers stay as the comparison and come from the same pipelines.

## The story in one line

v2: the magno stream's shape bias *starts in its input* and survives training under the diet.

v3: the magno stream's shape bias is *learned through the developmental order*. The same
network on the same adult input learns it only if the input grew up from coarse to fine.
Everything downstream holds or gets stronger: the texture-biased stream carries extrastriate
cortex, and behaviour is a readout over both. The one exception is error consistency (§2.5).

## Section by section

| v2 claim | dvd300 evidence | action |
|---|---|---|
| §2.1 diet costs accuracy (0.556 best / 0.445 final vs 0.701) | 0.683 ± 0.002 vs control 0.761 (1 seed); converges, no late decline; best-checkpoint policy no longer needed (final EMA throughout) | rewrite §2.1 cost paragraph; Methods 4.2 recipe + 4.1 diet schedule; drop the best-checkpoint machinery for the model of record |
| §2.1 single-stream controls (0.599 / 0.623) | not run for the ramp (user did not approve) | keep as BA-diet context or drop |
| §2.2 Table 1 probe, held out +0.337 / −0.168 | **+0.316 / −0.156**, every seed on its side | swap in |
| Table 1 probe, shared +0.819 / −0.518 | +0.578 / −0.454 | swap in |
| Table 1 cue-conflict 0.878 / 0.418 / 0.675 | 0.850 / 0.477 / 0.682 | swap in |
| Table 1 untrained +0.210 (shape bias "starts in the input") | untrained on the dvd300 input **+0.087** = no-diet level (+0.090); learning adds +0.229 (BA +0.127); the control learns to −0.180 | **reframe the mechanism**: learned, and only in order |
| §2.2 no diet −0.122 / −0.118 | control −0.180 / −0.079 (1 seed) | swap in once seeds 1–2 land (~10 h after start) |
| §2.2 single ResNet-18 loses the bias ("survives beside a second stream") | not run | **drop** for v3 (or state as BA-only) |
| §2.2 2 × 2 carrier × transform, interaction +0.36 | 3 of 4 cells in hand (dvd300 magno, dvd300 parvo, control magno); swap cell training | recompute when swap seeds land |
| §2.2 accuracy-matched control (4.3% / 6.1%) | moot: the control is the *more* accurate network | drop; say why |
| §2.3 gradient −0.745 to −0.834, ρ, r²/NC 0.405 vs 0.464, cc +1.53 | gradient **−1.27** (8/8 steeper than BA), ρ −0.97, r²/NC **0.471**, cc +1.32; control −0.06 (8/8) | swap in; diet's BOLD cost gone |
| §2.3 untrained −0.544 / −0.479 ("diet does not create the gradient") | untrained dvd300 encoding running (26671457) | rewrite after it lands: expect learned, not given |
| §2.3 atlas (ED Fig 4), Um/Up beyond V3 | all regions −1.12 to −1.27 vs V1, 8/8; Um .013 / Up .149 | swap in |
| §2.4 integrated 0.675 ± 0.025; integration cost 0.203 | 0.682; cost 0.850 − 0.682 = 0.168 | swap in |
| §2.4 w sweep 0.40 → 0.90 | 0.48 → 0.85 (final); control flat 0.38 | swap in |
| §2.4 refit nonlinear head 92% of affine | running (26671456 → 26671458); summary step has no producer yet | write the summary step |
| §2.4 cortex w* 0.58 → 0.33, 8/8 | 0.64 → 0.32, 8/8; control 0.51 → 0.76 | swap in |
| §2.4 behaviour w* not identified (2/30 beat null) | median 0.79, 10/30 beat null, CI width 0.70 | report; still not identified |
| §2.4 frozen-network zoo, Stylized-ImageNet, Fig 6 | independent of the TSN | unchanged; the concat row (0.50 vs 0.25) is recomputed (generality 26671454) |
| §2.5 OOD: diet less robust on 4/5 families, robust to low-pass | running (26671464, 6–9 h); seed 2 shows low-pass robustness | compare dvd300 vs control |
| §2.5 κ: most shape-biased stream has the lowest κ (3.5% of ceiling) | **magno alone 17.3%** (5.1 / 25.6 / 21.3), integrated 13.2%, parvo 10.3%, best 29.0% | **claim fails on dvd300.** κ now tracks the rate across readouts, as the paper's own κ-bound account of the zoo predicts. Re-lead the paragraph with the bound; recompute the equivalence interval honestly |
| §2.5 dimensionality (PR ratio) | from encoding features (CPU) | recompute (fig1_data.py) |
| Discussion "most of the shape bias is present before training" | reversed | rewrite |
| Discussion "less accurate, less stable, less robust, worse BOLD predictor" | more accurate than the BA, stable, BOLD better than the BA (still below the control) | rewrite |
| Limitations "a fixed 48-month front end rather than a trajectory" | resolved | delete; add "one control seed" if seeds 1–2 fail |

## Work list

1. Wait: control seeds 1–2, swap seeds 0–2 (A5000, ~10 h each once started), OOD, generality,
   nonlinear head, untrained encoding. Rerun the phase-B wrappers with `RUN_LIST` for the new
   seeds.
2. Figure scripts: add the fs2 names to the fixed BA lists (exemplar_aggregate.py:184-199,
   fig2_streams.py:60,126,153, fig1_data.py:52-62, fig4_readout.py:111-136,
   fig5_dissociations.py:52-131, ed_tables.py:154-170, generality_behaviour.py:24,
   ed_tables_generality.py:32). Filter the unfiltered globs (fig4_readout.py:40,
   ed_tables.py:187, wfit_summary.py:56, check_attainable_counts.py:7,
   verify_new_claims.py:1398).
3. Rewrite Abstract, §1 last paragraph, §2.1–2.5, Discussion, Methods 4.1/4.2/4.7, Limitations.
4. Verifier: every literal and every `dkl_bcs` glob in tmp_analysis/verify_new_claims.py
   moves to dvd300; the BA checks stay as a comparison block.

## Status 2026-10-19 22:50

Item 3 drafted: `docs/PAPER_v3.md` (Abstract 198 words; Results, Discussion, Limitations,
Methods 4.1–4.7 rewritten; ED Tables 1–2 rewritten by hand; ED Table 3 gains the model's
concatenation row). Numbers still open are marked ⟦PENDING …⟧ in the text: control seeds 1–2
(26671132), swap seeds 0–2 (26671793) and the 2 × 2 interaction, untrained encoding 8/8
(26671457), per-image correspondence, nuisance-axis controls (ED Fig. 3), compute accounting.
Changes from the table above, found while writing: the profile crosses zero (V1 magno-explained);
on the raw statistic training in order does not steepen the untrained gradient (−0.21 → −0.19),
so §2.3 claims preservation, not steepening; the magno stream decodes held-out texture above
chance (0.148), so "texture at chance" is dropped; κ re-led as planned (model 13.2% vs control
10.3%, magno 17.3%; seven-model slope 0.020 [0.013, 0.028] over the span). Items 2 and 4
(figure scripts, verifier) are next.

## Status 2026-10-20 11:25



User "1 2 3 yes": PAPER_final_2026-10-20.md adopted as the route R-C paper of record; PAPER_v3 is the lead paper. Items 2 and 4 done: figures on the model in `results/figures_v3/` (style.py writes there by default, `FIG_OUT` overrides; v2 BA figures untouched), and `tmp_analysis/verify_v3_claims.py` (450 phrase-anchored checks, then the BA checker as the comparison block; both pass). The checker found 11 text errors, fixed (training-time ranges 0.72–0.92 / 0.32–0.45 and the first-epoch onset, control never above 0.45, epoch 7 of 50, BA −0.833, ventral −1.25, dorsal IPS0–3 in the atlas contrast, hV4 U_m 0.015, control sweep 0.37–0.39, 37–55%, control w* 6/8 + 1 tie, BOLD cost 0.026). Phase-B wrappers now accept the swap run (607378e). Remaining: item 1 (control seeds 1–2, swap; training ends ~22:00 10-20).
