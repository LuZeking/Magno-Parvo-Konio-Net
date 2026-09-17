# Pre-registration: what the readout cannot move — agreement with human observers at matched rate

Written 2026-09-17, 18:40, before `src/analysis/generality/kappa_sweep.py` was run on any
network. Committed before the sweep. Extends `docs/PLAN_ZOO_2026-09-15.md` (W3), whose networks,
subspace rule, estimator, gate and probe are used unchanged; no network is retrained and no
feature is re-extracted. This analysis is **post hoc with respect to the manuscript's three
pre-registered plans** — it was prompted by the fifth referee simulation
(`docs/NMI_REVIEW_v5_2026-09-17.md`) — and will be labelled post hoc wherever it is reported.
What is pre-registered here is the prediction set, written before the numbers existed.

## Question

The manuscript shows that the cue-conflict rate is a property of the readout: one gain on a
network's own shape-predictive subspace moves it by 0.20–0.50 within five points of top-1, and
re-orders a quarter of all network pairs. That is a negative result about a measure the field
uses. Its positive companion is the complementary question, which the same data can answer:
**is there a quantity the readout does not control?**

Error consistency with the ten Geirhos observers (Cohen's κ on trial-by-trial agreement) is the
candidate. Extended Data Table 16 already reports it at three points per network — the trained
readout and the two ends of the gate — and two facts stand out: within every network κ rises
from the texture end to the shape end of the gate (38 of 38, median +13.5 points of the
ceiling), yet no ImageNet-trained readout anywhere in a gate exceeds 36.7% of the human–human
ceiling, while the Stylized-ImageNet ResNet-50 — the one network whose bias is representational
by the paper's own probe — reaches 39.5% at its trained readout and 49.7% at its shape end.

If κ is carried by the representation and the rate by the readout, then at a **matched rate**
networks should still differ in κ, and that difference should track the probe's representational
bias rather than the rate. If instead κ is a deterministic function of the rate, then κ carries
no information the rate does not, and the paper's reporting recommendation should say so.

## Design

**Networks.** The 42 of W3, as stored in `/scratch/byrong/mpk/generality/{features,behaviour}/
2026-09-15_zoo`: 38 torchvision `IMAGENET1K_V1` classifiers, ResNet-50 under the `IMAGENET1K_V2`
recipe, and the three Stylized-ImageNet ResNet-50s of Geirhos et al. (2019). Analyses of the
ranking-type statistics use the 38 ImageNet-trained networks; the recipe variant and the three
SIN controls are reported separately, as in W3.

**Sweep.** For each network and each w ∈ {0, 0.05, …, 1} — the grid of Extended Data Table 13 —
the stored cue-conflict features y, the network's own head (W, b) and the all-image bases
B_S, B_T give logits(w) = W·(P_R y + 2w·P_S y + 2(1 − w)·P_T y) + b, exactly the gain of
Methods 4.8. From those logits:

- the **rate** R(w) by the Geirhos 16-class rule of Methods 4.8, with its decided-image count;
- **κ(w)** by Methods 4.3 against each of the ten observers over the images both parties
  decided, reported as the mean over observers and as a percentage of the human–human ceiling
  (κ = 0.234, recomputed and asserted as in `error_consistency.py`);
- the **gate** G = {w : |top-1(w) − top-1 trained| ≤ 5 points} read from the stored
  `<tag>_gain_accuracy_subspace.json`; sweeps are reported over the full grid and over G.

Nothing else changes: same images, same estimator, same observers, same bases, same gate.

**Matched-rate comparison.** Target rate **r\* = 0.40**, tolerance ± 0.05, fixed now. For each
network, w_n(r\*) is the **gated** w whose rate is closest to r\*; a network enters the
comparison only if some gated w satisfies |R(w) − r\*| ≤ 0.05, and the number excluded is
reported. Secondary target **r\* = 0.30**, same rule, as a robustness check. No other target
will be examined.

**Statistics.** The zoo is a fixed set of networks, not a sample, so across-network statistics
are descriptive and carry no p-values, as in W3. The only sampled unit is the ten observers:
where an interval is quoted for a κ, it is a percentile bootstrap over the ten observers (1,000
draws). Rank statistics are Spearman ρ. κ values are used as percentages of the ceiling
throughout.

## Predictions

Each is directional with a threshold, and each is reported whether met or not.

- **K-1 (the readout moves the rate, not agreement).** Within each of the 38 ImageNet-trained
  networks, the gated rate range is ≥ 0.20 while κ stays below 40% of the ceiling at **every**
  gated w, in at least 36 of 38 networks. *Failed if fewer than 36.*
- **K-2 (agreement is not fixed by the rate).** At the matched rate r\* = 0.40 ± 0.05, κ across
  the entering ImageNet-trained networks varies by a factor of at least 2 (max ÷ min, on the
  percentage-of-ceiling scale). *Failed if the ratio is below 2.*
- **K-3 (the representation predicts agreement at matched rate).** Across the entering
  ImageNet-trained networks, Spearman ρ between κ at r\* and the probe's representational bias
  (shape − texture accuracy; less negative = more shape information) is ≥ +0.30. *Failed
  otherwise, including if the sign reverses.*
- **K-4a (representation against readout, as sources of variation in κ).** The standard
  deviation of κ across networks at the matched rate is at least as large as the median
  within-network standard deviation of κ across the gated sweep. *Failed if smaller.*
- **K-4b (the rate does not exhaust κ).** In a rank regression over all gated readouts of the 38
  networks pooled, adding the network's probe bias to the readout's rate raises R² by at least
  0.05 over the rate alone. *Failed if below 0.05.*
- **K-5 (the informative exception).** At the matched rate the Stylized-ImageNet ResNet-50's κ
  exceeds that of every ImageNet-trained network at the same rate. *Failed if any
  ImageNet-trained network matches or exceeds it.* (If SIN cannot reach r\* = 0.40 inside its
  gate, the comparison is made at the closest rate it does reach and the difference in rate is
  stated.)
- **K-6 (no readout reaches the human regime).** Over the **whole** grid, gate ignored, no
  ImageNet-trained network exceeds 50% of the ceiling at any w. *Failed if any does* — which
  would show the gate, not the representation, was the binding constraint.

## What is reported regardless of outcome

The full 21-point (rate, κ) trajectory of every network; the matched-rate table; the outcome of
each prediction above; and, if K-2, K-3 and K-4 fail together, the conclusion that κ adds
nothing to the rate in these networks, which contradicts the manuscript's current suggestion
that error consistency "belongs beside" the three numbers and would be stated as such in the
Discussion.

## Decision rules fixed in advance

1. If fewer than 20 ImageNet-trained networks reach r\* = 0.40 inside their gate, the primary
   target becomes r\* = 0.30 and the switch is reported.
2. κ is undefined where fewer than 30 images are decided by both parties (`kappa()` returns
   NaN); such readouts are dropped and counted, not imputed.
3. No network is added to or removed from the zoo for this analysis.
4. The ceiling, the human shape-bias mean and the estimator are asserted against their published
   values at run time, as the existing script does; a failed assertion stops the run.

## Outputs

- `src/analysis/generality/kappa_sweep.py` → `<tag>_kappa_sweep.json` per network in
  `/scratch/byrong/mpk/generality/behaviour/2026-09-15_zoo/`, driven by `slurm/zoo_kappa.sh`
  (one CPU job over the 42 stored feature files; no GPU, no extraction).
- `tmp_analysis/figs_v2/kappa_rate.py` → `results/figures_v2/kappa_rate.json` (committed) with
  the trajectories, the matched-rate table and the outcome of K-1 … K-6, plus an Extended Data
  table and the numbers for the manuscript.
- Outcomes appended to this file after the run, as in the other plan files.

## Outcomes

*(appended after the run)*
