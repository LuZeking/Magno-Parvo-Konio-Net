# Figure 4 — caption

**Shape bias does not confer out-of-distribution robustness.**

Corruptions are applied to the image **before** the front end and scored in the
models' own 112-way label space, at each seed's best saved checkpoint
(PLAN §71.6). Thin lines are individual training seeds (n = 3 per condition);
thick lines their mean. The y-axis is accuracy divided by that seed's **own**
clean accuracy, which is the statistic §2.6.1 reports: the diet arm is less
accurate before any corruption is applied (§2.7), so an absolute plot would
confound robustness with clean accuracy. This is the same computation as the
AUCs in the text (`tmp_analysis/ood_verdict.py`), not a re-derivation of them.

**(a–g) The diet-trained model degrades faster, not slower.** Of the five
corruptions the diet was not trained on, it is less robust on four — uniform
noise, high-pass, false colour and rotation — with complete seed separation on
each. The two conditions that carry no shape bias (`swap`, `parvo_highsf`)
track the no-diet control rather than the diet, which is what makes the pattern
a property of the diet and not of any front-end change.

*The two families marked "trained in".* The developmental front end is itself a
low-pass blur with a contrast-sensitivity cut, so low-pass and contrast are the
two corruptions the diet has effectively seen. Marking them is the point rather
than an apology: **the diet is more robust on low-pass — the one place
robustness was built in — and is *less* robust on contrast even though its
front end includes a contrast-sensitivity cut.** Robustness appears where it
was trained in and nowhere else, and being trained in does not by itself
confer it. This is the strongest available evidence that the nulls are real
rather than a broken measurement.

*Phase-scrambling* is the one family where the two arms are indistinguishable.
It is drawn as a null and bounded as one — the difference is within ±0.033,
below the smallest OOD effect this paper claims — and it is the weakest of the
paper's four equivalence results, resting on a Welch interval at 2.3 degrees of
freedom (§2.6.1).

**(h) Summary.** Relative-accuracy AUC per family, one marker per training seed,
diet against no diet. ✱ marks complete seed separation. **No significance stars
are drawn**: at 3 seeds against 3 the exact two-sided Mann–Whitney floor is
p = 0.100, reached only on complete separation, so a threshold marker would
misrepresent a floor as a crossing. Separation is shown as separation instead.

*Scope.* This figure does not contradict Geirhos et al. (2019), who showed that
Stylized-ImageNet training yields models that are both shape-biased and robust.
It shows the coupling is a property of that route: a developmental route to
comparable shape bias *in rate* delivers none of the robustness.

Built by `tmp_analysis/fig4_ood_degradation.py` from `results/ood_bestep/`.
