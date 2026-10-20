# Extended Data Figure 1 — caption

**The out-of-distribution degradation curves behind Fig. 5a.** Accuracy relative to clean accuracy against corruption severity for the seven families, for the model and its control (final checkpoints; identical test input) and the BA network's two arms (best saved checkpoints); thin lines: seeds, thick: seed mean. Contrast and low-pass are trained in by the diets. The model is less robust than its control on the four untrained families other than phase scrambling, and more robust only to low-pass.

Sources: `results/ood/`, `results/ood_bestep/`; `tmp_analysis/figs_v2/figE1_ood_curves.py`.
