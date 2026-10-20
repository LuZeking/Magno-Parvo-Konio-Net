# Figure 2 — caption

**The two streams specialise in opposite directions, both by learning, and the magno stream's shape bias only with a maturing input.**

**(a)** Linear-probe shape − texture accuracy per stream on exemplars the probe was not trained on (exemplar-balanced, leave-one-exemplar-out; Methods 4.3), before and after training; solid: magno stream, dashed: parvo stream; bands: seed range. The model and its control start from one untrained network on one test input (magno +0.087, parvo −0.033). Training on the maturing input ends at +0.316 (magno) and −0.156 (parvo), means of three seeds; training on the adult input from the first step ends at −0.180 and −0.079 (control, seed 0). The BA network starts from its own blurred front end (+0.210, −0.033) and ends at +0.337 and −0.168.

**(b, c)** Carrier × ramp on the probe (b) and on cue-conflict shape bias (c). Achromatic carrier: the control's magno stream without the ramp, the model's with it (probe −0.180 → +0.316; behaviour 0.378 → 0.850). Chromatic carrier without the ramp: the control's parvo stream (probe −0.079; behaviour 0.388); open markers: the model's parvo stream, beside a ramped magno stream (−0.156; 0.477). The fourth cell, the ramp on L−M with luminance untransformed (`fs2_dvd300swap`), is training and is drawn when its results exist.

**(d)** How each readout decides the 1,200 cue-conflict images: decided as the shape category, as the texture category, or neither. Shape bias is computed over the decided images only, a count that varies across readouts (426 to 580 in the model, 224 to 675 in the control).

**(e)** Training-time evaluation (the shared head on the magno stream alone, seed 0): the model's magno stream is shape-biased at every epoch evaluated (0.72–0.92) and the control's texture-biased (0.32–0.45); the top axis gives the model's diet age. **(f)** The same, against the network's validation top-1: at the model's final accuracy (0.683, open blue), the control (epoch 7, top-1 0.683) has a magno-stream shape bias of 0.41 (open vermillion).

Sources: `results/figures_v2/exemplar_streams.json` (probe; `two_by_two_v3`), `results/cue_conflict/`, run histories; `tmp_analysis/figs_v2/fig2_streams.py`.
