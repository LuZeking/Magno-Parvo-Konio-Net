# Figure 4 — caption

**Behaviour reads out from both streams, and the shape-bias rate is a property of that readout.**

**(a)** Cue-conflict shape bias against the readout weight w on the magno stream, holding both streams' features fixed (21 values of w). The model (final checkpoints, three seeds; thin: seeds) runs from 0.48 at w = 0 to 0.85 at w = 1, and w = 0.5 reproduces each trained model; the control's curve is flat (0.37–0.39, one seed); the BA network (best saved checkpoints) for comparison. Grey band and line: the ten human observers (mean 0.958); diamonds: the CNNs of Geirhos et al. (2018a) on the identical images and statistic.

**(b)** The span of shape bias over w for the frozen affine head and for heads refitted on the frozen features at each w (linear; two-layer nonlinear), per seed of the model (bar: median). Median spans: 0.371 affine, 0.395 nonlinear (106%; weakest seed 96%). Dashed: the span over readouts within 5 points of peak validation accuracy (nonlinear median 0.241). Open circles: the BA network's medians.

**(c)** Shape bias against validation top-1 at the scored checkpoint, for each network's magno stream alone (▲), integrated readout (●) and parvo stream alone (▼); the model and its control at final checkpoints, the BA network at its best saved checkpoints, the BA no-diet arm at its final ones. Integration costs the model 0.168 of its magno stream's 0.850.

Sources: `results/fusion_sweep/`, `results/nonlinear_readout_summary.json`, `results/cue_conflict*/`, run histories; `tmp_analysis/figs_v2/fig4_readout.py`.
