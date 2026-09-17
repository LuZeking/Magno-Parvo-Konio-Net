# Shape bias is a readout property: a developmental two-stream model, frozen networks and human cortex

## Abstract

Humans name objects by shape when shape and texture disagree, yet the ventral visual stream is well described by texture statistics. We test this with a two-stream network approximating the magnocellular and parvocellular inputs: one ResNet-18 receives the luminance channel of a cone-opponent image under a developmental visual diet, the other its L−M channel. The magno stream's shape bias is supplied by its input and survives training only under the diet; the parvo stream's texture bias is learnt. Fitted to eight Natural Scenes Dataset subjects, the texture-biased stream carries every extrastriate region. Behaviour, reading out from both, is shape-biased; one readout weight moves the rate from 0.40 to 0.90, and the weight that best predicts extrastriate cortex (0.33) is not the one that best predicts human choices (0.67). In 38 frozen ImageNet networks with no stream split, one gain on the network's own shape-predictive subspace moves the rate by 0.20–0.50 within five points of top-1, re-ordering a quarter of all network pairs; a Stylized-ImageNet network moves least, its bias sitting in the representation; and in our single trunks extrastriate cortex loads on the texture-leaning units. Shape-biased behaviour does not require shape-biased representations; the cue-conflict rate is a readout statistic.

## 1. Introduction

Humans presented with an image whose shape and texture point to different categories name the shape: on the cue-conflict images of Geirhos et al. (2018a), ten observers did so on 96% of decided trials, while ImageNet-trained convolutional networks named the texture. Closing that gap has become a standard target for biologically motivated models (Hermann et al., 2020; Jang & Tong, 2021), and a developmental visual diet — training on input whose acuity, contrast sensitivity and chromatic sensitivity follow the human developmental trajectory — is among the most effective routes to it (Lu et al., 2026).

A second body of evidence sits uneasily beside the first: along the ventral stream, object representations are well described by texture statistics — category-selective human cortex fails to discriminate natural images from texture-matched scrambles yet still discriminates categories (Jagadeesh & Gardner, 2022), macaque V4 responses to natural textures are predicted by their image statistics (Okazawa et al., 2015), and in inferotemporal cortex texture is the more decodable cue (Jagadeesh & Livingstone, 2024; Komatsu & Goda, 2018). The premise is scoped — one human study with one manipulation, the rest macaque, and “texture-like” meaning that a texture-statistics description suffices, not that shape is absent — and the argument needs only that the ventral description is substantially texture-like.

If behaviour is shape-biased and the ventral stream texture-biased, either the two measures disagree about what shape bias is, or behaviour is not a readout of one representation. The second horn is a hypothesis about architecture that single-trunk models of the ventral stream cannot express, and two facts make it concrete: the visual system receives not one image but a luminance-dominated, spatially coarse magnocellular signal and a chromatically opponent, finer parvocellular one, separated before cortex and later recombined (Livingstone & Hubel, 1987; Kaplan & Shapley, 1982; Derrington et al., 1984; Nassi & Callaway, 2009; Ferrera et al., 1994), and that input is not stationary over development, so the system first learns from a low-passed world. The two-stream hypothesis (Mishkin et al., 1983; Goodale & Milner, 1992) and the emergence of ventral- and dorsal-like streams in a two-pathway network trained on one objective (Bakhtiari et al., 2021) motivate an architecture whose streams are defined by their subcortical-like inputs rather than by their cortical targets.

This paper asks two questions, in the order the originating study posed them (Manocchio, 2026): can two streams fed magnocellular- and parvocellular-like inputs diverge while optimising a single objective and producing shape-biased behaviour, and does such a system explain how the ventral stream can be texture-like while behaviour is shape-biased? We build it — two ResNet-18 trunks receiving the luminance and L−M channels of a cone-opponent image, with a developmental diet on the luminance channel (Fig. 1) — and examine it with linear probes and cue-conflict behaviour on the streams (Fig. 2), voxel-wise encoding models of eight subjects' 7T fMRI responses (Fig. 3) and the behavioural readout that combines the streams (Fig. 4). The streams diverge by different routes; the texture-biased one carries extrastriate cortex; and the integrated behaviour is shape-biased because a behavioural rate is a property of a readout over components, not a report of any one of them — a property that holds, in kind, in networks with no stream split at all (Fig. 6). Because the account predicts that properties bundled with shape bias need not travel with it, we test that too: robustness, agreement with human decisions and representational dimensionality all come apart from the rate (Fig. 5).

Relative to the originating thesis, every condition is trained with three seeds and every claim states its replication unit; the representational-similarity analysis that had found no stream-to-brain mapping is replaced by variance partitioning of encoding models, which removes the global stream preference that had defeated it; and the stream weighting the thesis fixed by hand is swept, so that the dependence of behaviour on the readout is measured rather than assumed.

## 2. Results

### 2.1 The model, and what the diet costs

The network splits its input once, before any convolution (Fig. 1a): each image is converted to the DKL cone-opponent space, stream 1 receives the luminance channel and stream 2 the L−M channel, and each is a ResNet-18 trunk. Under the developmental visual diet the luminance channel passes through a Gaussian blur (σ = 4 px) and the scale-free contrast-sensitivity cut of Lu et al. (2026) at a fixed age of 48 months, separating the transform from the trajectory (Methods 4.1). The trunks are initialised bit-identically — a control in which both streams receive the same image leaves all 120 tensor pairs identical after 150 epochs — so every stream difference below is attributable to the inputs, and a linear classifier reads out w·y₁ + (1−w)·y₂ of the two streams' penultimate features, with w = 0.5 during training.

Four front-end conditions are used throughout (Fig. 1a): the diet; the no-diet control; a channel-swap control applying the identical transform to the L−M channel instead; and a `parvo_highsf` control in which stream 2 additionally receives untransformed luminance, because real parvocellular cells carry both. Networks were trained on MiniEcoset (112 classes; Thorat & Kietzmann, 2024) for 150 epochs, three seeds per condition (Methods 4.2).

The diet costs accuracy (Fig. 1c): the no-diet conditions converge to 0.70 validation top-1, the diet arm reaches 0.556 at its best saved checkpoint against 0.701 and, uniquely, degrades after its peak, ending at 0.445 (range 0.347–0.527), so behavioural and out-of-distribution results use each seed's best saved checkpoint, stated wherever the policy matters (Methods 4.7). Two single-stream controls trained with the identical diet — a ResNet-18 (11.2M parameters) and a ResNet-34 parameter-matched to the two-stream model (21.3M against 22.4M) — reach 0.599 and 0.623, so the accuracy cost of the split is not a capacity artefact (Fig. 4c).

### 2.2 The two streams specialise in opposite directions, by different routes

Read out separately with a linear probe on the cue-conflict images, the two streams of the diet-trained network end on opposite sides of the shape–texture boundary: shape − texture accuracy +0.819 for the magno stream and −0.518 for the parvo stream, means of three seeds (Fig. 2a). Behaviour agrees: read out alone at the final checkpoint, the policy of the cortical fits, the magno stream's cue-conflict shape bias is 0.878 and the parvo stream's 0.418, against 0.675 for the integrated readout (Table 1; best-checkpoint values in Extended Data Table 10) — the divergence the thesis reported on a single model per condition, now replicated across seeds.

**Table 1. The two streams end on opposite sides of the boundary, by different routes.** Final checkpoint of each seed, the policy of the cortical fits (Methods 4.7); cue-conflict values at the best saved checkpoints are in Extended Data Table 10.

| | magno stream (luminance + transform) | parvo stream (L−M) | integrated |
|---|---|---|---|
| linear probe, shape − texture (0 = sign flip), 3 seeds | **+0.819** | **−0.518** | — |
| cue-conflict shape bias (0.5 = parity), 3 seeds | **0.878** | **0.418** | **0.675** |
| linear probe, untrained, seed 0 | **+0.796** | −0.087 | — |
| linear probe, trained, seed 0 | **+0.810** | **−0.482** | — |
| *contribution of learning* (trained, 3 seeds, minus untrained, seed 0) | **+0.024** | **−0.432** | — |

The decomposition arises asymmetrically (Fig. 2a, Table 1). Scored on the untrained network with the same front end, the magno stream's shape bias is essentially complete before any weight is updated — the low-pass and contrast-sensitivity transform does it — and training adds +0.024 (3-seed mean; +0.014 in seed 0); the parvo stream begins barely biased and acquires its texture preference over training. The shape-biased component is supplied by the input, the texture-biased component built by learning, and what ordinary training does to the input's contribution is the decisive contrast: with the transform removed, the untrained luminance stream starts lower (+0.417) and training drives it across the sign flip to −0.226 — it does not merely fail to become shape-biased, it becomes texture-biased — while the parvo stream ends texture-biased whether or not the other stream is dieted (−0.518 against −0.504).

Which property of the input supplies the shape bias? Crossing the carrier a stream receives (achromatic luminance or chromatic L−M) with whether the developmental transform is applied to it gives four cells, two of which the swap and `parvo_highsf` conditions provide (Fig. 2b, c): on the probe the chromatic carrier gives −0.504 without the transform and −0.145 with it, the achromatic carrier −0.266 without and +0.819 with, so each single factor moves a stream toward neutral and only the conjunction crosses zero, with an interaction of +0.73 over the additive prediction of the other three cells (+0.09) and no seed overlap between the conjunction cell and any other (seed ranges 0.014–0.043; cue-conflict shape bias on the same four cells replicates the interaction, Extended Data Table 4). The shape bias requires the developmental transform applied specifically to an achromatic channel — a relational fact about what a stream receives, not a property of a stream called “magno”.

Two qualifications belong here. First, cue-conflict shape bias is computed only over images a readout decides as the shape or the texture category, a count that varies four-fold across cells (165 to 584 of 1,200; Fig. 2d), so the three non-conjunction behavioural cells are read as “no strong preference”; the probe, which classifies every image, gives the same pattern, and so does a forced two-way estimator over every image, which compresses values toward 0.5 (the four cells read 0.44, 0.51, 0.35 and 0.76) while preserving every ordering this paper draws a conclusion from, the two no-preference cells trading places (Extended Data Table 10). Second, the diet-trained models are less accurate: we pre-registered the fatal outcome — a no-diet model at the diet's accuracy with a shape-biased magno stream — and retrained the no-diet network with per-epoch checkpointing, selecting by rule, in each of three seeds, the epoch nearest the diet's accuracy (0.546–0.554 against 0.556); matching accuracy closes 7.2% of the probe gap (5.8–8.1% across seeds) and 6.1% of the behavioural gap (2.2–9.1%), and the matched magno stream stays on the texture side in every seed (probe −0.14 to −0.17; Fig. 2e, f).

### 2.3 In cortex, the texture-biased stream carries the ventral end of the hierarchy

The thesis had asked whether the streams align with different cortical streams by representational similarity analysis (Kriegeskorte et al., 2008) against seven regions of the Natural Scenes Dataset, and found that whichever stream was the more predictive correlated better with every region — a general factor that a correlation cannot separate from regional structure. We replaced the instrument with voxel-wise banded-ridge encoding models for each of the eight NSD subjects, one band per stream, partitioning every voxel's cross-validated explained variance into the part only the magno band accounts for (U_m) and the part only the parvo band does (U_p) (Methods 4.5); the per-voxel index (U_m − U_p)/(U_m + U_p), median over reliable voxels per ROI, across V1v → hV4 is Fig. 3a.

The two trained models sit on opposite sides of the global stream-preference axis — without the diet every retinotopic area is magno-explained, with it every area is parvo-explained — and that global difference, the general factor that defeated the correlational analysis, is not the result. The result is the shape of the profile: only the diet arm descends steeply from V1 to hV4, early visual cortex relatively more magno-explained and hV4 relatively more parvo-explained, the direction a luminance/low-spatial-frequency versus chromatic split predicts, and the descent belongs to the model whose global preference points the other way, which a model simply preferring one stream everywhere could not produce.

Quantified as the contrast hV4 − mean(V1v, V1d) of the per-ROI median (Fig. 3b; Extended Data Table 1), the gradient is −0.745 to −0.834 across the three diet-trained seeds against −0.049 to −0.096 without the diet, with the same separation on the raw statistic U_m − U_p, and all six comparisons reach the exact Wilcoxon floor for eight subjects (p = .0078) with complete seed separation. The per-subject Spearman ρ between hierarchical rank and stream preference is −0.964 to −0.973 with the diet and −0.46 to −0.66 without (Fig. 3c; Extended Data Table 2); the rank vector [1, 1, 2, 2, 3, 3, 4] admits exactly 630 labellings of the seven ROIs, and against that enumerable null — which calibrates the extremity of a result that was not pre-specified rather than testing a prediction — the diet arm's ρ is the single most extreme in every seed and on both statistics (p = 1/630 = .0016) while the no-diet arm is not (p = .019 to .092), so we claim nothing there.

The diet does not create the gradient; it decides what training does to the one the input supplies. An untrained network on the diet front end already sits at −0.544 (8/8 subjects, p = .0078), and on the no-diet front end at almost the same value (−0.479), so the two front ends alone impose no different gradient; training with the diet steepens it and training without it erodes it to near zero (Fig. 3b). Correcting each arm for the signed bias of the estimator itself — a simulated floor of the wrong sign, +0.04 with the diet and +0.05 without — leaves a paired displacement of −0.871 with the diet against −0.130 without, each non-zero in every subject, and an independently trained checkpoint from the thesis's own pipeline, with its progressive developmental schedule, gives −0.793.

Pushed through models fitted on natural scenes only, the cue-conflict images yield a per-voxel predicted shape-bias index whose magno − parvo difference is +1.53 to +1.57 with the diet against +0.08 to +0.11 without (8/8 subjects, p = .0078 in every seed; Fig. 3d; Extended Data Table 3) — a test that needs neither probe nor readout.

**The gradient survives its nuisance axes and the instrument's own bias.** Receptive-field size, eccentricity coverage and the noise ceiling all change from V1 to hV4, and the diet's magno stream is a low-pass instrument by construction; yet the diet gradient is negative within every one of NSD's five eccentricity bands and every pRF-size tertile while the no-diet arm stays near zero in each, it holds on continuous pRF eccentricity and size within subject, and the estimator's own simulated bias is small, flat across ROIs and of the wrong sign (Extended Data Fig. 3; Extended Data Tables 8 and 9; Supplementary Note S3).

**The gradient does not stop at hV4, and it does not turn.** Relabelling the stored fits with NSD's full atlases (Extended Data Fig. 4; Extended Data Table 11), the diet index runs from −0.05 in V1v through −0.81 in hV4 and then sits between −0.81 and −0.94 in every extrastriate region — ventral, lateral and dorsal alike, in the Kastner2015 atlas, NSD's stream parcels and the fLoc category regions — with every region-minus-V1 contrast negative in 8/8 subjects (p = .0078) and no region-minus-region contrast departing from zero under either arm (all p > .05). Beyond V3 the diet's magno band contributes about 0.01 unique variance while the parvo band's rises from 0.10 in V1v to 0.26 in PHC2 (Extended Data Fig. 4b): the texture-biased stream carries extrastriate cortex as a whole, and the thesis's magno→dorsal, parvo→ventral mapping fails for want of any magno variance to map.

Three things this section does not settle. The diet predicts BOLD worse, not better (median r² normalised by the noise ceiling 0.405 against 0.464 without the diet, in 8/8 subjects; Fig. 3e), so its case rests on stream differentiation and the gradient, not on prediction accuracy. The two front-end controls are degenerate on this instrument (swap: U_p = 0.006; `parvo_highsf`: U_m = 0.007; one network each, seed 0, Methods 4.5), so only the diet on luminance yields two streams that both carry substantial unique cortical variance and a gradient between them (Fig. 3a). And the gradient was not pre-specified: it is the survivor of seven regional hypotheses, six of which failed — among them the thesis's magno→dorsal mapping, an hMT+ specificity and a foveal/peripheral contrast whose sign flipped when the blur was removed — reported because it alone survives removal of that manipulation, and read as sign and ordering rather than as a calibrated effect (Supplementary Notes S1 and S3; Extended Data Table 1 for the subject-01 influence check).

### 2.4 Behaviour reads out from both streams, and the rate is a property of the readout

The two streams are oppositely biased and the texture-biased one carries extrastriate cortex; what remains is what the behavioural readout does with them. The integrated readout's cue-conflict shape bias is 0.675 ± 0.025 with the diet against 0.416 ± 0.020 without (mean ± s.d. over three seeds, final checkpoints, as in Table 1; the thesis's checkpoint reproduces at 0.697 against its reported 0.677). On the identical images and statistic the ten human observers give 0.958 ± 0.019 (range 0.919–0.987) and the Geirhos et al. (2018a) CNNs 0.429, 0.312 and 0.172, values our pipeline reproduces. The diet-trained magno stream, at 0.878, closes most of the distance from the CNN regime toward the observers without matching the lowest of them, and integration costs 0.203 of shape bias at this checkpoint policy (0.878 → 0.675; 0.248 at the best saved checkpoints; Fig. 4c, Extended Data Tables 5 and 10).

The integrated readout lands between its two components — an outcome two oppositely biased components nearly force, so not a prediction — and what can be measured is how much of the rate the readout controls. Sweeping w while holding y₁ and y₂ exactly fixed (Fig. 4a) runs cue-conflict shape bias from 0.40 to 0.90 over 21 values of w, three seeds and both checkpoint policies; w = 0.5 reproduces each trained model exactly, and the two streams alone sit at 0.41 and 0.78 on the forced two-way estimator, so the span is not an artefact of the decided-image denominator (Extended Data Table 10). Because the affine sweep is exact interpolation between two measured endpoints, under a rule fixed before the test we refitted a two-layer nonlinear head, and a linear head as a control for the refit itself, on the frozen features at each w (Fig. 4b): the nonlinear head's median span over the three best-checkpoint seeds is 0.466 against 0.508 for the frozen affine head (92% of the pre-registered baseline, 86% in the weakest seed, threshold 80%), the refitted-linear and nonlinear medians (0.461 and 0.466) agree within 0.01, and restricted to readouts within 5 points of peak validation accuracy the nonlinear span is 0.283 (0.26–0.31 across seeds): the span is a property of two oppositely biased components under any readout, and the extreme rates near w = 1 are bought with accuracy.

On the ladder of Fig. 4c the two streams alone bracket the integrated readout (0.903 and 0.400 around 0.655) and the single-stream controls trained with the identical diet lie between (0.567 and 0.625), inside the 0.058 that capacity alone moves the rate, so the ordering among these three models is attributed to capacity and no shape-bias advantage is claimed for the split.

Nothing here fits w to humans, and it could not be: fitted to each of the ten observers individually on a 0.001 grid, no observer's rate is reachable in any run (0 of 90 observer-runs, the closest approach missing the lowest observer by 0.027), and a weight fitted to an observer's trial-by-trial choices does not cross-validate (held-out agreement +0.026 against +0.023 for an arbitrary weight, better in 45 of 90). The rate is a free parameter of the readout, and nothing in this family of readouts selects the human operating point.

**Cortex and behaviour prefer different readout weights.** The readout's input z(w) = w·y₁ + (1 − w)·y₂ is a feature vector like any other, so for each w it was the design of a ridge encoding model of every voxel in NSD's general visual mask, and the w that maximises an ROI's median r² is the weight that region prefers (Methods 4.5; Extended Data Fig. 5; Extended Data Table 12). Under the diet that weight descends the hierarchy — 0.58 in V1 to 0.33–0.35 in every extrastriate group, ventral, lateral and dorsal alike, below V1 in 8/8 subjects (Δ = −0.25, exact p = .0078) — whereas the weight that best reproduces the ten observers' trial-by-trial choices is 0.67 (range 0.49–0.89), and the two sets of optima do not overlap (extrastriate 0.27–0.40; Mann–Whitney p = .0004). On the behavioural sweep the two demands sit on opposite sides of the rate, 0.56 at the weight cortex prefers and 0.78 at the weight behaviour prefers, and without the diet the gradient reverses (0.65 in V1 to 0.75–1.00 beyond hV4). The curves are shallow (every ROI within 3% of its best r² at w = 0.5 with the diet and within 6% without), so this is an ordering of optima rather than a large loss — the dissociation of Fig. 3 in the readout's own currency.

**Without a stream split, one gain on a network's own shape-predictive subspace moves the rate across the field's range.** The span of Fig. 4a is given by construction in a network built around two streams, so we tested networks with none, under plans committed before each analysis was run (Methods 4.8). In any frozen network the penultimate features hold a shape-predictive and a texture-predictive subspace — the fifteen discriminant directions of a 16-class shape and of a 16-class texture linear discriminant, made mutually orthogonal and fitted on one half of the cue-conflict images — and a gain of 2w on the first and 2(1 − w) on the second, everything else including the network's own head untouched, is the analogue of the mixing weight; the rate is scored on the other half of the images and every w is gated on top-1 accuracy on held-out natural images (Fig. 6). In the six ImageNet networks of Fig. 6 the rate rises monotonically with w (ρ ≥ 0.997) by 0.25–0.33 with top-1 within 1.5 points of the trained value over the whole sweep, meeting the pre-stated threshold (≥ 0.20 in at least four of the six) in all six; the spans are half to two thirds of the two-stream span and as large as the range the six trained networks occupy (0.11–0.40), so within the gate each network can be moved past the trained rate of every other (VGG-16 past four of the five) and the six placed in the reverse of their trained order (Fig. 6a, b; Extended Data Table 13). Across the field's zoo the result holds and the ranking does not: in 38 torchvision ImageNet classifiers the whole sweep stays within the five-point gate (top-1 changes by at most 4.4 points), the reachable span is 0.20–0.50 and unrelated to the trained rate (ρ = +0.04), the rankings by trained rate and by the rate at the shape end of the gate agree at Kendall τ = 0.47 with 27% of the 703 pairs reversing, and the ranking by trained rate agrees with that by a linear probe's shape-minus-texture accuracy on the same features only at τ = 0.44 (Extended Data Fig. 6; Extended Data Table 16). Error consistency with the ten observers (Results 2.5) is 2–23% of the human–human ceiling at the trained readouts (median 8%); it rises with the trained rate across networks (ρ = 0.80) and, within every network, from the texture end to the shape end of its gate (38 of 38; median +14 points of the ceiling), yet no gated readout of an ImageNet-trained network exceeds 37% of the ceiling. The informative exception is a ResNet-50 trained on Stylized-ImageNet (Geirhos et al., 2019): the highest trained rate of the 42 networks tested (0.81), the smallest span (0.17), a probe bias of −0.11 against −0.30 to −0.66 for every ImageNet-trained network, and the highest agreement with the observers (40% of the ceiling at its trained readout, 53% at its shape end) — its bias sits in the representation, which is why the readout moves it least and which the rate alone could not have shown (pre-registered as P-Z1–P-Z4; the Stylized-ImageNet span fell short of its 0.20 threshold and the other three predictions were met). Our own single trunks move by 0.23–0.34 ungated but lose 7–15 points of MiniEcoset top-1 at the ends of the sweep (seed means; up to 12–17 points in single seeds, Extended Data Table 13), 0.14–0.22 within the gate, and a head refitted at each w on the lossy sum of the two subspaces, the residual passed unchanged, keeps validation accuracy within 1.2 points of its ceiling in every seed (0.5–0.8 points as family means) while the rate still moves by 0.10–0.15 (two-layer head; 0.08–0.13 linear). The information the gain uses is distributed — the two subspaces hold 4–15% (shape) and 20–33% (texture) of the feature variance — and the pre-specified unit-level rule, a median split of the units by their shape-minus-texture η², reaches gated spans of only 0.07–0.16 (Extended Data Table 14); on the two-stream network the data-driven subspaces are not the streams (63–70% of the shape basis's energy on stream 1, against the pre-stated 90%) and the subspace gain moves the rate by 0.32 where the streams themselves give 0.46 and the readout weight of Fig. 4a 0.50, so the architectural split remains that network's most effective factorisation.

**Cortex loads on the texture-leaning units, not on the discriminant subspaces.** With the two unit halves of the no-diet single trunk as encoding bands, the texture-leaning half carries every extrastriate group relative to V1 in 8/8 subjects (Δ = −0.32, exact p = .0078), and the cortex-optimal weight of the lossy sum falls from 0.80 in V1 to 0.37–0.48 beyond hV4 (7/8, p = .023) while the behavioural rate rises toward w = 1 (Fig. 6c, d; Extended Data Table 15); the diet-trained single ResNet-34 shows the same contrast (7/8, p = .016), and in ImageNet ResNet-50, whose units are almost all texture-leaning, that half carries V1 through hV4 and ventral cortex alike, so the V1-relative prediction fails there (extrastriate − V1 = +0.24, p = .039). These unique variances are small (≤ 0.04 for the texture-leaning half against 0.10–0.25 for the parvo band with the split, 80–96% of the explained variance being shared between the halves against 45–63% between the streams), and under the subspace rule they all but vanish: entered as three bands with the residual, the two 15-dimensional subspaces each add at most 1.3% of noise-ceiling-normalised unique variance in any region against 23–63% carried by the shared and residual variance, their index sits at noise level, and the pre-registered cortical predictions fail — the extrastriate − V1 contrast is negative in 2/8, 2/8 and 2/7 subjects in the no-diet trunk, the diet ResNet-34 and ResNet-50, in 7/8 in ConvNeXt-B (Δ = −0.44, p = .016) and in 3/8 in ViT-B/16, and the 15-dimensional lossy sum's cortex-optimal weight is lower beyond V1 in no family at the pre-registered 7/8. The directions that move the rate are nearly silent as unique cortical predictors — which component cortex loads on is decided by how the representation is partitioned — so the cortical claim is made under the unit rule only. The architecture is therefore not what makes the rate a readout property, nor, in the networks trained on our images, what makes extrastriate cortex load on the texture-biased component; it is what makes the components separable enough for one weight to move the rate across the whole range models occupy at no cost in accuracy, and for cortex to be carried almost entirely by one of them.

This is the mechanism we propose for the paradox: a system holding a shape-biased and a texture-biased component, whose behaviour reads out from both, will look shape-biased behaviourally and texture-biased wherever the texture-biased component dominates — and in this model of the subcortical split, the component that dominates the ventral end is the texture-biased one. It is an existence proof with a cortically grounded demonstration, not a measurement of the human readout.

### 2.5 Properties that do not follow from shape bias

If shape bias is a property of a readout over components rather than of a representation, quantities the field treats as travelling with it need not. Over identical inputs, three do not (Fig. 5).

**Robustness.** The standard reading of Geirhos et al. (2018a, 2021) is that shape bias buys out-of-distribution robustness. We pre-registered that prediction and tested it on corruptions applied before the front end, scored in the models' own label space at each seed's best saved checkpoint (Fig. 5a; Extended Data Fig. 1; Extended Data Table 6). The diet-trained model is less robust on four of the five corruption families it was not trained on — uniform noise, high-pass, false colour and rotation (relative-accuracy AUC 0.634 against 0.729, 0.678 against 0.827, 0.854 against 0.938 and 0.574 against 0.615), with complete seed separation on each — equivalent within ±0.033 on phase scrambling, more robust only to low-pass (0.444 against 0.356), declared trained-in before the run because the diet is a low-pass blur, and not to contrast reduction (0.481 against 0.549) although its front end includes a contrast-sensitivity cut. Robustness appears exactly where it was trained in and nowhere else, the strongest available evidence that the nulls are real rather than a broken measurement. This does not contradict Geirhos et al., whose stylisation route yields shape bias together with robustness — it shows the coupling belongs to that route — and it differs from the source of the diet, whose title reports robust, shape-based vision: we reproduce the shape-based half (their released model 0.822, our diet-trained magno stream 0.878) and not the robust half, and Methods 4.6 lists three implementation differences any of which could carry the divergence.

**Agreement with human decisions.** Two decision-makers with the same marginal rate can agree on no individual image beyond what those rates force; error consistency — Cohen's κ on trial-by-trial agreement — subtracts that forced agreement (Geirhos et al., 2020), and across the ten observers the human–human ceiling is κ = 0.234. We pre-registered that the diet-trained model would be more human-consistent than its control; it is not, on either readout, and the sharpest true statement is stronger: our most shape-biased model, the diet-trained magno stream at 0.878, has, averaged over seeds, the lowest error consistency of the seven models compared (κ = 0.0082, 3.5% of the ceiling, seeds spanning −6.1% to 15.5%, against 7.3% for VGG-16 at shape bias 0.172; Extended Data Table 7); across all 189 readouts of the w sweep, with w chosen post hoc per run to maximise agreement, the best κ never exceeds 20.7% of the ceiling (Fig. 5b), and equivalence intervals fixed in advance exclude a κ change larger than 0.038 across the 0.71 span of shape bias the readouts cover; across the field's zoo the same measure does rise with the rate, to at most 37% of the ceiling (Results 2.4). The measure does detect a correspondence when one is there — the same machinery recovers 24.1% of the split-half ceiling on whether an image resolves to either cue at all, against 8.7% on which cue is picked (Fig. 5c) — a positive control that invites a diet reading we do not make, because the crossed front-end 2×2 shows a train/test-matching pattern rather than a weights effect (Fig. 5d).

**Representational dimensionality.** The participation ratio of each stream's features on 1,000 NSD images per subject gives a magno/parvo ratio whose ordering across the four front ends is reproduced by a learning-free Gabor bank, so it tracks the bandwidth of the carrier a stream is handed; what the diet decides is which side of parity training ends on — every diet-trained seed ends magno-dominant (24/24 subject-seeds in each of three diet arms), including the `parvo_highsf` front end, which starts inverted in 8/8 subjects and which we had pre-registered would stay inverted, while every no-diet seed collapses to parity (Extended Data Fig. 2). No level is claimed, because the within-arm seed range exceeds the gaps between arms; and `parvo_highsf`'s second stream becomes the more dimensional stream while remaining texture-biased (probe −0.266), which separates dimensionality from shape bias.

## 3. Discussion

**The two questions.** The thesis asked whether magnocellular- and parvocellular-driven streams inside one network could diverge while the network behaves shape-biased, and whether such a system bears on why the ventral stream can be texture-like while behaviour is not. To the first the answer is yes, replicated across seeds and with the mechanism located: divergence requires the developmental transform on an achromatic carrier, the shape-biased component is a property of the input that the diet protects from being unlearned, and the texture-biased component is built by learning. To the second the answer is an existence proof the thesis could not reach with its instruments: fitted to human cortex, the texture-biased stream's unique variance carries the extrastriate end of the hierarchy — ventral, lateral and dorsal alike — while behaviour, reading out from both, is shape-biased. Read as competing claims about a single representation the two cannot both be right; read as claims about a readout over components, they are not in competition.

**What the readout account licenses.** A model matching the human shape-bias rate is routinely described as having human-like representations; that inference fails twice over here. The same network holds components at +0.819 and −0.518 and behaves at 0.675, and one scalar over frozen representations moves the rate across most of the range the field's models occupy, so the rate does not identify a representation; and across readouts spanning a five-fold range of shape bias, trial-by-trial agreement with ten observers never exceeds 21% of the human–human ceiling in our networks and is lowest, 3.5%, for our most shape-biased stream; across the zoo the two do move together (ρ = 0.80 over 38 networks, every network's agreement rising from the texture to the shape end of its gate), yet no gated readout exceeds 37% of the ceiling, so the rate is at best a weak proxy for the decision process. Prior work established that a representation can hold more shape than its behaviour reveals (Hermann et al., 2020; Islam et al., 2021), that the measured rate depends on the test design (Tartaglini et al., 2022), that a prompt — a readout-level manipulation — steers it in vision–language models (Gavrikov et al., 2024), and that it tracks a network's spatial-frequency channel (Subramanian et al., 2023), the quantity the diet constrains; new here is the form of the dissociation — both representations held fixed, the rate moved by the readout alone, a refitted nonlinear head preserving the span — and its cortical grounding: the same mixing weight prefers the parvo-heavy end when fitted to every region beyond V1 and the magno-heavy end when fitted to human choices (Extended Data Fig. 5). Nor does the account need the split: in 38 frozen ImageNet networks with none, one gain on the network's own shape-predictive subspace moves the rate by 0.20–0.50 within five points of top-1, re-ordering a quarter of all network pairs, and in our single trunks the texture-leaning units again carry extrastriate cortex relative to V1 (Fig. 6); the split amplifies the dissociation (0.50 against 0.32 on the same network), it does not create it.

**How to report shape bias.** A shape-bias claim should carry three numbers, each answering a different question: the rate at the trained readout (an operating point); the span the same frozen features reach under a gain on their own shape-predictive subspace within an accuracy budget (how much of the rate the readout controls); and a probe's shape-minus-texture accuracy (what the representation holds). Across 38 ImageNet networks the three rank the zoo differently (τ = 0.47 and 0.44), and the Stylized-ImageNet network shows what their agreement looks like when a bias really is representational; error consistency belongs beside them wherever human trial data exist — across the zoo it rises with the rate but stays below 37% of the human–human ceiling for every ImageNet-trained readout (Extended Data Table 16) — and none of the four alone licenses a claim about representations. One command in the repository prints the first three for any torchvision network (Methods 4.8).

**What a developmental constraint does.** The intuitive account of low-pass developmental input is informational — coarse input shields the learner from texture, so it learns shape — and our results do not support it: the transform removes bandwidth rather than adding structure, the shape bias it produces is present before training, and what the diet decides is what learning does not destroy (without it, training drives the luminance stream from +0.42 to −0.23; with it, +0.80 is still +0.81 after 150 epochs, and a second instrument replicates the side of parity rather than a level). This is a constraint on the solution learning converges to, not a better input representation, consistent with the thesis's observation that the transformed magno stream becomes less predictive of the category as the parvo stream becomes more so.

**What comes apart, and what the diet costs.** Shape bias, robustness, dimensionality and human-likeness are discussed as a package; over identical inputs this system separates them three times, and a readout account says why it should: none of the three is a property of a readout. The diet is also expensive — less accurate (0.556 against 0.701), less stable late in training, less robust on four of five untrained-on corruptions, a worse predictor of BOLD — so nothing here recommends it as a training method; its interest is as a model of development, and the thesis's central mechanistic reading, that divergence and shape bias are bought by degrading the magno stream's evidence, survives in every instrument.

**Limitations.** The load-bearing cortical result rests on eight subjects from one dataset; its magnitude is read as sign and ordering only, and it is the survivor of seven regional hypotheses, disclosed above. Beyond hV4 the index saturates near its bound in every atlas, so the extension establishes that the effect is hierarchy-wide but cannot order extrastriate regions against one another. The mapping between our chromatic stream and texture-selective cortex is by explanatory role, not content: nothing here asserts that the ventral stream is chromatic, the prediction that the chromatic stream should win in colour-biased cortex fails (pooled VO +0.005, bounded within ±0.017), and substituting the koniocellular axis makes it worse (Supplementary Note S2). The `parvo_highsf` condition varies architecture alongside the front end. The diet is a fixed 48-month front end rather than a trajectory, though a schedule-trained checkpoint behaves like it on every instrument we tried. Whether a two-stream architecture specialises on its own is untestable here, because the trunks are bit-identical at initialisation. Human comparisons are against ten published observers and of aggregate rate; the readout weight is a hyperparameter that nothing selects. The single-trunk subspaces are linear discriminants of one cue-conflict set, and their spans are lower bounds on what a readout can reach, not measurements of the best one. The generality test was pre-registered three times and all three ledgers are complete (Methods 4.8; outcomes in the plan files): of the unit-level plan's five predictions, the behavioural thresholds were met by direction but not by magnitude, the stream-recovery check was met, the V1-relative cortical contrast held in our two trunks and reversed in ImageNet ResNet-50, and the mixing-weight ordering held in one network of three; of the subspace plan's, the magnitude threshold was met in all six ImageNet networks but in one of our three trunks, the stream-recovery check failed (63–70% of the shape basis on stream 1, against 90%), the subspace span exceeded the unit-level span in every network (Extended Data Tables 13–16), and of its three cortical predictions only the ConvNeXt-B contrast held, the two discriminant subspaces carrying almost no unique cortical variance; of the zoo plan's four predictions three were met and the Stylized-ImageNet span fell short of its threshold, read, as pre-stated, as a bias held in the representation rather than in the readout. Of the main study's four pre-registered predictions, two failed and are reported as failures; claims deliberately not made are listed in Supplementary Note S1.

## 4. Methods

### 4.1 Front end

RGB images are converted to the DKL cone-opponent space through RGB→XYZ (D65)→LMS (Hunt–Pointer–Estévez)→DKL at 256 × 256 px, each channel normalised to [0, 1] with the training set's global range and quantised to 8 bits exactly as the training data were stored (Derrington et al., 1984). Stream 1 receives the luminance channel; under the diet it passes through an isotropic Gaussian blur (σ = 4 px, kernel 33, reflect padding) and the scale-free contrast-sensitivity cut of Lu et al. (2026) evaluated at a fixed age of 48 months (`cs_logspan_start` = 5 × 10⁻³, decomposition mode “keep”). Stream 2 receives the L−M channel untransformed. The four conditions are: `dkl_bcs` (diet: luminance + blur + CS | L−M); `dkl_base` (no diet: luminance | L−M); `dkl_bcs_swap` (luminance | L−M + blur + CS); `dkl_bcs_parvo_highsf` (luminance + blur + CS | L−M stacked with untransformed luminance, two channels). The last differs architecturally in its first-layer fan-in, and that difference is measured rather than argued away (Extended Data Fig. 2; Results 2.5). Evaluation stimuli, including NSD images and cue-conflict images, pass through the identical front end.

### 4.2 Architecture and training

Two ResNet-18 trunks (He et al., 2016), one per stream, initialised bit-identically, with a joint linear classification head over the weighted sum of their penultimate features (w = 0.5; cross-stream mixing off). MiniEcoset at 256 px, 112 classes (Thorat & Kietzmann, 2024). 150 epochs, batch 256, Adam (lr 1 × 10⁻⁴, β₁ = 0.9; Kingma & Ba, 2017), cross-entropy, with the augmentation set of Lu et al. (2026) applied on GPU (horizontal flip, rotation ±15°, colour jitter, random equalise, perspective, sharpness; random grayscale for RGB inputs only). Three seeds per condition; weights saved every 25 epochs. Each run used one NVIDIA RTX A5000 (24 GB); the 21 training runs behind this paper took 578 GPU-hours in total (19.5–51.4 h each, from the runs' own timing logs). All encoding, statistical and plotting work ran on CPU. Over the whole project the scheduler allocated 29,000 core-hours and 605 GPU-hours to 1,222 jobs between 2026-08-25 and 2026-09-17: 4,700 core-hours and 584 GPU-hours to the 21 training runs, 20 GPU-hours to the accuracy-matched retraining, 14,800 core-hours to the two-band encoding fits and their controls, 4,600 to the subspace-rule cortex fits, 2,000 to the single-trunk generality analysis, 980 to the cortex-optimal readout weight, 55 to the model zoo and 1,800 to smoke tests, caches, figures, the robustness runs and the verifier (`tmp_analysis/compute_accounting.py` reads the scheduler's own accounting and writes `results/figures_v2/compute_accounting.json`). Estimated with the Green Algorithms method (Lannelongue et al., 2021) at 12 W per allocated core, 0.3725 W per allocated gigabyte, the A5000's 230 W, a power usage effectiveness of 1.67 and Germany's grid intensity of 0.35 kg CO₂e per kWh, that is 930 kWh and 325 kg CO₂e. It is an upper bound: allocated cores and memory are counted as fully used for the whole of each job's wall-clock time. Not included are the pretraining of the public networks analysed in Results 2.4 and the acquisition of the two human datasets. Single-stream controls use one ResNet-18 or one ResNet-34 trunk fed the diet's luminance input with the identical schedule. The accuracy-matched control retrained `dkl_base` for 20 epochs with per-epoch checkpointing at three seeds and selected by rule, per seed, the epoch whose validation top-1 is nearest the diet arm's best-checkpoint accuracy of 0.556 (epochs 10, 10 and 11: 0.548, 0.546 and 0.554). No hyperparameter search was run for any arm: every condition uses the training recipe of Lu et al. (2026) unchanged, and the only selection is the checkpoint policy stated above.

### 4.3 Behaviour and probe

Cue-conflict shape bias is computed on the Geirhos et al. (2018a) style-transfer set restricted to the 16 categories present in MiniEcoset and to images whose shape and texture categories differ (1,200 images). Logits are restricted to the 16 columns; the argmax is scored as a shape or texture response; images decided as neither are excluded from the denominator and their count is reported. Bootstrap 95% confidence intervals are over images. A second, forced two-way estimator keeps every image and scores whether the logit of the image's shape category exceeds that of its texture category; it is reported beside the primary estimator for every readout (Extended Data Table 10) and is computed from the same stored logits. The readout is full = classifier(w·y₁ + (1−w)·y₂), magno_only = classifier(w·y₁), parvo_only = classifier((1−w)·y₂), all through the same classifier and bias term; the w sweep evaluates full at 21 values of w on stored features. The refitted heads (Linear(512, 112) and Linear(512, 512)→ReLU→Linear(512, 112)) were trained on w·F₁ + (1−w)·F₂ of MiniEcoset training features at each w with an identical budget (Adam, lr 10⁻³, batch 1024, 20 epochs) and the same initialisation seed at every w. Human observers are the ten of the same release (1,280 trials each), scored by the identical rule; our recomputation reproduces the published 0.958 and the published AlexNet, GoogLeNet and VGG-16 values. Error consistency is Cohen's κ (Cohen, 1960) on the binary shape/texture choice over images both parties decided, with expected agreement p_m·p_h + (1−p_m)(1−p_h), reported as a percentage of the human–human κ over all 45 observer pairs (Geirhos et al., 2020). Per-image correspondence uses the observers' per-image shape rate and cue-resolution rate (split-half reliabilities r = .658 and .794 after category removal), the models' shape-minus-texture logit margin, w chosen on half the images and scored on the held-out half, as a percentage of each target's own ceiling. The linear probe reduces each stream's features to 256 PCA components and fits multinomial logistic regression (C = 1.0) under 5-fold stratified cross-validation to shape and texture labels separately; bias is shape accuracy minus texture accuracy on the same folds.

### 4.4 Effective dimensionality

Participation ratio (Σλ)²/Σλ² of the stream feature matrix on 1,000 NSD images per subject (Gao et al., 2017), computed Gram-side, reported as the magno/parvo ratio with parity at 1.0. Subjects are the replication unit within a seed; seeds are reported separately and never pooled. A learning-free reference is the same ratio computed on a Gabor oriented-energy bank applied to each stream's own transformed input.

### 4.5 Encoding

NSD (Allen et al., 2022) `betas_fithrf_GLMdenoise_RR` single-trial betas, converted to percent signal change and z-scored per session per voxel, `nsdgeneral` mask, eight subjects. Splits are by 73k image id with the shared-515 images held out. Stream features for each subject's stimuli are reduced by PCA to 1,000 components per stream (fit on training images only) and entered as two bands into a banded ridge regression (himalaya; Dupré la Tour et al., 2022) with 3-fold GroupKFold by image id, per-voxel alphas and per-voxel band weights. Unique variance U_m (U_p) is the drop in cross-validated r² when the magno (parvo) band is removed. Noise ceilings are NSD's `ncsnr`; voxels below 0.1 are excluded and any ROI cell with fewer than 20 surviving voxels is reported as missing. ROIs are the `prf-visualrois` labels of the Wang et al. (2015) atlas as distributed with NSD (V1v, V1d, V2v, V2d, V3v, V3d, hV4), with hierarchical rank [1, 1, 2, 2, 3, 3, 4]. The canonical-correlation spectrum between the two bands runs from ≈0 to 0.93, so each band retains directions the other cannot predict and unique variance is identifiable; a recovery simulation with the fitted feature geometry and a known across-ROI split returns the true ordering exactly (ρ = +1.000 in 8/8 subjects, both arms), which is why the gradient is reported as sign and ordering rather than as calibrated magnitude. The swap and `parvo_highsf` arms are fitted for all three seeds, as the diet and no-diet arms are, and their behavioural and probe measures use the same three seeds (Extended Data Table 4). Both are degenerate by construction and in every seed — the swapped network leaves the parvo band a median unique variance 0.005–0.010 of the magno band's, the parvo + high-SF network leaves the magno band 0.003–0.005 of the parvo band's — so they enter no cortical claim beyond the sign and size of their unique variances (Results 2.3; Extended Data Tables 1 and 11). The fits of this section were originally submitted at about 160 GB of memory per subject, which is not the size of the data: the peak is the group-ridge random search, which forms an (n_alphas × n_test × n_train) array per gamma sample and fold, about 30 GB in float64 with all 21 alphas at once. Batching that search seven alphas at a time changes no estimate — a refit of the diet seed-0 subject-01 fit reproduces the stored one at voxelwise r = 1.00000 on both unique variances, with the largest per-ROI index difference below 10⁻⁴ — and the same fits then peak at 32.6 GB (`slurm/encode_equiv_probe.sh`, `slurm/encode_cc_controls.sh`). Eccentricity bands are NSD's `prf-eccrois` labels (0–0.5°, 0.5–1°, 1–2°, 2–4°, > 4°); continuous pRF eccentricity, size and R² are NSD's population-receptive-field estimates in the same functional space, and the within-subject regressions of Extended Data Table 9 use z-scored log(eccentricity + 0.5) and log(size) over voxels with pRF R² > 5%. The per-voxel predicted shape-bias index is obtained by pushing the cue-conflict set through each fitted encoding model. For Extended Data Fig. 4 and Table 11 the same voxelwise fits were relabelled, without refitting, with the full Kastner2015 atlas as distributed with NSD (adding VO1/2, PHC1/2, LO1/2, TO1/2, V3A/B and IPS0–2; IPS3–5, SPL1 and FEF have no voxels inside `nsdgeneral`), with NSD's `streams` parcellation and with the fLoc face, place, body and word ROIs; regional means are unweighted means of the ROI medians, and region contrasts are tested per subject with the exact Wilcoxon test. For the cortex-optimal readout weight (Extended Data Fig. 5; Extended Data Table 12) the classifier's own input z(w) = w·y₁ + (1 − w)·y₂, where y₁ and y₂ are the 512-dimensional stream outputs the classifier mixes, was the design of a single-band ridge model of every `nsdgeneral` voxel for w ∈ {0, 0.05, …, 1}, with the same split, grouped folds and alpha grid as the banded model and columns z-scored on the training trials; because z(w) sums the streams dimension by dimension the ridge cannot undo the mixture, so w is a constraint on the design rather than a re-parameterisation of the two-band model. w*_cortex is the w that maximises the median test r² over an ROI's reliable voxels (seeds averaged per subject), the 1,024-dimensional concatenation [y₁, y₂] is the ceiling any mixture can reach, and the behaviour-optimal weight is the trial-fit w of Fig. 4 (Methods 4.4) for each of the ten observers. The koniocellular control substitutes S−(L+M) for L−M in stream 2 of the untrained seed-0 network, contrast-matched to L−M per image, with the magno band bit-identical across arms (gate max|Δr²_magno| < 10⁻⁴, passed at 0.000).

### 4.6 Out-of-distribution robustness

Corruptions are applied to RGB before the front end and scored in the native 112-class label space. Five families were declared untrained-on before the run (uniform noise, high-pass, phase scrambling, rotation, false colour) and two as trained-in positive controls (low-pass, contrast). Reported as the area under the relative-accuracy curve across severities, at each seed's best saved checkpoint. Three implementation differences from Lu et al. (2026) are declared: our front end is evaluated at a fixed 48 months rather than scheduled (the one we can rule out, since a schedule-trained arm behaves like the static diet); our diet is a transform on one stream's input of a two-stream network rather than an augmentation on a single trunk; and we train on MiniEcoset and score corruptions in that label space, applied before the front end.

### 4.7 Statistics

Within-subject comparisons use the paired exact Wilcoxon signed-rank test, n = 8 (exact two-sided floor p = .0078); seed-level comparisons use the exact Mann–Whitney U, 3 v 3 (floor p = 0.100, reached only on complete separation), and seeds are never pooled with subjects. Every cortical contrast is computed both on the raw per-voxel difference U_m − U_p and on the normalised index (U_m − U_p)/(U_m + U_p), each summarised by its median over reliable voxels per ROI, and a claim is made only where the two agree; disagreements are reported as unresolved. Equivalence uses two one-sided tests with bounds fixed before any interval was computed (Schuirmann, 1987; Lakens, 2017). No multiple-comparison correction is claimed: with 28 p-values and an exact floor of .0078, Holm–Bonferroni over any family larger than six cannot yield a rejection, so claims rest on complete separation, agreement of both statistics, replication across seeds and effect size (dz = −6.60 for the diet gradient against −1.09 without the diet). Behavioural and robustness results use each seed's best saved checkpoint unless stated; cortical fits use the final checkpoint. Four predictions — the front-end 2×2, the inverted-front-end prediction, the corruption families and the error-consistency hypothesis — were written into a version-controlled plan and committed before the corresponding analysis was run (internal pre-specification, not registry-based); two failed and are reported as failures.

### 4.8 Readout re-weighting in networks without a stream split

Pre-specified three times — in `docs/PLAN_GENERALITY_2026-09-14.md` (the unit-level rule; five directional predictions P1–P5), `docs/PLAN_SUBSPACE_2026-09-15.md` (the subspace rule; predictions P-A, P-A′, P-B, P-C and P-D) and `docs/PLAN_ZOO_2026-09-15.md` (the model zoo; P-Z1–P-Z4) — each committed before its analysis was run, with every outcome reported (Results 2.4; Extended Data Tables 13–16). Both rules act on a frozen network with logits = W·y + b, y its D-dimensional penultimate features, and neither retrains anything. Subspace rule: on a fitting set of cue-conflict images y is centred and reduced by PCA to min(256, D) components; a 16-class linear discriminant analysis on the shape labels (scikit-learn, eigen solver, Ledoit–Wolf shrinkage) gives 15 shape-discriminant directions and one on the texture labels 15 texture-discriminant directions; mapped back to unit space and orthonormalised, the two bases are made mutually orthogonal by symmetric (Löwdin) orthogonalisation, which treats shape and texture identically (sequential Gram–Schmidt in either order is the reported sensitivity; Extended Data Table 13), giving B_S, B_T and an orthonormal basis B_R of the residual, with projectors P_S, P_T and P_R. Fixed-head gain sweep: logits(w) = W·(P_R y + 2w·P_S y + 2(1 − w)·P_T y) + b, w ∈ {0, 0.05, …, 1}, so that w = 0.5 is the trained network exactly; the bases are fitted on one half of the 1,200 images, stratified by shape × texture pair, and the rate scored on the other, over 10 random splits in both directions. Unit-level rule, the special case in which the two subspaces are spanned by disjoint sets of units: each unit j receives d_j = η²_shape(j) − η²_texture(j), the one-way fraction of its variance over the images lying between the 16 shape or the 16 texture categories; the median split of d gives the shape-leaning half S and the texture-leaning half T (equal-sized, as the two streams are), and logits(w) = W·(g(w) ⊙ y) + b with g_j = 2w for j ∈ S and 2(1 − w) for j ∈ T, on the same image splits. At a referee's request, and not pre-specified, this rule was also run with S the absolute shape-leaning set (units with d_j > 0) or the top decile of units by d_j, T the remainder (Extended Data Table 14, last two columns; `slurm/generality_rules.sh`): both leave S small — 2–89 units under the absolute rule (a single unit, hence undefined, for ConvNeXt-S) and a tenth of the units (51–409) under the decile rule — so that w → 1 silences at least nine tenths of the representation; the ungated spans are 0.11–0.40 with positive mean ρ (0.29–0.99) but not every split is monotone in our trunks, and within 5 points of the trained top-1 — a gate that closes at w = 0.72–0.95 in every case but ConvNeXt-B under the decile rule, where the whole sweep passes — the reachable span is 0.02–0.21. Estimators: our networks by Methods 4.3; torchvision ImageNet networks (ResNet-50, ResNet-152, VGG-16, ViT-B/16, ConvNeXt-S/B, IMAGENET1K_V1 weights, model-vs-human preprocessing) by the Geirhos 16-class rule, mean softmax probability over each category's ImageNet classes and argmax over the 16, which assigns every image to one of the 16 categories; as in Methods 4.3, the rate is computed over the images assigned to their shape or their texture category (814 of 1,200 for ResNet-50; per-network counts in Extended Data Table 16). Accuracy gate: top-1 of the gained features, with the bases or the split defined on all 1,200 images, on held-out natural images — the MiniEcoset validation split for our networks and a stratified 10-per-class subset of the ILSVRC-2012 validation set for the ImageNet networks; the gated span is max − min of the seed-mean held-out curve over the w at which top-1 lies within 5 points of the trained value. Model zoo (Extended Data Fig. 6, Extended Data Table 16): the same pipeline on 42 networks — 38 torchvision `IMAGENET1K_V1` classifiers, ResNet-50 under torchvision's `IMAGENET1K_V2` recipe, and the three ResNet-50s of Geirhos et al. (2019) trained on Stylized-ImageNet, on Stylized-ImageNet plus ImageNet, and on both then fine-tuned on ImageNet — with y the input of each network's last linear layer, captured on the network's own forward pass and checked against its logits; Resize 256 / CenterCrop 224 except Inception-v3 (342/299), EfficientNet-B3 (320/300) and EfficientNet-V2-S (384/384), which were not trained at 224. The probe bias is the linear probe of Methods 4.3 on the same 1,200 features, predicting the shape and the texture class in turn. Ranking statistics over the 38 ImageNet-trained networks are Kendall τ between rankings and the fraction of the 703 pairs whose order reverses, with no p-values, the networks being a fixed set rather than a sample. `python -m src.analysis.generality.report --model tv:<arch>` prints the three numbers for any torchvision classifier. Refit: because a refitted linear head can undo a per-direction gain, the refit test mixes the two components lossily, as the two-stream model does, and passes the rest of the representation unchanged — z(w) = [w·s + (1 − w)·t, B_R^T^y] with s = B_S^T^y and t = B_T^T^y paired by discriminant rank (under the unit-level rule z(w) = w·y_S + (1 − w)·y_T, the halves paired by rank of d) — and refits the linear and two-layer heads of Methods 4.3 at each w, with a head refitted on the unmixed y as the accuracy ceiling. Cortex: under the unit-level rule S and T (split on all 1,200 images) are the two bands of the banded ridge model of Methods 4.5 on the penultimate units of the single-trunk networks, giving U_S, U_T and the index (U_S − U_T)/(U_S + U_T) per voxel; the cortex-optimal w of the lossy sum follows the protocol of Extended Data Fig. 5 on an 11-point grid (0, 0.1, …, 1). Under the subspace rule the bands are s = B_S^T^y, t = B_T^T^y and the residual r = B_R^T^y (15, 15 and D − 30 dimensions; bases fitted on all 1,200 images), with U_S = r²_{STR} − r²_{TR} and U_T = r²_{STR} − r²_{SR} from the three-band model and the two two-band models (random search with 30 and 20 samples), and the w sweep on the 15-dimensional lossy sum w·s + (1 − w)·t alone; the five families are `rgb_tsn` and `dkl_bcs_single34` (three seeds each, averaged per subject), ResNet-50, and ViT-B/16 and ConvNeXt-B, whose NSD features were extracted for this test (predictions P-C1–P-C3; Extended Data Table 15, upper block). The `rgb_tsn` network — two trunks that receive the identical three-channel image and remain bit-identical — is one ResNet-18 on the untransformed image and is used as the no-diet single trunk. All of our networks enter this analysis at their final checkpoints, the policy of the cortical fits (Methods 4.7).

### 4.9 Use of large language models

No large language model is part of the scientific pipeline: no result, number, figure or statistical decision in this
paper is produced by one. A large language model (Anthropic Claude, through the Claude Code command-line interface) was
used as a coding and drafting assistant under the authors' direction — writing and refactoring analysis, figure and job
scripts, drafting and revising manuscript text, and producing internal referee simulations
(`docs/NMI_REVIEW_*.md`) used only to prioritise revisions. Its use is recorded commit by commit in the repository's
`Co-Authored-By` trailers. Every pre-registration in `docs/PLAN_*.md` was written and committed before the corresponding
analysis ran, every reported number is regenerated from committed result files by the scripts named in each caption and
checked by `tmp_analysis/verify_new_claims.py`, and the authors take full responsibility for all content, including any
text or code an assistant drafted.

## Data availability

The Natural Scenes Dataset (Allen et al., 2022) is available at naturalscenesdataset.org under its own data-use agreement, which requires registration; we used the `betas_fithrf_GLMdenoise_RR` (b3) preparation, the `nsdgeneral` mask, the `prf-visualrois` and `prf-eccrois` labels and the pRF estimates as distributed with NSD. The Geirhos cue-conflict stimuli and the ten observers' trial-level responses are distributed at github.com/rgeirhos/texture-vs-shape (directory `raw-data/style-transfer-512-nomask-experiment/`) under CC BY 4.0; the same experiment is mirrored at github.com/bethgelab/model-vs-human, and the two mirrors were parsed by two independently written scripts that agree to four decimals on the human reference. MiniEcoset is at osf.io/msna2 (Thorat & Kietzmann, 2024). Derived result tables and fitted encoding-model summaries sufficient to reproduce every figure and Extended Data table are in the code repository, and will be deposited with a DOI on acceptance.

## Reporting summary

Further information on research design is available in the Nature Portfolio Reporting Summary linked to this article; a draft is `docs/REPORTING_SUMMARY.md` in the repository.

## Code availability

Training, front-end, encoding, behavioural and analysis code is at github.com/LuZeking/Magno-Parvo-Konio-Net (branch `mpk-rewrite`, commit as stamped on this manuscript's title page), under the BSD-3-Clause licence, with a pinned environment manifest (`requirements.txt`, Python 3.11) and the committed result files behind every figure; the figures of this paper regenerate with the scripts in `tmp_analysis/figs_v2/` and the manuscript's numbers, including those of Fig. 6 and Extended Data Tables 13–16, are locked by `tmp_analysis/verify_new_claims.py`, which reads the committed result files (for the generality analysis, `results/figures_v2/generality_behaviour.json` and `generality_cortex.json`). The per-subject encoding outputs that Fig. 3 reads are large and are deposited alongside the derived tables. The 21 trained networks (final and best saved checkpoints, ≈ 17 GB) that Figs. 4 and 6 and Extended Data Tables 10–15 read are not in the repository; they will be deposited in the same DOI archive at acceptance and are available to referees through the private link. A private link is available to referees on request; a DOI-archived release accompanies acceptance.

## Ethics and human participants

This study collected no new human data. Both human datasets are existing, de-identified public releases analysed under their own terms. NSD was collected under the approval of the University of Minnesota institutional review board with written informed consent from all eight participants (Allen et al., 2022); the behavioural data were collected under the approval described in Geirhos et al. (2018a). Secondary analysis of two de-identified public datasets required no additional approval under our institution's policy, and no participant-identifying information was accessed.

**Sex and demographics.** Participant sex was not under our control and is not a design factor; no analysis is disaggregated by sex, and at eight participants and ten observers none could be, because the exact tests used already sit at their resolution floor on the full samples. We do not restate sex or age: the Geirhos per-observer files carry only trial-level fields, so no per-observer demographic exists in the data we analysed, and for NSD we have not verified values against the released subject-information file and therefore point to the NSD release rather than transcribe from prose.

**Constraints on generality.** The neuroimaging sample is eight adults with normal or corrected-to-normal vision in the United States; the behavioural reference is ten adults tested in a German university laboratory. Both are small, Western, convenience-recruited adult samples. Claims about “human observers” should be read as claims about these samples; nothing here establishes cross-cultural, developmental or clinical generality, and the developmental motivation of the model's front end is a motivation for the model, not a claim about any developmental population we measured.

## Author contributions

[To be completed at submission per Nature-portfolio policy: conceptualisation, model and training code, encoding analyses, behavioural analyses, figures, writing, supervision. The originating thesis (Manocchio, 2026) supplied the architecture, the front-end conditions and the first behavioural and RSA analyses; the seed replications, encoding models, controls and manuscript were developed subsequently.]

## Competing interests

The authors declare no competing interests.

## Extended Data tables

<!-- ED_TABLES_START -->

**Extended Data Table 1 | The cortical gradient hV4 − mean(V1v, V1d), per seed and statistic.** “normalised” is the per-voxel index (U_m − U_p)/(U_m + U_p), “raw” the per-voxel difference U_m − U_p, each summarised by its median over reliable voxels per ROI. Mean over 8 subjects of the per-subject contrast; “sep.” is the number of subjects in which the diet arm is more negative than the no-diet arm; p is the exact paired Wilcoxon (floor .0078). The n = 7 column drops subject 01 (Results 2.3). Source: `results/encoding_cc/`.

| statistic | seed | diet | no diet (n = 8) | no diet (n = 7) | sep. | p |
|---|---|---|---|---|---|---|
| normalised | 0 | −0.833 | −0.078 | −0.054 | 8/8 | 0.0078 |
|  | 1 | −0.785 | −0.096 | −0.084 | 8/8 | 0.0078 |
|  | 2 | −0.745 | −0.049 | −0.031 | 8/8 | 0.0078 |
| raw | 0 | −0.166 | −0.027 | −0.021 | 8/8 | 0.0078 |
|  | 1 | −0.145 | −0.031 | −0.030 | 8/8 | 0.0078 |
|  | 2 | −0.153 | −0.023 | −0.021 | 8/8 | 0.0078 |
| untrained, diet front end | — | −0.544 | | | 8/8 < 0 | 0.0078 |
| untrained, no-diet front end | — | | −0.479 | | 8/8 < 0 | 0.0078 |
| swap, seed 0 (degenerate: U_p ≈ 0) | 0 | −0.006 | | | | |
| parvo_highsf, seed 0 (degenerate: U_m ≈ 0) | 0 | −0.143 | | | | |
| thesis checkpoint (progressive schedule) | — | −0.793 | | | | |

**Extended Data Table 2 | Monotonicity of the profile and the exact permutation null.** Spearman ρ between hierarchical rank [1,1,2,2,3,3,4] and stream preference, mean over subjects; the null enumerates all 630 distinct labellings of the seven ROIs (one-sided in the pre-stated direction, floor 1/630 = .0016). “per-voxel index” is the manuscript's statistic (Fig. 3c; Results 2.3): the median over reliable voxels of (U_m − U_p)/(U_m + U_p) per ROI, as in Extended Data Table 1; “ratio of medians” is median(U_m − U_p)/median(U_m + U_p) per ROI, the statistic of an earlier draft, kept for comparison. Source: `results/hierarchy_permutation_null.json` (which labels the two `raw` and `norm`).

| arm | seed | statistic | observed mean ρ | exact p | most extreme of 630 | null 2.5th pct |
|---|---|---|---|---|---|---|
| diet | 0 | per-voxel index | -0.964 | 0.0016 | yes | -0.743 |
| diet | 0 | ratio of medians | -0.964 | 0.0016 | yes | -0.754 |
| diet | 1 | per-voxel index | -0.973 | 0.0016 | yes | -0.743 |
| diet | 1 | ratio of medians | -0.973 | 0.0016 | yes | -0.750 |
| diet | 2 | per-voxel index | -0.966 | 0.0016 | yes | -0.756 |
| diet | 2 | ratio of medians | -0.966 | 0.0016 | yes | -0.748 |
| no diet | 0 | per-voxel index | -0.571 | 0.0206 | no | -0.559 |
| no diet | 0 | ratio of medians | -0.502 | 0.0286 | no | -0.502 |
| no diet | 1 | per-voxel index | -0.661 | 0.0190 | no | -0.594 |
| no diet | 1 | ratio of medians | -0.606 | 0.0190 | no | -0.546 |
| no diet | 2 | per-voxel index | -0.463 | 0.0921 | no | -0.594 |
| no diet | 2 | ratio of medians | -0.580 | 0.0365 | no | -0.616 |

**Extended Data Table 3 | In-silico cue conflict through the fitted encoding models, per seed.** Median over reliable voxels of the per-voxel predicted shape-bias index for each stream, mean over 8 subjects; the contrast is magno − parvo, paired within subject against the no-diet arm of the same seed. Source: `results/encoding_cc/`.

| seed | arm | magno stream | parvo stream | magno − parvo | diet − no diet | sep. | p |
|---|---|---|---|---|---|---|---|
| 0 | diet | +0.791 | −0.743 | +1.534 | +1.424 | 8/8 | 0.0078 |
| 0 | no diet | −0.638 | −0.748 | +0.110 | | | |
| 1 | diet | +0.796 | −0.770 | +1.566 | +1.468 | 8/8 | 0.0078 |
| 1 | no diet | −0.601 | −0.699 | +0.098 | | | |
| 2 | diet | +0.799 | −0.759 | +1.558 | +1.474 | 8/8 | 0.0078 |
| 2 | no diet | −0.613 | −0.696 | +0.083 | | | |

**Extended Data Table 4 | The carrier × transform 2×2 on the probe and on behaviour.** Mean over training seeds with the seed range in brackets; “decided” is the mean number of the 1,200 cue-conflict images the readout classifies as the shape or the texture category (the shape-bias denominator). Cells are read out on stream 2 in three cases and on the diet's magno stream in the fourth. Source: `results/figures/fig1_data.npz` (from `results/shape_texture/`, `results/cue_conflict/`).

| carrier | transform | condition, stream | probe bias | cue-conflict shape bias | decided / 1,200 | seeds |
|---|---|---|---|---|---|---|
| chromatic (L−M) | none | no diet, parvo | −0.504 [0.035] | 0.364 [0.080] | 270 | 3 |
| chromatic (L−M) | yes | swap, parvo | −0.145 [0.043] | 0.495 [0.038] | 165 | 3 |
| achromatic (luminance) | none | parvo_highsf, stream 2 | −0.266 [0.017] | 0.396 [0.012] | 584 | 2 |
| achromatic (luminance) | yes | diet, magno | +0.819 [0.014] | 0.878 [0.022] | 488 | 3 |

**Extended Data Table 5 | Single-stream controls trained with the identical diet, and the readout ladder.** Best saved checkpoint per seed, except the no-diet row, which is scored at the final checkpoint because its accuracy does not degrade (Fig. 2d; Methods 4.7); mean [range] over three seeds. Accuracy is validation top-1 at the scored checkpoint. Sources: `results/cue_conflict_bestep/`, `results/cue_conflict_single34_bestep/`, `results/cue_conflict_single_bestep/`, run histories, `results/param_matched/`.

| model / readout | parameters | shape bias | decided / 1,200 | accuracy |
|---|---|---|---|---|
| diet, magno stream alone | 22,397,936 | 0.903 [0.888, 0.916] | 504 | 0.556 [0.552, 0.564] |
| diet, both streams (w = 0.5) | 22,397,936 | 0.655 [0.594, 0.718] | 502 | 0.556 [0.552, 0.564] |
| diet, parvo stream alone | 22,397,936 | 0.400 [0.372, 0.429] | 388 | 0.556 [0.552, 0.564] |
| single stream, ResNet-34 (matched) | 21,338,992 | 0.625 [0.604, 0.652] | 537 | 0.623 [0.616, 0.634] |
| single stream, ResNet-18 (half) | 11,230,832 | 0.567 [0.541, 0.590] | 501 | 0.599 [0.589, 0.608] |
| no diet, both streams | 22,397,936 | 0.416 [0.394, 0.433] | 593 | 0.701 [0.698, 0.706] |

**Extended Data Table 6 | Out-of-distribution robustness: relative-accuracy AUC per corruption family and condition.** Mean over three seeds [range]; the thesis's schedule-trained checkpoint is a single run. “sep.” marks complete seed separation between the diet and no-diet arms (the exact 3 v 3 floor, p = 0.100). Families marked ‡ were declared trained-in before the run. Source: `results/ood_bestep/`.

| family | diet | no diet | swap | parvo_highsf | thesis schedule | diet − no diet | sep. |
|---|---|---|---|---|---|---|---|
| uniform-noise | 0.634 [0.619, 0.647] | 0.729 [0.720, 0.735] | 0.749 [0.744, 0.758] | 0.767 [0.758, 0.773] | 0.640 | −0.095 | yes |
| high-pass | 0.678 [0.647, 0.696] | 0.827 [0.820, 0.831] | 0.831 [0.817, 0.841] | 0.783 [0.775, 0.791] | 0.680 | −0.149 | yes |
| false-colour | 0.854 [0.845, 0.860] | 0.938 [0.933, 0.943] | 0.966 [0.960, 0.971] | 0.874 [0.868, 0.880] | 0.839 | −0.084 | yes |
| rotation | 0.574 [0.567, 0.582] | 0.615 [0.613, 0.616] | 0.605 [0.597, 0.612] | 0.611 [0.602, 0.616] | 0.572 | −0.041 | yes |
| phase-scrambling | 0.382 [0.365, 0.391] | 0.373 [0.369, 0.377] | 0.371 [0.365, 0.375] | 0.382 [0.380, 0.383] | 0.362 | +0.009 | no |
| contrast ‡ | 0.481 [0.460, 0.493] | 0.549 [0.530, 0.578] | 0.543 [0.539, 0.550] | 0.511 [0.498, 0.526] | 0.489 | −0.068 | yes |
| low-pass ‡ | 0.444 [0.434, 0.464] | 0.356 [0.353, 0.359] | 0.359 [0.353, 0.362] | 0.385 [0.375, 0.392] | 0.436 | +0.089 | yes |

**Extended Data Table 7 | Error consistency (Cohen's κ) with the ten human observers.** The trained readout (w = 0.5), the magno stream alone (w = 1; the κ = 0.0082, 3.5% of Results 2.5 is the mean of the three final-checkpoint runs) and the best readout over the 21 values of w, per run; κ as a percentage of the human–human ceiling (κ = 0.234, 45 pairs). Published CNN values are from Geirhos et al. (2020). Source: `results/sweep_error_consistency/`.

| run | shape bias at w = 0.5 | κ at w = 0.5 | % ceiling | κ at w = 1 (magno alone) | % ceiling | rate at w = 1 | best κ over w | % ceiling | w at best | rate at best |
|---|---|---|---|---|---|---|---|---|---|---|
| dkl_base_seed0_final | 0.421 | +0.0309 | 13.2 | +0.0404 | 17.3 | 0.438 | +0.0404 | 17.3 | 1.00 | 0.438 |
| dkl_base_seed1_final | 0.433 | +0.0333 | 14.3 | +0.0390 | 16.7 | 0.431 | +0.0421 | 18.0 | 0.75 | 0.429 |
| dkl_base_seed2_final | 0.394 | +0.0372 | 15.9 | +0.0349 | 15.0 | 0.400 | +0.0412 | 17.6 | 0.35 | 0.410 |
| dkl_bcs_seed0_best | 0.594 | +0.0096 | 4.1 | +0.0178 | 7.6 | 0.890 | +0.0178 | 7.6 | 1.00 | 0.890 |
| dkl_bcs_seed0_final | 0.692 | +0.0234 | 10.0 | +0.0361 | 15.5 | 0.872 | +0.0405 | 17.3 | 0.90 | 0.857 |
| dkl_bcs_seed1_best | 0.718 | +0.0269 | 11.5 | +0.0058 | 2.5 | 0.913 | +0.0425 | 18.2 | 0.35 | 0.614 |
| dkl_bcs_seed1_final | 0.686 | +0.0319 | 13.7 | +0.0028 | 1.2 | 0.894 | +0.0437 | 18.7 | 0.45 | 0.651 |
| dkl_bcs_seed2_best | 0.652 | +0.0372 | 15.9 | -0.0055 | -2.3 | 0.908 | +0.0483 | 20.7 | 0.85 | 0.847 |
| dkl_bcs_seed2_final | 0.646 | +0.0280 | 12.0 | -0.0143 | -6.1 | 0.871 | +0.0405 | 17.3 | 0.75 | 0.784 |
| AlexNet (published) | 0.429 | +0.0320 | 13.7 | | | | | | | |
| GoogLeNet (published) | 0.312 | +0.0250 | 10.7 | | | | | | | |
| VGG-16 (published) | 0.172 | +0.0170 | 7.3 | | | | | | | |

**Extended Data Table 8 | The estimator's own bias, and the residual-misfit variants.** Synthetic voxels were generated from each arm's fitted 1,000-component feature spaces with the measured per-ROI noise ceilings imposed and the true magno − parvo difference set to zero everywhere; the recovered contrast is the instrument's signed floor. “Displacement” is the real contrast minus the floor, paired per subject. The six variants differ in how the floor is paired (group or per-subject) and whether the residual misfit of the recovery simulation is also removed (before or after rescaling by the recovered slope). Sources: `results/vp_recovery_realnc/diet_floor_paired_2026-09-07.txt`, `floor_pairing_2026-09-07.txt`, `prediction_scored_2026-09-07.txt`.

| quantity | diet | no diet |
|---|---|---|
| floor, normalised contrast (true = 0) | +0.0375 (8/8 > 0, p = .0078) | +0.0511 (8/8 > 0, p = .0078) |
| floor, raw contrast | −0.0021 (p = .15) | +0.0009 (p = .84) |
| real contrast, normalised (seed 0) | −0.8335 | −0.0785 |
| displacement real − floor | −0.871 (8/8, p = .0078) | −0.130 (8/8, p = .0078); n = 7: −0.103 (7/7, p = .0156) |
| recovered slope of estimate vs truth | 1.210 | 1.281 |
| variant: group floor, minimal | −0.720 (8/8, p = .0078) | −0.101 (8/8, p = .0078) |
| variant: per-subject floor, minimal | −0.720 (8/8, p = .0078) | −0.101 (8/8, p = .0078) |
| variant: group floor + residual before scaling | −0.626 (8/8, p = .0078) | −0.034 (6/8, p = .055) |
| variant: per-subject floor + residual before scaling | −0.626 (8/8, p = .0078) | −0.034 (5/8, p = .31) |
| variant: group floor + residual after scaling | −0.607 (8/8, p = .0078) | −0.015 (4/8, p = 1.00) |
| variant: per-subject floor + residual after scaling | −0.607 (8/8, p = .0078) | −0.015 (5/8, p = .84) |
| per-ROI normalised floor, V1v … hV4 (no-diet geometry) | | +0.025, +0.025, +0.026, +0.025, +0.027, +0.026, +0.026 (ρ vs rank +0.54, p = .22) |

**Extended Data Table 9 | Stream preference regressed on pRF eccentricity and pRF size within subject.** Per-voxel stream-preference index regressed on log pRF eccentricity and log pRF size (NSD pRF estimates, voxels with pRF R² > 5%, noise ceiling > 0.1); “ecc | size” is the partial slope for eccentricity controlling for size and vice versa; mean over 8 subjects, with the number of subjects on the stated side and the exact Wilcoxon p. Collinearity between the two predictors is modest (mean VIF 1.66). The arm interaction is the paired difference of partial slopes, seed-averaged within arm. Sources: `results/prf_continuous/dissociation_2026-09-08.txt`, `interaction_2026-09-08.txt`, `input_null_2026-09-08.txt`.

| model | statistic | ecc alone | size alone | ecc \| size | size \| ecc |
|---|---|---|---|---|---|
| diet, seed 0 | normalised | +0.066 (0/8 neg, .0078) | −0.013 (5/8 neg, .31) | +0.119 (0/8 neg, .0078) | −0.086 (8/8 neg, .0078) |
| diet, seed 1 | normalised | +0.065 (0/8, .0078) | −0.029 (7/8, .039) | +0.132 (0/8, .0078) | −0.110 (8/8, .0078) |
| diet, seed 2 | normalised | +0.064 (0/8, .0078) | −0.012 (5/8, .25) | +0.115 (0/8, .0078) | −0.083 (8/8, .0078) |
| no diet, seed 0 | normalised | +0.018 (4/8, .64) | +0.033 (1/8, .039) | −0.004 (4/8, 1.00) | +0.035 (0/8, .0078) |
| no diet, seed 1 | normalised | +0.004 (4/8, .95) | +0.015 (4/8, .38) | −0.009 (6/8, .46) | +0.021 (2/8, .11) |
| no diet, seed 2 | normalised | −0.007 (5/8, .38) | +0.012 (3/8, .31) | −0.022 (6/8, .25) | +0.024 (1/8, .016) |
| diet, seed 0 | raw | +0.103 (1/8, .039) | −0.158 (8/8, .0078) | +0.325 (0/8, .0078) | −0.356 (8/8, .0078) |
| no diet, seed 0 | raw | +0.130 (1/8, .023) | +0.181 (0/8, .0078) | +0.028 (3/8, .84) | +0.163 (0/8, .0078) |
| Gabor bank, diet front end (input null) | normalised | | | +0.035 (6/8 pos, .15) | +0.002 (4/8 neg, .95) |
| untrained network, diet front end | normalised | | | +0.083 (7/8 pos, .016) | −0.038 (6/8 neg, .039) |
| arm interaction, diet − no diet (seed-averaged) | normalised | | | +0.134 (8/8, .0078) | −0.119 (0/8 pos, .0078) |
| arm interaction, diet − no diet (seed-averaged) | raw | | | +0.305 (8/8, .0078) | −0.502 (0/8 pos, .0078) |

**Extended Data Table 10 | Cue-conflict shape bias under the forced two-way estimator.** The primary estimator (Methods 4.3) restricts the logits to the 16 categories, takes the argmax, and scores shape against texture over the images decided as one of the two, so images decided as a third category drop out of the denominator (“decided”). The forced estimator keeps all 1,200 images and scores, per image, whether the logit of the shape category exceeds the logit of the texture category. Mean over seeds [range]. Source: the stored per-image logits in `results/cue_conflict*/`.

| model / readout | restricted (primary) | decided / 1,200 | forced two-way | seeds |
|---|---|---|---|---|
| diet, magno stream alone (best ckpt) | 0.903 [0.888, 0.916] | 504 | 0.775 [0.767, 0.788] | 3 |
| diet, both streams w = 0.5 (best ckpt) | 0.655 [0.594, 0.718] | 502 | 0.588 [0.574, 0.609] | 3 |
| diet, parvo stream alone (best ckpt) | 0.400 [0.372, 0.429] | 388 | 0.414 [0.394, 0.428] | 3 |
| diet, magno stream alone (final ckpt) | 0.878 [0.870, 0.892] | 488 | 0.761 [0.757, 0.764] | 3 |
| diet, both streams w = 0.5 (final ckpt) | 0.675 [0.646, 0.692] | 475 | 0.608 [0.596, 0.618] | 3 |
| diet, parvo stream alone (final ckpt) | 0.418 [0.406, 0.438] | 353 | 0.449 [0.411, 0.472] | 3 |
| no diet, both streams | 0.416 [0.394, 0.433] | 593 | 0.375 [0.357, 0.388] | 3 |
| no diet, magno stream alone | 0.421 [0.398, 0.434] | 602 | 0.378 [0.352, 0.398] | 3 |
| no diet, parvo stream alone | 0.364 [0.311, 0.391] | 270 | 0.442 [0.410, 0.473] | 3 |
| swap, both streams | 0.408 [0.403, 0.412] | 586 | 0.365 [0.362, 0.367] | 3 |
| swap, parvo stream alone (L−M + transform) | 0.495 [0.478, 0.516] | 165 | 0.511 [0.491, 0.531] | 3 |
| parvo_highsf, both streams | 0.486 [0.450, 0.516] | 618 | 0.450 [0.438, 0.457] | 3 |
| parvo_highsf, stream 2 alone (L−M + luminance) | 0.389 [0.374, 0.402] | 598 | 0.346 [0.324, 0.358] | 3 |
| single stream, ResNet-34 (matched) | 0.625 [0.604, 0.652] | 537 | 0.561 [0.552, 0.573] | 3 |
| single stream, ResNet-18 (half) | 0.567 [0.541, 0.590] | 501 | 0.523 [0.506, 0.538] | 3 |
| no diet, accuracy-matched (epochs 10, 10, 11), magno alone | 0.450 [0.432, 0.465] | 533 | 0.364 [0.357, 0.370] | 3 |
| thesis checkpoint (progressive schedule), both streams | 0.697 | 620 | 0.636 | 1 |
| untrained, diet front end, magno alone | 0.513 | 154 | 0.510 | 1 |
| untrained, no-diet front end, magno alone | 0.568 | 169 | 0.508 | 1 |

**Extended Data Table 11 | Stream preference beyond hV4, on every NSD atlas.** Median over reliable voxels of the per-voxel index (Um − Up)/(Um + Up) (noise ceiling > 0.1, Um + Up > 10⁻³, ≥ 20 voxels per ROI per subject), mean over the eight subjects (n = subjects with enough voxels); the no-diet and diet arms are the mean over three training seeds per subject, whereas the swap, parvo + high-SF and thesis columns are single networks (seed 0, or the thesis's one checkpoint; Methods 4.5). Positive = magno band, negative = parvo band. Atlases: Wang et al. (2015) Kastner2015 ROIs beyond hV4 (the V1–hV4 rows are Extended Data Table 1 and Fig. 3), NSD's `streams` atlas, and the fLoc category-selective ROIs. Regional contrasts and the exact tests are in Extended Data Fig. 4c and `results/figures_v2/ext_gradient.md`. Source: the stored voxelwise fits in `results/encoding_cc/` and `results/encoding/`, relabelled; no refitting.

| atlas | ROI | untrained (diet front end) | no diet (3 seeds) | diet (3 seeds) | swap (seed 0) | parvo + high-SF (seed 0) | thesis ckpt (n = 1) | voxels (median, diet) |
|---|---|---|---|---|---|---|---|---|
| Kastner2015 | VO1 | +0.00 (n=8) | +0.92 (n=8) | −0.87 (n=8) | +0.98 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 196 |
| Kastner2015 | VO2 | +0.12 (n=8) | +0.97 (n=8) | −0.90 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 220 |
| Kastner2015 | PHC1 | +0.41 (n=8) | +0.98 (n=8) | −0.89 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 148 |
| Kastner2015 | PHC2 | +0.46 (n=8) | +0.99 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −0.99 (n=8) | −0.93 (n=8) | 150 |
| Kastner2015 | LO1 | +0.13 (n=8) | +0.92 (n=8) | −0.83 (n=8) | +0.99 (n=8) | −0.98 (n=8) | −0.90 (n=8) | 230 |
| Kastner2015 | LO2 | −0.14 (n=8) | +0.92 (n=8) | −0.90 (n=8) | +0.99 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 166 |
| Kastner2015 | TO1 | +0.25 (n=8) | +0.96 (n=8) | −0.86 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.91 (n=8) | 280 |
| Kastner2015 | TO2 | +0.20 (n=8) | +0.96 (n=8) | −0.89 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.90 (n=8) | 36 |
| Kastner2015 | V3A | +0.39 (n=8) | +0.91 (n=8) | −0.81 (n=8) | +0.99 (n=8) | −0.98 (n=8) | −0.84 (n=8) | 278 |
| Kastner2015 | V3B | +0.24 (n=8) | +0.96 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 254 |
| Kastner2015 | IPS0 | +0.15 (n=8) | +0.98 (n=8) | −0.87 (n=8) | +1.01 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 317 |
| Kastner2015 | IPS1 | −0.06 (n=8) | +0.98 (n=8) | −0.86 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 239 |
| Kastner2015 | IPS2 | +0.02 (n=8) | +1.00 (n=8) | −0.92 (n=8) | +1.01 (n=8) | −1.01 (n=8) | −0.94 (n=8) | 47 |
| streams | early | +0.27 (n=8) | +0.86 (n=8) | −0.42 (n=8) | +0.96 (n=8) | −0.92 (n=8) | −0.45 (n=8) | 3133 |
| streams | midventral | −0.15 (n=8) | +0.87 (n=8) | −0.84 (n=8) | +0.97 (n=8) | −0.98 (n=8) | −0.86 (n=8) | 708 |
| streams | midlateral | −0.04 (n=8) | +0.91 (n=8) | −0.86 (n=8) | +0.98 (n=8) | −0.99 (n=8) | −0.90 (n=8) | 571 |
| streams | midparietal | +0.32 (n=8) | +0.94 (n=8) | −0.84 (n=8) | +1.00 (n=8) | −0.99 (n=8) | −0.91 (n=8) | 512 |
| streams | ventral | +0.08 (n=8) | +0.96 (n=8) | −0.91 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.95 (n=8) | 2724 |
| streams | lateral | +0.25 (n=8) | +0.97 (n=8) | −0.87 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.91 (n=8) | 1842 |
| streams | parietal | +0.12 (n=8) | +0.99 (n=8) | −0.87 (n=8) | +1.01 (n=8) | −1.00 (n=8) | −0.93 (n=8) | 1082 |
| floc-faces | OFA | −0.07 (n=8) | +0.90 (n=8) | −0.89 (n=8) | +0.98 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 367 |
| floc-faces | FFA-1 | +0.01 (n=8) | +0.95 (n=8) | −0.91 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 399 |
| floc-faces | FFA-2 | +0.22 (n=7) | +0.98 (n=7) | −0.92 (n=7) | +0.99 (n=7) | −1.00 (n=7) | −0.95 (n=7) | 260 |
| floc-places | OPA | +0.34 (n=8) | +0.98 (n=8) | −0.86 (n=8) | +1.01 (n=8) | −1.00 (n=8) | −0.93 (n=8) | 1070 |
| floc-places | PPA | +0.44 (n=8) | +0.99 (n=8) | −0.88 (n=8) | +1.00 (n=8) | −0.99 (n=8) | −0.94 (n=8) | 620 |
| floc-bodies | EBA | +0.17 (n=8) | +0.97 (n=8) | −0.88 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.92 (n=8) | 2174 |
| floc-bodies | FBA-1 | −0.07 (n=6) | +0.96 (n=6) | −0.91 (n=6) | +1.00 (n=6) | −1.00 (n=6) | −0.95 (n=6) | 314 |
| floc-bodies | FBA-2 | +0.13 (n=8) | +0.98 (n=8) | −0.92 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.94 (n=8) | 252 |
| floc-words | OWFA | −0.29 (n=8) | +0.88 (n=8) | −0.85 (n=8) | +0.98 (n=8) | −0.99 (n=8) | −0.88 (n=8) | 392 |
| floc-words | VWFA-1 | −0.16 (n=8) | +0.96 (n=8) | −0.91 (n=8) | +1.00 (n=8) | −1.00 (n=8) | −0.96 (n=8) | 454 |
| floc-words | VWFA-2 | −0.02 (n=5) | +0.95 (n=6) | −0.94 (n=6) | +0.99 (n=6) | −1.00 (n=6) | −0.96 (n=6) | 150 |
| floc-words | mfs-words | −0.08 (n=8) | +0.97 (n=8) | −0.93 (n=8) | +0.99 (n=8) | −1.00 (n=8) | −0.96 (n=8) | 121 |


**Extended Data Table 12 | The readout weight cortex prefers against the one behaviour prefers.** For each mixing weight w the classifier's own input z(w) = w·y₁ + (1 − w)·y₂ (y₁ magno stream output, y₂ parvo stream output, 512-d each) is the design of a ridge encoding model of every nsdgeneral voxel (same split, folds and alpha grid as the banded model; Methods 4.5). w*_cortex is the w that maximises the median test R² over reliable voxels of the ROI (mean over seeds per subject; median [range] over 8 subjects). "R² at w = 0.5 / max" is the median R² at the paper's readout weight relative to the best w; "concat" is the median R² of the 1,024-d concatenation, the ceiling for any mixture. "Voxels magno / parvo" is the fraction of the ROI's voxels whose own best w is ≥ 0.75 / ≤ 0.25. The behaviour-optimal weight (trial fit to each of 10 human observers' cue-conflict choices, diet final checkpoints, 3 seeds) is given in the last row for comparison, with the cue-conflict shape-bias rate the diet network produces at each weight.

| arm | ROI | w*_cortex median [range] | R² at w = 0.5 / max | max R² (median) | concat R² | voxels best w ≥ 0.75 | voxels best w ≤ 0.25 | rate at w*_cortex | n |
|---|---|---|---|---|---|---|---|---|---|
| diet | V1 | 0.58 [0.50, 0.70] | 0.994 | 0.1313 | 0.1402 | 0.19 | 0.11 | 0.72 | 8 |
| diet | V2 | 0.50 [0.45, 0.55] | 0.999 | 0.1024 | 0.1066 | 0.12 | 0.20 | 0.67 | 8 |
| diet | V3 | 0.40 [0.30, 0.50] | 0.987 | 0.0847 | 0.0864 | 0.08 | 0.29 | 0.61 | 8 |
| diet | hV4 | 0.35 [0.20, 0.50] | 0.976 | 0.0945 | 0.0948 | 0.05 | 0.41 | 0.57 | 8 |
| diet | ventral (VO, PHC) | 0.33 [0.25, 0.35] | 0.971 | 0.1339 | 0.1325 | 0.02 | 0.46 | 0.56 | 8 |
| diet | lateral (LO, TO) | 0.33 [0.25, 0.45] | 0.974 | 0.1067 | 0.1060 | 0.03 | 0.40 | 0.56 | 8 |
| diet | dorsal (V3A/B, IPS0–2) | 0.35 [0.25, 0.45] | 0.982 | 0.0683 | 0.0673 | 0.05 | 0.39 | 0.57 | 8 |
| no diet | V1 | 0.65 [0.55, 0.80] | 0.987 | 0.1221 | 0.1259 | 0.43 | 0.09 | nan | 8 |
| no diet | V2 | 0.58 [0.40, 0.95] | 0.995 | 0.0968 | 0.1022 | 0.36 | 0.13 | nan | 8 |
| no diet | V3 | 0.58 [0.50, 1.00] | 0.991 | 0.0845 | 0.0890 | 0.40 | 0.13 | nan | 8 |
| no diet | hV4 | 0.62 [0.40, 0.70] | 0.988 | 0.0965 | 0.1014 | 0.40 | 0.09 | nan | 8 |
| no diet | ventral (VO, PHC) | 0.75 [0.65, 1.00] | 0.975 | 0.1557 | 0.1535 | 0.59 | 0.02 | nan | 8 |
| no diet | lateral (LO, TO) | 0.88 [0.60, 1.00] | 0.949 | 0.1190 | 0.1214 | 0.59 | 0.03 | nan | 8 |
| no diet | dorsal (V3A/B, IPS0–2) | 1.00 [0.75, 1.00] | 0.944 | 0.0805 | 0.0792 | 0.61 | 0.04 | nan | 8 |
| behaviour (diet) | 10 human observers, trial fit | 0.67 [0.49, 0.89] | — | — | — | — | — | 0.78 | 10 |

**Extended Data Table 13 | The readout span without a stream split: one gain on a network's own shape- and texture-predictive subspaces, head fixed.** For each frozen network with logits = W·y + b, the 15 discriminant directions of a 16-class shape LDA and of a 16-class texture LDA on the penultimate features y (PCA to 256 dimensions, Ledoit–Wolf shrinkage; fitted on one stratified half of the 1,200 cue-conflict images) are made mutually orthogonal by symmetric (Löwdin) orthogonalisation, giving orthonormal bases B_S and B_T (15 directions each) and the residual B_R. logits(w) = W·(P_R y + 2w·P_S y + 2(1 − w)·P_T y) + b with the network's own head untouched; w = 0.5 is the trained network. The rate is scored on the other half of the images (10 random splits, both directions; mean, with the range over splits and seeds in brackets), by the paper's primary estimator (112-class, decided images) for our networks and the Geirhos 16-class probability aggregation for the ImageNet networks; “forced-estimator span” is the same sweep on the forced two-way estimator. “gated span” is max − min of the seed-mean held-out curve over the w at which top-1 of the gained features (bases fitted on all 1,200 images) on held-out natural images (MiniEcoset validation for our networks; a stratified 10-per-class ILSVRC-2012 validation subset for the ImageNet networks) stays within 5 points of the trained value, followed by that w-range and the largest change of top-1 anywhere in the sweep. “refit span” is the block-form lossy sum z(w) = [w·s + (1 − w)·t, B_R^T^y], s = B_S^T^y and t = B_T^T^y paired by discriminant rank, with a linear and a two-layer head refitted at each w (Methods 4.3 budget), our networks only; in parentheses the largest shortfall of validation accuracy, over w, from a head refitted on the unmixed features. “variance in S / T” is the fraction of the feature variance on the fitting half lying in each subspace; “order sensitivity” replaces the symmetric orthogonalisation by sequential Gram–Schmidt with the shape basis or the texture basis first. “median-split gated span” is the gated span of the pre-specified unit-level rule of Extended Data Table 14 on the same networks and images. For the two-stream concatenation, which has no held-out natural-image features, the fraction of the shape basis's energy lying on stream-1 units (all-image bases; held-out bases in parentheses; range over seeds) replaces the gate. Pre-specification: `docs/PLAN_SUBSPACE_2026-09-15.md` (predictions P-A, P-A′, P-B, P-D). Source: `results/figures_v2/subspace_behaviour.json`, aggregated by `tmp_analysis/figs_v2/subspace_behaviour.py` from the outputs of `src/analysis/generality/subspace_factor.py`, `gain_accuracy.py` and `refit_sum.py` (`slurm/subspace_{cc,refit}.sh`); the per-network files are outside the repository.

| network | D | trained rate | w = 0 | w = 1 | span (held-out) [range] | ρ(w, rate) | gated span (w-range; largest change of top-1) | forced-estimator span | refit span linear / two-layer (largest shortfall of top-1) | variance in S / T | order sensitivity, S first / T first | median-split gated span | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two-stream, diet, streams concatenated (validation) | 1024 | 0.675 | 0.495 | 0.801 | 0.320 [0.18, 0.45] | 0.98 | — (shape basis on stream 1: 0.63–0.70) | 0.216 | — | 0.11 / 0.29 | 0.35 / 0.25 | — | 3 |
| single ResNet-34, diet | 512 | 0.625 | 0.516 | 0.735 | 0.226 [0.12, 0.37] | 0.98 | 0.19 (0.10–0.90; 11.9 pts) | 0.163 | 0.08 / 0.10 (0.8 pts) | 0.14 / 0.20 | 0.24 / 0.20 | 0.07 | 3 |
| single ResNet-18, diet | 512 | 0.557 | 0.433 | 0.674 | 0.249 [0.13, 0.41] | 0.98 | 0.14 (0.25–0.73; 16.7 pts) | 0.189 | 0.08 / 0.10 (0.6 pts) | 0.15 / 0.23 | 0.25 / 0.20 | 0.09 | 3 |
| single ResNet-18, no diet (`rgb_tsn`) | 512 | 0.445 | 0.311 | 0.651 | 0.344 [0.22, 0.47] | 0.99 | 0.22 (0.20–0.78; 13.4 pts) | 0.242 | 0.13 / 0.15 (0.5 pts) | 0.13 / 0.26 | 0.29 / 0.34 | 0.08 | 3 |
| ImageNet ViT-B/16 | 768 | 0.399 | 0.287 | 0.570 | 0.282 [0.22, 0.34] | 1.00 | 0.28 (0.00–1.00; 0.2 pts) | 0.318 | — | 0.11 / 0.23 | 0.22 / 0.31 | 0.08 | 1 |
| ImageNet ResNet-152 | 2048 | 0.293 | 0.209 | 0.455 | 0.246 [0.20, 0.33] | 1.00 | 0.25 (0.00–1.00; 1.2 pts) | 0.271 | — | 0.07 / 0.24 | 0.21 / 0.27 | 0.16 | 1 |
| ImageNet ConvNeXt-B | 1024 | 0.269 | 0.178 | 0.512 | 0.334 [0.29, 0.39] | 1.00 | 0.33 (0.00–1.00; 0.2 pts) | 0.371 | — | 0.08 / 0.33 | 0.31 / 0.35 | 0.11 | 1 |
| ImageNet ConvNeXt-S | 768 | 0.248 | 0.170 | 0.458 | 0.287 [0.24, 0.34] | 1.00 | 0.29 (0.00–1.00; 0.1 pts) | 0.353 | — | 0.08 / 0.33 | 0.27 / 0.30 | 0.12 | 1 |
| ImageNet ResNet-50 | 2048 | 0.222 | 0.157 | 0.447 | 0.290 [0.21, 0.34] | 1.00 | 0.29 (0.00–1.00; 1.5 pts) | 0.310 | — | 0.07 / 0.28 | 0.28 / 0.29 | 0.13 | 1 |
| ImageNet VGG-16 | 4096 | 0.110 | 0.061 | 0.329 | 0.268 [0.23, 0.31] | 1.00 | 0.27 (0.00–1.00; 0.5 pts) | 0.322 | — | 0.04 / 0.28 | 0.26 / 0.27 | 0.09 | 1 |

**Extended Data Table 14 | The readout span without a stream split under the pre-specified unit-level rule: fixed-head gain sweep on the shape- and texture-leaning halves of the units.** The special case of Extended Data Table 13 in which the two subspaces are spanned by disjoint sets of units. Each unit of a network's penultimate layer is scored by the fraction of its variance between shape categories minus that between texture categories on the 1,200 cue-conflict images (one-way η²); the median split gives a shape-leaning half S and a texture-leaning half T. logits(w) = head(g(w) ⊙ y) with g = 2w on S and 2(1 − w) on T, the network's own head untouched; w = 0.5 is the trained network. The split is defined on one stratified half of the images and the rate scored on the other (10 random splits, both directions, the same splits as Extended Data Table 13; mean, with the range over splits and seeds in brackets). Rates by the paper's primary estimator (112-class, decided images) for our networks and the Geirhos 16-class probability aggregation for the ImageNet networks; “forced-estimator span” is the same sweep on the forced two-way estimator. “shape-leaning units” is the fraction of units with η²_shape > η²_texture on all images; for the two-stream concatenation the fraction of S that belongs to stream 1 is given in parentheses (P3). “w within 5 pts” is the w-range over which top-1 on held-out natural images (MiniEcoset validation; a stratified 10-per-class ILSVRC-2012 validation subset for the ImageNet networks) stays within 5 points of the trained value. “refit span” is the lossy-sum protocol z(w) = w·y_S + (1 − w)·y_T with a linear and a two-layer head refitted at every w (Methods 4.3 budget), our networks only. The last two columns are a sensitivity analysis added after the pre-specified analysis, at a referee's request: the same held-out sweep with S defined as the absolute shape-leaning set (units with d_j > 0) or as the top decile of units by d_j, and T as the remainder; given are the held-out span with its range over splits and seeds, the minimum Spearman ρ, the mean size of S and the w-range within 5 points of the trained top-1 (no gate exists for the two-stream concatenation, which has no held-out natural-image features). Pre-specification: `docs/PLAN_GENERALITY_2026-09-14.md`. Source: `results/figures_v2/generality_behaviour.json`, aggregated by `tmp_analysis/figs_v2/generality_behaviour.py` from the per-network outputs of `src/analysis/generality/` (`slurm/generality_{cc,mini,inval,rules}.sh`); the per-network files are outside the repository.

| network | units | trained rate | w = 0 | w = 1 | span (held-out) [range] | ρ(w, rate) | forced-estimator span | shape-leaning units | w within 5 pts of trained top-1 | refit span linear / two-layer | span, S = {d > 0} | span, top decile | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two-stream, diet, streams concatenated (validation) | 1024 | 0.675 | 0.420 | 0.879 | 0.461 [0.39, 0.57] | 1.00 | 0.307 | 0.49 (S from stream 1: 0.99) | — | — | 0.46 [0.39, 0.59] (ρ ≥ 0.98; |S| = 504) | 0.20 [0.11, 0.30] (ρ ≥ 0.56; |S| = 102) | 3 |
| single ResNet-34, diet | 512 | 0.625 | 0.565 | 0.688 | 0.136 [0.03, 0.24] | 0.71 | 0.102 | 0.17 | 0.20–0.78 (top-1 0.626) | 0.06 / 0.05 | 0.12 [0.03, 0.23] (ρ ≥ -0.78; |S| = 89; gate 0.00–0.73) | 0.11 [0.03, 0.20] (ρ ≥ -0.95; |S| = 51; gate 0.00–0.77) | 3 |
| single ResNet-18, diet | 512 | 0.557 | 0.467 | 0.652 | 0.194 [0.11, 0.31] | 0.97 | 0.131 | 0.09 | 0.28–0.72 (top-1 0.547) | 0.09 / 0.06 | 0.11 [0.04, 0.18] (ρ ≥ -0.63; |S| = 44; gate 0.00–0.73) | 0.12 [0.04, 0.21] (ρ ≥ -0.42; |S| = 51; gate 0.00–0.72) | 3 |
| single ResNet-18, no diet (`rgb_tsn`) | 512 | 0.445 | 0.374 | 0.566 | 0.195 [0.11, 0.39] | 0.96 | 0.164 | 0.05 | 0.27–0.75 (top-1 0.678) | 0.10 / 0.09 | 0.15 [0.03, 0.31] (ρ ≥ -0.57; |S| = 24; gate 0.00–0.78) | 0.17 [0.07, 0.29] (ρ ≥ -0.55; |S| = 51; gate 0.00–0.75) | 3 |
| ImageNet ViT-B/16 | 768 | 0.399 | 0.371 | 0.455 | 0.086 [0.07, 0.11] | 0.98 | 0.138 | 0.01 | 0.00–1.00 (top-1 0.815) | — | 0.21 [0.10, 0.34] (ρ ≥ 0.53; |S| = 6; gate 0.00–0.95) | 0.19 [0.14, 0.26] (ρ ≥ 0.81; |S| = 76; gate 0.00–0.95) | 1 |
| ImageNet ResNet-152 | 2048 | 0.293 | 0.227 | 0.440 | 0.214 [0.18, 0.25] | 1.00 | 0.196 | 0.02 | 0.15–0.90 (top-1 0.785) | — | 0.27 [0.22, 0.33] (ρ ≥ 0.86; |S| = 45; gate 0.00–0.85) | 0.30 [0.26, 0.36] (ρ ≥ 0.94; |S| = 204; gate 0.00–0.80) | 1 |
| ImageNet ConvNeXt-B | 1024 | 0.269 | 0.229 | 0.343 | 0.115 [0.09, 0.14] | 0.99 | 0.173 | 0.00 | 0.00–1.00 (top-1 0.843) | — | 0.28 [0.20, 0.35] (ρ ≥ 0.14; |S| = 2; gate 0.00–0.95) | 0.21 [0.13, 0.29] (ρ ≥ 0.73; |S| = 102; gate 0.00–1.00) | 1 |
| ImageNet ConvNeXt-S | 768 | 0.248 | 0.211 | 0.335 | 0.124 [0.11, 0.15] | 0.99 | 0.179 | 0.00 | 0.00–1.00 (top-1 0.840) | — | undefined (empty half) | 0.17 [0.12, 0.20] (ρ ≥ 0.72; |S| = 76; gate 0.00–0.95) | 1 |
| ImageNet ResNet-50 | 2048 | 0.222 | 0.167 | 0.346 | 0.180 [0.14, 0.23] | 0.99 | 0.202 | 0.02 | 0.15–0.90 (top-1 0.760) | — | 0.27 [0.20, 0.32] (ρ ≥ 0.75; |S| = 34; gate 0.00–0.85) | 0.27 [0.20, 0.31] (ρ ≥ 0.85; |S| = 204; gate 0.00–0.80) | 1 |
| ImageNet VGG-16 | 4096 | 0.110 | 0.096 | 0.189 | 0.095 [0.07, 0.11] | 0.96 | 0.106 | 0.01 | 0.00–1.00 (top-1 0.711) | — | 0.40 [0.34, 0.46] (ρ ≥ -0.38; |S| = 56; gate 0.00–0.95) | 0.27 [0.22, 0.34] (ρ ≥ 0.23; |S| = 409; gate 0.00–0.90) | 1 |

**Extended Data Table 15 | Cortex without a stream split: the shape- and texture-predictive components as encoding bands.** Upper block (subspace bands): the three bands are the shape-predictive subspace S = Y·B_S (15 dimensions), the texture-predictive subspace T = Y·B_T (15) and the residual R = Y·B_R of Extended Data Table 13 (bases fitted on all 1,200 images), with U_S = r²_{STR} − r²_{TR} and U_T = r²_{STR} − r²_{SR} from the three-band model and the two two-band models (random search, 30 and 20 samples); “shared” is r²_{STR} − U_S − U_T and “r²” the three-band model's noise-ceiling-normalised r². w*_cortex there is the weight of the 15-dimensional lossy sum z(w) = w·s + (1 − w)·t alone. Lower block (unit halves): for each single-trunk network the penultimate units were split as in Extended Data Table 14 (all 1,200 images) and the two halves entered as two bands of a banded ridge model of every `nsdgeneral` voxel (same split, folds and alpha grid as Methods 4.5); U_shape and U_texture are the noise-ceiling-normalised unique variances and the index is (U_shape − U_texture)/(U_shape + U_texture), median over reliable voxels (noise ceiling > 0.1, ≥ 20 voxels), mean over eight subjects (seeds averaged per subject). Regions: V1–hV4 from `prf-visualrois`; ventral (VO1/2, PHC1/2), lateral (LO1/2, TO1/2) and dorsal (V3A/B, IPS0–2) from Kastner2015. The two-stream diet row applies the same computation to the magno (shape-biased) and parvo (texture-biased) bands of Fig. 3. “shared” is the cross-validated variance the bands account for jointly (the full model's r² minus the unique variances) and “r²” the full model's noise-ceiling-normalised r², medians over the same voxels. w*_cortex is the mixing weight of the lossy sum z(w) = w·y_S + (1 − w)·y_T that maximises an ROI's median test R² (Extended Data Fig. 5 protocol). Exact paired Wilcoxon over subjects. Source: `results/figures_v2/generality_cortex.json` and `results/figures_v2/subspace_cortex.json`, aggregated by `tmp_analysis/figs_v2/generality_cortex.py` from the per-subject fits of `src/analysis/generality/cortex_bands.py` (`slurm/generality_{nsd,cortex}.sh`, `slurm/subspace_{nsd,cortex}.sh`), which are deposited alongside the encoding outputs of Fig. 3 (Code availability).

| network | statistic | V1 | V2 | V3 | hV4 | ventral | lateral | dorsal | hV4 − V1 (n neg, p) | extrastriate − V1 (n neg, p) | ρ(rank, index) V1–hV4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **subspace bands S, T, R** | |  |  |  |  |  |  |  |  | | |
| single ResNet-18, no diet | index | +0.12 | -0.01 | +0.01 | +0.03 | +0.35 | +0.13 | +0.32 | -0.088 (5/8, 0.4609) | +0.151 (2/8, 0.0781) | -0.25 |
| | U_shape | 0.011 | 0.009 | 0.010 | 0.010 | 0.013 | 0.009 | 0.011 | | | |
| | U_texture | 0.008 | 0.009 | 0.009 | 0.009 | 0.006 | 0.007 | 0.006 | | | |
| | shared | 0.329 | 0.311 | 0.305 | 0.327 | 0.521 | 0.409 | 0.375 | | | |
| | r² (full model, nc-normalised) | 0.351 | 0.332 | 0.327 | 0.351 | 0.542 | 0.429 | 0.396 | | | |
| | w*_cortex (median) | 0.63 | 0.68 | 0.68 | 0.63 | 0.33 | 0.33 | 0.33 | | Δw* extrastriate − V1 -0.23 (6/8 neg, 0.0391) | |
| single ResNet-34, diet | index | -0.03 | -0.13 | -0.08 | +0.13 | +0.19 | -0.02 | +0.14 | +0.153 (2/8, 0.1094) | +0.130 (2/8, 0.3828) | +0.25 |
| | U_shape | 0.007 | 0.007 | 0.007 | 0.010 | 0.011 | 0.007 | 0.010 | | | |
| | U_texture | 0.007 | 0.008 | 0.008 | 0.007 | 0.008 | 0.007 | 0.008 | | | |
| | shared | 0.325 | 0.298 | 0.293 | 0.316 | 0.503 | 0.379 | 0.358 | | | |
| | r² (full model, nc-normalised) | 0.341 | 0.316 | 0.313 | 0.337 | 0.524 | 0.397 | 0.378 | | | |
| | w*_cortex (median) | 0.62 | 0.57 | 0.55 | 0.45 | 0.55 | 0.68 | 0.48 | | Δw* extrastriate − V1 -0.01 (4/8 neg, 0.9453) | |
| ImageNet ResNet-50 | index | -0.34 | -0.18 | -0.23 | +0.11 | -0.18 | -0.24 | -0.29 | +0.452 (0/7, 0.0156) | +0.106 (2/7, 0.2969) | +0.69 |
| | U_shape | 0.004 | 0.005 | 0.005 | 0.009 | 0.005 | 0.004 | 0.004 | | | |
| | U_texture | 0.008 | 0.006 | 0.007 | 0.006 | 0.006 | 0.005 | 0.006 | | | |
| | shared | 0.354 | 0.328 | 0.317 | 0.374 | 0.631 | 0.499 | 0.453 | | | |
| | r² (full model, nc-normalised) | 0.369 | 0.342 | 0.334 | 0.395 | 0.646 | 0.511 | 0.470 | | | |
| | w*_cortex (median) | 0.40 | 0.30 | 0.40 | 0.40 | 0.40 | 0.50 | 0.40 | | Δw* extrastriate − V1 -0.01 (4/7 neg, 0.8125) | |
| ImageNet ViT-B/16 | index | -0.45 | -0.54 | -0.45 | -0.25 | -0.50 | -0.43 | -0.70 | +0.193 (1/8, 0.0781) | -0.097 (3/8, 0.8438) | +0.37 |
| | U_shape | 0.004 | 0.003 | 0.004 | 0.007 | 0.004 | 0.003 | 0.002 | | | |
| | U_texture | 0.006 | 0.006 | 0.007 | 0.009 | 0.010 | 0.005 | 0.008 | | | |
| | shared | 0.254 | 0.242 | 0.254 | 0.292 | 0.555 | 0.404 | 0.382 | | | |
| | r² (full model, nc-normalised) | 0.267 | 0.256 | 0.271 | 0.310 | 0.573 | 0.416 | 0.398 | | | |
| | w*_cortex (median) | 0.50 | 0.40 | 0.45 | 0.15 | 0.50 | 0.40 | 0.55 | | Δw* extrastriate − V1 +0.08 (3/8 neg, 0.3828) | |
| ImageNet ConvNeXt-B | index | +0.17 | +0.01 | -0.17 | +0.38 | +0.00 | -0.30 | -0.51 | +0.212 (3/8, 0.1953) | -0.440 (7/8, 0.0156) | +0.08 |
| | U_shape | 0.006 | 0.005 | 0.005 | 0.008 | 0.006 | 0.004 | 0.003 | | | |
| | U_texture | 0.004 | 0.003 | 0.004 | 0.003 | 0.004 | 0.006 | 0.005 | | | |
| | shared | 0.256 | 0.240 | 0.233 | 0.285 | 0.540 | 0.408 | 0.369 | | | |
| | r² (full model, nc-normalised) | 0.267 | 0.253 | 0.246 | 0.301 | 0.554 | 0.422 | 0.385 | | | |
| | w*_cortex (median) | 0.45 | 0.55 | 0.55 | 0.00 | 0.75 | 0.70 | 0.75 | | Δw* extrastriate − V1 +0.20 (1/8 neg, 0.0391) | |
| **unit halves S, T** | |  |  |  |  |  |  |  |  | | |
| two-stream, diet (magno − parvo) | index | -0.06 | -0.44 | -0.70 | -0.84 | -0.88 | -0.87 | -0.87 | — | — | — |
| | U_shape | 0.087 | 0.045 | 0.026 | 0.016 | 0.014 | 0.014 | 0.013 | | | |
| | U_texture | 0.099 | 0.119 | 0.145 | 0.180 | 0.246 | 0.190 | 0.181 | | | |
| | shared | 0.289 | 0.286 | 0.261 | 0.219 | 0.230 | 0.184 | 0.168 | | | |
| | r² (full model, nc-normalised) | 0.485 | 0.457 | 0.439 | 0.428 | 0.496 | 0.390 | 0.373 | | | |
| single ResNet-34, diet | index | +0.05 | +0.02 | +0.05 | -0.03 | -0.22 | -0.09 | -0.06 | -0.080 (7/8, 0.0156) | -0.178 (7/8, 0.0156) | -0.25 |
| | U_shape | 0.032 | 0.030 | 0.028 | 0.025 | 0.017 | 0.016 | 0.020 | | | |
| | U_texture | 0.028 | 0.028 | 0.025 | 0.027 | 0.027 | 0.020 | 0.022 | | | |
| | shared | 0.263 | 0.245 | 0.245 | 0.274 | 0.458 | 0.353 | 0.320 | | | |
| | r² (full model, nc-normalised) | 0.327 | 0.304 | 0.300 | 0.329 | 0.506 | 0.392 | 0.366 | | | |
| | w*_cortex (median) | 0.53 | 0.60 | 0.57 | 0.53 | 0.53 | 0.55 | 0.57 | | Δw* extrastriate − V1 +0.01 (4/8 neg, 0.9453) | |
| single ResNet-18, no diet | index | +0.17 | +0.13 | +0.12 | +0.17 | -0.15 | -0.14 | -0.14 | -0.005 (5/8, 0.9453) | -0.316 (8/8, 0.0078) | -0.15 |
| | U_shape | 0.038 | 0.035 | 0.030 | 0.031 | 0.019 | 0.016 | 0.018 | | | |
| | U_texture | 0.026 | 0.026 | 0.024 | 0.022 | 0.025 | 0.022 | 0.024 | | | |
| | shared | 0.272 | 0.258 | 0.262 | 0.288 | 0.483 | 0.371 | 0.335 | | | |
| | r² (full model, nc-normalised) | 0.340 | 0.321 | 0.319 | 0.347 | 0.530 | 0.415 | 0.381 | | | |
| | w*_cortex (median) | 0.80 | 0.80 | 0.68 | 0.68 | 0.42 | 0.48 | 0.37 | | Δw* extrastriate − V1 -0.32 (7/8 neg, 0.0234) | |
| ImageNet ResNet-50 | index | -0.38 | -0.38 | -0.32 | -0.50 | -0.44 | +0.06 | -0.07 | -0.120 (5/8, 0.3828) | +0.235 (1/8, 0.0391) | -0.13 |
| | U_shape | 0.008 | 0.008 | 0.008 | 0.006 | 0.006 | 0.010 | 0.009 | | | |
| | U_texture | 0.019 | 0.018 | 0.017 | 0.020 | 0.015 | 0.009 | 0.011 | | | |
| | shared | 0.312 | 0.292 | 0.297 | 0.365 | 0.606 | 0.489 | 0.446 | | | |
| | r² (full model, nc-normalised) | 0.343 | 0.322 | 0.327 | 0.397 | 0.633 | 0.512 | 0.471 | | | |
| | w*_cortex (median) | 0.10 | 0.10 | 0.00 | 0.05 | 0.00 | 0.85 | 0.70 | | Δw* extrastriate − V1 +0.07 (3/8 neg, 0.8125) | |

**Extended Data Table 16 | The readout span across the field's model zoo.** Frozen ImageNet networks under the subspace rule of Extended Data Table 13 (PCA to 256 dimensions, 16-class shape and texture LDA, 15 directions each, symmetric orthogonalisation; bases fitted on one stratified half of the 1,200 cue-conflict images and the rate scored on the other, 10 splits both directions), each network's own head untouched, the Geirhos 16-class probability aggregation throughout; “images decided” is the number of the 1,200 assigned to their shape or their texture category at the trained readout, the denominator of the rate. “gate” is the w-range over which top-1 on a stratified 10-per-class ILSVRC-2012 validation subset (10,000 images; bases fitted on all cue-conflict images) stays within 5 points of the trained value, with the largest change of top-1 inside it; “texture end / shape end” are the held-out rates at the smallest and the largest gated w, and “gated span” is their difference, under the subspace rule and under the pre-specified unit-level median split of Extended Data Table 14. “probe” is the linear probe of Methods 4.3 (PCA to 256 dimensions, multinomial logistic regression, 5-fold stratified cross-validation) predicting the shape class and, separately, the texture class from the same features; bias is shape minus texture accuracy (chance 0.0625). Preprocessing is Resize 256 / CenterCrop 224 except for Inception-v3 (342/299), EfficientNet-B3 (320/300) and EfficientNet-V2-S (384/384), which were not trained at 224. “κ with observers” is error consistency (Cohen's κ on trial-by-trial agreement, Results 2.5; Extended Data Table 7) with each of the ten human observers, mean over observers, as a percentage of the human–human ceiling κ = 0.234, at the trained readout and, in parentheses, at the texture end and the shape end of the gate under the same gain; computed under this pipeline's preprocessing, so the VGG-16 value differs from the published response file's 7.3% (Extended Data Table 7) as its trained rate does. Over the 38 ImageNet-trained networks: Kendall τ between the ranking by trained rate and by the shape-end rate 0.47 (by the texture-end rate 0.78); 188 of 703 pairs (27%) reverse order; gated span 0.20–0.50 (median 0.30), ≥ 0.20 in 97% and the whole sweep inside the gate in 100%; Spearman ρ between gated span and trained rate +0.04, and trained top-1 -0.22; error consistency at the trained readout 2.4–23.0% of the ceiling (median 7.9%), at most 36.7% anywhere inside a gate, changing by a median of +13.5 points from the texture end to the shape end; ρ(κ, trained rate) +0.80; Kendall τ between trained rate and probe bias 0.44. The three Stylized-ImageNet networks are the published checkpoints of Geirhos et al. (2019), all ResNet-50: trained on Stylized-ImageNet (SIN), on SIN + IN, and on SIN + IN then fine-tuned on IN. Pre-specification: `docs/PLAN_ZOO_2026-09-15.md` (predictions P-Z1–P-Z4). Source: `results/figures_v2/zoo_behaviour.json`, aggregated by `tmp_analysis/figs_v2/zoo_behaviour.py` from the outputs of `slurm/zoo_cc.sh`; the per-network files are outside the repository.

| network | D | top-1 (subset) | trained rate (images decided) | gate: w-range (largest Δtop-1) | rate at texture end / shape end | gated span, subspace rule | ρ(w, rate) | gated span, unit rule | probe: shape / texture accuracy (bias) | κ with observers, % of ceiling: trained (texture end → shape end) | whole sweep in gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ImageNet-trained (torchvision IMAGENET1K_V1)** | | | | | | | | | | | |
| ViT-B/32 | 768 | 0.758 | 0.617 (767) | 0.00–1.00 (0.2 pts) | 0.499 / 0.738 | 0.239 | 1.00 | 0.069 | 0.67 / 0.97 (-0.30) | 23.0 (12.9 → 34.0) | yes |
| ViT-B/16 | 768 | 0.815 | 0.399 (818) | 0.00–1.00 (0.2 pts) | 0.287 / 0.570 | 0.282 | 1.00 | 0.085 | 0.56 / 0.99 (-0.43) | 11.6 (7.2 → 19.2) | yes |
| ViT-L/16 | 1024 | 0.801 | 0.319 (885) | 0.00–1.00 (0.2 pts) | 0.207 / 0.545 | 0.337 | 1.00 | 0.078 | 0.57 / 0.99 (-0.43) | 11.5 (5.9 → 21.9) | yes |
| Inception-v3 | 2048 | 0.759 | 0.305 (797) | 0.00–1.00 (0.6 pts) | 0.241 / 0.487 | 0.246 | 0.99 | 0.099 | 0.40 / 0.96 (-0.56) | 11.0 (5.5 → 22.5) | yes |
| MaxViT-T | 512 | 0.842 | 0.295 (899) | 0.00–1.00 (0.2 pts) | 0.230 / 0.469 | 0.239 | 1.00 | 0.114 | 0.40 / 0.97 (-0.57) | 9.1 (4.5 → 20.7) | yes |
| ResNeXt-101 32x8d | 2048 | 0.796 | 0.294 (732) | 0.00–1.00 (0.6 pts) | 0.228 / 0.511 | 0.283 | 0.99 | 0.220 | 0.47 / 1.00 (-0.53) | 7.9 (3.9 → 16.3) | yes |
| ConvNeXt-L | 1536 | 0.848 | 0.293 (924) | 0.00–1.00 (0.1 pts) | 0.196 / 0.528 | 0.332 | 1.00 | 0.079 | 0.51 / 1.00 (-0.49) | 9.2 (4.4 → 18.1) | yes |
| ResNet-152 | 2048 | 0.785 | 0.293 (781) | 0.00–1.00 (1.2 pts) | 0.209 / 0.455 | 0.246 | 1.00 | 0.159 | 0.51 / 1.00 (-0.49) | 9.1 (2.1 → 16.0) | yes |
| EfficientNet-B3 | 1536 | 0.818 | 0.291 (905) | 0.00–1.00 (0.2 pts) | 0.204 / 0.514 | 0.310 | 1.00 | 0.141 | 0.49 / 0.99 (-0.50) | 12.5 (6.8 → 26.0) | yes |
| ResNet-101 | 2048 | 0.777 | 0.284 (770) | 0.00–1.00 (1.4 pts) | 0.216 / 0.493 | 0.276 | 1.00 | 0.143 | 0.48 / 1.00 (-0.52) | 8.7 (5.1 → 14.8) | yes |
| Wide-ResNet-101-2 | 2048 | 0.790 | 0.279 (746) | 0.00–1.00 (0.7 pts) | 0.201 / 0.431 | 0.231 | 1.00 | 0.119 | 0.39 / 0.99 (-0.60) | 7.9 (3.4 → 18.5) | yes |
| DenseNet-201 | 1920 | 0.764 | 0.273 (792) | 0.00–1.00 (1.5 pts) | 0.206 / 0.540 | 0.334 | 1.00 | 0.085 | 0.51 / 1.00 (-0.49) | 8.5 (3.0 → 18.2) | yes |
| EfficientNet-B0 | 1280 | 0.777 | 0.271 (835) | 0.00–1.00 (0.4 pts) | 0.175 / 0.502 | 0.327 | 1.00 | 0.110 | 0.45 / 0.99 (-0.54) | 7.1 (3.6 → 19.0) | yes |
| ConvNeXt-B | 1024 | 0.843 | 0.269 (921) | 0.00–1.00 (0.2 pts) | 0.178 / 0.512 | 0.334 | 1.00 | 0.114 | 0.53 / 1.00 (-0.47) | 8.0 (2.1 → 21.6) | yes |
| MobileNet-V3-L | 1280 | 0.741 | 0.265 (796) | 0.00–1.00 (1.2 pts) | 0.156 / 0.591 | 0.435 | 1.00 | 0.102 | 0.53 / 1.00 (-0.47) | 10.8 (4.0 → 29.0) | yes |
| AlexNet | 4096 | 0.552 | 0.261 (735) | 0.00–1.00 (1.4 pts) | 0.120 / 0.624 | 0.504 | 1.00 | 0.142 | 0.55 / 1.00 (-0.45) | 7.1 (3.0 → 14.3) | yes |
| RegNet-Y-16GF | 3024 | 0.808 | 0.261 (797) | 0.00–1.00 (0.6 pts) | 0.180 / 0.480 | 0.300 | 1.00 | 0.208 | 0.50 / 1.00 (-0.50) | 7.7 (4.6 → 15.3) | yes |
| ResNet-34 | 512 | 0.732 | 0.259 (791) | 0.00–1.00 (2.7 pts) | 0.175 / 0.528 | 0.353 | 1.00 | 0.084 | 0.47 / 1.00 (-0.53) | 6.1 (3.5 → 15.9) | yes |
| GoogLeNet | 1024 | 0.691 | 0.251 (876) | 0.00–1.00 (1.4 pts) | 0.187 / 0.483 | 0.296 | 1.00 | 0.056 | 0.47 / 0.99 (-0.52) | 9.5 (5.4 → 22.1) | yes |
| ConvNeXt-S | 768 | 0.840 | 0.248 (901) | 0.00–1.00 (0.1 pts) | 0.170 / 0.458 | 0.287 | 1.00 | 0.124 | 0.47 / 1.00 (-0.53) | 8.1 (3.6 → 16.9) | yes |
| RegNet-Y-8GF | 2016 | 0.802 | 0.244 (788) | 0.00–1.00 (0.8 pts) | 0.151 / 0.516 | 0.365 | 1.00 | 0.140 | 0.50 / 1.00 (-0.50) | 6.7 (1.8 → 15.9) | yes |
| ConvNeXt-T | 768 | 0.830 | 0.241 (924) | 0.00–1.00 (0.4 pts) | 0.154 / 0.507 | 0.353 | 1.00 | 0.125 | 0.50 / 1.00 (-0.50) | 7.0 (2.4 → 22.3) | yes |
| Swin-S | 768 | 0.836 | 0.236 (880) | 0.00–1.00 (0.2 pts) | 0.154 / 0.440 | 0.287 | 1.00 | 0.077 | 0.44 / 0.99 (-0.55) | 8.8 (5.3 → 16.6) | yes |
| ShuffleNet-V2 x1.0 | 1024 | 0.686 | 0.234 (781) | 0.00–1.00 (2.5 pts) | 0.146 / 0.527 | 0.381 | 1.00 | 0.034 | 0.57 / 1.00 (-0.43) | 9.2 (4.0 → 25.7) | yes |
| Swin-B | 1024 | 0.839 | 0.233 (891) | 0.00–1.00 (0.1 pts) | 0.177 / 0.406 | 0.229 | 1.00 | 0.071 | 0.42 / 1.00 (-0.57) | 7.7 (4.0 → 17.0) | yes |
| ResNet-18 | 512 | 0.695 | 0.233 (849) | 0.00–1.00 (3.8 pts) | 0.163 / 0.479 | 0.316 | 1.00 | 0.050 | 0.47 / 0.99 (-0.53) | 6.3 (3.2 → 16.2) | yes |
| Wide-ResNet-50-2 | 2048 | 0.785 | 0.230 (792) | 0.00–1.00 (0.6 pts) | 0.166 / 0.427 | 0.261 | 0.99 | 0.105 | 0.38 / 0.99 (-0.62) | 5.0 (2.4 → 11.6) | yes |
| MNASNet-1.0 | 1280 | 0.734 | 0.226 (766) | 0.00–1.00 (1.3 pts) | 0.134 / 0.530 | 0.395 | 1.00 | 0.042 | 0.47 / 1.00 (-0.53) | 8.7 (3.7 → 19.7) | yes |
| Swin-T | 768 | 0.820 | 0.225 (902) | 0.00–1.00 (0.2 pts) | 0.150 / 0.460 | 0.309 | 1.00 | 0.084 | 0.48 / 1.00 (-0.52) | 8.4 (4.2 → 18.4) | yes |
| DenseNet-121 | 1024 | 0.740 | 0.224 (839) | 0.00–1.00 (1.9 pts) | 0.147 / 0.494 | 0.347 | 1.00 | 0.071 | 0.45 / 1.00 (-0.55) | 5.7 (3.7 → 14.9) | yes |
| ResNet-50 | 2048 | 0.760 | 0.222 (814) | 0.00–1.00 (1.5 pts) | 0.157 / 0.447 | 0.290 | 1.00 | 0.130 | 0.43 / 1.00 (-0.56) | 5.1 (2.7 → 12.2) | yes |
| ResNeXt-50 32x4d | 2048 | 0.774 | 0.219 (776) | 0.00–1.00 (0.9 pts) | 0.145 / 0.461 | 0.316 | 1.00 | 0.193 | 0.41 / 0.99 (-0.58) | 5.6 (2.1 → 14.1) | yes |
| EfficientNet-V2-S | 1280 | 0.832 | 0.184 (879) | 0.00–1.00 (0.2 pts) | 0.134 / 0.332 | 0.198 | 1.00 | 0.074 | 0.35 / 0.99 (-0.64) | 5.6 (3.6 → 8.1) | yes |
| MobileNet-V2 | 1280 | 0.715 | 0.184 (798) | 0.00–1.00 (4.4 pts) | 0.128 / 0.474 | 0.346 | 1.00 | 0.051 | 0.48 / 1.00 (-0.52) | 5.6 (1.9 → 18.0) | yes |
| VGG-19-BN | 4096 | 0.743 | 0.150 (867) | 0.00–1.00 (0.8 pts) | 0.094 / 0.335 | 0.241 | 1.00 | 0.093 | 0.37 / 0.99 (-0.62) | 4.0 (2.2 → 6.8) | yes |
| VGG-16-BN | 4096 | 0.733 | 0.115 (915) | 0.00–1.00 (0.7 pts) | 0.068 / 0.294 | 0.226 | 1.00 | 0.104 | 0.36 / 1.00 (-0.64) | 2.4 (2.9 → 5.2) | yes |
| VGG-16 | 4096 | 0.711 | 0.110 (918) | 0.00–1.00 (0.5 pts) | 0.061 / 0.329 | 0.268 | 1.00 | 0.093 | 0.33 / 0.99 (-0.66) | 3.4 (2.3 → 6.4) | yes |
| VGG-11-BN | 4096 | 0.703 | 0.105 (927) | 0.00–1.00 (1.0 pts) | 0.059 / 0.294 | 0.235 | 1.00 | 0.092 | 0.42 / 1.00 (-0.57) | 4.2 (2.3 → 4.8) | yes |
| **same architecture, different training recipe** | | | | | | | | | | | |
| ResNet-50 (V2 recipe) | 2048 | 0.814 | 0.185 (874) | 0.00–1.00 (0.2 pts) | 0.131 / 0.465 | 0.334 | 1.00 | 0.056 | 0.38 / 1.00 (-0.62) | 0.5 (1.9 → 7.8) | yes |
| **Stylized-ImageNet controls (Geirhos et al., 2019)** | | | | | | | | | | | |
| ResNet-50 SIN | 2048 | 0.582 | 0.810 (758) | 0.00–1.00 (3.9 pts) | 0.701 / 0.867 | 0.166 | 0.99 | 0.140 | 0.79 / 0.90 (-0.11) | 39.5 (28.7 → 49.7) | yes |
| ResNet-50 SIN+IN | 2048 | 0.743 | 0.363 (802) | 0.00–1.00 (2.4 pts) | 0.256 / 0.606 | 0.350 | 1.00 | 0.170 | 0.59 / 0.99 (-0.40) | 11.7 (7.1 → 18.5) | yes |
| ResNet-50 SIN+IN, ft IN | 2048 | 0.770 | 0.227 (797) | 0.00–1.00 (1.9 pts) | 0.170 / 0.447 | 0.277 | 1.00 | 0.154 | 0.41 / 0.99 (-0.58) | 5.9 (4.6 → 14.7) | yes |

<!-- ED_TABLES_END -->

## References

Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle,
L. T., Nau, M., Caron, B., Pestilli, F., Charest, I., Hutchinson, J. B.,
Naselaris, T., & Kay, K. (2022). A massive 7T fMRI dataset to bridge cognitive
neuroscience and artificial intelligence. *Nature Neuroscience*, 25(1), 116–126.
https://doi.org/10.1038/s41593-021-00962-x

Bakhtiari, S., Mineault, P., Lillicrap, T., Pack, C., & Richards, B. (2021). The
functional specialization of visual cortex emerges from training parallel
pathways with self-supervised predictive learning. *Advances in Neural
Information Processing Systems*, 34, 25164–25178.

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and
Psychological Measurement*, 20(1), 37–46.
https://doi.org/10.1177/001316446002000104

Derrington, A. M., & Lennie, P. (1984). Spatial and temporal contrast
sensitivities of neurones in lateral geniculate nucleus of macaque. *The Journal
of Physiology*, 357(1), 219–240. https://doi.org/10.1113/jphysiol.1984.sp015498

Derrington, A. M., Krauskopf, J., & Lennie, P. (1984). Chromatic mechanisms in
lateral geniculate nucleus of macaque. *The Journal of Physiology*, 357(1),
241–265. https://doi.org/10.1113/jphysiol.1984.sp015499

Dupré la Tour, T., Eickenberg, M., Nunez-Elizalde, A. O., & Gallant, J. L.
(2022). Feature-space selection with banded ridge regression. *NeuroImage*, 264,
119728. https://doi.org/10.1016/j.neuroimage.2022.119728

Ferrera, V. P., Nealey, T. A., & Maunsell, J. H. R. (1994). Responses in macaque
visual area V4 following inactivation of the parvocellular and magnocellular LGN
pathways. *Journal of Neuroscience*, 14(4), 2080–2088.
https://doi.org/10.1523/JNEUROSCI.14-04-02080.1994

Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K., &
Ganguli, S. (2017). A theory of multineuronal dimensionality, dynamics and
measurement. *bioRxiv*. https://doi.org/10.1101/214262

Gavrikov, P., Lukasik, J., Jung, S., Geirhos, R., Lamm, B., Mirza, M. J., Keuper,
M., & Keuper, J. (2024). Are vision language models texture or shape biased and
can we steer them? *arXiv preprint*, arXiv:2403.09193.

Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel,
W. (2018a). ImageNet-trained CNNs are biased towards texture; increasing shape
bias improves accuracy and robustness. *International Conference on Learning
Representations* (ICLR 2019). https://openreview.net/forum?id=Bygh9j09KX

Geirhos, R., Meding, K., & Wichmann, F. A. (2020). Beyond accuracy: Quantifying
trial-by-trial behaviour of CNNs and humans by measuring error consistency.
*Advances in Neural Information Processing Systems*, 33.
https://doi.org/10.48550/arXiv.2006.16736

Geirhos, R., Narayanappa, K., Mitzkus, B., Thieringer, T., Bethge, M.,
Wichmann, F. A., & Brendel, W. (2021). Partial success in closing the gap
between human and machine vision. *Advances in Neural Information Processing
Systems*, 34, 23885–23899. https://doi.org/10.48550/arXiv.2106.07411

Goodale, M. A., & Milner, A. D. (1992). Separate visual pathways for perception
and action. *Trends in Neurosciences*, 15(1), 20–25.
https://doi.org/10.1016/0166-2236(92)90344-8

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. *IEEE Conference on Computer Vision and Pattern Recognition*,
770–778. https://doi.org/10.1109/CVPR.2016.90

Hermann, K., Chen, T., & Kornblith, S. (2020). The origins and prevalence of
texture bias in convolutional neural networks. *Advances in Neural Information
Processing Systems*, 33. https://doi.org/10.48550/arXiv.1911.09071

Islam, M. A., Kowal, M., Esser, P., Jia, S., Ommer, B., Derpanis, K. G., &
Bruce, N. (2021). Shape or texture: Understanding discriminative features in
CNNs. *International Conference on Learning Representations*.
https://doi.org/10.48550/arXiv.2101.11604

Jagadeesh, A. V., & Gardner, J. L. (2022). Texture-like representation of objects
in human visual cortex. *Proceedings of the National Academy of Sciences*, 119,
e2115302119. https://doi.org/10.1073/pnas.2115302119

Jagadeesh, A. V., & Livingstone, M. S. (2024). Texture bias in primate ventral
visual cortex. *ICLR 2024 Workshop on Representational Alignment*.
https://openreview.net/forum?id=D6YvyvZ7P7

Jang, H., & Tong, F. (2021). Convolutional neural networks trained with a
developmental sequence of blurry to clear images reveal core differences between
face and object processing. *Journal of Vision*, 21(12), 6.
https://doi.org/10.1167/jov.21.12.6

Kaplan, E., & Shapley, R. M. (1982). X and Y cells in the lateral geniculate
nucleus of macaque monkeys. *The Journal of Physiology*, 330(1), 125–143.
https://doi.org/10.1113/jphysiol.1982.sp014333

Kingma, D. P., & Ba, J. (2017). Adam: A method for stochastic optimization.
*arXiv*. https://doi.org/10.48550/arXiv.1412.6980

Komatsu, H., & Goda, N. (2018). Neural mechanisms of material perception: Quest
on Shitsukan. *Neuroscience*, 392, 329–347.
https://doi.org/10.1016/j.neuroscience.2018.09.001

Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational
similarity analysis — connecting the branches of systems neuroscience.
*Frontiers in Systems Neuroscience*, 2, 4.
https://doi.org/10.3389/neuro.06.004.2008

Lakens, D. (2017). Equivalence tests: A practical primer for t tests,
correlations, and meta-analyses. *Social Psychological and Personality Science*,
8(4), 355–362. https://doi.org/10.1177/1948550617697177

Lannelongue, L., Grealey, J., & Inouye, M. (2021). Green algorithms: Quantifying
the carbon footprint of computation. *Advanced Science*, 8(12), 2100707.
https://doi.org/10.1002/advs.202100707

Livingstone, M. S., & Hubel, D. H. (1987). Psychophysical evidence for separate
channels for the perception of form, color, movement, and depth. *Journal of
Neuroscience*, 7(11), 3416–3468.
https://doi.org/10.1523/JNEUROSCI.07-11-03416.1987

Lu, Z., Thorat, S., Cichy, R. M., & Kietzmann, T. C. (2026). Adopting a human
developmental visual diet yields robust and shape-based AI vision. *Nature
Machine Intelligence*, 8(5), 735–748.
https://doi.org/10.1038/s42256-026-01228-6

Manocchio, B. (2026). *Investigating two-stream artificial neural networks using a developmental visual diet* [Bachelor's thesis, University of Osnabrück, Institute of Cognitive Science; supervised by Z. Lu and T. C. Kietzmann].

Mishkin, M., Ungerleider, L. G., & Macko, K. A. (1983). Object vision and spatial
vision: Two cortical pathways. *Trends in Neurosciences*, 6, 414–417.
https://doi.org/10.1016/0166-2236(83)90190-X

Nassi, J. J., & Callaway, E. M. (2009). Parallel processing strategies of the
primate visual system. *Nature Reviews Neuroscience*, 10(5), 360–372.
https://doi.org/10.1038/nrn2619

Okazawa, G., Tajima, S., & Komatsu, H. (2015). Image statistics underlying
natural texture selectivity of neurons in macaque V4. *Proceedings of the
National Academy of Sciences*, 112(4), E351–E360.
https://doi.org/10.1073/pnas.1415146112

Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and
the power approach for assessing the equivalence of average bioavailability.
*Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
https://doi.org/10.1007/BF01068419

Subramanian, A., Sizikova, E., Majaj, N. J., & Pelli, D. G. (2023). Spatial-frequency
channels, shape bias, and adversarial robustness. *Advances in Neural Information
Processing Systems*, 36.

Tartaglini, A. R., Vong, W. K., & Lake, B. M. (2022). A developmentally-inspired
examination of shape versus texture bias in machines. *Proceedings of the 44th
Annual Conference of the Cognitive Science Society*, 1180–1186.

Thorat, S., & Kietzmann, T. C. (2024). MiniEcoset.
https://doi.org/10.17605/OSF.IO/MSNA2

Wang, L., Mruczek, R. E. B., Arcaro, M. J., & Kastner, S. (2015). Probabilistic
maps of visual topography in human cortex. *Cerebral Cortex*, 25(10), 3911–3931.
https://doi.org/10.1093/cercor/bhu277

## Supplementary Note S1 — claims deliberately not made

- That the two-stream split is what makes shape bias a readout property: networks with no split show the same dissociation — one gain on their shape-predictive subspace moves the rate by 0.25–0.33 in six ImageNet networks within 1.5 points of top-1, and in our trunks extrastriate cortex loads on the texture-leaning half relative to V1 — with the cortical part failing in ImageNet ResNet-50 under the unit-level rule (Fig. 6; Results 2.4).
- That the two-stream split causes a shape-bias advantage over a single stream: at matched capacity the arms do not separate at n = 3, and capacity alone moves the rate by a comparable amount (Results 2.4).
- Any dorsal/ventral stream mapping. The thesis's framing “magno→dorsal, parvo→ventral” was tested rather than dropped: the pre-declared dorsal−ventral interaction is +0.0245 on the diet model and +0.0122 (p = .195) without the low-pass front end, so it does not survive its own control. Read directly on the Wang atlas, dorsal (V3A/B, IPS0–2) and ventral (VO1/2, PHC1/2) regions are indistinguishable under both arms (dorsal − ventral +0.015 with the diet, +0.005 without, p > .1; Extended Data Fig. 4c): the diet's magno stream contributes no unique variance anywhere beyond V3.
- hMT+ specificity: at n = 8 the two statistics disagree (p = .195 normalised, .039 raw) and the effect is dead in the blur control.
- A foveal/peripheral headline: the foveal band's sign flips when the blur is removed (−0.057 → +0.051), so it is a front-end artefact.
- That the chromatic stream predicts colour-biased cortex: pooled VO is +0.0045 (p = .46), bounded within ±0.017; hV4's parvo preference is weak (−0.012, 2/8). Substituting the koniocellular axis makes the prediction worse, and worse specifically in the colour set (hV4 −0.029, VO −0.024, each p = .0078, 0/8).
- Where in cortex the diet's shape-bias effect lands: a diet-specific hV4/LO step does not survive the full control set.
- That integration produces human-like shape bias, or that the diet improves neural predictivity: it does neither.
- That the diet makes the model's image-level difficulty profile more human-like: the crossed 2×2 shows train/test matching, not a weights effect.
- That human cortex uses this mechanism: a single-stream account in which texture-like ventral representations feed shape-biased behaviour directly is not excluded by these data.

Why the three regional claims above failed together, and what it implies for the survivor: without the low-pass front end essentially every ROI is magno-preferring, so the blur manufactures apparent regional specificity by handicapping the magno stream until the parvo stream can win somewhere — the same general-factor pathology that defeated the thesis's RSA, reappearing in encoding form. Any regional claim on these models must be shown to survive removal of the blur, and the ventral gradient is reported because it does.

## Supplementary Note S2 — the koniocellular control

The thesis omitted the koniocellular pathway for simplicity. Because L−M fails to predict colour-biased cortex, two readings were live: that the chromatic stream does not predict colour cortex, or that L−M is the wrong chromatic axis. Swapping S−(L+M) into stream 2 of the untrained network, contrast-matched to L−M per image, with the luminance stream bit-identical, makes the ventral prediction worse and worse where colour lives (hV4 −0.029, VO −0.024, the pre-declared colour set −0.026, each p = .0078 with 0/8 subjects positive on both statistics; colour set minus the other 17 ROIs −0.014, p = .0078), unchanged when the channel keeps its native 2.05× contrast. Both cone-opponent axes have now been tested against colour-biased cortex and neither predicts it. These fits are untrained and single-seed, no third stream was added, and no blur was applied to the S channel; whether a three-band model beats a two-band one is a separate question requiring three-way variance partitioning.

## Supplementary Note S3 — the cortical gradient against its nuisance axes and the instrument's bias

Results 2.3 states the outcome; this note carries the analyses. See Extended Data Fig. 3 and Extended Data Tables 8 and 9.

**The two nuisance axes that co-vary with hierarchical rank.** Receptive-field size and eccentricity coverage both rise from V1 to hV4, and the diet's magno stream is a low-pass instrument by construction, so a stream-preference gradient that merely tracked receptive-field size would be expected from the front end alone. We tested this three ways (Extended Data Fig. 3; Extended Data Table 9). First, the gradient computed within each of NSD's five eccentricity bands — V1 and hV4 voxels compared only with voxels of the same eccentricity — is negative under the diet in every band (6/7 subjects in the foveal band, 8/8 in the other four) while the no-diet arm stays between −0.10 and +0.07 in every band; the diet arm's gradient does grow with eccentricity (−0.26 in the foveal band, −0.93 beyond 4°), so eccentricity modulates it, but the contrast is present at matched eccentricity and is therefore not an eccentricity contrast in disguise. Second, stratifying the same voxels by pRF size gives the same result within size tertiles (−0.71, −0.77 and −0.90 under the diet against −0.08 in each tertile without it). Third, regressing the per-voxel index on log pRF eccentricity and log pRF size jointly within subject (NSD's pRF estimates, voxels with pRF R² > 5%; the two predictors are only modestly collinear, VIF 1.66) shows that under the diet, stream preference moves toward parvo with increasing pRF size at fixed eccentricity (partial slope −0.086, 8/8 subjects, p = .0078) and toward magno with increasing eccentricity at fixed size (+0.119, 8/8, p = .0078), while the no-diet arm shows neither the size effect in that direction (+0.035, opposite sign) nor any eccentricity effect (−0.004, p = 1.00); the arm interaction on the eccentricity partial is +0.134 (8/8, p = .0078). A learning-free Gabor bank on the diet front end carries no such structure (+0.035, p = .15), so the dissociation is something training installs, not a property of the input filter. Within the diet arm, then, the ventral gradient and the pRF-size dependence are the same fact seen on two axes — larger receptive fields, further along the hierarchy, are increasingly explained by the texture-biased stream — and neither is present without the diet.

**The estimator's own bias points the wrong way.** The mean noise ceiling declines from V1v to hV4 (ρ = −0.89 against rank), a nuisance falling along the axis of the effect, so we simulated the instrument: synthetic voxels from each arm's fitted feature spaces, the measured noise ceilings imposed, the true magno − parvo difference set to zero everywhere. The recovered contrast is a small, positive floor (+0.038 with the diet's feature geometry, +0.051 without, 8/8 subjects each), flat across ROIs (+0.025 to +0.027 per ROI), and indistinguishable from zero on the raw statistic — opposite in sign to the observed effect and without its gradient. Because the floor is a signed bias, the right comparison is a paired displacement: the real contrast minus each subject's own floor is −0.871 with the diet and −0.130 without, each non-zero in every subject, the diet arm's 6.7 times larger (Extended Data Table 8). The floor subtraction relocates the recovery simulation's residual misfit rather than removing it, and we do not correct that residual, because the correction is under-specified: six variants swing the no-diet arm between 4/8 and 6/8 subjects and between p = .055 and p = 1.00, while the diet arm is non-zero in all eight subjects at p = .0078 under all six (Extended Data Table 8). The no-diet arm's non-zero gradient therefore rests on the floor correction alone, and since the simulation is not magnitude-faithful (it recovers the true contrast with a slope of 1.21 to 1.28), we read −0.834 as a sign-and-ordering result, not a calibrated effect size. The raw statistic per ROI is drawn beside the normalised one in Extended Data Fig. 3a; the diet arm's normalised index approaches −0.84 at hV4, near the index's bound, and the raw profile carries the same ordering without that bound.
