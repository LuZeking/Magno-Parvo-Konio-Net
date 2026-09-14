# Magnocellular- and parvocellular-like streams diverge under a developmental visual diet: shape-biased behaviour from a texture-biased ventral map

## Abstract

Humans name objects by shape when shape and texture disagree, yet the ventral visual stream is well described by texture statistics. We ask whether this is a contradiction, with a two-stream network whose inputs approximate the magnocellular and parvocellular signals reaching cortex: one ResNet-18 receives the luminance channel of a cone-opponent image, the other its L−M channel, and a developmental visual diet — a blur and contrast-sensitivity cut matched to a 48-month-old — is applied to luminance. Trained on one recognition objective, the streams end on opposite sides of the shape–texture boundary by different routes: the magno stream's shape bias is supplied by its input and survives training only under the diet; the parvo stream's texture bias is learnt. Fitted as encoding models to eight subjects of the Natural Scenes Dataset, it is the texture-biased stream whose unique variance rises through hV4 into every extrastriate region, along a gradient that training without the diet erodes. Behaviour, reading out from both streams, is shape-biased; re-weighting the two frozen streams moves the rate from 0.40 to 0.90, and the weight that best predicts extrastriate cortex (0.33) is not the one that best predicts human choices (0.67). Shape-biased behaviour therefore does not require shape-biased representations along the ventral pathway, and the diet buys neither accuracy, robustness nor agreement with human decisions. The cue-conflict rate is a property of the readout over components and should be reported as such, not as evidence about representations.

## 1. Introduction

Humans presented with an image whose shape and texture point to different categories name the shape: on the cue-conflict images of Geirhos et al. (2018a), ten observers did so on 96% of decided trials, while ImageNet-trained convolutional networks named the texture. Closing that gap has become a standard target for biologically motivated models (Hermann et al., 2020; Jang & Tong, 2021), and a developmental visual diet — training on input whose acuity, contrast sensitivity and chromatic sensitivity follow the human developmental trajectory — is among the most effective routes to it (Lu et al., 2026).

A second body of evidence sits uneasily beside the first. Along the ventral stream, object representations are well described by texture statistics: category-selective human cortex fails to discriminate natural images from feature-matched scrambles that preserve texture statistics while destroying spatial arrangement, yet still discriminates between categories (Jagadeesh & Gardner, 2022); macaque V4 responses to natural textures are predicted by their image statistics (Okazawa et al., 2015); and in inferotemporal cortex texture is the more decodable cue (Jagadeesh & Livingstone, 2024; Komatsu & Goda, 2018). The premise is scoped: the human evidence is one study with one manipulation, the rest is macaque, and “texture-like” means that a texture-statistics description suffices, not that shape is absent. The argument needs only that the ventral description is substantially texture-like.

If behaviour is shape-biased and the ventral stream texture-biased, then either the two measures disagree about what shape bias is, or behaviour is not a readout of one representation. The second horn is a hypothesis about architecture — other pathways contribute to the behavioural bias — that models of ventral-stream computation, which almost universally take a single RGB image into a single trunk, cannot express. Two facts about the primate visual system make it concrete. The visual system does not receive one image: it receives a luminance-dominated, spatially coarse magnocellular signal and a chromatically opponent, finer parvocellular one, separated before cortex and later recombined (Livingstone & Hubel, 1987; Kaplan & Shapley, 1982; Derrington et al., 1984; Nassi & Callaway, 2009; Ferrera et al., 1994). And that input is not stationary over development: acuity and contrast sensitivity improve over years, so the system first learns from a low-passed world. The two-stream hypothesis (Mishkin et al., 1983; Goodale & Milner, 1992) and the emergence of ventral- and dorsal-like streams in a two-pathway network trained on one objective (Bakhtiari et al., 2021) motivated a two-stream architecture; here the streams are defined by their subcortical-like inputs rather than by their cortical targets.

This paper asks two questions, in the order the originating study posed them (Manocchio, 2026). Can two streams fed magnocellular- and parvocellular-like inputs diverge in their internal representations while optimising a single objective and producing shape-biased behaviour? And does such a system say anything about why the ventral stream can be texture-like while behaviour is shape-biased? We build the system — two ResNet-18 trunks receiving the luminance and L−M channels of a cone-opponent image, with a developmental diet on the luminance channel (Fig. 1) — and examine it with linear probes and cue-conflict behaviour on the streams (Fig. 2), voxel-wise encoding models of eight subjects' 7T fMRI responses to natural scenes (Fig. 3), and the behavioural readout that combines the streams (Fig. 4). The streams diverge, by different routes; the texture-biased one carries extrastriate cortex; and the integrated behaviour is shape-biased because a behavioural rate is a property of a readout over components, not a report of any one of them. Because the account predicts that properties bundled with shape bias need not travel with it, we test that too, and find that robustness, agreement with human decisions and representational dimensionality all come apart from the rate (Fig. 5).

Relative to the originating thesis, three things changed. Every condition is now trained with three seeds and every claim states its replication unit; the representational-similarity analysis that had found no stream-to-brain mapping is replaced by variance partitioning of encoding models, which removes the global stream preference that had defeated it; and the stream weighting that the thesis fixed by hand is swept, so that the dependence of behaviour on the readout is measured rather than assumed.

## 2. Results

### 2.1 The model, and what the diet costs

The network splits its input once, before any convolution (Fig. 1a). Each image is converted to the DKL cone-opponent space; stream 1 receives the luminance channel, stream 2 the L−M channel, and each is a ResNet-18 trunk. Under the developmental visual diet the luminance channel passes through a Gaussian blur (σ = 4 px) and the scale-free contrast-sensitivity cut of Lu et al. (2026) at a fixed age of 48 months, so that the transform is separated from the trajectory (Methods 4.1). The trunks are initialised bit-identically, and a control in which both streams receive the same image leaves all 120 tensor pairs identical after 150 epochs, so every stream difference below is attributable to the inputs. A linear classifier reads out w·y₁ + (1−w)·y₂ of the two streams' penultimate features, with w = 0.5 during training.

Four front-end conditions are used throughout (Fig. 1a, table): the diet; the no-diet control; a channel-swap control in which the identical transform is applied to the L−M channel instead; and a `parvo_highsf` control in which stream 2 additionally receives untransformed luminance, because real parvocellular cells carry both. Networks were trained on MiniEcoset (112 classes; Thorat & Kietzmann, 2024) for 150 epochs, three seeds per condition (Methods 4.2).

The diet costs accuracy (Fig. 1c). The no-diet conditions converge to 0.70 validation top-1; the diet arm reaches 0.556 at its best saved checkpoint against 0.701 and, uniquely, degrades after its peak, ending at 0.445 (range 0.347–0.527). Because that degradation would otherwise handicap the diet arm, behavioural and out-of-distribution results use each seed's best saved checkpoint, stated wherever the policy matters (Methods 4.7). Two single-stream controls trained with the identical diet — a ResNet-18 (11.2M parameters) and a ResNet-34 parameter-matched to the two-stream model (21.3M against 22.4M) — reach 0.599 and 0.623, so the accuracy cost of the split is not a capacity artefact (Fig. 4c).

### 2.2 The two streams specialise in opposite directions, by different routes

Read out separately with a linear probe on the cue-conflict images, the two streams of the diet-trained network end on opposite sides of the shape–texture boundary: shape − texture accuracy +0.819 for the magno stream and −0.518 for the parvo stream, means of three seeds (Fig. 2a). Behaviour agrees: read out alone, the magno stream's cue-conflict shape bias is 0.878 and the parvo stream's 0.418, against 0.675 for the integrated readout (Table 1). This is the divergence the thesis reported on a single model per condition, now replicated across seeds.

**Table 1. The two streams end on opposite sides of the boundary, by different routes.**

| | magno stream (luminance + transform) | parvo stream (L−M) | integrated |
|---|---|---|---|
| linear probe, shape − texture (0 = sign flip), 3 seeds | **+0.819** | **−0.518** | — |
| cue-conflict shape bias (0.5 = parity), 3 seeds | **0.878** | **0.418** | **0.675** |
| linear probe, untrained, seed 0 | **+0.796** | −0.087 | — |
| linear probe, trained, seed 0 | **+0.810** | **−0.482** | — |
| *contribution of learning*, seed 0 | **+0.014** | **−0.395** | — |

The decomposition arises asymmetrically. Scoring the same probe on the untrained network with the same front end separates two routes (Fig. 2a, Table 1). The magno stream's shape bias is essentially complete before any weight is updated — the low-pass and contrast-sensitivity transform does it — and training adds +0.014; the parvo stream begins barely biased and acquires its texture preference over training. The shape-biased component is supplied by the input; the texture-biased component is built by learning. What ordinary training does to the input's contribution is the decisive contrast: with the transform removed, the untrained luminance stream starts lower (+0.417) and training drives it across the sign flip to −0.226. The unblurred luminance stream does not merely fail to become shape-biased; it becomes texture-biased. The parvo stream ends texture-biased whether or not the other stream is dieted (−0.518 against −0.504).

Which property of the input supplies the shape bias? Decomposing the front end into two factors — which carrier a stream receives (achromatic luminance or chromatic L−M) and whether the developmental transform is applied to it — gives four cells, two of which the swap and `parvo_highsf` conditions provide (Fig. 2b, c). On the probe, the chromatic carrier gives −0.504 without the transform and −0.145 with it; the achromatic carrier gives −0.271 without and +0.819 with, seed ranges 0.014–0.043. Both single factors move a stream toward neutral; only the conjunction crosses zero, and it crosses all the way, by +0.936 with no seed overlap; cue-conflict shape bias on the same four cells replicates the interaction (Extended Data Table 4). The shape bias requires the developmental transform applied specifically to an achromatic channel — a relational fact about what a stream receives, not a property of a stream called “magno”.

Two qualifications belong here. First, cue-conflict shape bias is computed only over images a readout decides as the shape or the texture category, a count that varies four-fold across cells (165 to 583 of 1,200; Fig. 2d), so the three non-conjunction behavioural cells are read as “no strong preference”. The probe classifies every image and gives the same pattern, and so does a forced two-way estimator that scores every image by whether the shape-category logit exceeds the texture-category logit: it compresses values toward 0.5 but preserves every ordering in this paper (the four cells read 0.44, 0.51, 0.35 and 0.76; Extended Data Table 10). Second, the diet-trained models are less accurate. We pre-registered the fatal outcome — a no-diet model at the diet's accuracy with a shape-biased magno stream — and retrained the no-diet network with per-epoch checkpointing, selecting by rule the epoch nearest the diet's accuracy (0.548 against 0.556): matching accuracy closes 7.6% of the probe gap and 7.0% of the behavioural gap, and the matched magno stream stays on the texture side (Fig. 2e, f; n = 1).

### 2.3 In cortex, the texture-biased stream carries the ventral end of the hierarchy

The thesis had asked whether the two streams align with different cortical streams, using representational similarity analysis (Kriegeskorte et al., 2008) against seven regions of the Natural Scenes Dataset, and found that whichever stream was the more predictive correlated better with every region. We replicated that result, understood it as a general-factor problem — a correlation cannot separate a global stream preference from regional structure — and replaced the instrument: voxel-wise banded-ridge encoding models for each of the eight NSD subjects, one feature band per stream, with every voxel's cross-validated explained variance partitioned into the part only the magno band accounts for (U_m) and the part only the parvo band does (U_p) (Methods 4.5). The stream-preference profile — the per-voxel index (U_m − U_p)/(U_m + U_p), median over reliable voxels per ROI — across V1v → hV4 is Fig. 3a.

The two trained models sit on opposite sides of the global stream-preference axis: without the diet every retinotopic area is magno-explained, with it every area is parvo-explained. That global difference is the general factor that defeated the correlational analysis, and it is not the result. The result is the shape of the profile. Only the diet arm descends steeply from V1 to hV4 — early visual cortex relatively more magno-explained, hV4 relatively more parvo-explained, the direction a luminance/low-spatial-frequency versus chromatic split predicts — and the descent belongs to the model whose global preference points the other way, which a model simply preferring one stream everywhere could not produce.

Quantified as the contrast hV4 − mean(V1v, V1d) of the per-ROI median (Fig. 3b; Extended Data Table 1), the gradient is −0.745 to −0.834 across the three diet-trained seeds against −0.049 to −0.096 without the diet, with the same separation on the raw statistic U_m − U_p; all six comparisons reach the exact Wilcoxon floor for eight subjects (p = .0078) with complete seed separation. The per-subject Spearman ρ between hierarchical rank and stream preference is −0.964 to −0.973 with the diet and −0.503 to −0.606 without (Fig. 3c; Extended Data Table 2). Because the rank vector [1, 1, 2, 2, 3, 3, 4] admits exactly 630 labellings of the seven ROIs, the permutation null is enumerable; it calibrates the extremity of a result that was not pre-specified — the gradient is the survivor of seven regional hypotheses, six of which failed (below) — and does not test a prediction. The diet arm's ρ is the single most extreme of all 630 in every seed and on both statistics (p = 1/630 = .0016); the no-diet arm does not reach that floor (p = .019 to .092), so we claim nothing there.

The diet does not create the gradient; it decides what training does to the one the input supplies. An untrained network on the diet front end already sits at −0.544 (8/8 subjects, p = .0078), and on the no-diet front end at almost the same value (−0.479), so the two front ends alone impose no different gradient. Training with the diet steepens it; training without it erodes it to near zero (Fig. 3b). Correcting each arm for the signed bias of the estimator itself — a simulated floor of the wrong sign, +0.05 and +0.04 — leaves a paired displacement of −0.871 with the diet against −0.130 without, each non-zero in every subject. The gradient survives the two front-end controls and replicates across codebases: an independently trained checkpoint from the thesis's own pipeline, with its progressive developmental schedule, gives −0.793.

The fitted models allow one more test that needs neither probe nor readout: passing the 1,200 cue-conflict images through models fitted on natural scenes only yields a per-voxel predicted shape-bias index, whose magno − parvo difference is +1.53 to +1.57 with the diet against +0.08 to +0.11 without (8/8 subjects, p = .0078 in every seed), carried entirely by the magno stream (Fig. 3d; Extended Data Table 3) — the encoding models' own verdict on the stream contrast of Fig. 2.

**The gradient survives its nuisance axes and the instrument's own bias.** Receptive-field size, eccentricity coverage and the noise ceiling all change from V1 to hV4, and the diet's magno stream is a low-pass instrument by construction. The diet gradient is negative within every one of NSD's five eccentricity bands and every pRF-size tertile while the no-diet arm stays near zero in each; it grows with eccentricity (−0.26 in the fovea to −0.93 beyond 4°) but is present at matched eccentricity, the same dissociation holds on continuous pRF eccentricity and size within subject, and the estimator's own simulated bias is small, flat across ROIs and of the wrong sign (Extended Data Fig. 3; Extended Data Tables 8 and 9; the full account is Supplementary Note S3).

**The gradient does not stop at hV4, and it does not turn.** The stored fits cover every voxel of NSD's general visual mask, so the same index can be read on the regions the retinotopic hierarchy leads into (Extended Data Fig. 4; Extended Data Table 11). Under the diet the index runs from −0.05 in V1v through −0.81 in hV4 and then sits between −0.80 and −0.94 in every extrastriate region — ventral VO1/2 and PHC1/2, lateral LO1/2 and TO1/2, dorsal V3A/B and IPS0–2, NSD's stream parcels and the fLoc face, place, body and word regions alike. Every region-minus-V1 contrast is negative in 8/8 subjects (ventral −0.76, lateral −0.75, dorsal −0.75; p = .0078 each) and no region-minus-region contrast departs from zero under either arm (dorsal − ventral +0.015 with the diet, +0.005 without; all p > .05). The components show what saturates: with the diet the parvo band's unique variance rises from 0.10 in V1v to 0.26 in PHC2 while the magno band's collapses to about 0.01 beyond V3; without the diet the luminance stream rises over the same range and the L−M stream stays near 0.01 (Extended Data Fig. 4b). The texture-biased stream therefore carries extrastriate cortex as a whole, not the ventral end specifically, and the thesis's magno→dorsal, parvo→ventral mapping fails because the diet's magno stream contributes no unique variance anywhere beyond V3.

Three things this section does not settle are stated here rather than left to be found. The diet predicts BOLD worse, not better: median r² normalised by the noise ceiling is 0.405 against 0.464 without the diet, in 8/8 subjects (Fig. 3e), so the diet's case rests on stream differentiation and the gradient, not on prediction accuracy. The two front-end controls are degenerate on this instrument (swap: U_p = 0.006; `parvo_highsf`: U_m = 0.007), so the correct statement is that only the diet on luminance yields two streams that both carry substantial unique cortical variance and a hierarchical gradient between them (Fig. 3a). And the gradient was not pre-specified: it is the survivor of seven regional hypotheses, of which six failed — including the thesis's magno→dorsal mapping, an hMT+ specificity and a foveal/peripheral contrast whose sign flipped when the blur was removed — reported because it alone survives removal of that manipulation, and read as sign and ordering rather than as a calibrated effect (Supplementary Notes S1 and S3; the subject-01 influence check is in Extended Data Table 1).

### 2.4 Behaviour reads out from both streams, and the rate is a property of the readout

The two streams are oppositely biased and the texture-biased one carries extrastriate cortex. What remains is what the behavioural readout does with them. The integrated readout's cue-conflict shape bias is 0.675 ± 0.025 with the diet against 0.416 ± 0.020 without, three seeds each; the thesis's own checkpoint reproduces at 0.697 against its reported 0.677. On the identical images and statistic the ten human observers give 0.958 ± 0.019 (range 0.919–0.987) and the CNNs of the Geirhos et al. (2018a) release 0.429, 0.312 and 0.172, values our pipeline reproduces. The diet-trained magno stream, at 0.878, closes most of the distance from the CNN regime toward the observers; no match is claimed — 0.878 is below the lowest individual human — and integration costs 0.203 of shape bias (0.878 → 0.675).

The integrated readout lands between its two components rather than at the shape-biased one. Given two oppositely biased components that outcome is close to arithmetically forced, so we do not present it as a prediction that could have failed. What can be measured is how much of the rate the readout controls. Sweeping w moves the readout while holding y₁ and y₂ exactly fixed (Fig. 4a): at 21 values of w, three seeds and both checkpoint policies, cue-conflict shape bias runs from 0.40 to 0.90, w = 0.5 reproduces the trained model's per-seed values exactly, and on the forced two-way estimator the two streams read out alone sit at 0.41 and 0.78 (Extended Data Table 10), so the span is not an artefact of the decided-image denominator. Because the classifier is affine and the weights sum to one, the sweep is exact interpolation between two measured endpoints; that says the affine readout adds nothing, not that the span would vanish under a readout that is not affine. Under a rule fixed before the test we refitted a two-layer nonlinear head, and a linear head as a control for the refit itself, on the frozen features at each of the 21 weights (Fig. 4b). The nonlinear head's median span is 0.466 against 0.508 for the frozen affine head (86% of the pre-registered baseline, threshold 80%), and the refitted-linear and nonlinear spans agree within 1%: the span is a property of having two oppositely biased components and any readout over them. Restricted to readouts within 5 points of peak validation accuracy the span narrows to 0.283 (0.421 to 0.760), so the extreme rates near w = 1 are bought with accuracy.

Fig. 4c places every model on one pair of axes. The two streams read out alone bracket the integrated readout (0.903 and 0.400 around 0.655), and the single-stream controls trained with the identical diet lie between: 0.567 for the half-capacity ResNet-18 and 0.626 for the parameter-matched ResNet-34, 0.029 below the two-stream model and inside the 0.059 that capacity alone moves the rate. We therefore attribute the ordering among these three models to capacity rather than to the split and claim no shape-bias advantage for the split; its accuracy cost is real, and the large number on the ladder is the integration cost, 0.248 from the best single component to the integrated readout (Extended Data Table 5).

Nothing here fits w to humans. We tested whether it could be: fitting w to each of the ten observers individually on a 0.001 grid, no observer's rate is reachable in any run (0 of 90 observer-runs; the closest approach misses the lowest observer by 0.027), and a weight fitted to an observer's trial-by-trial choices does not cross-validate (held-out agreement +0.026 against +0.023 for an arbitrary weight, beating it in 45 of 90). The rate is a free parameter of the readout, and nothing in this family of readouts selects the human operating point.

**Cortex and behaviour prefer different readout weights.** The sweep of Fig. 4a can be run on cortex, because the readout's input z(w) = w·y₁ + (1 − w)·y₂ is a feature vector like any other: for each w it was the design of a ridge encoding model of every voxel in NSD's general visual mask, with the split and folds of the banded model (Methods 4.5), and the w that maximises an ROI's median r² is the weight that region prefers (Extended Data Fig. 5; Extended Data Table 12). Under the diet that weight descends the hierarchy — 0.58 in V1, 0.50 in V2, 0.40 in V3, 0.35 in hV4 and 0.33–0.35 in every extrastriate group, ventral, lateral and dorsal alike — with the extrastriate mean below V1 in 8/8 subjects (Δ = −0.25, exact p = .0078). The weight that best reproduces the ten observers' trial-by-trial choices is 0.67 (range 0.49–0.89), and the two sets of optima do not overlap (extrastriate 0.27–0.40; Mann–Whitney p = .0004). On the axis of the behavioural sweep the two demands sit on opposite sides of the rate: the weight extrastriate cortex prefers gives a cue-conflict shape bias of 0.56, the weight behaviour prefers gives 0.78, and the human rate lies beyond either. Without the diet the gradient reverses (0.65 in V1 to 0.75–1.00 beyond hV4). The curves are shallow — at w = 0.5 every ROI is within 3% of its best r² — so this is an ordering of optima rather than a large loss, and it is the dissociation of Fig. 3 stated in the readout's own currency: the mixture that best describes extrastriate cortex is parvo-heavy and barely shape-biased, the mixture that best describes behaviour is magno-heavy, and both read out the same two representations.

This is the mechanism we propose for the paradox: a system holding a shape-biased and a texture-biased component, whose behaviour reads out from both, will look shape-biased behaviourally and texture-biased wherever the texture-biased component dominates — and in this model of the subcortical split, the component that dominates the ventral end is the texture-biased one. It is an existence proof with a cortically grounded demonstration, not a measurement of the human readout.

### 2.5 Properties that do not follow from shape bias

If shape bias is a property of a readout over components rather than of a representation, quantities the field treats as travelling with it need not. Over identical inputs, three do not (Fig. 5).

**Robustness.** The standard reading of Geirhos et al. (2018a, 2021) is that shape bias buys out-of-distribution robustness. We pre-registered that prediction and tested it on corruptions applied to the image before the front end, scored in the models' own label space at each seed's best saved checkpoint (Fig. 5a; Extended Data Fig. 1; Extended Data Table 6). The diet-trained model is less robust on four of the five corruption families it was not trained on — uniform noise (relative-accuracy AUC 0.634 against 0.729), high-pass (0.678 against 0.827), false colour (0.854 against 0.938) and rotation (0.574 against 0.615) — with complete seed separation on each, and equivalent within ±0.033 on phase scrambling. The only family on which the diet is more robust is low-pass (0.444 against 0.356), declared trained-in before the run because the diet is a low-pass blur; it is not more robust to contrast reduction (0.481 against 0.549) although its front end includes a contrast-sensitivity cut. Robustness appears exactly where it was trained in and nowhere else, which is the strongest available evidence that the nulls are real rather than a broken measurement. This does not contradict Geirhos et al., whose stylisation route yields shape bias together with robustness; it shows the coupling belongs to that route. It differs from the source of the diet, whose title reports robust, shape-based vision: we reproduce the shape-based half (their released model 0.822, our diet-trained magno stream 0.878) and not the robust half, and Methods 4.6 lists the three implementation differences any of which could carry the divergence.

**Agreement with human decisions.** Two decision-makers with the same marginal rate can agree on no individual image beyond what those rates force. Error consistency — Cohen's κ on trial-by-trial agreement — subtracts that forced agreement (Geirhos et al., 2020); across the ten observers the human–human ceiling is κ = 0.234. We pre-registered that the diet-trained model would be more human-consistent than its control. It is not, on either readout, and the sharpest true statement is stronger: our most shape-biased model, the diet-trained magno stream at 0.878, has the lowest error consistency of all seven models compared (κ = 0.0089, 3.8% of the ceiling, against 7.3% for VGG-16 at shape bias 0.172). Across all 189 readouts of the w sweep, with w chosen post hoc per run to maximise agreement, the best κ never exceeds 20.7% of the ceiling (Fig. 5b; Extended Data Table 7), and equivalence intervals fixed in advance exclude a κ change larger than 0.037 across the 0.71 span of shape bias the readouts cover. The measure does detect a correspondence when one is there: the same machinery recovers 24.1% of the split-half ceiling on whether an image resolves to either cue at all, against 8.7% on which cue is picked (Fig. 5c). That positive control invites a diet reading, since the diet arm exceeds the no-diet arm on difficulty with no seed overlap; we do not make it, because completing the crossed front-end 2×2 shows a train/test-matching pattern rather than a weights effect (Fig. 5d).

**Representational dimensionality.** The participation ratio of each stream's features on 1,000 NSD images per subject gives a magno/parvo ratio whose ordering across the four front ends is reproduced by a learning-free Gabor bank, so it tracks the bandwidth of the carrier a stream is handed. What the diet decides is which side of parity training ends on: every diet-trained seed ends magno-dominant (24/24 subject-seeds in each of three diet arms), including the `parvo_highsf` front end, which starts inverted in 8/8 subjects and which we had pre-registered would stay inverted, while every no-diet seed collapses to parity (Extended Data Fig. 2). No level is claimed, because the within-arm seed range exceeds the gaps between arms. The same case separates dimensionality from shape bias: `parvo_highsf`'s second stream becomes the more dimensional stream while remaining texture-biased (probe −0.266).

## 3. Discussion

**The two questions.** The thesis asked whether magnocellular- and parvocellular-driven streams inside one network could diverge while the network behaves shape-biased, and whether such a system bears on why the ventral stream can be texture-like while behaviour is not. To the first the answer is yes, replicated across seeds and with the mechanism located: divergence requires the developmental transform on an achromatic carrier, the shape-biased component is a property of the input that the diet protects from being unlearned, and the texture-biased component is built by learning. To the second the answer is an existence proof the thesis could not reach with its instruments: fitted to human cortex, it is the texture-biased stream whose unique variance carries the extrastriate end of the hierarchy — ventral, lateral and dorsal alike — while behaviour, reading out from both, is shape-biased. Read as competing claims about a single representation, shape-biased behaviour and texture-like ventral representations cannot both be right; read as claims about a readout over components, they are not in competition.

**What the readout account licenses.** A model matching the human shape-bias rate is routinely described as having human-like representations. This system shows that inference fails twice over. The same network holds components at +0.819 and −0.518 and behaves at 0.675, and one scalar over frozen representations moves the rate across most of the range the field's models occupy, so the rate does not identify a representation; and across models spanning a five-fold range of shape bias, trial-by-trial agreement with ten observers stays between 7% and 18% of the human–human ceiling, our most shape-biased stream lowest of all, so the rate does not identify a decision process either. Hermann et al. (2020) and Islam et al. (2021) established that a representation can hold more shape than its behaviour reveals; what is new here is the form of the dissociation — both representations held fixed, the rate moved by the readout alone, a refitted nonlinear head preserving the span — and the cortical grounding, which neither prior study has: the same mixing weight, fitted to cortex, prefers the parvo-heavy end in every region beyond V1, and fitted to human choices prefers the magno-heavy end (Extended Data Fig. 5). The methodological consequence is direct: shape bias should be reported as the readout statistic it is, alongside error consistency where human trial data exist, and should not on its own license a claim about representations.

**What a developmental constraint does.** The intuitive account of developmental low-pass input is informational: coarse input protects the learner from texture, so it learns shape. Our results do not support it. The transform removes bandwidth rather than adding structure, and the shape bias it produces is present before training. What the diet decides is what learning does not destroy: without it, training drives the luminance stream from +0.42 to −0.23; with it, +0.80 is still +0.81 after 150 epochs, and a second instrument shows the same signature, the side of parity replicating rather than a level. This is a constraint on the solution learning converges to, not a better input representation — consistent with the thesis's observation that the transformed magno stream becomes less predictive of the category as the parvo stream becomes more so.

**What comes apart, and what the diet costs.** Shape bias, robustness, dimensionality and human-likeness are discussed as a package. Over identical inputs this system separates them three times, and a readout account says why it should: none of the three is a property of a readout. The diet is also expensive — less accurate (0.556 against 0.701), less stable late in training, less robust on four of five untrained-on corruptions, a worse predictor of BOLD. Nothing here recommends a developmental diet as a training method; its interest is as a model of development, and the thesis's central mechanistic reading — that divergence and shape bias are bought by degrading the magno stream's evidence — survives in every instrument.

**Limitations.** The load-bearing cortical result rests on eight subjects from one dataset; its magnitude is read as sign and ordering only, and it is the survivor of seven regional hypotheses, disclosed above. Beyond hV4 the index saturates near its bound in every atlas, so the extension establishes that the effect is hierarchy-wide but cannot order extrastriate regions against one another. The mapping between our chromatic stream and texture-selective cortex is by explanatory role, not content: nothing here asserts that the ventral stream is chromatic, the prediction that the chromatic stream should win in colour-biased cortex fails (pooled VO +0.005, bounded within ±0.017), and substituting the koniocellular axis makes it worse (Supplementary Note S2). The `parvo_highsf` condition varies architecture alongside the front end. The diet is a fixed 48-month front end rather than a trajectory, though a schedule-trained checkpoint behaves like it on every instrument we tried. Whether a two-stream architecture specialises on its own is untestable here, because the trunks are bit-identical at initialisation. Human comparisons are against ten published observers and of aggregate rate; the readout weight is a hyperparameter that nothing selects. Two of the four pre-registered predictions failed and are reported as failures; claims deliberately not made are listed in Supplementary Note S1.

## 4. Methods

### 4.1 Front end

RGB images are converted to the DKL cone-opponent space through RGB→XYZ (D65)→LMS (Hunt–Pointer–Estévez)→DKL at 256 × 256 px, each channel normalised to [0, 1] with the training set's global range and quantised to 8 bits exactly as the training data were stored (Derrington et al., 1984). Stream 1 receives the luminance channel; under the diet it passes through an isotropic Gaussian blur (σ = 4 px, kernel 33, reflect padding) and the scale-free contrast-sensitivity cut of Lu et al. (2026) evaluated at a fixed age of 48 months (`cs_logspan_start` = 5 × 10⁻³, decomposition mode “keep”). Stream 2 receives the L−M channel untransformed. The four conditions are: `dkl_bcs` (diet: luminance + blur + CS | L−M); `dkl_base` (no diet: luminance | L−M); `dkl_bcs_swap` (luminance | L−M + blur + CS); `dkl_bcs_parvo_highsf` (luminance + blur + CS | L−M stacked with untransformed luminance, two channels). The last differs architecturally in its first-layer fan-in, and that difference is measured rather than argued away (Extended Data Fig. 2; Results 2.5). Evaluation stimuli, including NSD images and cue-conflict images, pass through the identical front end.

### 4.2 Architecture and training

Two ResNet-18 trunks (He et al., 2016), one per stream, initialised bit-identically, with a joint linear classification head over the weighted sum of their penultimate features (w = 0.5; cross-stream mixing off). MiniEcoset at 256 px, 112 classes (Thorat & Kietzmann, 2024). 150 epochs, batch 256, Adam (lr 1 × 10⁻⁴, β₁ = 0.9; Kingma & Ba, 2017), cross-entropy, with the augmentation set of Lu et al. (2026) applied on GPU (horizontal flip, rotation ±15°, colour jitter, random equalise, perspective, sharpness; random grayscale for RGB inputs only). Three seeds per condition; weights saved every 25 epochs. Each run used one NVIDIA RTX A5000 (24 GB); the 21 training runs behind this paper took 578 GPU-hours in total (19.5–51.4 h each, from the runs' own timing logs). All encoding, statistical and plotting work ran on CPU. Single-stream controls use one ResNet-18 or one ResNet-34 trunk fed the diet's luminance input with the identical schedule. The accuracy-matched control retrained `dkl_base` with per-epoch checkpointing and selected by rule the epoch nearest the diet arm's best-checkpoint accuracy. No hyperparameter search was run for any arm: every condition uses the training recipe of Lu et al. (2026) unchanged, and the only selection is the checkpoint policy stated above.

### 4.3 Behaviour and probe

Cue-conflict shape bias is computed on the Geirhos et al. (2018a) style-transfer set restricted to the 16 categories present in MiniEcoset and to images whose shape and texture categories differ (1,200 images). Logits are restricted to the 16 columns; the argmax is scored as a shape or texture response; images decided as neither are excluded from the denominator and their count is reported. Bootstrap 95% confidence intervals are over images. A second, forced two-way estimator keeps every image and scores whether the logit of the image's shape category exceeds that of its texture category; it is reported beside the primary estimator for every readout (Extended Data Table 10) and is computed from the same stored logits. The readout is full = classifier(w·y₁ + (1−w)·y₂), magno_only = classifier(w·y₁), parvo_only = classifier((1−w)·y₂), all through the same classifier and bias term; the w sweep evaluates full at 21 values of w on stored features. The refitted heads (Linear(512, 112) and Linear(512, 512)→ReLU→Linear(512, 112)) were trained on w·F₁ + (1−w)·F₂ of MiniEcoset training features at each w with an identical budget (Adam, lr 10⁻³, batch 1024, 20 epochs) and the same initialisation seed at every w. Human observers are the ten of the same release (1,280 trials each), scored by the identical rule; our recomputation reproduces the published 0.958 and the published AlexNet, GoogLeNet and VGG-16 values. Error consistency is Cohen's κ (Cohen, 1960) on the binary shape/texture choice over images both parties decided, with expected agreement p_m·p_h + (1−p_m)(1−p_h), reported as a percentage of the human–human κ over all 45 observer pairs (Geirhos et al., 2020). Per-image correspondence uses the observers' per-image shape rate and cue-resolution rate (split-half reliabilities r = .658 and .794 after category removal), the models' shape-minus-texture logit margin, w chosen on half the images and scored on the held-out half, as a percentage of each target's own ceiling. The linear probe reduces each stream's features to 256 PCA components and fits multinomial logistic regression (C = 1.0) under 5-fold stratified cross-validation to shape and texture labels separately; bias is shape accuracy minus texture accuracy on the same folds.

### 4.4 Effective dimensionality

Participation ratio (Σλ)²/Σλ² of the stream feature matrix on 1,000 NSD images per subject (Gao et al., 2017), computed Gram-side, reported as the magno/parvo ratio with parity at 1.0. Subjects are the replication unit within a seed; seeds are reported separately and never pooled. A learning-free reference is the same ratio computed on a Gabor oriented-energy bank applied to each stream's own transformed input.

### 4.5 Encoding

NSD (Allen et al., 2022) `betas_fithrf_GLMdenoise_RR` single-trial betas, converted to percent signal change and z-scored per session per voxel, `nsdgeneral` mask, eight subjects. Splits are by 73k image id with the shared-515 images held out. Stream features for each subject's stimuli are reduced by PCA to 1,000 components per stream (fit on training images only) and entered as two bands into a banded ridge regression (himalaya; Dupré la Tour et al., 2022) with 3-fold GroupKFold by image id, per-voxel alphas and per-voxel band weights. Unique variance U_m (U_p) is the drop in cross-validated r² when the magno (parvo) band is removed. Noise ceilings are NSD's `ncsnr`; voxels below 0.1 are excluded and any ROI cell with fewer than 20 surviving voxels is reported as missing. ROIs are the `prf-visualrois` labels of the Wang et al. (2015) atlas as distributed with NSD (V1v, V1d, V2v, V2d, V3v, V3d, hV4), with hierarchical rank [1, 1, 2, 2, 3, 3, 4]. The canonical-correlation spectrum between the two bands runs from ≈0 to 0.93, so each band retains directions the other cannot predict and unique variance is identifiable; a recovery simulation with the fitted feature geometry and a known across-ROI split returns the true ordering exactly (ρ = +1.000 in 8/8 subjects, both arms), which is why the gradient is reported as sign and ordering rather than as calibrated magnitude. Eccentricity bands are NSD's `prf-eccrois` labels (0–0.5°, 0.5–1°, 1–2°, 2–4°, > 4°); continuous pRF eccentricity, size and R² are NSD's population-receptive-field estimates in the same functional space, and the within-subject regressions of Extended Data Table 9 use z-scored log(eccentricity + 0.5) and log(size) over voxels with pRF R² > 5%. The per-voxel predicted shape-bias index is obtained by pushing the cue-conflict set through each fitted encoding model. For Extended Data Fig. 4 and Table 11 the same voxelwise fits were relabelled, without refitting, with the full Kastner2015 atlas as distributed with NSD (adding VO1/2, PHC1/2, LO1/2, TO1/2, V3A/B and IPS0–2; IPS3–5, SPL1 and FEF have no voxels inside `nsdgeneral`), with NSD's `streams` parcellation and with the fLoc face, place, body and word ROIs; regional means are unweighted means of the ROI medians, and region contrasts are tested per subject with the exact Wilcoxon test. For the cortex-optimal readout weight (Extended Data Fig. 5; Extended Data Table 12) the classifier's own input z(w) = w·y₁ + (1 − w)·y₂, where y₁ and y₂ are the 512-dimensional stream outputs the classifier mixes, was the design of a single-band ridge model of every `nsdgeneral` voxel for w ∈ {0, 0.05, …, 1}, with the same split, grouped folds and alpha grid as the banded model and columns z-scored on the training trials; because z(w) sums the streams dimension by dimension the ridge cannot undo the mixture, so w is a constraint on the design rather than a re-parameterisation of the two-band model. w*_cortex is the w that maximises the median test r² over an ROI's reliable voxels (seeds averaged per subject), the 1,024-dimensional concatenation [y₁, y₂] is the ceiling any mixture can reach, and the behaviour-optimal weight is the trial-fit w of Fig. 4 (Methods 4.4) for each of the ten observers. The koniocellular control substitutes S−(L+M) for L−M in stream 2 of the untrained seed-0 network, contrast-matched to L−M per image, with the magno band bit-identical across arms (gate max|Δr²_magno| < 10⁻⁴, passed at 0.000).

### 4.6 Out-of-distribution robustness

Corruptions are applied to RGB before the front end and scored in the native 112-class label space. Five families were declared untrained-on before the run (uniform noise, high-pass, phase scrambling, rotation, false colour) and two as trained-in positive controls (low-pass, contrast). Reported as the area under the relative-accuracy curve across severities, at each seed's best saved checkpoint. Three implementation differences from Lu et al. (2026) are declared: our front end is evaluated at a fixed 48 months rather than scheduled (the one we can rule out, since a schedule-trained arm behaves like the static diet); our diet is a transform on one stream's input of a two-stream network rather than an augmentation on a single trunk; and we train on MiniEcoset and score corruptions in that label space, applied before the front end.

### 4.7 Statistics

Within-subject comparisons use the paired exact Wilcoxon signed-rank test, n = 8 (exact two-sided floor p = .0078); seed-level comparisons use the exact Mann–Whitney U, 3 v 3 (floor p = 0.100, reached only on complete separation), and seeds are never pooled with subjects. Every cortical contrast is computed both on the raw per-voxel difference U_m − U_p and on the normalised index (U_m − U_p)/(U_m + U_p), each summarised by its median over reliable voxels per ROI, and a claim is made only where the two agree; disagreements are reported as unresolved. Equivalence uses two one-sided tests with bounds fixed before any interval was computed (Schuirmann, 1987; Lakens, 2017). No multiple-comparison correction is claimed: with 28 p-values and an exact floor of .0078, Holm–Bonferroni over any family larger than six cannot yield a rejection, so claims rest on complete separation, agreement of both statistics, replication across seeds and effect size (dz = −6.60 for the diet gradient against −1.09 without the diet). Behavioural and robustness results use each seed's best saved checkpoint unless stated; cortical fits use the final checkpoint. Four predictions — the front-end 2×2, the inverted-front-end prediction, the corruption families and the error-consistency hypothesis — were written into a version-controlled plan and committed before the corresponding analysis was run (internal pre-specification, not registry-based); two failed and are reported as failures.

### 4.8 Readout re-weighting in networks without a stream split

Pre-specified in `docs/PLAN_GENERALITY_2026-09-14.md`, committed before the analysis was run, with five directional predictions whose outcomes are all reported (Results 2.4; Extended Data Tables 13 and 14). For a frozen network with final layer logits = W·y + b, each unit j of y receives a shape-leaning score d_j = η²_shape(j) − η²_texture(j), where η² is the one-way fraction of the unit's variance over the 1,200 cue-conflict images lying between the 16 shape categories or the 16 texture categories; the median split of d gives the shape-leaning half S and the texture-leaning half T (equal-sized, as the two streams are). Fixed-head gain sweep: logits(w) = W·(g(w) ⊙ y) + b with g_j = 2w for j ∈ S and 2(1 − w) for j ∈ T, w ∈ {0, 0.05, …, 1}, so that w = 0.5 is the trained network; the split is defined on one half of the images, stratified by shape × texture pair, and the rate scored on the other, over 10 random splits in both directions. Our networks are scored by the estimators of Methods 4.3; torchvision ImageNet networks (ResNet-50, ResNet-152, VGG-16, ViT-B/16, ConvNeXt-S/B, IMAGENET1K_V1 weights, model-vs-human preprocessing) by the Geirhos 16-class rule, mean softmax probability over each category's ImageNet classes and argmax over the 16, which decides every image. Accuracy under the sweep is top-1 with the same gained features on held-out natural images: the MiniEcoset validation split for our networks and a stratified 10-per-class subset of the ILSVRC-2012 validation set for the ImageNet networks. Because a refitted linear head can undo a per-unit gain, the refit test mixes the halves lossily as the two-stream model does, z(w) = w·y_S + (1 − w)·y_T with the halves paired by rank of d, and refits the linear and two-layer heads of Methods 4.3 at each w. For cortex, S and T (split on all 1,200 images) are the two bands of the banded ridge model of Methods 4.5 on the penultimate units of the single-trunk networks, giving U_S, U_T and the index (U_S − U_T)/(U_S + U_T) per voxel; the cortex-optimal w of the lossy sum follows the protocol of Extended Data Fig. 5 on an 11-point grid (0, 0.1, …, 1). The `rgb_tsn` network — two trunks that receive the identical three-channel image and remain bit-identical — is one ResNet-18 on the untransformed image and is used as the no-diet single trunk. All of our networks enter this analysis at their final checkpoints, the policy of the cortical fits (Methods 4.7).

## Data availability

The Natural Scenes Dataset (Allen et al., 2022) is available at naturalscenesdataset.org under its own data-use agreement, which requires registration; we used the `betas_fithrf_GLMdenoise_RR` (b3) preparation, the `nsdgeneral` mask, the `prf-visualrois` and `prf-eccrois` labels and the pRF estimates as distributed with NSD. The Geirhos cue-conflict stimuli and the ten observers' trial-level responses are distributed at github.com/rgeirhos/texture-vs-shape (directory `raw-data/style-transfer-512-nomask-experiment/`) under CC BY 4.0; the same experiment is mirrored at github.com/bethgelab/model-vs-human, and the two mirrors were parsed by two independently written scripts that agree to four decimals on the human reference. MiniEcoset is at osf.io/msna2 (Thorat & Kietzmann, 2024). Derived result tables and fitted encoding-model summaries sufficient to reproduce every figure and Extended Data table are in the code repository, and will be deposited with a DOI on acceptance.

## Reporting summary

Further information on research design is available in the Nature Portfolio Reporting Summary linked to this article; a draft is `docs/REPORTING_SUMMARY.md` in the repository.

## Code availability

Training, front-end, encoding, behavioural and analysis code is at github.com/LuZeking/Magno-Parvo-Konio-Net (branch `mpk-rewrite`, commit as stamped on this manuscript's title page), under the BSD-3-Clause licence, with a pinned environment manifest (`requirements.txt`, Python 3.11) and the committed result files behind every figure; the figures of this paper regenerate with the scripts in `tmp_analysis/figs_v2/` and the manuscript's numbers are locked by `tmp_analysis/verify_new_claims.py`. The per-subject encoding outputs that Fig. 3 reads are large and are deposited alongside the derived tables. A private link is available to referees on request; a DOI-archived release accompanies acceptance.

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

**Extended Data Table 2 | Monotonicity of the profile and the exact permutation null.** Spearman ρ between hierarchical rank [1,1,2,2,3,3,4] and stream preference, mean over subjects; the null enumerates all 630 distinct labellings of the seven ROIs (one-sided in the pre-stated direction, floor 1/630 = .0016). Source: `results/hierarchy_permutation_null.json`.

| arm | seed | statistic | observed mean ρ | exact p | most extreme of 630 | null 2.5th pct |
|---|---|---|---|---|---|---|
| diet | 0 | norm | -0.964 | 0.0016 | yes | -0.754 |
| diet | 0 | raw | -0.964 | 0.0016 | yes | -0.743 |
| diet | 1 | norm | -0.973 | 0.0016 | yes | -0.750 |
| diet | 1 | raw | -0.973 | 0.0016 | yes | -0.743 |
| diet | 2 | norm | -0.966 | 0.0016 | yes | -0.748 |
| diet | 2 | raw | -0.966 | 0.0016 | yes | -0.756 |
| no diet | 0 | norm | -0.502 | 0.0286 | no | -0.502 |
| no diet | 0 | raw | -0.571 | 0.0206 | no | -0.559 |
| no diet | 1 | norm | -0.606 | 0.0190 | no | -0.546 |
| no diet | 1 | raw | -0.661 | 0.0190 | no | -0.594 |
| no diet | 2 | norm | -0.580 | 0.0365 | no | -0.616 |
| no diet | 2 | raw | -0.463 | 0.0921 | no | -0.594 |

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

**Extended Data Table 5 | Single-stream controls trained with the identical diet, and the readout ladder.** Best saved checkpoint per seed; mean [range] over three seeds. Accuracy is validation top-1 at the scored checkpoint. Sources: `results/cue_conflict_bestep/`, `results/cue_conflict_single34_bestep/`, `results/cue_conflict_single_bestep/`, run histories, `results/param_matched/`.

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

**Extended Data Table 7 | Error consistency (Cohen's κ) with the ten human observers.** The trained readout (w = 0.5) and the best readout over the 21 values of w, per run; κ as a percentage of the human–human ceiling (κ = 0.234, 45 pairs). Published CNN values are from Geirhos et al. (2020). Source: `results/sweep_error_consistency/`.

| run | shape bias at w = 0.5 | κ at w = 0.5 | % ceiling | best κ over w | % ceiling | w at best | rate at best |
|---|---|---|---|---|---|---|---|
| dkl_base_seed0_final | 0.421 | +0.0309 | 13.2 | +0.0404 | 17.3 | 1.00 | 0.438 |
| dkl_base_seed1_final | 0.433 | +0.0333 | 14.3 | +0.0421 | 18.0 | 0.75 | 0.429 |
| dkl_base_seed2_final | 0.394 | +0.0372 | 15.9 | +0.0412 | 17.6 | 0.35 | 0.410 |
| dkl_bcs_seed0_best | 0.594 | +0.0096 | 4.1 | +0.0178 | 7.6 | 1.00 | 0.890 |
| dkl_bcs_seed0_final | 0.692 | +0.0234 | 10.0 | +0.0405 | 17.3 | 0.90 | 0.857 |
| dkl_bcs_seed1_best | 0.718 | +0.0269 | 11.5 | +0.0425 | 18.2 | 0.35 | 0.614 |
| dkl_bcs_seed1_final | 0.686 | +0.0319 | 13.7 | +0.0437 | 18.7 | 0.45 | 0.651 |
| dkl_bcs_seed2_best | 0.652 | +0.0372 | 15.9 | +0.0483 | 20.7 | 0.85 | 0.847 |
| dkl_bcs_seed2_final | 0.646 | +0.0280 | 12.0 | +0.0405 | 17.3 | 0.75 | 0.784 |
| AlexNet (published) | 0.429 | +0.0320 | 13.7 | | | | |
| GoogLeNet (published) | 0.312 | +0.0250 | 10.7 | | | | |
| VGG-16 (published) | 0.172 | +0.0170 | 7.3 | | | | |

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
| no diet, accuracy-matched (epoch 10), magno alone | 0.455 | 532 | 0.370 | 1 |
| thesis checkpoint (progressive schedule), both streams | 0.697 | 620 | 0.636 | 1 |
| untrained, diet front end, magno alone | 0.513 | 154 | 0.510 | 1 |
| untrained, no-diet front end, magno alone | 0.568 | 169 | 0.508 | 1 |

**Extended Data Table 11 | Stream preference beyond hV4, on every NSD atlas.** Median over reliable voxels of the per-voxel index (Um − Up)/(Um + Up) (noise ceiling > 0.1, Um + Up > 10⁻³, ≥ 20 voxels per ROI per subject), mean over the eight subjects (n = subjects with enough voxels); trained arms are the mean over training seeds per subject. Positive = magno band, negative = parvo band. Atlases: Wang et al. (2015) Kastner2015 ROIs beyond hV4 (the V1–hV4 rows are Extended Data Table 1 and Fig. 3), NSD's `streams` atlas, and the fLoc category-selective ROIs. Regional contrasts and the exact tests are in Extended Data Fig. 4c and `results/figures_v2/ext_gradient.md`. Source: the stored voxelwise fits in `results/encoding_cc/` and `results/encoding/`, relabelled; no refitting.

| atlas | ROI | untrained (diet front end) | no diet (3 seeds) | diet (3 seeds) | swap | parvo + high-SF | thesis ckpt | voxels (median, diet) |
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

**Extended Data Table 13 | The readout span without a stream split: fixed-head gain sweep on ten frozen networks.** Each unit of a network's penultimate layer is scored by the fraction of its variance between shape categories minus that between texture categories on the 1,200 cue-conflict images (one-way η²); the median split gives a shape-leaning half S and a texture-leaning half T. logits(w) = head(g(w) ⊙ y) with g = 2w on S and 2(1 − w) on T, the network's own head untouched; w = 0.5 is the trained network. The split is defined on one stratified half of the images and the rate scored on the other (10 random splits, both directions; mean, with the range over splits and seeds in brackets). Rates by the paper's primary estimator (112-class, decided images) for our networks and the Geirhos 16-class probability aggregation for the ImageNet networks; “forced-estimator span” is the same sweep on the forced two-way estimator. “shape-leaning units” is the fraction of units with η²_shape > η²_texture on all images; for the two-stream concatenation the fraction of S that belongs to stream 1 is given in parentheses (P3). “w within 5 pts” is the w-range over which top-1 on held-out natural images (MiniEcoset validation; a stratified 10-per-class ILSVRC-2012 validation subset for the ImageNet networks) stays within 5 points of the trained value. “refit span” is the lossy-sum protocol z(w) = w·y_S + (1 − w)·y_T with a linear and a two-layer head refitted at every w (Methods 4.3 budget), our networks only. Pre-specification: `docs/PLAN_GENERALITY_2026-09-14.md`. Source: `/scratch/byrong/mpk/generality/behaviour/2026-09-14_gen/`.

| network | units | trained rate | w = 0 | w = 1 | span (held-out) [range] | ρ(w, rate) | forced-estimator span | shape-leaning units | w within 5 pts of trained top-1 | refit span linear / two-layer | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| two-stream, diet, streams concatenated (validation) | 1024 | 0.675 | 0.420 | 0.879 | 0.461 [0.39, 0.57] | 1.00 | 0.307 | 0.49 (S from stream 1: 0.99) | — | — | 3 |
| single ResNet-34, diet | 512 | 0.625 | 0.565 | 0.688 | 0.136 [0.03, 0.24] | 0.71 | 0.102 | 0.17 | 0.20–0.78 (top-1 0.626) | 0.06 / 0.05 | 3 |
| single ResNet-18, diet | 512 | 0.557 | 0.467 | 0.652 | 0.194 [0.11, 0.31] | 0.97 | 0.131 | 0.09 | 0.28–0.72 (top-1 0.547) | 0.09 / 0.06 | 3 |
| single ResNet-18, no diet (`rgb_tsn`) | 512 | 0.445 | 0.374 | 0.566 | 0.195 [0.11, 0.39] | 0.96 | 0.164 | 0.05 | 0.27–0.75 (top-1 0.678) | 0.10 / 0.09 | 3 |
| ImageNet ViT-B/16 | 768 | 0.399 | 0.371 | 0.455 | 0.086 [0.07, 0.11] | 0.98 | 0.138 | 0.01 | 0.00–1.00 (top-1 0.815) | — | 1 |
| ImageNet ResNet-152 | 2048 | 0.293 | 0.227 | 0.440 | 0.214 [0.18, 0.25] | 1.00 | 0.196 | 0.02 | 0.15–0.90 (top-1 0.785) | — | 1 |
| ImageNet ConvNeXt-B | 1024 | 0.269 | 0.229 | 0.343 | 0.115 [0.09, 0.14] | 0.99 | 0.173 | 0.00 | 0.00–1.00 (top-1 0.843) | — | 1 |
| ImageNet ConvNeXt-S | 768 | 0.248 | 0.211 | 0.335 | 0.124 [0.11, 0.15] | 0.99 | 0.179 | 0.00 | 0.00–1.00 (top-1 0.840) | — | 1 |
| ImageNet ResNet-50 | 2048 | 0.222 | 0.167 | 0.346 | 0.180 [0.14, 0.23] | 0.99 | 0.202 | 0.02 | 0.15–0.90 (top-1 0.760) | — | 1 |
| ImageNet VGG-16 | 4096 | 0.110 | 0.096 | 0.189 | 0.095 [0.07, 0.11] | 0.96 | 0.106 | 0.01 | 0.00–1.00 (top-1 0.711) | — | 1 |

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

Thorat, S., & Kietzmann, T. C. (2024). MiniEcoset.
https://doi.org/10.17605/OSF.IO/MSNA2

Wang, L., Mruczek, R. E. B., Arcaro, M. J., & Kastner, S. (2015). Probabilistic
maps of visual topography in human cortex. *Cerebral Cortex*, 25(10), 3911–3931.
https://doi.org/10.1093/cercor/bhu277

## Supplementary Note S1 — claims deliberately not made

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
