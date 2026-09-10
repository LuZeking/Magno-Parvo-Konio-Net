# Magnocellular- and parvocellular-like streams diverge under a developmental visual diet: shape-biased behaviour from a texture-biased ventral map

## Abstract

Humans name objects by their shape when shape and texture disagree, yet the ventral visual stream — the pathway most identified with object recognition — is well described by texture statistics. We ask whether this is a contradiction, using a two-stream convolutional network whose inputs approximate the magnocellular and parvocellular signals that reach cortex: one ResNet-18 stream receives the luminance channel of a cone-opponent (DKL) image, the other its L−M chromatic channel, and a developmental visual diet — a blur and contrast-sensitivity cut matched to a 48-month-old — is applied to the luminance channel. Trained on a single object-recognition objective, the two streams end on opposite sides of the shape–texture boundary by different routes: the magno stream's shape bias is supplied by its input and merely survives training, the parvo stream's texture bias is built by learning, and only the conjunction of an achromatic carrier with the developmental transform produces a shape-biased stream. Fitted as voxel-wise encoding models to eight subjects of the Natural Scenes Dataset, it is the texture-biased stream whose unique variance rises toward hV4, along a gradient that is the most extreme of all 630 labellings of the retinotopic hierarchy in every seed and that training without the diet erodes. Behaviour, reading out from both streams, is shape-biased (0.68 against 0.42 without the diet); re-weighting the two frozen streams moves the rate from 0.40 to 0.90 without changing either representation, and a refitted nonlinear readout preserves that span. Shape-biased behaviour therefore does not require shape-biased representations along the ventral pathway. The same system shows what the diet does not buy: the network is less accurate, less robust to four of five untrained-on corruptions, and no closer to human trial-by-trial decisions than its control. The account is an existence proof with a cortically grounded demonstration, and we state its limits.

## 1. Introduction

Humans presented with an image whose shape and texture point to different categories name the shape; on the cue-conflict images of Geirhos et al. (2018a), ten observers did so on 96% of decided trials, while ImageNet-trained convolutional networks named the texture. Closing that gap has become a standard target for biologically motivated models (Hermann et al., 2020; Jang & Tong, 2021), and a developmental visual diet — training a network on input whose acuity, contrast sensitivity and chromatic sensitivity follow the human developmental trajectory — is among the most effective routes to it (Lu et al., 2026).

A second body of evidence sits uneasily beside the first. Along the ventral stream, object representations are well described by texture statistics: category-selective human cortex fails to discriminate natural images from feature-matched scrambles that destroy spatial arrangement while preserving texture statistics, yet still discriminates between categories (Jagadeesh & Gardner, 2022); macaque V4 responses to natural textures are predicted by their image statistics (Okazawa et al., 2015); and in inferotemporal cortex texture is the more accurately decodable of the two cues (Jagadeesh & Livingstone, 2024; Komatsu & Goda, 2018). We state the scope of this premise plainly, because the paper rests on it: the human evidence is one study with one image-synthesis manipulation, the remainder is macaque, and “texture-like” means that a texture-statistics description suffices for the measured responses, not that shape is absent. Our argument needs only that the ventral description is substantially texture-like.

If behaviour is shape-biased and the ventral stream texture-biased, then either the two measures disagree about what shape bias is, or behaviour is not a readout of one representation. The second horn is a hypothesis about architecture: other pathways contribute to the behavioural bias, and models of ventral-stream computation, which almost universally take a single RGB image into a single trunk, cannot express that contribution. Two facts about the primate visual system make the hypothesis concrete. First, the visual system does not receive one image. It receives a luminance-dominated, spatially coarse magnocellular signal and a chromatically opponent, spatially finer parvocellular one, separated in the retina and lateral geniculate nucleus before either reaches cortex (Livingstone & Hubel, 1987; Kaplan & Shapley, 1982; Derrington et al., 1984), with the two later recombined rather than kept apart (Nassi & Callaway, 2009; Ferrera et al., 1994). Second, that input is not stationary over development: acuity and contrast sensitivity are poor in infancy and improve over years, so the visual system first learns from a low-passed world. The two-stream hypothesis of cortical vision (Mishkin et al., 1983; Goodale & Milner, 1992) and the demonstration that ventral- and dorsal-like streams can emerge in a two-pathway network trained on one objective (Bakhtiari et al., 2021) motivated a two-stream architecture; here the streams are defined by their subcortical-like inputs rather than by their cortical targets.

This paper asks two questions, in the order the originating study posed them (Manocchio, 2026). Can two streams fed magnocellular- and parvocellular-like inputs diverge in their internal representations while optimising a single objective and producing clearly shape-biased behaviour? And does such a system say anything about why the ventral stream can be texture-like while behaviour is shape-biased? We build the system — two ResNet-18 trunks receiving the luminance and L−M channels of a cone-opponent image, with a developmental diet on the luminance channel (Fig. 1) — and examine it with three instruments: linear probes and cue-conflict behaviour on the streams (Fig. 2), voxel-wise encoding models of eight subjects' 7T fMRI responses to natural scenes (Fig. 3), and the behavioural readout that combines the streams (Fig. 4). The two streams do diverge, and by different routes; the texture-biased one is the one whose unique variance carries the ventral end of the visual hierarchy; and the integrated behaviour is shape-biased because a behavioural rate is a property of a readout over components, not a report of any one of them. Because the account predicts that properties commonly bundled with shape bias need not travel with it, we test that too, and find that robustness, agreement with human decisions and representational dimensionality all come apart from the rate (Fig. 5).

Relative to the originating thesis, three things changed. Every condition is now trained with three seeds and every claim states its replication unit; the representational-similarity analysis that had found no stream-to-brain mapping is replaced by variance partitioning of encoding models, which removes the global stream preference that had defeated it; and the stream weighting that the thesis fixed by hand is swept, so that the dependence of behaviour on the readout is measured rather than assumed.

## 2. Results

### 2.1 The model, and what the diet costs

The network splits its input once, before any convolution (Fig. 1a). Each image is converted to the DKL cone-opponent space; stream 1 receives the luminance channel, stream 2 the L−M chromatic channel, and each is a ResNet-18 trunk. Under the developmental visual diet the luminance channel passes through a Gaussian blur (σ = 4 px) and the scale-free contrast-sensitivity cut of Lu et al. (2026), evaluated at a fixed age of 48 months rather than scheduled over training, so that the transform is separated from the trajectory (Methods 4.1). The two trunks are initialised bit-identically: 0 of 120 stream-1/stream-2 tensor pairs differ in any condition, and a control in which both streams receive the same image returns all 120 pairs still identical after 150 epochs, with a stream difference of exactly +0.0000 on every fold of every seed. Every stream difference reported below is therefore attributable to the inputs. A linear classifier reads out the weighted sum w·y₁ + (1−w)·y₂ of the two streams' penultimate features, with w fixed at 0.5 during training (Fig. 1a).

Four front-end conditions are used throughout (Fig. 1a, table): the diet; the no-diet control; a channel-swap control in which the identical transform is applied to the L−M channel instead; and a `parvo_highsf` control in which stream 2 additionally receives untransformed luminance, because real parvocellular cells carry both. Networks were trained on MiniEcoset (112 classes; Thorat & Kietzmann, 2024) for 150 epochs, three seeds per condition (Methods 4.2).

The diet costs accuracy, and Fig. 1c shows how. The three no-diet conditions converge to 0.70 validation top-1; the diet arm plateaus near 0.50, reaches 0.556 at its best saved checkpoint against 0.701 for the control, and — uniquely — degrades after its per-epoch peak, ending at 0.445 (range 0.347–0.527 across seeds). Weights were saved every 25 epochs, and because late-training degradation would otherwise handicap the diet arm, the behavioural and out-of-distribution results below use each seed's best saved checkpoint, stating so wherever the policy matters (Methods 4.7). Two single-stream controls trained with the identical diet — one ResNet-18 trunk (11.2M parameters) and one ResNet-34 trunk parameter-matched to the two-stream model (21.3M against 22.4M) — reach 0.599 and 0.623, so the accuracy cost of the split is not a capacity artefact and is larger at matched parameters (Fig. 4c).

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

Which property of the input supplies the shape bias? Decomposing the front end into two factors — which carrier a stream receives (achromatic luminance or chromatic L−M) and whether the developmental transform is applied to it — gives four cells, two of which the swap and `parvo_highsf` conditions provide (Fig. 2b, c). On the probe, the chromatic carrier gives −0.504 without the transform and −0.145 with it; the achromatic carrier gives −0.271 without and +0.819 with, seed ranges 0.014–0.043. Both single factors move a stream toward neutral; neither crosses zero; only the conjunction crosses, and it crosses all the way, by +0.936 with no seed overlap. Cue-conflict shape bias on the same four cells replicates the interaction (0.364, 0.495, 0.396, 0.878). The shape bias requires the developmental transform applied specifically to an achromatic channel — a relational fact about what a stream receives, not a property of a stream called “magno”.

Two qualifications belong here rather than in a footnote. First, cue-conflict shape bias is computed only over images a readout decides as the shape or the texture category, and that count varies about four-fold across cells (165 to 583 of 1,200); a readout deciding 13% of the set returns something near a coin flip by construction, so the three non-conjunction behavioural cells are read as “no strong preference” rather than as precise values. Fig. 2d draws the decision composition of every readout so that this is visible rather than footnoted; the probe classifies every image and gives the same pattern. Second, the diet-trained models are less accurate, so the conjunction effect could in principle be an accuracy effect. We pre-registered the fatal outcome — a no-diet model at the diet's accuracy showing a shape-biased magno stream — and retrained the no-diet network with per-epoch checkpointing, selecting by rule the epoch nearest the diet's accuracy (epoch 10, 0.548 against 0.556). Matching accuracy closes 7.6% of the probe gap and 7.0% of the behavioural gap, two instruments agreeing to within 0.6 percentage points, and the matched model's magno stream stays on the texture side (Fig. 2e, f). The control rests on one matched run and is read as n = 1.

### 2.3 In cortex, the texture-biased stream carries the ventral end of the hierarchy

The thesis had asked whether the two streams align with different cortical streams, using representational similarity analysis (Kriegeskorte et al., 2008) against seven regions of the Natural Scenes Dataset, and found that they do not: whichever stream was the more predictive correlated better with every region. We replicated that result and understood it as a general-factor problem — a correlation cannot separate a global stream preference from regional structure — so we replaced the instrument. Voxel-wise banded-ridge encoding models were fitted for each of the eight NSD subjects with one feature band per stream, and every voxel's cross-validated explained variance was partitioned into the part only the magno band accounts for (U_m) and the part only the parvo band does (U_p) (Methods 4.5). The stream-preference profile — the per-voxel index (U_m − U_p)/(U_m + U_p), median over reliable voxels in each ROI — across the retinotopic hierarchy V1v → hV4 is Fig. 3a.

The two trained models sit on opposite sides of the global stream-preference axis: without the diet, every retinotopic area is magno-explained; with it, every area is parvo-explained. That global difference is the general factor that defeated the correlational analysis, and it is not the result. The result is the shape of the profile. Only the diet arm descends steeply from V1 to hV4 — early visual cortex relatively more magno-explained, hV4 relatively more parvo-explained, the direction a luminance/low-spatial-frequency versus chromatic split predicts — and the descent belongs to the model whose global preference points the other way, which a model simply preferring one stream everywhere could not produce.

Quantified as the contrast hV4 − mean(V1v, V1d) of the per-ROI median (Fig. 3b), the gradient is −0.834, −0.785 and −0.745 across the three diet-trained seeds against −0.079, −0.096 and −0.049 without the diet, and −0.166/−0.145/−0.153 against −0.027/−0.031/−0.023 on the raw statistic; every one of these six comparisons reaches the exact Wilcoxon floor for eight subjects (p = .0078) with complete seed separation. Monotonicity follows it: the per-subject Spearman ρ between hierarchical rank and stream preference is −0.964 to −0.973 with the diet and −0.503 to −0.606 without (Fig. 3c). Because the rank vector [1, 1, 2, 2, 3, 3, 4] has ties, the seven ROIs admit exactly 630 distinct labellings, and the permutation null is enumerable rather than sampled: the diet arm's observed ρ is the single most extreme of all 630 in every seed and on both statistics (p = 1/630 = .0016), while the no-diet arm does not reach that floor (p = .019 to .092) and in one seed its two statistics disagree, so we claim nothing there.

The diet does not create the gradient; it decides what training does to the one the input supplies. An untrained network on the diet front end already sits at −0.544 (8/8 subjects, p = .0078), and the untrained network on the no-diet front end at almost the same value (−0.479); the two front ends alone impose no different gradient. Training with the diet then steepens it, and training without it erodes it to near zero (Fig. 3b). Correcting each arm for the signed bias of the estimator itself — measured by simulation with the fitted feature spaces and the real noise ceilings, which returns a floor of the wrong sign, +0.05 and +0.04 — leaves a paired displacement of −0.871 with the diet against −0.130 without, each non-zero in every subject, the diet arm's 6.7 times larger. The gradient also survives the two front-end controls and replicates across codebases: an independently trained checkpoint from the thesis's own pipeline, which carries the progressive developmental schedule rather than our fixed age, gives −0.793 (p = .20 against −0.834).

The fitted models allow one more test that needs neither probe nor readout. Passing the 1,200 cue-conflict images through models fitted on natural scenes only yields a per-voxel predicted shape-bias index; the magno − parvo difference in it is +1.534 with the diet and +0.110 without (8/8 subjects, p = .0078; seeds 1 and 2: +1.566/+0.098 and +1.558/+0.083), carried entirely by the magno stream (+0.791 against −0.638), the parvo streams being indistinguishable (Fig. 3d). This is the encoding models' own verdict on the stream contrast of Fig. 2.

Three things this section does not settle are stated here rather than left to be found. The diet predicts BOLD worse, not better: median r² normalised by the noise ceiling is 0.405 with the diet against 0.464 without, in 8/8 subjects (Fig. 3e), a 13% cost, so the diet's case rests on stream differentiation and the gradient, not on prediction accuracy. The two front-end controls are degenerate on this instrument — in the swap condition the parvo stream contributes median U_p = 0.006, in `parvo_highsf` the magno stream U_m = 0.007 — so the correct statement is not that the controls abolish the gradient but that only the diet on luminance yields two streams that both carry substantial unique cortical variance and a hierarchical gradient between them (Fig. 3a). And the gradient was not pre-specified: it is the survivor of a family of seven regional hypotheses tested on these fitted models, of which six failed, including the thesis's own framing of a magno→dorsal, parvo→ventral mapping, an hMT+ specificity, and a foveal/peripheral contrast whose sign flipped when the blur was removed (Discussion; Supplementary Note S1). The survivor is reported because it alone survives removal of the very front-end manipulation that manufactures apparent regional structure in the others: run at the input level, the gradient gets larger without the blur on the raw statistic and is unchanged on the normalised one. Its magnitude is read as sign and ordering rather than as a calibrated effect size. Subject 01 carries about a third of the no-diet arm's contrast; excluding it moves that arm from −0.079 to −0.054, still non-zero in 7/7.

### 2.4 Behaviour reads out from both streams, and the rate is a property of the readout

The two streams are oppositely biased and the texture-biased one carries the ventral end. What remains is what the behavioural readout does with them. The integrated readout's cue-conflict shape bias is 0.675 ± 0.025 with the diet against 0.416 ± 0.020 without, three seeds each; the thesis's own checkpoint reproduces at 0.697 against its reported 0.677. On the identical images and statistic the ten human observers give 0.958 ± 0.019 (range 0.919–0.987) and the CNNs of the Geirhos et al. (2018a) release 0.429, 0.312 and 0.172, values our pipeline reproduces. The diet-trained magno stream, at 0.878, closes most of the distance from the CNN regime toward human observers; the no-diet model stays inside that regime. No match is claimed: 0.878 is below the lowest individual human, and integration costs 0.203 of shape bias (0.878 → 0.675).

The integrated readout lands between its two components rather than at the shape-biased one. Given two oppositely biased components that outcome is close to arithmetically forced, so it is not by itself a prediction that could have failed, and we do not present it as one. What can be measured is how much of the rate the readout controls. Sweeping w moves the readout while holding y₁ and y₂ exactly fixed (Fig. 4a): at 21 values of w, three seeds and both checkpoint policies, cue-conflict shape bias runs from 0.40 to 0.90 at the best saved checkpoints (0.42 to 0.88 at the final ones), and w = 0.5 reproduces the trained model's per-seed values exactly, which is the anchor showing that sweep and main pipeline are the same computation. Because the classifier is affine and the weights sum to one, clf(w·y₁ + (1−w)·y₂) = w·clf(y₁) + (1−w)·clf(y₂) holds exactly (maximum deviation 1.1 × 10⁻⁶ on a logit scale of 0.32), so the sweep is exact interpolation between two measured endpoints and contains no information beyond w = 0 and w = 1. That says the affine readout adds nothing; it does not say the span would vanish under a readout that is not affine. Under a rule fixed before the test, we refitted a two-layer nonlinear head, and a linear head as a control for the refit itself, on the frozen features at each of the 21 weights (Fig. 4b). The nonlinear head's median span is 0.466 against 0.508 for the frozen affine head (per seed 90%, 86% and 100% of the affine span; 86% of the 0.540 affine baseline registered before the run, against a survival threshold of 80%), and the refitted-linear and nonlinear spans agree to within 1%. The span is a property of having two oppositely biased components and any readout over them, not of the head's functional form. Restricted to readouts within 5 points of peak validation accuracy the nonlinear span narrows to 0.283 (shape bias 0.421 to 0.760), so the extreme rates near w = 1 are bought with accuracy — the cost accounting of Fig. 1c seen from the readout side.

Fig. 4c places every model on one pair of axes. The two streams read out alone bracket the integrated readout (0.903 and 0.400 around 0.655 at the best saved checkpoints), and the single-stream controls trained with the identical diet lie between: 0.567 for the half-capacity ResNet-18 and 0.626 for the parameter-matched ResNet-34, which is 0.029 below the two-stream model's 0.655 and inside the 0.059 that capacity alone moves the rate between the two single-stream arms. We therefore attribute the shape-bias ordering among these three models to capacity rather than to the split, and make no claim that the split causes a shape-bias advantage; the accuracy ordering, in which both single-stream arms beat the two-stream model, is the split's, and is larger at matched parameters. The large number on this ladder is the integration cost, 0.248 from the best single component to the integrated readout.

Nothing here fits w to humans. We tested whether it could be: fitting w to each of the ten observers individually on a 0.001 grid, no observer's rate is reachable in any run (0 of 90 observer-runs; the closest approach misses the lowest observer by 0.027), and a weight fitted to an observer's trial-by-trial choices does not cross-validate (held-out agreement +0.026 against +0.023 for an arbitrary weight, beating it in 45 of 90). The rate is a free parameter of the readout, and nothing in this family of readouts selects the human operating point.

This is the mechanism we propose for the paradox: a system holding a shape-biased and a texture-biased component, whose behaviour reads out from both, will look shape-biased behaviourally and texture-biased wherever the texture-biased component dominates — and in this model of the subcortical split, the component that dominates the ventral end is the texture-biased one. It is an existence proof with a cortically grounded demonstration, not a measurement of the human readout.

### 2.5 Properties that do not follow from shape bias

If shape bias is a property of a readout over components rather than of a representation, quantities the field treats as travelling with it need not. Over identical inputs, three do not (Fig. 5).

**Robustness.** The standard reading of Geirhos et al. (2018a, 2021) is that shape bias buys out-of-distribution robustness. We pre-registered that prediction and tested it on corruptions applied to the image before the front end, scored in the models' own label space at each seed's best saved checkpoint (Fig. 5a; curves in Extended Data Fig. 1). The diet-trained model is less robust on four of the five corruption families it was not trained on — uniform noise (relative-accuracy AUC 0.634 against 0.729), high-pass (0.678 against 0.827), false colour (0.854 against 0.938) and rotation (0.574 against 0.615) — with complete seed separation on each; on phase scrambling the arms are equivalent to within ±0.033. The only family on which the diet is more robust is low-pass (0.444 against 0.356), which we declared trained-in before the run because the diet is a low-pass blur, and it is not more robust to contrast reduction (0.481 against 0.549) even though its front end includes a contrast-sensitivity cut. Robustness appears exactly where it was trained in and nowhere else, which is the strongest available evidence that the nulls are real rather than a broken measurement. The two conditions without a shape bias track the no-diet control, and the thesis's schedule-trained checkpoint behaves like the static diet (inside its three-seed range on five of seven families, closer to it on six). This does not contradict Geirhos et al., whose stylisation route yields shape bias together with robustness; it shows the coupling belongs to that route. It also differs from the source of the diet, whose title reports robust, shape-based vision: we reproduce the shape-based half — their released model's mean cue-conflict shape bias is 0.822, our diet-trained magno stream reaches 0.878 — and not the robust half, and Methods 4.6 lists the three implementation differences any of which could carry the divergence.

**Agreement with human decisions.** Two decision-makers with the same marginal rate can agree on no individual image beyond what those rates force. Error consistency — Cohen's κ on trial-by-trial agreement — subtracts that forced agreement (Geirhos et al., 2020); across the ten observers the human–human ceiling is κ = 0.234 (45 pairs). We pre-registered that the diet-trained model would be more human-consistent than its control. It is not, on either readout, and the sharpest true statement is stronger: our most shape-biased model, the diet-trained magno stream at 0.878, has the lowest error consistency of all seven models compared (κ = 0.0089, 3.8% of the ceiling, against 7.3% for VGG-16 at shape bias 0.172). Across all 189 readouts of the w sweep, with w chosen post hoc per run to maximise agreement, the best κ never exceeds 20.7% of the ceiling (Fig. 5b); equivalence intervals fixed in advance exclude a κ change larger than 0.037 (15.8% of the ceiling) across the 0.71 span of shape bias the readouts cover. The measure does detect a correspondence when one is there: on the same images and observers, the same machinery recovers 24.1% of the split-half ceiling on whether an image resolves to either cue at all against 8.7% on which cue is picked (Fig. 5c). That positive control invites a diet reading, since the diet arm exceeds the no-diet arm on difficulty (24.1% against 9.2%) with no seed overlap; we do not make that claim, because completing the crossed front-end 2×2 shows a train/test-matching pattern rather than a weights effect — both matched cells positive, both crossed cells not (−0.6%, −3.7%) — and the design cannot separate the learned representation from the input transform, because separating them breaks the model (Fig. 5d).

**Representational dimensionality.** The participation ratio of each stream's features on 1,000 NSD images per subject gives a magno/parvo ratio whose ordering across the four front ends is reproduced by a learning-free Gabor bank, so it tracks the bandwidth of the carrier a stream is handed and not anything the architecture contributes. What the diet decides is which side of parity training ends on: every diet-trained seed ends magno-dominant (24/24 subject-seeds above parity in each of three diet arms) from front ends spanning 3.3-fold — including the `parvo_highsf` front end, which starts inverted in 8/8 subjects and which we had pre-registered would stay inverted — while every no-diet seed collapses to parity (Extended Data Fig. 2). No level is claimed, because the within-arm seed range exceeds the gaps between arms; two earlier readings of these data, a convergence to within 2% and a channel-specific set point, are retired by the seed replication. And the same case separates dimensionality from shape bias: `parvo_highsf`'s second stream becomes the more dimensional stream while remaining texture-biased (probe −0.266).

## 3. Discussion

**The two questions.** The thesis asked whether magnocellular- and parvocellular-driven streams inside one network could diverge in their representations while the network behaves shape-biased, and whether such a system bears on why the ventral stream can be texture-like while behaviour is not. To the first the answer is yes, replicated across seeds and with the mechanism located: divergence requires the developmental transform on an achromatic carrier, the shape-biased component is a property of the input that the diet protects from being unlearned, and the texture-biased component is built by learning. To the second the answer is an existence proof that the thesis could not reach with its instruments. Fitted to human cortex, it is the texture-biased stream whose unique variance carries the ventral end of the hierarchy, while behaviour, reading out from both, is shape-biased. Read as competing claims about a single representation, shape-biased behaviour and texture-like ventral representations cannot both be right; read as claims about a readout over components, they are not in competition.

**What the readout account licenses.** A model matching the human shape-bias rate is routinely described as having human-like representations. This system shows that inference fails twice over. The same network holds components at +0.819 and −0.518 and behaves at 0.675, and one scalar over frozen representations moves the rate across most of the range the field's models occupy, so the rate does not identify a representation; and across models spanning a five-fold range of shape bias, trial-by-trial agreement with ten observers stays between 7% and 18% of the human–human ceiling, our most shape-biased stream lowest of all, so the rate does not identify a decision process either. Hermann et al. (2020) and Islam et al. (2021) established that a representation can hold more shape than its behaviour reveals; what is new here is the form of the dissociation — both representations held fixed, the rate moved by the readout alone, a refitted nonlinear head preserving the span — and the cortical grounding, which neither prior study has. The methodological consequence is direct: shape bias should be reported as the readout statistic it is, alongside error consistency where human trial data exist, and should not on its own license a claim about representations.

**What a developmental constraint does.** The intuitive account of developmental low-pass input is informational: coarse input protects the learner from texture, so it learns shape. Our results do not support it. The transform removes bandwidth rather than adding structure, and the shape bias it produces is present before training. What the diet decides is what learning does not destroy: without it, training drives the luminance stream from +0.42 to −0.23; with it, +0.80 is still +0.81 after 150 epochs. A second instrument shows the same signature, and what replicates there is the side of parity rather than a level. This is a constraint on the solution learning converges to, not a better input representation — consistent with the thesis's observation that the transformed magno stream becomes less predictive of the category as the parvo stream becomes more so.

**What comes apart, and what the diet costs.** Shape bias, robustness, dimensionality and human-likeness are discussed as a package. Over identical inputs this system separates them three times, and a readout account says why it should: none of the three is a property of a readout. The diet is also expensive. It makes the network less accurate (0.556 against 0.701), less stable late in training, less robust on four of five untrained-on corruptions, and a worse predictor of BOLD. Nothing here recommends a developmental diet as a training method; its interest is as a model of development, and the thesis's central mechanistic reading — that divergence and shape bias are bought by degrading the magno stream's evidence — survives in every instrument.

**Limitations.** The load-bearing cortical result rests on eight subjects from one dataset and on a single retinotopic atlas; its magnitude is read as sign and ordering only, and it is the survivor of a family of seven regional hypotheses, disclosed above. The mapping between our chromatic stream and texture-selective ventral cortex is by explanatory role, not by content: nothing here asserts that the ventral stream is chromatic, and the natural prediction that the chromatic stream should win in colour-biased cortex fails (pooled VO +0.005, bounded within ±0.017; Supplementary Note S2). The `parvo_highsf` condition varies architecture alongside the front end. The diet is a fixed 48-month front end rather than a trajectory, though a schedule-trained checkpoint behaves like it on every instrument we tried. The koniocellular pathway is absent, and substituting S−(L+M) for L−M in an untrained network makes the ventral prediction worse rather than better. Whether a two-stream architecture specialises on its own is untestable here, because the trunks are bit-identical at initialisation. Human comparisons are against ten published observers and of aggregate rate; the readout weight is a hyperparameter that nothing selects, and fitting it to observers does not repair that. Two of the four pre-registered predictions in this paper failed and are reported as failures. Claims deliberately not made are listed in Supplementary Note S1.

## 4. Methods

### 4.1 Front end

RGB images are converted to the DKL cone-opponent space through RGB→XYZ (D65)→LMS (Hunt–Pointer–Estévez)→DKL at 256 × 256 px, each channel normalised to [0, 1] with the training set's global range and quantised to 8 bits exactly as the training data were stored (Derrington et al., 1984). Stream 1 receives the luminance channel; under the diet it passes through an isotropic Gaussian blur (σ = 4 px, kernel 33, reflect padding) and the scale-free contrast-sensitivity cut of Lu et al. (2026) evaluated at a fixed age of 48 months (`cs_logspan_start` = 5 × 10⁻³, decomposition mode “keep”). Stream 2 receives the L−M channel untransformed. The four conditions are: `dkl_bcs` (diet: luminance + blur + CS | L−M); `dkl_base` (no diet: luminance | L−M); `dkl_bcs_swap` (luminance | L−M + blur + CS); `dkl_bcs_parvo_highsf` (luminance + blur + CS | L−M stacked with untransformed luminance, two channels). The last differs architecturally in its first-layer fan-in, and that difference is measured rather than argued away (Extended Data Fig. 2; Results 2.5). Evaluation stimuli, including NSD images and cue-conflict images, pass through the identical front end.

### 4.2 Architecture and training

Two ResNet-18 trunks (He et al., 2016), one per stream, initialised bit-identically, with a joint linear classification head over the weighted sum of their penultimate features (w = 0.5; cross-stream mixing off). MiniEcoset at 256 px, 112 classes (Thorat & Kietzmann, 2024). 150 epochs, batch 256, Adam (lr 1 × 10⁻⁴, β₁ = 0.9; Kingma & Ba, 2017), cross-entropy, with the augmentation set of Lu et al. (2026) applied on GPU (horizontal flip, rotation ±15°, colour jitter, random equalise, perspective, sharpness; random grayscale for RGB inputs only). Three seeds per condition; weights saved every 25 epochs. Single-stream controls use one ResNet-18 or one ResNet-34 trunk fed the diet's luminance input with the identical schedule. The accuracy-matched control retrained `dkl_base` with per-epoch checkpointing and selected by rule the epoch nearest the diet arm's best-checkpoint accuracy.

### 4.3 Behaviour and probe

Cue-conflict shape bias is computed on the Geirhos et al. (2018a) style-transfer set restricted to the 16 categories present in MiniEcoset and to images whose shape and texture categories differ (1,200 images). Logits are restricted to the 16 columns; the argmax is scored as a shape or texture response; images decided as neither are excluded from the denominator and their count is reported. Bootstrap 95% confidence intervals are over images. The readout is full = classifier(w·y₁ + (1−w)·y₂), magno_only = classifier(w·y₁), parvo_only = classifier((1−w)·y₂), all through the same classifier and bias term; the w sweep evaluates full at 21 values of w on stored features. The refitted heads (Linear(512, 112) and Linear(512, 512)→ReLU→Linear(512, 112)) were trained on w·F₁ + (1−w)·F₂ of MiniEcoset training features at each w with an identical budget (Adam, lr 10⁻³, batch 1024, 20 epochs) and the same initialisation seed at every w. Human observers are the ten of the same release (1,280 trials each), scored by the identical rule; our recomputation reproduces the published 0.958 and the published AlexNet, GoogLeNet and VGG-16 values. Error consistency is Cohen's κ (Cohen, 1960) on the binary shape/texture choice over images both parties decided, with expected agreement p_m·p_h + (1−p_m)(1−p_h), reported as a percentage of the human–human κ over all 45 observer pairs (Geirhos et al., 2020). Per-image correspondence uses the observers' per-image shape rate and cue-resolution rate (split-half reliabilities r = .658 and .794 after category removal), the models' shape-minus-texture logit margin, w chosen on half the images and scored on the held-out half, as a percentage of each target's own ceiling. The linear probe reduces each stream's features to 256 PCA components and fits multinomial logistic regression (C = 1.0) under 5-fold stratified cross-validation to shape and texture labels separately; bias is shape accuracy minus texture accuracy on the same folds.

### 4.4 Effective dimensionality

Participation ratio (Σλ)²/Σλ² of the stream feature matrix on 1,000 NSD images per subject (Gao et al., 2017), computed Gram-side, reported as the magno/parvo ratio with parity at 1.0. Subjects are the replication unit within a seed; seeds are reported separately and never pooled. A learning-free reference is the same ratio computed on a Gabor oriented-energy bank applied to each stream's own transformed input.

### 4.5 Encoding

NSD (Allen et al., 2022) `betas_fithrf_GLMdenoise_RR` single-trial betas, converted to percent signal change and z-scored per session per voxel, `nsdgeneral` mask, eight subjects. Splits are by 73k image id with the shared-515 images held out. Stream features for each subject's stimuli are reduced by PCA to 1,000 components per stream (fit on training images only) and entered as two bands into a banded ridge regression (himalaya; Dupré la Tour et al., 2022) with 3-fold GroupKFold by image id, per-voxel alphas and per-voxel band weights. Unique variance U_m (U_p) is the drop in cross-validated r² when the magno (parvo) band is removed. Noise ceilings are NSD's `ncsnr`; voxels below 0.1 are excluded and any ROI cell with fewer than 20 surviving voxels is reported as missing. ROIs are the `prf-visualrois` labels of the Wang et al. (2015) atlas as distributed with NSD (V1v, V1d, V2v, V2d, V3v, V3d, hV4), with hierarchical rank [1, 1, 2, 2, 3, 3, 4]. The canonical-correlation spectrum between the two bands runs from ≈0 to 0.93, so each band retains directions the other cannot predict and unique variance is identifiable; a recovery simulation with the fitted feature geometry and a known across-ROI split returns the true ordering exactly (ρ = +1.000 in 8/8 subjects, both arms), which is why the gradient is reported as sign and ordering rather than as calibrated magnitude. The per-voxel predicted shape-bias index is obtained by pushing the cue-conflict set through each fitted encoding model. The koniocellular control substitutes S−(L+M) for L−M in stream 2 of the untrained seed-0 network, contrast-matched to L−M per image, with the magno band bit-identical across arms (gate max|Δr²_magno| < 10⁻⁴, passed at 0.000).

### 4.6 Out-of-distribution robustness

Corruptions are applied to RGB before the front end and scored in the native 112-class label space. Five families were declared untrained-on before the run (uniform noise, high-pass, phase scrambling, rotation, false colour) and two as trained-in positive controls (low-pass, contrast). Reported as the area under the relative-accuracy curve across severities, at each seed's best saved checkpoint. Three implementation differences from Lu et al. (2026) are declared: our front end is evaluated at a fixed 48 months rather than scheduled (the one we can rule out, since a schedule-trained arm behaves like the static diet); our diet is a transform on one stream's input of a two-stream network rather than an augmentation on a single trunk; and we train on MiniEcoset and score corruptions in that label space, applied before the front end.

### 4.7 Statistics

Within-subject comparisons use the paired exact Wilcoxon signed-rank test, n = 8 (exact two-sided floor p = .0078); seed-level comparisons use the exact Mann–Whitney U, 3 v 3 (floor p = 0.100, reached only on complete separation), and seeds are never pooled with subjects. Every cortical contrast is computed both on the raw per-voxel difference U_m − U_p and on the normalised index (U_m − U_p)/(U_m + U_p), each summarised by its median over reliable voxels per ROI, and a claim is made only where the two agree; disagreements are reported as unresolved. Equivalence uses two one-sided tests with bounds fixed before any interval was computed (Schuirmann, 1987; Lakens, 2017). No multiple-comparison correction is claimed: with 28 p-values and an exact floor of .0078, Holm–Bonferroni over any family larger than six cannot yield a rejection, so claims rest on complete separation, agreement of both statistics, replication across seeds and effect size (dz = −6.60 for the diet gradient against −1.09 without the diet). Behavioural and robustness results use each seed's best saved checkpoint unless stated; cortical fits use the final checkpoint. Four predictions — the front-end 2×2, the inverted-front-end prediction, the corruption families and the error-consistency hypothesis — were written into a version-controlled plan and committed before the corresponding analysis was run (internal pre-specification, not registry-based); two failed and are reported as failures.

## Data and code availability

NSD is available at naturalscenesdataset.org under its data-use agreement; the Geirhos cue-conflict stimuli and observer data at github.com/rgeirhos/texture-vs-shape (CC BY 4.0); MiniEcoset at osf.io/msna2. Training, front-end, encoding, behavioural and analysis code, the pinned environment and the derived result tables behind every figure are in the accompanying repository; figures are regenerated by the scripts in `tmp_analysis/figs_v2/`.


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
- Any dorsal/ventral stream mapping. The thesis's framing “magno→dorsal, parvo→ventral” was tested rather than dropped: the pre-declared dorsal−ventral interaction is +0.0245 on the diet model and +0.0122 (p = .195) without the low-pass front end, so it does not survive its own control.
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
