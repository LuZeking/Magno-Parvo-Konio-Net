# TSN x TDANN spatial-loss pilot: pre-registration (2026-09-25)

**Question.** Give a network the two retinal pathways (magno = blurred, contrast-
sensitivity-filtered luminance; parvo = L−M) and let them merge into one topographic
cortical sheet trained with the TDANN spatial loss. Do the sheet units that win
**dorsal** voxels in Finzi et al.'s 1-to-1 assignment come out **magno-like**? That is
the property the TDANN probe found missing in Finzi's own network
(`results/finzi_1to1/MAGNO_PROBE_REPORT.md`: DV NOT MAGNO-LIKE, and slightly
anti-magno).

**Why now.** The trigger registered in PLAN_TDANN_MAGNO_PROBE §8 fired through idea 1
(MAGNO -> DORSAL, +0.125, 8/8 subjects, `results/finzi_1to1/REPORT.md`). The user gave the
go on 2026-09-25.

**Registered before any pilot training.** At commit time only the GPU smoke test has run,
at 300 steps; it produces no analysis checkpoint. No pilot features, assignments or
probe responses exist.

Code:
- `src/models/fused_two_stream.py` (network);
- `src/models/topographic.py` (sheets and spatial loss, ported from `spacetorch`);
- `src/training/standalone_trainer.py` (conditions `dkl_bcs_fused_a0`, `dkl_bcs_fused_a2p5`);
- `slurm/train_spatial_pilot.sh`.

## 0. Why the design changed from the earlier sketch
The earlier sketch put the existing TSN's magno and parvo layer-4 units on one shared
sheet. In that network every unit is magno or parvo by construction, which leaves two
possibilities:

- **Segregated positions** (for example from TDANN's swap pre-optimisation, which sorts
  units by response correlation): the sheet separates magno from parvo by placement
  alone, so "the dorsal cluster is magno" would be built in. That is why the sketch needed
  position-only and random-position controls.
- **Interleaved positions:** the loss can only make neighbouring magno and parvo units
  correlate, pulling the two streams towards each other.

The pilot instead keeps the pathways separate through `layer2` (retina/V1 in TDANN's
area map) and merges them into one shared `layer3`-`layer4` (V2/V4/VTC). Every sheet
unit then learns its own mix of magno and parvo input, so a magno-dominated dorsal region
can only emerge; it cannot be placed. With no stream identity to leak, the two position
controls are no longer needed.

## 1. Arms
Two arms, seed 0, identical except for the spatial-loss weight alpha:

| arm | condition | alpha per depth |
|---|---|---|
| A0 | `dkl_bcs_fused_a0` | 0 (sheets built, loss logged, not trained on) |
| A1 | `dkl_bcs_fused_a2p5` | 2.5 |

**Shared settings:**
- **Network:** `FusedTwoStreamNetwork`, 12.2M parameters. Two ResNet-18 trunks through `layer2` with 1-channel stems; the 256-channel concatenation feeds ResNet-18's `layer3` and `layer4`, then avgpool and a linear 512 -> 112 head.
- **Data and front end:** miniecoset, 256 px, DKL uint8, 112 classes. The diet front end is the `dkl_bcs` condition (blur sigma = 4 plus CSF at 48 months on luminance -> magno trunk; rg -> parvo trunk).
- **Training:** Adam at lr 1e-4, batch 256, 150 epochs, bf16 autocast, DVD-B augmentations. Everything else is identical to the `dkl_bcs` TSN runs.
- **Spatial loss:** a port of `spacetorch` `spatial_correlation_loss`, SL = (1 − corr(r_ij, 1/(1 + d_ij))) / 2, over one random neighbourhood per sheet per step.
  - It is applied to all 8 ResNet-18 depths.
  - Trunk depths (`layer1.x`, `layer2.x`) have one sheet per pathway, and their two SLs are averaged, so each depth carries weight alpha once.
  - Total loss = CE + alpha · sum over the 8 depths of SL.
  - alpha = 2.5 is Finzi et al.'s supervised optimum (TDANN config `lwx10`: 2.5 on each of 8 layers).
- **Sheets:**
  - TDANN retinotopic initialisation, rebuilt for this network's 64/32/16/8 grids: `place_conv` with random within-hypercolumn offsets, `jitter 0.3`, TDANN tissue sizes and neighbourhood widths, and at most 500 units per neighbourhood.
  - The sheets depend only on layer shapes and seed, so A0 and A1 share identical sheets. They are saved in each run's `sheets/`.
  - **Not ported:** swap pre-optimisation. It needs a pretrained network of the same architecture, and with no stream identity there is nothing for it to leak.
- **Analysis checkpoint:** `checkpoint_final.pt` (epoch 150), fixed now. It is never chosen by validation accuracy.

## 2. Units and assignment
- **Units:** the output of the shared `layer4.1`, 512 x 8 x 8 = 32,768 units.
- **NSD image path:** exactly the TSN extraction path (`extract.py`, 256 px, `magno_parvo_inputs` with the diet front end), then trunks -> `layer3` -> `layer4`.
- **Assignment:** Finzi's all-units rectangular Hungarian assignment (cost = 1 − train Pearson r), per hemisphere per subject. It uses the idea-1 pipeline's `finzi_all_units` path unchanged: NSD `streams` ROIs with NC >= 10%, and the winner of each voxel saved to `_assign.npz`.

## 3. Indices per unit (layer4.1)
- **ACH and LSF** exactly as in PLAN_TDANN_MAGNO_PROBE §3-§4:
  - same NSD shared-515 stimuli (orig / lum / iso / low / high);
  - float DKL path (Amendment 2);
  - residualised on grid position (8 x 8 here) and standardised by the SD over units.
- **MR (magno reliance), secondary.** For each unit, compare its response to orig images
  when one pathway's input is replaced by that pathway's input for a uniform mid-grey
  image:
  - sd_M = response SD across images with the parvo input neutralised;
  - sd_P = the same with the magno input neutralised;
  - MR = sd_M / (sd_M + sd_P).

  Float path, then the same residualisation and standardisation. MR is the direct
  analogue of idea 1's magno share for units that mix both pathways.

## 4. Per-subject statistic
As in PLAN_TDANN_MAGNO_PROBE §5, with one seed:

- Delta^DV_s = the NC-quintile-stratified mean index of dorsal-winning units minus that of ventral-winning units (`mapping.strat_delta`: bins need >= 20 voxels per stream; hemispheres pooled).
- Delta^LV_s = the same for lateral minus ventral.

## 5. Decision rule
Inference is on n = 8 subjects, with SESOI delta = 0.2 SD. Each index gets an exact
two-sided Wilcoxon test and a 90% t-interval. The labels are those of
PLAN_TDANN_MAGNO_PROBE §6 with no seed rule (one seed):

| label | condition |
|---|---|
| MAGNO-LIKE | p < .05, mean >= +0.2 |
| PARVO-LIKE | p < .05, mean <= −0.2 |
| SMALL | p < .05, abs(mean) < 0.2 |
| NO DIFFERENCE | 90% CI inside (−0.2, +0.2) |
| UNRESOLVED | otherwise |

The combined verdict per contrast (MAGNO-LIKE / PARTLY / NOT / UNRESOLVED) combines ACH
and LSF as in the probe.

- **Primary:** A1 DV combined verdict.
- **Secondary** (they do not change the primary):
  1. A0 DV combined verdict.
  2. A1 − A0 paired by subject, per index: **SL-AMPLIFIED** if both ACH and LSF are MAGNO-LIKE on the difference.
  3. LV for both arms.
  4. MR, DV and LV, both arms.

**What happens next:**

| outcome | reading | next step |
|---|---|---|
| A1 DV MAGNO-LIKE, and SL-AMPLIFIED | M/P inputs **and** the spatial constraint together make a magno-dominated dorsal stream | GO: seed grid (3 seeds x {A0, A1}) + nodiet A1 |
| A1 DV MAGNO-LIKE, not SL-AMPLIFIED | the input pathways, not the spatial loss, make dorsal magno-like; topography comes on top. Paired with TDANN's NOT MAGNO-LIKE, this is still a positive claim | GO (same grid) |
| A1 DV UNRESOLVED | underpowered at one seed | run seed 1 of both arms once, pool seed means as in the probe, re-decide; if still UNRESOLVED, stop |
| A1 DV NOT MAGNO-LIKE or PARVO-LIKE | the hybrid does not give a magno-like dorsal stream | stop; one paragraph in the idea-1 supplement |

## 6. Gates (all must pass, otherwise every verdict is UNRESOLVED)
- **G-train.** Both runs finish 150 epochs, A0 final val top-1 is >= 0.35, and A1 final
  val top-1 is >= 0.8 x A0's.
- **G-SL (the manipulation took).** Mean `layer4.1` SL over 1,000 random neighbourhoods,
  computed on the orig shared-515 images (float path, batch = all 515): A1 <= A0 − 0.05.
- **G-probe (instrument).** In both arms, on the float path:
  - the median ACH of the magno trunk's `layer2.1` units (pure magno by construction) is >= 0.95;
  - the parvo trunk's median is <= 0.05.

  This is the probe's G1, moved to the trunks. The front end is unchanged, so the probe's
  G2 carries over.
- The mapping pipeline is unchanged from idea 1. Its positive and negative controls passed
  there (PC-TDANN, PC-TSN, NEG) and are not re-run.

## 7. Topography (secondary, descriptive with permutation p)
- **T1, MR map.** Over unit pairs within 50 fixed `layer4.1` neighbourhoods (500 units
  each), the Pearson correlation between |MR_i − MR_j| and sheet distance, averaged over
  the neighbourhoods.
  - It is compared between arms.
  - Each arm gets a p from a null of 200 shuffles of MR among the units of each grid
    cell. The shuffles keep any retinotopic gradient and remove the within-cell map.
- **T2, stream clustering.** Per subject, the fraction of each assigned unit's 10 nearest
  assigned sheet neighbours that won the same stream.
  - It is compared with 1,000 label shuffles among the assigned units.
  - A1 vs A0 is tested with a paired Wilcoxon over subjects.
- **Not done:** Finzi's spatial similarity (sheet distance vs cortical-surface distance).
  It needs surface coordinates, which the volume pipeline does not carry; it is a
  follow-up only if the primary is GO.

## 8. Descriptive
- Functional correspondence r/√NC per stream.
- Val top-1 curves.
- Per-depth SL trajectories.
- The MR distribution: the fraction of units with MR > 0.8 and with MR < 0.2 per arm.
- The same indices on the `layer3.1` sheet.

## 9. Limitations, stated up front
- **One seed.** The UNRESOLVED path buys at most one more.
- **The architecture is not the manuscript TSN.** Fusion is after `layer2` rather than a
  weighted sum after `layer4`, with 12.2M parameters against 22.4M. A0 is the matched
  reference; the idea-1 networks are not.
- **The objective is not Finzi's best.** It is supervised miniecoset (112 classes), not
  SimCLR on ImageNet. Finzi's supervised TDANN needed alpha = 2.5 and matched cortex less
  well. alpha is not re-tuned for Adam.
- **No swap pre-optimisation.** Early-layer maps will be rougher than TDANN's, but the
  question is about `layer4.1`.
- **The parvo luminance leak** (uint8 rounding, `tsn-parvo-uint8-luminance-leak`) is in
  the training data of both arms. The probe itself runs on the float path.

## 10. Budget and where it runs
- **Two single-GPU runs on g017 (B200).** On 2026-09-25 another user's ~190-task array
  holds every agcichy A5000 at 6x our fairshare priority, and g016's H100s are allocated.
- **g017 has no /scratch**, and compute nodes cannot ssh to the login node. So:
  - the 59 GB dataset is staged on `/home/byrong/_staging/TSN/spatial_pilot_data`
    (`slurm/spatial_pilot_stage_data.sh`, checked byte for byte with `cmp`) and cached in
    RAM (`--cache-ram`);
  - outputs are staged under `/home/byrong/_staging/TSN/spatial_pilot/runs`;
  - a chained `main` job (`slurm/spatial_pilot_migrate.sh`) moves them to
    `/scratch/byrong/TSN/spatial_pilot/runs` with a per-file manifest check, then deletes
    both staged copies.
- **Smoke test 26532505 (g017, 2 x 150 steps per arm, with a resume in between):**
  - both arms exit 0, with no NaN or inf values;
  - the resume restarts at epoch 1, and the saved sheets reload and match a rebuild;
  - the final checkpoints load into `FusedTwoStreamNetwork` with every key matched;
  - A0 and A1 sheets are identical;
  - by step 300, A1's SL has fallen at every depth (`layer3.1` 0.50 -> 0.31,
    `layer4.1` 0.50 -> 0.40), while A0's stays at 0.50;
  - A1's CE falls more slowly early on (4.60 vs 4.15 at step 300; val top-1 0.017 vs
    0.046). G-train judges this at epoch 150, not here.
- **Speed:** ~0.43 s/step, i.e. ~440 s/epoch and **~18.5 h per run** (both in parallel).
  The reference is 26.9 h on an A5000 for the 22.4M-parameter `dkl_bcs`.
- **Memory:** peak MaxRSS equalled the 72G request, because the cgroup also counts the page
  cache from reading the dataset into RAM. Training requests 80G.
- Then about 1 day of CPU analysis: extraction for 8 subjects x 2 networks, mapping, probe
  (`src/analysis/finzi_1to1/spatial_pilot.py`).

## Amendments
(none)
