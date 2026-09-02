# TSN → Paper Plan: Magno-Parvo-Konio-Net

**Goal.** Turn the BA thesis (Manocchio, "Investigating two-stream artificial neural networks
using a developmental visual diet") into a high-tier paper (NMI / NeurIPS-tier). Working title
claim: *biologically constrained magno/parvo(/konio) input channels induce specialized
shape-/texture-biased streams under a single objective; their integration yields human-like
shape-biased behavior; and each stream uniquely explains distinct, biologically predicted
variance in visual cortex.*

Repo: `/home/byrong/projects/TSN/Magno-Parvo-Konio-Net` (clone of the thesis codebase
`two_stream_dvd`). Thesis PDF: `/home/byrong/projects/TSN/BA_Manocchio_Benito.pdf`.

---

## 0. Audited ground truth (verified on this cluster, 2026-08-25)

| Item | Status | Location / note |
|---|---|---|
| NSD betas b3 (`betas_fithrf_GLMdenoise_RR`, func1pt8mm, session HDF5s) | ✅ present, 8 subjects | `/scratch/byrong/encoding/datasets/nsd/betas/subjXX/betas_fithrf_GLMdenoise_RR/` |
| NSD noise-ceiling SNR (`ncsnr.nii.gz`, + split1/2) | ✅ present | same dir as betas |
| ROIs: `nsdgeneral`, `streams`, `prf-eccrois`, `prf-visualrois`, `floc-*`, `Kastner2015`, `HCP_MMP1` | ✅ present | `/scratch/byrong/encoding/datasets/nsd/rois/subjXX/func1pt8mm/roi/` |
| NSD stimuli + design (`nsd_stimuli.hdf5`, `nsd_stim_info_merged.pkl`) | ✅ present | `/scratch/byrong/encoding/datasets/nsd/stimuli/` |
| Continuous pRF maps (`prf_eccentricity/angle/size.nii.gz`) | ❌ not local | optional small download from `nsddata/ppdata`; the 4-band `prf-eccrois` labels suffice for the main analysis |
| **Trained TSN checkpoints from the thesis** | ❌ **NOT local** | they live on the Osnabrück cluster (`/share/klab/...`). Blocks trained-model feature extraction until fetched or retrained. Pipeline is validated with the untrained-network control first (needed anyway). |
| MiniEcoset 256px | ✅ present | `/scratch/byrong/VST_data/VST_datasets/texture2shape_miniecoset_256px.h5` (verify class count = 112 before retraining) |
| Python env with torch/himalaya | ❌ none exists | create env `mpk` (plan §4, step 0) |
| DVD package (scale_free branch) | ✅ public | `github.com/KietzmannLab/DVD` branches `paper`, `scale_free` |
| `lightning_kietzmannlab` (training pipeline dep) | ❌ not on PyPI | needed only for retraining with the original pipeline; either obtain from lab GitHub or write a minimal local trainer. **Open item.** |
| Slurm | ✅ | `main` (CPU, 166 nodes, 14d), `gpu`, `agcichy`, `scavenger`. Rules: array jobs when N>1; CPU work on `main`; GPU only when CUDA used; watch `prio` GPU cap → fall back `--qos=standard`. |
| **Subject neural RDMs + 12 models' layer RDMs** | ✅ in repo `RDMs/` | `streams` + V1v/V1d/V2v/V2d/V3v/V3d/hV4/PPA per subject (special515) — repaired RSA (Phase 1.2) unblocked WITHOUT checkpoints |
| Design assumptions validated | ✅ | shared-515 logic reproduces exactly 515 across 8 subjects; `nsdId` = 0-based row index; sessions on disk: 40/40/32/30/40/32/40/30 |
| MiniEcoset h5 | ✅ verified | 112 classes, train/val/test/testplus splits — it IS the thesis dataset |
| Known code bug: Interconnected-TSN mixing | ✅ confirmed | `src/models/pytorch_models.py:225-226, 258-259`: `y2` is mixed with the **already-mixed** `y1` ⇒ stream 2 receives 0.8125·y2 + 0.1875·y1, not 75/25; mixing asymmetric. Fix with a legacy flag (old checkpoints trained under buggy forward must be evaluated with the same forward). |

Thesis-inherited issues the paper must fix:
- Single seed per condition; no error bars/tests anywhere.
- Dataset-version confound (DKL Base & DKL CS trained on uint8 dataset; others on float64).
- Linear readouts trained for 1 epoch only.
- Neural-alignment section is a null (group-averaged RSA, 515 images, no ceiling normalization,
  no per-subject stats).

---

## 1. Track A — Neural alignment redesign (no retraining required to build)

**Reframe.** Drop "magno→dorsal, parvo→ventral" as the headline (static images cannot carry it;
parietal reliability is low). New target claim: **the two input-defined streams uniquely explain
complementary, biologically predicted variance in visual cortex** along axes where a
luminance/low-SF vs. chromatic(L−M) contrast has strong priors: eccentricity, spatial-frequency
preference, color selectivity, texture-vs-shape selectivity.

### A1. Core analysis: voxelwise banded-ridge encoding + stream-wise variance partitioning
- Betas: b3 session HDF5s, int16/300 → PSC, **z-score per voxel within session**, pool sessions.
  Mask: `nsdgeneral`. All 8 subjects, each subject's full available design.
- Split **by 73k image ID**: test = shared-515 (all 8 subjects, 3 reps each — average reps);
  train = all remaining trials (single-trial). Exclude any trial of a test image from training.
  Hyperparameter CV: GroupKFold grouped by image ID.
- Features: NSD images through the model's **own training front end** (RGB→XYZ(D65)→LMS(HPE)→DKL;
  magno band = luminance + blur σ=4 + 48-month CSF cut; parvo band = L−M untransformed), then
  per-stream activations from all 4 ResNet stages, adaptive spatial pooling, **PCA to matched
  dimensionality per stream** (default 1000; robustness at 500/4000; PCA fit on train images only).
- Model: himalaya `MultipleKernelRidgeCV` (torch_cuda backend), 2 bands; plus 2 single-band
  ridge fits with identical CV.
- Quantities: Unique_magno = R²_joint − R²_parvo; Unique_parvo = R²_joint − R²_magno;
  Shared = R²_m + R²_p − R²_joint; SPI = (Uₘ−Uₚ)/(Uₘ+Uₚ) on voxels where joint model > chance.
- **Noise-ceiling normalization**: NC = ncsnr²/(ncsnr² + 1/3) (3 reps averaged); report R²/NC;
  analyze voxels above NC threshold (ncsnr > 0.2 default; show threshold robustness).

### A2. Inference: stream × cortical-axis interaction, subject-level
- Unit of inference = subject (n=8). Per-subject ROI summaries of NC-normalized unique variance.
- Interaction: paired difference-of-differences across subjects (Wilcoxon/permutation), all 8
  subject dots plotted; mixed model as omnibus. A **double dissociation additionally requires both
  simple effects** significant (avoid the Nieuwenhuis error).
- Stimulus bootstrap (resample 515 test images, 1000×) for per-subject CIs.
- FDR across ROIs for maps; one pre-declared headline contrast.

### A3. Hypothesis-targeted analyses (ordered by prior)
1. **Eccentricity gradient**: SPI parvo→magno from fovea to periphery within V1–V3/hV4
   (`prf-eccrois` bands; trend test across subjects). Near-free; likely the strongest panel.
2. **Spatial-frequency preference**: small Gabor-pyramid encoding model per voxel → preferred SF;
   correlate with SPI (low-SF voxels ↔ magno-unique).
3. **Color cortex**: parvo-unique variance peaks in hV4/VO (`prf-visualrois`/Kastner labels).
4. **Texture/shape voxels + in-silico cue-conflict** (flagship link to behavior): define
   texture- vs shape-selective voxels split-half (Portilla–Simoncelli / Gram-feature control
   models); test stream-unique variance in each set. Then push Geirhos cue-conflict images
   through fitted encoding models → per-voxel predicted shape-bias index → correlate with SPI.
5. Dorsal/MT: explicitly a *limitation* (static stimuli), not a result.

### A4. Controls (built in from the start)
- **Untrained (random) TSN**, identical inputs & pipeline → isolates learning from input filtering.
  (Also unblocks development while trained checkpoints are missing.)
- **Input-level bands** (Gabor energy / GIST on each stream's transformed input) as extra bands →
  learned-nonlinear vs. filter-induced variance.
- Matched PCA dims; no best-layer selection on test data; report magno↔parvo feature-space
  correlation (they were trained jointly); state unique-variance caveats.

### A6. Developmental EEG alignment (7 sub-age bins) — NEW
Time-resolved RSA of TSN stream RDMs against the developmental EEG dataset
(infant project, `/scratch/byrong/infant`): bins infants_6mo, infants_12mo,
youngerChildren_4yo/6yo, olderChildren_8yo/12yo, adults_all (target months
6/12/48/72/96/144/240); ~360 subjects, each seeing one of 4 subsets of 12
stimuli (80-image pool at `infantEEG/stimuli`); neural RDMs 12x12x301 tp
(-500 ms + 20 ms steps), sources occipital/wholebrain, from
`rsa2026/inputs/brain_rdms`; campaign noise ceilings per bin x tp.

**Hypothesis:** relative magno-vs-parvo stream alignment shifts across
development — infant EEG (low acuity, immature chromatic pathways) should
align relatively better with the blurred/low-SF magno stream, with
parvo-stream alignment increasing toward adulthood. This connects the TSN
directly to the DVD developmental story with human developmental data.

Implementation (worktree `src/analysis/eeg/`):
- `tsn_rdm_extractor.py`: EEG stimuli → training-exact DKL front end →
  stream-wise layer RDMs (magno_/parvo_ conv1, layer1-4, avgpool +
  combined_avgpool), emitted in the rsa2026 campaign npz schema
  (`<root>/<subset>/<model>/rdms_epoch_NNN.npz`, pearson+spearman ×
  zscore/no_zscore via the infant project's own `compute_rdm`) — caches are
  interchangeable with the existing runner.
- `temporal_rsa_tsn.py`: per-subject time-resolved Spearman per layer,
  aggregated per sub-age bin with campaign noise ceilings; plus the key
  statistic: layer-matched magno-minus-parvo contrast per bin x stage x tp
  (Wilcoxon across subjects) → developmental trajectory of stream alignment.
Caches: `/scratch/byrong/infant/tsn_rdms_orig_input_v1{,_spearman}`.

### A5. Repaired RSA → supplementary
Per-subject inference, crossnobis distances (rsatoolbox), both noise ceilings, stimulus×subject
bootstrap, stream × ROI interaction on ceiling-normalized values. Recompute for existing RDMs
first — the thesis null may partly dissolve with correct statistics alone.

**Figures:** N1 pipeline + R²/NC flatmap + ROI bars w/ ceiling; N2 SPI flatmap + paired
unique-variance dots + shared-variance panel; N3 eccentricity & SF gradients + color ROIs;
N4 texture/shape voxels + in-silico cue-conflict map.

---

## 2. Track B — Model modifications (ranked)

**Diagnosis.** Supervised parallel pathways don't segregate (Bakhtiari et al. 2021); genuine
stream→brain mapping came from magno/parvo-like inputs **plus different objectives per stream**
(Choi & Han 2023). Summing stream features into one shared FC forces a common representational
basis. Accuracy collapse (0.47 vs 0.76) is an input-information problem: no stream ever sees
high-SF luminance; biologically the parvo pathway is the *high-acuity luminance* channel
(fine SF + L−M), not chroma-only.

1. **B1 — Objective asymmetry + unshared readout** (highest leverage, ~1.5–2× compute).
   Separate heads; parvo: classification (or SimCLR); magno: CPC-style prediction over simulated
   gaze sequences (3–5 fixational-jitter/saccade crops per image; optional frame-difference
   input). Gives the 2×2 design (input split × objective split) in which the thesis result is the
   control cell. Full video training (Kinetics/Ego4D) deferred.
2. **B2 — Per-pathway DVD developmental schedule + corrected parvo input** (~zero extra compute).
   Parvo input := high-SF luminance + L−M. Full DVD trajectory per pathway with biologically
   ordered maturation (magno early ~4–6 mo; parvo over years; Dobkins 1999, Hammarrenger 2003),
   ending at mature inputs (recovers accuracy, keeps shape bias per Lu et al. 2026).
   Ablation: magno-first vs parvo-first vs simultaneous.
3. **B3 — Konio third stream** (after B1+B2). Thin (¼-width) S−(L+M) stream, heavy blur;
   dual projection: → parvo stream after block 1 (K→blobs→thin stripes→V4) and late skip →
   magno stream (LGN-K→MT bypass; Sincich 2004). Prediction: selective gain in color-biased
   ventral ROIs. Delivers the repo title.
4. B4 — Learnable L1-sparse cross-stream gates: one ablation; report learned gate values as a
   result. Fix the 75/25 mixing bug first (with legacy flag).

Required regardless: ≥3 seeds per condition, single dataset version (uint8), readouts trained to
convergence, blur-only / channel-swap / RGB-two-stream controls (is it the DKL split or the
transformation?), report accuracy–shape-bias tradeoff curve.

---

## 3. Statistics & reporting standards (whole paper)
- Every bar/point: mean ± CI over ≥3 seeds; subject-level tests for neural data.
- Behavioral shape bias: bootstrap over cue-conflict items; compare conditions with permutation.
- Disclose all limitations (static stimuli for dorsal claims; MiniEcoset scale; supervised
  objective in the base 2×2 cell).

---

## 4. Execution phases

**Working copy:** all code changes live in the git worktree
`Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev` (branch `mpk-dev`);
the user's checkout stays untouched.

**Phase 0 — Infrastructure (now, this machine)**
0.1 Create conda env `mpk` (`/scratch/byrong/anaconda3/envs/mpk`): python 3.11, torch (CUDA),
    himalaya, nibabel, h5py, scipy, pandas, scikit-learn, matplotlib, colour-science, rsatoolbox.
0.2 Fix Interconnected-TSN mixing bug (`symmetric mixing` + `legacy` flag).
0.3 Implement encoding pipeline package `src/analysis/encoding/`:
    `nsd_data.py` (betas/z-scoring/ncsnr/ROIs/design/shared-515), `dkl_frontend.py`
    (RGB→DKL + per-stream transforms, matching training), `features.py` (stream activations,
    pooling, PCA), `banded_ridge.py` (himalaya fits, partitioning, SPI, NC-normalization),
    `stats.py` (subject-level interaction, bootstraps), `run_encoding.py` (CLI, one subject),
    `slurm/encode_subjects.sh` (array 0-7).
0.4 Validate end-to-end on subj01 with an **untrained** TSN (control model), CPU or 1 GPU.

**Phase 1 — Existing-model rescue (needs checkpoints)**
1.1 Obtain thesis checkpoints from Osnabrück (`/share/klab/lzejin/mbenito/...`) — **ask/rsync**;
    otherwise proceed to 2.1 and treat retrained models as canonical.
1.2 Repaired RSA on existing RDMs (in-repo `RDMs/` pickles) with proper stats → quick read on
    whether the null survives.
1.3 Full A1–A3 on trained + untrained models, 8 subjects (Slurm array; GPU for feature
    extraction, `main` for ridge fits if CPU-feasible, else GPU/`standard` QOS).

**Phase 2 — Retraining with fixes (GPU)**
2.1 ≥3 seeds × {RGB baseline, DVD baseline, RGB TSN, DVD TSN, DKL Base, DKL B&CS, M-heavy,
    Inter (fixed mixing)} on uint8 dataset; plus controls (blur-only RGB TSN, channel swap).
    One Slurm **array**, `gpu` partition, watch QOS caps. Blocked on `lightning_kietzmannlab`
    or a minimal local trainer (open item in §0).
2.2 Stream-ablation/logit-decomposition mechanism analysis on cue-conflict (why does behavior
    follow the magno stream when parvo is more predictive?).
**Phase 3 — B1 objective-asymmetry 2×2; Phase 4 — B3 konio stream; Phase 5 — writing.**

## 5. Status log

**2026-08-25 (Phase 0 executed):**
- Env `mpk` built (torch 2.13 cu130, himalaya, kornia, nibabel; DVD scale_free
  via `/scratch/byrong/software/DVD_scale_free` + .pth — its pip packaging omits
  the nested `dvd_scale_free.dvd_scale_free` module).
- Mixing bug fixed (`legacy_asymmetric_mixing` flag); encoding package written
  and committed on branch `mpk-dev` (worktree `.claude/worktrees/mpk-dev`,
  commit 2697b9f). Push blocked: no GitHub credentials in this environment.
- **Smoke test passed**: NSD→DKL front end→untrained-TSN features→banded
  ridge→variance partitioning recovers a synthetic magno/parvo voxel gradient
  (SPI vs ground truth ρ=0.996).
- **Repaired RSA on existing RDMs done** (`results/repaired_rsa/` in worktree):
  noise ceilings vary hugely (ventral .71, parietal .36, hV4 .11, V3d .01);
  general-factor pattern confirmed (one stream better across ALL ROI classes,
  every DKL model, p=.008 simple effects); dorsal/ventral interaction null
  except a suggestive DKL-B effect (magno relatively more parietal, p=.014
  uncorrected; DKL Base same direction p=.08) — reinforces the plan to move
  to encoding-model axes (eccentricity/SF/color/texture).
- Slurm: job 26348386 (DKL min/max, `main`) pending on QOS job cap;
  job 26348518 (subj01 untrained-control encoding pilot, `gpu`) queued with
  dependency on it.

**2026-08-25 (EEG track added & first result):**
- Built + committed (c93f1b7, c13408c) the developmental-EEG RSA track (§A6):
  TSN RDM extractor (campaign-schema caches at
  `/scratch/byrong/infant/tsn_rdms_orig_input_v1{,_spearman}`) and the
  7-sub-age-bin temporal RSA runner.
- **First result (untrained-control TSN, occipital, post-stimulus, matched
  stages):** infants (6/12 mo) show NO stream preference (contrast ≈ 0);
  from 4 yo the EEG aligns increasingly better with the PARVO (L−M) stream,
  peaking at 8 yo (mean magno−parvo ≈ −0.11 at conv1; ~55–59% of timepoints
  individually significant) and persisting into adulthood. Untrained network
  ⇒ the developmental emergence of parvo-stream alignment is carried by the
  input-statistics split itself — consistent with late maturation of
  chromatic/parvocellular vision, and a clean baseline against which trained
  TSNs (pending checkpoints) can be compared.
- Caveat for interpretation: `group_summary.csv` n_subj counts
  subject×subset observations; `stream_contrast.csv` n_subj is true subject
  count (subjects contribute up to 4 subsets, averaged within subject before
  the Wilcoxon).
- **SUPERSEDED 2026-08-26 — do not cite the wording above.** The claim of a
  developmental *emergence* of parvo alignment does not survive the noise-
  ceiling check; see the corrected entry below.

**2026-08-26 (checkpoint-independent work):**
- **Fixed a silent Slurm bug**: the subj01 encoding pilot sat in
  `Reason=BadConstraints` — `gpu` nodes g[001-008] have 95250 MB RealMemory
  and the script asked for 100G, so it could never be scheduled (not merely
  queued). Now `--mem=80G`. Cancelled 26348518; resubmit after smoke test.
- **Solver changed to primal banded ridge** (`himalaya.ridge.BandedRidgeCV`,
  `Y_in_cpu=True`) instead of `MultipleKernelRidgeCV`: with ~1k PCA features
  per band, the kernel formulation would build an n_trials x n_trials kernel
  per band (~7 GB at 30k trials) for no benefit. Primal is cheaper in both
  time and GPU memory, and fits the 11 GB cards on the `gpu` partition.
  (Cluster GPU inventory: gpu=gtx1080ti/93GB RAM, agcichy=a5000/95GB +
  g017 B200 8x/2TB, agkeller=rtx2080ti/192GB, begendiv=h100 4x/515GB.)
- **EEG control battery submitted** (job 26358875, 5-task array, `main`):
  B&CS seeds 0/1/2 (seed robustness), DKL-Base (no blur/CS — are the
  transforms necessary?), channel-swap (blur/CS on rg instead of lum —
  transformation vs channel identity). All runs now also emit input-level
  pseudo-layers (`magno_input`/`parvo_input`: RDMs of the stimuli themselves),
  which separate input statistics from network features as the source of the
  developmental alignment effect.
- **Local training unblocked**: `lightning_kietzmannlab` is unreachable (no
  GitHub credentials, not on PyPI), so `src/training/standalone_trainer.py`
  reimplements the needed pieces on top of the in-repo `TwoStreamNetwork`
  and the verified DVD transforms — thesis defaults (150 ep, Adam 1e-4,
  bs 256, DVD-B augmentations), plus seeds, corrected symmetric mixing, and
  a condition registry covering thesis replications, the blur-only /
  channel-swap controls, and Track B's high-SF-luminance parvo input.
  Prerequisite `scripts/convert_miniecoset_dkl.py` (single pass, reuses the
  cached min/max, uncompressed uint8 for dataloader throughput) is RUNNING
  as job 26358896.
- Cue-conflict images confirmed local (1280 canonical Geirhos stimuli, one
  subdirectory per class); behavioural eval implemented as
  `src/analysis/behavior/cue_conflict.py` — shape bias with item-bootstrap
  CIs **plus a per-stream decomposition** (`full` / `magno_only` /
  `parvo_only`, exact because the classifier is linear on the weighted
  feature sum). This is the analysis that answers the thesis's open question
  §5.2. All 16 Geirhos categories verified present in MiniEcoset
  (airplane→89, bear→103, …, truck→87).
- Smoke tests all green on a compute node (job 26358947 / local): encoding
  solver recovers a synthetic magno/parvo voxel gradient; trainer runs
  `rgb_tsn`, `dkl_bcs`, `dkl_bcs_inter`, `dkl_bcs_parvo_highsf` end to end
  (loss ≈ ln 112 at init, checkpoints written).
- Infrastructure note: **always smoke-test through Slurm**. The login node
  has `RLIMIT_NPROC=4096` and is often saturated, so BLAS thread spawning
  fails mid-import and surfaces as a bogus `KeyboardInterrupt` during
  `import scipy`. Local quick checks need
  `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` capped.

**2026-08-26 — CORRECTED reading of the EEG result (supersedes §A6 log above).**
Occipital noise ceilings (NC_lower, post-stimulus, campaign CSV) rise
monotonically with age: 6mo **−0.007**, 12mo **0.005**, 4yo 0.147, 6yo 0.176,
8yo 0.285, 12yo 0.324, adults 0.364. Consequences:
1. **Infant bins carry no reliable stimulus-driven signal** (NC ≈ 0), so the
   earlier "infants show no stream preference" is an absence of measurable
   signal, not a finding. No model claim of any kind can be made for 6/12 mo
   from this measure.
2. Raw magno-minus-parvo contrast tracks NC almost exactly, so the apparent
   "developmental increase" is confounded with data reliability and must not
   be reported as development.
3. What the data actually show, from 4 yo on (where NC is measurable): the
   **parvo (L−M) stream aligns positively** (mean ρ ≈ +0.02…+0.08, peak at
   8 yo) while the **magno stream does not align at all** (ρ ≈ 0 to −0.04 at
   every bin). So the honest claim is *stream-selective* alignment, not
   *developmentally emerging* alignment.
4. Absolute ρ (0.02–0.08) is small against NC (0.15–0.36): an untrained TSN
   captures only a small share of reliable EEG geometry, as expected.
5. Controls (job 26358875): effect is **seed-robust** (3 seeds agree to
   ~0.01); the **network amplifies it over raw input statistics** (fraction
   of significant timepoints at 8 yo: input 0.05 vs conv1 0.59 vs avgpool
   0.57) — so it is not a pure stimulus-statistics artefact; blur/CS are
   **not necessary** but strengthen it (adults −0.035 → −0.049); the
   **channel swap attenuates it** (adults −0.049 → −0.011), so channel
   identity and the transformation both contribute.
6. **Methods fix needed**: the scale-free ratio index
   (ρm−ρp)/(|ρm|+|ρp|) degenerates to exactly ±1 whenever the two streams
   have opposite signs (which is the common case here). For the paper use the
   **noise-ceiling-normalized difference** (ρp−ρm)/NC_lower with
   subject-level CIs, and restrict age comparisons to bins with NC above a
   pre-declared threshold.

**2026-08-27 — FIRST REAL ENCODING RESULT (2 of 8 subjects).**
Two elements of an earlier array completed before the runtime fix (subj04 in
10h41m, subj08 in 19h56m); the other six were OOM-killed. Feature caches for
all 8 subjects survived, so reruns skip extraction.

Untrained-control TSN, `nsdgeneral`, voxels with NC>0.1, median per ROI of
noise-ceiling-normalized unique variance:

| ROI | subj04 uM-uP | subj08 uM-uP | consistent? |
|---|---|---|---|
| early | +0.066 | +0.045 | yes (magno) |
| midventral | **-0.016** | **-0.044** | **yes (parvo)** |
| midparietal | **+0.057** | **+0.044** | **yes (magno)** |
| midlateral | +0.004 | -0.005 | no |
| ventral | +0.013 | -0.009 | no |
| lateral | +0.027 | -0.004 | no |
| parietal | +0.015 | -0.004 | no |

Difference-of-differences (dorsal-leaning minus ventral-leaning ROIs):
subj04 **+0.0375**, subj08 **+0.0470** — same sign, similar magnitude, from
independently fitted subjects.

**This is the crossover the thesis looked for and did not find**: magno
favours midparietal, parvo favours midventral. The encoding-model +
variance-partitioning approach recovers a dissociation that group-averaged
RSA missed, vindicating the Track A redesign. Model quality is modest but
real: R2_joint/NC median ~0.09 overall, ~0.23-0.27 in early visual cortex.

Caveats that must travel with this number:
1. **n=2 subjects — descriptive only, no group inference.** The pre-declared
   subject-level interaction test needs all 8.
2. **Untrained network**, so this reflects input statistics plus architecture,
   not learned representations. Interesting in itself, but the trained
   comparison is what licenses any claim about learning.
3. Voxelwise Wilcoxon p-values in the scratch analysis are anti-conservative
   (voxels are not independent) and are within-subject descriptors only.
4. Only early / midventral / midparietal are consistent across the two
   subjects; the higher-level ROIs disagree in sign.


**2026-08-28 — GROUP RESULT, ALL 8 SUBJECTS. Track A core analysis complete.**
Array 26368287 finished the remaining five subjects overnight; all 8
`untrained_seed0_subj*.npz` exist. Coverage is uniform (12.7k–17.9k voxels,
8.5k–13.5k with NC>0.1, median R²_joint/NC 0.11–0.17), so no subject is
carried by a degenerate fit. Analyses in `tmp_analysis/{group_interaction,
ecc_gradient,visualrois}.py`; unit of inference is the subject throughout.

**A2 — pre-declared stream × cortical-axis interaction (n=8).**
Difference-of-differences, dorsal{midparietal,parietal} minus
ventral{midventral,ventral}: **+0.0266**, sign-flip permutation
**p = .0079**, **8/8 subjects positive**, **dz = +1.64**. With 8 subjects
there are only 2⁸ = 256 sign patterns, so p = .0079 is the second-smallest
attainable value — the test is at its resolution limit, not weakly
significant. Per-ROI simple effects (FDR q=.05): early **+0.052** (p=.008,
survives), midparietal **+0.030** (p=.008, survives), midventral **−0.023**
(p=.016, survives), lateral +0.021 (p=.039, does not survive FDR);
midlateral, ventral, parietal null.

*The double dissociation is incomplete and must be reported as such.* On the
pooled ROI sets the dorsal simple effect is significant (median +0.018,
p=.016) but the ventral one is not (median −0.012, **p=.078**). Per
Nieuwenhuis et al. (2011) the interaction is the valid claim; "magno→dorsal
AND parvo→ventral" as a double dissociation is not yet licensed.

**A3.1 — eccentricity gradient (n=8). This is the strongest panel, as §1
predicted, and it is now the headline candidate.**
Median SPI by `prf-eccrois` band, averaged over subjects:

| ecc band | 0.5° | 1° | 2° | 4° |
|---|---|---|---|---|
| mean SPI | **−0.449** | −0.065 | +0.211 | **+0.326** |

A full **sign reversal**: parvo-dominant at the fovea, magno-dominant in the
periphery. Per-subject slope of SPI on log-eccentricity: mean **+0.322**,
sign-flip **p = .0079**, Wilcoxon p=.0078, **dz = +1.77**, **8/8 subjects
positive**, and monotonic in every subject. Companion contrast
SPI(4°)−SPI(0.5°) = **+0.775**, 8/8 positive. This is the textbook M/P
retinotopic signature recovered from a model that was never fitted to
retinotopy.

**A3.3 — retinotopic areas V1→hV4 (n=8).**
hV4 is the **only** area where parvo-unique variance exceeds magno-unique
(mean SPI **−0.223**); every other area is magno-preferring. SPI(hV4) −
SPI(V1) = **−0.538**, **0/8 subjects positive**, p=.0078.
Mechanism, stated precisely because it is not what the naive prediction says:
magno-unique variance **falls monotonically** along the hierarchy (V1v 0.207
→ hV4 0.055, a 3.8× drop) while parvo-unique is roughly **flat** (0.111 →
0.087). The crossover is magno *losing* explanatory power, not parvo gaining
it. Matched-eccentricity control — dorsal vs ventral bank of V1/V2/V3 —
is **null** (+0.091, p=.109, 6/8), so the hV4 effect is not a generic
ventral-stream effect.

**THE CONTROL THAT DECIDES WHETHER ANY OF THIS IS PUBLISHABLE.**
All three results are consistent with a deflationary account: the magno
stream is low-pass by construction (blur σ=4 + 48-month CSF cut), so it
should predict low-SF, peripheral, non-chromatic voxels better *for trivial
reasons*, with no biology involved. Job **26373186** (array 0-7%3) reruns
the identical pipeline and the identical untrained network with
`--no-lum-blur --no-lum-cs`, i.e. magno = raw luminance, parvo = raw L−M.
- If the eccentricity gradient **survives**, it is carried by
  achromatic-vs-chromatic channel content — the real biological claim.
- If it **collapses**, the gradient is a restatement of the blur and must be
  demoted from a finding to a front-end validation.
Nothing in this section should be written up before that array returns.

Further caveats carried forward: the network is **untrained**, so these are
input-statistics-plus-architecture effects, not learned representations —
the trained comparison is what licenses any claim about learning; and
`prf-eccrois` bands are coarse (4 rings), so the gradient is a trend, not a
continuous pRF fit.

**2026-08-28 — GPU throughput is a blocker for Track B.**
Timing probe 26368283 on an A5000: 60 steps at bs=256 in 270.8 s ≈ **57
img/s** → ~1.3 h/epoch → **~8 days for one 150-epoch run**, which makes a
seed × condition array impossible as designed. Two candidate causes, being
separated by profiler 26373188 (h5 dataloader vs the per-batch DVD front end
vs model fwd/bwd, at two worker counts): the probe ran on g017 alongside
three of my other jobs, and `StreamInputBuilder` calls
`apply_scale_free_contrast` on **every batch** inside the training loop.
If the DVD front end dominates, the fix is to precompute the transform —
the schedule is fixed at 48 months, so the transform is a constant function
of the image and can be baked into the h5 exactly as the DKL conversion was.

**2026-08-28 (later) — statistic harmonized, figure built, training array launched.**

*Reporting-consistency fix.* Two summary statistics were in use across my own
analyses: `stats.py` summarizes an ROI as median(U_m) − median(U_p) (a
difference of medians), while the eccentricity, robustness and figure code
use median(U_m − U_p) (the median of the paired per-voxel difference). They
are not identical. All headline numbers are now reported with the **paired**
statistic (`tmp_analysis/group_interaction_paired.py`), which is the more
natural summary of a within-voxel contrast and the one the figure draws.

Harmonized values, n=8, NC>0.1:

| ROI | median(U_m−U_p) | p | subjects | dz |
|---|---|---|---|---|
| early | **+0.0556** | .0078 | 8/8 | +3.65 |
| midparietal | **+0.0286** | .0078 | 8/8 | +1.75 |
| lateral | +0.0174 | .023 | 7/8 | +1.29 |
| parietal | +0.0106 | .078 | 6/8 | +0.81 |
| ventral | +0.0065 | .109 | 6/8 | +0.74 |
| midlateral | −0.0033 | .055 | 1/8 | −0.79 |
| midventral | **−0.0162** | .039 | 1/8 | −0.89 |

Pre-declared interaction: **+0.0245, p_perm = .0079, 8/8 subjects, dz = 1.46**
(was +0.0266 under the difference-of-medians statistic; the conclusion does
not move). Simple effects: dorsal **+0.0196, p = .016**; ventral **−0.0049,
p = .31**.

*The double-dissociation caveat gets stronger, not weaker.* Under the paired
statistic the ventral simple effect falls from p = .078 to **p = .31**. The
pre-declared pooling is the reason: `midventral` on its own is
parvo-preferring (−0.0162, 7/8 subjects) but `ventral` leans very slightly
the other way (+0.0065), and averaging them cancels most of the effect. The
pooled contrast was fixed in advance and is reported as primary; the
per-ROI breakdown showing that the parvo effect is specifically
**midventral** is exploratory and must be labelled as such.

*Figure.* `tmp_analysis/fig_streams.py` → `results/figures/fig_streams.{png,pdf}`.
Three panels (eccentricity, V1→hV4, the interaction), all 8 subjects drawn
individually since the subject is the unit of inference. `--compare` overlays
a second model, which is how the blur control will be shown: if the gradient
is a restatement of the low-pass front end, the red lines in panel A flatten.

**Training array 26373625 launched: 5 conditions x 3 seeds = 15 runs.**
Conditions: dkl_bcs, dkl_base, rgb_tsn, dkl_bcs_swap, dkl_bcs_parvo_highsf.
Deferred: dkl_b, dkl_cs, dkl_bcs_mheavy, dkl_bcs_inter, rgb_tsn_blur.

Measured first-epoch times (8 jobs, 4 per node, g010/g011): **583–734 s**,
i.e. 359–452 img/s — better than the 130–200 img/s I projected, so
**~24–31 h per run and ~2.6 days for all 15**. Two checks that the
performance model is right: `dkl_base` (585 s) and `dkl_bcs` (734 s) differ
by 149 s, almost exactly the profiled DVD front-end cost (143 ms × 1028
steps = 147 s); and val top-1 after one epoch is 0.07–0.23 against 1/112 =
0.009 chance, so every condition is learning. `dkl_base` learns fastest
(0.18–0.23) and `dkl_bcs` slowest (0.07–0.14), as expected when blur+CSF
removes information.

*Self-imposed constraint while the array runs:* no further edits to
`standalone_trainer.py`. A job that resumes from `checkpoint_last.pt` after
an edit would run different code than its own earlier epochs. The
RandomEqualize optimization (a ~20% win) waits until the array drains.

**Still the gating item:** the `--no-lum-blur --no-lum-cs` control array
(26373186) has not returned. Until it does, none of the three neural results
above can be written up as biology rather than as a consequence of the magno
stream being low-pass by construction.
## 6. Risks / open items
- `lightning_kietzmannlab` not public → minimal local trainer fallback (plain Lightning loop
  reusing `TwoStreamNetwork`).
- Trained thesis checkpoints not on this cluster → Phase 1 blocked on transfer; Phase 0/2 are not.
- MiniEcoset h5 here is the `texture2shape` variant → verify it matches the thesis's 112-class
  256px version before retraining.
- himalaya on GPU with ~30k trials × ~8k features × ~30k voxels: fits in A100 80GB with
  kernelized solver; fall back to chunked voxel batches.
- DVD CSF-cut implementation must exactly match the `scale_free` branch used in the thesis
  (import DVD package rather than re-implementing).

---

## 7. Analysis review, 2026-08-28 — gaps and additions

Review of what is built vs. what the claim needs. Two items were executed
during the review (§7.1); the rest is a ranked queue (§7.2–7.4).

### 7.1 Done during the review (no refitting required)

**Kastner2015 ROIs attached to the existing fits.** The encoding runs used a
deterministic `nsdgeneral` mask, so any ROI atlas on disk can be indexed onto
the stored per-voxel arrays after the fact. `Kastner2015.nii.gz` is present
for all 8 subjects and contains TO1/TO2 (= hMT+), VO1/VO2, V3A/V3B and
IPS0–5. `tmp_analysis/kastner_rois.py`, n=8, NC>0.1:

| ROI | median(Uₘ−Uₚ) | subjects |
|---|---|---|
| V1d / V1v | +0.086 / +0.083 | 8/8 |
| V2v / V2d | +0.077 / +0.058 | 8/8 |
| PHC2 | +0.054 | 8/8 |
| V3d / V3A / PHC1 / V3v | +0.041 / +0.041 / +0.038 / +0.036 | 8/8 |
| **TO2 / TO1 (hMT+)** | +0.021 / +0.018 | 6/8, 7/8 |
| VO2 / VO1 | +0.011 / −0.002 | 7/8, 4/8 |
| LO2 | −0.010 | 1/8 |
| **hV4** | **−0.012** | 2/8 |

- **hMT+ is magno-preferring: +0.0197, p = .023, 7/8 subjects, dz = 1.29.**
  §A3.5 wrote the dorsal/MT test off as a limitation of static stimuli. It is
  testable and it comes out in the predicted direction. It should be promoted
  from "limitation" to a reported (secondary) result.
- **The colour prediction fails at VO: +0.0044, p = .46.** hV4 remains the
  only parvo-preferring area. The hMT+ − VO interaction is **not**
  significant (+0.0153, p = .109), so this is *not* a second double
  dissociation. Report the null.
- **The 20-ROI profile is highly reproducible**: leave-one-subject-out
  Spearman **ρ = +0.79** (range .59–.88). Useful reviewer-proofing: the map
  is not carried by one or two subjects.
- The profile ranks roughly by hierarchical level *and* eccentricity
  (V1 > V2 > V3 > rest; peripheral-biased PHC/V3A high; foveal hV4/LO2 low),
  consistent with the already-reported mechanism (magno-unique *declines*
  along the hierarchy) rather than with a clean dorsal/ventral split.

**Limitation discovered while building it.** `prf-eccrois` labels **0%** of
TO1/TO2 and VO2 voxels, ~9–23% of VO1, and only **~33% of the reliable mask
overall**. So (i) the eccentricity gradient is estimated on the retinotopic
core only, not on the mask it is implicitly generalized to, and (ii) the
eccentricity-residualized version of the hMT+/VO contrast **cannot be
computed** with this atlas. Controlling retinotopy outside early visual
cortex requires the continuous pRF maps (§7.2).

**Bug fix: a whole eccentricity band was being dropped.** `prf-eccrois` has
**five** labels (`ecc0pt5, ecc1, ecc2, ecc4, ecc4+`); `stats.py:ECC_LABELS`
defines four. Every eccentricity number reported so far silently excluded
`ecc4+` — which is the **largest** band (median 948 voxels/subject) and the
most peripheral, i.e. the most magno-diagnostic voxels in the mask.
`tmp_analysis/ecc_gradient5.py`:

| band | 0.5° | 1° | 2° | 4° | **4+°** |
|---|---|---|---|---|---|
| mean SPI | −0.449 | −0.065 | +0.211 | +0.326 | **+0.342** |

The gradient stays monotonic across all five bands and the effect **grows**:
dz **+1.77 → +2.01**, SPI(outer)−SPI(0.5°) = +0.791, still 8/8 subjects and
still at the permutation floor (p = .0079). The slope statistic changes
(+0.322 → +0.261) only because the x-range extends. Fix `ECC_LABELS` and
re-report. Note the gradient **saturates** between 4° and 4+°, which is worth
stating rather than smoothing over.

### 7.2 Highest-value additions (ranked)

1. **Input-level null model — the control that decides how much the network
   matters.** Add, as extra bands, a Gabor-energy (or pixel-PCA)
   representation of each stream's *own transformed input*, and refit. If the
   eccentricity gradient and the V1→hV4 crossover are fully carried by the
   input bands, the untrained ResNet is doing nothing and those panels are
   front-end validations, not model results. This is the exact control that
   rescued the EEG track (§log 2026-08-26: input 0.05 vs conv1 0.59 of
   timepoints significant) and a top-tier reviewer will demand it. Note this
   is a *different* question from the blur control (26373186): blur asks
   whether the magno *input* is trivially low-pass; this asks whether the
   *network* adds anything at all.

2. **Continuous pRF maps** (`prf_eccentricity/prf_size/prf_angle.nii.gz`
   from `nsddata/ppdata`, small download). Buys three things at once:
   (a) turns a 5-bin trend into a continuous within-subject regression over
   the whole mask instead of the 33% that carries a band label;
   (b) **dissociates eccentricity from pRF size**, which the bands confound
   entirely — "magno explains peripheral voxels" and "magno explains
   large-pRF voxels" are different claims and only one is about retinotopy;
   (c) makes the residualized hMT+/VO contrast in §7.1 computable.

3. **Spatial frequency as a mediator, not as an extra panel.** §A3.2 is
   currently framed as an independent result, but preferred SF and
   eccentricity are the same axis in cortex, so an SF panel would be the
   eccentricity panel twice. The informative version is mediation: fit a
   Gabor-pyramid model per voxel for preferred SF, then test whether
   SPI ~ log-ecc survives partialling SF, and vice versa. Infrastructure is
   shared with item 1, so it is nearly free once that exists.

4. **Reliability of the SPI map itself.** Split the training trials in half,
   fit twice, correlate the two SPI maps within subject. Gives a noise
   ceiling for SPI and licenses statements of the form "the eccentricity
   gradient explains X% of the *reliable* SPI variance". Two extra fits.

5. **Stimulus bootstrap** (§A2, still unrun). With n=8 the sign-flip test is
   pinned at its floor (p = .0039), so subject-level inference has no
   headroom left; resampling the 515 test images gives a second, independent
   inferential axis and per-subject CIs. Requires storing test predictions —
   a cheap re-run.

6. **Encoding-form general-factor control.** The RSA general-factor confound
   (a global stream main effect of +0.0218, §2026-08-28 novelty check) has a
   direct analogue here: the two streams' feature spaces differ in total
   variance and effective dimensionality, which alone can shift unique
   variance. Report participation ratio and feature-norm per stream and show
   the gradients survive matching. Cheap, and it closes the objection that
   the thesis's method lost to.

### 7.3 The biggest hole in the story

The working claim has three legs. Leg (c) — each stream uniquely explains
distinct cortical variance — is done (gated on the blur control). Leg (b) —
integration yields human-like shape bias — is implemented
(`src/analysis/behavior/cue_conflict.py`, with the exact per-stream
decomposition) and waits on trained checkpoints. **Leg (a) — that the input
split induces shape- vs texture-biased streams — currently has no direct
measurement anywhere in the project.** It is asserted in the title and tested
nowhere.

- **7.3a Stream-level shape/texture probe (do this next; runnable today).**
  Push the Geirhos cue-conflict / stylized stimuli through each stream and
  measure shape-vs-texture decodability *from each stream's features*
  (linear probe on shape label vs texture label, matched dimensionality).
  This is the mechanistic middle link between the input split and the
  behavioural shape bias, it needs no fMRI and no trained checkpoints to
  produce a baseline, and it re-runs unchanged on the trained models when the
  array lands.
- **7.3b In-silico cue-conflict → cortex (§A3.4, the flagship figure).**
  Push cue-conflict images through the *fitted encoding models* → per-voxel
  predicted shape-bias index → correlate with SPI across voxels. This is the
  single panel that makes Track A and the behavioural result one paper
  instead of two. Blocked on nothing but effort.

### 7.4 Framing decisions to make *before* the checkpoints land

- **Pre-declare the untrained-vs-trained comparison now.** The untrained
  network already produces the headline, so "what does training add?" is the
  question the paper will be judged on, and deciding what each outcome means
  *after* seeing it would be storytelling. Write down now what "training
  increases / does not change / decreases stream differentiation" each imply.
  If the answer is that architecture plus input statistics suffice, that is a
  strong and topical claim in its own right (cf. the untrained-network
  critiques in the Brain-Score literature) — but only if declared in advance.
- **Single-stream and pretrained baselines.** Reviewers will ask whether the
  two-stream split buys anything over a matched single-stream ResNet-18 on
  RGB and over an ImageNet-pretrained network, in *total* explained variance
  as well as in differentiation. `rgb_tsn` in array 26373625 covers part of
  this; a single-stream control does not yet exist.
- **Konio (§B3) via the cheap route.** A third ¼-width S−(L+M) stream run
  **untrained** through the existing encoding pipeline costs one array and no
  training, and delivers the repository's title with a pre-registered
  prediction (K → VO1/VO2/hV4). Given that VO came out null for parvo
  (§7.1), this is also the honest test of whether the colour prediction fails
  because the *parvo* channel is the wrong model of colour cortex.

---

## 8. 2026-08-29 — THE BLUR CONTROL RETURNED. Headline reassigned.

Array 26373186 finished all 8 subjects (`untrained_seed0_noblur`): identical
pipeline, identical untrained network, `--no-lum-blur --no-lum-cs`, so magno
= raw luminance and parvo = raw L−M. The two models differ **only** in
whether the luminance channel is low-passed. Per §A3/§5 this was pre-declared
as the gate on every neural result.

**Sanity first: the control is not degenerate.** It fits *better* than the
main model (median R²_joint/NC 0.1435 vs 0.1191), as expected when the blur
stops destroying information.

### 8.1 A statistical fork that had to be resolved first

In the control, magno (raw luminance) dominates unique variance almost
everywhere, so SPI = (Uₘ−Uₚ)/(Uₘ+Uₚ) sits at +0.44…+0.80 — deep in the
compressive tail of a bounded ratio. A real gradient could be present there
and still look flat. Every contrast is therefore reported under **both**:

- `raw`  = median(Uₘ−Uₚ) — unbounded; *does magno's absolute advantage grow?*
- `norm` = median(Uₘ−Uₚ)/median(Uₘ+Uₚ) — scale-free; *does the relative
  stream preference shift?*

A result is claimed only where the two agree
(`tmp_analysis/blur_verdict.py`, `blur_control_unbounded.py`).

### 8.2 Verdict table (n=8, paired statistic, NC>0.1)

| contrast | blur | no blur | verdict |
|---|---|---|---|
| ecc slope, raw | +0.0470 (p=.008, 8/8) | +0.0413 (p=.008, 8/8) | **survives** (diff p=.109) |
| ecc slope, norm | +0.2518 (p=.008, 8/8) | +0.0891 (p=.055, 6/8) | **collapses** (diff p=.016) |
| foveal band (<0 ⇒ parvo wins) | −0.0574 (0/8) | **+0.0513 (7/8)** | **DEAD — sign flips** |
| **hV4 − V1v, raw** | −0.1179 (p=.008, 0/8) | **−0.2081 (p=.008, 0/8)** | **survives, larger** |
| **hV4 − V1v, norm** | −0.4691 (p=.008, 0/8) | **−0.4877 (p=.008, 0/8)** | **survives** (diff p=.383) |
| interaction dorsal−ventral | +0.0245 (p=.008, 8/8) | +0.0122 (p=.195, 6/8) | **DEAD** |
| hMT+ | +0.0181 (p=.016) | +0.0443 (p=.008) | survives but **not specific** |
| hMT+ − VO | +0.0128 (p=.109) | −0.0087 (p=.383) | **DEAD** |

### 8.3 What this means — the eccentricity headline must be given up

**The sign reversal is an artefact of the blur.** Without the low-pass front
end the foveal band flips from parvo-preferring (−0.057) to magno-preferring
(+0.051), 7/8 subjects, and SPI is positive in all five bands. "Parvo
dominates the fovea, magno the periphery" — the candidate headline of
2026-08-28 — **is not a biological result** and must not be reported as one.

What honestly survives of §A3.1 is weaker and must be stated in these terms:
magno's *absolute* unique-variance advantage grows with eccentricity in both
models (raw slope +0.047 vs +0.041, 8/8 both, difference n.s.), but the
*relative* preference shifts with eccentricity only when the magno stream is
blurred. So the gradient is real in magnitude and artefactual in sign.

**The dorsal/ventral interaction is dead** (p=.195 without blur), and with it
the last remnant of "magno→dorsal, parvo→ventral". §1's reframe was right to
drop it; it should now be dropped entirely rather than reported weakly.

**hMT+ is not a specific result.** It gets *stronger* without blur (+0.044,
8/8, dz=3.09) — but so does VO (+0.053, 8/8), so hMT+−VO is null and even
flips sign. Without blur essentially everything is magno-preferring; the blur
manufactures apparent regional specificity by handicapping magno until parvo
can win somewhere. This is the same general-factor pathology that defeated
the thesis's RSA, reappearing in encoding form — worth reporting *as* a
methodological point.

### 8.4 THE NEW HEADLINE: the V1→hV4 decline, which the control strengthens

The one contrast that survives on both statistics, and the only one that is
*larger* without blur:

> Magno-unique variance falls from V1v to hV4 while parvo-unique variance
> holds up. norm: −0.4691 (blur) vs −0.4877 (no blur), difference p=.383,
> **0/8 subjects positive in either model**.

Because it is undiminished when the low-pass filter is removed entirely, it
**cannot** be a spatial-frequency artefact — the strongest possible outcome
for this panel.

**Eccentricity-matched control** (`tmp_analysis/hv4_matched.py`): hV4 and V1v
compared within the *same* prf-eccrois band, so retinotopy cannot drive it.

| within band | 0.5° | 1° | 2° | 4° | 4+° | pooled |
|---|---|---|---|---|---|---|
| blur, raw | −0.007 | −0.033 | −0.084** | −0.161** | −0.166** | **−0.092**, 0/8 |
| **no blur, raw** | −0.085* | −0.143** | −0.181** | −0.246** | −0.252** | **−0.183**, 0/8 |
| **no blur, norm** | −0.503* | −0.447* | −0.413** | −0.487** | −0.524** | **−0.468**, 0/8 |

(* p<.05, ** p=.0078 = permutation floor.) In the control the effect holds in
**every band including the fovea**, 0–1/8 subjects positive, dz −1.2 to −5.5.
In the blurred model the two foveal bands are null — i.e. the blur *weakens*
this result rather than creating it.

### 8.5 Consequences for the paper

1. Figure panel A (eccentricity) is demoted to a front-end validation and a
   supplementary panel. Panel B (V1→hV4) becomes the lead.
2. The claim to lead with: **the chromatic and achromatic input channels
   diverge along the ventral hierarchy — the achromatic stream's unique
   contribution declines from V1 to hV4 at matched eccentricity — and this
   does not depend on any spatial-frequency manipulation.** That is a
   colour-cortex claim, and hV4 is the right place for it.
3. Every remaining ROI-specificity claim (hMT+, dorsal/ventral, VO) is
   reported as null, with the global magno dominance in the control given as
   the reason. Do not bury this: it is the general-factor problem, and naming
   it is what separates this analysis from the thesis's.
4. Still untrained. §7.4's pre-declaration is now urgent, because the
   surviving result is precisely the one the trained models must be tested
   against. `dkl_base` ×3 seeds and `rgb_tsn` ×2 seeds completed on
   2026-08-29 (~20 h each); `dkl_bcs` ×3 still running.

---

## 9. 2026-08-29 — Leg (a) measured for the first time, and a caught bug

### 9.1 A silent failure that would have become a "finding"

`features.build_model` assumed a Lightning checkpoint and unconditionally
kept only keys prefixed `network.`. `standalone_trainer.py` saves
`model.state_dict()` directly, with no prefix, so the filter matched nothing,
the state dict became empty, and `load_state_dict(strict=False)` **loaded 0
of 242 parameters while returning normally**. Every "trained" model was a
random network.

Caught because the probe returned byte-identical numbers for
`untrained_noblur`, `dkl_base_seed0_trained` and `dkl_base_seed1_trained`.
The log said so plainly — "202 missing, 0 unexpected keys" — but nothing
raised.

This is the worst failure mode available to this project: it produces a
confident **"training changes nothing"**, which §7.4 had already flagged as a
live and *publishable-sounding* outcome. It would have been read as a result.
`build_model` now detects the dialect and **raises** when a checkpoint
populates no parameters. Jobs 26383592/26383633 were cancelled, outputs and
feature caches deleted, and relaunched as **26383806** (encoding) and
**26383807** (probe).

Standing lesson: every trained-vs-untrained comparison in this project must
carry a positive control that the checkpoint actually loaded. Identical
numbers across conditions are a bug signature, not a null.

### 9.2 The stream shape/texture probe (§7.3a) — first direct test of leg (a)

`src/analysis/behavior/stream_shape_texture.py`. Geirhos cue-conflict images
(1200 usable, 16 shape × 16 texture classes, orthogonal by construction);
cross-validated linear probes on stage-pooled features, both streams reduced
to the same 256 PCA components. `bias = acc(shape) − acc(texture)`; chance
.0625 for both probes, so the two accuracies are directly comparable.

| model | magno shape/tex → bias | parvo shape/tex → bias | **M−P** |
|---|---|---|---|
| untrained, blur+CS | .977 / .181 → **+0.796** | .325 / .412 → −0.087 | **+0.882** |
| untrained, no blur | .860 / .443 → **+0.417** | .325 / .412 → −0.087 | **+0.503** |
| **trained dkl_base** s0 | .734 / .953 → **−0.219** | .352 / .871 → −0.519 | **+0.300** |
| trained dkl_base s1 | .725 / .938 → −0.213 | .370 / .879 → −0.509 | +0.296 |
| trained dkl_base s2 | .710 / .956 → −0.246 | .388 / .872 → −0.484 | +0.238 |
| trained swap s0 | .781 / .966 → −0.185 | .315 / .521 → −0.206 | **+0.021** |
| trained swap s1 | .772 / .938 → −0.166 | .267 / .512 → −0.244 | +0.078 |

**The predicted direction holds everywhere: M−P > 0 in all seven models.**
Leg (a) has its first direct evidence — the magno stream is reliably *more
shape-biased relative to* the parvo stream, and this is a property of the
representations, not of the classifier's decisions.

Three things must be reported with it, none of them flattering:

1. **Most of the absolute shape bias is the blur, not the split.** Untrained
   magno bias falls +0.796 → +0.417 when the low-pass front end is removed.
   And .977 shape / .181 texture is near ceiling and near floor respectively,
   so that +0.796 is range-limited and should not be quoted as an effect
   size.
2. **Training makes both streams texture-biased and *shrinks* the
   asymmetry.** On a fixed front end, M−P falls +0.503 (untrained) → +0.300
   (trained), and magno flips from shape-preferring (+0.417) to
   texture-preferring (−0.219). This is the classic Geirhos CNN texture bias
   appearing, and it works *against* the input-induced specialisation rather
   than consolidating it.
3. **The channel swap nearly abolishes the asymmetry** (+0.021 / +0.078 vs
   +0.300 / +0.296 / +0.238 for `dkl_base`), so the effect tracks which
   channel carries the developmental transform, not channel identity alone.
   Caveat: `dkl_base` and `dkl_bcs_swap` differ in *both* front end and
   training, so this is suggestive, not a single-variable contrast. The
   contrast that closes it is `dkl_bcs` (transform on magno), still training.

### 9.3 PRE-DECLARATION, made before `dkl_bcs` finishes

Point 2 is measured on `dkl_base` — the condition **without** the
developmental visual diet. The DVD claim is precisely that the diet protects
shape bias. So the sharp, currently-open question is what `dkl_bcs`
(blur + CS, still training as 26373625 tasks 0–2) does. Writing the reading
down now, so it is not chosen after seeing the number:

- **M−P(dkl_bcs trained) ≈ M−P(dkl_bcs untrained) = +0.882**, or magno bias
  stays positive → the developmental diet *preserves* the input-induced
  specialisation that ordinary training erodes. This supports the project's
  working claim and is the strong outcome.
- **M−P(dkl_bcs trained) ≈ +0.30, magno bias goes negative** → the diet
  delays but does not prevent the collapse to texture bias; the honest claim
  becomes that the input split sets the *initial* geometry and training
  overwrites it. Leg (a) then has to be restated as a claim about
  initialisation, not about learned streams.
- **M−P(dkl_bcs trained) < M−P(dkl_base trained)** → the diet actively hurts
  specialisation. This would contradict the working claim and must be
  reported as such.

The comparison must be trained-vs-untrained *within* `dkl_bcs`, since the
untrained blur+CS number (+0.882) is inflated by the probe ceiling noted
above.

---

## §10 Status check 2026-08-29 — two corrections to §9

### 10.1 The swap numbers in §9 are from a model that was 20% trained

`dkl_bcs_swap_seed0/1` were still training when the shape/texture probe read
their `checkpoint_last.pt`. Timeline from the run directories:

| | started | first ckpt (ep 25) | probe read `checkpoint_last.pt` |
|---|---|---|---|
| `dkl_bcs_swap_seed0` | 12:05 | 16:54 | **17:45** |

So the probe measured roughly epoch 28 of 150, whereas the `dkl_base` numbers
it was compared against are from fully converged 150-epoch runs. `checkpoint_last.pt`
is overwritten continuously, so nothing in the output file records this — the
JSON looks identical to a converged one. **Any probe that reads
`checkpoint_last.pt` must first assert `history.json` has the full epoch count.**

Direction of the bias, stated before re-running: the §9 trend is that *more*
training *shrinks* M−P, so an undertrained model should sit **between**
untrained (+0.503) and converged `dkl_base` (+0.238…+0.300). The swap sits far
below all of them (+0.021, +0.078). Undertraining therefore does not explain
the swap result under the trend already established, and if anything understates
it. But this is not a clean contrast (front end, training length, and loss level
all differ), so **the swap claim is provisional and must be re-run on the
converged checkpoints** before it appears in any figure or text.

`dkl_base_seed0/1/2` and `untrained_*` are unaffected — those checkpoints are final.

### 10.2 Classification accuracy is now on the record

Fully converged, top-1 on held-out miniEcoset (112 classes):

| condition | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `dkl_base` | 0.6998 | 0.6982 | 0.7055 |
| `rgb_tsn`  | 0.6643 | 0.6836 | — (training) |
| `dkl_bcs`  | ~0.43–0.48 at ep 146 (still training) | | |

Two things follow. First, **the DKL front end is not a handicap**: `dkl_base`
beats the RGB two-stream control by ~2–4 points, so the opponent-channel split
is at worst free and plausibly helpful. Second, **the developmental diet costs
~25 points of accuracy**. That is expected — blur plus reduced contrast destroys
information — but it must be stated plainly in the paper rather than omitted,
and it means every `dkl_bcs` representational claim has to be defended against
"this model is simply worse". The matched comparison for that is `dkl_base`,
not `rgb_tsn`.

### 10.3 A scheduling bug that cost a day

`slurm/encode_subjects.sh` requested `--mem=190G`. Slurm reads `G` as GiB, so
that is 194560 MB, while the 50 mid-tier `main` nodes advertise 192080 MB. The
request therefore fit on only the 18 large-memory nodes — the comment in the
script claiming "68 main nodes have >=190G" was wrong by exactly this unit
conversion. Measured peak is 137G (`sacct` MaxRSS), so the request is now
`--mem=160G` (163840 MB): fits on 68 nodes, 20% headroom. Pending job 26383806
updated in place via `scontrol`.

---

## §11 The thesis checkpoints arrived (2026-08-29) — what still needs training

`/scratch/byrong/tsn_checkpoints/` now holds the two models the BA thesis
actually reports, with a full provenance record (SwissTransfer link, md5 of the
archive, per-file sha256, `unzip -t` verified). This closes the §6 risk item
"obtain thesis checkpoints from Osnabrück OR treat local retrains as canonical" —
we now have both, which is strictly better than either.

| downloaded | our equivalent |
|---|---|
| `dkl_blur_cs_dvd_trans` | `dkl_bcs` |
| `dkl_blur_cs_s1_90` | `dkl_bcs_mheavy` (deferred) |

Verified by loading: Lightning 2.6.1, epoch 147 / step 152292, 242 tensors,
`network.stream{1,2}_*` prefixes, **1 input channel per stream**, 112-class head
over a 512-d shared FC, `num_classes=112`, `out_size=224`. Architecturally
identical to ours; `features.build_model` loads all 242/242 parameters clean.
40/242 tensors are byte-identical between the two (buffers), so they are
genuinely distinct models.

**The mixing flag is not a confound.** Our retrains set
`legacy_asymmetric_mixing=False`, the thesis runs used the pre-fix asymmetric
path. But `_mix_streams` is only reached inside `if self.stream_concat:`, and
`stream_concat` is false for every condition in play. The flag is therefore a
no-op here and the two sets are directly comparable. It would matter only for
`dkl_bcs_inter`, which is still deferred.

### 11.1 What the download does and does not remove from the training queue

Of the five conditions in array 26373625, **one overlaps and four do not**:

| condition | in download? | why it is still needed |
|---|---|---|
| `dkl_bcs` | **yes** | 3 seeds vs the thesis's 1; and it is the reproduction test |
| `dkl_base` | no | isolates the diet from the DKL front end — the control leg (a) needs |
| `rgb_tsn` | no | isolates the DKL front end from two-stream architecture |
| `dkl_bcs_swap` | no | the causal control: transform on parvo instead of magno |
| `dkl_bcs_parvo_highsf`| no | the second causal control |

So the training is not redundant. Two changes follow:

1. **`dkl_bcs_mheavy` is dropped from the deferred queue** — `s1_90` is exactly
   that condition, already trained. This is ~10 GPU-hours saved.
2. **`dkl_bcs` ×3 is kept** and finishes within the hour anyway. A single
   checkpoint cannot give seed variance, and the retrain is the only way to
   check that our pipeline reproduces the thesis's model rather than assuming it.

### 11.2 The larger point: every encoding result so far is on an untrained net

The V1→hV4 headline, the blur control, the eccentricity work — all of it is
`untrained_seed0`. That was the right starting point (it isolates architecture
and input from learning), but the paper is about a *trained* model, and we now
have the exact trained model the thesis reports. **Running the encoding
pipeline on `thesis_dvd_trans` is now the highest-value queued analysis**, and
it no longer depends on our own training finishing.

### 11.3 This also resolves §9.3 without waiting

`thesis_dvd_trans` *is* a trained blur+CS model, so the shape/texture probe on
it answers the §9.3 pre-declaration directly. Submitted as 26385545 (array
tasks 7,8). One assumption is on the record: the checkpoints carry no
LightningModule `hyper_parameters`, so **which stream received the DVD
transform is not recorded in the file**. We run it as transform-on-magno, our
default. If the probe returns a strongly *negative* M−P, mis-assigned
transform target is the first thing to check, not a finding.

---

## §12 §9.3 RESOLVED — the diet protects magno; training specialises parvo

Probe on the thesis checkpoints (26385545), against the pre-declaration written
in §9.3 before any of these numbers existed. Chance = 0.0625.

| model | M:shape | M:tex | **M bias** | P:shape | P:tex | **P bias** | **M−P** |
|---|---|---|---|---|---|---|---|
| untrained blur+CS | .977 | .181 | **+0.796** | .325 | .412 | −0.087 | +0.882 |
| untrained no-blur | .860 | .443 | +0.417 | .325 | .412 | −0.087 | +0.503 |
| **thesis `dvd_trans`** (trained, diet) | .983 | .153 | **+0.831** | .443 | .948 | **−0.506** | **+1.337** |
| **thesis `s1_90`** (trained, diet, w=.9) | .992 | .173 | +0.818 | .332 | .948 | −0.617 | **+1.435** |
| `dkl_base` ×3 (trained, **no** diet) | .71–.73 | .94–.96 | **−0.21…−0.25** | .35–.39 | .87–.88 | −0.48…−0.52 | +0.24…+0.30 |

**Outcome 1 of the three pre-declared readings** — the strong one: with the
diet, M−P does not merely survive training, it *grows* (+0.882 → +1.337), and
magno stays firmly shape-biased. Without the diet, training drives magno from
+0.417 to **−0.219**: the shape bias present at initialisation is destroyed and
the stream becomes texture-preferring like everything else.

### 12.1 The mechanism is legible, and it is not the one we assumed

The asymmetry is built from two independent halves that move for different
reasons:

- **Magno's shape bias is created by the input filter, not by learning.** It is
  already +0.796 at initialisation under blur+CS. Training with the diet leaves
  it there (+0.831 — and both are at probe ceiling, .977 vs .983, so this is
  "preserved", not "increased"). Training *without* the diet destroys it.
- **Parvo's texture bias is created by learning, not by the input.** It is
  −0.087 at initialisation — essentially neutral — and training drives it to
  −0.506. Notably it reaches −0.48…−0.52 in `dkl_base` too, i.e. **parvo
  specialises into a texture stream whether or not the diet is applied.**

So the honest claim for leg (a) is a conjunction, and it is sharper than
"the input split induces two streams":

> The chromatic/achromatic split gives magno a shape bias at initialisation and
> lets training specialise parvo toward texture. The developmental diet's role
> is specifically **protective**: it prevents training from erasing magno's
> shape bias, which is otherwise lost. The diet does not create the
> specialisation; it preserves the half that learning would destroy.

This is a better result than the one we set out to show, because the protective
role is exactly what the developmental-diet literature predicts, and the
initialisation-vs-learning split is directly measured rather than argued.

### 12.2 Dose-response

`s1_90` weights magno at 0.9 and shows a *larger* asymmetry (+1.435 vs +1.337),
driven entirely by parvo (−0.617 vs −0.506). Weak evidence — n=1 per condition,
and the two differ only in the readout weight — but it is in the predicted
direction and is free to report.

### 12.3 What must be disclosed

- **The magno probe is at ceiling** (.977–.992 shape accuracy). Differences
  among magno biases above ~+0.8 are not interpretable; all real movement in
  this table is in parvo. A harder probe would be needed to rank the magno
  conditions against each other.
- **`thesis_dvd_trans` vs `dkl_base` is not a matched comparison.** They differ
  in the diet *and* in codebase, trainer, and augmentation. Our own `dkl_bcs`
  ×3 (finishing within the hour) is the matched contrast and **§12's claim is
  provisional until it agrees.** If it does not, the codebase difference — not
  the diet — is the explanation.
- The transform-target assumption from §11.3 held: a mis-assigned target would
  have produced a negative M−P, and we observe +1.337.
- Absolute accuracies are not comparable across front ends; only the within-model
  shape-minus-texture contrast is.

### 12.4 Priority change

Encoding on `thesis_dvd_trans` is now the top queued analysis. Every encoding
result to date (V1→hV4 headline, blur control, eccentricity) is from an
*untrained* network, and we now hold the trained model the thesis reports, with
a measured, interpretable representational difference between its streams. The
leg-(a) result above and the leg-(c) encoding result should be computed on the
same checkpoint.


---

## §13 Two controls §12 did not run, and one correction to §11

§11/§12 were written from job 26385545. Job **26385558/26385589** ran the same
probe independently and reproduced §12's numbers exactly (M−P +1.337 and +1.435,
magno +0.831/+0.818, parvo −0.506/−0.617), so that table is confirmed by two
independent submissions. What follows is only what those sections do not cover.

### 13.1 `stream1_weight` is unrecorded and must be asserted, not assumed

§12 labels `s1_90` as "w=.9", but **nothing in the checkpoint carries that
value**. `stream1_weight` is a plain forward-pass scalar
(`pytorch_models.py:250`), absent from the state dict and from the (nonexistent)
LightningModule `hyper_parameters`. Passing the default 0.5 for `s1_90` loads a
valid-looking wrong model with no error — the same silent-failure shape as §9.1.

2×2 cross-check, top-1 on miniEcoset val (n=5600), each checkpoint run under
both weights:

| checkpoint | w=0.5 | w=0.9 |
|---|---|---|
| `dkl_blur_cs_dvd_trans` | **0.5686** | 0.2929 |
| `dkl_blur_cs_s1_90` | 0.5707 | **0.5834** |

Each peaks at its named weight, so the mapping holds. Quote the asymmetry
honestly: for `dvd_trans` it is decisive (−0.276 at the wrong weight), for
`s1_90` it is weak on accuracy alone (+0.013) and rests mainly on the filename
and the `model_zoo` / figure-legend mapping. Operationally: **always pass
`--stream1-weight 0.9` for `s1_90`** in anything routed through the classifier
(accuracy, behavioural shape bias). It does not affect §12's probe or the
encoding features — `extract_stream_features` runs the streams independently and
never reads it, which is why §12's numbers are safe as published.

### 13.2 The accuracy positive control does *not* reproduce the thesis

§12 says absolute accuracies are not comparable across front ends, which is
true, but the thesis reports val accuracy for *these two checkpoints*
(`plot_classification_performance.py`: 0.469 and 0.524) and that is a
within-model target we should be able to hit. We do not:

| model | ours (val, n=5600, 256 px) | thesis | Δ |
|---|---|---|---|
| DKL B&CS (`dvd_trans`) | 0.5686 | 0.469 | **+0.100** |
| M-heavy (`s1_90`) | 0.5834 | 0.524 | **+0.059** |

Chance is 1/112 = 0.009, so this is emphatically not a loading failure, and the
M-heavy > standard ordering is preserved. But a 10-point overshoot is not
"reproduced", and the direction is the awkward one — we look *better* than the
thesis, which is what a subtly easier evaluation looks like. Most likely causes,
untested: we evaluate clean (no augmentation) on the uint8 DKL h5 at 256 px,
whereas the thesis logged val accuracy *during training* at `out_size=224` on
the float64 dataset (see §13.3 — that 224 is not runnable, which is itself a
clue that the thesis's logged number came from a differently-configured pass).

Scope of the remaining doubt, after commit a7c30aa: the **representational**
reproduction is settled — our `dkl_bcs` seeds 0/2 give M−P +1.292/+1.349 against
the thesis's +1.337, agreement to 0.016 across two independent codebases. What
is *not* settled is the **absolute accuracy** of the thesis checkpoints under our
evaluation. So the honest statement is: our pipeline reproduces the thesis
model's representational geometry, and we cannot yet reproduce its reported
top-1. Nothing in §12 or in the matched contrast depends on the latter — both
are within-model contrasts — but no text should claim the thesis's *performance*
numbers were reproduced until this is closed.

### 13.3 Correction to §11: the model cannot be run at 224 px

§11 records `out_size=224` from the checkpoint's datamodule hparams. That value
cannot be honoured by this front end. `apply_scale_free_contrast` constructs
`DVDConfig(image_size=256)` (`dkl_frontend.py:161`, `IMAGE_SIZE = 256`), so it
returns a **256 px** magno channel regardless of input size; feed 224 px and
parvo stays 224 while magno comes back 256, and the two streams fail to
broadcast at the fusion (`RuntimeError: size of tensor a (8) must match tensor b
(7)`). 256 px is the resolution the transform — and therefore the trained
model — actually operates at. The `out_size=224` in the hparams describes the
thesis *datamodule*, not the front end that ran inside it, and is a likely
contributor to §13.2's gap.

### 13.4 Artefacts

- `tmp_analysis/thesis_ckpt_accuracy.py` — accuracy control (native 256 px).
- `slurm/thesis_ckpt.sh` — array: probe ×2 + accuracy ×2 (26385558).
- `slurm/thesis_ckpt_w.sh` — array: `stream1_weight` off-diagonal (26385589).
- `results/thesis_ckpt_accuracy/*.json` — all four accuracy cells.
Both arrays are CPU-only on `main`, ~2–3 min per task.

---

## §14 Leg (c) on the TRAINED thesis model — the headline survives and sharpens

First encoding run on a trained network (26385643). **n=7 of 8; subject 7 still
fitting — every number below is provisional until it lands.** Paired against
`untrained_seed0` on the same subjects.

| contrast | untrained | **thesis trained** | verdict |
|---|---|---|---|
| **hV4 − V1v** (norm) | −0.480 (0/7) | **−0.851 (0/7)** | **survives, ~2× larger** (p=.016) |
| foveal band | −0.416 | −0.757 | larger — but still blur-dependent (§8) |
| ecc slope | +0.258 | +0.124 | weakened / absent |
| dorsal−ventral interaction | +0.283 | −0.019 (p=.375) | **dead** (dead in the blur control too) |
| hMT+ absolute | +0.267 | **−0.892** | **SIGN FLIP** |
| hMT+ − VO | +0.221 | +0.070 (p=.047) | survives, weaker |

### 14.1 Training flips global stream dominance from magno to parvo

Per-subject global stream bias: **+0.19 untrained (8/8 magno-dominant) →
−0.47 trained (0/7 magno-dominant)**. This single fact explains most of the
table, and it is the encoding-side echo of §13: training makes parvo strongly
texture-selective and therefore highly predictive of image-driven BOLD almost
everywhere. hMT+'s "sign flip" is this global shift, not a change in hMT+'s
relative position — hMT+ − VO stays positive.

### 14.2 The headline is NOT a general factor — this is the decisive control

The confound that killed the thesis's RSA is at its most dangerous here, because
the hV4−V1v effect now points in the *same* direction as the global bias.
Subtracting each subject's own across-ROI mean removes any global additive
stream bias; whatever survives is regional by construction.

| | untrained | **thesis trained** |
|---|---|---|
| hV4−V1v after global centring (norm) | −0.469, p=.008, 0/8, dz=−2.56 | **−0.851, p=.016, 0/7, dz=−10.43** |
| V1v relative to subject's own mean | +0.101 (8/8 above) | **+0.451 (7/7 above)** |
| hV4 relative to subject's own mean | −0.368 (0/8 above) | **−0.400 (0/7 above)** |
| Spearman ρ(V1→V2→V3→hV4, value) | −0.698, 8/8 negative | **−0.973, 7/7 negative** |

The effect is *undiminished* by centring, V1 sits significantly above and hV4
significantly below each subject's own mean, and the gradient along the ventral
hierarchy is **near-perfectly monotonic in every subject**. A global stream
factor cannot produce a within-subject rank ordering of ρ = −0.97. Training
does not create this gradient — it *sharpens* it (−0.70 → −0.97).

### 14.3 What must be restated: gradient, not double dissociation

In the untrained model V1v is magno-preferring and hV4 parvo-preferring, so the
two ROIs sit on opposite sides of zero — a genuine crossover in **7/8** subjects.
In the trained model V1v is **not** magno-preferring in absolute terms
(−0.017, p=.81, 3/7) and the crossover holds in only **3/7**.

By the Nieuwenhuis standard a double dissociation needs both simple effects, and
in the trained model only one of them is present. So the claim has to be stated
as what it is:

> Along the ventral hierarchy, magno-unique variance declines monotonically and
> parvo-unique variance rises, in every subject, and this gradient is
> independent of each subject's global stream bias. It is a **relative
> gradient**, not an absolute double dissociation: after training, hV4 is
> strongly parvo-preferring while V1 is merely *less* parvo-preferring than the
> rest of cortex.

The absolute crossover is a property of the untrained network and should be
reported as such rather than folded into the trained-model claim.

### 14.4 Eccentricity-matched, trained model

The gradient is not eccentricity in disguise. Restricting V1v and hV4 to the
same `prf-eccrois` band (norm): −0.543 (1°), −0.755 (2°), −0.954 (4°), −1.053
(4+°), all p=.016, 0/7, dz −2.8 to −7.6; **pooled within-band −0.726, p=.016,
0/7, dz=−5.36** — larger than untrained (−0.356, dz=−2.36) in every band. The
0.5° band is null in both models (n=6, few voxels), a limitation carried from §8
that needs continuous pRF maps to resolve.

### 14.5 Consequence for the figure

Panel A is now the **trained** V1→hV4 gradient with the global-centring control
beside it, since that is the analysis that survives the confound the project has
repeatedly fallen to. The untrained model becomes the reference showing the
gradient exists before learning and sharpens with it.

---

## §15 THE UNIFYING RESULT — the diet decides whether learning builds or destroys the cortical gradient

Each contrast below holds the encoding front end constant, so the only thing
that varies within a row is training. (n=7 per cell; the pairs use slightly
different subject sets — see §15.3.)

| | untrained | trained | change | A−B p |
|---|---|---|---|---|
| **without diet** (no-blur front end) | **−0.478** | **−0.072** | **destroyed (−85%)** | .016 |
| **with diet** (blur+CS front end) | **−0.480** | **−0.851** | **amplified (+77%)** | .016 |

hV4−V1v, scale-free statistic. The two untrained models start at
**−0.478 and −0.480** — indistinguishable. The initial cortical gradient does
not depend on the input front end at all (this is the §8 blur-control result,
restated). What the diet determines is the **direction learning moves it**.

This is the same phenomenon as §13, measured in cortex instead of in the
network's representations:

| | magno shape bias (§13) | cortical V1→hV4 gradient (§15) |
|---|---|---|
| at initialisation | +0.417 / +0.796 | −0.478 / −0.480 |
| trained **without** diet | **−0.226** (destroyed, sign flip) | **−0.072** (destroyed) |
| trained **with** diet | **+0.819** (preserved) | **−0.851** (amplified) |

**Legs (a) and (c) are not two findings. They are one finding measured two
ways.** The developmental visual diet preserves magno's shape bias through
learning, and a network whose magno stream keeps its shape bias is the one whose
unique variance tracks the human ventral hierarchy. Without the diet, learning
collapses both together.

### 15.1 Why this is the paper

The claim the project set out to make — biologically constrained inputs induce
specialised streams that explain cortical variance — is true only in a much more
specific and more interesting form:

> The input split alone is not sufficient and the diet alone is not
> constructive. The input split creates a magno shape bias and an initial
> cortical gradient; ordinary training destroys both; the developmental visual
> diet's function is to protect them. Brain-likeness here is not something
> training produces — it is something training removes unless development is
> constrained to prevent it.

That last sentence is the contribution. It inverts the usual goal-driven framing
(train harder on more data to become more brain-like) and it is supported by a
2×2 in which both untrained cells are identical and the two trained cells move
in opposite directions.

### 15.2 What is already controlled

- **General factor** (§14.2): the gradient survives removing each subject's
  global stream bias undiminished (dz −10.4), with ρ(hierarchy) = −0.973 in 7/7
  subjects. This is the confound that killed the thesis's RSA.
- **Eccentricity** (§14.4): holds within every pRF band except the 0.5° one
  (which is null in all models), pooled dz −5.36.
- **Front end** (§15, top row): both untrained models agree to 0.002, so the
  low-pass filter does not set the initial gradient.
- **Codebase** (§13): our retrain reproduces the thesis checkpoint to 0.016 on
  the behavioural probe.

### 15.3 What is NOT yet controlled — must be closed before submission

1. **One seed per trained cell, and one model in the diet cell.** The "destroy"
   row rests on `dkl_base_seed0` alone and the "amplify" row on the thesis
   checkpoint alone. Queued: encoding for `dkl_base_seed1` and for our own
   `dkl_bcs_seed0`, which also makes the amplify cell a cross-codebase
   replication. **Until those land, §15 is a single-seed result and must be
   described that way.**
2. **The two rows use different subject sets** (one is missing subject 1, the
   other subject 7, both currently re-fitting). The within-row pairing is exact,
   which is what the p-values rest on, but the between-row comparison is not yet
   on identical subjects.
3. `dkl_bcs` differs from `dkl_base` in *both* blur and contrast-sensitivity.
   `dkl_b` and `dkl_cs` (deferred) would separate them. Not required for the
   claim as stated, which is about the diet as a package.
4. The absolute crossover is an untrained-model property only (§14.3). The
   trained claim is a relative gradient.


---

## §16 Final n=8 numbers, and one more claim withdrawn

Subject 7 landed; `thesis_dvd_trans` is complete at n=8. Everything in §14 is
confirmed and slightly stronger. **The n=7 figures in §14 are superseded by
these.**

| contrast (norm) | untrained n=8 | thesis trained n=8 | verdict |
|---|---|---|---|
| **hV4 − V1v** | −0.4691 (p=.008, 0/8) | **−0.8593 (p=.008, 0/8)** | **survives, 1.8× larger** (A−B p=.008) |
| after removing global bias | −0.4691, dz −2.56 | **−0.8593, p=.008, 0/8, dz −10.86** | general factor excluded |
| ρ(V1→V2→V3→hV4) | −0.698, 8/8 neg | **−0.973, 8/8 neg** | monotonic in every subject |
| eccentricity-matched, pooled | −0.356, dz −2.36 | **−0.7363, p=.008, 0/8, dz −5.72** | not eccentricity |
| V1v vs own mean | +0.101 (8/8 above) | **+0.454 (8/8 above)** | |
| hV4 vs own mean | −0.368 (0/8 above) | **−0.406 (0/8 above)** | |

Significant within-band in 4 of 5 pRF bands (1°, 2°, 4°, 4+°; dz −3.0 to −7.9).
The 0.5° band remains null in every model (n=7, few voxels) — the limitation
from §8 that needs continuous pRF maps.

### 16.1 hMT+ is dead — withdraw it

At n=7 hMT+ − VO looked like it survived (p=.047 norm). **At n=8 it does not:
p=.195 on the scale-free statistic and p=.039 on the raw one.** The two
statistics disagree, and the project's own rule (§8) is to claim a result only
where they agree. It is also already dead in the blur control.

So the dorsal/MT leg is withdrawn entirely, in all models. This matters because
§7 promoted it from "limitation" to "result" on the strength of an n=8 untrained
fit (+0.0197, p=.023) — that number does not replicate on the final subject set
(raw +0.0128, p=.109). **The honest record is that hMT+ specificity was a
false positive from an incomplete subject set, and it is now retracted.**

Nothing else in §14/§15 depends on it. What survives is the ventral gradient,
which is the stronger and better-controlled claim anyway.

### 16.2 Crossover, final

4/8 subjects in the trained model (was 3/7). The §14.3 restatement stands
unchanged: **relative gradient, not double dissociation.** The absolute
crossover is a property of the untrained network (7/8) and is reported as such.

### 16.3 The gradient is not "the diet model predicts cortex better" — it predicts worse

Median R²_joint / noise ceiling, over reliable voxels, averaged across subjects:

| model | R²/NC | hV4−V1v (norm) |
|---|---|---|
| untrained, blur+CS | 0.119 | −0.469 |
| untrained, no blur | 0.144 | −0.478 |
| **trained, no diet** (`dkl_base`) | **0.461** | **−0.072** |
| **trained, with diet** (`thesis`) | **0.423** | **−0.859** |

Two things follow, and the second is the important one.

First, **training improves cortical prediction ~3.5×** (0.12–0.14 → 0.42–0.46).
The trained models are good encoding models, explaining >40% of the noise
ceiling, so nothing in §14–§16 rests on a degenerate fit.

Second, and this closes the obvious objection to §15: **the diet model is the
*worse* aggregate encoder** — 0.423 against 0.461, ~8% lower — while carrying
essentially all of the ventral gradient (−0.859 against −0.072). Aggregate
variance explained and regional structure therefore dissociate cleanly, and in
opposite directions.

So the §15 claim cannot be restated as "the developmental diet makes a better
model of visual cortex". It does not. It makes a model whose *stream-level
division of labour* matches the human ventral hierarchy, at a small cost in
total variance explained. That is a sharper and more falsifiable claim than a
goodness-of-fit story, and it is the one the data support.


---

## §17 The 2×2 is complete — n=8 everywhere, two models per trained cell

All three queued encoding arrays landed. Every cell below is n=8 on the **same
eight subjects**, and each trained cell now has two independent models.
hV4−V1v, scale-free statistic; `dz` and `ρ` are from the global-centring
control (§14.2).

| | model | hV4−V1v | centred dz | ρ(hierarchy) |
|---|---|---|---|---|
| **untrained**, no-blur front end | — | **−0.4877** | −2.56 | −0.698 |
| **untrained**, blur+CS front end | — | **−0.4691** | −2.56 | −0.698 |
| **trained, NO diet** | `dkl_base` s0 | −0.1669 | **−0.61** | −0.512 |
| | `dkl_base` s1 | −0.1318 | — | — |
| **trained, WITH diet** | thesis `dvd_trans` | **−0.8593** | **−10.86** | −0.973 |
| | **ours** `dkl_bcs` s0 | **−0.8618** | **−11.70** | −0.964 |

All within-row model-vs-untrained contrasts p ≤ .039, 0/8 or 1/8 subjects.

**The diet cell replicates across codebases to 0.0025.** Our standalone-trained
`dkl_bcs_seed0` and the thesis Lightning checkpoint give −0.8618 and −0.8593,
with matching control statistics (dz −11.70 / −10.86, ρ −0.964 / −0.973). Two
trainers, two augmentation pipelines, two seeds. This is the strongest
reproducibility evidence in the project.

### 17.1 Correction to §15: "destroyed" was too strong

§15 reported the no-diet trained cell as **−0.072, "destroyed (−85%)"**. That
was n=7 — computed while subject 1 was still re-fitting after the disk-quota
failure. **With subject 1 included the value is −0.1669**, and the second seed
gives −0.1318. The corrected figure is a mean of **−0.149, a ~69% reduction**
from −0.4877, not 85%, and the effect remains statistically detectable
(p=.016, though only 1/8 subjects).

> **SUPERSEDED 2026-09-01 (§29).** That subject-1 file was the contaminated
> one (§22) — encoded through the diet front end despite being a no-diet
> model. After re-encoding, seed0 gives **−0.1013**, not −0.1669, and the
> two-seed mean is **−0.117**, a ~76% reduction. Quote −0.117, not −0.149.
> The direction of the correction is favourable: the no-diet cell moves
> *further* from the diet cell.

The claim must therefore be stated as **attenuation, not abolition**:

> Training without the developmental diet reduces the ventral gradient by
> roughly two thirds and collapses its effect size by a factor of ~19
> (dz −11.70 → −0.61), leaving a weak residual gradient that is monotonic in
> far fewer subjects (ρ −0.964 → −0.512). Training with the diet nearly doubles
> it.

The effect-size and monotonicity collapse is the more honest headline than the
point estimate, because it is the part that does not move between n=7 and n=8:
dz −0.61 versus −11.70 is not a marginal difference.

### 17.2 A detail worth reporting: the two trained models are globally opposite

Per-subject global stream bias (norm): **no-diet trained is +0.727, magno-
dominant in 7/8 subjects; diet-trained is −0.454, parvo-dominant in 0/8.** The
low-pass front end is what hands parvo the global advantage, exactly as §8's
blur control predicted.

This makes the general-centring control more valuable, not less: the two trained
models sit on *opposite sides* of the global-bias axis, yet the diet model shows
the strong ventral gradient and the no-diet model does not. A general stream
factor cannot explain a regional effect that is largest in the model whose
global bias points the other way.

### 17.3 §15 and §16 stand otherwise

The unifying claim is unchanged and now rests on two models per trained cell:
the diet determines whether learning strengthens or attenuates the correspondence
between the network's stream-level division of labour and the human ventral
hierarchy. Legs (a) and (c) remain the same finding measured two ways.

---

## §18 LEG (b) MEASURED — the third leg agrees with the other two

Leg (b) — "their integration yields human-like shape-biased behaviour" — had
never been run on any model. `cue_conflict.py` was written before the
checkpoints existed and then never invoked, so the claim in the working title
had numbers for legs (a) and (c) and none for (b). Array 26391416, twelve
converged models, 1200 usable Geirhos cue-conflict images, B_shape =
N_shape/(N_shape+N_texture), CI from 2000-item bootstrap.

| model | **B_shape** [CI95] | magno_only | parvo_only | 16-way acc |
|---|---|---|---|---|
| untrained, no blur | 0.500 [.416,.583] | 0.568 | 0.503 | 0.125 |
| untrained, blur+CS | 0.503 [.418,.587] | 0.513 | 0.503 | 0.124 |
| `dkl_base` s0/s1/s2 (no diet) | .421 / .433 / .394 → **0.416 ± .020** | ~0.42 | ~0.36 | 0.47–0.51 |
| `dkl_bcs` s0/s1/s2 (diet) | .692 / .686 / .646 → **0.675 ± .025** | **0.87–0.89** | ~0.42 | 0.37–0.45 |
| `dkl_bcs_swap` s0/s1 | .412 / .409 → **0.411 ± .002** | 0.40 | 0.48 | 0.47–0.49 |
| thesis `dvd_trans` | **0.697** [.661,.732] | **0.911** | 0.407 | 0.517 |
| thesis `s1_90` | **0.712** [.678,.748] | 0.904 | 0.299 | 0.538 |

### 18.1 The same 2×2, a third time

| | untrained | trained, no diet | trained, with diet |
|---|---|---|---|
| leg (a) magno shape bias (§13) | +0.796 | **−0.226** | **+0.819** |
| leg (b) behavioural B_shape (§18) | **0.50** | **0.416** | **0.675** |
| leg (c) cortical hV4−V1v (§17) | −0.488 | **−0.117** | **−0.861** |

Behaviour starts at the neutral point — untrained models are at B_shape = 0.50,
i.e. no preference at all — and training moves it in *opposite directions*
depending on the diet. Three independent measurements (representations,
behaviour, cortex), three different data types, one pattern. Leg (b) was the
one that could have broken the story and it does not.

### 18.2 The thesis's behavioural number reproduces

`thesis_dvd_trans` gives **0.697** against the thesis's reported **0.677**
(`plot_behavioral_shape_bias.py`), a difference of 0.020 that sits inside our
bootstrap CI. This is worth contrasting with §13.2, where the same checkpoint's
*accuracy* overshot the thesis by 10 points and remains unexplained: the
behavioural pipeline reproduces the thesis, the accuracy pipeline does not.
That narrows §13.2 — whatever the discrepancy is, it is in the classification
evaluation path, not in the checkpoint or the front end.

Less good: `thesis_s1_90` gives 0.712 against a reported 0.756, and 0.756 falls
just outside our CI [.678,.748]. The M-heavy > standard ordering is preserved
(0.712 > 0.697, as 0.756 > 0.677) but the gap is smaller than the thesis's.
Report the ordering, not the magnitude.

### 18.3 The swap control is now decisive, and §10.1 is discharged for behaviour

`dkl_bcs_swap` puts blur+CSF on the chromatic channel instead of the luminance
one. Behaviour collapses to **0.411**, statistically indistinguishable from the
no-diet `dkl_base` (0.416) and nowhere near the diet models (0.675). Both swap
seeds are converged 150/150, so this is *not* the §10.1 problem — the earlier
swap numbers came from 20%-trained checkpoints and were quarantined in
`results/shape_texture/partial_invalid/`. For leg (b) that caveat is now
discharged on converged models.

The control therefore rules out the obvious alternative: it is not "whichever
stream gets the developmental transform becomes the shape stream and drives
behaviour". Applying the identical transform to parvo does **not** produce a
shape-biased network. The effect requires the transform on the *luminance*
channel specifically.

### 18.4 The per-stream decomposition answers §2.2

The thesis's open question was why behaviour follows the magno stream when
parvo is the better classifier. Zeroing one stream before the weighted sum
isolates the other exactly (the classifier is linear on that sum):

- In the diet models, **magno alone is far more shape-biased than the full
  network** (0.87–0.91 vs 0.65–0.70) while parvo alone is texture-biased
  (~0.41). The full network's shape bias is magno's, diluted by parvo.
- In the swap models the asymmetry weakly *reverses* (magno 0.40, parvo 0.48) —
  the transformed channel is the more shape-biased one — but neither stream is
  strongly shape-biased and the network is not.

So the answer is that behaviour follows magno because magno is the only stream
that ever becomes shape-biased, and only the luminance channel can be made so.

### 18.5 What must be disclosed

- **The untrained B_shape = 0.50 rests on very few decisions.** Untrained
  16-way accuracy is 0.124, so only ~150 of 1200 images produce a shape-or-
  texture answer (n_shape=75, n_texture=74) against ~1050 "other". The CI is
  correspondingly wide [.416,.587]. Read it as "no preference", not as a
  precise 0.50.
- **The diet costs accuracy here too**: 0.37–0.45 against 0.47–0.51 for
  `dkl_base`, the same ~25%-relative penalty as §10.2. The shape-bias gain is
  not free and the tradeoff curve must be reported (§2, "required regardless").
- **`rgb_tsn` is not covered.** It is a 3-channel RGB two-stream model and this
  entry point builds the 1-channel DKL front end only. The architecture-only
  control for leg (b) therefore does not yet exist.
- `dkl_bcs_swap_seed2` (96/150) and all three `parvo_highsf` seeds were still
  training and are excluded by the convergence guard, which was added to
  `cue_conflict.py` for this run — it had been added to
  `stream_shape_texture.py` after §10.1 but never to this entry point.

---

## §19 THE INPUT-LEVEL NULL MODEL — the trained result survives, the untrained panels do not

§7.2 item 1, "the control that decides how much the network matters", run at
last. A bank of oriented energy filters (4 scales × 8 orientations, quadrature
energy, 2048 dims/stream) is applied to **each stream's own transformed input**
and pushed through the identical PCA → banded-ridge → variance-partition path
(`--features gabor`, jobs 26391744/26391745, n=8 both front ends). If the
gradient is carried by the input, the ResNet is doing nothing.

### 19.1 The result, hV4−V1v (norm), n=8

| front end | **Gabor input-only** | untrained net | trained net |
|---|---|---|---|
| no-blur (no diet) | **−0.6209** | −0.4877 | −0.1669 (`dkl_base` s0) |
| blur+CS (diet) | **−0.3666** | −0.4691 | −0.8593 / −0.8618 |

Fit quality, median R²/NC over reliable voxels:

| | Gabor input | untrained net | trained net |
|---|---|---|---|
| blur+CS | **0.1408** | **0.1384** | 0.386–0.403 |
| no-blur | **0.1627** | **0.1610** | 0.424 |

### 19.2 What survives: the trained diet result, on both statistics

`gabor_blur_cs` vs the diet-trained models is the one contrast where the raw
and scale-free statistics **agree**, which is this project's bar (§8):

| contrast | raw | norm | verdict |
|---|---|---|---|
| gabor vs thesis `dvd_trans` | −0.0419 vs −0.1746, **p=.008** | −0.3666 vs −0.8593, **p=.008** | **larger in trained** |
| gabor vs our `dkl_bcs` s0 | −0.0419 vs −0.1759, **p=.008** | −0.3666 vs −0.8618, **p=.008** | **larger in trained** |

**The headline of §15/§17 survives the control it was most exposed to.** The
diet-trained network's ventral gradient is roughly 2.3× the input model's on
the scale-free statistic and 4.2× on the raw one, in both codebases. It is not
a front-end artefact.

### 19.3 What does not survive: the untrained network adds nothing to fit

The untrained ResNet predicts BOLD **no better than oriented energy on its own
input** — 0.1384 vs 0.1408 (blur+CS) and 0.1610 vs 0.1627 (no-blur), the input
model marginally *ahead* in both. Whatever the random network computes, it buys
zero additional explained variance over a Gabor bank.

This forces a correction to §14.5, which proposed keeping the untrained model as
"the reference showing the gradient exists before learning and sharpens with
it". That framing is still available but the wording must change: the
pre-learning gradient is a property of the **front end**, not of the untrained
network. Any panel that shows the untrained model as evidence about
architecture is a front-end validation and must be labelled as one.

### 19.4 Two contrasts where the statistics disagree — both unresolved

Per §8 neither may be claimed.

1. **Untrained net vs input null.** raw says the network roughly *doubles* the
   absolute gradient (−0.1179 vs −0.0419, p=.008; no-blur −0.2081 vs −0.1009,
   p=.008); norm says the two are indistinguishable (p=.109 and p=.055). The
   cause is visible in the denominator: total stream-unique variance is
   0.0294 for the untrained net against 0.0208 for the input model, so the
   network moves variance from shared into unique **without predicting better**
   (§19.3). Absolute and relative statements therefore diverge, and both are
   true of different quantities. The honest summary is that the untrained
   network sharpens the *magnitude* of stream separation while leaving the
   *relative* stream preference profile where the input already had it.
2. **No-diet trained vs input null.** norm says training drops the model below
   its own input (−0.1669 vs −0.6209, p=.016); raw says no difference
   (p=.055). Tempting to read as "learning without the diet actively destroys
   structure the input handed it for free", and the monotonicity is consistent
   (input ρ=−0.819, 8/8 negative; no-diet trained ρ=−0.512), but one statistic
   is not enough. **Do not put this in the paper until it is resolved.**

### 19.5 The eccentricity gradient is a front-end property, confirmed

Gabor +0.2221 vs untrained +0.2518, norm **p=.844** — indistinguishable. (Raw
disagrees, p=.008, for the §19.4 denominator reason.) §8 already demoted
eccentricity to a front-end validation on the blur control; the input model
confirms it directly. It is also *absent* in the trained diet model (+0.1009,
p=.055), so it is not part of the trained claim at all.

### 19.6 Monotonicity, and a subtlety worth keeping

Spearman ρ along V1→V2→V3→hV4, from the global-centring control:

| model | centred hV4−V1v | dz | ρ |
|---|---|---|---|
| Gabor, blur+CS | −0.3666 | −2.93 | −0.326 (7/8) |
| Gabor, no-blur | −0.6209 | −4.04 | −0.819 (8/8) |
| untrained net | −0.4691 | −2.56 | −0.698 |
| diet-trained | −0.8593 | −10.86 | −0.973 (8/8) |

The diet's front end *lowers* the input-level gradient (−0.62 → −0.37) and
weakens its monotonicity (ρ −0.82 → −0.33): blurring luminance removes the
fine-scale content that gives V1 its magno advantage. Learning then builds it
back to −0.86 with ρ=−0.97. So the diet does not hand the network a better
starting point — it hands it a *worse* one, and the diet-trained network
overtakes anyway. That is a stronger version of §15's claim, and unlike §19.4
it rests on the contrast that passes both statistics.

### 19.7 Consequences

1. §15/§17's central claim stands and is now defended against the control the
   plan predicted a reviewer would demand.
2. Every untrained-model panel is re-labelled a front-end validation (§19.3).
3. The eccentricity panel is settled as front-end (§19.5), consistent with §8.
4. Two contrasts are unresolved pending a statistic that both agree on (§19.4).
5. Still to do: the input-null column for `dkl_base` seed1 and `dkl_bcs` seed2+,
   and an input-null run against the `swap` front end once its seeds converge.

---

## §20 Three additions to §18, and a coordination note

§18 and this section were written by two sessions independently from the same
JSONs (see §20.4). §18 stands as the primary write-up; the following three
points are not in it and change what leg (b) may claim.

### 20.1 Quantify the swap: it is a CONJUNCTION, not a transform effect

§18.3 concludes the swap "does not produce a shape-biased network", which is
true but leaves the effect unquantified. `transform_target` moves *only* the
transform between streams — stream 1 always receives luminance, stream 2 always
red-green (`dkl_frontend.magno_parvo_inputs`) — so the per-stream deltas
isolate the carrier:

| transform applied to | that stream, no diet | that stream, with transform | Δ |
|---|---|---|---|
| **luminance** (magno) | 0.421 | 0.878 | **+0.457** |
| **red-green** (parvo) | 0.418 | 0.484 | **+0.066** |

The identical transform buys **7x less** on the chromatic carrier — 14% of the
effect. So the *direction* does follow the transform (in swap the transformed
stream is the more shape-biased of the two, 0.484 vs 0.399, consistent with
§18.4's "weakly reverses"), but the *magnitude* requires luminance.

State the claim as a conjunction: **the shape bias requires both the
developmental low-pass transform and an achromatic carrier; neither alone
suffices.** This is the biologically motivated version — magnocellular cells
are both luminance-driven and low-pass — and it is what answers a reviewer
asking "isn't this just blur?": the same blur on the chromatic channel buys
14%.

### 20.2 "Integration yields shape-biased behaviour" is unsupported — the
readout AVERAGES, it does not amplify

§18.4 notes the full network's bias is "magno's, diluted by parvo". That
dilution is fatal to the working claim's wording for leg (b) ("*their
integration* yields human-like shape-biased behaviour") and should be recorded
as a refutation, not an aside. In **every** model the integrated readout is at
or below the better stream:

    full − max(magno, parvo):  diet −0.203, thesis −0.214, s1_90 −0.191,
                               no-diet −0.005, swap −0.073

For the diet cell full 0.675 [.653,.729] vs magno 0.878 [.839,.901] — CIs do
not overlap. There is no integration synergy anywhere in the model set, and
`s1_90` confirms the mechanism rather than contradicting it: up-weighting magno
to 0.9 yields *both* the highest integrated shape bias (0.712) and the highest
accuracy (0.538), exactly as a weighted average of two streams predicts.

**Surviving claim**: the diet gives the network one shape-biased stream, and
the integrated readout inherits a fraction of it set by the mixing weight.
Integration is where shape bias is *lost*, not created. Do not write
"integration yields".

### 20.3 "Human-like" is not established, and the human data is not here

Best integrated shape bias in the model set is 0.712; human shape bias on these
stimuli is far higher. Leg (b) supports a *directional* statement (the diet
moves the network toward the human end; the no-diet model sits in the
texture-biased CNN regime) and **not** a matching one. Geirhos's human observer
data is **not in this repo** — no human number goes into the manuscript until
it is downloaded and cited. Flagged, not assumed, and not estimated from
memory.

Consequence for figures, reinforcing §18.5: because the untrained models decide
only ~150/1200 images and return a coin-flip 0.50, **leg (b) cannot be drawn as
an untrained -> trained slopegraph** the way Figure 1A/1C are. §18.1's "training
moves it in opposite directions" overstates what the untrained cell can carry —
by §18.5's own disclosure that cell means "no preference", so the honest
rendering is a trained-only contrast. Leg (a)'s probe works untrained precisely
because it does not require the network to classify; leg (b) does.

### 20.4 Coordination note (process, worth fixing)

Two sessions submitted the *same* 12-task cue-conflict array over the identical
config 20 min apart — 26391416 (23:02) and 26391506 (23:24) — and both wrote a
§18. The files in `results/cue_conflict/` are from 26391506, which overwrote the
earlier set; the numbers are identical, so nothing is compromised. Separately,
this session briefly clobbered §19 and the original §18 by copying the
out-of-repo `/home/byrong/projects/TSN/PAPER_PLAN.md` (which lagged the repo
copy) over `docs/PAPER_PLAN.md`; restored from 927f49a.

**Fix**: `docs/PAPER_PLAN.md` in the repo is now the authoritative copy. The
out-of-repo file is stale and should not be copied forward again — edit the
repo copy directly.

### 20.5 §19's consequence for Figure 1 (open)

§19 finds a Gabor bank predicts BOLD as well as the untrained ResNet, so the
untrained arm is a **front-end validation, not evidence about architecture**.
Figure 1A/1B currently lead on the untrained -> trained divergence and the
caption implies the shared start is a property of the networks. It is not; it is
a property of the input filters. Either relabel the untrained arm in the caption
or move it to a supplementary panel. Additionally §19's finding that the diet's
front end *lowers* the input-level gradient (−0.62 -> −0.37) while the
diet-trained network overtakes anyway is a strong addition to §15 — it rules out
"the diet just supplies a better input representation". Not yet acted on.

---

## §21 The swap probe on converged checkpoints — quarantine vindicated, §20.1 refined

Job 26399145 (`shape_texture` indices 5–6) re-ran the leg (a) linear probe on
both `dkl_bcs_swap` seeds at 150/150, replacing the epoch-33 files in
`results/shape_texture/partial_invalid/`.

| cell | magno | parvo | magno − parvo | n |
|---|---|---|---|---|
| untrained, no diet | +0.417 | −0.087 | +0.503 | 1 |
| untrained, with diet | +0.796 | −0.087 | +0.883 | 1 |
| trained, no diet | −0.226 | −0.504 | +0.278 | 3 |
| trained, with diet | **+0.819** | −0.518 | **+1.338** | 3 |
| **trained, SWAP (converged)** | **−0.236** | **−0.155** | **−0.081** | 2 |
| trained, SWAP (quarantined, ep33) | −0.175 | −0.225 | +0.050 | 2 |
| thesis `dvd_trans` | +0.831 | −0.506 | +1.337 | 1 |
| thesis `s1_90` | +0.818 | −0.617 | +1.435 | 1 |

### 21.1 The quarantine was correct — the sign was not stable

At epoch 33 the swap asymmetry was **+0.050**; at convergence it is **−0.081**.
The sign flips. Both values are near zero, so no strong claim was ever
available from either, but the withdrawn §10.1 statement ("the channel swap
nearly abolishes the asymmetry") was resting on a number that does not survive
the remaining 78% of training. Quarantining rather than patching was the right
call, and `partial_invalid/` should be kept as the record.

The converged swap also does not merely *abolish* the asymmetry — it slightly
**reverses** it (−0.081), which agrees in sign with the cue-conflict swap result
(§18: magno 0.399 vs parvo 0.484). Two independent measures, same weak
reversal. Report it as "abolished, with a weak reversal that both measures
agree on", not as "abolished".

Also clean: swap magno (−0.236) is statistically indistinguishable from no-diet
magno (−0.226). Removing the diet from the luminance channel returns that
channel exactly to its no-diet value — no residual effect of the diet being
present elsewhere in the network.

### 21.2 §20.1's "14%" is a behavioural number, not a universal one — corrected

§20.1 quantified the conjunction from cue-conflict behaviour: the transform
buys +0.457 on luminance and +0.066 on red-green, a **7x** ratio, "14% of the
effect". The probe gives the same direction but a different ratio:

| transform applied to | that stream, no diet | with transform | Δ |
|---|---|---|---|
| **luminance** | −0.226 | +0.819 | **+1.046** |
| **red-green** | −0.504 | −0.155 | **+0.349** |

Ratio **3.0x** on representations vs **7x** on behaviour. So "14%" must not be
quoted as the conjunction's magnitude in general — on the probe the chromatic
carrier captures **33%** of the benefit. Corrected in the text; §20.1's ratio
stands only for the behavioural readout.

### 21.3 The sharper form of the conjunction claim

The probe makes the conjunction crisper than a ratio does. The transform helps
*whichever* carrier receives it, substantially so — red-green moves from −0.504
to −0.155, i.e. from clearly texture-biased to nearly neutral. But it **never
crosses zero**. Only luminance is pushed into positive, shape-biased territory
(+0.819).

**State it as**: the developmental transform reduces texture bias on either
carrier, but only the achromatic channel can be driven into genuine shape
bias. This is the quantitative version of §18.4's sentence ("magno is the only
stream that ever becomes shape-biased, and only the luminance channel can be
made so") — that sentence is now supported by both the behavioural and the
representational measure, with the probe supplying the zero-crossing that makes
it a qualitative distinction rather than a difference of degree.

### 21.4 Consequence for Figure 1C

Figure 1C currently plots magno/parvo arms for diet vs no-diet only. The swap
arm is now available on converged checkpoints for both measures and is the
strongest causal control in the project. Adding it makes 1C a three-arm panel
(diet / no-diet / swap) and lets the zero-crossing in §21.3 be read directly
off the axis, since the panel already has a zero line. Recommended, together
with the §20.5 relabelling of the untrained arm. Not yet acted on.

---

## §31 Claiming §7.3b (in-silico cue-conflict → cortex), 2026-08-31

**Coordination (per §20.4): this session is taking §7.3b and nothing else.**
§20.5 (Figure 1 relabel) and the human-data download are explicitly left for
the other session. Do not start §7.3b in parallel — see the collision log in
§20.4.

### 21.1 Why this and not the polish queue

The remaining queue is seeds, a single-stream baseline, and extra input-null
columns. All of it is reviewer-proofing for claims that already stand. §7.3b
is the only open item that changes what the paper *is*: leg (b) currently
lives entirely in behaviour (miniEcoset logits) and leg (c) entirely in
cortex (NSD voxels), and no analysis connects them. Without it the manuscript
reads as two results about the same network; with it, one.

### 21.2 What has to be built (the estimate in §7.3 was low)

`results/encoding/*.npz` stores per-voxel summary statistics only —
`fit_banded` returns test R2 and discards the fitted estimators. So the
cue-conflict images cannot be pushed through the *existing* fits; §7.3b needs

1. `fit_banded` to return (or re-apply) the fitted magno-only, parvo-only and
   joint estimators;
2. the same PCA/scaling transform used on the NSD features applied to the
   cue-conflict features — the band scaling in `prepare_band` is fit on the
   NSD training set and must be carried over, not refit on 1280 images;
3. a refit array over the models we want the panel for.

This is a code change plus one array, not a pure analysis pass.

### 21.3 Per-voxel shape-bias index — declared before fitting

For each of the 1280 cue-conflict images the filename gives a shape label S
and a texture label T (16 levels each, crossed). For every voxel, from its
*predicted* responses to those images:

    eta2_shape   = between-shape-group variance / total variance
    eta2_texture = between-texture-group variance / total variance
    VSB          = (eta2_shape - eta2_texture) / (eta2_shape + eta2_texture)

VSB is the voxel-level analogue of the behavioural B_shape, bounded [-1, +1],
and needs no original (non-conflict) images.

**The panel**: VSB against SPI across reliable voxels, within subject, then
the correlation aggregated over the 8 subjects with the same sign-flip test
used everywhere else.

**Both statistics rule (§8) applies**: report the raw and the noise-ceiling-
normalised version of the voxel correlation and claim only where they agree.

### 21.4 Pre-declaring the outcomes (§7.4's rule, applied to this panel)

This is written before the fit because the sign is genuinely uncertain and
deciding afterwards what it means would be storytelling.

- **VSB correlates negatively with SPI** (magno-preferring voxels are *less*
  shape-selective). This is the outcome the existing maps predict: magno-
  unique variance peaks in V1/V2 (§7.1) and shape selectivity should peak in
  LO/hV4. It would mean the stream that drives shape-biased *behaviour*
  (§18/§20.1, magno 0.878) is not the stream that predicts shape-selective
  *cortex* — a dissociation, and a genuinely interesting one, but it refutes
  the simple "same stream, both levels" reading of the working claim.
- **VSB correlates positively with SPI.** The strong unifying result: the
  magno stream's shape preference is visible in the cortical voxels it
  uniquely explains. This is the flagship as originally imagined.
- **No correlation.** The two legs are independent measurements of the same
  network and the paper stays two results side by side. Report it; do not
  re-cut the voxel set looking for a subset that works.

A fourth possibility must also be pre-empted: VSB may be dominated by overall
predictability rather than by stream preference, since a voxel the model
predicts poorly has low eta2 on *both* factors. VSB is a ratio so it is
scale-free by construction, but it will be unstable where the denominator is
small. **Restrict to the existing reliable mask and additionally report the
correlation with voxels binned by r2_joint_nc**, so a predictability
confound is visible rather than hidden.

### 21.5 Model set

Minimum for the panel: `dkl_bcs_seed0` and `thesis_dvd_trans` (the two diet
cells) against `dkl_base_seed0` (no diet). If the diet cells show the effect
and the no-diet cell does not, the panel carries §15's claim into cortex as
well, which is the strongest available version.


---

## §22 A CONTAMINATED SUBJECT IN THE NO-DIET CELL (found 2026-08-31)

Found while recovering the original submission commands with `sacct
--format=SubmitLine` in order to reuse them for §7.3b. Not caused by §7.3b;
it has been in the results since 2026-08-30 06:06.

### 22.1 What happened

`dkl_base_seed0_trained` subject 1 was killed by the /home disk-quota failure
(§ the Errno 122 incident, job `26383806_0`, 2026-08-29 19:03). It was
re-submitted the next morning as

    26386734_0  sbatch --array=0 slurm/encode_subjects.sh \
                  dkl_base_seed0_trained .../dkl_base_seed0/checkpoint_last.pt

— **without the `--no-lum-blur --no-lum-cs` flags** that the original
`26383806` array carried on every task. `dkl_base` is the *no-diet* model, so
those flags are what make its front end match its training diet. Without them
the run applied the blur + contrast-sensitivity front end to a network trained
on unblurred input.

The feature cache did not save it. `26386734_0`'s log shows it *extracting*
features (`features 512/10000`), not hitting a cache, and the cache file's
mtime (02:38) falls inside that job's window — the quota failure had left no
usable cache to hit.

### 22.2 Verified directly, not inferred

Recomputing the first six images' features both ways and comparing against the
cached matrix (`$CLAUDE_JOB_DIR/tmp/check_frontend.py`):

| front end | max abs delta vs cache, magno | parvo |
|---|---|---|
| no-blur / no-CS (intended) | 16.1679 | 0 |
| **blur + CS (what the re-run defaulted to)** | **0** | 0 |

An exact zero on the magno band. The parvo band is identical either way, as
expected — the blur and CS cut touch luminance only — so the damage is
confined to the magno stream, which is the band carrying the headline.

### 22.3 Scope: exactly one file

Every other run of a no-blur model carried the flags on all eight tasks
(`26383806_1..7`, `26386799_0..7` = `dkl_base_seed1`, `26373186_0..7` =
`untrained_seed0_noblur`, `26391745_0..7` = `gabor_noblur`). `26386734_0` is
the only offender. Models whose front end *is* blur+CS (`dkl_bcs_*`,
`thesis_*`) cannot be affected by a missing flag, since the flag's absence is
their correct setting.

Damage: `results/encoding/dkl_base_seed0_trained_subj01.npz`, and every n=8
statistic computed from `dkl_base_seed0_trained`.

### 22.4 How much it moved the headline

Per-subject hV4−V1v for the no-diet cell:

| | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
|---|---|---|---|---|---|---|---|---|
| seed0 norm | **−0.830** | −0.102 | −0.038 | −0.100 | −0.057 | −0.138 | +0.019 | −0.088 |
| seed1 norm (clean) | −0.223 | −0.100 | −0.125 | −0.213 | −0.132 | −0.002 | −0.098 | −0.161 |

Subject 1 of seed0 is a **six-fold outlier** against its own seven siblings and
against the same subject in the clean seed1 run. Effect of dropping it:

| statistic | n=8 (as reported) | n=7, subj01 dropped |
|---|---|---|
| raw | −0.0452 (p=.039) | −0.0281 (p=.078) |
| norm | **−0.1669** (p=.016) | **−0.0721** (p=.031) |

**The contaminated subject carries 57% of the reported no-diet effect size**,
and on the raw statistic it is the difference between significant and not. The
−0.1669 in §15/§17 is not a number the paper can keep.

### 22.5 What this does to the claims — it strengthens §15

Corrected, the no-diet cell moves from −0.167 to roughly −0.07/−0.13 (n=7
seed0; clean seed1 gives −0.1318 at n=8). The diet-trained cell is −0.859.
The contamination had made the no-diet model look **more** like the diet model,
so removing it **widens** the gap that §15 is built on. The direction of every
§15/§17 claim survives and the dissociation gets cleaner, not weaker.

Two further reassurances:
- `dkl_base_seed1_trained` is a fully clean n=8 no-diet run and reproduces
  seed0 on every row of `model_verdict.py` ("same in both" throughout). The
  no-diet cell was never resting on the broken file alone.
- §19's input-null comparisons used the *diet* and *gabor* models for their
  headline; the affected contrast is the already-UNRESOLVED
  no-diet-trained-vs-input one (§19.4), which must be recomputed anyway.

### 22.6 Root cause and the fix (committed, `ad64e58`)

The feature cache is keyed on `${MODEL_NAME}_subj%02d.npz` and recorded
**nothing** about the front end the features came from, so a model name and a
set of flags could disagree with no error at any point. `run_encoding.py` now
writes `prov_lum_blur / prov_lum_cs / prov_age_months / prov_features /
prov_checkpoint / prov_stream1_weight / prov_seed` into every cache and
refuses to reuse a file whose provenance disagrees — or that predates the
guard and therefore records no provenance at all. It recomputes instead of
failing, so every existing cache gets one clean regeneration and is
self-validating from then on.

This is the same *class* of failure as §10.1 (reading `checkpoint_last.pt`
mid-training): a silent reuse of an artefact that looks right and is not.
§10.1's fix was `assert_converged`; this is its counterpart for the front end.

### 22.7 Repair, folded into the §7.3b array

`slurm/encode_cue_conflict.sh`, **job 26399760**, array 0-23%6, one array over
(model x subject), front-end flags selected from the task id:

| tasks | model | front end |
|---|---|---|
| 0-7 | `dkl_bcs_seed0_trained` | blur + CS |
| 8-15 | `thesis_dvd_trans` | blur + CS |
| 16-23 | `dkl_base_seed0_trained` | **`--no-lum-blur --no-lum-cs`** |

Output goes to `results/encoding_cc/`, not `results/encoding/`, so the
existing fits are not destroyed and reproducing their R2 in a separate
directory doubles as a determinism check on subjects 2-8. Tasks 16-23 are the
repair; task 16 is the file that was wrong.

**Do not re-report any `dkl_base_seed0_trained` n=8 number until 26399760
lands.** Until then quote seed1, or seed0 at n=7.

### 22.8 Figure 1 contains the contaminated subject (for the other session)

`results/figures/fig1_data.npz` was built 2026-08-30 22:59, i.e. *after* the
bad `dkl_base_seed0_trained_subj01.npz` landed at 06:06 that morning. Its
`nodiet_s0__profile / __centred / __rho` arrays are (8, 7) over subjects 1-8
and therefore include the contaminated subject in row 0. The `nodiet_s1`,
`diet_thesis`, `diet_ours_s0` and both `untrained_*` panels are unaffected.

`fig1_diet_gradient.pdf/png` and the caption must be regenerated once 26399760
lands. This is not a redraw for taste — one of the four trained rows is wrong.
Per §22.5 the correction moves the no-diet row *further* from the diet rows,
so the figure's message does not change and its contrast gets cleaner.

Since §20.5 already requires a Figure 1 revision (relabelling the untrained arm
as a front-end validation after §19), both changes should be made in the same
pass by whoever owns the figure. This session is not touching it — see §31.

### 22.9 Verification status of the §7.3b code (job 26399760)

Tested before submission, not after:

- `prepare_band` with an extra design: extra receives the training mu, the
  training degenerate-dimension mask and the training scale; with
  `Xb_extra=None` the train/test outputs are bit-identical to before.
- `fit_banded` with a synthetic design whose magno band carries the shape
  label and whose parvo band carries the texture label: `pred_extra_*` come
  back at (n_extra, n_voxels), the parvo-band VSB is strongly negative as
  constructed, and with no extra design the returned keys and `r2_joint` are
  unchanged.
- Full cue-conflict path end to end on a 64-image subset: 1200 images listed
  (same-cue excluded, 16 shapes x 75, 16 textures), front end, features, PCA
  projection, VSB, npz round-trip. NB the subset returns VSB = -1 because the
  first 64 images are all `airplane` and shape variance is identically zero
  there -- the correct degenerate answer, not a failure.

Not yet exercised: the voxel-chunked concatenation of `pred_extra_*` along the
voxel axis (the array runs with `--voxel-chunk 0`, a single chunk, so this
path is not hit) and the real 1200-image fit.

---

## §25 `parvo_highsf` closes the conjunction into a 2×2 on the INPUT factors

**n=2, confirmed** (`dkl_bcs_parvo_highsf` seeds 1 and 2, cue-conflict; seed 0
is at 139/150 and excluded). This was the prediction the conjunction claim was
most exposed to, and the two seeds agree to within 0.006 on every condition.

The condition gives stream 2 **untransformed high-acuity luminance + rg** (2
channels) — the biologically corrected parvo input, since real parvocellular
cells carry both. So parvo gets the achromatic *carrier* without the
developmental *transform*.

| condition | seed 1 | seed 2 | mean |
|---|---|---|---|
| magno_only | 0.798 [.756,.839] | 0.803 [.762,.841] | **0.801** |
| **parvo_only** | 0.390 [.352,.431] | 0.402 [.364,.443] | **0.396** |
| full | 0.516 [.476,.555] | 0.491 [.451,.532] | **0.504** |

### 25.1 The prediction holds — and the input factors now form a clean 2×2

§21.3 claimed only an achromatic carrier can be driven into shape bias. The
sharper test is whether the carrier *alone* suffices. It does not:

| | **no transform** | **with transform** |
|---|---|---|
| **chromatic carrier (rg)** | 0.364 (no-diet parvo) | 0.484 (swap parvo) |
| **achromatic carrier (lum)** | **0.396 (parvo_highsf parvo)** | **0.878 (diet magno)** |

Both single factors land at 0.36–0.48. Only the **conjunction** reaches 0.878.
That is an interaction, not two additive main effects, and it is now measured on
all four cells rather than inferred from two. This is the strongest causal
statement in the project: **the shape bias is produced by the developmental
low-pass transform applied specifically to an achromatic channel, and neither
factor alone does anything much.**

**The linear probe replicates the same 2x2 independently** (job 26399698,
`results/shape_texture/dkl_bcs_parvo_highsf_seed{1,2}_trained.json`,
magno/parvo shape−texture bias, n=2 agreeing to 0.017):

| | **no transform** | **with transform** |
|---|---|---|
| **chromatic carrier (rg)** | −0.504 (no-diet parvo) | −0.155 (swap parvo) |
| **achromatic carrier (lum)** | **−0.266 (parvo_highsf parvo)** | **+0.819 (diet magno)** |

The probe version is the more interpretable of the two because zero is
meaningful (it is the shape/texture sign flip), and the result is sharper:
**both single factors move parvo substantially toward neutral — −0.504 to
−0.266 for the carrier alone, −0.504 to −0.155 for the transform alone — and
neither crosses zero. Only the conjunction does, and it goes all the way to
+0.819.** Two measurement modalities, four cells each, same interaction.

### 25.2 The capacity confound is not merely argued away — it is measured

`parvo_highsf` gives stream 2 an extra input channel, an architectural
difference from every other condition (disclose this). The confound would
matter if the extra channel were *weakening* parvo and thereby faking the null.
It is not — the extra channel demonstrably **helped** parvo:

- parvo 16-way accuracy **0.486** (0.496 / 0.477), up from ~0.379 in the
  standard diet models -- a ~28% relative gain
- full accuracy 0.500 (0.515 / 0.485), up from 0.447

So parvo received materially more usable information, got materially better at
the task, and *still* did not become shape-biased (0.396 vs 0.418 standard).
A capacity confound can only push parvo's competence up, which is the direction
that would have *helped* it become shape-biased if the carrier were sufficient.
The null survives its own most favourable confound.

### 25.3 It independently confirms §20.2's averaging account

The integrated readout **drops** to 0.504, from 0.675 in the standard diet
models, even though magno is still strongly shape-biased (0.801). §20.2 argued
the readout is a competence-weighted average rather than a synergy. This is that
account's prediction: making parvo a *better* classifier gives its
texture-biased logits more pull, so the integrated shape bias falls. Predicted
before it was measured, and the accuracy numbers in §22.2 supply the mechanism.

Together with `s1_90` (up-weighting magno raises integrated shape bias to 0.712)
this brackets the effect from both sides — the readout tracks the mixing weight
and the relative competence of the streams, and never exceeds its best stream.

### 25.4 The watch item resolves — but NOT the way I first wrote it

**Correction to the first draft of this subsection.** From behaviour alone,
magno read 0.798 / 0.803 against 0.870–0.892 in the standard diet models, and I
wrote that "when parvo is made competent, magno specialises less". The linear
probe, which arrived afterwards, **refutes the representational version of that
claim**:

| measure | standard diet magno | parvo_highsf magno |
|---|---|---|
| probe (shape − texture) | +0.819 | **+0.852** (unchanged, if anything higher) |
| behavioural shape bias | 0.878 | **0.801** (lower) |
| behavioural 16-way accuracy | ~0.397 | **0.303** (much lower) |

Magno's *representation* is exactly as shape-biased as before. What falls is
magno's **readout competence** — it classifies far fewer images correctly
(0.303 vs ~0.397), and behavioural shape bias is computed only over decided
images, so a weaker decider contributes less to the integrated readout.

Meanwhile parvo's accuracy rises (0.486 vs ~0.379) and full accuracy rises
(0.500 vs 0.447). So the division of labour is real, but it is a division of
**classification**, not of representation: parvo takes over more of the task
while magno keeps representing shape exactly as it did.

That is a better claim than the one I first wrote, and it only became available
because the two modalities disagreed. It also removes the "specialisation is
relational" speculation — the front end fixes what magno represents, and what
changes is how much the readout leans on it. Consistent with §25.3 and §20.2:
the readout is a competence-weighted average, and here the competences moved
without the representations moving.

### 25.5 Disclosure

The extra input channel is an architectural difference from every other
condition (a 2-channel vs 1-channel stem conv, ~64 extra parameters). §25.2
argues and measures why it cannot manufacture the null, but it must be stated
in the text rather than left for a reviewer to find. `parvo_highsf` is also not
yet run through the encoding path, so it contributes to legs (a)/(b) only, not
to leg (c). Both seeds are covered on both behavioural measures (n=2); seed 0
(139/150) would take it to n=3.

---

## §26 Numbering collision, and Figure 1 is quarantined until the repair lands

### 26.1 Duplicate section numbers (two sessions appending concurrently)

The document currently has **two §21s and two §22s**, because two sessions
appended without seeing each other's headers:

| number | section | author |
|---|---|---|
| §21 | The swap probe on converged checkpoints (line ~1822) | this session |
| §21 | Claiming §7.3b, in-silico cue-conflict → cortex (line ~1902) | other session |
| §22 | A contaminated subject in the no-diet cell (line ~1990) | other session |
| §22 → **§25** | `parvo_highsf` closes the conjunction (line ~2155) | this session, **renumbered** |

I renamed only my own newest section (§22 → §25) rather than editing the other
session's headers, since they may have edits in flight. The remaining duplicate
§21/§22 pair should be resolved in one pass by whoever next edits the document;
cross-references in §25 and §26 use the *titles*, not the numbers, so they
survive renumbering either way. Same root cause as §20.4 — append-only edits to
a shared file from two sessions.

### 26.2 Figure 1 is quarantined — I own it and I am not regenerating it yet

Acknowledging §22.8 ("Figure 1 contains the contaminated subject", addressed to
this session). Confirmed: `results/figures/fig1_data.npz` was built at 22:59 on
2026-08-30, after the bad `dkl_base_seed0_trained_subj01.npz` landed at 06:06,
so `nodiet_s0__profile / __centred / __rho` include the contaminated subject in
row 0.

**Numbers I reported that are contaminated, and must be withdrawn:**

- Figure 1's headline no-diet trained value **−0.149** averaged the
  contaminated `nodiet_s0` (−0.1669) with the clean `nodiet_s1` (−0.1318).
  Withdrawn. Per §22.7, quote **seed1 at n=8 (−0.1318)** or **seed0 at n=7
  (−0.0721)** until job 26399760 lands, and no `dkl_base_seed0_trained` n=8
  number at all.
- The caption's panel-A sentence "to −0.149 without the diet" and panel-D's
  no-diet ρ = −0.56 are both computed from the affected arrays.

Not affected: `untrained_nodiet`, `untrained_diet`, `nodiet_s1`,
`diet_thesis`, `diet_ours_s0`. So panels B and C, and the diet arm throughout,
stand.

**Direction of the correction**: per §22.5 the contamination made the no-diet
model look *more* like the diet model, so removing it widens the gap Figure 1 is
built on. The figure's message does not change and the contrast gets cleaner —
this is a correction, not a retraction.

**Why I am not regenerating now**: job 26399760 (24 tasks at %6, ~3 h each) is
~12 h from completion, and regenerating against the seed1-only data would
produce a third set of numbers that also has to be replaced. One regeneration,
after the repair, folding in all three pending changes:

1. the repaired `dkl_base_seed0_trained` subject 1 (§22.7),
2. relabelling the untrained arm as a front-end validation, not evidence about
   architecture (§20.5, from §19's Gabor null),
3. adding the swap arm to panel C, now available on converged checkpoints for
   both measures (§21.4), and the `parvo_highsf` cell if n=2 confirms (§25).

Until then `fig1_diet_gradient.{png,pdf}` and `fig1_caption.md` must not be used
in any draft. A warning banner has been written into `fig1_caption.md` itself so
the file cannot be picked up innocently.

---

## §23 Process note, 2026-08-31 — two files committed by the wrong session

Commit `ad64e58` (this session, the §7.3b code) used an unrestricted
`git add -A` and swept in two files belonging to the *other* session:

    results/cue_conflict/dkl_bcs_parvo_highsf_seed2.json
    results/cue_conflict/dkl_bcs_parvo_highsf_seed2_logits.npz

Verified intact, not truncated mid-write: the npz loads with all five arrays
at full size (`shapes`/`textures` 1200, `full`/`magno_only`/`parvo_only`
1200x112), the json parses, and both are byte-identical to the working tree.
Nothing is lost and nothing needs re-running — they are simply recorded under
a commit whose message does not mention them.

History is **not** being rewritten to extract them: `mpk-dev` is shared with a
live session and rewriting under it is worse than a mislabelled commit.

**Rule going forward, both sessions**: in this worktree use path-limited
`git add <paths>`, never `git add -A`/`git add .`. Two sessions write to
`results/` and `docs/` concurrently, so an unrestricted add commits whatever
the other session happens to have on disk at that instant — and unlike this
case, it could catch a file *during* a write.

Also noted: the index currently holds another session's staged, uncommitted
work (`docs/PAPER_PLAN.md`, a `parvo_highsf_seed1` result, and
`tmp_analysis/fix_subsection_numbers.py`). The staging area is shared across
the worktree, so `git commit` without explicit paths would commit that too.
This session has committed only explicit paths since.

---

## §24 First §7.3b result (subject 6, diet cell) — and a circularity caveat
declared before the group analysis

`results/encoding_cc/dkl_bcs_seed0_trained_subj06.npz`, the first task of
26399760 to land. Recorded now, with the caveat, so the group result cannot be
framed after seeing it (§7.4's rule).

### 24.1 The refit reproduces the original bit-for-bit

Against `results/encoding/dkl_bcs_seed0_trained_subj06.npz`:

| quantity | max abs difference |
|---|---|
| r2_joint, r2_magno, r2_parvo | **0.000e+00** |
| unique_magno, unique_parvo, spi | **0.000e+00** |

Exactly zero, not merely small. Three things follow: the pipeline is
deterministic end to end; the §7.3b code changes perturb nothing upstream of
the extra design; and the un-keyed feature cache for this model/subject was
in fact sound, since the forced recompute reproduced it exactly. The
provenance guard (§22.6) costs a recompute and buys proof.

### 24.2 Leg (a), measured for the first time — in the encoding model

Median VSB over reliable voxels, by band (n = 16503):

| band | median VSB | eta2_shape | eta2_texture |
|---|---|---|---|
| **magno** | **+0.770** [IQR +0.61, +0.88] | 0.352 | 0.045 |
| **parvo** | **−0.748** [IQR −0.85, −0.60] | 0.049 | 0.354 |
| joint | −0.425 | 0.111 | 0.277 |

A near-perfect mirror image, with nothing pinned at the ±1 bounds (0.0% of
voxels), so it is not a degenerate ratio. The magno stream's predicted
responses to cue-conflict images are dominated by the **shape** cue by ~8x in
explained variance; the parvo stream's by the **texture** cue by ~7x.

§7.3 called leg (a) "asserted in the title and tested nowhere". This is a
direct measurement of it, and it agrees in sign and in ordering with the
behavioural decomposition (§18/§20.1: magno B_shape 0.878, parvo 0.484).

### 24.3 The panel — and why its headline number is partly circular

    rho(VSB, SPI) across voxels, subject 6:
      joint model   +0.2090  (p = 2.6e-162)
      magno band    -0.0495
      parvo band    -0.0021

The joint result falls in §31.4's **positive** branch. But it must not be
reported as the pre-declared "strong unifying result" without this caveat,
which the within-band rows make unavoidable:

Within a single band, VSB barely varies with SPI at all (−0.05, −0.00). So the
entire +0.209 comes from **band mixing** — and given §24.2, that is close to
definitional. SPI measures which band explains a voxel; the magno band's
predictions are shape-dominated and the parvo band's texture-dominated; so a
voxel weighted toward magno must have shape-dominated joint predictions. The
correlation is largely guaranteed once §24.2 is known.

What is *not* circular: SPI is estimated from brain responses to NSD images
and VSB from Geirhos cue-conflict images, so the two are computed on disjoint
stimulus sets. The conjunction is real. Its **effect size is inflated** by the
shared band structure, and +0.209 is not evidence of a cortical fact
independent of §24.2.

**Consequence for the manuscript, decided now**: the flagship panel is
§24.2 (the streams' shape/texture dissociation, measured through cortex),
with the VSB-SPI correlation as a consistency check carrying the caveat in
its caption — not as an independent result. Do not lead with rho = +0.209.

The genuinely independent test, which the group data can support and which
should be run when all 24 tasks land: does the predicted VSB map reproduce the
**known cortical organisation** of shape vs texture selectivity (higher shape
weighting in LO/hV4 than in V1/V2)? That is checkable against the literature
rather than against SPI, and it is a real prediction the model can fail.

### 24.4 Status

n = 1 subject, one model, one cell of the design. Nothing above is a group
claim. 23 tasks outstanding.

### 24.5 Correction to §24.3: the ROI test is circular too (n=2, preliminary)

§24.3 proposed "does predicted VSB rise along V1 -> hV4/LO?" as the
*independent* test that could fail. It is not independent, and running it makes
the reason obvious.

Joint-model VSB by ROI, `dkl_bcs_seed0_trained`, subjects 4 and 6 only:

| V1v | V1d | V2v | V2d | V3v | V3d | hV4 | LO |
|---|---|---|---|---|---|---|---|
| −0.008 | +0.127 | +0.070 | +0.027 | −0.114 | −0.179 | **−0.529** | **−0.534** |

monotonicity rho over V1v..hV4 = **−0.661** (0/2 subjects positive).

VSB *falls* along the hierarchy. But the joint model's VSB is set by band
mixing (§24.3), and band mixing across ROIs **is** the §15 gradient — magno
weighting is highest in V1/V2 and lowest at hV4. So a declining VSB profile is
the §15 profile re-expressed in shape/texture units, not a second measurement.
The ROI test inherits exactly the confound §24.3 identified in the voxel test;
proposing it as the escape from that confound was wrong.

**What this means for §7.3b as a whole.** With this VSB definition the joint
model's cortical shape/texture profile is largely a re-description of leg (c).
The panel does not supply an independent cortical test, and the paper should
not claim one. The genuinely new content of §7.3b is **§24.2** — the per-band
dissociation, which is a fact about the streams and is *not* determined by the
variance partition.

**The tension worth reporting anyway** (n=2, must be re-checked at n=8): the
model predicts a **texture-dominated hV4 and LO** (−0.53, −0.53). Human LO is
strongly shape/object selective. If this survives to n=8 it is a real,
interpretable failure of the two-stream model — its shape-carrying stream is
weighted *out* exactly where cortex is most shape-selective — and it should be
disclosed as such rather than dropped. It is a consequence of §15's gradient,
so it is a prediction the architecture actually makes.

A genuinely independent test would need measured shape/texture selectivity in
the same voxels, which requires cue-conflict fMRI this project does not have.
Absent that, §7.3b is: leg (a) measured (§24.2), plus a disclosed tension.

### 24.6 §24.5 was too strong — the profile survives residualisation (n=3)

§24.5 argued from the *mechanism* that the ROI profile must be the band-mixing
gradient re-expressed, and concluded §7.3b "does not supply an independent
cortical test". That was an inference, not a measurement. Measuring it:

**SPI explains 0.161 of the variance in joint VSB** (per subject 0.18, 0.17,
0.14). The confound is real but accounts for a *sixth* of the signal, not the
bulk of it. "Largely a re-description of leg (c)" is wrong.

Removing the dependence non-parametrically — subtract the median VSB within
each of 20 SPI quantile bins, which kills any monotone relation, not merely a
linear one — and again within each cell of a 10x10 (SPI decile) x
(r2_joint_nc decile) grid, which additionally removes fit quality:

| profile | V1v | V1d | V2v | V2d | V3v | V3d | hV4 | LO | hV4−V1 |
|---|---|---|---|---|---|---|---|---|---|
| raw | −0.101 | +0.014 | −0.014 | −0.076 | −0.157 | −0.181 | −0.543 | −0.536 | — |
| − SPI | +0.008 | +0.137 | +0.167 | +0.129 | +0.189 | +0.137 | **−0.111** | −0.039 | −0.184 |
| − SPI x predictability | −0.093 | +0.058 | +0.038 | +0.088 | +0.102 | +0.099 | **−0.105** | −0.046 | −0.088 |

The effect attenuates substantially (hV4 −0.543 → −0.111 → −0.105) but does
not vanish, and hV4/LO remain the most texture-shifted areas under both
controls. Whatever survives cannot be band preference and cannot be fit
quality, because both were removed within-subject at decile resolution.

**Why a remainder exists at all**, which §24.5 missed: SPI summarises *which
band* explains a voxel, but not *which features inside that band* it reads
out. Two voxels with identical SPI can weight shape-tuned and texture-tuned
magno features differently, and that degree of freedom is not touched by the
variance partition.

**Revised status of §7.3b.** It supplies (i) leg (a), §24.2, untouched by any
of this; and (ii) a genuine, if modest, cortical prediction — the model places
its texture-dominated readout at hV4/LO, beyond what band weighting explains.
Since human LO is strongly shape-selective, that remains the tension flagged
in §24.5, and it is now a *sharper* tension because it cannot be dismissed as
an artefact of the variance partition.

**n = 3 subjects, one model.** No Wilcoxon is possible below n=5 and none is
reported. Everything here is provisional until the cell completes; the
attenuation from −0.184 to −0.088 between the two controls is itself large
enough that the n=8 numbers could land either side of meaningful.

### 24.7 §24.2 is NOT the first measurement of leg (a) — correcting the claim

§24.2 called the per-band VSB dissociation "leg (a), measured for the first
time", quoting §7.3's "asserted in the title and tested nowhere". That
sentence was written 2026-08-28 and was already superseded:
`src/analysis/behavior/stream_shape_texture.py` is the §7.3a probe, it has
run, and `results/shape_texture/dkl_bcs_seed0_trained.json` holds a **cleaner
and stronger** measurement of the same thing on the same 1200 images:

| stream | shape acc | texture acc | bias |
|---|---|---|---|
| magno | **0.969** | 0.159 | **+0.810** |
| parvo | 0.454 | **0.936** | −0.482 |

magno − parvo = **+1.292**, consistent across all 5 folds (1.26–1.35), with
both streams matched at 256 PCA components so the general-factor confound is
controlled. Chance is 0.0625.

My VSB version (magno +0.770, parvo −0.748) agrees in sign and ordering, but
it is a *weaker instrument*: a variance ratio on predicted BOLD rather than a
cross-validated decoding accuracy, with no fold-level reliability.

**What §24.2 actually contributes**, stated correctly: the stream
dissociation survives projection through the *voxel-predictive* dimensions —
the probe shows the information is in the stream features, and VSB shows the
encoding model's readout preserves it. That is a consistency check between two
independent methods, worth one sentence, and it is not a first measurement.
Cite `stream_shape_texture.py` for leg (a); do not lead with VSB.

**Consequence: the real contribution of §7.3b is §24.6**, the cortical VSB
profile — the model's texture-dominated hV4/LO, surviving residualisation on
SPI and on predictability. That has no counterpart in
`stream_shape_texture.py`, which has no cortical axis at all. §7.3b should be
written up around the profile, with §24.2 as its sanity check.

**Housekeeping**: §7.3's "leg (a) has no direct measurement anywhere in the
project" is stale and should be struck when that section is next revised, so
it stops misleading readers of the plan (it misled this session).

## §32 The input-level null for the §7.3b profile — pre-declared 2026-08-31

§24.7 concluded the real contribution of §7.3b is §24.6's cortical VSB profile
(texture-dominated hV4/LO, surviving residualisation on SPI and on
predictability). §19 established that this project cannot report a cortical
profile without asking whether the *front end alone* produces it — the
untrained-network panels failed exactly that test and had to be relabelled.
§24.6 currently has no such column, so it is at present an unguarded claim of
precisely the kind §19 disallows.

**Job 26401261, `slurm/encode_cc_gabor.sh`, 8 tasks**: `gabor_blur_cs` with
`--cue-conflict`, i.e. oriented-energy features on each stream's own
transformed input, blur+CS front end to match `dkl_bcs_seed0`. The Gabor bank
is the same one used in §19 (`input_bands.py`, 4 scales x 8 orientations,
2048 dims/stream). Untrained-network columns are *not* run: §19 showed the
untrained ResNet predicts BOLD no better than this bank (0.1408 vs 0.1384), so
the bank is the cheaper and stricter null.

### 25.1 Outcomes, written before the fit

- **The Gabor null reproduces the declining profile** (hV4/LO texture-shifted,
  monotonicity rho negative). Then §24.6 is a **front-end fact**, not a model
  prediction, and §7.3b collapses to §24.2's consistency check. The
  "texture-dominated LO contradicts human LO" tension would then be a
  statement about blurred luminance versus red-green input, not about the
  trained two-stream network, and must be re-worded accordingly.
- **The Gabor null is flat, or declines much less.** Then the profile is
  *built by the network* and §24.6 stands as a genuine model prediction — and
  the tension with human LO's shape selectivity is a real property of the
  trained model, worth reporting as a limitation with teeth.
- **The Gabor null declines more steeply.** The network partially *corrects*
  an input-level bias, which would be the most interesting outcome of the
  three and would parallel §19.6 (the diet hands the network a worse starting
  point and learning overtakes).

Both statistics (§8) and the same residualisation ladder (§24.6 sections 4-5)
apply to the null exactly as to the trained model; a difference claimed on the
raw profile but not on the residualised one is not claimed.

### 25.2 Note on what this cannot settle

The Gabor null shares the *front end* with `dkl_bcs_seed0` but not the
architecture. A flat null therefore licenses "the network builds it", not "the
diet builds it". Separating those needs the no-diet trained column
(`dkl_base_seed0`, tasks 16-23 of 26399760, already running) against this one.
The three columns together — input-only, no-diet-trained, diet-trained — are
the same design §19 used, and should be reported the same way.

---

## §27 The architecture-only control is now runnable — and `rgb_tsn` is a misnomer

Written 2026-08-31, this session. Numbered 27 because 21, 23, 24 and 25 are all
either duplicated or out of order at the tail of this file (see §26.1); 27 is
the lowest number that collides with nothing.

### 27.1 Why this control matters more than its §7.4 placement suggests

Every claim in leg (a) has the same logical shape: *biologically constrained
input channels induce stream specialization*. The obvious way that claim dies
is if a two-stream architecture specializes on its own, for reasons that have
nothing to do with what enters each stream. Until now that possibility was
argued against rather than measured, because the entry point built a
1-channel DKL front end only and `rgb_tsn` could not be run through it.

`rgb_tsn` is the exact control: both streams receive the **identical**
untransformed 3-channel image. There is no input asymmetry of any kind, so
the two streams differ by nothing except their random initialization. If a
shape/texture division appears here anyway, the causal story in §21/§25 is in
serious trouble. If it does not, the conjunction result is the cleanest it
can be — specialization requires the input manipulation, full stop.

All three seeds converged at 150/150, so this is available at n=3 immediately.

### 27.2 The name is wrong, and the paper must say so

`slurm/train_array.sh:45,61` feeds `miniecoset_256px_dkl_uint8.h5` to **every**
condition, `rgb_tsn` included. So the three channels this model sees are the
DKL axes — luminance, red-green, blue-yellow — not R/G/B. What
`input_mode="rgb"` actually selects in `standalone_trainer.StreamInputBuilder`
is "hand both streams the whole 3-channel image and apply no transform".

This is not a bug, and it does not weaken the control — arguably it
strengthens it, because the comparison against `dkl_base` and `dkl_bcs` now
holds the *data* fixed and varies only how channels are routed to streams. But
the manuscript cannot call this condition "RGB". It should be named for what
it is: **full-input two-stream**, or `dkl_full`. Flagging rather than renaming
the run directories, which would break every existing checkpoint path.

One further disclosure: `build_augmentations` appends `RandomGrayscale(p=0.5)`
when `input_mode == "rgb"`, so this condition — and only this condition — was
trained with an augmentation that averages the three DKL axes together half
the time. That is a real difference from the other cells, it was not chosen
deliberately for DKL data, and it belongs in the methods text. It cannot
manufacture a false negative (it destroys chromatic information, which if
anything should *encourage* the streams to differentiate), but it is not
nothing.

### 27.3 Implementation, and the pre-flight that gates it

`dkl_frontend.rgb_to_dkl3` was factored out of `rgb_to_stream_inputs` (which
now calls it and slices), and `magno_parvo_inputs(rgb_mode=True)` returns that
full tensor to both streams, ignoring blur/CS because the trained condition
applies neither. Both behaviour entry points gained `--rgb-mode`, and the
`in_channels` decision moved into one shared `cue_conflict.stream_in_channels`
rather than being duplicated as an inline conditional in each file.

`tmp_analysis/check_rgbmode_load.py` gates the run, in the same spirit as the
`parvo_highsf` pre-flight: a mismatched stem would leave random weights and
fabricate exactly the "no specialization" answer this control is supposed to
be able to give honestly. It reports **242/242 tensors matching the
checkpoint**, 3-channel stems on both streams, and byte-identical stream
inputs. Verdict SAFE — recorded before the jobs were submitted, not after.

### 27.4 What is reported, and how to read it

> **SUPERSEDED by §28.2 (2026-09-01).** The test proposed below is
> vacuous: the two streams are bit-identical at initialization, so
> with identical inputs they can never diverge. Read §28 instead.

Because both streams see the same input, `magno_only` / `parvo_only` are
misnomers here too; they are stream 1 and stream 2, and the prediction is that
they are statistically indistinguishable. The quantity of interest is the
**absolute** stream difference |bias(S1) − bias(S2)| at n=3, against the diet
cell's ~1.1 (probe) and ~0.46 (behaviour). A near-zero difference here is the
result the causal claim needs.

Note this control cannot be one-sided-tested into significance at n=3 and is
not meant to be: it is reported as a magnitude comparison against the diet
cell, with the seed spread shown, not as a p-value.

### 27.5 Submitted

- `26401266` / `26401267` — probe and cue-conflict for `dkl_bcs_swap_seed2` and
  `dkl_bcs_parvo_highsf_seed0`, both held on `afterok:26385433` (the last
  training task, at 148/150 when submitted). These take swap and
  `parvo_highsf` from n=2 to **n=3**, matching every other cell.
- `26401311` / `26401312` — probe and cue-conflict for `rgb_tsn` seeds 0–2.

All four on `--partition=main`, no `--gres`, as CPU-only work.

## §33 The §7.3b profile at n=7, both statistics — and what actually survives

`dkl_bcs_seed0_trained`, subjects 1-6 and 8 (subject 7 still running in
26399760). n=7 puts the exact Wilcoxon floor at p=.0156, so significance is
reachable for the first time. `tmp_analysis/cc_panel.py` now computes both
statistics required by §8:

    norm = (eta2_s - eta2_t)/(eta2_s + eta2_t)   [VSB, bounded]
    raw  =  eta2_s - eta2_t                      [unbounded]

### 26.1 Everything below agrees on both statistics

| quantity | norm | raw | subjects | p |
|---|---|---|---|---|
| stream dissociation magno − parvo | +1.541 | +0.620 | 7/7 | .0156 |
| profile hV4 − V1 (uncorrected) | −0.467 | −0.190 | 7/7 | .0156 |
| after removing SPI | −0.150 | −0.071 | 6/7 | .0312 |
| after removing SPI x predictability | −0.096 | −0.057 | 6/7 | .0312 |

The confound share is stable across statistics and subjects (SPI removes
.170/.166 of the variance; SPI x predictability .219/.215). So §24.6's
conclusion holds at n=7 and under the project's both-statistics rule: the
hV4/LO texture shift is **not** reducible to band weighting or to fit quality.

### 26.2 But it is a step at hV4/LO, NOT a hierarchical gradient

This wording matters and §24.6 got it wrong by implication. Monotonicity
along V1v..hV4:

| | uncorrected | − SPI | − SPI x predictability |
|---|---|---|---|
| rho (norm) | **−0.663** (0/7 positive) | −0.087 (2/7) | −0.066 (2/7) |
| rho (raw) | **−0.597** (1/7) | −0.077 (3/7) | −0.010 (3/7) |

The monotone gradient **is** the band-weighting effect: it collapses to
approximately zero once SPI is removed, and the sign becomes inconsistent
across subjects. What survives residualisation is not a gradient at all --
the residual profile is *positive* through V1d..V3d (+0.03 to +0.17, norm)
and negative only at hV4 and LO.

So the correct statement is: **after controlling band preference and fit
quality, hV4 and LO stand apart from every earlier area as texture-shifted.**
It is a categorical step at the two areas, not a progressive shift along the
hierarchy. Do not write "the model predicts a shape-to-texture gradient";
write that it places a texture-dominated readout specifically at hV4/LO.

This is still the claim that collides with human LO's strong shape/object
selectivity, so §24.6's tension stands and is if anything more pointed -- it
is localised to exactly the areas where the discrepancy is sharpest.

### 26.3 Not yet settled

- Subject 7 outstanding; n=8 will move the floor to p=.0078.
- **The input-level null (§32, job 26401261) is the control that decides
  whether any of this is about the network at all.** Nothing in §33 should be
  written into the manuscript before that column exists -- §19 is the
  precedent, where a comparable-looking cortical profile turned out to be a
  property of the front end.
- The no-diet column (tasks 16-23 of 26399760) is needed to separate "the
  network builds it" from "the diet builds it" (§32.2).

### 26.4 The hV4/LO step is not an eccentricity artefact

hV4 and LO are foveally biased and this dataset has a strong eccentricity
gradient (§7.1, §A3.2), so "hV4/LO are texture-shifted" and "foveal voxels are
texture-shifted" are competing explanations of §33.2. Tested with a third
control factor (`$CLAUDE_JOB_DIR/tmp/ecc_confound.py`, n=7).

**First, the eccentricity effect is real and large.** Residual VSB by
`prf-eccrois` band after removing SPI x predictability (all five bands, per
§7.1's ECC_LABELS fix):

| 0.5° | 1° | 2° | 4° | 4+° |
|---|---|---|---|---|
| −0.199 | −0.221 | −0.116 | +0.111 | **+0.323** |

Monotonic, range 0.52. The model predicts foveal voxels texture-dominated and
peripheral voxels shape-dominated, *beyond* band weighting and fit quality.
This is the §A3.2 eccentricity story restated in shape/texture units and is a
new observation, not a restatement of SPI (SPI was residualised out).

**But it does not explain the ROI step.** Adding the eccentricity band as a
third residualisation factor:

| controls | hV4 − V1 | p | subjects |
|---|---|---|---|
| SPI x predictability | −0.0964 | .0312 | 6/7 |
| SPI x predictability x **ecc band** | **−0.0926** | **.0156** | **7/7** |

The step barely moves (−0.096 → −0.093) and becomes *more* consistent, not
less — 7/7 subjects at the n=7 floor. So §33.2 survives three controls: band
preference, fit quality, and retinotopic eccentricity. That is a considerably
harder target for a reviewer than it was an hour ago.

Caveat to carry: `prf-eccrois` labels only ~33% of the reliable mask (§7.1),
so unlabeled voxels form a single residualisation cell rather than being
eccentricity-matched. hV4 and V1 are retinotopic and largely labeled, so the
contrast tested here is well covered, but the control is weaker for LO.

**Both of these still await §32's input-level null.** A Gabor bank on blurred
luminance versus red-green input could plausibly produce *both* the
eccentricity effect and the hV4/LO step by itself. Nothing in §33 is a claim
about the network until 26401261 says otherwise.

---

## §28 The architecture has no symmetry breaking — §27.4 was the wrong test

Written 2026-09-01. Supersedes §27.4, which framed `rgb_tsn` as an empirical
test it cannot be.

### 28.1 What came back

All ten queued tasks completed. Every cell is now at **n=3** on both measures:

| cell (what stream 2 receives) | probe S1 | probe S2 | S1−S2 | cc S1 | cc S2 | cc full |
|---|---|---|---|---|---|---|
| no diet — rg, untransformed | −0.2261 | −0.5042 | +0.2781 | 0.4210 | 0.3643 | 0.4162 |
| diet on lum — rg, untransformed | +0.8194 | −0.5183 | **+1.3378** | 0.8778 | 0.4177 | 0.6749 |
| diet on rg (swap) — rg, transformed | −0.2286 | −0.1453 | −0.0833 | 0.4007 | 0.4949 | 0.4083 |
| diet + hi-SF parvo — raw lum+rg | +0.8442 | −0.2708 | +1.1150 | 0.7966 | 0.3886 | 0.4859 |
| no split (`rgb_tsn`) — identical to S1 | −0.1425 | −0.1425 | **+0.0000** | 0.4446 | 0.4446 | 0.4449 |

Seed spread is tight enough to quote: swap −0.0883/−0.0733/−0.0883,
parvo_highsf +1.1100/+1.1100/+1.1250. The n=2 numbers in §21 and §25 survive
the third seed without qualification, and the conjunction 2×2 on the S2 cell
is unchanged: −0.504 (chromatic, no transform), −0.145 (chromatic,
transform), −0.271 (achromatic, no transform), **+0.819** (achromatic,
transform). Neither single factor crosses zero; only the conjunction does.

### 28.2 The control's zero is exact, and that is the problem

`rgb_tsn` returned S1 − S2 = **+0.0000 on every fold of every seed**. An
exact zero is not a measurement, so I checked the weights rather than
reporting it. Two facts, verified directly:

1. In the trained `rgb_tsn` checkpoints, **all 120 stream-1/stream-2 tensor
   pairs are bit-identical** (max |S1 − S2| = 0.000e+00, seeds 0 and 1). The
   diet conditions have max |S1 − S2| of 34–53 over the same pairs.
2. Reproducing the trainer's own construction, `TwoStreamNetwork` initializes
   the two streams **bit-identically** — 0/120 pairs differ, for `rgb_tsn`
   and `dkl_bcs` alike, before and after `patch_stream_input_channels`.

So `rgb_tsn` could never have specialized: identical initialization plus
identical input gives identical gradients, and the two streams stay the same
tensor for all 150 epochs. The zero is an algebraic identity, not evidence
about learning.

**§27.4 is therefore wrong** where it said the quantity of interest is
|bias(S1) − bias(S2)| tested against the diet cell. That comparison is
vacuous. What follows replaces it.

### 28.3 What the run is actually worth — and what it must not be said to show

It must **not** be written as "two streams do not spontaneously specialize."
That hypothesis is untestable in this architecture, and a reviewer who checks
will find the streams are the same tensor. Claiming it would be an overclaim
that the code refutes in one line.

What it does establish is narrower and, for this paper, more useful:

> The two streams of the TSN are bit-identical at initialization. Every
> divergence between them is therefore attributable to the difference in
> their inputs — not to a lucky draw in one stream's weights.

That eliminates a real alternative explanation *by construction* rather than
by argument. It is the reason the conjunction 2×2 in §21/§25 has to carry the
causal weight, and can: since input is the only thing that can differentiate
the streams, showing *which property* of the input does it (achromatic
carrier × developmental transform, neither alone) is the whole claim.

The 150-epoch run is still worth one sentence as an empirical confirmation
that nothing in the training loop — augmentation sampling, dropout,
non-deterministic kernels — breaks that symmetry in practice. It did not:
after 150 epochs and ~40k steps the streams were still bit-identical. That is
a non-trivial check on the implementation, just not the check §27.4 claimed.

### 28.4 The honest limitation, for the paper

If a reviewer asks whether a two-stream architecture would specialize on its
own, the answer we can support is: *not this one, by construction.* Testing
the general question needs symmetry broken at initialization — a per-stream
seed — which is a new set of training runs (3 seeds × ~2 days GPU).

I am **not** queueing that. It answers a question the paper does not ask, the
narrative is already carried by the conjunction result, and the standing
research-output rule is to reframe rather than spend compute when the claim
is the thing that needs fixing. It goes in Limitations as a stated,
unexplored alternative, with the cost named so a reviewer can judge whether
the omission is reasonable.

### 28.5 Also still true from §27.2

`rgb_tsn`'s channels are DKL lum/rg/by, not R/G/B, and this condition alone
was trained with `RandomGrayscale(p=0.5)`. Both remain disclosure items
regardless of how the condition is now framed.

### 26.5 FINAL n=8 for the diet cell (`dkl_bcs_seed0_trained`)

All 8 subjects, exact Wilcoxon floor p=.0078. Supersedes the n=5/n=7 tables
above; those were interim and should not be quoted.

| quantity | norm | raw | p (norm/raw) | subjects |
|---|---|---|---|---|
| stream dissociation magno − parvo | **+1.534** | **+0.616** | .0078 / .0078 | 8/8 |
| profile hV4 − V1, uncorrected | −0.447 | −0.182 | .0078 / .0078 | 8/8 |
| − SPI | −0.128 | −0.063 | .0391 / .0234 | 6/8, 7/8 |
| − SPI x predictability | −0.080 | −0.050 | .0391 / .0234 | 6/8, 7/8 |
| **− SPI x predictability x ecc** | **−0.095** | — | **.0078** | **8/8** |

Both statistics significant on every row, so the §8 rule is met throughout.

**Adding the eccentricity control makes the step stronger, not weaker**
(−0.080 at 6/8 -> −0.095 at 8/8, p .0391 -> .0078). Eccentricity was
*opposing* the effect: hV4's foveal bias pushes its residual VSB in the same
direction as the ROI step, so leaving eccentricity uncontrolled mixes two
effects of opposite consistency. Matching on it isolates the ROI step and
makes it unanimous. This is the strongest form of the §26.2 result.

**Monotonicity is gone**, confirming §26.2's wording: rho after residualisation
is −0.022 (norm) and −0.013 (raw), with 3/8 and 4/8 subjects positive — i.e.
indistinguishable from zero. The uncorrected rho of −0.59 is band weighting.
The manuscript says "a texture-dominated readout localised to hV4/LO", never
"a gradient".

The eccentricity gradient itself is stable at n=8: −0.212, −0.222, −0.110,
+0.121, +0.325 across the five bands.

**Still gated on §25.** `thesis_dvd_trans` has begun landing (subject 3), so
the second diet model will provide a within-cell replication; the no-diet
column and the Gabor null remain outstanding.

## §34 Participation ratio — the §32 comparison is not dimensionality-matched

§7.2 item 6 asked for participation ratio per stream, to close the
general-factor objection that defeated the thesis's RSA. Computed from the
feature caches (2000 NSD images, subject 1):

| model | magno D | magno PR | parvo D | parvo PR |
|---|---|---|---|---|
| `dkl_bcs_seed0_trained` | 15360 | **269.9** | 15360 | 160.1 |
| `gabor_blur_cs` | 2048 | **15.3** | 2048 | 11.8 |
| `dkl_base_seed0_trained` (subj01, contaminated) | 15360 | **32.2** | 15360 | 189.5 |

### 27.1 Consequence for §32, stated before the null lands

The Gabor bank has roughly **18x fewer effective dimensions** than the trained
network. Both are projected to 1000 PCA components, but PCA cannot manufacture
dimensions the bank does not have. So the two columns are *not* matched in
representational capacity, and §32's comparison inherits that.

The caveat is **asymmetric**, which is what makes it tolerable:

- If the null **reproduces** the hV4/LO step, the conclusion is safe and in
  fact strengthened: a 15-dimensional input model producing the effect means
  the network is not needed for it, and no dimensionality argument rescues the
  network's claim.
- If the null **fails to reproduce** it, the result is **confounded**: the
  bank may lack the step because it lacks the capacity to express any
  fine-grained voxel structure, not because the front end lacks the
  information. That outcome cannot be reported as "the network builds it"
  without a capacity-matched control.

So §32.1's second and third outcomes both require a follow-up if they occur:
refit the trained model with its PCA truncated to the bank's effective
dimensionality (~16 components), and check the step survives. That is one
extra array and should be pre-committed now rather than argued about later.

### 27.2 Independent corroboration of the §22 contamination

`dkl_base_seed0_trained` subject 1 shows magno PR = **32.2** against parvo
PR = 189.5. Every other model here has the *magno* stream at least as
high-dimensional as parvo (dkl_bcs: 270 vs 160), which is what an unblurred or
diet-matched front end should give. A magno stream collapsed to a sixth of its
partner is exactly what applying a blur the network never trained on would do,
and this cache is the one §22 identified as contaminated — note it is also the
only row still reading `prov=unstamped`, i.e. written before the guard.

This is independent of the §22.2 feature-identity check and of the §22.4
outlier analysis, and agrees with both. Tasks 16-23 of 26399760 will overwrite
this cache with a provenance-stamped, correctly-blurred version; the PR asymmetry
should disappear, and that is a cheap post-hoc verification the repair worked.

### 27.3 Capacity ladder submitted pre-emptively — job 26402571

`slurm/encode_cc_pca.sh`, array 0-15%6, one array over (pca-dim x subject):

| tasks | output model | PCA dims |
|---|---|---|
| 0-7 | `dkl_bcs_seed0_pca16` | 16 (matched to the bank's PR ~15) |
| 8-15 | `dkl_bcs_seed0_pca64` | 64 (intermediate) |

Run **before** the null reports, deliberately: §34.1 says a negative null would
be uninterpretable without it, and pre-committing removes any appearance of
having chosen the control after seeing which way the result went.

Implementation note worth keeping: the job passes a `CACHE_NAME`
(`dkl_bcs_seed0_trained`) distinct from its output `MODEL_NAME`. PCA is applied
*after* caching, so the features are byte-identical to the parent run and the
provenance-stamped cache is reused — the ~25 min extraction is skipped and the
ridge over 16-64 features is cheap, so these are much shorter than their
parents. The provenance guard (§22.6) still validates the reuse, which is
exactly the case it was written for: a deliberate cache share across two model
names, checked rather than assumed.

Reading the ladder, decided now: if the hV4/LO step is present at 1000 dims and
absent at 16, capacity matters and a negative Gabor null is uninterpretable. If
the step is present at all three, capacity does not explain it and a negative
null means the front end genuinely lacks the structure.

### 26.6 Second diet model replicates (n=4, descriptive) — and a fixed display bug

`thesis_dvd_trans` (Lightning checkpoint, thesis codebase) against
`dkl_bcs_seed0_trained` (our standalone trainer). Different codebases,
different training runs, same diet:

| quantity | dkl_bcs_seed0 (n=8) | thesis_dvd_trans (n=4) |
|---|---|---|
| magno − parvo, norm | +1.534 | +1.572 (4/4) |
| magno − parvo, raw | +0.616 | **+0.615** (4/4) |
| profile hV4 − V1, norm | −0.447 | −0.489 (4/4 negative) |
| profile hV4 − V1, raw | −0.182 | **−0.183** (4/4 negative) |
| − SPI, norm | −0.128 | −0.172 (4/4 negative) |
| − SPI x predictability, norm | −0.080 | −0.121 (4/4 negative) |

The raw statistic agrees to three decimals on two independent rows. n=4 admits
no Wilcoxon (floor is p=.125 at n=4, and even that needs 4/4), so this is
descriptive — but the agreement is close enough that the §33.5 result is
clearly not a property of one training run.

**Bug fixed in `tmp_analysis/cc_panel.py`.** Its `wil()` helper returned a
hardcoded `0` for the positive count whenever n < 5, so the n=4 table first
printed "`magno - parvo +1.5715 ... 0/4 positive`" — a +1.57 mean next to a
count implying every subject was negative. The p-value withholding below n=5 is
correct and stays; only the count was wrong. No number previously recorded in
this plan came from that path (all committed tables are n>=5), but the interim
n=3/n=4 console output in this session's log should not be trusted for counts.

## §35 FIRST NULL RESULT (n=1, subject 3) — the dissociation is front-end, the
cortical step is not

`gabor_blur_cs` with `--cue-conflict`, job 26401261, subject 3 only. **n=1, no
statistic is computable, nothing here is a claim.** Recorded now because it
already separates the two §33 findings in opposite directions, and because
§32.1's outcomes were pre-declared.

### 28.1 The stream dissociation is largely an input-level fact

| | trained diet (n=8) | Gabor null (n=1) |
|---|---|---|
| magno | +0.791 | +0.604 |
| parvo | −0.750 | −0.815 |
| **magno − parvo** | **+1.534** | **+1.419** |

The Gabor bank reproduces **~93%** of the stream dissociation. Oriented energy
on blurred luminance versus red-green input is already shape-dominated in one
stream and texture-dominated in the other, with no network and no learning.

This qualifies §24.2 and §33.1 substantially, and in the direction §19 warned
about. The magno/parvo shape-texture split — the project's title claim — is
mostly a property of **what the two channels are fed**, not of what the network
learns from them. `stream_shape_texture.py`'s +1.292 (§24.7) should be re-run
on an untrained/gabor front end before leg (a) is written up as a result about
representations; if it behaves like this, leg (a) is a statement about the
input transform. That is still a real and defensible claim — it is the
biologically motivated part of the design — but it is not "training induces
specialised streams".

### 28.2 The cortical profile is NOT reproduced

| profile | V1v | V1d | V2v | V2d | V3v | V3d | hV4 | LO | hV4−V1 | rho |
|---|---|---|---|---|---|---|---|---|---|---|
| trained (n=8) | −0.055 | +0.048 | −0.046 | −0.033 | −0.227 | −0.148 | **−0.471** | −0.466 | −0.447 | −0.59 |
| Gabor (n=1) | −0.781 | −0.810 | −0.711 | −0.733 | −0.752 | −0.696 | **−0.809** | −0.689 | **−0.014** | **+0.25** |

The null is **flat**: uniformly texture-dominated at about −0.75 everywhere,
with hV4 − V1 = −0.014 against the trained model's −0.447, and monotonicity of
the *opposite* sign. After residualisation the null's residual hV4 − V1 is
**+0.070** with rho +0.86/+0.93 — it tilts the other way.

This is §32.1's second pre-declared outcome. The hV4/LO texture step is not
produced by the front end.

### 28.3 Why this is NOT yet "the network builds it"

§34.1 pre-committed the reason: the bank has ~15 effective dimensions against
the network's ~270/160, so a *failure* to reproduce fine-grained voxel
structure is exactly what a capacity limit also predicts. The null's flatness
is equally consistent with "the front end lacks the structure" and "a
15-dimensional model cannot express any ROI structure at all" — and the fact
that it is flat at a *constant* −0.75, rather than noisy, is mildly suggestive
of the latter.

**The capacity ladder (26402571, `dkl_bcs_seed0_pca16/pca64`) decides this**,
by the rule written down in §34.3 before either result existed: step present at
16 dims -> capacity is not the explanation and §35.2 stands; step absent at 16
dims -> §35.2 is uninterpretable and the honest report is that the comparison
cannot separate the two.

### 28.4 Status

n=1 of 8 for the null. Seven subjects outstanding, plus the whole no-diet
column and the capacity ladder. Do not quote §35 outside this file yet.

### 28.5 Correcting §35.1 — the front end sets MAGNO, learning sets PARVO

§35.1 said the dissociation is "mostly a property of what the two channels are
fed, not of what the network learns", and proposed re-running
`stream_shape_texture.py` on an untrained front end. That re-run was
unnecessary — `results/shape_texture/untrained_*.json` already exist — and the
existing numbers show the summary was too coarse:

| model | magno bias | parvo bias | M − P |
|---|---|---|---|
| `untrained_blur_cs` (front end only) | **+0.796** | −0.087 | +0.882 |
| `dkl_bcs_seed0_trained` (diet, trained) | **+0.810** | **−0.482** | +1.292 |
| `untrained_noblur` | +0.417 | −0.087 | +0.503 |
| `dkl_base_seed0_trained` (no diet, trained) | **−0.219** | −0.519 | +0.300 |
| `rgb_tsn_seed*_trained` (architecture only) | −0.141 | −0.141 | **+0.000** |

Decomposed per stream, the picture is not "mostly front-end" — it is a clean
division of labour:

- **Magno's shape bias is entirely the front end.** +0.796 untrained ->
  +0.810 trained: learning adds **+0.014**. The blur + CS cut does all of it.
- **Parvo's texture specialisation is entirely learning.** −0.087 untrained ->
  −0.482 trained: learning adds **−0.395**, and the untrained parvo stream is
  barely biased at all.

So the +0.882 -> +1.292 gain is not a uniform 68/32 split; the two streams
arrive at their specialisations by different routes. This *is* §12's finding
("the diet protects magno; training specialises parvo"), now quantified per
stream, and §35.1 should not have restated it as a single ratio.

**And training without the diet reverses magno outright**: +0.417 untrained ->
**−0.219** trained. The unblurred magno stream does not merely fail to become
shape-biased, it becomes *texture*-biased, more so than its own parvo partner
is in relative terms. That is §15's mechanism visible in the probe.

`rgb_tsn` returning exactly +0.000 is a good architecture-only control: with
identical input to both streams, two-stream architecture alone buys nothing.

**Revised statement for the manuscript**: the input split is necessary and
sufficient for magno's shape bias; training is necessary for parvo's texture
specialisation; and without the diet, training actively destroys the shape
bias the architecture would otherwise inherit. The Gabor null (§35.1) is
consistent with this — it has a front end and no learning, and it lands near
the untrained network.

## §36 The untrained network has ~4 effective dimensions — training raises PR ~70x

Checked while considering an untrained-network cue-conflict column as a
capacity-matched alternative to the Gabor null (§34.1). The check killed that
plan and produced a more useful number. Participation ratio, 2000 NSD images,
subject 1:

| model | magno PR | parvo PR |
|---|---|---|
| `untrained_seed0` (blur+CS front end) | **3.7** | **2.8** |
| `untrained_seed0_noblur` | 4.3 | 2.8 |
| `gabor_blur_cs` | 15.3 | 11.8 |
| `dkl_bcs_seed0_trained` | **269.9** | **160.1** |

A random ResNet with 15360 output dimensions carries roughly **four**
effective dimensions of variance across natural images — *fewer than the Gabor
bank*. Training multiplies that by ~70x.

### 29.1 Consequences, several of them uncomfortable

1. **An untrained cue-conflict column would not have been a capacity-matched
   control.** It is even lower-dimensional than the bank. Not run; the pca
   ladder (§34.3) remains the right instrument, and its 16-dim rung is a
   reasonable match to the bank's 15.3 while still 4x the untrained network's.
2. **§19 now has a mechanism.** "The untrained ResNet predicts BOLD no better
   than a Gabor bank" (0.1384 vs 0.1408) is what a 4-dimensional feature space
   should do. §19.3's relabelling of the untrained panels as front-end
   validations is not just defensible, it is explained.
3. **The capacity confound is pervasive, not local to §35.** Every null this
   project uses — untrained network, Gabor bank — is low-dimensional, and every
   trained model is high-dimensional. Any trained-vs-null comparison in §14.5,
   §19, §33 and §35 is confounded to some degree by effective dimensionality.
   This should be stated once, prominently, in the manuscript's methods rather
   than defended piecemeal per figure.
4. **It sharpens §7.2 item 6.** The "general factor" the thesis's RSA lost to is
   not a subtlety in the untrained network — with PR 3.7 the untrained network
   very nearly *is* a general factor.

### 29.2 What it does NOT undermine

The trained-vs-trained contrasts are unaffected: `dkl_bcs` vs `dkl_base` vs
`thesis_dvd_trans` vs `swap` are all trained networks and there is no reason to
expect the 70x gap between them. §15's headline (diet vs no-diet, both trained)
and §33.6's two-diet-model replication do not rest on a trained-vs-null
comparison. The PR of `dkl_base_seed0` should be measured on a *repaired*
subject to confirm this (its current cache is the §22-contaminated one, §34.2),
which tasks 16-23 of 26399760 will provide.

### 29.3 Action

- Do not run an untrained cue-conflict column. (Decision recorded so it is not
  re-proposed.)
- Measure PR for every model in the paper once the repairs land, and report the
  table in supplementary methods.
- The §35.3 reading rule stands unchanged: the pca ladder decides whether the
  §35.2 flatness is a front-end fact or a capacity artefact.

### 29.4 Capacity ladder, first subject (PRELIMINARY, n=1 — subject 4)

Same subject, three capacities, so nothing is confounded by subject:

| model | median r2_joint_nc | magno − parvo | hV4 − V1 raw | − SPI x pred | − SPI x pred x ecc |
|---|---|---|---|---|---|
| `dkl_bcs_seed0_pca16` | 0.337 | +1.528 | **−0.600** | −0.016 | −0.059 |
| `dkl_bcs_seed0_trained` (1000) | 0.423 | +1.537 | **−0.591** | −0.072 | −0.096 |
| `gabor_blur_cs` (PR 15.3) | — | +1.419 | **−0.014** | — | — |

**The gross hV4/LO step is capacity-independent**: −0.600 at 16 PCA dimensions
against −0.591 at 1000. A 16-dimensional *trained* model produces it; a
15-dimensional *Gabor* model does not (−0.014). By the rule pre-committed in
§34.3, the §35.2 flatness of the null is therefore a front-end fact, not a
capacity artefact — the trained model does not need 270 dimensions to show the
step.

**The residualised step is partly capacity-dependent**: −0.096 -> −0.059 with
the full control set (~60% retained), and −0.072 -> −0.016 without the
eccentricity factor. So the component that survives band preference, fit
quality and retinotopy does shrink when capacity is cut, and the manuscript
should not claim that part is capacity-free.

**n = 1.** Seven pca16 subjects, the pca64 rung and seven null subjects are
outstanding. Do not quote §36.4 until the ladder is complete; recorded now only
so the reading is fixed before the rest arrives.

## §37 Replication at n=7 and the null at n=3 — the separation is now large

### 30.1 `thesis_dvd_trans` replicates `dkl_bcs_seed0` on both statistics

n=7 (subject 7 outstanding), floor p=.0156. Against §33.5's n=8 diet cell:

| quantity | dkl_bcs_seed0 (n=8) | thesis_dvd_trans (n=7) | p (thesis) |
|---|---|---|---|
| magno − parvo, norm | +1.534 | +1.579 (7/7) | .0156 |
| magno − parvo, raw | +0.616 | +0.635 (7/7) | .0156 |
| profile hV4 − V1, norm | −0.447 | −0.474 (7/7) | .0156 |
| profile hV4 − V1, raw | −0.182 | −0.189 (7/7) | .0156 |
| − SPI, norm | −0.128 | −0.178 (6/7) | .0312 |
| − SPI x predictability, norm | −0.080 | −0.116 (7/7) | .0156 |
| − SPI x predictability, raw | −0.050 | −0.058 (6/7) | .0312 |

Every row significant on both statistics, in two networks trained in different
codebases. §33.5 is not a property of one training run.

### 30.2 The null is not flat — it goes the OTHER way, consistently

`gabor_blur_cs`, n=3 (subjects 2, 3, 4). No p is computable, but the
*consistency* is informative on its own:

| quantity | trained diet | Gabor null (n=3) |
|---|---|---|
| magno − parvo, norm | +1.53 / +1.58 | **+1.438 (3/3)** |
| profile hV4 − V1, norm | −0.447 / −0.474 | **−0.003** (1/3 pos) |
| profile hV4 − V1, raw | −0.182 / −0.189 | **−0.013** |
| monotonicity rho | −0.59 / −0.50 | **+0.20 / +0.25** |
| residual − SPI, norm | −0.128 / −0.178 | **+0.057 (3/3 positive)** |
| residual − SPI x pred, norm | −0.080 / −0.116 | **+0.034 (2/3)** |

The gross step is ~150x smaller in the null (−0.003 vs −0.45) and the
monotonicity has the opposite sign.

### 30.3 An argument against the capacity explanation that does not need the ladder

§34.1/§35.3 worried the null's flatness might be a capacity artefact: a
15-dimensional model may be unable to express *any* ROI structure. But a
capacity-limited model should produce **noise** — near-zero effects with
random signs across subjects. Instead the null's residualised step is
**+0.057 with 3/3 subjects positive**, and its monotonicity is +0.65 with 3/3
positive. It is consistently structured, just in the opposite direction.

A model that reliably expresses an ROI gradient of the *opposite* sign is not
a model that cannot express ROI gradients. This is independent evidence that
§35.2 reflects the front end rather than capacity, and it agrees with the
ladder's first subject (§36.4), where the trained model kept its step at 16
dimensions.

Both arguments remain n<5 and neither is yet a claim. But they are now two
independent lines pointing the same way, which is worth more than either alone.

### 30.4 Outstanding

thesis subject 7; five more null subjects; the pca16 rung beyond subject 4; the
whole pca64 rung; and the entire `dkl_base_seed0` no-diet column, which is the
one that separates "the network builds it" from "the *diet* builds it".

### 37.5 Numbering collision resolved (2026-09-01)

§26.1 (other session) flagged duplicate §21/§22 and renumbered its own newest
section. Since then the collision had grown to **five duplicated numbers** —
§21, §25, §26, §27 and §28 each existed twice — because both sessions kept
appending.

Resolved by following §26.1's stated convention: **renumber only your own
headers, never the other session's.** This session's sections moved

    §21 → §31    §25 → §32    §26 → §33    §27 → §34
    §28 → §35    §29 → §36    §30 → §37

and every `§N.x` cross-reference *inside this session's own line ranges* was
remapped with them. The other session's sections — §21 (swap probe), §25
(`parvo_highsf`), §26 (numbering/Figure 1), §27 (`rgb_tsn`), §28 (architecture
symmetry) — are **byte-identical**, verified line by line against a pre-edit
copy: 49 lines changed, all inside this session's ranges, none outside.

Scoping mattered and was not cosmetic: §26.2 cites "§21.4" meaning the *other*
session's §21, so a document-wide find-and-replace would have silently
redirected their reference. Anyone repeating this exercise must scope by line
range, not by pattern.

§29 and §30 are now unused; the gap is deliberate and harmless.

**Convention going forward, for both sessions**: before appending a numbered
section, run

    grep -n "^## §\?[0-9]" docs/PAPER_PLAN.md | tail -5

and take the next free number. Prefer citing sections by **title** as well as
number — §26.1's own advice, and the only thing that survives a renumber.

## §38 BOTH DIET MODELS FINAL AT n=8 — the §7.3b diet cell is closed

`thesis_dvd_trans` completed (job 26399760 tasks 8-15). Both diet models now
have all 8 subjects. Floor p=.0078.

| quantity | `dkl_bcs_seed0` | `thesis_dvd_trans` | p (both) |
|---|---|---|---|
| magno − parvo, norm | +1.534 (8/8) | +1.574 (8/8) | .0078 |
| magno − parvo, raw | +0.616 (8/8) | +0.631 (8/8) | .0078 |
| profile hV4 − V1, norm | −0.447 (8/8) | −0.449 (8/8) | .0078 |
| profile hV4 − V1, raw | **−0.182** (8/8) | **−0.181** (8/8) | .0078 |
| − SPI, norm | −0.128 | −0.173 (7/8) | .0391 / .0156 |
| − SPI x predictability, norm | −0.080 | −0.112 (8/8) | .0391 / .0078 |
| − SPI x predictability x ecc | −0.095 (8/8) | −0.103 (8/8) | .0078 / .0078 |

Two networks, two codebases (standalone trainer vs thesis Lightning
checkpoint), two independently initialised training runs. The uncorrected raw
profile agrees to three decimals. Every row significant on both statistics in
both models.

Eccentricity gradient, `thesis_dvd_trans`: −0.219, −0.209, −0.121, +0.143,
**+0.404** — matching `dkl_bcs_seed0`'s −0.212 … +0.325 in shape and sign.

Monotonicity after residualisation is again ~0 in both (−0.067/+0.027 norm,
−0.067/−0.036 raw, 2-4 of 8 positive), confirming §33.2: **a step at hV4/LO,
not a gradient.**

### 38.1 What is now established, and what is not

**Established** (diet cell, two models, n=8, both statistics, three controls):
a texture-dominated readout localised to hV4 and LO, not reducible to band
preference, fit quality or retinotopic eccentricity, replicating across
training runs and codebases.

**Not established.** Whether it is a fact about *the network* rather than the
front end rests on the Gabor null (n=4 of 8, §37.2 shows it flat-to-opposite)
and the capacity ladder (n=2 of 8+8, §36.4 shows the step surviving at 16
dims). Whether it is a fact about *the diet* rather than training in general
rests entirely on the `dkl_base_seed0` column, whose tasks (16-23) are now
running. Neither is a claim yet.

## §39 The capacity objection is answered (paired ladder, PRELIMINARY n=3/n=4)

`tmp_analysis/cc_ladder.py`, paired within subject against the
full-dimensional `dkl_bcs_seed0_trained` fit, so between-subject variance —
the dominant term — is removed.

| rung | n | median fit | uncorrected step | − SPI x pred | − SPI x pred x ecc |
|---|---|---|---|---|---|
| `dkl_bcs_seed0_pca16` | 3 | 0.303 | **100.3% retained** | 69.0% | 63.4% |
| `gabor_blur_cs` | 4 | 0.137 | **8.4% retained** | −3.0% | 10.7% |

A **16-dimensional trained model keeps the entire gross step** (−0.5184 against
its own reference −0.5166). A **~15-dimensional Gabor model keeps 8%**
(−0.0414 against −0.4943). Same dimension count, opposite result.

This satisfies §36.4's pre-committed rule — "step present at 16 dims ->
capacity is not the explanation and §35.2 stands" — and it converges with
§37.3's independent argument (the null is *consistently* oppositely
structured, which noise from a capacity ceiling would not be).

### 39.1 Precision about what has been refuted

The objection refuted is specifically **dimension count**: "a ~15-dimensional
model cannot express ROI structure" is false, because a 16-dimensional trained
model expresses it in full.

What is *not* refuted, and should not be claimed: the two rungs are not matched
on **information content**. PCA-16 of a trained network keeps its top sixteen
principal components — highly informative, learned directions — while the
Gabor bank's ~15 effective dimensions are whatever oriented energy provides.
The pca16 rung's fit quality is 0.303 against the bank's 0.137, so it is a
materially better model at the same dimensionality.

But that difference *is* the front end plus learning, which is the thing under
test. "The Gabor features carry less of the relevant information" is not a
competing explanation for §35.2 — it is a restatement of it. The capacity
objection mattered only because it offered a *methodological* alternative
(a ceiling artefact) to a *scientific* conclusion; that alternative is closed.

### 39.2 Status

n=3 on the pca16 rung and n=4 on the null, so no p-value is computable and
this is a read, not a result. The pca64 rung has not started. Recorded now
because the retention percentages are large enough (100% vs 8%) that the
direction is unlikely to reverse, and because fixing the interpretation before
the remaining subjects arrive is the point of §36.4's pre-commitment.

### 39.3 n=5 update — the capacity argument holds for the GROSS step only

Two more pca16 subjects landed. The paired retention percentages moved, and
not uniformly:

| measure | n=3 (subj 3,4,6) | **n=5 (subj 2,3,4,5,6)** | Gabor (n=4) | diff vs full, p |
|---|---|---|---|---|
| hV4−V1 uncorrected | 100.3% | **93.0%** | 8.4% | .4375 |
| hV4−V1 − SPI x pred | 69.0% | **52.8%** | −3.0% | .1250 |
| hV4−V1 − SPI x pred x ecc | 63.4% | **27.5%** | 10.7% | .1250 |

(The reference values change with the paired subject subset, so the drift is
subject sampling, not a computation change.)

**What still holds.** The gross step is capacity-independent: 93% retained at
16 dimensions, with no detectable difference from the full fit (p=.4375),
against the Gabor bank's 8.4% at comparable dimensionality. §39.1's refutation
of the *dimension-count* objection stands on this row, and it is the row with
the largest separation by an order of magnitude.

**What weakens.** The fully-controlled residual step retains only **27.5%** at
16 dimensions, down from 63.4% at n=3, against the bank's 10.7%. That
separation is no longer an order of magnitude and both values are small. So
for the residualised effect — which §33 and §38 identified as the *real*
§7.3b contribution — capacity cannot be dismissed as an explanation on this
evidence.

**Revised claim.** Say: the hV4/LO texture step is not an artefact of feature
dimensionality (gross step, 93% vs 8%). Do **not** say the residualised
component is capacity-free; at 16 dimensions it is substantially attenuated,
and the honest statement is that the residualised effect requires
representational capacity the Gabor bank does not have — which is a weaker and
more careful claim than §39 originally implied.

§37.3's independent argument is unaffected: the null's *consistent opposite
structure* (3/3 subjects, rho +0.65) is not what a capacity ceiling produces,
and that argument never relied on the ladder.

n=5 of 8 on this rung and n=4 of 8 on the null; the pca64 rung would locate the
attenuation curve between 16 and 1000 dimensions and has not started.

## §40 §22 CLOSED — the no-diet cell repaired, and the corrected numbers

`dkl_base_seed0_trained` subject 1 was refit with the correct front end
(26399760 task 16), verified, and installed over the contaminated file by
`tmp_analysis/apply_repair.py`. The model is back to n=8.

### 40.1 The corrected headline

| version | raw | norm | n |
|---|---|---|---|
| contaminated (withdrawn) | −0.0452 (p=.039) | **−0.1669** (p=.016) | 8 |
| interim, subject 1 dropped | −0.0281 (p=.078) | −0.0721 (p=.031) | 7 |
| **repaired** | **−0.0342** (p=.039) | **−0.1013** (p=.016) | **8** |
| clean `dkl_base_seed1` | −0.0406 (p=.016) | −0.1318 (p=.008) | 8 |

The repaired seed0 and the never-contaminated seed1 now agree on both
statistics — `model_verdict.py` reports "same in both", A−B p=.383 (raw) and
p=.547 (norm). Quote **−0.1013 (norm) / −0.0342 (raw)** for
`dkl_base_seed0_trained` from here on. The interim n=7 figures in §22.4 are
superseded and should not be quoted either.

### 40.2 §22.5's prediction was correct, and the gap widened

The diet-trained cell is −0.859. Against a *contaminated* no-diet value of
−0.167 the ratio was 5.1x; against the **repaired** −0.101 it is **8.5x**. The
contamination had made the no-diet model look more like the diet model, so
removing it widens the contrast §15/§17 is built on, exactly as §22.5
predicted. This is a correction that strengthens the paper's central claim.

Figure 1 must be regenerated with −0.1013 (the other session owns this, §26.2);
its withdrawn headline of −0.149 averaged the contaminated seed0 with seed1.
The correct average of the two clean seeds is **−0.1166**.

### 40.3 The verification guard blocked first, and was itself wrong

Worth recording because the lesson generalises. `apply_repair.py` refused to
install: the repaired value (−0.3053 on the hV4−V1 SPI contrast used for the
check) fell outside the range of its seven siblings.

Investigation showed the **guard** was miscalibrated, not the data. Subject 1
is *systematically* the most extreme subject for the no-diet model — in the
clean, never-contaminated `dkl_base_seed1` it also ranks 1/8 (−0.2234), while
in the diet model it ranks only 4/8. A sibling-range test therefore rejects a
legitimate value for this subject and model combination.

Recalibrated to the principled comparison — the same subject in the clean seed,
with tolerance set by the observed seed-to-seed gap on subjects 2-8:

- repaired −0.3053 vs clean seed1 −0.2234: gap **0.0819**, inside the maximum
  seed-to-seed gap of **0.1365**;
- the repair moved subject 1 from **0.6071** away from the clean seed to
  **0.0819** away, a 7.4x improvement.

Both checks pass and the install proceeded. The general point: a guard that
blocks is doing its job even when it turns out to be wrong, because the cost of
investigating a false block is small and the cost of a silent bad install is
what produced §22 in the first place.

### 39.4 §39.3 over-read an unstable estimate — hold the ladder until n=8

n=6 on the pca16 rung:

| measure | n=3 | n=5 | n=6 | diff vs full, p (n=6) |
|---|---|---|---|---|
| hV4−V1 uncorrected | 100.3% | 93.0% | **95.6%** | .6875 |
| − SPI x pred | 69.0% | 52.8% | **76.5%** | .4375 |
| − SPI x pred x ecc | 63.4% | 27.5% | **47.1%** | .0625 |

The fully-controlled row has now read 63%, 28% and 47% at three consecutive
sample sizes. That is not a trend, it is an unstable estimate — the paired
subject subset changes with each arrival and the residualised quantity is small
enough that one subject moves it substantially.

**§39.3 was therefore an over-read.** Its conclusion ("the capacity argument
weakens sharply for the residualised step") was drawn from the n=5 value, which
the n=6 value does not support. The correct statement at present is that the
residualised retention **cannot be estimated** from n<=6 with useful precision.
§39.3's revised *wording* remains the safe one to use — do not claim the
residualised component is capacity-free — but its stated reason (a sharp
weakening) is not established.

What is stable across all three sample sizes is the **uncorrected** row: 100%,
93%, 96%, never significantly different from the full fit, against the Gabor
bank's 8.4%. §39.1's refutation of the dimension-count objection rests on that
row and is unaffected.

**Process note.** This session has been re-running the ladder at each arrival
and writing up each result. That produced two corrections (§39.3, §39.4) which
would both have been avoided by waiting for the complete rung. Hold the ladder
until n=8 on the pca16 arm; the same applies to the Gabor null (n=4) and the
no-diet column (n=1).

## §41 Repair confirmed independently by participation ratio; and the diet
raises magno's effective dimensionality

### 41.1 Fourth independent confirmation of the §22 repair

§36.2 predicted the contaminated subject's magno/parvo PR asymmetry should
disappear once refit. Measured on the repaired cache (2000 NSD images):

| model / subject | magno PR | parvo PR | ratio |
|---|---|---|---|
| `dkl_base_seed0` s1, **repaired** | **173.4** | 189.5 | **0.92** |
| `dkl_base_seed0` s1, *contaminated* | *32.2* | *189.5* | *0.17* |
| `dkl_base_seed0` s2, never contaminated | 175.0 | 187.5 | 0.93 |
| `dkl_base_seed1` s1, independent clean seed | 175.6 | 183.2 | 0.96 |

Magno PR moved 32.2 -> 173.4 and now agrees with its own sibling (175.0) and
with the same subject in a different seed (175.6) to within 2%. This is the
fourth independent line confirming §22, after the exact feature-identity match
(§22.2), the outlier analysis (§22.4) and the seed-agreement check in §40.3 —
and unlike those it is computed from the feature cache, not from any fit.

Note `dkl_base_seed1`'s cache is still `UNSTAMPED` (written before the §22.6
guard). The sacct audit in §22.3 confirms its flags were correct on all eight
tasks and its PR agrees, so it is trusted; it will acquire a stamp whenever it
is next regenerated.

### 41.2 New: the diet makes the magno stream richer, not poorer

| model | magno PR | parvo PR | magno/parvo |
|---|---|---|---|
| `dkl_bcs_seed0` (diet) | **269.9** | 160.1 | **1.69** |
| `dkl_base_seed0` (no diet) | 173.4 | 189.5 | 0.92 |

Without the diet the two streams end up at comparable effective dimensionality
(ratio 0.92). With it, magno ends up **56% richer than parvo** and 56% richer
than its own no-diet counterpart (269.9 vs 173.4).

This is counterintuitive and worth stating plainly: the diet *removes*
information from the magno input — blur plus a 48-month CSF cut — and the
network nonetheless learns a **higher**-dimensional magno representation from
it. The natural reading is that without the diet the two streams converge on
similar solutions and duplicate each other, while the low-pass constraint
forces them apart; but that is an interpretation, and the measured fact is the
PR asymmetry.

It sits alongside §36.6's finding that the diet's front end *lowers* the
input-level cortical gradient while the trained model overtakes anyway. Both
say the same thing in different currencies: the diet's value is not that it
hands the network better inputs.

Measured on subject 1 only. Extend to all 8 subjects and to
`thesis_dvd_trans` / `swap` before this goes in the manuscript.

### 41.3 n=8 correction to §41.2 — the front end sets the asymmetry, no-diet
training destroys it

§41.2 said "the diet makes the magno stream richer", from subject 1 alone. At
n=8 across five models (`tmp_analysis/pr_table.py`, 1000 images, Gram-side
eigendecomposition) that is the wrong causal reading:

| model | magno PR | parvo PR | magno/parvo |
|---|---|---|---|
| `untrained_seed0` (front end, no learning) | 3.7 ±0.1 | 2.6 ±0.1 | **1.40 ±0.09** |
| `gabor_blur_cs` (input only) | 15.5 ±0.5 | 11.3 ±0.7 | **1.38 ±0.09** |
| `dkl_bcs_seed0` (diet, trained) | 233.2 ±2.4 | 145.5 ±2.0 | **1.60 ±0.03** |
| `thesis_dvd_trans` (diet, trained) | 205.9 ±3.8 | 177.9 ±2.4 | **1.16 ±0.03** |
| `dkl_base_seed0` (no diet, trained) | 158.9 ±1.3 | 168.0 ±5.5 | **0.95 ±0.04** |

Diet vs no-diet ratio: 1.60 vs 0.95, **paired p=.0078, 8/8 subjects**.

**The asymmetry is already in the front end.** Untrained 1.40 and Gabor 1.38 —
magno is the more dimensional stream before any learning. The diet does not
create it (1.60 is a modest increase on 1.40); **no-diet training abolishes it**
(0.95). This is structurally identical to §28.5's finding for shape bias: the
front end sets the property, diet-training preserves it, no-diet training
destroys it. §41.2's "the diet makes magno richer" should be replaced with
"the diet preserves an asymmetry the input transform creates and that training
otherwise erases".

The §41.2 paradox dissolves with it: the diet is not conjuring dimensionality
out of blurred input. Blurred luminance is *already* the more dimensional
channel relative to red-green, and the interesting fact is that unconstrained
training collapses that difference.

**Caveat — the two diet models disagree here (1.60 vs 1.16).** Every §7.3b
cortical measure matched between them to three decimals (§38), but this one
does not. PR is evidently more sensitive to training details (optimiser,
schedule, codebase) than the cortical measures are. Report the range, not a
point estimate, and do not treat 1.60 as "the" diet value.

Absolute PRs are also worth stating: training raises magno PR from 3.7 to
159-233, a 43-63x increase, which is the §36 finding at n=8.

## §42 CAPACITY LADDER COMPLETE (pca16, n=8) — the objection is closed, and
§39.3 was noise

Paired within subject against the full-dimensional fit, all 8 subjects:

| measure | pca16 retained | diff vs full, p | Gabor (n=4) |
|---|---|---|---|
| hV4−V1 uncorrected | **101.2%** | .7422 | 8.4% |
| hV4−V1 − SPI x pred | **106.3%** | .9453 | −3.0% |
| hV4−V1 − SPI x pred x ecc | **65.1%** | .0547 | 10.7% |

At sixteen PCA dimensions the trained model retains **the entire step** on two
of three measures — statistically indistinguishable from its own
full-dimensional fit — and 65% on the strictest, at p=.055.

### 42.1 The trajectory, and the lesson

The fully-controlled row read 63.4% (n=3), 27.5% (n=5), 47.1% (n=6), **65.1%
(n=8)**. §39.3 treated the n=5 dip as a finding and wrote a correction around
it; §39.4 caught that and imposed a hold; the completed rung confirms §39.4.
**§39.3's stated reason is withdrawn** — there is no sharp weakening. Its
cautious wording was harmless, but it was reasoning from noise.

The lesson is now recorded twice in this document and should be followed:
do not analyse a partially-filled cell. Every one of this session's
self-corrections on the ladder (§39.3, §39.4, this section) traces to reading
an incomplete rung.

### 42.2 What is now established about capacity

A 16-dimensional trained model reproduces the hV4/LO step in full; a
~15-dimensional Gabor bank reproduces 8% of it. The capacity objection raised
in §34.1 — that the null's flatness might be a ceiling artefact rather than a
front-end fact — is **closed**. The null is flat because the front end lacks
the structure, not because a low-dimensional model cannot express it.

This converges with §37.3's independent argument (the null is *consistently*
oppositely structured, 3/3 subjects, rho +0.65, which a ceiling does not
produce). Two unrelated lines, same conclusion.

Remaining caveat, unchanged from §39.1: pca16 and the Gabor bank are matched on
dimension count but not on information content, and that difference *is* the
front end plus learning, which is the thing under test.

### 42.3 Outstanding

The Gabor null is at n=4 of 8 and the no-diet column at n=4 of 8; neither is
yet testable. The pca64 rung has not started. §42 concerns capacity only — it
says nothing about whether the effect is due to the diet, which only the
no-diet column can settle.

## §43 THE DECISIVE CONTRAST — the diet owns the stream dissociation, the
NETWORK owns the cortical step

`dkl_bcs_seed0_trained` (diet) vs `dkl_base_seed0_trained` (no diet), paired
within subject, n=8, both statistics. This is the comparison §32.2 identified
as the one that separates "the network builds it" from "the diet builds it".

| measure | diet | no diet | A−B | p (norm/raw) | verdict |
|---|---|---|---|---|---|
| median r2/NC | +0.405 | **+0.464** | −0.059 | .0078 / .0078 | **no-diet fits BETTER** |
| magno stream | +0.791 | **−0.638** | +1.429 | .0078 / .0078 | diet, 8/8 |
| parvo stream | −0.743 | −0.748 | +0.005 | .6406 / .0078 | **UNRESOLVED** |
| **magno − parvo** | **+1.534** | **+0.110** | +1.424 | .0078 / .0078 | **diet, 8/8** |
| hV4−V1 uncorrected | −0.447 | −0.202 | −0.245 | .0078 / .0234 | diet larger |
| hV4−V1 − SPI x pred | −0.080 | **−0.159** | +0.079 | .0391 / .0234 | **no-diet larger** |
| **hV4−V1 − SPI x pred x ecc** | −0.095 | −0.060 | −0.035 | **.1484 / .9453** | **NO DIFFERENCE** |

### 43.1 The diet owns the stream dissociation, decisively

magno−parvo is **+1.534 with the diet and +0.110 without** — the no-diet
network has essentially no stream dissociation at all. Driven entirely by the
magno stream (+0.791 vs **−0.638**: without the diet the magno stream is
*texture*-dominated), while the parvo streams are indistinguishable (−0.743 vs
−0.748). 8/8 subjects, both statistics, p=.0078.

This confirms §28.5's probe finding inside the encoding model, on independent
machinery. It is the strongest single result in the §7.3b set.

### 43.2 The cortical step is NOT diet-specific — this is a reframing

Under the full control set the hV4/LO step is **−0.095 with the diet and
−0.060 without, p=.1484 (norm) and p=.9453 (raw)**. Both statistics agree there
is no difference, so by §8 the answer is "no difference", not "a trend".

The uncorrected profile *does* look diet-specific (−0.447 vs −0.202), but that
gap is carried by band weighting — which is itself diet-dependent (§43.1) — and
it disappears once band preference, fit quality and eccentricity are matched.

**Consequence.** §35.2/§38's cortical result must be reported as a property of
**training a two-stream network on this input split**, not of the developmental
diet. §32.2's pre-declaration anticipated exactly this: "a flat null licenses
'the network builds it', not 'the diet builds it'." The null and ladder
established the former; this contrast rules out the latter.

The intermediate control (− SPI x pred) actually shows the **no-diet** model
with the *larger* step (−0.159 vs −0.080, p=.039/.023, both statistics). That
reverses once eccentricity is matched, so it is not claimed either way, but it
is further evidence against a diet-specific cortical effect.

### 43.3 The no-diet model predicts BOLD better

median R2/NC **0.464 without the diet against 0.405 with it**, p=.0078 on both
statistics, 8/8 subjects. The diet costs ~13% of explained variance.

This must be stated plainly in the manuscript. It is consistent with the
existing "fit quality dissociates from regional structure" result (commit
6a7543f) and with §36's PR finding, but it is a real cost and a reviewer will
find it immediately. The diet's case cannot rest on prediction accuracy; it
rests on the stream dissociation (§43.1) and on the §15/§17 gradient.

### 43.4 One unresolved cell

The parvo stream disagrees between statistics: norm says no difference
(+0.005, p=.6406) while raw says the diet's parvo is *less* texture-dominated
(−0.032, p=.0078, 0/8). Recorded as UNRESOLVED per §8. It does not affect
§43.1, which is carried by magno and is unambiguous on both statistics.

## §44 ALL CELLS COMPLETE — final §7.3b verdict, and it is mixed

`gabor_blur_cs` and `dkl_bcs_seed0_pca64` both reached n=8. Every cell of the
§7.3b design is now full: two diet models, one no-diet model, the input null,
and two capacity rungs, all at n=8.

### 44.1 Diet vs the input null, n=8 paired, both statistics

| measure | diet | Gabor null | p (norm) | p (raw) | §8 verdict |
|---|---|---|---|---|---|
| median r2/NC | +0.405 | +0.132 | .0078 | .0078 | **trained fits 3x better** |
| magno − parvo | +1.534 | +1.413 | .0078 | .0078 | trained slightly larger |
| **hV4−V1 uncorrected** | −0.447 | −0.017 | **.0078** | **.0078** | **ESTABLISHED** |
| hV4−V1 − SPI x pred | −0.080 | +0.015 | .0078 | **.1953** | **UNRESOLVED** |
| hV4−V1 − SPI x pred x ecc | −0.095 | +0.005 | .0078 | **.1953** | **UNRESOLVED** |

The gross profile difference is established on both statistics and is large
(−0.447 vs −0.017, 0/8 subjects overlapping). **But the residualised
comparisons — the ones §33 and §38 identified as the real contribution — are
UNRESOLVED.** The norm statistic says the null is clearly different; the raw
statistic says it is not (−0.050 vs −0.024, p=.1953). Per §8 neither may be
claimed.

### 44.2 The capacity ladder, complete

| rung | uncorrected | − SPI x pred | − SPI x pred x ecc |
|---|---|---|---|
| pca16 | 101.2% (p=.74) | 106.3% (p=.95) | 65.1% (p=.055) |
| pca64 | 84.0% (p=.0078) | 90.6% (p=.95) | 63.4% (p=.023) |
| Gabor | **3.8%** (p=.0078) | **−18.9%** (p=.0078) | **−5.2%** (p=.0078) |

Note the ladder tests the *norm* statistic only, so its p-values do not
discharge §8; §44.1 is the authority. Retention is non-monotonic between the
rungs (pca16 101% vs pca64 84% on the uncorrected row), which is a further sign
that these retention estimates carry more noise than their tight paired
p-values suggest. Do not build an argument on the rung ordering.

### 44.3 The three findings, ranked by what survives

1. **The diet determines the stream dissociation.** +1.534 vs +0.110 against
   no-diet (§43.1), 8/8, both statistics, p=.0078, driven entirely by magno
   (+0.791 vs −0.638). Replicated in a second diet model and in an entirely
   separate probe (§28.5). **This is the surviving positive claim.**
2. **The gross hV4/LO profile is a training effect, not a diet effect and not
   a front-end effect.** Trained > null on both statistics (§44.1); but diet
   and no-diet do not differ once controls are matched (§43.2). Report as a
   property of training a two-stream network on this input split.
3. **The residualised hV4/LO step does not survive the controls it needs.**
   Diet vs no-diet: no difference (§43.2, both statistics). Diet vs null:
   unresolved (§44.1). It cannot carry a claim.

### 44.4 Reframing, per the project's own standard

The research standard says a result a control kills is dead, and the fix for
messy results is reframing rather than more compute. Applying it:

- **Do not** write §7.3b around the cortical profile. §33/§38 identified the
  residualised step as the contribution; §43.2 and §44.1 have removed its
  support from both directions.
- **Do** write §7.3b around §44.3 item 1, with the encoding model as
  independent confirmation of `stream_shape_texture.py` — which per §31.7 was
  always the stronger instrument for that claim.
- **Do** report item 2 honestly as a training effect, and report the
  fit-quality cost (§43.3: the diet loses ~13% of explained variance) in the
  same breath.

What §7.3b bought, stated plainly: it did not produce the flagship cortical
panel it was commissioned for. It produced a rigorous negative — the cortical
step is not diet-specific — plus independent confirmation of the stream
dissociation, plus three methodological artefacts of lasting value (the
provenance guard §22.6, the PR table §41.3, and the §22 repair). That is a
smaller result than §31 hoped for and it is the one the data supports.

---

## §29 Figure 1 regenerated — the quarantine is lifted

Written 2026-09-01. Closes §22, §26.2 and §20.5.

### 29.1 The repair landed and the correction went the predicted way

Job 26399760 completed all 24 tasks. The contaminated `dkl_base_seed0_trained`
subject 1 — encoded through the diet front end despite being a no-diet model
— was re-encoded with `--no-lum-blur --no-lum-cs` (confirmed in the task log,
not assumed). Its hV4−V1v moves from the contaminated **−0.1669** to
**−0.1013**.

| | withdrawn | corrected |
|---|---|---|
| panel A, no-diet trained | −0.149 | **−0.117** |
| panel D, no-diet ρ | −0.56 | **−0.554** |

Both p-values stay at the n=8 floor (.0078, 8/8 subjects). §26.2 predicted the
correction would *widen* the diet/no-diet gap, and it does: the separation
goes from 0.712 to **0.744**. This was a correction, not a retraction, and the
figure's claim is stronger after it than before.

The quarantine banner is removed from `fig1_caption.md`. The caption now
carries a **correction history** paragraph naming both withdrawn numbers and
why they were wrong, so the record survives in the artefact rather than only
in this file. §17's two stale `−0.149` references were corrected in place with
a superseded note; the §22/§26 passages that *describe* the withdrawal are
history and were left alone.

### 29.2 The untrained arm is relabelled (§20.5)

Panel A's caption now states plainly that the untrained points are a
**front-end validation, not an architectural baseline**: §19's Gabor
oriented-energy null reproduces them from each stream's transformed input
alone, so an untrained network's stream preference reflects the input filter
and not the architecture. Their job is to show neither front end by itself
imposes a cortical gradient — which is what licenses reading the
post-training divergence as an effect of learning. This closes the relabelling
§20.5 asked for.

### 29.3 The swap arm is in panel C (§21.4)

Panel C had two arms and now has three. This required an `untrained_swap`
probe, which did not exist — the untrained end of the swap slope. It returned
magno **+0.4167**, which reproduces `untrained_noblur`'s magno (+0.4167) to
four decimals, exactly as it must since both hand stream 1 untransformed
luminance. That agreement is a useful check that the swap front end is doing
only what it claims.

What the third arm shows:

- **Magno, swap = −0.229 against no-diet's −0.226.** The two lines are
  collinear from untrained to trained. Taking the transform off luminance
  returns that channel exactly to its undieted state. Drawn dashed-over-solid
  so both remain visible; drawn solid the green simply hid the orange and the
  panel looked like it had one arm fewer.
- **Parvo, swap = −0.145 against −0.504/−0.515.** Putting the diet on parvo's
  own channel relieves its texture bias only partially and **does not cross
  zero**.

So panel C now carries the conjunction directly: neither the transform alone
nor the achromatic carrier alone produces a shape-biased stream. That is the
claim §21 and §25 established numerically, and it is now visible in the
figure rather than only in the text.

### 29.4 What is still open on this figure

- `parvo_highsf` is **not** in panel C. It belongs to the same conjunction and
  its probe numbers exist (magno +0.844, parvo −0.271, n=3), but a fourth arm
  makes the panel unreadable. The 2×2 is better served by its own small panel
  or a table; deferred rather than crammed in.
- Panel B still shows only untrained / no-diet / diet. The swap profile exists
  for legs (a) and (b) but has no encoding run yet — leg (c) could not express
  `--transform-target` until today (§30-to-be). Once those 8 subjects land,
  panel B gets its third curve.

---

## §45 §41.3 CORRECTED — the diet does not supply the asymmetry, it makes
training keep it

`tmp_analysis/pr_retention.py`, n=8, 1000 images, Gram-side eigendecomposition.

### 45.1 The error in §41.3

§41.3 concluded "the front end sets the asymmetry, no-diet training destroys
it" from a table in which **every** trained model was compared against a single
front-end row, `untrained_seed0` — which is the **diet** front end (blur+CS).
The no-diet model is not trained through that front end. Referencing it there
asks the wrong question.

The right reference was already on disk and never read: `untrained_seed0_noblur`
and `gabor_noblur`, the no-diet front ends. Their asymmetry is **larger**, not
smaller:

| front end (no training) | magno/parvo PR |
|---|---|
| `untrained_seed0` (diet, blur+CS) | 1.40 ±0.09 |
| `untrained_seed0_noblur` (no diet) | **1.63 ±0.12** |
| `gabor_blur_cs` (diet, learning-free) | 1.38 ±0.09 |
| `gabor_noblur` (no diet, learning-free) | **2.23 ±0.13** |

This inverts the causal story. The developmental transform *removes* asymmetry
from the input — it low-passes the luminance channel, so magno's effective
dimensionality falls (untrained magno PR 4.3 → 3.7; Gabor 25.1 → 15.5) while
parvo's rg channel is untouched. **The no-diet network is handed the larger
magno/parvo asymmetry and still ends up with none.**

### 45.2 Retention: each model against its OWN front end

retention_norm = ratio_trained / ratio_frontend; retention_raw = the difference.
Both statistics, per §8.

| trained model | diet? | norm | >1 | raw | >0 |
|---|---|---|---|---|---|
| `dkl_bcs_seed0` | diet | **1.15** ±0.08 | 8/8 | **+0.20** ±0.10 | 8/8 |
| `thesis_dvd_trans` | diet | 0.83 ±0.06 | 0/8 | −0.24 ±0.10 | 0/8 |
| `dkl_base_seed0` | no diet | **0.58** ±0.05 | 0/8 | **−0.69** ±0.13 | 0/8 |
| `dkl_base_seed1` | no diet | **0.61** ±0.05 | 0/8 | **−0.65** ±0.13 | 0/8 |

Against the learning-free Gabor front end instead of an untrained network, the
separation is starker still (diet 1.17 / 0.84; no-diet **0.43 / 0.44**, raw
−1.28 / −1.24), so the conclusion does not rest on one random initialisation.

**Every one of the 16 diet-vs-no-diet contrasts** — 2 diet models × 2 no-diet
models × 2 front-end references × 2 statistics — gives **p=.0078, 8/8
subjects**. Both statistics agree everywhere, so §8 is satisfied without
qualification. This is the most unanimous contrast in the project.

### 45.3 What may and may not be claimed

**May be claimed.** Diet-trained networks retain far more of their front end's
magno/parvo dimensionality asymmetry than no-diet networks do. No-diet training
discards roughly 40% of it against an untrained reference and ~57% against the
input-level bank, on every subject, in both seeds, on both statistics.

**May NOT be claimed.** That the diet *preserves* the asymmetry. The two diet
models straddle 1.0 — `dkl_bcs_seed0` amplifies it (1.15, 8/8 above 1),
`thesis_dvd_trans` reduces it (0.83, 0/8 above 1). "Preserves" is true of one
diet model and false of the other. The defensible statement is comparative
(diet retains more than no-diet), not absolute (diet retains all of it).
§41.3's replacement wording — "the diet preserves an asymmetry the input
transform creates" — is therefore **also wrong on both halves**: the transform
does not create the asymmetry (it shrinks it), and the diet does not reliably
preserve it.

**Superseded.** §41.2 ("the diet makes magno richer") and §41.3 ("the front end
sets the asymmetry") are both withdrawn. §41.1's use of PR to confirm the §22
repair is unaffected — that is a within-condition comparison and needs no
front-end reference.

### 45.4 Disclosures

- `thesis_dvd_trans` is referenced to `untrained_seed0`'s front end. If the
  thesis codebase's diet schedule differs in detail from ours, that reference
  is approximate — which is itself the most likely explanation for why the two
  diet models differ (1.60 vs 1.16 trained). Report the diet arm as a range.
- PR is computed on 1000 images per subject from the feature cache; the subject
  dimension is a resample of images, not independent brains, so the n=8 paired
  test speaks to stability across image samples, not to population inference.
- This is a statement about effective dimensionality of stream features. It is
  not a claim about task performance, and §43.3's cost disclosure (the diet
  loses ~13% of explained BOLD variance) still stands alongside it.

### 45.5 PRE-REGISTERED — the front-end 2×2, submitted before it was measured

Job **26406434** (`slurm/encode_frontend_2x2.sh`, array 0-31%8) fills the two
missing cells of the input-factor 2×2 that §25 established behaviourally, on
both the Gabor bank and the untrained network, using `--cache-only` so no ridge
fit is paid for. Smoke-tested first: `swap`'s magno input is bit-identical to
the no-diet magno (max|Δ| = 0.0000) and `parvo_highsf`'s magno is bit-identical
to the diet magno, so neither flag is a silent no-op.

| front end | magno stream gets | parvo stream gets | magno/parvo PR |
|---|---|---|---|
| `parvo_highsf` | lum, transformed | **lum+rg**, untransformed | *predicted lowest* |
| `blur_cs` (diet) | lum, transformed | rg, untransformed | 1.38 / 1.40 |
| `noblur` (no diet) | lum, untransformed | rg, untransformed | 1.63 / 2.23 |
| `swap` | lum, untransformed | rg, **transformed** | *predicted highest* |

**The prediction, from one principle** — PR tracks the bandwidth of the carrier
a stream receives, and the developmental transform removes bandwidth from
whichever stream it is applied to:

> **parvo_highsf < blur_cs < noblur < swap**, with `parvo_highsf` plausibly
> below 1.0 (parvo gains the luminance carrier outright).

Recording it here **before** the job returns, because the persistent failure
mode in this document is reading a story into a partially-filled table
(§39.3, §24.3, §28.1, §41.2, and now §41.3). If the ordering does not hold, the
"PR follows the carrier" account is wrong and §45.2 stands as a bare empirical
contrast with no mechanism attached — which is still publishable and must not
be quietly upgraded.

**Do not read this table until all 32 tasks are COMPLETE.** Partial cells are
how §41.2 happened.

---

## §46 THE §7.3b RE-LEAD — what the flagship figure becomes

§44.4 decided to reframe §7.3b around the stream dissociation. This is that
write-up: the claim, the instruments, the figure, and what is demoted.

### 46.1 The commission, and why it could not be met

§7.3 commissioned §7.3b as "the single panel that makes Track A and the
behavioural result one paper instead of two": cue-conflict images through the
fitted encoding models → per-voxel predicted shape-bias index → correlate with
SPI across voxels. The join was supposed to be **cortical** — the diet-trained
model's shape bias would be shown to land on the right part of visual cortex.

It does not. §43.2: with controls matched, the hV4/LO step is −0.095 (diet) vs
−0.060 (no diet), p=.1484 norm / .9453 raw — no difference. §44.1: against the
input-level null the residualised step is UNRESOLVED (norm .0078, raw .1953).
The cortical join is dead in both directions, and per the project standard a
result a control kills is dead.

### 46.2 The join that survives — and it is stronger than the one commissioned

The link between Track A and the behavioural result is not cortical
*localisation*. It is that **the diet determines whether the two streams stay
differentiated at all**, and this is now measured in four independent
currencies that agree:

| # | instrument | stimuli | what it measures | diet vs no diet |
|---|---|---|---|---|
| 1 | behavioural cue-conflict (§25) | Geirhos 1200 | shape bias of the readout | 0.878 vs 0.364 (magno vs no-diet parvo), 2×2 interaction |
| 2 | linear probe on stream features (§25) | Geirhos 1200 | shape−texture decodability | +0.819 vs −0.504, same 2×2 |
| 3 | **in-silico cue-conflict → BOLD (§43.1)** | Geirhos 1200 | per-voxel VSB, magno−parvo | **+1.534 vs +0.110**, 8/8, both statistics, p=.0078 |
| 4 | **participation-ratio retention (§45.2)** | NSD naturals | effective-dimensionality asymmetry kept through training | **1.15/0.83 vs 0.58/0.61**, 16/16 contrasts, p=.0078 |

Instruments 3 and 4 are the new ones and they are the reason this is a
publishable join rather than a restatement. Instrument 3 is the encoding model
— it is Track A, and it confirms the behavioural claim *through the brain data*
without needing the cortical gradient to cooperate. Instrument 4 uses **no
cue-conflict stimuli at all**: it is a purely geometric property of the
features on natural images, so it cannot be an artefact of the Geirhos set, of
the shape/texture labels, or of the readout.

**The claim to lead with:** *a developmental visual diet is what makes a
two-stream network preserve the separation between its streams; without it,
training collapses the two streams onto each other — in shape/texture selectivity
(1–3) and in representational dimensionality (4) alike.* Every one of the four
instruments is at its exact-test floor with unanimous subject agreement, and
they use different stimuli, different features, and different statistics.

**AMENDED by §47.4 — convergence on a contrast, not on a construct.** Do not
read the table as four measurements of one property. §47 ran the PR measure
across the same four front ends §25 used behaviourally and found they
**factorise differently**: shape bias needs the carrier×transform conjunction,
PR is additive in the two factors. The decisive case is `parvo_highsf`, where
parvo becomes the *more* dimensional stream (ratio 0.56) while staying
texture-biased (probe −0.266) — high dimensionality does not confer shape bias.
What instruments 1–4 agree on is the **diet-vs-no-diet contrast**, each at
p=.0078: no-diet training collapses the streams in shape/texture selectivity
*and*, separately, in dimensionality. That is a stronger claim than either
alone, but PR must never be described as the geometric version of the shape
bias.

### 46.3 Proposed figure

Replace the commissioned cortical panel with a four-panel convergence figure:

- **(a)** the §25 2×2 on input factors (carrier × transform), behaviour and
  probe side by side — establishes that the effect needs the conjunction.
- **(b)** in-silico cue-conflict: per-voxel VSB, magno−parvo, diet vs no-diet,
  8 subjects paired (§43.1). This is the encoding model's own verdict.
- **(c)** PR retention (§45.2): each trained model against its own front end,
  8 subjects paired, with the front-end reference row shown explicitly so the
  reader can see the no-diet model was handed *more* asymmetry.
- **(d)** the cost, plotted honestly: r²/NC, diet 0.405 vs no-diet 0.464
  (§43.3). The diet buys stream differentiation and pays ~13% of explained
  variance for it.

Panel (d) is not optional. A four-panel figure whose first three panels all
favour the diet and which omits the one measure that does not would be
cherry-picking.

### 46.4 Demoted to supplementary, with their negatives intact

- **The cortical VSB profile** (§33, §38, §43.2, §44.1). Report as: the gross
  hV4/LO profile is a *training* effect (trained > input-level null on both
  statistics, §44.1), but it is not diet-specific (§43.2) and the residualised
  version is unresolved. State all three.
- **The capacity ladder** (§42, §44.2) — it closes a reviewer objection to a
  panel that is now supplementary, so it goes with it. Keep the caveat that
  retention is non-monotonic between rungs.
- **§41.2 / §41.3** are withdrawn outright (§45.3), not demoted. The PR result
  in the main text is §45.2's retention framing, never §41.3's table.

### 46.5 What this costs the paper, stated plainly

The paper loses its cortical-localisation claim. It cannot say where in visual
cortex the diet's effect lands, and it should not imply it. What it gains is a
convergence argument across four instruments, which is a different and in some
respects sturdier kind of evidence than one panel would have been — but it must
be sold as convergence, not as localisation.

Still absent and still blocking: **Geirhos human observer data is not in the
repo**, so no "human-like" comparison can appear in any of the four
instruments. Either download it or drop the word.

### 46.6 Process note — a second cross-session commit

Commit `7bc7911` (the other session, Figure 1 regeneration) swept up this
session's then-uncommitted `run_encoding.py` changes (the result-file
provenance stamp and `--cache-only`). Content verified intact — worktree and
HEAD agree exactly, all three call sites present. Shared history was **not**
rewritten. This is the mirror image of §23, where this session's `git add -A`
captured two of theirs. Two incidents in two days from the same cause: both
sessions edit one worktree and neither can see the other's staged state. The
durable fix is path-limited `git add` on both sides; §23 adopted it here, and
this note records that the other direction is still open.

### 45.6 Front-end audit of the §45.2 references — and a disclosure corrected

Two of the four front-end references are UNSTAMPED (written before the §22.6
guard). Audited by submit line, the method §22.3 and §41.1 established:

| reference | stamp | submit line | front end |
|---|---|---|---|
| `untrained_seed0` | unstamped | `encode_subjects.sh untrained_seed0` (no extra args, all submissions) | defaults = blur+CS @48 |
| `untrained_seed0_noblur` | unstamped | `... untrained_seed0_noblur --no-lum-blur --no-lum-cs` | no diet |
| `gabor_blur_cs` | stamped | `--features gabor`, defaults | blur+CS @48 |
| `gabor_noblur` | stamped | `--features gabor --no-lum-blur --no-lum-cs` | no diet |

All four are what §45.2 assumed. The retention result stands as reported.

**§45.4's first disclosure was wrong and is replaced.** It said
`thesis_dvd_trans` is referenced to `untrained_seed0`'s front end
"approximately". The cache is stamped, and the stamp says otherwise:

    thesis_dvd_trans      prov_lum_blur=True  prov_lum_cs=True  age=48
    dkl_bcs_seed0_trained prov_lum_blur=True  prov_lum_cs=True  age=48

The *extraction* front end is identical to the other diet model's, and to
`untrained_seed0`'s. The reference is **exact**, not approximate.

The real caveat is a different and more interesting one. `thesis_dvd_trans` was
**trained** by the thesis codebase under its own DVD schedule, but its features
are **extracted** here through blur+CS @48. Where those two disagree the model
is being read out through a front end it was not trained under. That is a
train/extract mismatch specific to this one model, and it is a better
explanation for the diet arm's spread (1.60 vs 1.16 trained; 1.15 vs 0.83
retention) than anything about the reference. `dkl_bcs_seed0` has no such
mismatch — it was trained and extracted under the same front end.

**Consequence for §45.3.** The instruction to report the diet arm as a range
stands, but the reason changes: the range is not irreducible model-to-model
noise, it is one clean model plus one with a documented front-end mismatch. If
the diet arm ever needs a point estimate, `dkl_bcs_seed0` is the defensible one
and `thesis_dvd_trans` is the one to disclose. Do not average them.

---

## §47 THE FRONT-END 2×2 — pre-registered prediction CONFIRMED, and PR is
explained entirely by carrier bandwidth

Job 26406434, **32/32 COMPLETED, zero failures**. Read only after all 32 landed,
per §45.5. `tmp_analysis/pr_frontend_2x2.py`.

### 47.1 The result

magno/parvo participation ratio, n=8 subjects per cell, two references:

| front end | magno gets | parvo gets | Gabor | untrained |
|---|---|---|---|---|
| `parvo_highsf` | lum + transform | **lum+rg**, no transform | **0.56** ±0.02 | **0.59** ±0.01 |
| `blur_cs` (diet) | lum + transform | rg, no transform | 1.38 ±0.09 | 1.40 ±0.09 |
| `noblur` (no diet) | lum, no transform | rg, no transform | 2.23 ±0.13 | 1.63 ±0.12 |
| `swap` | lum, no transform | rg + **transform** | **3.37** ±0.16 | **1.91** ±0.12 |

Pre-registered ordering **parvo_highsf < blur_cs < noblur < swap**:

- **CONFIRMED on both references**, in **8/8 subjects**, with every adjacent
  pair at **p=.0078** (the exact Wilcoxon floor for n=8).
- `parvo_highsf` falls **below 1.0** as predicted — 0.56 / 0.59, 8/8 subjects
  below 1. Give parvo the luminance carrier and **parvo becomes the more
  dimensional stream**. The asymmetry reverses.

### 47.2 The mechanism check — only the stream whose input changed moves

Per-subject PR relative to the `blur_cs` baseline:

| front end | magno | parvo | |
|---|---|---|---|
| `parvo_highsf` | **×1.00** | ×2.45 / ×2.39 | parvo gains the lum carrier |
| `noblur` | ×1.62 / ×1.17 | **×1.00** | magno gains back its high SF |
| `swap` | ×1.62 / ×1.17 | ×0.66 / ×0.86 | magno gains SF, parvo loses it |

The two **×1.00** entries are exact and are the controls: `parvo_highsf` does
not touch the magno input and magno does not move; `noblur` does not touch the
parvo input and parvo does not move. Every cell moves **only** the stream whose
input actually changed, in the predicted direction, on both references.

**The carrier account is therefore not merely consistent with the ordering — it
predicts which stream moves in each cell, and it is right four times out of
four.** There is nothing "magno" about the high-dimensional stream. PR tracks
the bandwidth of the carrier a stream is handed: luminance beats red-green, and
the developmental transform subtracts bandwidth from whichever stream it lands
on.

### 47.3 What this does to §45 — it strengthens the claim by removing a rival

§45.2 showed diet-trained networks retain their front end's asymmetry and
no-diet networks destroy it (16/16 contrasts, p=.0078). The obvious rival
explanation was that the asymmetry is some privileged property of the magno
pathway which the diet protects. §47 kills that: the asymmetry is a generic
consequence of carrier bandwidth, reproducible with **no network at all** (the
Gabor rows), and **reversible** — `parvo_highsf` inverts it.

So the diet does not protect anything special about magno. It changes what
training does to a dimensionality difference that the input hands over for
purely bandwidth reasons. §45.1's correction stands and is now mechanistic:
the no-diet front end supplies *more* asymmetry (2.23 vs 1.38 Gabor) precisely
because it withholds nothing from the luminance carrier — and no-diet training
still throws it away.

### 47.4 A DISSOCIATION that must be disclosed — PR is not a proxy for shape bias

§25's behavioural 2×2 found an **interaction**: neither the achromatic carrier
alone (parvo_highsf parvo, probe −0.266) nor the transform alone (swap parvo,
−0.155) moves parvo across zero; only the conjunction reaches +0.819. §47's PR
2×2 on the *same four front ends* is **monotone and additive** — each factor
moves its own stream independently, with no interaction needed.

The two measures therefore factorise **differently over identical inputs**. The
sharpest case: in `parvo_highsf`, parvo becomes the **more dimensional** stream
(ratio 0.56) while remaining **texture-biased** (probe −0.266). High effective
dimensionality does not confer shape bias.

**Consequence for §46.2.** The four-instrument convergence table must not imply
instrument 4 measures the same property as instruments 1–3. It does not.
Instruments 1–3 measure shape/texture selectivity; instrument 4 measures
representational dimensionality; §47 proves these dissociate across input
conditions. What the four still agree on — and all at p=.0078 — is the
**diet-vs-no-diet contrast specifically**: no-diet training collapses the two
streams in *both* respects, which is a stronger statement than either alone,
but it is convergence on a contrast, not on a construct. §46.2 is amended
accordingly; do not describe PR as "the geometric version of the shape bias".

### 47.5 Disclosures

- The Gabor reference shows larger spreads than the untrained network
  (3.37 vs 1.91 at `swap`) because the untrained ResNet compresses everything
  to PR ≈ 2–6 (§36). The *ordering* is identical; only the dynamic range
  differs. Claims should use the ordering, not the magnitudes.
- `parvo_highsf` gives stream 2 a second input channel (§25.5's architectural
  disclosure applies here too). For the Gabor rows this is not a confound —
  there is no network and no learned capacity, so the doubled parvo PR is a
  direct statement about the input's bandwidth.
- n=8 is 8 image samples from 8 subjects' NSD stimuli, not 8 independent
  brains; the paired tests speak to stability across image samples.

### 47.6 Coordination — trained `parvo_highsf` is the other session's

`dkl_bcs_parvo_highsf_seed0_trained` feature caches appeared 2026-09-01
15:24–15:31 (subjects 1–4), i.e. the other session is running trained
`parvo_highsf` through leg (c) — the gap §25.5 recorded. This session produced
only the learning-free `gabor_parvo_highsf` / `untrained_parvo_highsf` front
ends, so there is no duplication.

**The opportunity, for whoever gets there:** when their trained cell reaches
n=8, §45.2's retention analysis can run on a **third** condition, and the
prediction is sharp — if the diet's role is to make training keep whatever
asymmetry the input supplies, then a diet-trained `parvo_highsf` model should
retain its *inverted* (below-1.0) asymmetry rather than drift back toward
magno-dominance. That is a falsifiable test of §45.3 on a front end whose
asymmetry points the other way, and it is the single most informative run left
in this thread.

## §48 LEG (c) CONTROLS COMPLETE (n=8) — the diet owns the cortical gradient,
but both controls destroy the two-stream decomposition and that must be said

Jobs 26407250 / 26407256 (8 subjects each) plus 26411650 (one repair). The two
front-end controls that §29.4 left unencoded now have full NSD encoding runs,
so leg (c) has the same 2×2 the behaviour (§25) and PR (§47) legs already had.

### 48.1 The result

Voxelwise banded ridge, `prf-visualrois` V1v→hV4, `norm` = median(Um−Up) /
median(Um+Up), noise-ceiling > 0.1. Um / Up are the median unique variance of
each stream over all retinotopic voxels. n=8 subjects, seed0 throughout.

| condition | what parvo gets | Um | Up | level | hV4−V1v | ρ |
|---|---|---|---|---|---|---|
| no diet | rg, no transform | 0.164 | 0.013 | +0.838 | −0.101 | −0.502 |
| **diet on luminance** | rg, no transform | 0.041 | 0.121 | −0.454 | **−0.862** | **−0.964** |
| swap | rg + **transform** | 0.288 | **0.006** | +0.957 | −0.027 | −0.259 |
| `parvo_highsf` | **lum+rg**, no transform | **0.007** | 0.186 | −0.909 | −0.119 | −0.780 |

Paired Wilcoxon, diet against each control, n=8 (exact floor p=.0078):

| contrast | gradient p | ρ p |
|---|---|---|
| diet vs no diet | **.0078** | **.0078** |
| diet vs swap | **.0078** | **.0078** |
| diet vs `parvo_highsf` | **.0078** | .0274 |

The diet arm's gradient is 7–33× larger than every control's and separates from
all three at the floor. **Leg (c)'s central claim survives its own controls.**

### 48.2 The caveat that must not be buried — both controls kill a stream

Read the Um / Up columns, not just the gradient. In `swap` the parvo stream
contributes Up = 0.006 — a twentieth of the diet arm's 0.121, and four times
*less* than the no-diet arm's already small 0.013. In `parvo_highsf` the magno
stream contributes Um = 0.007. **Each control leaves one stream contributing
essentially nothing unique.**

This matters because `norm` is a ratio. When one stream is ~0 everywhere the
statistic saturates (level +0.957 and −0.909 respectively, against −0.454 for
the diet) and there is almost no dynamic range left in which a gradient could
appear. So the honest statement is **not** "the controls abolish the gradient",
which would credit the manipulation with an effect the statistic could not have
shown either way. It is:

> Only the diet-on-luminance configuration yields two streams that *both* carry
> substantial unique cortical variance **and** a hierarchical gradient between
> them. The controls do not relocate the diet within a fixed two-stream
> decomposition — they destroy the decomposition.

That is a narrower claim than the one §29.4 anticipated, and it is the one the
data support.

### 48.3 ρ separates the two controls, and only one of them is really flat

`swap` collapses on both measures: ρ = −0.259, below even the no-diet arm's
−0.502. `parvo_highsf` does not — it retains ρ = −0.780, between diet (−0.964)
and no diet, while its amplitude is 7× smaller than the diet's. Because ρ is
rank-based it survives the saturation that flattens the amplitude, so it is the
fairer cross-condition read here, and by ρ the ordering is

    diet (−0.964)  >  parvo_highsf (−0.780)  >  no diet (−0.502)  >  swap (−0.259)

Note the diet-vs-`parvo_highsf` ρ contrast is p = .0274, the only one of the six
tests above the floor. **Do not write that the controls flatten the gradient.**
`parvo_highsf` keeps a real monotonic ordering; what it loses is amplitude, and
what it loses completely is its magno stream.

### 48.4 Convergence with §47 — the carrier account predicts which stream dies

§47 established on PR that a stream's dimensionality tracks the bandwidth of the
carrier it is handed, and that the transform subtracts bandwidth from whichever
stream it lands on. The encoding data agree, on a different instrument:

| condition | §47 PR ratio (magno/parvo) | §48 dominant stream |
|---|---|---|
| `swap` | 3.37 — magno far more dimensional | magno (Um 0.288, Up 0.006) |
| `parvo_highsf` | 0.56 — **parvo** more dimensional | parvo (Up 0.186, Um 0.007) |

The stream holding the full-bandwidth luminance carrier dominates both the PR
and the unique cortical variance, and `parvo_highsf` inverts both together. So
§48.2's degeneracy is **not an artifact of the encoding model** — it is what
§47's carrier account predicts: hand a stream a gutted carrier (rg blurred at
σ=4 in `swap`; dieted luminance competing against a parvo stream holding raw
luminance in `parvo_highsf`) and it has nothing unique left to contribute.

This is convergence on a *mechanism*, and unlike §46.2's four-instrument table
it is not vulnerable to §47.4's dissociation warning, because the claim here is
about which stream carries information, not about shape bias.

### 48.5 Provenance — every file now says what produced it

All 16 outputs carry the nine `prov_*` keys added in `5b42163` (27 arrays each:
18 base + 9 provenance), recording `transform_target`, `parvo_highsf`, and the
exact checkpoint. The pre-existing `parvo_highsf` subject 1 from validation job
26402643 had **18 arrays and no provenance** — its feature cache was labelled
but its output was not, because the cache patch (01:38) preceded the output
patch. It was deliberately re-run rather than kept, so no unlabelled file sits
in a set of labelled ones. That is the §22 lesson applied prospectively.

### 48.6 Operational — the §22 disk-quota failure recurred, and why

`26407250_7` (swap subject 8) died with `OSError: [Errno 122] Disk quota
exceeded` **at the final `np.savez`**, after its full 7595 s fit had completed,
leaving a 0-byte output. `/home` had reached 410G against a 400G soft / 410G
hard quota. The dominant consumer is **`~/projects/feedback_vlm` at 247G**;
this repo is 12G. Nothing in this project can fix that and it will recur.

Two further points worth recording:

- 2.1G of stale one-epoch AMP/timing benchmark checkpoints was archived off
  `/home` to `/scratch/byrong/VST_data/tmp_analysis_archive/`. The first attempt
  used a bare `mv` of the tree, which the Data Saving Rules forbid; it was
  interrupted having copied 2 of 7 subdirectories and removed nothing, so no
  data was lost. Redone with `rsync` + a verify-before-delete gate. That gate
  then fired a *false* mismatch because `du -sb` counts directory entries and
  Lustre directories are 33280 B against `/home`'s 4096 B — comparing file bytes
  only, all four matched exactly.
- The repair run then wedged **twice** on specific nodes (c165, c123): CPU,
  RSS and disk-read counters frozen byte-identical across 45 min, at 2 s of CPU
  in 46 min of wall clock. `/scratch` was healthy throughout and the same job
  ran normally on c132 (86% CPU). Node-local I/O stalls, not a code or
  filesystem fault. If an encoding task shows a flat `sstat` AveCPU, cancel and
  resubmit with `--exclude=` rather than waiting out the 12 h wall clock.

## §49 THE §47.6 PREDICTION IS FALSIFIED — and what replaces it is better:
the diet sets a dimensionality SET POINT, it does not retain the input's

`tmp_analysis/pr_retention_highsf.py`, n=8, NIMG=1000, Gram-side. Run the
moment `dkl_bcs_parvo_highsf_seed0_trained` reached 8/8 subjects (2026-09-02).

### 49.1 The prediction, and its failure

§47.6 pre-registered the sharpest available test of §45.3. If the diet's role
is to make training **keep** whatever asymmetry the input supplies, then a
diet-trained model on the `parvo_highsf` front end — whose asymmetry is
*inverted*, magno/parvo = 0.59, below 1.0 in 8/8 subjects (§47.1) — should
stay inverted rather than drift back toward magno-dominance.

It does not. It flips completely.

| | front end | trained | below 1.0 |
|---|---|---|---|
| `dkl_bcs_parvo_highsf_seed0` | **0.586** ±0.013 | **1.640** ±0.030 | front 8/8 → trained **0/8** |

Paired Wilcoxon trained vs its own front end: p = .0078. Per-subject trained
ratios 1.60–1.68, not one below 1.0. **The KEEP account is falsified**, and it
was this session's own account, pre-registered in writing before the trained
cell existed.

### 49.2 What the same numbers say instead — a set point

The reference is exact here in a way none of §45.2's were: `untrained_parvo_highsf`
carries provenance identical to the trained model on every key except
`prov_checkpoint` (`lum_blur=True, lum_cs=True, parvo_highsf=True, age=48,
seed=0`). So this is the cleanest front-end pairing in the series, and the
comparison it licenses is the informative one — **the same training recipe, the
same seed, two front ends 2.4× apart**:

| arm | model | front end | trained |
|---|---|---|---|
| diet | `dkl_bcs_seed0` | 1.402 ±0.09 | **1.603** ±0.032 |
| diet | `dkl_bcs_parvo_highsf_seed0` | **0.586** ±0.013 | **1.640** ±0.030 |
| diet | `thesis_dvd_trans` | 1.402 ±0.09 | 1.158 ±0.034 |
| no diet | `dkl_base_seed0` | 1.634 ±0.118 | 0.947 ±0.035 |
| no diet | `dkl_base_seed1` | 1.634 ±0.118 | 0.988 ±0.040 |

The two `dkl_bcs` models are handed front ends that differ by **0.816** in
magno/parvo PR (2.39×, p=.0078). After training they differ by **0.037** —
a **22× compression**. Training does not carry the input asymmetry forward;
it drives the ratio to a value the input does not choose.

**Every diet × no-diet contrast on the trained ratio** (3 diet models × 2
no-diet models) is **p=.0078, 8/8 subjects**, and diet trained (1.62 ±0.04 for
the two `dkl_bcs` models) sits far above no-diet trained (0.97 ±0.04). The
no-diet models land essentially at parity — 0.95 and 0.99 — i.e. the two
streams end up equally dimensional, which is the collapse §46.2 describes,
measured directly rather than as a ratio-of-ratios.

**The claim that replaces §45.3's:**

> A developmental visual diet does not preserve the magno/parvo dimensionality
> asymmetry the front end supplies. It installs one. Diet-trained networks
> converge on a magno-dominant ratio (~1.6 for our recipe) from front ends that
> are magno-dominant (1.40) *and* from front ends that are parvo-dominant
> (0.59); no-diet networks converge on parity (~0.97) from a front end that is
> strongly magno-dominant (1.63). What training ends with is set by the diet,
> not by the input.

This is a stronger statement than retention, and it is the one the data
support. It also explains the anomaly §45.3 had to concede: the two diet models
"straddled 1.0" in *retention* only because retention divides by a moving
denominator. Their **trained** values are the stable quantity.

### 49.3 §45.2 is reframed, not withdrawn

§45.2's 16/16 contrasts at p=.0078 are arithmetically correct and stand as
computed. What changes is their interpretation: "diet retains more of the
input's asymmetry" was the right description of a two-condition dataset in
which every front end happened to be magno-dominant, so *retain* and *install*
made the same prediction. §47.6's third condition separates them and selects
*install*. Report the trained ratios (49.2's table) as the primary quantity and
retention as a derived one, never the reverse.

§45.1's correction of §41.3 is untouched and is now doubly safe: the no-diet
front end supplies more asymmetry and no-diet training still ends at parity.

### 49.4 Disclosures — four, and the third is the one that limits the claim

1. **The set point is per-recipe, not universal.** `thesis_dvd_trans` lands at
   1.158, not ~1.62, on the same front end as `dkl_bcs_seed0`. It is a
   different codebase and diet schedule (§45.4), and it was extracted through
   blur+CS@48 rather than its own training schedule (§45.6) — so it is the
   least comparable row in the table. The tight convergence claim is made for
   the two `dkl_bcs` models only; the diet-vs-no-diet claim holds across all
   three diet models.
2. **The two diet arms are not statistically identical** — 1.603 vs 1.640,
   p = .0391, 8/8. They differ. The point is the magnitude: a 2.3% difference
   in the trained ratio from a 139% difference in the input. Do not write
   "identical"; write "converge to within 2%".
3. **`parvo_highsf` changes the architecture, not only the front end.** Stream
   2 receives 2 input channels instead of 1 (§25.2). So this row varies input
   *and* first-layer fan-in together, and the two cannot be separated with the
   models that exist. §25.2 measured that the extra channel *helped* parvo
   (16-way accuracy 0.486 vs ~0.379), which rules out the confound that would
   fake the effect here — a weakened parvo stream — but it does not make the
   comparison architecture-matched. **State this wherever 49.2's table appears.**
4. **n=8 is subjects-as-image-samples**, not independent brains (§45.4). PR is
   computed on 1000 NSD images per subject from the feature cache; the paired
   test speaks to stability across image samples.

### 49.5 Convergence with §48, and what it costs

§48 found on the encoding side that `parvo_highsf` leaves the *magno* stream
contributing essentially no unique cortical variance (Um = 0.007 vs Up = 0.186)
— parvo dominates the cortex in that condition. §47.1 found the same inversion
in the front end's PR (0.56). But §49 now shows the **trained** network's PR
inverts back to magno-dominance (1.64) in that same condition.

These are not in conflict — they are different objects (§48's Um/Up are unique
*cortical* variance from the fitted encoding model; §49's PR is the geometry of
the features themselves) — but the combination is awkward for any narrative
that treats the two as one story, and it must not be smoothed over. The honest
summary is that `parvo_highsf` is a condition where the input, the trained
feature geometry, and the cortical fit do **not** point the same way, and it is
therefore poor material for a headline panel. Its value is exactly what it was
used for here: as the falsifying case.

### 49.6 Status of the thread

> **CORRECTED BY §50 (same day).** The paragraph below was wrong twice over.
> There *was* a cheap measurement left — `dkl_bcs_swap_seed0_trained` had full
> 8-subject caches from §48 and had never been put on the PR instrument — and it
> *did* change §49.2's claim. The diet-on-rg arm lands at 2.225, not with the
> other two diet arms at ~1.62, so there is no single set point; §49.2's 22×
> compression figure is computed on the two arms that agree and must not be
> quoted alone. Read §50.4 for the claim that survives (which side of parity),
> and §50.3 for what is withdrawn. §49.1's falsification of the KEEP account is
> unaffected, and §49.3's reframing of §45.2 stands.

§47.6 called this "the single most informative run left in this thread." It
was, and it cost one analysis script on caches that already existed. The thread
is now closed: there is no cheap measurement left that would change §49.2's
claim. The remaining gaps are §46.5's (Geirhos human observer data absent, so
no "human-like" comparison may be written) and the fact that **no manuscript
text exists yet** — the project has results and no draft.

---

## §50 THE SET POINT IS CHANNEL-SPECIFIC, NOT RECIPE-LEVEL — §49.2's headline
compression figure is selective and must be restated

`tmp_analysis/pr_setpoint_swap.py`, n=8, NIMG=1000, Gram-side, on caches that
already existed. Prediction and the three admissible outcomes were written into
the script docstring and committed as `6baad39` **before** the run, so nothing
here was fitted to the result.

### 50.1 Why a fourth front end was available at zero cost

§49.6 closed the PR thread with "there is no cheap measurement left that would
change §49.2's claim." That was wrong on one point. `dkl_bcs_swap_seed0_trained`
had full 8-subject feature caches from §48's encoding runs and had **never** been
put on the PR instrument. It is the best-qualified fourth case in the set:

- Its front end is the **most extreme of the four** (§47.1): 1.91 untrained /
  3.37 gabor, against `parvo_highsf`'s 0.59. Diet-trained front ends now span
  **3.26×** instead of 2.4×.
- It dissociates *diet-trained* from *diet on the achromatic carrier*. Every arm
  in §49.2's table confounds the two; swap puts the identical transform on rg
  and leaves luminance untouched.
- Its reference is **provenance-exact** — `untrained_swap` matches the trained
  model on every `prov_*` key but `prov_checkpoint`. Only the `parvo_highsf`
  pair is this clean; the diet/normal arm's reference is unstamped and audited
  only by submit line (§45.6).
- Unlike `parvo_highsf` it is **architecture-matched** to the diet arm — same
  channel counts, same first-layer fan-in, only the transform target differs.
  It therefore carries none of §49.4's disclosure-3 confound.

### 50.2 The result

| arm | model | front end (untr / gabor) | **trained** |
|---|---|---|---|
| diet on lum | `dkl_bcs_seed0` | 1.40 / 1.38 | 1.603 ±0.032 |
| diet on lum | `dkl_bcs_parvo_highsf_seed0` | 0.59 / 0.56 | 1.640 ±0.030 |
| **diet on rg** | **`dkl_bcs_swap_seed0`** | **1.91 / 3.37** | **2.225 ±0.079** |
| no diet | `dkl_base_seed0` | 1.63 / 2.23 | 0.947 ±0.035 |
| no diet | `dkl_base_seed1` | 1.63 / 2.23 | 0.988 ±0.040 |

swap trained is **2.225**, and it differs from **both** diet-on-luminance arms
at **p=.0078, 8/8 subjects** (per-subject 2.09–2.38, not one below 2.0). It does
not join the set point. Outcome **A is refuted** — and so is outcome B: it does
not collapse to parity either (vs no diet, p=.0078, 8/8).

The pre-declared outcome that fits is the third, in its milder form: swap is
diet-like in **sign** but has its **own level**.

### 50.3 What this kills — and it is in the abstract right now

§49.2's compression figure is computed on the two arms that agree. With the
third diet arm included it does not survive:

| reference | arms used | front-end span | trained span | compression |
|---|---|---|---|---|
| untrained | two diet-on-lum only | 0.816 | 0.037 | **22.1×** |
| untrained | **all three diet arms** | 1.324 | 0.622 | **2.1×** |
| gabor | two diet-on-lum only | 0.814 | 0.037 | 22.1× |
| gabor | **all three diet arms** | 2.803 | 0.622 | **4.5×** |

`docs/PAPER_DRAFT.md`'s abstract currently reads "A front-end gap of 0.816
becomes 0.037 after training." That sentence is arithmetically correct and
selects the two arms out of three that happen to converge. **Reporting 22×
while a third, cheaper, better-provenanced diet arm on the same instrument
gives 2.1× is cherry-picking by this project's own standard.** Either report
the three-arm number, or state explicitly that the 22× figure is the
diet-on-luminance pair and that the diet-on-rg arm lands elsewhere. The
abstract sentence must not stand as written.

### 50.4 What survives, and it is stronger than before

The claim §49.2 actually leads with — the diet-vs-no-diet contrast on the
**trained** ratio — is untouched and now rests on three front ends instead of
two:

| | vs `dkl_base_seed0` (0.947) | vs `dkl_base_seed1` (0.988) |
|---|---|---|
| diet on lum (1.603) | p=.0078, 8/8 | p=.0078, 8/8 |
| diet on lum/highsf (1.640) | p=.0078, 8/8 | p=.0078, 8/8 |
| diet on rg (2.225) | p=.0078, 8/8 | p=.0078, 8/8 |

**6/6 contrasts at the exact-test floor with unanimous subject agreement**,
across diet-trained front ends spanning 3.26×. Every diet arm ends
magno-dominant (≥1.60); both no-diet arms end at parity (0.95, 0.99).

**The claim that replaces §49.2's:**

> A developmental visual diet does not preserve the magno/parvo dimensionality
> asymmetry the front end supplies — a diet-trained network handed an inverted
> front end (0.59) ends magno-dominant (1.64), overtaking a model handed 1.40.
> But neither does it drive training to one universal value. What the diet
> fixes is **which side of parity training ends on**: every diet-trained model
> ends magno-dominant regardless of a 3.26× spread in what it was handed, and
> every no-diet model collapses to parity from a front end that is strongly
> magno-dominant. The *level* it converges to is set by **which channel the
> diet is applied to** — ~1.62 on luminance, 2.23 on red-green — not by the
> input asymmetry and not by the schedule alone.

This is weaker than "a set point" and stronger than "retention". It is what
three front ends support.

### 50.5 A second reason not to lean on direction-of-movement

§47.5 warned that the untrained ResNet compresses magnitudes while the Gabor
bank does not, so claims should use ordering rather than magnitudes. §50 shows
the warning has teeth: swap's direction of movement from its front end **flips
sign with the reference** — up against `untrained_swap` (1.91→2.23), down
against `gabor_swap` (3.37→2.23). The two diet-on-luminance arms move up on
both references and both no-diet arms move down on both.

So "the diet moves the ratio up" is **not reference-robust**. Only the
diet-vs-no-diet contrast on the *trained level* (50.4) is reference-free, which
is a further reason to make that the reported quantity and to drop retention
and compression framings entirely.

### 50.6 Process note — a decision rule that printed the wrong verdict

The script's first verdict rule was `|swap − setpoint| < |swap − parity| / 2`,
i.e. "closer to the set point than to parity". It printed **A: SET POINT IS
RECIPE-LEVEL** for a value that differs from both diet arms at p=.0078. The
rule was satisfiable by almost anything above ~1.3 and encoded no test of the
claim. Replaced with the actual test — outcome A requires swap to be
statistically *indistinguishable* from the two diet-on-luminance arms — which
returns "neither A nor B". Recorded because a pre-registered outcome rule that
cannot fail is worse than none: it launders a refutation as a confirmation.

### 50.7 Coordination

The PR thread is the other session's (§47, §49). This section adds a
measurement on caches that already existed and touches no prose of theirs;
`fig1_data.py` was modified in their worktree while this ran and was left
alone. §49.2–§49.4 are **not** rewritten here — 50.4 states what should replace
the claim, and one owner should apply it to §49 and to the draft's abstract
together.

## §51 FIGURE 1 REBUILT AS THE FOUR-PANEL CONVERGENCE FIGURE — and one cell
grew a seed

`tmp_analysis/fig1_data.py` + `tmp_analysis/fig1.py` →
`results/figures/fig1_convergence.{png,pdf}`, caption in
`fig1_convergence_caption.md`. Built to §46.3's spec with panel (c) replaced
per §49.

### 51.1 What changed from §46.3's spec

§46.3 specified panel (c) as "PR retention (§45.2): each trained model against
its own front end". §49 superseded the retention framing: retention divides by
a moving denominator, and the stable quantity is the **trained ratio**. Panel
(c) is therefore a front-end → trained slopegraph over three arms, which shows
the convergence directly — three arms enter at 1.63 / 1.40 / 0.59 and two leave
at the same place. Retention is a derived quantity and is not plotted.

The form matters here. A bar chart of trained values alone would delete the
entire argument, because the argument *is* that the starting points differ and
the endpoints do not.

### 51.2 A seed rejoined the swap cell — §25.1's table is superseded

Building panel (a) surfaced that `dkl_bcs_swap_seed2` now has both a
cue-conflict and a probe result on disk (re-run 2026-08-31 22:50/22:51), where
§20.2 had excluded it by the convergence guard at "96/150". On the re-run it
decides **182/1200** images — *more* than seeds 0 (159) and 1 (155) — so it
passes the guard and belongs in the cell.

| swap parvo | §25.1 (n=2) | now (n=3) |
|---|---|---|
| behavioural shape bias | 0.484 | **0.495** |
| probe shape−texture | −0.155 | **−0.145** |

Neither moves a claim: the cell is still far below the conjunction's 0.878 /
+0.819, and the probe still does not cross zero. §25.1's numbers are superseded
rather than withdrawn — they were correct for the two seeds available then.
`PAPER_DRAFT.md` §2.1 quotes the new values.

### 51.3 A denominator caveat that the 2×2 had not made explicit

The four cells' behavioural shape bias rests on very different numbers of
decided images — the count of images the stream answers as shape *or* texture:

| cell | decided / 1200 |
|---|---|
| chromatic + transform (`swap` parvo) | **165** |
| chromatic, no transform (no-diet parvo) | 270 |
| achromatic + transform (diet magno) | 488 |
| achromatic, no transform (`parvo_highsf` parvo) | 583 |

§18.5 and §20.3 made this point about the *untrained* models (~150/1200, "read
it as no preference, not as a precise 0.50"). It applies with equal force to the
`swap` cell at 165. The consequence is favourable to the claim but must be
stated the right way round: the three non-conjunction cells should be read as
**"no strong preference"**, not as precise values, and the interaction is
carried by the conjunction cell being unambiguous rather than by fine
distinctions among the other three. The probe has no such problem — it
classifies every image — and gives the same pattern, which is why both
modalities are drawn.

Added to the caption. This is a caveat §25 should have carried and did not.

### 51.4 The old Figure 1 is not deleted

`fig1_diet_gradient.{png,pdf}` and its caption remain. That figure is the
ventral-gradient result (§17/§29/§48), which `PAPER_DRAFT.md` §2.4(b) still
leads with — it is diet-specific, replicates across codebases to 0.0025, and
survives its front-end controls. It is no longer *Figure 1*; renumber it when
the figure order is fixed. Do not confuse it with the VSB cortical step
(§43.2/§44.1), which is the measure that does not survive and which the main
text does not raise.

### 51.5 Panel (c) rebuilt after §50 — and the numbering collision

Two things happened between §51's first build and this amendment, both from the
parallel session's §50 landing (`8cad8cf`) while this figure was being drawn.

**The numbering collided.** Both sessions wrote a §50 within two hours. Theirs
was committed first, so this section renumbered itself to §51; references in
`fig1_convergence_caption.md` and `PAPER_DRAFT.md` were updated with it. This is
the third cross-session collision in three days (§20.4, §23, §46.6) and the first
on section numbers rather than files. No content was lost either way.

**Panel (c) was wrong and is rebuilt.** As first drawn it showed three arms —
two diet-on-luminance and one no-diet — and was titled "The diet installs a set
point". §50 added the diet-on-rg arm at 2.225, which does not join the other two
diet arms, and the panel as drawn would have shown a set point only because the
refuting arm was absent. It now draws **four** arms and is titled "The diet fixes
which side of parity". The parity rule is drawn heavier than the reference rules
in the other panels because it is now the claim rather than a decoration.

The swap arm was **verified independently in this session** before the figure was
changed, rather than taken on the other session's word: front end 1.909 ±0.125 →
trained 2.225 ±0.079, per-subject 2.09–2.38, p=.0078 8/8 against both
diet-on-luminance arms and both no-diet arms, three-arm compression 2.1×.
Provenance checked first per §45.6 — `untrained_swap` matches
`dkl_bcs_swap_seed0_trained` on every `prov_*` key but the checkpoint. All
figures agree with §50.2 exactly.

**Consequences carried into the draft**, so the correction is not left in this
file only:

- The abstract's "a front-end gap of 0.816 becomes 0.037 after training" is
  **deleted**. It was arithmetically correct on two of three diet arms and
  selective by this project's standard — §50.3 caught it in the draft, which is
  the correct outcome of writing the draft before the analysis thread was closed.
- §2.3 is retitled and restated around the side-of-parity claim, with the swap
  arm in its table.
- Limitation 4 now says there is **no universal set point** and states why no
  compression factor is reported anywhere in the paper.

**The process lesson, which is the one worth keeping.** §49.6 declared the thread
closed on the grounds that no cheap measurement remained. A cheap measurement did
remain, on caches this session had already used for a different purpose, and it
overturned the headline. The failure was not the analysis; it was declaring a
thread closed while an unused instrument×condition cell existed. Before writing
"closed" again, enumerate the cells.

## §52 THE GEIRHOS HUMAN DATA IS IN HAND — the blocker in §20.3 and §46.5 is
lifted, and the number is 0.958

Downloaded 2026-09-02 to `/scratch/byrong/VST_data/geirhos_human_data/cue_conflict`
(13 files, 1.4 MB, on scratch not `/home` per the Data Saving Rules and §48.6).
Computed by `tmp_analysis/geirhos_humans.py`.

### 52.1 Why it was hard to find

`rgeirhos/texture-vs-shape` has **no directory called `cue-conflict`**. The
cue-conflict stimuli *are* the style-transfer images, so the human data lives in

    raw-data/style-transfer-512-nomask-experiment/

as `..._subject-01_session_1.csv` … `subject-10`, plus `alexnet`, `googlenet`
and `vgg` from the same experiment. That naming is the entire reason this sat
open as a blocker for four days.

### 52.2 The numbers, and the check that our arithmetic matches theirs

Shape bias as Geirhos defines it: over trials answered as either the shape or
the texture category, the fraction that were shape. Non-conflict trials (shape
category == texture category) excluded.

| observer | shape bias | decided / 1200 |
|---|---|---|
| **10 human observers** | **0.9576 ±0.0191** | 791–1055 |
| | range 0.919–0.987 | |
| AlexNet | 0.4294 | 659 |
| GoogLeNet | 0.3119 | 731 |
| VGG-16 | 0.1722 | 755 |

**The three CNN rows reproduce the published values.** That is the point of
computing them: it verifies our parser, our conflict-trial exclusion and our
denominator against numbers Geirhos already reported, rather than trusting a
pipeline we wrote against data we had never seen.

### 52.3 What may now be said — and it is still directional

| | shape bias | vs human mean |
|---|---|---|
| human | 0.958 | — |
| **diet, magno stream** | **0.878** | **−0.080** |
| diet, integrated readout | 0.675 | −0.283 |
| no diet, integrated | 0.416 | −0.542 |
| no diet, parvo stream | 0.364 | −0.594 |

The diet-trained magno stream reaches **0.878 against a human mean of 0.958**,
closing most of the distance from the texture-biased CNN regime (0.17–0.43) to
human observers. But **0.878 is below the lowest individual human (0.919)**, and
the integrated readout falls back to 0.675.

**Therefore: still directional, now quantified.** The permitted statement is that
the diet moves the network most of the way from the CNN regime toward human
observers on this measure and the no-diet model stays in the CNN regime — with
the numbers given. The forbidden statement is unchanged: no match, no
"human-like", no claim that any stream is in the human range. §20.3 stands as
written; what changes is that the gap is now measured (−0.080 for the best
stream) instead of gestured at.

Note also that §20.2 is sharpened by this: integration costs 0.203 of shape bias
(0.878 → 0.675), which is 72% of the remaining distance to humans. "Integration
loses shape bias" now has a magnitude attached and is worth a sentence.

### 52.4 Citation

Geirhos, Rubisch, Michaelis, Bethge, Wichmann & Brendel (2019), *ImageNet-trained
CNNs are biased towards texture; increasing shape bias improves accuracy and
robustness*, ICLR. Data: github.com/rgeirhos/texture-vs-shape (CC-BY 4.0 — confirm
the licence file before submission). **The paper must cite the data release, not
only the paper.**

## §53 PRE-REGISTERED — does the diet's shape bias buy OOD robustness?

Written and committed **before** the comparison was run. `tmp_analysis/ood_robustness.py`,
`slurm/ood_robustness.sh`, job array 0-11 (4 conditions × 3 seeds).

This is the experiment §7 never commissioned and the one that decides the
venue. Geirhos et al.'s claim is that shape bias buys robustness. This project
established the shape bias (§25) and never tested the robustness.

### 53.1 Design, and why not the model-vs-human OOD sets

Corruptions are applied to the RGB image **before the front end** — that is
where a real corruption happens; the retina sees the corrupted world — and
evaluated in the models' **own 112-way MiniEcoset label space**.

The obvious alternative was Geirhos's 17 OOD datasets. Rejected: they are
16-class ImageNet-derived, our models are MiniEcoset-trained, and that stacks a
domain shift on top of the corruption. The cue-conflict evaluation already pays
that cost (shape accuracy ~0.31) because it must — those stimuli are the
measure. For robustness there is no such obligation, so the corruption is
applied to the models' own test distribution and nothing else changes.

**n = 3 seeds per condition.** This is the first result in the project with
seed-level replication on all four cells, which is the gap the internal review
flagged as the most serious methodological hole (the PR and encoding results
are n=1 per arm). Whatever this experiment shows, it shows it over training
runs, not over a resample of one run.

### 53.2 Families, and which ones may support a claim

| family | severities | counts? |
|---|---|---|
| uniform-noise | 0 → 0.35 | **yes** |
| high-pass | none → σ=3 | **yes** |
| phase-scrambling | 0° → 180° | **yes** |
| rotation | 0/90/180/270 | **yes** |
| false-colour (opponent inversion) | off / on | **yes** |
| low-pass | σ 0 → 15 | **no — trained-in** |
| contrast | 1.0 → 0.03 | **no — trained-in** |

**The diet IS a low-pass blur plus a contrast-sensitivity cut on luminance.**
Low-pass and contrast robustness are therefore partly trained in, and cannot
support a robustness claim. They are run and reported because hiding them would
be worse, and they are never counted.

### 53.3 Two statistics, and here it is not a formality

The diet costs clean accuracy (§18: 0.37–0.45 vs 0.47–0.51), so absolute
accuracy under corruption is confounded by where each model starts.

    absolute = top-1 accuracy at severity s
    relative = accuracy(s) / accuracy(clean)

`relative` is the one that asks "is this model more robust" rather than "is
this model better". Per §8 a claim is made only where both agree.

### 53.4 THE PREDICTIONS, declared now

**H1 (Geirhos).** If shape bias confers robustness, the diet arm degrades more
slowly than no-diet in `relative` terms on uniform-noise, high-pass and
false-colour.

**H2 (the sharp one — a crossover, not a main effect).** Phase-scrambling
destroys global form while preserving local texture and the amplitude spectrum.
A shape-reliant model should therefore degrade **more**. So the signature of
genuine shape reliance is a **crossover**: diet better under noise and
high-pass, **worse** under phase-scrambling. A uniform "diet wins everywhere"
would be evidence for a generic regularisation account, not for shape.

**H3 (the causal control, and the one that decides the interpretation).**
`swap` applies the identical developmental transform to the chromatic channel
and has **no** shape bias (§25: probe −0.145). `parvo_highsf` likewise
(−0.266). If robustness tracks shape bias, both must pattern with **no-diet**.
If `swap` patterns with the diet instead, then the transform is acting as a
generic blur regulariser and the shape bias is incidental to the robustness.

### 53.5 Outcomes, declared in advance

- **A.** Diet > no-diet on the counted families, `swap`/`parvo_highsf` with
  no-diet, and diet < no-diet on phase-scrambling. → Shape bias confers
  robustness, causally isolated by the 2×2. This is the NMI/NeurIPS result.
- **B.** Diet > no-diet everywhere *including* `swap`. → The transform is a
  generic regulariser; robustness is real but not attributable to shape bias.
  A different, weaker, still-publishable claim.
- **C.** No `relative` difference. → **The diet's shape bias does not buy
  robustness.** Then the paper is a neuroscience paper, the NeurIPS framing
  should be dropped, and this section is reported as a negative result in the
  supplement rather than quietly not run.

C is a real possibility and it is written down here so that it cannot later be
described as something we always expected. Whatever lands, §53.4's predictions
are the ones on record.

### 53.6 Validation gate, and what was seen before pre-registering

The script refuses to report numbers unless our RGB→DKL front end reproduces
the h5 the models were actually trained on. It does, **exactly**: mean and max
absolute difference 0.000 uint8 levels over the first 64 test images, and the
RGB and DKL test labels are identical. So the corruption is being pushed through
the same retina the weights were fitted for, and the two h5s are the same split.

**Disclosed:** a 128-image smoke test was run on `dkl_bcs_seed0` alone to check
runtime and to find that the original high-pass severities saturated at chance
(σ=3.0 gave 0.008, chance is 0.009); the family was rescaled to σ 30→3 as a
result. No second model was run, so the diet-vs-no-diet contrast that §53.4
predicts is unseen at the time of writing.


---

## §54 FOUR ADDITIONS TO §52 — the decision-rate caveat, all arms against the
human CI, seed stability, and what integration costs leg (b)

§52 is the section of record for the human reference; nothing here restates it.
This section was written independently and in parallel (the two sessions
downloaded the same experiment from two different mirrors within the same hour
— see 54.4), so it is deliberately trimmed to the four things §52 does not
cover. `tmp_analysis/human_shapebias.py`, reading §52's canonical copy.

### 54.1 The decision-rate caveat — the models decide far less often than
either humans or the ImageNet CNNs

§52.2 records `decided / 1200` for humans (791–1055) and for the three
reference CNNs (659–755). It does not record it for **our** models, and that is
where the problem is:

| | in-pair decision rate |
|---|---|
| human observers | **0.803** ± 0.069 |
| AlexNet / GoogLeNet / VGG-16 (§52.2) | 0.55 – 0.63 |
| **our models, all arms** | **0.24 – 0.53** |
| diet on lum, magno stream | **0.41** |
| untrained (either front end) | **0.12** |

Shape bias is a ratio computed **only over decided trials**. Our diet-trained
magno stream therefore reaches 0.878 on ~41% of the stimuli, against the human
0.958 on ~80% of them — the two ratios have denominators differing by a factor
of two, and the model's decided subset is not guaranteed to be a random half of
the trials.

**Consequence.** Every model shape bias in the paper must carry its decision
rate. §52.3's forbidden-statement list is correct and this is a second,
independent reason for it: even if a stream *had* landed inside the human
range, equal ratios on unequal denominators would not license "behaves like".
The untrained rows (12%) are the extreme case and should not be reported as
shape biases at all — a ratio over 12% of trials is noise with a decimal point.

### 54.2 All arms against the human CI, not just four

§52.3 tabulates four rows. The full set, all k=3 seeds where they exist, gives
the human 95% CI as **0.9439 .. 0.9713** and:

| arm | condition | shape bias | gap | inside human CI |
|---|---|---|---|---|
| diet on lum | magno only | 0.8778 ±0.0122 | −0.080 | no |
| diet on lum | integrated | 0.6749 ±0.0249 | −0.283 | no |
| `thesis_dvd_trans` | magno only | 0.9109 | −0.047 | no |
| `thesis_s1_90` | magno only | 0.9035 | −0.054 | no |
| parvo_highsf | magno only | 0.7966 | −0.161 | no |
| rgb (arch control) | integrated | 0.4449 | −0.513 | no |
| diet on rg (swap) | magno only | 0.4007 | −0.557 | no |
| no diet | magno only | 0.4210 | −0.537 | no |
| no diet | parvo only | 0.3643 | −0.593 | no |

**Not one arm of any condition falls inside the human 95% CI**, the closest
being `thesis_dvd_trans` at −0.047. This is the stronger form of §52.3's point
that 0.878 is below the lowest individual human: it holds for every model we
have, not only the best one, so the "still directional" verdict is not an
artefact of which model was picked.

Note the two thesis-codebase checkpoints (0.911, 0.904) beat our own best
(0.878) on this measure. §45.6's train/extract mismatch does not apply here —
cue-conflict is a behavioural readout with no feature extraction — so this is a
real, unexplained advantage for the thesis recipe on the behavioural
instrument, and it should not be quietly omitted from a table that otherwise
leads with our models.

### 54.3 Integration and leg (b)

§52's note that "integration costs 0.203 of shape bias" is arithmetically the
same observation as this, but the framing matters for the paper's claim
structure. The project's working claim has carried leg **(b)** since §7.3:
*"integration yields human-like shape-biased behaviour."*

    diet on lum, magno stream alone   0.8778   closes 85% of no-diet -> human
    diet on lum, integrated readout   0.6749   closes 47%

Integration moves the readout **away** from humans, in 3/3 seeds. **Leg (b) as
worded is refuted by our own data and must not be written in any form.** What
survives is a claim about the *stream*: the diet produces a stream whose shape
bias approaches the human level; the summing readout does not inherit it.

This is not merely a subtraction. §46.2's surviving join is about whether the
two streams stay *differentiated*, and a readout that averages a shape-biased
stream with a texture-biased one toward the middle is precisely what
differentiated streams plus a summing FC predict. The refutation of leg (b) is
evidence **for** the differentiation account. Say it that way rather than
burying it.

### 54.4 Provenance strength: two mirrors, one number

This session pulled the experiment from **bethgelab/model-vs-human**
`raw-data/cue-conflict/` (10 observer files); §52 pulled it from
**rgeirhos/texture-vs-shape** `raw-data/style-transfer-512-nomask-experiment/`
(10 observers + 3 CNNs). Different repositories, different filename
conventions, two independently written parsers. Both return

    mean 0.9576, sd 0.0191, range 0.919-0.987, decision rate 0.803

to four decimals. Together with §52.2's reproduction of the published
AlexNet/GoogLeNet/VGG values, the human reference is now about as well
validated as it can be without re-running the experiment.

The duplicate download was **deleted** (`/scratch/byrong/encoding/datasets/
geirhos_cue_conflict/human_raw_data`, 38 KB) so that one copy and one citation
remain; `human_shapebias.py` reads §52's path. §52.4's citation requirement
covers both mirrors — cite the data release, and prefer the `rgeirhos` one that
§52.4 names.

### 54.5 Seed stability — behavioural only

Across 3 seeds the diet arm's magno shape bias is 0.8718 / 0.8918 / 0.8697,
**sd 0.0122**; the integrated readout 0.6922 / 0.6861 / 0.6464, **sd 0.0249**.
This is the first seed-variance estimate anywhere in the project and it is
reassuring.

**It licenses nothing about the other instruments.** The PR ratios (§47, §49,
§50) and the cortical numbers (§48) each rest on one checkpoint per condition,
and §50 made that load-bearing: §49.4 asks for "converge to within 2%" on a
0.037 gap that has never been compared against seed noise. Job **26413641**
(56 elements, `--cache-only`, seeds 1–2 of the three diet conditions plus
`dkl_base_seed2`) is measuring exactly that. Do not transfer 54.5's stability
to those tables before it lands.

---

## §55 EFFECT SIZES — the p-values are uninformative by construction, §48 is
independently reproduced, and one of its six tests is fragile

`tmp_analysis/effect_sizes.py`. No new compute; per-subject values from
`results/encoding`, the PR store and `fig1_data.npz`.

### 55.1 The problem this fixes

With n=8 the two-sided exact paired Wilcoxon has a **floor of p=.0078**,
attained whenever all 8 pairs agree in sign. Nearly every contrast in this
project attains it. PAPER_PLAN currently reports p=.0078 for the cortical
gradient, the PR set point, the in-silico cue-conflict dissociation, the
front-end orderings and the leg-(c) controls alike — so the reported statistic
**cannot distinguish an enormous effect from a merely consistent one**, and a
reader cannot tell which claims are load-bearing.

The matched-pairs rank-biserial correlation does **not** fix this: it is ±1.0
exactly when all pairs agree, so it saturates on the same contrasts. The
quantity that separates them is the standardized paired difference
**d_z = mean(diff) / sd(diff)**, reported with an interval on the raw
difference in its own units.

Worked demonstration: five headline contrasts all report the identical
p=.0078 while their |d_z| runs from **5.95 to 28.20**, a factor of 4.7. The
p-value is constant across that range by construction.

### 55.2 §48 reproduced exactly, from the raw encoding outputs

Recomputed independently from `results/encoding/*.npz` using §48's `norm`
statistic — median(Uₘ−Uₚ)/median(Uₘ+Uₚ), guarded at den>1e-4, as implemented
in `tmp_analysis/apply_repair.py`:

| arm | ρ (here) | ρ (§48) | hV4−V1v (here) | hV4−V1v (§48) |
|---|---|---|---|---|
| diet | −0.9637 | −0.964 | −0.8618 | −0.8618 |
| no diet | −0.5025 | −0.502 | −0.1013 | −0.1013 |
| swap | −0.2593 | −0.259 | −0.0265 | −0.0265 |
| parvo_highsf | −0.7801 | −0.780 | −0.1186 | −0.1186 |

**Eight of eight to four decimals.** §48 stands as computed.

One numerical note: §48.1 quotes the diet-vs-`parvo_highsf` ρ test as p=.0274.
This recomputation gives **p=.0312 (6/8)**. For n=8 the achievable exact
two-sided p-values are multiples of 1/128 (.0078, .0156, .0234, .0312), so
.0274 is not an attainable exact value and presumably came from a different
`scipy.stats.wilcoxon` mode or tie correction. The conclusion §48 drew from it
("above the floor") is unaffected; the number should be corrected to .0312.

**A trap worth recording.** Two stream-preference statistics live in this
codebase and are *not* interchangeable: `norm` (ratio of medians, above) and
`visualrois.py`'s `spi` (median of per-voxel ratios, with a per-voxel
(Uₘ+Uₚ)>1e-3 filter). They agree on the diet arm to four decimals and diverge
on the degenerate controls, where one stream contributes almost nothing unique
and per-voxel ratios saturate at ±1 (§48.2). This section's first draft used
the per-voxel one and appeared to contradict §48 before the cause was found.
Any future recomputation must state which statistic it uses.

### 55.3 The effect sizes, and what they reorder

**Cortical gradient, hV4−V1v (`norm`).** All three at p=.0078, 8/8:

| contrast | mean diff | 95% CI | d_z |
|---|---|---|---|
| diet vs swap | −0.8353 | [−0.895, −0.776] | **−11.70** |
| diet vs parvo_highsf | −0.7432 | [−0.814, −0.672] | **−8.74** |
| diet vs no diet | −0.7605 | [−0.867, −0.654] | **−5.95** |

The p-values are identical; d_z differs twofold. Note the ordering: the
**swap** control produces the largest, tightest separation and **no diet** the
smallest and widest — the reverse of the emphasis the write-ups give, where
no-diet is treated as the primary comparison. All three are overwhelming, but
if a single contrast is quoted it should be diet-vs-swap, not diet-vs-no-diet.

**PR trained ratio.** Six diet×no-diet contrasts, all p=.0078, 8/8, with d_z
from **+13.78 to +28.20** — the largest effects anywhere in the project. The
two diet-on-luminance arms against each other give d_z **+1.02**, CI
[+0.0067, +0.0672] on a raw gap of 0.037, p=.0391. That modest d_z is the
quantitative support for §49.4's "converge to within 2%" instruction, and it is
the right way to state it: not "indistinguishable", but a small standardized
difference against enormous between-condition ones.

**Other instruments.** In-silico cue-conflict magno−parvo, diet vs no diet:
+1.4236, d_z **+23.53**. The r²/NC cost: −0.0590, CI [−0.068, −0.050], d_z
**−5.42** — the cost is small in raw units but highly consistent, which is
exactly why it must be disclosed rather than dismissed as noise.

### 55.4 The fragile test — diet vs `parvo_highsf` on ρ

This is the weakest link in leg (c) and it should be stated as such:

| statistic | mean diff | 95% CI | d_z | p | sign |
|---|---|---|---|---|---|
| `norm` (published) | −0.1836 | **[−0.443, +0.076]** | −0.59 | .0312 | 6/8 |
| per-voxel ratio | −0.0734 | [−0.205, +0.058] | −0.47 | .3125 | 4/8 |

Three independent reasons for caution converge: the 95% CI on the raw
difference **includes zero** under the published statistic despite the rank
test's p=.0312; the effect flips to clearly non-significant under an equally
defensible statistic; and d_z is small (−0.59) where every other leg-(c)
contrast exceeds 5.

**Consequence for §48.3.** Its instruction *"Do not write that the controls
flatten the gradient"* is **reinforced** — `parvo_highsf` retains ρ ≈ −0.78
(`norm`) to −0.89 (per-voxel), nowhere near flat. But its further claim that ρ
*separates the diet arm from* `parvo_highsf` is **not supportable**: on ρ the
two are statistically indistinguishable once the CI and the statistic choice
are taken into account. What kills the `parvo_highsf` control is the
**gradient** (d_z −8.74, p=.0078, 8/8), not ρ.

The clean statement: `parvo_highsf` loses the *magnitude* of the cortical
gradient decisively while retaining its *monotonic ordering*. Both halves are
now backed by the appropriate statistic, and neither is being carried by a
test that a change of statistic overturns.

### 55.5 Disclosures

- n=8 makes the bootstrap CI on d_z coarse (percentile, 10k resamples of
  subjects, seed fixed). It is reported to show the ordering is not fragile,
  not as a precise interval.
- d_z is a within-subject standardized difference and is **not** comparable
  across instruments in different units. Compare within a block only; the
  PR block's d_z ≈ 20 and the gradient block's d_z ≈ 8 do not mean PR is
  "more significant".
- For the PR block, n=8 is 8 image samples from 8 subjects' NSD stimuli, not 8
  independent brains (§47.5, §49.4). The same caveat that limits the p-values
  limits these intervals.

### 55.6 What the manuscript should do

Report the **raw difference with its 95% CI** as the primary quantity, **d_z**
as the standardized effect, and give the p-value **once** with a footnote that
.0078 is the n=8 exact floor and that many contrasts attain it. Do not rank
claims by p. Do not substitute rank-biserial r — it saturates identically. And
flag the one contrast (55.4) whose significance depends on the statistic
chosen, rather than leaving it in a list of p-values where it looks like the
others.

## §56 OUTCOME C — the diet's shape bias does NOT buy robustness. It costs it.

Job array 26413592, **12/12 COMPLETED, zero failures**, ~2h per model on `main`.
`tmp_analysis/ood_verdict.py`. Scored against §53.4/§53.5 as written.

### 54.1 The result, in the statistic that matters

Relative accuracy (accuracy at severity ÷ clean accuracy), area under the
degradation curve, mean over **3 seeds** per condition:

| family | counts | diet | no diet | swap | parvo_highsf | diet − no diet |
|---|---|---|---|---|---|---|
| uniform-noise | yes | 0.687 | 0.729 | 0.745 | 0.767 | **−0.042** |
| high-pass | yes | 0.619 | 0.827 | 0.821 | 0.783 | **−0.208** |
| phase-scrambling | yes | 0.351 | 0.373 | 0.368 | 0.382 | **−0.023** |
| rotation | yes | 0.552 | 0.615 | 0.602 | 0.611 | **−0.063** |
| false-colour | yes | 0.842 | 0.938 | 0.969 | 0.874 | **−0.096** |
| low-pass | **NO — trained-in** | **0.441** | 0.356 | 0.349 | 0.386 | **+0.085** |
| contrast | **NO — trained-in** | 0.327 | 0.549 | 0.528 | 0.511 | −0.222 |

**The diet is less robust on every counted family**, in both statistics
(absolute agrees everywhere, and more strongly). Complete seed separation on
4 of 5; uniform-noise overlaps. §8 is satisfied for the negative direction.

### 54.2 H1 refuted. H2's "support" is vacuous and must not be claimed.

**H1 is refuted outright** — the diet degrades *faster*, not slower, on
uniform-noise, high-pass and false-colour.

**H2 must not be reported as supported.** The mechanical scorer marks it so
because the diet does degrade more under phase-scrambling, as predicted. But
§53.4 predicted a **crossover** — better under noise, worse under scrambling —
and there is no crossover. The diet is worse in the same direction on every
family, so the phase-scrambling result carries no information about shape
reliance; it is the same main effect. **Recording this because the temptation
to bank H2 as a hit is exactly the failure mode §39.3 and §41.2 are in this
document for.**

**H3 holds, 9/10.** `swap` and `parvo_highsf` — the two conditions with no
shape bias — pattern with no-diet throughout (the single exception is
`parvo_highsf` on false-colour, which is unsurprising given it is the condition
whose second stream carries luminance). The controls behave, so this is not a
quirk of one condition.

### 54.3 The positive control fires exactly where it should

The **only** family on which the diet is more robust is `low-pass`
(+0.085, complete separation) — the family we declared trained-in before
running, because the diet *is* a low-pass blur. The experiment detects
robustness precisely where robustness was literally trained in, and nowhere
else. That is the strongest available evidence that the null on the other five
families is a real null and not a broken measurement.

`contrast` is the instructive counterpart: the diet is much **worse** (−0.222)
despite carrying a contrast-sensitivity cut. A fixed CS cut at 48 months does
not confer tolerance to further contrast reduction; it consumes the headroom.

### 54.4 A robustness check on our own verdict, and it strengthens it

`dkl_bcs_seed1` has anomalous clean accuracy (0.336 against 0.503 and 0.479).
Dropping it entirely:

| family | diet (n=2) | no diet | |
|---|---|---|---|
| uniform-noise | 0.653 | 0.729 | overlap **resolves** — diet lower |
| high-pass | 0.638 | 0.827 | diet lower |
| phase-scrambling | 0.348 | 0.373 | diet lower |
| rotation | 0.560 | 0.615 | diet lower |
| false-colour | 0.843 | 0.938 | diet lower |

The one overlap in §54.1 traces entirely to that seed, and removing it makes
the verdict **cleaner**, not weaker. Disclosed rather than used: the headline
table keeps all three seeds.

### 54.5 The accuracy cost is larger than §18 recorded

112-way MiniEcoset test accuracy, 3 seeds: **diet 0.439 ±0.091 against no-diet
0.695 ±0.006** — a 37% relative cost, not the ~25% §18 estimated from 16-way
cue-conflict accuracy. `swap` (0.692) and `parvo_highsf` (0.696) sit with
no-diet, so the cost is specific to the diet on luminance.

Note also the **variance**: the diet arm spans 0.336–0.503 across seeds while
every other condition spans <0.02. The diet does not only cost accuracy, it
destabilises training. Neither fact was visible before this run because no
condition had ever been evaluated on its own test set at seed level.

### 54.6 Consequences, per §53.5 — and they are not negotiable

- **Drop the NeurIPS/NMI framing.** There is no machine-learning payoff. There
  is a machine-learning *cost*: less accurate, less stable, less robust. A
  venue that wants "and therefore the model is better" has no result here.
- **The paper is a neuroscience paper.** NHB is the target; eLife / PLoS CB /
  Journal of Vision the comfortable fallback.
- **This is reported, not buried.** It goes in the main text, not the supplement.
  §53.5 said "report as a negative in the supplement"; that was too weak. A
  reader who takes the shape-bias result seriously will assume robustness
  follows, because that is what the field assumes. Leaving the correction to a
  supplement would let the assumption stand.

### 54.7 What this actually buys the paper — a second dissociation

This is not merely a null. The paper now contains two dissociations of
properties the field treats as coupled:

- **§47.4** — shape bias and representational dimensionality factorise
  differently over identical inputs.
- **§54** — shape bias and OOD robustness come apart: a developmental route to
  shape bias produces a strongly shape-biased stream (probe +0.819, behaviour
  0.878 against a human 0.958) that is *less* robust than a texture-biased
  control on every corruption it was not trained on.

Stated carefully, this does **not** contradict Geirhos et al. They showed that
*Stylized-ImageNet training* yields models that are both shape-biased and more
robust. We show the coupling is not intrinsic to shape bias: **a different
route to comparable shape bias delivers none of the robustness.** The route
matters, and "increase shape bias to get robustness" does not generalise across
the manipulations that produce it.

That is a real, positive, falsifiable claim, and it is the honest thing this
experiment bought. It also fits the through-line the project keeps finding:
properties assumed to travel together do not.

---

## §57 PRE-REGISTRATION — error consistency against the 10 human observers

Written and committed BEFORE the analysis is run. §53's OOD pre-registration is
the template: the prediction, the decision rule, and the outcome that would
damage the paper are all fixed in advance.

### §57.1 Why this, and why now

The draft's title claims **"human-like"** shape bias. The evidence for that word
is currently a *rate*: a shape-bias scalar near the human one. Geirhos et al.
(2020, "Beyond accuracy") showed precisely why a matching rate is weak evidence
— two decision-makers can produce the same shape-bias fraction while agreeing
on no particular image beyond chance. Their remedy is **error consistency**,
Cohen's kappa on trial-by-trial agreement, which subtracts exactly the agreement
that the two marginal rates force.

An NHB reviewer will ask this. Better that we ask it.

### §57.2 A comparison error found while setting this up — and corrected

The abstract compares model **0.878** to human **0.958**. Checking
`results/cue_conflict/dkl_bcs_seed0.json`: 0.872 (seed 0) is the `magno_only`
readout — ONE STREAM with the other lesioned. The whole model's shape bias is
`full` = **0.692**. A whole human observer is not a lesioned stream, so
0.878-vs-0.958 flatters us by comparing different units.

The 2x2 of §2.1 is internally consistent (all four cells are stream readouts,
which is the right unit for an *interaction* about carriers), so §2.1 stands.
What must change is the HUMAN comparison, which has to be made at the whole-
system level, or explicitly labelled as a stream-vs-observer comparison. Both
numbers are now to be reported wherever the human benchmark appears.

### §57.3 The metric

Per image, each decision-maker chooses shape or texture; images decided as
neither are dropped for that pair. For a model-human pair with observed
agreement c_obs and marginal shape rates p_m, p_h:

    c_exp  = p_m * p_h + (1 - p_m) * (1 - p_h)
    kappa  = (c_obs - c_exp) / (1 - c_exp)

Ceiling: human-human kappa over all 45 observer pairs. A model kappa is
meaningful only relative to that ceiling.

Computed for BOTH readouts (`full` and `magno_only`), reported for both. n=3
seeds per condition; four conditions; 10 observers.

### §57.4 Predictions and decision rule

- **H1 (the one that matters).** Diet > no-diet in model-human kappa.
  SUPPORTED only on complete separation at the seed level (all 3 diet seeds
  above all 3 no-diet seeds) in BOTH readouts, per the §8 two-statistics rule.
- **H2.** Diet kappa is a non-trivial fraction of the human-human ceiling.
  Descriptive; no threshold pre-set, because none is defensible in advance.

### §57.5 The outcome that damages the paper, named in advance

If every model's kappa is near zero — which is what Geirhos found for standard
CNNs against humans — then no model here matches human trial-by-trial structure,
and **"human-like" in the title and abstract is not earned at the trial level**.
The commitment, made now: in that case the word is removed from the title, the
claim is restricted explicitly to the aggregate rate, and the null is reported
in the main text, not the supplement. The paper would still have its
interaction, its mechanism, its cortex result and its two dissociations; it
would not have a trial-level human-likeness claim.

If H1 is supported, the diet gains a genuinely human-facing result of the kind
NHB weights most heavily, and it belongs in the abstract.
