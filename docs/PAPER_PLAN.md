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
| V1d / V1v | +0.0853 / +0.0831 | 8/8 |
| V2v / V2d | +0.0764 / +0.0573 | 8/8 |
| PHC2 | +0.0546 | 8/8 |
| V3d / V3A / PHC1 / V3v | +0.0404 / +0.0397 / +0.0377 / +0.0355 | 8/8 |
| **TO2 / TO1 (hMT+)** | +0.0209 / +0.0181 | 6/8, 7/8 |
| VO2 / VO1 | +0.0105 / −0.0016 | 7/8, 4/8 |
| LO2 | −0.0100 | 2/8 |
| **hV4** | **−0.0123** | 2/8 |

Values recomputed after §69's reference refit; see §70.6. Every entry is
unchanged to the precision originally reported except LO2's subject count
(1/8 → 2/8) and third-decimal rounding on the four largest early-visual ROIs.

- **hMT+ is magno-preferring: +0.0195, p = .0234, 7/8 subjects, dz = +1.30.**
  §A3.5 wrote the dorsal/MT test off as a limitation of static stimuli. It is
  testable and it comes out in the predicted direction. It should be promoted
  from "limitation" to a reported (secondary) result.
- **The colour prediction fails at VO: +0.0045, p = .4609.** hV4 remains the
  only parvo-preferring area. The hMT+ − VO interaction is **not**
  significant (+0.0150, p = .1094), so this is *not* a second double
  dissociation. Report the null.
- **The 20-ROI profile is highly reproducible**: leave-one-subject-out
  Spearman **ρ = +0.794** (range .589–.896). Useful reviewer-proofing: the map
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

`fig2_ventral_gradient.pdf/png` and the caption must be regenerated once 26399760
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

Until then `fig2_ventral_gradient.{png,pdf}` and `fig2_ventral_gradient_caption.md` must not be used
in any draft. A warning banner has been written into `fig2_ventral_gradient_caption.md` itself so
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

The quarantine banner is removed from `fig2_ventral_gradient_caption.md`. The caption now
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

`fig2_ventral_gradient.{png,pdf}` and its caption remain. That figure is the
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

---

## §58 SEED REPLICATION OF THE PR SET POINT — the level is noise, the side is not

Job 26413641 (56 cache-only tasks, all COMPLETED) -> `tmp_analysis/pr_seeds.py`.
n=8 subjects x 3 seeds x 4 conditions, NIMG=1000. This is the first time the
seed axis has been measured on any instrument in this project.

### §58.1 The table

| arm | front end | seed 0 | seed 1 | seed 2 | seed range |
|---|---|---|---|---|---|
| diet / luminance | 1.402 | 1.603 | 2.507 | 1.823 | **0.904** |
| diet / parvo_highsf | **0.586** | 1.640 | 1.497 | 1.664 | 0.166 |
| diet / red−green (swap) | 1.909 | 2.225 | 4.383 | 1.995 | **2.388** |
| no diet | 1.634 | 0.947 | 0.988 | 0.914 | 0.074 |

### §58.2 Q1 — the side of parity replicates 12/12

Every one of the nine diet seeds is above parity in **8/8 subjects**. All three
no-diet seeds are below parity in the mean (subject-level 7/8, 5/8, 8/8 below).
Seed-level diet-vs-no-diet is **complete separation for all three diet arms**,
p=0.100 — the exact two-sided 3v3 floor, i.e. the strongest value the test can
return. The §2.2 claim, as currently worded, survives without amendment.

Note the honest wrinkle: no-diet seed 1 has only 5/8 subjects below parity and a
mean of 0.988. "Collapses to parity" is the accurate phrase; "falls below parity"
would not be.

### §58.3 Q2 — two claims die here, one of them mine and one of them §50's

**§49.4's "the two diet-on-luminance arms converge to within 2%" is dead.** The
gap was 0.037. The within-arm seed range on that same arm is **0.904**, ~24x
larger. The gap was never distinguishable from seed noise, and no run had
measured that noise. Withdrawn.

**§50's "the set point is channel-specific" is not established either.** That
claim rested on swap 2.225 against luminance 1.603, both n=1. Across seeds,
swap spans 1.995–4.383 and luminance spans 1.603–2.507 — overlapping ranges.
The swap mean is higher (2.867 vs 1.978) and may well be a real effect, but
n=3 seeds cannot separate them. Not refuted; **not established**.

This is the symmetric outcome. §50 correctly caught that my §49 was selective;
the seed axis now shows that the counter-claim rested on the same n=1 footing.
What survives is what both accounts agreed on and neither tested: the SIDE.

### §58.4 Why this strengthens rather than weakens the paper

A claim about a *level* would have been unreproducible — a reviewer running one
more seed would have broken it. The claim the draft actually makes is about
which side of parity training ends on, and that is now replicated across three
seeds in four conditions with complete separation. The restructure that demoted
the level to mechanism was made before this ran; the result vindicates it.

Consequence for the draft: every "after training" number in §2.2 must be
reported as a seed range, never as a condition constant. No compression factor,
no set point, no "converges to" anywhere in the paper.

### §58.5 The developmental-schedule check (`tmp_analysis/schedule_check.py`)

Our diet is a STATIC front end at age 48 months, not a trajectory. The most
likely reviewer objection is that nothing in the paper develops. `thesis_dvd_trans`
— the BA checkpoint, trained on an actual progressive age transition, in a
different codebase — lands at **1.158, 8/8 subjects above parity**: the same
side as every diet arm, at a level below all of them.

So the schedule does not change the side, which is the claim. Reported as a
convergent check across independent implementations (§45.6), not as a matched
arm, because the codebase and architecture differ.

---

## §59 ERROR CONSISTENCY — OUTCOME: H1 REFUTED, and §57.5 is now binding

`tmp_analysis/error_consistency.py`. Pre-registered in §57 before running.

### §59.1 Validation, run before the result was read

- The 10 observers reproduce the **published 0.958** shape bias exactly
  (mean 0.958, range 0.919–0.987). The parser is right.
- kappa returns **+1.000** for identical raters, **−0.043** for independent
  raters at a matched 0.9 rate, negative for anti-correlated. The metric is right.
- A simulated model matching a human's rate exactly but choosing independently
  scores **−0.036**. Rate-matching alone earns nothing, which is the point.

### §59.2 The result

Human–human ceiling: **kappa = 0.234** (45 pairs, sd 0.095).

| | shape bias | kappa vs human | % of ceiling |
|---|---|---|---|
| diet, whole model (3 seeds) | 0.646–0.692 | 0.023–0.032 | 10–14% |
| no diet, whole model | 0.394–0.433 | 0.031–0.037 | 13–16% |
| diet, magno stream | 0.870–0.892 | −0.015–0.036 | −6–15% |
| no diet, magno stream | 0.398–0.434 | 0.036–0.041 | 15–18% |
| *AlexNet (published)* | *0.429* | *0.032* | *14%* |
| *GoogLeNet (published)* | *0.312* | *0.025* | *11%* |
| *VGG (published)* | *0.172* | *0.017* | *7%* |

**H1 is refuted, in both readouts.** The diet-trained model is not more
consistent with human observers than the no-diet control; it is numerically
*less* so on the stream readout (0.009 vs 0.039). No model of ours exceeds
AlexNet.

### §59.3 What this means, stated plainly

Our diet stream carries **twice AlexNet's shape bias** (0.872 vs 0.429) and
**no more agreement with human decisions** (0.036 vs 0.032). Matching the human
shape-bias rate buys nothing at the trial level. The models that Geirhos showed
were texture-biased and the model we built to be shape-biased are equally far
from human decision structure.

### §59.4 Honouring §57.5

The commitment made before running: if kappa is near zero, **"human-like" comes
out of the title**, the claim is restricted to the aggregate rate, and the null
goes in the main text. All three are now done. The title drops the phrase; §2.1's
human comparison is relabelled as a rate comparison between a stream and an
observer (§57.2); the null becomes §2.6.

### §59.5 The caveat that does NOT rescue it

Kappa is compressed when marginals are skewed, and humans are at 0.958. That is
real — it is why the human–human ceiling is only 0.234 rather than something
near 1. It is handled by reporting every model as a **percentage of that
ceiling**, which is the same skew. It does not explain why the diet model scores
*below* its own no-diet control.

### §59.6 Why this is a result and not a wound

It converts a soft claim ("our model is human-like") into a sharp one: **shape
bias, the field's most-cited human-alignment metric for vision models, does not
track human decision structure at all.** Across six models spanning 0.17 to 0.89
shape bias — a 5x range — human agreement is flat at 7–18% of ceiling. That is
the third and cleanest of the paper's dissociations, and unlike the other two it
directly concerns human behaviour, which is the axis NHB weights most.

---

## §60 WHY §58's SEED SPREAD HAPPENED — the flagship arm never converged, and
§10.1's guard is necessary but not sufficient

`tmp_analysis/pr_seed_variance.py` plus the checkpoint histories. Additive to
§58, which measured the spread; this section explains it and records three
consequences §58 does not cover. §58's table is independently reproduced here
to four decimals from a separate script and is not restated.

### 60.1 §10.1's convergence guard passes, and is still not enough

§10.1 installed the rule *"any probe that reads `checkpoint_last.pt` must first
assert `history.json` has the full epoch count"*, after a swap model was probed
at ~epoch 28 of 150. That assertion **passes for all twelve checkpoints used in
§58**: every one reports epoch 149 with 150 history entries.

It is nevertheless insufficient. A full epoch count does not mean converged, and
`dkl_bcs` — the flagship diet arm, the model behind legs (a), (b) and (c) — is
still oscillating at epoch 149:

| condition | sd(val_top1) over epochs 130–149 | best − last | best epoch |
|---|---|---|---|
| **diet / luminance** | **0.0417 – 0.0709** | **0.060 / 0.246 / 0.136** | **120, 59, 141** |
| diet / parvo_highsf | 0.0054 – 0.0085 | 0.000 / 0.003 / 0.016 | 149, 146, 143 |
| diet / red−green | 0.0060 – 0.0095 | 0.015 / 0.012 / 0.008 | 137, 93, 128 |
| no diet | 0.0065 – 0.0098 | 0.008 / 0.013 / 0.003 | 145, 142, 138 |

`dkl_bcs` is the **only** condition that has not settled: its late-training
scatter is ~8× every other arm's, its final epoch sits 0.06–0.25 below its own
best, and its best epoch lands at 120 / 59 / 141 rather than at the end. Every
other condition finishes within 0.016 of its best.

**The rule §10.1 should have installed**, and which should now replace it: assert
the full epoch count *and* that sd(val_top1) over the last 20 epochs is below a
stated threshold, and record both in any output the probe writes.

### 60.2 So §58's spread is checkpoint timing, not learned-solution variability

> **REFUTED by §61.3 — do not cite this subsection.** Direct measurement of the
> epoch axis (job 26416534) shows PR moves only 0.100 across epochs within
> `dkl_bcs_seed0` while that run's accuracy swings 0.230, so PR does not track
> accuracy within a run and the Spearman −1.00 below was a three-point
> coincidence. §58's spread is **genuine seed-to-seed variability**, not
> checkpoint timing. §58's conclusion is unaffected; the explanation offered
> here was wrong. Retained for the record and for the accuracy table in 60.3,
> which stands.

Two facts separate those two readings, and they point the same way.

**(i) What the condition achieves is stable; where the last checkpoint lands is
not.** Across the three diet/lum seeds:

| | seed 0 | seed 1 | seed 2 | sd |
|---|---|---|---|---|
| val_top1 at epoch 149 | 0.5273 | 0.3470 | 0.4595 | **0.0911** |
| val_top1 at that run's best | 0.5877 | 0.5929 | 0.5957 | **0.0041** |

The *best* accuracies agree to 0.004 — as tight as any converged arm. Only the
*last* accuracies scatter, by 22×  more. The three seeds are therefore not
finding different solutions; they are being sampled at different phases of the
same oscillation.

**(ii) The PR ratio ranks inversely with how well that particular checkpoint
scored.** Within diet/lum, Spearman(PR ratio, val_top1) = **−1.00** (n=3):

| run | PR ratio | val_top1 |
|---|---|---|
| `dkl_bcs_seed0` | 1.603 | 0.5273 |
| `dkl_bcs_seed2` | 1.823 | 0.4595 |
| `dkl_bcs_seed1` | 2.507 | 0.3470 |

n=3 makes this suggestive, not established. But it is the wrong direction for
§58's spread to be read as seed-to-seed variability in a learned target: the
ratio is tracking checkpoint quality, and the worst checkpoint gives the most
extreme ratio. §49.2's "set point at 1.62" came from `seed0`, which is the
*best* of the three — i.e. the arm's reported level was, unknowingly, the
best-case draw.

The same ordering holds in diet/rg (Spearman −1.00) and no diet (−1.00), where
the accuracy spread is only ~0.010, so the effect is not confined to the
unstable arm. It does **not** hold in diet/parvo_highsf (−0.50), the fourth
arm — stated because with n=3 seeds a −1.00 is one of only three possible
values and three arms out of four agreeing is weak evidence, not a result.
Across all 12 runs Spearman(PR, val_top1) = −0.65.

### 60.3 §10.2's accuracy table completed, and the cost halves under a fair
checkpoint

§10.2 recorded `dkl_bcs` as "~0.43–0.48 at ep 146 (still training)" and was
never revisited. Completed, val_top1 on held-out miniEcoset (112 classes):

| condition | seed 0 | seed 1 | seed 2 | mean ± sd | vs no diet |
|---|---|---|---|---|---|
| no diet | 0.6998 | 0.6982 | 0.7055 | 0.7012 ± 0.0038 | — |
| diet / parvo_highsf | 0.7054 | 0.6991 | 0.6948 | 0.6998 ± 0.0053 | **−0.001** |
| diet / red−green | 0.6920 | 0.6895 | 0.6996 | 0.6937 ± 0.0053 | **−0.008** |
| diet / luminance | 0.5273 | 0.3470 | 0.4595 | 0.4446 ± 0.0911 | **−0.257** |

Two corrections to §10.2 follow.

**The ~25-point cost is not a cost of "the diet".** It is a cost of the diet
applied to luminance *only*. Moving the same transform to red−green (−0.008) or
restoring raw luminance into stream 2 (−0.001) costs essentially nothing. The
plausible mechanism, consistent with but not established by these numbers: the
diet blurs the sole channel carrying high spatial frequency, and red−green is
inherently low-SF, so `dkl_bcs` alone is starved of high-SF information
altogether. `parvo_highsf` restores it and recovers the accuracy exactly.

**And the size of the cost depends on the checkpoint.** Against best-epoch
accuracies (0.5921 vs no diet's 0.7089) the gap is **−0.117**, not −0.257. So
§10.2's "~25 points" is a last-checkpoint artefact of an oscillating run; the
defensible figure is ~12 points, and the paper should quote whichever it uses
with the checkpoint rule stated. This matters directly for §10.2's own warning
that every `dkl_bcs` claim must be defended against "this model is simply
worse" — the defence is roughly twice as strong as recorded.

### 60.4 §49.2's compression figure does not merely shrink — it inverts

§58.3 withdrew "converge to within 2%" and correctly declined to establish
§50's channel-specific level. Neither computes what the compression figure
becomes with all nine diet checkpoints, which is worth stating because §49.2's
22× is the number that reached the draft abstract:

| reference | front-end span | trained span (9 diet ckpts) | ratio |
|---|---|---|---|
| untrained | 1.324 | 2.885 | **0.46× — training EXPANDS the spread 2.2×** |
| gabor | 2.803 | 2.885 | **0.97× — no change** |

Trained diet checkpoints span **1.497 – 4.383**, wider than the 0.586 – 1.909
front-end range that fed them. So the direction of §49.2's claim is wrong, not
just its magnitude: on this instrument, training does not pull heterogeneous
front ends toward a common level. **No compression factor, in either direction,
should appear in the paper** — the honest statement is that the level is not
measurable from these checkpoints at all.

### 60.5 The exposure this creates for leg (c), and the control now running

> **ANSWERED by §61 — job 26416534 has landed.** Outcome: the diet arm is
> epoch-stable (range 0.100) and the *control* arm is not (0.558), inverting
> the expectation set out below. Leg (c)'s exposure is on the no-diet baseline,
> not on `dkl_bcs_seed0`. The pre-registration is retained unedited so the
> prediction can be read against the result.

Every cortical number in §48 and every variance-partitioning result in legs (a)
and (c) is computed from **`dkl_bcs_seed0_trained` — one draw from the one
oscillating run**. §58 did not reach this because it measured PR only.

The features themselves have never been checked for epoch stability, so the
exposure is unquantified in either direction. **Job 26416534**
(`slurm/cache_epoch_jitter.sh`, 40 elements, cache-only) measures it:
`dkl_bcs_seed0` at epochs 100 / 125 / 150 and `dkl_base_seed0` at 100 / 125,
all 8 subjects. Pre-registered reading, before the result:

- If PR moves across epochs within `seed0` as much as it moves across seeds
  (range ≈ 0.9), the level is a checkpoint-timing artefact, §49.2/§49.4 are
  withdrawn rather than re-worded, and every leg-(c) number needs an
  epoch-stability disclosure.
- If PR is flat across epochs, §58's spread is genuine seed-to-seed variability
  in the learned solution and is reportable as such.
- `dkl_base_seed0` is the control: it is converged, so its PR must be flat. If
  it is not, the instrument is epoch-sensitive and none of this is about the
  diet.
- `dkl_bcs_seed0` @ epoch 150 is a **reproduction check**: it must return 1.603,
  the value already cached from `checkpoint_last.pt`. If it does not, those two
  files are not the same weights and nothing else here is believed until that
  is resolved.

### 60.6 A note on §58.2's seed-level test

§58.2 reports the side-of-parity result as a 3-vs-3 seed-level comparison at
p=0.100, "the exact two-sided 3v3 floor". That is a sound test but the weakest
available framing of the strongest result in the section. Run instead within
each seed, so every p-value keeps n=8 subject pairs, the same claim gives:

- **27/27** diet × no-diet seed pairings at p=.0078 with 8/8 subjects positive,
  smallest mean difference **+0.509**;
- all three diet arms above parity in **8/8 subjects in all 3 seeds** (72/72
  model-subject cells), no-diet above parity in only 1, 3 and 0 of 8.

This needs no pooling and inflates no n — it is sign-consistency across 27
independent subject-level tests. Prefer it to the 3v3 p=0.100. Conversely,
`diet/rg` vs `diet/lum` — §50's channel-specific claim — holds in only **6/9**
pairings and reverses in 3, which is the same conclusion §58.3 reached by
inspecting overlapping ranges, now with a test attached.

---

## §61 THE EPOCH-JITTER CONTROL — §60.2's diagnosis is REFUTED, and the
control arm turns out to be the unstable one

`tmp_analysis/pr_epoch_jitter.py`, job 26416534 (40 cache-only tasks, all
COMPLETED). Read against the criteria pre-registered in §60.5 before the caches
existed.

### 61.1 Reproduction check: PASS, exactly

`dkl_bcs_seed0` @ `checkpoint_150.pt` returns **1.6030 ± 0.0318**; the value
already cached from `checkpoint_last.pt` is **1.6030 ± 0.0318**. Maximum
per-subject difference **0.000000**. The two files are the same weights, the
new extraction path reproduces the old one bit-for-bit, and §49.2's published
1.603 is confirmed a third time.

Front ends were separately verified from the `prov_*` keys: the epoch caches
carry `lum_blur/lum_cs/transform_target/parvo_highsf` identical to the
originals for both arms. So nothing below is a front-end mismatch (§22).

### 61.2 The result, and it is the opposite of what §60.5 expected

> **OVERSTATED — corrected by §62.3.** The ranges below were measured on
> epochs 100–150 only. Over the full 25–150 trajectory the diet arm's range is
> **0.356**, not 0.100, so the dissociation is **1.6×, not 5.6×**. "The arm
> with unstable accuracy has a stable representation" is too strong and should
> be struck: both arms' representations move substantially, and neither is
> stable enough for one checkpoint to characterise a level. The direction of
> the effect stands; the magnitude below does not.

| epoch | `dkl_bcs_seed0` (diet, NOT converged) | `dkl_base_seed0` (no diet, converged) |
|---|---|---|
| 100 | 1.6204 ± 0.0415 | 1.5046 ± 0.0562 |
| 125 | 1.7033 ± 0.0551 | 1.0185 ± 0.0430 |
| 150 | 1.6030 ± 0.0318 | 0.9466 ± 0.0352 |
| **epoch range** | **0.1003** | **0.5580** |

Set against each arm's accuracy over the same epochs:

| arm | val_top1 spread, ep100–149 | PR spread, ep100–150 |
|---|---|---|
| `dkl_bcs` (diet) | **0.2302** | 0.1003 |
| `dkl_base` (no diet) | 0.0441 | **0.5580** |

**The two are inverted.** The arm with unstable accuracy has a stable
representation; the arm with converged accuracy has a representation still
migrating.

### 61.3 CORRECTION — §60.2's checkpoint-timing diagnosis is refuted

§60.2 argued that §58's seed spread was checkpoint timing rather than genuine
seed variability, on two grounds: `dkl_bcs`'s accuracy oscillates, and its PR
ratio ranked inversely with final accuracy across three seeds (Spearman −1.00).
Direct measurement refutes this.

Within `dkl_bcs_seed0`, accuracy swings **0.2302** across epochs 100–149 while
PR moves only **0.1003**. PR is therefore *not* tracking accuracy within a run,
and the cross-seed Spearman −1.00 was a coincidence of three points — which
§60.2 hedged as "suggestive, not established", but the hedge is not enough: the
claim is now actively refuted and should be struck, not softened.

**The corrected reading.** Epoch jitter within one seed is **0.100 = 11%** of
the 0.904 seed spread. So §58's spread is **genuine seed-to-seed variability in
the learned solution**, not an artefact of which epoch the checkpoint caught.
§58's own conclusion stands unchanged and is now positively supported rather
than merely asserted; it was §60.2's proposed *explanation* that was wrong.

`dkl_bcs`'s trained level is a real property of each training run that varies
between runs. That is worse for §49.2 than §60.2 suggested, not better: a
checkpoint artefact could have been fixed by checkpoint selection, whereas
genuine run-to-run variability cannot be fixed at all. §49.2/§49.4 stay
withdrawn.

### 61.4 The control "failed", and the failure is the finding

§60.5 pre-registered `dkl_base_seed0` as the control on the grounds that it is
converged, so its PR must be flat; if not, "the instrument is epoch-sensitive
and none of this is about the diet". Its PR is not flat — range 0.5580 — so by
the letter of the rule the control failed.

The rule was wrong, and it is worth recording why, because the mistake is
reusable. It equated *accuracy* convergence with *representational*
convergence. `dkl_base` has flat accuracy (sd 0.0052–0.0098 over trailing
10-epoch windows) and a representation moving from clearly magno-dominant to
parity over the same epochs. **Accuracy convergence does not imply
representational convergence**, and in this project the two are anti-correlated
across conditions.

This does not mean the instrument is broken: the PR measure reproduces to
0.000000 on identical weights (61.1) and is stable across epochs on the diet
arm. It means the *quantity being measured is still changing* in the no-diet
network at the point we measure it.

Consequence for §60.1, which proposed strengthening §10.1's guard to "full
epoch count **and** sd(val_top1) over the last 20 epochs below a threshold":
that is still necessary but still not sufficient. `dkl_base` passes both and is
representationally unconverged. **The guard must be on the reported measure
itself** — extract at two or more epochs and report the movement — because no
accuracy-based criterion detects this.

### 61.5 Leg (c)'s exposure is on the BASELINE, not the diet arm

§60.5 flagged leg (c) as exposed because every §48 cortical number comes from
`dkl_bcs_seed0`. That specific worry is now answered: the diet arm is
epoch-stable (0.100), so its features are epoch-safe to within ~6% of level.

The exposure is on `dkl_base_seed0`, the matched no-diet control, which appears
in every leg-(a) and leg-(c) contrast as the comparison condition. At epoch 100
its ratio is **1.5046 — above parity, and within 0.12 of the diet arm's
1.6204**. Had the baseline been extracted 50 epochs earlier, the headline
diet-vs-no-diet separation would have been near zero.

Both arms are in fact extracted at epoch 150, so the published contrast is
matched and fair. But it must be disclosed as **a statement about networks
compared at a fixed training duration, not an asymptotic property**. The
no-diet arm is still descending at 150, though sharply decelerating (−0.486
from ep100→125, −0.072 from ep125→150), so it is close to an asymptote near
~0.93–0.95 and longer training would if anything widen the contrast. That is
the honest form of the caveat: the direction of the residual drift favours the
claim, and the claim should not be stated as if drift were absent.

### 61.6 A replacement mechanism for the set point, and the run that tests it

> **SUPERSEDED by §62.6 — job 26417753 has landed.** Both falsifiers cleared,
> but the no-diet trajectory is **non-monotone** (it drops, climbs back to
> within 0.12 of its front end by ep75, then falls twice), so the "training
> **erodes** the dominance" wording below describes a smooth process that did
> not occur and must not be used. The reworded claim — that the diet flips the
> **sign** of training's effect on stream asymmetry, with the arms separated
> 8/8 at every epoch 25–150 and moving in opposite directions from their own
> front ends — is in §62.6.

§58 killed the set-point account and §60.4 showed the compression figure
inverts, leaving the paper without a mechanism for what the diet does to stream
specialisation. These trajectories suggest one, and it fits the surviving
side-of-parity result rather than competing with it:

> **Both arms begin magno-dominant. Training erodes that dominance in the
> no-diet network; the diet prevents the erosion.**

Front ends (§58.1): no diet **1.634** untrained, diet **1.402**. The no-diet
arm then runs 1.634 → 1.505 (ep100) → 1.019 (ep125) → 0.947 (ep150), while the
diet arm holds 1.40 → 1.62 → 1.70 → 1.60. On this account the diet is not
installing a level — which is why no set point was ever found — it is
protecting an initial asymmetry from being trained away.

**This is a hypothesis on three points from one seed per arm, and must not be
written up as a result yet.** Job 26417753 (`slurm/cache_epoch_trajectory.sh`,
48 elements, cache-only) extracts epochs 25/50/75 for both arms, giving 6-point
curves. Falsifiers stated in the submit script before it ran: if `dkl_base` is
flat from 25 to 100 and only drops afterwards, the "erosion during training"
reading is wrong and the drop belongs to late training (LR schedule) rather
than to the diet contrast; if `dkl_bcs` is not flat over 25–75, its
epoch-stability at 100–150 is a late-training coincidence and the dissociation
in 61.2 is not real.

Scope disclosure that travels with any result: **one seed per arm**. §58 showed
the trained level varies across seeds on the diet arm by 0.904, so the
trajectory *shape* may vary too. Seed generality of the shape is untested and
must not be claimed.

---

## §62 THE FULL EPOCH TRAJECTORIES — the falsifiers clear, the *shape* story
does not, and §61.2's dissociation was measured on the wrong third of training

`tmp_analysis/pr_epoch_trajectory.py`, job 26417753 (48 cache-only tasks, all
COMPLETED). Six epochs per arm plus each arm's own untrained front end, n=8
subjects, one seed per arm.

### 62.1 The trajectories

| arm | untrained | ep25 | ep50 | ep75 | ep100 | ep125 | ep150 |
|---|---|---|---|---|---|---|---|
| diet (`dkl_bcs_seed0`) | 1.402 | 1.752 | **1.959** | 1.733 | 1.620 | 1.703 | 1.603 |
| no diet (`dkl_base_seed0`) | **1.634** | 1.156 | 1.237 | 1.513 | 1.505 | 1.019 | **0.947** |

Subjects above parity, no-diet arm: 8/8 at epochs 25–100, **6/8** at 125,
**1/8** at 150.

### 62.2 Both pre-registered falsifiers clear — and the hypothesis still needs
rewording

F1 (no-diet flat to ep100, dropping only after) **not triggered**: the arm
moves −0.478 to ep25 then +0.357 to ep100, so the change is not confined to
late training. F2 (diet arm not flat over 25–75) **not triggered**: its range
there is 0.226, under the 0.3 threshold.

But the falsifiers were the wrong instrument for the claim, and §61.6's wording
does not survive contact with the curve. **The no-diet trajectory is
non-monotone**: 1.634 → 1.156 → 1.237 → 1.513 → 1.505 → 1.019 → 0.947. It
drops sharply, climbs *back* to within 0.12 of its front end by epoch 75, sits
flat to 100 (p=0.84), then falls twice. 71% of the net decline happens in the
single ep100→125 step.

So "**training erodes the dominance**" — §61.6's phrasing — describes a smooth
process that did not occur. **That wording must not be used.** The net movement
is downward and highly reliable, but the path is not a progressive erosion.

Also recorded as a design error: 62's own "share of total decline before epoch
100" metric returned **−167%**, which is meaningless. A share-of-total is
undefined when the series is non-monotone and the early change has the opposite
sign to the net. The falsifier was coded for a monotone curve and silently
produced a nonsense number rather than failing; the threshold test around it
happened to be robust to this, but the metric should not be reused.

### 62.3 CORRECTION — §61.2's dissociation is 1.6×, not 5.6×

§61.2 reported that the diet arm is epoch-stable (range 0.100) while the
no-diet arm is not (0.558), calling the two "inverted". That was measured on
epochs 100–150 only, i.e. the last third of training. Over the full 25–150
range:

| arm | range, ep100–150 (§61.2) | range, ep25–150 (here) |
|---|---|---|
| diet | 0.100 | **0.356** |
| no diet | 0.558 | **0.566** |
| ratio | 5.6× | **1.6×** |

The no-diet figure barely changes; the **diet** figure more than triples. The
diet arm is not epoch-stable — it is *less* epoch-unstable, by a factor of 1.6
rather than 5.6. §61.2's "the arm with unstable accuracy has a stable
representation" is therefore too strong and should be struck; the accurate
statement is that both arms' representations move substantially across
training, and neither is stable enough for a single checkpoint to characterise
a level.

This weakens, but does not by itself remove, the set-point reading: within the
diet arm's *own single run* the ratio spans **1.603–1.959**, which reaches to
within **0.036** of the bottom of the range §50 attributed to a different
channel (swap, 1.995–4.383 across seeds). One run therefore traverses almost
the whole distance that was being read as a channel difference.

**Correction to an earlier draft of this subsection**, which said the two
ranges "overlap": they do not — 1.959 < 1.995, adjacent with a 0.036 gap. The
weaker statement above is what the numbers support. See §63 for the
arm-by-arm audit, which finds the swap-vs-`parvo_highsf` ordering intact and
only the swap-vs-luminance ordering unsupported.

### 62.4 What is robust, and it is better than the shape story

Two statements need no trajectory shape and no seed generality.

**(i) The arms are separated at every epoch measured**, 8/8 subjects,
p=.0078 throughout:

| epoch | diet | no diet | diff | d_z |
|---|---|---|---|---|
| 25 | 1.752 | 1.156 | +0.596 | +11.30 |
| 50 | 1.959 | 1.237 | +0.722 | +13.78 |
| 75 | 1.733 | 1.513 | +0.220 | +4.12 |
| 100 | 1.620 | 1.505 | +0.116 | +2.41 |
| 125 | 1.703 | 1.019 | +0.685 | +20.01 |
| 150 | 1.603 | 0.947 | +0.656 | +28.20 |

The separation is present from epoch 25, never reverses, and narrows to
d_z +2.41 at its weakest (epoch 100) — still 8/8. This is a duration-robust
claim, unlike anything about levels.

**(ii) The two arms move in opposite directions from their own front ends**,
each against its provenance-matched untrained reference:

| arm | untrained → ep150 | change | p | subjects down |
|---|---|---|---|---|
| diet | 1.402 → 1.603 | **+0.201** | .0078 | **0/8** |
| no diet | 1.634 → 0.947 | **−0.687** | .0078 | **8/8** |

Note the crossover: the no-diet arm **starts higher** (1.634 vs 1.402) and
**ends lower** (0.947 vs 1.603). The diet does not begin with a magno-dominance
advantage — it begins with less of one — so the epoch-150 separation cannot be
attributed to the front end. That is the strongest form of the leg-(a) argument
available in this project, and it does not depend on trajectory shape, on a set
point, or on the level being reproducible.

### 62.5 A fragility this exposes in the surviving side-of-parity claim

§58/§60 left the side-of-parity result as the paper's load-bearing claim. These
curves qualify it. The no-diet arm is above parity in **8/8 subjects at epochs
25 through 100** and crosses only between 125 (6/8) and 150 (1/8). Its
epoch-150 value of 0.947 is **not an equilibrium** — it is a point on a curve
still descending at −0.072 per 25 epochs, 8/8 subjects.

So "no diet collapses to parity" is true at 150 and is not a property of the
condition; it is where the descent happened to be when training stopped. The
honest form: **the diet arm ends above parity and the no-diet arm ends at or
below it, at a matched 150 epochs**, with the residual drift carrying the
no-diet arm further away from the diet arm — so the direction of the remaining
uncertainty favours the claim. Do not write "converges to parity" or imply an
asymptote. And any reviewer asking "what happens at 300 epochs?" has a fair
question we cannot currently answer.

### 62.6 The mechanism claim, reworded to what the data supports

Replacing §61.6:

> Both arms start from magno-dominant front ends — the no-diet arm *more*
> magno-dominant (1.634) than the diet arm (1.402). Training then moves them in
> **opposite directions**: the diet arm upward (+0.201, 0/8 subjects down), the
> no-diet arm downward (−0.687, 8/8 down), crossing over. The two are
> separated in 8/8 subjects at every epoch from 25 to 150. What the diet
> changes is the **sign of training's effect on stream asymmetry**, not a
> target level — no level is reproducible across seeds (§58) or even stable
> within one run (62.3).

Scope, to travel with this wherever it appears: **one seed per arm**. The
trajectory shape has no seed replication, and §58 measured a 0.904 seed range
on the diet arm's endpoint, so the shape may well vary. What is seed-replicated
is the endpoint separation (§60.6: 27/27 pairings, 8/8 subjects) — not the path.

---

## §63 HOW DEAD IS EACH DEAD CLAIM? — an audit, because two are refuted and
the third was overstated

`tmp_analysis/deadness_audit.py`. All inputs are the seed
table from job 26413641 and the epoch table from job 26417753. No new compute.
Written because §60–§62 retired three level-based claims as a group, and they
are not equally dead. Grouping them overstates one and understates why the
other two fail.

### 63.1 §49.4's "converge to within 2%" (the 0.037 gap) — REFUTED

Three independent noise sources each dwarf the effect:

| quantity | value | vs the 0.037 gap |
|---|---|---|
| seed range, diet/lum | 0.904 | **24×** |
| epoch range within diet/lum seed 0 | 0.356 | **10×** |
| seed range, diet/highsf (the *tightest* arm) | 0.167 | **5×** |

And the sign of the gap is not determined: across the nine seed pairings the
diet/lum-minus-diet/highsf difference is positive in 7 and negative in 2, with
per-subject splits of 2/8, 8/8 and 0/8 on the three seed-0-involving pairings.

There is no measurement protocol under which 0.037 is a quantity. This is not
"unsupported pending more seeds" — more seeds cannot help, because the
within-arm variation is an order of magnitude larger than the between-arm
effect on every axis measured. **Refuted. Do not restate in any weakened form.**

### 63.2 §49.2's 22× compression — REFUTED, by arm selection rather than by
reference choice

| checkpoint set | ref = untrained | ref = gabor |
|---|---|---|
| 2 seed-0 arms (as published) | 0.816 → 0.037 = **22.1×** | 0.814 → 0.037 = **22.0×** |
| 3 arms, seed 0 only | 1.323 → 0.622 = 2.1× | 2.803 → 0.622 = 4.5× |
| 3 arms, all 9 seeds | 1.323 → 2.886 = **0.46× (expansion)** | 2.803 → 2.886 = **1.0× (no change)** |

To be fair to §49.2: the original figure was **not** reference-fragile — it is
22× under both references. It fails for a different reason. Its denominator
*is* the 0.037 of §63.1, i.e. a noise quantity, and a ratio whose denominator
is noise is not a measurement. Adding the third diet arm drops it to 2.1–4.5×;
adding the seed axis removes it entirely.

Two things must not be written. Not "training compresses front-end
heterogeneity" — false over the full checkpoint set. And not "training expands
it" either: that is 2.2× under one reference and 1.0× under the other, so the
expansion is not a finding. **The only robust statement is the null: the
trained spread is not smaller than the front-end spread.**

### 63.3 §50's channel-specific set point — OVERSTATED AS DEAD. It splits.

§60–§62 retired this with the other two. That was wrong; audited arm by arm it
separates cleanly into one surviving ordering and one unsupported one.

**swap vs `parvo_highsf` — SURVIVES.** Seed ranges are **disjoint**:

| arm | seeds | range |
|---|---|---|
| diet/rg (swap) | 1.995, 2.225, 4.383 | — |
| diet/highsf | 1.497, 1.640, 1.664 | 0.167 |

min(swap) = 1.995 exceeds max(highsf) = 1.664 by **0.331**, which is 2× the
highsf arm's entire seed range. It holds in **9/9** seed pairings, and it
survives deleting swap's extreme seed (without 4.383: 1.995/2.225 still
disjoint from 1.664). This is a real difference between two conditions, not
seed noise.

**swap vs luminance — NOT SUPPORTED, and for a locatable reason.** Ranges
overlap (swap 1.995–4.383, lum 1.603–2.507); it holds in 7/9 pairings; and it
depends **entirely on swap's 4.383 seed**. Delete that one checkpoint and swap
becomes 1.995/2.225 against a luminance arm reaching 2.507 — the ordering
breaks. A claim resting on one of three seeds is not established.

So the correct verdict is neither "dead" nor "channel-specific set point": **one
channel ordering is seed-robust and disjoint, the other is carried by a single
checkpoint.** §58.3's "not refuted; not established" was right about the
general claim and too pessimistic about the swap/highsf half.

### 63.4 What this changes about how to write it

The general form — "the diet installs a channel-specific set point" — stays
retired, because it requires *both* orderings and one of them fails, and
because §62.3 shows a single run traverses almost the whole distance being
attributed to a channel difference (1.603–1.959 within one run against swap's
1.995 floor: a 0.036 margin).

What is reportable, with the arms named rather than generalised: **applying the
diet to red−green yields a higher magno/parvo ratio than restoring raw
luminance to stream 2 does, with disjoint seed ranges (9/9 pairings, gap 0.331
against a 0.167 within-arm range).** That is a two-condition comparison, not a
set point, and it should be written as one.

### 63.5 Summary table, for whoever revises §49–§50

| claim | status | why |
|---|---|---|
| §49.4 "converge to within 2%" | **refuted** | effect is 1/24 of seed noise, 1/10 of epoch noise; sign undetermined |
| §49.2 "22× compression" | **refuted** | denominator is the above noise; null under the full checkpoint set |
| §50 swap vs `parvo_highsf` | **survives** | disjoint seed ranges, 9/9, robust to dropping the extreme seed |
| §50 swap vs luminance | **not established** | overlapping ranges; depends on 1 of 3 seeds |
| §50 "set point is channel-specific" (general) | **retired** | requires both orderings; one fails |
| side of parity, diet vs no diet | **survives** | 27/27 seed pairings, 8/8 subjects, 6/6 epochs |

---

## §64 PRE-REGISTERED — what does splitting the input across two streams buy?
The single-stream control (job 26421237)

Written and committed **before** the run finishes. `dkl_bcs_single`, 3 seeds,
`slurm/train_single_stream.sh`; wiring smoke-tested by
`tmp_analysis/smoke_single_stream.py` before any GPU time was spent.

### 64.1 The gap this fills

Leg (a) claims that biologically constrained magno/parvo input channels induce
specialised streams. Every existing control varies the **input** (`dkl_base`:
no diet; `dkl_bcs_swap`: diet on the other channel; `dkl_bcs_parvo_highsf`:
corrected parvo) or the **colour space** (`rgb_tsn`: two streams, no M/P
split). None asks whether the two-stream architecture is needed at all.

`dkl_bcs_single` feeds **one** trunk both DKL channels stacked, diet on
luminance exactly as in `dkl_bcs`. Identical input information, identical diet,
identical classifier, identical hyperparameters, same 150 epochs, same
`--ckpt-every 25` grid. Only the architectural split differs. It uses the
`baseline=True` path that already existed in `src/models/pytorch_models.py`;
this commit only wires it to the trainer.

### 64.2 What this arm cannot do

It has no streams. **The magno/parvo PR instrument and stream-wise variance
partitioning are unavailable** — every headline number in §47–§63 is
stream-relative and none of them can be computed here. Do not attempt it, and
do not report this arm in any table whose rows are stream statistics. It is
measurable on: top-1 accuracy, whole-model Geirhos cue-conflict shape bias
through the classifier, and whole-model cortical encoding without partitioning.

### 64.3 The capacity confound, and why one arm first

Measured, not estimated: `dkl_bcs` **22,397,936** trainable parameters,
`dkl_bcs_single` **11,230,832** — exactly **0.501×** (one ResNet18 trunk of
11,170,240 against two, plus the shared 57,456-parameter classifier).

The confound is **asymmetric**, which is the whole reason to run one arm before
any width-matched variant:

- If the single stream **wins or ties**, it does so with half the capacity.
  The confound strengthens the conclusion and is irrelevant.
- If it **loses**, nothing follows until a parameter-matched control is run
  (a `resnet34` trunk, ~21.3M against ~22.4M for two `resnet18` trunks). That
  run should be submitted **then**, not now.

### 64.4 The reference values, fixed now

From §54 (human data) and §60.3 (accuracy), so no number here can be chosen
after seeing the result:

| quantity | value |
|---|---|
| human shape bias | 0.9576 ± 0.0191, 95% CI [0.9439, 0.9713] |
| `dkl_bcs` magno stream alone | 0.8778 ± 0.0122 (3 seeds) |
| `dkl_bcs` **integrated** readout | **0.6749** |
| `dkl_bcs` top-1, last epoch | 0.4446 ± 0.0911 |
| `dkl_bcs` top-1, best epoch | 0.5921 ± 0.0041 |
| `dkl_base` top-1, best epoch | 0.7089 |

The behavioural comparison that matters is against **0.6749**, the integrated
two-stream readout, because that is the quantity the single-stream classifier
replaces. The comparison against 0.8778 is a *different* question — whether one
stream can reach what the two-stream model's best single stream reaches.

### 64.5 The four outcomes, declared in advance

Thresholds are stated numerically. §50.6's lesson is binding: a
pre-registered rule that cannot fail launders a refutation as a confirmation,
so each of these is falsifiable and they are mutually exclusive.

**A. THE SPLIT BUYS NOTHING.** Single-stream shape bias ≥ 0.6749 **and** best-
epoch top-1 within 0.02 of 0.5921. With half the capacity, this says the
two-stream architecture is decorative on both instruments the paper reports
behaviourally. Leg (a)'s architectural claim would have to be restated as a
claim about the *input transformation* alone — which the diet-vs-no-diet
result (§60.6, 27/27) already supports without any reference to streams.

**B. THE SPLIT HELPS BEHAVIOUR.** Single-stream shape bias < 0.6749 by more
than the between-seed sd, with accuracy within 0.02. The split buys shape bias
specifically. Capacity confound applies — a `resnet34` arm is then required.

**C. THE SPLIT IS A CAPACITY STORY.** Single-stream loses on **accuracy** by
more than 0.02 as well. Then the arm is uninterpretable as run and the
parameter-matched control is mandatory before any statement is made.

**D. THE SPLIT ACTIVELY HURTS LEG (b).** Single-stream shape bias ≥ 0.85, i.e.
approaching the 0.8778 that `dkl_bcs`'s magno stream reaches alone. This is the
live possibility flagged in §54: the two-stream **integration step** drops the
shape bias from 0.878 to 0.675, and if one stream carrying both channels
retains ~0.87, then summing the streams is what destroys the behavioural
result, not the split per se. Under D the paper's leg (b) needs rewriting a
second time and the architecture becomes a liability rather than a mechanism.

Outcome D is the one that costs the paper most, and §54 already showed
integration moves the network *away* from humans in 3/3 seeds, so it is not a
remote possibility.

### 64.6 Analysis protocol, fixed now

1. **Convergence first.** §60.1/§61.4 are binding: assert 150 epochs, report
   sd(val_top1) over the last 20 epochs, and report best-epoch as well as
   last-epoch accuracy. `dkl_bcs` is the one arm that never converged; if this
   arm is also unstable, every comparison uses best-epoch values and says so.
2. **Three seeds both sides.** No single-seed comparison (§60.6).
3. **3v3 seed-level tests floor at p=0.10**, so report complete separation and
   the raw difference with its range, not a p-value ranking (§55, §60.6).
4. **Decision rate.** §54 found our models decide in-pair on only 24–53% of
   cue-conflict trials against humans' 80%. Report it for this arm too; a
   shape bias computed over a small decided subset is not comparable to one
   computed over a large subset.
5. **No stream statistics** (64.2).

### 64.7 Disclosure

One condition, one diet placement (luminance). This tests whether the split
matters *for the diet-on-luminance arm*; it says nothing about the swap or
`parvo_highsf` arms, and must not be generalised to "the split does/does not
matter" without them. Runtime ~55–85 h per seed on one a5000 (from
`slurm/train_array.sh`'s measured 130–200 img/s), so a matched single-stream
series for the other diet placements is a separate decision, not an
afterthought.

---

## §65 PRE-REGISTERED — the konio arm: is the colour null a failed colour
prediction, or is L−M the wrong model of colour cortex?

Written and committed **before** the run finishes. `untrained_konio_matched`
and `untrained_konio_raw`, 8 subjects each, `slurm/encode_konio.sh`; front
end smoke-tested by `tmp_analysis/smoke_konio.py` before any compute.

### 65.1 The gap this fills

§7.1 attached the Kastner2015 atlas to the existing fits and found the
paper's colour prediction fails where it was supposed to succeed: **VO
median(Uₘ−Uₚ) = +0.0044, p = .46** — VO is, if anything, magno-preferring —
and **hV4 (−0.012, 2/8) is the only parvo-preferring area among 20 ROIs**.

Two readings have been indistinguishable ever since:

1. the model's chromatic stream does not predict colour-biased cortex at all;
2. **L−M is the wrong chromatic axis**, and S−(L+M) is the right one.

Nothing in §7–§64 can separate them, because every arm run so far has used
L−M as *the* colour channel. The repository is called
Magno-Parvo-**Konio**-Net and the third axis has never been run.

### 65.2 Why this is nearly free

The DKL dataset already stores all three opponent axes — `rgb_to_dkl3`
returns (lum, rg, **by**) and the training h5 quantises all three — so the
koniocellular S−(L+M) input needs **no new data conversion and no training**.
An untrained stream is the reference model for this claim anyway: §8 item 4
records that the surviving cortical results are the untrained ones.

Cost: one 16-task array on `main`, no GPU. The wiring is a channel slice
(`CHROMA_INDEX`), not a new model.

### 65.3 What changes — and the confound that was measured first

Stream 2 receives `by` instead of `rg`. Everything else is held: same
untrained network at seed 0 (weights bit-identical across builds, verified),
same luminance stream (σ=4 blur + CS cut at 48 months — the defaults
`untrained_seed0` was actually submitted with, recovered via `sacct
SubmitLine`), same architecture, same CV splits, same `nsdgeneral` mask.
`tmp_analysis/smoke_konio.py` verified that the magno **feature matrix** is
bit-identical between the two arms, so a change in unique chromatic variance
cannot come from the luminance side.

**But the channels are not on equal footing, and the imbalance flatters
konio.** Each DKL channel is normalised to [0,1] with its *own* global
min/max (rg spans 1.048 of cone contrast, by spans 8.189), which re-centres
but does not equalise. Measured on 100 shared-515 NSD images
(`tmp_analysis/konio_input_stats.py`):

| channel | per-image RMS contrast | ratio to rg | power above 1/8 cyc/px |
|---|---|---|---|
| lum (L+M) | 0.2047 ± 0.0662 | 4.45× | 1.29e+07 |
| rg (L−M) | 0.0460 ± 0.0234 | 1.00× | 8.28e+05 |
| **by (S−(L+M))** | **0.0941 ± 0.0315** | **2.05×** | **2.91e+06 (3.5×)** |

So an **unmatched konio win would be uninterpretable** — chromatic axis and
input contrast confounded, in konio's favour. Hence two cells, and the
asymmetry is the whole design (the same logic as §64.3's capacity argument):

- **cell 0 `untrained_konio_matched`** — by, per-image RMS matched to rg by
  `match_chroma_contrast`. **The primary arm.** After matching, the only
  remaining difference from the published parvo fit is which chromatic axis
  stream 2 carries. Matching is a pure contraction (rg has the lower
  contrast), so nothing leaves [0,1] and no clipping is applied.
- **cell 1 `untrained_konio_raw`** — by as the pipeline delivers it. **The
  sensitivity check.** A *loss* here is decisive (konio fails while holding a
  2× contrast advantage); a *win* here proves nothing alone and must be read
  against cell 0.

The reference cell needs no run: `untrained_seed0` is on disk for all 8
subjects with exactly this front end.

### 65.4 The hard positive control — a gate, not a diagnostic

`r2_magno` comes from a `RidgeCV` fit on the magno band **alone**, and
`prepare_band` centres/scales each band using only that band's own training
statistics. With bit-identical magno features and identical splits,
`r2_magno` must therefore come out **the same in the konio run as in the
published parvo run**.

**Gate: max|Δr2_magno| < 1e-4 across all voxels, all 8 subjects.** Report the
measured value. If it exceeds that, something differed that was supposed to
be held fixed and **nothing else in the arm may be interpreted** — §9.1's
standing lesson is that this project's worst failure mode is a pipeline
difference that reads as a result. A value in 1e-6…1e-4 is solver
nondeterminism and passes; a value above 1e-3 invalidates the arm.

This gate also makes the primary statistic exact: since `r2_magno` cancels,

    Δ  =  r2_joint(konio) − r2_joint(parvo)  =  U_k − U_p

per voxel, with no estimation error from the magno side at all.

### 65.5 What this arm CANNOT do

**No eccentricity residualisation at VO.** §7.1 found `prf-eccrois` labels
**0% of VO2 voxels** and only 9–23% of VO1, so the residualised VO contrast
is not computable with this atlas — that is why the hMT+/VO test was reported
raw. Do **not** pre-register or report an ecc-residualised VO effect here.
ROI specificity is instead tested by the **20-ROI profile difference** (65.6),
which is computable, and which is the instrument that correctly killed hMT+
in §16.1.

### 65.6 The four outcomes, declared in advance

Reference values, fixed now from §7.1 (n=8, NC>0.1, raw median(Uₘ−Uₚ)) so no
threshold can be chosen after seeing the result:

| quantity | value |
|---|---|
| VO pooled | +0.0044 (p = .46) |
| VO2 / VO1 | +0.011 (7/8) / −0.002 (4/8) |
| **hV4** | **−0.012 (2/8)** |
| V1d / V1v | +0.086 / +0.083 (8/8) |
| leave-one-subject-out profile ρ | +0.79 (range .59–.88) |

Primary statistic: **Δ = median over ROI voxels of (U_k − U_p)**, per subject,
paired Wilcoxon across n=8 (exact two-sided floor **.0078**; attainable values
are multiples of 1/128). Both statistics rule (§8) applies: report raw and
NC-normalised and claim only where they agree.

**Units.** Every threshold below is in **NC-normalised** units — the units of
§7.1's reference table, so that ±0.005 is commensurable with the −0.012 hV4
effect it has to explain. The raw version is reported alongside and the signs
must agree; naming the units matters because `variance_partition` returns both
(`unique_parvo` and `unique_parvo_raw`) and they differ by a per-voxel factor
of 1/NC. §50.6's lesson is binding —
each outcome below is falsifiable and they are mutually exclusive.

**A. KONIO RESCUES THE COLOUR PREDICTION.** Δ > 0 in hV4 **and** in VO with
≥7/8 subjects and p ≤ .0156, **and** Δ in {hV4, VO1, VO2} exceeds the median
Δ over the remaining Kastner ROIs (paired across subjects, p ≤ .05; the
number of ROIs clearing the ≥20-voxel floor is reported, not assumed — §7.1
got finite values for about 20 of the 25). The colour null was an L−M
problem, not a
colour problem: B3's prediction is delivered on the input side and the paper
gains a genuine third-channel result.

**B. KONIO WINS GLOBALLY, WHICH IS NOT A COLOUR RESULT.** Δ > 0 at A's
strength but *not* concentrated in colour-biased cortex — the colour-set-minus-
rest contrast fails at p ≤ .05. This is the general-factor problem that killed
hMT+ in §16.1
recurring on a new channel. It licenses "S−(L+M) is a better chromatic
regressor overall" and **nothing about VO**. It must not be written as a
colour-cortex result.

**C. NO DIFFERENCE.** |Δ| < 0.005 in both hV4 and VO — less than half the
−0.012 hV4 effect it would have to explain — with p > .05. Then the colour
null is not about which chromatic axis, and the honest statement is that both
opponent axes were tested and neither predicts colour-biased cortex. This
closes a question the paper currently leaves open, and is worth a paragraph
rather than a panel.

**D. KONIO IS WORSE.** Δ < 0 in VO and hV4. Decisive against B3 — and *more*
decisive in cell 1, where konio loses while carrying 2× the contrast. Under D
the konio channel is retired with the same finality as §49.4, and the paper
says so rather than omitting the arm.

Outcome B is the most likely a priori, because the by channel's advantage in
65.3 is global by construction and every ROI-specificity claim this project
has tested (hMT+, dorsal/ventral, VO) has died the same way.

### 65.7 Disclosure

- **Untrained, and that bounds the claim.** The networks were never trained on
  a by channel, so there is no konio checkpoint and none is implied. This arm
  is about the *input* as a model of colour cortex, evaluated through the
  architecture prior. A trained konio stream is a training run (~55–85 h/seed),
  not this array.
- **One seed.** An untrained network still has a seed, so this is one draw
  from the architecture prior. The reference cell is also one seed, so the
  comparison is matched — but if the result is positive, a 3-seed untrained
  replication is **required** before it is claimed, per §60.6's rule against
  one-sided replication.
- **No blur on the konio stream, deliberately.** B3 specified "heavy blur"
  from the K pathway's low spatial acuity. The measured radial spectrum of
  this pipeline's by channel has nearly the same *shape* as rg's (0.614 vs
  0.570 of power in the lowest octave band, near-identical above that), so
  the low-acuity property **is not present in the input**. Blurring would
  impose an assumption rather than reproduce a data property, and it would
  break the single-factor swap. A blur-on-K variant tests that assumption and
  is a separate, later question.
- **This swaps the chromatic axis; it does not add a third stream.** B3's
  architecture — a thin ¼-width K stream with dual projection (K→parvo after
  block 1, late skip→magno; Sincich 2004) — is **not** implemented and not
  tested here. So this arm does not deliver the repository title in the
  architectural sense. It delivers the konio *input* as a competing model of
  colour cortex, which is the part that is answerable without training.
- **Whole-model total variance is not the target.** Δ is a swap of one band;
  it says nothing about whether a three-band model would beat a two-band one.
  That comparison needs 3-way variance partitioning, which `banded_ridge.py`
  does not implement (it is hardcoded 2-band, and 3-way unique estimates with
  correlated bands are substantially noisier). Explicitly out of scope.

---

## §66 THE CORTICAL CLAIM IS SEED-REPLICATED — and it is the most reproducible
result in the paper

Array **26416428**, 32/32 COMPLETED (`slurm/encode_cc_seeds.sh`): seeds 1–2 of
`dkl_bcs` and `dkl_base` on `results/encoding_cc`, 8 subjects each, scored with
the unmodified `tmp_analysis/cc_contrast.py`. §2.3 was the last n=1 claim in the
paper and the only human-brain evidence; §58 had just shown that
single-checkpoint *levels* on the PR instrument do not survive a seed change, so
this was the outstanding risk to the whole NHB case.

### §66.1 Stream dissociation (magno − parvo VSB), diet vs no diet

| seed | diet | no diet | difference | p | subjects |
|---|---|---|---|---|---|
| 0 | +1.534 | +0.110 | **+1.424** | .0078 | 8/8 |
| 1 | +1.566 | +0.098 | **+1.468** | .0078 | 8/8 |
| 2 | +1.558 | +0.083 | **+1.474** | .0078 | 8/8 |

**Diet seed range 0.032; no-diet seed range 0.027.** The raw statistic agrees on
every seed (diet 0.616/0.658/0.632 against no-diet −0.033/−0.039/−0.053, p=.0078,
8/8 throughout), so the §8 two-statistics rule is satisfied three times over.

### §66.2 The contrast with the PR instrument is the point

| instrument | diet-arm seed range |
|---|---|
| magno/parvo PR ratio (§58) | **0.904** |
| cortical stream dissociation (here) | **0.032** |

Same three checkpoints. A 28× difference in seed stability. The instability that
retired §49.4 and §49.2 is a property of the participation-ratio measure, not of
these models and not of this project's instruments in general. §2.1's probe
(0.014) and the cortical dissociation (0.032) are both an order of magnitude
tighter. This should be stated explicitly, because a reader who meets §2.2's
2.39 seed range first will otherwise carry that skepticism into §2.3, where it
is not warranted.

### §66.3 The ventral gradient — replicates, with one honest qualification

hV4−V1v stream-bias gradient, normalised statistic: diet −0.447/−0.424/−0.501
against no-diet −0.202/−0.166/−0.160, p=.0078 and 0/8 in all three seeds.

The raw statistic agrees in direction and sign in all three seeds but reaches the
exact floor only in seed 2 (p = .0234, .0156, .0078; subject splits 1/8, 1/8,
0/8). Disclose this: the gradient is a weaker instrument than the stream
dissociation, and the two must not be quoted with equal confidence.

### §66.4 The cost replicates too

Median r²/NC: diet 0.405/0.382/0.399 against no-diet 0.464/0.464/0.457 —
p=.0078, 0/8 (diet lower) in every seed. The ~13% predictivity cost is not a
one-checkpoint artefact either. Panel (d) of Figure 1 stands.

### §66.5 Consequence

§2.3 loses its SEED STATUS flag. The paper now has **no n=1 claim in its main
results.** Every headline — the 2×2 (§2.1), the side of parity (§2.2), the
cortical dissociation (§2.3), the robustness null (§2.5), the error-consistency
null (§2.6) and the cost (§2.7) — rests on three training seeds.

## §69 THE §65.4 GATE FIRED — and it caught a stale reference, not a bad arm

§65.4 pre-registered a hard gate: `r2_magno` is a `RidgeCV` fit on the magno
band alone, `prepare_band` uses only that band's own statistics, and
`smoke_konio.py` proved the magno feature matrix is bit-identical across the
konio and parvo arms. So `r2_magno` MUST come out identical. Gate:
max|Δr2_magno| < 1e-4; above 1e-3 the arm is invalid and nothing else may be
interpreted.

Run on the first 7 konio-matched subjects as they landed, against
`untrained_seed0`:

| subj | max\|Δr2_magno\| | mean\|Δ\| | gate |
|---|---|---|---|
| 1 | 0.000e+00 | 0.000e+00 | PASS |
| 2 | 0.000e+00 | 0.000e+00 | PASS |
| 3 | 0.000e+00 | 0.000e+00 | PASS |
| 4 | **2.818e-02** | 9.728e-04 | **INVALID** |
| 5 | 0.000e+00 | 0.000e+00 | PASS |
| 6 | 0.000e+00 | 0.000e+00 | PASS |
| 8 | **2.786e-02** | 9.687e-04 | **INVALID** |

Five subjects are bit-exact, which is the strongest possible confirmation that
the front-end patch left the rg path untouched end to end — through the cache,
the PCA and the fit, not merely in the transform that `smoke_konio.py` tested.
Two subjects are not.

### §69.1 The cause: the reference is stale, the arm is fine

Noise ceiling, ROI labels and voxel count are IDENTICAL between the konio and
reference files for subj04 and subj08, so the betas, trials and mask are the
same data. The difference is on the fit side.

`untrained_seed0_subj04.npz` (Aug 27 02:26) and `untrained_seed0_subj08.npz`
(Aug 27 12:40) are the only two result files in `results/encoding` — 2 of 86 —
that predate commit **e3b8667** (Aug 27 15:04, "Cut encoding runtime ~30x").
That commit changed the banded-ridge hyperparameter search: `n_iter` 100 → 20
and CV folds 5 → 3. Different search, different selected (alpha, gamma) per
voxel, different r². The two failing subjects are exactly the two fitted before
it; every other result file, including all six passing konio fits, is after it.

So this is not a konio defect. It is a §22-class staleness that has been
sitting in the REFERENCE condition since August, and the konio gate is what
exposed it — which is the argument for pre-registering a gate that can only
fail for one reason.

### §69.2 Scope of the contamination

e3b8667 changed the ridge SEARCH only, so the contamination reaches exactly
those quantities derived from the fit, and no further. Checked, not assumed:

AFFECTED — everything downstream of `r2_joint`/`r2_magno`/`r2_parvo`.
`variance_partition` (`banded_ridge.py:221-223`) builds `unique_magno`,
`unique_parvo` and `shared` by differencing those three, and `spi`
(`banded_ridge.py:241-244`) from the uniques. So §7.1's Kastner variance
partitioning — the VO median +0.0044 / hV4 −0.012 result — mixed 6 subjects
fitted one way with 2 fitted another, as does any paired-subject statistic over
8 subjects that read a `untrained_seed0` fit.

NOT AFFECTED — the participation-ratio and effective-dimension tables in §36,
§41 and §45, and the front-end reference `thesis_dvd_trans` is normalised
against (§46). `tmp_analysis/pr_table.py` reads the FEATURE CACHE
(`/scratch/byrong/VST_data/mpk_feature_cache`, lines 15 and 32), never
`results/encoding`, and the caches predate e3b8667 on both sides of every
comparison. The 3.7 / 2.8 / 1.40 ±0.09 figures stand as reported.

The effect size on the affected numbers is small — mean|Δr2_magno| ≈ 1e-3, max
2.8e-2 — so the qualitative conclusions are unlikely to move. That is a
prediction, not a finding, and it is not a licence to leave it. Every fit-derived
number quoted from `untrained_seed0` must be recomputed once the refit lands,
and any that changes must be corrected in place rather than silently re-quoted.

### §69.3 Fix

Job **26423414**, `sbatch --array=3,7 slurm/encode_subjects.sh untrained_seed0`
— explicit array indices so the invocation is byte-identical to the original
submission recorded in §46 (`SUBJECT=$((SLURM_ARRAY_TASK_ID + 1))`, so 3,7 →
subjects 4, 8). `main`, no GPU. The Aug-26 feature caches are reused unchanged;
e3b8667 touched the ridge search and the stage logging, not feature extraction,
and the six bit-exact subjects prove the cache is sound. Fit only.

The pre-e3b8667 files are preserved at
`/scratch/byrong/mpk/encoding/backup_pre_e3b8667/` so the numbers that were
reported from them stay auditable.

### §69.4 What this blocks

The konio verdict CANNOT be read until the refit lands and the gate passes on
all 8 subjects. `tmp_analysis/konio_verdict.py` already refuses to interpret
anything when the gate fails, so the protocol holds without amendment — but it
means item #5's answer is gated on 26423414, not on 26422215 alone.

---

## §67 PRE-REGISTRATION — the accuracy-matched control, and a checkpoint-selection
error found while designing it

Prompted by an NMI reviewer simulation whose major comment M1 was: the diet arm
reaches 0.439 clean accuracy against 0.695, no per-condition hyperparameter
search was run, and therefore the dissociations of §2.4/§2.5/§2.6 may follow
from an accuracy gap rather than from the diet. Written before any measurement.

### §69.1 The error found first — `checkpoint_last.pt` is not the right checkpoint

`history.json` logs `val_top1` for all 150 epochs of all 12 runs. Reading it:

| arm | best epoch | seed range at best | final | seed range at final |
|---|---|---|---|---|
| diet (`dkl_bcs`) | **0.592** | **0.008** | 0.445 | **0.180** |
| no diet (`dkl_base`) | 0.709 | 0.003 | 0.701 | 0.007 |

Per-seed drop from own peak — diet: −0.060 (ep121), **−0.246** (ep60), −0.136
(ep142). No diet: −0.008, −0.013, −0.003.

Three consequences, all of which the paper currently states incorrectly:

1. **"Markedly higher seed variance" (Limitation 1, §2.5) is an artefact.** At
   best epoch the diet seeds agree to 0.008, against 0.003 for no diet. The
   0.180 spread is entirely produced by taking the last checkpoint.
2. **The accuracy cost is overstated.** 0.592 vs 0.709 is **16.5%** relative,
   not the 37% now reported.
3. **Every result in the paper was computed on `checkpoint_last.pt`** — behaviour,
   probe, PR, encoding, OOD, error consistency — including `dkl_bcs_seed1`, a
   model 0.246 below its own peak. The training script (§4.2) has no
   validation-based model selection; it saves last and final. That is defensible
   when training is stable and it is not stable for the diet arm.

The late-training degradation may itself be a finding about the diet. It must
not be reported as, or conflated with, the diet's accuracy cost.

### §69.2 What is being run

**Test A — best-epoch re-measurement (cheap, decides the artefact question).**
Re-run the cue-conflict behaviour and the shape/texture probe on each diet
seed's best *saved* checkpoint (ep125 = 0.564, ep75 = 0.553, ep50 = 0.552;
mean 0.556, range 0.012) and on the matched no-diet checkpoints.

**Test B — accuracy-matched no-diet control (the actual M1 control).** No saved
no-diet checkpoint is as low as 0.556 — the earliest saved, ep25, is already
0.658. A matched control therefore requires retraining `dkl_base` with dense
early checkpointing and selecting the epoch whose `val_top1` is nearest 0.556.

**Test C — within-arm accuracy correlation (free).** Across the three diet
seeds, final accuracy spans 0.347–0.527. If shape bias, PR ratio, OOD AUC and
error-consistency kappa do not track accuracy across that range, accuracy is not
the driver.

### §69.3 Predictions, fixed now

- **P1.** The 2×2 conjunction (§2.1) survives at best epoch: the diet magno
  probe bias stays above 0, and above all three other cells. *If it does not,
  §2.1 was a property of degraded checkpoints and the paper's lead result falls.*
- **P2.** The diet arm's best-epoch seed range on behaviour is smaller than its
  last-checkpoint range, mirroring accuracy.
- **P3 (the M1 test).** An accuracy-matched no-diet model does **not** acquire
  the diet's shape bias. If a no-diet model at 0.556 shows a shape-biased magno
  stream, then §2.1 is an accuracy effect and the paper's central claim is wrong.
- **P4.** Within the diet arm, shape bias does not correlate with accuracy
  across seeds.

### §69.4 The outcome that would be fatal, named in advance

If P3 fails — if matching accuracy is sufficient to produce the shape bias
without the diet — the conjunction result is confounded and the paper does not
have its central claim. There is no reframing that survives that; it would have
to be reported as a negative result about the whole approach.

If P1 fails, the lead result is a checkpoint artefact and every headline needs
recomputing on selected checkpoints before anything is claimed.

### §67.5 Disclosure regardless of outcome

The checkpoint-selection error of §67.1 is reported in the paper whatever Tests
A–C return, because three currently-stated numbers (seed variance, 37% cost, and
the implicit claim that reported values characterise the trained condition) are
wrong as they stand.

---

## §68 §67 TESTS A AND C — the lead result is not an accuracy artefact; §2.2's
instrument is exposed

Test B (the matched control, array 26423530) is still queued. A and C are done.

### §70.1 Test C — which measures track accuracy inside the diet arm

The diet arm's three seeds span 0.347–0.527 final accuracy at identical
architecture, data, schedule and front end — the widest within-condition
accuracy range available. `tmp_analysis/accuracy_confound.py`:

| measure | range over that 0.18 accuracy span | r with accuracy | verdict |
|---|---|---|---|
| probe bias (magno) | 0.014 | −0.79 | flat |
| behaviour shape bias | 0.022 | −0.89 | flat |
| **PR ratio (magno/parvo)** | **0.904** | **−0.99** | **EXPOSED** |

n=3, so r is descriptive. But the *ranges* are the argument: accuracy moves 0.180
and the probe moves 0.014, while the PR ratio moves 0.904 in near-perfect
inverse order with accuracy — the collapsed seed has the highest ratio.

**This reinterprets §58.** What was recorded there as seed noise on the PR
instrument is at least partly checkpoint degradation. It does not by itself
refute the side-of-parity claim — the no-diet arm sits at parity with *higher*
accuracy, the opposite of what a pure accuracy account predicts — but §2.2
cannot be cleared without Test B.

### §70.2 Test A — the 2×2 at best epoch. P1 CONFIRMED.

Cue-conflict re-run on each diet seed's best saved checkpoint (ep125 = 0.564,
ep75 = 0.553, ep50 = 0.552):

| | best epoch | final (published) |
|---|---|---|
| magno-stream shape bias | **0.903** (range 0.028) | 0.878 (range 0.022) |
| minimum across seeds | 0.888 | 0.870 |

The conjunction cell stays far above the 0.5 flip on every non-degraded
checkpoint, and the shape bias is **higher** there than on the published models.
Degradation does not manufacture the §2.1 effect; it slightly dilutes it. The
lead result is not an accuracy artefact.

### §70.3 P2 REFUTED — and it was my prediction

§67.3 predicted the best-epoch seed range on behaviour would be smaller than the
last-checkpoint range, mirroring accuracy. It is not: 0.028 against 0.022. The
behavioural measure was already seed-tight at the final checkpoint; only accuracy
and the PR ratio were not. Recorded as a failed prediction, and it sharpens the
picture — checkpoint degradation moves accuracy and the PR ratio, and leaves the
shape-bias measures alone.

### §70.4 Draft corrections made now, per §67.5

Applied to `PAPER_DRAFT.md` regardless of Test B:
- The 37% accuracy cost → **16.5%** (0.592 vs 0.709 at best epoch).
- "Markedly higher seed variance" → **withdrawn as an artefact** (0.008 vs 0.003
  at best epoch), replaced by the real and separate finding that the diet
  destabilises *late training*.
- Limitation 1 now names the superseded numbers explicitly rather than silently
  replacing them.

### §70.5 What Test B still has to decide

Whether a no-diet model **held at the diet's accuracy** stays at parity on the PR
ratio and stays texture-biased. P3 remains the fatal test: if it acquires the
shape bias, §2.1 is confounded despite §68.2.

## §70 THE KONIO ARM ANSWERS: OUTCOME D — the third channel is worse, and
## worse specifically where colour lives

Item #5 is decided. Job 26422215 cell 0 (`untrained_konio_matched`) is complete
for all 8 subjects, and after §69's reference refit the §65.4 gate passes
**bit-exact on every subject** (max|Δr2_magno| = 0.000e+00, worst across
subjects 0.000e+00). So Δ = U_konio − U_parvo is exactly
r2_joint(konio) − r2_joint(parvo), with zero magno-side estimation error, as
§65.4 predicted it would be.

### §70.1 The pre-declared contrasts, all four at the exact floor

Primary arm, contrast-matched, n=8, NC-normalised (raw units in parentheses;
§65.6 required sign agreement before any claim, and all four agree):

| pre-declared contrast | Δ (NC) | Δ (raw) | subjects | p | d_z |
|---|---|---|---|---|---|
| hV4 | **−0.0293** | −0.0099 | **0/8** | **.0078** | −2.09 |
| VO (VO1,VO2) | **−0.0238** | −0.0075 | **0/8** | **.0078** | −1.95 |
| colour set (hV4,VO1,VO2) | **−0.0256** | −0.0083 | **0/8** | **.0078** | −2.06 |
| colour set − other 17 ROIs | **−0.0140** | −0.0049 | **0/8** | **.0078** | −1.51 |

.0078 is the exact two-sided Wilcoxon floor for n=8 (§55), and 0/8 is complete
separation. This is **outcome D** of §65.6: konio is worse. Every arm of the
pre-registration points the same way, and the coded verdict in
`konio_verdict.py` returns D without touching the UNCLASSIFIED branch.

### §70.2 The specificity contrast is the load-bearing one

Outcome B — "konio wins globally, which is not a colour result" — was the
outcome §65.6 called most likely a priori, because the contrast advantage is
global by construction and because every ROI-specificity claim this project has
tested (hMT+, dorsal/ventral, VO) has died that way. **That expectation was
wrong, and in the informative direction.** Konio does not lose uniformly: in the
matched arm it *wins* in V1d/V1v/V2d (+0.025/+0.014/+0.011) and loses
everywhere else. Its worst ROIs of twenty are **hV4 (−0.0293), VO1 (−0.0274),
V3v (−0.0209) and VO2 (−0.0202)** — so the pre-declared colour set occupies
ranks **1, 2 and 4**, with ventral V3 interleaved between VO1 and VO2. The
colour-set-minus-rest contrast is itself at 0/8 and p = .0078, so the deficit is
not a global shift; it is concentrated in colour-biased cortex.

*(Correction, §76.1: this paragraph previously read "its three worst ROIs of
twenty being hV4, VO1 and VO2 — precisely the colour set". That is false: V3v
at −0.0209 sits between VO1 and VO2, in the matched arm under both statistics
(raw: hV4 −0.00991, VO1 −0.00880, V3v −0.00675, VO2 −0.00615), so the
interleaving is not a units artefact. The colour set is three of the worst
**four**, not the worst three. Nothing else in §70 depends on the ranking — the
load-bearing test is the colour-set-minus-rest contrast, which is a paired
Wilcoxon and does not use ranks at all.)* The same instrument that
correctly killed hMT+ in §16.1 here returns a positive verdict on specificity,
which is the strongest available evidence that this is a colour result and not
a general-factor artefact.

### §70.3 The raw cell, COMPLETE at n=8 — and the deficit is contrast-invariant

Job 26422215 finished all 16 tasks COMPLETED 0:0. The raw cell gates bit-exact
on all 8 subjects (max|Δr2_magno| = 0.000e+00), so both cells are now valid at
full n. §65.3 built the asymmetry deliberately: in the raw cell konio carries
**2.05× rg's per-image RMS contrast**, so a loss there is decisive while a win
proves nothing.

| contrast | matched Δ (NC) | raw Δ (NC) | raw subj | raw p | raw d_z |
|---|---|---|---|---|---|
| hV4 | −0.0293 | **−0.0059** | **0/8** | **.0078** | −1.08 |
| VO (VO1,VO2) | −0.0238 | −0.0061 | 2/8 | .1094 | −0.74 |
| colour set | −0.0256 | −0.0060 | 2/8 | .0391 | −1.01 |
| colour set − other 17 | −0.0140 | **−0.0139** | **0/8** | **.0078** | −2.80 |

Konio loses in colour cortex while holding a 2× contrast advantage, and NC and
raw units agree in sign on all four contrasts. Outcome D is returned
independently by both cells at full n.

**The load-bearing number is the last row.** The contrast manipulation moves
konio's global level enormously — mean Δ over the other 17 ROIs goes from
**−0.0116** (matched) to **+0.0079** (raw), and in the raw cell konio wins
early visual cortex outright (V1d +0.077, 8/8). Yet the colour-set-minus-rest
deficit is **−0.0140 matched against −0.0139 raw**: essentially unchanged. The
colour-specific deficit is therefore **invariant to a 2× input-contrast
manipulation that swamps the global effect**. That is much stronger than a
simple loss: it shows the deficit does not live on the contrast/SNR axis at
all, which is the axis a sceptic would reach for first.

**Stated honestly, three limitations of the raw cell.** (1) **VO pooled does
not reach significance even at n=8** — p = .1094, 2/8, d_z = −0.74. Only hV4
and the specificity contrast hit the .0078 floor; the pooled-VO claim rests on
the matched cell alone. (2) The absolute deficits are ~4–5× smaller than in the
matched cell, which is expected — the contrast advantage partly offsets them —
but it means the raw cell corroborates direction and specificity, not effect
size. (3) The n=5 interim read reported here previously was misleading in
places (it gave VO 1/5 and d_z = −3.15 on specificity); at n=5 the exact
two-sided floor is .0625, so nothing there could have been significant. Those
interim numbers are superseded by the table above and should not be quoted.

*(Numbering note: a concurrent session's §68 contains subsections mislabelled
"§70.3" and "§70.4"; the §70.3 meant here is this one. Left untouched rather
than edited across sessions — see §70.7.)*

### §70.4 What this closes, and what it does not

CLOSES: the two-reading ambiguity §65 was built to resolve. Since §7.1, two
readings of the failed colour prediction have been indistinguishable — the
chromatic stream does not predict colour cortex at all, or L−M is the wrong
chromatic axis. It is now the **first**: S−(L+M) is not merely no better, it is
reliably and specifically worse in exactly the ROIs where the colour prediction
was supposed to succeed. Both chromatic axes have been tested against colour
cortex and neither predicts it. Per §65.6 outcome D the question closes, and B3
loses its motivation: there is no evidence here for adding a koniocellular
channel.

DOES NOT CLOSE, and must be disclosed rather than glossed (§65.7):
- **Untrained, one seed.** Nothing follows about a *trained* konio stream. §60.6
  requires 3-seed untrained replication for a positive result; this is a
  negative one for konio, so the requirement does not bind, but the arm remains
  a single random initialisation and should be described as such.
- **This is an axis SWAP, not a third stream.** B3 proposed dual chromatic
  projections; the architecture that would deliver the repository's title is
  untested. What is refuted is the premise that motivated it, not the
  architecture itself. 3-way variance partitioning stays out of scope —
  `banded_ridge.py` is hardcoded 2-band.
- **No blur on K**, deliberately (§65.7): the measured radial spectrum of this
  pipeline's `by` channel has nearly rg's shape (0.614 vs 0.570 of power in the
  lowest octave), so blurring would impose an assumption rather than reproduce
  a property.

### §70.5 Where it goes in the paper

Not a section. Konio is a **negative control that strengthens leg (c)** and it
belongs in one paragraph plus a supplementary panel. The positive claim it
supports is that the parvo stream's unique variance is not a generic
"chromatic-channel" effect that any opponent axis would produce: swap L−M for
S−(L+M), hold the luminance stream bit-identical, and the ventral prediction
gets *worse*, specifically in colour cortex. Per the project's positioning rule,
lead with that surviving positive claim — the specificity of L−M — and report
the konio failure as its control, not as a finding in its own right.

### §70.6 §7.1 revalidated on the corrected reference

With §69's refit in place, `kastner_rois.py` was re-run on 8 consistently
fitted subjects. The predicted-negligible effect is confirmed:

| statistic | as reported | recomputed |
|---|---|---|
| hMT+ (TO1,TO2) | +0.0197, p=.023, 7/8, dz=1.29 | **+0.0195, p=.0234, 7/8, dz=+1.30** |
| VO (VO1,VO2) | +0.0044, p=.46 | **+0.0045, p=.4609** |
| hMT+ − VO interaction | +0.0153, p=.109 | **+0.0150, p=.1094** |
| hV4 | −0.012, 2/8 | **−0.0123, 2/8** |
| LOSO profile ρ | +0.79 (.59–.88) | **+0.794 (.589–.896)** |

Every conclusion in §7.1 stands. The only changes at reported precision are
LO2's subject count (1/8 → 2/8) and third-decimal rounding on four early-visual
ROIs; the §7.1 table has been corrected in place. The eccentricity-residualised
rows still return "n=0 / n=3 too few subjects", independently confirming §65.5's
decision not to pre-register a test the atlas cannot support.

---

## §71 A SECOND CHECKPOINT ERROR, IN MY OWN CORRECTION — "best epoch" named an
## epoch whose weights do not exist

§67.1 caught that every published result used `checkpoint_last.pt`, and §68
re-ran cue-conflict on "the best epoch". Re-reading the artefacts before
reporting status, the two halves of that correction do not refer to the same
thing.

### 71.1 What the numbers actually are

`standalone_trainer.py:414` writes `checkpoint_{epoch+1:03d}.pt` every 25
epochs. `history.json` is 0-indexed, 150 entries. So `checkpoint_125.pt` is
`history[124]`, and only six weight files exist per run. The per-epoch
validation maximum almost never lands on one of them.

| definition | diet mean | range | no-diet mean | range | rel. gap |
|---|---|---|---|---|---|
| per-epoch max (log only, **no weights**) | 0.5921 | 0.0080 | 0.7089 | 0.0029 | 16.5% |
| **best saved checkpoint** | **0.5562** | 0.0123 | **0.7012** | 0.0073 | **20.7%** |
| final checkpoint (published) | 0.4446 | 0.1804 | 0.7012 | 0.0073 | 36.6% |

Best saved checkpoints: diet epochs 125 / 75 / 50, no-diet 150 / 150 / 150.
The no-diet arm's best saved checkpoint *is* its final one, which is why only
the diet arm was ever affected.

### 71.2 The error

§68 quoted **16.5%** — row 1 — as the corrected accuracy cost, while the
shape-bias re-run (0.903) was computed on row 2's checkpoints, which sit 0.036
lower. The manuscript therefore paired an accuracy figure from weights that do
not exist with a behavioural figure from weights that do. Row 1 characterises
the training schedule; it is not the cost of any model that can be evaluated.

Corrected in `PAPER_DRAFT.md` §2.5 and Limitations 1: the cost is **20.7%**
(0.556 vs 0.701), with row 1 retained explicitly labelled as a schedule
property. Limitation 1 now names both superseded figures (37%, then 16.5%)
rather than silently replacing the second.

### 71.3 What does not change

- **§68's Test A conclusion stands, and strengthens.** The 0.903 shape bias was
  always measured on the best *saved* checkpoints. Those models are 0.111 more
  accurate than the final ones and *more* shape-biased, not less. Degradation
  dilutes the §2.1 effect; it does not manufacture it.
- **The seed-variance withdrawal stands.** 0.012 vs 0.180 is still an order of
  magnitude, so "the diet arm is markedly less stable" remains an artefact of
  last-checkpoint selection, and "the diet destabilises late training" remains
  the real, separate finding.
- **§67's Test B is unaffected** — it trains fresh models to a target accuracy
  and never referenced these checkpoints. Its target band should bracket
  **0.556**, the diet arm's best evaluable accuracy, not 0.592.

### 71.4 Standing rule

Any accuracy quoted beside a behavioural, cortical, or representational number
must come from the same weight file that produced it. A validation-log maximum
is a property of the schedule and may only be reported as such.

### 71.5 The §2.1 probe at n=3 on best saved checkpoints

The third diet seed's probe finished after §71 was written. All three:

| | magno | parvo | magno − parvo |
|---|---|---|---|
| final ckpt (published) | +0.8194 [0.0142] | −0.5183 [0.0667] | +1.3378 [0.0808] |
| best saved ckpt | +0.8061 [0.0125] | −0.4733 [0.0250] | +1.2794 [0.0342] |

Per-seed magno at best checkpoint: 0.8008, 0.8042, 0.8133 — 3/3 positive.

The headline cell moves **−0.013**, an order of magnitude below the effect it
supports, and the seed range on the difference more than halves. Note the two
measures move in opposite directions under the checkpoint switch — behaviour up
(0.878 → 0.903), probe down (0.819 → 0.806) — which is what independent
measurements of a checkpoint-insensitive quantity should look like, and is not
what a shared accuracy driver would produce.

Together with §68's Test A and Test C this closes the checkpoint question for
§2.1 in both instruments. §2.2 (PR ratio) remains exposed at r = −0.99 and is
not cleared by this; only Test B can settle it.

### 71.6 §2.5 OOD re-launched on best saved checkpoints (array 26424303)

`slurm/ood_robustness_bestep.sh`, 12 elements, `--partition=main`, no `--gres`
(resnet18 inference; the project Slurm rules send non-CUDA work to `main`).
Writes to `results/ood_bestep/`, leaving the published `results/ood/` intact so
the two policies can be compared rather than one silently overwriting the other.

The job resolves the checkpoint itself from `history.json` and fails loudly if
the file or the resolved `.pt` is missing, so the §71.1 off-by-one cannot be
reintroduced by hand. Resolved epochs, confirmed in the logs against an
independent offline computation:

| cell | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `dkl_bcs` (diet) | 125 (0.564) | 75 (0.553) | 50 (0.552) |
| `dkl_base` | 150 (0.700) | 150 (0.698) | 150 (0.706) |
| `dkl_bcs_swap` | 100 (0.701) | 100 (0.697) | 150 (0.700) |
| `dkl_bcs_parvo_highsf` | 150 (0.705) | 150 (0.699) | 150 (0.695) |

The policy is applied to **all four cells**, not only the diet arm, so the
comparison stays checkpoint-policy-consistent. Only the diet arm moves much;
`swap` shifts by ≤0.009 and the other two not at all.

**Pre-commitment.** §2.5's claim is that shape bias and OOD robustness are not
intrinsically coupled — the diet arm delivers none of the robustness Geirhos
et al. obtained via Stylized-ImageNet. The published numbers used degraded diet
weights, which is the reading most favourable to that claim. If the diet arm's
relative-accuracy AUC rises materially at its best checkpoint, the dissociation
weakens and §2.5 must be restated in whatever terms survive; the section is not
load-bearing for §2.1 either way. If it does not rise, the published
dissociation stands on undegraded weights and the limitation is retired.

### 71.7 OOD on best saved checkpoints — the §71.6 pre-commitment resolves

Array 26424303, 12/12 COMPLETED, `results/ood_bestep/`. `ood_verdict.py` now
takes the results directory as an argument, so both policies are scored by
identical code.

Relative-accuracy AUC, diet arm, final → best saved:

| corruption | final | best saved | Δ | separation (relative) |
|---|---|---|---|---|
| uniform-noise | 0.6869 | 0.6341 | **−0.053** | overlap → **complete sep.** |
| high-pass | 0.6188 | 0.6781 | +0.059 | sep. → sep. |
| phase-scrambling | 0.3507 | 0.3825 | +0.032 | **sep. → overlap** |
| rotation | 0.5524 | 0.5738 | +0.021 | sep. → sep. |
| false-colour | 0.8419 | 0.8536 | +0.012 | sep. → sep. |
| *low-pass (trained-in)* | *0.4407* | *0.4444* | *+0.004* | sep. → sep. |
| *contrast (trained-in)* | *0.3269* | *0.4813* | *+0.154* | sep. → sep. |

**The pre-commitment asked whether the diet arm's relative AUC rises materially.
On the corruptions that count, it does not.** Four of five move by ≤0.059 and
one moves *down*. The diet arm stays below no-diet on every one of the five, and
`OUTCOME C` is returned under both policies. §2.5's claim survives on undegraded
weights, which is the outcome that was not guaranteed.

The one large move is **contrast, +0.154** — a trained-in family that counts
toward nothing. That is the §68 Test C pattern again: late-training degradation
selectively destroys the diet's robustness to the very manipulation its front
end applies, which is a coherent story about degradation and not about shape
bias.

### 71.7.1 What changed, and what it costs

1. **H1 strengthens.** All three pre-registered H1 corruptions (uniform noise,
   high-pass, false colour) now show complete seed separation in the predicted-
   against direction; at final checkpoint uniform noise merely overlapped. H1
   remains *refuted* — the diet is less robust, not more — and it is refuted
   more cleanly.
2. **H2 dies.** Phase-scrambling flips from "diet worse, complete separation"
   to "indistinguishable, p = 0.70". H2 was the only pre-registered hypothesis
   the mechanical scorer ever marked supported. **This costs the paper nothing,
   because §54.2 already ruled that H2 must not be reported as supported** —
   its "support" was a main effect masquerading as the predicted crossover. The
   draft never claimed it. A refusal taken on conceptual grounds in §54.2 is now
   independently confirmed by a checkpoint control, which is the strongest form
   of evidence that the refusal was right.
3. **H3 weakens slightly**, 90% → 80% of control comparisons patterning with
   no-diet. The new defector is `parvo_highsf` on phase-scrambling, where the
   diet/no-diet gap is 0.009 — "closer to diet" is not meaningful across a gap
   that small, and the scorer's binary nearest-neighbour rule overstates it.
   §2.5 now reports 8 of 10 rather than "throughout".
4. **A new observation worth one sentence:** the diet is *not* more robust to
   contrast reduction (0.481 vs 0.549) despite its front end containing a
   contrast-sensitivity cut. Trained-in does not automatically mean robust, and
   that makes the low-pass positive control more informative rather than less.

Draft updated: §2.5's table, its headline sentence ("every corruption" → "four
of the five"), the Discussion, and Limitation 1.

### 71.7.2 Checkpoint policy is now split across the paper

§2.1 (both instruments) and §2.5 are scored on best saved checkpoints. §2.4
(PR), §2.6 (error consistency) and §2.7 (cortex) are still on final
checkpoints. §68 Test C showed the PR instrument tracks checkpoint degradation
at r = −0.99, so **§2.4 is the section most likely to move** and is the next
re-run to schedule. This split must be stated in Methods before submission; it
is currently only in the PLAN.

### §70.7 Numbering collisions on the shared `mpk-dev` branch

Recorded because it is now recurring, and because it makes cross-references in
this file unreliable rather than merely untidy.

Two sessions are committing to `mpk-dev` and appending to this file
concurrently. Since `c9f688a` the collisions are:

- §67 claimed twice — by `c432dfd` (the gate / stale reference) and by
  `333b697` (the accuracy-matched control). Resolved by renumbering **mine** to
  §69, since `333b697` and `c34fd51` cross-reference each other and were the
  larger change to unpick.
- §68 claimed twice — by my konio verdict and by `c34fd51`. Mine became §70.
- **"§70.3" and "§70.4" are still ambiguous.** `c34fd51`'s §68 section contains
  subsections labelled `### §70.3` and `### §70.4` (apparently intended as
  §68.3/§68.4), which collide with §70.3 and §70.4 of this section. Left
  untouched: editing another session's committed prose while it is still
  working in the same file is what produced the earlier duplication incidents.

Consequence for anyone reading: a bare "§70.3" in this document resolves to two
different places. One owner should reconcile the numbering in a single pass —
that is a documentation fix, not a result, and it should not be done piecemeal
by whichever session notices it. This joins the older §21/§25/§23/§24/§29
disorder already recorded in §26.1.

## §71.8 PRE-REGISTERED — §2.4's PR dissociation at the best saved checkpoint

§68 Test C is the reason this has to be run: across the diet arm's own 0.18
accuracy span the magno/parvo PR ratio moved 0.904 at r = −0.99, with the
collapsed seed highest. Every PR number in §2.4 came from caches extracted at
`checkpoint_last.pt`. §2.4 is the last load-bearing section still scored on
weights §71 showed to be the wrong ones, and it is the section most likely to
move.

### 71.8.1 What is being run

`slurm/cache_bestep.sh`, array **26427718**, 32 elements, `main`, no `--gres`,
cache-only. Only four model×epoch combinations were missing:

| run | best saved ckpt | cache status |
|---|---|---|
| `dkl_bcs_seed0` | ep125 (0.564) | **already cached** (epoch trajectory) |
| `dkl_bcs_seed1` | ep075 (0.553) | extracting |
| `dkl_bcs_seed2` | ep050 (0.552) | extracting |
| `dkl_bcs_swap_seed0` | ep100 (0.701) | extracting |
| `dkl_bcs_swap_seed1` | ep100 (0.697) | extracting |
| `dkl_bcs_swap_seed2` | ep150 | == `checkpoint_last`, cached |
| `dkl_base_seed{0,1,2}` | ep150 | == `checkpoint_last`, cached |
| `dkl_bcs_parvo_highsf_seed{0,1,2}` | ep150 | == `checkpoint_last`, cached |

`checkpoint_last.pt` is rewritten every epoch, so after a full 150-epoch run it
holds the same weights as `checkpoint_150.pt`; the three arms whose best saved
checkpoint is 150 therefore need no re-extraction. Each task re-derives the best
checkpoint from `history.json` and aborts if the path it was handed is not that
file, so the §71.1 off-by-one cannot return through a hand-edited array.

### 71.8.2 The prediction, written before the caches exist

§2.4's claim is a **side-of-parity** claim, not a level claim — §50 already
killed the "universal set point" reading, and Figure 1(c) carries the footnote
"no level is claimed". The claim is: diet-on-luminance arms sit above parity
(magno-dominant), no-diet arms below.

- **P1.** The side of parity survives: diet arms > 1, no-diet arms < 1, in all
  three seeds. This is what §2.4 actually asserts.
- **P2.** The *level* moves, and probably downward, since Test C found the
  most-degraded seed carried the highest ratio. A level shift refutes nothing,
  because no level is claimed — but it must be reported, and the §2.4 numbers
  replaced rather than kept alongside.
- **P3 (fatal).** If the side of parity does **not** survive — if the diet arm
  falls to or below parity on undegraded weights — then §2.4's dissociation was
  an artefact of late-training collapse and the section is dead. §2.1 does not
  depend on it (§71.5 closed both of §2.1's instruments independently), so the
  paper survives; but §2.4 would have to be withdrawn, not reframed, and
  Figure 1(c) rebuilt.

Recorded now, before any best-epoch PR number has been computed, so P3 cannot
be quietly softened after the fact. Same discipline as §57.5 and §67.4, both of
which refuted my own hypothesis and were honoured.

### 71.8.3 Free preview available immediately

`dkl_bcs_seed0` has a full cached epoch trajectory (ep025–150), so the PR ratio
can be plotted against that seed's own validation accuracy across six
checkpoints before the array finishes. That is a within-seed version of Test C
on the actual §2.4 statistic, and it costs nothing.

## §71.9 THE WITHIN-SEED TRAJECTORY — §68 Test C was over-read, and §2.2 has a
## real vulnerability that best-vs-last would never have found

`tmp_analysis/pr_epoch_within_seed.py`. The cached epoch trajectories (ep025–150,
8 subjects each) allow accuracy to be varied **within** a seed, which is the test
§68 Test C should have run.

### 71.9.1 Correcting §68 Test C

Test C correlated the magno/parvo PR ratio against `val_top1` across the three
diet seeds at their final checkpoints — n=3, perfectly monotone, r = −0.99 — and
I read that as the PR instrument tracking checkpoint degradation. **That was an
over-read.** Three points in which seed identity and accuracy co-vary cannot
separate the two, and with n=3 a monotone triple gives |r| ≈ 1 almost by
construction.

Holding the seed fixed and varying the checkpoint instead:

| run | accuracy span | PR-ratio range | within-seed r | p |
|---|---|---|---|---|
| `dkl_bcs_seed0` | 0.340–0.564 | 0.356 | **−0.169** | 0.75 |
| `dkl_base_seed0` | 0.658–0.699 | 0.494 | **−0.184** | 0.77 |

The diet arm's accuracy span here (0.224) is **wider** than the cross-seed span
that produced the −0.99, and the correlation collapses to nothing. The 0.904
cross-seed range in §2.2's replication table is seed-to-seed variation, not
checkpoint degradation. §68's "§2.2's instrument is EXPOSED" verdict is
withdrawn and replaced by this.

### 71.9.2 But the trajectory found something worse, in the control arm

| ckpt | diet `dkl_bcs_seed0` | | no-diet `dkl_base_seed0` | |
|---|---|---|---|---|
| | ratio | side | ratio | side |
| front end (untrained) | 1.402 | — | 1.634 | — |
| 25 | 1.752 | 8/8 > 1 | 1.156 | 8/8 > 1 |
| 50 | 1.959 | 8/8 > 1 | 1.237 | 8/8 > 1 |
| 75 | 1.733 | 8/8 > 1 | 1.513 | 8/8 > 1 |
| 100 | 1.620 | 8/8 > 1 | 1.505 | 8/8 > 1 |
| 125 | 1.703 | 8/8 > 1 | 1.019 | 6/8 > 1 |
| 150 / last | 1.603 | 8/8 > 1 | **0.947** | **1/8 > 1** |

**The no-diet arm is above parity for the first two-thirds of training and
crosses below only in the final quarter** — at essentially constant accuracy
(0.658 → 0.699 across the whole span). The published sub-parity value is a
late-training endpoint, not a stable property of no-diet training.

The diet arm, by contrast, is above parity at **every** checkpoint from 25
onward, 8/8 subjects each time.

### 71.9.3 What this does and does not cost

- **It does not refute §2.2.** The section is titled "The diet fixes which side
  of parity training *ends* on", and the claim is evaluated at the end of
  training. Under both checkpoint policies — final, and best-saved (§71.1, where
  no-diet's best *is* its final) — the dissociation holds.
- **It does cost the paper an unstated assumption.** At epochs 75–100 both arms
  sit above parity (1.62–1.73 vs 1.50–1.51) and there is no dissociation at all.
  A reviewer who asked "when does this appear?" would find that, and finding it
  unprompted is much worse than us reporting it.
- **It sharpens the asymmetry into a better claim.** The diet's side is stable
  across training; the control's is not. "The diet *stabilises* which side of
  parity the network sits on, while no-diet training drifts and ends below" is
  both more accurate and more interesting than the current wording.
- **n=1 seed per arm.** Only seed 0 has a cached trajectory. This is a
  disclosure, not a headline, until more seeds have one. Do not promote it.

### 71.9.4 Correction to §71.7.2

§71.7.2 named §2.4 as the section most exposed by the checkpoint issue. Wrong
section: §2.4's table is computed on the **Gabor bank and untrained networks**,
which have no checkpoints, so §2.4 is structurally immune. The trained PR
statistic lives in **§2.2**. Array 26427718 is still the right job — it supplies
seeds 1 and 2 at their best saved checkpoints for §2.2's replication table — but
it is no longer the load-bearing evidence, which is §71.9.2.

---

## §71.10 THE §71.8 PRE-REGISTRATION RESOLVES: P1 HELD, P3 NOT TRIGGERED — and
## most of §2.2's "seed irreproducibility" was late-training degradation

Array 26427718, 32/32 COMPLETED. `tmp_analysis/pr_bestep_score.py` scores the
§2.2 participation-ratio statistic at each seed's **best saved checkpoint**, with
an in-script verifier that re-derives that checkpoint from `history.json` and
aborts on a mismatch (the §71.1 off-by-one cannot return).

### 71.10.1 The pipeline validates itself before it is read

Six of the twelve seeds have their best saved checkpoint at epoch 150, which is
the same weight file as `checkpoint_last` and therefore the *published* number.
Those six must reproduce §2.2's table exactly, and they do:

| arm | published | recomputed | match |
|---|---|---|---|
| no diet, seeds 0/1/2 | 0.947 / 0.988 / 0.914 | 0.9466 / 0.9885 / 0.9143 | ✅ |
| `parvo_highsf`, seeds 0/1/2 | 1.640 / 1.497 / 1.664 | 1.6399 / 1.4974 / 1.6638 | ✅ |

So any movement below is a real checkpoint effect, not a re-extraction artefact.

### 71.10.2 P1 — HELD, with complete separation

| arm | best-saved seeds | mean | seed range | side |
|---|---|---|---|---|
| diet / luminance | 1.703, 1.722, 1.602 | 1.676 | **0.120** | 3/3 > 1, 8/8 subjects |
| diet / `parvo_highsf` | 1.640, 1.497, 1.664 | 1.600 | 0.166 | 3/3 > 1, 8/8 subjects |
| diet / red−green (`swap`) | 2.327, 1.541, 1.995 | 1.954 | **0.787** | 3/3 > 1, 8/8 subjects |
| no diet | 0.947, 0.988, 0.914 | 0.950 | 0.074 | 0/3 > 1 |

All nine diet seeds above parity, all three no-diet seeds below, **complete
seed-level separation** (diet minimum 1.602 vs no-diet maximum 0.988, gap
**+0.614**). P3 — the fatal branch, "the diet arm falls to or below parity on
undegraded weights" — is **not triggered**. §2.2 stands, and Figure 1(c) does
not need rebuilding.

### 71.10.3 P2 — the level moved, and it moved the *opposite* way from the guess

§71.8.2 predicted the level would "probably move downward, since Test C found the
most-degraded seed carried the highest ratio". Test C was withdrawn in §71.9.1,
and its prediction fails too: the diet/luminance mean is essentially unchanged
(1.978 published → 1.676) but the **seed range collapses from 0.904 to 0.120, a
7.6× tightening**; `swap` collapses from 2.388 to 0.787, a 3.0× tightening. The
two arms whose best checkpoint is not 150 are exactly the two that moved.

This is the *same pattern* §71 found on accuracy — the diet arm's seed range
there went 0.180 → 0.012 when scored on weights that exist. Two independent
instruments now agree: **the diet arm is not markedly less reproducible than the
control; it looked that way because it was being scored at a degraded final
checkpoint.** That is a property of the training schedule, not of the diet.

### 71.10.4 What this does NOT license

The tightening must not be spent on a level claim. Three checks, all of which
must pass before "converges to X" could be written, and one fails:

- §58.3 retired a "convergence to within 2%" reading because the 0.037 gap
  between the two diet-on-luminance arms was ~24× smaller than that arm's seed
  noise. At best saved checkpoints the gap between arm means is 0.075 and the
  seed noise is 0.120 — still **larger than the gap**. The retraction stands.
- The largest within-arm seed range is still 0.787 (`swap`), an 80% swing.
- n=3 per arm cannot separate a level from a seed draw regardless.

So §2.2 continues to claim the side and no level. What changes is the *reason
given*: the honest statement is no longer "the within-arm seed range reaches
2.39", which was a final-checkpoint artefact, but "the within-arm seed range
reaches 0.79 and exceeds the between-arm gap".

### 71.10.5 Consequential edits

1. §2.2's seed-replication table → best-saved values, with the checkpoint stated.
2. §2.2's "seed range reaches 2.39" → 0.79, and the §58.3 retraction re-derived
   on the new noise estimate.
3. §2.3's "0.904 ... a 28-fold difference in seed stability" → 0.120, **3.7-fold**.
   The cortical instrument is still the tighter one, but not by the margin
   claimed, and the sentence asserting §2.2's instability "belongs to the
   participation-ratio measure" is now wrong and must go — the instability
   belonged to the checkpoint.
4. Methods: state the split checkpoint policy (§2.1, §2.2, §2.5 on best saved;
   §2.6, §2.7 on final) — still outstanding.

---

## §72 THE §64 PRE-REGISTRATION RESOLVES: the split buys shape bias and costs
## accuracy — and §64.5's outcome set could not name it

Job 26421237, 3/3 seeds COMPLETED, 150 epochs each (~26 h/seed). Cue-conflict
run as array 26428876, 6/6 COMPLETED, `main`, no `--gres`.

### 72.1 The answer, before the bookkeeping

| | single stream | two-stream `dkl_bcs` | Δ |
|---|---|---|---|
| shape bias, best saved ckpt | **0.5672** | 0.6547 | **−0.0875** |
| shape bias, final ckpt | **0.5571** | 0.6749 | **−0.1178** |
| top-1, best saved ckpt | **0.5990** | 0.5562 | **+0.0428** |
| top-1, final ckpt | **0.5505** | 0.4446 | **+0.1059** |
| trainable parameters | 11,230,832 | 22,397,936 | **0.501×** |

**The two-stream split buys shape bias and pays for it in accuracy.** One trunk
carrying both DKL channels, with the identical diet, is *less* shape-biased and
*more* accurate than two trunks — at half the parameters, under both checkpoint
policies.

Per-seed, with complete separation (§64.6 item 3; 3v3 floors at p=0.10 so
separation is the reportable statistic, not a p-value):

| policy | single stream | two-stream | separation |
|---|---|---|---|
| best saved | 0.5414 / 0.5899 / 0.5703 | 0.5942 / 0.7177 / 0.6522 | max single 0.5899 < min two 0.5942 — **complete** |
| final | 0.5146 / 0.5865 / 0.5703 | 0.6922 / 0.6861 / 0.6464 | **complete** |

### 72.2 This is the outcome that most helps leg (a), and it was not the
### expected one

§64.5 flagged **D** as "the one that costs the paper most" and "not a remote
possibility": if one stream carrying both channels retained ~0.87, then
*integration* — not the split — would be what destroys the behavioural result,
and the architecture would be a liability.

**D is refuted, and not narrowly.** 0.5672 against a 0.85 threshold. A single
stream carrying both channels does not retain the magno stream's 0.903; it does
not even retain the *integrated* two-stream 0.6547. So summing the streams is
not what destroys the shape bias — the ordering is

    magno alone 0.903  >  two-stream integrated 0.655  >  single stream 0.567

Integration costs 0.248 relative to the best stream, but the split still buys
0.088 over having no split at all. The architecture is doing work rather than
merely surviving. §54's worry that integration moves the network away from
humans stands as a description of the 0.903 → 0.655 step, but the remedy is not
"drop the split" — dropping it makes the behaviour worse.

**A is refuted too**: the split is not decorative on either instrument.

### 72.3 §64.5 could not name this result, and the defect is in the band

The grader returns **UNCLASSIFIED under both policies**, for the same reason
each time. B's substantive clause — shape bias below the reference by more than
the between-seed sd — **is met, with room**: 0.0875 against sd 0.0244
(best saved), 0.1178 against sd 0.0377 (final). B fails only on its accuracy
clause, which §64.5 wrote as a two-sided band ("with accuracy within 0.02"),
and the arm is **+0.0428 outside it in the winning direction**.

So the pre-registration refuses to fire because the model was *too accurate*.
That is a defect in the rule, not in the arm, and §64.3 had already anticipated
the asymmetry it failed to encode: "If the single stream wins or ties, it does
so with half the capacity. The confound strengthens the conclusion and is
irrelevant." §64.5's B should have carried a one-sided accuracy floor
(≥ reference − 0.02), not a band.

Recorded as UNCLASSIFIED rather than quietly read as B. §50.6 is binding: a
pre-registered rule that cannot fail launders a refutation as a confirmation,
and a rule silently widened after the fact does the same thing. The substantive
reading — the split buys shape bias, costs accuracy — is stated as the
interpretation it is, on complete separation under both policies, not as a
threshold that fired.

**C is refuted under the corrected reference.** The single stream does not lose
on accuracy; it wins by 0.043 (best saved) and 0.106 (final). No
parameter-matched `resnet34` control is triggered by C.

### 72.4 Two errors caught in this section's own instruments

1. **`cue_conflict.py --baseline` would have produced a fabricated number.**
   It builds a *torchvision* ResNet on raw 3-channel RGB with no DKL conversion
   and no diet — the thesis's RGB-baseline anchor. `dkl_bcs_single` is a
   `TwoStreamNetwork(baseline=True, combined_color=True)` taking two DKL
   channels. Because `load_state_dict(strict=False)` tolerates missing keys, it
   would have scored a **randomly initialised network** and printed a shape bias
   without erroring. `tmp_analysis/cue_conflict_single.py` mirrors
   `StreamInputBuilder` exactly and **hard-fails** on any missing or unexpected
   key; all six tasks logged `122 tensors, 0 missing, 0 unexpected`. Verified in
   the other direction too: pointed at a two-stream checkpoint it exits 1.
2. **A §71.4 violation inside the §71-aware grader.** The final-checkpoint cell
   initially compared single-stream accuracy against `dkl_bcs`'s **0.5921** —
   the per-epoch maximum naming weights that do not exist — instead of its
   final-checkpoint **0.4446**. That mispairing alone returned a spurious
   `OUTCOME C`. Corrected; `ACC_BCS_PEREPOCH` is retained in the source
   explicitly labelled a schedule property, per §71.4.

### 72.5 Controls that make the comparison readable

- **Decision rate** (§64.6 item 4): single 0.417 vs two-stream 0.418 (best
  saved), 0.399 vs 0.396 (final), all over the same 1200 scored images. The
  shape biases are computed over comparably sized decided subsets, so the
  comparison is not the §54 artefact.
- **Pipeline validation**: seed 2's best saved checkpoint *is* its final one, so
  `ep150` and `final` are the same weights. They return bit-identical 0.570265.
- **Convergence** (§64.6 item 1): this arm did **not** converge either.
  sd(val_top1) over the last 20 epochs = 0.0653 / 0.0286 / 0.0275, and seed 0
  falls 0.589 → 0.454 between epochs 148 and 149. Best-saved is therefore the
  primary policy, with final reported alongside; both give the same answer.

### 72.6 Disclosure

- **One diet placement.** §64.7 stands: this tests the split *for the
  diet-on-luminance arm* and must not be generalised to the swap or
  `parvo_highsf` arms.
- **The capacity confound is not fully retired.** It is irrelevant for the
  accuracy comparison (the smaller model wins). It is **not** disposed of for
  the shape-bias comparison: a parameter-matched single stream (`resnet34`,
  ~21.3M) might close the 0.0875 gap. §64.5's B text pre-registered that arm as
  required, and the substantive outcome is B, so **the `resnet34` control is
  triggered by the letter of the pre-registration** even though C — the branch
  §64.3 tied it to — is refuted. It is 3 seeds × ~26 h and has not been
  submitted; that is a decision, not an oversight.
- **Untrained-side comparisons are unaffected**; this arm is trained-only and
  carries no stream statistics (§64.2), so nothing in §47–§63 moves.
- The single-stream numbers are **3 seeds**, and the 0.5414–0.5899 spread is
  wide relative to the 0.0875 effect. Complete separation, not a p-value, is
  what is claimed.

---

## §73 ALIGNMENT AUDIT — §2.3(b) was quoting the wrong measure, and was
## undersold by roughly an order of magnitude as a result

Full recomputation of §2.3 from the stored per-voxel arrays
(`$CLAUDE_JOB_DIR/tmp/audit23b.py`, `confirm_mislabel.py`), independent of every
previous script. Two directories hold encoding output — `results/encoding` and
`results/encoding_cc` — and the first check was that they agree: on the five
models present in both they are **bit-identical** on the gradient. `encoding_cc`
is a strict superset (it adds the `cc_*` VSB fields), so it is the one to use.

### 73.1 The error

§66.3, and §2.3(b) after it, reported:

> hV4−V1v stream-bias gradient, normalised: diet −0.447/−0.424/−0.501 against
> no-diet −0.202/−0.166/−0.160 … the raw statistic reaches the exact floor only
> in seed 2 (p = .0234, .0156, .0078).

Those numbers are **not the stream-bias gradient**. They are the **joint-VSB
profile step, hV4 − V1(pooled)** — the ROI profile of the *predicted shape-bias
index*, not of the stream-preference index. Recomputing that measure returns
−0.4465/−0.4242/−0.5007 against −0.2017/−0.1664/−0.1601 and raw p = .0234/.0156/
.0078, matching the quoted figures on every digit. The identification is certain.

This matters twice over. First, the VSB profile is a **different quantity** from
the SPI gradient — one is about what the streams represent, the other about where
they explain variance. Second, the VSB profile step is the measure this paper
**explicitly disclaims**: under the full control set it is not diet-specific
(−0.095 vs −0.060, p = .1484/.9453; §43.2). §2.3(b) was therefore supporting a
live claim with numbers from a retired one.

### 73.2 The actual gradient is much stronger

| statistic | seed | diet | no diet | p | subjects |
|---|---|---|---|---|---|
| norm | 0 | −0.8618 | −0.1013 | .0078 | 8/8 |
| norm | 1 | −0.8018 | −0.1318 | .0078 | 8/8 |
| norm | 2 | −0.7813 | −0.0732 | .0078 | 8/8 |
| raw | 0 | −0.1759 | −0.0342 | .0078 | 8/8 |
| raw | 1 | −0.1507 | −0.0406 | .0078 | 8/8 |
| raw | 2 | −0.1595 | −0.0303 | .0078 | 8/8 |

**Six of six cells at the exact floor with complete separation.** The
"weaker instrument … the gradient is corroboration" framing was an artefact of
the mislabel and is withdrawn; §2.3(b) is a claim, not corroboration.

*A definitional trap that nearly produced a second error.* This project uses
`raw`/`norm` to mean *with and without the median(Uₘ+Uₚ) denominator*, both
computed on the **noise-ceiling-normalised** `unique_magno`/`unique_parvo`
(`blur_verdict.py`, `stats.py`, `effect_sizes.py`). The stored
`unique_magno_raw`/`unique_parvo_raw` fields are a *different* thing (un-normalised
by NC) and are used only by `banded_ridge.py` and `group_check.py`. Substituting
them makes the raw between-arm contrast collapse to p = .109/.195/.055 and would
have "killed" §2.3(b). Any future analysis must use the `blur_verdict.py`
convention. **Standing rule: `raw` means no denominator, never the `_raw` fields.**

### 73.3 The controls are stronger too

All on both statistics, paired, n=8, seed 0:

| control | norm | raw | p (both) | subjects |
|---|---|---|---|---|
| no diet | −0.101 | −0.034 | .0078 | 8/8 |
| `swap` | −0.027 | −0.043 | .0078 | 8/8 |
| `parvo_highsf` | −0.119 | −0.108 | .0078 | 8/8 |
| untrained, diet front end | −0.471 | −0.118 | .0078 | 8/8 |
| Gabor bank, diet front end | −0.367 | −0.042 | .0078 | 8/8 |
| Gabor bank, no-blur | −0.621 | −0.101 | .0078 | 8/8 |
| `thesis_dvd_trans` | −0.859 | −0.175 | **.8438** | 5/8, 4/8 |

§48.1 claimed three controls on the normalised statistic only. It is six controls
on both. And `thesis_dvd_trans` **fails to separate** — the correct behaviour for
a cross-codebase convergent replication, now demonstrated rather than asserted.

### 73.4 Supporting numbers corrected

| quantity | was | is | why |
|---|---|---|---|
| no-diet gradient | −0.117 | **−0.101** | −0.117 is the mean of seeds 0–1, paired against a seed-0 diet number |
| ρ, no diet | −0.512 | **−0.502** | §48.1 already said −0.502 |
| dz, no diet | −0.61 | **−1.06** | recomputed |
| "untrained front ends" | −0.488 | **−0.471 / −0.488** | two different baselines were conflated: diet front end vs no-blur front end |
| global stream bias | −0.454 / +0.727 | **−0.495 / +0.849** | retinotopic voxels, the §48.1 definition |
| no-diet reduction | ~76% | **79%** | follows from −0.101 / −0.488 |

### 73.5 What verified clean

§2.3(a) reproduces to four decimals on every number: +1.5340/+1.5659/+1.5576
against +0.1104/+0.0980/+0.0832; per-stream +0.7910 / −0.6376 and −0.7430 /
−0.7479; seed range 0.0319. It also **gains** its raw statistic, which separates
the arms by *sign* (+0.616/+0.658/+0.632 against −0.033/−0.039/−0.053) — a
stronger statement than the normalised one and previously unreported.

§2.7's cost reproduces: 0.4046/0.3817/0.3992 against 0.4637/0.4639/0.4565,
p=.0078 and 8/8 in all three seeds.

### 73.6 Gaps found

1. **`swap` and `parvo_highsf` have cortical fits for seed 0 only.** Two of
   §2.3(b)'s controls are n=1 seed. Now disclosed in §2.3(b); the "no n=1 claim
   in the main results" statement of §66.5 was too strong as written.
2. **The konio result (§70) appears nowhere in the draft** — zero occurrences.
   A complete four-contrast pre-registered result at the exact floor, including
   its specificity contrast, is currently invisible to a reader.

---

## §74 §63.3's SURVIVING HALF DOES NOT SURVIVE §71.10 — cancel the pending
## instruction to fold it into §2.2

§63.3 audited §50's "channel-specific set point" and concluded it splits: the
**swap vs `parvo_highsf`** ordering survives (disjoint seed ranges, gap 0.331,
9/9 pairings) while swap vs luminance does not (overlapping, 7/9, carried by one
checkpoint). An open instruction has stood since then to fold the surviving half
into §2.2.

**Do not.** §63.3 was computed on final checkpoints. Under the best-saved policy
that §71.10 established for §2.2, `swap`'s seeds move (ep100, ep100, ep150) while
`parvo_highsf`'s do not (all three are already ep150):

| policy | swap | `parvo_highsf` | verdict |
|---|---|---|---|
| final (used by §63.3) | 1.995 / 2.225 / 4.383 | 1.497 / 1.640 / 1.664 | disjoint, gap **+0.331**, **9/9** |
| **best saved (§71.10)** | **1.541** / 1.995 / 2.327 | 1.497 / 1.640 / 1.664 | **overlap 0.123**, **7/9** |

7/9 with overlapping ranges is *precisely* the pattern §63.3 used to reject the
swap-vs-luminance half — same pairing count, same overlap structure. Applying
§63.3's own criterion to its own surviving half retires it.

**So §50 is now dead in both halves**, and the reason is the same one that
inflated §2.2's seed range (§71.10): the ordering was carried by `swap` seed 1's
final checkpoint (4.383), a weight file 0.786 above where that seed's own best
saved checkpoint sits (1.541). §2.2 continues to claim the side of parity and no
level, no set point, and no channel ordering.

This is the fourth claim retired by the checkpoint correction (§49.4, §49.2,
§50's luminance half, and now §50's channel half) and the second time a result
that looked seed-robust turned out to rest on late-training degradation.

---

## §75 THE FATAL TEST PASSES — P3 HOLDS. The conjunction is not an accuracy
## artefact (seed 0; seeds 1–2 still queued)

Test B (§67/§69.2), the accuracy-matched no-diet control, is the last fatal test
in this project. It exists because the NMI reviewer simulation's major comment M1
was that the diet arm's dissociations might follow from its accuracy gap rather
than from the diet. §69.3 P3 and §69.4 named the fatal outcome in advance:

> **P3.** An accuracy-matched no-diet model does **not** acquire the diet's shape
> bias. If a no-diet model at 0.556 shows a shape-biased magno stream, then §2.1
> is an accuracy effect and the paper's central claim is wrong. … There is no
> reframing that survives that.

### 75.1 The checkpoint, chosen by rule

Array 26423530 task 0 COMPLETED (13 h 22 m), 12 epochs at `--ckpt-every 1` into
`/scratch/byrong/VST_data/mpk_runs_matched/dkl_base_seed0`. Target 0.556, the
diet arm's best-saved mean (§71.3). Selected in-script by minimising
|val_top1 − 0.556| over the checkpoints that exist, with an assertion:

| ckpt | val_top1 | \|diff\| |
|---|---|---|
| 009 | 0.5416 | 0.0144 |
| **010** | **0.5480** | **0.0080** |
| 011 | 0.5893 | 0.0333 |
| 012 (unsaved) | 0.5680 | 0.0120 |

`checkpoint_010.pt`, val_top1 **0.5480** against the diet's 0.556 — matched to
0.008, tighter than the diet arm's own 0.012 seed range.

### 75.2 The result, on both instruments

| instrument | diet (best saved, 0.556) | no diet (final, 0.70) | **matched (0.548)** |
|---|---|---|---|
| probe, magno bias | **+0.806** | −0.226 | **−0.148** |
| cue-conflict, magno-only | **0.903** | 0.421 | **0.455** |

**Matching accuracy closes 7.6% of the probe gap and 7.0% of the behavioural
gap.** Two independent instruments, built on different machinery, agree to
within 0.6 percentage points on how little of the effect accuracy explains.

The matched model stays inside the no-diet regime on both. Its probe bias is
still **negative** — texture-biased, on the same side of the sign flip as every
no-diet seed — and its cue-conflict magno bias sits 0.021 above the no-diet seed
maximum, a shift smaller than that arm's own seed range would need to be
surprising (width 0.036).

**P3 HOLDS. The fatal branch of §69.4 is not taken.** Accuracy is not what
produces the conjunction effect of §2.1, and M1 is answered with the control it
asked for rather than with an argument.

### 75.3 Its strength, stated honestly

- **n = 1 seed.** Tasks 1–2 of 26423530 are still PENDING on `agcichy`. The
  claim above rests on one matched training run. Report it as such until they land.
- The matched run reached 12 of its 20 epochs; that is *why* it can be matched
  at all (no saved 150-epoch no-diet checkpoint is below 0.658), and the
  selection rule only ever needed a checkpoint bracketing 0.556, which epochs
  9–11 supply.
- The direction of the residual shift is *toward* shape (+0.079 probe, +0.034
  behaviour). That is worth reporting rather than hiding: lowering a no-diet
  model's accuracy does move it slightly toward the diet, just nowhere near far
  enough, and the honest number is 7%, not 0%.

### 75.4 What is now closed

With P3 held, every fatal pre-registration in this project has resolved in the
paper's favour or been survived: §57.5 (error consistency — refuted my own
hypothesis, honoured), §67.4/§69.4 (accuracy confound — held), §71.6 (OOD at
best checkpoints — held), §71.8 P3 (PR side of parity — held), §64/§72 (the
split — refuted outcome D). **No fatal test remains outstanding.**

---

## §76 THE KONIO ARM IS WRITTEN INTO THE DRAFT — verified by re-run, with one
## ranking claim in §70.2 corrected

The user asked for the konio arm to be added to the paper. Before writing it in,
`tmp_analysis/konio_verdict.py` was re-run end to end against the on-disk fits
rather than transcribing §70's tables.

### 76.1 Everything reproduces, except one sentence

**Reproduced exactly**, matched cell (primary), n=8, NC-normalised, with the raw
statistic in parentheses:

| contrast | Δ | subjects | p | d_z |
|---|---|---|---|---|
| hV4 | −0.02930 (−0.00991) | 0/8 | .0078 | −2.09 |
| VO (VO1,VO2) | −0.02381 (−0.00748) | 0/8 | .0078 | −1.95 |
| colour set | −0.02564 (−0.00829) | 0/8 | .0078 | −2.06 |
| colour set − other 17 | −0.01402 (−0.00493) | 0/8 | .0078 | −1.51 |

The §65.4 gate passes **bit-exact, 0.000e+00, every voxel of all 8 subjects in
both cells** — so Δ really is R²_joint(konio) − R²_joint(parvo) with zero
magno-side estimation error, as pre-registered. The unmatched cell reproduces
too (hV4 −0.00586 0/8 .0078; VO −0.00609 2/8 .1094; colour set −0.00601 2/8
.0391; specificity −0.01388 0/8 .0078), including the load-bearing invariance:
specificity −0.0140 matched against −0.0139 unmatched, while the mean over the
other 17 ROIs moves from −0.0116 to +0.0079 and konio wins V1d outright
(+0.077, 8/8). `konio_verdict.py` returns **OUTCOME D** from both cells.

**Corrected.** §70.2 claimed konio's "three worst ROIs of twenty" were hV4, VO1
and VO2, "precisely the colour set". They are not. Sorted worst-first in the
matched cell: hV4 −0.02930, VO1 −0.02744, **V3v −0.02087**, VO2 −0.02019. The
colour set is ranks 1, 2 and **4**; ventral V3 interleaves. The same
interleaving holds under the raw statistic (hV4 −0.00991, VO1 −0.00880, V3v
−0.00675, VO2 −0.00615), so it is not a units artefact. §70.2 is corrected in
place.

Nothing else in §70 moves, because the ranking was never load-bearing: the
specificity test is a paired Wilcoxon on colour-set-minus-rest and does not use
ranks. The draft states the corrected version and does not repeat the ranking
claim at all.

### 76.2 A terminology collision, resolved in the draft

"Raw" now means three different things across this project, which is how §73.2's
near-miss happened:

1. **§2.3(b) / `blur_verdict.py`** — the SPI *without its (Uₘ+Uₚ) denominator*.
   Both raw and norm take NC-normalised `unique_magno`/`unique_parvo` as inputs.
2. **§70 / `konio_verdict.py`** — the `unique_*_raw` fields, i.e. *without the
   noise-ceiling denominator*. Verified against `banded_ridge.py:226-234`, where
   `unique_magno_raw = unique_m` and `unique_magno = unique_m / nc`.
3. **§65.3** — the `untrained_konio_raw` **cell**, i.e. the *unmatched input
   contrast* arm of the design.

All three are internally correct; the word is not. The draft therefore calls the
cells **contrast-matched** and **unmatched**, never "raw", so that "raw" in the
paper only ever means a statistic. This does not change any number.

### 76.3 Where it landed

Per §70.5 ("not a section; one paragraph plus a supplementary panel, strengthening
leg (c)"), and since leg (c) is the cortical hV4−V1v gradient (§9's three legs),
the arm went into **§2.3(b) as a fifth bullet** — the point being that the
gradient's ventral end is specific to L−M and not to chromaticity as such, which
is a control on the gradient's *interpretation*, not a finding of its own. The
positioning rule is honoured: the surviving positive claim leads, the konio
failure is its control.

Also added: a Methods paragraph (§4.5) covering the two cells, the gate and the
atlas, and a **Claims deliberately not made** entry recording the three things
§70.4 says the arm does *not* close — it is an axis swap and not a third stream,
it is untrained and single-seed, and no blur was applied to K.

Still true and still disclosed in the draft: pooled VO reaches the floor in the
matched cell only.

---

## §77 PRE-REGISTERED — equivalence bounds for the three dissociations

Written and committed **before** any interval is computed. Prompted by the NHB
referee simulation (`docs/NHB_REVIEW_2026-09-06.md`, major comment 4): the
paper's three most-promoted contributions each rest on failing to reject, and
NHB's reporting policy forbids stating a null as "no effect" without an
equivalence bound or Bayes factor. The referee's exact ask is the deliverable:
*"state, for each, what difference the design could have detected and what
interval the data exclude."*

### 77.1 What is and is not blind here

The **bounds** below are blind: none is chosen from the variance or the interval
of the comparison it is applied to. Each is anchored to a *different*, already-
published quantity, fixed by claims the paper has been making for weeks.

The **point estimates are not blind** — §2.5's phase-scrambling Δ = +0.009 and
§2.3's VO median = +0.0044 are already in the draft, and I have read them.
Disclosed rather than concealed, because an equivalence bound chosen after
seeing a point estimate is worthless and the reader must be able to check that
these were not. What protects this pre-registration is that the SESOIs are
functions of *other* results, so knowing the point estimate cannot move them.

### 77.2 The principle for every bound

**SESOI = the smallest effect the paper already reports as real, in the same
measure and the same design.** The argument is symmetric and hard to game: if a
difference here were as large as the smallest difference we do claim elsewhere,
we would be obliged to claim it here too. So excluding that magnitude is exactly
the statement "this null is not hiding an effect we would have called real."

### 77.3 The four tests, with bounds fixed now

| # | null | design | SESOI | anchored to |
|---|---|---|---|---|
| 1 | §2.5 phase-scrambling, diet vs no-diet relative AUC | 3 v 3 seeds, independent | **±0.041** | rotation (−0.041), the smallest OOD effect the paper claims |
| 2 | §2.6 diet vs no-diet error consistency κ, whole model | 3 v 3 seeds, independent | **±0.0234** | 10% of the human–human ceiling (0.234), the width that moves a model across the observed 7–18% band |
| 3 | §2.6 slope of κ on shape bias across 6 models | n = 6 models | **±0.1625 κ per unit shape bias** | a slope carrying κ by half the ceiling (0.117) over the observed 0.720 shape-bias span |
| 4 | §2.3 VO median(Uₘ − Uₚ) | n = 8 subjects, paired | **±0.0195** | hMT+ (+0.0195, p = .0234, 7/8), the smallest ROI effect this measure reports as real |

### 77.4 Procedure, fixed now

- **TOST** at α = .05, reporting the **90% CI** on the difference (the interval
  whose containment within ±SESOI is equivalent to both one-sided tests
  rejecting). Tests 1–2: Welch. Test 4: paired. Test 3: OLS slope CI.
- **Sensitivity** reported alongside: the effect detectable at 80% power given
  the observed sd and n. This is the half of the referee's question that TOST
  alone does not answer.
- **Nonparametric floor, stated up front.** At 3 v 3 the exact rank-sum p floors
  at 0.10, so tests 1–2 *cannot* return a nonparametric equivalence result at
  any bound. The parametric interval is reported with that limitation named, not
  as though the choice of test were free.

### 77.5 The outcome that would hurt

If a 90% CI is **wider** than its ±SESOI, the honest report is *"the design
cannot exclude an effect as large as the smallest one we claim,"* which weakens
the corresponding dissociation. §2.5 (3 v 3) is the most exposed. That outcome
will be reported as a limitation of the claim, not buried in a supplement and
not re-tested at a wider bound: **the bounds in §77.3 are final.** §50.6 is
binding — a bound widened after seeing the interval launders a failure.

### 77.6 §2.4 is deliberately excluded, and why

The referee grouped §2.4 with the nulls. It does not belong there. §2.4's claim
("dimensionality is not a proxy for shape bias") rests on a **sign reversal in a
decisive case** — `parvo_highsf`'s second stream is the *more* dimensional one
(0.56) while remaining texture-biased (probe −0.266) — not on a failure to
reject. An equivalence test on a correlation across n = 4 front ends would be
uninformative at any bound, and computing one would dress a qualitative
dissociation in inferential clothing it does not need. The correlation and its
interval will be reported with "n = 4, uninformative" stated explicitly, and the
claim will continue to rest on the decisive case.

---

## §78 THE §77 EQUIVALENCE TESTS RESOLVE — all four nulls are bounded, and two
## draft claims do not survive the arithmetic

`tmp_analysis/equivalence_tests.py`, output in
`results/equivalence/equivalence_tests_2026-09-06.txt`. Bounds fixed in 03ca8fc
before any interval was computed; none was changed afterwards (§77.5).

### 78.1 The pipeline reproduces the published numbers first

Nothing below was read until these matched:

- human–human ceiling **κ = 0.234** over 45 pairs — asserted in code, not eyeballed;
- VO Wilcoxon **p = 0.4609** against the draft's p = .46;
- every κ and shape-bias value inside the draft's published per-condition ranges
  (diet κ .0234/.0319/.0280 vs the table's .023–.032; no-diet .0309/.0333/.0372
  vs .031–.037).

### 78.2 All four nulls clear their declared bound

| # | null | Δ | 90% CI | SESOI | TOST p | verdict |
|---|---|---|---|---|---|---|
| 1 | phase-scrambling, diet − no diet | +0.0092 | [−0.0145, +0.0330] | ±0.041 | .0287 | **equivalent** |
| 2 | κ, diet − no diet (whole model) | −0.0060 | [−0.0127, +0.0006] | ±0.0234 | .0030 | **equivalent** |
| 3 | slope of κ on shape bias | −0.0154 | [−0.0524, +0.0217] | ±0.1625 | — | **equivalent** |
| 4 | VO median(Uₘ − Uₚ) | +0.0045 | [−0.0082, +0.0171] | ±0.0195 | .0297 | **equivalent** |

**Report the achieved bound, not the declared one.** The declared SESOIs were
deliberately generous (§77.2 anchors them to the smallest effect the paper
claims). The data support tighter statements, and those are what the manuscript
should quote:

- phase-scrambling: the difference is **within ±0.033**, below the smallest OOD
  effect the paper claims (rotation, 0.041);
- κ diet vs no-diet: **within ±0.013**, i.e. ±5.4% of the ceiling — the declared
  ±0.0234 was nearly the whole between-model κ spread and is the weaker claim;
- VO: **within ±0.017**, below hMT+'s +0.0195;
- slope: across the observed 0.706 shape-bias span the data exclude a κ change
  larger than **0.037, i.e. 15.8% of the human–human ceiling**.

That last line is the one to put in the manuscript. It converts "shape bias does
not track human decision structure" from a failure to reject into: *spanning 5×
in shape bias moves trial-level human agreement by at most 16% of the
human–human ceiling.*

### 78.3 Two caveats that must travel with these numbers

1. **Test 1 rests on a Welch df of 2.3.** Three seeds per arm. The CI half-width
   (0.0238) sits at 58% of the bound, so it clears — but on a t interval with
   ~2 df, and §77.4 declared in advance that at 3v3 no *nonparametric*
   equivalence result is obtainable at any bound. This is the weakest of the
   four and must be labelled as such, not listed as a peer of tests 2–4.
2. **Test 4's design is marginally underpowered relative to its own bound.** The
   effect detectable at 80% power is **0.0220**, slightly *above* the ±0.0195
   SESOI. Equivalence is achieved because the point estimate sits near zero
   (+0.0045), not because n = 8 is highly sensitive. Both facts belong in the
   text; quoting the TOST pass alone would overstate the design. The percentile
   bootstrap agrees independently ([−0.0065, +0.0142], 20k resamples).

### 78.4 Two claims in `PAPER_DRAFT.md` §2.6 that the arithmetic contradicts

Found while assembling the slope, not sought. **Not edited — `PAPER_DRAFT.md` is
another session's file** (seventh cross-session collision avoided); flagged here
for its owner.

1. **"no model of ours exceeds AlexNet" is false, by the draft's own table.**
   AlexNet κ = 0.032. The table's own "no diet, magno stream" row is
   **0.036–0.041** (computed mean 0.0386), and "no diet, whole model" is
   0.031–0.037 (0.0338). Both exceed AlexNet. The sentence contradicts the table
   three lines above it.
2. **"six models" is a miscount**, appearing 4× including the abstract (lines 69,
   136, 727, 841). The §2.6 table has **seven** rows: four of ours (diet and
   no-diet × full and magno readouts) plus AlexNet, GoogLeNet, VGG. The slope in
   §78.2 uses all seven and says so. If "six" was meant to count distinct trained
   networks it should be five (diet, no-diet, and the three published), so
   neither reading gives six.

**A stronger, true replacement is available for claim 1.** The rhetorical point
survives in sharper form: our **most** shape-biased model has the **lowest**
error consistency of all seven. `diet/magno` carries shape bias 0.878 — 5× VGG's
0.172 — and returns κ = 0.0089 (3.8% of ceiling) against VGG's 0.0170 (7.3%).
Not "no more agreement than AlexNet" but *less agreement than the least
shape-biased model in the comparison*, which is the dissociation stated at its
strongest and is what the table actually shows.

### 78.5 What this does and does not buy

Answers NHB major comment 4 for §2.5, §2.6 and the VO null, and converts the
Open Science audit row "Null results reported with equivalence/BF — **Fail**" to
a pass for these four. It does **not** address §2.4, excluded with reasons in
§77.6 — that claim rests on a sign reversal, not on a failure to reject, and no
equivalence test is owed there. It also does not touch the referee's items 1
(References), 5 (multiplicity) or 6 (ethics block).

---

## §79 WHAT `±` MEANS — resolved by recomputation, and the manuscript-wide
## definition that was about to be written wrong

### §79.1 Why this is worth a section

The NHB v2 referee simulation (`docs/NHB_REVIEW_2026-09-06_v2.md`, commit
`0cb7255`) records **two** hard overrides holding the paper at Major revision,
and both are clerical rather than empirical:

> **Fatal statistical item → YES.** `±` remains undefined manuscript-wide while
> carrying the §2.5 headline table and §2.3's `0.675 ±0.025`. **Caps at Major
> revision.**

The Statistical Rigor Audit scores the same row **Fail (fatal)**. Closing it
costs one sentence in §4.7. The trap is that the sentence has to be *true*, and
nobody now remembers which estimator produced each printed value — the numbers
were emitted by half a dozen scripts over several weeks.

The failure mode this section exists to prevent: a definition written to satisfy
a referee item, which misdescribes the numbers it defines. That is strictly
worse than leaving `±` undefined, because it converts a presentational gap into
a factual error — the §50.6 pattern, where a rule written to close a hole
launders the hole instead of fixing it. A referee who checks one cell (and this
one demonstrably checks) then finds not a missing definition but a wrong one.

### §79.2 The definition that was about to ship

A draft of the fix appeared in `PAPER_DRAFT.md` (~line 1297) as:

> **Uncertainty notation.** Throughout, `±` denotes the **range across the three
> training seeds** (max − min), not a standard deviation, standard error or
> confidence interval [...] the human observer figure 0.958 ±0.019 is the
> **range across the ten observers** [...] Where a within-subject quantity is
> summarised across the eight NSD subjects, the subject-level spread is
> reported as an explicit range or as dz, **never as `±`**.

All three of those statements are false. They are not close.

### §79.3 The arithmetic

`tmp_analysis/uncertainty_notation.py` recomputes every plausible estimator from
the source data and matches it against what the manuscript prints, at the
precision the manuscript prints it. Saved output:
`results/uncertainty_notation/pm_notation_2026-09-06.txt`.

| cell | unit | printed | SD(ddof=1) | SD(ddof=0) | SEM | range | half-range |
|---|---|---|---|---|---|---|---|
| §2.3 diet, integrated | 3 seeds | **±0.025** | **0.0249** | 0.0203 | 0.0144 | 0.0458 | 0.0229 |
| §2.3 no diet, integrated | 3 seeds | **±0.020** | 0.0197 | 0.0161 | 0.0114 | 0.0384 | 0.0192 |
| §2.3 human observers | 10 obs | **±0.019** | **0.0191** | 0.0181 | 0.0060 | 0.0675 | 0.0338 |

Per-seed values are 0.6922 / 0.6861 / 0.6464 (diet) and 0.4212 / 0.4328 /
0.3944 (no diet), from `results/cue_conflict/*.json`.

**The identification is by intersection, not by any single cell.** The no-diet
cell is printed to two decimals, where SD(ddof=1), SD(ddof=0) and half-range all
round to 0.020 — it constrains but cannot identify. The other two are printed to
three decimals and each identifies SD(ddof=1) uniquely. Since the manuscript
uses one notation, the estimator must satisfy every cell simultaneously, and
exactly one does:

> **`±` is the sample standard deviation (ddof=1) about the mean, over the unit
> named in each case.**

`range (max − min)` is excluded by all three cells independently. The human cell
refutes it on its own: the range is 0.0675, **3.6×** the printed 0.019 — and the
draft already prints that range in the same sentence, as "range 0.919–0.987".
The value and its own stated refutation sit nine words apart.

### §79.4 Provenance, independently of the arithmetic

The arithmetic is confirmed by the emitting code, which was found afterwards and
agrees:

* `tmp_analysis/geirhos_humans.py:61-62` builds the human string as
  `± {v.std(ddof=1):.4f}` and `, range {v.min():.3f}–{v.max():.3f}` — **two
  separate fields**. Running it prints, verbatim:
  `human mean: 0.9576 ± 0.0191 (n=10, range 0.919–0.987)`.
  So the `±` is the SD and the range is the parenthetical beside it. The draft's
  claim that the `±` *is* the range mistakes one field for the other.
* `tmp_analysis/human_shapebias.py:118,120` likewise uses `std(ddof=1)`.

### §79.5 The second estimator — `±` is not one thing

The §2.4/§2.5 PR and SPI tables (draft ~lines 557-561 and 685-688) carry `±`
across **8 NSD subjects**, e.g. `1.402 ±0.089`. These come from
`tmp_analysis/pr_table.py:41`, which prints `r.std()` — numpy's **default
ddof=0**, a population SD. That is a different estimator from §79.3's.
`tmp_analysis/pr_seeds.py:118` uses `ddof=1` again.

So the manuscript currently uses **at least two** dispersion estimators under
one unexplained symbol. This also refutes the third clause of §79.2: those
tables *are* subject-level spread reported as `±`, which the proposed definition
asserts never happens.

At n=8 the ddof=0/ddof=1 gap is √(8/7) ≈ **7%**, so this is not cosmetic — it is
a systematic ~7% understatement of the spread in every PR/SPI cell.

**Confirmed numerically**, by recomputing the participation ratios per subject
from `/scratch/byrong/VST_data/mpk_feature_cache` (n=8 for every cell). `sd0`
reproduces the printed value in **eight independent cells**; `sd1` reproduces
none of them:

| draft cell | printed | mean | **sd0** | sd1 |
|---|---|---|---|---|
| `dkl_bcs_seed0` front end | 1.402 ±0.089 | 1.402 | **0.089** | 0.095 |
| `dkl_bcs_parvo_highsf_seed0` front end | 0.586 ±0.013 | 0.586 | **0.013** | 0.014 |
| `dkl_bcs_swap_seed0` front end | 1.909 ±0.125 | 1.909 | **0.125** | 0.133 |
| `dkl_base_seed0` front end | 1.634 ±0.118 | 1.634 | **0.118** | 0.126 |
| `dkl_base_seed0` trained | 0.947 ±0.035 | 0.947 | **0.035** | 0.038 |
| `dkl_base_seed1` trained | 0.988 ±0.040 | 0.988 | **0.040** | 0.043 |
| `dkl_bcs_parvo_highsf_seed0` trained | 1.640 ±0.030 | 1.640 | **0.030** | 0.032 |
| §2.5 `gabor_swap` | 3.37 ±0.16 | 3.366 | **0.163** | 0.174 |

(The `dkl_bcs_seed0`/`swap` *trained* cells in the §2.4 table are the best-saved
checkpoints per §71.8, not the `_trained` cache entries used above, so they are
not directly comparable and are omitted rather than mismatched.)

So the two-estimator finding is established, not inferred from the source alone:
**behavioural and human values are ddof=1; every PR/SPI cell is ddof=0.**

**Recommendation:** recompute those tables at ddof=1 so a single sentence covers
the manuscript, rather than documenting two estimators.

### §79.6 A notation collision that inflates the multiplicity count

The draft's new Multiplicity paragraph opens "This paper reports 36 p-values."
That count is not defensible, because ~7 of the matches are not p-values: they
are kornia augmentation probabilities — "horizontal flip (p = 0.25)", "rotation
±15° (p = 0.25)", "colour jitter (brightness 0.4, p = 0.5)" (~lines 1132-1134)
and `RandomGrayscale(p=0.5)` (~line 1650). A referee counting `p =` to check the
family sizes will hit these, as this audit did.

Rename them `prob.` or `p_aug`. Note the same lines also contain `rotation
±15°`, where `±` is a stimulus parameter and not a dispersion at all — the
definition sentence should say it applies to reported measurements.

### §79.7 What to write instead

> **Uncertainty notation.** Unless stated otherwise, `±` denotes one sample
> standard deviation (ddof = 1) about the mean, computed over the unit named in
> each case: three training seeds for model behavioural values, ten observers
> for the human reference, eight subjects for the PR and SPI tables. Ranges,
> where given, are printed separately and labelled as ranges. Cue-conflict
> values additionally carry a bootstrap 95% CI over cue-conflict images (§4.3).
> `±` in a stimulus specification (e.g. rotation ±15°) is a parameter, not a
> dispersion.

Conditional on §79.5 being actioned; if the PR/SPI tables are left at ddof=0,
that clause must say so explicitly instead.

### §79.8 Ownership

`PAPER_DRAFT.md` is held by a parallel session, which is actively editing it
(the Uncertainty-notation and Multiplicity paragraphs and the Reporting block
all appeared during this audit, and it holds array 26439369, the fusion sweep).
Per §70.7 this session did **not** edit the draft. The findings above were sent
to it directly with the arithmetic. This section is the durable record; the fix
belongs to whoever holds the file.

Also passed to that session: its fusion sweep will **not** reproduce the
published `magno_only` 0.8877 / `parvo_only` 0.3724 at the sweep endpoints, and
that is correct rather than a bug. `magno_only` is `classifier(0.5·y1)` — an
ablation holding w = 0.5 — whereas sweep `w = 1` is `classifier(1.0·y1)`. The
classifier is affine (AdaptiveAvgPool → Flatten → Linear **with bias**), so
scaling the feature vector does move the argmax. Ablation and reweighting are
different interventions; only `w = 0.5` is a reproduction anchor, and it must
match `full` exactly.

---

## §80 THE CONFIRMATORY FAMILY DOES NOT SURVIVE HOLM — and the sentence saying
## it does is checkable and false

### §80.1 The claim

The draft's new Multiplicity paragraph (the other half of the referee's fatal
statistical item) states:

> (i) The **pre-specified confirmatory family** [...] comprises 18 tests
> declared before the corresponding analyses; Holm–Bonferroni across these
> leaves every claim we make significant, because each rests on the exact
> Wilcoxon floor p=.0078 at n=8 and 18 × .0078 = .140 is the largest adjusted
> value only for the single least extreme test, which we do not claim.

`tmp_analysis/multiplicity_check.py` computes this. Saved output:
`results/multiplicity/holm_family_2026-09-06.txt`. The sentence is wrong twice,
and the underlying situation is worse than the sentence admits.

### §80.2 The arithmetic

The exact two-sided Wilcoxon signed-rank test at n=8 has a **floor** — a
smallest attainable p-value, reached when all eight subjects move the same way.
Verified three independent ways, all agreeing to machine precision:

| route | value |
|---|---|
| enumeration over all 2⁸ sign assignments | 0.0078125 |
| `scipy.stats.wilcoxon(..., mode="exact")` on an all-same-sign vector | 0.0078125 |
| closed form 2 / 2⁸ | 0.0078125 |

Holm's first step compares the **smallest** p against α/m:

    α/m = .05 / 18 = .0027778
    floor          = .0078125        ->  floor > threshold

The smallest attainable p-value in the family is already larger than the
threshold the smallest p-value must clear. Holm therefore stops at step 1 and
**rejects nothing**: 0 of 18, every adjusted p = .1406.

### §80.3 Both halves of the sentence fail

1. **"only for the single least extreme test"** inverts the correction. Holm
   attaches the **largest** multiplier (m) to the **smallest** p-value, not the
   largest. With all 18 tests tied at the floor, every one of them adjusts to
   .1406 — the .140 figure is universal in this family, not a worst case that
   applies to a test we decline to claim.
2. **"leaves every claim we make significant"** is the opposite of the result.
   It leaves *none* significant.

### §80.4 The real finding, which is structural

This is not a defect of these data and cannot be fixed by better effects. An
n=8 exact test **cannot emit a p below .0078125 no matter how large the
effect** — a perfect 8/8 separation with dz = −11.70 returns exactly .0078125,
the same as a marginal 8/8. So the largest family in which such a test can
reach corrected significance at α=.05 is

    m ≤ α / floor = .05 / .0078125 = 6.4   ->   m_max = 6

| m | α/m | can an n=8 exact test survive? |
|---|---|---|
| 6 | .00833 | yes |
| 7 | .00714 | **no** |
| 18 | .00278 | **no** |

Any confirmatory family larger than six tests is unreachable. The paper declares
eighteen.

### §80.5 What to do about it — and what not to

**Not** this: shrinking the family post hoc until it fits under six. Choosing
the family after seeing which tests need to survive is exactly the practice
multiplicity control exists to prevent, and §77's own principle (bounds fixed
before computing) forbids it.

**Not** this either: switching to FDR to obtain a friendlier number. BH at m=18
with all p tied at .0078125 gives adjusted .1406 as well; the floor problem is
not a choice-of-procedure problem.

The defensible move is the one the paper's own paragraph already reaches for in
its last sentence, promoted from caveat to the actual position:

> With n = 8, the exact signed-rank test is resolution-limited: its smallest
> attainable two-sided p-value is .0078, so within a pre-specified family of 18
> tests no result can reach Holm- or FDR-corrected significance at α = .05
> regardless of effect size. We therefore do not rest any confirmatory claim on
> a corrected p-value. For every such claim we report the complete subject
> separation (8/8 or 0/8) and the paired effect size alongside the uncorrected
> exact p, and treat the separation and effect size as the evidence.

That is honest, it is accurate, and it costs the paper nothing it actually has —
the claims were never resting on the p-values, as §2.1's dz = −11.70 and the
8/8 counts throughout make clear. It also converts the Statistical Rigor Audit's
"Multiple comparisons declared and corrected — **Fail**" into a declared family
with a stated, correct treatment.

### §80.6 Relation to §79

§79 and §80 are the two halves of the same referee override, and they fail the
same way: a sentence written to close a checkable item, which is itself
checkable and does not check out. §79's `±` definition misdescribes an
estimator; §80's multiplicity sentence misdescribes a correction. Both would
have been caught by the first referee who reran one line of arithmetic — and the
v2 simulation demonstrates that this referee reruns arithmetic.

The general rule for this manuscript, stated once: **any sentence added to
satisfy a reviewer must be verified against the data it describes before it is
committed**, on the same standard as a result. A clerical fix is still a claim.

---

## §81 §79.5 REVISED — there are THREE estimators, not two, and one table
## column is internally mixed

### §81.1 How this was found

§79.5 concluded that the manuscript carries two dispersion estimators: ddof=1
for the behavioural and human values, ddof=0 for the PR/SPI tables, and
recommended unifying the latter at ddof=1. The draft owner recorded that as an
open item (`bf8d90a`) and asked for the replacement numbers.

`tmp_analysis/pr_ddof1_table.py` produces them. It was written so that a cell is
never "corrected" on the strength of a remembered mapping: each printed cell is
tied to a feature-cache entry, and **both** its mean and its currently-printed
dispersion are re-derived and checked at the printed precision before any
replacement is emitted. A mismatch fails the run.

On the first run it failed, on exactly two of seventeen cells.

### §81.2 What the guard caught

| cell | printed | recomputed mean | recomputed sd0 |
|---|---|---|---|
| `dkl_bcs_seed0` / after training | 1.703 ±0.059 | 1.7033 ✓ | 0.0551 ✗ |
| `dkl_bcs_swap_seed0` / after training | 2.327 ±0.085 | 2.3273 ✓ | 0.0793 ✗ |

The **means reproduce to four significant figures** while the dispersions do
not. That combination is diagnostic: the cache mapping is right — it is the same
model on the same subjects — and the disagreement is in the estimator alone.

And the printed values are exactly the ddof=1 ones:

    0.05507 × √(8/7) = 0.0589  ->  printed 0.059
    0.07933 × √(8/7) = 0.0848  ->  printed 0.085

### §81.3 Why, and what it means

Both are the **best-saved** cells re-scored under §71.8 (ep125 and ep100). That
re-scoring ran through `pr_seeds.py:118`, which uses `std(ddof=1)`, whereas the
original table came from `pr_table.py:41` with numpy's default. The estimator
changed when the checkpoint policy changed, and nobody noticed because the two
scripts print into the same table.

So:

* the manuscript carries **three** groups, not two — behavioural/human (ddof=1),
  original PR/SPI cells (ddof=0), and the §71.8 re-scored cells (ddof=1);
* the §2.4 **"after training" column is internally mixed**: two of its five
  cells use one estimator and three use the other, in adjacent rows of one
  table.

### §81.4 The averted error

§79.5's recommendation as written — "recompute those tables at ddof=1" — is
**too broad, and following it literally would have introduced two new errors**,
re-scaling already-correct values to 0.063 and 0.091. A blanket correction
applied to a set that is not uniform corrupts the members that were already
right.

This is the §79/§80 lesson recurring one level up, and this time it was my own
recommendation rather than the draft's: a fix written to close an audit finding
is itself a claim, and needs verifying against the data before it ships. The
only reason it did not ship is that the module checks each cell instead of
trusting the class. **A correction pass must verify per item, never per
category** — the categories are exactly what turned out to be wrong.

### §81.5 Disposition

The remaining 15 cells reproduce their printed ddof=0 values exactly, so their
mapping is confirmed and they are the ones that change. Replacement values are
in `results/pr_ddof1/pr_spi_ddof1_2026-09-06.txt`, with the two already-ddof=1
cells explicitly marked *unchanged*.

No mean moves. Every dispersion that moves moves up by the same √(8/7) = 1.0690,
so no ordering, no separation and no claim in either table is affected; this is
a notation unification and nothing else. Sent to the draft owner, who holds
`PAPER_DRAFT.md`; this session did not edit it.

---

## §82 THE Uₘ/Uₚ PARTITIONING IS VALIDATED — it recovers a known split and does
## not manufacture a gradient

### §82.1 Why this was owed

§2.2's entire claim runs through `variance_partition`
(`banded_ridge.py:208-223`), which is pure subtraction:

    unique_m = r2_joint - r2_parvo
    unique_p = r2_joint - r2_magno

Its own docstring warns that "with correlated bands, unique components
understate a band's total contribution". Our two bands are strongly correlated —
same images, same architecture, adjacent front ends — so the instrument is used
exactly in the regime its author flagged, and it had never been validated. The
NN referee simulation ranked this first, correctly: if the pipeline can produce a
V1v→hV4 gradient out of correlated bands and ROI-varying SNR alone, §2.2 is an
artefact and no amount of confound control rescues it.

### §82.2 Design

`tmp_analysis/vp_recovery.py`. Real cached magno/parvo features, so the
inter-band correlation is the true one; only the voxel responses are synthetic.
Everything runs through the production `fit_banded` + `variance_partition`, not
a reimplementation.

Ground truth comes from **canonical correlations**. For orthonormal bases the
singular values of Qmᵀ Qp *are* the canonical correlations, so a source built
from the σ≈0 direction is genuinely unpredictable from the other band and one
built from the σ≈1 direction is shared. A canonical direction with correlation σ
has population R² = σ² from the other band, which makes the split exact:

    R2_magno = [a²  + b²σ_p² + c²      ] / v
    R2_parvo = [a²σ_m² + b²  + c²σ_s²  ] / v
    R2_joint = 1
    unique_m = 1 - R2_parvo,   unique_p = 1 - R2_magno

**Condition A (recovery)** — the true split varies monotonically across seven
simulated ROIs. Does the estimate track it?

**Condition B (manufacture)** — every ROI gets an *identical* true split, and
ROIs differ only in noise ceiling. Any gradient here is manufactured. B is the
decisive one, and it is robust to residual uncertainty in the true magnitudes:
it needs the truth to be *the same* across ROIs, not exactly known.

### §82.3 Result (pilot, 60 components, subject 1)

| ROI | nc | A: true Uₘ−Uₚ | A: est | error |
|---|---|---|---|---|
| 0 | 0.15 | −0.2001 | −0.1927 | +0.0074 |
| 1 | 0.20 | −0.0780 | −0.0535 | +0.0245 |
| 2 | 0.25 | +0.0571 | +0.0805 | +0.0234 |
| 3 | 0.30 | +0.1971 | +0.2106 | +0.0135 |
| 4 | 0.35 | +0.3326 | +0.3388 | +0.0062 |
| 5 | 0.40 | +0.4548 | +0.4545 | −0.0003 |
| 6 | 0.45 | +0.5572 | +0.5505 | −0.0067 |

**A: ρ(est, rank) = +1.000**, matching ρ(true, rank) = +1.000. The ordering is
recovered exactly.

**B: ρ(est, rank) = −0.321** (n=7, p≈.48, and *negative*) with the truth
constant. **No gradient is manufactured.**

B was then re-run with the truth set to exactly zero (§82.5). Same verdict on
the gradient, and a second finding it had been hiding:

| ROI | nc | true Uₘ−Uₚ | est |
|---|---|---|---|
| 0 | 0.15 | 0.0000 | **+0.0319** |
| 1 | 0.20 | 0.0000 | +0.0198 |
| 2 | 0.25 | 0.0000 | −0.0003 |
| 3 | 0.30 | 0.0000 | −0.0012 |
| 4 | 0.35 | 0.0000 | +0.0201 |
| 5 | 0.40 | 0.0000 | −0.0032 |
| 6 | 0.45 | 0.0000 | +0.0198 |

ρ(est, rank) = −0.321 again — no gradient — but the estimates are **not centred
on zero**: mean **+0.0124**, six of seven at or above zero.

### §82.4 The bias that is real, and must be reported

Both conditions show the same thing, and it is the most consequential result
here.

Condition A: the bias is **largest (≈ +0.024) when the true difference is near
zero** and shrinks to ≈0 as the true difference grows. Condition B at a true
zero: mean estimate **+0.0124**, six of seven ROIs at or above zero.

So the estimator is **biased toward magno, by roughly +0.012 to +0.024, exactly
where the true stream difference is near zero**. Where the true difference is
large the estimate is essentially unbiased (roi5-6 in A: −0.0003, −0.0067).

This does **not** threaten the gradient. B shows the bias does not vary with
noise ceiling in a way that produces one, and A recovers the ordering exactly.
The gradient is a *large* effect and lives in the regime where the estimator is
accurate.

It does threaten the near-zero claims, and that has to be said plainly:

* **§2.3's VO null is the exposed one.** It reports Uₘ−Uₚ = **+0.0045** and
  bounds it by equivalence to **±0.0195**. A null-condition bias of **+0.0124**
  toward magno is larger than the effect itself and covers most of the bound.
  The honest reading is that VO's small positive value is **consistent with
  being wholly an artefact of the estimator's null bias**, and the equivalence
  statement should be read as bounding the *observed* quantity rather than a
  bias-free one.
* The same caution applies to any other cell whose Uₘ−Uₚ is within ~0.02 of
  zero.

This is a limitation to disclose in §2.3 and §4.5, not a reason to withdraw
anything: the VO null was already reported as a null, and a bias toward magno
makes a *null* result more conservative, not less. What it forbids is reading
VO's +0.0045 as a small real magno preference.

**Caveat on the magnitude.** These are pilot numbers: 60 components, 20 voxels
per ROI, one subject. The bias magnitude may change at the production 1000
components. It must be re-measured there before any of the above is written
into the draft, and the direction matters more than the exact size until then.

A second oddity to check at production scale: the null estimates cluster near
two values (≈ +0.020 and ≈ 0.000) rather than scattering continuously, which
suggests the alpha grid or the random gamma search is discretising the result.
If that persists it is worth reporting, since it means neighbouring ROIs can
differ by 0.02 for purely numerical reasons.

### §82.5 A design flaw the pilot exposed, and the fix

The pilot's condition B had a true difference of **+0.197, not 0**. Setting
a = b does not give a symmetric truth, because the shared source is a *magno*
canonical variate that parvo predicts only at σ_s² = 0.66; the remaining
(1 − σ_s²) of it is genuinely magno-unique. So B tested "does a constant stay
constant" rather than the stronger "does zero stay zero".

Confirmed by deriving the offset by hand from the σ formula: predicted +0.1974
against the code's +0.1971. That agreement is itself the check that the truth
formula is correct.

The production run solves for b so the true difference is exactly zero,

    b² = [a²(1 − σ_m²) + c²(1 − σ_s²)] / (1 − σ_p²)

and asserts |true difference| < 1e-9 before fitting — the null must actually be
the null.

### §82.6 Identifiability, which is a separate result

The canonical spectrum is itself a measured property worth reporting: at 60
components the smallest canonical correlation is **0.0029** and the largest
**0.813**.

The smallest being ≈0 means **unique variance is identifiable in these feature
spaces** — there really exist directions one band predicts and the other cannot.
Nobody had checked this, and it is a precondition for the whole of §2.2 rather
than a detail: had σ_min been near 1, the decomposition would have been ill-posed
before any noise entered. The largest being 0.813 rather than ~1 is the other
half — even the most shared direction is only R² = 0.66 predictable across bands.

These are two different claims on two different evidence bases, and §4.5 should
keep them apart: *unique variance is identifiable in these feature spaces* is
what the spectrum supports; *the partitioning recovers it correctly* is what
§82.3 supports.

### §82.7 Three design errors, all caught before execution

Recorded because the pattern matters more than the result:

1. **Vacuous ground truth.** The first construction defined magno-unique as
   span(Zm) ∩ span(Zp)^⊥. For two generic 1000-dim subspaces of a 10,000-dim
   space that intersection is {0}. Had it run it would have produced a confident
   vacuous pass — zero true unique variance everywhere, with the estimator
   looking perfect or catastrophic depending on sign convention and no way to
   tell which from the output.
2. **A tolerance that could not fail.** The orthogonality assertion used
   `< 1e-3`. But S_m and S_p are the *matched* last canonical pair, so their
   correlation is σ_min by construction, not zero. Replaced with
   `|corr − σ_min| < 1e-6`.
3. **σ was not the correlation.** σ from the SVD is an *uncentred* inner
   product while `np.corrcoef` centres, and PCA centres on the train mean, so
   the stacked design carried a residual offset. They disagreed in the fourth
   decimal — small enough to pass any "is it small" check, and large enough to
   bias every truth value the formula produces while the analysis still looked
   like it worked. Fixed by centring before the QR.

The general rule, which §79/§80/§81 all instantiate too: **an assertion that a
quantity is *near* an expected value can pass for the wrong reason; an assertion
that it *equals* a separately-derived value cannot.** Prefer the latter.

---

## §83 THE §82.4 BIAS DOES NOT REACH VSB — and a §2.2(a) number I cannot
## reproduce from the stored results

### §83.1 The question

§82.4 found that `variance_partition` is biased toward magno by ~+0.012 where
the true stream difference is near zero. §2.2(a) reads a sign off a near-zero
value — "−0.033 ... if anything, marginally *more* texture- than
shape-organised" — and §4.5 describes the VSB index as coming from stream-wise
variance partitioning. So: does the bias propagate?

### §83.2 No, and the reason is structural

**VSB never touches `variance_partition`.** `run_encoding.py:409-414` computes
it per band, from that band's own fitted estimator's predictions:

    VSB = (eta2_shape - eta2_texture) / (eta2_shape + eta2_texture)

over `pred_extra_{magno,parvo,joint}`. `variance_partition` is called
separately at line 419. The two share the fitted models but no arithmetic.

The §82.4 bias arises specifically from the **nested** subtraction
`r2_joint - r2_parvo`: ridge fitting noise puts the joint model's test R²
systematically above a nested band's, so the difference has a positive floor at
a true zero. An η² contrast on a *single* model's predictions contains no
nested subtraction and cannot inherit it.

This also disposes of the sign-convention question rather than answering it:
which way the bias would push VSB does not arise, because it does not reach VSB.

### §83.3 VSB's own near-zero pathology — checked, not biting

VSB is a ratio and its docstring flags it as "unstable where the denominator is
small" (§21.4). That is a *different* mechanism and it would genuinely threaten
a −0.033 sign read. Measured over NC>0.1 voxels, 8 subjects:

| band | arm | median denominator | min |
|---|---|---|---|
| joint | diet | 0.421 | 0.406 |
| joint | no diet | 0.478 | 0.443 |
| magno | diet | 0.406 | 0.393 |
| magno | no diet | 0.476 | 0.446 |
| parvo | diet | 0.405 | 0.380 |
| parvo | no diet | 0.364 | 0.344 |

Healthy everywhere. Near-zero VSB in this dataset is not a small-denominator
artefact.

### §83.4 OPEN — the −0.033 cell does not reproduce for me

Attempting to check the cell itself, median VSB over NC>0.1 voxels, seed 0,
n=8 subjects:

| band | diet | no diet |
|---|---|---|
| joint | −0.400 | −0.672 |
| magno | **+0.791** | −0.644 |
| parvo | −0.739 | −0.748 |

§2.2(a)'s quantity is the stream dissociation VSB_magno − VSB_parvo (§2934,
§3049, §6579). From the medians above that is **+1.53** for diet and **+0.104**
for no diet, against the draft's **+0.616** and **−0.033**.

The diet value +1.53 does match the **+1.534** printed in the other column of
the §2934 table, which suggests those two columns are different statistics and
that I have reproduced one of them — not the one §2.2(a) quotes.

**This is not a claim that the draft is wrong.** The likeliest explanation is
that I used the wrong mask, or the wrong member of the raw/normalised pair. But
it is flagged rather than dropped for one reason: **§73 is on record as having
found "the mislabel, the raw/_raw trap" in precisely this measure.** A number
that will not reproduce from the stored results, in a measure with a documented
mislabelling history, is worth pinning down. Resolving it needs the exact mask
and variant §2.2(a) quotes, which belongs to the draft owner.

Note the sign question is unaffected either way: on my numbers the no-diet
dissociation is **+0.104**, positive, where the draft reads −0.033 as negative.
If my aggregation were the right one the clause would be wrong in sign, not
merely fragile — which is the reason not to leave this unresolved.

### §83.5 RESOLVED — `raw` is the un-normalised numerator, and the draft was right

The draft owner's hypothesis, tested directly rather than inferred: `norm` is
the stored ratio and **`raw` is its numerator alone**, η²_shape − η²_texture.
The stream dissociation is then
median(η²_s − η²_t | magno) − median(η²_s − η²_t | parvo).

Both η² fields are stored per band, so this is a direct check. Median over
NC>0.1 voxels, n=8 subjects:

| arm | seed 0 | seed 1 | seed 2 | draft quotes |
|---|---|---|---|---|
| diet, RAW | +0.6062 | +0.6475 | +0.6180 | 0.616 / 0.658 / 0.632 |
| no diet, RAW | **−0.0283** | **−0.0478** | **−0.0505** | −0.033 / −0.039 / −0.053 |
| diet, NORM | +1.5353 | +1.5724 | +1.5590 | (+1.534 in §2934's other column) |
| no diet, NORM | +0.1123 | +0.0884 | +0.0799 | — |

**The sign alarm in §83.4 is withdrawn.** All three no-diet seeds are negative
on the raw statistic, matching the draft. My +0.104 was the *normalised* ratio —
the wrong statistic for that cell. The draft's −0.033 reproduces in sign and to
within ~0.01 in magnitude, the residual being a mask detail (the draft drops ROI
cells under 20 voxels; I did not).

This also confirms the §2934 table's two columns: **+1.534 is `norm` and +0.616
is `raw`**, exactly as the owner read them.

I raised this as a possible sign error and it was not one. The escalation was
still correct — a sign discrepancy in a clause whose content is the sign cannot
be left on "probably my mask" — but the outcome is that the number was right and
my comparison was wrong, and that is what the record should say.

### §83.6 The §73 trap is LIVE, and there are three senses of `raw`

Confirmed in `tmp_analysis/cc_ladder.py:64`. The key literally named `"raw"` is

    "raw": step(v, ...)   where  v = z["cc_joint_vsb"]

i.e. the **hV4 − V1 profile step on the joint band**, unresidualised. The stream
dissociation is a *different* key, `"mp"`, and it uses the **normalised**
`cc_magno_vsb` / `cc_parvo_vsb`.

So the project currently carries three quantities across two uses of the word:

| where | `raw` means | value (diet, seed 0) |
|---|---|---|
| §2.2(a) / §2934 | un-normalised dissociation numerator | +0.616 |
| `cc_ladder.py` `"raw"` | unresidualised hV4−V1 step, joint band | (a step, not a dissociation) |
| `cc_ladder.py` `"mp"` | normalised dissociation | +1.535 |

Anyone reading `"raw"` out of `cc_ladder.py` and pairing it with the §2.2(a)
sense gets two different quantities under one label. This is precisely what §73
documented and it is still in the tree. **Rename `cc_ladder.py`'s `"raw"` to
`"step_unresid"`** — it is a tmp_analysis script with no downstream consumer of
the string, so the rename is cheap and removes the collision at its source.

---

## §84 CONDITION B RAN THE NOISE-CEILING RAMP BACKWARDS — the artefact is a
## 1/nc normalisation effect, and its sign was therefore inverted

### §84.1 The error, which is mine

`vp_recovery.py` hardcoded

    roi_nc = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

**ascending** with hierarchical rank, under a comment claiming the values were
"taken from the real fitted results" and a docstring claiming ROIs "differ only
in noise ceiling and voxel count, drawn from the real per-ROI values". Both
statements were false: the values are invented, and they run the wrong way.

Measured from `results/encoding/dkl_bcs_seed0_trained_subj*.npz`, NC>0.1,
`roi_prf-visualrois`, 8 subjects, median per ROI:

| V1v | V1d | V2v | V2d | V3v | V3d | hV4 |
|---|---|---|---|---|---|---|
| 0.421 | 0.429 | 0.390 | 0.343 | 0.350 | 0.314 | 0.313 |

**ρ(NC, hierarchical rank) = −0.929, p = .0025.** The real noise ceiling
**declines** V1v→hV4. The simulated ramp climbs (ρ = +1.000) and spans a range
~2.5× too wide (0.15–0.45 against 0.313–0.429).

Caught by the draft owner, who measured the real profile rather than accepting
the simulation's premise. Independently reproduced here at ρ = −0.929 against
their −0.893 (mask differences).

### §84.2 The mechanism — it is a normalisation artefact

`variance_partition` divides the unique components by the noise ceiling, and
`vp_recovery.py` reports the normalised values. From this session's own
condition-B output, 8 subjects:

| roi | nc | est diff (normalised) | raw = est × nc |
|---|---|---|---|
| 0 | 0.15 | 0.0617 | 0.0093 |
| 1 | 0.20 | 0.0568 | 0.0114 |
| 2 | 0.25 | 0.0610 | 0.0152 |
| 3 | 0.30 | 0.0426 | 0.0128 |
| 4 | 0.35 | 0.0333 | 0.0117 |
| 5 | 0.40 | 0.0380 | 0.0152 |
| 6 | 0.45 | 0.0375 | 0.0169 |

* normalised bias: ρ vs rank **−0.857**, spread 0.0284,
  **corr with 1/nc = +0.854**
* raw bias: spread only 0.0076, growing mildly with nc (≈ nc^0.5)

So the raw bias is small and slowly increasing, and the **apparent gradient is
produced by dividing it by an ascending nc**. The artefact's sign is set by the
direction of the nc ramp, not by anything about the streams.

### §84.3 What follows, quantitatively

Under the **real descending** profile the same mechanism runs the other way.
With raw ≈ nc^0.5, the normalised bias goes as nc^(−0.5), so it **rises** with
hierarchical rank — i.e. the artefact would be **positive** where §2.2's
observed gradient is **negative**. The artefact would *oppose* the observed
gradient rather than produce it.

Magnitude should also fall sharply. The real nc range is a factor 0.429/0.313 =
**1.37**, against the simulated 0.45/0.15 = **3.0**. At nc^(−0.5) that is a
~16% variation in the normalised bias across the real hierarchy, against the
~65% the backwards ramp produced.

This is a prediction, not a result. The draft owner has submitted array
26440398 (`vp_recovery_realnc.py`, `--roi-nc` defaulting to the measured
profile, both diet and no-diet arms, 8 subjects, `main`, no GPU) to test it.

### §84.4 What §2.2 may claim, either way

Independently of the sign, §84.2 establishes that **the instrument has a
non-zero artefact floor whenever noise ceiling varies across ROIs**, which it
does. So a statement of the form "the profile declines steeply with the diet and
is nearly flat without it" is implicitly comparing against **zero**, when the
correct baseline is whatever condition B yields under the real profile.

That baseline belongs in the paper as a floor the diet arm has to clear. This
holds whichever way the corrected sign lands, and it is the durable consequence
of this exercise regardless of the re-run.

### §84.5 Why it was diagnosable at all

The bug is one line in an otherwise correct experiment, and it survived three
rounds of assertion-checking because every assertion was about the *ground
truth* — the orthogonality of the sources, σ equalling the measured
correlation, the solved-to-zero null — and none was about whether the
*simulated design* matched the real one.

The general lesson, extending §82.7: **verifying that a simulation is internally
correct is not verifying that it simulates the right thing.** Every parameter
asserted to come from data should be read from that data in code, not typed in
as a literal. Had `roi_nc` been loaded from `results/encoding/` — as its own
comment claimed it was — the error could not have occurred.

Concretely: a literal that a comment describes as "measured" is a defect
whatever its value.

---

## §85 The corrected recovery run: the artefact opposes §2.2, but "the floor"
## is two different numbers and the paper must pick the right units

> **READ §86 FIRST — four claims below are superseded.** §85 was written
> against a median over the ROI profile; §2.2's statistic is an hV4 − V1
> contrast (§86), and in that statistic the floor is **+0.051**, not the
> +0.0256 or +0.1244 argued over here. Specifically:
>
> | §85 claim | status |
> |---|---|
> | §85.2's withdrawal of "no gradient" | **reversed** — it was correct for the difference statistic (§86.2) |
> | §85.5's "prediction FAILED by 60×" | **wrong** — scored against the wrong statistic; it passes (§86.1) |
> | §85.4's `est_diff × nc` = +0.0094 as the "raw" figure | **conceded wrong** — the paper's raw is also ÷nc (§86.3) |
> | "−0.862 is a floor, not a ceiling" (below) | **withdrawn** — condition A biases the two statistics in opposite directions (§86.7) |
>
> What survives: the headline sign argument, §85.6–85.9, and the naming
> problem, which §86.3 sharpens rather than resolves. Quote §86 and §90, not
> the numbers in this section.

Job 26440398 ran `vp_recovery_realnc.py` with the measured, descending
noise-ceiling profile. §84.3 recorded a quantitative prediction before the run
existed, so it could fail. Condition B's null is exact — max |true_diff| =
1.11e-16 across all 56 cells — so every non-zero estimate is pure instrument.

**The headline, which b5faaee and this section agree on:** the artefact is
*positive* and *rising* with hierarchical rank, while §2.2's effect is negative
(−0.862) and falling (ρ = −0.964). The artefact has the opposite sign on both
the contrast and the gradient. It cannot manufacture §2.2's result; it opposes
it, so the reported contrast **understates** the truth. Condition A returns
ρ(est, rank) = +1.000 in 8/8 subjects with magnitudes compressed toward zero
(`est = 0.713 × true`, 1.40× attenuation), so −0.862 is a floor, not a ceiling.

### §85.1 Two statistics, both called "the floor", differing 5×

b5faaee reports "+0.126 normalised / +0.025 raw … ρ = +0.68". Reproducing every
candidate definition against the same eight files identifies all three exactly:

| printed | definition that reproduces it | value |
|---|---|---|
| "+0.025 raw" | the **stored** `est_diff` = `est_unique_m − est_unique_p` | mean **+0.0256** |
| "+0.126 normalised" | `(um − up)/(um + up)`, a VSB-style ratio index | mean **+0.1244**, median +0.1250 |
| "ρ = +0.68" | **median of per-subject** Spearman ρ on the ratio index | **+0.679**, 8/8 positive |

Those numbers are correct. The problem is not arithmetic, it is units.

### §85.2 The gradient exists in one parameterisation and not the other

| statistic | mean | ROI-mean ρ vs rank | p |
|---|---|---|---|
| stored `est_diff` (already ÷nc) | +0.0256 | +0.536 | .215 |
| `est_diff × nc` (nc division undone) | +0.0094 | −0.893 | .007 |
| ratio index `(um−up)/(um+up)` | +0.1244 | **+0.964** | **.0005** |

An earlier draft of this section claimed condition B "manufactures essentially
no gradient (+0.0008)". **That is true only of the stored `est_diff`.** The
ratio index manufactures a significant rising gradient, and `est_diff × nc`
manufactures a significant *falling* one. Three parameterisations of one
simulation give ρ = +0.96, +0.54, and −0.89. Reporting any of them as "the"
artefact gradient without naming the statistic is not defensible.

The ratio index cancels nc exactly (both `um` and `up` carry the same 1/nc), so
its rise is *not* a normalisation effect: total unique variance falls with rank
faster than the difference does. The §84 diagnosis — that the ascending ramp's
−0.351 was a 1/nc artefact — remains correct for that quantity, but it does not
explain the ratio index's gradient, which has a different cause.

### §85.3 Which floor the paper should disclose

**The floor must be in the same units as the effect it bounds.** §2.2 reports
−0.862, a magnitude only a bounded ratio index can take. If §2.2's contrast is
the ratio index, the correct floor is **+0.124** — 14.4% of |−0.862|, opposite
in sign — with a rising gradient (median per-subject ρ = +0.679, 8/8 subjects,
Wilcoxon p = .0078, which is the exact n=8 floor per §80).

**Action for §E2.1:** state which statistic §2.2 reports, then quote the floor
computed in that statistic and no other. Quoting +0.126 beside a −0.862
computed differently, or +0.025 beside it, would be a units error either way.
This is a question for the draft owner, not a number I can settle from the
recovery run alone.

### §85.4 "raw" now has three incompatible meanings in this project

1. §73 / `cc_ladder.py`: the VSB **numerator** η²_shape − η²_texture, versus the
   ratio. (Caught in §83.5; the file was renamed.)
2. §84–85 here: `est_diff × nc`, i.e. the nc division **undone**.
3. b5faaee / §E2.1: the **stored** `est_diff`, which is already divided by nc.

Sense 3 is not raw with respect to the noise ceiling. A referee reading
"+0.025 raw" will reasonably take it as an un-normalised variance difference;
that quantity is **+0.0094**. This is the same trap as §73, third occurrence.
Recommendation: retire the word. Write `est_diff (÷nc)`, `est_diff × nc`, and
`ratio index` explicitly wherever these appear.

### §85.5 The prediction scored: sign right, magnitude 60× too large

> **SUPERSEDED BY §86.1.** The verdict below is wrong. §84.3's prediction was
> registered as a contrast and is scored here against a profile median. In
> §2.2's actual statistic the artefact is +0.0511 against a predicted +0.05 —
> the prediction PASSES on both sign and magnitude.

Against the stored `est_diff`, which is what §84.3 predicted:

| claim | predicted | observed | verdict |
|---|---|---|---|
| sign of the manufactured contrast | positive | +0.0008 | PASS |
| relative variation across hierarchy | ~16% | 7.4% | PASS (within 2×) |
| absolute contrast | order +0.05 | +0.0008 | **FAIL, 60× too large** |

I scaled from the backwards ramp's **−0.351**, a 60-component pilot figure.
Re-running the *same* backwards ascending ramp at 1000 components gives a
contrast of **−0.0242** — 15× smaller. The percentage reasoning was sound; the
baseline it multiplied had never been checked at production scale.

**The −0.351 must not enter the paper.** It is a low-rank pilot artefact that
does not survive at production dimensionality. b5faaee already treats the
−0.351 alarm as closed, which is right, but for the additional reason that the
figure itself does not reproduce.

### §85.6 The 1/nc mechanism is real but range-limited

| run | nc range | raw ~ nc^p | corr(norm, 1/nc) | norm spread |
|---|---|---|---|---|
| ascending 0.15–0.45 (factor 3.0) | 0.150–0.450 | nc^+0.43 | **+0.854** | 0.0284 |
| real, descending (factor 1.29) | 0.320–0.413 | nc^+0.90 | +0.381 | **0.0019** |

Over a 3× ceiling range the normalisation dominates. Over the real 1.29× range
it is negligible. The exponent difference is not a contradiction: 7 points over
a 1.29× span cannot constrain a power law, and the low-nc end where the effect
lives does not occur in the data at all.

The severe shrinkage flagged earlier (first A cell true −0.3037 vs est −0.0453,
~7×) was itself largely an artefact of the wrong ceiling: under the real
profile the same cell reads true −0.3315 vs est −0.2335, a 1.42× compression.

### §85.7 Discretisation: no evidence of quantisation

The pilot's estimates appeared to cluster near ≈+0.020 and ≈0.000, suggesting
the alpha grid or the 20-sample random gamma search discretised the output.
It does not: **112 of 112** estimates are distinct at 6 dp in both runs, with
minimum gaps of 2.8e-05 (ascending) and 2.3e-06 (real). The apparent clustering
was small-n coincidence. Worry withdrawn.

### §85.8 Identifiability at 1000 components

| run | σ_shared | σ_min | σ_mean | σ_median |
|---|---|---|---|---|
| ascending | 0.9258 | 1.61e-04 | 0.3065 | 0.2933 |
| real | 0.9539 | 3.00e-04 | 0.3243 | 0.3068 |

(pilot, n_comp=60: σ_min 0.0029, σ_max 0.813.)

At production dimensionality the bands hold a **near-degenerate leading shared
direction** (σ ≈ 0.93–0.95, i.e. 86–91% shared variance) *and* directions that
are essentially exactly separable (σ ≈ 1e-4). The typical pair shares only
σ² ≈ 9%. Both facts belong in §4.5: the bands are neither redundant nor
independent, which is why a subtraction-based partition needed validating
rather than assuming.

### §85.9 The diet arm was lost to an output-name collision

All 16 tasks of 26440398 reported COMPLETED but only 8 files exist. The submit
script maps tasks 0–7 to `dkl_bcs_seed0_trained` and 8–15 to
`dkl_base_seed0_trained`, while the writer named outputs by subject alone — so
the no-diet half (finishing 04:27–06:13) overwrote the diet half (finishing
~03:00). Provenance was recoverable only because each JSON carries its own
`model` field; the filenames did not.

The writer is now model-tagged. The lost arm has been re-submitted as
**26442351** (`--array=0-7%6`, `main`, no `--gres`), which cannot collide with
the surviving files. **Every number in §85 is the no-diet arm**; the diet arm is
expected to differ, because the artefact depends on feature geometry and the two
models differ there (§2.7).

### §85.10 The lesson, distinct from §84's

§84's lesson was that a simulation can be internally correct while simulating
the wrong thing. §85 has two more, both about *reporting* rather than computing:

1. A prediction scaled from an earlier number inherits that number's errors.
   State the baseline as a separately falsifiable quantity, or a passing sign
   and a failing magnitude will look like one result instead of two.
2. Two sessions computed this floor correctly and got +0.0256 and +0.1244. The
   disagreement was never arithmetic — it was that neither number carried its
   units. This is the third time the same ambiguity has cost us (§73, §83.5,
   here). A statistic reported without its denominator is not a result.

---

## §86 The units question resolved: §2.2 is a contrast, the floor is +0.051,
## and §2.2's published values do not exactly reproduce

6f336f5 answered §85.3's question with code rather than a guess:
`visualrois.py:72-75` computes `paired(spi[:, hV4], v1)` with
`v1 = mean(V1v, V1d)`, so §2.2 reports an **hV4 − V1 contrast**, not a value at
hV4 and not a median over the ROI profile. Recomputed here independently from
the same eight condition-B files:

| statistic, in §2.2's contrast form | floor | subjects | p |
|---|---|---|---|
| ratio index `(uₘ−uₚ)/(uₘ+uₚ)` | **+0.0511** | 8/8 positive | **.0078** |
| difference `uₘ−uₚ` | **+0.0009** | 4/8 | .84 |

Both match 6f336f5 to four decimals. Two consequences for §85, which was
written against the wrong statistic:

### §86.1 My §84.3 prediction did not fail — §85.5's scoring is withdrawn

§84.3 registered "something small and positive, order **+0.05** normalised".
In §2.2's actual statistic the artefact is **+0.0511**. §85.5 scored this as
"FAIL, 60× too large" because it compared against a median over the ROI
profile (+0.0008), a statistic the paper does not report. **The scoring table
in §85.5 is wrong and is superseded by this section.**

| claim | predicted | observed (contrast) | verdict |
|---|---|---|---|
| sign | positive | +0.0511 | PASS |
| magnitude | order +0.05 | **+0.0511** | **PASS** |

The §85.5 lesson — that a prediction scaled from an unchecked baseline
inherits its errors — survives only as it applies to the **−0.351**, which
still does not reproduce (−0.0242 at 1000 components). What actually went wrong
in §85 was different and worse: I scored a registered prediction against a
statistic other than the one it was registered in.

### §86.2 My withdrawn "no gradient" claim was correct, for one statistic

§85.2 withdrew "condition B manufactures essentially no gradient (+0.0008)".
On the **difference** statistic the contrast is **+0.0009**, 4/8, p = .84 —
indistinguishable from zero. The withdrawal was itself an error. The accurate
statement names both:

> The ratio index manufactures a small positive artefact (+0.051, 8/8,
> p = .0078); the difference statistic manufactures nothing measurable
> (+0.0009, p = .84). Neither is negative, so neither can produce §2.2's
> negative contrast.

### §86.3 The nc-units objection is conceded, and the naming problem is worse

`variance_partition` returns **both** forms (`banded_ridge.py:225-237`):
`unique_magno_raw = r2_joint − r2_parvo`, and `unique_magno = unique_m / nc`.
`visualrois.py:34` reads `z["unique_magno"]` — the **÷nc** form. So the paper's
"raw" (the difference statistic) is also ÷nc, and §85.4's proposed **+0.0094**
(`est_diff × nc`) is in units the paper never reports. Conceded.

But this makes the naming worse, not better: the codebase's own `_raw` suffix
means *not divided by nc*, which is the **opposite** of the paper's "raw",
which means *the difference rather than the ratio* and is divided by nc. That
is a fourth incompatible sense:

1. §73 / `cc_ladder.py` — the VSB **numerator**
2. `variance_partition`'s `*_raw` keys — **not ÷nc**
3. §2.2 / §E2.1 — the **difference** statistic, which *is* ÷nc
4. §84–85 (mine, withdrawn) — `est_diff × nc`

Sense 2 and sense 3 are both live in code that the paper depends on, and they
disagree. E2.1 defining both constructions in a table is the right fix; the
codebase keys should be renamed too, or a reader following `_raw` from code to
paper lands on the wrong quantity.

### §86.4 NEW: §2.2's four published values do not exactly reproduce

> **Conclusion withdrawn in §87.7; observation stands.** The "one-directional
> across four numbers, therefore not rounding" inference does not survive the
> wider set: `untrained_seed0` recomputes at −0.5441 against a published
> −0.471, i.e. more negative the *other* way. It is twelve values, not four,
> and per-number provenance drift is the right reading. §88 then dated the
> no-diet pair to a pre-quarantine fit, which explains those two and only
> those two.

Recomputing §2.2's contrast from `results/encoding/` under both possible
ratio constructions, all n=8:

| arm | construction | recomputed | published | gap |
|---|---|---|---|---|
| diet | ratio, median-of-ratios | −0.8335 | −0.862 | **+0.0285** |
| diet | ratio, ratio-of-medians | −0.8165 | −0.862 | +0.0455 |
| diet | difference of medians | −0.1650 | −0.176 | **+0.0110** |
| no diet | ratio, median-of-ratios | −0.0785 | −0.101 | **+0.0225** |
| no diet | ratio, ratio-of-medians | −0.0693 | −0.101 | +0.0317 |
| no diet | difference of medians | −0.0258 | −0.034 | **+0.0082** |

Median-of-ratios is closest in both arms, which **confirms the contrast
reading** — no other construction gets the right order of magnitude on all
four. But "all four reproduce" overstates it. **All four gaps carry the same
sign**: every published value is more negative than any recomputation, by
0.008–0.046. A one-directional offset across four independent numbers is not
rounding; it points to a selection or provenance difference — `NC_MIN`,
`MIN_VOX`, the `(uₘ+uₚ) > 1e-3` mask, a checkpoint/epoch, or a subject subset.

This does not threaten the sign or the ordering, and the floor-to-effect ratio
is nearly unchanged (6.1% of −0.834 vs 5.9% of −0.862). But E2.1 should not
quote a +0.051 floor against a −0.862 that the current code returns as −0.834
without stating which is authoritative.

### §86.5 NEW: the floor is a different estimator from the effect it bounds

§2.2 takes **median-of-ratios** over voxels. The recovery JSONs store only
per-ROI medians, so the +0.0511 floor is necessarily **ratio-of-medians**. On
real data the two differ at the contrast level by **0.017 (diet)** and
**0.009 (no diet)** — the same order as a third of the floor itself.

So +0.051 carries an estimator-mismatch uncertainty it does not currently
disclose. Fixing it means re-running condition B recording the per-voxel ratio
median, not just the medians of `uₘ` and `uₚ`. Until then E2.1 should give the
floor as approximate and say why.

### §86.6 NEW: the floor is negligible for the diet arm and not for no-diet

| arm | contrast (recomputed) | floor | floor as % of effect |
|---|---|---|---|
| diet | −0.8335 | +0.0511 | **6.1%** |
| no diet | −0.0785 | +0.0511 | **65%** |

Against the published values, 5.9% and 51%. The floor is positive and the
effects negative, so both arms are **understated**; correcting both by +0.051
gives diet ≈ −0.885, no-diet ≈ −0.130, and shrinks the between-arm ratio from
**8.5× to 6.8×** — a ~20% reduction that leaves the diet/no-diet separation
large but should not be silently absorbed.

The consequence for wording: any phrase like "nearly flat without the diet" is
partly describing the artefact, since the artefact is 65% of the no-diet
contrast and points the other way. The honest form is that the no-diet arm
shows a shallow but non-zero negative gradient whose magnitude is comparable to
the instrument's floor.

**Caveat:** this floor was measured on **no-diet feature geometry only** — the
diet arm was lost to the filename collision. The submit script's own rationale
says the artefact depends on feature geometry, so applying +0.051 to the diet
arm assumes what that script denies. The diet-arm floor comes from 26442320.

### §86.7 Condition A is not magnitude-faithful, and the two figures disagree

| statistic | est/true | direction |
|---|---|---|
| ratio index | **1.338** (6f336f5 reports 1.255) | inflates |
| difference | **0.713** (agrees) | compresses 1.403× |

Direction agrees on both, so 6f336f5's withdrawal of "−0.862 is a floor rather
than a ceiling" is correct and §2.2 should read as **sign-and-ordering only**.
§85's "1.40× compression, so −0.862 is a floor" is withdrawn with it; that
figure was the difference statistic's and does not carry to the ratio.

But 1.338 vs 1.255 is a 6% disagreement on a number neither of us stated an
estimator for — the same failure that produced this whole exchange. Mine is an
OLS slope of est on true pooled over 56 cells (r = +0.987).

### §86.8 Job hygiene: the duplicate array was correctly cancelled

26442351 (mine) and 26442320 (theirs) both had `WorkDir` in `mpk-dev` and, with
the writer now model-tagged, both wrote the identical eight paths — 12 tasks
racing on 8 files, the same failure that lost the diet arm originally.
26442320 was submitted first and survives (6 running / 2 queued at check).
Cancelling the second was right; I should have checked for an in-flight
equivalent before submitting, since the owner had just diagnosed the collision
and re-running was the obvious next step for either of us.

---

## §87 subj01's provenance is settled from Slurm, and the file is legitimate:
## the "arm-asymmetric outlier" is a ratio artefact, not a bad fit

8bfa2c5 left the subj01 question as an open blocker because the file carries no
provenance stamp. Slurm settles it without a re-fit.

### §87.1 The file is the repair, proven three ways

The quarantine README names the repair as tasks 16-23 of job 26399760. That
job's records and logs both survive:

```
slurm/logs/enccc_26399760_16.out, line 1:
  task=16 model=dkl_base_seed0_trained subject=1
          frontend='--no-lum-blur --no-lum-cs'
  [cache] no provenance recorded in this cache file (missing 7 keys)
          -- recomputing features rather than trusting it
  [stage +13782.3s] saved -> .../results/encoding_cc/
                              dkl_base_seed0_trained_subj01.npz
```

- **The front end is correct.** `--no-lum-blur --no-lum-cs` is exactly the pair
  the defective job `26386734_0` omitted, and the run refused to trust the
  cached features and recomputed them.
- **The timestamps match to the microsecond.**
  `results/encoding/dkl_base_seed0_trained_subj01.npz` has mtime
  `2026-09-01 05:27:47.293126`; task 16 ended `05:27:50`. Its siblings in
  `results/encoding/` date from 2026-08-29 and are 1.5-1.7 MB against
  subj01's 2.85 MB, i.e. subj01 is the only file there from the cc round —
  consistent with a `cp -p` of the repair, as the README instructed.
- **The two copies are byte-identical** (`cmp` clean), and the defective job
  ran 2026-08-30 02:00-06:06, which matches neither copy.

### §87.2 The missing stamp is not evidence

`results/encoding_cc/dkl_base_seed0_trained_subj02.npz` — an uncontested file
from the same repair round — has the same 28 keys and no provenance key either.
**No file the writer produces carries a stamp.** So "unstamped" cannot
discriminate the repair from the original; it is a property of the writer. The
README's promise of a "provenance-stamped version" was never fulfilled by the
code, which is a real gap, but not a reason to distrust this file.

### §87.3 The outlier survives the repair — so it is not a front-end artefact

subj01 is still extreme: no-diet ratio contrast **−0.2487** against a sibling
median of −0.0535 (**4.65×**, sibling range −0.0845..−0.0266). Since §87.1
establishes the front end was correct, **the outlier cannot be attributed to
the wrong front end.** The README's original 6× shrank to 4.65×, but did not go
away.

### §87.4 The arm asymmetry is an artefact of dividing by different baselines

8bfa2c5's inference was: "A subject-level peculiarity would show in both arms;
a wrong-front-end no-diet fit shows in one." The ratios do look arm-specific
(4.65× no-diet, 1.23× diet). But the arms' baselines differ ~15-fold, so a
ratio against each baseline is not a comparison. The **absolute displacement**
is:

| arm | subj01 | sibling median | ratio | displacement |
|---|---|---|---|---|
| no diet | −0.2487 | −0.0535 | 4.65× | **−0.1952** |
| diet | −0.9970 | −0.8077 | 1.23× | **−0.1894** |

The same shift, to within 0.006. **The asymmetry is entirely a ratio
artefact.** This is the fourth appearance of one failure mode in this thread
(§73, §83.5, §86, here): a ratio and its numerator answering different
questions while sharing a name.

### §87.5 The displacement is in every fit, including ones the bad job never touched

| model | subj01 | sibling median | ratio | displacement |
|---|---|---|---|---|
| dkl_base_seed0 | −0.2487 | −0.0535 | 4.65× | −0.1952 |
| dkl_base_seed1 | −0.1798 | −0.0892 | 2.02× | −0.0907 |
| dkl_base_seed2 | −0.1746 | −0.0305 | 5.73× | −0.1441 |
| dkl_bcs_seed0 | −0.9970 | −0.8077 | 1.23× | −0.1894 |
| dkl_bcs_seed1 | −0.9237 | −0.7185 | 1.29× | −0.2053 |
| dkl_bcs_seed2 | −0.8865 | −0.7711 | 1.15× | −0.1154 |

Mean displacement: no-diet **−0.143**, diet **−0.170**, differing by 0.027 —
while the *ratios* span 1.15× to 5.73× on the same underlying shift.

`26386734_0` touched only seed 0, subject 1, of the no-diet arm. Seeds 1 and 2
of that arm were never re-fitted, and subj01 is displaced by −0.091 and −0.144
in them. **subj01 is a subject-level effect** — a genuinely steeper hV4−V1
gradient in this subject — present in six independent fits.

### §87.6 Recommendation: keep subj01, report n=7 as influence, not exclusion

The provenance question is closed and the file is legitimate, so n=8 is the
correct primary analysis and subj01 must not be dropped as contaminated.

The n=7 sensitivity still belongs in the paper, but as an **influence check**:
subj01 alone moves the no-diet contrast from −0.0542 to −0.0785, so ~31% of the
no-diet effect rests on one subject. That is worth disclosing on its own terms.
It does not weaken §2.2's reframed claim — at n=8 the no-diet contrast is ~1.5×
the artefact floor and at n=7 ~1.1×, so "a large gradient with the diet and
none resolvable without it" holds under either, and now holds *without* an
unresolved provenance caveat attached.

### §87.7 §86.4 corrected: twelve values, and the offset is not one-directional

8bfa2c5's two corrections are accepted, and the second is verified here:
`untrained_seed0` recomputes at **−0.5441** against a published −0.471 — more
negative in the *recomputation*, the opposite direction to all six §2.2 values.
So my §86.4 inference ("one-directional across four numbers is not rounding")
does not survive the wider set: a single selection difference cannot produce
gaps in both directions, and per-number provenance drift is the better reading.
§86.4's *observation* stands for the six §2.2 values; its *conclusion* is
withdrawn.

Accepted without re-deriving: the ~200-configuration search, the twelve-value
table from `encoding_cc`, and the decision to stop searching rather than fit
the convention to the numbers. That is the right call under the standing rule
that the fix for messy results is reframing the claim, not more compute.

One gap to note: `untrained_seed0` exists only in `results/encoding/`, not in
`results/encoding_cc/`. If the verifier locks `encoding_cc` as the sole
declared source, the untrained control cannot be recomputed from it.

### §87.8 The `_raw` rename is done

`variance_partition` now returns `unique_magno_pre_nc`, `unique_parvo_pre_nc`,
`shared_pre_nc` (`banded_ridge.py:225-237`). Checked before changing: only one
reader (`tmp_analysis/group_check.py:7`) and nothing globs or pattern-matches
`_raw`.

`run_encoding.py:446` splats the dict straight into `np.savez`, so the rename
propagates to all future fits while the ~188 stored `.npz` files keep the old
keys. `group_check.py` therefore reads the new name with a fallback to the old,
commented as such. Verified after the change: `group_check.py` still runs on
the stored files and independently reports
`median (unique_magno/pre_nc)*nc = 1.0`, confirming §86.3's claim that
`unique_*` is the ÷nc form; and a synthetic check confirms
`unique_magno == unique_magno_pre_nc / nc` with the old key absent.

---

## §88 The published table is dated: the no-diet numbers were computed before
## the 2026-09-01 quarantine, with the defective subj01 still in the set

b778201 left one thread open — dating the published table — as worth more than
further configuration search. It is datable, and the answer explains the
no-diet gap exactly while confirming that the diet gap has a different cause.

### §88.1 When the numbers were written

`git log -S"-0.862" -- docs/` gives three commits; the earliest introduces the
value:

| commit | date | how |
|---|---|---|
| **c0002a2** | **2026-09-06 14:24** | added as new prose ("declines … by **-0.862** … against **-0.101**") |
| 391e336 | 2026-09-06 16:26 | edited |
| 8d4f22d | 2026-09-06 22:30 | edited |

The value entered the repository as written text, not as generated output.
(The pickaxe also hits three older commits, but those are `results/eeg_rsa/*
group_summary.csv` files that contain −0.862 coincidentally — unrelated.)

### §88.2 Nothing the table depends on changed after it was written

| dependency | last change |
|---|---|
| `tmp_analysis/visualrois.py` | **2026-08-28** (4ef1a3f) |
| `src/analysis/encoding/stats.py` (`load_results`) | **2026-08-29** (f9a465b) |
| diet fits `results/encoding/*bcs_seed0*` | **2026-08-30** |
| `results/encoding_cc/*` | **2026-09-01** |
| table written | 2026-09-06 14:24 |

The only change to `src/analysis/encoding/` after 2026-09-06 is §87's rename
today, which the nc-invariant ratio cannot feel. **Post-hoc code or data drift
is ruled out**: every input was already frozen when the numbers were written.

### §88.3 The committed script cannot have produced the table

`visualrois.py:22` reads

```python
res = load_results("results/encoding", "untrained_seed0")
```

The model is **hardcoded**, and running the script unmodified today prints
`SPI hV4 - V1  mean diff=-0.5441  p=0.0078  0/8 positive` — the untrained
number, not the diet one. So the diet and no-diet table values required an
**uncommitted edit** of that line, run and then reverted. That is why they are
unauditable: the configuration that produced them was never recorded, and this
is a stronger reason than "we cannot find the right settings".

It also explains the untrained row's direction: −0.5441 is what the committed
script emits, against a published −0.471, and that is the one row the script
can still be checked against.

### §88.4 The no-diet numbers date to before the quarantine

Solving for the subj01 value that the published n=8 mean requires, against the
seven siblings' actual values:

| statistic | published | needs subj01 | × sibling median | share of effect |
|---|---|---|---|---|
| ratio | −0.101 | **−0.4289** | **8.0×** | 53% |
| difference | −0.034 | **−0.1275** | **6.0×** | 47% |

The quarantine README describes the defective file, independently and before
any of this, as a **"six-fold outlier"** that **"contributed 57% of the
reported no-diet effect size"** and "was the difference between significant and
not on the raw statistic". The solved difference statistic lands at **exactly
6.0×**, and the shares at 53%/47% against 57%.

**So §2.2's no-diet values were computed with the defective subj01 in the n=8
set — i.e. before 2026-09-01 — and transcribed into the draft on 2026-09-06,
five days after that file had been quarantined for being wrong.** The gap was
never a configuration difference. The data those numbers came from was
deliberately removed, which is why no configuration reproduces them.

Consistency check: the repaired file gives a marginal raw no-diet contrast
(p = .0391, 2/8 positive), matching the README's "difference between
significant and not".

### §88.5 The diet gap is not a subject substitution

Applying the same solve to the diet arm, which never had a defective file:

| statistic | published | needs subj01 | sibling range |
|---|---|---|---|
| ratio | −0.862 | **−1.2252** | −0.9643..−0.6227 |
| difference | −0.176 | −0.2721 | −0.2068..−0.1336 |

−1.2252 lies far outside all 48 observed subject×model cells. It is not
strictly impossible — with correlated bands a voxel can have negative unique
variance while `uₘ+uₚ > 0`, so the ratio is not bounded at ±1 — but nothing
remotely like it occurs anywhere in the data. **The diet gap therefore has a
different cause from the no-diet gap**, which supports b778201's per-number
provenance-drift reading over any single missing configuration.

### §88.6 What this settles for the paper

- The decision to restate the table from `encoding_cc` and stop searching was
  correct, and now has a mechanism rather than a shrug: the no-diet numbers
  were computed from data that was intentionally deleted.
- §2.2's supersession note can say **why** the old numbers do not regenerate,
  which is much stronger than recording that they do not.
- The remaining unexplained gaps are the diet arm (+0.029 ratio, +0.011
  difference) and the untrained row (−0.073, the other direction). Neither is
  attributable to subj01, and both should stay disclosed as drift.
- Standing lesson: `visualrois.py` hardcodes its model on line 22. Any table
  built by editing that line and re-running leaves no record. The fix is to
  take the model as an argument and write the emitted numbers to a
  provenance-stamped file under `results/`, so a published table always has a
  regenerable source. This is the same defect class as §84's hardcoded
  `roi_nc` — a value whose provenance lives only in someone's shell history.

---

## §89 The §88.6 fix landed: `visualrois.py` takes `--model` and stamps its output

The defect §88.3 identified — a published table produced by editing a
hardcoded model string and re-running, leaving no record — is now closed at
the source rather than only described.

### §89.1 What changed

- `--model` (default `untrained_seed0`), `--src` (default `results/encoding`),
  `--out-dir` (default `results/visualrois`), and `--no-write`.
- Every run writes `<model>_<date>.json` + `.txt` carrying: the git commit and
  whether the tree was dirty, the model and source directory, the subject list,
  **the convention** (`nc_min`, `min_voxels`, the denominator mask,
  median-of-ratios, the SPI definition, the V1 reference, the test), the exact
  input files with sizes and mtimes, the per-ROI SPI means, and all three
  paired contrasts including per-subject differences.
- The convention is recorded in the artifact because inferring it from code is
  what §86 cost a full exchange, and the input sizes/mtimes are recorded
  because that is what settled §87 and §88.

### §89.2 The audit anchor is preserved, and proved so

The default is deliberately unchanged behaviour. Verified by diffing the new
script's default output against the pre-refactor output captured before any
edit:

```
diff baseline.txt after.txt  ->  ANCHOR BYTE-IDENTICAL TO BASELINE
```

`SPI hV4 - V1  mean diff=-0.5441  W=0  p=0.0078  0/8 positive` still reproduces
with no arguments. That row is the only one with continuous provenance back to
2026-08-28, so it stays the thing a future reader can check the script against.

A refactor of a script that produces published numbers is only safe if the old
numbers are captured *first*; a diff after the fact against a remembered value
would not have been evidence.

### §89.3 The other paths exercised

| invocation | `SPI hV4 − V1` | matches |
|---|---|---|
| (no arguments) | −0.5441 | §88.3 anchor |
| `--model dkl_bcs_seed0_trained` | **−0.8335** | §86.4 recomputation |
| `--src results/encoding_cc --model dkl_base_seed0_trained` | **−0.0785** | §86.4 / §87.6 |
| `--model nonexistent_model` | clean message, exit 1 | — |

The restated table now has a regenerable source: the three artifacts in
`results/visualrois/` were produced by the committed script with no
uncommitted edits, which is precisely what §2.2's superseded numbers lacked.

### §89.4 What this does not fix

The artifact records the tree as **dirty** when it is, which is honest but
means an artifact is only as trustworthy as that flag. It does not capture the
`src` directory's full contents, so a fit added between two runs of the same
model would change the numbers without changing the recorded convention — the
input list would show it, but only if someone reads it.

`verify_new_claims.py` deliberately reimplements the convention rather than
importing this script, so the twelve locked table values stay independently
checked regardless of what happens here. That separation should be preserved:
a verifier that imports the thing it verifies checks nothing.

### §89.5 The anchor now runs bare, from any directory

22c5b92 flagged that `python tmp_analysis/visualrois.py` exited 1 with
`ModuleNotFoundError: No module named 'src'` without `PYTHONPATH=.`. Not a
regression — the pre-§89 script had the same import — but a bad property for
the one row a referee may be pointed at: the failure mode for someone
following the documentation was an exception rather than a number.

Fixed three things, since the import was hiding two others:

1. The script inserts its own repo root on `sys.path` (`E402` on the following
   import is deliberate and marked).
2. `--src` and `--out-dir` **defaults** are now anchored at the repo root, so a
   zero-argument run works from any cwd. A path passed explicitly is still
   used as given, i.e. relative to cwd. The import fix alone would have turned
   a foreign-cwd run from an import error into a "no fits matched" error.
3. `git_head()` runs with `cwd=REPO`. Without it, a run from another directory
   would have stamped **whatever repository the invoker happened to be
   standing in** — a provenance stamp that silently describes the wrong tree is
   worse than none, and this is the §84 defect class again.

Verified, all against the baseline captured before §89 touched the file:

| invocation | result |
|---|---|
| `python tmp_analysis/visualrois.py --no-write` | byte-identical to baseline |
| same, run from `/scratch/.../tmp` | byte-identical to baseline |
| `PYTHONPATH=. python ...` (old style) | byte-identical — no regression |
| `--model dkl_bcs_seed0_trained` | −0.8335 |
| `--src results/encoding_cc --model dkl_base_seed0_trained` | −0.0785 |

The foreign-cwd run stamped `commit 22c5b92`, matching the repo HEAD rather
than the invocation directory, confirming (3).

Not re-run on the other two paths by the draft owner, deliberately and
correctly: `−0.8335` and `−0.0785` are locked in `verify_new_claims.py` from an
independent reimplementation of the convention, so agreement between this CLI
and that verifier is a stronger check than re-running this script would be.
They agree.

---

## §90 §86.5's estimator mismatch is bounded from existing data, and the
## per-voxel re-run is not warranted

§86.5 recorded that the +0.051 floor is **ratio-of-medians** (the recovery
JSONs store only per-ROI medians) while §2.2's effect is **median-of-ratios**,
and named a per-voxel re-run of condition B as the proper fix — roughly
8 × 1.5 h of CPU. Before spending that, the gap is measurable on real fits,
where both estimators are computable.

### §90.1 The gap scales with effect magnitude

| model | median-of-ratios | ratio-of-medians | gap | gap/\|effect\| |
|---|---|---|---|---|
| dkl_base_seed0 | −0.0785 | −0.0693 | −0.0091 | 11.6% |
| dkl_base_seed1 | −0.0964 | −0.0968 | +0.0004 | 0.4% |
| dkl_base_seed2 | −0.0489 | −0.0486 | −0.0003 | 0.6% |
| dkl_bcs_seed0 | −0.8335 | −0.8165 | −0.0170 | 2.0% |
| dkl_bcs_seed1 | −0.7850 | −0.7758 | −0.0092 | 1.2% |
| dkl_bcs_seed2 | −0.7447 | −0.7360 | −0.0087 | 1.2% |

`corr(|effect|, |gap|) = +0.754`. Absolute gap: mean 0.0074, range
0.0003–0.0170. Proportional gap: mean 2.8%, range 0.4–11.6%.

Two things worth naming. The gap is **directional** — median-of-ratios is the
larger magnitude in 5 of 6 fits — so for a positive floor the same tendency
would make it slightly *larger*, not smaller. And the 11.6% outlier is
`dkl_base_seed0`, the fit containing subj01 (§87); the other two no-diet seeds
sit at 0.4% and 0.6%, so the proportional mean of 2.8% is itself inflated by
one influential subject and is conservative for a typical fit.

### §90.2 Neither extrapolation moves a claim

| extrapolation | floor gap | floor range | diet / floor | no-diet / floor |
|---|---|---|---|---|
| proportional (2.8%) | 0.0015 | 0.0496–0.0526 | 15.9–16.8× | 1.5–1.6× |
| constant (0.0074) | 0.0074 | 0.0437–0.0585 | **14.2–19.1×** | **1.3–1.8×** |

The constant-gap row is the conservative bound, and it is almost certainly too
pessimistic given the +0.754 scaling. §2.2's reframed claim — a large gradient
with the diet, none resolvable without it — requires diet ≫ 1 and no-diet ≈ 1.
Both hold across the whole of both ranges.

### §90.3 Therefore: no re-run

The re-run would refine +0.051 to about ±0.002 (proportional) from a
conservative ±0.007, and change **no** stated conclusion. Under the standing
rule — do not stall on marginal gains; when results are messy the fix is
reframing, not more compute — that compute is not justified.

What §E2.1 should say changes, though: not "approximate, proper fix pending"
but **"approximate, bounded at ±0.007 conservatively and ±0.002 on the
observed scaling, and the bound does not move the comparison"**. A disclosed
limitation with a bound is a much stronger disclosure than one with a promise
attached, and it does not leave a referee wondering what the missing run would
have shown.

The estimator mismatch stays disclosed. It is the *re-run* that is closed, not
the caveat.

### §90.4 What would change this

If the diet-arm floor from 26442320 comes back much smaller than +0.051, the
ratios shift and the no-diet arm could fall below 1×, at which point a ±0.007
uncertainty starts to matter for whether the no-diet effect is distinguishable
from the instrument at all. So this conclusion is contingent on the diet-arm
floor being of the same order, and should be re-checked when that lands rather
than treated as settled independent of it.

---

## §91 Inventory of the two encoding directories: neither is a superset, and
## the control table can only come from `encoding/`

389bfb9 asked whether the six front-end controls are fitted under
`encoding_cc`. Enumerated rather than eyeballed, because "looked to hold" is
how §88 happened.

| model | `encoding` | `encoding_cc` |
|---|---|---|
| dkl_base_seed0_trained | n=8 | n=8 |
| dkl_base_seed1_trained | n=8 | n=8 |
| **dkl_base_seed2_trained** | **–** | **n=8** |
| dkl_bcs_seed0_trained | n=8 | n=8 |
| **dkl_bcs_seed1_trained** | **–** | **n=8** |
| **dkl_bcs_seed2_trained** | **–** | **n=8** |
| dkl_bcs_seed0_pca16 / pca64 | – | n=8 each |
| **dkl_bcs_parvo_highsf_seed0_trained** | **n=8** | **–** |
| **dkl_bcs_swap_seed0_trained** | **n=8** | **–** |
| gabor_blur_cs | n=8 | n=8 |
| **gabor_noblur** | **n=8** | **–** |
| **untrained_konio_matched / _raw** | **n=8 each** | **–** |
| **untrained_seed0** | **n=8** | **–** |
| **untrained_seed0_noblur** | **n=8** | **–** |
| thesis_dvd_trans | n=8 | n=8 |

### §91.1 The answer: yes, `encoding_cc` holds only one of the six controls

Of the six controls, only **`gabor_blur_cs`** is in `encoding_cc`. Seven models
exist in `encoding/` and nowhere else: `dkl_bcs_parvo_highsf_seed0_trained`,
`dkl_bcs_swap_seed0_trained`, `gabor_noblur`, `untrained_konio_matched`,
`untrained_konio_raw`, `untrained_seed0`, `untrained_seed0_noblur`.

So the control table **cannot** be regenerated from the declared source. That
is a genuine open item, and leaving it visibly flagged rather than
half-rebuilt is the right call.

### §91.2 NEW: neither directory is a superset of the other

`dkl_base_seed2_trained`, `dkl_bcs_seed1_trained` and `dkl_bcs_seed2_trained`
exist **only** in `encoding_cc`. The controls and the untrained baseline exist
**only** in `encoding`. The two directories are *complementary*, not one a
refit of the other.

This matters for how the declared-source statement is phrased. "`encoding_cc`
is authoritative" is correct and sufficient **for the twelve table values** —
all six diet/no-diet seed fits are there. It cannot be a blanket rule, because:

- the three-seed table needs `encoding_cc` (seeds 1–2 are not in `encoding`);
- the control table and the untrained baseline need `encoding`.

§87.7 noted this for `untrained_seed0` alone. It is seven models, not one, and
the right form is a **per-table** source declaration rather than a global one.

### §91.3 NEW: the untrained-on-diet control has no encoding fit at all

Matching the six labels against filenames, five map to `encoding`-only fits and
one to `encoding_cc`. For **untrained-on-diet** there is no encoding fit under
any name: no `.npz` in either directory matches
`untrained.*(bcs|dvd|diet)` or the reverse. The only place that model exists is
`results/eeg_rsa/tsn_untrained_bcs_seed{0,1,2}` — an EEG-RSA analysis, a
different measurement entirely.

So either that row of E2's table maps to a fit under a different name, or its
source is the EEG-RSA result rather than an encoding fit, or the row has no
source. This needs identifying before the table is restated; it is not
something to infer. (Note the label mapping here is mine, taken from 389bfb9's
description, so the mismatch may be in my mapping rather than in the table.)

### §91.4 Consequence

The control table's diet and no-diet reference columns share the superseded
provenance, and the controls themselves are only in `encoding`. Regenerating it
means reading two directories with a stated reason for each, which is exactly
the kind of thing that should be recorded in the artifact rather than in a
shell history — the §89 stamp already records `src`, so a per-table
regeneration would carry its own source declaration.

---

## §92 The structural lead is dead, the paper's raw estimator is identified,
## and §88's "exactly 6.0×" was my own estimator artefact

07d4df6 offered the raw-reproduces/normalised-doesn't split as the one
structural lead, having searched denominator conventions and stopped on
principle. One candidate lay outside that space and is tested here.

### §92.1 The reference hypothesis: pre-specified, and dead

`visualrois.py` prints two contrasts adjacently — `SPI hV4 - V1` and
`SPI hV4 - V1/V2/V3`. For untrained the second is −0.4868 against a published
−0.471, much closer than the V1 reference's −0.5441, so a reference mix-up was
a live candidate — and exactly the kind of error this thread keeps producing.

**Prediction stated before computing:** if the published normalised values are
V1/V2/V3 contrasts, V123 must win *every* normalised row while V1 keeps
winning every raw row.

| model | stat | V1 ref | V123 ref | published | winner |
|---|---|---|---|---|---|
| untrained_seed0 | norm | −0.5441 | **−0.4868** | −0.471 | V123 |
| untrained_seed0 | raw | −0.1199 | −0.0874 | −0.118 | V1 |
| dkl_bcs_seed0 | norm | **−0.8335** | −0.4630 | −0.862 | V1 |
| dkl_bcs_seed0 | raw | −0.1660 | −0.1051 | −0.176 | V1 |
| dkl_base_seed0 | norm | **−0.0785** | −0.0333 | −0.101 | V1 |
| dkl_base_seed0 | raw | −0.0271 | +0.0001 | −0.034 | V1 |

(Raw rows regenerated under the paper's median-of-differences estimator after
§92.2 identified it. The first run of this grid used difference-of-medians and
printed −0.1251 / −0.0919 / −0.1650 / −0.1041 / −0.0258 / +0.0017; every
verdict is unchanged, since V1 wins each raw row under either estimator. The
0.0045 disagreement with 5a6459f on untrained V123 raw was this and is now
resolved — the numbers above match on both sides.)

V123 wins **1 of 6**, and loses catastrophically where it matters: diet
normalised is −0.4630 against a published −0.862, off by 0.399. **The
hypothesis is dead.** The untrained near-match is coincidence, and keeping it
while discarding the diet row is precisely the "fitting the convention to the
numbers" that 07d4df6 declined to do.

Recorded as a failure, not omitted. A pre-specified test that fails is worth
more than an unstated search that succeeds.

### §92.2 The paper's raw statistic is median-of-DIFFERENCES

07d4df6 reported untrained raw as −0.1199; I had −0.1251. Enumerating:

| variant | untrained | \|gap to −0.118\| |
|---|---|---|
| difference of medians `med(uₘ) − med(uₚ)` | −0.1251 | 0.0071 |
| **median of differences `med(uₘ − uₚ)`** | **−0.1199** | **0.0019** |
| difference of medians, pre-nc | −0.0491 | 0.0689 |
| median of differences, pre-nc | −0.0424 | 0.0756 |

So the paper's "raw" is **`median(uₘ − uₚ)`** over voxels, not the difference
of the two medians — the same distinction as median-of-ratios vs
ratio-of-medians, one level down. It tightens the untrained match from 0.007 to
**0.002**, which slightly *strengthens* the raw-reproduces/normalised-doesn't
split.

The two pre-nc rows are off by 0.069–0.076, a **third independent
confirmation** of §86.3: the paper's raw is the ÷nc form.

Under the correct estimator, §86.4's raw gaps become diet **0.010** (was 0.011)
and no-diet **0.007** (was 0.008) — immaterial, but restated for accuracy.
§90's bound concerns the normalised statistic and is unaffected.

### §92.3 §88's solve corrected: 6.3×, not "exactly 6.0×"

§88 reported the required subj01 raw value as −0.1275, **6.0×** the sibling
median, 47% of the effect, and made rhetorical use of "exactly 6.0×" matching
the quarantine README's "six-fold outlier". 07d4df6's solve gave −0.1241,
6.3×, 46%, and attributed the 0.003 gap to rounding off the 3-dp published
−0.034.

It was not rounding. Redoing the solve under the paper's actual estimator:

| raw as | required subj01 | × sibling median | share |
|---|---|---|---|
| median of differences (correct) | **−0.1241** | **6.3×** | **46%** |
| difference of medians (§88's) | −0.1275 | 6.0× | 47% |

The first row reproduces 07d4df6's numbers exactly, so **the inter-session gap
was the estimator, not rounding.** The correct figure is 6.3×.

§88's conclusion is unaffected — 6.3× still matches "six-fold", and the ratio
statistic's 8.0×/53% is unchanged — but **"exactly 6.0×" was an artefact of my
using the wrong estimator**, and the exactness was doing rhetorical work it had
not earned. The honest statement is that the solve lands at 6.3× against a
README that said six-fold, and at 46% against a README that said 57%.

### §92.4 Stopping here, for the same reason

The reference hypothesis was the only structural candidate outside the
denominator space, and it failed. Continuing would mean searching for a variant
that fits, which is what 07d4df6 correctly declined. The remaining unexplained
gaps — diet normalised +0.029, diet raw +0.010, untrained normalised −0.073 —
stay disclosed as per-number drift with no established cause.

Two constraints worth recording for anyone who picks this up: a candidate must
reproduce raw **and** normalised simultaneously across all twelve, and it must
not disturb the untrained raw match, which is now good to 0.002 and is the one
row with continuous provenance to 2026-08-28.

---

## §93 The diet-arm floor verifies, but the no-diet gradient IS resolvable:
## comparing a signed bias to an effect as a magnitude ratio is the wrong test

cedadc6 delivered the diet-arm floor and asked directly whether the direction of
one inference was read the same way here. It is not, and the disagreement
changes a claim rather than a number.

### §93.1 Everything in cedadc6 verifies

| quantity | cedadc6 | recomputed here |
|---|---|---|
| diet normalised floor | +0.0375, 8/8, p=.0078 | **+0.0375, 8/8, p=.0078** |
| diet range | [+0.0141, +0.0651] | **identical** |
| diet raw floor | −0.00213, 2/8, p=.148 | **−0.00213, 2/8, p=.1484** |
| no-diet normalised floor | +0.0511, 8/8, p=.0078 | **+0.0511, 8/8, p=.0078** |
| Mann–Whitney between geometries | p = 0.328 | **U=42.0, p=0.3282** |

The two known estimator differences recur and are not errors: their
monotonicity ρ (+0.804 diet) is the *median* of per-subject ρ where mine
(+0.746) is the mean (§85.1), and their condition-A ratio fidelity (×1.151) is
a ratio of across-subject means where mine (×1.242) is the OLS slope over 56
cells (§86.7). The difference-statistic fidelity agrees closely (×0.686 vs
×0.694). The opposite-direction bias on the two statistics replicates on diet
geometry, so §86.7's withdrawal of "−0.862 is a floor rather than a ceiling"
now rests on both.

**Their headline stands and is worth having:** the floor is indistinguishable
between geometries at 8v8, so it is a property of the instrument rather than of
the model being fitted. §86.6's single-geometry caveat lifts and §90's bound
stands. Writing that as a positive finding rather than a null is right.

### §93.2 The inference asked about: a signed bias is not a noise level

cedadc6 compares |effect| to |floor| as a ratio — no-diet 0.054 against
0.051 ± 0.007, spanning 0.93×–1.23× — and reads "it can sit below its own
floor" as *strengthening* "no gradient resolvable without the diet".

**The floor and the effect have opposite signs.** The floor is **+0.051**
(toward magno); the no-diet contrast is **−0.054**. A ratio of magnitudes
treats the floor as if it were dispersion, when condition B measures a *signed
bias*: it is what the instrument returns when the truth is exactly zero. Two
quantities of opposite sign being similar in magnitude says nothing about
whether one could have produced the other — it cannot, in either direction.

The right comparison is per-subject and paired: is the real contrast displaced
from what the **same subject's own instrument** returns under a true zero?

| arm | real | own floor | real − floor | subjects negative | p |
|---|---|---|---|---|---|
| no diet | −0.0785 | +0.0511 | **−0.1295** | **8/8** | **.0078** |
| no diet, n=7 (drop subj01) | −0.0542 | +0.0500 | **−0.1032** | **7/7** | **.0156** |
| diet | −0.8335 | +0.0375 | **−0.8709** | 8/8 | .0078 |

Since the instrument returns `true + bias`, `real − floor` estimates the true
contrast — recovering ≈ −0.130 for no-diet, exactly the §86.6 correction — and
the Wilcoxon on it tests whether that true contrast differs from zero.

**It does, in every subject, at the exact n=8 and n=7 Wilcoxon floors.** The
no-diet gradient is not at the instrument's floor and does not disappear under
leave-one-out. Dropping subj01 *reduces* the displacement from −0.130 to
−0.103 but leaves it 7/7 consistent.

### §93.3 What this means for §2.2's wording

"A large gradient with the diet and **none resolvable** without it" is not what
the data support. The supported claim is:

> a large gradient with the diet (**−0.87** displacement from its own
> instrument null) and a **much smaller but still resolvable** one without it
> (**−0.13**, or −0.10 at n=7; 8/8 and 7/7 subjects, p = .0078 / .0156).

That is a factor of **6.7×** between arms, which carries the mechanism claim
§2.5 needs perfectly well. The diet is what makes the gradient *large*, not
what makes it *exist* — a weaker rhetorical claim and a true one, and the
standing rule is explicit that overclaiming is not licensed by a cleaner story.

Note this cuts against the direction cedadc6 read: the ratio framing made the
no-diet arm look marginal, and the paired test shows it is not. So the previous
"1.5×, or 1.1× under leave-one-out" framing understated the no-diet arm rather
than flattering it.

### §93.4 The assumption this rests on, stated

`real − floor` assumes the instrument's bias is approximately **additive** —
that the +0.051 measured under a zero-truth structure also applies under the
no-diet arm's actual structure. That is an assumption, not a measurement.

Two things support it and neither proves it: the bias is statistically
indistinguishable across two very different feature geometries (p = .33) whose
real contrasts differ 15-fold (−0.83 vs −0.05), and condition A shows the
estimator preserves ordering with a stable multiplicative distortion. If the
bias were strongly structure-dependent, the diet and no-diet floors would be
expected to differ, and they do not.

So the honest form for §2.2: report the displacement test as the primary
evidence that the no-diet gradient is non-zero, and disclose that it assumes an
additive instrument bias supported by the cross-geometry equivalence. That is a
stronger and more defensible statement than either the magnitude ratio or
"none resolvable".

---

## §94 The estimated-slope correction verifies, my additivity justification was
## wrong, and the fitting unit they chose is the badly-conditioned one

e1ce0d3 replaced §93.4's justification with something better and was right to.
Verified, then refined.

### §94.1 My §93.4 reasoning was wrong

I justified `real − floor` by the two floors being statistically
indistinguishable across geometries whose real contrasts differ fifteen-fold.
**Condition B sets truth to exactly zero in both arms.** So that comparison
demonstrates independence from *feature geometry*, not independence from the
*true contrast* — and the latter is the property the subtraction needs. The
cross-geometry equivalence was the wrong evidence for the claim I attached it
to. Conceded.

### §94.2 Their slope estimate reproduces exactly

Fitting `est = a·truth + b` with `b` fixed at the independently measured floor,
over **per-subject contrasts**:

| arm | a | recovered true contrast |
|---|---|---|
| no diet | **1.217** | **−0.1065** |
| diet | **1.122** | **−0.7765** |

Ratio **7.3×**. Identical to e1ce0d3 to four decimals. Estimating the
multiplicative term rather than assuming it away is a genuinely better footing
than my additivity argument.

### §94.3 But that fitting unit carries almost no slope information

Condition A's true split varies across ROIs by construction and is *the same
for every subject*, so the eight per-subject true contrasts are nearly
identical — there is almost no spread on the x-axis. The constrained fit is
well-determined only because `b` is held fixed; the free fit in that unit
collapses:

| arm | unit | a (b fixed) | free OLS |
|---|---|---|---|
| no diet | per-subject contrast (n=8) | 1.217 | **a = 3.003, b = −2.369** |
| diet | per-subject contrast (n=8) | 1.122 | a = 1.514, b = −0.466 |
| no diet | per-ROI ratio (n=56) | 1.348 | **a = 1.338, b = +0.0775** |
| diet | per-ROI ratio (n=56) | 1.302 | a = 1.242, b = +0.1378 |

The per-ROI unit spans the real range of truth, and there the free and
constrained slopes agree to within 0.01 (no diet) and 0.06 (diet) — the mark of
a well-conditioned fit. In the per-subject unit they differ by a factor of 2.5.
So the per-ROI estimate is the better-founded one:

| arm | a | recovered true |
|---|---|---|
| no diet | 1.348 | **−0.0961** |
| diet | 1.302 | **−0.6692** |

Ratio **7.0×**.

### §94.4 A model check neither of us ran

> **WITHDRAWN — SUPERSEDED BY §95.1.** This check is mis-specified. Condition
> B's bias varies across ROIs (0.099→0.148 no diet, 0.132→0.164 diet), so a
> pooled free intercept estimates the *mean* per-ROI bias while the contrast
> floor is a *difference* of per-ROI biases. Comparing them is not a test of
> the model. Against the correct reference the direction **reverses**: the form
> fits the diet arm at 0.96× and the no-diet arm at 0.62×, i.e. well for diet
> and loosely for no-diet — the opposite of what this subsection concludes.
> The valid form check is the residual intercept after removing each ROI's own
> bias (§95.3).

If `est = a·truth + b` is the right form, a **free** fit's intercept should
recover the independently measured condition-B floor. It partly does:

| arm | free intercept (per-ROI) | measured floor | agreement |
|---|---|---|---|
| no diet | +0.0775 | +0.0511 | same order, 1.5× |
| diet | +0.1378 | +0.0375 | **3.7× off** |

So the linear-plus-offset form is an approximation, better for the no-diet arm
than the diet arm, and the recovered magnitudes should be read as estimates
with real slack rather than corrections. This is worth stating in E2.1
alongside the slope, because it is the check that tests the *model* rather than
its parameters — and it is the one step of this argument that neither session
had performed.

### §94.5 The conclusion is robust to all of it

| method | no diet | diet | ratio |
|---|---|---|---|
| additive (§93) | −0.1295 | −0.8709 | 6.7× |
| slope, per-ROI (§94.3) | −0.0961 | −0.6692 | 7.0× |
| slope, per-subject (e1ce0d3) | −0.1065 | −0.7765 | 7.3× |

**6.7–7.3× across every treatment.** Neither sign changes, both arms stay
non-zero, and the between-arm factor moves by less than 10%. The sign and the
per-subject consistency (8/8 and 7/7 at the exact Wilcoxon floors) come from
the displacement itself, not from `a` — any `a > 0` rescales both arms
identically. So §2.2's "the diet makes the gradient large; it does not make it
exist" holds under all three, and the reduction is 85–86% however it is
computed.

### §94.6 On the framing that got reinstated

e1ce0d3 notes this partly restores the paper's original "nearly flat without
it" with its 79% reduction — structurally right, numerically superseded, now
85–86%. That is worth recording as it stands: the record now carries **two
withdrawn framings pointing in opposite directions**, each with the error
named, rather than quietly arriving at a third. The first withdrawal was mine
in effect (§86.6 argued the no-diet arm was at its floor), the second theirs,
and the surviving claim is closer to the original than either intermediate
position. That is what the sequence actually shows and it should not be tidied.

---

## §95 §94.4's model check was mis-specified and its direction reverses — but
## the identity that replaces it is arithmetic, and the assumption is still there

26bc5a3 is right on all four points and every number verifies. One step of its
argument goes further than the evidence does.

### §95.1 §94.4 is withdrawn: it compared a mean to a difference

Condition B's bias is **not constant across ROIs**:

| arm | per-ROI bias profile | spread |
|---|---|---|
| no diet | 0.0986, 0.0956, 0.1066, 0.1332, 0.1425, 0.1464, 0.1481 | 0.0526 |
| diet | 0.1320, 0.1220, 0.1326, 0.1455, 0.1398, 0.1724, 0.1644 | 0.0505 |

So a pooled free intercept estimates the **mean** per-ROI bias, while the
contrast floor is a **difference** of per-ROI biases. §94.4 compared the first
to the second, which the model never predicts should match. Against the correct
reference the direction reverses:

| arm | pooled intercept | mean per-ROI bias | ratio | §94.4's (wrong) ratio |
|---|---|---|---|---|
| diet | +0.1378 | +0.1441 | **0.96×** | 3.68× |
| no diet | +0.0775 | +0.1244 | **0.62×** | 1.52× |

**The form fits the diet arm nearly exactly and the no-diet arm loosely — the
opposite of what §94.4 concluded.** Withdrawn.

### §95.2 But the identity is arithmetic, not evidence

26bc5a3 says the model form "makes the subtraction exact … so `real − floor`
needs no structure-independence assumption". The identity
`b(hV4) − b(V1) = contrast floor` verifies at a difference of **2.78e-17** —
machine epsilon, in both arms. That is the signature of a definitional identity,
not an empirical agreement: the contrast floor *is defined as* the contrast of
condition B's profile, and `b(ROI)` *is* condition B's profile. The two
quantities are the same arithmetic, so their agreement carries no evidential
weight.

What the subtraction actually needs is that **`b(ROI)` is the same when truth ≠ 0
as when truth = 0**. Condition B measures `b(ROI)` only at truth = 0. The
identity says nothing about that transfer. So the structure-independence
assumption has been **relocated, not removed** — from "the contrast bias is
constant" to "each ROI's bias is independent of that ROI's truth", which is a
weaker and more plausible assumption but still an assumption.

This matters because it is the third time in this thread an exact numerical
agreement has been mistaken for a check (cf. §82's `max_corr < 1e-3` assertion
that could not fail, and §84's simulation verified internally while simulating
the wrong thing). **An identity that holds to machine precision is usually
telling you it is a tautology.**

### §95.3 What does test it, and what it says

26bc5a3 correctly identifies the residual intercept as the real form check, and
here it is decisive in a way worth quantifying:

| arm | bias-removed a | residual intercept | recovered true | residual as share of recovered |
|---|---|---|---|---|
| diet | 1.210 | **−0.0016** | −0.7198 | **0%** |
| no diet | 1.281 | **−0.0428** | −0.1011 | **42%** |

All reproducing 26bc5a3 exactly. For the diet arm the model transfers
essentially perfectly. **For the no-diet arm — the arm whose non-zero-ness is
the contested claim — an uncaptured additive term equal to 42% of the recovered
magnitude remains.**

Sign consequence: the residual has the *same* sign as the recovered contrast, so
treating it as unremoved bias gives ≈ −0.068 and treating it as signal gives
more negative than −0.101. **The no-diet contrast spans roughly −0.07 to −0.14
and is negative throughout.** So the sign survives and the magnitude carries
real slack — which is what 26bc5a3's "approximate for the no-diet arm" already
says, now with a number on it.

### §95.4 The bracket, properly specified

| method | no diet | diet | factor |
|---|---|---|---|
| additive (§93) | −0.1295 | −0.8709 | 6.7× |
| per-ROI, pooled intercept (§94.3) | −0.0961 | −0.6692 | 7.0× |
| **per-ROI, bias-removed (correct)** | **−0.1011** | **−0.7198** | **7.1×** |
| per-subject, b pinned (e1ce0d3) | −0.1065 | −0.7765 | 7.3× |

§94.3's row used the mis-specified pooled intercept, so the properly specified
per-ROI figure is 26bc5a3's 7.1×, not my 7.0×. The bracket is **6.7–7.3×**,
spread under 10%, no sign changes, both arms non-zero throughout. The claim
rests on the factor, which is stable; the magnitudes carry slack, the no-diet
one considerably more than the diet one.

### §95.5 Agreed on what survives

26bc5a3's closing point is the right generalisation and stronger than mine: the
inverted noise-ceiling ramp, the profile-median floor, subj01's arm-specific
ratio, the `_raw` collision, the ratio-to-floor comparison, and §94.4's
intercept are all **one quantity used without its definition travelling with
it**. Six instances, in both directions, across two sessions.

Which is why the two durable artefacts are structural, not numerical:
`visualrois.py` writing its convention into its own output (§89), and
`verify_new_claims.py` pinning both the value and the statistic so a number
cannot silently become a different number. Those hold after both of us have
forgotten the argument, and §95.2 is the reason the verifier should pin
*statistics* and not just values — an identity pinned without its definition is
exactly the tautology this section is about.

---

## §96 The residual is identified: it tracks condition A's source-power
## schedule, not the truth — so it must not be transferred to the real data

a9ba8d7 subtracts a residual contrast as well as the condition-B floor, taking
the no-diet arm to −0.0337 and the factor to 18.6×, and flags the correction's
legitimacy as open. It is no longer open.

### §96.1 My nonlinearity hypothesis was wrong

I predicted the residual was a nonlinear response to truth, which would make it
non-transferable. A quadratic term in truth absorbs **57%** (no diet) and 28%
(diet) of the residual *variance* but **0% of the residual contrast**
(−0.0864 → −0.0889; −0.1132 → −0.1241). The residual is not simple
nonlinearity. Hypothesis dead.

### §96.2 What it actually is

Condition A generates its sources on a fixed schedule, `a = 0.30 + 0.10i`,
`b = 0.90 − 0.10i`, so total source power `a² + b²` is **U-shaped** in ROI
index: 0.90, 0.80, 0.74, 0.72, 0.74, 0.80, 0.90. Against that:

| arm | corr(residual, total power) | corr(residual, truth) |
|---|---|---|
| no diet | **−0.847** | +0.080 |
| diet | **−0.820** | +0.005 |

And across the two arms, whose real effects differ tenfold:

> **corr(no-diet residual profile, diet residual profile) = +0.989**

So the residual tracks the *magnitude schedule condition A was built with*,
carries essentially no relationship to the truth ratio, and is
near-identical in two arms whose effects differ by a factor of ten. It is a
property of the simulation's construction, not of the estimator's response to
either model.

**The real data has no such schedule.** Its source magnitudes are whatever they
are, unrelated to ROI index. So condition A's residual contrast measures the
estimator's response to a synthetic power ramp that the real data does not
share, and subtracting it from the real contrast is not a correction — it is
importing an artefact of the generator. a9ba8d7's own suspicion ("a property of
condition A's construction rather than of either model") was right, and is now
measured at r = −0.83 rather than suspected.

### §96.3 The correction is also not well enough defined to carry a claim

Three equally defensible ways of removing the same residual contrast:

| implementation | no-diet result | subjects | p |
|---|---|---|---|
| group residual, before scaling by a | −0.0337 | 6/8 | **.0547** |
| per-subject residual | −0.0337 | 6/8 | **.1094** |
| residual removed after scaling | −0.0148 | 4/8 | **1.0000** |

None reproduces a9ba8d7's 5/8, p = .3125, so a fourth variant exists that I
have not identified. That is the point: a correction whose p-value spans
**.055 to 1.00** across arbitrary implementation choices cannot be the basis
for concluding that the no-diet arm "does not hold". The spread is not evidence
about the data; it is evidence that the quantity is under-specified.

### §96.4 Recommendation: the minimal treatment is the best-founded one

| treatment | no diet | diet | factor | status |
|---|---|---|---|---|
| floor only (minimal) | **−0.1011, 8/8, p=.0078** | −0.7198, 8/8, p=.0078 | 7.1× | **best founded** |
| floor + residual contrast | −0.034 to −0.015, p=.055–1.00 | −0.6262, 8/8 | 18.6× | imports a generator artefact |

So on the evidence the no-diet arm's gradient *is* resolvable, and a9ba8d7's
earlier position was the correct one. The residual correction should be
reported as a sensitivity check that fails for identifiable reasons, not as a
bound on the claim.

**One honest limitation in the other direction.** Showing that condition A's
residual is non-transferable does not show the estimator has *no* misfit on
real data — it shows condition A cannot measure it. So the no-diet claim rests
on the floor correction alone, with real-data misfit unquantified and disclosed.
That is a weaker footing than the diet arm's, which survives every treatment.

### §96.5 Where to stop

The comparative claim — the diet arm's gradient is non-zero in every subject
under **every** treatment, and is 7–19× the no-diet arm's — is robust across
all four treatments and both estimator families. §2.5's mechanism needs only
that.

Whether the no-diet arm's own gradient is separately resolvable now turns on a
question n=8 cannot settle: whether an estimator misfit exists on real data
that no simulation here measures. Under the standing rule the answer is to
state what the data support and stop, not to build a fifth treatment. The
recommendation is: claim the comparison, report the no-diet arm as resolvable
under the best-founded correction with the sensitivity check and its diagnosis
disclosed, and stop refining.

Nine sections of this exchange have improved §2.2 materially. This one does not
change the paper's claim, and that is the signal to stop.

### §96.6 The variant set is closed, and the minimal treatment is the only
### one invariant to the choice

6684a41 identified §96.3's unplaced variant as the group-versus-per-subject
**floor pairing**. Confirmed: all six combinations of {group, per-subject}
floor × {minimal, residual-before-scaling, residual-after-scaling} account for
every number either session has produced. No fifth variant exists.

| floor pairing | treatment | no-diet | subjects | p |
|---|---|---|---|---|
| group | **minimal** | −0.1011 | **8/8** | **.0078** |
| per-subject | **minimal** | −0.1011 | **8/8** | **.0078** |
| group | + residual before scaling | −0.0337 | 6/8 | .0547 |
| per-subject | + residual before scaling | −0.0337 | 5/8 | .3125 |
| group | + residual after scaling | −0.0148 | 4/8 | 1.0000 |
| per-subject | + residual after scaling | −0.0148 | 5/8 | .8438 |

An argument for the minimal treatment neither of us had made: **it is the only
treatment invariant to the pairing choice.** Its sign count and p-value are
identical under both floor pairings, while every residual-corrected variant
swings 4/8–6/8 and p = .055–1.00 on choices that carry no scientific content.
A statistic that moves that much on an arbitrary implementation detail is not
measuring the data; one that does not move on it is at least measuring
something stable.

The diet arm is 8/8 at p = .0078 in **all six** combinations.

So the settled position is doubly supported: the residual is a generator
artefact (§96.2), and the treatment that ignores it is the only one that does
not depend on how the floor is paired.

---

## §97 The parameter-matched control is submitted — §64.5's B branch is being
## resolved rather than carried as a caveat

§72.6 recorded the `resnet34` arm as "3 seeds × ~26 h and has not been
submitted; that is a decision, not an oversight." That decision is reversed.
Job **26444916** (`--array=0-2`, `agcichy`, `qos=standard`, `gpu:a5000:1`,
32G, 4-day limit) trains three seeds of `dkl_bcs_single34`.

### §97.1 Why it had to be run

`train_single_stream.sh` pre-committed the trigger: the param-matched arm is
needed "only if this arm LOSES … and that run should be submitted then, not
now." It lost. §72.1 recorded outcome **B** — the split buys shape bias — and
§64.5's B branch names a `resnet34` arm as *required* in exactly that case.

Two readings currently survive the result that a half-capacity single stream
beat the two-stream model's 0.6749 integrated shape bias: the **split** buys
shape bias, or **twice the trunk parameters** do. This arm holds parameters
fixed and varies only the split, so:

- if `resnet34`-single still beats 0.6749, the effect is **capacity**, and leg
  (b) must stop attributing it to the two-stream architecture;
- if it does not, the split is doing the work and the 0.0875 gap stands with
  the confound **retired** rather than disclosed.

Either outcome resolves §64.5's B. Leaving it open means a referee who reads
the pre-registration finds a branch the paper declared binding and then did not
follow.

### §97.2 The premise was measured, not assumed

§72.6 asserted "~21.3M against ~22.4M". Verified before spending the GPU time:

| model | architecture | in-channels | parameters | vs two-stream |
|---|---|---|---|---|
| `dkl_bcs` | two ResNet18 trunks | (1, 1) | **22,397,936** | — |
| `dkl_bcs_single` | one ResNet18 trunk | (2, 0) | 11,230,832 | **50.1%** |
| **`dkl_bcs_single34`** | **one ResNet34 trunk** | (2, 0) | **21,338,992** | **95.3%** |

A 4.7% mismatch, against the 49.9% deficit of the arm that triggered this. The
control is genuinely parameter-matched, and §72.6's figures are confirmed to
three significant figures. Forward pass checked on the real input shape
(2-channel, 256², logits (2, 112)) for both single-stream arms.

### §97.3 The collision was designed out, not remembered

The run directory is `{condition}_seed{n}` with **no architecture in it**. So
`--condition dkl_bcs_single --like-resnet resnet34` would have written a
ResNet34 run into `dkl_bcs_single_seed0`, overwritten the existing ResNet18
checkpoints, and let `--resume` load them — silently, since the array would
report success. That is §87's failure exactly, which cost a whole arm of a
16-task array.

Fixed structurally rather than by remembering the flag:

- `like_resnet` now lives in the **condition** (`dkl_bcs_single34`), so the
  name and the architecture cannot diverge;
- `--like-resnet` defaults to `None` and still overrides for one-off probes,
  resolved as `args.like_resnet or cfg.get("like_resnet", "resnet18")`;
- the resolved architecture and total parameter count are printed at startup
  **and** written to `config.json` as `like_resnet_resolved` / `n_trunk_params`,
  because `vars(args)` alone would have recorded `like_resnet: null`.

The last point is §89's lesson applied to the trainer: an artefact should state
what was built, not leave it inferable.

### §97.4 Resources sized from measurement

`mpk_single` (26421237, ResNet18) ran 150 epochs in 1d02–03h with
**MaxRSS 23,067,728K = 22.0 GiB against a 22G request** — sitting exactly at
its limit with no headroom. ResNet34 is deeper, so `--mem=32G` (measured +
~45%), inside the 64G project cap. Expect ~2 days per seed at ~1.8× the
ResNet18 FLOPs; `--time=4-00:00:00` leaves margin. Checked before submitting:
no `QOSMaxGRESPerUser` pending, 12 GPU jobs running, `standard` caps at 16.

### §97.5 The analysis protocol is already fixed

§64.6 binds, and is not renegotiable after seeing the numbers: assert 150
epochs and report sd(val_top1) over the last 20 plus best-epoch as well as
last-epoch accuracy; three seeds both sides; report complete separation and the
raw difference with its range, **not** a p-value, since 3v3 floors at p = 0.10;
report the cue-conflict decision rate alongside the shape bias; no stream
statistics, as this arm has none.

One addition from §92–§96: whichever shape-bias statistic is quoted must be
named with its estimator. The single-stream comparison is a difference of two
scalars, so it has none of §2.2's ratio-versus-numerator ambiguity — but the
decision-rate caveat in §64.6(4) is the same shape of problem, a ratio whose
denominator changes between arms, and should be reported as a count over a
named subset rather than as a bare percentage.

---

## §98 §7.2 item 2 done: the eccentricity bands were confounding two effects
## of opposite sign, and controlling pRF size doubles the eccentricity result

§7.2 item 2 asked for NSD's continuous pRF maps on the grounds that
"magno explains *peripheral* voxels" and "magno explains *large-pRF* voxels"
are different claims that the 5-bin eccentricity analysis cannot separate.
They are different, they point in **opposite directions**, and separating them
strengthens the eccentricity result rather than qualifying it.

Data: `prf_eccentricity`, `prf_size`, `prf_angle`, `prf_R2` for all 8 subjects,
fetched into the existing NSD tree on `/scratch` (32 volumes, ~44 MB, each
verified against its Content-Length and gzip-tested — a truncated volume would
have decompressed to garbage and failed late).

### §98.1 Two preconditions, both tested before any model was fitted

**Alignment.** The pRF volumes are indexed with the same `nsdgeneral` mask the
fits used, via the project's own `_load_volume_zyx` — but that is a claim until
tested. `prf-eccrois` is an ordinal band label attached through the same mask,
so median pRF eccentricity must rise monotonically across it:

| band | 0–0.5° | 0.5–1 | 1–2 | 2–4 | 4+ |
|---|---|---|---|---|---|
| median pRF ecc (subj 1) | 0.28 | 0.69 | 1.36 | 2.58 | 4.34 |

Monotonic in **8/8 subjects**, and the values land inside the intervals their
labels name. Indexing confirmed rather than assumed. `load_prf` lives in
`nsd_data.py` beside `load_roi_labels` so the mask convention cannot drift from
the fits.

**Collinearity — the enabling result.** Eccentricity and pRF size both grow
with distance from the fovea, so the question is only answerable if they are
separable within subject. Measured: **r(log ecc, log size) = +0.611**
(range +0.391..+0.698), **VIF = 1.66**. Had this been ~0.9 the partial betas
would have been noise and the honest report would have been "these data cannot
control for size". They can.

Coverage also improves as §7.2 predicted: ~5.8k–9.8k voxels per subject at
`prf_R2 > 5`, against the 33% of the mask that carries a band label.

### §98.2 The result: a suppression effect the bands could not see

Standardised OLS betas within subject, aggregated over 8 subjects (subject as
the resampling unit), Wilcoxon signed-rank. Diet arm, seed 0, both statistics:

| term | norm | raw |
|---|---|---|
| ecc alone | +0.066, 0/8 neg, p=.0078 | +0.103, 1/8, p=.0391 |
| size alone | −0.013, 5/8, p=.3125 | −0.158, 8/8, p=.0078 |
| **ecc \| size** | **+0.119, 0/8, p=.0078** | **+0.325, 0/8, p=.0078** |
| **size \| ecc** | **−0.086, 8/8, p=.0078** | **−0.356, 8/8, p=.0078** |

**Controlling pRF size nearly doubles the eccentricity beta** (+0.066 → +0.119
norm; +0.103 → +0.325 raw). That is a textbook suppression pattern and it is
mathematically expected here rather than surprising: the two predictors
correlate at +0.61 and act with opposite signs, so each was masking the other
in the marginal analysis. The banded result **understated** eccentricity.

All three diet seeds agree on all four terms.

Read plainly: within the diet arm, magno-unique variance rises toward the
periphery **and**, at matched eccentricity, falls as pRF size grows. Those are
two separate facts about the same cortex, and the 5-bin analysis reported their
sum.

### §98.3 Sensitivity, with the criterion stated before it was run

Two conventions carry no scientific content — the `prf_R2` threshold and the
log transform — so both were varied over 16 cells per arm, against a rule
fixed in advance: the dissociation is a property of the data only if
diet keeps `ecc|size > 0, size|ecc < 0` and no-diet keeps
`ecc|size ≈ 0, size|ecc > 0` in **every** cell.

**Diet passes all 16.** ecc|size runs +0.010 to +0.231 and size|ecc −0.009 to
−0.231 across both parameterisations and R2 ∈ {0, 5, 10, 20}, with 8/8 sign
consistency and p = .0078 in 14 of 16; the weakest cell is linear/R2>0
(p = .0781), where including unfittable voxels attenuates everything.

**No-diet fails, and the failing half is not claimed.** The absence of an
eccentricity effect is robust (ns in all 8 cells, |β| ≤ 0.026 under log). But
the apparently positive **size** effect is log-specific: +0.035 (p=.0078) under
log at R2>5, yet it **flips sign** under the linear parameterisation at R2>5
(−0.022) and R2>10 (−0.054), both ns. By the pre-stated rule that is the filter
talking. §96's lesson applies unchanged — a statistic that moves on an
arbitrary implementation choice is not measuring the data — so the no-diet
size effect is recorded as **not established**, not as a finding.

### §98.4 What this supports, and what it does not

> **Claim (ii) below is WITHDRAWN — see §99.** "The eccentricity component is
> absent in the no-diet arm … an arm difference" rested on *significant in one
> arm, not significant in the other*, which is not a test of the difference
> (Gelman & Stern) and is a move §8 refuses elsewhere. The paired
> within-subject interaction was then run and the arm difference does hold
> (+0.134 norm / +0.305 raw, 8/8, at the n=8 floor) — but **conditionally on a
> pRF-quality filter**, not "under every convention" as claimed here, and it
> holds for the *partial* while the arms are indistinguishable on the
> *marginal* slope. §99 supersedes this subsection's (ii) and its robustness
> wording. Claims (i) and (iii) stand.

**Supported.** (i) In the diet arm, stream preference has an eccentricity
component and a pRF-size component of opposite sign, both robust across every
convention tested. (ii) The eccentricity component is **absent** in the no-diet
arm under every convention — an arm difference on the retinotopic claim
specifically, which is what leg (c) needs. (iii) §7.2 item 2's premise was
correct: the two claims were confounded, and the confound was costing the
eccentricity result about half its size.

**Not supported.** Any statement about pRF size in the no-diet arm. And the
suppression means the *marginal* eccentricity beta reported from bands is not
the quantity to quote going forward — the partial is, with the size term
reported beside it.

### §98.5 Limitations

- `prf_R2 > 5` retains 45–60% of the mask; the excluded voxels are those the
  pRF model could not fit, which are not missing at random and are plausibly
  the noisiest. The R2>0 row is the closest thing to an unfiltered check and
  the diet dissociation survives it, weakly (p=.078 linear).
- Betas are standardised within subject, so they are comparable across
  subjects but are not effect sizes in degrees.
- This is a *correlational* dissociation over voxels. It says the two axes
  carry different information about stream preference; it does not establish
  that either is the causal variable.
- `prf_angle` was downloaded and is unused here. Polar angle is the obvious
  next control — a genuine retinotopic account should be angle-invariant, and
  anything that is not would point at a coverage artefact instead.

---

## §99 §98.4's arm-difference claim was the significant/not-significant
## fallacy — tested properly it holds, and it reconciles with the draft

`950c6df`'s session caught that §98.4 asserted "the eccentricity component is
absent in the no-diet arm … an arm difference" on the strength of *significant
in one arm, not significant in the other*. That is not a test of the difference
(Gelman & Stern 2006), and §8 (line 346) already refuses exactly this move
elsewhere in the paper. **§98.4's phrasing is withdrawn** and replaced by the
paired test below.

### §99.1 The interaction, tested the way the claim requires

Per subject, per arm: the ecc|size partial. Differenced **within subject**,
tested across the 8 subjects with the same estimator and resampling unit.

| pairing | mean diff | range | subjects | p |
|---|---|---|---|---|
| seed0 vs seed0 | +0.123 | −0.011..+0.234 | 7/8 | .0156 |
| seed1 vs seed1 | +0.142 | +0.034..+0.345 | 8/8 | .0078 |
| seed2 vs seed2 | +0.137 | +0.015..+0.218 | 8/8 | .0078 |
| **seed-averaged, norm** | **+0.134** | +0.044..+0.252 | **8/8** | .0078 |
| **seed-averaged, raw** | **+0.305** | +0.024..+0.700 | **8/8** | .0078 |
| seed-averaged, size\|ecc, norm | −0.119 | −0.172..−0.073 | 8/8 neg | .0078 |
| seed-averaged, size\|ecc, raw | −0.502 | −0.755..−0.243 | 8/8 neg | .0078 |

The arms differ on **both** partial terms, not only eccentricity.

**Note on every p in §98 and here**: the two-sided Wilcoxon at n=8 floors at
2/2⁸ = **.0078125**. A p at that value is *8/8 sign consistency*, not an
independently strong test, and §98 should have said so. The interaction is
therefore not a ninth confirmation stacked on the eight before it — it is one
test, of the quantity the claim actually names.

### §99.2 Robustness: 11 of 16, and the failures are not arbitrary

Same 16-cell grid as §98.3, applied to the interaction:

| filter | log/norm | log/raw | linear/norm | linear/raw |
|---|---|---|---|---|
| R2>0 | .0781 ✗ | .1484 ✗ | .5469 ✗ | .8438 ✗ |
| R2>5 | .0156 ✓ | .0156 ✓ | .0234 ✓ | .3125 ✗ |
| R2>10 | .0078 ✓ | .0078 ✓ | .0078 ✓ | .0391 ✓ |
| R2>20 | .0156 ✓ | .0156 ✓ | .0078 ✓ | .0078 ✓ |

**By §98.3's stated bar — every cell — this fails.** Stating that plainly
rather than moving the bar: five cells do not clear .05.

All four failures at R2>0 share one cause, and it is not an arbitrary
convention. R2>0 admits voxels the pRF model could not fit, whose eccentricity
and size values are noise; including them attenuates every effect toward zero,
which is visible in the within-arm betas too (§98.3's diet linear/R2>0 cell
fell to +0.010). R2>0 is not a threshold choice on the same footing as 5 vs 10
vs 20 — it is the absence of a quality filter. The fifth failure
(linear/raw, R2>5) has no such account and is simply a miss.

**The defensible claim is therefore conditional and should be written that
way**: the arm difference holds wherever the pRF estimates are usable — 7 of 8
cells at R2 ≥ 5, and 4 of 4 at R2 ≥ 10 — and does not hold without a pRF-quality
filter.

### §99.3 The reconciliation: the draft's sentence is right

> **WITHDRAWN — SUPERSEDED BY §100.1/§100.2.** The verdict ("line 1890 is
> right, no correction needed") happens to be correct, but it was reached on
> the wrong contrast. `blur_verdict.py:24` pins line 1890 as
> `untrained_seed0` vs `untrained_seed0_noblur` — one **untrained** network,
> diet front end against no-blur front end. This subsection compares **trained**
> diet against **trained** no-diet across three seeds. Matching "difference
> n.s." was a coincidence of outcome, not a reproduction, and the mechanism
> story below ("the size confound was hiding the arm difference") cannot apply,
> because line 1890 was never about arms. Run on its own contrast the partial
> does **not** separate the two front ends (−0.068, p=.055, wrong direction),
> so §99.4's proposed companion is retracted too. §99.1 and §99.2 stand as
> claims about trained arms.

The draft's live eccentricity sentence (line 1890) reports the arms as **not**
differing: "raw slope +0.047 vs +0.041, 8/8 both, difference n.s." That is a
**marginal** slope. §99.1 tests a **partial**. Running both on the same data:

| statistic | term | diet | no diet | difference | subjects | p |
|---|---|---|---|---|---|---|
| raw | **marginal** | +0.089 | +0.096 | **−0.008** | 3/8 | **.8438** |
| raw | **partial** | +0.300 | −0.006 | **+0.305** | 8/8 | **.0078** |
| norm | marginal | +0.065 | +0.005 | +0.060 | 7/8 | .0156 |
| norm | partial | +0.122 | −0.012 | +0.134 | 8/8 | .0078 |

On the raw statistic — the one the draft names — the marginal interaction is
emphatically null (p = .84, 3/8), reproducing the draft's "difference n.s."
exactly. **The draft is correct about the quantity it names, and does not need
correcting.**

What the partial adds is the reason: in the no-diet arm the apparent
eccentricity trend is **entirely carried by pRF size** — marginal +0.096
collapses to **−0.006** once size is controlled. In the diet arm a genuine
eccentricity effect remains (+0.300) after the same control. So the marginal
comparison cannot separate the arms because it is comparing a real
eccentricity effect against a size effect wearing eccentricity's clothes.

That is a better result than "the draft was wrong", and it is the third time
in this project that a confound has been hiding a difference rather than
manufacturing one.

### §99.4 What leg (c) can claim

> Controlling pRF size, magno-unique variance rises with eccentricity in the
> diet arm and not in the no-diet arm (paired within-subject difference
> +0.134 norm / +0.305 raw, 8/8 subjects, at the n=8 sign-consistency floor),
> while the two arms are indistinguishable on the *marginal* eccentricity
> slope (raw, p = .84) because the no-diet arm's apparent trend is carried by
> pRF size. Holds at every pRF-quality threshold tested (R2 ≥ 5 in 7 of 8
> convention cells, R2 ≥ 10 in 4 of 4); not established without a quality
> filter.

The existing "difference n.s." sentence stays true and gains a companion rather
than a correction.

### §99.5 The angle control, with the aperture wrinkle

`950c6df` adds the constraint that matters for the next step: NSD's aperture is
~8.4°, so coverage is anisotropic near the edge and eccentricity and polar
angle are not independent there. An apparent angle effect at high eccentricity
could be aperture geometry rather than retinotopy, so the angle control is
informative **within** an eccentricity band, not pooled — pooled, a null could
be masking two artefacts that cancel. Recorded for whoever runs it.

---

## §100 §99.3 was the wrong contrast. Running the right one, plus the input
## null, reframes the result: the diet PRESERVES an architectural property

### §100.1 §99.3 withdrawn

§99.3 claimed my trained diet-vs-no-diet result "reproduced" the draft's
line 1890 "difference n.s." exactly. It does not. `blur_verdict.py:24` pins
that sentence's contrast:

```python
MAIN, CTRL = "untrained_seed0", "untrained_seed0_noblur"
```

One **untrained** network, diet front end against no-blur front end — an
input-level control where the blur can actually be removed. The draft sentence
sits inside "The foveal band's sign flips **when the blur is removed**", so
"in both models" means both front ends, untrained, not both training arms.
Same verdict, different quantity: matching n.s. was a coincidence of outcome.
**§99.3's reconciliation is withdrawn**; §99.1's interaction and §99.2's
robustness stand, as claims about trained arms only.

### §100.2 Line 1890 gets no companion — the partial does not separate the front ends

Running the §98 decomposition on the pair line 1890 actually names:

| term | diet FE | no-blur FE | difference | subjects | p |
|---|---|---|---|---|---|
| ecc marginal (raw) | +0.286 | +0.292 | −0.006 | 3/8 | .9453 |
| ecc\|size (raw) | +0.401 | +0.470 | **−0.068** | 2/8 | .0547 |
| size\|ecc (raw) | −0.187 | −0.292 | +0.105 | 8/8 | .0078 |
| ecc\|size (norm) | +0.083 | +0.089 | −0.006 | 4/8 | .9453 |

Controlling pRF size does **not** separate the two front ends, and what
movement there is runs the *wrong* way. So line 1890 stands alone, exactly as
written, and §99.4's suggested companion is retracted.

### §100.3 The finding underneath: this is architectural, not diet-created

> **CORRECTED BY §101 — use that table, not this one.** Three defects here:
> (a) the no-diet arm is contrasted against `untrained_seed0`, the **diet**
> front end, when its own initialisation is `untrained_seed0_noblur` — so
> −0.374 confounds front end with training; corrected it is **−0.442**;
> (b) "untrained already exceeds trained diet" reads a direction off a null
> (−0.077, p=.1094) and is withdrawn — the supported statement is "essentially
> unchanged"; (c) four of the five contrasts cross `encoding_cc` against
> `encoding`, a boundary that is removable since all six models exist in
> `encoding`. §101 redoes the whole table within-pipeline. The conclusion
> survives and strengthens, but is **raw-statistic-specific** (§101.4).

Both untrained models carry the dissociation strongly — ecc|size **+0.401** and
**+0.470** raw, 8/8, larger than the trained diet arm's +0.325. It is present
at initialisation regardless of front end.

Against the §7.2-item-1 input null, all raw, paired within subject:

| model | ecc\|size | size\|ecc |
|---|---|---|
| `gabor_blur_cs` (input null, diet FE) | +0.096, 7/8, p=.109 | +0.004, ns |
| `gabor_noblur` (input null, no-blur FE) | +0.201, 8/8, p=.0078 | −0.137, 8/8 |
| untrained, diet FE | +0.401, 8/8 | −0.187, 8/8 |
| untrained, no-blur FE | +0.470, 8/8 | −0.292, 8/8 |
| **trained diet** | **+0.325, 8/8** | **−0.356, 8/8** |
| **trained no diet** | **+0.028, 5/8, ns** | **+0.163, 0/8 neg** |

| contrast (ecc\|size, paired) | diff | subjects | p |
|---|---|---|---|
| untrained diet − input null | **+0.306** | 8/8 | .0078 |
| trained diet − input null | **+0.229** | 7/8 | .0156 |
| trained no-diet − input null | −0.068 | 2/8 | .2500 |
| trained diet − untrained diet | −0.077 | 2/8 | .1094 |
| **trained no-diet − untrained diet** | **−0.374** | **0/8** | **.0078** |

Three things follow, and none of them is what §98 implied.

1. **It survives the input null on the diet side.** The diet front end's *input*
   carries +0.096 (ns); the network adds +0.306 over its own input. So this is
   not a front-end validation — the control §7.2 item 1 was written to catch
   does not catch it. (Asymmetry worth noting: `gabor_noblur`'s input *does*
   carry +0.201 significantly, so the no-blur front end's input carries more of
   the structure than the diet one's.)
2. **Training does not create it.** Untrained diet (+0.401) already exceeds
   trained diet (+0.325), and trained − untrained is −0.077, n.s.
3. **Training without the diet destroys it.** Trained no-diet sits 0.374 below
   untrained, 0/8 subjects, at the floor — and its `size|ecc` term **flips
   positive** (+0.163), the only model of the six that does.

### §100.4 The reframing for leg (c)

Not "the diet produces an eccentricity organisation" but:

> The eccentricity/pRF-size dissociation is present at initialisation and is
> largely not inherited from the input. Training on an undieted diet destroys
> it — the eccentricity term collapses to zero and the size term reverses sign.
> Training with the developmental diet preserves it.

That is a **better** claim than §98's. It is developmental in the right way: the
diet protects a structure the architecture already has, rather than imposing
one, which is what a critical-period account should predict. It also removes
the awkwardness of asserting that a randomly-initialised network has no
retinotopic organisation — it plainly does.

It also means §98.4's claim (i) needs its scope corrected: the diet arm's
dissociation is real, but "in the diet arm" was doing work that "at
initialisation, and retained under the diet" does properly.

### §100.5 Caveats

- Trained models come from `results/encoding_cc`, untrained and Gabor from
  `results/encoding` (§91: neither directory is a superset). §91 verified the
  two agree where they overlap, but `untrained_*` and `gabor_*` have no cc
  counterpart, so that check cannot be run for these specific fits.
- The untrained-vs-trained comparison is across models with very different
  overall R²; the betas are standardised within subject, which makes the
  *direction* comparable but not the magnitudes.
- Everything here is seed 0 for the trained arms. §98 showed all three seeds
  agree on the within-arm terms; the untrained and Gabor arms have one seed
  each by construction.

### §100.6 Accepting the caution on the "three confounds" pattern

`950c6df` asked me to hold the "three confounds hiding a difference" pattern
loosely until this ran, on the grounds that two of them may be one phenomenon
counted twice. That is right and the count should be **two**, not three: §98's
suppression and §99's marginal-vs-partial gap are the same eccentricity/size
structure seen at within-arm and between-arm level, not two independent
instances. Recorded rather than quietly dropped.

---

## §101 §100.3 corrected: within-pipeline, each arm against its own
## initialisation — the destruction is larger, and it is raw-statistic-specific

`950c6df` raised three defects in §100.3. All three are real and all three are
fixed here. Every fit below is from `results/encoding`, so **no contrast
crosses the pipeline boundary**.

### §101.1 Each arm against its OWN initialisation

§100.3 contrasted the no-diet arm against `untrained_seed0` — the **diet**
front end. Its own initialisation is `untrained_seed0_noblur`, so the published
−0.374 confounded front end with training. Corrected, and the legs become
parallel:

| arm | initialisation | trained | change | subjects | p |
|---|---|---|---|---|---|
| diet | +0.401 | +0.325 | **−0.077** | 2/8 | .1094 |
| no diet | +0.470 | +0.028 | **−0.442** | **0/8** | **.0078** |

The destruction is **larger** than §100.3 claimed, not smaller. The `size|ecc`
term, within arm:

| arm | change in size\|ecc | subjects | p |
|---|---|---|---|
| diet | −0.169 (more negative) | 0/8 | .0078 |
| no diet | **+0.454 (reverses sign)** | 8/8 | .0078 |

### §101.2 "Untrained exceeds trained diet" is withdrawn

> **ITS REPLACEMENT IS ALSO WITHDRAWN — see §102.3.** "Training with the diet
> leaves it essentially unchanged" treats this as a null, and it is not one:
> the raw and norm point estimates have **opposite signs** (−0.077 vs +0.036)
> and **both** 90% intervals exclude zero. Neither "exceeds", "reduced" nor
> "preserved" is supportable; the term cannot be characterised from these data.

§100.3 said untrained diet (+0.401) "already exceeds" trained diet (+0.325).
That difference is −0.077, 2/8, **p = .1094** — a direction read off a null,
the §98.4 fallacy one notch quieter. Withdrawn.

The supported statement is **"training with the diet leaves it essentially
unchanged"**, which is also the better fit to the reframing: a critical-period
account predicts structure *held*, not structure reduced, and the stronger
phrasing invited a question the evidence cannot answer.

### §101.3 The input-null asymmetry, answered

§100.5 flagged that `gabor_noblur` (+0.201) carries more than `gabor_blur_cs`
(+0.096) and offered no account. The matched contrast settles what that does
and does not mean:

| contrast | diff | subjects | p |
|---|---|---|---|
| untrained diet − input null **diet** | +0.306 | 8/8 | .0078 |
| untrained no-blur − input null **no-blur** | **+0.268** | **8/8** | **.0078** |
| trained diet − input null diet | +0.229 | 7/8 | .0156 |
| **trained no-diet − input null no-blur** | **−0.173** | 2/8 | .1094 |

**The asymmetry is in the inputs, not in what the network adds.** Both
untrained models exceed their own input null by a similar margin (+0.306,
+0.268), so "present at initialisation and largely not inherited from the
input" holds for **both** front ends, symmetrically. I still have no account
for why the two Gabor banks differ, and 950c6df notes the obvious guess —
blur removes high-SF, which should *raise* an eccentricity gradient — predicts
the wrong sign. Left unexplained rather than furnished with a story.

The fourth row is the sharpest single result here: **after no-diet training the
model carries no more eccentricity structure than a Gabor bank on its own
input** (−0.173, n.s.), while the diet-trained model clearly does (+0.229,
p = .0156).

### §101.4 The honest split: this is a raw-statistic result

> **WITHDRAWN — SUPERSEDED BY §102.** "Write it up as a difference-statistic
> result" is not an available remedy: §8.1 is pre-declared and says a result is
> **claimed only where the two statistics agree**, and the draft already spends
> that rule at line 346. Two contrasts are therefore REFUSED, not relabelled —
> the no-diet ecc|size destruction (−0.442 vs −0.093 n.s.) and untrained-diet
> vs its input null (+0.306 vs +0.048 n.s.). §102.2 carries what survives.

On the norm statistic the same table is much weaker, and two load-bearing
contrasts do not reach significance:

| contrast | raw | norm |
|---|---|---|
| no diet: trained − own init | −0.442, 0/8, **.0078** | −0.093, 2/8, **.1484** |
| untrained diet − input null diet | +0.306, 8/8, **.0078** | +0.048, 6/8, **.1094** |
| trained diet − input null diet | +0.229, 7/8, .0156 | +0.084, 8/8, .0078 |
| untrained no-blur − input null no-blur | +0.268, 8/8, .0078 | +0.067, 8/8, .0078 |
| no diet: size\|ecc reversal | +0.454, 8/8, .0078 | +0.073, 8/8, .0078 |

So the claim must be stated as resting on the **difference** statistic. What
survives on both is the `size|ecc` **sign reversal** in the no-diet arm (8/8,
p = .0078 on each) — that is the single most robust element of the whole
result, and it is the one to lead with if only one is quoted.

The norm index is bounded and divides by (uₘ+uₚ), which shrinks exactly where
the unique variances are small; that is a plausible reason for the attenuation
but it is a rationalisation, not a test, and the split should be reported
rather than argued away.

### §101.5 The cost of going within-pipeline, stated

`dkl_bcs_seed1/2` and `dkl_base_seed1/2` are `encoding_cc`-only (§91.2), so
this table is **seed 0 by construction**. Leg (c) therefore has a choice, and
must say which it took:

- **within-pipeline at one seed** (this table), or
- **three seeds across the pipeline boundary** (§100.3's version).

I take within-pipeline first, on 950c6df's reasoning that a confound you can
eliminate beats one you can only disclose. Seed stability for the *trained
arms* is already established separately in `encoding_cc`: §98.2 showed all
three seeds agree on every within-arm term and §99.1 showed the arm interaction
holds for all three seed pairings. What is seed-0-only is specifically the
**comparison against untrained and against the input null**, because those
models have one seed by construction and no `encoding_cc` counterpart.

---

## §102 §8 applied to the pRF result: one claimable dissociation, two
## contrasts refused, two nulls that cannot be stated at all

§101.4 proposed writing the eccentricity result up as "a difference-statistic
result". That is not an available remedy. §8.1 is pre-declared and explicit:

> Every contrast is therefore reported under **both** … **A result is claimed
> only where the two agree** (`blur_verdict.py`, `blur_control_unbounded.py`).

The draft already spends this rule — line 346 refuses hMT+ specificity for
exactly this reason. Labelling a raw-only result as raw-only is weaker than
the rule the project pre-committed to, and §101.4 is withdrawn accordingly.

### §102.1 The verdict table

| contrast | raw | norm | §8 |
|---|---|---|---|
| **size\|ecc**, diet: trained − own init | −0.169, .0078 | −0.048, .0156 | **CLAIMABLE** |
| **size\|ecc**, no diet: trained − own init | +0.454, .0078 | +0.073, .0078 | **CLAIMABLE** |
| ecc\|size, trained diet − input null diet | +0.229, .0156 | +0.084, .0078 | **CLAIMABLE** |
| ecc\|size, untrained no-blur − null no-blur | +0.268, .0078 | +0.067, .0078 | **CLAIMABLE** |
| ecc\|size, no diet: trained − own init | −0.442, .0078 | −0.093, **.1484** | **REFUSED** |
| ecc\|size, untrained diet − input null diet | +0.306, .0078 | +0.048, **.1094** | **REFUSED** |
| ecc\|size, diet: trained − own init | −0.077 ns | +0.036 ns | null, see §102.3 |
| ecc\|size, trained no-diet − null no-blur | −0.173 ns | −0.026 ns | null, see §102.3 |

`950c6df` also corrects itself here and the correction matters: its earlier
"`untrained diet − input null diet` = +0.306 is the load-bearing half that
survives" is **raw-only** and fails the rule. What carries "present at
initialisation, largely not inherited from the input" on both statistics is the
**no-blur** front end (+0.268 / +0.067), not the diet one.

### §102.2 The claimable result is a double dissociation on the size term

Both `size|ecc` rows clear §8, and they run in **opposite directions**, each arm
measured against its own initialisation:

| arm | size\|ecc at init | after training | change (raw / norm) |
|---|---|---|---|
| diet | −0.187 | **−0.356** | −0.169 (.0078) / −0.048 (.0156) |
| no diet | −0.292 | **+0.163** | +0.454 (.0078) / +0.073 (.0078) |

Training with the diet makes magno preference fall *faster* with pRF size;
training without it **reverses the term's sign entirely**. Both statistics
agree in both arms — the only part of the whole pRF analysis that does.

This is stronger than §101.4's proposed headline (the no-diet sign flip alone),
because it is a dissociation rather than one arm's behaviour, and it is the
only piece of the §100 reframing that survives §8 intact.

### §102.3 Both nulls fail, and "preserved" fails in a way I did not expect

> **BOTH REVERSALS WITHDRAWN — SUPERSEDED BY §103.** The intervals below are
> **paired-t** (`_tost_from`), while the declared test is the paired exact
> Wilcoxon (`PAPER_DRAFT.md:1351`). Using a t-TOST interval to *bound* a null
> is existing practice and fine; using it to *reverse* the declared verdict is
> not, and all three affected cells straddle α = .10. The exact
> Hodges–Lehmann/Walsh intervals contain zero in **all four** cells, so "N1 is
> not a null at all" and "sits significantly below its own input null" are both
> withdrawn. Under the declared test both nulls are **unbounded** — neither
> asserted nor overturned. This does not reinstate §101.2's "essentially
> unchanged" either; that term is simply unresolved.

Neither null can be stated. §77.5 forbids choosing a SESOI after seeing the
interval and none of §77.3's pre-registered bounds covers this contrast, so
these are reported as intervals and MDEs, not as equivalence:

| null | raw diff, 90% CI | norm diff, 90% CI |
|---|---|---|
| **N1** diet "preserved" | **−0.077 [−0.146, −0.007]** | **+0.036 [+0.003, +0.068]** |
| **N2** no-diet "no more than a Gabor bank" | **−0.173 [−0.308, −0.039]** | −0.026 [−0.089, +0.037] |

**N1 is not a null at all.** The two statistics have **opposite signs** and
*both* 90% intervals exclude zero. So the diet arm's ecc|size term neither
stays put nor moves in an agreed direction — §8 refuses any claim, and
"preserved" (§101.2's replacement wording) is withdrawn along with §100.3's
"exceeds". Both formulations were wrong; the term simply cannot be
characterised from these data.

**N2 is worse than "no difference".** On raw the trained no-diet model sits
*significantly below* its own input null (−0.173, CI excludes zero), while norm
shows nothing. So "carries no more eccentricity structure than a Gabor bank" —
§101.3's "sharpest single result" — is not supportable, and the raw direction
suggests it carries *less*, which is a different and unexplained claim.
**Withdrawn.**

MDE at 80% power is **+0.232 raw / +0.110 norm** (dz = 1.16). This design
detects only large effects, so both nulls are underpowered independently of
everything above.

### §102.4 What leg (c) can say

> Training reorganises the pRF-size term in opposite directions depending on
> the diet: with the developmental diet, magno preference comes to fall more
> steeply with pRF size (−0.187 → −0.356); without it the term reverses sign
> (−0.292 → +0.163). Both the difference and normalised statistics agree in
> both arms, each measured against its own initialisation. The eccentricity
> term's fate differs between the two statistics and is therefore not claimed.

Two supporting contrasts also clear §8 and can be quoted: the diet-trained
model exceeds its own input null on eccentricity (+0.229 / +0.084), and the
untrained no-blur model exceeds its own input null (+0.268 / +0.067).

### §102.5 On the denominator rationalisation

§101.4 offered "the norm index divides by (uₘ+uₚ) and shrinks where unique
variances are small" as a reason for the attenuation. `950c6df` notes it is
testable — the norm/raw ratio should track median(uₘ+uₚ) across models and
subjects — but that running it *after* seeing which side failed cannot lift a
pre-declared rule; at best it would explain the failure. Agreed. Not run, and
the split stands as reported rather than argued away.

### §102.6 What this cost

§100's reframing is now roughly a third of its original size: the
"destroyed / preserved" pair is gone (one half refused by §8, the other half
uncharacterisable), and "no more than its own input" is gone. What replaced it
is one dissociation that clears every gate the project has. That is the trade
the §8 rule was written to force, and it worked as intended on a result I had
already written up twice.

---

## §103 §102.3's two reversals were an estimator switch — withdrawn. Both
## nulls are unbounded under the declared test, neither asserted nor overturned

`950c6df` caught that §102.3's reversals rest on an estimator the paper does
not declare. Verified: `tost_paired` → `_tost_from` builds a **paired-t**
interval (`se = d.std(ddof=1)/√n`, `t(.95, n−1)`), while
`PAPER_DRAFT.md:1351` declares "**the paired exact Wilcoxon signed-rank test,
n = 8**" and every p-value in this thread is Wilcoxon — the discrete
.0078/.0156/.1094 ladder is the signature.

### §103.1 The distinction that matters

Using a t-TOST interval to **bound** a null is this paper's existing practice
and is defensible: §2.6.2's VO entry quotes a TOST interval beside a Wilcoxon
p, and §2.6.1 openly calls phase-scrambling "a Welch interval at 2.3 degrees of
freedom". *How large an effect could the data have hidden* is a different
question from *did the declared test reject*.

What that interval cannot do is **reverse** the declared verdict. "N1 is not a
null at all" and "the trained no-diet model sits significantly below its own
input null" are both reversals, and both were produced by switching estimator
after Wilcoxon returned n.s. Back-solved, all three affected cells straddle
α = .10 (t-p ≈ .075, .077, .044 against Wilcoxon .1094).

### §103.2 The interval the declared test implies

Hodges–Lehmann with the exact Walsh-average distribution-free CI — the interval
Wilcoxon implies, exact at n = 8:

| null | statistic | HL | exact CI (**92.19%**) | Wilcoxon p | excludes 0 |
|---|---|---|---|---|---|
| N1 diet: trained − own init | raw | −0.082 | [−0.148, +0.001] | .1094 | **no** |
| N1 diet: trained − own init | norm | +0.036 | [−0.002, +0.085] | .1094 | **no** |
| N2 no-diet − input null | raw | −0.176 | [−0.343, +0.001] | .1094 | **no** |
| N2 no-diet − input null | norm | −0.029 | [−0.101, +0.046] | .4609 | **no** |

**Not a 90% interval.** Discreteness at n=8 means the nominal level is not
attainable: α = .10 admits k = 5, P(W⁺ ≤ 5) = 10/256 = .0391, so attained
coverage is 1 − 2(.0391) = **92.19%**. (Checkable by hand — the subsets of
{1..8} summing to ≤ 5 are {}, {1}, {2}, {3}, {1,2}, {4}, {1,3}, {5}, {1,4},
{2,3}: ten of 256.) The construction rounds **conservative**, so the interval
is *wider* than a 90% label implies and "all four contain zero" is a stronger
statement than the label suggests — the withdrawals hold a fortiori. `hl_ci`
now returns its attained coverage rather than carrying a nominal label, since
the draft names the .0078 floor explicitly rather than rounding it and a
nominal "90%" would be out of character in the same document.

**All four contain zero.** §102.3's reversals are withdrawn.

So under the declared test both N1 and N2 are **unbounded nulls** — neither
asserted nor overturned. §102.3's "N1 is not a null at all" is wrong: it is a
null, whose two point estimates happen to have opposite signs and are both
consistent with zero. §102.3's "carries *less* than a Gabor bank" is likewise
withdrawn.

`950c6df` is right that this does **not** reinstate §101.2's "essentially
unchanged": that wording was withdrawn on evidence which does not support the
withdrawal, but it still needs a bound before it can be asserted. The ecc|size
term is **unresolved**, which is where §8 leaves it regardless.

### §103.3 The implementation bug the verification caught

The first version of `hl_ci` took `walsh[k+1]` where the exact interval is
`[W_(k+1), W_(M−k)]` = `walsh[k] .. walsh[m−1−k]` zero-indexed. That off-by-one
narrows the interval, and it made **N1 raw exclude zero** — reproducing
§102.3's false reversal through a second, independent route.

It was caught by an assertion written before the result was looked at: the HL
interval must exclude zero **iff** the two-sided Wilcoxon p < α, checked on 300
random samples at α ∈ {.05, .10}. That assertion is the only reason the bug did
not survive into this section, and it is the §82 lesson again — an assertion
that a quantity *equals* a separately-derived value cannot pass for the wrong
reason. The identity is exact at n=8 rather than approximate: the achievable
two-sided ladder is .0078/.0156/.0234/.0391/.0547/.0781/.1094, so "p < .10"
means "p ≤ .0781" means "W⁺ ≤ 5" means "the k=5 interval excludes zero".

**The lesson from the convergence, which is the more useful half.** Two
unrelated errors — a t-interval substituted for the declared test, then an
off-by-one in the exact interval built to replace it — produced the *same*
false reversal. When a result reappears after the thing that caused it has been
fixed, that is evidence the analysis is **fragile at that point**, not evidence
the result is robust. The natural reading is the opposite one, which is what
makes it worth writing down.

### §103.4 A durable fix in the module

`equivalence_tests.py` was built for bounding nulls, where parametric is fine,
and it silently carries a different test into contexts where the verdict is at
stake. A comment now says so at the top of `_tost_from`, since the next reader
will reach for it exactly as I did.

### §103.5 Unaffected

The double dissociation and the two supporting contrasts rest on Wilcoxon
p-values under both statistics with **no interval doing any work**:

| contrast | raw | norm |
|---|---|---|
| size\|ecc, diet: trained − own init | −0.169, .0078 | −0.048, .0156 |
| size\|ecc, no diet: trained − own init | +0.454, .0078 | +0.073, .0078 |
| ecc\|size, trained diet − input null diet | +0.229, .0156 | +0.084, .0078 |
| ecc\|size, untrained no-blur − null no-blur | +0.268, .0078 | +0.067, .0078 |

§102.4's leg (c) wording stands as written, including "the eccentricity term's
fate differs between statistics and is therefore not claimed".

### §103.6 Note on where the shrinkage came from

Across §100–§103 this result went from a four-part reframing to one
dissociation. Every reduction came from applying standards the project had
already written down — §8.1's both-statistics rule, §77.5's bound-fixing rule,
§4.7's declared test — and none from new data. Three of the four retractions
were of claims I had already committed.


## §104 PRE-REGISTERED — does the readout span survive a nonlinear head?

Written and committed **before** any feature is extracted or any head is fitted.
Prompted by the NMI referee simulation (`docs/NMI_REVIEW_2026-09-10.md`, major
comment 2) and scheduled as T1 of `docs/REVISION_PLAN_2026-09-10.md`. It is the
only open item that crosses the Minor-revision band on its own, and it is the
one that can take the title away.

### 104.1 The question, and why the current answer is not enough

§2.3 sweeps the fusion weight `w` over frozen representations and moves the
cue-conflict shape bias from 0.373 to 0.913. §2.3 also establishes, correctly,
that this is an **identity**: the classifier is affine and the mixture weights
sum to one, so `clf(w·y₁+(1−w)·y₂) = w·clf(y₁)+(1−w)·clf(y₂)` to 1.1 × 10⁻⁶,
and the sweep contains no information beyond `w` = 0 and `w` = 1.

The paper's title — *shape-biased behaviour does not require shape-biased
representations* — is a claim about readouts. If the free-parameter status of
the rate follows from affineness, the claim is established for **affine**
readouts and for no others, which covers neither a nonlinear network head nor
any plausible biological readout. That is the gap.

### 104.2 What is and is not blind

**Blind:** every number this test produces. No nonlinear head has been fitted;
no penultimate features have been extracted for this purpose. The features do
not currently exist on disk — `tmp_analysis/perimage_logits.py` computes `F1`
and `F2` internally and saves only the logits — so there is no cached artefact
anyone could have peeked at.

**Not blind:** the affine span (0.373–0.913) is published in §2.3 and I have
read it. It is the *baseline* this test is measured against, not an outcome of
it, and the thresholds below are fractions of it fixed now.

### 104.3 Design

Three candidates were considered; the one registered here is **A**.

- **A (registered).** Re-fit a small nonlinear head — `Linear(512→h) → ReLU →
  Linear(h→112)` — on `w·y₁ + (1−w)·y₂` at each of the same 21 values of `w`,
  independently per seed, on MiniEcoset training features. Score cue-conflict
  shape bias on the 1,200-image set at each `w`.
- B (fallback, only if A's fits prove unstable). One head trained on `[y₁,y₂]`,
  gate swept at test time. Weaker: the head is not matched to each `w`.
- C (rejected). Frozen classifier with an inserted nonlinearity. Ad hoc; it
  tests the nonlinearity, not the readout.

Fixed now: `h = 512`; the same optimiser family and epoch budget for every `w`
so no cell gets a larger search than another (the tuning-fairness item the DOME
audit has marked Partial twice); 3 seeds; both checkpoint policies; features
extracted once and reused across all `w`, so every cell sees identical inputs.

**The representations stay frozen.** Only the head is re-fitted. This preserves
the property that makes §2.3's demonstration cleaner than a probe-decoder
design, and it is the difference between B and A that matters.

### 104.4 The decision rule

Let **S_nl** be the span of cue-conflict shape bias over the 21 `w`, taken as
the median span across seeds, and **S_aff = 0.913 − 0.373 = 0.540**.

| outcome | rule | what the paper then says |
|---|---|---|
| **survives** | S_nl ≥ 0.80 · S_aff (≥ 0.432) | The claim is not an artefact of affineness. §2.3 and the title stand, strengthened; state that a nonlinear fitted readout reproduces the span. |
| **partial** | 0.50 · S_aff ≤ S_nl < 0.80 · S_aff | Report the measured span and claim the weaker, quantified version. No re-running to taste, no third design. |
| **does not survive** | S_nl < 0.50 · S_aff (< 0.270) | **The title takes a qualifier** — "in affine readouts of these components" or equivalent — and §2.3 states the boundary as a result, not a caveat. |

**Accuracy is a gate, not a footnote.** Report MiniEcoset validation accuracy at
every `w` alongside the shape bias. If a span appears *only* where head accuracy
has collapsed, it is not a readout result and must not be counted as one: the
registered treatment is to report the span restricted to `w` whose accuracy is
within 5 points of the best cell, and to report the unrestricted span beside it.

### 104.5 What this cannot do, stated now

- It cannot establish that a *biological* readout behaves this way. It widens
  the claim from affine to affine-plus-one-nonlinear family; it does not close
  the class.
- A re-fitted head means the readout is *learned* while the representation is
  frozen. That is a different — and, if it survives, stronger — claim than
  §2.3's, and it must be described as such rather than folded into the existing
  sentence.
- It does not address selection (§2.2(b)) or capacity (Limitation 16). Those are
  separate items with separate owners in the revision plan.

### 104.6 The commitment

The outcome is reported whichever way it lands, in the main text, with the span
and the accuracy profile. This project has reported two failed pre-registrations
and six failed regional hypotheses in the main text already; this one is
registered under the same terms. A result that qualifies the title is not a
reason to leave the test unrun — it is the reason to run it before a referee
does.

### 104.7 Amendment, recorded before any result exists

Added while the feature extraction was still running (job 26459572), and
committed before the first sweep was fitted, so the ordering is checkable in the
history rather than asserted.

**A refitted *linear* head is fitted at every `w` alongside the nonlinear one.**
§104.3 registered design A as "re-fit a small nonlinear head". That confounds two
manipulations: the head becomes nonlinear *and* it becomes refitted. If the span
changed, design A as registered could not say which did it. Three heads are now
scored at each `w`:

| head | what it is | isolates |
|---|---|---|
| `affine` | the trained classifier's own Linear, unchanged | §2.3's published baseline |
| `linear` | `Linear(512,112)` refitted at this `w` | the effect of **refitting** |
| `mlp` | `Linear(512,512) → ReLU → Linear(512,112)` refitted at this `w` | the effect of **nonlinearity**, given the linear control |

This is strictly an addition: it removes an interpretation the registered design
would have left open, and it cannot make the test easier to pass. The decision
rule of §104.4 is unchanged and is applied to the **`mlp`** span, which is the
one §104.1 asks about. The `linear` span is reported beside it.

Fixed now, identical for every `w` and every head: Adam at lr 1e-3, batch 1024,
20 epochs, hidden width 512, and the same initialisation seed at every `w` so a
difference between weights is not a difference between draws.

### 104.8 OUTCOME — the span survives, and nonlinearity is not what carries it

Jobs 26459572 (extraction) and 26459577 (fits), both complete. Scored against
§104.4's rule exactly as fixed, with no denominator or threshold changed after
the fact.

| quantity | value | rule | verdict |
|---|---|---|---|
| nonlinear median span, best checkpoint | **0.4655** | ≥ 0.432 survives | **SURVIVES** (86.2% of the registered 0.540) |
| same, against the matched per-run affine span (0.5084) | 90.2% | — | passes on the fairer denominator too |
| nonlinear ÷ refitted-linear span | **1.01** | — | nonlinearity changes nothing |
| accuracy-gated nonlinear span | 0.283 (52.4%) | gate: is the span *only* where accuracy collapsed? | **no** — a substantial span survives among accurate readouts |

**What this settles.** §2.3's claim was established for affine readouts only, and
a referee was right to say so. It now holds for a fitted two-layer nonlinear
readout as well, at 86% of the affine span. The §104.7 control is what makes the
result interpretable: the refitted *linear* head gives the same span as the
nonlinear one to within 1%, so the span is not a property of the head's
functional form. It is a property of having two oppositely biased components and
a readout over them — which is the claim the title makes, now established
outside the case where it is an identity.

**What it does not settle**, restating §104.5 rather than quietly dropping it:
this widens the class from affine to affine-plus-one-fitted-nonlinear family. It
does not close the class, and it says nothing about a biological readout.

**The honest qualification, reported in the main text and not only here.**
Restricting to readouts within 5 points of peak validation accuracy roughly
halves the span (0.283; shape bias 0.421 to 0.760 over `w` = 0 to 0.75). The
gate was written to catch a span that exists only where the head has collapsed;
that is not what happened, but the extreme rates near `w` = 1 do cost accuracy.
That is §2.7's cost accounting seen from the readout side.

**Registered and unchanged.** The threshold (80%), the floor (50%), the gate
(±5 accuracy points), the statistic (median span across seeds at the best
checkpoint) and the head (512-wide, ReLU) were all fixed in §104.3–104.4 before
extraction, and §104.7's linear control was committed while extraction was still
running. Nothing was tuned to the outcome.

---

## §105 The parameter-matched control LANDED and it KILLS §2.3's "the split is
## doing the work". Capacity, not the split, accounts for the shape-bias gap.

*Jobs 26444916 (training, 3 seeds, COMPLETED 1-00:44 to 1-02:05) and 26459565
(cue-conflict, 6 tasks, COMPLETED ~4.5 min each). Analysis protocol fixed in
§64.6 before the runs existed and not renegotiated after seeing them.*

### §105.1 The comparator side reproduces the published numbers exactly

Before reading the new arm: the same collection script, run over the two arms
already in the paper, returns the draft's own figures.

| quantity | draft | recomputed |
|---|---|---|
| single-stream shape bias, best saved | 0.567 | 0.5672 |
| two-stream shape bias, best saved | 0.655 | 0.6547 |
| single-stream accuracy | 0.599 | 0.5990 |
| two-stream accuracy | 0.556 | 0.5562 |

So the matched arm is being compared against the published baselines, not
against a re-derivation of them.

### §105.2 Shape bias: the separation that carried the claim does not survive

Three seeds per arm, both checkpoint policies, ranges rather than p-values
(§64.6: 3v3 floors at .10, so complete separation is the declared rule).

**Best saved checkpoint**

| arm | params | mean | range | vs two-stream |
|---|---|---|---|---|
| two-stream `dkl_bcs` | ~22.4M | 0.6547 | [0.5942, 0.7177] | — |
| 1-stream `single` r18 | 11.2M | 0.5672 | [0.5414, 0.5899] | **COMPLETE separation** |
| 1-stream `single34` r34 | 21.3M | 0.6255 | [0.6043, 0.6524] | **overlap** |

**Final checkpoint**

| arm | params | mean | range | vs two-stream |
|---|---|---|---|---|
| two-stream `dkl_bcs` | ~22.4M | 0.6749 | [0.6464, 0.6922] | — |
| 1-stream `single` r18 | 11.2M | 0.5571 | [0.5146, 0.5865] | **COMPLETE separation** |
| 1-stream `single34` r34 | 21.3M | 0.6255 | [0.6043, 0.6524] | **overlap** |

The gap §2.3 quotes as **0.088** (0.655 − 0.567) falls to **0.029** at the best
saved checkpoint and **0.049** at the final one, with overlapping seed ranges
at both. Under the rule declared in advance, that is **no separation and
therefore no claim**.

And the mechanism of the confound is directly visible rather than merely
hypothesised: **r18 single vs r34 single is COMPLETE separation at both
policies** ([0.5414, 0.5899] vs [0.6043, 0.6524]). Identical input, identical
diet, identical schedule, one trunk in both cases — capacity alone raises
shape bias by roughly the amount the split was credited with.

### §105.3 What dies

§2.3's paragraph headed **"The split is doing the work: without it the
behaviour is worse."** Specifically:

- "seed-level separation complete under both checkpoint policies" — true of the
  r18 control, **false** of the parameter-matched one.
- "The split buys shape bias" — not supported at matched capacity.
- The ladder sentence "having two components to integrate is still worth 0.088
  over having one" — the 0.088 is a capacity difference as much as a split
  difference, and shrinks to 0.029 when capacity is held.

This is Limitation 16's own hedge firing. It read: *"The confound runs against
the accuracy result and in favour of the shape-bias result."* Both halves are
now settled, and the shape-bias half is fatal.

**Do not overcorrect into the opposite claim.** The matched arm does not show
the split buys *nothing*. Overlapping ranges at n=3 are an absence of
separation, not evidence of equality, and reading them as "no effect" would be
the §99 significant/not-significant fallacy in its other direction. The honest
statement is that this design cannot attribute the shape-bias difference to the
split.

### §105.4 What strengthens

The accuracy half moves the other way, and in the paper's favour:

| arm | best-saved val_top1 | range |
|---|---|---|
| two-stream | 0.5562 | [0.5516, 0.5639] |
| 1-stream r18 | 0.5990 | [0.5889, 0.6075] |
| 1-stream r34 | 0.6232 | [0.6157, 0.6338] |

All three completely separated. At matched parameters the single stream's
accuracy advantage **widens** (0.6232 vs 0.5562), so "the split pays for it in
accuracy" is not a capacity artefact — it is real and larger than the draft
reports. §2.3 can keep that half and should strengthen it.

Late-training stability also favours the matched arm: sd(val_top1) over the
last 20 epochs is [0.0152, 0.0406] for r34 single against [0.0428, 0.0727] for
two-stream, complete separation. For all three r34 seeds the best saved
checkpoint IS epoch 149, so §71.4's checkpoint-policy concern is moot for this
arm and both policies score identical weights.

### §105.5 A complication that must travel with the numbers

§64.6 required the decision rate as a **count over a named subset** precisely
because its denominator moves. It moved:

| arm | decided / 1200, best saved |
|---|---|
| two-stream | 502.0 [483, 520] |
| 1-stream r18 | 501.0 [491, 517] |
| 1-stream r34 | 536.7 [515, 556] |

§2.3's claim that decision rates match "to within 0.003" holds for the r18
control (501.0 vs 502.0) and **fails for the matched one**: 536.7 vs 502.0 is
0.029 of 1200, ten times larger. The r34 arm resolves more images to a cue, so
its shape bias is computed over a partly different and larger subset. This
weakens the comparison in an undetermined direction and must be disclosed
alongside it; it does not rescue the claim.

### §105.6 What is untouched

The paper's central claim never rested on this comparison. "Shape-biased
behaviour does not require shape-biased representations" is an existence proof
about a readout over oppositely-biased components:

- §2.1's opposite stream biases — untouched.
- §2.2's ventral gradient — untouched.
- §2.3's w-sweep spanning 0.40–0.90 over **frozen** components — untouched, and
  it is the section's real load-bearing result.
- §2.6's three properties that do not travel with shape bias — untouched.

What is lost is a subsidiary causal claim about the architecture, in a section
whose thesis is that the *readout* rather than the components sets the
behavioural rate. Losing it costs the paper an argument it did not need.

### §105.7 A candidate reframing, flagged not asserted

The r18→r34 result is that shape bias rises with trunk capacity at fixed input
and fixed diet. That is another instance of the section's actual thesis —
shape bias moving under a manipulation that is not about shape. It is *not* as
clean as the w-sweep, because capacity changes the representation as well as
the readout, so it cannot be presented as a frozen-component result. Whether it
belongs in §2.3 as a second demonstration or in Extended Data as a control is a
judgement for whoever owns the section; recorded here so the option is visible.

### §105.8 Consequences for the draft (peer-owned; not edited by me)

1. §2.3's "The split is doing the work" paragraph needs rewriting, not patching.
2. The ladder quote (magno 0.903 > two-stream 0.655 > single 0.567) needs the
   matched arm inserted at 0.626 and the 0.088 sentence removed.
3. Limitation 16 must change from "has not been run" to the outcome, and it is
   no longer a limitation about a missing control but a disclosure that a
   control removed a claim.
4. "Claims deliberately not made" gains an entry: *that the two-stream split
   causes the shape-bias advantage over one stream*.
5. §105.5's decision-count divergence needs a line wherever the matched arm's
   figure appears.

### §105.9 Method note

`tmp_analysis/cue_conflict_single.py` now reads `like_resnet` from the
checkpoint rather than hardcoding it (c44aae4). §97 bound the architecture to
the CONDITION, so it rides inside every checkpoint's saved config; the r18
blobs carry no key and default correctly, which is why §105.1 reproduces. The
file's pre-existing strict load is an independent check — a wrong architecture
cannot yield 0 missing and 0 unexpected keys — and it confirmed 122 tensors /
11,230,832 params for r18 and 218 / 21,338,992 for r34, the latter matching the
count the training log stamped at startup by a different route.


---

## §106 PRE-REGISTERED — fit `w` to the ten Geirhos observers. Written BEFORE
## any fit is computed, because the grid it would run on is already on disk.

*Authorised by the user 2026-09-10. This section is timestamped by its commit;
the fit is not run until it is committed.*

### §106.1 Why this is worth running, given what §2.3 and §2.6.2 already have

The draft already reports (a) the rate sweep, max **0.913** over 21 weights × 3
seeds × 2 policies, below the lowest individual observer (0.919) and 0.045 below
the human mean; and (b) error consistency recomputed at all 21 weights for 9
runs against the same 10 observers, never exceeding 18% of the human–human
ceiling. Neither is a fit. Limitation 14 says so in as many words:

> "No w was fitted to human data and nothing here estimates the weighting a
> human readout would use."

That is the gap. A sweep reports a *range*; a fit reports an **estimate**, and
an estimate can be pinned at a boundary, which is a qualitatively different and
stronger statement than "the range does not reach". This section converts
Limitation 14 from an admission into a measurement, and Limitation 7's
"directional only" into a bound with an interval.

**This is a bound, not a new lead.** It sharpens §2.6.2 and rewrites two
limitations. It does not get a section of its own and must not be written up as
a negative result (Research Output Standard: no negative/audit papers).

### §106.2 The estimand

For observer *i*, the readout weight that best reproduces that observer:

    E1 (rate fit)   w_i^rate  = argmin_w | sb(w) - rate_i |
    E2 (trial fit)  w_i^trial = argmax_w  kappa( model_w , observer_i )

Both are reported for all ten observers, per run. **§8's two-statistics rule
applies: a claim is made only where E1 and E2 agree.** They answer different
questions on purpose — E1 asks what weighting reproduces a person's *rate*, E2
what weighting best predicts their *trials* — and the contrast between them is
the point, not a nuisance.

kappa is the instrument already used in §2.6.2 and validated in this codebase
against the published human reference (0.958) and the human–human ceiling
(0.234); both assertions are kept in the new script. It is not swapped for a
likelihood, because the model's decision at fixed w is deterministic and a
likelihood would require inventing a noise model that nothing here constrains.
That limitation is stated rather than papered over.

### §106.3 Boundary handling, declared in advance

If argmin/argmax lands at w = 0 or w = 1, the fit is **pinned at the boundary**
and is reported as *"no interior estimate exists"* — NOT as an estimate of 1.0.
A boundary-pinned fit means the observer lies outside the family, which is a
result about the family, and writing it as a number would disguise that.

### §106.4 Resolution, and why the fit costs almost nothing

The classifier is affine and the weights sum to one, so

    clf(w*F1 + (1-w)*F2) = w*clf(F1) + (1-w)*clf(F2)

**exactly** (§2.3 measures the deviation at 1.1e-6). So the two endpoint logit
matrices (1200 x 112 each, ~0.5 MB) determine every w. The new script caches
those once per run and evaluates a grid of step **0.001** — 1001 points against
the cached grid's 21 — at no meaningful cost. No GPU, no retraining.

Uncertainty on w^trial: **item bootstrap over the 1,200 cue-conflict images,
B = 1000**, matching the bootstrap already used by
`src.analysis.behavior.cue_conflict`. Reported as a percentile interval on the
fitted w, and separately as the fraction of bootstrap replicates whose argmax is
at a boundary.

### §106.5 Arms and aggregation

Nine runs, as §2.6.2: diet arm 3 seeds × 2 checkpoint policies, no-diet arm 3
seeds (whose best saved checkpoint IS epoch 150). Per-observer fits are reported
per run with the **range across seeds**, and complete separation rather than a
p-value, per §64.6. No stream statistics.

### §106.6 The decision-denominator complication, carried forward from §105.5

The model does not decide every image, and the observers do not either. kappa
already masks to images where both decided. The **rate** comparison does not:
sb(w) is over the model's decided subset and rate_i over the observer's. Both
counts are reported alongside every fit, as counts over a named subset, exactly
as §64.6 required and §105.5 had to invoke.

### §106.7 Predictions, recorded so they cannot be fitted afterwards

1. **E1 is boundary-pinned for 10/10 observers on the diet arm.** The family's
   maximum rate is 0.913 and the lowest observer is 0.919, so no interior
   solution can exist unless the finer grid pushes the maximum above 0.919 —
   which the 0.05 grid cannot rule out and the 0.001 grid will settle.
2. **E2 is interior, and w^trial != w^rate.** If both hold, the sharper
   statement is available: *the weighting you would infer from a person's rate
   is not the weighting that best predicts their choices* — a second form of
   §2.6.2's dissociation, obtained within one network rather than across a model
   zoo.
3. **kappa at w^trial stays below 25% of the human–human ceiling for every
   observer.** The existing sweep never exceeds 18%; the finer grid may improve
   on that slightly and should not improve on it much.

### §106.8 What would count as a POSITIVE result, declared now

If any observer's fitted w yields kappa approaching the human–human ceiling,
the readout family contains a human-like decision-maker and that is a positive
finding which **must be led with**, not buried — it would materially change
§2.6.2's conclusion and the Discussion's "shape bias is a readout property"
framing. Similarly, if E1 returns an interior estimate for any observer, the
family does reach that person's rate and Limitation 7's "below the lowest
individual observer" needs restating. Recorded here so the analysis is not
rigged toward the bound I expect.

### §106.9 What this cannot do

It cannot estimate "the" human readout weight: it fits a scalar in OUR
architecture over OUR two components to a human's choices, so a fitted w is a
statement about this family's ability to imitate an observer, not about human
visual cortex. Limitation 14's replacement text must say that. It also inherits
every scope limit already on the human comparison (Limitation 15, and the
ten-observer sample).

### §106.10 AMENDMENT — §106.7's prediction 3 FAILED on the smoke run. Two
### controls added, and they are POST-HOC.

*Written after seeing one run (dkl_bcs_seed0, best-saved checkpoint) at the
coarse 0.05 grid. This is not a pre-registration and must not be read as one.
It is recorded here, before the nine-run array is submitted, so that the
sequence — prediction, falsification, added control — is on the record in the
order it happened.*

**What happened.** The smoke run passed every assertion: human anchors 0.9576
and 0.2335, the vectorised statistics agree with the validated scalar ones to
0.00e+00 at all 21 coarse points, the affine identity holds at 3.6e-05.
Predictions 1 and 2 came out as declared — E1 unattainable for 10/10 observers,
E2 interior for 6/10 and spread from w = 0.30 to w = 1.00 across observers.
**Prediction 3 did not.** It declared kappa at the fitted w would stay below 25%
of the human–human ceiling for every observer. The best observer (subject-03)
reached **+0.0893 = 38.2%**, and three others exceeded 23%.

**Why this is not yet a contradiction of §2.6.2.** The published "never exceeds
18% of the ceiling" is the *mean over the ten observers* at each w. 38.2% is a
*per-observer maximum over 1001 weights*, then the largest of ten such maxima.
Those are different statistics and the second is selected twice over. My
prediction was simply wrong to compare them as if they were the same quantity;
that error is mine and is why the number came in high.

**But it cannot be waved away either**, because §106.8 pre-declared that a
fitted w approaching the ceiling is a positive finding that must be led with.
38.2% is not "approaching" — it is still 62% short, and the human–human pairs
average 0.234 with sd 0.095 while the best model–observer pair reaches 0.0893.
The honest position is that the number is inflated by selection by an unknown
amount, and an unknown amount is not a result. Two controls settle it.

**Control A — split-half cross-validation of the fitted weight.** The direct
test of whether a per-observer fitted w is real or is fitting 1,200 trials of
noise. Items are split at random in half; w is fitted by argmax kappa on half A
and kappa is evaluated at that w on held-out half B. **R = 200 random splits**,
reporting the mean held-out kappa against two references: the in-sample kappa at
the all-data fitted w (the inflated number), and the mean kappa over all w with
no fitting at all (the no-selection floor). If held-out kappa collapses to the
no-fitting floor, the 38.2% is selection and the family contains no
observer-specific readout.

**Control B — permutation null for the selected maximum.** Each observer's
trial vector is shuffled across items **P = 1000** times, and the whole
best-over-1001-weights statistic is recomputed each time. This gives the
distribution of the selected maximum under no model–observer correspondence,
and the observed maximum is reported as a position in it. This measures exactly
the inflation that argmax-over-w buys for free.

Both are cheap: kappa's four sufficient statistics are matmuls of the model's
fixed decision matrix against the observer vectors, so P permutations or R
splits are one matmul each, not a loop.

**Declared now, before the controls run:** if held-out kappa (Control A) stays
near the in-sample value and the observed maximum sits outside the permutation
null (Control B), then the fitted readout weight carries real observer-specific
structure and §2.6.2's "rate-matching is not decision-matching" needs
qualifying — some readout of these components *does* track an individual
observer, which would be a positive and would be led with per §106.8. If
held-out kappa collapses toward the no-fitting floor, the bound stands and
strengthens: not only does the family fail to reach any observer's rate, its
apparent trial-level agreement does not survive cross-validation either.

**Prediction 3 is withdrawn, not revised.** It compared two different
statistics and its replacement would be a number chosen after seeing the data.
Controls A and B are decision rules, not predictions, and that is deliberate.
