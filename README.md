# Shape bias is a readout property: a developmental two-stream model, frozen networks and human cortex

Code and analysis for a two-stream network that receives approximations of the
magnocellular and parvocellular signals reaching cortex, and the encoding,
behavioural, readout and single-trunk generality analyses built on it.

The claim: a behavioural shape-bias rate is a property of a *readout over
components*, not a report of any one component. The luminance stream ends
training shape-biased and the chromatic stream texture-biased; fitted to eight
subjects of the Natural Scenes Dataset it is the **texture**-biased stream whose
unique variance carries extrastriate cortex; behaviour, reading out from both, is
shape-biased; and in nine frozen networks with no stream split at all, a gain on
their own shape-leaning units moves the rate and reorders the field's ranking.

Manuscript: `docs/PAPER_v2.md` (the Extended Data tables are embedded between the
`ED_TABLES` markers; figure legends are `results/figures_v2/*_caption.md`; the
compiled PDF is `docs/pdf/PAPER_v2.pdf`, built by
`docs/pdf/build_manuscript_pdf.py --profile v2`). Methods §4.1–§4.8 name, by
path, the module that produces every reported number. The pre-specification of
the generality test, with its recorded outcomes, is
`docs/PLAN_GENERALITY_2026-09-14.md`. `docs/PAPER_DRAFT.md` is the superseded
earlier draft, kept for provenance.

> **Status:** manuscript in preparation. This repository is the working tree, not
> an archived release. A DOI-archived version carrying the derived result tables
> and the 21 trained checkpoints (≈ 17 GB) is committed to at acceptance — see
> *Data availability* and *Code availability* in the manuscript.

---

## Verify the reported numbers (no dataset download, no GPU)

The fastest real check on this repository. `verify_new_claims.py` re-derives
numbers the manuscript states directly from the committed result files and exits
non-zero if any of them has drifted:

```bash
pip install -r requirements.txt
PYTHONPATH=.:tmp_analysis python tmp_analysis/verify_new_claims.py
```

Expected final line:

```
FAILURES: none
```

637 checks, about 95 seconds and 1.5 GB of memory on four CPU cores. If it takes
tens of minutes instead, the cost is almost certainly not the arithmetic: one
check imports `shape_bias`, and that module chain imports `torch` at module
level, so a slow or contended filesystem under the interpreter turns a pure-numpy
suite into a full deep-learning cold start. Put the environment on a local or
otherwise fast filesystem before concluding anything about the analysis.

It checks, among others: the §2.3 untrained cortical anchor (−0.544 normalised,
−0.120 raw, 8/8 subjects at the exact Wilcoxon floor p = .0078); that
`results/encoding/` and `results/encoding_cc/` are numerically identical across
the 40 files they share (max |Δr²| = 0.000e+00); the §2.5 out-of-distribution
relative-accuracy AUCs and per-image human-correspondence percentages; that no
effect size exceeds the |dz| = 6.60 the manuscript calls its largest; and, from
`results/figures_v2/generality_behaviour.json` and `generality_cortex.json`, the
Results 2.4 / Fig. 6 / Extended Data Fig. 6 / Extended Data Tables 13–16 numbers of the single-trunk and model-zoo
generality test (spans, monotonicity, the accuracy-gated re-ordering of the
ImageNet networks, and the cortical contrasts of the three networks fitted to
NSD).

On a cluster node, also point `MPLCONFIGDIR` at a writable path.

**Reproduce a headline number from scratch (CPU, minutes).** The script a referee
can actually run:

```bash
PYTHONPATH=. python notebooks/readout_gain_torchvision.py        # add --model tv:<arch> for any torchvision network
```

It downloads the Geirhos et al. (2019) cue-conflict stimuli from the authors'
repository, runs torchvision ResNet-50 on CPU, and prints the trained rate
(0.222, 814/1,200 images decided), the held-out readout curve under one gain on
the network's own shape-predictive subspace, its span and ρ(w, rate) — each
beside the committed value from `results/figures_v2/subspace_behaviour.json`
(Fig. 6b, Extended Data Table 13). With `IMAGENET_VAL` pointing at the
ILSVRC-2012 validation images it also runs the 5-point accuracy gate. About
five minutes on a laptop after a 200 MB download.

## Repository layout

```
src/training/standalone_trainer.py              the trainer that produced the reported models
src/models/pytorch_models.py                    TwoStreamNetwork (two ResNet-18 trunks + joint head)
src/analysis/encoding/dkl_frontend.py           RGB->DKL, and the blur + contrast-sensitivity front end
src/analysis/encoding/run_encoding.py           banded-ridge voxelwise encoding on NSD
src/analysis/encoding/readout_w_cortex.py       cortex-optimal readout weight (ED Fig. 5, ED Table 12)
src/analysis/behavior/cue_conflict.py           Geirhos cue-conflict scoring
src/analysis/behavior/stream_shape_texture.py   per-stream linear probe
src/analysis/generality/                        readout re-weighting without a stream split and the model zoo
                                                (Results 2.4, Fig. 6, ED Tables 13-16): models.py (frozen-network
                                                loader: our runs, any torchvision classifier, the Geirhos SIN
                                                ResNet-50s), extract.py, subspace_factor.py, unit_factor.py,
                                                gain_accuracy.py, refit_sum.py, cortex_bands.py, probe_bias.py,
                                                report.py (the one-command report), scoring.py
notebooks/readout_gain_torchvision.py           referee script: the Fig. 6b ResNet-50 curve from a fresh download, CPU, minutes
docs/reproducibility/mpk-verify.def             Apptainer definition of the pinned CPU environment that runs the verification
environment.lock.yml, requirements.lock.txt     the complete environment behind the numbers (conda export, pip freeze)
tmp_analysis/                                   the analysis entry points named in Methods §4.1-§4.7
tmp_analysis/figs_v2/                           figure, aggregator and Extended Data table scripts of the manuscript
slurm/                                          the submission script each run used
results/                                        committed result files behind the tables
docs/                                           manuscript, plan, reporting summary, reviews, PDF build
```

`tmp_analysis/` holds analysis entry points despite the name; they are the
scripts the Methods cite, and they are version-controlled deliberately.

## Reproducing from scratch

Four stages, in order. Stages 2–4 need the datasets listed below.
Submission scripts live in `slurm/`; adapt the directives to your scheduler.

| stage | script | resources as submitted |
|---|---|---|
| 1. train | `slurm/train_array.sh` | GPU, 22G RAM, 150 epochs, batch 256, 3 seeds per condition |
| 2. encode | `slurm/encode_subjects.sh <model_tag>` | **CPU only**, ~160G RAM per array task, 8 subjects |
| 3. analyse | `tmp_analysis/*.py`, `tmp_analysis/figs_v2/*.py` | CPU, modest memory |
| 4. readout re-weighting (Results 2.4) | `slurm/generality_{cc,nsd,mini,inval,cortex,rules}.sh` (unit rule), `slurm/subspace_{cc,refit,nsd,cortex}.sh` (subspace rule), `slurm/zoo_cc.sh` (model zoo), then `tmp_analysis/figs_v2/{generality,subspace,zoo}_behaviour.py` and `generality_cortex.py` | **CPU only**, 8–48G per array task; ≈ 1,600 allocated core-hours for the unit-rule pass, ≈ 1,500 for the subspace pass, ≈ 50 for the zoo |

Training is the only stage that needs a GPU. Everything else — feature
extraction, ridge fits, statistics, plotting — runs on CPU.

**A note on the encoding footprint.** The stage-2 jobs that produced the
committed results were submitted at ~160G per task, which was never the size of
the data (the stream features are 15,360-d, about 0.6 GB per stream, and the PCA
takes 25 s). The peak is himalaya's group-ridge random search, which forms an
(n_alphas x n_test x n_train) array for every gamma sample and CV fold — about
30 GB in float64 with all 21 alphas at once, whatever the voxel chunk. Passing
`n_alphas_batch = 7` batches that search over alphas and changes no estimate:
the equivalence probe (`slurm/encode_equiv_probe.sh`, then
`tmp_analysis/equiv_check.py`) refits diet seed 0, subject 01 and reproduces the
stored fit at voxelwise r = 1.00000 on both unique variances, largest per-ROI
index difference 0.0000. The same fits then peak at 32.6 GB measured and are
submitted at 34G — see `slurm/encode_cc_controls.sh`, which is the path to
reproduce stage 2 on, and which produced the three-seed swap and `parvo_highsf`
arms of the manuscript. Several scripts in `slurm/`
carry historical resource requests from the runs that produced the committed
results and are deliberately left as they were rather than rewritten, so check
the directives before submitting anything. Stage 4's banded fits use
`--ridge-batched` settings that stay under 48G.

Individual analysis entry points are argument-driven and self-documenting:

```bash
PYTHONPATH=. python src/analysis/behavior/cue_conflict.py --help
PYTHONPATH=.:tmp_analysis python tmp_analysis/fusion_sweep.py --help
PYTHONPATH=.:tmp_analysis python tmp_analysis/sweep_error_consistency.py --help
PYTHONPATH=. python -m src.analysis.generality.unit_factor --help
```

### Report shape bias for your own network

```bash
PYTHONPATH=. python -m src.analysis.generality.report --model tv:resnet50
```

prints the three numbers a shape-bias claim should carry (Methods 4.3, 4.8;
`docs/PLAN_ZOO_2026-09-15.md`): the cue-conflict rate at the trained readout;
the span of rates the same frozen features reach under one gain on the network's
own shape-predictive subspace within 5 points of the trained top-1 (the
unit-level median split beside it); and the probe-based representational bias
(shape minus texture accuracy of a linear probe). Any `tv:<arch>[:<weights>]`
torchvision classifier works — the loader takes the input of the network's last
linear layer and checks it reproduces the logits. Stages are cached under
`--work-dir`, so a re-run is free; needs the cue-conflict stimuli
(`CUE_CONFLICT_DIR`, or run the referee script above once to download them) and,
for the gate, `IMAGENET_VAL`.

## Data

None of it is redistributed here.

| dataset | where | terms |
|---|---|---|
| Natural Scenes Dataset | `naturalscenesdataset.org` | own data-use agreement, requires registration. We use the `betas_fithrf_GLMdenoise_RR` (b3) preparation and the `nsdgeneral` mask |
| Geirhos cue-conflict stimuli and 10 human observers | `github.com/rgeirhos/texture-vs-shape`, under `raw-data/style-transfer-512-nomask-experiment/` | CC BY 4.0 |
| MiniEcoset | `osf.io/msna2` | see the OSF record |
| ILSVRC-2012 validation set (accuracy gate of the ImageNet networks, Fig. 6a) | `image-net.org` | own terms; a stratified 10-per-class subset is used, indices in `src/analysis/generality/imagenet_val_labels.npz` |

The torchvision ImageNet networks of Fig. 6 (ResNet-50/152, VGG-16, ViT-B/16,
ConvNeXt-S/B) are the `IMAGENET1K_V1` weights, downloaded by torchvision into
`TORCH_HOME`.

## Environment

`requirements.txt` pins the exact versions that produced the reported numbers,
each read from installed package metadata rather than transcribed. Python
3.11.15. `environment.lock.yml` (`conda env export --no-builds`) and
`requirements.lock.txt` (`pip freeze`) record the complete environment,
exported on 2026-09-15 from the environment the runs used; numpy 2.4.6 there
carries scipy-openblas 0.3.31 (`numpy.show_config()`), torch 2.13.0 is the
CUDA 13.0 build (GPU needed for training only). `docs/reproducibility/mpk-verify.def`
is an Apptainer definition of the same package set on CPU wheels; its header
says how to build it and run the verification or the referee script inside it.

`setup.cfg` additionally lists four KietzmannLab git dependencies. None of them
is imported by any module that produces a reported number — the header of
`requirements.txt` records where each one is actually used, and why the `DVD`
entry is correctly commented out: the developmental front-end transform is
implemented here, in `dkl_frontend.py`, not imported from that package.

## License

BSD-3-Clause, see `LICENSE`. The `licenses/` directory holds the license texts
of vendored third-party code.

## Citation

`CITATION.cff` carries the title and repository URL; the author list and the
Zenodo DOI are added at release. Until then, please cite this repository and the
commit you used.
