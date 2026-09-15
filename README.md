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

It checks, among others: the §2.3 untrained cortical anchor (−0.544 normalised,
−0.120 raw, 8/8 subjects at the exact Wilcoxon floor p = .0078); that
`results/encoding/` and `results/encoding_cc/` are numerically identical across
the 40 files they share (max |Δr²| = 0.000e+00); the §2.5 out-of-distribution
relative-accuracy AUCs and per-image human-correspondence percentages; that no
effect size exceeds the |dz| = 6.60 the manuscript calls its largest; and, from
`results/figures_v2/generality_behaviour.json` and `generality_cortex.json`, the
Results 2.4 / Fig. 6 / Extended Data Table 13–14 numbers of the single-trunk
generality test (spans, monotonicity, the accuracy-gated re-ordering of the
ImageNet networks, and the cortical contrasts of the three networks fitted to
NSD).

On a cluster node, also point `MPLCONFIGDIR` at a writable path.

## Repository layout

```
src/training/standalone_trainer.py              the trainer that produced the reported models
src/models/pytorch_models.py                    TwoStreamNetwork (two ResNet-18 trunks + joint head)
src/analysis/encoding/dkl_frontend.py           RGB->DKL, and the blur + contrast-sensitivity front end
src/analysis/encoding/run_encoding.py           banded-ridge voxelwise encoding on NSD
src/analysis/encoding/readout_w_cortex.py       cortex-optimal readout weight (ED Fig. 5, ED Table 12)
src/analysis/behavior/cue_conflict.py           Geirhos cue-conflict scoring
src/analysis/behavior/stream_shape_texture.py   per-stream linear probe
src/analysis/generality/                        single-trunk generality test (Results 2.4, Fig. 6, ED Tables 13-14):
                                                models.py (frozen-network loader), extract.py, unit_factor.py,
                                                gain_accuracy.py, refit_sum.py, cortex_bands.py, scoring.py
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
| 4. generality (Results 2.4) | `slurm/generality_{cc,nsd,mini,inval,cortex,rules}.sh`, then `tmp_analysis/figs_v2/generality_{behaviour,cortex}.py` | **CPU only**, 8–48G per array task; ≈ 1,600 allocated core-hours in total |

Training is the only stage that needs a GPU. Everything else — feature
extraction, ridge fits, statistics, plotting — runs on CPU.

**A note on the encoding footprint.** Stage 2 is submitted at ~160G per task
because it holds a subject's full feature matrices in memory alongside the
banded-ridge solve. If your nodes are smaller, reduce the footprint rather than
the problem: PCA the streams to fewer components, read the beta files in
chunks or memory-mapped, and drop cached tensors between folds. This is also why
the swap and `parvo_highsf` cortical fits of the manuscript are single networks
(seed 0): that allocation was not available for the two controls once the diet
and no-diet arms had been fitted (Methods 4.5). Several scripts in `slurm/`
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
3.11.15.

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
