# Shape-biased behaviour does not require shape-biased representations

Code and analysis for a two-stream network that receives approximations of the
magnocellular and parvocellular signals reaching cortex, and the encoding,
behavioural and statistical analyses built on it.

The claim: a behavioural shape-bias rate is a property of a *readout over
components*, not a report of any one component. The luminance stream ends
training shape-biased and the chromatic stream texture-biased; fitted to eight
subjects of the Natural Scenes Dataset it is the **texture**-biased stream whose
unique variance carries the ventral end of the hierarchy; and behaviour, reading
out from both, is shape-biased.

Manuscript: `docs/PAPER_DRAFT.md`. Extended Data: `docs/PAPER_EXTENDED_DATA.md`.
Methods §4.1–§4.7 name, by path, the module that produces every reported number.

> **Status:** manuscript in preparation. This repository is the working tree, not
> an archived release. A DOI-archived version carrying the derived result tables
> is committed to at acceptance — see *Data availability* and *Code
> availability* in the draft.

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

It checks, among others: the §2.2 untrained cortical anchor (−0.544 normalised,
−0.120 raw, 8/8 subjects at the exact Wilcoxon floor p = .0078); that
`results/encoding/` and `results/encoding_cc/` are numerically identical across
the 40 files they share (max |Δr²| = 0.000e+00); the §2.6.1 out-of-distribution
relative-accuracy AUCs; the §2.6.2 per-image human-correspondence percentages;
and that no effect size in the §2.2 table exceeds the |dz| = 6.60 the draft
calls its largest.

On a cluster node, also point `MPLCONFIGDIR` at a writable path.

## Repository layout

```
src/training/standalone_trainer.py              the trainer that produced the reported models
src/models/pytorch_models.py                    TwoStreamNetwork (two ResNet-18 trunks + joint head)
src/analysis/encoding/dkl_frontend.py           RGB->DKL, and the blur + contrast-sensitivity front end
src/analysis/encoding/run_encoding.py           banded-ridge voxelwise encoding on NSD
src/analysis/behavior/cue_conflict.py           Geirhos cue-conflict scoring
src/analysis/behavior/stream_shape_texture.py   per-stream linear probe
tmp_analysis/                                   the analysis entry points named in Methods §4.1-§4.7
slurm/                                          the submission script each run used
results/                                        committed result files behind the tables
docs/                                           manuscript, extended data, reporting summary, reviews
```

`tmp_analysis/` holds analysis entry points despite the name; they are the
scripts the Methods cite, and they are version-controlled deliberately.

## Reproducing from scratch

Three stages, in order. Stages 2 and 3 need the datasets listed below.
Submission scripts live in `slurm/`; adapt the directives to your scheduler.

| stage | script | resources as submitted |
|---|---|---|
| 1. train | `slurm/train_array.sh` | GPU, 22G RAM, 150 epochs, batch 256, 3 seeds per condition |
| 2. encode | `slurm/encode_subjects.sh <model_tag>` | **CPU only**, ~160G RAM per array task, 8 subjects |
| 3. analyse | `tmp_analysis/*.py` | CPU, modest memory |

Training is the only stage that needs a GPU. Everything else — feature
extraction, ridge fits, statistics, plotting — runs on CPU.

**A note on the encoding footprint.** Stage 2 is submitted at ~160G per task
because it holds a subject's full feature matrices in memory alongside the
banded-ridge solve. If your nodes are smaller, reduce the footprint rather than
the problem: PCA the streams to fewer components, read the beta files in
chunks or memory-mapped, and drop cached tensors between folds. Several scripts
in `slurm/` carry historical resource requests from the runs that produced the
committed results and are deliberately left as they were rather than rewritten,
so check the directives before submitting anything.

Individual analysis entry points are argument-driven and self-documenting:

```bash
PYTHONPATH=. python src/analysis/behavior/cue_conflict.py --help
PYTHONPATH=.:tmp_analysis python tmp_analysis/fusion_sweep.py --help
PYTHONPATH=.:tmp_analysis python tmp_analysis/sweep_error_consistency.py --help
```

## Data

None of it is redistributed here.

| dataset | where | terms |
|---|---|---|
| Natural Scenes Dataset | `naturalscenesdataset.org` | own data-use agreement, requires registration. We use the `betas_fithrf_GLMdenoise_RR` (b3) preparation and the `nsdgeneral` mask |
| Geirhos cue-conflict stimuli and 10 human observers | `github.com/rgeirhos/texture-vs-shape`, under `raw-data/style-transfer-512-nomask-experiment/` | CC BY 4.0 |
| MiniEcoset | `osf.io/msna2` | see the OSF record |

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

A citation will be added when the manuscript is public. Until then, please cite
this repository and the commit you used.
