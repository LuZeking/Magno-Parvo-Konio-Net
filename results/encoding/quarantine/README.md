# Quarantined encoding results

Files here are **excluded from analysis on purpose**. `stats.load_results()`
globs `<out_dir>/<model>_subj*.npz` non-recursively, so anything in this
subdirectory is invisible to every analysis script without any code change.

## `dkl_base_seed0_trained_subj01.npz` — quarantined 2026-09-01

Fitted with the **wrong front end**. See PAPER_PLAN §22 for the full account.

Job `26386734_0` re-ran this subject after a disk-quota failure but omitted
`--no-lum-blur --no-lum-cs`, so a network trained on unblurred input was
encoded through the blur + contrast-sensitivity front end. Confirmed three
independent ways:

1. Recomputed features match the blur+CS front end **exactly** (max abs
   delta 0.000) and differ from the intended one by 16.17 on the magno band
   (§22.2).
2. Its hV4−V1 is a six-fold outlier against its own seven siblings and
   against the same subject in the clean `dkl_base_seed1` run (§22.4).
3. Its magno participation ratio is 32.2 against parvo's 189.5 — every other
   model has magno at least as high-dimensional as parvo (§27.2).

It contributed **57%** of the reported no-diet effect size and was the
difference between significant and not on the raw statistic.

**Effect of this quarantine**: `dkl_base_seed0_trained` now loads as n=7, which
is the correct state. Analyses that previously reported n=8 for this model must
be re-run; their numbers are superseded.

**When the repair lands** (tasks 16-23 of job 26399760 write a correctly
fronted, provenance-stamped version to `results/encoding_cc/`), copy that file
into `results/encoding/` and delete this one. Do not simply restore this file.
