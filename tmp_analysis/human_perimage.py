"""The human side of the per-image cue-conflict comparison, and its noise ceiling.

2.6.2 asks whether a model's trial-level choices agree with an individual human
observer (Cohen's kappa, ceiling 0.234). That ceiling is low because ONE
observer's ONE binary response per image is a noisy quantity.

This script builds the better-estimated target: the per-image human shape rate
h_i, averaged over all 10 observers, which is what "the images humans find hard"
actually means. Its own reliability sets the ceiling for any model.

Two profiles are measured, because humans can respond in three ways -- the shape
category, the texture category, or neither:
  * h_i  = P(shape | responded shape or texture)   -- the cue-conflict quantity
  * v_i  = P(responded shape or texture)           -- whether either cue won at all

and each is also measured AFTER removing shape-category and texture-category
main effects. That residual is the one that matters for the paper's claim:
without it, a correlation could mean nothing more than "humans and models agree
that cats are easier than bottles", which is a claim about the 16 categories,
not about images.

The ceiling is exhaustive, not sampled: 10 observers admit only
C(10,5)/2 = 126 distinct half-splits, so all of them are enumerated and
Spearman-Brown corrected.

Usage: PYTHONPATH=. python tmp_analysis/human_perimage.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import re

import numpy as np

HUMAN_DIR = "/scratch/byrong/VST_data/geirhos_human_data/cue_conflict"
STEM = re.compile(r"([a-z]+)\d+-([a-z]+)\d+$")


def load_choice_matrix():
    """(n_obs, n_images) with 1.0 = shape, 0.0 = texture, nan = neither."""
    # Imported here, not at module scope: this pulls in torch via the encoding
    # package, and perimage_correspondence.py imports this module purely for
    # residualise/_pearson/_rank. Keeping it lazy makes that import cheap.
    from src.analysis.behavior.cue_conflict import list_cue_conflict_images
    items = list_cue_conflict_images()
    paths    = [i[0] for i in items]
    shapes   = [i[1] for i in items]
    textures = [i[2] for i in items]
    stems = [os.path.splitext(os.path.basename(q))[0] for q in paths]
    index = {st: i for i, st in enumerate(stems)}

    subs, rows = [], []
    for f in sorted(glob.glob(f"{HUMAN_DIR}/*subject-*.csv")):
        subj = re.search(r"(subject-\d+)", os.path.basename(f)).group(1)
        ch = np.full(len(stems), np.nan)
        with open(f) as fh:
            for row in csv.DictReader(fh):
                m = STEM.search(os.path.splitext(row["imagename"])[0])
                if not m:
                    continue
                i = index.get(m.group(0))
                if i is None:
                    continue
                r = row["object_response"]
                if r == shapes[i]:
                    ch[i] = 1.0
                elif r == textures[i]:
                    ch[i] = 0.0
        subs.append(subj)
        rows.append(ch)
    return np.vstack(rows), subs, stems, shapes, textures


def residualise(y, shapes, textures):
    """Remove shape-category and texture-category main effects from y.

    Least squares on an additive design with one indicator per shape category
    and one per texture category (intercept absorbed). What survives is
    image-specific: the part of the profile that two images sharing both a
    shape class and a texture class still disagree about.
    """
    finite = np.isfinite(y)
    sh = sorted(set(shapes))
    tx = sorted(set(textures))
    X = np.zeros((len(y), 1 + len(sh) + len(tx)))
    X[:, 0] = 1.0
    for j, c in enumerate(sh):
        X[np.asarray(shapes) == c, 1 + j] = 1.0
    for j, c in enumerate(tx):
        X[np.asarray(textures) == c, 1 + len(sh) + j] = 1.0
    beta, *_ = np.linalg.lstsq(X[finite], y[finite], rcond=None)
    out = np.full_like(y, np.nan)
    out[finite] = y[finite] - X[finite] @ beta
    return out


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def split_half_ceiling(mat, shapes=None, textures=None, resid=False):
    """Spearman-Brown-corrected split-half reliability over ALL 126 partitions."""
    n_obs = mat.shape[0]
    half = n_obs // 2
    seen, pr, sr = set(), [], []
    for combo in itertools.combinations(range(n_obs), half):
        other = tuple(sorted(set(range(n_obs)) - set(combo)))
        key = tuple(sorted([combo, other]))
        if key in seen:
            continue
        seen.add(key)
        with np.errstate(invalid="ignore"):
            a = np.nanmean(mat[list(combo)], 0)
            b = np.nanmean(mat[list(other)], 0)
        if resid:
            a = residualise(a, shapes, textures)
            b = residualise(b, shapes, textures)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            continue
        rp = _pearson(a[m], b[m])
        rs = _pearson(_rank(a[m]), _rank(b[m]))
        # Spearman-Brown: reliability of the full 10-observer mean, from halves.
        pr.append(2 * rp / (1 + rp) if rp > -1 else np.nan)
        sr.append(2 * rs / (1 + rs) if rs > -1 else np.nan)
    return np.array(pr, float), np.array(sr, float), len(seen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    C, subs, stems, shapes, textures = load_choice_matrix()
    N = C.shape[1]
    valid = np.isfinite(C)
    print(f"images {N}   observers {len(subs)}")
    print(f"valid shape-or-texture responses: {valid.sum()}/{C.size} "
          f"= {valid.mean():.4f}")
    print(f"aggregate shape bias: {np.nanmean(C):.4f}")

    nvalid = valid.sum(0)
    with np.errstate(invalid="ignore"):
        h = np.where(nvalid > 0, np.nansum(C, 0) / np.maximum(nvalid, 1), np.nan)
    v = nvalid / C.shape[0]

    print(f"\nper-image shape rate h: mean {np.nanmean(h):.4f}  "
          f"sd {np.nanstd(h):.4f}   images with no valid response: "
          f"{int((nvalid == 0).sum())}")
    for lo, hi in [(0.0, .001), (.001, .25), (.25, .5), (.5, .75),
                   (.75, .999), (.999, 1.001)]:
        n = int(np.sum((h >= lo) & (h < hi)))
        print(f"   h in [{lo:.3f},{hi:.3f}): {n:5d}  ({100*n/N:5.1f}%)")
    print(f"   images where >=1 observer chose texture: "
          f"{int(np.nansum(h < 1.0))}")
    print(f"per-image valid rate v: mean {v.mean():.4f}  sd {v.std():.4f}  "
          f"min {v.min():.2f}")

    results = {}
    for tag, mat, kw in [
        ("shape_rate",          C,                   dict()),
        ("shape_rate_resid",    C,                   dict(resid=True)),
        ("valid_rate",          valid.astype(float), dict()),
        ("valid_rate_resid",    valid.astype(float), dict(resid=True)),
    ]:
        pr, sr, npart = split_half_ceiling(
            mat, shapes=shapes, textures=textures, **kw)
        results[tag] = {
            "pearson_mean": float(np.nanmean(pr)),
            "pearson_lo": float(np.nanpercentile(pr, 2.5)),
            "pearson_hi": float(np.nanpercentile(pr, 97.5)),
            "spearman_mean": float(np.nanmean(sr)),
            "spearman_lo": float(np.nanpercentile(sr, 2.5)),
            "spearman_hi": float(np.nanpercentile(sr, 97.5)),
            "n_partitions": npart,
        }
        r = results[tag]
        print(f"\nceiling [{tag}]  ({npart} exhaustive half-splits)")
        print(f"   Pearson  SB {r['pearson_mean']:+.4f}  "
              f"[{r['pearson_lo']:+.4f}, {r['pearson_hi']:+.4f}]")
        print(f"   Spearman SB {r['spearman_mean']:+.4f}  "
              f"[{r['spearman_lo']:+.4f}, {r['spearman_hi']:+.4f}]")

    np.savez_compressed(
        os.path.join(args.out_dir, "human_perimage.npz"),
        C=C, subs=np.array(subs), stems=np.array(stems),
        shapes=np.array(shapes), textures=np.array(textures),
        h=h, v=v, h_resid=residualise(h, shapes, textures),
        v_resid=residualise(v, shapes, textures))
    with open(os.path.join(args.out_dir, "human_ceiling.json"), "w") as f:
        json.dump({"n_images": int(N), "n_observers": len(subs),
                   "aggregate_shape_bias": float(np.nanmean(C)),
                   "valid_fraction": float(valid.mean()),
                   "ceilings": results}, f, indent=2)
    print(f"\nsaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
