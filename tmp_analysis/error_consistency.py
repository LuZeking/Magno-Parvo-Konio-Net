"""PAPER_PLAN §57: error consistency against the 10 Geirhos human observers.

Pre-registered in §57 BEFORE running. A matching shape-bias RATE is weak
evidence for "human-like": two decision-makers with the same marginal rate can
agree on no particular image beyond what those rates force. Cohen's kappa
subtracts exactly that forced agreement.

    c_exp = p_m*p_h + (1-p_m)*(1-p_h)
    kappa = (c_obs - c_exp) / (1 - c_exp)

Both readouts are scored (§57.3): `full` is the whole model and is the only
unit comparable to a whole human observer; `magno_only` is the shape-biased
stream the paper's 2x2 is built on, with the other stream lesioned. §57.2
records that the draft's 0.878-vs-0.958 line compared a stream to an observer.

Usage: PYTHONPATH=. python tmp_analysis/error_consistency.py
"""
from __future__ import annotations
import glob, os, re, csv, itertools
import numpy as np
from src.analysis.behavior.cue_conflict import (
    GEIRHOS_CLASSES, load_class_index, list_cue_conflict_images,
)

HUMAN_DIR = "/scratch/byrong/VST_data/geirhos_human_data/cue_conflict"
CC = "results/cue_conflict"
STEM = re.compile(r"([a-z]+)\d+-([a-z]+)\d+$")

CONDITIONS = [
    ("diet",          [f"dkl_bcs_seed{i}"              for i in (0, 1, 2)]),
    ("no diet",       [f"dkl_base_seed{i}"             for i in (0, 1, 2)]),
    ("diet / swap",   [f"dkl_bcs_swap_seed{i}"         for i in (0, 1, 2)]),
    ("diet / highsf", [f"dkl_bcs_parvo_highsf_seed{i}" for i in (0, 1, 2)]),
]


def kappa(a, b):
    """Error consistency between two aligned choice vectors of 'S'/'T'/None."""
    m = np.array([x is not None and y is not None for x, y in zip(a, b)])
    if m.sum() < 30:
        return np.nan, int(m.sum())
    x = np.array([v == "S" for v in np.asarray(a, dtype=object)[m]])
    y = np.array([v == "S" for v in np.asarray(b, dtype=object)[m]])
    c_obs = float((x == y).mean())
    px, py = float(x.mean()), float(y.mean())
    c_exp = px * py + (1 - px) * (1 - py)
    if c_exp >= 1.0:
        return np.nan, int(m.sum())
    return (c_obs - c_exp) / (1 - c_exp), int(m.sum())


# ---------------------------------------------------------------- stimuli ---
images = list_cue_conflict_images()          # same sorted order the npz used
stems = [os.path.splitext(os.path.basename(p))[0] for p, _, _ in images]
shapes = [s for _, s, _ in images]
textures = [t for _, _, t in images]
index = {st: i for i, st in enumerate(stems)}
print(f"cue-conflict images: {len(stems)}")

# ----------------------------------------------------------------- humans ---
humans = {}
for f in sorted(glob.glob(f"{HUMAN_DIR}/*subject-*.csv")):
    subj = re.search(r"(subject-\d+)", os.path.basename(f)).group(1)
    choice = [None] * len(stems)
    with open(f) as fh:
        for row in csv.DictReader(fh):
            m = STEM.search(os.path.splitext(row["imagename"])[0])
            if not m:
                continue
            st = m.group(0)
            i = index.get(st)
            if i is None:                      # shape==texture, excluded
                continue
            r = row["object_response"]
            if r == shapes[i]:
                choice[i] = "S"
            elif r == textures[i]:
                choice[i] = "T"
    humans[subj] = choice
print(f"human observers: {len(humans)}  "
      f"(median trials decided: {int(np.median([sum(c is not None for c in v) for v in humans.values()]))})")

# ------------------------------------------------------- human-human floor --
hh = [kappa(humans[a], humans[b])[0]
      for a, b in itertools.combinations(sorted(humans), 2)]
hh = np.array(hh, dtype=float)
CEIL = float(np.nanmean(hh))
print(f"\nHUMAN-HUMAN CEILING: kappa = {CEIL:.3f} "
      f"(n={np.isfinite(hh).sum()} pairs, sd {np.nanstd(hh):.3f}, "
      f"range {np.nanmin(hh):.3f}-{np.nanmax(hh):.3f})")

# ------------------------------------------------------------------ models --
cls = load_class_index()
cols = [cls[c] for c in GEIRHOS_CLASSES]
sh_a = np.asarray(shapes, dtype=object)
tx_a = np.asarray(textures, dtype=object)


def model_choice(model, readout):
    z = np.load(f"{CC}/{model}_logits.npz", allow_pickle=True)
    if readout not in z.files:
        return None
    dec = np.asarray(GEIRHOS_CLASSES, dtype=object)[z[readout][:, cols].argmax(1)]
    out = np.full(len(stems), None, dtype=object)
    out[dec == sh_a] = "S"
    out[dec == tx_a] = "T"
    return list(out)


for readout in ("full", "magno_only"):
    print("\n" + "=" * 94)
    print(f"READOUT: {readout}"
          + ("   <- whole model; the unit comparable to a whole human observer"
             if readout == "full" else
             "   <- shape-biased stream, other stream lesioned"))
    print("=" * 94)
    print(f"  {'condition':16s} {'seed':>4s} {'shape bias':>11s} {'kappa vs human':>15s} "
          f"{'% of ceiling':>13s} {'n img':>7s}")
    means = {}
    for label, models in CONDITIONS:
        vals = []
        for si, m in enumerate(models):
            ch = model_choice(m, readout)
            if ch is None:
                print(f"  {label:16s} {si:>4d}   (readout absent)"); continue
            ks, ns = zip(*[kappa(ch, humans[h]) for h in sorted(humans)])
            k = float(np.nanmean(ks))
            nS = sum(c == "S" for c in ch); nT = sum(c == "T" for c in ch)
            sb = nS / (nS + nT) if (nS + nT) else np.nan
            vals.append(k)
            print(f"  {label:16s} {si:>4d} {sb:11.3f} {k:15.3f} "
                  f"{100*k/CEIL:12.1f}% {int(np.mean(ns)):7d}")
        means[label] = np.array(vals, dtype=float)
    d, nd = means.get("diet"), means.get("no diet")
    if d is not None and nd is not None and d.size == 3 and nd.size == 3:
        sep = d.min() > nd.max()
        print(f"\n  H1  diet {d.mean():.3f} vs no diet {nd.mean():.3f}   "
              f"diff {d.mean()-nd.mean():+.3f}   "
              f"{'COMPLETE SEPARATION' if sep else 'OVERLAP'}")
