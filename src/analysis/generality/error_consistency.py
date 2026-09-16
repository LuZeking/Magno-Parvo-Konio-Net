"""Error consistency (Cohen's kappa) of a frozen network with the ten Geirhos observers, at the
trained readout and at the two ends of its accuracy gate.

  python -m src.analysis.generality.error_consistency --features FEAT/<tag>_cue_conflict.npz \
      [--factor BEH/<tag>_subspace.npz --gate BEH/<tag>_gain_accuracy_subspace.json] --out-dir BEH

The fourth number of the Discussion's reporting box, computed exactly as Results 2.5 / Extended
Data Table 7 compute it for our own networks (tmp_analysis/sweep_error_consistency.py, whose
`kappa` and `load_humans` are reproduced here): every cue-conflict image the network decides as
the shape or the texture category gives a choice 'S' or 'T', the human observer's choice on the
same image likewise, and kappa subtracts the agreement the two marginal rates force. Mean over
the ten observers; reported also as a percentage of the human-human ceiling, which is recomputed
from the observers and asserted to be the published 0.234. With a subspace factor and its gate,
kappa is also computed at the texture end and the shape end of the gate (all-image bases, the
gain of Methods 4.8), so that agreement can be read against the reachable span of the rate.
Writes <tag>_error_consistency.json.
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

from src.analysis.generality.scoring import Scorer, GEIRHOS_CLASSES

HUMAN_DIR = os.environ.get("GEIRHOS_HUMAN_DIR", "/scratch/byrong/VST_data/geirhos_human_data/cue_conflict")
STEM = re.compile(r"([a-z]+)\d+-([a-z]+)\d+$")


def kappa(a, b):
    """Cohen's kappa on aligned 'S'/'T'/None choice vectors (Results 2.5)."""
    m = np.array([x is not None and y is not None for x, y in zip(a, b)])
    if m.sum() < 30:
        return np.nan
    x = np.array([v == "S" for v in np.asarray(a, dtype=object)[m]])
    y = np.array([v == "S" for v in np.asarray(b, dtype=object)[m]])
    c_obs = float((x == y).mean())
    px, py = float(x.mean()), float(y.mean())
    c_exp = px * py + (1 - px) * (1 - py)
    return np.nan if c_exp >= 1.0 else (c_obs - c_exp) / (1 - c_exp)


def load_humans(stems, shapes, textures):
    index = {st: i for i, st in enumerate(stems)}
    humans = {}
    for f in sorted(glob.glob(f"{HUMAN_DIR}/*subject-*.csv")):
        subj = re.search(r"(subject-\d+)", os.path.basename(f)).group(1)
        choice = [None] * len(stems)
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
                    choice[i] = "S"
                elif r == textures[i]:
                    choice[i] = "T"
        humans[subj] = choice
    return humans


def choices(logits, scorer: Scorer, shapes, textures):
    r = scorer._restricted(np.asarray(logits, dtype=np.float64))
    dec = np.asarray(GEIRHOS_CLASSES, dtype=object)[r.argmax(1)]
    ch = np.full(len(shapes), None, dtype=object)
    ch[dec == np.asarray(shapes, dtype=object)] = "S"
    ch[dec == np.asarray(textures, dtype=object)] = "T"
    return list(ch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--factor", default=None, help="<tag>_subspace.npz (B_S, B_T) for the gate ends")
    p.add_argument("--gate", default=None, help="<tag>_gain_accuracy_subspace.json (w_within_5pts)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    z = np.load(args.features, allow_pickle=True)
    tag, kind = str(z["tag"]), str(z["scorer"])
    stems = [str(s) for s in z["stems"]]
    shapes = [str(s) for s in z["shapes"]]
    textures = [str(t) for t in z["textures"]]
    scorer = Scorer(kind)
    humans = load_humans(stems, shapes, textures)
    if len(humans) != 10:
        raise SystemExit(f"expected 10 observers, found {len(humans)} in {HUMAN_DIR}")
    hsb = np.array([sum(v == "S" for v in c) / sum(v is not None for v in c) for c in humans.values()])
    ceil = float(np.mean([kappa(humans[a], humans[b]) for a, b in itertools.combinations(sorted(humans), 2)]))
    assert abs(hsb.mean() - 0.958) < 0.002, f"human shape bias {hsb.mean():.4f}"
    assert abs(ceil - 0.234) < 0.005, f"human-human ceiling {ceil:.4f}"

    def evaluate(logits, label):
        ch = choices(logits, scorer, shapes, textures)
        ks = {h: kappa(ch, humans[h]) for h in sorted(humans)}
        kbar = float(np.nanmean(list(ks.values())))
        sc = scorer.score(logits, shapes, textures)
        print(f"[error_consistency] {tag} {label:12s} rate {sc['shape_bias']:.3f}  kappa {kbar:+.4f}  ({100 * kbar / ceil:5.1f}% of ceiling)", flush=True)
        return dict(kappa_mean=kbar, kappa_pct_ceiling=100 * kbar / ceil, kappa_per_human=ks,
                    shape_bias=sc["shape_bias"], decided=sc["n_shape"] + sc["n_texture"])

    out = dict(tag=tag, scorer=kind, ceiling=ceil, human_shape_bias=float(hsb.mean()), n_images=len(stems),
               trained=evaluate(z["logits"], "trained"))
    if args.factor and args.gate:
        from src.analysis.generality.subspace_factor import gain_features
        fac = np.load(args.factor, allow_pickle=True)
        g = json.load(open(args.gate))
        B_S, B_T = fac["B_S"].astype(np.float64), fac["B_T"].astype(np.float64)
        y = z["features"].astype(np.float64)
        W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)
        w_lo, w_hi = float(min(g["w_within_5pts"])), float(max(g["w_within_5pts"]))
        out["gate"] = [w_lo, w_hi]
        out["texture_end"] = evaluate(gain_features(y, B_S, B_T, w_lo) @ W.T + b, f"w={w_lo:.2f}")
        out["shape_end"] = evaluate(gain_features(y, B_S, B_T, w_hi) @ W.T + b, f"w={w_hi:.2f}")
        # the largest kappa over the gated sweep, as Fig. 5b reports for our own readouts
        best = None
        for w in np.arange(w_lo, w_hi + 1e-9, 0.05):
            ch = choices(gain_features(y, B_S, B_T, float(w)) @ W.T + b, scorer, shapes, textures)
            kb = float(np.nanmean([kappa(ch, humans[h]) for h in humans]))
            if best is None or kb > best[1]:
                best = (float(w), kb)
        out["best_in_gate"] = dict(w=best[0], kappa_mean=best[1], kappa_pct_ceiling=100 * best[1] / ceil)
        print(f"[error_consistency] {tag} best in gate: w={best[0]:.2f} kappa {best[1]:+.4f} ({100 * best[1] / ceil:.1f}%)", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, f"{tag}_error_consistency.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
