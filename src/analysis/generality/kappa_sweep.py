"""Cue-conflict rate and error consistency with the ten Geirhos observers at every gain w.

  python -m src.analysis.generality.kappa_sweep --features FEAT/<tag>_cue_conflict.npz \
      --factor BEH/<tag>_subspace.npz --gate BEH/<tag>_gain_accuracy_subspace.json --out-dir BEH
  python -m src.analysis.generality.kappa_sweep --features-dir FEAT --out-dir BEH   # whole zoo

`error_consistency.py` evaluates kappa at four points of a network's sweep (trained, the two
ends of the accuracy gate, the best gated w). This walks the whole 21-point grid of Extended
Data Table 13, so that the rate and the agreement can be read against each other along the same
trajectory, and so that networks can be compared at a *matched* rate rather than at whichever
rate each one happens to occupy (`docs/PLAN_KAPPA_2026-09-17.md`, predictions K-1 to K-6).

Estimator, images, observers, bases and gate are those of Methods 4.8; kappa, the human loader
and the choice rule are imported from `error_consistency.py`, so nothing is recomputed a second
way. Writes <tag>_kappa_sweep.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from src.analysis.generality.error_consistency import choices, kappa, load_humans
from src.analysis.generality.scoring import Scorer
from src.analysis.generality.subspace_factor import gain_features

WS = [round(0.05 * i, 2) for i in range(21)]


def kappa_max(a, b):
    """The largest kappa attainable at these two marginals (Cohen 1960).

    Two decision-makers whose shape rates differ cannot agree on every image: the most they can
    agree is 1 - |p_a - p_b|, so kappa is bounded by (1 - |p_a - p_b| - p_e)/(1 - p_e). A model
    at rate 0.2 against observers at 0.96 is capped near 2% of the human-human kappa whatever it
    does per image, which is why kappa rises with any manipulation that moves the rate toward
    the human one. Returned alongside kappa so that agreement can also be read as a fraction of
    what is attainable (kappa / kappa_max).
    """
    m = np.array([x is not None and y is not None for x, y in zip(a, b)])
    if m.sum() < 30:
        return np.nan, np.nan, np.nan, int(m.sum())
    x = np.array([v == "S" for v in np.asarray(a, dtype=object)[m]])
    y = np.array([v == "S" for v in np.asarray(b, dtype=object)[m]])
    px, py = float(x.mean()), float(y.mean())
    p_e = px * py + (1 - px) * (1 - py)
    if p_e >= 1.0:
        return np.nan, px, py, int(m.sum())
    return float((1 - abs(px - py) - p_e) / (1 - p_e)), px, py, int(m.sum())


def run_one(features, factor, gate_json, out_dir, cache):
    z = np.load(features, allow_pickle=True)
    tag, kind = str(z["tag"]), str(z["scorer"])
    stems = [str(s) for s in z["stems"]]
    shapes = [str(s) for s in z["shapes"]]
    textures = [str(t) for t in z["textures"]]
    scorer = Scorer(kind)

    key = tuple(stems)
    if key not in cache:                      # the observers' choices depend only on the image order
        humans = load_humans(stems, shapes, textures)
        if len(humans) != 10:
            raise SystemExit(f"expected 10 observers, found {len(humans)} in the human directory")
        names = sorted(humans)
        hsb = float(np.mean([sum(v == "S" for v in humans[h]) / sum(v is not None for v in humans[h])
                             for h in names]))
        pairs = list(itertools.combinations(names, 2))
        ceil = float(np.mean([kappa(humans[a], humans[b]) for a, b in pairs]))
        # the same ceiling on the attainable scale: observers differ little in rate, so it is near 1
        ceil_frac = float(np.mean([kappa(humans[a], humans[b]) / kappa_max(humans[a], humans[b])[0]
                                   for a, b in pairs]))
        assert abs(hsb - 0.958) < 0.002, f"human shape bias {hsb:.4f}"
        assert abs(ceil - 0.234) < 0.005, f"human-human ceiling {ceil:.4f}"
        cache[key] = (humans, names, hsb, ceil, ceil_frac)
    humans, names, hsb, ceil, ceil_frac = cache[key]

    fac = np.load(factor, allow_pickle=True)
    B_S, B_T = fac["B_S"].astype(np.float64), fac["B_T"].astype(np.float64)
    y = z["features"].astype(np.float64)
    W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)

    g = json.load(open(gate_json))
    gate = [float(min(g["w_within_5pts"])), float(max(g["w_within_5pts"]))]
    top1 = {round(float(w), 2): float(t) for w, t in zip(g["ws"], g["top1"])}

    rows = []
    for w in WS:
        logits = gain_features(y, B_S, B_T, float(w)) @ W.T + b
        sc = scorer.score(logits, shapes, textures)
        ch = choices(logits, scorer, shapes, textures)
        ks = {h: kappa(ch, humans[h]) for h in names}
        km = {h: kappa_max(ch, humans[h]) for h in names}
        vals = np.array([ks[h] for h in names], dtype=np.float64)
        mx = np.array([km[h][0] for h in names], dtype=np.float64)
        frac = np.array([ks[h] / km[h][0] if np.isfinite(km[h][0]) and km[h][0] > 0 else np.nan
                         for h in names], dtype=np.float64)
        kbar = float(np.nanmean(vals))
        rows.append(dict(
            w=w, rate=sc["shape_bias"], forced=sc["forced_two_way"],
            decided=sc["n_shape"] + sc["n_texture"],
            kappa_mean=kbar, kappa_pct_ceiling=100 * kbar / ceil,
            kappa_max_mean=float(np.nanmean(mx)), kappa_over_max=float(np.nanmean(frac)),
            kappa_per_human={h: (None if np.isnan(ks[h]) else float(ks[h])) for h in names},
            kappa_max_per_human={h: (None if not np.isfinite(km[h][0]) else float(km[h][0])) for h in names},
            n_human_nan=int(np.isnan(vals).sum()),
            top1=top1.get(w), in_gate=bool(gate[0] - 1e-9 <= w <= gate[1] + 1e-9),
        ))
        print(f"[kappa_sweep] {tag} w={w:.2f} rate {sc['shape_bias']:.3f} "
              f"kappa {kbar:+.4f} ({100 * kbar / ceil:5.1f}% of ceiling, "
              f"{100 * rows[-1]['kappa_over_max']:5.1f}% of attainable) decided {rows[-1]['decided']:4d}"
              f"{' gated' if rows[-1]['in_gate'] else ''}", flush=True)

    out = dict(tag=tag, scorer=kind, ceiling=ceil, ceiling_over_max=ceil_frac, human_shape_bias=hsb,
               n_images=len(stems), observers=names, gate=gate,
               trained_top1=float(g["trained_top1"]), sweep=rows)
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{tag}_kappa_sweep.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest, flush=True)
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", help="one <tag>_cue_conflict.npz")
    p.add_argument("--factor", help="<tag>_subspace.npz (all-image bases B_S, B_T)")
    p.add_argument("--gate", help="<tag>_gain_accuracy_subspace.json (w_within_5pts, top1)")
    p.add_argument("--features-dir", help="run every *_cue_conflict.npz in this directory in one process")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    cache: dict = {}

    if args.features_dir:
        import glob
        done = skipped = 0
        for cc in sorted(glob.glob(os.path.join(args.features_dir, "*_cue_conflict.npz"))):
            tag = os.path.basename(cc)[: -len("_cue_conflict.npz")]
            fac = os.path.join(args.out_dir, f"{tag}_subspace.npz")
            gat = os.path.join(args.out_dir, f"{tag}_gain_accuracy_subspace.json")
            if not (os.path.exists(fac) and os.path.exists(gat)):
                print(f"[kappa_sweep] SKIP {tag}: no stored subspace or gate", flush=True)
                skipped += 1
                continue
            run_one(cc, fac, gat, args.out_dir, cache)
            done += 1
        print(f"[kappa_sweep] {done} sweeps written, {skipped} skipped", flush=True)
        return
    if not (args.features and args.factor and args.gate):
        raise SystemExit("either --features-dir or all of --features/--factor/--gate")
    run_one(args.features, args.factor, args.gate, args.out_dir, cache)


if __name__ == "__main__":
    main()
