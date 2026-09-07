"""Attribute the per-image difficulty effect: learned weights, or input transform?

The matched comparison evaluates each arm with its own front end, so the diet
and no-diet arms differ in BOTH the weights and the test-time input. This reads
the completed 2x2 and separates them:

                     front end ON        front end OFF
    diet weights     dkl_bcs_*_final     dkl_bcs_*_final_nofe
    no-diet weights  dkl_base_*_final_fe dkl_base_*_final

The attribution contrasts hold one factor fixed:
  * weights, at fixed front end   -- diet vs no-diet within a column
  * transform, at fixed weights   -- on vs off within a row
If the effect follows the column, it is a claim about what the diet taught the
network. If it follows the row, it is a claim about the input at test time.

Statistics are exact by enumeration: 3 seeds per cell, so the two-sided
Mann-Whitney floor is 2/C(6,3) = .100 and the paired-sign floor is 2/2^3 = .250.
Neither can go lower; both are reported as attained, not as evidence of a
larger effect than the design can show.

Usage: PYTHONPATH=. python tmp_analysis/perimage_frontend_2x2.py --run-dir <dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

SEEDS = (0, 1, 2)
CELLS = {
    ("diet", "on"):    "dkl_bcs_seed{s}_final",
    ("diet", "off"):   "dkl_bcs_seed{s}_final_nofe",
    ("nodiet", "on"):  "dkl_base_seed{s}_final_fe",
    ("nodiet", "off"): "dkl_base_seed{s}_final",
}
TARGETS = ("valid_rate_resid", "shape_rate_resid", "valid_rate", "shape_rate")


def exact_mw(a, b):
    a, b = list(a), list(b)
    n, m = len(a), len(b)
    pool = a + b
    U = lambda x, y: sum((xi > yj) + 0.5 * (xi == yj) for xi in x for yj in y)
    obs, cnt, tot = U(a, b), 0, 0
    for combo in itertools.combinations(range(n + m), n):
        x = [pool[i] for i in combo]
        y = [pool[i] for i in range(n + m) if i not in combo]
        tot += 1
        if abs(U(x, y) - n * m / 2) >= abs(obs - n * m / 2) - 1e-9:
            cnt += 1
    return cnt / tot


def sign_p(diffs):
    """Exact two-sided sign test on paired differences (n = 3 -> floor .25)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = sum(x > 0 for x in d)
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def verify_parvo_identity(run_dir):
    """The front end must not touch the parvo stream. Assert it, do not assume.

    magno_parvo_inputs applies the blur and the contrast-sensitivity cut with
    transform_target="magno", so the parvo (L-M) input is untransformed and the
    parvo-stream logits L2 must be bit-identical between the front-end-on and
    front-end-off runs of the SAME checkpoint. If they are not, the front end is
    leaking into the chromatic stream and the whole 2x2 is misattributed --
    every "transform" contrast would silently carry a parvo change too.
    """
    import glob
    pairs = [("dkl_bcs_seed{s}_final", "dkl_bcs_seed{s}_final_nofe"),
             ("dkl_base_seed{s}_final_fe", "dkl_base_seed{s}_final")]
    for a, b in pairs:
        for s in SEEDS:
            fa = os.path.join(run_dir, "logits",
                              a.format(s=s) + "_perimage_logits.npz")
            fb = os.path.join(run_dir, "logits",
                              b.format(s=s) + "_perimage_logits.npz")
            if not (os.path.exists(fa) and os.path.exists(fb)):
                raise SystemExit(f"missing for parvo check: {fa} / {fb}")
            L2a = np.load(fa, allow_pickle=True)["L2"]
            L2b = np.load(fb, allow_pickle=True)["L2"]
            dev = float(np.abs(L2a - L2b).max())
            if dev > 1e-5:
                raise SystemExit(
                    f"front end changed the parvo stream: {a.format(s=s)} vs "
                    f"{b.format(s=s)} differ by {dev:.3e} -- the 2x2 cannot "
                    f"separate weights from transform if this is nonzero")
            # And the magno stream MUST differ, or the flags did nothing.
            L1a = np.load(fa, allow_pickle=True)["L1"]
            L1b = np.load(fb, allow_pickle=True)["L1"]
            mdev = float(np.abs(L1a - L1b).max())
            if mdev < 1e-3:
                raise SystemExit(
                    f"front end did NOT change the magno stream for "
                    f"{a.format(s=s)}: max dev {mdev:.3e} -- flags not applied")
            print(f"  parvo identical ({dev:.2e}), magno differs "
                  f"({mdev:.3f}): {a.format(s=s)} vs {b.format(s=s)}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--kind", default="pearson", choices=["pearson", "spearman"])
    args = p.parse_args()

    verify_parvo_identity(args.run_dir)

    R = json.load(open(os.path.join(args.run_dir,
                                    "perimage_correspondence.json")))
    M = R["models"]
    missing = [n.format(s=s) for (_, _), n in CELLS.items() for s in SEEDS
               if n.format(s=s) not in M]
    if missing:
        raise SystemExit("missing cells:\n  " + "\n  ".join(missing))

    def cell(w, fe, tgt):
        return [M[CELLS[(w, fe)].format(s=s)][f"{tgt}__{args.kind}"]
                ["oos_pct_ceiling"] for s in SEEDS]

    print(f"per-image correspondence, out of sample, % of human ceiling "
          f"({args.kind}); 3 seeds per cell\n")
    for tgt in TARGETS:
        print(f"=== {tgt} ===")
        g = {k: cell(k[0], k[1], tgt) for k in CELLS}
        print(f"{'':16s} {'front end ON':>22s} {'front end OFF':>22s}")
        for w in ("diet", "nodiet"):
            on, off = g[(w, "on")], g[(w, "off")]
            print(f"  {w+' weights':14s} "
                  f"{np.mean(on):6.1f} [{min(on):5.1f},{max(on):5.1f}] "
                  f"{np.mean(off):6.1f} [{min(off):5.1f},{max(off):5.1f}]")

        # attribution
        print("  attribution:")
        for fe in ("on", "off"):
            d, n = g[("diet", fe)], g[("nodiet", fe)]
            sep = ("diet>nodiet" if min(d) > max(n) else
                   "nodiet>diet" if max(d) < min(n) else "overlap")
            print(f"    weights   @ front end {fe:3s}: diet {np.mean(d):5.1f} "
                  f"vs no-diet {np.mean(n):5.1f}   MW p={exact_mw(d, n):.3f}  "
                  f"[{sep}]  paired sign p={sign_p([a-b for a, b in zip(d, n)]):.3f}")
        for w in ("diet", "nodiet"):
            on, off = g[(w, "on")], g[(w, "off")]
            print(f"    transform @ {w+' weights':15s}: on {np.mean(on):5.1f} "
                  f"vs off {np.mean(off):5.1f}   MW p={exact_mw(on, off):.3f}  "
                  f"paired sign p={sign_p([a-b for a, b in zip(on, off)]):.3f}")
        print()

    print("exact floors at this n: Mann-Whitney 3v3 = .100; "
          "paired sign test n=3 = .250")


if __name__ == "__main__":
    main()
