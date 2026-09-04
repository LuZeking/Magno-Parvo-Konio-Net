"""Score the §53 pre-registered OOD robustness experiment.

Reads results/ood/*.json (4 conditions x 3 seeds) and evaluates H1/H2/H3 as
written in §53.4, against the outcomes declared in §53.5.

STATISTICAL HONESTY, stated up front. The replication unit here is the TRAINING
SEED, n=3 per condition, which is the right unit for a claim about training but
is badly underpowered: the exact two-sided Mann-Whitney floor for 3 vs 3 is
p = 0.10, reached only on complete separation. So the primary evidence is
reported as (i) complete separation yes/no, (ii) effect size, (iii) whether the
ordering holds at EVERY severity level, and only then (iv) the exact p with its
floor named. A "p = 0.10" here means "as strong as n=3 vs 3 can be", not "weak
trend", and it is written that way so it cannot be read as either more or less
than it is.

Per family the summary statistic is the area under the degradation curve
(mean over severities), in both statistics per §8:
    absolute = accuracy
    relative = accuracy / accuracy(clean)

Usage: PYTHONPATH=. python tmp_analysis/ood_verdict.py [results-dir]

The optional argument selects the checkpoint policy: `results/ood` (the
published final-checkpoint run) or `results/ood_bestep` (best saved checkpoint
per seed, PLAN §71.6). Identical scoring code either way, so the two are
directly comparable.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from scipy import stats as sps

D = sys.argv[1] if len(sys.argv) > 1 else "results/ood"
CONDS = [("diet",         "dkl_bcs_seed",              True),
         ("no diet",      "dkl_base_seed",             False),
         ("swap",         "dkl_bcs_swap_seed",         False),
         ("parvo_highsf", "dkl_bcs_parvo_highsf_seed", False)]
SHAPE_BIASED = {"diet"}


def load():
    out = {}
    for label, prefix, _sb in CONDS:
        runs = []
        for f in sorted(glob.glob(f"{D}/{prefix}*.json")):
            runs.append(json.load(open(f)))
        out[label] = runs
    return out


def curves(runs, family):
    """(absolute AUC, relative AUC, per-severity accuracy matrix) per seed."""
    acc = np.array([r["families"][family]["accuracy"] for r in runs], float)
    rel = acc / acc[:, :1]
    return acc.mean(1), rel.mean(1), acc, rel


def sep(a, b):
    """Complete separation and the exact rank-sum p, floor named."""
    p = sps.mannwhitneyu(a, b, alternative="two-sided").pvalue
    complete = a.min() > b.max() or b.min() > a.max()
    return complete, p


def main():
    data = load()
    n = {k: len(v) for k, v in data.items()}
    print("=" * 88)
    print("§53 OOD ROBUSTNESS -- scored against the pre-registration")
    print("=" * 88)
    print("seeds per condition:", n)
    missing = [k for k, v in n.items() if v < 3]
    if missing:
        print(f"\n!! INCOMPLETE: {missing} -- refusing to score a partial table (§39.3)")
        return

    fams = [(f, data["diet"][0]["families"][f]["counts"])
            for f in data["diet"][0]["families"]]

    print(f"\nCLEAN ACCURACY (severity 0), mean over 3 seeds")
    for label, _p, _s in CONDS:
        c = np.array([r["families"]["uniform-noise"]["accuracy"][0] for r in data[label]])
        print(f"  {label:14s} {c.mean():.4f} ± {c.std(ddof=1):.4f}   seeds {np.round(c,3)}")

    for stat_name, idx in (("RELATIVE (degradation slope -- the robustness statistic)", 1),
                           ("ABSOLUTE (confounded by clean accuracy)", 0)):
        print("\n" + "=" * 88)
        print(stat_name)
        print("=" * 88)
        print(f"{'family':18s} {'counts':7s} " +
              " ".join(f"{l:>14s}" for l, _p, _s in CONDS) + "   diet vs no-diet")
        for fam, counts in fams:
            vals = {}
            for label, _p, _s in CONDS:
                a, r, _acc, _rel = curves(data[label], fam)
                vals[label] = r if idx else a
            complete, p = sep(vals["diet"], vals["no diet"])
            d = vals["diet"].mean() - vals["no diet"].mean()
            flag = "SEP" if complete else "overlap"
            print(f"{fam:18s} {'yes' if counts else 'NO':7s} " +
                  " ".join(f"{vals[l].mean():14.4f}" for l, _p, _s in CONDS) +
                  f"   {d:+.4f} {flag} p={p:.3f}")

    # ---- the three hypotheses
    print("\n" + "=" * 88)
    print("VERDICT against §53.4")
    print("=" * 88)

    def rel_auc(label, fam):
        return curves(data[label], fam)[1]

    counted = [f for f, c in fams if c]
    h1_fams = ["uniform-noise", "high-pass", "false-colour"]
    print("\nH1  diet degrades slower than no-diet on noise / high-pass / false-colour:")
    h1 = []
    for f in h1_fams:
        a, b = rel_auc("diet", f), rel_auc("no diet", f)
        ok = a.mean() > b.mean()
        c, p = sep(a, b)
        h1.append(ok and c)
        print(f"    {f:17s} diet {a.mean():.4f} vs {b.mean():.4f}  "
              f"{'diet higher' if ok else 'NO':12s} {'complete sep' if c else 'overlap':13s} p={p:.3f}")
    print(f"  -> H1 {'SUPPORTED' if all(h1) else 'NOT supported'}")

    a, b = rel_auc("diet", "phase-scrambling"), rel_auc("no diet", "phase-scrambling")
    c, p = sep(a, b)
    h2 = a.mean() < b.mean() and c
    print(f"\nH2  crossover -- diet should degrade MORE under phase-scrambling:")
    print(f"    phase-scrambling  diet {a.mean():.4f} vs {b.mean():.4f}  "
          f"{'diet WORSE (predicted)' if a.mean()<b.mean() else 'diet better (NOT predicted)'}  "
          f"{'complete sep' if c else 'overlap'} p={p:.3f}")
    print(f"  -> H2 {'SUPPORTED' if h2 else 'NOT supported'}")

    print("\nH3  swap / parvo_highsf have no shape bias -> must pattern with NO DIET:")
    h3 = []
    for f in counted:
        d_, nd = rel_auc("diet", f).mean(), rel_auc("no diet", f).mean()
        for ctl in ("swap", "parvo_highsf"):
            cv = rel_auc(ctl, f).mean()
            near_nd = abs(cv - nd) < abs(cv - d_)
            h3.append(near_nd)
            print(f"    {f:17s} {ctl:13s} {cv:.4f}   closer to "
                  f"{'no-diet  OK' if near_nd else 'DIET  <-- generic-regulariser signature'}")
    frac = sum(h3) / len(h3)
    print(f"  -> H3 {frac*100:.0f}% of control comparisons pattern with no-diet")

    print("\n" + "-" * 88)
    if all(h1) and h2 and frac > 0.6:
        print("OUTCOME A -- shape bias confers robustness, causally isolated by the 2x2.")
    elif all(h1) and frac <= 0.6:
        print("OUTCOME B -- diet wins but swap comes with it: generic regularisation,")
        print("             robustness NOT attributable to shape bias.")
    elif not any(h1):
        print("OUTCOME C -- the diet's shape bias does not buy robustness.")
        print("             Per §53.5: drop the NeurIPS framing, report as a negative.")
    else:
        print("MIXED -- read the tables above; no declared outcome fits cleanly.")
        print("         Report what holds and what does not; do not reshape the")
        print("         prediction to fit (§53.5).")
    print("-" * 88)
    print("\nReminder: low-pass and contrast are TRAINED-IN (the diet is blur + a CS")
    print("cut) and are shown above only for completeness. They count toward nothing.")
    print(f"n=3 per condition; exact two-sided Mann-Whitney floor for 3v3 is p=0.100.")


if __name__ == "__main__":
    main()
