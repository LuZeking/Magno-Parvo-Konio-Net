"""Does the confirmatory family actually survive Holm-Bonferroni? It does not.

The draft's new Multiplicity paragraph claims:

    "[the confirmatory family] comprises 18 tests declared before the
     corresponding analyses; Holm-Bonferroni across these leaves every claim we
     make significant, because each rests on the exact Wilcoxon floor p=.0078
     at n=8 and 18 x .0078 = .140 is the largest adjusted value only for the
     single least extreme test, which we do not claim."

That is arithmetically false, and in an instructive way. The exact two-sided
Wilcoxon signed-rank test at n=8 has a *floor* -- a smallest attainable p-value,
2/2^8 = .0078125, reached when all eight subjects move the same way. A
multiplicity correction compares that floor against alpha/m. Once m is large
enough that alpha/m drops below the floor, NO result can reach corrected
significance, however large the effect: the test has run out of resolution
before the correction has run out of penalty.

At m=18 and alpha=.05 the Holm threshold for the very first (smallest) p-value
is .05/18 = .00278, which is below the .0078 floor. So Holm rejects nothing --
not "everything except the least extreme test", but nothing at all.

This module computes that rather than asserting it, and reports the largest
family size at which an n=8 exact test could survive.

    PYTHONPATH=. python tmp_analysis/multiplicity_check.py
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats as sps

ALPHA = 0.05
N_SUBJ = 8
M_FAMILY = 18   # the draft's own stated size for the confirmatory family


def exact_wilcoxon_floor(n: int) -> float:
    """Smallest attainable two-sided exact signed-rank p at this n.

    Computed from the null distribution rather than quoted, so the number is
    checked rather than remembered.
    """
    ranks = np.arange(1, n + 1)
    totals = []
    for signs in itertools.product([-1, 1], repeat=n):
        totals.append(float(np.sum(ranks * np.array(signs))))
    totals = np.array(totals)
    # W+ statistic: sum of positive ranks. Extreme case = all one direction.
    wplus = np.array([
        float(np.sum(ranks[np.array(s) > 0]))
        for s in itertools.product([-1, 1], repeat=n)
    ])
    n_perm = len(wplus)
    most_extreme = wplus.max()
    # two-sided: both tails
    p = (np.sum(wplus >= most_extreme) + np.sum(wplus <= wplus.min())) / n_perm
    return float(p)


def holm(pvals: list[float], alpha: float = ALPHA) -> list[tuple[float, float, bool]]:
    """Holm-Bonferroni step-down. Returns (p, adjusted p, reject)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)          # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return [(pvals[i], float(adj[i]), bool(adj[i] <= alpha)) for i in range(m)]


def main() -> int:
    print("=" * 78)
    print("MULTIPLICITY: can the confirmatory family survive Holm-Bonferroni?")
    print("=" * 78)

    floor = exact_wilcoxon_floor(N_SUBJ)
    # cross-check against scipy on an all-same-sign vector
    d = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.05, 0.4, 0.35])
    sp = sps.wilcoxon(d, alternative="two-sided", mode="exact").pvalue
    print(f"\n  exact two-sided signed-rank floor at n={N_SUBJ}")
    print(f"    enumerated over 2^{N_SUBJ} sign flips : {floor:.7f}")
    print(f"    scipy, all-same-sign vector          : {sp:.7f}")
    print(f"    2/2^{N_SUBJ}                              : {2 / 2**N_SUBJ:.7f}")
    assert abs(floor - sp) < 1e-12, (floor, sp)
    assert abs(floor - 2 / 2**N_SUBJ) < 1e-12

    print(f"\n  Holm's threshold for the SMALLEST p in a family of m:")
    print(f"    alpha/m = {ALPHA}/{M_FAMILY} = {ALPHA / M_FAMILY:.7f}")
    print(f"    floor   = {floor:.7f}")
    verdict = "BELOW the floor -> nothing can be rejected" \
        if ALPHA / M_FAMILY < floor else "above the floor -> rejection possible"
    print(f"    {verdict}")

    # The draft's stated family: 18 tests, each at the floor.
    fam = [floor] * M_FAMILY
    res = holm(fam)
    n_rej = sum(r[2] for r in res)
    print(f"\n  Family as the draft describes it: {M_FAMILY} tests, each at the floor")
    print(f"    Holm-adjusted p (all tied)  : {res[0][1]:.4f}")
    print(f"    rejected at alpha={ALPHA}       : {n_rej} of {M_FAMILY}")
    print(f"    Bonferroni {M_FAMILY} x {floor:.6f}   : {M_FAMILY * floor:.4f}")

    print(f"\n  Largest family in which an n={N_SUBJ} exact test can survive:")
    m_max = int(np.floor(ALPHA / floor))
    print(f"    m <= alpha/floor = {ALPHA}/{floor:.7f} = {ALPHA / floor:.2f}"
          f"  ->  m_max = {m_max}")
    for m in (6, 7, 18):
        print(f"    m={m:2d}: alpha/m = {ALPHA / m:.5f}  "
              f"{'survives' if ALPHA / m >= floor else 'CANNOT survive'}")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    print(f"""
  The draft's claim -- that Holm leaves every claim significant and that
  .140 applies "only for the single least extreme test" -- is wrong twice.

  1. With all {M_FAMILY} tests tied at the floor, Holm assigns EVERY one an
     adjusted p of {res[0][1]:.3f}, not just the least extreme. Ties propagate
     through the step-down; the largest multiplier attaches to the SMALLEST
     p-value, which is the opposite of what the sentence says.

  2. Holm therefore rejects {n_rej} of {M_FAMILY}. The family does not survive.

  This is structural, not a defect of these data: an n={N_SUBJ} exact test cannot
  produce a p below {floor:.5f}, so any family larger than m={m_max} is
  unreachable at alpha={ALPHA} no matter how large the effect. Even a perfect
  8/8 separation with a huge effect size returns exactly {floor:.5f}.

  The remedy is NOT to correct the claims into significance, and not to shrink
  the family post hoc to make it fit. It is to say plainly what the paper
  already half-says at the end of that same paragraph: at n=8 the exact test
  is resolution-limited, so the evidence is complete subject separation (8/8)
  and the effect size, reported alongside every p-value -- and a corrected
  p-value is not informative in this regime. That is honest and defensible;
  the current sentence is checkable and false.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
