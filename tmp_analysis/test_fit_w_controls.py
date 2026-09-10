"""Exercise ModelDecisions / Controls A and B on synthetic decisions.

No model, no forward pass. Checks the matmul kappa against the VALIDATED
scalar kappa under every weighting the script actually uses: unweighted,
bootstrap multiplicities, a cross-validation half mask, and a permuted
observer. A shape or indexing bug in the new control code shows up here in
seconds instead of after nine 13-minute forward passes.
"""
import numpy as np
from sweep_error_consistency import kappa as scalar_kappa
from fit_w_observers import (ModelDecisions, obs_indicators, nanargmax_cols,
                             SHAPE, TEXTURE, NEITHER)

rng = np.random.default_rng(7)
n_w, n_items = 40, 1200

# Correlated model/observer so kappa is not degenerate, with a realistic
# fraction of "decided neither" on both sides.
truth = rng.integers(0, 2, n_items)
M = np.empty((n_w, n_items), dtype=np.int8)
for i in range(n_w):
    flip = rng.random(n_items) < (0.1 + 0.6 * i / n_w)
    d = np.where(flip, 1 - truth, truth)
    d = np.where(rng.random(n_items) < 0.15, NEITHER, d)
    M[i] = d
h = np.where(rng.random(n_items) < 0.2, 1 - truth, truth)
h = np.where(rng.random(n_items) < 0.10, NEITHER, h).astype(np.int8)

md = ModelDecisions(M)


def to_ch(v):
    return [None if x == NEITHER else ("S" if x == SHAPE else "T") for x in v]


# --- 1. unweighted, every w ------------------------------------------------
k_mm = md.kappa_against(*obs_indicators(h))
err = max(abs(float(scalar_kappa(to_ch(M[i]), to_ch(h))) - float(k_mm[i]))
          for i in range(n_w))
print(f"unweighted           max |d kappa| = {err:.3e}")
assert err < 1e-9, err

# --- 2. a cross-validation half mask ---------------------------------------
mask = np.zeros(n_items, dtype=np.float32)
mask[rng.choice(n_items, n_items // 2, replace=False)] = 1.0
k_half = md.kappa_against(*obs_indicators(h, mask))
sel = mask.astype(bool)
err = max(abs(float(scalar_kappa(to_ch(M[i][sel]), to_ch(h[sel])))
              - float(k_half[i])) for i in range(n_w))
print(f"half mask            max |d kappa| = {err:.3e}")
assert err < 1e-9, err

# --- 3. bootstrap multiplicities (integer weights == repeated items) -------
mult = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items)).astype(np.float32)
k_boot = md.kappa_against(*obs_indicators(h, mult[:, None]))[:, 0]
rep = np.repeat(np.arange(n_items), mult.astype(int))
err = max(abs(float(scalar_kappa(to_ch(M[i][rep]), to_ch(h[rep])))
              - float(k_boot[i])) for i in range(n_w))
print(f"bootstrap weights    max |d kappa| = {err:.3e}")
assert err < 1e-9, err

# --- 4. permuted observer, batched over columns ----------------------------
P = 8
perm = np.stack([h[rng.permutation(n_items)] for _ in range(P)], axis=1)
k_perm = md.kappa_against((perm == SHAPE).astype(np.float32),
                          (perm == TEXTURE).astype(np.float32),
                          (perm != NEITHER).astype(np.float32))
err = 0.0
for q in range(P):
    ref = np.array([scalar_kappa(to_ch(M[i]), to_ch(perm[:, q]))
                    for i in range(n_w)], dtype=float)
    err = max(err, float(np.nanmax(np.abs(ref - k_perm[:, q]))))
print(f"permuted, batched    max |d kappa| = {err:.3e}   shape {k_perm.shape}")
assert err < 1e-9, err

# --- 5. the CV indexing the script uses, done the slow way -----------------
R = 12
SA = np.zeros((n_items, R), dtype=np.float32)
for r in range(R):
    SA[rng.choice(n_items, n_items // 2, replace=False), r] = 1.0
SB = 1.0 - SA
kA = md.kappa_against(*obs_indicators(h, SA))
kB = md.kappa_against(*obs_indicators(h, SB))
idx = nanargmax_cols(kA)
held = kB[idx, np.arange(R)]
for r in range(R):
    a_sel, b_sel = SA[:, r].astype(bool), SB[:, r].astype(bool)
    ref_A = np.array([scalar_kappa(to_ch(M[i][a_sel]), to_ch(h[a_sel]))
                      for i in range(n_w)], dtype=float)
    best = int(np.nanargmax(ref_A))
    assert best == idx[r], f"split {r}: argmax {idx[r]} != {best}"
    ref_B = scalar_kappa(to_ch(M[best][b_sel]), to_ch(h[b_sel]))
    assert abs(float(ref_B) - float(held[r])) < 1e-9, r
print(f"CV fit-on-A/score-on-B agrees on all {R} splits "
      f"(held-out mean {np.nanmean(held):+.4f} vs in-sample "
      f"{np.nanmean(kA[idx, np.arange(R)]):+.4f})")

# --- 6. rate() against a direct count --------------------------------------
sb, dc = md.rate()
for i in (0, n_w // 2, n_w - 1):
    ns, nt = int((M[i] == SHAPE).sum()), int((M[i] == TEXTURE).sum())
    assert abs(sb[i] - ns / (ns + nt)) < 1e-12 and int(dc[i]) == ns + nt
print("rate() matches direct counts")

print("\nALL CONTROL-CODE CHECKS PASSED")
