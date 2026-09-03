import numpy as np
from scipy import stats as sps
import itertools

# Seed-level trained PR ratios (subject-means), from job 26413641
SEED = {"diet/lum":    [1.603, 2.507, 1.823],
        "diet/highsf": [1.640, 1.497, 1.664],
        "diet/rg":     [2.225, 4.383, 1.995],
        "no diet":     [0.947, 0.988, 0.914]}
# diet/lum epoch range within seed0, from job 26417753
LUM_EPOCH = [1.752, 1.959, 1.733, 1.620, 1.703, 1.603]

print("=== CLAIM 3: is the swap ordering seed-robust, arm by arm? ===")
for other in ("diet/lum", "diet/highsf"):
    a, b = SEED["diet/rg"], SEED[other]
    print(f"\n  diet/rg {sorted(a)}  vs  {other} {sorted(b)}")
    print(f"    min(rg)={min(a):.3f}  max({other})={max(b):.3f}"
          f"  -> {'DISJOINT, gap %.3f' % (min(a)-max(b)) if min(a) > max(b) else 'OVERLAPPING'}")
    print(f"    {other} own seed range = {max(b)-min(b):.3f}")
    wins = sum(1 for x, y in itertools.product(a, b) if x > y)
    print(f"    rg > other in {wins}/9 seed pairings")
    # does it survive dropping rg's extreme seed?
    a2 = sorted(a)[:-1]
    print(f"    without rg's highest seed ({max(a):.3f}): rg {a2} vs max(other) {max(b):.3f}"
          f"  -> {'still disjoint' if min(a2) > max(b) else 'breaks'}")

print("\n=== §62.3's 'overlaps' sentence: check it ===")
lo, hi = min(LUM_EPOCH), max(LUM_EPOCH)
slo, shi = min(SEED["diet/rg"]), max(SEED["diet/rg"])
print(f"  diet/lum within-one-run epoch range : {lo:.3f} .. {hi:.3f}")
print(f"  diet/rg across-seed range           : {slo:.3f} .. {shi:.3f}")
print(f"  overlap? {'YES' if hi >= slo else 'NO -- adjacent, gap %.3f' % (slo-hi)}")

print("\n=== CLAIM 1: how big is the 0.037 gap against each noise source? ===")
gap = 0.037
seed_rng = max(SEED["diet/lum"]) - min(SEED["diet/lum"])
ep_rng = hi - lo
print(f"  gap                        : {gap:.3f}")
print(f"  seed range  (diet/lum)     : {seed_rng:.3f}  = {seed_rng/gap:.0f}x the gap")
print(f"  epoch range (diet/lum s0)  : {ep_rng:.3f}  = {ep_rng/gap:.0f}x the gap")
print(f"  highsf seed range          : {max(SEED['diet/highsf'])-min(SEED['diet/highsf']):.3f}"
      f"  = {(max(SEED['diet/highsf'])-min(SEED['diet/highsf']))/gap:.0f}x the gap")

print("\n=== CLAIM 2: compression under every checkpoint set and reference ===")
FRONT_U = {"diet/lum": 1.402, "diet/highsf": 0.586, "diet/rg": 1.909}
FRONT_G = {"diet/lum": 1.377, "diet/highsf": 0.563, "diet/rg": 3.366}
for label, arms in (("2 seed-0 arms (as published)", ["diet/lum", "diet/highsf"]),
                    ("3 arms, seed 0 only", ["diet/lum", "diet/highsf", "diet/rg"]),
                    ("3 arms, all 9 seeds", ["diet/lum", "diet/highsf", "diet/rg"])):
    allseeds = "all 9" in label
    for rn, F in (("untrained", FRONT_U), ("gabor", FRONT_G)):
        fs = max(F[a] for a in arms) - min(F[a] for a in arms)
        if allseeds:
            vals = [v for a in arms for v in SEED[a]]
        else:
            vals = [SEED[a][0] for a in arms]
        ts = max(vals) - min(vals)
        r = fs / ts if ts else float("inf")
        verd = "COMPRESSION %.1fx" % r if r > 1 else "EXPANSION %.1fx" % (1 / r)
        print(f"  {label:28s} ref={rn:9s} {fs:.3f} -> {ts:.3f}  {verd}")
