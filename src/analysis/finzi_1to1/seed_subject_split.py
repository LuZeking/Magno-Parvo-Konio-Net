"""Seed vs subject variance of the training x diet contrasts (NOTE_TRAINING_X_DIET_2026-09-27).

    python -m src.analysis.finzi_1to1.seed_subject_split

Post hoc. Reads the idea-1 mapping outputs (NC-stratified dDV, plan §5) per arm x seed x
subject and splits each contrast into subject means, seed means and the seed x subject
residual. Seeds of a trained arm and of its untrained control are paired by index only;
the pairing is arbitrary (no epoch-0 checkpoint shows they share an initialisation).
"""

from __future__ import annotations

import numpy as np

from src.analysis.finzi_1to1.stats import ARMS, SUBJECTS, decide, per_subject


def values(arm: str) -> np.ndarray:
    """(seeds, subjects) dDV_strat."""
    return np.stack([per_subject(n, "main", "draws", "dDV_strat") for n in ARMS[arm]])


def split(x: np.ndarray) -> dict:
    subj, seed = x.mean(0), x.mean(1)
    res = x - subj[None] - seed[:, None] + x.mean()
    df = (x.shape[0] - 1) * (x.shape[1] - 1)
    return dict(seed_means=seed.round(3).tolist(), subject_means=subj.round(3).tolist(),
                sd_subject=round(float(subj.std(ddof=1)), 3),
                sd_seed=round(float(seed.std(ddof=1)), 3),
                sd_residual=round(float(np.sqrt((res ** 2).sum() / df)), 3))


def main() -> None:
    v = {a: values(a) for a in ("diet", "nodiet", "untrained_diet", "untrained_nodiet")}
    contrasts = {
        "T_diet = diet - untrained_diet": v["diet"] - v["untrained_diet"],
        "T_nodiet = nodiet - untrained_nodiet": v["nodiet"] - v["untrained_nodiet"],
        "untrained diet - untrained nodiet": v["untrained_diet"] - v["untrained_nodiet"],
        "interaction = T_diet - T_nodiet": (v["diet"] - v["untrained_diet"])
                                           - (v["nodiet"] - v["untrained_nodiet"]),
    }
    for name, x in {**{a: v[a] for a in v}, **contrasts}.items():
        d, s = decide(x.mean(0)), split(x)
        print(f"\n{name}")
        print(f"  subject-level: mean {d['mean']:+.3f} ci90 [{d['ci90'][0]:+.3f}, "
              f"{d['ci90'][1]:+.3f}] {d['n_positive']}/{len(SUBJECTS)} > 0 "
              f"p {d['wilcoxon_p']:.4f} -> {d['label']}")
        print(f"  seed means {s['seed_means']}")
        print(f"  subject means {s['subject_means']}")
        print(f"  SD subject {s['sd_subject']}  SD seed {s['sd_seed']}  "
              f"SD seed x subject {s['sd_residual']}")
        if name in contrasts:
            print("  per seed x subject:\n  " + str(x.round(3)).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
