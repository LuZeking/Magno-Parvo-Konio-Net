"""Apply the registered decision rule and gates (plan §5-§8) to the mapping outputs.

    python -m src.analysis.finzi_1to1.stats --out results/finzi_1to1

Aggregation (plan §5): mean over draws -> mean over hemispheres -> mean over an
arm's seeds, per subject; subject is the unit of inference (n = 8).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats as st

from src.analysis.finzi_1to1.mapping import out_stem

DELTA = 0.05
SUBJECTS = list(range(1, 9))
ARMS = {
    "diet": [f"diet_s{k}" for k in range(3)],
    "nodiet": [f"nodiet_s{k}" for k in range(3)],
    "swap": [f"swap_s{k}" for k in range(3)],
    "untrained_diet": [f"untrained_diet_s{k}" for k in range(3)],
    "untrained_nodiet": [f"untrained_nodiet_s{k}" for k in range(3)],
}
KEYS = ["dDV_strat", "dLV_strat", "dDV_raw", "dLV_raw", "share_all",
        "share_ventral", "share_lateral", "share_dorsal",
        "fr_ventral", "fr_lateral", "fr_dorsal",
        "fr_ventral_A", "fr_ventral_B", "fr_lateral_A", "fr_lateral_B",
        "fr_dorsal_A", "fr_dorsal_B"]


def per_subject(network: str, mode: str, field: str, key: str) -> np.ndarray:
    """(8,) mean over draws then hemispheres; NaN where the file is missing."""
    out = np.full(len(SUBJECTS), np.nan)
    for i, s in enumerate(SUBJECTS):
        path = f"{out_stem(network, s)}__{mode}.json"
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))[field]
        hemis = [np.nanmean([r[key] for r in rows if r["hemi"] == h]) for h in (0, 1)]
        out[i] = np.nanmean(hemis)
    return out


def arm_values(arm: str, key: str, field: str = "draws") -> np.ndarray:
    return np.nanmean(np.stack([per_subject(n, "main", field, key) for n in ARMS[arm]]), 0)


def decide(x: np.ndarray, delta: float = DELTA) -> dict:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"label": "MISSING", "n": n}
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    tq = float(st.t.ppf(0.95, n - 1))
    lo, hi = mean - tq * se, mean + tq * se
    p = float(st.wilcoxon(x, method="exact").pvalue) if np.any(x != 0) else 1.0
    if p < 0.05 and abs(mean) >= delta:
        label = "POSITIVE (A -> first stream)" if mean > 0 else "NEGATIVE (A -> second stream)"
    elif p < 0.05:
        label = "SMALL"
    elif -delta < lo and hi < delta:
        label = "NO PREFERENCE"
    else:
        label = "UNRESOLVED"
    return {"label": label, "n": n, "mean": mean, "ci90": [lo, hi], "wilcoxon_p": p,
            "n_positive": int((x > 0).sum()), "values": x.round(4).tolist()}


def gate_pc(x: np.ndarray) -> dict:
    d = decide(x)
    ok = (d.get("n_positive", 0) >= 7 and d.get("wilcoxon_p", 1) < 0.05
          and d.get("mean", 0) >= 2 * DELTA)
    return dict(d, gate="PASS" if ok else "FAIL")


def gate_neg(x: np.ndarray) -> dict:
    d = decide(x)
    ok = "ci90" in d and -DELTA < d["ci90"][0] and d["ci90"][1] < DELTA
    return dict(d, gate="PASS" if ok else "FAIL")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/finzi_1to1")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R: dict = {"delta": DELTA, "gates": {}, "primary": {}, "secondary": {}, "descriptive": {}}

    for net, name in (("tdann_a025_s0", "PC-TDANN"), ("diet_s0", "PC-TSN")):
        R["gates"][name] = gate_pc(per_subject(net, "pc", "pc", "dDV_strat"))
        # the registered mean-r grouping, reported beside the amended p90 gate
        R["gates"][name + " (registered mean-r groups, descriptive)"] = gate_pc(
            per_subject(net, "pc", "pc_meanr", "dDV_strat"))
    R["gates"]["NEG"] = gate_neg(per_subject("diet_s0", "pc", "neg", "dDV_strat"))
    R["gates"]["NEG-TDANN (descriptive)"] = decide(per_subject("tdann_a025_s0", "pc", "neg", "dDV_strat"))

    R["primary"]["diet dDV_strat"] = decide(arm_values("diet", "dDV_strat"))
    for arm in ARMS:
        for key in ("dDV_strat", "dLV_strat", "dDV_raw", "dLV_raw"):
            if arm == "diet" and key == "dDV_strat":
                continue
            R["secondary"][f"{arm} {key}"] = decide(arm_values(arm, key))
    for key in ("dDV_strat", "dLV_strat"):
        R["secondary"][f"diet - untrained_diet {key}"] = decide(
            arm_values("diet", key) - arm_values("untrained_diet", key))
        R["secondary"][f"diet - nodiet {key}"] = decide(
            arm_values("diet", key) - arm_values("nodiet", key))
    for arm in ARMS:
        R["descriptive"][arm] = {
            k: float(np.nanmean(arm_values(arm, k))) for k in KEYS}
        R["descriptive"][arm + " (unforced)"] = {
            k: float(np.nanmean(arm_values(arm, k, "unforced"))) for k in KEYS}
    for net in ("tdann_a025_s0", "tdann_a0_s0"):
        R["descriptive"][net + " (Finzi all-units)"] = {
            k: float(np.nanmean(per_subject(net, "main", "finzi_all_units", k)))
            for k in ("fr_ventral", "fr_lateral", "fr_dorsal")}

    g = R["gates"]
    if g["PC-TDANN"]["gate"] == "FAIL" or g["NEG"]["gate"] == "FAIL":
        verdict = "PIPELINE NOT TRUSTED (PC-TDANN or NEG failed): no TSN result interpreted"
    elif g["PC-TSN"]["gate"] == "FAIL":
        verdict = ("TSN layer-4 units carry no subject-replicable stream preference on this "
                   "instrument; primary reported descriptively, no NO PREFERENCE label")
    else:
        verdict = f"primary label: {R['primary']['diet dDV_strat']['label']}"
    R["verdict"] = verdict
    with open(os.path.join(args.out, "finzi_1to1_stats.json"), "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps({"verdict": verdict, "gates": {k: v.get("gate", v.get("label"))
                                                    for k, v in g.items()},
                      "primary": R["primary"]}, indent=1))


if __name__ == "__main__":
    main()
