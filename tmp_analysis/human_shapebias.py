"""Model-vs-human shape bias on the Geirhos cue-conflict set.

Unblocks the item PAPER_PLAN §46.5 / PAPER_DRAFT "Claims deliberately not
made" left open: no human observer data was in the repo, so the word
"human-like" could not appear anywhere. It is now here.

Source: bethgelab/model-vs-human `raw-data/cue-conflict/` -- the standardized
release of Geirhos et al. 2018's human experiment, 10 observers x 1280 trials.
Downloaded to
/scratch/byrong/encoding/datasets/geirhos_cue_conflict/human_raw_data/
(scratch, next to the stimuli, per the Data Saving Rules).

Both sides use the IDENTICAL statistic and the identical exclusion, which is
what makes the comparison legitimate rather than rhetorical:
  * drop trials whose shape and texture category coincide (80 of 1280),
    leaving the canonical 1200;
  * among the rest count responses equal to the shape category and equal to
    the texture category, and report n_shape / (n_shape + n_texture).
`src/analysis/behavior/cue_conflict.py` already restricts model decisions to
the 16 categories and applies the same rule, so no re-scoring is needed --
the model numbers are read from its committed JSON.

Sanity anchor: the human mean reproduces the published ~0.96 to three
decimals, which is the check that the imagename parsing here is right.
"""
from __future__ import annotations
import csv
import glob
import json
import os

import numpy as np
from scipy import stats as sps

HUMAN = "/scratch/byrong/encoding/datasets/geirhos_cue_conflict/human_raw_data"
MODELS = "results/cue_conflict"


def _strip_digits(s: str) -> str:
    return "".join(c for c in s if not c.isdigit())


def human_observers() -> list[dict]:
    """Per-observer shape bias, plus the in-pair decision rate."""
    out = []
    for f in sorted(glob.glob(f"{HUMAN}/*.csv")):
        rows = list(csv.DictReader(open(f)))
        ns = nt = nother = 0
        for r in rows:
            pair = os.path.splitext(r["imagename"])[0].split("_")[-1]
            shp, tex = (_strip_digits(p) for p in pair.split("-"))
            if shp == tex:            # same-category pair, excluded
                continue
            resp = r["object_response"]
            if resp == shp:
                ns += 1
            elif resp == tex:
                nt += 1
            else:
                nother += 1
        n = ns + nt + nother
        out.append(dict(who=os.path.basename(f)[13:23], n_images=n,
                        n_shape=ns, n_texture=nt, n_other=nother,
                        shape_bias=ns / (ns + nt),
                        decided=(ns + nt) / n))
    return out


def model_rows() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(f"{MODELS}/*.json")):
        d = json.load(open(f))
        for cond, c in d["conditions"].items():
            rows.append(dict(model=d["model_name"], cond=cond,
                             shape_bias=c["shape_bias"],
                             ci=c.get("shape_bias_ci95"),
                             decided=(c["n_shape"] + c["n_texture"]) / c["n_images"],
                             n_images=c["n_images"]))
    return rows


def arm(model: str) -> str:
    if model.startswith("dkl_bcs_swap"):
        return "diet on rg (swap)"
    if model.startswith("dkl_bcs_parvo_highsf"):
        return "parvo_highsf"
    if model.startswith("dkl_bcs"):
        return "diet on lum"
    if model.startswith("dkl_base"):
        return "no diet"
    if model.startswith("rgb_tsn"):
        return "rgb (arch control)"
    return model


H = human_observers()
hsb = np.array([h["shape_bias"] for h in H])
hdec = np.array([h["decided"] for h in H])
n = len(hsb)
tcrit = sps.t.ppf(0.975, n - 1)
hse = hsb.std(ddof=1) / np.sqrt(n)

print("=" * 88)
print("HUMAN OBSERVERS (Geirhos et al. 2018, standardized release)")
print("=" * 88)
print(f"{'observer':12s} {'shape bias':>11s} {'in-pair decided':>16s}")
for h in H:
    print(f"{h['who']:12s} {h['shape_bias']:11.4f} {h['decided']:16.3f}")
print(f"\n  n={n}  mean={hsb.mean():.4f}  sd={hsb.std(ddof=1):.4f}"
      f"  95% CI {hsb.mean()-tcrit*hse:.4f}..{hsb.mean()+tcrit*hse:.4f}")
print(f"  in-pair decision rate: {hdec.mean():.3f} ± {hdec.std(ddof=1):.3f}")
print(f"  published value ~0.96 -> parsing check {'PASS' if abs(hsb.mean()-0.96)<0.01 else 'CHECK'}")

R = model_rows()
print("\n" + "=" * 88)
print("MODELS, same statistic and same 1200-image exclusion")
print("=" * 88)
print(f"{'arm':19s} {'model':30s} {'cond':11s} {'shape bias':>11s} {'gap to human':>13s} {'decided':>8s}")
by_arm: dict[tuple[str, str], list[float]] = {}
for r in sorted(R, key=lambda r: (arm(r["model"]), r["model"], r["cond"])):
    gap = r["shape_bias"] - hsb.mean()
    print(f"{arm(r['model']):19s} {r['model']:30s} {r['cond']:11s} "
          f"{r['shape_bias']:11.4f} {gap:+13.4f} {r['decided']:8.3f}")
    by_arm.setdefault((arm(r["model"]), r["cond"]), []).append(r["shape_bias"])

print("\n" + "=" * 88)
print("ARM MEANS vs HUMAN  (seeds pooled; one-sample t against the human mean)")
print("=" * 88)
print(f"{'arm':19s} {'cond':11s} {'k':>2s} {'shape bias':>11s} {'gap':>9s} {'inside human 95% CI':>21s}")
lo, hi = hsb.mean() - tcrit * hse, hsb.mean() + tcrit * hse
for (a, c), v in sorted(by_arm.items()):
    v = np.array(v)
    inside = "yes" if lo <= v.mean() <= hi else "no"
    print(f"{a:19s} {c:11s} {len(v):2d} {v.mean():11.4f} {v.mean()-hsb.mean():+9.4f} {inside:>21s}")

print("\n" + "=" * 88)
print("THE CAVEAT THAT MUST TRAVEL WITH ANY 'HUMAN-LIKE' SENTENCE")
print("=" * 88)
mdec = np.array([r["decided"] for r in R])
print(f"  humans decide in-pair on {hdec.mean()*100:.0f}% of cue-conflict trials;")
print(f"  these models on {mdec.mean()*100:.0f}% (range {mdec.min()*100:.0f}-{mdec.max()*100:.0f}%).")
print("  Shape bias is a ratio computed ONLY over decided trials, so the model")
print("  numbers condition on a much smaller and possibly non-random subset of")
print("  the same stimuli. A model and a human with equal shape bias are not")
print("  therefore behaving alike. Report the decision rate alongside, and do")
print("  not write that any model 'matches' human shape bias on the strength of")
print("  the ratio alone.")
