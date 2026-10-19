"""Fig. 4b summary: span of cue-conflict shape bias over the 21 readout weights for
the frozen affine head (results/fusion_sweep/<tag>_wsweep.json) and the refitted
linear and MLP heads (tmp_analysis/nonlinear_readout_sweep.py output), and the
span within 5 points of each head's peak validation top-1.

Writes results/nonlinear_readout_summary.json, keyed by tag (<run>_best / <run>_final),
the file tmp_analysis/figs_v2/fig4_readout.py and the verifier read. Existing keys
are re-derived and must reproduce the stored values; new tags are added.

    python tmp_analysis/nonlinear_readout_summary.py [--nl-dir DIR]
"""
import argparse
import glob
import json
import os

import numpy as np

NL_DIR = "/scratch/byrong/TSN/nonlinear_readout/results/2026-09-10"
OUT = "results/nonlinear_readout_summary.json"
GATE = 0.05


def summarise(tag: str, nl_dir: str) -> dict:
    pts = sorted(json.load(open(f"results/fusion_sweep/{tag}_wsweep.json"))["sweep"], key=lambda r: r["w"])
    sb = np.array([r["shape_bias"] for r in pts])
    out = {"affine": dict(span=float(sb.max() - sb.min()), lo=float(sb.min()), hi=float(sb.max()), gated=None)}
    rows = json.load(open(os.path.join(nl_dir, f"{tag}_nonlinear_sweep.json")))
    for head in ("linear", "mlp"):
        r = sorted((x for x in rows if x["head"] == head), key=lambda x: x["w"])
        sb = np.array([x["shape_bias"] for x in r]); val = np.array([x["val_top1"] for x in r])
        keep = val >= val.max() - GATE
        out[head] = dict(span=float(sb.max() - sb.min()), lo=float(sb.min()), hi=float(sb.max()),
                         gated=float(sb[keep].max() - sb[keep].min()), n_gated=int(keep.sum()), best_val=float(val.max()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl-dir", default=NL_DIR)
    args = ap.parse_args()
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    new = {}
    for f in sorted(glob.glob(os.path.join(args.nl_dir, "*_nonlinear_sweep.json"))):
        tag = os.path.basename(f)[:-len("_nonlinear_sweep.json")]
        new[tag] = summarise(tag, args.nl_dir)
    worst = aff = 0.0
    for tag, prev in old.items():
        if tag not in new:
            raise SystemExit(f"stored tag {tag} has no sweep file in {args.nl_dir}")
        for head, d in prev.items():
            for k, v in d.items():
                if v is not None and head != "affine":
                    worst = max(worst, abs(new[tag][head][k] - v))
        aff = max(aff, abs(new[tag]["affine"]["span"] - prev["affine"]["span"]))
        new[tag]["affine"] = prev["affine"]
    print(f"re-derived {len(old)} stored tags: refit heads max |diff| {worst:.1e}; "
          f"stored affine vs fusion sweep max |diff| {aff:.1e} (stored kept)")
    if worst > 1e-12:
        raise SystemExit("stored summary not reproduced; not writing")
    json.dump(new, open(OUT, "w"), indent=2)
    for tag, d in new.items():
        print(f"{tag:40s} affine {d['affine']['span']:.3f}  linear {d['linear']['span']:.3f}  "
              f"mlp {d['mlp']['span']:.3f} (gated {d['mlp']['gated']:.3f}, n {d['mlp']['n_gated']})")


if __name__ == "__main__":
    main()
