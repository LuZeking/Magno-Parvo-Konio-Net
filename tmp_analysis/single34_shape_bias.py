"""§64.6's leg-(b) comparison for the parameter-matched arm.

Protocol fixed in advance: 3 seeds each side, BOTH checkpoint policies,
complete separation + raw range rather than a p-value (3v3 floors at .10),
and the decision rate as a COUNT OVER A NAMED SUBSET, not a bare percentage --
its denominator (n_shape + n_texture) changes between arms, so a percentage
silently compares different subsets.
"""
import glob
import json
import os

ARMS = [
    ("two-stream  dkl_bcs      (~22.4M)", "results/cue_conflict_bestep",
     "results/cue_conflict", "dkl_bcs_seed*"),
    ("1-stream r18 single      (11.2M)", "results/cue_conflict_single_bestep",
     "results/cue_conflict_single", "dkl_bcs_single_seed*"),
    ("1-stream r34 single34    (21.3M)", "results/cue_conflict_single34_bestep",
     "results/cue_conflict_single34", "dkl_bcs_single34_seed*"),
]


def rows(d, pat):
    out = []
    for f in sorted(glob.glob(os.path.join(d, pat + ".json"))):
        if f.endswith("_logits.npz"):
            continue
        j = json.load(open(f))
        c = j.get("conditions", {}).get("full")
        if c is None:
            continue
        out.append((os.path.basename(f)[:-5], c))
    return out


for policy, idx in (("BEST SAVED CHECKPOINT", 1), ("FINAL CHECKPOINT", 2)):
    print(f"\n{'='*78}\n{policy}\n{'='*78}")
    print(f"{'arm / run':44s} {'shape_bias':>10s} {'decided':>9s} "
          f"{'shape':>6s} {'texture':>7s}")
    summary = {}
    for label, bd, fd, pat in ARMS:
        d = bd if idx == 1 else fd
        rr = rows(d, pat)
        if not rr:
            print(f"  {label:42s} -- no files in {d}")
            continue
        sbs = []
        for name, c in rr:
            dec = c["n_shape"] + c["n_texture"]
            sbs.append(c["shape_bias"])
            print(f"  {name:42s} {c['shape_bias']:10.4f} "
                  f"{dec:4d}/{c['n_images']:<4d} {c['n_shape']:6d} "
                  f"{c['n_texture']:7d}")
        summary[label] = sbs
    print(f"\n  {'-- ranges --':44s}")
    for label, sbs in summary.items():
        print(f"  {label:42s} n={len(sbs)} "
              f"[{min(sbs):.4f}, {max(sbs):.4f}]")
    ks = list(summary)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = summary[ks[i]], summary[ks[j]]
            sep = (min(a) > max(b)) or (min(b) > max(a))
            print(f"  separation {ks[i][:22]:24s} vs {ks[j][:22]:24s} -> "
                  f"{'COMPLETE' if sep else 'overlap'}")
