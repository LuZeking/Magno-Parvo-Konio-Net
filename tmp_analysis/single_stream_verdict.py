"""The §64.5 verdict, coded mechanically from the pre-registered thresholds.

§50.6's lesson is binding: a pre-registered rule that cannot fail launders a
refutation as a confirmation. So the thresholds are transcribed here verbatim
and applied without judgement, and the two places where §64.5 turns out to be
under-specified are reported rather than silently resolved:

1. **Which accuracy reference.** §64.4 declared 0.5921 for `dkl_bcs`, which
   §71.1 later showed is a per-epoch validation maximum naming weights that do
   not exist. §71.4 now forbids pairing it with a behavioural number. The
   best-saved-checkpoint value is 0.5562. Both are applied; if the grade
   differs between them that is reported as a conflict, not resolved here.

2. **A and D are not disjoint as written.** A requires shape bias >= 0.6749;
   D requires >= 0.85, which also satisfies A. §64.5 asserts the outcomes are
   mutually exclusive, so this is a defect in the pre-registration. D is the
   more specific threshold and takes precedence; the overlap is printed.

    python -m tmp_analysis.single_stream_verdict
"""
import glob
import json
import os
import sys

import numpy as np

# --- §64.4 reference values, fixed before the run --------------------------
SB_INTEGRATED_FINAL = 0.6749   # dkl_bcs integrated readout, final ckpt
SB_MAGNO_ALONE = 0.8778        # dkl_bcs magno stream alone, final ckpt
ACC_BCS_PEREPOCH = 0.5921      # §64.4 as declared -- a SCHEDULE property.
# §71.4: never use it as a reference beside a behavioural number. Kept here
# only so the superseded figure stays visible rather than being deleted.
ACC_BCS_BESTSAVED = 0.5562     # §71.1 correction, weights that exist
ACC_BCS_FINAL = 0.4446         # §64.4, dkl_bcs top-1 at the final checkpoint
# dkl_bcs integrated readout at best saved ckpt, measured (results/cue_conflict_bestep)
SB_INTEGRATED_BESTSAVED = 0.6547
ACC_TOL = 0.02                 # §64.5's declared band
D_THRESHOLD = 0.85             # §64.5 outcome D

RUNS = "/scratch/byrong/VST_data/mpk_runs"


def load_single(d: str) -> dict[int, float]:
    """seed -> integrated shape bias, from a results dir."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "dkl_bcs_single_seed*.json"))):
        r = json.load(open(f))
        seed = int(os.path.basename(f).split("seed")[1][0])
        out[seed] = float(r["conditions"]["full"]["shape_bias"])
    return out


def acc_single(policy: str) -> dict[int, float]:
    """seed -> val_top1 at the weight file actually scored."""
    out = {}
    for s in range(3):
        h = json.load(open(f"{RUNS}/dkl_bcs_single_seed{s}/history.json"))
        v = [r["val_top1"] for r in h]
        if policy == "final":
            out[s] = float(v[149])
        else:
            avail = [n for n in (25, 50, 75, 100, 125, 150)
                     if os.path.exists(
                         f"{RUNS}/dkl_bcs_single_seed{s}/checkpoint_{n:03d}.pt")]
            best = max(avail, key=lambda n: v[n - 1])
            out[s] = float(v[best - 1])
    return out


def grade(sb: float, acc: float, acc_ref: float, sb_ref: float,
          sb_sd: float) -> tuple[str, list[str]]:
    """Apply §64.5 verbatim. Returns (code, notes)."""
    notes = []
    d_acc = acc - acc_ref
    loses_acc = d_acc < -ACC_TOL
    ties_acc = abs(d_acc) <= ACC_TOL

    if sb >= D_THRESHOLD:
        if sb >= sb_ref:
            notes.append(
                "A's shape-bias clause is also satisfied (>= %.4f); D is the "
                "more specific threshold and takes precedence." % sb_ref)
        return "D", notes

    # C is checked before A/B: §64.5 makes an accuracy loss disqualifying.
    if loses_acc:
        return "C", notes

    if sb >= sb_ref and ties_acc:
        return "A", notes

    if sb < sb_ref - sb_sd and ties_acc:
        return "B", notes

    # Report WHICH clause failed. Naming the wrong one would misdescribe how
    # the pre-registration behaved, which is the §50.6 failure mode.
    if sb < sb_ref - sb_sd:
        notes.append(
            "B's SHAPE-BIAS clause IS met: %.4f is below %.4f by %.4f, more "
            "than the between-seed sd (%.4f)."
            % (sb, sb_ref, sb_ref - sb, sb_sd))
        notes.append(
            "B fails ONLY on its accuracy clause: delta = %+.4f, outside "
            "+-%.2f." % (d_acc, ACC_TOL))
    else:
        notes.append(
            "shape bias %.4f is below %.4f but not by more than the "
            "between-seed sd (%.4f), so B's shape clause is not met."
            % (sb, sb_ref, sb_sd))
    if d_acc > ACC_TOL:
        notes.append(
            "accuracy is %+.4f, ABOVE the reference and outside the band in "
            "the WINNING direction. §64.3 pre-committed that a win with half "
            "the capacity makes the confound irrelevant, so this is not a "
            "reason to discount the arm -- but §64.5 wrote B's clause as a "
            "two-sided band and so cannot fire. The band is the defect."
            % d_acc)
    return "UNCLASSIFIED", notes


def main() -> int:
    cells = [
        ("best saved ckpt", "results/cue_conflict_single_bestep", "best",
         SB_INTEGRATED_BESTSAVED, ACC_BCS_BESTSAVED),
        # Each policy pairs the shape bias with the accuracy of the SAME
        # weight file (§71.4). The final-checkpoint cell must therefore use
        # dkl_bcs's final-checkpoint top-1 (0.4446), NOT the 0.5921 per-epoch
        # maximum §64.4 declared -- pairing those two is the §71.2 error.
        ("final ckpt", "results/cue_conflict_single", "final",
         SB_INTEGRATED_FINAL, ACC_BCS_FINAL),
    ]
    rc = 0
    for label, d, policy, sb_ref, acc_ref in cells:
        print(f"\n{'=' * 72}\n{label}   ({d})\n{'=' * 72}")
        sb = load_single(d)
        if len(sb) < 3:
            print(f"INCOMPLETE: {len(sb)}/3 seeds on disk -- not graded.")
            rc = max(rc, 2)
            continue
        acc = acc_single(policy)
        sb_v = np.array([sb[s] for s in sorted(sb)])
        acc_v = np.array([acc[s] for s in sorted(acc)])
        sb_sd = float(sb_v.std(ddof=1))

        print(f"{'seed':>5} {'shape_bias':>11} {'val_top1':>10}")
        for s in sorted(sb):
            print(f"{s:>5} {sb[s]:>11.4f} {acc[s]:>10.4f}")
        print(f"{'mean':>5} {sb_v.mean():>11.4f} {acc_v.mean():>10.4f}"
              f"   sd(sb)={sb_sd:.4f}")
        print(f"\nreferences: dkl_bcs integrated sb={sb_ref:.4f}, "
              f"top-1={acc_ref:.4f}, tol=+-{ACC_TOL}")
        print(f"delta: sb={sb_v.mean() - sb_ref:+.4f}  "
              f"acc={acc_v.mean() - acc_ref:+.4f}")

        code, notes = grade(sb_v.mean(), acc_v.mean(), acc_ref, sb_ref, sb_sd)
        names = {"A": "THE SPLIT BUYS NOTHING",
                 "B": "THE SPLIT HELPS BEHAVIOUR",
                 "C": "THE SPLIT IS A CAPACITY STORY",
                 "D": "THE SPLIT ACTIVELY HURTS LEG (b)",
                 "UNCLASSIFIED": "no pre-registered outcome matches"}
        print(f"\n  OUTCOME {code}: {names[code]}")
        for n in notes:
            print(f"  note: {n}")

        # Complete separation, per §64.6 item 3 (3v3 floors at p=0.10).
        print(f"\n  per-seed sb range [{sb_v.min():.4f}, {sb_v.max():.4f}] "
              f"vs reference {sb_ref:.4f}: "
              f"{'ALL above' if (sb_v > sb_ref).all() else ('ALL below' if (sb_v < sb_ref).all() else 'OVERLAP')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
