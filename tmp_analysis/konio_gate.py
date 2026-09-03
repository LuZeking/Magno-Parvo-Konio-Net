"""The §65.4 gate, standalone: r2_magno must be identical across the konio and
parvo arms.

r2_magno is a RidgeCV fit on the magno band ALONE and prepare_band uses only
that band's own statistics, so with a bit-identical magno feature matrix
(proved by smoke_konio.py) it cannot legitimately differ. Any difference means
something that was supposed to be held fixed was not, and per §65.4 NOTHING
else in the arm may be interpreted until it is explained.

This is what caught the stale pre-e3b8667 reference fits for subjects 4 and 8
(§67). Run it after every konio cell lands, not once at the end.

    python -m tmp_analysis.konio_gate [konio_model_name]
"""
import os
import sys

import numpy as np

R = "results/encoding"
REF = "untrained_seed0"
PASS, WARN = 1e-4, 1e-3


def main(model: str = "untrained_konio_matched") -> int:
    print(f"gate: {model} vs {REF}   (pass < {PASS:g}, invalid > {WARN:g})\n")
    print(f"{'subj':>4} {'nvox':>6} {'max|d|':>12} {'mean|d|':>11}  verdict")
    worst, missing, bad = 0.0, [], []
    for s in range(1, 9):
        fk = f"{R}/{model}_subj{s:02d}.npz"
        fp = f"{R}/{REF}_subj{s:02d}.npz"
        if not (os.path.exists(fk) and os.path.exists(fp)):
            missing.append(s)
            print(f"{s:>4} {'-':>6} {'(pending)':>12}")
            continue
        zk, zp = np.load(fk, allow_pickle=True), np.load(fp, allow_pickle=True)
        # Same voxels and same data, or the comparison is meaningless.
        assert zk["noise_ceiling"].shape == zp["noise_ceiling"].shape, f"subj{s}: voxel count differs"
        assert np.array_equal(zk["noise_ceiling"], zp["noise_ceiling"]), f"subj{s}: noise ceiling differs -- different betas/trials"
        d = np.abs(zk["r2_magno"] - zp["r2_magno"])
        m = float(d.max())
        worst = max(worst, m)
        v = "PASS" if m < PASS else ("WARN" if m < WARN else "INVALID")
        if v != "PASS":
            bad.append(s)
        print(f"{s:>4} {d.size:>6} {m:>12.3e} {d.mean():>11.3e}  {v}")

    print()
    if missing:
        print(f"INCOMPLETE: subjects {missing} not yet on disk.")
    if bad:
        print(f"GATE FAILED for subjects {bad} (worst {worst:.3e}).")
        print("Per §65.4 nothing else in the arm may be interpreted. See §67.")
        return 1
    if missing:
        return 2
    print(f"GATE PASSED on all 8 subjects (worst {worst:.3e}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
