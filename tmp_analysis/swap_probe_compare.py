"""Compare the converged swap-seed shape/texture probe against the quarantined
mid-training run, and against the diet / no-diet cells.

The quarantined files (results/shape_texture/partial_invalid/) were read at
epoch 33/150 and compared against converged 150-epoch models, which confounded
condition with training length (§10.1). Job 26399145 re-ran them at 150/150.

Usage: PYTHONPATH=. python tmp_analysis/swap_probe_compare.py
"""

from __future__ import annotations

import json
import os

import numpy as np

SETS = {
    "untrained, no diet":   ["untrained_noblur"],
    "untrained, with diet": ["untrained_blur_cs"],
    "trained, no diet":     [f"dkl_base_seed{s}_trained" for s in (0, 1, 2)],
    "trained, with diet":   [f"dkl_bcs_seed{s}_trained" for s in (0, 1, 2)],
    "trained, SWAP (converged)":
        [f"dkl_bcs_swap_seed{s}_trained" for s in (0, 1)],
    "trained, SWAP (quarantined, ep33)":
        [f"partial_invalid/dkl_bcs_swap_seed{s}_trained" for s in (0, 1)],
    "thesis dvd_trans":     ["thesis_dvd_trans"],
    "thesis s1_90":         ["thesis_s1_90"],
}

print(f"{'cell':<36} {'magno':>9} {'parvo':>9} {'magno-parvo':>12}  n")
vals = {}
for lab, names in SETS.items():
    m, p = [], []
    for n in names:
        fp = f"results/shape_texture/{n}.json"
        if not os.path.exists(fp):
            print(f"{lab:<36} MISSING {fp}")
            continue
        d = json.load(open(fp))["streams"]
        m.append(d["magno"]["bias"])
        p.append(d["parvo"]["bias"])
    if not m:
        continue
    vals[lab] = (float(np.mean(m)), float(np.mean(p)))
    print(f"{lab:<36} {np.mean(m):>+9.4f} {np.mean(p):>+9.4f} "
          f"{np.mean(m)-np.mean(p):>+12.4f}  {len(m)}")

# The conjunction test, on the probe rather than on behaviour (§20.1 did this
# for cue-conflict). stream1 always carries luminance, stream2 red-green;
# transform_target only moves the diet transform between them. So the delta
# for the stream that RECEIVED the transform isolates the carrier.
print("\n--- conjunction test: delta for the stream that received the diet ---")
base_m, base_p = vals["trained, no diet"]
if "trained, with diet" in vals:
    d_m, _ = vals["trained, with diet"]
    print(f"transform on luminance (magno): {base_m:+.4f} -> {d_m:+.4f}  "
          f"delta {d_m-base_m:+.4f}")
if "trained, SWAP (converged)" in vals:
    _, s_p = vals["trained, SWAP (converged)"]
    print(f"transform on red-green (parvo): {base_p:+.4f} -> {s_p:+.4f}  "
          f"delta {s_p-base_p:+.4f}")
