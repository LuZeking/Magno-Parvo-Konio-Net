"""Sanity-check the 8 completed subjects before group inference."""

import numpy as np

z = np.load("results/encoding/untrained_seed0_subj01.npz", allow_pickle=True)
nc = z["noise_ceiling"]
raw = z["unique_magno_raw"]
norm = z["unique_magno"]
ok = (nc > 0.1) & (raw != 0)
ratio = (norm[ok] / raw[ok]) * nc[ok]
print("subj01: median (unique_magno/raw)*nc =", round(float(np.nanmedian(ratio)), 4),
      " -> 1.0 means unique_* is raw/nc")

print()
print(f"{'subj':6s}{'voxels':>8s}{'reliable':>10s}{'medR2j/nc':>11s}"
      f"{'medNC':>8s}")
for s in range(1, 9):
    zz = np.load(f"results/encoding/untrained_seed0_subj{s:02d}.npz",
                 allow_pickle=True)
    m = zz["reliable_mask"]
    print(f"{s:<6d}{zz['noise_ceiling'].size:>8d}{int(m.sum()):>10d}"
          f"{np.nanmedian(zz['r2_joint_nc'][m]):>11.3f}"
          f"{np.nanmedian(zz['noise_ceiling'][m]):>8.3f}")

print()
print("ROI voxel counts (nc>0.1) per subject")
LAB = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal",
       5: "ventral", 6: "lateral", 7: "parietal"}
hdr = f"{'subj':6s}" + "".join(f"{n:>13s}" for n in LAB.values())
print(hdr)
for s in range(1, 9):
    zz = np.load(f"results/encoding/untrained_seed0_subj{s:02d}.npz",
                 allow_pickle=True)
    roi, nc = zz["roi_streams"], zz["noise_ceiling"]
    line = f"{s:<6d}"
    for lab in LAB:
        line += f"{int(((roi == lab) & (nc > 0.1)).sum()):>13d}"
    print(line)
