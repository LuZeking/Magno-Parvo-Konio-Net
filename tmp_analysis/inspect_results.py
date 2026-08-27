import numpy as np

STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal",
           5: "ventral", 6: "lateral", 7: "parietal"}

for subj in (4, 8):
    z = np.load(f"results/encoding/untrained_seed0_subj{subj:02d}.npz",
                allow_pickle=True)
    nc = z["noise_ceiling"]
    r2j, r2m, r2p = z["r2_joint"], z["r2_magno"], z["r2_parvo"]
    um, up = z["unique_magno"], z["unique_parvo"]
    roi = z["roi_streams"]
    rel = nc > 0.1
    print(f"\n{'=' * 72}")
    print(f"subject {subj:02d}  ({nc.size} voxels, {int(rel.sum())} NC>0.1)")
    print(f"{'=' * 72}")
    print(f"  R2 joint median (NC>0.1) : {np.nanmedian(r2j[rel]):+.4f}")
    print(f"  R2 magno median          : {np.nanmedian(r2m[rel]):+.4f}")
    print(f"  R2 parvo median          : {np.nanmedian(r2p[rel]):+.4f}")
    print(f"  R2 joint / noise ceiling : "
          f"{np.nanmedian(r2j[rel] / nc[rel]):+.4f}")
    print(f"  {'ROI':<13}{'n':>6}{'uniqM':>10}{'uniqP':>10}{'R2j/NC':>9}")
    for lab, name in STREAMS.items():
        v = (roi == lab) & rel
        if v.sum() < 20:
            continue
        print(f"  {name:<13}{int(v.sum()):>6}"
              f"{np.nanmedian(um[v]):>10.4f}{np.nanmedian(up[v]):>10.4f}"
              f"{np.nanmedian(r2j[v] / nc[v]):>9.3f}")
