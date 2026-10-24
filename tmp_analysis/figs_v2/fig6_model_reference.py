"""The model's magno − parvo profile for Fig. 6c, under the rule generality_cortex.py applies to the BA network's
"two_stream_diet" reference: voxels pooled per region (V1–hV4 from prf-visualrois, the ventral, lateral and dorsal
groups from Kastner2015), the index the median over reliable voxels, seeds averaged per subject. Reads the per-voxel
fits of PAPER_v3's model of record on /scratch; writes results/figures_v3/fig6_model_reference.json.

  PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig6_model_reference.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from generality_cortex import REGIONS, index_stats, kastner_labels, region_masks

ENC = os.environ.get("FS2_ENCODING_DIR", "/scratch/byrong/mpk/encoding/results/2026-10-16_fs2")
MODEL = [f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)]


def profile(enc: str = ENC) -> dict:
    """Per-subject seed means of the index (and its parts) per region; tmp_analysis/verify_v3_claims.py calls this too."""
    ref = {}
    for s in range(1, 9):
        kas = kastner_labels(s, None)
        for run in MODEL:
            z = np.load(f"{enc}/{run}_subj{s:02d}.npz", allow_pickle=True)
            assert len(kas) == len(z["noise_ceiling"]), (run, s)
            masks = region_masks(z["roi_prf-visualrois"], kas, z["noise_ceiling"])
            ref.setdefault(s, []).append({r: index_stats(z["unique_magno"], z["unique_parvo"], masks[r], z["shared"], z["r2_joint_nc"])
                                          for r in REGIONS})
    subjects = sorted(ref)
    return dict(subjects=subjects, runs=MODEL, source=enc,
                **{k: {r: [float(np.nanmean([st[r][k] for st in ref[s]])) for s in subjects] for r in REGIONS}
                   for k in ("index", "uS", "uT", "shared", "r2")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures_v3/fig6_model_reference.json")
    args = ap.parse_args()
    fam = profile()
    json.dump({"rule": "tmp_analysis/figs_v2/generality_cortex.py two_stream_diet reference", "families": {"two_stream_model": fam}},
              open(args.out, "w"), indent=1)
    for r in REGIONS:
        print(f"  {r:8s} index {np.nanmean(fam['index'][r]):+.3f}  U_m {np.nanmean(fam['uS'][r]):.3f}  U_p {np.nanmean(fam['uT'][r]):.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
