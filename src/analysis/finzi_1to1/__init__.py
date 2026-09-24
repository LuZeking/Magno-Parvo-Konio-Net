"""1-to-1 unit-to-voxel stream-preference test (docs/PLAN_FINZI_1TO1_2026-09-24.md)."""

import os

OUT_ROOT = os.environ.get("FINZI_OUT_ROOT", "/scratch/byrong/TSN/finzi_1to1")
STREAM_LABELS = {5: "ventral", 6: "lateral", 7: "dorsal"}
NC_MIN_PERCENT = 10.0


def betas_cache_path(subject: int) -> str:
    return os.path.join(OUT_ROOT, "cache", "betas", f"subj{subject:02d}.npz")


def features_path(network: str, subject: int) -> str:
    return os.path.join(OUT_ROOT, "features", network, f"subj{subject:02d}.npz")
