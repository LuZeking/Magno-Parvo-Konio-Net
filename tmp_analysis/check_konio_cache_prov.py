"""Verify the konio cache records its front end, and that a MISMATCHED front
end is rejected rather than silently reused.

This is the one behaviour tmp_analysis/smoke_konio.py could not cover: it
tested the front end in isolation, not the cache round-trip. It is also the
exact failure that contaminated a subject in §22 -- a cache keyed only on
model name and subject, silently reused across front ends.
"""
import argparse
import sys

import numpy as np

from src.analysis.encoding.run_encoding import (
    BACKFILL_DEFAULTS, cache_is_valid, cache_provenance,
)

# Any konio-matched cache will do; pass a different path as argv[1]. Only the
# small prov_* arrays are touched, so opening a ~1.1 GB npz here is cheap.
CACHE = sys.argv[1] if len(sys.argv) > 1 else (
    "/scratch/byrong/VST_data/mpk_feature_cache/"
    "untrained_konio_matched_subj04.npz"
)


def args_for(**over):
    """The argparse namespace run_encoding would build for a given cell."""
    d = dict(no_lum_blur=False, no_lum_cs=False, age_months=48,
             features="network", transform_target="magno",
             parvo_highsf=False, chroma_channel="rg", chroma_rescale="none",
             checkpoint=None, stream1_weight=0.5, seed=0)
    d.update(over)
    return argparse.Namespace(**d)


z = np.load(CACHE, allow_pickle=True)
print(f"cache: {CACHE}")
print(f"arrays: {[k for k in z.files if not k.startswith('prov')]}")
print("\nrecorded provenance:")
for k in sorted(k for k in z.files if k.startswith("prov")):
    print(f"  {k:26s} = {z[k].item()!r}")

KONIO = dict(chroma_channel="by", chroma_rescale="match-rg")

print("\n" + "=" * 70)
print("the cache must VALIDATE for the cell that wrote it")
print("=" * 70)
ok = cache_is_valid(z, cache_provenance(args_for(**KONIO)))
print(f"  konio-matched front end -> valid={ok}   (expect True)")
assert ok, "the cache does not validate against its own front end"

print("\n" + "=" * 70)
print("...and must be REJECTED for every other front end")
print("=" * 70)
CASES = {
    "parvo (rg, the reference cell)": dict(),
    "konio RAW (by, unmatched)": dict(chroma_channel="by"),
    "konio matched but no blur": dict(no_lum_blur=True, **KONIO),
    "konio matched, swapped target": dict(transform_target="parvo", **KONIO),
    "konio matched, gabor features": dict(features="gabor", **KONIO),
}
for label, over in CASES.items():
    v = cache_is_valid(z, cache_provenance(args_for(**over)))
    print(f"  {label:34s} -> valid={v}   (expect False)")
    assert not v, f"{label} WRONGLY accepted this cache"

print("\n" + "=" * 70)
print("backfill defaults cover the new keys (old caches stay usable)")
print("=" * 70)
for k in ("prov_chroma_channel", "prov_chroma_rescale"):
    print(f"  {k:26s} backfill = {BACKFILL_DEFAULTS[k]!r}")
    assert k in BACKFILL_DEFAULTS

print("\nALL CACHE-PROVENANCE CHECKS PASSED")
