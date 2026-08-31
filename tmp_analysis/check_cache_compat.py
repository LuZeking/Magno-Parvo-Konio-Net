"""Does the feature cache still fail safe after the new front-end flags?

The §22 contamination was a cache serving features from the wrong front end,
so any change to cache validation has to be shown not to reopen that hole.
Three cases matter:

  1. a cache written BEFORE --transform-target/--parvo-highsf existed, read by
     a default run -> must HIT (otherwise the in-flight repair job recomputes
     features on all its remaining tasks for nothing)
  2. the same old cache, read by a --parvo-highsf run   -> must MISS
  3. the same old cache, read by a --transform-target parvo run -> must MISS

Usage: PYTHONPATH=. python tmp_analysis/check_cache_compat.py
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from src.analysis.encoding.run_encoding import cache_is_valid, cache_provenance


def make_args(**over):
    a = argparse.Namespace(
        no_lum_blur=True, no_lum_cs=True, age_months=48, features="network",
        transform_target="magno", parvo_highsf=False, checkpoint="/ck.pt",
        stream1_weight=0.5, seed=0,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


base = make_args()
want = cache_provenance(base)
# An old cache: exactly the keys that existed before 2026-09-01.
old_keys = {k: v for k, v in want.items()
            if k not in ("prov_transform_target", "prov_parvo_highsf")}

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "old_cache.npz")
    np.savez(p, image_ids=np.arange(3), magno=np.zeros((3, 2)),
             parvo=np.zeros((3, 2)), **old_keys)
    z = np.load(p, allow_pickle=False)

    cases = [
        ("default run vs pre-flag cache", make_args(), True),
        ("--parvo-highsf vs pre-flag cache", make_args(parvo_highsf=True), False),
        ("--transform-target parvo vs pre-flag cache",
         make_args(transform_target="parvo"), False),
    ]
    ok = True
    for label, args, expect in cases:
        got = cache_is_valid(z, cache_provenance(args))
        mark = "ok" if got is expect else "FAIL"
        ok &= got is expect
        print(f"  [{mark}] {label:<44} valid={got} (want {expect})")

print("\nVERDICT:", "cache still fails safe" if ok else "REGRESSION -- fix before use")
