"""Smoke test for the konio front end, before any encoding time is spent.

The expensive risk here is not that the konio arm fails -- it is that adding
`--chroma-channel` silently changes the DEFAULT rg path, which 256 feature
caches and 81 fitted result files on disk depend on. So check 1 compares the
patched front end against the pre-patch module recovered from git, bit for
bit, rather than trusting inspection.

Checks, in order:
  1. REGRESSION: at default flags the patched front end is bit-identical to
     the pre-patch one, for every existing condition (base, bcs, swap,
     parvo_highsf, rgb_mode).
  2. The konio arm returns the by channel (index 2), and its magno stream is
     bit-identical to the parvo arm's -- so the swap is single-factor.
  3. Contrast matching does what it claims: per-image RMS equals rg's and
     the result stays in [0, 1]. Note the input here is uniform random RGB,
     which has no natural chromatic structure, so the unmatched by/rg ratio
     comes out ~1.0 rather than the 2.05x measured on real NSD stimuli. This
     check tests the MECHANISM, not the size of the confound; the confound
     itself is quantified on NSD images (§65.3).
  4. The nonsense flag combinations raise instead of silently no-opping.
  5. The untrained network is bit-identical across arms at the same seed, so
     the only difference between the konio run and the published parvo run
     is the input -- and the resulting magno features match exactly while
     the chromatic features do not.

Run with PYTHONPATH set to the worktree root.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

from src.analysis.encoding import dkl_frontend as fe
from src.analysis.encoding import features

PREPATCH = os.path.join(
    os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp",
    "dkl_frontend_prepatch.py",
)


def load_prepatch():
    """Import the pre-patch dkl_frontend as a separate module.

    Its MINMAX_CACHE constant is derived from its own __file__ and so points
    somewhere useless, but that is harmless: min/max enter through the
    function arguments, and nothing in the call path recomputes them.
    """
    spec = importlib.util.spec_from_file_location("fe_old", PREPATCH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["fe_old"] = m
    spec.loader.exec_module(m)
    return m


def p(msg=""):
    print(msg, flush=True)


ch_min, ch_max = fe.compute_dkl_channel_minmax()
rng = np.random.default_rng(0)
rgb = rng.integers(0, 256, size=(6, 425, 425, 3), dtype=np.uint8)

p("=" * 78)
p("1. REGRESSION -- default paths bit-identical to the pre-patch module")
p("=" * 78)
if not os.path.exists(PREPATCH):
    sys.exit(f"pre-patch module not found at {PREPATCH}; run\n"
             f"  git show HEAD:src/analysis/encoding/dkl_frontend.py > "
             f"{PREPATCH}")
old = load_prepatch()

CASES = {
    "dkl_base           ": dict(lum_blur=False, lum_cs=False),
    "dkl_bcs            ": dict(lum_blur=True, lum_cs=True),
    "dkl_bcs_swap       ": dict(lum_blur=True, lum_cs=True,
                                transform_target="parvo"),
    "dkl_bcs_parvo_highsf": dict(lum_blur=True, lum_cs=True,
                                 parvo_highsf=True),
    "rgb_tsn            ": dict(rgb_mode=True),
}
for name, kw in CASES.items():
    a1, a2 = old.magno_parvo_inputs(rgb, ch_min, ch_max, **kw)
    b1, b2 = fe.magno_parvo_inputs(rgb, ch_min, ch_max, **kw)
    d1 = (a1 - b1).abs().max().item()
    d2 = (a2 - b2).abs().max().item()
    p(f"  {name}  x1 max|diff|={d1:.3e}  x2 max|diff|={d2:.3e}  "
      f"shapes {tuple(b1.shape)} {tuple(b2.shape)}")
    assert d1 == 0.0 and d2 == 0.0, f"{name} CHANGED -- caches invalidated"
p("  -> the rg default is untouched; existing caches remain valid")

p()
p("=" * 78)
p("2. the konio arm is a single-factor swap")
p("=" * 78)
x3 = fe.rgb_to_dkl3(rgb, ch_min, ch_max)
m_p, x_p = fe.magno_parvo_inputs(rgb, ch_min, ch_max, chroma_channel="rg")
m_k, x_k = fe.magno_parvo_inputs(rgb, ch_min, ch_max, chroma_channel="by")

p(f"  parvo stream == DKL channel 1?  "
  f"max|diff|={(x_p - x3[:, 1:2]).abs().max().item():.3e}")
p(f"  konio stream == DKL channel 2?  "
  f"max|diff|={(x_k - x3[:, 2:3]).abs().max().item():.3e}")
assert (x_p - x3[:, 1:2]).abs().max().item() == 0.0
assert (x_k - x3[:, 2:3]).abs().max().item() == 0.0

d_m = (m_p - m_k).abs().max().item()
d_c = (x_p - x_k).abs().max().item()
p(f"  magno stream identical across arms?  max|diff|={d_m:.3e}  (expect 0)")
p(f"  chromatic stream differs?            max|diff|={d_c:.3e}  (expect > 0)")
assert d_m == 0.0, "the konio arm changed the luminance stream too"
assert d_c > 1e-3, "by and rg are the same tensor -- wrong slice"

p()
p("=" * 78)
p("3. contrast matching")
p("=" * 78)
_, x_km = fe.magno_parvo_inputs(rgb, ch_min, ch_max, chroma_channel="by",
                                chroma_rescale="match-rg")


def rms(t):
    return t.reshape(t.shape[0], -1).std(dim=1)


r_p, r_k, r_km = rms(x_p), rms(x_k), rms(x_km)
p(f"  per-image RMS  rg          : {r_p.mean():.4f} +- {r_p.std():.4f}")
p(f"  per-image RMS  by unmatched: {r_k.mean():.4f} +- {r_k.std():.4f}"
  f"   ratio to rg = {(r_k / r_p).mean():.3f}x")
p(f"  per-image RMS  by matched  : {r_km.mean():.4f} +- {r_km.std():.4f}"
  f"   ratio to rg = {(r_km / r_p).mean():.3f}x")
p(f"  max per-image |RMS_matched - RMS_rg| = "
  f"{(r_km - r_p).abs().max().item():.3e}")
assert (r_km - r_p).abs().max().item() < 1e-6, "matching did not match"
p(f"  matched range: [{x_km.min():.4f}, {x_km.max():.4f}] "
  f"(expect inside [0, 1]; matching is a contraction)")
assert x_km.min() >= 0.0 and x_km.max() <= 1.0
p("  mean is preserved (matching rescales about it): "
  f"max|diff| = "
  f"{(x_km.mean(dim=(1, 2, 3)) - x_k.mean(dim=(1, 2, 3))).abs().max():.3e}")

p()
p("=" * 78)
p("4. nonsense combinations raise")
p("=" * 78)
for kw, why in [
    (dict(chroma_channel="rg", chroma_rescale="match-rg"),
     "match-rg on rg is the identity"),
    (dict(rgb_mode=True, chroma_channel="by"),
     "rgb_mode ignores chroma_channel"),
    (dict(chroma_channel="lum"), "unknown channel"),
    (dict(chroma_channel="by", chroma_rescale="match-lum"),
     "unknown rescale"),
]:
    try:
        fe.magno_parvo_inputs(rgb, ch_min, ch_max, **kw)
    except ValueError as e:
        p(f"  raised as expected ({why}): {e}")
    else:
        sys.exit(f"NO RAISE for {kw} -- {why}")

p()
p("=" * 78)
p("5. same untrained network, different input")
p("=" * 78)
m1 = features.build_model(checkpoint=None, seed=0, device="cpu")
m2 = features.build_model(checkpoint=None, seed=0, device="cpu")
s1, s2 = m1.state_dict(), m2.state_dict()
worst = max((s1[k] - s2[k]).abs().max().item() for k in s1
            if s1[k].is_floating_point())
p(f"  two seed-0 builds identical? max|diff over all params| = {worst:.3e}")
assert worst == 0.0, "untrained weights are not reproducible at fixed seed"

f_p = features.extract_stream_features(m1, m_p, x_p, batch_size=6)
f_k = features.extract_stream_features(m1, m_k, x_k, batch_size=6)
f_km = features.extract_stream_features(m1, m_k, x_km, batch_size=6)
p(f"  feature shapes: magno {f_p['magno'].shape} parvo {f_p['parvo'].shape}")
dm = np.abs(f_p["magno"] - f_k["magno"]).max()
dp = np.abs(f_p["parvo"] - f_k["parvo"]).max()
dpm = np.abs(f_k["parvo"] - f_km["parvo"]).max()
p(f"  magno FEATURES identical across arms?  max|diff|={dm:.3e} (expect 0)")
p(f"  chromatic FEATURES differ?             max|diff|={dp:.3e} (expect >0)")
p(f"  matching changes chromatic features?   max|diff|={dpm:.3e} (expect >0)")
assert dm == 0.0, "magno features moved; the comparison is not single-factor"
assert dp > 0.0 and dpm > 0.0
p("  -> Um is estimated from an identical feature matrix in both arms, so a")
p("     change in Up/Uk cannot come from the luminance side.")

p()
p("ALL CHECKS PASSED")
