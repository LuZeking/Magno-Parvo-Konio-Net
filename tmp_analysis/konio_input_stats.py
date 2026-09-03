"""Input-level statistics of the three DKL channels on NSD stimuli.

Written BEFORE the konio encoding arm is submitted, because the cached
training-set min/max (dkl_channel_minmax.json) shows very different raw
ranges per channel:

    lum  [-1.414,  5.193]   span 6.607
    rg   [-0.524,  0.524]   span 1.048
    by   [-5.039,  3.150]   span 8.189

Every channel is then normalised to [0, 1] with its OWN global min/max and
quantised to uint8. A channel whose global range is set by rare extreme
pixels therefore lands in a narrow band around the middle of [0, 1] for
typical images -- i.e. low effective contrast -- purely as an artefact of the
normalisation, not because of anything biological.

If `by` has much lower post-normalisation sd than `rg`, then a konio stream
that explains less cortical variance than the parvo stream would be
uninterpretable: the numerical contrast difference and the chromatic-axis
difference would be confounded. That has to be known before the arm runs,
not diagnosed afterwards (§9.1).

Also reports the radial power spectrum, because the S-(L+M) axis is expected
to carry less high-spatial-frequency energy in natural images, which is a
real property rather than an artefact and must be told apart from one.
"""
import numpy as np
import torch

from src.analysis.encoding import dkl_frontend, nsd_data

N_IMAGES = 100
CH = ["lum (L+M)", "rg (L-M)", "by (S-(L+M))"]

ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
print("training-set global DKL range (used for the [0,1] normalisation):")
for i, name in enumerate(CH):
    print(f"  {name:16s} [{ch_min[i]:+.4f}, {ch_max[i]:+.4f}]  "
          f"span {ch_max[i]-ch_min[i]:.4f}")

stim = nsd_data.load_stim_info()
ids = nsd_data.shared515_image_ids(stim)[:N_IMAGES]
print(f"\nloading {len(ids)} shared-515 NSD images", flush=True)
rgb = dkl_frontend.load_nsd_images(ids)

x = dkl_frontend.rgb_to_dkl3(rgb, ch_min, ch_max)   # (N, 3, 256, 256) in [0,1]
print(f"dkl3 tensor {tuple(x.shape)}  min {x.min():.4f} max {x.max():.4f}")

print("\n" + "=" * 74)
print("post-normalisation statistics, in the units the network actually sees")
print("=" * 74)
print(f"{'channel':16s} {'mean':>8s} {'sd':>8s} {'p1':>7s} {'p99':>7s} "
      f"{'p99-p1':>8s} {'sd/rg':>7s}")
sds = []
for i, name in enumerate(CH):
    v = x[:, i].flatten().numpy()
    sd = float(v.std())
    sds.append(sd)
    p1, p99 = np.percentile(v, [1, 99])
    print(f"{name:16s} {v.mean():8.4f} {sd:8.4f} {p1:7.4f} {p99:7.4f} "
          f"{p99-p1:8.4f} {sd/1.0:7.4f}")
print(f"\n  sd ratio by/rg  = {sds[2]/sds[1]:.4f}")
print(f"  sd ratio by/lum = {sds[2]/sds[0]:.4f}")

# Per-image RMS contrast, the quantity a conv stem's first layer responds to.
print("\nper-image RMS contrast (sd within image, averaged over images):")
for i, name in enumerate(CH):
    per_img = x[:, i].reshape(x.shape[0], -1).std(dim=1)
    print(f"  {name:16s} {per_img.mean():.4f} +- {per_img.std():.4f}")

# uint8 quantisation levels actually occupied: a channel squeezed into a
# narrow band loses precision as well as contrast.
print("\ndistinct uint8 levels occupied (of 256):")
for i, name in enumerate(CH):
    q = (x[:, i].numpy() * 255).round().astype(np.uint8)
    print(f"  {name:16s} {len(np.unique(q)):3d}")

# Radial power spectrum, mean over images, normalised to unit total power so
# the SHAPE is comparable across channels of different contrast.
print("\nradial power spectrum (fraction of total power per octave band):")
S = x.shape[-1]
fy = np.fft.fftfreq(S)[:, None]
fx = np.fft.fftfreq(S)[None, :]
r = np.sqrt(fy**2 + fx**2)
edges = [1/256, 1/64, 1/32, 1/16, 1/8, 1/4, 0.5 + 1e-9]
labels = [f"{edges[k]:.4f}-{edges[k+1]:.4f}" for k in range(len(edges) - 1)]
print(f"{'channel':16s} " + " ".join(f"{l:>13s}" for l in labels))
for i, name in enumerate(CH):
    xi = x[:, i].numpy()
    xi = xi - xi.mean(axis=(1, 2), keepdims=True)
    P = np.abs(np.fft.fft2(xi))**2
    P = P.mean(axis=0)
    tot = P.sum()
    fr = []
    for k in range(len(edges) - 1):
        m = (r >= edges[k]) & (r < edges[k + 1])
        fr.append(P[m].sum() / tot)
    print(f"{name:16s} " + " ".join(f"{v:13.4f}" for v in fr))

print("\nabsolute power in the top two octaves (contrast-dependent, not "
      "normalised):")
for i, name in enumerate(CH):
    xi = x[:, i].numpy()
    xi = xi - xi.mean(axis=(1, 2), keepdims=True)
    P = np.abs(np.fft.fft2(xi))**2
    P = P.mean(axis=0)
    m = r >= 1/8
    print(f"  {name:16s} {P[m].sum():.4e}")
