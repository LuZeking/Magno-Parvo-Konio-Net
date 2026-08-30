"""Input-level null model: Gabor-energy features on each stream's own input.

PAPER_PLAN §7.2 item 1 — "the control that decides how much the network
matters". Every leg-(c) result so far is stream-*network* features regressed
onto BOLD. If a bank of oriented energy filters applied to the very same
transformed inputs reproduces the V1->hV4 gradient, then the ResNet is doing
nothing the front end had not already done, and those panels are front-end
validations rather than model results.

This is a different question from the blur control (26373186). Blur asks
whether the magno *input* is trivially low-pass; this asks whether the
*network* adds anything at all on top of the input.

The bank is deliberately matched to the network readout it replaces:
four scales (one per ResNet stage) x eight orientations, quadrature-pair
energy, adaptive-average-pooled to a fixed grid and flattened, so the output
is a (N, D) matrix per stream that drops straight into the same PCA ->
banded-ridge -> variance-partition path with no downstream change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# One scale per ResNet stage (features.STAGE_INDICES has four entries).
# Sigmas are in pixels at the 256 px working resolution of the front end.
SIGMAS = (2.0, 4.0, 8.0, 16.0)
N_ORIENT = 8
POOL_GRID = 8  # 4 scales * 8 orientations * 8 * 8 = 2048 dims per stream


def _gabor_pair(sigma: float, theta: float, lambd: float | None = None):
    """Even/odd Gabor kernels (quadrature pair) for one scale/orientation."""
    if lambd is None:
        lambd = sigma * 2.0  # ~1 octave bandwidth
    half = int(np.ceil(3.0 * sigma))
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    x_t = x * np.cos(theta) + y * np.sin(theta)
    y_t = -x * np.sin(theta) + y * np.cos(theta)
    env = np.exp(-(x_t**2 + y_t**2) / (2.0 * sigma**2))
    even = env * np.cos(2.0 * np.pi * x_t / lambd)
    odd = env * np.sin(2.0 * np.pi * x_t / lambd)
    # Zero-mean the even phase so it does not respond to mean luminance,
    # which would make the "energy" partly a DC term.
    even = even - even.mean()
    odd = odd - odd.mean()
    # Unit L2 norm keeps scales on a comparable footing.
    even /= np.linalg.norm(even) + 1e-12
    odd /= np.linalg.norm(odd) + 1e-12
    return even, odd


def build_bank(device: str = "cpu"):
    """[(even_kernel, odd_kernel, half_width)] for every scale x orientation."""
    bank = []
    for sigma in SIGMAS:
        for k in range(N_ORIENT):
            theta = np.pi * k / N_ORIENT
            even, odd = _gabor_pair(sigma, theta)
            half = even.shape[0] // 2
            e = torch.tensor(even, dtype=torch.float32, device=device)[None, None]
            o = torch.tensor(odd, dtype=torch.float32, device=device)[None, None]
            bank.append((e, o, half))
    return bank


@torch.no_grad()
def gabor_energy(x: torch.Tensor, bank, pool_grid: int = POOL_GRID):
    """(N,1,H,W) -> (N, n_filters * pool_grid**2) oriented-energy features."""
    outs = []
    for even, odd, half in bank:
        # 'reflect' matches the border handling of the training-time blur.
        xp = F.pad(x, (half, half, half, half), mode="reflect")
        re = F.conv2d(xp, even)
        im = F.conv2d(xp, odd)
        energy = torch.sqrt(re * re + im * im + 1e-12)
        outs.append(F.adaptive_avg_pool2d(energy, pool_grid).flatten(1))
    return torch.cat(outs, dim=1)


@torch.no_grad()
def extract_input_features(
    x_magno: torch.Tensor,
    x_parvo: torch.Tensor,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Drop-in analogue of features.extract_stream_features, input-level.

    Same return contract: {"magno": (N, D), "parvo": (N, D)} float32.
    """
    bank = build_bank(device)
    out: dict[str, list[np.ndarray]] = {"magno": [], "parvo": []}
    for start in range(0, x_magno.shape[0], batch_size):
        xm = x_magno[start : start + batch_size].to(device)
        xp = x_parvo[start : start + batch_size].to(device)
        # The parvo stream can carry 2 channels (parvo_highsf); treat each
        # channel as an independent map and concatenate, so the feature space
        # still describes only the input.
        for name, t in (("magno", xm), ("parvo", xp)):
            per_ch = [
                gabor_energy(t[:, c : c + 1], bank) for c in range(t.shape[1])
            ]
            out[name].append(
                torch.cat(per_ch, dim=1).float().cpu().numpy()
            )
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
