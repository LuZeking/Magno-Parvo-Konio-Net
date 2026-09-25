"""TDANN-style cortical-sheet positions and spatial correlation loss.

Port of Margalit et al. (2024) `spacetorch` (github.com/neuroailab/TDANN):
- positions: `scripts/save_retinotopic_positions.py` + `utils/spatial_utils.py`
  (`place_conv` with random within-hypercolumn offsets, `jitter_positions(0.3)`,
  `precompute_neighborhoods`, `collapse_and_trim_neighborhoods(keep_fraction=0.95,
  keep_limit=500)`), tissue sizes and neighbourhood widths from
  `models/positions.py` / `constants.py`;
- loss: `losses/losses_torch.py::spatial_correlation_loss`,
  SL = (1 - corr(lower_tri(corrcoef(responses)), lower_tri(1 / (d + 1)))) / 2 for one
  random neighbourhood per layer per step (`neighborhoods_per_batch: 1`).

Not ported: the swap pre-optimisation (`swapopt.py`), which needs a pretrained network
of the same architecture; the pilot plan (docs/PLAN_TSN_SPATIAL_PILOT_2026-09-25.md)
records that deviation. Positions are generated from a seeded RNG so a run's sheet can be
rebuilt exactly from (layer dims, area, seed); they are also saved next to the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# spacetorch/constants.py and models/positions.py (mm)
TISSUE_SIZE = {"retina": 2.4, "V1": 36.75, "V2": 35.0, "V4": 22.4, "VTC": 70.0}
NEIGHBORHOOD_WIDTH = {"retina": 0.0475, "V1": 1.626, "V2": 3.977, "V4": 2.545,
                      "VTC": 31.818}
# spacetorch/models/__init__.py BRAIN_MAPPING["resnet18"]; "IT" -> VTC tissue.
RESNET18_AREA = {"layer1.0": "retina", "layer1.1": "retina", "layer2.0": "V1",
                 "layer2.1": "V1", "layer3.0": "V2", "layer3.1": "V4",
                 "layer4.0": "VTC", "layer4.1": "VTC"}
INITIAL_RF_OVERLAP = 0.1
POS_JITTER = 0.3
N_NEIGHBORHOODS = 20_000
KEEP_FRACTION = 0.95
KEEP_LIMIT = 500
MIN_NB_SIZE = 25  # spacetorch collapse_and_trim_neighborhoods default


@dataclass
class LayerSheet:
    name: str
    dims: tuple[int, int, int]      # (C, H, W) of the layer output
    coordinates: np.ndarray         # (C*H*W, 2) mm, index = flat CHW index
    neighborhoods: np.ndarray       # (P, Q) flat unit indices, Q <= KEEP_LIMIT
    neighborhood_width: float

    def to_npz(self, path: str) -> None:
        np.savez(path, name=self.name, dims=np.asarray(self.dims),
                 coordinates=self.coordinates, neighborhoods=self.neighborhoods,
                 neighborhood_width=self.neighborhood_width)

    @classmethod
    def from_npz(cls, path: str) -> "LayerSheet":
        z = np.load(path)
        return cls(str(z["name"]), tuple(int(v) for v in z["dims"]), z["coordinates"],
                   z["neighborhoods"], float(z["neighborhood_width"]))


def _rf_centers(n: int, overlap: float, width: float) -> tuple[np.ndarray, float]:
    rf_width = width / (n - n * overlap + overlap)
    r = rf_width / 2.0
    return np.linspace(r, width - r, n), r


def place_conv(dims, tissue, overlap, rng) -> np.ndarray:
    """(C*H*W, 2) positions in flat CHW order (spacetorch `place_conv`, 'random')."""
    c, h, w = dims
    if h != w:
        raise ValueError("square feature maps only")
    centers, r = _rf_centers(w, overlap, tissue)
    xx, yy = np.meshgrid(centers, centers)
    anchors = np.stack((xx.ravel(), yy.ravel()), 1)                # (H*W, 2)
    offsets = rng.uniform(-r, r, size=(len(anchors), c, 2))
    pos = anchors[:, None, :] + offsets                             # (H*W, C, 2)
    return np.swapaxes(pos, 0, 1).reshape(-1, 2)                    # (C*H*W, 2)


def jitter(pos, amount, rng) -> np.ndarray:
    j = pos + rng.normal(0.0, amount, size=pos.shape)
    j = j / (np.ptp(j, axis=0) / np.ptp(pos, axis=0))
    return j - j.min(0) + pos.min(0)


def neighborhoods(pos, radius, rng, n=N_NEIGHBORHOODS, keep_fraction=KEEP_FRACTION,
                  keep_limit=KEEP_LIMIT) -> np.ndarray:
    """Square windows of half-width `radius`, trimmed to a common size.

    spacetorch keeps the neighbourhoods at or above max(the (1 - keep_fraction) size
    percentile, MIN_NB_SIZE) and subsamples each to min(that size, keep_limit) units.
    """
    xs = rng.uniform(pos[:, 0].min() + radius, pos[:, 0].max() - radius, n)
    ys = rng.uniform(pos[:, 1].min() + radius, pos[:, 1].max() - radius, n)
    order = np.argsort(pos[:, 0], kind="stable")
    px = pos[order, 0]
    nb = []
    for x0, y0 in zip(xs, ys):
        lo, hi = np.searchsorted(px, [x0 - radius, x0 + radius], side="left")
        cand = order[lo:hi]
        keep = cand[np.abs(pos[cand, 1] - y0) <= radius]
        nb.append(keep)
    sizes = np.array([len(v) for v in nb])
    size = min(max(int(np.percentile(sizes, (1 - keep_fraction) * 100)), MIN_NB_SIZE),
               keep_limit)
    out = [rng.choice(v, size, replace=False) for v in nb if len(v) >= size]
    if len(out) < 100:
        raise ValueError(f"only {len(out)} neighbourhoods reach {size} units")
    return np.stack(out).astype(np.int64)


def build_sheet(name: str, dims, area: str, first_hw: int, seed: int) -> LayerSheet:
    """One layer's sheet. `first_hw` = H of the first sheet layer (overlap scaling)."""
    rng = np.random.default_rng(seed)
    overlap = min(INITIAL_RF_OVERLAP * first_hw / dims[-1], 1.0)
    pos = jitter(place_conv(dims, TISSUE_SIZE[area], overlap, rng), POS_JITTER, rng)
    width = NEIGHBORHOOD_WIDTH[area]
    return LayerSheet(name, tuple(dims), pos.astype(np.float32),
                      neighborhoods(pos, width / 2, rng), width)


# ---------------------------------------------------------------- loss


def _corrcoef_cols(x: torch.Tensor) -> torch.Tensor:
    """Correlation between columns of x (B, Q) -> (Q, Q)."""
    x = x - x.mean(0, keepdim=True)
    x = x / (x.norm(dim=0, keepdim=True) + 1e-8)
    return x.t() @ x


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def gather_units(features: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, Q) responses of the flat-CHW units `nb_idx`, without
    materialising a flattened copy (the layer1 maps are 67M values per batch)."""
    _, _, h, w = features.shape
    c = torch.div(nb_idx, h * w, rounding_mode="floor")
    rem = nb_idx - c * h * w
    y = torch.div(rem, w, rounding_mode="floor")
    return features[:, c, y, rem - y * w]


def spatial_correlation_loss(features: torch.Tensor, coords: torch.Tensor,
                             nb_idx: torch.Tensor) -> torch.Tensor:
    """SL for one neighbourhood; features (B, C, H, W), nb_idx (Q,) flat indices.

    Computed in float32 (call it outside autocast). Units constant over the batch
    (dead ReLUs) get correlation 0 through the epsilon where spacetorch's `corrcoef`
    would return NaN.
    """
    f = gather_units(features, nb_idx).float()
    p = coords[nb_idx].float()
    d = torch.cdist(p, p)
    iu = torch.triu_indices(len(nb_idx), len(nb_idx), 1, device=f.device)
    resp = _corrcoef_cols(f)[iu[0], iu[1]]
    dist_sim = (1.0 / (d + 1.0))[iu[0], iu[1]]
    return (1.0 - _pearson(resp, dist_sim)) / 2.0


class SheetLoss(torch.nn.Module):
    """Holds sheets as buffers; `forward(feats)` -> {sheet: SL} for one random
    neighbourhood per sheet (spacetorch `neighborhoods_per_batch: 1`)."""

    def __init__(self, sheets: list[LayerSheet]):
        super().__init__()
        self.names = [s.name for s in sheets]
        for i, s in enumerate(sheets):
            self.register_buffer(f"coords_{i}", torch.from_numpy(s.coordinates),
                                 persistent=False)
            self.register_buffer(f"nb_{i}", torch.from_numpy(s.neighborhoods),
                                 persistent=False)

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for i, name in enumerate(self.names):
            nb = getattr(self, f"nb_{i}")
            k = torch.randint(len(nb), (1,), device=nb.device)
            out[name] = spatial_correlation_loss(feats[name],
                                                 getattr(self, f"coords_{i}"),
                                                 nb[k].squeeze(0))
        return out


class SheetCapture:
    """Forward hooks that keep each sheet module's output while `active`."""

    def __init__(self, modules: dict[str, torch.nn.Module]):
        self.feats: dict[str, torch.Tensor] = {}
        self.active = False
        self.handles = [m.register_forward_hook(self._hook(n)) for n, m in modules.items()]

    def _hook(self, name):
        def hook(_mod, _inp, out):
            if self.active:
                self.feats[name] = out
        return hook

    def clear(self) -> None:
        self.feats = {}


def build_model_sheets(dims: dict[str, tuple[int, int, int]], area_of, seed: int
                       ) -> list[LayerSheet]:
    """Sheets for every entry of `dims` (name -> CHW); `area_of(name)` gives the
    TDANN area. RF overlap scales from the largest grid, as in spacetorch. Each
    sheet's RNG seed is derived from (seed, position in `dims`)."""
    first_hw = max(d[-1] for d in dims.values())
    return [build_sheet(n, d, area_of(n), first_hw, seed=1000 * seed + i)
            for i, (n, d) in enumerate(dims.items())]
