"""Self-contained TSN trainer (no lightning_kietzmannlab dependency).

The thesis pipeline inherits from the lab's private `lightning_kietzmannlab`
package (TrainingPipeline / H5dataset / BaseModule), which is not obtainable
here. This module reimplements exactly the parts the paper needs, reusing the
in-repo TwoStreamNetwork and the verified DVD transforms, so models can be
retrained locally with:
  * a single dataset version (removes the thesis's uint8/float64 confound)
  * multiple seeds
  * the corrected symmetric cross-stream mixing
  * per-stream input configuration (incl. the high-SF-luminance parvo fix and
    the channel-swap / blur-only controls)

Training defaults replicate the thesis: 150 epochs, Adam lr 1e-4 (beta1 0.9),
batch 256, cross-entropy, and the DVD-B augmentation set (random grayscale
skipped for DKL inputs, as in the original).

Run (single condition):
    python -m src.training.standalone_trainer \
        --dataset /scratch/.../miniecoset_256px_dkl_uint8.h5 \
        --condition dkl_bcs --seed 0 --out-dir /scratch/.../runs
Smoke test:
    ... --max-steps 4 --epochs 1 --batch-size 8 --num-workers 0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models import topographic
from src.models.fused_two_stream import FusedTwoStreamNetwork, depth_of
from src.models.late_fusion import (LateFusionNetwork,
                                    make_magno_stem64)
from src.models.pytorch_models import TwoStreamNetwork

# --------------------------------------------------------------------------
# Conditions: which channels feed which stream, and what is applied to them.
#   input_mode "dkl"  -> stream1 = lum, stream2 = rg (1 channel each)
#   input_mode "rgb"  -> both streams get the 3-channel image
#   parvo_highsf      -> stream2 receives lum+rg stacked (2 channels): the
#                        biologically corrected parvo input (high-acuity
#                        luminance + red-green), Track B item B2
# --------------------------------------------------------------------------
CONDITIONS = {
    # thesis replications
    "rgb_tsn":    dict(input_mode="rgb", blur=False, cs=False),
    "dkl_base":   dict(input_mode="dkl", blur=False, cs=False),
    "dkl_b":      dict(input_mode="dkl", blur=True,  cs=False),
    "dkl_cs":     dict(input_mode="dkl", blur=False, cs=True),
    "dkl_bcs":    dict(input_mode="dkl", blur=True,  cs=True),
    "dkl_bcs_mheavy": dict(input_mode="dkl", blur=True, cs=True,
                           stream1_weight=0.9),
    "dkl_bcs_inter":  dict(input_mode="dkl", blur=True, cs=True,
                           stream_concat=True),
    # controls the paper needs
    "dkl_bcs_swap":   dict(input_mode="dkl", blur=True, cs=True,
                           transform_target="parvo"),
    "rgb_tsn_blur":   dict(input_mode="rgb", blur=True, cs=True),
    # Track B: corrected parvo input (high-SF luminance + L-M)
    "dkl_bcs_parvo_highsf": dict(input_mode="dkl", blur=True, cs=True,
                                 parvo_highsf=True),
    # What does splitting the input across two streams buy? One stream fed
    # BOTH DKL channels, diet on luminance as usual: identical input
    # information and an identical diet, without the architectural split.
    # `rgb_tsn` answers "two streams, no split" but confounds the colour
    # space; nothing answers "one stream at all". Uses the TwoStreamNetwork
    # baseline=True path, so the classifier and every hyperparameter are
    # unchanged. Capacity is HALVED (one ResNet18 trunk, not two) -- see the
    # §64 write-up: that confound only bites if this arm LOSES.
    "dkl_bcs_single": dict(input_mode="dkl", blur=True, cs=True,
                           single_stream=True),
    # The parameter-matched version of the arm above, triggered by the letter
    # of §64.5's outcome B: the split bought shape bias, so the capacity
    # confound has to be retired before that is attributed to the split rather
    # than to having twice the trunk parameters. One ResNet34 trunk (~21.3M)
    # against two ResNet18 trunks (~22.4M) -- see §97 for the measured counts.
    #
    # `like_resnet` lives in the CONDITION, not only in the CLI flag, because
    # the run directory is named `{condition}_seed{n}` with no architecture in
    # it. Passing --like-resnet alone would write a ResNet34 run into
    # `dkl_bcs_single_seed0`, overwriting the existing ResNet18 checkpoints and
    # letting --resume load them. Binding the architecture to the condition
    # name makes the two impossible to separate (§87 lost an arm to exactly
    # this class of collision).
    "dkl_bcs_single34": dict(input_mode="dkl", blur=True, cs=True,
                             single_stream=True, like_resnet="resnet34"),
    # TDANN spatial-loss pilot (docs/PLAN_TSN_SPATIAL_PILOT_2026-09-25.md):
    # magno/parvo trunks through layer2, shared layer3-layer4
    # (FusedTwoStreamNetwork), TDANN relative spatial loss on all 8 ResNet-18
    # depths with weight `spatial_alpha` per depth (Finzi et al.'s supervised
    # optimum, TDANN config lwx10 = 2.5). alpha = 0 builds the same sheets and
    # logs their loss without training on it. Architecture and alpha live in
    # the condition name for the reason given above dkl_bcs_single34.
    "dkl_bcs_fused_a0":   dict(input_mode="dkl", blur=True, cs=True,
                               fused=True, spatial_alpha=0.0),
    "dkl_bcs_fused_a2p5": dict(input_mode="dkl", blur=True, cs=True,
                               fused=True, spatial_alpha=2.5),
}

# Fusion Stage 1 (TSN_FUSION_PROPOSAL_2026-10-09.md §4): 50-epoch pilots of a
# late-fusion readout on the two ResNet-18 trunks (src/models/late_fusion.py),
# all under one recipe: AdamW + cosine decay + EMA, and per-stream auxiliary CE
# heads (weight 0.3 each) so neither trunk is gradient-starved by the joint
# head. The recipe lives in the condition for the reason given above
# dkl_bcs_single34: run directories carry only the condition name, so a
# recipe set by CLI flag could be resumed under a different one.
# Weight decay: the proposal's 5e-5 is an L2-style coefficient. Decoupled
# AdamW multiplies it by the lr (1e-4 x 5e-5 = 5e-9 shrink per step), which is
# a no-op, so the pilots use the standard AdamW pairing of lr 1e-3 with wd
# 0.05, no decay on biases and norm parameters, 2 warm-up epochs.
STAGE1_RECIPE = dict(optimizer="adamw", lr=1e-3, weight_decay=0.05,
                     schedule="cosine", warmup_epochs=2, ema_decay=0.9995,
                     aux_weight=0.3)


def _stage1(fusion: str, *, highsf: bool = False, blur: bool = True,
            **kw) -> dict:
    return dict(input_mode="dkl", blur=blur, cs=True, parvo_highsf=highsf,
                fusion=fusion, recipe=STAGE1_RECIPE, **kw)


CONDITIONS.update({
    # A0: the current tied average head, new recipe + aux heads
    "fs1_bcs_avg":         _stage1("average"),
    # A1: untied additive evidence z = w z_M + (1 - w) z_P, dieted input
    "fs1_bcs_add_w050":    _stage1("additive", w=0.5),
    "fs1_bcs_add_w070":    _stage1("additive", w=0.7),
    "fs1_bcs_add_w085":    _stage1("additive", w=0.85),
    # A1': the same on the parvo_highsf input
    "fs1_hsf_add_w050":    _stage1("additive", w=0.5, highsf=True),
    "fs1_hsf_add_w070":    _stage1("additive", w=0.7, highsf=True),
    "fs1_hsf_add_w085":    _stage1("additive", w=0.85, highsf=True),
    # A2: coarse-to-fine hinge z = z_P + lam min(0, z_M - z_M,(k))
    "fs1_hsf_hinge_k3_l2": _stage1("hinge", k=3, lam=2.0, highsf=True),
    "fs1_hsf_hinge_k3_l4": _stage1("hinge", k=3, lam=4.0, highsf=True),
    # A3: A1 w = 0.7 with the magno trunk at 64 px (3x3/1 stem, no max-pool)
    "fs1_bcs_add_w070_mstem64": _stage1("additive", w=0.7, magno_stem64=True),
    # A4: A1 w = 0.7 at two lighter diet doses
    "fs1_cs_add_w070":     _stage1("additive", w=0.7, blur=False),
    "fs1_b2cs_add_w070":   _stage1("additive", w=0.7, blur_sigma=2.0),
})
STAGE1_CONDITIONS = [c for c in CONDITIONS if c.startswith("fs1_")]

# Fusion Stage 2 (2026-10-12): the new main TSN keeps the BA's single
# objective -- one cross-entropy on the network output. The per-stream heads
# are trained on detached features (detach_aux), so they are linear readouts
# of each stream (thesis Fig. 7) and send no gradient into the trunks. Every
# arm uses the BA's tied average head with magno weight w in training and at
# test time (the thesis's M-heavy variant is w = 0.9; stream1_weight records
# it for TwoStreamNetwork-based tools).
STAGE2_RECIPE = dict(STAGE1_RECIPE, aux_weight=1.0, label_smoothing=0.0)


def _stage2(w: float = 0.5, *, highsf: bool = False,
            label_smoothing: float = 0.0, ogm: dict | None = None,
            epochs: int | None = None) -> dict:
    recipe = dict(STAGE2_RECIPE, label_smoothing=label_smoothing)
    if ogm:
        recipe["ogm"] = ogm
    if epochs:
        recipe["epochs"] = epochs
    return dict(_stage1("average", highsf=highsf, w=w),
                stream1_weight=w, detach_aux=True, recipe=recipe)


# Stage 2b (2026-10-13): fs2_bcs_avg_w050 learns most of the task in the
# parvo trunk (alone: parvo 0.62, magno 0.25 top-1 at epoch 49), so the
# shape-biased magno stream barely shapes the decision. OGM-GE (class OGMGE
# below) slows the dominant trunk while it dominates, with the loss unchanged.
# alpha = 0.1 and modulation over the first half of training (epochs 0-25 of
# 50) are the reference defaults (0.1 for VGGSound; epochs 0-50 of 100).
OGM_DEFAULT = dict(alpha=0.1, start=0, end=25, ge=True)


CONDITIONS.update({
    # the BA's TSN under the new optimiser, single objective: main candidate
    "fs2_bcs_avg_w050":    _stage2(0.5),
    # more magno weight in training and test: shape bias up
    "fs2_bcs_avg_w070":    _stage2(0.7),
    "fs2_bcs_avg_w090":    _stage2(0.9),
    # accuracy lever: label smoothing 0.1
    "fs2_bcs_avg_w050_ls": _stage2(0.5, label_smoothing=0.1),
    # accuracy lever: parvo also gets full-acuity luminance; magno-heavy w
    # to hold shape bias
    "fs2_hsf_avg_w085":    _stage2(0.85, highsf=True),
    # Stage 2b: the main candidate with balanced stream learning (OGM-GE)
    "fs2_bcs_avg_w050_ogm": _stage2(0.5, ogm=OGM_DEFAULT),
})

# Stage 2c (2026-10-14): OGM-GE did not help (0.641 @ SB 0.61; magno alone
# 0.25 as without it). In every 50-epoch arm the joint training CE falls to
# ~0.1 through the parvo trunk (magno train CE stays ~2.5), and from epoch
# ~22 val top-1, joint shape bias and the magno readout all decline together
# (w050: 0.669 / 0.68 / 0.27 at epoch 20 -> 0.653 / 0.63 / 0.25 at 49).
# Stage 2c combines the two levers that helped (label smoothing 0.1: +1.6 pp;
# w = 0.7: +0.02 SB) and ends the cosine schedule at epoch 25, before the
# memorisation phase. The schedule length is fixed in advance, not chosen on
# the validation curve.
CONDITIONS.update({
    "fs2_bcs_avg_w070_ls":     _stage2(0.7, label_smoothing=0.1),
    "fs2_bcs_avg_w050_ls_e25": _stage2(0.5, label_smoothing=0.1, epochs=25),
    "fs2_bcs_avg_w070_ls_e25": _stage2(0.7, label_smoothing=0.1, epochs=25),
})
STAGE2_CONDITIONS = [c for c in CONDITIONS if c.startswith("fs2_")]


class H5ImageDataset(Dataset):
    """Lazily-opened HDF5 image/label dataset (worker-safe).

    `cache_ram` reads the whole split into a resident uint8 tensor once, up
    front, and serves batches from it. The dataset lives on Lustre, where the
    per-image random reads of the uncached path are latency-bound: measured
    throughput was 127 img/s at 8 workers and still only 591 at 32, against a
    model that can consume 670 img/s. Caching removes that ceiling for a
    one-off sequential read (~48 GB for the 263200-image train split, ~150 s).

    That 48 GB fits an a5000 node (95 GB) only if the job takes essentially
    the whole node, so caching trades three of the node's four GPUs for the
    speedup -- worth it because a cached run is ~16 h against ~36 h for two
    concurrent uncached ones. The larger-memory alternatives are not usable
    here: begendiv rejects this account, and g017 (2 TB, 8x B200) does not
    mount /scratch at all, so neither the dataset nor the conda env is
    visible from it.
    """

    def __init__(self, path: str, split: str, image_key: str = "data",
                 label_key: str = "labels", cache_ram: bool = False):
        self.path, self.split = path, split
        self.image_key, self.label_key = image_key, label_key
        self._f = None
        self.cache = None
        with h5py.File(path, "r") as f:
            d = f[split][image_key]
            self.length = d.shape[0]
            self.labels = np.asarray(f[split][label_key][:]).reshape(-1)
            if cache_ram:
                gb = d.size / 2**30
                print(f"  caching {split} into RAM ({gb:.1f} GB)...",
                      flush=True)
                t0 = time.time()
                self.cache = torch.empty(d.shape, dtype=torch.uint8)
                # Sequential block reads: Lustre serves these far better than
                # the random single-image reads of the uncached path.
                block = 2048
                buf = self.cache.numpy()
                for i in range(0, self.length, block):
                    d.read_direct(buf, np.s_[i:i + block],
                                  np.s_[i:i + block])
                # Deliberately NOT share_memory_(): that would copy all 48 GB
                # into /dev/shm, which is typically far smaller. Forked
                # DataLoader workers already see these pages copy-on-write,
                # and the cache is only ever read, so no copy is triggered.
                print(f"  cached in {time.time() - t0:.0f}s "
                      f"({gb / (time.time() - t0):.2f} GB/s)", flush=True)

    def __len__(self) -> int:
        return self.length

    def _file(self):
        if self._f is None:  # opened per worker process
            self._f = h5py.File(self.path, "r", swmr=True)
        return self._f

    def __getitem__(self, idx: int):
        if self.cache is not None:
            return self.cache[idx].permute(2, 0, 1), int(self.labels[idx])
        img = self._file()[self.split][self.image_key][idx]  # (H, W, C) uint8
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return x, int(self.labels[idx])


def build_augmentations(input_mode: str):
    """Thesis augmentation set (DVD-B); grayscale skipped for DKL inputs."""
    import kornia.augmentation as K

    aug = [
        K.RandomHorizontalFlip(p=0.25),
        K.RandomRotation(degrees=15.0, p=0.25),
    ]
    if input_mode == "rgb":
        aug.append(K.RandomGrayscale(p=0.5))
    aug += [
        K.ColorJitter(brightness=0.4, p=0.5),
        K.RandomEqualize(p=0.5),
        K.RandomPerspective(0.3, p=0.5),
        K.RandomSharpness(0.5, p=0.5),
        K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
    ]
    return nn.Sequential(*aug)


class StreamInputBuilder:
    """Turns a raw batch into per-stream inputs for a given condition."""

    def __init__(self, cfg: dict, age_months: int = 48):
        self.cfg = cfg
        self.age_months = age_months
        self.input_mode = cfg["input_mode"]
        self.blur, self.cs = cfg["blur"], cfg["cs"]
        # None keeps dkl_frontend.BLUR_SIGMA (4), the diet of every run
        # before the Stage-1 dose arm.
        self.blur_sigma = cfg.get("blur_sigma")
        self.target = cfg.get("transform_target", "magno")
        self.parvo_highsf = cfg.get("parvo_highsf", False)
        self.single_stream = cfg.get("single_stream", False)

    def __call__(self, x: torch.Tensor):
        from src.analysis.encoding.dkl_frontend import (
            apply_gaussian_blur,
            apply_scale_free_contrast,
        )

        if self.input_mode == "rgb":
            x1 = x2 = x
            if self.blur:
                x1 = apply_gaussian_blur(x1)
            if self.cs:
                x1 = apply_scale_free_contrast(x1, self.age_months)
            return x1, x2

        lum, rg = x[:, 0:1], x[:, 1:2]
        tgt = lum if self.target == "magno" else rg
        if self.blur:
            tgt = (apply_gaussian_blur(tgt) if self.blur_sigma is None
                   else apply_gaussian_blur(tgt, sigma=self.blur_sigma))
        if self.cs:
            tgt = apply_scale_free_contrast(tgt, self.age_months)
        if self.target == "magno":
            lum = tgt
        else:
            rg = tgt
        if self.single_stream:
            # One stream, both channels stacked. x_2=None selects the
            # TwoStreamNetwork baseline forward path.
            return torch.cat([lum, rg], dim=1), None
        if self.parvo_highsf:
            # parvo = high-acuity luminance + red-green (untransformed lum)
            return lum, torch.cat([x[:, 0:1], x[:, 1:2]], dim=1)
        return lum, rg


def in_channels_for(cfg: dict) -> tuple[int, int]:
    if cfg.get("single_stream"):
        # Stream 2 does not exist; 0 records that in config.json rather than
        # leaving a misleading 1 for a stream that was never built.
        return 2, 0
    if cfg["input_mode"] == "rgb":
        return 3, 3
    return 1, (2 if cfg.get("parvo_highsf") else 1)


def patch_stream_input_channels(model: TwoStreamNetwork, c1: int, c2: int):
    """Set each stream's stem conv to its required channel count."""
    for block, c in ((model.stream1_block1, c1), (model.stream2_block1, c2)):
        if block is None:
            continue
        old = block[0]
        if isinstance(old, nn.Conv2d) and old.in_channels != c:
            block[0] = nn.Conv2d(
                c, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=old.bias is not None,
            )


def setup_sheets(model, train_ds, out_dir: str, seed: int, device: str):
    """Cortical sheets for every `model.sheet_modules()` output + capture hooks.

    Layer shapes come from one dummy forward at the dataset's image size. On a
    resumed run the saved sheets are reloaded and checked against a rebuild, so
    a resumed run can never train against a different sheet.
    """
    capture = topographic.SheetCapture(model.sheet_modules())
    _, h, w = train_ds[0][0].shape
    was_training = model.training
    model.eval()
    capture.active = True
    with torch.no_grad():
        model(torch.zeros(1, 1, h, w, device=device),
              torch.zeros(1, 1, h, w, device=device))
    capture.active = False
    dims = {n: tuple(f.shape[1:]) for n, f in capture.feats.items()}
    capture.clear()
    model.train(was_training)

    sheet_dir = os.path.join(out_dir, "sheets")
    area = lambda n: topographic.RESNET18_AREA[depth_of(n)]  # noqa: E731
    sheets = topographic.build_model_sheets(dims, area, seed)
    if os.path.isdir(sheet_dir):
        for s in sheets:
            old = topographic.LayerSheet.from_npz(
                os.path.join(sheet_dir, f"{s.name}.npz"))
            if not (np.array_equal(old.coordinates, s.coordinates)
                    and np.array_equal(old.neighborhoods, s.neighborhoods)):
                raise RuntimeError(f"saved sheet {s.name} differs from rebuild")
    else:
        os.makedirs(sheet_dir + ".tmp", exist_ok=True)
        for s in sheets:
            s.to_npz(os.path.join(sheet_dir + ".tmp", f"{s.name}.npz"))
        os.replace(sheet_dir + ".tmp", sheet_dir)
    for s in sheets:
        print(f"  sheet {s.name} dims={s.dims} neighbourhoods="
              f"{s.neighborhoods.shape} width={s.neighborhood_width} mm",
              flush=True)
    return capture, topographic.SheetLoss(sheets).to(device)


def depth_losses(sl: dict) -> dict:
    """Per-sheet SL -> per-depth SL (trunk depths average magno and parvo), so
    alpha weights each of TDANN's 8 ResNet-18 depths once."""
    by_depth: dict = {}
    for name, v in sl.items():
        by_depth.setdefault(depth_of(name), []).append(v)
    return {d: torch.stack(v).mean() for d, v in by_depth.items()}


@torch.no_grad()
def evaluate(model, builder, loader, device, max_batches=None, amp=False):
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True).float() / 255.0
        y = y.to(device, non_blocking=True)
        x1, x2 = builder(x)
        if amp:
            x1 = x1.contiguous(memory_format=torch.channels_last)
            if x2 is not None:
                x2 = x2.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x1, x2)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


# ---------------------------- fusion Stage 1 ------------------------------

@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float, n: int) -> None:
    """EMA of parameters and float buffers (BN running statistics, as in
    timm's ModelEma), with the usual short-run warm-up of the decay; integer
    buffers (num_batches_tracked) are copied."""
    d = min(decay, (1.0 + n) / (10.0 + n))
    e_t, m_t = [], []
    for e, m in zip(list(ema.parameters()) + list(ema.buffers()),
                    list(model.parameters()) + list(model.buffers())):
        if e.dtype.is_floating_point:
            e_t.append(e)
            m_t.append(m.detach())
        else:
            e.copy_(m)
    torch._foreach_mul_(e_t, d)
    torch._foreach_add_(e_t, m_t, alpha=1.0 - d)


def load_cue_conflict(path: str) -> dict:
    """Cue-conflict cache (scripts/fusion_stage1_cue_conflict_cache.py):
    x (N, H, W, 3) uint8 DKL exactly as the training h5 stores it, the shape
    and texture labels, and the MiniEcoset columns of the 16 Geirhos classes."""
    z = np.load(path, allow_pickle=False)
    return {"x": torch.from_numpy(z["x"]), "shapes": z["shapes"].astype(str),
            "textures": z["textures"].astype(str),
            "classes": z["classes"].astype(str),
            "cols": torch.from_numpy(z["cols"].astype(np.int64))}


def cue_conflict_scores(logits: torch.Tensor, cc: dict) -> dict:
    """16-way restricted decision -> shape bias and coverage."""
    dec = cc["classes"][logits[:, cc["cols"]].argmax(1).numpy()]
    n_s = int((dec == cc["shapes"]).sum())
    n_t = int((dec == cc["textures"]).sum())
    n = len(cc["shapes"])
    return {"sb": n_s / max(n_s + n_t, 1), "cover": (n_s + n_t) / n,
            "n_shape": n_s, "n_texture": n_t, "n": n}


def _heads(model, builder, x, amp):
    x1, x2 = builder(x)
    if amp:
        x1 = x1.contiguous(memory_format=torch.channels_last)
        x2 = x2.contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        return model.forward_heads(x1, x2)


@torch.no_grad()
def evaluate_fusion(ema, raw, builder, loader, cc, device, amp,
                    max_batches=None) -> tuple[dict, dict]:
    """One pass over val (and the cue-conflict set) for a late-fusion network.

    The EMA weights are the model of record: every head is scored, and the
    two component logits are swept as w z_M + (1 - w) z_P (for `average` the
    tied head's own w-sweep, which is the A0 frontier of the decision rule,
    and the aux heads' sweep). The raw weights get joint top-1 only.
    Returns (metrics, logits) with logits on the CPU for the final dump.
    """
    ema.eval()
    raw.eval()
    keep: dict[str, list] = {}
    ys, raw_correct = [], 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True).float() / 255.0
        y = y.to(device, non_blocking=True)
        for k, v in _heads(ema, builder, x, amp).items():
            keep.setdefault(f"val_{k}", []).append(v.cpu())
        raw_correct += (_heads(raw, builder, x, amp)["joint"].argmax(1)
                        == y).sum().item()
        ys.append(y.cpu())
    if cc is not None:
        bs = loader.batch_size
        n_cc = len(cc["x"]) if max_batches is None else min(
            len(cc["x"]), max_batches * bs)
        for s in range(0, n_cc, bs):
            x = cc["x"][s:min(s + bs, n_cc)].to(device).permute(0, 3, 1, 2)
            x = x.float() / 255.0
            for k, v in _heads(ema, builder, x, amp).items():
                keep.setdefault(f"cc_{k}", []).append(v.cpu())
    raw.train()
    ema.eval()
    logits = {k: torch.cat(v) for k, v in keep.items()}
    y = torch.cat(ys)
    logits["val_y"] = y

    def top1(z):
        return float((z.argmax(1) == y).float().mean())

    heads = [k[4:] for k in logits if k.startswith("val_") and k != "val_y"]
    m = {"val": {h: top1(logits[f"val_{h}"]) for h in heads},
         "val_top1_raw": raw_correct / max(len(y), 1)}
    sub = None
    if cc is not None:
        sub = {**cc, "shapes": cc["shapes"][:len(logits["cc_joint"])],
               "textures": cc["textures"][:len(logits["cc_joint"])]}
        m["cc"] = {h: cue_conflict_scores(logits[f"cc_{h}"], sub)
                   for h in heads}
    pairs = [("untied", "magno", "parvo")]
    if "val_tied_magno" in logits:
        pairs.insert(0, ("tied", "tied_magno", "tied_parvo"))
    m["sweep"] = {}
    for name, a, b in pairs:
        rows = []
        for w in [i / 10 for i in range(11)]:
            r = {"w": w, "val_top1": top1(w * logits[f"val_{a}"]
                                          + (1 - w) * logits[f"val_{b}"])}
            if sub is not None:
                s = cue_conflict_scores(w * logits[f"cc_{a}"]
                                        + (1 - w) * logits[f"cc_{b}"], sub)
                r.update(sb=s["sb"], cover=s["cover"])
            rows.append(r)
        m["sweep"][name] = rows
    return m, logits


def build_stage1_optimizer(model, recipe, steps_per_epoch, epochs):
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": recipe["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    if recipe.get("ogm"):
        # OGM-GE sets each trunk's conv-weight lr per step, so those weights
        # get their own groups. Same hyperparameters as the decay group and
        # AdamW is per-parameter, so at k = 1 the optimisation is unchanged.
        conv = {name: [p for p in getattr(model, block).parameters()
                       if p.ndim == 4]
                for name, block in (("magno_conv", "stream1_block1"),
                                    ("parvo_conv", "stream2_block1"))}
        moved = {id(p) for ps in conv.values() for p in ps}
        groups[0]["params"] = [p for p in decay if id(p) not in moved]
        groups += [{"params": ps, "weight_decay": recipe["weight_decay"],
                    "name": name} for name, ps in conv.items()]
    optimizer = torch.optim.AdamW(groups, lr=recipe["lr"], betas=(0.9, 0.999))
    total = epochs * steps_per_epoch
    warm = recipe["warmup_epochs"] * steps_per_epoch

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        t = min(1.0, (s - warm) / max(1, total - warm))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class OGMGE:
    """On-the-fly gradient modulation with generalisation enhancement (Peng et
    al., CVPR 2022; github.com/GeWu-Lab/OGM-GE_CVPR2022) for the `average`
    head. Each step, s_M and s_P are the summed softmax probabilities of the
    true class under each stream's share of the decision
    (LateFusionNetwork.stream_contributions); with r = s_P / s_M, the
    dominant stream gets k = 1 - tanh(alpha * r) (r > 1: parvo) or
    1 - tanh(alpha / r) (else: magno), the other stream 1.

    Two departures from the reference code, both forced by AdamW:
      - k scales the dominant trunk's conv-weight learning rate for the step
        instead of its gradient. Adam's update m / sqrt(v) does not change
        when a gradient is rescaled, so the reference's SGD form would do
        almost nothing here; under SGD the two are the same.
      - GE as in the reference: Gaussian noise with the gradient tensor's own
        std is added to every conv-weight gradient of both trunks.
    Active for epochs start..end inclusive (the reference's convention). The
    loss is unchanged: one cross-entropy on the network output.
    """

    def __init__(self, optimizer, cfg: dict):
        self.alpha = float(cfg["alpha"])
        self.start, self.end = int(cfg["start"]), int(cfg["end"])
        self.ge = bool(cfg.get("ge", True))
        self.groups = {g["name"]: g for g in optimizer.param_groups
                       if g.get("name") in ("magno_conv", "parvo_conv")}
        assert sorted(self.groups) == ["magno_conv", "parvo_conv"]
        self.reset()

    def reset(self) -> None:
        self.n, self.n_active, self.sums = 0, 0, [0.0, 0.0, 0.0]

    @torch.no_grad()
    def modulate(self, model, out, y, epoch: int):
        """Between backward() and optimizer.step(); returns the learning rates
        to restore after the step (None outside the window)."""
        c_m, c_p = model.stream_contributions(out)
        s_m = F.softmax(c_m, dim=1).gather(1, y[:, None]).sum()
        s_p = F.softmax(c_p, dim=1).gather(1, y[:, None]).sum()
        r = float(s_p / s_m.clamp_min(1e-12))
        k_m, k_p = ((1.0, 1.0 - math.tanh(self.alpha * r)) if r > 1
                    else (1.0 - math.tanh(self.alpha / r), 1.0))
        self.n += 1
        for i, v in enumerate((r, k_m, k_p)):
            self.sums[i] += v
        if not self.start <= epoch <= self.end:
            return None
        self.n_active += 1
        if self.ge:
            for g in self.groups.values():
                for p in g["params"]:
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad)
                                    * (p.grad.std() + 1e-8))
        saved = {name: g["lr"] for name, g in self.groups.items()}
        self.groups["magno_conv"]["lr"] *= k_m
        self.groups["parvo_conv"]["lr"] *= k_p
        return saved

    def restore(self, saved) -> None:
        if saved:
            for name, lr in saved.items():
                self.groups[name]["lr"] = lr

    def summary(self) -> dict:
        n = max(self.n, 1)
        return {"ratio_parvo_over_magno": self.sums[0] / n,
                "k_magno": self.sums[1] / n, "k_parvo": self.sums[2] / n,
                "active_steps": self.n_active, "steps": self.n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=112)
    # Default None, not "resnet18": the condition supplies the architecture
    # for arms that pin one (dkl_bcs_single34), and a truthy default here
    # would silently win over it. Falls back to resnet18 when neither sets it.
    p.add_argument("--like-resnet", default=None)
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop early (smoke test)")
    p.add_argument("--val-batches", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=10)
    # Measured on an A5000 (tmp_analysis/profile_opt.py): bf16 autocast plus
    # channels_last takes the model's fwd/bwd from 314 to 670 img/s and peak
    # memory from 20.5 to 11.2 GB at bs=256, cutting a 150-epoch run from
    # ~35 h to ~16 h. Adam keeps fp32 master weights, so this changes speed,
    # not the optimization. --no-amp restores the fp32 path.
    p.add_argument("--no-amp", action="store_true",
                   help="disable bf16 autocast + channels_last")
    p.add_argument("--resume", action="store_true",
                   help="continue from checkpoint_last.pt if present")
    p.add_argument("--cache-ram", action="store_true",
                   help="load the whole dataset into RAM (~48 GB train); "
                        "big-memory nodes only")
    p.add_argument("--cue-conflict-npz", default=None,
                   help="Stage-1 conditions: score cue-conflict shape bias "
                        "every epoch from this cache "
                        "(scripts/fusion_stage1_cue_conflict_cache.py)")
    args = p.parse_args()

    cfg = CONDITIONS[args.condition]
    # Stage-1 conditions carry their own optimizer recipe; --lr is ignored
    # for them (config.json records the recipe that was used).
    recipe = cfg.get("recipe")
    # A recipe that fixes its schedule length overrides --epochs, so the run
    # directory (named by condition only) cannot be resumed under another one.
    if recipe and recipe.get("epochs") and recipe["epochs"] != args.epochs:
        print(f"{args.condition}: recipe sets epochs = {recipe['epochs']} "
              f"(--epochs {args.epochs} ignored)", flush=True)
        args.epochs = recipe["epochs"]
    fusion = cfg.get("fusion")
    if fusion is not None and not recipe:
        raise ValueError(f"{args.condition}: late fusion needs the Stage-1 "
                         "recipe (EMA weights are what it evaluates)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Explicit --like-resnet wins (one-off probes), then the condition's own
    # architecture, then the historical default. Resolved before the run
    # directory is created so config.json records what was actually built.
    arch = args.like_resnet or cfg.get("like_resnet", "resnet18")

    run = f"{args.condition}_seed{args.seed}"
    out_dir = os.path.join(args.out_dir, run)
    os.makedirs(out_dir, exist_ok=True)

    train_ds = H5ImageDataset(args.dataset, "train",
                              cache_ram=args.cache_ram)
    val_ds = H5ImageDataset(args.dataset, "val",
                            cache_ram=args.cache_ram)
    # With the dataset already resident, worker processes buy nothing: there
    # is no I/O left to overlap, only a ~50 MB per-batch memcpy against a
    # ~380 ms GPU step. They also cost a great deal -- forking 24 children
    # off a 48 GB parent fails outright with OSError(12) under the job's
    # memory cgroup, even though the pages are copy-on-write.
    n_workers = args.num_workers
    if args.cache_ram and n_workers > 0:
        print(f"  cache_ram: forcing num_workers {n_workers} -> 0", flush=True)
        n_workers = 0
    common = dict(num_workers=n_workers, pin_memory=True,
                  persistent_workers=n_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, **common)

    c1, c2 = in_channels_for(cfg)
    single = cfg.get("single_stream", False)
    if cfg.get("fused"):
        model = FusedTwoStreamNetwork(num_classes=args.num_classes,
                                      in_channels=(c1, c2))
        arch = "fused_resnet18_after_layer2"
    else:
        model = TwoStreamNetwork(
            num_classes=args.num_classes,
            like_resnet=arch,
            seperate_from_layer=0,
            # baseline=True builds ONE trunk (stream2_block1 = None) and takes
            # the x_2=None forward path; combined_color makes its stem
            # 2-channel so the single stream can receive both DKL channels.
            baseline=single,
            combined_color=single,
            dvd_strategy="dkl" if cfg["input_mode"] == "dkl" else "dvd",
            stream_concat=cfg.get("stream_concat", False),
            stream1_weight=cfg.get("stream1_weight", 0.5),
            legacy_asymmetric_mixing=False,  # corrected symmetric 75/25
        )
        patch_stream_input_channels(model, c1, c2)
        if cfg.get("magno_stem64"):
            make_magno_stem64(model.stream1_block1, c1)
        if fusion is not None:
            model = LateFusionNetwork(model, fusion, w=cfg.get("w", 0.5),
                                      k=cfg.get("k", 3),
                                      lam=cfg.get("lam", 0.0),
                                      detach_aux=cfg.get("detach_aux", False))
    model = model.to(device)

    # Spatial-loss conditions: sheets are built (or reloaded on resume) before
    # the first step and saved next to the checkpoints; they are a pure
    # function of the layer shapes and the seed, so both pilot arms of a seed
    # share them.
    spatial = "spatial_alpha" in cfg
    alpha = float(cfg.get("spatial_alpha", 0.0))
    capture = sheet_loss = None
    if spatial:
        capture, sheet_loss = setup_sheets(model, train_ds, out_dir, args.seed,
                                           device)

    amp = (not args.no_amp) and device == "cuda" and torch.cuda.is_bf16_supported()
    if amp:
        model = model.to(memory_format=torch.channels_last)

    builder = StreamInputBuilder(cfg, age_months=args.age_months)
    augment = build_augmentations(cfg["input_mode"]).to(device)
    scheduler = ema = cc = None
    if recipe:
        optimizer, scheduler = build_stage1_optimizer(
            model, recipe, len(train_loader), args.epochs)
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     betas=(0.9, 0.999))
    ogm = None
    if recipe and recipe.get("ogm"):
        if fusion != "average":
            raise ValueError("OGM-GE is implemented for the average head only")
        ogm = OGMGE(optimizer, recipe["ogm"])
    if fusion is not None and args.cue_conflict_npz:
        cc = load_cue_conflict(args.cue_conflict_npz)
        print(f"  cue-conflict cache: {len(cc['x'])} images", flush=True)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=(recipe or {}).get("label_smoothing", 0.0))

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        # `like_resnet` in vars(args) is None whenever the condition supplied
        # the architecture, so record what was actually built rather than
        # leaving the resolved value inferable only from condition_cfg.
        json.dump({**vars(args), "condition_cfg": cfg,
                   "like_resnet_resolved": arch,
                   "n_trunk_params": sum(p.numel() for p in model.parameters()),
                   "in_channels": [c1, c2],
                   "spatial": ({"alpha_per_depth": alpha,
                                "sheets": sheet_loss.names,
                                "sheet_seed": args.seed,
                                "neighborhoods_per_step": 1}
                               if spatial else None)}, f, indent=2)

    print(f"[{run}] train={len(train_ds)} val={len(val_ds)} device={device} "
          f"arch={arch} params={sum(p.numel() for p in model.parameters()):,} "
          f"in_channels={c1},{c2} amp={amp} workers={n_workers}",
          flush=True)

    # Resume: these runs are ~3 days each, so a node failure or a
    # walltime kill must not cost the whole run. checkpoint_last.pt carries
    # the optimizer state too -- restoring weights alone would silently
    # restart Adam's moments and change the optimization.
    history, step, start_epoch = [], 0, 0
    last_path = os.path.join(out_dir, "checkpoint_last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ck["scheduler"])
        if ema is not None:
            ema.load_state_dict(ck["ema_state_dict"])
        start_epoch = ck["epoch"] + 1
        history = ck.get("history", [])
        step = ck.get("step", 0)
        print(f"[{run}] resumed from epoch {ck['epoch']} "
              f"-> starting at {start_epoch}", flush=True)

    def snapshot(epoch):
        """For Stage-1 arms `state_dict` is the EMA network (the one every
        logged metric scores) and the raw weights ride along."""
        snap = {"epoch": epoch, "config": {**cfg, "in_channels": [c1, c2]}}
        if ema is not None:
            snap.update(state_dict=ema.state_dict(), weights="ema",
                        state_dict_raw=model.state_dict())
        else:
            snap["state_dict"] = model.state_dict()
        return snap

    logits = None
    for epoch in range(start_epoch, args.epochs):
        t0, running, seen = time.time(), 0.0, 0
        running_ce, running_sl, n_steps = 0.0, {}, 0
        running_parts = torch.zeros(3, device=device)
        if ogm is not None:
            ogm.reset()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).float() / 255.0
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                x = augment(x)
            x1, x2 = builder(x)
            if amp:
                x1 = x1.contiguous(memory_format=torch.channels_last)
                if x2 is not None:
                    x2 = x2.contiguous(memory_format=torch.channels_last)
            if spatial:
                capture.active = True
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                if fusion is not None:
                    out = model.forward_heads(x1, x2)
                    ce_parts = torch.stack([criterion(out[h], y) for h in
                                            ("joint", "magno", "parvo")])
                    loss = ce_parts[0] + recipe["aux_weight"] * ce_parts[1:].sum()
                    running_parts = running_parts + ce_parts.detach() * y.numel()
                else:
                    loss = criterion(model(x1, x2), y)
            ce = loss
            if spatial:
                # Outside autocast: the correlations are computed in fp32.
                capture.active = False
                with torch.set_grad_enabled(alpha > 0):
                    sl = depth_losses(sheet_loss(capture.feats))
                capture.clear()
                if alpha > 0:
                    loss = ce + alpha * torch.stack(list(sl.values())).sum()
                # Accumulated on the GPU: an .item() here would sync before
                # the backward pass is queued.
                for d, v in sl.items():
                    running_sl[d] = running_sl.get(d, 0.0) + v.detach()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            saved_lr = (ogm.modulate(model, out, y, epoch)
                        if ogm is not None else None)
            optimizer.step()
            if ogm is not None:
                ogm.restore(saved_lr)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema_update(ema, model, recipe["ema_decay"], step)
            running += loss.item() * y.numel()
            running_ce += ce.detach() * y.numel()
            seen += y.numel()
            n_steps += 1
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        if fusion is not None:
            # val_top1 is the EMA network's joint decision, the model of
            # record for the Stage-1 arms; the raw weights are logged beside.
            m, logits = evaluate_fusion(ema, model, builder, val_loader, cc,
                                        device, amp,
                                        max_batches=args.val_batches)
            val_acc = m["val"]["joint"]
        else:
            val_acc = evaluate(model, builder, val_loader, device,
                               max_batches=args.val_batches, amp=amp)
        rec = {"epoch": epoch, "train_loss": running / max(seen, 1),
               "val_top1": val_acc, "sec": round(time.time() - t0, 1)}
        if fusion is not None:
            rec["lr"] = scheduler.get_last_lr()[0]
            rec["train_ce"] = dict(zip(("joint", "magno", "parvo"),
                                       (running_parts / max(seen, 1)).tolist()))
            rec.update(m)
        if ogm is not None:
            rec["ogm"] = ogm.summary()
        if spatial:
            rec["train_ce"] = float(running_ce) / max(seen, 1)
            rec["train_sl"] = {d: float(v) / max(n_steps, 1)
                               for d, v in running_sl.items()}
        history.append(rec)
        print(f"[{run}] {rec}", flush=True)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        # Resume point, rewritten every epoch: cheap next to an epoch, and
        # it bounds the loss from an unexpected kill to one epoch rather
        # than to ckpt_every. Written to a temp path first so a kill during
        # the write cannot leave a truncated file as the only resume point.
        tmp_path = last_path + ".tmp"
        state = {"epoch": epoch, "step": step, "history": history,
                 "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "config": {**cfg, "in_channels": [c1, c2]}}
        if ema is not None:
            state["ema_state_dict"] = ema.state_dict()
            state["scheduler"] = scheduler.state_dict()
        torch.save(state, tmp_path)
        os.replace(tmp_path, last_path)
        snap = snapshot(epoch)
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            torch.save(snap, os.path.join(out_dir,
                                          f"checkpoint_{epoch + 1:03d}.pt"))
        if fusion is not None and val_acc >= max(h["val_top1"] for h in history):
            torch.save(snap, os.path.join(out_dir, "checkpoint_best.pt"))
        if args.max_steps is not None and step >= args.max_steps:
            break

    torch.save({k: v for k, v in snapshot(None).items() if k != "epoch"},
               os.path.join(out_dir, "checkpoint_final.pt"))
    if logits is not None:
        # Per-image logits of the final EMA network (val + cue-conflict), so
        # frontiers and bootstraps need no GPU re-run.
        np.savez_compressed(os.path.join(out_dir, "final_logits.npz"),
                            **{k: v.numpy() for k, v in logits.items()})
    print(f"[{run}] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
