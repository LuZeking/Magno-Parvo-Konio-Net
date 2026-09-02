"""OOD robustness of the two-stream models on corrupted MiniEcoset.

PAPER_PLAN §53 (pre-registered). Geirhos et al.'s claim is that shape bias buys
robustness. This project has established the shape bias (§25) and never tested
the robustness, which is the single result that decides whether this is a
machine-learning paper or a neuroscience one.

DESIGN CHOICE: corruptions are applied to the RGB image BEFORE the front end,
which is where a real-world corruption happens -- the retina sees the corrupted
world. They are evaluated in the model's OWN 112-way MiniEcoset label space, so
there is no domain shift stacked on top of the corruption, unlike the
16-class ImageNet-derived OOD sets.

TWO STATISTICS (§8), and here they are not optional. The diet costs clean
accuracy (~0.37-0.45 vs 0.47-0.51, §18), so absolute accuracy under corruption
is confounded by where each model starts:

    absolute  = top-1 accuracy at severity s
    relative  = accuracy(s) / accuracy(clean)      <- degradation slope

A robustness claim is made only where both agree, and `relative` is the one
that addresses "is this model more robust", not "is this model better".

CONFOUND, DECLARED IN ADVANCE: the diet IS a low-pass blur plus a
contrast-sensitivity cut on the luminance channel. `low-pass` and `contrast`
are therefore partly trained-in and CANNOT support a robustness claim. They are
run and reported, never counted. The clean tests are uniform-noise, high-pass,
phase-scrambling, rotation and false-colour.

Usage:
    python -m tmp_analysis.ood_robustness --validate-frontend
    python tmp_analysis/ood_robustness.py --model-name dkl_bcs_seed0 \
        --checkpoint <ckpt> --out-dir results/ood
"""
from __future__ import annotations

import argparse, json, os
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.encoding import dkl_frontend, features

RGB_H5 = "/scratch/byrong/VST_data/VST_datasets/texture2shape_miniecoset_256px.h5"
DKL_H5 = "/scratch/byrong/VST_data/VST_datasets/miniecoset_256px_dkl_uint8.h5"


# ----------------------------------------------------------------- corruptions
# x is float32 (N,3,H,W) in [0,1]. Every corruption returns the same shape and
# range. Severity 0 of every family is the identity, which is the internal check
# that a family is wired up correctly: all seven clean accuracies must agree.

def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(x, sigma):
    if sigma <= 0:
        return x
    k = _gauss_kernel(sigma, x.device)
    p = (k.numel() - 1) // 2
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"),
                 k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"),
                 k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def c_uniform_noise(x, w, g):
    if w <= 0:
        return x
    return (x + (torch.rand(x.shape, generator=g, device=x.device) * 2 - 1) * w).clamp(0, 1)


def c_contrast(x, c, g):
    return ((x - 0.5) * c + 0.5).clamp(0, 1) if c < 1.0 else x


def c_low_pass(x, s, g):
    return _blur(x, s)


def c_high_pass(x, s, g):
    if s is None or s <= 0:
        return x
    return (x - _blur(x, s) + 0.5).clamp(0, 1)


def c_phase_scrambling(x, deg, g):
    """Add uniform phase noise in [-deg, deg]. One shared phase field across
    the three channels, so colour stays coherent and the manipulation is to
    spatial structure alone."""
    if deg <= 0:
        return x
    f = torch.fft.rfft2(x)
    noise = (torch.rand((x.shape[0], 1, f.shape[-2], f.shape[-1]),
                        generator=g, device=x.device) * 2 - 1) * (deg * np.pi / 180.0)
    return torch.fft.irfft2(torch.polar(f.abs(), f.angle() + noise),
                            s=x.shape[-2:]).clamp(0, 1)


def c_rotation(x, deg, g):
    k = {0: 0, 90: 1, 180: 2, 270: 3}[int(deg)]
    return x if k == 0 else torch.rot90(x, k, dims=(-2, -1))


def c_false_colour(x, on, g):
    """Opponent inversion: keep luminance, negate the colour difference. This
    is the manipulation the parvo stream should be most exposed to."""
    if not on:
        return x
    lum = x.mean(1, keepdim=True)
    return (2 * lum - x).clamp(0, 1)


# (name, fn, severities, counts_toward_the_claim)
FAMILIES = [
    ("uniform-noise",    c_uniform_noise,    [0, .03, .06, .10, .16, .25, .35], True),
    ("high-pass",        c_high_pass,        [None, 30.0, 20.0, 12.0, 8.0, 5.0, 3.0], True),
    ("phase-scrambling", c_phase_scrambling, [0, 30, 60, 90, 120, 150, 180],    True),
    ("rotation",         c_rotation,         [0, 90, 180, 270],                 True),
    ("false-colour",     c_false_colour,     [False, True],                     True),
    # trained-in: the diet IS blur + a CS cut. Reported, never counted.
    ("low-pass",         c_low_pass,         [0, 1, 3, 5, 7, 10, 15],           False),
    ("contrast",         c_contrast,         [1.0, .5, .3, .15, .10, .05, .03], False),
]


# ------------------------------------------------------------------ validation
def validate_frontend(n=64, tol=2.0):
    """Does our RGB->DKL front end reproduce the h5 the models actually trained
    on? If not, every number below is measured through a different retina than
    the one the weights were fitted for, and the experiment is void.

    Also confirms the RGB and DKL h5 test splits are the same images."""
    with h5py.File(RGB_H5, "r") as fr, h5py.File(DKL_H5, "r") as fd:
        rgb = fr["test"]["data"][:n]
        ref = fd["test"]["data"][:n].astype(np.float32)
        lab_r, lab_d = fr["test"]["labels"][:n], fd["test"]["labels"][:n]
        ch_min, ch_max = fd["dkl_channel_min"][:], fd["dkl_channel_max"][:]
    assert np.array_equal(lab_r, lab_d), "test labels differ between the two h5"
    got = dkl_frontend.rgb_to_dkl3(rgb, ch_min, ch_max)
    got = np.asarray(got.cpu() if torch.is_tensor(got) else got, dtype=np.float32)
    if got.ndim == 4 and got.shape[1] == 3:
        got = np.transpose(got, (0, 2, 3, 1))
    if got.max() <= 1.01:
        got = got * 255.0
    d = np.abs(got - ref)
    print(f"[validate] labels match; |RGB->DKL - stored DKL|: "
          f"mean {d.mean():.3f} max {d.max():.3f} (uint8 levels, tol {tol})")
    return d.mean() < tol


# ------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, rgb_u8, labels, ch_min, ch_max, args, device, bs=64):
    """Top-1 over every family x severity. Returns {family: [acc per severity]}."""
    g = torch.Generator(device=device).manual_seed(args.seed)
    out = {}
    for name, fn, sevs, _counts in FAMILIES:
        accs = []
        for s in sevs:
            correct = total = 0
            for i in range(0, len(rgb_u8), bs):
                chunk = rgb_u8[i:i + bs]
                x = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2).float() / 255.0
                x = fn(x, s, g)
                u8 = (x.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                xm, xp = dkl_frontend.magno_parvo_inputs(
                    u8, ch_min, ch_max,
                    lum_blur=not args.no_lum_blur, lum_cs=not args.no_lum_cs,
                    age_months=args.age_months, device=device,
                    transform_target=args.transform_target,
                    parvo_highsf=args.parvo_highsf,
                )
                y1 = model.stream1_block1(xm.to(device))
                y2 = model.stream2_block1(xp.to(device))
                if model.stream_concat:
                    y1, y2 = model._mix_streams(y1, y2)
                    y1, y2 = model.stream1_block2(y1), model.stream2_block2(y2)
                w = float(model.stream1_weight)
                pred = model.classifier(w * y1 + (1 - w) * y2).argmax(1).cpu().numpy()
                correct += int((pred == labels[i:i + bs]).sum()); total += len(chunk)
            accs.append(correct / total)
            print(f"  {name:17s} sev={str(s):6s} acc={accs[-1]:.4f}", flush=True)
        out[name] = accs
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate-frontend", action="store_true")
    p.add_argument("--model-name"); p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", default="results/ood")
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--transform-target", default="magno", choices=["magno", "parvo"])
    p.add_argument("--parvo-highsf", action="store_true")
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--stream-concat", action="store_true")
    p.add_argument("--stream1-weight", type=float, default=0.5)
    p.add_argument("--n-images", type=int, default=5600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ok = validate_frontend()
    if args.validate_frontend:
        raise SystemExit(0 if ok else 1)
    if not ok:
        raise SystemExit("front-end validation FAILED -- refusing to report numbers")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with h5py.File(RGB_H5, "r") as fr:
        rgb = fr["test"]["data"][:args.n_images]
        labels = fr["test"]["labels"][:args.n_images].astype(np.int64)
    with h5py.File(DKL_H5, "r") as fd:
        ch_min, ch_max = fd["dkl_channel_min"][:], fd["dkl_channel_max"][:]

    if args.checkpoint:
        features.assert_converged(args.checkpoint, False)
    model = features.build_model(
        checkpoint=args.checkpoint, stream_concat=args.stream_concat,
        stream1_weight=args.stream1_weight, device=device,
        in_channels=(1, 2 if args.parvo_highsf else 1),
    ).eval()

    print(f"[{args.model_name}] device={device} n={len(rgb)}", flush=True)
    accs = evaluate(model, rgb, labels, ch_min, ch_max, args, device)

    os.makedirs(args.out_dir, exist_ok=True)
    rec = {"model_name": args.model_name, "checkpoint": str(args.checkpoint),
           "n_images": int(len(rgb)),
           "frontend": {"lum_blur": not args.no_lum_blur, "lum_cs": not args.no_lum_cs,
                        "transform_target": args.transform_target,
                        "parvo_highsf": bool(args.parvo_highsf),
                        "age_months": args.age_months},
           "families": {n: {"severities": [str(s) for s in sv],
                            "accuracy": accs[n], "counts": c}
                        for n, _f, sv, c in FAMILIES}}
    path = os.path.join(args.out_dir, f"{args.model_name}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print("saved ->", path, flush=True)


if __name__ == "__main__":
    main()
