"""Cue-conflict shape bias for the SINGLE-STREAM arm (§64), scored by the
two-stream scorer so the decision rule is identical.

`src.analysis.behavior.cue_conflict --baseline` cannot do this. That flag
builds a torchvision ResNet on raw 3-channel RGB with no DKL conversion and
no developmental diet -- it is the thesis's "RGB Baseline" anchor. Our
`dkl_bcs_single` is a `TwoStreamNetwork(baseline=True, combined_color=True)`
taking TWO DKL channels with the diet on luminance. Pointing `--baseline` at
it would mismatch both the architecture and the input, and because
`load_state_dict(strict=False)` tolerates missing keys it would have scored a
randomly initialised network without failing. Hence a separate entry point
with a STRICT load.

Mirrors `standalone_trainer.StreamInputBuilder` exactly: the diet is applied
to luminance, then `cat([lum, rg], dim=1)` feeds the one trunk with x_2=None.

Only the integrated readout exists here -- there are no streams, so per §64.2
no stream statistics are produced.

    python -m tmp_analysis.cue_conflict_single --checkpoint <pt> \
        --model-name dkl_bcs_single_seed0_ep050 --out-dir results/cue_conflict_single
"""
import argparse

import numpy as np
import torch

from src.analysis.behavior import cue_conflict as cc
from src.analysis.encoding import dkl_frontend
from src.models.pytorch_models import TwoStreamNetwork
from src.training.standalone_trainer import patch_stream_input_channels


def build_single(checkpoint: str, num_classes: int = 112, seed: int = 0,
                 device: str = "cpu") -> TwoStreamNetwork:
    """The trainer's own single-stream build, then a strict load."""
    torch.manual_seed(seed)
    model = TwoStreamNetwork(
        num_classes=num_classes,
        like_resnet="resnet18",
        seperate_from_layer=0,
        baseline=True,        # one trunk; stream2_block1 is None
        combined_color=True,  # 2-channel stem
        dvd_strategy="dkl",
        stream_concat=False,
        stream1_weight=0.5,
        legacy_asymmetric_mixing=False,
    )
    patch_stream_input_channels(model, 2, 0)

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # standalone_trainer saves the CONDITION config flattened, plus
    # in_channels: {'input_mode','blur','cs','single_stream','in_channels'}.
    cfg = blob.get("config", {}) if isinstance(blob, dict) else {}
    if cfg:
        if not cfg.get("single_stream", False):
            raise SystemExit(
                f"{checkpoint} is not a single_stream run (config={cfg}); "
                "this entry point is only for the §64 arm."
            )
        if list(cfg.get("in_channels", [])) != [2, 0]:
            raise SystemExit(
                f"{checkpoint} records in_channels="
                f"{cfg.get('in_channels')}, expected [2, 0]."
            )
        # The diet must match what the front end below will apply.
        print(f"[single] checkpoint config: input_mode={cfg.get('input_mode')} "
              f"blur={cfg.get('blur')} cs={cfg.get('cs')}", flush=True)

    state = blob.get("state_dict", blob.get("model", blob))
    if any(k.startswith("network.") for k in state):
        state = {k.removeprefix("network."): v for k, v in state.items()
                 if k.startswith("network.")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    # THE guard. strict=False is used only so both lists can be reported; a
    # nonempty either way means the architecture does not match the weights
    # and every number downstream would be partly random.
    if missing or unexpected:
        raise SystemExit(
            f"architecture/weights mismatch on {checkpoint}: "
            f"{len(missing)} missing {missing[:4]}, "
            f"{len(unexpected)} unexpected {unexpected[:4]}"
        )
    print(f"[single] strict load OK: {len(state)} tensors, "
          f"0 missing, 0 unexpected", flush=True)
    return model.eval().to(device)


@torch.no_grad()
def single_logits(model, x_lum, x_rg, batch_size: int = 32,
                  device: str = "cpu") -> dict[str, np.ndarray]:
    """cat the two DKL channels into the one trunk (StreamInputBuilder)."""
    assert x_lum.shape[1] == 1 and x_rg.shape[1] == 1, (
        f"expected 1 channel each, got {tuple(x_lum.shape)}, "
        f"{tuple(x_rg.shape)}")
    x = torch.cat([x_lum, x_rg], dim=1)
    out = []
    for s in range(0, x.shape[0], batch_size):
        # x_2 defaults to None, which selects the baseline forward path.
        out.append(model(x[s:s + batch_size].to(device)).float().cpu().numpy())
    return {"full": np.concatenate(out)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--age-months", type=int, default=48)
    p.add_argument("--no-lum-blur", action="store_true")
    p.add_argument("--no-lum-cs", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_index = cc.load_class_index()
    items = cc.list_cue_conflict_images()
    if not items:
        raise SystemExit("no cue-conflict images found")
    paths = [i[0] for i in items]
    shapes = [i[1] for i in items]
    textures = [i[2] for i in items]
    print(f"[{args.model_name}] {len(items)} cue-conflict images", flush=True)

    model = build_single(args.checkpoint, seed=args.seed, device=device)
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()

    parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(paths), 128):
        rgb = cc.load_images(paths[start:start + 128])
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb, ch_min, ch_max,
            lum_blur=not args.no_lum_blur, lum_cs=not args.no_lum_cs,
            age_months=args.age_months, device=device,
        )
        for k, v in single_logits(model, x_m, x_p, device=device).items():
            parts.setdefault(k, []).append(v)
        print(f"  {min(start + 128, len(paths))}/{len(paths)}", flush=True)

    logits = {k: np.concatenate(v) for k, v in parts.items()}
    cc.write_report(logits, shapes, textures, class_index, args)


if __name__ == "__main__":
    main()
