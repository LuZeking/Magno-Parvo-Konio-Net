"""Does features.build_model load a parvo_highsf checkpoint correctly?

parvo_highsf trains stream2 with TWO input channels (untransformed luminance +
red-green). build_model has no in_channels argument, so if it builds a
1-channel stream2 the first conv shape mismatches, load_state_dict(strict=False)
reports it as missing+unexpected, and the guard -- which only fires when ALL
parameters are missing -- lets it through with that conv left RANDOM.

That failure mode would make parvo look catastrophically texture-biased for a
spurious reason, i.e. it would falsely confirm the §21.3 conjunction prediction.
Check before running the probe, not after.

Usage: PYTHONPATH=. python tmp_analysis/check_highsf_load.py
"""

from __future__ import annotations

import torch

from src.analysis.encoding import features

CK = ("/scratch/byrong/VST_data/mpk_runs/dkl_bcs_parvo_highsf_seed1"
      "/checkpoint_last.pt")

state = torch.load(CK, map_location="cpu", weights_only=False)
if "state_dict" in state:
    state = state["state_dict"]
ck_keys = {k: tuple(v.shape) for k, v in state.items() if hasattr(v, "shape")}

print("checkpoint first-conv shapes:")
for k, s in ck_keys.items():
    if "conv1.weight" in k and "layer" not in k:
        print(f"  {k:<34} {s}")

model = features.build_model(checkpoint=None)
print("\nbuild_model() first-conv shapes:")
built = {k: tuple(v.shape) for k, v in model.state_dict().items()}
for k, s in built.items():
    if "conv1.weight" in k and "layer" not in k:
        print(f"  {k:<34} {s}")

mismatch = {k: (built.get(k), s) for k, s in ck_keys.items()
            if k in built and built[k] != s}
only_ck = [k for k in ck_keys if k not in built]
print(f"\nshape mismatches: {len(mismatch)}")
for k, (b, c) in mismatch.items():
    print(f"  {k:<34} built {b}  ckpt {c}")
print(f"keys only in checkpoint: {len(only_ck)}")

print("\n--- what build_model actually does with it ---")
try:
    m = features.build_model(checkpoint=CK)
    # Did stream2's first conv actually receive the checkpoint's weights?
    sd = m.state_dict()
    for k in ck_keys:
        if "conv1.weight" in k and "layer" not in k and k in sd:
            same = (sd[k].shape == state[k].shape
                    and torch.allclose(sd[k], state[k]))
            print(f"  {k:<34} loaded={same}  shape={tuple(sd[k].shape)}")
    print("\nVERDICT: build_model returned a model. Any 'loaded=False' above is "
          "a SILENTLY RANDOM layer -- do not run the probe on this condition "
          "until build_model takes in_channels.")
except Exception as e:
    print(f"  raised {type(e).__name__}: {e}")
    print("\nVERDICT: it refuses. Safe -- needs an in_channels argument before "
          "parvo_highsf can be probed.")
