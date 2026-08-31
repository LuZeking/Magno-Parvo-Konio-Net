"""Are rgb_tsn's two streams identical tensors?

The probe returned bias(S1) - bias(S2) = +0.0000 on every fold. Either the
streams learned indistinguishable solutions, or they are the same weights.
Those mean completely different things, so check the checkpoint directly.
"""
import torch

for cond in ("rgb_tsn_seed0", "rgb_tsn_seed1", "dkl_bcs_seed0", "dkl_base_seed0"):
    ck = f"/scratch/byrong/VST_data/mpk_runs/{cond}/checkpoint_last.pt"
    st = torch.load(ck, map_location="cpu", weights_only=False)
    st = st.get("state_dict", st)
    pairs = [(k, k.replace("stream1", "stream2")) for k in st
             if k.startswith("stream1") and k.replace("stream1", "stream2") in st]
    n_same = 0
    maxdiff = 0.0
    for a, b in pairs:
        x, y = st[a].float(), st[b].float()
        if x.shape != y.shape:
            continue
        d = (x - y).abs().max().item()
        maxdiff = max(maxdiff, d)
        n_same += int(d == 0.0)
    print(f"{cond:<18} pairs={len(pairs):>3}  bit-identical={n_same:>3}  "
          f"max|S1-S2|={maxdiff:.3e}")
