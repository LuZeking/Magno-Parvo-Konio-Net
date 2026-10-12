"""Late-fusion readouts over the two TwoStreamNetwork trunks (fusion Stage 1).

TSN_FUSION_PROPOSAL_2026-10-09.md §2 and §4. The trunks are the unchanged
`stream1_block1` / `stream2_block1` of a TwoStreamNetwork (same module names,
so trunk weights load into the existing analysis tools); only the readout on
the pooled 512-d stream features changes:

  average   z = W(w h_M + (1 - w) h_P) + b      the BA's tied head (A0: w = 0.5)
  additive  z = w z_M + (1 - w) z_P             untied, §2a (A1, A1', A3, A4)
  hinge     z = z_P + lam * min(0, z_M - z_M,(k))   coarse-to-fine, §2b (A2)

with z_M = W_M h_M + b_M and z_P = W_P h_P + b_P. Every Stage-1 arm also trains
z_M and z_P with their own cross-entropy (aux weight in the trainer), so each
component is readable alone and its logit scale is pinned by its own loss --
which is what makes w binding. For `average`, z_M / z_P are auxiliary heads
that do not enter the decision.

Stage 2 keeps the BA's single objective: `detach_aux=True` trains z_M and z_P
on detached features, so they are linear readouts of each stream (the
thesis's Fig. 7) that send no gradient into the trunks; only the joint
cross-entropy trains the network.

`magno_stem64` (A3): the magno trunk reads its dieted 256-px input average-
pooled to 64 px (the sigma = 4 blur leaves < 1 % amplitude at the 64-px
Nyquist frequency, so the pooling discards almost nothing), through a 3x3
stride-1 stem with no max-pool. Layer1 then sees the same 64x64 grid as the
standard 7x7/2 + max-pool stem, so everything downstream is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.pytorch_models import TwoStreamNetwork

FUSIONS = ("average", "additive", "hinge")


def hinge_penalty(z_m: torch.Tensor, k: int, lam: float) -> torch.Tensor:
    """lam * min(0, z_M - z_M,(k)): zero on magno's top-k, negative below."""
    kth = torch.topk(z_m, k, dim=1).values[:, k - 1:k]
    return lam * torch.clamp(z_m - kth, max=0.0)


def fuse(fusion: str, z_m: torch.Tensor, z_p: torch.Tensor, *, w: float = 0.5,
         k: int = 3, lam: float = 0.0) -> torch.Tensor:
    """The integration rule on per-stream logits (`average` is handled by the
    model, because its tied head is applied to the mixed features)."""
    if fusion == "additive":
        return w * z_m + (1.0 - w) * z_p
    if fusion == "hinge":
        return z_p + hinge_penalty(z_m, k, lam)
    raise ValueError(f"fuse() does not handle {fusion!r}")


def make_magno_stem64(block: nn.Sequential, in_channels: int) -> None:
    """Replace a ResNet stem (conv7x7/2 at [0], max-pool at [3]) in place."""
    old = block[0]
    block[0] = nn.Sequential(
        nn.AvgPool2d(4),
        nn.Conv2d(in_channels, old.out_channels, kernel_size=3, stride=1,
                  padding=1, bias=False),
    )
    block[3] = nn.Identity()


class LateFusionNetwork(nn.Module):
    def __init__(self, trunk: TwoStreamNetwork, fusion: str, w: float = 0.5,
                 k: int = 3, lam: float = 0.0, detach_aux: bool = False):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        if trunk.stream2_block1 is None or trunk.stream_concat:
            raise ValueError("late fusion needs two separate trunks")
        if detach_aux and fusion != "average":
            raise ValueError("detach_aux leaves the decision without trained "
                             "components unless fusion is 'average'")
        self.fusion, self.w, self.k, self.lam = fusion, float(w), int(k), float(lam)
        self.detach_aux = bool(detach_aux)
        self.stream1_block1 = trunk.stream1_block1
        self.stream2_block1 = trunk.stream2_block1
        # classifier = [AdaptiveAvgPool2d, Flatten, Linear]; only `average`
        # decides through it, the other rules drop it.
        linear = trunk.classifier[-1]
        n_feat, n_cls = linear.in_features, linear.out_features
        self.classifier = trunk.classifier if fusion == "average" else None
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head_m = nn.Linear(n_feat, n_cls)
        self.head_p = nn.Linear(n_feat, n_cls)

    def features(self, x_1, x_2):
        return (self.pool(self.stream1_block1(x_1)),
                self.pool(self.stream2_block1(x_2)))

    def forward_heads(self, x_1, x_2) -> dict[str, torch.Tensor]:
        """{'joint', 'magno', 'parvo'} logits in fp32 (the rule runs in fp32
        so top-k ties and the hinge are not decided in bf16).

        For `average` the dict also carries the tied head applied to each
        stream alone ('tied_magno', 'tied_parvo'); since the head is linear,
        w * tied_magno + (1 - w) * tied_parvo is that network's w-sweep, and
        the decision is the sweep at the network's own w (the BA's
        TwoStreamNetwork with stream1_weight = w).
        """
        h_m, h_p = self.features(x_1, x_2)
        a_m, a_p = (h_m.detach(), h_p.detach()) if self.detach_aux else (h_m, h_p)
        out = {"magno": self.head_m(a_m).float(),
               "parvo": self.head_p(a_p).float()}
        if self.fusion == "average":
            lin = self.classifier[-1]
            out["tied_magno"] = lin(h_m).float()
            out["tied_parvo"] = lin(h_p).float()
            out["joint"] = (self.w * out["tied_magno"]
                            + (1.0 - self.w) * out["tied_parvo"])
        else:
            out["joint"] = fuse(self.fusion, out["magno"], out["parvo"],
                                w=self.w, k=self.k, lam=self.lam)
        return out

    def forward(self, x_1, x_2):
        return self.forward_heads(x_1, x_2)["joint"]
