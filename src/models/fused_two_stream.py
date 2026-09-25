"""Magno/parvo trunks that merge after layer2 into one shared layer3-layer4.

For the TDANN spatial-loss pilot (docs/PLAN_TSN_SPATIAL_PILOT_2026-09-25.md). In
`TwoStreamNetwork` every layer4 unit belongs to one stream by construction, so a
shared cortical sheet could only segregate magno from parvo by where units are put.
Here the streams stay separate through layer2 (retina/V1 in the TDANN area mapping)
and every layer3/layer4 unit (V2/V4/VTC) takes both, so a unit's reliance on magno
versus parvo input is learned, and the spatial loss can organise it.

Trunks are torchvision ResNet-18 children 0-5 with a 1-channel stem, as built by
`TwoStreamNetwork` + `patch_stream_input_channels` (default-initialised stem). The
shared layer3 takes the 256-channel concatenation of the two layer2 outputs; layer3
and layer4 otherwise match ResNet-18 and get torchvision's initialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

# TDANN's 8 ResNet-18 depths; trunk depths exist once per stream.
TRUNK_DEPTHS = ("layer1.0", "layer1.1", "layer2.0", "layer2.1")
SHARED_DEPTHS = ("layer3.0", "layer3.1", "layer4.0", "layer4.1")


def _trunk(in_ch: int) -> nn.Sequential:
    r = resnet18(weights=None)
    stem = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return nn.Sequential(stem, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)


class FusedTwoStreamNetwork(nn.Module):
    def __init__(self, num_classes: int = 112, in_channels: tuple[int, int] = (1, 1)):
        super().__init__()
        self.magno = _trunk(in_channels[0])
        self.parvo = _trunk(in_channels[1])
        shared = resnet18(weights=None)
        shared.inplanes = 256  # concat of two 128-channel layer2 outputs
        self.layer3 = shared._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = shared._make_layer(BasicBlock, 512, 2, stride=2)
        for m in (*self.layer3.modules(), *self.layer4.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def trunks(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.magno(x1), self.parvo(x2)], dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.layer4(self.layer3(self.trunks(x1, x2))))

    def sheet_modules(self) -> dict[str, nn.Module]:
        """Block whose output carries each sheet, keyed 'magno.layer1.0', ...,
        'layer4.1' (spatial loss is applied to these outputs, as in TDANN)."""
        mods = {}
        for stream in ("magno", "parvo"):
            trunk = getattr(self, stream)
            for d in TRUNK_DEPTHS:
                layer, block = d.split(".")
                mods[f"{stream}.{d}"] = trunk[4 if layer == "layer1" else 5][int(block)]
        for d in SHARED_DEPTHS:
            layer, block = d.split(".")
            mods[d] = getattr(self, layer)[int(block)]
        return mods


def depth_of(sheet: str) -> str:
    """'magno.layer2.1' -> 'layer2.1'; shared sheets are their own depth."""
    return sheet.split(".", 1)[1] if sheet.startswith(("magno.", "parvo.")) else sheet
