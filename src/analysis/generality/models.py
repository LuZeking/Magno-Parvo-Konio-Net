"""One interface to a frozen network: penultimate features y and the linear head.

Every network here ends in  logits = W y + b  for some feature vector y (the
input of its last linear layer). The generality test manipulates y with a
per-unit gain or a lossy mixture and reads the rate through the network's own
W, b -- so the loader has to hand back exactly that pair and exactly that y,
and it checks the identity  head(y) == network(x)  on the first batch it sees.

Specs
-----
  run:<run_name>[:<checkpoint>]     one of our runs under MPK_RUNS. Single-stream
                                    conditions (dkl_bcs_single, dkl_bcs_single34)
                                    and rgb_tsn (both trunks receive the same
                                    image and are bit-identical, so it is one
                                    trunk) give a 512-d y.
  concat:<run_name>[:<checkpoint>]  a two-stream run with y = [y1, y2] (1,024-d)
                                    and head [W/2, W/2], b -- the validation
                                    case, where the factorisation should recover
                                    the streams.
  tv:<arch>                         torchvision ImageNet weights (IMAGENET1K_V1),
                                    Geirhos preprocessing, 16-class scorer.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MPK_RUNS = os.environ.get("MPK_RUNS", "/scratch/byrong/VST_data/mpk_runs")
os.environ.setdefault("TORCH_HOME", "/scratch/byrong/caches/torch")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FrozenNet:
    """Penultimate features + linear head of one frozen network."""

    def __init__(self, spec: str, device: str = "cpu"):
        self.spec = spec
        self.device = device
        self.kind, rest = spec.split(":", 1)
        self._checked = False
        if self.kind in ("run", "concat"):
            parts = rest.split(":")
            self.run = parts[0]
            self.ckpt_name = parts[1] if len(parts) > 1 else "checkpoint_last.pt"
            self._load_run()
        elif self.kind == "tv":
            self.arch = rest
            self._load_tv()
        else:
            raise ValueError(spec)
        self.W = self.W.detach().float().cpu().numpy()
        self.b = self.b.detach().float().cpu().numpy()

    # ------------------------------------------------------------------ ours
    def _load_run(self):
        from src.analysis.encoding import dkl_frontend, features
        from src.models.pytorch_models import TwoStreamNetwork
        from src.training.standalone_trainer import patch_stream_input_channels

        run_dir = os.path.join(MPK_RUNS, self.run)
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        self.cfg, self.ccfg = cfg, cfg["condition_cfg"]
        ckpt = os.path.join(run_dir, self.ckpt_name)
        features.assert_converged(ckpt)
        self.checkpoint = ckpt
        self.scorer = "miniecoset112"
        self.tag = f"{self.kind}_{self.run}"
        self.single = bool(self.ccfg.get("single_stream", False))
        self.rgb_mode = self.ccfg.get("input_mode") == "rgb"
        arch = cfg.get("like_resnet_resolved") or self.ccfg.get("like_resnet") or cfg.get("like_resnet") or "resnet18"

        if self.single:
            if self.kind == "concat":
                raise ValueError("concat: needs a two-stream run")
            torch.manual_seed(cfg.get("seed", 0))
            model = TwoStreamNetwork(
                num_classes=cfg.get("num_classes", 112), like_resnet=arch,
                seperate_from_layer=0, baseline=True, combined_color=True,
                dvd_strategy="dkl", stream_concat=False, stream1_weight=0.5,
                legacy_asymmetric_mixing=False)
            patch_stream_input_channels(model, 2, 0)
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = blob.get("state_dict", blob.get("model", blob)) if isinstance(blob, dict) else blob
            if any(k.startswith("network.") for k in state):
                state = {k.removeprefix("network."): v for k, v in state.items() if k.startswith("network.")}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"strict load failed: {len(missing)} missing, {len(unexpected)} unexpected")
            model.eval().to(self.device)
        else:
            model = features.build_model(
                checkpoint=ckpt, like_resnet=arch, stream_concat=False,
                stream1_weight=0.5, seed=cfg.get("seed", 0), device=self.device,
                in_channels=tuple(cfg["in_channels"]))
            if self.kind == "run" and not self.rgb_mode:
                raise ValueError(f"{self.run}: a two-stream run needs the concat: spec")
        self.model = model
        lin = [m for m in model.classifier.modules() if isinstance(m, nn.Linear)]
        if len(lin) != 1:
            raise RuntimeError(f"expected one Linear in the classifier, found {len(lin)}")
        W, b = lin[0].weight, lin[0].bias
        if self.kind == "concat":
            self.W, self.b = torch.cat([0.5 * W, 0.5 * W], dim=1), b
            self.stream_of_unit = np.r_[np.zeros(W.shape[1], int), np.ones(W.shape[1], int)]
        else:
            self.W, self.b = W, b
            self.stream_of_unit = None
        self.ch_min, self.ch_max = dkl_frontend.compute_dkl_channel_minmax()
        self.age = cfg.get("age_months", 48)

    def _run_inputs_from_rgb(self, rgb_uint8: np.ndarray):
        from src.analysis.encoding import dkl_frontend
        x_m, x_p = dkl_frontend.magno_parvo_inputs(
            rgb_uint8, self.ch_min, self.ch_max,
            lum_blur=bool(self.ccfg.get("blur", True)), lum_cs=bool(self.ccfg.get("cs", True)),
            age_months=self.age, device=self.device,
            transform_target=self.ccfg.get("transform_target", "magno"),
            parvo_highsf=bool(self.ccfg.get("parvo_highsf", False)),
            rgb_mode=self.rgb_mode)
        if self.single:
            # the trainer's StreamInputBuilder: one trunk, both channels stacked
            return torch.cat([x_m, x_p], dim=1), None
        return x_m, x_p

    def _run_inputs_from_h5(self, x_dkl_float: torch.Tensor):
        """MiniEcoset tensors (N,3,H,W) in [0,1] -> stream inputs, the trainer's own builder."""
        from src.training.standalone_trainer import StreamInputBuilder
        builder = StreamInputBuilder(self.ccfg, age_months=self.age)
        x1, x2 = builder(x_dkl_float.to(self.device))
        return x1, x2

    @torch.no_grad()
    def _run_forward(self, x1: torch.Tensor, x2: torch.Tensor | None, batch: int = 64) -> np.ndarray:
        m = self.model
        outs = []
        for s in range(0, x1.shape[0], batch):
            a = x1[s:s + batch].to(self.device)
            if self.single:
                f1 = m.stream1_block1(a)
                y = F.adaptive_avg_pool2d(f1, 1).flatten(1)
                if not self._checked:
                    self._check(y, m.classifier(f1))
            elif self.kind == "concat":
                bb = x2[s:s + batch].to(self.device)
                f1, f2 = m.stream1_block1(a), m.stream2_block1(bb)
                y1, y2 = F.adaptive_avg_pool2d(f1, 1).flatten(1), F.adaptive_avg_pool2d(f2, 1).flatten(1)
                y = torch.cat([y1, y2], 1)
                if not self._checked:
                    self._check(y, m.classifier(0.5 * f1 + 0.5 * f2))
            else:  # rgb_tsn: both trunks identical, take trunk 1
                f1 = m.stream1_block1(a)
                y = F.adaptive_avg_pool2d(f1, 1).flatten(1)
                if not self._checked:
                    f2 = m.stream2_block1(x2[s:s + batch].to(self.device))
                    dev = (f1 - f2).abs().max().item()
                    if dev > 1e-3:
                        raise RuntimeError(f"rgb_tsn trunks differ (max |y1-y2| = {dev:.2e}); not a single trunk")
                    self._check(y, m.classifier(0.5 * f1 + 0.5 * f2))
            outs.append(y.float().cpu().numpy())
        return np.concatenate(outs)

    # ----------------------------------------------------------- torchvision
    def _load_tv(self):
        import torchvision
        self.model = torchvision.models.get_model(self.arch, weights="IMAGENET1K_V1").eval().to(self.device)
        self.scorer = "imagenet16"
        self.tag = f"tv_{self.arch}"
        self.checkpoint = f"torchvision {self.arch} IMAGENET1K_V1"
        self.stream_of_unit = None
        m = self.model
        if self.arch.startswith("resnet"):
            self.head = m.fc
        elif self.arch.startswith("vgg"):
            self.head = m.classifier[6]
        elif self.arch.startswith("vit"):
            self.head = m.heads.head
        elif self.arch.startswith("convnext"):
            self.head = m.classifier[2]
        else:
            raise ValueError(self.arch)
        self.W, self.b = self.head.weight, self.head.bias

    def _tv_prep(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """Geirhos / model-vs-human default: Resize(256), CenterCrop(224), Normalize."""
        t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(0, 3, 1, 2).float() / 255.0
        t = F.interpolate(t, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
        t = t[:, :, 16:240, 16:240]
        return (t - IMAGENET_MEAN) / IMAGENET_STD

    @torch.no_grad()
    def _tv_features(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        if self.arch.startswith("resnet"):
            x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
            x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
            return torch.flatten(m.avgpool(x), 1)
        if self.arch.startswith("vgg"):
            x = torch.flatten(m.avgpool(m.features(x)), 1)
            return m.classifier[:6](x)
        if self.arch.startswith("vit"):
            x = m._process_input(x)
            n = x.shape[0]
            x = torch.cat([m.class_token.expand(n, -1, -1), x], dim=1)
            x = m.encoder(x)
            return x[:, 0]
        if self.arch.startswith("convnext"):
            x = m.avgpool(m.features(x))
            return m.classifier[1](m.classifier[0](x))
        raise ValueError(self.arch)

    @torch.no_grad()
    def _tv_forward(self, rgb_uint8: np.ndarray, batch: int = 32) -> np.ndarray:
        outs = []
        for s in range(0, len(rgb_uint8), batch):
            x = self._tv_prep(rgb_uint8[s:s + batch]).to(self.device)
            y = self._tv_features(x)
            if not self._checked:
                self._check(y, self.model(x))
            outs.append(y.float().cpu().numpy())
        return np.concatenate(outs)

    # ------------------------------------------------------------------ API
    def _check(self, y: torch.Tensor, ref_logits: torch.Tensor):
        """head(y) must reproduce the network's own logits."""
        W = self.W if torch.is_tensor(self.W) else torch.from_numpy(self.W)
        b = self.b if torch.is_tensor(self.b) else torch.from_numpy(self.b)
        mine = y.float() @ W.float().t().to(y.device) + b.float().to(y.device)
        dev = (mine - ref_logits.float()).abs().max().item()
        scale = ref_logits.float().abs().mean().item()
        if dev > 1e-3 * max(scale, 1.0):
            raise RuntimeError(f"head(y) != network(x): max dev {dev:.3e} on logit scale {scale:.3f}")
        print(f"[models] {self.tag}: head identity holds (max dev {dev:.1e}); D = {y.shape[1]}", flush=True)
        self._checked = True

    def features_from_rgb(self, rgb_uint8: np.ndarray, chunk: int = 256) -> np.ndarray:
        """uint8 RGB (N,H,W,3) -> penultimate features (N, D) float32."""
        outs = []
        for s in range(0, len(rgb_uint8), chunk):
            part = rgb_uint8[s:s + chunk]
            if self.kind == "tv":
                outs.append(self._tv_forward(part))
            else:
                x1, x2 = self._run_inputs_from_rgb(part)
                outs.append(self._run_forward(x1, x2))
        return np.concatenate(outs)

    def features_from_h5(self, x_dkl_uint8: np.ndarray) -> np.ndarray:
        """MiniEcoset uint8 (N,H,W,3) DKL images -> features, through the trainer's own input builder."""
        if self.kind == "tv":
            raise ValueError("torchvision networks have no MiniEcoset head")
        x = torch.from_numpy(np.ascontiguousarray(x_dkl_uint8)).permute(0, 3, 1, 2).float() / 255.0
        x1, x2 = self._run_inputs_from_h5(x)
        return self._run_forward(x1, x2)

    @property
    def n_units(self) -> int:
        return int(self.W.shape[1])
