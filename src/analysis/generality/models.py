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
                                    the streams. A Stage-2 run name
                                    (fs2_dvd300_avg_w050_ls_seed0) resolves under
                                    dkl_frontend.FS2_RUNS, default checkpoint
                                    checkpoint_final.pt (the EMA network).
  tv:<arch>[:<weights>]             torchvision ImageNet weights (IMAGENET1K_V1 unless
                                    named), Geirhos preprocessing (Resize 256, CenterCrop
                                    224; the weights' own sizes for networks not trained
                                    at 224), 16-class scorer. y is the input of the last
                                    nn.Linear, captured by a forward hook on the full
                                    forward pass, so every torchvision classifier loads.
  hub:resnet50_sin | resnet50_sin_in | resnet50_sin_in_ft
                                    Geirhos et al. (2019) ResNet-50 trained on
                                    Stylized-ImageNet, on SIN + IN, and SIN + IN then
                                    fine-tuned on IN (checkpoints under
                                    $TORCH_HOME/hub/checkpoints, see HUB_CKPTS).
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

# Geirhos et al. (2019) checkpoints (bitbucket robert_geirhos/texture-vs-shape-pretrained-models,
# fetched 2026-09-15): file name, label. DataParallel state dicts ("module." prefix), 1000 classes.
HUB_CKPTS = {
    "resnet50_sin": ("geirhos_resnet50_SIN-c8e5653e.pth.tar", "trained on Stylized-ImageNet"),
    "resnet50_sin_in": ("geirhos_resnet50_SIN_IN-2a0d100e.pth.tar", "trained on SIN + IN"),
    "resnet50_sin_in_ft": ("geirhos_resnet50_SIN_IN_IN-ca06340c.pth.tar", "trained on SIN + IN, fine-tuned on IN"),
}


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
            self.ckpt_name = parts[1] if len(parts) > 1 else None
            self._load_run()
        elif self.kind == "tv":
            parts = rest.split(":")
            self.arch = parts[0]
            self.weights_name = parts[1] if len(parts) > 1 else "IMAGENET1K_V1"
            self._load_tv()
        elif self.kind == "hub":
            self.hub_name = rest
            self._load_hub()
        else:
            raise ValueError(spec)
        self.W = self.W.detach().float().cpu().numpy()
        self.b = self.b.detach().float().cpu().numpy()

    # ------------------------------------------------------------------ ours
    def _load_run(self):
        from src.analysis.encoding import dkl_frontend, features
        from src.models.pytorch_models import TwoStreamNetwork
        from src.training.standalone_trainer import patch_stream_input_channels

        fs2 = bool(dkl_frontend.FS2_RUN_NAME.fullmatch(self.run))
        run_dir = (dkl_frontend.resolve_run_dir(self.run) if fs2
                   else os.path.join(MPK_RUNS, self.run))
        if self.ckpt_name is None:
            self.ckpt_name = "checkpoint_final.pt" if fs2 else "checkpoint_last.pt"
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        # The g017 runs name a node-local copy of the training h5.
        cfg["dataset"] = dkl_frontend.run_dataset(cfg["dataset"])
        self.cfg, self.ccfg = cfg, cfg["condition_cfg"]
        # The run's own front end: a DVD-diet run keeps blur/cs = true beside
        # dvd_end_months, so reading blur/cs alone feeds it the thesis diet.
        self.frontend = dkl_frontend.checkpoint_frontend(cfg)
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
            if float(self.ccfg.get("w", 0.5)) != 0.5:
                raise ValueError(f"{self.run}: trained at w = {self.ccfg['w']}; "
                                 f"the concat head is [W/2, W/2]")
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
            rgb_uint8, self.ch_min, self.ch_max, device=self.device,
            **self.frontend)
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
    # Every torchvision classifier ends in one nn.Linear. y is that layer's input,
    # captured with a forward hook during the network's own forward pass, so any
    # architecture (Inception's transform_input, MaxViT's Tanh-projected pooled
    # token, DenseNet's ReLU-then-pool) hands back exactly the vector its head
    # reads; _check verifies head(y) == logits on the first batch.
    def _load_tv(self):
        import torchvision
        self.model = torchvision.models.get_model(self.arch, weights=self.weights_name).eval().to(self.device)
        wenum = torchvision.models.get_model_weights(self.arch)[self.weights_name]
        self._set_tv_prep(wenum.transforms())
        self.scorer = "imagenet16"
        self.tag = f"tv_{self.arch}" + ("" if self.weights_name == "IMAGENET1K_V1" else f"_{self.weights_name}")
        self.checkpoint = f"torchvision {self.arch} {self.weights_name}"
        self.stream_of_unit = None
        self._attach_head()

    def _load_hub(self):
        """Geirhos et al. (2019) ResNet-50s (torchvision architecture, DataParallel state dict)."""
        import torchvision
        if self.hub_name not in HUB_CKPTS:
            raise ValueError(f"hub:{self.hub_name}; known: {sorted(HUB_CKPTS)}")
        fname, label = HUB_CKPTS[self.hub_name]
        path = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", fname)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in blob["state_dict"].items()}
        model = torchvision.models.resnet50(weights=None)
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(self.device)
        self.arch, self.weights_name = "resnet50", self.hub_name
        self.resize_size, self.crop_size = 256, 224
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        self.scorer = "imagenet16"
        self.tag = f"hub_{self.hub_name}"
        self.checkpoint = f"Geirhos et al. 2019 ResNet-50 {label} ({fname}, epoch {blob.get('epoch')})"
        self.stream_of_unit = None
        self._attach_head()

    def _set_tv_prep(self, tf):
        crop = int(tf.crop_size[0] if isinstance(tf.crop_size, (list, tuple)) else tf.crop_size)
        resize = int(tf.resize_size[0] if isinstance(tf.resize_size, (list, tuple)) else tf.resize_size)
        # Geirhos / model-vs-human default for every network trained at 224 (as in Fig. 6);
        # the weights' own sizes otherwise (Inception-v3 342/299, EfficientNet-B3 320/300,
        # EfficientNet-V2-S 384/384).
        self.resize_size, self.crop_size = (256, 224) if crop == 224 else (resize, crop)
        self.mean = torch.tensor([float(v) for v in tf.mean]).view(1, 3, 1, 1)
        self.std = torch.tensor([float(v) for v in tf.std]).view(1, 3, 1, 1)

    def _attach_head(self):
        lins = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        if not lins:
            raise RuntimeError(f"{self.tag}: no nn.Linear head")
        self.head = lins[-1]
        self.W = self.head.weight
        # MaxViT's head has bias=False; a zero bias keeps logits = W y + b exact
        self.b = self.head.bias if self.head.bias is not None else torch.zeros(self.W.shape[0])
        self._captured = None
        self.head.register_forward_hook(self._capture_input)

    def _capture_input(self, module, inputs, output):
        self._captured = inputs[0].detach()

    def _tv_prep(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """Square resize, centre crop, normalise (256 -> 224 unless the weights were not trained at 224)."""
        t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(0, 3, 1, 2).float() / 255.0
        r, c = self.resize_size, self.crop_size
        t = F.interpolate(t, size=(r, r), mode="bilinear", antialias=True, align_corners=False)
        o = (r - c) // 2
        t = t[:, :, o:o + c, o:o + c]
        return (t - self.mean) / self.std

    @torch.no_grad()
    def _tv_forward(self, rgb_uint8: np.ndarray, batch: int = 32) -> np.ndarray:
        outs = []
        for s in range(0, len(rgb_uint8), batch):
            x = self._tv_prep(rgb_uint8[s:s + batch]).to(self.device)
            self._captured = None
            logits = self.model(x)
            y = self._captured
            if y is None:
                raise RuntimeError(f"{self.tag}: the head hook captured nothing")
            if y.ndim > 2:
                y = y.flatten(1)
            if not self._checked:
                self._check(y, logits)
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
            if self.kind in ("tv", "hub"):
                outs.append(self._tv_forward(part))
            else:
                x1, x2 = self._run_inputs_from_rgb(part)
                outs.append(self._run_forward(x1, x2))
        return np.concatenate(outs)

    def features_from_h5(self, x_dkl_uint8: np.ndarray) -> np.ndarray:
        """MiniEcoset uint8 (N,H,W,3) DKL images -> features, through the trainer's own input builder."""
        if self.kind in ("tv", "hub"):
            raise ValueError("torchvision networks have no MiniEcoset head")
        x = torch.from_numpy(np.ascontiguousarray(x_dkl_uint8)).permute(0, 3, 1, 2).float() / 255.0
        x1, x2 = self._run_inputs_from_h5(x)
        return self._run_forward(x1, x2)

    @property
    def n_units(self) -> int:
        return int(self.W.shape[1])
