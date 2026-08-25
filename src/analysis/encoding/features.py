"""Stream-wise feature extraction from the TwoStreamNetwork for encoding.

Extracts activations at the output of the four ResNet stages (layer1..4) of
each stream, spatially pools them to a fixed grid, concatenates stages, and
reduces each stream to a matched dimensionality with PCA (fit on training
images only). Matched dimensionality removes the "one stream simply has more
parameters" confound in variance partitioning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.models.pytorch_models import TwoStreamNetwork

# Children of torchvision resnet* inside stream{1,2}_block1 (sequential):
# 0 conv1, 1 bn1, 2 relu, 3 maxpool, 4 layer1, 5 layer2, 6 layer3, 7 layer4
STAGE_INDICES = (4, 5, 6, 7)
POOL_GRID = 4  # adaptive avg pool each stage to POOL_GRID x POOL_GRID


def build_model(
    checkpoint: str | None = None,
    like_resnet: str = "resnet18",
    stream_concat: bool = False,
    stream1_weight: float = 0.5,
    legacy_asymmetric_mixing: bool = True,
    num_classes: int = 112,
    seed: int = 0,
    device: str = "cpu",
) -> TwoStreamNetwork:
    """Build a DKL-input TSN; load a checkpoint or return an untrained net.

    ``legacy_asymmetric_mixing`` defaults to True because all existing
    checkpoints were trained with the pre-fix forward pass; untrained
    control nets keep the flag of whatever they are compared against.
    """
    torch.manual_seed(seed)
    model = TwoStreamNetwork(
        num_classes=num_classes,
        like_resnet=like_resnet,
        seperate_from_layer=0,
        baseline=False,
        dvd_strategy="dkl",
        stream_concat=stream_concat,
        stream1_weight=stream1_weight,
        legacy_asymmetric_mixing=legacy_asymmetric_mixing,
    )
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in state:  # lightning checkpoint
            state = {
                k.removeprefix("network."): v
                for k, v in state["state_dict"].items()
                if k.startswith("network.")
            }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[features] load_state_dict: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys"
            )
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_stream_features(
    model: TwoStreamNetwork,
    x_magno: torch.Tensor,
    x_parvo: torch.Tensor,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Stage-pooled features per stream for a stack of images.

    Returns {"magno": (N, D), "parvo": (N, D)} float32 arrays, where D =
    sum over stages of channels * POOL_GRID^2 (15,360 for ResNet18).
    Handles both the separated architecture (block1 holds all stages) and
    the interconnected one (stages split across block1/block2 with mixing).
    """
    feats: dict[str, list[np.ndarray]] = {"magno": [], "parvo": []}

    def run_stages(seq: torch.nn.Sequential, x: torch.Tensor):
        stage_outs = []
        for i, layer in enumerate(seq):
            x = layer(x)
            if i in STAGE_INDICES:
                stage_outs.append(
                    F.adaptive_avg_pool2d(x, POOL_GRID).flatten(1)
                )
        return x, stage_outs

    for start in range(0, x_magno.shape[0], batch_size):
        xm = x_magno[start : start + batch_size].to(device)
        xp = x_parvo[start : start + batch_size].to(device)

        if model.stream_concat:
            y1, s1 = run_stages(model.stream1_block1, xm)
            y2, s2 = run_stages(model.stream2_block1, xp)
            y1, y2 = model._mix_streams(y1, y2)
            n1 = len(model.stream1_block1)
            for i, layer in enumerate(model.stream1_block2):
                y1 = layer(y1)
                if (i + n1) in STAGE_INDICES:
                    s1.append(F.adaptive_avg_pool2d(y1, POOL_GRID).flatten(1))
            for i, layer in enumerate(model.stream2_block2):
                y2 = layer(y2)
                if (i + n1) in STAGE_INDICES:
                    s2.append(F.adaptive_avg_pool2d(y2, POOL_GRID).flatten(1))
        else:
            _, s1 = run_stages(model.stream1_block1, xm)
            _, s2 = run_stages(model.stream2_block1, xp)

        feats["magno"].append(
            torch.cat(s1, dim=1).float().cpu().numpy()
        )
        feats["parvo"].append(
            torch.cat(s2, dim=1).float().cpu().numpy()
        )

    return {k: np.concatenate(v, axis=0) for k, v in feats.items()}


def fit_pca(
    train_features: np.ndarray, n_components: int = 1000, seed: int = 0
):
    """PCA fit on TRAINING images only (leak-free)."""
    from sklearn.decomposition import PCA

    n_components = min(
        n_components, train_features.shape[0] - 1, train_features.shape[1]
    )
    pca = PCA(
        n_components=n_components, svd_solver="randomized", random_state=seed
    )
    pca.fit(train_features)
    return pca
