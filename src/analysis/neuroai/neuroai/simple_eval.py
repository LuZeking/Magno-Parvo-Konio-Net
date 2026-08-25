#!/usr/bin/env python3
"""
simple_eval.py
==============

A minimal, flat, easy-to-use evaluator for shape bias + accuracy metrics.

This is a thin wrapper on top of the existing `neuroai` toolkit.  It keeps
the original functionality intact (model loading, dataset loading,
plotting) but exposes it through a single entry point so that it can be
called from a training loop with just a model name and a checkpoint path.

Typical usage
-------------

>>> from neuroai.simple_eval import evaluate
>>> results = evaluate(
...     model_name="resnet50_DVD_B",
...     checkpoint_path="/path/to/checkpoint.pth",
...     datasets=["cue-conflict", "edge", "silhouette"],
...     dataset_trained_on="imagenet",
... )
>>> print(results["cue-conflict"]["mean_shape_bias"])
>>> print(results["cue-conflict"]["median_shape_bias"])
>>> print(results["edge"]["accuracy"])

Return value
------------
A dict keyed by dataset name.  For each dataset:

    {
        "dataset"           : str,
        "accuracy"          : float,              # overall top-1 accuracy
        "accuracy_per_cond" : dict[str -> float], # per-condition accuracies
        # extra keys for "cue-conflict":
        "mean_shape_bias"   : float,
        "median_shape_bias" : float,
        "shape_bias_per_cat": dict[str -> float],
        "shape_acc_per_cat" : dict[str -> float],
        "texture_acc_per_cat": dict[str -> float],
        "classes"           : list[str],
    }
"""

from __future__ import annotations

import contextlib
import fnmatch
import logging
import os
from os.path import join as pjoin
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.helpers import (
    apply_gaussian_blur,
    apply_scale_free_contrast,
    convert_rgb_to_dkl,
    normalize_images_channels,
)

from . import constants as c
from .evaluation.evaluate import ResultPrinter, print_performance_to_csv
from .plotting import analyses as a
from .repr_shape_bias import (
    compute_repr_shape_bias as _compute_repr_shape_bias,
)
from .utils import (
    load_dataset,
    load_model,
)

logger = logging.getLogger(__name__)


SIMPLE_EVAL_PLOT_CONTEXTS = ("model", "humans", "all")
_SIMPLE_EVAL_PLOT_CONTEXT_ALIASES = {
    "self": "model",
    "model-only": "model",
    "with-humans": "humans",
    "references": "all",
}


# ---------------------------------------------------------------------------
# In-place model registration
# ---------------------------------------------------------------------------


def register_pytorch_model(
    name: str,
    checkpoint_path: Optional[str] = None,
    out_class_dim: Optional[int] = None,
    overwrite: bool = False,
):
    """Register a new PyTorch model on-the-fly (no need to edit model_zoo.py).

    The `name` must contain a recognizable architecture substring so that
    `texture_shape_models.load_model` can build the backbone:

        resnet50, resnet18, resnet101, resnext50_32x4d, vgg16, alexnet,
        mobilenet_v3_small, shufflenet_v2_x1_0, vit_b_16, swin_b,
        deit_base_patch16_224

    The classifier head dimension is inferred from the name when possible
    ("imagenet" -> 1000, "miniecoset" -> 112, "ecoset" -> 565), or from
    `out_class_dim` / `dataset_trained_on` at load time.

    Example
    -------
    >>> from neuroai import register_pytorch_model, evaluate
    >>> register_pytorch_model(
    ...     name="resnet50_my_new_model",
    ...     checkpoint_path="/path/to/ckpt.pth",
    ... )
    >>> evaluate(model_name="resnet50_my_new_model",
    ...          datasets=["cue-conflict"])
    """
    from .models.pytorch import model_zoo as pytorch_model_zoo
    from .models.pytorch.shapenet import texture_shape_models as tsm
    from .models.registry import _model_registry, register_model
    from .models.wrappers.pytorch import PytorchModel

    if name in _model_registry["pytorch"] and not overwrite:
        logger.info("Model '%s' is already registered — skipping.", name)
        return
    if name in _model_registry["pytorch"] and overwrite:
        _model_registry["pytorch"].remove(name)

    def _loader(model_name_str, *args, **kwargs):
        # Checkpoint resolution: a checkpoint explicitly passed to
        # load_model(...) at call time wins over the one baked in here.
        if checkpoint_path and not kwargs.get("load_checkpoint_path"):
            kwargs["load_checkpoint_path"] = checkpoint_path
        if out_class_dim is not None:
            # Expose the preferred head dim via a dataset alias tsm understands.
            kwargs.setdefault(
                "dataset_trained_on",
                {1000: "imagenet", 565: "ecoset", 112: "miniecoset_zejin"}.get(
                    out_class_dim
                ),
            )
        model = tsm.load_model(model_name_str, *args, **kwargs)
        return PytorchModel(model, model_name_str, *args)

    _loader.__name__ = name
    register_model("pytorch")(_loader)
    # Expose on the model_zoo module so `eval(f"pytorch_model_zoo.{name}")`
    # in `utils.load_model` can find it.
    setattr(pytorch_model_zoo, name, _loader)
    logger.info(
        "Registered pytorch model '%s' (ckpt=%s).", name, checkpoint_path
    )


# ---------------------------------------------------------------------------
# Custom output paths: temporarily redirect RAW_DATA_DIR / FIGURE_DIR /
# PERFORMANCES_DIR. This lets us reuse the existing ResultPrinter and the
# plotting pipeline without any code changes.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _override_paths(
    output_dir: Optional[str],
):
    """Temporarily re-point the package's output directories.

    If `output_dir` is given, all writes (raw per-sample CSVs, performance
    CSVs, and figures) are redirected under it:

        <output_dir>/raw-data/<dataset>/...
        <output_dir>/performances/<model>.csv
        <output_dir>/plots/<figure_directory_name>/...

    When the context exits the original module-level paths are restored.
    """
    if not output_dir:
        yield None
        return

    os.makedirs(output_dir, exist_ok=True)
    new_raw = pjoin(output_dir, "raw-data")
    new_perf = pjoin(output_dir, "performances")
    new_fig = pjoin(output_dir, "plots")
    for d in (new_raw, new_perf, new_fig):
        os.makedirs(d, exist_ok=True)

    old_raw, old_perf, old_fig = (
        c.RAW_DATA_DIR,
        c.PERFORMANCES_DIR,
        c.FIGURE_DIR,
    )
    c.RAW_DATA_DIR = new_raw
    c.PERFORMANCES_DIR = new_perf
    c.FIGURE_DIR = new_fig
    try:
        yield new_fig
    finally:
        c.RAW_DATA_DIR = old_raw
        c.PERFORMANCES_DIR = old_perf
        c.FIGURE_DIR = old_fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device() -> torch.device:
    """Pick the best available torch device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _normalize_images(images: torch.Tensor, norm_type: str) -> torch.Tensor:
    """Normalize a batch of images according to `norm_type`.

    - "0-1"     : rescale to [0, 1] using min/max (used by our custom models)
    - "mean-std": assume the dataloader already applied ImageNet normalization
    """
    if norm_type == "0-1":
        images = (images - images.min()) / (
            images.max() - images.min() + 1e-12
        )
    elif norm_type == "mean-std":
        # The loader is assumed to have normalized with ImageNet stats.
        pass
    else:
        raise ValueError(f"Unsupported data_normalization_type: {norm_type}")
    return images


# ---------------------------------------------------------------------------
# Core: run a single forward pass over a dataset, return a DataFrame
# ---------------------------------------------------------------------------


def _forward_pass_to_df(
    model,
    dataset,
    model_name: str,
    norm_type: str = "0-1",
    save_csv: bool = True,
    verbose: bool = False,
    dvd_strategy: str = "adult",
    age_m: int = None,
    lum: bool = None,
    rg: bool = None,
    by: bool = None,
    lum_blur: bool = None,
    lum_cs: bool = None,
    rg_blur: bool = None,
    rg_cs: bool = None,
    by_blur: bool = None,
    by_cs: bool = None,
    combined_color: bool = None,
    epoch: int = -1,
):
    """Run the model on `dataset` and return ``(DataFrame, path_list)``.

    DataFrame columns:
        subj, session, trial, rt, object_response, category,
        condition, imagename, image_id
    The path_list preserves the *absolute* image paths yielded by the
    dataloader, in the same order as the DataFrame rows.  This lets the
    representational-shape-bias analysis look up the original shape /
    texture reference images by filename.

    Optionally mirrors the predictions to the raw-data CSVs that the
    existing plotting code expects (so the visualization pipeline still
    works untouched).
    """
    device = _device()
    rows: List[Dict] = []
    all_paths: List[str] = []
    if epoch != -1:
        model_name += f"_epoch_{epoch}"
    writer = (
        ResultPrinter(model_name=model_name, dataset=dataset)
        if save_csv
        else None
    )

    with torch.no_grad():
        for images, targets, paths in tqdm(
            dataset.loader, desc=f"[{dataset.name}]"
        ):
            images = images.to(device)
            images = _normalize_images(images, norm_type)

            if dvd_strategy in ["dkl", "dkl_dvd"]:
                images = convert_rgb_to_dkl(images, device)
                images = normalize_images_channels(images, norm_type)

            if verbose:
                print(
                    f"[{dataset.name}] normalization={norm_type} "
                    f"range=({images.min():.3f},{images.max():.3f}) "
                    f"mean={images.mean():.3f} std={images.std():.3f}"
                )

            if dvd_strategy in ["dkl", "dkl_dvd"]:
                # get the magno and parvo channels

                # magno/lum/dorsal
                x_1 = images[:, 0, :, :]
                # parvo/rg/ventral
                x_2 = images[:, 1, :, :]
                # konio/by
                x_3 = images[:, 2, :, :]

                # unsqueeze
                x_1 = x_1.unsqueeze(1)
                x_2 = x_2.unsqueeze(1)
                x_3 = x_3.unsqueeze(1)

                if lum:

                    if lum_blur:
                        x_1 = apply_gaussian_blur(x_1)

                    if lum_cs:
                        x_1 = apply_scale_free_contrast(x_1, age_m, 224)

                    if not rg and not by:
                        logits = model.forward_batch(x_1)

                if rg:

                    if rg_blur:
                        x_2 = apply_gaussian_blur(x_2)

                    if rg_cs:
                        x_2 = apply_scale_free_contrast(x_2, age_m, 224)

                    if not lum and not by:
                        logits = model.forward_batch(x_2)

                if by:

                    if by_blur:
                        x_3 = apply_gaussian_blur(x_3)

                    if by_cs:
                        x_3 = apply_scale_free_contrast(x_3, age_m, 224)

                    if not lum and not rg:
                        logits = model.forward_batch(x_3)

                if combined_color:
                    x_combined = images[:, 1:3, :, :]
                    x_combined[:, 0, :, :] = x_2.squeeze(1)
                    x_combined[:, 1, :, :] = x_3.squeeze(1)

                    if lum and not model.model.baseline:
                        logits = model.forward_batch(x_1, x_combined)
                    else:
                        logits = model.forward_batch(x_combined)

                else:
                    if lum and rg and not by:
                        logits = model.forward_batch(x_1, x_2)
                    if lum and by and not rg:
                        logits = model.forward_batch(x_1, x_3)
                    if rg and by and not lum:
                        logits = model.forward_batch(x_2, x_3)

            else:
                logits = model.forward_batch(images)

            softmax_output = model.softmax(logits)
            predictions = dataset.decision_mapping(softmax_output)

            if isinstance(targets, torch.Tensor):
                targets_np = model.to_numpy(targets)
            else:
                targets_np = targets

            # Mirror to the standard raw-data CSV if requested (keeps
            # the existing plotting scripts working unchanged).
            if writer is not None:
                writer.print_batch_to_csv(
                    object_response=predictions,
                    batch_targets=targets_np,
                    paths=paths,
                )

            # Build an in-memory DataFrame row for each sample.
            for resp, path in zip(predictions, paths):
                session_name, img_name, condition, category = (
                    dataset.info_mapping(path)
                )
                session_num = int(session_name.split("-")[-1])
                rows.append(
                    {
                        "subj": model_name,
                        "session": session_num,
                        "trial": len(rows) + 1,
                        "rt": "NaN",
                        "object_response": resp[0],
                        "category": category,
                        "condition": condition,
                        "imagename": img_name,
                        "image_id": img_name,
                    }
                )
                all_paths.append(str(path))

    return pd.DataFrame(rows), all_paths


# ---------------------------------------------------------------------------
# Metric computation (flat, DataFrame-based)
# ---------------------------------------------------------------------------


def compute_shape_bias(df: pd.DataFrame) -> Dict:
    """Compute shape-bias metrics from a cue-conflict prediction DataFrame.

    Returns mean/median shape bias plus per-category breakdown.
    Uses the existing `ShapeBias` analysis to stay consistent with the
    original pipeline.
    """
    analysis = a.ShapeBias()
    result = analysis.analysis(df=df)

    classes = sorted(df["category"].unique())
    shape_bias_per_cat: Dict[str, float] = {}
    shape_acc_per_cat: Dict[str, float] = {}
    texture_acc_per_cat: Dict[str, float] = {}

    for cl in classes:
        sub = df[df["category"] == cl]
        if len(sub) == 0:
            continue
        r = analysis.analysis(df=sub)
        shape_bias_per_cat[cl] = r["shape-bias"]
        shape_acc_per_cat[cl] = r["fraction-correct-shape"]
        texture_acc_per_cat[cl] = r["fraction-correct-texture"]

    per_cat = np.array(list(shape_bias_per_cat.values()))

    # Overall accuracy (fraction of correct responses, either shape or texture).
    total_correct = (df["object_response"] == df["category"]).sum()
    accuracy = float(total_correct) / max(len(df), 1)

    return {
        "dataset": "cue-conflict",
        "accuracy": accuracy,
        "mean_shape_bias": float(result["shape-bias"]),
        "median_shape_bias": (
            float(np.median(per_cat)) if len(per_cat) else 0.0
        ),
        "shape_bias_per_cat": shape_bias_per_cat,
        "shape_acc_per_cat": shape_acc_per_cat,
        "texture_acc_per_cat": texture_acc_per_cat,
        "classes": list(shape_bias_per_cat.keys()),
    }


def compute_accuracy(df: pd.DataFrame, dataset_name: str) -> Dict:
    """Compute overall + per-condition accuracy for a dataset."""
    overall = (
        float((df["object_response"] == df["category"]).mean())
        if len(df)
        else 0.0
    )
    per_cond: Dict[str, float] = {}
    for cond, sub in df.groupby("condition"):
        per_cond[str(cond)] = float(
            (sub["object_response"] == sub["category"]).mean()
        )
    return {
        "dataset": dataset_name,
        "accuracy": overall,
        "accuracy_per_cond": per_cond,
    }


# ---------------------------------------------------------------------------
# Optional: visualization (re-uses the original plotting pipeline)
# ---------------------------------------------------------------------------


def _normalize_plot_context(plot_context: str) -> str:
    """Normalize aliases and validate the simple-eval plotting context."""

    normalized = _SIMPLE_EVAL_PLOT_CONTEXT_ALIASES.get(
        plot_context, plot_context
    )
    if normalized not in SIMPLE_EVAL_PLOT_CONTEXTS:
        raise ValueError(
            "plot_context must be one of "
            f"{SIMPLE_EVAL_PLOT_CONTEXTS} (got {plot_context!r})"
        )
    return normalized


def _has_subject_match(subjects: Sequence[str], name_pattern) -> bool:
    patterns = (
        [name_pattern] if isinstance(name_pattern, str) else list(name_pattern)
    )
    return any(
        fnmatch.fnmatch(subject, pattern)
        for subject in subjects
        for pattern in patterns
    )


def _same_subject_name(left: str, right: str) -> bool:
    return left.replace("-", "_") == right.replace("-", "_")


def _make_simple_decision_maker(
    model_name: str,
    plotting_name: Optional[str] = None,
    plot_context: str = "model",
):
    """Build a flexible decision_maker function for the plotting API.

    ``plot_context`` controls how much existing reference data should be
    shown alongside ``model_name``:

    - ``"model"``: only the evaluated model
    - ``"humans"``: model + humans + age cohorts/adults
    - ``"all"``: model + humans + age cohorts/adults + all other models
    """
    from .plotting.colors import rgb
    from .plotting.decision_makers import (
        DecisionMaker,
        is_human_decision_maker,
    )

    plot_context = _normalize_plot_context(plot_context)

    human_specs = [
        ("4_6_years_old", "humans(4-6 y/o)", rgb(207, 138, 153)),
        ("7_9_years_old", "humans(7-9 y/o)", rgb(207, 138, 153)),
        ("10_12_years_old", "humans(10-12 y/o)", rgb(207, 138, 153)),
        ("13_15_years_old", "humans(13-15 y/o)", rgb(207, 138, 153)),
        ("adults", "humans(adults)", rgb(207, 138, 153)),
    ]

    def _dm_fun(df):
        subjects = [str(subj) for subj in df.subj.unique()]
        decision_makers = [
            DecisionMaker(
                name_pattern=model_name,
                color=rgb(0, 0, 0),
                marker="o",
                df=df,
                plotting_name=plotting_name or model_name,
            )
        ]

        if plot_context in {"humans", "all"}:
            for cohort_name, cohort_plot_name, cohort_color in human_specs:
                if _has_subject_match(subjects, cohort_name):
                    decision_makers.append(
                        DecisionMaker(
                            name_pattern=cohort_name,
                            color=cohort_color,
                            marker="D",
                            df=df,
                            plotting_name=cohort_plot_name,
                        )
                    )

            if _has_subject_match(subjects, "subject-*"):
                decision_makers.append(
                    DecisionMaker(
                        name_pattern="subject-*",
                        color=rgb(165, 30, 55),
                        marker="D",
                        df=df,
                        plotting_name="humans",
                    )
                )

        if plot_context == "all":
            for subject in sorted(subjects):
                if is_human_decision_maker(subject):
                    continue
                if _same_subject_name(subject, model_name):
                    continue
                decision_makers.append(
                    DecisionMaker(
                        name_pattern=subject,
                        color=rgb(150, 150, 150),
                        marker="o",
                        df=df,
                        plotting_name=subject,
                    )
                )

        return decision_makers

    return _dm_fun


def visualize(
    model_name: str,
    datasets: Sequence[str] = ("cue-conflict",),
    plot_types: Sequence[str] = ("shape-bias",),
    figure_dirname: str = "simple_eval_figures",
    plotting_name: Optional[str] = None,
    plot_context: str = "model",
    output_dir: Optional[str] = None,
) -> str:
    """Render the standard plots for `model_name` using existing raw-data CSVs.

    Parameters
    ----------
    figure_dirname : str
        Subdirectory under the plots root (`<output_dir>/plots/` if
        `output_dir` is given, otherwise `results/plots/`).
    output_dir : str, optional
        Custom root for reading raw-data CSVs and writing figures.  Must
        match the `output_dir` used during `evaluate(...)`.
    plot_context : {"model", "humans", "all"}
        Which existing raw-data references to overlay next to the
        evaluated model.
    """
    from .plotting.plot import plot as _plot

    dm_fun = _make_simple_decision_maker(
        model_name,
        plotting_name=plotting_name,
        plot_context=plot_context,
    )
    with _override_paths(output_dir):
        _plot(
            plot_types=list(plot_types),
            plotting_definition=dm_fun,
            dataset_names=list(datasets) if datasets else None,
            figure_directory_name=figure_dirname,
        )
        return pjoin(c.FIGURE_DIR, figure_dirname)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    datasets: Sequence[str] = ("cue-conflict",),
    dataset_trained_on: str = "imagenet",
    data_normalization_type: str = "0-1",
    image_size: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 10,
    save_csv: bool = True,
    save_performance_csv: bool = True,
    plot: bool = False,
    plot_types: Sequence[str] = ("shape-bias",),
    figure_dirname: str = "simple_eval_figures",
    output_dir: Optional[str] = None,
    plotting_name: Optional[str] = None,
    plot_context: str = "model",
    compute_repr_shape_bias: bool = False,
    repr_mode: str = "winner-take-all",
    repr_shape_dir: Optional[str] = None,
    repr_texture_dir: Optional[str] = None,
    verbose: bool = False,
    like_resnet: str = "resnet18",
    seperate_from_layer: int = 0,
    baseline: bool = None,
    dvd_strategy: str = "adult",
    age_m: int = None,
    stream_concat: bool = None,
    stream1_weight: float = 0.5,
    lum: bool = None,
    rg: bool = None,
    by: bool = None,
    lum_blur: bool = None,
    lum_cs: bool = None,
    rg_blur: bool = None,
    rg_cs: bool = None,
    by_blur: bool = None,
    by_cs: bool = None,
    combined_color: bool = None,
    epoch: int = -1,
) -> Dict[str, Dict]:
    """Evaluate a single model on one or several datasets and return metrics.

    Parameters
    ----------
    model_name : str
        Model name that must be registered in the neuroai model zoo
        (e.g. "resnet50_DVD_B", "resnet50_baseline_imagenet", ...).
    checkpoint_path : str, optional
        Path to a checkpoint (.pth) to load into the model.
    datasets : sequence of str
        Dataset names, e.g. ["cue-conflict", "edge", "silhouette",
        "sketch", "stylized", "colour", "contrast", ...].
    dataset_trained_on : str
        The training dataset identifier (affects classifier head size
        for custom models); "ecoset", "imagenet", "miniecoset_zejin", ...
    data_normalization_type : {"0-1", "mean-std"}
        Image normalization applied before the forward pass.
    image_size : int, optional
        Override image size (forwarded to the model loader).
    batch_size, num_workers : int
        DataLoader parameters.
    save_csv : bool
        If True, also write per-sample predictions to the standard
        raw-data CSVs so the existing plotting code still works.
    save_performance_csv : bool
        If True, write summary metrics to `raw-data/performances/<model>.csv`.
    plot : bool
        If True, render the visualizations specified by `plot_types`.
    plot_types : sequence of str
        Any of the plot types supported by `neuroai.plotting.plot`,
        e.g. ["shape-bias", "accuracy", "benchmark-barplot", ...].
    figure_dirname : str
        Sub-directory under the plots root where figures are written.
        When `output_dir` is given, plots land in
        `<output_dir>/plots/<figure_dirname>/`; otherwise they land in
        the default `results/plots/<figure_dirname>/`.
    output_dir : str, optional
        Custom root for this run's artifacts.  If given, raw per-sample
        CSVs, performance CSVs, and figures are all written under it:

            <output_dir>/raw-data/<dataset>/<model>_<ds>_session-1.csv
            <output_dir>/performances/<model>.csv
            <output_dir>/plots/<figure_dirname>/...

        Defaults to the package-level paths in `neuroai.constants`.
    plotting_name : str, optional
        Display name used on figures (legend/title).  Defaults to
        `model_name`.
    plot_context : {"model", "humans", "all"}
        Which existing references to overlay on the plots:
        only the evaluated model, the model plus human groups, or the
        model plus human groups and all other models found in raw-data.
    compute_repr_shape_bias : bool
        If True, additionally compute the *representational* shape-texture
        bias via triple correlation between each cue-conflict image and
        its matching original-shape / preprocessed-texture reference
        images.  Only applied to the "cue-conflict" dataset.  The results
        are merged into ``results["cue-conflict"]`` under the key
        ``"repr_shape_bias"``.
    repr_mode : {"winner-take-all", "ratio"}
        Aggregation mode for the representational score.
        - "winner-take-all" (default): each sample contributes 1 iff
              cosine(cc, shape) > cosine(cc, texture); the reported
              bias is the fraction of shape-favouring samples.
        - "ratio": each sample contributes
              cos(cc, shape) / (cos(cc, shape) + cos(cc, texture));
              the reported bias is the mean across samples.
    repr_shape_dir, repr_texture_dir : str, optional
        Reference-image directories. If ``None`` they are auto-discovered
        as siblings of the ``cue-conflict`` dataset folder:
            ``<parent>/original_shape_images/<cls>/<cls><i>.png``
            ``<parent>/preprocessed_textures_224/<cls>/<cls><i>.png``
    verbose : bool
        Print per-batch normalization stats.

    Returns
    -------
    dict
        Mapping dataset_name -> metrics dict (see module docstring).
    """
    # Redirect all writes (raw CSVs, performance CSVs, figures) to
    # `output_dir` for this run. When `output_dir` is None the context
    # manager is a no-op and the default paths are used.
    with _override_paths(output_dir):
        # 1) Load the model + checkpoint once.
        model, framework = load_model(
            model_name,
            load_checkpoint_path=checkpoint_path,
            dataset_trained_on=dataset_trained_on,
            image_size=image_size,
            like_resnet=like_resnet,
            seperate_from_layer=seperate_from_layer,
            baseline=baseline,
            dvd_strategy=dvd_strategy,
            combined_color=combined_color,
            stream_concat=stream_concat,
            stream1_weight=stream1_weight,
        )
        if framework != "pytorch":
            # The simplified evaluator targets PyTorch; tf support stays in
            # the original ModelEvaluator.
            raise NotImplementedError(
                f"simple_eval only supports pytorch models; got framework='{framework}'. "
                "Use the original ModelEvaluator for tensorflow models."
            )
        logger.info("Loaded model %s (framework=%s)", model_name, framework)

        # 2) Iterate datasets, run forward pass, compute metrics.
        results: Dict[str, Dict] = {}
        for ds_name in datasets:
            dataset = load_dataset(
                ds_name,
                batch_size=batch_size,
                num_workers=num_workers,
                dataset_trained_on=dataset_trained_on,
                data_normalization_type=data_normalization_type,
                image_size=image_size,
            )

            df, paths_list = _forward_pass_to_df(
                model=model,
                dataset=dataset,
                model_name=model_name,
                norm_type=data_normalization_type,
                save_csv=save_csv,
                verbose=verbose,
                dvd_strategy=dvd_strategy,
                age_m=age_m,
                lum=lum,
                rg=rg,
                by=by,
                lum_blur=lum_blur,
                lum_cs=lum_cs,
                rg_blur=rg_blur,
                rg_cs=rg_cs,
                by_blur=by_blur,
                by_cs=by_cs,
                combined_color=combined_color,
                epoch=epoch,
            )

            if ds_name == "cue-conflict":
                metrics = compute_shape_bias(df)
                # Optional: representational shape-texture bias.

                if compute_repr_shape_bias:
                    try:
                        # keep former functionality
                        if "ts_" not in model_name:
                            repr_metrics = _compute_repr_shape_bias(
                                model=model,
                                cue_conflict_paths=paths_list,
                                original_shape_dir=repr_shape_dir,
                                texture_dir=repr_texture_dir,
                                mode=repr_mode,
                                image_size=image_size or 224,
                                batch_size=batch_size,
                                norm_type=data_normalization_type,
                            )

                            metrics["repr_shape_bias"] = repr_metrics
                            if save_performance_csv:
                                print_performance_to_csv(
                                    model_name=model_name,
                                    dataset_name=ds_name,
                                    performance=repr_metrics[
                                        "mean_repr_shape_bias"
                                    ],
                                    metric_name=(
                                        f"mean_repr_shape_bias ({repr_mode})"
                                        f"{f' epoch: {epoch}' if epoch != -1 else ''}"
                                    ),
                                )
                                print_performance_to_csv(
                                    model_name=model_name,
                                    dataset_name=ds_name,
                                    performance=repr_metrics[
                                        "median_repr_shape_bias"
                                    ],
                                    metric_name=(
                                        f"median_repr_shape_bias ({repr_mode})"
                                        f"{f' epoch: {epoch}' if epoch != -1 else ''}"
                                    ),
                                )
                        else:
                            repr_metrics = _compute_repr_shape_bias(
                                model=model,
                                cue_conflict_paths=paths_list,
                                original_shape_dir=repr_shape_dir,
                                texture_dir=repr_texture_dir,
                                mode=repr_mode,
                                image_size=image_size or 224,
                                batch_size=batch_size,
                                norm_type=data_normalization_type,
                                baseline=baseline,
                                stream=1,
                                dvd_strategy=dvd_strategy,
                                age_m=age_m,
                                stream_concat=stream_concat,
                                lum=lum,
                                rg=rg,
                                by=by,
                                lum_blur=lum_blur,
                                lum_cs=lum_cs,
                                rg_blur=rg_blur,
                                rg_cs=rg_cs,
                                by_blur=by_blur,
                                by_cs=by_cs,
                                combined_color=combined_color,
                            )
                            metrics["repr_shape_bias_stream_1"] = repr_metrics
                            if save_performance_csv:
                                print_performance_to_csv(
                                    model_name=model_name,
                                    dataset_name=ds_name,
                                    performance=repr_metrics[
                                        "mean_repr_shape_bias"
                                    ],
                                    metric_name=(
                                        f"mean_repr_shape_bias_stream_1 ({repr_mode})"
                                        f"{f' epoch: {epoch}' if epoch != -1 else ''}"
                                    ),
                                )
                                print_performance_to_csv(
                                    model_name=model_name,
                                    dataset_name=ds_name,
                                    performance=repr_metrics[
                                        "median_repr_shape_bias"
                                    ],
                                    metric_name=(
                                        f"median_repr_shape_bias_stream_1 ({repr_mode})"
                                        f"{f' epoch: {epoch}' if epoch != -1 else ''}"
                                    ),
                                )

                            if not baseline:
                                repr_metrics = _compute_repr_shape_bias(
                                    model=model,
                                    cue_conflict_paths=paths_list,
                                    original_shape_dir=repr_shape_dir,
                                    texture_dir=repr_texture_dir,
                                    mode=repr_mode,
                                    image_size=image_size or 224,
                                    batch_size=batch_size,
                                    norm_type=data_normalization_type,
                                    baseline=baseline,
                                    stream=2,
                                    dvd_strategy=dvd_strategy,
                                    age_m=age_m,
                                    stream_concat=stream_concat,
                                    lum=lum,
                                    rg=rg,
                                    by=by,
                                    lum_blur=lum_blur,
                                    lum_cs=lum_cs,
                                    rg_blur=rg_blur,
                                    rg_cs=rg_cs,
                                    by_blur=by_blur,
                                    by_cs=by_cs,
                                    combined_color=combined_color,
                                )
                                metrics["repr_shape_bias_stream_2"] = (
                                    repr_metrics
                                )
                                if save_performance_csv:
                                    print_performance_to_csv(
                                        model_name=model_name,
                                        dataset_name=ds_name,
                                        performance=repr_metrics[
                                            "mean_repr_shape_bias"
                                        ],
                                        metric_name=(
                                            f"mean_repr_shape_bias_stream_2 ({repr_mode})"
                                            f"{f' epoch: {epoch}' if epoch != -1 else ''}"
                                        ),
                                    )
                                    print_performance_to_csv(
                                        model_name=model_name,
                                        dataset_name=ds_name,
                                        performance=repr_metrics[
                                            "median_repr_shape_bias"
                                        ],
                                        metric_name=(
                                            f"median_repr_shape_bias_stream_2 ({repr_mode})"
                                            f"{f' epoch: {epoch}' if epoch != -1 else ''}"
                                        ),
                                    )

                    except Exception as exc:  # pragma: no cover
                        logger.warning(
                            "compute_repr_shape_bias failed: %s",
                            exc,
                        )
                        metrics["repr_shape_bias"] = {"error": str(exc)}
            else:
                metrics = compute_accuracy(df, dataset_name=ds_name)

            # 3) Optionally also dump summary metrics to the legacy CSV layout.
            if save_performance_csv:
                print_performance_to_csv(
                    model_name=model_name,
                    dataset_name=ds_name,
                    performance=metrics["accuracy"],
                    metric_name=(
                        "accuracy (top-1)"
                        f"{f' epoch: {epoch}' if epoch != -1 else ''}"
                    ),
                )
                if ds_name == "cue-conflict":
                    print_performance_to_csv(
                        model_name=model_name,
                        dataset_name=ds_name,
                        performance=metrics["mean_shape_bias"],
                        metric_name=(
                            "mean_shape_bias"
                            f"{f' epoch: {epoch}' if epoch != -1 else ''}"
                        ),
                    )
                    print_performance_to_csv(
                        model_name=model_name,
                        dataset_name=ds_name,
                        performance=metrics["median_shape_bias"],
                        metric_name=(
                            "median_shape_bias"
                            f"{f' epoch: {epoch}' if epoch != -1 else ''}"
                        ),
                    )

            results[ds_name] = metrics
            print(f"[{ds_name}]  " + _pretty_print(metrics))

        # 4) Optionally render figures (still inside the path override so
        #    both the reads and the writes go to `output_dir`).
        if plot:
            if epoch == -1:
                out_dir = visualize(
                    model_name=model_name,
                    datasets=datasets,
                    plot_types=plot_types,
                    figure_dirname=figure_dirname,
                    plotting_name=plotting_name,
                    plot_context=plot_context,
                    # output_dir is None here on purpose: the outer context
                    # manager has already redirected c.FIGURE_DIR etc., and
                    # visualize() would try to redirect them again otherwise.
                    output_dir=None,
                )
                print(f"Figures saved to: {out_dir}")

        return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _pretty_print(metrics: Dict) -> str:
    """Return a one-line human-readable summary of a metrics dict."""
    parts = [f"accuracy={metrics['accuracy']*100:.2f}%"]
    if "mean_shape_bias" in metrics:
        parts.append(f"mean_shape_bias={metrics['mean_shape_bias']:.3f}")
        parts.append(f"median_shape_bias={metrics['median_shape_bias']:.3f}")
    if (
        "repr_shape_bias" in metrics
        and "mean_repr_shape_bias" in metrics["repr_shape_bias"]
    ):
        r = metrics["repr_shape_bias"]
        parts.append(
            f"repr_shape_bias[{r['mode']}]="
            f"mean={r['mean_repr_shape_bias']:.3f},"
            f"median={r['median_repr_shape_bias']:.3f}"
        )
    if "accuracy_per_cond" in metrics and metrics["accuracy_per_cond"]:
        conds = ", ".join(
            f"{k}={v*100:.1f}%"
            for k, v in metrics["accuracy_per_cond"].items()
        )
        parts.append(f"per_cond=[{conds}]")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Command-line / training-loop usage
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple shape-bias / accuracy evaluator."
    )
    parser.add_argument(
        "--model", required=True, help="Registered model name."
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Path to checkpoint .pth."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cue-conflict"],
        help="Datasets to evaluate on (space-separated).",
    )
    parser.add_argument(
        "--trained-on", default="imagenet", help="dataset_trained_on."
    )
    parser.add_argument("--norm", default="0-1", choices=["0-1", "mean-std"])
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument(
        "--plot", action="store_true", help="Render visualization PDFs."
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        default=["shape-bias"],
        help="Plot types: shape-bias, accuracy, benchmark-barplot, ...",
    )
    parser.add_argument("--figure-dir", default="simple_eval_figures")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom root for raw CSVs, performance CSVs, and figures.",
    )
    parser.add_argument(
        "--plotting-name",
        default=None,
        help="Display name shown on plots (defaults to --model).",
    )
    parser.add_argument(
        "--plot-context",
        default="model",
        choices=list(SIMPLE_EVAL_PLOT_CONTEXTS)
        + sorted(_SIMPLE_EVAL_PLOT_CONTEXT_ALIASES.keys()),
        help="Which references to overlay: model, humans, or all.",
    )
    parser.add_argument(
        "--repr-shape-bias",
        action="store_true",
        help="Also compute the representational shape-texture bias via "
        "triple correlation (cue-conflict dataset only).",
    )
    parser.add_argument(
        "--repr-mode",
        default="winner-take-all",
        choices=["winner-take-all", "ratio"],
        help="Aggregation mode for the representational score.",
    )
    parser.add_argument(
        "--repr-shape-dir",
        default=None,
        help="Directory of original shape images (auto-detected by default).",
    )
    parser.add_argument(
        "--repr-texture-dir",
        default=None,
        help="Directory of preprocessed texture images (auto-detected by default).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        datasets=args.datasets,
        dataset_trained_on=args.trained_on,
        data_normalization_type=args.norm,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        plot=args.plot,
        plot_types=args.plot_types,
        figure_dirname=args.figure_dir,
        output_dir=args.output_dir,
        plotting_name=args.plotting_name,
        plot_context=args.plot_context,
        compute_repr_shape_bias=args.repr_shape_bias,
        repr_mode=args.repr_mode,
        repr_shape_dir=args.repr_shape_dir,
        repr_texture_dir=args.repr_texture_dir,
        verbose=args.verbose,
    )
