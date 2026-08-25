#!/usr/bin/env python3
"""Minimal example of the simplified evaluator.

    python scripts/simple_evaluate.py

Edit the MODEL / CHECKPOINT / DATASETS variables below to match what you
want to evaluate.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from src.analysis.neuroai.neuroai import evaluate

# ----- edit me ----------------------------------------------------------
MODEL = "ts_resnet18_dkl_blur_cs_medium"
CHECKPOINT = "/share/klab/lzejin/mbenito/two_stream_dvd/logs/dkl_blur_cs_medium/epoch=149-step=154350.ckpt"  # e.g. "/path/to/checkpoint_best.pth"
DATASETS = ["cue-conflict"]  # + ["edge", "silhouette", "sketch", "stylized"]
TRAINED_ON = (
    "texture2shape_miniecoset"  # "ecoset", "miniecoset_zejin" or "imagenet"
)
NORM = "0-1"  # or "mean-std"
DO_PLOT = True
PLOT_CONTEXT = "model"  # "model", "humans" (humans + current model), or "all" (everything)

# Set to a folder path to save everything (raw CSVs + performances + plots)
# under that folder.  Leave as None to use the package defaults.
OUTPUT_DIR = "/share/klab/lzejin/mbenito/two_stream_dvd/src/visualizations/assets/texture_shape_output"  # all artifacts go here
FIGURE_DIRNAME = f"simple_eval/{MODEL}"  # sub-folder for plots

# Representational shape-texture bias (triple correlation).  Set to True
# to enable.  The reference dirs default to siblings of cue-conflict:
#   <parent>/cue-conflict/<cls>/<cls><i>-<cls><j>.png
#   <parent>/original_shape_images/<cls>/<cls><i>.png
#   <parent>/preprocessed_textures_224/<cls>/<cls><j>.png
DO_REPR_SHAPE_BIAS = True  # the ONE option that adds this analysis
REPR_MODE = "winner-take-all"  # "winner-take-all" or "ratio"
REPR_SHAPE_DIR = "/share/klab/lzejin/mbenito/two_stream_dvd/datasets/original_shape_images"  # auto-detect unless given explicitly, you can copy from "/share/klab/lzejin/lzejin/codebase/neuro-vs-ai/datasets/original_shape_images"
REPR_TEXTURE_DIR = "/share/klab/lzejin/mbenito/two_stream_dvd/datasets/original_texture_images"  # auto-detect unless given explicitly, you can copy from  "/share/klab/lzejin/lzejin/codebase/neuro-vs-ai/datasets/preprocessed_textures_224"


## Register a brand-new model in-place (uncomment to use):
# register_pytorch_model(
#     name            = "resnet50_my_new_experiment",
#     checkpoint_path = "/path/to/my_checkpoint_best.pth",
#     # out_class_dim = 1000,   # optional, auto-inferred from dataset_trained_on
#     # overwrite    = True,    # optional, re-register if name exists
# )
# MODEL = "resnet50_my_new_experiment"

## Two stream dvd additional parameters
LIKE_RESNET = "resnet18"
SEPERATE_FROM_LAYER = 0
BASELINE = False
DVD_STRATEGY = "dkl"
AGE_M = 24
STREAM_CONCAT = False
STREAM1_WEIGHT = 0.5
LUM = True
RG = True
BY = False
LUM_BLUR = True
LUM_CS = True
RG_BLUR = False
RG_CS = False
BY_BLUR = False
BY_CS = False
COMBINED_COLOR = False
EPOCH = -1

# ------------------------------------------------------------------------


def main():
    results = evaluate(
        model_name=MODEL,
        checkpoint_path=CHECKPOINT,
        datasets=DATASETS,
        dataset_trained_on=TRAINED_ON,
        data_normalization_type=NORM,
        batch_size=64,
        num_workers=10,
        plot=DO_PLOT,
        plot_types=["shape-bias"],
        figure_dirname=FIGURE_DIRNAME,
        output_dir=OUTPUT_DIR,
        plot_context=PLOT_CONTEXT,
        compute_repr_shape_bias=DO_REPR_SHAPE_BIAS,
        repr_mode=REPR_MODE,
        repr_shape_dir=REPR_SHAPE_DIR,
        repr_texture_dir=REPR_TEXTURE_DIR,
        verbose=False,
        like_resnet=LIKE_RESNET,
        seperate_from_layer=SEPERATE_FROM_LAYER,
        baseline=BASELINE,
        dvd_strategy=DVD_STRATEGY,
        age_m=AGE_M,
        stream_concat=STREAM_CONCAT,
        stream1_weight=STREAM1_WEIGHT,
        lum=LUM,
        rg=RG,
        by=BY,
        lum_blur=LUM_BLUR,
        lum_cs=LUM_CS,
        rg_blur=RG_BLUR,
        rg_cs=RG_CS,
        by_blur=BY_BLUR,
        by_cs=BY_CS,
        combined_color=COMBINED_COLOR,
        epoch=EPOCH,
    )

    # Short summary to stdout.
    for ds_name, metrics in results.items():
        print(f"\n=== {ds_name} ===")
        for k, v in metrics.items():
            if isinstance(v, dict):
                continue  # skip per-category dicts in the summary print
            print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    main()
