#!/usr/bin/env python3
"""Minimal example of the simplified evaluator.

    python scripts/simple_evaluate.py

Edit the MODEL / CHECKPOINT / DATASETS variables below to match what you
want to evaluate.  
"""

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from neuroai import evaluate, register_pytorch_model  


# ----- edit me ----------------------------------------------------------
MODEL       = "resnet50_DVD_B"
CHECKPOINT  = "/share/klab/lzejin/lzejin/codebase/P001_DVD/logs/shared_weights/resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_best.pth"  # e.g. "/path/to/checkpoint_best.pth"
DATASETS    = ["cue-conflict"]  # + ["edge", "silhouette", "sketch", "stylized"]
TRAINED_ON  = "ecoset"        # "ecoset", "miniecoset_zejin" or "imagenet"
NORM        = "0-1"             # or "mean-std"
DO_PLOT     = True
PLOT_CONTEXT = "model"          # "model", "humans" (humans + current model), or "all" (everything)

# Set to a folder path to save everything (raw CSVs + performances + plots)
# under that folder.  Leave as None to use the package defaults.
OUTPUT_DIR    = "/share/klab/lzejin/lzejin/codebase/neuro-vs-ai/results/shape_bias/test_run"            # all artifacts go here
FIGURE_DIRNAME = f"simple_eval/{MODEL}" # sub-folder for plots

# Representational shape-texture bias (triple correlation).  Set to True
# to enable.  The reference dirs default to siblings of cue-conflict:
#   <parent>/cue-conflict/<cls>/<cls><i>-<cls><j>.png
#   <parent>/original_shape_images/<cls>/<cls><i>.png
#   <parent>/preprocessed_textures_224/<cls>/<cls><j>.png
DO_REPR_SHAPE_BIAS = True      # the ONE option that adds this analysis
REPR_MODE          = "winner-take-all"   # or "ratio"
REPR_SHAPE_DIR     = None       # auto-detect unless given explicitly, you can copy from "/share/klab/lzejin/lzejin/codebase/neuro-vs-ai/datasets/original_shape_images"
REPR_TEXTURE_DIR   = None       # auto-detect unless given explicitly, you can copy from  "/share/klab/lzejin/lzejin/codebase/neuro-vs-ai/datasets/preprocessed_textures_224"


## Register a brand-new model in-place (uncomment to use):
# register_pytorch_model(
#     name            = "resnet50_my_new_experiment",
#     checkpoint_path = "/path/to/my_checkpoint_best.pth",
#     # out_class_dim = 1000,   # optional, auto-inferred from dataset_trained_on
#     # overwrite    = True,    # optional, re-register if name exists
# )
# MODEL = "resnet50_my_new_experiment"
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
