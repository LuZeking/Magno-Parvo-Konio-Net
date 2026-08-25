"""
Read PyTorch model from .pth.tar checkpoint.
Code from:
https://github.com/rgeirhos/texture-vs-shape/blob/master/models/load_pretrained_models.py
Accessed 02.03.2020
"""

import os
from collections import OrderedDict
from itertools import product

import timm
import torch
import torchvision.models as tv_models
from torch.utils import model_zoo

from src.models.pytorch_models import TwoStreamNetwork

###############################################################################
# Global Dictionaries and Variables
###############################################################################

model_urls = {
    "resnet50_trained_on_SIN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar",
    "resnet50_trained_on_SIN_and_IN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar",
    "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar",
    "vgg16_trained_on_SIN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar",
    "alexnet_trained_on_SIN": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar",
}


# Directory containing all locally saved checkpoints.
source_dir = "/share/klab/lzejin/lzejin/codebase/P001_DVD/logs/shared_weights/"  #! /share/klab/lzejin/lzejin/codebase/P001_DVD/logs/open_source_weights
# source_dir = '/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/logs/' #! /share/klab/lzejin/lzejin/codebase/P001_DVD/logs/open_source_weights

# from '/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/logs/', copy the 44 epochs weights under '/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/logs/'+ "model name" + 44 epoch (the last one move 49 epoch), then finally rename the to best epoch in the new dir ('/share/klab/lzejin/lzejin/codebase/P001_DVD/logs/shared_weights/' ). --> also keep the same strcute like model name/weights/checkpoint_best.pth
# Corresponding epochs for each checkpoint.
epochs = [
    "best",
    "best",
    "best",
    "best",
    "best",
    "best",
] + [
    "best"
] * 120  # adjust if you change epochs above

# ''' Weight paths for imagenet trained DVD models:
#     "resnet50_baseline_imagenet": "/share/klab/lzejin/lzejin/codebase/P001_DVD/logs/shared_weights/resnet50_imagenet_256_0.0001_dev_adult_seed_1/weights/checkpoint_best.pth",
#     "resnet50_DVD_P_imagenet":    "/share/klab/lzejin/lzejin/codebase/P001_DVD/logs/shared_weights/resnet50_mpe4.0_alpha0.4_dn50.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_best.pth",
#     "resnet50_DVD_B_imagenet":    "/share/klab/lzejin/lzejin/codebase/P001_DVD/logs/shared_weights/resnet50_mpe2.0_alpha0.1_dn150.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_best.pth",
#     "resnet50_DVD_S_imagenet":   "/share/klab/lzejin/lzejin/codebase/P001_DVD/logs/shared_weights/resnet50_mpe1.0_alpha0.1_dn150.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_best.pth",
# '''

# Map of model names to their folder names, used to construct full checkpoint paths.
model_to_savename = {
    # 1a ecoset – main models
    "resnet50_baseline": "resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1",
    "resnet50_DVD_P": "resnet50_mpe4.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_DVD_B": "resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_DVD_S": "resnet50_mpe1.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_DVD_SS": "resnet50_mpe1.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_DVD_PP": "resnet50_mpe8.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 1b ImageNet – main models
    # "resnet50_baseline_imagenet": "resnet50_imagenet_256_0.0001_dev_adult_seed_1",
    # "resnet50_DVD_B_imagenet":    "resnet50_mpe2.0_alpha0.1_dn150.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # "resnet50_DVD_S_imagenet":    "resnet50_mpe1.0_alpha0.1_dn100.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # "resnet50_DVD_SS_imagenet":   "resnet50_mpe1.0_alpha0.1_dn150.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # "resnet50_DVD_P_imagenet":    "resnet50_mpe4.0_alpha0.4_dn50.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # "resnet50_DVD_PP_imagenet":   "resnet50_mpe8.0_alpha0.8_dn50.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_baseline_imagenet": "baseline",
    "resnet50_DVD_P_imagenet": "resnet50_imagenet_DVD_P",
    "resnet50_DVD_B_imagenet": "resnet50_imagenet_DVD_B",
    "resnet50_DVD_S_imagenet": "resnet50_imagenet_DVD_S",
    # 1c mini-ecoset – main models
    "resnet50_baseline_miniecoset": "resnet50_texture2shape_miniecoset_256_0.0001_dev_adult_seed_1",
    "resnet50_DVD_B_miniecoset": "resnet50_mpe2.0_alpha0.1_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_DVD_S_miniecoset": "resnet50_mpe1.0_alpha0.1_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_DVD_SS_miniecoset": "resnet50_mpe1.0_alpha0.2_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_DVD_P_miniecoset": "resnet50_mpe4.0_alpha0.8_dn50.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_DVD_PP_miniecoset": "resnet50_mpe8.0_alpha0.2_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 2. Additional architectures (baseline vs balanced)
    "alexnet_baseline": "alexnet_ecoset_square256_256_0.0001_dev_adult_seed_1",
    "alexnet_DVD_B": "alexnet_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "vgg16_baseline": "vgg16_ecoset_square256_256_0.0001_dev_adult_seed_1",
    "vgg16_DVD_B": "vgg16_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "mobilenet_v3_small_baseline": "mobilenet_v3_small_ecoset_square256_256_0.0001_dev_adult_seed_1",
    "mobilenet_v3_small_DVD_B": "mobilenet_v3_small_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "shufflenet_v2_x1_0_baseline": "shufflenet_v2_x1_0_ecoset_square256_256_0.0001_dev_adult_seed_1",
    "shufflenet_v2_x1_0_DVD_B": "shufflenet_v2_x1_0_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet18_baseline": "resnet18_ecoset_square256_256_0.0001_dev_adult_seed_1",
    "resnet18_DVD_B": "resnet18_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet101_baseline": "resnet101_ecoset_square256_256_0.0001_dev_adult_seed_1",
    "resnet101_DVD_B": "resnet101_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnext50_32x4d_baseline": "resnext50_32x4d_ecoset_square256_256_0.0001_dev_adult_seed_1",
    "resnext50_32x4d_DVD_B": "resnext50_32x4d_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "vit_b_16_baseline": "vit_b_16_ecoset_square256_256_0.0001_dev_adult_seed_1",
    "vit_b_16_DVD_B": "vit_b_16_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "deit_base_patch16_224_baseline": "deit_base_patch16_224_ecoset_square256_256_0.0001_dev_adult_seed_1",
    "deit_base_patch16_224_DVD_B": "deit_base_patch16_224_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 3. Ablation models
    "resnet50_DVD_B_blur_only": "resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c0cs0_T_normal_seed_1",
    "resnet50_DVD_B_color_only": "resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b0c1cs0_T_normal_seed_1",
    "resnet50_DVD_B_contrast_only": "resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b0c0cs1_T_normal_seed_1",
    "resnet50_DVD_B_blur_color_only": "resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs0_T_normal_seed_1",
    "resnet50_DVD_B_blur_contrast_only": "resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c0cs1_T_normal_seed_1",
    "resnet50_DVD_B_color_contrast_only": "resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b0c1cs1_T_normal_seed_1",
    # extra: adversarially trained models
    "resnet50_adult_adv_pgd_linf_eps0_001_k10": "resnet50_ecoset_square256_256_0.0001_dev_adult_seed_1_adv-pgd_linf_eps0.001_k10",
    "resnet50_adult_adv_pgd_linf_eps0_001_k10_seed2": "resnet50_ecoset_square256_256_0.0001_dev_adult_seed_2_adv-pgd_linf_eps0.001_k10",
    "resnet50_adult_adv_pgd_linf_eps0_001_k10_seed3": "resnet50_ecoset_square256_256_0.0001_dev_adult_seed_3_adv-pgd_linf_eps0.001_k10",
    "resnet50_adult_adv_pgd_linf_eps0_0039_k10_seed_1": "resnet50_ecoset_square256_256_0.0001_dev_adult_seed_1_adv-pgd_linf_eps0.0039_k10",
    "resnet50_adult_adv_pgd_linf_eps0_01568_k10_seed_1": "resnet50_ecoset_square256_256_0.0001_dev_adult_seed_1_adv-pgd_linf_eps0.01568_k10",
    # * SSL models ms 08
    # miniecsoet
    "resnet50_finetuned_DVD_B_SimCLR_miniecoset_ms08_dvd_data_aug_seed1": "miniecoset_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed0/linear_from_e0149",
    "resnet50_finetuned_adult_SimCLR_miniecoset_ms08_dvd_data_aug_seed1": "miniecoset_resnet50_sz224_norm01_ms08_dvd_baseline_fast_seed0/linear_from_e0149/",
    # ecoset
    "resnet50_finetuned_DVD_B_SimCLR_ecoset_ms08_dvd_data_aug_seed1": "ecoset_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed1/linear_from_e0149_full_ckpt",
    "resnet50_finetuned_adult_SimCLR_ecoset_ms08_dvd_data_aug_seed1": "ecoset_resnet50_sz224_norm01_ms08_dvd_baseline_fast_seed1/linear_from_e0074_full_ckpt",
    # imagenet 1k
    "resnet50_finetuned_DVD_B_SimCLR_imagenet_1k_ms08_dvd_data_aug_seed1": "imagenet_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed1/linear_from_e0114",
    "resnet50_finetuned_adult_SimCLR_imagenet_1k_dvd_data_aug_seed1": "",
    # BYOL & DINO
    # 'byol_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1', 'dino_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1','dino_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1','byol_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1','dino_lin_eval_DVD_B_imagenet_ms08_dvd_data_aug_seed1','byol_lin_eval_DVD_B_magenet_ms08_dvd_data_aug_seed1'
    #! ino need retrain fro mlast ep: logs/byol_miniecoset_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed1/checkpoint_0149.pth
    "resnet50_byol_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1": "byol_miniecoset_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed1/linear_from_e0149/",
    # 89 ep
    "resnet50_dino_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1": "dino_miniecoset_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed1/linear_from_e0124/",
    # best 0.78 --> need to traverse to see  --> need train more epoch
    # ep 19
    "resnet50_dino_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1": "dino_ecoset_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed1/linear_from_e0139/",
    # best is fine
    "resnet50_byol_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1": "byol_ecoset_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed1/linear_from_e0124/",
    # best 0.78 --> need to traverse to see
    # ep 19
    "resnet50_dino_lin_eval_DVD_B_imagenet_ms08_dvd_data_aug_seed1": "dino_imagenet_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed1/linear_from_e0114/",
    # # last ok
    "resnet50_byol_lin_eval_DVD_B_magenet_ms08_dvd_data_aug_seed1": "byol_imagenet_resnet50_sz224_norm01_ms08_dvd_baseline_fast_dev_dvd_normal_mpe2_alpha01_beta150_seed1/linear_from_e0149/",
    # * other Custom (no default built-in path, need input)
    "swin_b_ours": None,
    "vit_b_16_ours": None,
    "resnet50_ours": None,
    "resnet18_ours": None,
    "resnet101_ours": None,
    "mobilenet_v3_small_ours": None,
    "shufflenet_v2_x1_0_ours": None,
    "resnext50_32x4d_ours": None,
    "deit_base_patch16_224_ours": None,
}

# * tarverse hyperparameters to include trade-off models in mini-ecoset
# Hyperparameters for DVD
months_per_epochs = [
    1.0,
    2.0,
    4.0,
    8.0,
][:4]
contrast_thresh = [0.1, 0.2, 0.4, 0.8][:4]  # Alpha for contrast
contrast_spd = [50.0, 100.0, 150.0][:3]  # Beta for contrast
for mpe, alpha, beta in product(
    months_per_epochs, contrast_thresh, contrast_spd
):
    model_type = "resnet50"
    DATASET_TRAINED_ON = "texture2shape_miniecoset"
    model_save_name = f"{model_type}_mpe{mpe}_alpha{alpha}_dn{beta}".replace(
        ".", "_"
    )  # * Setting name for models     #! notice model_save_name .replace(".", "_")
    model_subdir = (
        f"{model_type}_mpe{mpe}_alpha{alpha}_dn{beta}_"
        f"{DATASET_TRAINED_ON}256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1"
    )

    model_to_savename[model_save_name] = model_subdir

# Build dictionary of each model_name -> local checkpoint path (if applicable).
model_paths = {
    name: (
        f"{source_dir}/{path}/weights/checkpoint_{epochs[list(model_to_savename.keys()).index(name)]}.pth"
        if path
        else None
    )
    for name, path in model_to_savename.items()
}


###############################################################################
# Public interface
###############################################################################


def load_model(model_name, *args, **kwargs):
    """
    Main entry point. Supports ResNet50/18/101, ResNeXt-50, VGG16, AlexNet,
    MobileNetV3-Small, ShuffleNetV2 x1.0, ViT-B/16, Swin-B, and timm DeiT-B/16.
    """
    dataset_trained_on = kwargs.pop("dataset_trained_on", None)

    # Decide classifier dim
    if "imagenet" in model_name.lower():
        out_class_dim = 1000
    elif "miniecoset" in model_name.lower():
        out_class_dim = 112
    elif "ecoset" in model_name.lower():
        out_class_dim = 565
    else:
        out_class_dim = (
            get_output_dim(dataset_trained_on) if dataset_trained_on else 1000
        )

    if "ts_" in model_name:
        return load_two_stream_model(model_name, out_class_dim, **kwargs)
    elif "resnet50" in model_name:
        return load_cnn_model(model_name, "resnet50", out_class_dim, **kwargs)
    elif "resnet18" in model_name:
        return load_cnn_model(model_name, "resnet18", out_class_dim, **kwargs)
    elif "resnet101" in model_name:
        return load_cnn_model(model_name, "resnet101", out_class_dim, **kwargs)
    elif "resnext50_32x4d" in model_name:
        return load_cnn_model(
            model_name, "resnext50_32x4d", out_class_dim, **kwargs
        )
    elif "vgg16" in model_name:
        return load_cnn_model(model_name, "vgg16", out_class_dim, **kwargs)
    elif "alexnet" in model_name:
        return load_cnn_model(model_name, "alexnet", out_class_dim, **kwargs)
    elif "mobilenet_v3_small" in model_name:
        return load_cnn_model(
            model_name, "mobilenet_v3_small", out_class_dim, **kwargs
        )
    elif "shufflenet_v2_x1_0" in model_name:
        return load_cnn_model(
            model_name, "shufflenet_v2_x1_0", out_class_dim, **kwargs
        )
    elif "vit_b_16" in model_name:
        return load_swin_transformer_model(
            model_name, "vit_b_16", out_class_dim, **kwargs
        )
    elif "swin_b" in model_name:
        return load_swin_transformer_model(
            model_name, "swin_b", out_class_dim, **kwargs
        )
    elif "deit_base_patch16_224" in model_name:
        return load_timm_vit_model(
            model_name, "deit_base_patch16_224", out_class_dim, **kwargs
        )
    else:
        raise ValueError(f"Unknown model architecture for '{model_name}'.")


###############################################################################
# Helper Functions
###############################################################################


def get_output_dim(dataset_identifier):
    if dataset_identifier in ["ecoset", "ecoset_square256"]:
        return 565
    elif str(dataset_identifier) in [
        "texture2shape_miniecoset",
        "miniecoset_zejin",
    ]:
        return 112
    elif "16" in str(dataset_identifier):
        return 16
    elif "imagenet" in str(dataset_identifier):
        return 1000
    else:
        raise RuntimeError(
            f"Unsupported dataset identifier: {dataset_identifier}"
        )


def _make_torchvision_model(arch):
    """Construct model with modern 'weights=None' API (avoids deprecation warnings)."""
    try:
        return tv_models.__dict__[arch](weights=None)
    except TypeError:
        # fallback for very old torchvision
        return tv_models.__dict__[arch](pretrained=False)


def _resolve_checkpoint(model_name, **kwargs):
    """Pick custom path > local path dict > remote URL dict. Load the checkpoint object."""
    custom = kwargs.get("load_checkpoint_path", None)
    if custom and os.path.exists(custom):
        print(f"Loading local checkpoint from: {custom}")
        return _safe_torch_load(
            custom,
            want_optimizer=False,
            map_location=("cuda" if torch.cuda.is_available() else "cpu"),
        )

    path = model_paths.get(model_name)
    if path and os.path.exists(path):
        print(f"Loading local checkpoint from: {path}")
        return _safe_torch_load(
            path,
            want_optimizer=False,
            map_location=("cuda" if torch.cuda.is_available() else "cpu"),
        )

    if model_name in model_urls:
        print(
            f"Loading remote checkpoint from model_urls for {model_name} ..."
        )
        return model_zoo.load_url(model_urls[model_name])

    print(f"No valid checkpoint found for {model_name}.")
    # import pdb; pdb.set_trace()
    raise EOFError


def _load_state_into_model(model, checkpoint):
    """Canonicalize the checkpoint and load into model with strict=False and a summary."""
    raw = _ensure_state_dict_obj(checkpoint)
    sd = canonicalize_state_dict_for_model(raw, model)
    result = model.load_state_dict(sd, strict=False)
    _summarize_load(result)


def load_two_stream_model(model_name, out_class_dim=112, **kwargs):
    """Load the custom Two Stream Model based on ResNet*"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    like_resnet = kwargs.get("like_resnet", "resnet18")
    seperate_from_layer = kwargs.get("seperate_from_layer", 0)
    baseline = kwargs.get("baseline", False)
    dvd_strategy = kwargs.get("dvd_strategy", "dvd")
    combined_color = kwargs.get("combined_color", False)
    stream_concat = kwargs.get("stream_concat", False)
    stream1_weight = kwargs.get("stream1_weight", False)

    model = TwoStreamNetwork(
        num_classes=out_class_dim,
        dropout=0.0,
        like_resnet=like_resnet,
        seperate_from_layer=seperate_from_layer,
        baseline=baseline,
        dvd_strategy=dvd_strategy,
        combined_color=combined_color,
        stream_concat=stream_concat,
        stream1_weight=stream1_weight,
    ).to(device)

    checkpoint_path = kwargs.get("load_checkpoint_path")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # get the state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise ValueError("Checkpoint does not contain 'state_dict'")

    # rename the keys
    renamed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("network."):
            renamed_key = key[len("network.") :]
        else:
            renamed_key = key
        renamed_state_dict[renamed_key] = value

    model.load_state_dict(renamed_state_dict)
    return model


def load_cnn_model(model_name, arch, out_class_dim=1000, **kwargs):
    """
    Loads CNNs (ResNet*, ResNeXt-50, VGG16, AlexNet, MobileNetV3, ShuffleNetV2).
    Handles DP prefixes + wrapper remaps.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _make_torchvision_model(arch).to(device)

    if any(s in arch.lower() for s in ("resnet", "resnext", "shufflenet_v2")):
        model.fc = torch.nn.Linear(model.fc.in_features, out_class_dim).to(
            device
        )
    elif any(s in arch.lower() for s in ("vgg", "alexnet", "mobilenet_v3")):
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, out_class_dim
        ).to(device)
    else:
        raise ValueError(f"Unknown CNN architecture: {arch}")

    ckpt = _resolve_checkpoint(model_name, **kwargs)
    _load_state_into_model(model, ckpt)
    return model


def load_swin_transformer_model(
    model_name, arch, out_class_dim=1000, **kwargs
):
    """
    Loads ViT-B/16 (torchvision) and Swin-B (torchvision).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _make_torchvision_model(arch).to(device)

    if "vit_b_16" in arch:
        model.heads.head = torch.nn.Linear(
            model.heads.head.in_features, out_class_dim
        ).to(device)
    else:
        model.head = torch.nn.Linear(model.head.in_features, out_class_dim).to(
            device
        )

    ckpt = _resolve_checkpoint(model_name, **kwargs)
    _load_state_into_model(model, ckpt)
    return model


def load_timm_vit_model(model_name, arch, out_class_dim=1000, **kwargs):
    """
    Loads DeiT-B/16 (timm).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(arch, pretrained=False).to(device)
    model.head = torch.nn.Linear(model.head.in_features, out_class_dim).to(
        device
    )

    ckpt = _resolve_checkpoint(model_name, **kwargs)
    _load_state_into_model(model, ckpt)
    return model


###############################################################################
# Canonicalization helpers
###############################################################################


def _safe_torch_load(path, want_optimizer=False, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


from collections import OrderedDict


def _ensure_state_dict_obj(ckpt):
    """Return the actual state_dict mapping from various checkpoint formats."""
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state", "weights", "params"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            for k in ("state_dict", "model_state", "weights", "params"):
                if k in ckpt["model"] and isinstance(ckpt["model"][k], dict):
                    return ckpt["model"][k]
    return ckpt


def _strip_known_prefixes_once(k: str) -> str:
    prefixes = (
        "module._orig_mod.",
        "module.",
        "_orig_mod.",
        "model.",
        "network.",
        "net.",
        "student.",
        "ema.",
        "encoder.",
        "backbone.body.",
        "visual.",
    )
    out = k
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if out.startswith(p):
                out = out[len(p) :]
                changed = True
                break
    return out


def _remap_backbone_key(k: str) -> str:
    """
    Handle wrapper backbone names like:
      backbone.0.* -> conv1.*
      backbone.1.* -> bn1.*
      backbone.4.* -> layer1.*
      backbone.5.* -> layer2.*
      backbone.6.* -> layer3.*
      backbone.7.* -> layer4.*
    """
    if not k.startswith("backbone."):
        return k

    rest = k[len("backbone.") :]
    mapping = {
        "0.": "conv1.",
        "1.": "bn1.",
        "4.": "layer1.",
        "5.": "layer2.",
        "6.": "layer3.",
        "7.": "layer4.",
    }
    for src, tgt in mapping.items():
        if rest.startswith(src):
            return tgt + rest[len(src) :]

    return k


def _candidate_keys_for_target(k: str, model_keys_bare: set[str]):
    """
    Produce plausible aliases for a checkpoint key.
    Keep it conservative so AlexNet/VGG keep working while ResNet trunk keys load cleanly.
    """
    k = _strip_known_prefixes_once(k)
    k = _remap_backbone_key(k)

    cands = [k]

    # classifier.fc.* -> fc.*
    if k.startswith("classifier.fc."):
        cands.append("fc." + k[len("classifier.fc.") :])

    # fc.* <-> classifier.*
    if k.startswith("fc."):
        suffix = k[len("fc.") :]
        cands.append("classifier." + suffix)
        if suffix in ("weight", "bias"):
            cands.append(
                f"classifier.0.{suffix}"
            )  # VGG/AlexNet first layer special case

    if k.startswith("classifier."):
        suffix = k[len("classifier.") :]
        cands.append("fc." + suffix)
        cands.append("head." + suffix)
        if suffix in ("weight", "bias"):
            cands.append(f"classifier.0.{suffix}")

    # torchvision ViT <-> timm-style
    if k.startswith("heads.head."):
        cands.append("head." + k[len("heads.head.") :])
    if k.startswith("head."):
        cands.append("heads.head." + k[len("head.") :])

    # keep only unique
    seen = set()
    out = []
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _is_head_key(k: str) -> bool:
    return (
        k.startswith("fc.")
        or k.startswith("classifier.")
        or k.startswith("head.")
        or k.startswith("heads.")
    )


def canonicalize_state_dict_for_model(raw_state_dict, model):
    """
    Robustly map checkpoint keys to exactly what `model.state_dict()` expects.
    Works for bare models (your current texture-shape setup), and preserves
    AlexNet/VGG behavior while fixing ResNet trunk loading.
    """
    own = model.state_dict()
    own_keys = list(own.keys())
    own_keys_bare = {k[7:] if k.startswith("module.") else k for k in own_keys}
    model_has_module = len(own_keys) > 0 and all(
        k.startswith("module.") for k in own_keys
    )

    out = OrderedDict()

    for raw_k, v in raw_state_dict.items():
        chosen = None

        for cand in _candidate_keys_for_target(raw_k, own_keys_bare):
            final = "module." + cand if model_has_module else cand
            if final in own:
                # for heads, require shape match; for trunk, exact-key match is enough
                if _is_head_key(cand):
                    if tuple(own[final].shape) == tuple(v.shape):
                        chosen = final
                        break
                else:
                    if tuple(own[final].shape) == tuple(v.shape):
                        chosen = final
                        break

        if chosen is not None:
            out[chosen] = v

    return out


def _summarize_load(result):
    if hasattr(result, "missing_keys"):
        missing = result.missing_keys
        unexpected = result.unexpected_keys
    else:
        missing, unexpected = result
    if missing:
        print(
            f"[load] Missing keys: {len(missing)} (showing up to 10): {missing[:10]}"
        )
    if unexpected:
        print(
            f"[load] Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}"
        )
    if not missing and not unexpected:
        print("[load] Perfect key match.")
