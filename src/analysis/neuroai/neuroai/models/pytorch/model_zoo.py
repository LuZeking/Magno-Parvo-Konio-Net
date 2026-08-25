#!/usr/bin/env python3
import torch

# * VLMs
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from ..registry import register_model
from ..wrappers.pytorch import (  # , CogVLMModel
    Blip2PytorchModel,
    ClipPytorchModel,
    EfficientNetPytorchModel,
    LlavaPytorchModel,
    OpenCLIPPytorchModel,
    PyContrastPytorchModel,
    PytorchModel,
    QwenVLPytorchModel,
    SwagPytorchModel,
    TwoStreamDVDModel,
    ViTPytorchModel,
)

# QwenVLPytorchMode #, BlipV2Model #, #LlavaVLPytorchModel


_PYTORCH_IMAGE_MODELS = "rwightman/pytorch-image-models"

_EFFICIENTNET_MODELS = "rwightman/gen-efficientnet-pytorch"

# custom models
# SSL
# -------------------------------------------------------------------------
# List SSL model names here
# -------------------------------------------------------------------------
SSL_MODEL_NAMES = [
    # miniecoset
    "resnet50_finetuned_DVD_B_SimCLR_miniecoset_ms08_dvd_data_aug_seed1",
    "resnet50_finetuned_adult_SimCLR_miniecoset_ms08_dvd_data_aug_seed1",
    # ecoset
    "resnet50_finetuned_DVD_B_SimCLR_ecoset_ms08_dvd_data_aug_seed1",
    "resnet50_finetuned_adult_SimCLR_ecoset_ms08_dvd_data_aug_seed1",
    # imagenet-1k
    "resnet50_finetuned_DVD_B_SimCLR_imagenet_1k_ms08_dvd_data_aug_seed1",
    "resnet50_finetuned_adult_SimCLR_imagenet_1k_dvd_data_aug_seed1",
    # BYOL, DINO
    "resnet50_byol_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1",
    "resnet50_dino_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1",
    "resnet50_dino_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1",
    "resnet50_byol_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1",
    "resnet50_dino_lin_eval_DVD_B_imagenet_ms08_dvd_data_aug_seed1",
    "resnet50_byol_lin_eval_DVD_B_magenet_ms08_dvd_data_aug_seed1",
]

# -------------------------------------------------------------------------
# Auto-register each model by creating a function dynamically
# -------------------------------------------------------------------------

for model_name in SSL_MODEL_NAMES:
    # Import registration and loader once
    exec(f"""
@register_model("pytorch")
def {model_name}(model_name_str, *args, **kwargs):
    model = tsm.load_model(model_name_str, *args, **kwargs)
    return PytorchModel(model, model_name_str, *args)
""")


def model_pytorch(model_name, *args, **kwargs):
    import torchvision.models as zoomodels

    model = zoomodels.__dict__[model_name](pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


# -- THE MODELS FOR THE TWO STREAM DVD --


@register_model("pytorch")
def ts_final_ss_resnet_50(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_final_resnet_18(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_baseline_resnet50_DVD_Scale_Free_DVD_trans(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dvd_final(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_base(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_blur_final(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_contrast_final(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_blur_cs_medium(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_blur_cs_high(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_blur_cs_medium_s1_90(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_blur_cs_high_s1_90(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_blur_cs_dvd_trans(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_blur_cs_s1_90(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_blur_cs_concat(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_blur_cs_concat_s1_90(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_lum_blur_cs_final(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


@register_model("pytorch")
def ts_resnet18_dkl_rg_dvd_trans(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return TwoStreamDVDModel(model, model_name, *args)


# -- END


@register_model("pytorch")
def resnet50_trained_on_SIN(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN_and_IN(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN(
    model_name, *args, **kwargs
):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def swin_b_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg16_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet101_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet18_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vit_b_16_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def mobilenet_v3_small_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def shufflenet_v2_x1_0_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnext50_32x4d_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


# deit_base_patch16_224
@register_model("pytorch")
def deit_base_patch16_224_ours(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


## register DVD models for ecoset | Baseline, Balanced, Shape bias favored, Performance favored
@register_model("pytorch")
def resnet50_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B2(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B3(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_S(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_SS(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_S3(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_P(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_PP(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_P3(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_SimCLR_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_SimCLR_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


# * ablation models
# 'resnet50_DVD_B_blur_only', 'resnet50_DVD_B_color_only', 'resnet50_DVD_B_contrast_only', 'resnet50_DVD_B_blur_color_only', 'resnet50_DVD_B_blur_contrast_only', 'resnet50_DVD_B_color_contrast_only',
@register_model("pytorch")
def resnet50_DVD_B_blur_only(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B_color_only(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B_contrast_only(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B_blur_color_only(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B_blur_contrast_only(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B_color_contrast_only(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


## register DVD models for imagenet | Baseline, Balanced, Shape bias favored, Performance favored
# 'resnet50_baseline_imagenet', 'resnet50_DVD_B_imagenet', 'resnet50_DVD_B2_imagenet', 'resnet50_DVD_B3_imagenet', 'resnet50_DVD_S_imagenet', 'resnet50_DVD_SS_imagenet', 'resnet50_DVD_S3_imagenet', 'resnet50_DVD_P_imagenet', 'resnet50_DVD_PP_imagenet', 'resnet50_DVD_P3_imagenet'
@register_model("pytorch")
def resnet50_baseline_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B2_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B3_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_S_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_SS_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_S3_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_P_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_PP_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_P3_imagenet(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


## register DVD models for miniecoset | Baseline, Balanced, Shape bias favored, Performance favored
# 'resnet50_baseline_miniecoset', 'resnet50_DVD_B_miniecoset', 'resnet50_DVD_S_miniecoset', 'resnet50_DVD_SS_miniecoset', 'resnet50_DVD_P_miniecoset','resnet50_DVD_PP_miniecoset',
@register_model("pytorch")
def resnet50_baseline_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B2_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_B3_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_S_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_SS_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_S3_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_P_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_PP_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_DVD_P3_miniecoset(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


## register DVD models for different architectures | 'alexnet','vgg16', 'resnet101', 'mobilenet_v3_small', 'shufflenet_v2_x1_0', 'resnet18', 'resnext50_32x4d', 'deit_base_patch16_224', 'vit_b_16',
# 'alexnet_baseline', 'alexnet_DVD_B', 'vgg16_baseline', 'vgg16_DVD_B', 'resnet101_baseline', 'resnet101_DVD_B', 'mobilenet_v3_small_baseline', 'mobilenet_v3_small_DVD_B', 'shufflenet_v2_x1_0_baseline', 'shufflenet_v2_x1_0_DVD_B', 'resnet18_baseline', 'resnet18_DVD_B', 'resnext50_32x4d_baseline', 'resnext50_32x4d_DVD_B', 'deit_base_patch16_224_baseline', 'deit_base_patch16_224_DVD_B', 'vit_b_16_baseline', 'vit_b_16_DVD_B'
@register_model("pytorch")
def alexnet_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def alexnet_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg16_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vgg16_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet101_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet101_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def mobilenet_v3_small_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def mobilenet_v3_small_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def shufflenet_v2_x1_0_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def shufflenet_v2_x1_0_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet18_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet18_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnext50_32x4d_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnext50_32x4d_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def deit_base_patch16_224_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def deit_base_patch16_224_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vit_b_16_baseline(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def vit_b_16_DVD_B(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


# * sweeped miniecoset models
# @register_model("pytorch")
# def resnet50_mpe1.0_alpha0.1_dn50.0(model_name, *args, **kwargs):
#     from .shapenet import texture_shape_models as tsm
#     model = tsm.load_model(model_name, *args, **kwargs)
#     return PytorchModel(model, model_name, *args)
from itertools import product

# Define sweep parameters
months_per_epochs = [1.0, 2.0, 4.0, 8.0]
contrast_thresh = [0.1, 0.2, 0.4, 0.8]  # alpha
contrast_spd = [50.0, 100.0, 150.0]  # beta

# Generate and register models
for mpe, alpha, beta in product(
    months_per_epochs, contrast_thresh, contrast_spd
):

    # Import registration and loader
    model_name = f"resnet50_mpe{mpe}_alpha{alpha}_dn{beta}".replace(".", "_")

    exec(f"""
@register_model("pytorch")
def {model_name}(model_name_str, *args, **kwargs):
    model = tsm.load_model(model_name_str, *args, **kwargs)
    return PytorchModel(model, model_name_str, *args)
""")


#! adversarially robust models
# # extra: adversarially trained models
# 'resnet50_adult_adv_pgd_linf_eps0_001_k10':'resnet50_ecoset_square256_256_0.0001_dev_adult_seed_1_adv-pgd_linf_eps0.001_k10',
# 'resnet50_adult_adv_pgd_linf_eps0_001_k10_seed2':'resnet50_ecoset_square256_256_0.0001_dev_adult_seed_2_adv-pgd_linf_eps0.001_k10',
# 'resnet50_adult_adv_pgd_linf_eps0_001_k10_seed3':'resnet50_ecoset_square256_256_0.0001_dev_adult_seed_3_adv-pgd_linf_eps0.001_k10',
# 'resnet50_adult_adv_pgd_linf_eps0_0039_k10_seed_1':'resnet50_ecoset_square256_256_0.0001_dev_adult_seed_1_adv-pgd_linf_eps0.0039_k10',
# 'resnet50_adult_adv_pgd_linf_eps0_01568_k10_seed_1':'resnet50_ecoset_square256_256_0.0001_dev_adult_seed_1_adv-pgd_linf_eps0.01568_k10',


# 'resnet50_adult_adv_pgd_linf_eps0_001_k10', 'resnet50_adult_adv_pgd_linf_eps0_001_k10_seed2', 'resnet50_adult_adv_pgd_linf_eps0_001_k10_seed3', 'resnet50_adult_adv_pgd_linf_eps0_0039_k10_seed_1', 'resnet50_adult_adv_pgd_linf_eps0_01568_k10_seed_1'
@register_model("pytorch")
def resnet50_adult_adv_pgd_linf_eps0_001_k10(model_name, *args, **kwargs):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_adult_adv_pgd_linf_eps0_001_k10_seed2(
    model_name, *args, **kwargs
):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_adult_adv_pgd_linf_eps0_001_k10_seed3(
    model_name, *args, **kwargs
):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_adult_adv_pgd_linf_eps0_0039_k10_seed_1(
    model_name, *args, **kwargs
):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_adult_adv_pgd_linf_eps0_01568_k10_seed_1(
    model_name, *args, **kwargs
):
    from .shapenet import texture_shape_models as tsm

    model = tsm.load_model(model_name, *args, **kwargs)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet9(model_name, *args, **kwargs):
    from .bagnets.pytorchnet import bagnet9

    model = bagnet9(pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet17(model_name, *args, **kwargs):
    from .bagnets.pytorchnet import bagnet17

    model = bagnet17(pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def bagnet33(model_name, *args, **kwargs):
    from .bagnets.pytorchnet import bagnet33

    model = bagnet33(pretrained=True)
    model = torch.nn.DataParallel(model)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x1_supervised_baseline(model_name, *args, **kwargs):
    from .simclr import simclr_resnet50x1_supervised_baseline

    model = simclr_resnet50x1_supervised_baseline(
        pretrained=True, use_data_parallel=True
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x4_supervised_baseline(model_name, *args, **kwargs):
    from .simclr import simclr_resnet50x4_supervised_baseline

    model = simclr_resnet50x4_supervised_baseline(
        pretrained=True, use_data_parallel=True
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x1(model_name, *args, **kwargs):
    from .simclr import simclr_resnet50x1

    model = simclr_resnet50x1(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x2(model_name, *args, **kwargs):
    from .simclr import simclr_resnet50x2

    model = simclr_resnet50x2(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def simclr_resnet50x4(model_name, *args, **kwargs):
    from .simclr import simclr_resnet50x4

    model = simclr_resnet50x4(pretrained=True, use_data_parallel=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def InsDis(model_name, *args, **kwargs):
    from .pycontrast.pycontrast_resnet50 import InsDis

    model, classifier = InsDis(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def MoCo(model_name, *args, **kwargs):
    from .pycontrast.pycontrast_resnet50 import MoCo

    model, classifier = MoCo(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def MoCoV2(model_name, *args, **kwargs):
    from .pycontrast.pycontrast_resnet50 import MoCoV2

    model, classifier = MoCoV2(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def DINOv2(model_name, *args, **kwargs):
    from .pycontrast.pycontrast_resnet50 import DINOv2

    model, classifier = DINOv2(
        model_variant="dinov2_vitb14", pretrained=True, classes=1000
    )
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def PIRL(model_name, *args, **kwargs):
    from .pycontrast.pycontrast_resnet50 import PIRL

    model, classifier = PIRL(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def InfoMin(model_name, *args, **kwargs):
    from .pycontrast.pycontrast_resnet50 import InfoMin

    model, classifier = InfoMin(pretrained=True)
    return PyContrastPytorchModel(*(model, classifier), model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps0

    model = resnet50_l2_eps0()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_01(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_01

    model = resnet50_l2_eps0_01()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_03(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_03

    model = resnet50_l2_eps0_03()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_05(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_05

    model = resnet50_l2_eps0_05()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_1(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_1

    model = resnet50_l2_eps0_1()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_25(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_25

    model = resnet50_l2_eps0_25()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps0_5(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps0_5

    model = resnet50_l2_eps0_5()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps1(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps1

    model = resnet50_l2_eps1()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps3(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps3

    model = resnet50_l2_eps3()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_l2_eps5(model_name, *args, **kwargs):
    from .adversarially_robust.robust_models import resnet50_l2_eps5

    model = resnet50_l2_eps5()
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_b0(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_es(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_b0_noisy_student(model_name, *args, **kwargs):
    model = torch.hub.load(
        _PYTORCH_IMAGE_MODELS, "tf_efficientnet_b0_ns", pretrained=True
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def efficientnet_l2_noisy_student_475(model_name, *args, **kwargs):
    model = torch.hub.load(
        _EFFICIENTNET_MODELS, "tf_efficientnet_l2_ns_475", pretrained=True
    )
    return EfficientNetPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_B16_IN21K(model_name, *args, **kwargs):
    from pytorch_pretrained_vit import ViT

    model = ViT("B_16_imagenet1k", pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_B32_IN21K(model_name, *args, **kwargs):
    from pytorch_pretrained_vit import ViT

    model = ViT("B_32_imagenet1k", pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_L16_IN21K(model_name, *args, **kwargs):
    from pytorch_pretrained_vit import ViT

    model = ViT("L_16_imagenet1k", pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def transformer_L32_IN21K(model_name, *args, **kwargs):
    from pytorch_pretrained_vit import ViT

    model = ViT("L_32_imagenet1k", pretrained=True)
    return ViTPytorchModel(model, model_name, *args)


@register_model("pytorch")
def vit_small_patch16_224(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def vit_base_patch16_224(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def vit_large_patch16_224(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    img_size = (224, 224)
    return ViTPytorchModel(model, model_name, img_size, *args)


@register_model("pytorch")
def cspresnet50(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cspresnext50(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def cspdarknet53(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def darknet53(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn68(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn68b(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn92(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn98(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn131(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def dpn107(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18_small_v2(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w18(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w30(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w40(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w44(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w48(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def hrnet_w64(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls42(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls84(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls42b(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls60(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def selecsls60b(model_name, *args, **kwargs):
    model = torch.hub.load(_PYTORCH_IMAGE_MODELS, model_name, pretrained=True)
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def clip(model_name, *args, **kwargs):
    import clip

    model, _ = clip.load("ViT-B/32")
    return ClipPytorchModel(model, model_name, *args)


@register_model("pytorch")
def clipRN50(model_name, *args, **kwargs):
    import clip

    model, _ = clip.load("RN50")
    return ClipPytorchModel(model, model_name, *args)


# * VLMs
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


@register_model("pytorch")
def qwen2_vl_7B_instruct(model_name, *args, **kwargs):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        device_map="auto",
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),  # Use float16 for speed on GPU
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return QwenVLPytorchModel(model, processor, model_name, *args)


@register_model("pytorch")
def qwen2_vl_2B_instruct(model_name, *args, **kwargs):
    # This assumes "Qwen/Qwen2-VL-2B-Instruct" exists. Please verify.
    print(
        f"Attempting to load Qwen2-VL-2B-Instruct. Please verify model existence."
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        device_map="auto",
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return QwenVLPytorchModel(model, processor, model_name, *args)


@register_model("pytorch")
def qwen2_5_vl_7B_instruct(
    model_name: str, *args, **kwargs
) -> QwenVLPytorchModel:
    """
    Loads the Qwen2.5-VL-7B-Instruct model from Hugging Face.
    Note: You might need to install 'transformers' from source if you encounter
    'KeyError: qwen2_5_vl' for the latest models.
    pip install git+https://github.com/huggingface/transformers accelerate
    """
    print(f"Attempting to load Qwen2.5-VL-7B-Instruct model: {model_name}")

    # Qwen2.5 models might use a slightly different class name, verify on Hugging Face if issues occur
    # For now, assuming Qwen2VLForConditionalGeneration works for Qwen2.5-VL too as per some examples,
    # but the exact class might be Qwen2_5_VLForConditionalGeneration if it exists.
    # We will stick with Qwen2VLForConditionalGeneration for broader compatibility if it works.

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto",
        torch_dtype=torch_dtype,
        # attn_implementation="flash_attention_2",
        **kwargs,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    return QwenVLPytorchModel(model, processor, model_name, *args)


# * LLaVA
@register_model("pytorch")
def llava1_5_7b(model_name, *args, **kwargs):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    llava_model_id = "llava-hf/llava-1.5-7b-hf"  # Or another LLaVA model like "llava-hf/llava-1.5-13b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        llava_model_id,
        device_map="auto",
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),
        # load_in_8bit=True if torch.cuda.is_available() else False
    )
    processor = AutoProcessor.from_pretrained(llava_model_id)
    return LlavaPytorchModel(model, processor, model_name, *args)


# From your earlier snippet:
# neuroai/models/pytorch/model_zoo.py
@register_model("pytorch")
def llava_1_5_13b_hf(model_name, *args, **kwargs):
    """
    Registers and initializes a LLaVA v1.5-13b-hf model.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    llava_model_id = "llava-hf/llava-1.5-13b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        llava_model_id,
        device_map="auto",  # This is the key part
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),
    )
    processor = AutoProcessor.from_pretrained(llava_model_id)

    return LlavaPytorchModel(model, processor, model_name, *args, **kwargs)


@register_model("pytorch")
def blip2_opt_2_7b(model_name, *args, **kwargs):
    """
    Loads BLIP-2 with OPT-2.7B as the language model.
    """
    # Use 'auto' for device_map for better resource utilization if available,
    # otherwise default to 'cuda' or 'cpu'.
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),
        device_map="auto",  # Use "auto" to let transformers handle device placement
    )
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    return Blip2PytorchModel(model, processor, model_name, *args, **kwargs)


# --
@register_model("pytorch")
def clipRN50_CCM_12m_all_epochs(model_name, *args, **kwargs):
    import open_clip
    import torch

    # checkpoint_path = "/share/klab/vkapoor/models_clip_cc12m_openai_config/epoch=1670-step=5010.ckpt"
    checkpoint_path = "/share/klab/vkapoor/models_clip_cc12m_openai_config/epoch=4369-step=13107.ckpt"

    # Load model structure without weights
    model, _, preprocess_fn = open_clip.create_model_and_transforms(
        model_name="RN50",
        pretrained=None,  # Important: don't load default weights
        device="cuda",
    )
    tokenizer = open_clip.get_tokenizer("RN50")

    # Load your lightning-style checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Remove "model." prefix from all keys
    new_state_dict = {
        k[len("model.") :]: v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }

    # Load cleaned state dict into the OpenCLIP model
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print(f"[INFO] Missing keys: {missing}")
    print(f"[INFO] Unexpected keys: {unexpected}")

    # Wrap and return
    return OpenCLIPPytorchModel(
        model, tokenizer, preprocess_fn, model_name, *args
    )


@register_model("pytorch")
def clipRN50_CCM_12m_DVD(model_name, *args, **kwargs):
    import open_clip
    import torch

    checkpoint_path = "/share/klab/vkapoor/models_clip_cc12m_openai_config_evd/epoch=1670-step=5013.ckpt"

    # Load model structure without weights
    model, _, preprocess_fn = open_clip.create_model_and_transforms(
        model_name="RN50",
        pretrained=None,  # Important: don't load default weights
        device="cuda",
    )
    tokenizer = open_clip.get_tokenizer("RN50")

    # Load your lightning-style checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Remove "model." prefix from all keys
    new_state_dict = {
        k[len("model.") :]: v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }

    # Load cleaned state dict into the OpenCLIP model
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print(f"[INFO] Missing keys: {missing}")
    print(f"[INFO] Unexpected keys: {unexpected}")

    # Wrap and return
    return OpenCLIPPytorchModel(
        model, tokenizer, preprocess_fn, model_name, *args
    )


@register_model("pytorch")
def resnet50_swsl(model_name, *args, **kwargs):
    model = torch.hub.load(
        "facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_swsl"
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def ResNeXt101_32x16d_swsl(model_name, *args, **kwargs):
    model = torch.hub.load(
        "facebookresearch/semi-supervised-ImageNet1K-models",
        "resnext101_32x16d_swsl",
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_50x1(model_name, *args, **kwargs):
    model = torch.hub.load(
        _PYTORCH_IMAGE_MODELS, "resnetv2_50x1_bitm", pretrained=True
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_50x3(model_name, *args, **kwargs):
    model = torch.hub.load(
        _PYTORCH_IMAGE_MODELS, "resnetv2_50x3_bitm", pretrained=True
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_101x1(model_name, *args, **kwargs):
    model = torch.hub.load(
        _PYTORCH_IMAGE_MODELS, "resnetv2_101x1_bitm", pretrained=True
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_101x3(model_name, *args, **kwargs):
    model = torch.hub.load(
        _PYTORCH_IMAGE_MODELS, "resnetv2_101x3_bitm", pretrained=True
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_152x2(model_name, *args, **kwargs):
    model = torch.hub.load(
        _PYTORCH_IMAGE_MODELS, "resnetv2_152x2_bitm", pretrained=True
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def BiTM_resnetv2_152x4(model_name, *args, **kwargs):
    model = torch.hub.load(
        _PYTORCH_IMAGE_MODELS, "resnetv2_152x4_bitm", pretrained=True
    )
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_clip_hard_labels(model_name, *args, **kwargs):
    import torchvision.models as zoomodels

    model = zoomodels.__dict__["resnet50"](pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/bethgelab/model-vs-human/releases/download/v0.3"
        "/ResNet50_clip_hard_labels.pth",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["state_dict"])
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def resnet50_clip_soft_labels(model_name, *args, **kwargs):
    import torchvision.models as zoomodels

    model = zoomodels.__dict__["resnet50"](pretrained=False)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/bethgelab/model-vs-human/releases/download/v0.3"
        "/ResNet50_clip_soft_labels.pth",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["state_dict"])
    return PytorchModel(model, model_name, *args)


@register_model("pytorch")
def swag_regnety_16gf_in1k(model_name, *args, **kwargs):
    model = torch.hub.load("facebookresearch/swag", model="regnety_16gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_regnety_32gf_in1k(model_name, *args, **kwargs):
    model = torch.hub.load("facebookresearch/swag", model="regnety_32gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_regnety_128gf_in1k(model_name, *args, **kwargs):
    model = torch.hub.load("facebookresearch/swag", model="regnety_128gf_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_vit_b16_in1k(model_name, *args, **kwargs):
    model = torch.hub.load("facebookresearch/swag", model="vit_b16_in1k")
    return SwagPytorchModel(model, model_name, input_size=384, *args)


@register_model("pytorch")
def swag_vit_l16_in1k(model_name, *args, **kwargs):
    model = torch.hub.load("facebookresearch/swag", model="vit_l16_in1k")
    return SwagPytorchModel(model, model_name, input_size=512, *args)


@register_model("pytorch")
def swag_vit_h14_in1k(model_name, *args, **kwargs):
    model = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")
    return SwagPytorchModel(model, model_name, input_size=518, *args)
