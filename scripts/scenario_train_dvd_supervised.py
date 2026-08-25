from dataclasses import dataclass

import torch


@dataclass
class Params:
    slurm_auto_requeue: bool
    epochs: int
    batch_size_train: int
    batch_size_val_test: int
    learning_rate: float
    devices: int
    num_workers: int
    in_memory: bool
    num_classes: int
    loss_func: torch.nn.Module
    decay: float
    momentum: float
    label_smoothing: float
    hflip_prob: float
    rotation_prob: float
    rotation_degrees: float
    brightness_prob: float
    grayscale_prob: float
    equalize_prob: float
    distortion_scale: float
    perspective_prob: float
    sharpness_prob: float
    rand_blur_prob: float
    augment_train: list
    augment_test_val: list
    epsilon: float
    dropout: float
    like_resnet: str
    seperate_from_layer: int
    baseline: bool
    apply_blur: int
    apply_color: int
    apply_contrast: int
    contrast_amplitude_beta: float
    contrast_amplitude_lamda: float
    apply_threshold_color: int
    image_size: int
    cs_logspan_start: float
    dvd_strategy: str
    fully_random: bool
    months_per_epoch: int
    age_m: int
    stream_concat: bool
    stream1_weight: float
    lum: bool
    rg: bool
    by: bool
    lum_blur: bool
    lum_cs: bool
    rg_blur: bool
    rg_cs: bool
    by_blur: bool
    by_cs: bool
    combined_color: bool


@dataclass
class Train_Data_Paths:
    log_path: str
    dataset_dir: str
    dataset_name: str
    experiment_name: str


@dataclass
class TwoStreamDVD:
    parameters: Params
    train_data_paths: Train_Data_Paths
