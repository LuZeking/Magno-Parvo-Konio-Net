import json
from pathlib import Path

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from lightning_kietzmannlab.pytorch_callbacks import (
    CheckpointModel,
    CustomProgressBar,
    EpochDurationCallback,
)
from lightning_kietzmannlab.pytorch_loggers import CustomNPZLogger
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from src.training.lightning_trainer import TwoStreamTrainingPipeline

from .scenario_train_dvd_supervised import TwoStreamDVD


def save_config_as_json(config, log_path):
    "Save config utilizing OmegaConf"
    conf = OmegaConf.to_container(config, resolve=True)

    Path(log_path).mkdir(exist_ok=True)
    file = Path(log_path) / "training_conf.json"
    with open(file, "w") as f:
        json.dump(conf, f, indent=2)

    print(f"Conf saved to: {file}")

    return conf


configstore = ConfigStore.instance()
configstore.store(name="TwoStreamDVD", node=TwoStreamDVD)


@hydra.main(
    config_path="../scripts/conf",
    config_name="scenario_train_dvd_supervised",
)
def main(config: TwoStreamDVD):
    log_path = config.train_data_paths.log_path
    save_config_as_json(config, log_path)

    slurm_auto_requeue = config.parameters.slurm_auto_requeue
    epochs = config.parameters.epochs
    batch_size_train = config.parameters.batch_size_train
    batch_size_val_test = config.parameters.batch_size_val_test
    learning_rate = config.parameters.learning_rate
    devices = config.parameters.devices
    num_workers = config.parameters.num_workers
    in_memory = config.parameters.in_memory
    num_classes = config.parameters.num_classes
    decay = config.parameters.decay
    momentum = config.parameters.momentum
    augment_train = config.parameters.augment_train
    augment_test_val = config.parameters.augment_test_val
    epsilon = config.parameters.epsilon
    dropout = config.parameters.dropout
    like_resnet = config.parameters.like_resnet
    seperate_from_layer = config.parameters.seperate_from_layer
    baseline = config.parameters.baseline
    label_smoothing = config.parameters.label_smoothing
    hflip_prob = config.parameters.hflip_prob
    rotation_prob = config.parameters.rotation_prob
    rotation_degrees = config.parameters.rotation_degrees
    brightness_prob = config.parameters.brightness_prob
    grayscale_prob = config.parameters.grayscale_prob
    equalize_prob = config.parameters.equalize_prob
    distortion_scale = config.parameters.distortion_scale
    perspective_prob = config.parameters.perspective_prob
    sharpness_prob = config.parameters.sharpness_prob
    rand_blur_prob = config.parameters.rand_blur_prob
    apply_blur = config.parameters.apply_blur
    apply_color = config.parameters.apply_color
    apply_contrast = config.parameters.apply_contrast
    contrast_amplitude_beta = config.parameters.contrast_amplitude_beta
    contrast_amplitude_lamda = config.parameters.contrast_amplitude_lamda
    apply_threshold_color = config.parameters.apply_threshold_color
    image_size = config.parameters.image_size
    cs_logspan_start = config.parameters.cs_logspan_start
    dvd_strategy = config.parameters.dvd_strategy
    fully_random = config.parameters.fully_random
    months_per_epoch = config.parameters.months_per_epoch
    age_m = config.parameters.age_m
    stream_concat = config.parameters.stream_concat
    stream1_weight = config.parameters.stream1_weight
    lum = config.parameters.lum
    rg = config.parameters.rg
    by = config.parameters.by
    lum_blur = config.parameters.lum_blur
    lum_cs = config.parameters.lum_cs
    rg_blur = config.parameters.rg_blur
    rg_cs = config.parameters.rg_cs
    by_blur = config.parameters.by_blur
    by_cs = config.parameters.by_cs
    combined_color = config.parameters.combined_color

    dataset_name = config.train_data_paths.dataset_name
    dataset_dir = config.train_data_paths.dataset_dir
    experiment_name = config.train_data_paths.experiment_name

    loss_function = hydra.utils.instantiate(
        config.parameters.loss_func, label_smoothing=label_smoothing
    )

    progress = CustomProgressBar()
    modelcheckpoint = CheckpointModel(save_dir=log_path)
    epoch_duration = EpochDurationCallback()
    callbacks = [
        progress.progress_bar,
        modelcheckpoint.checkpoint_callback,
        epoch_duration,
    ]
    npz_logger = CustomNPZLogger(
        save_dir=log_path, experiment_name=experiment_name
    )
    logger = [npz_logger]

    seed_everything(42, workers=True)
    train_pipeline = TwoStreamTrainingPipeline(
        num_workers=num_workers,
        epochs=epochs,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        log_path=log_path,
        in_memory=in_memory,
        batch_size_train=batch_size_train,
        batch_size_val_test=batch_size_val_test,
        devices=devices,
        augment_train=augment_train,
        augment_test_val=augment_test_val,
        loss_function=loss_function,
        experiment_name=experiment_name,
        learning_rate=learning_rate,
        momentum=momentum,
        decay=decay,
        epsilon=epsilon,
        slurm_auto_requeue=slurm_auto_requeue,
        num_classes=num_classes,
        dropout=dropout,
        like_resnet=like_resnet,
        seperate_from_layer=seperate_from_layer,
        baseline=baseline,
        apply_blur=apply_blur,
        apply_color=apply_color,
        apply_contrast=apply_contrast,
        contrast_amplitude_beta=contrast_amplitude_beta,
        contrast_amplitude_lamda=contrast_amplitude_lamda,
        apply_threshold_color=apply_threshold_color,
        image_size=image_size,
        cs_logspan_start=cs_logspan_start,
        dvd_strategy=dvd_strategy,
        fully_random=fully_random,
        months_per_epoch=months_per_epoch,
        age_m=age_m,
        stream_concat=stream_concat,
        stream1_weight=stream1_weight,
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

    train_pipeline.setup_preset_twostream_transforms(
        hflip_prob=hflip_prob,
        rotation_prob=rotation_prob,
        rotation_degrees=rotation_degrees,
        brightness_prob=brightness_prob,
        grayscale_prob=grayscale_prob,
        equalize_prob=equalize_prob,
        distortion_scale=distortion_scale,
        perspective_prob=perspective_prob,
        sharpness_prob=sharpness_prob,
        rand_blur_prob=rand_blur_prob,
    )

    if dataset_name in [
        "texture2shape_miniecoset_256px.h5",
        "texture2shape_miniecoset_256px_dkl_kapoor.h5",
    ]:
        train_pipeline.setup_datasets(num_classes=num_classes)
    else:
        train_pipeline.setup_datasets(
            num_classes=num_classes, image_dtype=np.float32
        )

    train_pipeline.setup_age_months_curve()

    train_pipeline.setup_two_stream_model()

    train_pipeline.setup_adam()

    train_pipeline.setup_two_stream_lightning_module()

    train_pipeline.train(logger=logger, callbacks=callbacks)


if __name__ == "__main__":
    main()
