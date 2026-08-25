from typing import List

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from dvd_scale_free.datasets.dataset_loader import KorniaTransform
from dvd_scale_free.dvd_scale_free.development import generate_age_months_curve
from lightning_kietzmannlab.lightning_trainer import TrainingPipeline
from lightning_kietzmannlab.pytorch_datasets import (
    GenericDataModule,
    H5dataset,
)

from models.lightning_two_stream_model import TwoStreamModule
from models.pytorch_models import TwoStreamNetwork


class TwoStreamTrainingPipeline(TrainingPipeline):
    """
    Training pipeline for a two stream approach inhereting from TrainingPipeline.

    Args:

        # Args from TrainingPipeline
        num_workers (int, optional): Number of workers for data loading (8)
        epochs (int, optional): Number of training epochs (100)
        dataset_name (str, optional): Name of dataset ("miniecoset")
        dataset_dir (str, optional): Directory where the dataset is ("/share/klab/datasets/")
        log_path (str, optional): Path for logs ("LOG_PATH")
        in_memory (bool, optional): Whether to load the dataset in memory. (False)
        batch_size_train (int, optional): Batch size for training (64)
        batch_size_val_test (int, optional): Batch size for validation and testing (64)
        devices (int, optional): # Devices (1)
        augment_train (list, optional): Augmentations for training ([])
        augment_test_val (list, optional): Augmentations for testing and validation ([])
        loss_function (nn.Module, optional): Loss Function (torch.nn.CrossEntropyLoss)
        scheduler (str, optional): Learning rate scheduler (None)
        experiment_name (str, optional): Specified name for the running experiment ("two_stream_miniecoset")
        learning_rate (float, optional): Learning rate in the beginning of the training (0.001)
        momentum (float, optional): Strength of momentum for optimiter (0.9)
        decay (float, optional): Weight decay parameter for optimizer (1e-4)
        epsilon (float, optional: Term added to the denominator in optimizer (1e-8).
        slurm_auto_requeue (bool, optional): Whether to automatically requeue on SLURM (True)
        train_precision (str, optional): Level of precision in training ("32")
        gradient_clip_val (float, optional): Maximum norm for gradient clipping (None),
        gradient_clip_algorithm (float, optional): Gradient clipping algorithm (None),
        strategy (str, optional): Distributed training strategy ("auto")

        # Args for TwoStreamNetwork
        num_classes (int): Number of classification categories (100)
        dropout (float, optional): Value for the dropout (0.25)
        like_resnet (str, optional): Specification if the network should be put together directly from Resnet blocks ("resnet34")
        seperate_from_layer (int): The layer at which the two streams seperate (0)
        baseline (bool): Whether to use the baseline architecture (False)
        apply_blur (int, optional): Apply gaussian blur dvd transformation (1)
        apply_color (int, optional): Apply color dvd transformation (1)
        apply_contrast (int, optional): Apply contrast dvd transformation (1)
        contrast_amplitude_beta (float, optional): Magnitude of contrast beta (0.1)
        contrast_amplitude_lamda (float, optional): Magnitude of contrast lambda (150.0)
        cs_logspan_start(float, optional): Sets the minimum starting point of the log-scale range (5e-3)
        apply_threshold_color (int, optional): Threshold when to apply color (0)
        image_size (int, optional): Size of img (None)
        dvd_strategy (str, optional): The dvd strategy used ("dvd")
        fully_random (bool, optional): Randomize the application of the transformations (False)
        age_months_curve (List, optional): The generated age months curve needed for image transformation in training (None)
        months_per_epoch (int, optional): Number of months an epoch represents (2)
        len_train_loader (int, optional): Length of the train loader (1)
        age_m (int, optional): The specific age in months for constant transformations. (0)
        stream_concat (bool, optional): Share information in between the streams in a 3:1 ratio; 3 parts own stream and 1 part other stream (False)
        stream1_weight (float, optional): Indicates the proportion of information coming from stream1 for the classification

        # dkl
        lum (bool, optional): The lum channel is active (False)
        rg (bool, optional): The red-green channel is active (False)
        by (bool, optional): The blue-yellow channel is active (False)
        lum_blur (bool, optional): Apply blurring to the lum channel (False)
        lum_cs (bool, optional): Apply contrast transformation to lum channel (False)
        rg_blur (bool, optional): Apply blurring to the rg channel (False)
        rg_cs (bool, optional): Apply contrast transformation to rg channel (False)
        by_blur (bool, optional): Apply blurring to the by channel (False)
        by_cs (bool, optional): Apply contrast transformation to rg channel (False)
        combined_color (bool, optional): Combine the two color channels (False)

    """

    def __init__(
        self,
        num_workers: int = 8,
        epochs: int = 100,
        dataset_name: str = "texture2shape_miniecoset_256px.h5",
        dataset_dir: str = "/share/klab/datasets/",
        log_path: str = "/share/klab/lzejin/mbenito/two_stream_dvd",
        in_memory: bool = False,
        batch_size_train: int = 64,
        batch_size_val_test: int = 64,
        devices: int = 1,
        augment_train: list = [],
        augment_test_val: list = [],
        loss_function: nn.Module = nn.CrossEntropyLoss,
        scheduler: str = None,
        experiment_name: str = "two_stream_miniecoset",
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-4,
        epsilon: float = 1e-8,
        slurm_auto_requeue: bool = True,
        train_precision: str = "32",
        gradient_clip_val: float = None,
        gradient_clip_algorithm: str = None,
        strategy: str = "auto",
        num_classes: int = 100,
        dropout: float = 0.25,
        like_resnet: str = "resnet34",
        seperate_from_layer: int = 0,
        baseline: bool = False,
        apply_blur: int = 1,
        apply_color: int = 1,
        apply_contrast: int = 1,
        contrast_amplitude_beta: float = 0.1,
        contrast_amplitude_lamda: float = 150.0,
        cs_logspan_start: float = 5e-3,
        apply_threshold_color: int = 0,
        image_size: int = None,
        dvd_strategy: str = "dvd",
        fully_random: bool = False,
        age_months_curve: List[float] = None,
        months_per_epoch: int = 2,
        len_train_loader: int = 1,
        age_m: int = 0,
        lum: bool = False,
        rg: bool = False,
        by: bool = False,
        lum_blur: bool = False,
        lum_cs: bool = False,
        rg_blur: bool = False,
        rg_cs: bool = False,
        by_blur: bool = False,
        by_cs: bool = False,
        combined_color: bool = False,
        stream_concat: bool = False,
        stream1_weight: float = 0.5,
    ):
        super().__init__(
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
            scheduler=scheduler,
            experiment_name=experiment_name,
            learning_rate=learning_rate,
            momentum=momentum,
            decay=decay,
            epsilon=epsilon,
            slurm_auto_requeue=slurm_auto_requeue,
            train_precision=train_precision,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            strategy=strategy,
        )

        self.num_classes = num_classes
        self.dropout = dropout
        self.like_resnet = like_resnet
        self.seperate_from_layer = seperate_from_layer
        self.baseline = baseline

        self._apply_blur = apply_blur
        self._apply_color = apply_color
        self._apply_contrast = apply_contrast
        self._contrast_amplitude_beta = contrast_amplitude_beta  # old dvd
        self._contrast_amplitude_lamda = contrast_amplitude_lamda  # old dvd
        self._apply_threshold_color = apply_threshold_color
        self._image_size = image_size
        self._len_train_loader = len_train_loader
        self.cs_logspan_start = cs_logspan_start  # scale free dvd
        self.dvd_strategy = dvd_strategy
        self.fully_random = fully_random
        self.age_months_curve = age_months_curve
        self.months_per_epoch = months_per_epoch
        self.age_m = age_m
        self.stream_concat = stream_concat
        self.stream1_weight = stream1_weight

        self.lum = lum
        self.rg = rg
        self.by = by
        self.lum_blur = lum_blur
        self.lum_cs = lum_cs
        self.rg_blur = rg_blur
        self.rg_cs = rg_cs
        self.by_blur = by_blur
        self.by_cs = by_cs
        self.combined_color = combined_color

    def setup_two_stream_model(self):
        "Setup the two stream model given a specific configuration."

        self.model = TwoStreamNetwork(
            num_classes=self.num_classes,
            dropout=self.dropout,
            like_resnet=self.like_resnet,
            seperate_from_layer=self.seperate_from_layer,
            baseline=self.baseline,
            dvd_strategy=self.dvd_strategy,
            combined_color=self.combined_color,
            stream_concat=self.stream_concat,
            stream1_weight=self.stream1_weight,
        )
        model_hyperparameters = {
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "model_path": self.log_path,
            "model_name": self.experiment_name,
        }

        self._store_hyperparameters(model_hyperparameters)

    def setup_two_stream_lightning_module(self):
        "Setup the two stream lightning module. Call setup_two_stream_model() first to create the neccessary model."

        self.lightning_model = TwoStreamModule(
            network=self.model,
            optim_func=self.optimizer,
            loss_func=self.loss_function,
            num_classes=self.num_classes,
            apply_blur=self._apply_blur,
            apply_color=self._apply_color,
            apply_contrast=self._apply_contrast,
            contrast_amplitude_beta=self._contrast_amplitude_beta,
            contrast_amplitude_lamda=self._contrast_amplitude_lamda,
            apply_threshold_color=self._apply_threshold_color,
            image_size=self._image_size,
            cs_logspan_start=self.cs_logspan_start,
            len_train_loader=self._len_train_loader,
            dvd_strategy=self.dvd_strategy,
            fully_random=self.fully_random,
            age_months_curve=self.age_months_curve,
            age_m=self.age_m,
            lum=self.lum,
            rg=self.rg,
            by=self.by,
            lum_blur=self.lum_blur,
            lum_cs=self.lum_cs,
            rg_blur=self.rg_blur,
            rg_cs=self.rg_cs,
            by_blur=self.by_blur,
            by_cs=self.by_cs,
            combined_color=self.combined_color,
        )

    def setup_age_months_curve(self):
        "Setup the age month curve needed for DVD training"
        train_loader = self.data_module.train_dataloader()
        self._len_train_loader = len(train_loader)

        self.age_months_curve = generate_age_months_curve(
            self.epochs,
            self._len_train_loader,
            self.months_per_epoch,
        )

    def setup_datasets(
        self,
        image_dict: str = "data",
        label_dict="labels",
        num_classes=None,
        optional_dict=None,
        image_dtype=np.uint8,
        label_dtype=np.int64,
        normalize_embedding=False,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        multi_label=False,
        use_foveation: bool = False,
        out_size: float = 224,
        fov_min: float = 1,
        fov_max: float = 30,
        step: float = 0.2,
        use_simple_val: bool = False,
    ):
        self.label_dict = label_dict
        self.dataset_train = H5dataset(
            "train",
            self.dataset_path,
            transform=(
                self.train_transformations
                if self.train_transformations is not None
                else None
            ),
            image_dict=image_dict,
            label_dict=label_dict,
            label_dtype=label_dtype,
            optional_dict=optional_dict,
            in_memory=self.in_memory,
            num_classes=num_classes,
            normalize_embedding=normalize_embedding,
            image_dtype=image_dtype,
        )
        self.dataset_val = H5dataset(
            "val",
            self.dataset_path,
            transform=(
                self.val_test_transformations
                if self.val_test_transformations is not None
                else None
            ),
            image_dict=image_dict,
            label_dict=label_dict,
            label_dtype=label_dtype,
            optional_dict=optional_dict,
            in_memory=self.in_memory,
            num_classes=num_classes,
            normalize_embedding=normalize_embedding,
            image_dtype=image_dtype,
        )
        self.dataset_test = H5dataset(
            "val",
            self.dataset_path,
            transform=(
                self.val_test_transformations
                if self.val_test_transformations is not None
                else None
            ),
            image_dict=image_dict,
            label_dict=label_dict,
            label_dtype=label_dtype,
            optional_dict=optional_dict,
            in_memory=self.in_memory,
            num_classes=num_classes,
            normalize_embedding=normalize_embedding,
            image_dtype=image_dtype,
        )

        self.dataset_train.open_on_first_read()
        self.dataset_val.open_on_first_read()
        self.dataset_test.open_on_first_read()
        self.num_classes = self.dataset_train.num_classes
        print(f"Number of classes: {self.num_classes}")
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.data_module = GenericDataModule(
            self.dataset_train,
            self.dataset_test,
            self.dataset_val,
            batch_size_train=self.batch_size_train,
            batch_size_val=self.batch_size_val_test,
            batch_size_test=self.batch_size_val_test,
            num_workers_train=self.num_workers,
            num_workers_val=self.num_workers,
            num_workers_test=self.num_workers,
            sampler=self.sampler,
            num_classes=self.num_classes,
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            multi_label=multi_label,
            use_foveation=use_foveation,
            out_size=out_size,
            fov_min=fov_min,
            fov_max=fov_max,
            step=step,
            use_simple_val=use_simple_val,
            prefetch_factor_train=2,
        )

    def setup_preset_twostream_transforms(
        self,
        hflip_prob=0.25,
        rotation_prob: float = 0.25,
        rotation_degrees: float = 15.0,
        brightness_prob: float = 0.5,
        grayscale_prob: float = 0.5,
        equalize_prob: float = 0.5,
        distortion_scale: float = 0.5,
        perspective_prob: float = 0.5,
        sharpness_prob: float = 0.5,
        rand_blur_prob: float = 0.5,
    ):

        # use the DVD supervised transforms
        train_trans = []
        train_trans.append(T.ConvertImageDtype(torch.float))
        train_trans.append(K.RandomHorizontalFlip(p=hflip_prob))
        train_trans.append(
            K.RandomRotation(degrees=rotation_degrees, p=rotation_prob)
        )
        train_trans.append(
            K.RandomBrightness(brightness=(0.8, 1.2), p=brightness_prob)
        )
        if self.dvd_strategy not in ["dkl", "dkl_dvd"]:
            train_trans.append(K.RandomGrayscale(p=grayscale_prob))
        train_trans.append(K.RandomEqualize(p=equalize_prob))
        train_trans.append(
            K.RandomPerspective(
                distortion_scale=distortion_scale, p=perspective_prob
            )
        )
        train_trans.append(K.RandomSharpness(p=sharpness_prob))
        train_trans.append(
            K.RandomGaussianBlur(
                kernel_size=(5, 5), sigma=(0.1, 2.0), p=rand_blur_prob
            )
        )

        trans = nn.Sequential(*train_trans)
        self.train_transformations = KorniaTransform(trans, "0-1", None)

        # apply nothing except the type conversion
        val_trans = []
        val_trans.append(T.ConvertImageDtype(torch.float))

        trans = nn.Sequential(*val_trans)
        self.val_test_transformations = KorniaTransform(trans, "0-1", None)
