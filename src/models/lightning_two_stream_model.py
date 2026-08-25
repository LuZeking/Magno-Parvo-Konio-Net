from typing import List

import kornia.filters as KF
import torch
import torch.nn as nn
from dvd_scale_free.dvd_scale_free.development import DVDConfig, DVDTransformer
from lightning_kietzmannlab import schedulers
from lightning_kietzmannlab.lightning_base_model import BaseModule
from torch.nn.modules.module import Module
from torchmetrics.classification import MultilabelAveragePrecision

from .pytorch_models import TwoStreamNetwork


class TwoStreamModule(BaseModule):
    """
    Args:

        # Args from Basemodule
        network (TwoStreamNetwork): The model.
        optim_func (Module): The optimizer
        loss_func (nn.Module, optional): The loss function (nn.CrossEntropyLoss)
        num_classes (int, optional): Number of classes (100)
        on_step (bool, optional): Perform training step (True)
        on_epoch (bool, optional): Perform training on epoch (True)
        frozen_layers (list, optional): List of frozen layers (None)
        scheduler (schedulers._Schedulers, optional): Learning rate scheduler (None)
        reduce_lr_metric (str, optional): Metric for reducing lr ("val_loss")
        automatic_optimization (bool, optional): Bool to define whether to use automatic optimization or not (True)
        sync_dist (bool, optional): Bool to define whether to synchronize distributed processes (True)
        rank_zero_only (bool, optional): Whether the value will be loggd only on rank 0 (False)
        custom_metrics (dict, optional): Dictionary of custom metrics to compute ([])
        loss_weight_decay (int, optional): Loss weight decay (0)
        multihot (bool, optional): Is the classification multihot (False)
        gpu_gaussian_blur (bool, optional): Perform gpu gaussian blur (False)
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
        len_train_loader (int, optional): Length of the train loader (1)
        age_m (int, optional): The specific age in months for constant transformations. (0)

        # dkl
        lum (bool, optional): The lum channel is active (False)
        rg (bool, optional): The red-green channel is active (False)
        by (bool, optional): The blue-yellow channel is active (False)
        lum_blur (bool, optional): Apply blurring to the lum channel (False)
        lum_cs (bool, optional): Apply contrast transformation to lum channel (False)
        rg_blur (bool, optional): Apply blurring to the rg channel (False)
        rg_cs (bool, optional): Apply contrast transformation to rg channel (False)
        by_blur (bool, optional): Apply blurring to the rg channel (False)
        by_cs (bool, optional): Apply contrast transformation to rg channel (False)
        combined_color (bool, optional): Combine the two color channels (False)
    """

    def __init__(
        self,
        network: TwoStreamNetwork,
        optim_func: Module,
        loss_func: nn.Module = nn.CrossEntropyLoss,
        num_classes: int = 100,
        on_step: bool = True,
        on_epoch: bool = True,
        frozen_layers: list = None,
        scheduler: schedulers._Schedulers = None,
        reduce_lr_metric="val_loss",
        automatic_optimization=True,
        sync_dist: bool = True,
        rank_zero_only: bool = False,
        custom_metrics: dict = None,
        loss_weight_decay: int = 0,
        multihot: bool = False,
        gpu_gaussian_blur: bool = False,
        apply_blur: int = 1,
        apply_color: int = 1,
        apply_contrast: int = 1,
        contrast_amplitude_beta: float = 0.1,
        contrast_amplitude_lamda: float = 150.0,
        apply_threshold_color: int = 0,
        image_size: int = None,
        cs_logspan_start: float = 5e-3,
        dvd_strategy: str = "dvd",
        fully_random: bool = False,
        age_months_curve: List[float] = None,
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
    ):
        super().__init__(
            network=network,
            optim_func=optim_func,
            loss_func=loss_func,
            num_classes=num_classes,
            on_step=on_step,
            on_epoch=on_epoch,
            frozen_layers=frozen_layers,
            scheduler=scheduler,
            reduce_lr_metric=reduce_lr_metric,
            automatic_optimization=automatic_optimization,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
            custom_metrics=custom_metrics,
            loss_weight_decay=loss_weight_decay,
            multihot=multihot,
            gpu_gaussian_blur=gpu_gaussian_blur,
        )

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
        self.age_m = age_m

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

    def forward(self, x_1, x_2=None):
        "Forward pass"
        return self.network(x_1, x_2)

    def _apply_dvd_augmentations(self, x, batch_idx):
        "Apply the blur, color and contrast augmentations"
        age_months = self.age_months_curve[
            self.trainer.current_epoch * self._len_train_dataloader + batch_idx
        ]
        dvdt = DVDTransformer(
            DVDConfig(
                apply_blur=self.apply_blur,
                apply_color=self.apply_color,
                apply_contrast=self.apply_contrast,
                cs_logspan_start=self.cs_logspan_start,
                apply_threshold_color=self.apply_threshold_color,
                image_size=self.image_size,
                fully_random=self.fully_random,
                age_months_curve=self.age_months_curve,
            )
        )
        img = dvdt(x, age_months)

        return img

    def _apply_gaussian_blur(self, x):
        blur_sigma = 4
        k = int(8 * blur_sigma)
        kernel_size = k + (1 if k % 2 == 0 else 0)  # 33
        x = KF.gaussian_blur2d(
            x,
            (kernel_size, kernel_size),
            (blur_sigma, blur_sigma),
            border_type="reflect",
        )
        return x

    def _apply_scale_free_contrast(self, x):
        dvdt = DVDTransformer(
            DVDConfig(
                apply_blur=0,
                apply_color=0,
                apply_contrast=1,
                cs_logspan_start=5e-3,
                image_size=256,
                decomposition_mode="keep",
            )
        )
        x = dvdt(x, self.age_m)
        return x

    def training_step(self, batch, batch_idx):
        "Custom training step for dvd transformations"
        self.to(self.device)

        if self.combined_color and not (self.rg and self.by):
            raise ValueError(
                "Combined color cannot be true if not both rg and by are true."
            )

        x, y = batch
        if self.dvd_strategy == "split_dvd":
            x_aug = self._apply_dvd_augmentations(x, batch_idx)
            y_hat = self(x, x_aug)

        elif self.dvd_strategy == "dvd":
            x = self._apply_dvd_augmentations(x, batch_idx)
            y_hat = self(x)

        elif self.dvd_strategy in  ["dkl", "dkl_dvd"]:
            
            if self.dvd_strategy == "dkl_dvd":
                if (self.lum_blur + self.lum_cs + self.rg_blur + self.rg_cs + self.by_blur + self.by_cs) > 0:
                    raise ValueError(
                    "The DKL DVD strategy handles the blurring and contrast augmentations. " \
                    "Please set the blur and contrast variables to False")
                x = self._apply_dvd_augmentations(x, batch_idx)

            # get the channels
            # magno/lum/dorsal
            x_1 = x[:, 0, :, :]
            # parvo/rg/ventral
            x_2 = x[:, 1, :, :]
            # konio/by
            x_3 = x[:, 2, :, :]

            # unsqueeze
            x_1 = x_1.unsqueeze(1)
            x_2 = x_2.unsqueeze(1)
            x_3 = x_3.unsqueeze(1)

            if self.lum:
                if self.lum_blur:
                    x_1 = self._apply_gaussian_blur(x_1)

                if self.lum_cs:
                    x_1 = self._apply_scale_free_contrast(x_1)

                if not self.rg and not self.by:
                    y_hat = self(x_1)

            if self.rg:
                if self.rg_blur:
                    x_2 = self._apply_gaussian_blur(x_2)

                if self.rg_cs:
                    x_2 = self._apply_scale_free_contrast(x_2)

                if not self.lum and not self.by:
                    y_hat = self(x_2)

            if self.by:
                if self.by_blur:
                    x_3 = self._apply_gaussian_blur(x_3)

                if self.by_cs:
                    x_3 = self._apply_scale_free_contrast(x_3)

                if not self.lum and not self.rg:
                    y_hat = self(x_3)

            if self.combined_color:
                x_combined = x[:, 1:3, :, :]
                x_combined[:, 0, :, :] = x_2.squeeze(1)
                x_combined[:, 1, :, :] = x_3.squeeze(1)

                if self.lum and not self.network.baseline:
                    y_hat = self(x_1, x_combined)
                else:
                    y_hat = self(x_combined)
            else:
                if self.lum and self.rg and not self.by:
                    y_hat = self(x_1, x_2)
                if self.lum and self.by and not self.rg:
                    y_hat = self(x_1, x_3)
                if self.rg and self.by and not self.lum:
                    y_hat = self(x_2, x_3)

        else:
            y_hat = self(x)

        loss = self.loss(y_hat, y).to(self.device)

        if self.loss_weight_decay > 0:
            loss = loss + self.regularization_loss()

        self.log_metrics(
            "train_loss",
            loss,
            on_step=self.on_step,
        )

        acc1, acc5 = self.compute_accuracy(y_hat, y)

        self.log_metrics("train_top1", acc1, on_step=self.on_step)
        if acc5 is not None:
            self.log_metrics("train_top5", acc5, on_step=self.on_step)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_metrics(
            "learning_rate",
            current_lr,
            on_step=self.on_step,
        )

        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        return loss

    def _shared_eval(self, batch, batch_idx, prefix):
        """Shared evaluation step.

        self:
            batch: Input batch.
            batch_idx: Index of the batch.
            prefix: Prefix for logging.
        """

        x, y = batch

        if self.dvd_strategy in ["dkl", "dkl_dvd"]:
            # get the magno and parvo channels

            # magno/lum/dorsal
            x_1 = x[:, 0, :, :]
            # parvo/rg/ventral
            x_2 = x[:, 1, :, :]
            # konio/by
            x_3 = x[:, 2, :, :]

            # unsqueeze
            x_1 = x_1.unsqueeze(1)
            x_2 = x_2.unsqueeze(1)
            x_3 = x_3.unsqueeze(1)

            if self.lum:
                if self.lum_blur:
                    x_1 = self._apply_gaussian_blur(x_1)

                if self.lum_cs:
                    x_1 = self._apply_scale_free_contrast(x_1)

                if not self.rg and not self.by:
                    y_hat = self(x_1)

            if self.rg:
                if self.rg_blur:
                    x_2 = self._apply_gaussian_blur(x_2)

                if self.rg_cs:
                    x_2 = self._apply_scale_free_contrast(x_2)

                if not self.lum and not self.by:
                    y_hat = self(x_2)

            if self.by:
                if self.by_blur:
                    x_3 = self._apply_gaussian_blur(x_3)

                if self.by_cs:
                    x_3 = self._apply_scale_free_contrast(x_3)

                if not self.lum and not self.rg:
                    y_hat = self(x_3)

            if self.combined_color:
                x_combined = x[:, 1:3, :, :]
                x_combined[:, 0, :, :] = x_2.squeeze(1)
                x_combined[:, 1, :, :] = x_3.squeeze(1)

                if self.lum and not self.network.baseline:
                    y_hat = self(x_1, x_combined)
                else:
                    y_hat = self(x_combined)
            else:
                if self.lum and self.rg and not self.by:
                    y_hat = self(x_1, x_2)
                if self.lum and self.by and not self.rg:
                    y_hat = self(x_1, x_3)
                if self.rg and self.by and not self.lum:
                    y_hat = self(x_2, x_3)

        else:
            y_hat = self(x)

        loss = self.loss(y_hat, y).to(self.device)
        self.log_metrics(f"{prefix}_loss", loss)

        acc1, acc5 = self.compute_accuracy(y_hat, y)

        self.log_metrics(f"{prefix}_top1", acc1)
        if acc5 is not None:
            self.log_metrics(f"{prefix}_top5", acc5)

        if self.multihot:
            metric = MultilabelAveragePrecision(num_labels=self.num_classes)
            y_thresh = (y > 0.5).long()
            y_prob = torch.sigmoid(y_hat)
            meanap = metric(y_prob, y_thresh)
            self.log_metrics(f"{prefix}_mean_av_precision", meanap)

        if self.custom_metrics:
            for metric_name, metric_instance in self.custom_metrics.items():
                custom_value = metric_instance(y_hat, y)
                self.log_metrics(
                    f"{prefix}_{metric_name}",
                    custom_value,
                )
