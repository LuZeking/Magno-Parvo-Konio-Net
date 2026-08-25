import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from hydra.core.config_store import ConfigStore
from lightning_kietzmannlab.pytorch_datasets import H5dataset
from omegaconf import OmegaConf
from torchmetrics.classification import Accuracy

from src.analysis.two_stream_rdm import load_model
from src.models.pytorch_models import LinearReadoutModel
from src.utils.helpers import apply_gaussian_blur, apply_scale_free_contrast

from .scenario_linear_readout import LinearReadout


def get_transforms():
    transf = T.Compose(
        [
            T.Resize(256, antialias=True),
            T.CenterCrop(256),
            T.ConvertImageDtype(torch.float),
        ]
    )

    return transf


def get_linear_models(trained_model, num_classes, device, stream_concat):
    # get the output dimensionality of the streams
    if trained_model.dvd_strategy in ["dkl", "dkl_dvd"]:
        rand_input = torch.randn(256, 1, 256, 256)
    else:
        rand_input = torch.randn(256, 3, 256, 256)

    rand_input = rand_input.to(device)
    output = trained_model.stream1_block1(rand_input)
    if stream_concat:
        output = trained_model.stream1_block2(output)
    output_dim = output.shape[1] * output.shape[2] * output.shape[3]

    # create the two readout models
    linear_model_1 = LinearReadoutModel(
        input_dim=output_dim,
        output_dim=num_classes,
    )
    linear_model_2 = LinearReadoutModel(
        input_dim=output_dim,
        output_dim=num_classes,
    )

    linear_model_1.to(device)
    linear_model_2.to(device)

    return linear_model_1, linear_model_2


def load_data(
    dataset_path,
    transforms,
    num_classes=112,
    batch_size=256,
    num_workers=8,
    dvd_strategy="dkl",
):
    train_dataset = H5dataset(
        "train",
        dataset_path,
        transform=transforms,
        image_dict="data",
        label_dict="labels",
        label_dtype=np.int64,
        optional_dict=None,
        in_memory=False,
        num_classes=num_classes,
        normalize_embedding=False,
        image_dtype=(
            np.float32
            if dataset_path
            in ["/share/klab/datasets/texture2shape_miniecoset_256px_dkl.h5"]
            else np.uint8
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2,
    )

    val_dataset = H5dataset(
        "val",
        dataset_path,
        transform=transforms,
        image_dict="data",
        label_dict="labels",
        label_dtype=np.int64,
        optional_dict=None,
        in_memory=False,
        num_classes=num_classes,
        normalize_embedding=False,
        image_dtype=(
            np.float32
            if dataset_path
            in ["/share/klab/datasets/texture2shape_miniecoset_256px_dkl.h5"]
            else np.uint8
        ),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
    )

    return train_dataset, train_loader, val_dataset, val_loader


def get_loss_optim(
    model, learning_rate, weight_decay, epsilon, label_smoothing
):
    loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=epsilon,
    )

    return loss, optimizer


def compute_accuracy(y_hat, y, num_classes, device):
    if y.ndim == 2:
        y = y.argmax(dim=1)

    top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(
        device
    )

    top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(
        device
    )

    acc1 = top1(y_hat, y)
    acc5 = top5(y_hat, y)

    return acc1, acc5


def calculate_epoch_avg(step_accs, normal_batch_size, last_batch_size):
    last_el = step_accs.pop()
    numerator = (
        sum(acc * normal_batch_size for acc in step_accs)
        + last_el * last_batch_size
    )
    avg = numerator / (len(step_accs) * normal_batch_size + last_batch_size)
    return avg


def visualize_accuracy(
    stream1_train_epoch_acc1,
    stream1_train_epoch_acc5,
    stream2_train_epoch_acc1,
    stream2_train_epoch_acc5,
    stream1_val_epoch_acc1,
    stream1_val_epoch_acc5,
    stream2_val_epoch_acc1,
    stream2_val_epoch_acc5,
    experiment_name,
):
    print(f"stream1_train_epoch_acc1: {stream1_train_epoch_acc1}")
    print(f"stream1_train_epoch_acc5: {stream1_train_epoch_acc5}")
    print(f"stream2_train_epoch_acc1: {stream2_train_epoch_acc1}")
    print(f"stream2_train_epoch_acc5: {stream2_train_epoch_acc5}")
    print(f"stream1_val_epoch_acc1: {stream1_val_epoch_acc1}")
    print(f"stream1_val_epoch_acc5: {stream1_val_epoch_acc5}")
    print(f"stream2_val_epoch_acc1: {stream2_val_epoch_acc1}")
    print(f"stream2_val_epoch_acc5: {stream2_val_epoch_acc5}")

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].scatter(
        range(len(stream1_train_epoch_acc1)),
        stream1_train_epoch_acc1,
        label="Stream1",
    )
    axs[0, 0].scatter(
        range(len(stream2_train_epoch_acc1)),
        stream2_train_epoch_acc1,
        label="Stream2",
    )
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Average Accuracy")
    axs[0, 0].legend()
    axs[0, 0].set_title("Train top1-accuracy")

    axs[0, 1].scatter(
        range(len(stream1_val_epoch_acc1)),
        stream1_val_epoch_acc1,
        label="Stream1",
    )
    axs[0, 1].scatter(
        range(len(stream2_val_epoch_acc1)),
        stream2_val_epoch_acc1,
        label="Stream2",
    )
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Average Accuracy")
    axs[0, 1].legend()
    axs[0, 1].set_title("Val top1-accuracy")

    axs[1, 0].scatter(
        range(len(stream1_train_epoch_acc5)),
        stream1_train_epoch_acc5,
        label="Stream1",
    )
    axs[1, 0].scatter(
        range(len(stream2_train_epoch_acc5)),
        stream2_train_epoch_acc5,
        label="Stream2",
    )
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Average Accuracy")
    axs[1, 0].legend()
    axs[1, 0].set_title("Train top5-accuracy")

    axs[1, 1].scatter(
        range(len(stream1_val_epoch_acc5)),
        stream1_val_epoch_acc5,
        label="Stream1",
    )
    axs[1, 1].scatter(
        range(len(stream2_val_epoch_acc5)),
        stream2_val_epoch_acc5,
        label="Stream2",
    )
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Average Accuracy")
    axs[1, 1].legend()
    axs[1, 1].set_title("Val top5-accuracy")

    plt.savefig(
        f"/share/klab/lzejin/mbenito/two_stream_dvd/metric_images/linear_readout/{experiment_name}.png"
    )


def linear_readout(
    experiment_name,
    epochs,
    batch_size,
    trained_model,
    linear_model_1,
    linear_model_2,
    train_loader,
    val_loader,
    criterion1,
    optimizer1,
    criterion2,
    optimizer2,
    dvd_strategy,
    num_classes,
    device,
    age_m,
    stream_concat,
    lum,
    rg,
    by,
    lum_blur,
    lum_cs,
    rg_blur,
    rg_cs,
    by_blur,
    by_cs,
    combined_color,
):
    "Function to fit the linear models"
    stream1_train_epoch_acc1 = []
    stream1_train_epoch_acc5 = []
    stream2_train_epoch_acc1 = []
    stream2_train_epoch_acc5 = []

    stream1_val_epoch_acc1 = []
    stream1_val_epoch_acc5 = []
    stream2_val_epoch_acc1 = []
    stream2_val_epoch_acc5 = []

    for i in range(epochs):
        print(f"Epoch: {i}")
        stream1_train_step_acc1 = []
        stream1_train_step_acc5 = []
        stream2_train_step_acc1 = []
        stream2_train_step_acc5 = []
        # -- Train --
        for b_idx, batch in enumerate(train_loader):
            print(f"Batch: {b_idx+1}/ {len(train_loader)}")
            is_last = b_idx == len(train_loader) - 1
            x, y = batch
            x, y = x.to(device), y.to(device)
            # get the activations of the two streams
            with torch.no_grad():
                if dvd_strategy in ["dkl", "dkl_dvd"]:
                    # get the magno and parvo channels
                    if (lum + rg + by) < 2:
                        raise ValueError(
                            "At least two channels need to be active for the linear readout."
                        )

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

                    if lum:
                        if lum_blur:
                            x_1 = apply_gaussian_blur(x_1)

                        if lum_cs:
                            x_1 = apply_scale_free_contrast(x_1, age_m)

                    if rg:
                        if rg_blur:
                            x_2 = apply_gaussian_blur(x_2)

                        if rg_cs:
                            x_2 = apply_scale_free_contrast(x_2, age_m)

                    if by:
                        if by_blur:
                            x_3 = apply_gaussian_blur(x_3)

                        if by_cs:
                            x_3 = apply_scale_free_contrast(x_3, age_m)

                    if combined_color:
                        if (lum + rg + by) != 3:
                            raise ValueError(
                                "All three channels need to be active!"
                            )
                        x_combined = x[:, 1:3, :, :]
                        x_combined[:, 0, :, :] = x_2.squeeze(1)
                        x_combined[:, 1, :, :] = x_3.squeeze(1)

                        y_hat_s1 = trained_model.stream1_block1(x_1)
                        y_hat_s2 = trained_model.stream2_block1(x_combined)
                        if stream_concat:
                            y_hat_s1 = (3 * y_hat_s1 + y_hat_s2) / 4
                            y_hat_s2 = (3 * y_hat_s2 + y_hat_s1) / 4

                            y_hat_s1 = trained_model.stream1_block2(y_hat_s1)
                            y_hat_s2 = trained_model.stream2_block2(y_hat_s2)

                    else:
                        if lum and rg and not by:
                            s1_input = x_1
                            s2_input = x_2
                        elif lum and by and not rg:
                            s1_input = x_1
                            s2_input = x_3
                        elif rg and by and not lum:
                            s1_input = x_2
                            s2_input = x_3

                        y_hat_s1 = trained_model.stream1_block1(s1_input)
                        y_hat_s2 = trained_model.stream2_block1(s2_input)
                        if stream_concat:
                            y_hat_s1 = (3 * y_hat_s1 + y_hat_s2) / 4
                            y_hat_s2 = (3 * y_hat_s2 + y_hat_s1) / 4

                            y_hat_s1 = trained_model.stream1_block2(y_hat_s1)
                            y_hat_s2 = trained_model.stream2_block2(y_hat_s2)

                else:
                    if trained_model.shared_block is not None:
                        x = trained_model.shared_block(x)

                    y_hat_s1 = trained_model.stream1_block1(x)
                    y_hat_s2 = trained_model.stream2_block1(x)
                    if stream_concat:
                        y_hat_s1 = (3 * y_hat_s1 + y_hat_s2) / 4
                        y_hat_s2 = (3 * y_hat_s2 + y_hat_s1) / 4

                        y_hat_s1 = trained_model.stream1_block2(y_hat_s1)
                        y_hat_s2 = trained_model.stream2_block2(y_hat_s2)

            linear_model_1.train()
            linear_model_2.train()
            # Linear Model 1
            y_hat_s1 = linear_model_1(y_hat_s1)
            loss_1 = criterion1(y_hat_s1, y)

            s1_acc1, s1_acc5 = compute_accuracy(
                y_hat_s1, y, num_classes, device
            )
            print(f"Stream 1: acc1:{s1_acc1.item()}, acc5:{s1_acc5.item()}")
            stream1_train_step_acc1.append(s1_acc1.item())
            stream1_train_step_acc5.append(s1_acc5.item())

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()

            # Linear Model 2
            y_hat_s2 = linear_model_2(y_hat_s2)
            loss_2 = criterion2(y_hat_s2, y)

            s2_acc1, s2_acc5 = compute_accuracy(
                y_hat_s2, y, num_classes, device
            )
            print(f"Stream 2: acc1:{s2_acc1.item()}, acc5:{s2_acc5.item()}")
            stream2_train_step_acc1.append(s2_acc1.item())
            stream2_train_step_acc5.append(s2_acc5.item())

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            if is_last:
                avg = calculate_epoch_avg(
                    step_accs=stream1_train_step_acc1,
                    normal_batch_size=batch_size,
                    last_batch_size=x.shape[0],
                )
                stream1_train_epoch_acc1.append(avg)

                avg = calculate_epoch_avg(
                    step_accs=stream1_train_step_acc5,
                    normal_batch_size=batch_size,
                    last_batch_size=x.shape[0],
                )
                stream1_train_epoch_acc5.append(avg)

                avg = calculate_epoch_avg(
                    step_accs=stream2_train_step_acc1,
                    normal_batch_size=batch_size,
                    last_batch_size=x.shape[0],
                )
                stream2_train_epoch_acc1.append(avg)

                avg = calculate_epoch_avg(
                    step_accs=stream2_train_step_acc5,
                    normal_batch_size=batch_size,
                    last_batch_size=x.shape[0],
                )
                stream2_train_epoch_acc5.append(avg)

        # -- Val --
        stream1_val_step_acc1 = []
        stream1_val_step_acc5 = []
        stream2_val_step_acc1 = []
        stream2_val_step_acc5 = []
        for b_idx, batch in enumerate(val_loader):
            print(f"Batch: {b_idx+1}/ {len(val_loader)}")
            is_last = b_idx == len(val_loader) - 1
            x, y = batch
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                if dvd_strategy in ["dkl", "dkl_dvd"]:
                    # get the magno and parvo channels
                    if (lum + rg + by) < 2:
                        raise ValueError(
                            "At least two channels need to be active for the linear readout."
                        )

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

                    if lum:
                        if lum_blur:
                            x_1 = apply_gaussian_blur(x_1)

                        if lum_cs:
                            x_1 = apply_scale_free_contrast(x_1, age_m)

                    if rg:
                        if rg_blur:
                            x_2 = apply_gaussian_blur(x_2)

                        if rg_cs:
                            x_2 = apply_scale_free_contrast(x_2, age_m)

                    if by:
                        if by_blur:
                            x_3 = apply_gaussian_blur(x_3)

                        if by_cs:
                            x_3 = apply_scale_free_contrast(x_3, age_m)

                    if combined_color:
                        if (lum + rg + by) != 3:
                            raise ValueError(
                                "All three channels need to be active!"
                            )
                        x_combined = x[:, 1:3, :, :]
                        x_combined[:, 0, :, :] = x_2.squeeze(1)
                        x_combined[:, 1, :, :] = x_3.squeeze(1)

                        y_hat_s1 = trained_model.stream1_block1(x_1)
                        y_hat_s2 = trained_model.stream2_block1(x_combined)
                        if stream_concat:
                            y_hat_s1 = (3 * y_hat_s1 + y_hat_s2) / 4
                            y_hat_s2 = (3 * y_hat_s2 + y_hat_s1) / 4

                            y_hat_s1 = trained_model.stream1_block2(y_hat_s1)
                            y_hat_s2 = trained_model.stream2_block2(y_hat_s2)

                    else:
                        if lum and rg and not by:
                            s1_input = x_1
                            s2_input = x_2
                        elif lum and by and not rg:
                            s1_input = x_1
                            s2_input = x_3
                        elif rg and by and not lum:
                            s1_input = x_2
                            s2_input = x_3

                        y_hat_s1 = trained_model.stream1_block1(s1_input)
                        y_hat_s2 = trained_model.stream2_block1(s2_input)
                        if stream_concat:
                            y_hat_s1 = (3 * y_hat_s1 + y_hat_s2) / 4
                            y_hat_s2 = (3 * y_hat_s2 + y_hat_s1) / 4

                            y_hat_s1 = trained_model.stream1_block2(y_hat_s1)
                            y_hat_s2 = trained_model.stream2_block2(y_hat_s2)

                else:
                    y_hat_s1 = trained_model.stream1_block1(x)
                    y_hat_s2 = trained_model.stream2_block1(x)
                    if stream_concat:
                        y_hat_s1 = (3 * y_hat_s1 + y_hat_s2) / 4
                        y_hat_s2 = (3 * y_hat_s2 + y_hat_s1) / 4

                        y_hat_s1 = trained_model.stream1_block2(y_hat_s1)
                        y_hat_s2 = trained_model.stream2_block2(y_hat_s2)

            linear_model_1.eval()
            linear_model_2.eval()
            with torch.no_grad():
                y_hat_s1 = linear_model_1(y_hat_s1)

                s1_acc1, s1_acc5 = compute_accuracy(
                    y_hat_s1, y, num_classes, device
                )
                stream1_val_step_acc1.append(s1_acc1.item())
                stream1_val_step_acc5.append(s1_acc5.item())

                y_hat_s2 = linear_model_2(y_hat_s2)

                s2_acc1, s2_acc5 = compute_accuracy(
                    y_hat_s2, y, num_classes, device
                )
                stream2_val_step_acc1.append(s2_acc1.item())
                stream2_val_step_acc5.append(s2_acc5.item())

            if is_last:
                avg = calculate_epoch_avg(
                    step_accs=stream1_val_step_acc1,
                    normal_batch_size=batch_size,
                    last_batch_size=x.shape[0],
                )
                stream1_val_epoch_acc1.append(avg)

                avg = calculate_epoch_avg(
                    step_accs=stream1_val_step_acc5,
                    normal_batch_size=batch_size,
                    last_batch_size=x.shape[0],
                )
                stream1_val_epoch_acc5.append(avg)

                avg = calculate_epoch_avg(
                    step_accs=stream2_val_step_acc1,
                    normal_batch_size=batch_size,
                    last_batch_size=x.shape[0],
                )
                stream2_val_epoch_acc1.append(avg)

                avg = calculate_epoch_avg(
                    step_accs=stream2_val_step_acc5,
                    normal_batch_size=batch_size,
                    last_batch_size=x.shape[0],
                )
                stream2_val_epoch_acc5.append(avg)

    visualize_accuracy(
        stream1_train_epoch_acc1,
        stream1_train_epoch_acc5,
        stream2_train_epoch_acc1,
        stream2_train_epoch_acc5,
        stream1_val_epoch_acc1,
        stream1_val_epoch_acc5,
        stream2_val_epoch_acc1,
        stream2_val_epoch_acc5,
        experiment_name,
    )


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
configstore.store(name="LinearReadout", node=LinearReadout)


@hydra.main(
    config_path="../scripts/conf",
    config_name="scenario_linear_readout",
)
def main(config: LinearReadout):
    # hyperparameters
    num_classes = config.parameters.num_classes
    batch_size = config.parameters.batch_size
    num_workers = config.parameters.num_workers
    epochs = config.parameters.epochs
    learning_rate = config.parameters.learning_rate
    weight_decay = config.parameters.weight_decay
    epsilon = config.parameters.epsilon
    label_smoothing = config.parameters.label_smoothing
    like_resnet = config.parameters.like_resnet
    seperate_from_layer = config.parameters.seperate_from_layer
    dvd_strategy = config.parameters.dvd_strategy
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

    experiment_name = config.train_data_paths.experiment_name
    dataset_dir = config.train_data_paths.dataset_dir
    dataset_name = config.train_data_paths.dataset_name
    log_path = config.train_data_paths.log_path

    save_config_as_json(config, log_path)

    # -- Load the Model --
    trained_model, _, device = load_model(
        num_classes=num_classes,
        like_resnet=like_resnet,
        baseline=False,
        seperate_from_layer=seperate_from_layer,
        experiment_name=experiment_name,
        dvd_strategy=dvd_strategy,
        combined_color=combined_color,
        stream_concat=stream_concat,
        stream1_weight=stream1_weight,
    )

    transforms = get_transforms()

    # -- Set the model to eval mode & freeze all its layers --
    trained_model.eval()
    for param in trained_model.parameters():
        param.requires_grad = False

    # -- Create the linear readout models --
    linear_model_1, linear_model_2 = get_linear_models(
        trained_model, num_classes, device, stream_concat
    )

    # -- Load the data --
    train_dataset, train_loader, val_dataset, val_loader = load_data(
        dataset_path=dataset_dir + dataset_name,
        transforms=transforms,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        dvd_strategy=dvd_strategy,
    )

    # -- Get the loss & optimizer --
    criterion1, optimizer1 = get_loss_optim(
        model=linear_model_1,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epsilon=epsilon,
        label_smoothing=label_smoothing,
    )

    criterion2, optimizer2 = get_loss_optim(
        model=linear_model_2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epsilon=epsilon,
        label_smoothing=label_smoothing,
    )

    # -- Fit the linear Model --
    linear_readout(
        experiment_name=experiment_name,
        epochs=epochs,
        batch_size=batch_size,
        trained_model=trained_model,
        linear_model_1=linear_model_1,
        linear_model_2=linear_model_2,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion1=criterion1,
        optimizer1=optimizer1,
        criterion2=criterion2,
        optimizer2=optimizer2,
        dvd_strategy=dvd_strategy,
        num_classes=num_classes,
        device=device,
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


if __name__ == "__main__":
    main()
