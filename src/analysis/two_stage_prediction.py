import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from lightning_kietzmannlab.pytorch_datasets import H5dataset
from torchmetrics.classification import Accuracy

from scripts.linear_readout import calculate_epoch_avg
from src.analysis.two_stream_rdm import load_model
from src.models.pytorch_models import TwoStreamNetwork
from src.utils.helpers import apply_gaussian_blur, apply_scale_free_contrast


def get_transforms():
    transf = T.Compose(
        [
            T.Resize(256, antialias=True),
            T.CenterCrop(256),
            T.ConvertImageDtype(torch.float),
        ]
    )

    return transf


def load_data(
    dataset_path,
    transforms,
    num_classes=112,
    batch_size=256,
    num_workers=8,
    dvd_strategy="dkl",
):

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
        image_dtype=np.float64 if dvd_strategy in ["dkl", "dkl_dvd"] else np.uint8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
    )

    return val_dataset, val_loader


def run_batches(
    model1: TwoStreamNetwork,
    model2: TwoStreamNetwork,
    val_loader: torch.utils.data.DataLoader,
    topk: int = 10,
    device: str = "cuda",
    num_classes: int = 112,
    batch_size: int = 256,
    m1_age_m: int = 0,
    m1_lum: bool = True,
    m1_rg: bool = False,
    m1_by: bool = False,
    m1_blur: bool = False,
    m1_cs: bool = False,
    m1_combined_color: bool = False,
    m2_age_m: int = 0,
    m2_lum: bool = False,
    m2_rg: bool = True,
    m2_by: bool = False,
    m2_blur: bool = False,
    m2_cs: bool = False,
    m2_combined_color: bool = False,
):
    model2_acc1 = []
    for b_idx, batch in enumerate(val_loader):
        print(f"Batch: {b_idx+1}/ {len(val_loader)}")
        is_last = b_idx == len(val_loader) - 1

        x, y = batch
        x, y = x.to(device), y.to(device)

        # model 1 input
        if m1_combined_color:
            m1_input = x[:, 1:3, :, :]
        elif m1_lum:
            m1_input = x[:, 0, :, :].unsqueeze(1)
        elif m1_rg:
            m1_input = x[:, 1, :, :].unsqueeze(1)
        elif m1_by:
            m1_input = x[:, 2, :, :].unsqueeze(1)

        if m1_blur:
            m1_input = apply_gaussian_blur(m1_input)
        if m1_cs:
            m1_input = apply_scale_free_contrast(m1_input, m1_age_m)

        # model 2 input
        if m2_combined_color:
            m2_input = x[:, 1:3, :, :]
        elif m2_lum:
            m2_input = x[:, 0, :, :].unsqueeze(1)
        elif m2_rg:
            m2_input = x[:, 1, :, :].unsqueeze(1)
        elif m2_by:
            m2_input = x[:, 2, :, :].unsqueeze(1)

        if m2_blur:
            m2_input = apply_gaussian_blur(m2_input)
        if m2_cs:
            m2_input = apply_scale_free_contrast(m2_input, m2_age_m)

        # forward pass
        m1_logits = model1(m1_input)
        m2_logits = model2(m2_input)

        # get the topk indices from model1
        _, idxs = torch.topk(m1_logits, k=topk, dim=1)

        # apply the softmax on model2 with the topk preselection of model1
        softmax_input = torch.full_like(m2_logits, float("-inf"))
        # for every batch just keep the top k indices from model1
        for b, b_idxs in enumerate(idxs):
            softmax_input[b, b_idxs] = m2_logits[b, b_idxs]
        softmax = torch.nn.Softmax(dim=1)
        softmax_output = softmax(softmax_input)

        # calculate the accuracy
        y_hat = softmax_output.argmax(dim=1)
        if y.ndim == 2:
            y = y.argmax(dim=1)

        top1 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        ).to(device)

        top1_acc = top1(y_hat, y)

        model2_acc1.append(top1_acc)

        if is_last:
            avg = calculate_epoch_avg(
                model2_acc1,
                normal_batch_size=batch_size,
                last_batch_size=x.shape[0],
            )

    return avg.cpu().numpy()


def plot(acc, exp1, exp2):
    fig = plt.bar(range(1), acc)
    plt.savefig(f"two_stage_prediction/{exp1}_{exp2}.png")


def two_stage_prediction(
    dataset_path="/share/klab/datasets/texture2shape_miniecoset_256px_dkl.h5",
    num_classes=112,
    batch_size=256,
    baseline=True,
    seperate_from_layer=0,
    like_resnet="resnet18",
    dvd_strategy="dkl",
    stream_concat=False,
    stream1_weight=1,
    topk=10,
    m1_experiment_name="dkl_lum",
    m1_age_m=0,
    m1_lum=True,
    m1_rg=False,
    m1_by=False,
    m1_blur=False,
    m1_cs=False,
    m1_combined_color=False,
    m2_experiment_name="dkl_rg",
    m2_age_m=0,
    m2_lum=False,
    m2_rg=True,
    m2_by=False,
    m2_blur=False,
    m2_cs=False,
    m2_combined_color=False,
):
    # load model 1
    model1, _, device = load_model(
        num_classes,
        like_resnet,
        baseline,
        seperate_from_layer,
        m1_experiment_name,
        dvd_strategy,
        m1_combined_color,
        stream_concat,
        stream1_weight,
    )
    model1.eval()

    # load model 2
    model2, _, _ = load_model(
        num_classes,
        like_resnet,
        baseline,
        seperate_from_layer,
        m2_experiment_name,
        dvd_strategy,
        m2_combined_color,
        stream_concat,
        stream1_weight,
    )
    model2.eval()

    transf = get_transforms()

    # load the data
    val_dataset, val_loader = load_data(
        dataset_path=dataset_path,
        transforms=transf,
        batch_size=batch_size,
        num_classes=num_classes,
        dvd_strategy=dvd_strategy,
    )

    avg_acc1 = run_batches(
        model1,
        model2,
        val_loader,
        topk,
        device,
        num_classes,
        batch_size,
        m1_age_m,
        m1_lum,
        m1_rg,
        m1_by,
        m1_blur,
        m1_cs,
        m1_combined_color,
        m2_age_m,
        m2_lum,
        m2_rg,
        m2_by,
        m2_blur,
        m2_cs,
        m2_combined_color,
    )

    plot(avg_acc1, m1_experiment_name, m2_experiment_name)


if __name__ == "__main__":
    # shared parameters
    dataset_path = "/share/klab/datasets/texture2shape_miniecoset_256px_dkl.h5"
    num_classes = 112
    batch_size = 256
    baseline = True
    seperate_from_layer = 0
    like_resnet = "resnet18"
    dvd_strategy = "dkl"
    stream_concat = False
    stream1_weight = 1
    topk = 10

    # individual parameters
    m1_experiment_name = "dkl_lum"
    m1_age_m = 0
    m1_lum = True
    m1_rg = False
    m1_by = False
    m1_blur = False
    m1_cs = False
    m1_combined_color = False

    m2_experiment_name = "dkl_rg"
    m2_age_m = 0
    m2_lum = False
    m2_rg = True
    m2_by = False
    m2_blur = False
    m2_cs = False
    m2_combined_color = False

    two_stage_prediction(
        dataset_path=dataset_path,
        num_classes=num_classes,
        batch_size=batch_size,
        baseline=baseline,
        seperate_from_layer=seperate_from_layer,
        like_resnet=like_resnet,
        dvd_strategy=dvd_strategy,
        stream_concat=stream_concat,
        stream1_weight=stream1_weight,
        topk=10,
        m1_experiment_name=m1_experiment_name,
        m1_age_m=m1_age_m,
        m1_lum=m1_lum,
        m1_rg=m1_rg,
        m1_by=m1_by,
        m1_blur=m1_blur,
        m1_cs=m1_cs,
        m1_combined_color=m1_combined_color,
        m2_experiment_name=m2_experiment_name,
        m2_age_m=m2_age_m,
        m2_lum=m2_lum,
        m2_rg=m2_rg,
        m2_by=m2_by,
        m2_blur=m2_blur,
        m2_cs=m2_cs,
        m2_combined_color=m2_combined_color,
    )
