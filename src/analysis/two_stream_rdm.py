"""
Script to compute all layer RDMs for a specified ANN.
"""

import gc
import os
import pickle
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import squareform
from torchvision import transforms

from src.models.pytorch_models import TwoStreamNetwork
from src.utils.helpers import (
    apply_gaussian_blur,
    apply_scale_free_contrast,
    convert_rgb_to_dkl,
    normalize,
)


class MiniEcoset(torch.utils.data.Dataset):
    "Test Dataset to validate the functionality of the code"

    def __init__(self, path, transf):
        self.dataset = h5py.File(path, "r")
        self.transf = transf

        img = self.dataset["test"]["data"]
        self.images = img[:118]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transf is not None:
            img = Image.fromarray(img)
            return self.transf(img)
        else:
            return torch.Tensor(img)


class NSD(torch.utils.data.Dataset):
    "NSD subset of COCO"

    def __init__(self, path, transf):
        self.dataset = h5py.File(path, "r")
        self.transf = transf

        idxs_use = np.load(
            "/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds515.npy"
        )
        img = self.dataset["test"]["data"]
        self.images = img[idxs_use]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transf is not None:
            img = Image.fromarray(img)
            return self.transf(img)
        else:
            return torch.Tensor(img)


def load_model(
    num_classes=112,
    like_resnet="resnet18",
    baseline=False,
    seperate_from_layer=0,
    experiment_name="split_dvd",
    dvd_strategy="dvd",
    combined_color=False,
    stream_concat=False,
    stream1_weight=0.5,
):
    "Load the custom Two-Stream Network"

    # set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create the model
    model = TwoStreamNetwork(
        num_classes=num_classes,
        like_resnet=like_resnet,
        baseline=baseline,
        seperate_from_layer=seperate_from_layer,
        dvd_strategy=dvd_strategy,
        combined_color=combined_color,
        stream_concat=stream_concat,
        stream1_weight=stream1_weight,
    )

    # load the last checkpoint
    checkpoint = torch.load(
        f"/share/klab/lzejin/mbenito/two_stream_dvd/logs/{experiment_name}/epoch=149-step=154350.ckpt",
        map_location=device,
    )

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

    # load the state_dict to the model and set the device
    model.load_state_dict(renamed_state_dict)
    model.to(device)

    # set up the transforms
    transf = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )

    return model, transf, device


def load_data(name="miniecoset", transf=None):
    "Load a specific dataset"

    if name == "miniecoset":
        path = "/share/klab/datasets/texture2shape_miniecoset_256px.h5"
        dataset = MiniEcoset(path, transf)
        batch_size = 2
    if name == "nsd":
        path = "/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze_16_fixations.h5"
        dataset = NSD(path, transf)
        batch_size = 8
    else:
        raise ValueError("This Dataset is not supported")

    # create dataloader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )

    return dataset, loader, batch_size


def select_layers(model, layers="all", baseline=False):
    "Select all layers of the model"

    layer_names = [
        layer_name
        for layer_name, _ in model.named_modules()
        if len(layer_name) > 0
    ]

    if layers == "conv":
        print("Selecting only convolutional layers")
        if baseline:
            first_layer = ["stream1_block1.0"]
        else:
            first_layer = ["stream1_block1.0", "stream2_block1.0"]

        layer_names = first_layer + [
            name for name in layer_names if "conv" in name
        ]

    return layer_names


def get_activations(
    model,
    images,
    layers,
    dvd_strategy,
    age_m,
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
    "Get the activations utilizing forward hooks"

    activations: Dict[str, torch.Tensor] = {}
    hooks = []
    batch_size = images.shape[0]

    def mk(name: str):
        if len(name) == 0:
            name = f"unnamed_module_{np.random.randint(0,100)}"
            print(f"Warning: empty module name, renaming to {name}")

        def _hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor):
                print(f"Warning: {name} output is not a tensor, skipping")
                return
            if t.shape[0] != batch_size:  # in case batch dim is not first
                print(
                    f"Warning: {name} output batch size {t.shape[0]} != expected {batch_size}, trying second dim"
                )
                if t.shape[1] == batch_size:
                    t = t.permute(1, 0, *range(2, t.ndim))
                    print(f"  • {name} permuted to put batch dim first")
                    activations[name] = (
                        t.detach().cpu().reshape(batch_size, -1)
                    )
                else:
                    print(
                        f"Error: {name} output batch size not found in first two dims, skipping"
                    )
            #     bdim = list(t.shape).index(batch_size)
            #     t = t.permute([bdim] + [d for d in range(t.ndim) if d != bdim])
            # acts[name] = t.detach().cpu().reshape(bs, -1).to(torch.float16)
            else:
                activations[name] = t.detach().cpu().reshape(batch_size, -1)

        return _hook

    for n, m in model.named_modules():
        if n in layers:
            hooks.append(m.register_forward_hook(mk(n)))

    with torch.inference_mode():
        if dvd_strategy in ["dkl", "dkl_dvd"]:
            # get the magno and parvo channels

            # magno/lum/dorsal
            x_1 = images[:, 0, :, :]
            # parvo/rg/ventral
            x_2 = images[:, 1, :, :]
            # konio/by
            x_3 = images[:, 2, :, :]

            # unsqueeze
            x_1 = x_1.unsqueeze(1)
            x_2 = x_2.unsqueeze(1)
            x_3 = x_3.unsqueeze(1)

            if lum:

                if lum_blur:
                    x_1 = apply_gaussian_blur(x_1)

                if lum_cs:
                    x_1 = apply_scale_free_contrast(x_1, age_m)

                if not rg and not by:
                    _ = model(x_1)

            if rg:

                if rg_blur:
                    x_2 = apply_gaussian_blur(x_2)

                if rg_cs:
                    x_2 = apply_scale_free_contrast(x_2, age_m)

                if not lum and not by:
                    _ = model(x_2)

            if by:

                if by_blur:
                    x_3 = apply_gaussian_blur(x_3)

                if by_cs:
                    x_3 = apply_scale_free_contrast(x_3, age_m)

                if not lum and not rg:
                    _ = model(x_3)

            if combined_color:
                x_combined = images[:, 1:3, :, :]
                x_combined[:, 0, :, :] = x_2.squeeze(1)
                x_combined[:, 1, :, :] = x_3.squeeze(1)

                if lum and not model.baseline:
                    _ = model(x_1, x_combined)
                else:
                    _ = model(x_combined)

            else:
                if lum and rg and not by:
                    _ = model(x_1, x_2)
                if lum and by and not rg:
                    _ = model(x_1, x_3)
                if rg and by and not lum:
                    _ = model(x_2, x_3)

        else:
            _ = model(images)

    for h in hooks:
        h.remove()

    return activations


def run_batches(
    model,
    loader,
    device,
    layers,
    h5_path,
    N_images,
    batch_size=2,
    dvd_strategy="dvd",
    age_m=48,
    lum=False,
    rg=False,
    by=False,
    lum_blur=False,
    lum_cs=False,
    rg_blur=False,
    rg_cs=False,
    by_blur=False,
    by_cs=False,
    combined_color=False,
):
    "Get the activations of al batches"

    written = 0
    if dvd_strategy in ["dkl", "dkl_dvd"]:
        data = []
        for batch in loader:
            data.append(batch)
        data = torch.cat(data, dim=0)

        data = convert_rgb_to_dkl(data, device)

        min_max = []
        # lum
        lum_min = data[:, 0, :, :].min().item()
        lum_max = data[:, 0, :, :].max().item()
        min_max.append((lum_min, lum_max))

        # rg
        rg_min = data[:, 1, :, :].min().item()
        rg_max = data[:, 1, :, :].max().item()
        min_max.append((rg_min, rg_max))

        # by
        by_min = data[:, 2, :, :].min().item()
        by_max = data[:, 2, :, :].max().item()
        min_max.append((by_min, by_max))

    # -- First Batch (initial write of h5 file)--
    images = next(iter(loader))
    if dvd_strategy in ["dkl", "dkl_dvd"]:
        images = convert_rgb_to_dkl(images, device)
        for i in range(images.shape[1]):
            channel = images[:, i, :, :]
            c_min, c_max = min_max[i]
            normalized_channel = normalize(channel, c_min, c_max)
            images[:, i, :, :] = normalized_channel

    with torch.no_grad():
        activations = get_activations(
            model,
            images.to(device),
            layers,
            dvd_strategy,
            age_m,
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
        )

    h5 = h5py.File(h5_path, "w")
    for key, value in activations.items():
        # check if v is empty
        if value.numel() == 0:
            print(f"Warning: Layer {key} produced empty activations, skipping")
            continue
        F = value.shape[1]
        h5.create_dataset(
            key, shape=(N_images, F), dtype="float32", chunks=(batch_size, F)
        )
        h5[key][0 : value.shape[0]] = value
    written = value.shape[0]

    del images, activations, value
    torch.cuda.empty_cache()
    gc.collect()

    # -- Remaining Batches --
    loader_iter = iter(loader)
    next(loader_iter)  # skip first batch
    for b_idx, images in enumerate(loader_iter, start=1):
        if dvd_strategy in ["dkl", "dkl_dvd"]:
            images = convert_rgb_to_dkl(images, device)
            for i in range(images.shape[1]):
                channel = images[:, i, :, :]
                c_min, c_max = min_max[i]
                normalized_channel = normalize(channel, c_min, c_max)
                images[:, i, :, :] = normalized_channel
        with torch.no_grad():
            activations = get_activations(
                model,
                images.to(device),
                layers,
                dvd_strategy,
                age_m,
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
            )
        for key, value in activations.items():
            h5[key][written : written + value.shape[0]] = value
        written += value.shape[0]
        print(f"▶ {written}/{N_images} images processed")
        del images, activations, value
        torch.cuda.empty_cache()
        gc.collect()

    assert written == N_images, "Dataset size mismatch during write"

    return h5


def compute_single_RDM(h5, key, device):
    "Calculate the RDM for a single layer"

    # correlation
    ds = h5[key]
    ds = ds[()]
    N = ds.shape[0]

    ds = ds.reshape(N, -1)
    out = np.empty(N * (N - 1) // 2, dtype=np.float32)

    use_features = ds.std(axis=0) > 1e-6
    if not use_features.any():
        print(
            f"Warning: No features in {key} with std > 1e-6, returning empty RDM"
        )
        return
    print(f"▶ Using {use_features.sum()} features in {key}")

    ds = torch.Tensor(ds[:, use_features]).to(device)

    # row‐center & normalize so corr(u,v) = u_norm·v_norm
    mean = ds.mean(dim=1, keepdim=True)
    ds = ds - mean
    norm = ds.norm(dim=1, keepdim=True)
    zero_rows = norm.squeeze(1) < 1e-12
    ds = ds / norm.clamp_min(1e-12)

    # full correlation matrix = X @ X^T
    #  then correlation‐distance = 1 – corr
    corr = ds @ ds.t()
    corr.clamp_(-1.0, 1.0)
    dist = 1.0 - corr

    if zero_rows.any():
        zr = zero_rows.nonzero(as_tuple=True)[0]
        dist[zr, :] = 1.0
        dist[:, zr] = 1.0
        dist[zr, zr] = 0.0

    dist = dist.cpu().numpy()  # move to CPU for squareform
    dist = dist + dist.T  # make symmetric
    np.fill_diagonal(dist, 0)  # set diagonal to 0

    # dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    out = squareform(dist)

    print(f"▶ RDM for {key} has shape {out.shape} and dtype {out.dtype}")

    return out


def compute_RDMs(h5, layers, device):
    "Calculate all RDMs"

    RDMs = {}

    for key in layers:
        if key not in h5:
            print(
                f"⚠️  Skipping RDM for {key} as no activations were recorded"
            )
            continue
        print(f"▶ Computing RDM for {key}")
        RDMs[key] = compute_single_RDM(h5, key, device)

    return RDMs


def get_ann_rdms(
    num_classes=112,
    like_resnet="resnet18",
    baseline=False,
    seperate_from_layer=0,
    experiment_name="split_dvd",
    dataset_name="nsd",
    layers="conv",
    dvd_strategy="dvd",
    age_m=48,
    stream_concat=False,
    stream1_weight=0.5,
    lum=False,
    rg=False,
    by=False,
    lum_blur=False,
    lum_cs=False,
    rg_blur=False,
    rg_cs=False,
    by_blur=False,
    by_cs=False,
    combined_color=False,
):
    "Main function to compute the RDMs"

    # paths to save the data to
    h5_path = (
        Path("/share/klab/lzejin/mbenito/two_stream_dvd/RDMs")
        / f"{experiment_name}_{dataset_name}_{like_resnet}_seperate_from_layer{seperate_from_layer}_acts.h5"
    )
    rdm_path = (
        Path("/share/klab/lzejin/mbenito/two_stream_dvd/RDMs")
        / f"{experiment_name}_{dataset_name}_{like_resnet}_seperate_from_layer{seperate_from_layer}_RDMs.pkl"
    )

    # load the model, transforms and get device
    model, transf, device = load_model(
        num_classes,
        like_resnet,
        baseline,
        seperate_from_layer,
        experiment_name,
        dvd_strategy,
        combined_color,
        stream_concat,
        stream1_weight,
    )

    # put model into eval mode
    model.eval()

    # get the data
    dataset, loader, batch_size = load_data(name=dataset_name, transf=transf)
    N_images = len(dataset)
    print(
        f"▶ Dataset {dataset_name} loaded with {N_images} images and batch size {batch_size}"
    )

    # select the layers
    layers = select_layers(model, layers, baseline)
    print(f" Selected layers: {layers}")

    # get all activations and save them
    h5 = run_batches(
        model,
        loader,
        device,
        layers,
        h5_path,
        N_images,
        batch_size,
        dvd_strategy,
        age_m,
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
    )

    RDMs = compute_RDMs(h5, layers, device)

    with open(rdm_path, "wb") as f:
        pickle.dump(RDMs, f)
    print(f"✔ RDMs written → {rdm_path}")

    h5.close()
    if h5_path.exists():
        os.remove(h5_path)
        print(f"🗑 Deleted activations file {h5_path}")


if __name__ == "__main__":
    # split_dvd config
    num_classes = 112
    like_resnet = "resnet18"
    baseline = True
    seperate_from_layer = 0
    experiment_name = "dkl_rg_dvd_trans"
    dataset_name = "nsd"
    layers = "conv"
    dvd_strategy = "dkl"
    age_m = 0
    stream_concat = False
    stream1_weight = 1
    lum = False
    rg = True
    by = False
    lum_blur = False
    lum_cs = False
    rg_blur = False
    rg_cs = False
    by_blur = False
    by_cs = False
    combined_color = False

    get_ann_rdms(
        num_classes=num_classes,
        like_resnet=like_resnet,
        baseline=baseline,
        seperate_from_layer=seperate_from_layer,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        layers=layers,
        dvd_strategy=dvd_strategy,
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
