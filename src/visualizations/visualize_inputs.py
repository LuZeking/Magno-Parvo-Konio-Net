import h5py
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

from src.utils.helpers import (
    apply_gaussian_blur,
    apply_scale_free_contrast,
)


# -----------------------------
# RGB
# -----------------------------
def show_rgb(ax, image):
    ax.imshow(image)
    ax.text(
        0.5,
        -0.10,
        "RGB",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )


# -----------------------------
# DKL dataset channels
# -----------------------------
def show_dataset_dkl(ax, image):
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
    image = T.ConvertImageDtype(torch.float)(image)
    image = image.squeeze(0).permute(1, 2, 0)

    ax[0, 1].imshow(image[:, :, 0], cmap="gray")
    ax[0, 1].text(
        0.5,
        -0.10,
        "Luminance",
        transform=ax[0, 1].transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    ax[0, 2].imshow(image[:, :, 1], cmap="PiYG_r")
    ax[0, 2].text(
        0.5,
        -0.10,
        "Red-Green",
        transform=ax[0, 2].transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )


# -----------------------------
# Transformations (bottom row)
# -----------------------------
def show_transformations(ax, image, age_m):
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
    image = T.ConvertImageDtype(torch.float)(image)
    image = image.squeeze(0).permute(1, 2, 0)

    lum = image[:, :, 0].unsqueeze(0).unsqueeze(0)

    blur = apply_gaussian_blur(lum).cpu().numpy()
    contrast = apply_scale_free_contrast(lum, age_m).cpu().numpy()

    blur_contrast = (
        apply_scale_free_contrast(apply_gaussian_blur(lum), age_m)
        .cpu()
        .numpy()
    )

    imgs = [blur, contrast, blur_contrast]
    labels = ["Blur", "Contrast", "Blur + Contrast"]

    for i, (img, label) in enumerate(zip(imgs, labels)):
        ax[1, i].imshow(img[0, 0], cmap="gray")
        ax[1, i].text(
            0.5,
            -0.10,
            label,
            transform=ax[1, i].transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )


# -----------------------------
# Main
# -----------------------------
def main():
    miniecoset_path = "/share/klab/datasets/texture2shape_miniecoset_256px.h5"
    dkl_path = (
        "/share/klab/datasets/texture2shape_miniecoset_256px_dkl_kapoor.h5"
    )

    img_number = 0
    age_m = 48

    with h5py.File(miniecoset_path, "r") as f:
        eco_image = f["train"]["data"][img_number]

    with h5py.File(dkl_path, "r") as f:
        dkl_image = f["train"]["data"][img_number]

    # -----------------------------
    # Figure
    # -----------------------------
    fig, ax = plt.subplots(2, 3, figsize=(8.5, 5.5))

    fig.subplots_adjust(wspace=0.35, hspace=0.55)

    # remove axes
    for a in ax.ravel():
        a.axis("off")

    # top row
    show_rgb(ax[0, 0], eco_image)
    show_dataset_dkl(ax, dkl_image)

    # bottom row
    show_transformations(ax, dkl_image, age_m)

    # -----------------------------
    # Panel labels (a, b, c)
    # -----------------------------
    fig.text(0.10, 0.94, "a   Original Image", fontsize=13, fontweight="bold")
    fig.text(
        0.40, 0.94, "b   DKL Input Channels", fontsize=13, fontweight="bold"
    )
    fig.text(
        0.10,
        0.46,
        "c   Luminance Transformations",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------
    # Save
    # -----------------------------
    fig.savefig(
        f"/share/klab/lzejin/mbenito/two_stream_dvd/src/visualizations/assets/model_input_{img_number}.png",
        dpi=900,
        bbox_inches="tight",
    )


if __name__ == "__main__":  # PiYG_r
    main()
