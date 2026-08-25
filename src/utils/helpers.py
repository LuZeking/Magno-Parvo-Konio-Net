import colour
import kornia.filters as KF
import numpy as np
import torch
from dvd_scale_free.dvd_scale_free.development import DVDConfig, DVDTransformer

##################################################################################

# -- Augmentations --

##################################################################################


def apply_gaussian_blur(x):
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


def apply_scale_free_contrast(x, age_m, image_size=256):
    dvdt = DVDTransformer(
        DVDConfig(
            apply_blur=0,
            apply_color=0,
            apply_contrast=1,
            cs_logspan_start=5e-3,
            image_size=image_size,
            decomposition_mode="keep",
        )
    )
    x = dvdt(x, age_m)
    return x


##################################################################################

# -- RGB colour space images to DKL colour space  --

##################################################################################


def convert_rgb_to_dkl(img, device):
    """
    Convert the image to DKL color space, to simulate magno-parvo inputs.
    Code provided from Zejin.
    """

    # Standard XYZ → LMS (Hunt-Pointer-Estevez)
    M_XYZ2LMS = np.array(
        [
            [0.4002, 0.7075, -0.0807],
            [-0.2263, 1.1653, 0.0457],
            [0.0, 0.0, 0.9182],
        ]
    )
    img = img.cpu().numpy().transpose(0, 2, 3, 1)

    # RGB → XYZ
    xyz = colour.sRGB_to_XYZ(img)

    # XYZ → LMS
    lms = np.tensordot(xyz, M_XYZ2LMS.T, axes=1)

    # Define proper gray background
    gray_rgb = np.array([0.5, 0.5, 0.5])
    gray_xyz = colour.sRGB_to_XYZ(gray_rgb)
    lms_bg = gray_xyz @ M_XYZ2LMS.T

    # cone contrast
    d = (lms - lms_bg) / (lms_bg + 1e-6)

    dL, dM, dS = d[..., 0], d[..., 1], d[..., 2]

    # opponent axes
    lum = (dL + dM) / np.sqrt(2)
    rg = (dL - dM) / np.sqrt(2)
    by = (dS - (dL + dM)) / np.sqrt(2)

    dkl = np.stack([lum, rg, by], axis=-1)
    dkl = torch.from_numpy(dkl).float().to(device).permute(0, 3, 1, 2)

    return dkl


def convert_rgb_to_dkl_numpy(img, needs_norm=False):
    """
    Convert the image to DKL color space, to simulate magno-parvo inputs.
    Code provided from Zejin.
    """
    img = img / 255.0  # Normalize RGB to [0, 1]

    # Standard XYZ → LMS (Hunt-Pointer-Estevez)
    M_XYZ2LMS = np.array(
        [
            [0.4002, 0.7075, -0.0807],
            [-0.2263, 1.1653, 0.0457],
            [0.0, 0.0, 0.9182],
        ]
    )

    # RGB → XYZ
    xyz = colour.sRGB_to_XYZ(img)

    # XYZ → LMS
    lms = np.tensordot(xyz, M_XYZ2LMS.T, axes=1)

    # Define proper gray background
    gray_rgb = np.array([0.5, 0.5, 0.5])
    gray_xyz = colour.sRGB_to_XYZ(gray_rgb)
    lms_bg = gray_xyz @ M_XYZ2LMS.T

    # cone contrast
    d = (lms - lms_bg) / (lms_bg + 1e-6)

    dL, dM, dS = d[..., 0], d[..., 1], d[..., 2]

    # opponent axes
    lum = (dL + dM) / np.sqrt(2)
    rg = (dL - dM) / np.sqrt(2)
    by = (dS - (dL + dM)) / np.sqrt(2)

    dkl = np.stack([lum, rg, by], axis=-1)
    # dkl = torch.from_numpy(dkl).float()

    return dkl


##################################################################################

# -- Normalization --

##################################################################################


def normalize(x, min, max):
    x = (x - min) / ((max - min) + 1e-12)
    return x


def _normalize_images(images: torch.Tensor, norm_type: str) -> torch.Tensor:
    """Normalize a batch of images according to `norm_type`.

    - "0-1"     : rescale to [0, 1] using min/max (used by our custom models)
    - "mean-std": assume the dataloader already applied ImageNet normalization
    """
    if norm_type == "0-1":
        images = (images - images.min()) / (
            images.max() - images.min() + 1e-12
        )
    elif norm_type == "mean-std":
        # The loader is assumed to have normalized with ImageNet stats.
        pass
    else:
        raise ValueError(f"Unsupported data_normalization_type: {norm_type}")
    return images


def normalize_images_channels(
    images: torch.Tensor, norm_type: str
) -> torch.Tensor:
    """Normalize a batch of images per channel according to `norm_type`.

    - "0-1"     : rescale to [0, 1] using min/max (used by our custom models)
    - "mean-std": assume the dataloader already applied ImageNet normalization
    """
    if norm_type == "0-1":
        # images = (images - images.min()) / (
        #     images.max() - images.min() + 1e-12
        # )
        c_1 = images[:, 0, :, :]
        c_2 = images[:, 1, :, :]
        c_3 = images[:, 2, :, :]

        c_1_norm = _normalize_images(c_1, norm_type)
        c_2_norm = _normalize_images(c_2, norm_type)
        c_3_norm = _normalize_images(c_3, norm_type)

        images[:, 0, :, :] = c_1_norm
        images[:, 1, :, :] = c_2_norm
        images[:, 2, :, :] = c_3_norm

    else:
        raise ValueError(f"Unsupported data_normalization_type: {norm_type}")
    return images
