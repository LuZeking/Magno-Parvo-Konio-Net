import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# RSA plot direcories
direct = "/share/klab/lzejin/mbenito/two_stream_dvd/RDMs/"

rgb_base = (
    direct
    + "final_ss_resnet_50_nsd_resnet50_seperate_from_layer0_RDMs_streams.png"
)
dvd_base = (
    direct
    + "baseline_dvd_sf_dvd_trans_nsd_resnet50_seperate_from_layer0_RDMs_streams.png"
)
dkl_b = (
    direct
    + "final_ts_dkl_blur_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)
dkl_cs = (
    direct
    + "final_ts_dkl_contrast_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)
dkl_b_cs = (
    direct
    + "dkl_blur_cs_dvd_trans_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)
int_dkl_b_cs = (
    direct
    + "dkl_blur_cs_concat_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)
int_m_heavy_dkl_b_cs = (
    direct
    + "dkl_blur_cs_concat_s1_90_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)
m_driven = (
    direct
    + "final_ss_dkl_lum_blur_cs_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)
p_driven = (
    direct
    + "dkl_rg_dvd_trans_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)

# load plots
img1 = mpimg.imread(rgb_base)
img2 = mpimg.imread(dvd_base)
img8 = mpimg.imread(m_driven)
img9 = mpimg.imread(p_driven)


img3 = mpimg.imread(dkl_b)
img4 = mpimg.imread(dkl_cs)
img5 = mpimg.imread(dkl_b_cs)
img6 = mpimg.imread(int_dkl_b_cs)
img7 = mpimg.imread(int_m_heavy_dkl_b_cs)


fig, axes = plt.subplots(4, 1, figsize=(18, 18))

axes[0].imshow(img1)
axes[0].text(
    0.01,
    1.0,
    "RGB Baseline",
    transform=axes[0].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes[1].imshow(img2)
axes[1].text(
    0.01,
    1.0,
    "DVD Baseline",
    transform=axes[1].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes[2].imshow(img8)
axes[2].text(
    0.01,
    1.0,
    "M-driven SSN B&CS",
    transform=axes[2].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes[3].imshow(img9)
axes[3].text(
    0.01,
    1.0,
    "P-driven SSN",
    transform=axes[3].transAxes,
    fontsize=18,
    fontweight="bold",
)

# Achsen ausblenden
for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "src/visualizations/assets/neural_data_appendix1.pdf",
    dpi=600,
    bbox_inches="tight",
)


fig2, axes2 = plt.subplots(5, 1, figsize=(22, 18))


axes2[0].imshow(img3)
axes2[0].text(
    0.01,
    1.0,
    "DKL B",
    transform=axes2[0].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes2[1].imshow(img4)
axes2[1].text(
    0.01,
    1.0,
    "DKL CS",
    transform=axes2[1].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes2[2].imshow(img5)
axes2[2].text(
    0.01,
    1.0,
    "DKL B&CS",
    transform=axes2[2].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes2[3].imshow(img6)
axes2[3].text(
    0.01,
    1.0,
    "Inter. DKL B&CS",
    transform=axes2[3].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes2[4].imshow(img7)
axes2[4].text(
    0.01,
    1.0,
    "Inter. M-heavy DKL B&CS",
    transform=axes2[4].transAxes,
    fontsize=18,
    fontweight="bold",
)

# Achsen ausblenden
for ax in axes2.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "src/visualizations/assets/neural_data_appendix2.pdf",
    dpi=600,
    bbox_inches="tight",
)
