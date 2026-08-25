import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# RSA plot direcories
direct = "/share/klab/lzejin/mbenito/two_stream_dvd/RDMs/"

rgb_tsn = (
    direct
    + "final_ts_resnet_18_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)
dvd_tsn = direct + "ts_dvd_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
sep_dkl_base = (
    direct
    + "final_ts_dkl_base_lum_rg_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)
sep_m_heavy_dkl_b_cs = (
    direct
    + "dkl_blur_cs_s1_90_nsd_resnet18_seperate_from_layer0_RDMs_streams.png"
)

# load plots
img1 = mpimg.imread(rgb_tsn)
img2 = mpimg.imread(dvd_tsn)
img3 = mpimg.imread(sep_dkl_base)
img4 = mpimg.imread(sep_m_heavy_dkl_b_cs)


fig, axes = plt.subplots(4, 1, figsize=(18, 18))

axes[0].imshow(img1)
axes[0].text(
    0.01,
    1.0,
    "RGB TSN",
    transform=axes[0].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes[1].imshow(img2)
axes[1].text(
    0.01,
    1.0,
    "DVD TSN",
    transform=axes[1].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes[2].imshow(img3)
axes[2].text(
    0.01,
    1.0,
    "DKL Base",
    transform=axes[2].transAxes,
    fontsize=18,
    fontweight="bold",
)

axes[3].imshow(img4)
axes[3].text(
    0.01,
    1.0,
    "M-heavy DKL B&CS",
    transform=axes[3].transAxes,
    fontsize=18,
    fontweight="bold",
)

# Achsen ausblenden
for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "src/visualizations/assets/neural_data.pdf", dpi=600, bbox_inches="tight"
)
plt.show()
