import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# data manually retrieved from output files that logged the accuracies during training
# these files are to find under: /share/klab/lzejin/mbenito/two_stream_dvd/output&error/linear_readout/...
linear_readout_acc = {                                           
    "RGB TSN": {"stream1": 0.705, "stream2": 0.702}, #...final_ts_resnet_18.output
    "DVD TSN": {"stream1": 0.662, "stream2": 0.662}, #...ts_dvd.output
    "DKL Base": {"stream1": 0.667, "stream2": 0.305},  # ...final_ts_dkl_base_lum_rg.output
    "DKL B": {"stream1": 0.558, "stream2": 0.320}, # ...final_ts_dkl_blur.output
    "DKL CS": {"stream1": 0.309, "stream2": 0.511}, # ...final_ts_dkl_contrast.output
    "DKL B&CS": {"stream1": 0.236, "stream2": 0.456}, # ...dkl_blur_cs_dvd_trans.output
    "M-heavy DKL B&CS": {"stream1": 0.268, "stream2": 0.450}, # ...dkl_blur_cs_s1_90.output
    "Inter. DKL B&CS": {"stream1": 0.489, "stream2": 0.482}, # ...dkl_blur_cs_concat.output
    "Inter. M-heavy DKL B&CS": {"stream1": 0.572, "stream2": 0.295}, # ...dkl_blur_cs_concat_s1_90.output
}


# -------------------------
# Model splits
# -------------------------
A_models = [
    "DKL Base",
    "DKL B",
    "DKL CS",
    "DKL B&CS",
    "M-heavy DKL B&CS",
    "Inter. DKL B&CS",
    "Inter. M-heavy DKL B&CS",
]

B_models = [
    "RGB TSN",
    "DVD TSN",
]

# -------------------------
# Color function
# -------------------------
def get_color(model):
    if model.startswith("RGB"):
        return "#4C72B0"
    elif model.startswith("DVD"):
        return "#55A868"
    elif "DKL" in model:
        return "#C44E52"
    return "gray"

# -------------------------
# Subplots
# -------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

width = 0.25

# =========================================================
# Panel A 
# =========================================================
xA = np.arange(len(A_models))

for i, m in enumerate(A_models):
    c = get_color(m)

    # stream 1 (magno-driven)
    axA.bar(xA[i] - width/2,
            linear_readout_acc[m]["stream1"],
            width=width,
            color=c,
            alpha=0.65,
            edgecolor="black",
            linewidth=0.6)

    # stream 2 (parvo-driven)
    axA.bar(xA[i] + width/2,
            linear_readout_acc[m]["stream2"],
            width=width,
            color=c,
            alpha=0.35,
            edgecolor="black",
            linewidth=0.6)

axA.set_xticks(xA)
axA.set_xticklabels([m.replace(" ", "\n") for m in A_models], fontsize=11)
axA.set_ylabel("Top-1 Validation Accuracy", fontsize=14)
axA.text(-0.05, 1.15, "a   Magno- and Parvo-driven Streams of DKL models", transform=axA.transAxes,
                  fontsize=14, fontweight="bold")
axA.set_ylim(0,0.8)

# A-panel legend
legend_A = [
    mpatches.Patch(facecolor="gray", alpha=0.65, edgecolor="black", label="Magno-driven"),
    mpatches.Patch(facecolor="gray", alpha=0.35, edgecolor="black", label="Parvo-driven"),
]
axA.legend(handles=legend_A, frameon=False, loc="upper left")

axA.grid(True, axis="y", linestyle="--", alpha=0.4)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)


# =========================================================
# Panel B
# =========================================================

width = 0.1

xB = np.arange(len(B_models))

for i, m in enumerate(B_models):
    c = get_color(m)

    axB.bar(xB[i] - width/2,
             linear_readout_acc[m]["stream1"],
             width=width,
             color=c,
             alpha=0.65,
             edgecolor="black",
             linewidth=0.6)


    axB.bar(xB[i] + width/2,
                linear_readout_acc[m]["stream2"],
                width=width,
                color=c,
                alpha=0.35,
                edgecolor="black",
                linewidth=0.6)

axB.set_xticks(xB)
axB.set_xticklabels([m.replace(" ", "\n") for m in B_models], fontsize = 11)
axB.set_ylabel("Top-1 Validation Accuracy", fontsize=14)
axB.text(-0.10, 1.15, "b   Streams of RGB and DVD Models", transform=axB.transAxes,
                  fontsize=14, fontweight="bold")
axB.set_ylim(0,0.8)

# B-panel legend
legend_B = [
    mpatches.Patch(facecolor="gray", alpha=0.65, edgecolor="black", label="Stream 1"),
    mpatches.Patch(facecolor="gray", alpha=0.35, edgecolor="black", label="Stream 2"),
]
axB.legend(handles=legend_B, frameon=False, loc="upper right")

axB.grid(True, axis="y", linestyle="--", alpha=0.4)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("/share/klab/lzejin/mbenito/two_stream_dvd/src/visualizations/assets/linear_readout_AB.pdf", bbox_inches="tight")
plt.show()