import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# data manually retrieved from csv files that logged the shabe biases and accuracies
# these files are to find under: /share/klab/lzejin/mbenito/two_stream_dvd/src/analysis/neuroai/raw-data/performances...
beh_shape_biases = {
    "RGB Baseline": 0.494,#...ts_final_ss_resnet_50.csv                                          
    "RGB TSN": 0.534,#...ts_final_resnet_18.csv
    "DVD Baseline": 0.833,#...ts_baseline_resnet50_DVD_Scale_Free_DVD_trans.csv
    "DVD TSN": 0.802,#...ts_resnet18_dvd_final.csv
    "DKL Base": 0.424, #...ts_resnet18_dkl_base.csv
    "DKL B": 0.664, #...ts_resnet18_dkl_blur_final.csv
    "DKL CS": 0.681,#...ts_resnet18_dkl_contrast_final.csv
    "DKL B&CS": 0.677,#...ts_resnet18_dkl_blur_cs_dvd_trans.csv
    "M-heavy DKL B&CS": 0.756,#...ts_resnet18_dkl_blur_cs_s1_90.csv
    "Inter. DKL B&CS": 0.609,#...ts_resnet18_dkl_blur_cs_concat.csv
    "Inter. M-heavy DKL B&CS": 0.615, #...ts_resnet18_dkl_blur_cs_concat_s1_90.csv
    "M-driven SSN B&CS": 0.927,#...ts_resnet18_dkl_lum_blur_cs_final.csv
    "P-driven SSN": 0.527,#...ts_resnet18_dkl_rg_dvd_trans.csv
}

repr_shape_biases = {
    "RGB Baseline": {"stream1": 0.345},                                                
    "RGB TSN": {"stream1": 0.374, "stream2": 0.374}, 
    "DVD Baseline": {"stream1": 0.771}, 
    "DVD TSN": {"stream1": 0.760, "stream2": 0.760},
    "DKL Base": {"stream1": 0.263, "stream2": 0.101},
    "DKL B": {"stream1": 0.689, "stream2": 0.078},
    "DKL CS": {"stream1": 0.937, "stream2": 0.184},
    "DKL B&CS": {"stream1": 0.986, "stream2": 0.113},
    "M-heavy DKL B&CS": {"stream1": 0.985, "stream2": 0.131},
    "Inter. DKL B&CS": {"stream1": 0.677, "stream2": 0.158},
    "Inter. M-heavy DKL B&CS":{"stream1": 0.540, "stream2": 0.139},
    "M-driven SSN B&CS": {"stream1": 0.962},
    "P-driven SSN": {"stream1": 0, "stream2": 0.206},
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
    "M-driven SSN B&CS",
    "P-driven SSN",
]

B_models = [
    "RGB Baseline",
    "RGB TSN",
    "DVD Baseline",
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
    elif "SSN" in model:
        return "#8172B2"
    return "gray"

# -------------------------
# Figure with 2 panels
# -------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]})

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
    axA.bar(xA[i] - width,
            repr_shape_biases[m]["stream1"],
            width=width,
            color=c,
            alpha=0.65,
            edgecolor="black",
            linewidth=0.6)

    # behavior
    axA.bar(xA[i],
            beh_shape_biases[m],
            width=width,
            color=c,
            edgecolor="black",
            linewidth=0.9)

    # stream 2 (parvo-driven if exists)
    if "stream2" in repr_shape_biases[m]:
        axA.bar(xA[i] + width,
                repr_shape_biases[m]["stream2"],
                width=width,
                color=c,
                alpha=0.35,
                edgecolor="black",
                linewidth=0.6)

axA.set_xticks(xA)
axA.set_xticklabels([m.replace(" ", "\n") for m in A_models], fontsize=11)
axA.set_ylabel("Shape Bias", fontsize=14)
axA.axhline(0.5, linestyle="--", color="black", alpha=0.25)
axA.text(-0.05, 1.15, "a   Magno- and Parvo-driven Streams of DKL models", transform=axA.transAxes,
                  fontsize=16, fontweight="bold")

# A-panel legend
legend_A = [
    mpatches.Patch(facecolor="gray", alpha=0.65, edgecolor="black", label="Magno-driven"),
    mpatches.Patch(facecolor="gray", alpha=1.0, edgecolor="black", label="Behavioral"),
    mpatches.Patch(facecolor="gray", alpha=0.35, edgecolor="black", label="Parvo-driven"),
]
axA.legend(handles=legend_A, frameon=False, loc="upper left")

axA.grid(True, axis="y", linestyle="--", alpha=0.4)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)


# =========================================================
# Panel B 
# =========================================================


yB = np.arange(len(B_models))

for i, m in enumerate(B_models):
    c = get_color(m)

    axB.bar(yB[i] - width,
             repr_shape_biases[m]["stream1"],
             width=width,
             color=c,
             alpha=0.65,
             edgecolor="black",
             linewidth=0.6)

    axB.bar(yB[i],
             beh_shape_biases[m],
             width=width,
             color=c,
             edgecolor="black",
             linewidth=0.9)

    if "stream2" in repr_shape_biases[m]:
        axB.bar(yB[i] + width,
                 repr_shape_biases[m]["stream2"],
                 width=width,
                 color=c,
                 alpha=0.35,
                 edgecolor="black",
                 linewidth=0.6)

axB.set_xticks(yB)
axB.set_xticklabels([m.replace(" ", "\n") for m in B_models], fontsize = 11)
axB.set_ylabel("Shape Bias", fontsize=14)
axB.set_ylim(0,1)
axB.axhline(0.5, linestyle="--", color="black", alpha=0.25)
axB.text(-0.10, 1.15, "b   Streams of RGB and DVD Models", transform=axB.transAxes,
                  fontsize=16, fontweight="bold")

# B-panel legend
legend_B = [
    mpatches.Patch(facecolor="gray", alpha=0.65, edgecolor="black", label="Stream 1"),
    mpatches.Patch(facecolor="gray", alpha=1.0, edgecolor="black", label="Behavioral"),
    mpatches.Patch(facecolor="gray", alpha=0.35, edgecolor="black", label="Stream 2"),
]
axB.legend(handles=legend_B, frameon=False, loc="upper left")

axB.grid(True, axis="y", linestyle="--", alpha=0.4)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

plt.tight_layout()
#plt.savefig("/share/klab/lzejin/mbenito/two_stream_dvd/src/visualizations/assets/shape_bias_panel_AB.pdf", bbox_inches="tight")
plt.savefig("shape_bias_panel_AB.pdf", bbox_inches="tight")

plt.show()