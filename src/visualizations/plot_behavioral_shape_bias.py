import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data
# -------------------------
# data manually retrieved from csv files that logged the shabe biases and accuracies
# these files are to find under: /share/klab/lzejin/mbenito/two_stream_dvd/src/analysis/neuroai/raw-data/performances...
models = [
    "RGB Baseline",#...ts_final_ss_resnet_50.csv          
    "RGB TSN", #...ts_final_resnet_18.csv
    "DVD Baseline",#...ts_baseline_resnet50_DVD_Scale_Free_DVD_trans.csv
    "DVD TSN",#...ts_resnet18_dvd_final.csv
    "DKL Base", #...ts_resnet18_dkl_base.csv
    "DKL B", #...ts_resnet18_dkl_blur_final.csv
    "DKL CS",#...ts_resnet18_dkl_contrast_final.csv
    "DKL B&CS",#...ts_resnet18_dkl_blur_cs_dvd_trans.csv
    "Inter. DKL B&CS",#...ts_resnet18_dkl_blur_cs_concat.csv
    "M-heavy DKL B&CS",#...ts_resnet18_dkl_blur_cs_s1_90.csv
    "Inter. M-heavy DKL B&CS", #...ts_resnet18_dkl_blur_cs_concat_s1_90.csv
    "M-driven SSN B&CS", #...ts_resnet18_dkl_lum_blur_cs_final.csv
    "P-driven SSN"#...ts_resnet18_dkl_rg_dvd_trans.csv
]

beh_sh_bias = [
    0.494, 
    0.534, 
    0.833, 
    0.802,
    0.424, 
    0.664, 
    0.681,
    0.678, 
    0.609, 
    0.756, 
    0.615,
    0.927, 
    0.527
]

accuracies = [
    0.293, 
    0.300, 
    0.493, 
    0.453,
    0.235, 
    0.309, 
    0.352,
    0.372, 
    0.285, 
    0.416, 
    0.310,
    0.491, 
    0.213
] 

# -------------------------
# Architecture grouping colors
# -------------------------
def get_color(model):
    if model.startswith("RGB"):
        return "#4C72B0"  # blue
    elif model.startswith("DVD"):
        return "#55A868"  # green
    elif "DKL" in model:
        return "#C44E52"  # red
    elif "SSN" in model:
        return "#8172B2"  # purple

colors = [get_color(m) for m in models]

offsets = {
    "DKL B": (-0.025, 0.020),
    "DKL CS": (-0.030, 0.020),
    "DKL B&CS": (0.010, 0.0),
    "M-heavy DKL B&CS": (0.01, 0.0),
    "Inter. DKL B&CS": (-0.050, 0.020),
    "Inter. M-heavy DKL B&CS": (0.010, -0.010),
}

fig, ax = plt.subplots(figsize=(8, 6))

for model, acc, sb, color in zip(models, accuracies, beh_sh_bias, colors):
    ax.scatter(acc, sb, s=120, color=color,
               edgecolor="black", linewidth=0.7)


    dx, dy = offsets.get(model, (0.005, 0))

    ax.text(acc + dx, sb + dy, model, fontsize=9)


ax.set_xlabel("Shape Classification Accuracy")
ax.set_ylabel("Shape Bias")
ax.set_xlim(0.15, 0.55)
ax.set_ylim(0.35, 1.0)

ax.grid(True, linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig("/share/klab/lzejin/mbenito/two_stream_dvd/src/visualizations/assets/behavioral_shape_bias.pdf", bbox_inches="tight")

plt.show()
