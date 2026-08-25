import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data
# -------------------------
# data manually retrieved from output files that logged the accuracies during training
# these files are to find under: /share/klab/lzejin/mbenito/two_stream_dvd/output&error/...
models = [
    "RGB Baseline", #...final_ss_resnet_50.output
    "RGB TSN", #...final_ts_resnet_18.output
    "DVD Baseline", #...baseline_dvd_sf_dvd_trans.output
    "DVD TSN", #...ts_dvd.output
    "DKL Base",  # ...final_ts_dkl_base_lum_rg.output
    "DKL B", # ...final_ts_dkl_blur.output
    "DKL CS",# ...final_ts_dkl_contrast.output
    "DKL B&CS",# ...dkl_blur_cs_dvd_trans.output
    "M-heavy DKL B&CS",# ...dkl_blur_cs_s1_90.output
    "Inter. DKL B&CS", # ...dkl_blur_cs_concat.output
    "Inter. M-heavy DKL B&CS",# ...dkl_blur_cs_concat_s1_90.output
    "M-driven SSN B&CS", # ...final_ss_dkl_lum_blur_cs.output
    "P-driven SSN"# ...dkl_rg_dvd_trans.output
]

val_accuracy = [
    0.757, 0.721, 0.672, 0.672,
    0.673, 0.532, 0.585,
    0.469, 0.524, 0.611, 0.582,
    0.354, 0.445
]

# -------------------------
# Reverse order
# -------------------------
models = models[::-1]
val_accuracy = val_accuracy[::-1]

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

# -------------------------
# Styling
# -------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "sans-serif"
})

fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.barh(
    models,
    val_accuracy,
    color=colors,
    edgecolor="black",
    linewidth=0.6
)

# -------------------------
# Labels & formatting
# -------------------------
ax.set_title("Validation Accuracy Across Models")
ax.set_xlabel("Top-1 Accuracy")

ax.set_xlim(0, 0.85)

# Grid
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

# Remove spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# -------------------------
# Group separators
# -------------------------
group_breaks = [
    "DVD Baseline",
    "DKL Base",
    "M-driven SSN B&CS"
]

for i, model in enumerate(models):
    if model in group_breaks:
        ax.axhline(i + 0.5, color="black", linewidth=0.6, alpha=0.3)

# -------------------------
# Value labels
# -------------------------
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center",
        fontsize=10
    )

plt.tight_layout()

plt.savefig("/share/klab/lzejin/mbenito/two_stream_dvd/src/visualizations/assets/validation_accuracy.pdf", bbox_inches="tight")

plt.show()