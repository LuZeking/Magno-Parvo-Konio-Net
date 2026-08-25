import matplotlib.pyplot as plt

epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
          80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]

# data manually retrieved from output files that logged the accuracies during training
# these files are to find under: /share/klab/lzejin/mbenito/two_stream_dvd/output&error/...
accuracies = {"DKL Base": [0.196, 0.448, 0.529, 0.607, 0.633, 0.643, 0.634, 0.642, 0.656, 0.678, 0.691, 0.695, 0.697, 0.713, 0.660, 0.690, 0.719, 0.702, 0.693, 0.695, 0.688, 0.694, 0.699, 0.707, 0.701, 0.714, 0.716, 0.711, 0.702, 0.698, 0.673], # ...final_ts_dkl_base_lum_rg.output
              "DKL B": [0.102, 0.331, 0.483, 0.491, 0.570, 0.579, 0.527, 0.609, 0.603, 0.606, 0.572, 0.544, 0.532, 0.660, 0.628, 0.526, 0.580, 0.629, 0.572, 0.638, 0.655, 0.633, 0.640, 0.569, 0.567, 0.601, 0.631, 0.671, 0.660, 0.547, 0.532], # ...final_ts_dkl_blur.output
              "DKL CS": [0.126, 0.269, 0.215, 0.408, 0.474, 0.522, 0.421, 0.565, 0.455, 0.544, 0.575, 0.455, 0.528, 0.584, 0.394, 0.480, 0.597, 0.323, 0.562, 0.417, 0.558, 0.602, 0.589, 0.450, 0.484, 0.567, 0.428, 0.567, 0.504, 0.517, 0.585], # ...final_ts_dkl_contrast.output
              "DKL B&CS": [0.073, 0.151, 0.186, 0.280, 0.462, 0.513, 0.430, 0.539, 0.477, 0.547, 0.583, 0.388, 0.519, 0.589, 0.515, 0.566, 0.585, 0.580, 0.407, 0.516, 0.535, 0.565, 0.572, 0.551, 0.434, 0.489, 0.430, 0.428, 0.586, 0.532, 0.469], # ...dkl_blur_cs_dvd_trans.output
              "M-heavy DKL B&CS": [0.117, 0.234, 0.267, 0.365, 0.454, 0.428, 0.343, 0.514, 0.408, 0.524, 0.485, 0.407, 0.481, 0.560, 0.434, 0.503, 0.576, 0.373, 0.519, 0.503, 0.463, 0.516, 0.576, 0.407, 0.460, 0.570, 0.459, 0.545, 0.382, 0.542, 0.524], # ...dkl_blur_cs_s1_90.output
              "Inter. DKL B&CS": [0.120, 0.211, 0.374, 0.382, 0.463, 0.535, 0.504, 0.488, 0.539, 0.526, 0.478, 0.610, 0.625, 0.611, 0.447, 0.601, 0.618, 0.427, 0.523, 0.567, 0.494, 0.552, 0.627, 0.537, 0.566, 0.445, 0.587, 0.531, 0.525, 0.536, 0.611], # ...dkl_blur_cs_concat.output
              "Inter. M-heavy DKL B&CS": [0.061, 0.267, 0.301, 0.534, 0.492, 0.563, 0.447, 0.509, 0.341, 0.530, 0.470, 0.482, 0.622, 0.417, 0.528, 0.574, 0.528, 0.517, 0.594, 0.592, 0.570, 0.531, 0.589, 0.576, 0.522, 0.426, 0.628, 0.615, 0.468, 0.535, 0.582], # ...dkl_blur_cs_concat_s1_90.output
              "M-driven SSN B&CS": [0.110, 0.229, 0.287, 0.303, 0.331, 0.340, 0.352, 0.356, 0.363, 0.365, 0.372, 0.370, 0.368, 0.369, 0.370, 0.373, 0.368, 0.369, 0.367, 0.367, 0.366, 0.369, 0.367, 0.363, 0.364, 0.361, 0.363, 0.358, 0.355, 0.352, 0.354], # ...final_ss_dkl_lum_blur_cs.output
              "P-driven SSN": [0.092, 0.156, 0.211, 0.332, 0.374, 0.505, 0.404, 0.374, 0.479, 0.504, 0.392, 0.518, 0.468, 0.459, 0.390, 0.320, 0.309, 0.481, 0.510, 0.448, 0.470, 0.484, 0.434, 0.511, 0.545, 0.533, 0.463, 0.433, 0.450, 0.502, 0.445] # ...dkl_rg_dvd_trans.output
              }


# -------------------------
# Styling
# -------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif"
})

# -------------------------
# Color mapping
# -------------------------
def get_color(model):
    if "DKL" in model:
        return "#C44E52"   # red
    elif "SSN" in model:
        return "#8172B2"   # purple
    else:
        return "#4C72B0"

# -------------------------
# Create 3x3 figure
# -------------------------
fig, axes = plt.subplots(
    3, 3,
    figsize=(12, 8),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, (model, values) in zip(axes, accuracies.items()):

    color = get_color(model)

    # thin connecting line
    ax.plot(
        epochs,
        values,
        color=color,
        linewidth=1.2,
        alpha=0.7
    )

    # scatter points
    ax.scatter(
        epochs,
        values,
        color=color,
        edgecolors="black",
        linewidths=0.4,
        s=35,
        zorder=3
    )

    # title
    ax.set_title(model)

    # grid
    ax.grid(
        linestyle="--",
        alpha=0.4
    )
    ax.set_axisbelow(True)

    # remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# -------------------------
# axis labels
# -------------------------
fig.supxlabel("Epoch")
fig.supylabel("Validation Accuracy")

# Axis limits and ticks
for ax in axes:
    ax.set_xlim(0, 152)
    ax.set_ylim(0.0, 0.8)

plt.tight_layout()

plt.savefig(
    "/share/klab/lzejin/mbenito/two_stream_dvd/src/visualizations/assets/training_curves_3x3.pdf",
    bbox_inches="tight"
)

plt.show()