"""
Script to compare the model and neural RDMs.
"""

import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def load_ann_rdms(ann_file):
    "Load the ANN RDMs (if it is the baseline model stream1 will be filled and stream2 will be empty)"

    stream1_rdms = {}
    stream2_rdms = {}

    with open(
        f"/share/klab/lzejin/mbenito/two_stream_dvd/RDMs/{ann_file}",
        # f"RDMs/{ann_file}", # local testing
        "rb",
    ) as f:
        RDMs = pickle.load(f)

    for key, value in RDMs.items():
        if "stream1" in key:
            stream1_rdms[key] = value
        elif "stream2" in key:
            stream2_rdms[key] = value

    return stream1_rdms, stream2_rdms


def load_nsd_rdms(ventral_rois=[], dorsal_rois=[], streams_rois=[]):
    "Load the neural RDMs for the given ROIs"

    ventral_rdms = {}
    dorsal_rdms = {}
    streams_rdms = {}

    for roi in ventral_rois + dorsal_rois + streams_rois:
        nsd_file = f"{roi}_all_neural_rdms_correlation_special515.pkl"

        with open(
            f"/share/klab/lzejin/mbenito/two_stream_dvd/RDMs/{nsd_file}",
            # f"RDMs/{nsd_file}", # local testing
            "rb",
        ) as f:
            RDMs = pickle.load(f)

        if roi == "streams":
            for _, dictionary in RDMs.items():
                for area, rdm in dictionary.items():
                    if area not in streams_rdms:
                        streams_rdms[area] = []
                    streams_rdms[area].append(rdm)

            for area in streams_rdms.keys():
                streams_rdms[area] = np.mean(streams_rdms[area], axis=0)

        else:
            subject_rdms = []
            for _, v in RDMs.items():
                for _, value in v.items():
                    subject_rdms.append(value)

            mean_rdm = np.mean(subject_rdms, axis=0)

            if roi in ventral_rois:
                ventral_rdms[roi] = mean_rdm
            else:
                dorsal_rdms[roi] = mean_rdm

    return ventral_rdms, dorsal_rdms, streams_rdms


def calculate_noise_ceilings(ventral_rois=[], dorsal_rois=[], streams_rois=[]):
    # the lower noise ceiling is the average distance between a subjects RDM and the mean RDM of the other subjects
    ventral_lnc = {}
    dorsal_lnc = {}
    streams_lnc = {}

    for roi in ventral_rois + dorsal_rois + streams_rois:
        nsd_file = f"{roi}_all_neural_rdms_correlation_special515.pkl"

        with open(
            f"/share/klab/lzejin/mbenito/two_stream_dvd/RDMs/{nsd_file}",
            # f"RDMs/{nsd_file}", # local testing
            "rb",
        ) as f:
            RDMs = pickle.load(f)

        if roi == "streams":
            streams_rdms = {}

            for _, dictionary in RDMs.items():
                for area, rdm in dictionary.items():
                    if area not in streams_rdms:
                        streams_rdms[area] = []
                    streams_rdms[area].append(rdm)

            for area in streams_rdms.keys():
                correlations = []
                for i, subject_rdm in enumerate(streams_rdms[area]):
                    other_rdms = (
                        streams_rdms[area][:i] + streams_rdms[area][i + 1 :]
                    )
                    other_rdms_mean = np.mean(other_rdms, axis=0)
                    corr, _ = spearmanr(subject_rdm, other_rdms_mean)
                    correlations.append(corr)
                mean_corr = np.mean(correlations)
                streams_lnc[area] = mean_corr

        else:
            # get all subject rdms for this roi
            subject_rdms = []
            correlations = []
            for _, v in RDMs.items():
                for _, value in v.items():
                    subject_rdms.append(value)

            # calculate the correlations
            for i, subject_rdm in enumerate(subject_rdms):
                other_rdms = subject_rdms[:i] + subject_rdms[i + 1 :]
                other_rdms_mean = np.mean(other_rdms, axis=0)
                corr, _ = spearmanr(subject_rdm, other_rdms_mean)
                correlations.append(corr)

            mean_corr = np.mean(correlations)

            if roi in ventral_rois:
                ventral_lnc[roi] = mean_corr
            else:
                dorsal_lnc[roi] = mean_corr

    return ventral_lnc, dorsal_lnc, streams_lnc


def compare_specified_regions_rdms(
    stream1_rdms={},
    stream2_rdms={},
    ventral_rdms={},
    dorsal_rdms={},
):
    "Compare the specified ventral&dorsal RDMs with the ANN's using scipy spearmanr"

    st1_v_corrs = {}
    st1_d_corrs = {}
    st2_v_corrs = {}
    st2_d_corrs = {}

    # stream 1 spearman correlation
    for st1_key, st1_rdm in stream1_rdms.items():
        for ventral_key, ventral_rdm in ventral_rdms.items():
            corr, _ = spearmanr(st1_rdm, ventral_rdm)
            st1_v_corrs[(st1_key, ventral_key)] = corr

        for dorsal_key, dorsal_rdm in dorsal_rdms.items():
            corr, _ = spearmanr(st1_rdm, dorsal_rdm)
            st1_d_corrs[(st1_key, dorsal_key)] = corr

    # stream 2 spearman correlation
    for st2_key, st2_rdm in stream2_rdms.items():
        for ventral_key, ventral_rdm in ventral_rdms.items():
            corr, _ = spearmanr(st2_rdm, ventral_rdm)
            st2_v_corrs[(st2_key, ventral_key)] = corr

        for dorsal_key, dorsal_rdm in dorsal_rdms.items():
            corr, _ = spearmanr(st2_rdm, dorsal_rdm)
            st2_d_corrs[(st2_key, dorsal_key)] = corr

    return (
        st1_v_corrs,
        st1_d_corrs,
        st2_v_corrs,
        st2_d_corrs,
    )


def compare_streams_rdms(
    stream1_rdms={},
    stream2_rdms={},
    streams_rdms={},
):
    "Compare the streams RDMs with the ANN's using scipy spearmanr"

    st1_s_corrs = {}
    st2_s_corrs = {}

    # stream 1 spearman correlation
    for st1_key, st1_rdm in stream1_rdms.items():
        for stream_key, stream_rdm in streams_rdms.items():
            corr, _ = spearmanr(st1_rdm, stream_rdm)
            st1_s_corrs[(st1_key, stream_key)] = corr

    # stream 2 spearman correlation
    for st2_key, st2_rdm in stream2_rdms.items():
        for stream_key, stream_rdm in streams_rdms.items():
            corr, _ = spearmanr(st2_rdm, stream_rdm)
            st2_s_corrs[(st2_key, stream_key)] = corr

    return st1_s_corrs, st2_s_corrs


def plot_specified_regions_data(
    st1_v_corrs={},
    st1_d_corrs={},
    st2_v_corrs={},
    st2_d_corrs={},
    ventral_lnc={},
    dorsal_lnc={},
    ventral_rois=[],
    dorsal_rois=[],
    baseline=False,
    png_name="rdm_comparison.png",
):
    "Plot the correlation with matplotlib"

    plot_data = {}
    for ventral_roi in ventral_rois:
        plot_data[ventral_roi] = {
            "stream1": [
                st1_v_corrs.get((key[0], ventral_roi), None)
                for key in st1_v_corrs.keys()
                if key[1] == ventral_roi
            ],
            "stream2": [
                st2_v_corrs.get((key[0], ventral_roi), None)
                for key in st2_v_corrs.keys()
                if key[1] == ventral_roi
            ],
        }

    for dorsal_roi in dorsal_rois:
        plot_data[dorsal_roi] = {
            "stream1": [
                st1_d_corrs.get((key[0], dorsal_roi), None)
                for key in st1_d_corrs.keys()
                if key[1] == dorsal_roi
            ],
            "stream2": [
                st2_d_corrs.get((key[0], dorsal_roi), None)
                for key in st2_d_corrs.keys()
                if key[1] == dorsal_roi
            ],
        }

    print("Plot data:")
    for roi, data in plot_data.items():
        print(f"ROI: {roi}")
        for key, values in data.items():
            print(f"  {key}: {values}")

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    for i, ventral_roi in enumerate(ventral_rois):
        if not baseline:
            axs[0, i].scatter(
                range(len(plot_data[ventral_roi]["stream1"])),
                plot_data[ventral_roi]["stream1"],
                label="S1",
            )
            axs[0, i].scatter(
                range(len(plot_data[ventral_roi]["stream2"])),
                plot_data[ventral_roi]["stream2"],
                label="S2",
            )
        else:
            axs[0, i].scatter(
                range(len(plot_data[ventral_roi]["stream1"])),
                plot_data[ventral_roi]["stream1"],
                label="Bl",
            )
        axs[0, i].hlines(
            y=ventral_lnc[ventral_roi],
            xmin=0,
            xmax=len(plot_data[ventral_roi]["stream1"]) - 1,
            linestyle="--",
            label="LNC",
        )
        axs[0, i].set_title(f"Ventral ROI: {ventral_roi}")
        if i == 0:
            axs[0, i].set_ylabel("Spearman Correlation")
            axs[0, i].legend()
    for i, dorsal_roi in enumerate(dorsal_rois):
        if not baseline:
            axs[1, i].scatter(
                range(len(plot_data[dorsal_roi]["stream1"])),
                plot_data[dorsal_roi]["stream1"],
                label="S1",
            )
            axs[1, i].scatter(
                range(len(plot_data[dorsal_roi]["stream2"])),
                plot_data[dorsal_roi]["stream2"],
                label="S2",
            )
        else:
            axs[1, i].scatter(
                range(len(plot_data[dorsal_roi]["stream1"])),
                plot_data[dorsal_roi]["stream1"],
                label="Bl",
            )
        axs[1, i].hlines(
            y=dorsal_lnc[dorsal_roi],
            xmin=0,
            xmax=len(plot_data[dorsal_roi]["stream1"]) - 1,
            linestyle="--",
            label="LNC",
        )
        axs[1, i].set_title(f"Dorsal ROI: {dorsal_roi}")
        axs[1, i].set_xlabel("Layer Position")
        if i == 0:
            axs[1, i].set_ylabel("Spearman Correlation")
            axs[1, i].legend()
    png_name = png_name.replace(".png", "_specified_regions.png")
    plt.savefig(png_name)


# def plot_streams_data(
#     st1_s_corrs={},
#     st2_s_corrs={},
#     streams_lnc={},
#     baseline=False,
#     png_name="rdm_comparison.png",
# ):
#     "Plot the correlation with matplotlib"
#     plot_data = {}

#     for area in streams_lnc.keys():
#         plot_data[area] = {
#             "stream1": [
#                 st1_s_corrs.get((key[0], area), None)
#                 for key in st1_s_corrs.keys()
#                 if key[1] == area
#             ],
#             "stream2": [
#                 st2_s_corrs.get((key[0], area), None)
#                 for key in st2_s_corrs.keys()
#                 if key[1] == area
#             ],
#         }

#     fig, axs = plt.subplots(1, len(streams_lnc.keys()), figsize=(20, 3), sharey=True)
#     for i, area in enumerate(streams_lnc.keys()):
#         if not baseline:
#             axs[i].scatter(
#                 range(len(plot_data[area]["stream1"])),
#                 plot_data[area]["stream1"],
#                 label="S1",
#             )
#             axs[i].scatter(
#                 range(len(plot_data[area]["stream2"])),
#                 plot_data[area]["stream2"],
#                 label="S2",
#             )
#         else:
#             axs[i].scatter(
#                 range(len(plot_data[area]["stream1"])),
#                 plot_data[area]["stream1"],
#                 label="Bl",
#             )
#         axs[i].hlines(
#             y=streams_lnc[area],
#             xmin=0,
#             xmax=len(plot_data[area]["stream1"]) - 1,
#             linestyle="--",
#             label="LNC",
#         )
#         axs[i].set_title(f"{area}", )
#         if i == 0:
#             axs[i].set_ylabel("Spearman Correlation")
#             axs[i].legend()


#     png_name = png_name.replace(".png", "_streams.png")
#     plt.savefig(png_name)
def plot_streams_data(
    st1_s_corrs={},
    st2_s_corrs={},
    streams_lnc={},
    baseline=False,
    png_name="rdm_comparison.png",
):

    plot_data = {}

    for area in streams_lnc.keys():
        plot_data[area] = {
            "stream1": [
                st1_s_corrs.get((key[0], area), None)
                for key in st1_s_corrs.keys()
                if key[1] == area
            ],
            "stream2": [
                st2_s_corrs.get((key[0], area), None)
                for key in st2_s_corrs.keys()
                if key[1] == area
            ],
        }

    fig, axs = plt.subplots(
        1, len(streams_lnc.keys()), figsize=(18, 4), sharey=True
    )

    if len(streams_lnc) == 1:
        axs = [axs]

    for i, area in enumerate(streams_lnc.keys()):

        x = range(len(plot_data[area]["stream1"]))

        if not baseline:
            axs[i].plot(
                x,
                plot_data[area]["stream1"],
                marker="o",
                linewidth=2,
                markersize=5,
                label="Magno-driven" if "dkl" in png_name else "Stream 1",
            )

            axs[i].plot(
                x,
                plot_data[area]["stream2"],
                marker="s",
                linewidth=2,
                markersize=5,
                label="Parvo-driven" if "dkl" in png_name else "Stream 2",
            )

        else:
            axs[i].plot(
                x,
                plot_data[area]["stream1"],
                marker="o",
                linewidth=2,
                markersize=5,
                label="Stream 1",
            )

        axs[i].axhline(
            y=streams_lnc[area],
            linestyle="--",
            linewidth=2,
            label="Lower Noise Ceiling",
        )

        axs[i].set_title(area, fontsize=13)
        axs[i].set_xlabel("Layer Position")
        axs[i].grid(alpha=0.3)

        # cleaner look
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

    axs[0].set_ylabel("Spearman Correlation", fontsize=12)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    png_name = png_name.replace(".png", "_streams.png")
    plt.savefig(png_name, dpi=300, bbox_inches="tight")


def save_results_to_csv(
    st1_v_corrs={},
    st1_d_corrs={},
    st2_v_corrs={},
    st2_d_corrs={},
    st1_s_corrs={},
    st2_s_corrs={},
    output_dir="",
):
    "Save all the correlations to a csv"

    # Ventral & Dorsal
    rows = []

    # Stream 1
    for (model_layer, ventral_region), corr in st1_v_corrs.items():
        rows.append(
            {
                "stream": "stream1",
                "region": ventral_region,
                "layer": model_layer,
                "correlation": corr,
            }
        )

    for (model_layer, dorsal_region), corr in st1_d_corrs.items():
        rows.append(
            {
                "stream": "stream1",
                "region": dorsal_region,
                "layer": model_layer,
                "correlation": corr,
            }
        )

    # Stream 2
    for (model_layer, ventral_region), corr in st2_v_corrs.items():
        rows.append(
            {
                "stream": "stream2",
                "region": ventral_region,
                "layer": model_layer,
                "correlation": corr,
            }
        )

    for (model_layer, dorsal_region), corr in st2_d_corrs.items():
        rows.append(
            {
                "stream": "stream2",
                "region": dorsal_region,
                "layer": model_layer,
                "correlation": corr,
            }
        )

    output_dir_v_d = output_dir.replace(".png", "_specified_regions.csv")
    with open(output_dir_v_d, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stream", "region", "layer", "correlation"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Streams
    rows = []

    for (model_layer, streams_region), corr in st1_s_corrs.items():
        rows.append(
            {
                "stream": "stream1",
                "region": streams_region,
                "layer": model_layer,
                "correlation": corr,
            }
        )

    for (model_layer, streams_region), corr in st2_s_corrs.items():
        rows.append(
            {
                "stream": "stream2",
                "region": streams_region,
                "layer": model_layer,
                "correlation": corr,
            }
        )

    output_dir_s = output_dir.replace(".png", "_streams.csv")
    with open(output_dir_s, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stream", "region", "layer", "correlation"],
        )
        writer.writeheader()
        writer.writerows(rows)


def full_rdms_comparison(
    ann_file,
    ventral_rois,
    dorsal_rois,
    streams_rois,
    baseline=False,
    png_name="rdm_comparison.png",
):
    "Full rdm comparison pipeline"

    # load ANN RDMs
    stream1_rdms, stream2_rdms = load_ann_rdms(ann_file)

    # load NSD RDMs
    ventral_rdms, dorsal_rdms, streams_rdms = load_nsd_rdms(
        ventral_rois, dorsal_rois, streams_rois
    )

    # calculate lower noise ceiling for each ROI
    ventral_lnc, dorsal_lnc, streams_lnc = calculate_noise_ceilings(
        ventral_rois, dorsal_rois, streams_rois
    )

    # compare specified ventral&dorsal RDMs with the ANN's using Spearman correlation
    (
        st1_v_corrs,
        st1_d_corrs,
        st2_v_corrs,
        st2_d_corrs,
    ) = compare_specified_regions_rdms(
        stream1_rdms, stream2_rdms, ventral_rdms, dorsal_rdms
    )

    # compare streams RDMs with the ANN's using Spearman correlation
    st1_s_corrs, st2_s_corrs = compare_streams_rdms(
        stream1_rdms, stream2_rdms, streams_rdms
    )

    # plot the results
    plot_specified_regions_data(
        st1_v_corrs,
        st1_d_corrs,
        st2_v_corrs,
        st2_d_corrs,
        ventral_lnc,
        dorsal_lnc,
        ventral_rois,
        dorsal_rois,
        baseline,
        png_name,
    )

    plot_streams_data(
        st1_s_corrs,
        st2_s_corrs,
        streams_lnc,
        baseline,
        png_name,
    )

    save_results_to_csv(
        st1_v_corrs,
        st1_d_corrs,
        st2_v_corrs,
        st2_d_corrs,
        st1_s_corrs,
        st2_s_corrs,
        png_name,
    )


if __name__ == "__main__":
    # directories
    ann_file = "ts_dvd_nsd_resnet18_seperate_from_layer0_RDMs.pkl"
    png_name = "/share/klab/lzejin/mbenito/two_stream_dvd/RDMs/ts_dvd_nsd_resnet18_seperate_from_layer0_RDMs.png"

    # variables
    baseline = False
    ventral_rois = [
        "probmap_V1v",
        "probmap_V2v",
        "probmap_V3v",
        "probmap_hV4",
        "probmap_PPA",
    ]
    dorsal_rois = [
        "probmap_V1d",
        "probmap_V2d",
        "probmap_V3d",
    ]
    streams_rois = ["streams"]

    full_rdms_comparison(
        ann_file, ventral_rois, dorsal_rois, streams_rois, baseline, png_name
    )
