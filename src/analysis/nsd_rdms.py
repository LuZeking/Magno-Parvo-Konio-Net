"""
Script to calculate the neural RDMs for all subjects for NSD.
Change the 'which_rois' variable to select different ROIs.
"""

import os

from src.analysis.lab_analyses.roi_utils import (
    get_all_subj_all_roi_neural_rdms,
)

if __name__ == "__main__":

    # directories
    nsd_dir = "/share/klab/datasets/NSD_for_visuo_semantics"
    rois_dir = os.path.join(nsd_dir, "nsddata/freesurfer/fsaverage/label")
    nsd_derivatives_dir = (
        "/share/klab/datasets/NSD_for_visuo_semantics_derivatives/"
    )
    betas_dir = f"{nsd_derivatives_dir}/betas"
    save_dir = "/share/klab/lzejin/mbenito/two_stream_dvd"

    # variables
    targetspace = "fsaverage"
    which_data = "special515"
    rdm_distance = "correlation"
    which_rois = [
        "probmap_V1v",
        "probmap_V2v",
        "probmap_V3v",
        "probmap_V1d",
        "probmap_V2d",
        "probmap_V3d",
        "probmap_hV4",
        "probmap_PPA",
        # "streams"
    ]

    for roi in which_rois:
        get_all_subj_all_roi_neural_rdms(
            nsd_dir,
            betas_dir,
            roi,
            rois_dir,
            which_data,
            rdm_distance,
            save_dir,
            targetspace,
        )
