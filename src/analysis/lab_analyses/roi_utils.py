import os
import pickle

import nibabel as nb
import numpy as np
from nsd_visuo_semantics.utils.nsd_get_data_light import (  # get_conditions_1000, get_conditions_485, get_subject_conditions
    get_conditions,
    get_conditions_100,
    get_conditions_515,
    load_or_compute_betas_average,
)
from scipy.spatial.distance import pdist


def get_rdm(data, rdm_distance, roi_name="name_not_given", verbose=True):

    if verbose:
        print(f"\tcomputing neural RDM for roi: {roi_name}")
    rdm = pdist(data, metric=rdm_distance)
    if np.any(np.isnan(rdm)):
        raise ValueError(f"nan found in RDM for ROI {roi_name}")
    return rdm


def get_subj_all_roi_rdms(subj, betas, ROIS, maskdata, rdm_distance):

    print(f"computing neural RDMs for {subj}")

    subj_rdms = {}

    for roi in range(1, len(ROIS)):

        roi_name = ROIS[roi]

        # maskdata is an array of shape [n_voxels,], with a number corresponding to the
        # ROI of each voxel (e.g. 0 means no ROI is associated with this voxel, 1 means voxel
        # is in ROIS[1] (EVC for example), etcetc).
        # so vs_mask is a logical array of mask vertices, with True in ROI vertices
        vs_mask = maskdata == roi

        # betas is [n_voxels, n_subj_conditions] (ordered by nsd_id, see thorough comments above).
        # masked betas is [n_roi_betas, n_conditions]
        masked_betas = betas[vs_mask, :]
        # remove vertices with a nan
        good_vox = [
            True if np.sum(np.isnan(x)) == 0 else False for x in masked_betas
        ]

        if np.sum(good_vox) != len(good_vox):
            print(f"found some NaN for ROI: {roi_name} - {subj}")
        masked_betas = masked_betas[good_vox, :]

        X = (
            masked_betas.T
        )  # [n_conditions, n_roi_betas], i.e., we make an [n_conditionsxn_conditions] rdm

        subj_rdms[roi_name] = get_rdm(X, rdm_distance, roi_name)

    return subj_rdms


def get_subj_all_roi_betas(subj, betas, ROIS, maskdata):

    print(f"Getting neural betas for {subj}")

    subj_betas = {}

    for roi in range(1, len(ROIS)):

        roi_name = ROIS[roi]

        # maskdata is an array of shape [n_voxels,], with a number corresponding to the
        # ROI of each voxel (e.g. 0 means no ROI is associated with this voxel, 1 means voxel
        # is in ROIS[1] (EVC for example), etcetc).
        # so vs_mask is a logical array of mask vertices, with True in ROI vertices
        vs_mask = maskdata == roi

        # betas is [n_voxels, n_subj_conditions] (ordered by nsd_id, see thorough comments above).
        # masked betas is [n_roi_betas, n_conditions]
        masked_betas = betas[vs_mask, :]
        # remove vertices with a nan
        good_vox = [
            True if np.sum(np.isnan(x)) == 0 else False for x in masked_betas
        ]

        if np.sum(good_vox) != len(good_vox):
            print(f"found some NaN for ROI: {roi_name} - {subj}")
        masked_betas = masked_betas[good_vox, :]

        X = (
            masked_betas.T
        )  # [n_conditions, n_roi_betas], i.e., we make an [n_conditionsxn_conditions] rdm

        subj_betas[roi_name] = X

    return subj_betas


def get_all_subj_all_roi_neural_rdms(
    nsd_dir,
    betas_dir,
    which_rois,
    rois_dir,
    which_data,
    rdm_distance,
    save_dir,
    targetspace,
):
    """Get all neural RDMs for all ROIs for all subjects.
    which_rois should refer to a NSD ROI definition (e.g. 'streams')
    rois_dir should be the path to the directory containing the ROIs.
    which_data should be either 'fullnsd', 'special1000', 'special515' or 'special100'.
    rdm_distance: 'correlation' or 'cosine', ...
    """

    all_rdms = {}

    n_sessions = 40
    n_subjects = 8
    subs = [f"subj0{x+1}" for x in range(n_subjects)]
    maskdata, ROIS = get_rois(which_rois, rois_dir)
    targetspace = "fsaverage"

    print(
        f'Computing neural RDMs for all subjects for the "{which_rois}" ROIs'
    )

    # if 'special1000' in which_data:
    #     print('\tUsing special 1000 conditions')
    #     subconds = np.asarray(get_conditions_1000(nsd_dir)).ravel()
    if "special515" in which_data:
        print("\tUsing special 515 conditions")
        subconds = np.asarray(get_conditions_515(nsd_dir)).ravel()
    elif "special100" in which_data:
        print("\tUsing special 100 conditions")
        subconds = np.asarray(get_conditions_100(nsd_dir)).ravel()
    elif "fullnsd" in which_data:
        print("\tUsing all conditions")
    else:
        raise ValueError(
            "which_data should be either all, special1000, special515 or special100"
        )

    for s_n, subj in enumerate(subs):

        conditions, conditions_sampled, sample = get_subject_conditions(
            nsd_dir, subj, n_sessions, keep_only_3repeats=True
        )

        # Betas per subject
        betas_file = os.path.join(
            betas_dir, f"{subj}_betas_average_{targetspace}.npy"
        )
        betas = load_or_compute_betas_average(
            betas_file,
            nsd_dir,
            subj,
            n_sessions,
            conditions,
            conditions_sampled,
            targetspace,
        )
        if which_data != "fullnsd":
            subinds = [x for x, j in enumerate(sample) if j in subconds]
            betas = betas[:, subinds]

        all_rdms[subj] = get_subj_all_roi_rdms(
            subj, betas, ROIS, maskdata, rdm_distance
        )

        del betas

    with open(
        f"{save_dir}/{which_rois}_all_neural_rdms_{rdm_distance}_{which_data}.pkl",
        "wb",
    ) as pickle_file:
        pickle.dump(all_rdms, pickle_file)

    return all_rdms


def get_all_subj_all_roi_neural_betas(
    nsd_dir, betas_dir, which_rois, rois_dir, which_data, save_dir, targetspace
):
    """Get all neural betas for all ROIs for all subjects.
    which_rois should refer to a NSD ROI definition (e.g. 'streams')
    rois_dir should be the path to the directory containing the ROIs.
    which_data should be either 'fullnsd', 'special1000', 'special515' or 'special100'.
    """

    all_betas = {}

    n_sessions = 40
    n_subjects = 8
    subs = [f"subj0{x+1}" for x in range(n_subjects)]
    maskdata, ROIS = get_rois(which_rois, rois_dir)
    targetspace = "fsaverage"

    print(f'Getting neural betas for all subjects for the "{which_rois}" ROIs')

    # if 'special1000' in which_data:
    #     print('\tUsing special 1000 conditions')
    #     subconds = np.asarray(get_conditions_1000(nsd_dir)).ravel()
    if "special515" in which_data:
        print("\tUsing special 515 conditions")
        subconds = np.asarray(get_conditions_515(nsd_dir)).ravel()
    # elif 'special485' in which_data:
    #     print('\tUsing special 515 conditions')
    #     subconds = np.asarray(get_conditions_485(nsd_dir)).ravel()
    elif "special100" in which_data:
        print("\tUsing special 100 conditions")
        subconds = np.asarray(get_conditions_100(nsd_dir)).ravel()
    elif "fullnsd" in which_data:
        print("\tUsing all conditions")
    else:
        raise ValueError(
            "which_data should be either all, special1000, special515 or special100"
        )

    for s_n, subj in enumerate(subs):

        conditions, conditions_sampled, sample = get_subject_conditions(
            nsd_dir, subj, n_sessions, keep_only_3repeats=True
        )

        # Betas per subject
        betas_file = os.path.join(
            betas_dir, f"{subj}_betas_average_{targetspace}.npy"
        )
        betas = load_or_compute_betas_average(
            betas_file,
            nsd_dir,
            subj,
            n_sessions,
            conditions,
            conditions_sampled,
            targetspace,
        )
        if which_data != "fullnsd":
            subinds = [x for x, j in enumerate(sample) if j in subconds]
            betas = betas[:, subinds]

        all_betas[subj] = get_subj_all_roi_betas(subj, betas, ROIS, maskdata)

        del betas

    with open(
        f"{save_dir}/{which_rois}_all_neural_betas_{which_data}.pkl", "wb"
    ) as pickle_file:
        pickle.dump(all_betas, pickle_file)

    return None


def get_subject_conditions(nsd_dir, subj, n_sessions, keep_only_3repeats=True):

    # extract conditions data.
    # NOTES ABOUT HOW THIS WORKS:
    # get_conditions returns a list with one item for each session the subject attended. Each of these items contains
    # the NSD_ids for the images presented in that session. Then, we reshape all this into a single array, which now
    # contains all the NSD_ids for the subject, in the order in which they were shown. Next, we create a boolean list of
    # the same size as the conditions array, which assigns True to NSD_ids that are present 3x in the condition array.
    # We use this boolean to create conditions_sampled, which now contains all NSD_indices for stimuli the subject has
    # seen 3x. This list still contains the 3 repetitions of each stimulus, and is still in the stimulus presentation
    # order. For example: [46003, 61883,   829, ...]
    # Hence, we need to only keep each NSD_id once (since we compute everything on the average fMRI data over
    # the 3 presentations), and we also need to order them in increasing NSD_id order (so that we can then easily
    # for all subjects/models). Both of these desiderata are addressed by using np.unique (which sorts the unique idx).
    # So sample contains the unique NSD_ids for that subject, in increasing order (e.g. [ 14,  28,  72, ...]).
    # Importantly, the average betas loaded above are arranged in the same way, so that if we want to find the betas
    # for NSD_id=72, we just need to find the idx of 72 in sample (in the present example: 2). Using this method, we can
    # find the avg_betas corresponding to the shared 515 images as done below with subj_indices_515 (hint: the trick to
    # go from an ordered list of nsd_ids to finding the idx as described above is to use enumerate).
    # For example sample[subj_indices_515[0]] = conditions_515[0].

    # extract conditions data
    conditions = get_conditions(nsd_dir, subj, n_sessions)
    # we also need to reshape conditions to be ntrials x 1
    conditions = np.asarray(conditions).ravel()
    if keep_only_3repeats:
        # then we find the valid trials for which we do have 3 repetitions.
        conditions_bool = [
            True if np.sum(conditions == x) == 3 else False for x in conditions
        ]
    else:
        conditions_bool = [True for x in conditions]
    # and identify those.
    conditions_sampled = conditions[conditions_bool]
    # find the subject's condition list (sample pool)
    # this sample is the same order as the betas
    sample = np.unique(conditions[conditions_bool])

    return conditions, conditions_sampled, sample


def get_rois(which_rois, roi_defs_dir):
    roi_names_file = os.path.join(roi_defs_dir, f"{which_rois}.mgz.ctab")
    try:
        with open(roi_names_file) as f:
            # get ROI names automatically. If you don't have the .ctab file
            # you can also enter them by hand. 0 is always "Unknown")
            roi_id2name = {int(x[0]): x[2:-1] for x in f}
    except FileNotFoundError:
        print(
            f"roi_names_file not found. Requested {roi_names_file}. Using {which_rois} as single ROI name."
        )
        roi_id2name = {0: "Unknown"}
        roi_id2name[1] = which_rois

    # load the roi masks
    try:
        lh_file = os.path.join(roi_defs_dir, f"lh.{which_rois}.mgz")
        rh_file = os.path.join(roi_defs_dir, f"rh.{which_rois}.mgz")
        maskdata_lh = nb.load(lh_file).get_fdata().squeeze()
        maskdata_rh = nb.load(rh_file).get_fdata().squeeze()
    except ValueError:
        lh_file = os.path.join(roi_defs_dir, f"lh.{which_rois}.npy")
        rh_file = os.path.join(roi_defs_dir, f"rh.{which_rois}.npy")
        maskdata_lh = np.load(lh_file, allow_pickle=True)
        maskdata_rh = np.load(rh_file, allow_pickle=True)

    maskdata = np.hstack((maskdata_lh, maskdata_rh))

    return maskdata, roi_id2name
