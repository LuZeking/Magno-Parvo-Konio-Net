import os
import shutil
import subprocess
import warnings

from .. import constants as c


def get_short_imagename(imagename):
    """Return image-specific suffix of imagename.

    This excludes a possible experiment-specific prefix,
    such as 0001_... for trial #1.
    """

    splits = imagename.split("_")
    if len(splits) > 1:
        name = splits[-2:]
        if name[0].startswith("n0"):
            # ImageNet image: keep n0... prefix
            name = name[0] + "_" + name[1]
        else:
            name = name[1]
    else:
        name = splits[0]
    return name


def read_data(path):
    """Read experimental data from csv file."""

    import pandas as pd

    assert os.path.exists(path)
    df = pd.read_csv(path)
    return df


def read_all_csv_files_from_directory(dir_path):
    import pandas as pd

    assert os.path.exists(dir_path)
    assert os.path.isdir(dir_path)

    df = pd.DataFrame()
    for f in sorted(os.listdir(dir_path)):
        if f.endswith(".csv"):
            df2 = read_data(os.path.join(dir_path, f))
            df2.columns = [c.lower() for c in df2.columns]
            df = pd.concat([df, df2])
    return df


def _candidate_experiment_paths(dataset_name):
    """Yield likely on-disk locations for a dataset's raw CSV directory."""

    raw_data_roots = [
        c.RAW_DATA_DIR,
        os.path.join(c.PROJ_DIR, "raw-data"),
        os.path.join(os.getcwd(), "raw-data"),
        os.path.join(c.PROJ_DIR, "results", "shape_bias", "raw-data"),
        os.path.join(os.getcwd(), "results", "shape_bias", "raw-data"),
    ]

    seen = set()
    for raw_root in raw_data_roots:
        candidate = os.path.abspath(os.path.join(raw_root, dataset_name))
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def find_experiment_path(dataset_name):
    """Return the first existing raw-data directory for ``dataset_name``."""

    for candidate in _candidate_experiment_paths(dataset_name):
        if os.path.isdir(candidate):
            return candidate
    return None


def get_experiment_path(dataset_name):
    """Resolve a dataset's raw-data directory across new and legacy layouts."""

    experiment_path = find_experiment_path(dataset_name)
    if experiment_path is not None:
        return experiment_path

    searched_paths = "\n".join(_candidate_experiment_paths(dataset_name))
    raise AssertionError(
        f"No raw-data directory found for '{dataset_name}'. Searched:\n{searched_paths}"
    )


def get_experimental_data(dataset, print_name=False):
    """Read all available data for an experiment."""

    if print_name:
        print(dataset.name)
    experiment_path = get_experiment_path(dataset.name)

    df = read_all_csv_files_from_directory(experiment_path)
    df.condition = df.condition.astype(str)

    for experiment in dataset.experiments:
        if not set(experiment.data_conditions).issubset(set(df.condition.unique())):
            print(set(experiment.data_conditions))
            print(set(df.condition.unique()))
            raise ValueError("Condition mismatch")

    df = df.copy()
    df["image_id"] = df["imagename"].apply(get_short_imagename)

    return df


def crop_pdfs_in_directory(dir_path, suppress_output=True):
    """Crop all PDF plots in a directory (removing white borders),

    Args:
        dir_path: path to directory
    """
    assert os.path.exists(dir_path)
    assert os.path.isdir(dir_path)

    pdfcrop = shutil.which("pdfcrop")
    if pdfcrop is None:
        warnings.warn("pdfcrop not found; skipping PDF cropping.", RuntimeWarning)
        return

    stdout = subprocess.DEVNULL if suppress_output else None
    stderr = subprocess.DEVNULL if suppress_output else None

    for file in sorted(os.listdir(dir_path)):
        if file.endswith(".pdf"):
            fullpath = os.path.join(dir_path, file)
            subprocess.run(
                [pdfcrop, fullpath, fullpath],
                check=False,
                stdout=stdout,
                stderr=stderr,
            )


##################################################################
# QUICK TESTS
##################################################################

assert get_short_imagename("0000_cl_s01_bw_boat_40_n02951358_5952.JPEG") == "n02951358_5952.JPEG"
assert get_short_imagename("airplane1-bicycle2.png") == "airplane1-bicycle2.png"
