"""
Script to visualize the training and testing metrics of the two stream network models.
"""

import os
from pathlib import Path

from lightning_kietzmannlab.metrics import plot_npz_files_interactive

if __name__ == "__main__":

    # directories
    npz_directory_current = Path(
        "/share/klab/lzejin/mbenito/two_stream_dvd/logs/final_ts_dkl_base_lum_rg"
    )

    npz_file_current = [
        npz_directory_current / file
        for file in os.listdir(npz_directory_current)
        if file.endswith(".npz")
    ][-1]

    filepaths = [npz_file_current]

    plot_npz_files_interactive(
        filepaths, save_plots=True, page_output_dir=npz_directory_current.stem
    )
