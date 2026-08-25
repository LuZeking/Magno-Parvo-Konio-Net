import shutil

import h5py
import numpy as np

from src.utils.helpers import convert_rgb_to_dkl_numpy, normalize

if __name__ == "__main__":

    src_path = "/share/klab/datasets/texture2shape_miniecoset_256px.h5"
    dst_path = "/share/klab/datasets/texture2shape_miniecoset_256px_dkl.h5"

    # Copy entire file (keeps structure perfectly)
    shutil.copy(src_path, dst_path)

    with h5py.File(dst_path, "r+") as f:
        # load the data
        X_train = f["train"]["data"]
        X_val = f["val"]["data"]
        X_test = f["test"]["data"]

        print(X_train.dtype)

        # create empty arrays with same shape
        X_train_dkl = np.zeros(X_train.shape)
        X_val_dkl = np.zeros(X_val.shape)
        X_test_dkl = np.zeros(X_test.shape)

        # augment
        print("Train Dataset:")
        for bs in range(0, X_train.shape[0], 100):
            print(f"Processing entries {bs} to {bs+100}")
            X_batch = X_train[bs : bs + 100]
            X_dkl_batch = convert_rgb_to_dkl_numpy(X_batch)
            X_train_dkl[bs : bs + 100] = X_dkl_batch

        print("Validation Dataset:")
        for bs in range(0, X_val.shape[0], 100):
            print(f"Processing entries {bs} to {bs+100}")
            X_batch = X_val[bs : bs + 100]
            X_dkl_batch = convert_rgb_to_dkl_numpy(X_batch)
            X_val_dkl[bs : bs + 100] = X_dkl_batch

        print("Test Dataset:")
        for bs in range(0, X_test.shape[0], 100):
            print(f"Processing entries {bs} to {bs+100}")
            X_batch = X_test[bs : bs + 100]
            X_dkl_batch = convert_rgb_to_dkl_numpy(X_batch)
            X_test_dkl[bs : bs + 100] = X_dkl_batch

        # normalize
        min_max = []
        # lum
        lum_min = min(
            X_train_dkl[:, :, :, 0].min(),
            X_val_dkl[:, :, :, 0].min(),
            X_test_dkl[:, :, :, 0].min(),
        )
        lum_max = max(
            X_train_dkl[:, :, :, 0].max(),
            X_val_dkl[:, :, :, 0].max(),
            X_test_dkl[:, :, :, 0].max(),
        )
        min_max.append((lum_min, lum_max))
        # rg
        rg_min = min(
            X_train_dkl[:, :, :, 1].min(),
            X_val_dkl[:, :, :, 1].min(),
            X_test_dkl[:, :, :, 1].min(),
        )
        rg_max = max(
            X_train_dkl[:, :, :, 1].max(),
            X_val_dkl[:, :, :, 1].max(),
            X_test_dkl[:, :, :, 1].max(),
        )
        min_max.append((rg_min, rg_max))
        # by
        by_min = min(
            X_train_dkl[:, :, :, 2].min(),
            X_val_dkl[:, :, :, 2].min(),
            X_test_dkl[:, :, :, 2].min(),
        )
        by_max = max(
            X_train_dkl[:, :, :, 2].max(),
            X_val_dkl[:, :, :, 2].max(),
            X_test_dkl[:, :, :, 2].max(),
        )
        min_max.append((by_min, by_max))

        print("Train Dataset Normalization:")
        for i in range(X_train_dkl.shape[3]):
            channel = X_train_dkl[:, :, :, i]
            c_min, c_max = min_max[i]
            for bs in range(0, channel.shape[0], 100):
                print(f"Normalizing entries {bs} to {bs+100}")
                channel_batch = channel[bs : bs + 100]
                normalized_batch = normalize(channel_batch, c_min, c_max)
                channel[bs : bs + 100] = normalized_batch
            X_train_dkl[:, :, :, i] = channel

        print("Validation Dataset Normalization:")
        for i in range(X_val_dkl.shape[3]):
            channel = X_val_dkl[:, :, :, i]
            c_min, c_max = min_max[i]
            for bs in range(0, channel.shape[0], 100):
                print(f"Normalizing entries {bs} to {bs+100}")
                channel_batch = channel[bs : bs + 100]
                normalized_batch = normalize(channel_batch, c_min, c_max)
                channel[bs : bs + 100] = normalized_batch
            X_val_dkl[:, :, :, i] = channel

        print("Test Dataset Normalization:")
        for i in range(X_test_dkl.shape[3]):
            channel = X_test_dkl[:, :, :, i]
            c_min, c_max = min_max[i]
            for bs in range(0, channel.shape[0], 100):
                print(f"Normalizing entries {bs} to {bs+100}")
                channel_batch = channel[bs : bs + 100]
                normalized_batch = normalize(channel_batch, c_min, c_max)
                channel[bs : bs + 100] = normalized_batch
            X_test_dkl[:, :, :, i] = channel

        # overwrite datasets
        del f["train"]["data"]
        del f["val"]["data"]
        del f["test"]["data"]

        f["train"].create_dataset("data", data=X_train_dkl)
        f["val"].create_dataset("data", data=X_val_dkl)
        f["test"].create_dataset("data", data=X_test_dkl)
