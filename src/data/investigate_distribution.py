import h5py
import matplotlib.pyplot as plt
import numpy as np


def get_data():
    # read in the dataset
    dataset = h5py.File(
        "/share/klab/datasets/texture2shape_miniecoset_256px_dkl.h5", "r"
    )

    train_data = dataset["train"]["data"]
    print(f"Length of Train Dataset: {len(train_data)}")
    val_data = dataset["val"]["data"]
    print(f"Length of Validation Dataset: {len(val_data)}")

    return train_data, val_data


def get_histograms_per_channel(data, batch_step=1000):
    # create absolute count arrays
    lum = np.zeros(100)
    rg = np.zeros(100)
    by = np.zeros(100)

    # process batch wise
    for b in range(0, data.shape[0], batch_step):
        print(f"Processing entries {b} to {b+batch_step}")
        # get channels
        lum_batch = data[b : b + batch_step, :, :, 0]
        rg_batch = data[b : b + batch_step, :, :, 1]
        by_batch = data[b : b + batch_step, :, :, 2]

        # flatten the data
        lum_batch = lum_batch.reshape(lum_batch.shape[0], -1)
        rg_batch = rg_batch.reshape(rg_batch.shape[0], -1)
        by_batch = by_batch.reshape(by_batch.shape[0], -1)

        # get the histograms
        lum_histogram, _ = np.histogram(lum_batch, 100, (0.0, 1.0))
        rg_histogram, _ = np.histogram(rg_batch, 100, (0.0, 1.0))
        by_histogram, _ = np.histogram(by_batch, 100, (0.0, 1.0))

        # ad to the absolute counts
        lum += lum_histogram
        rg += rg_histogram
        by += by_histogram

    # normalize the histograms
    lum /= lum.sum()
    rg /= rg.sum()
    by /= by.sum()

    return lum, rg, by


def main():

    train_data, val_data = get_data()

    lum_train, rg_train, by_train = get_histograms_per_channel(train_data)

    lum_val, rg_val, by_val = get_histograms_per_channel(val_data)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(np.linspace(0, 1, 100), lum_train, label="Training Data")
    axs[0].plot(np.linspace(0, 1, 100), lum_val, label="Validation Data")
    axs[0].set_title("Lum Channel")
    axs[0].legend()

    axs[1].plot(np.linspace(0, 1, 100), rg_train, label="Training Data")
    axs[1].plot(np.linspace(0, 1, 100), rg_val, label="Validation Data")
    axs[1].set_title("RG Channel")
    axs[1].legend()

    axs[2].plot(np.linspace(0, 1, 100), by_train, label="Training Data")
    axs[2].plot(np.linspace(0, 1, 100), by_val, label="Validation Data")
    axs[2].set_title("BY Channel")
    axs[2].legend()

    fig.savefig(
        f"/share/klab/lzejin/mbenito/two_stream_dvd/src/visualizations/assets/distributions.png",
        dpi=300,
    )


if __name__ == "__main__":
    main()
