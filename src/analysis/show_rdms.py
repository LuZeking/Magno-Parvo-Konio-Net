"""
Simple script to quickly visualize single RDMS from a saved RDMs.pkl file.
Change the layer_name and file to select the desired RDM.
"""

import pickle

import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

# variables
layer_name = "stream2.7.1.conv2"
file = "/share/klab/lzejin/mbenito/two_stream_dvd/RDMs/split_dvd_nsd_resnet18_seperate_from_layer0_RDMs.pkl"

# load the RDMs
with open(file, "rb") as f:
    RSMs = pickle.load(f)

# select the desired RDM and make it square
matrix = RSMs[layer_name]
matrix = squareform(matrix)

# plotting
plt.imshow(matrix, cmap="viridis")
plt.title(f"RDM for layer {layer_name}")
plt.colorbar(label="Correlation")
plt.xlabel("Images")
plt.ylabel("Images")
plt.savefig(f"{layer_name}_nsd.png")
