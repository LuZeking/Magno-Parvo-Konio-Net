"""What did the thesis's own RSA see in the ROIs it did analyse?

The eccentricity reversal cannot be in the thesis: it never used prf-eccrois.
But the thesis DID analyse V1v..hV4, so the hV4 crossover is a place where it
had the data. This asks whether its method could have surfaced that, by
computing its own stream1-vs-stream2 contrast from its published RDM
correlations.

Note what RSA gives and does not: a correlation per (stream, region, layer).
There is no per-voxel decomposition, so no unique variance and no
stream-preference index -- the quantity my encoding analysis reports simply
does not exist in this file. The closest available analogue is the
difference in RDM correlation between the two streams.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

RDM_DIR = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/RDMs"
MAIN = "dkl_blur_cs_dvd_trans_nsd_resnet18_seperate_from_layer0"

for suffix, title in (("specified_regions", "V1v..hV4 / PPA"),
                      ("streams", "stream ROIs")):
    path = os.path.join(RDM_DIR, f"{MAIN}_RDMs_{suffix}.csv")
    df = pd.read_csv(path)
    df = df[df["region"] != "region"]
    df["correlation"] = pd.to_numeric(df["correlation"], errors="coerce")
    print(f"\n=== {title} ({suffix}) ===")
    print("columns:", list(df.columns),
          "| n rows:", len(df),
          "| streams:", sorted(df['stream'].unique()),
          "| layers:", df['layer'].nunique())

    # Deepest layer of each stream, the closest analogue to the pooled
    # features the encoding model uses.
    for stream in sorted(df["stream"].unique()):
        lay = sorted(df[df["stream"] == stream]["layer"].unique())
        print(f"  {stream}: {len(lay)} layers, deepest = {lay[-1]}")

    s1 = df[df["stream"] == "stream1"]
    s2 = df[df["stream"] == "stream2"]
    if s1.empty or s2.empty:
        continue
    # Average over layers within stream, then contrast: the thesis reports
    # per-layer curves, so this is the summary it would have had.
    a = s1.groupby("region")["correlation"].mean()
    b = s2.groupby("region")["correlation"].mean()
    best_a = s1.groupby("region")["correlation"].max()
    best_b = s2.groupby("region")["correlation"].max()
    out = pd.DataFrame({"stream1(magno)": a, "stream2(parvo)": b,
                        "diff": a - b, "peak_diff": best_a - best_b})
    print(out.round(4).to_string())
