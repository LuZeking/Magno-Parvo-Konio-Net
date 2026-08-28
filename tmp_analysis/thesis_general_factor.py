"""Is the thesis's stream contrast a stream preference, or a global offset?

In the thesis's published RSA numbers stream2 correlates better than stream1
in almost every higher-level ROI. That is what an overall difference in how
much structure each stream's RDM carries looks like -- the unblurred parvo
stream simply has richer features -- and it is NOT the same claim as "this
region prefers parvo". Removing the stream main effect leaves the
region-specific part, which is the quantity comparable to unique variance.
"""

from __future__ import annotations

import pandas as pd

RDM = ("/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/RDMs/"
       "dkl_blur_cs_dvd_trans_nsd_resnet18_seperate_from_layer0_RDMs_")

for suffix in ("streams", "specified_regions"):
    df = pd.read_csv(RDM + suffix + ".csv")
    df = df[df["region"] != "region"].copy()
    df["correlation"] = pd.to_numeric(df["correlation"], errors="coerce")
    m = df.groupby(["stream", "region"])["correlation"].mean().unstack(0)
    s1, s2 = m["stream1"], m["stream2"]

    off = s2.mean() - s1.mean()   # global stream main effect
    raw = s1 - s2
    adj = raw + off               # region-specific deviation from it

    out = pd.DataFrame({"magno": s1, "parvo": s2, "raw_diff": raw,
                        "diff_minus_global": adj})
    print(f"\n=== {suffix} ===")
    print(f"global offset: mean(parvo) - mean(magno) = {off:+.4f}")
    print(out.round(4).sort_values("diff_minus_global",
                                   ascending=False).to_string())
