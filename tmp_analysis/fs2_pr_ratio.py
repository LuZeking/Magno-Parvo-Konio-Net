"""Results 2.5 / ED Fig. 2 on the TSN of record (2026-10-19): magno/parvo
participation-ratio ratio on the first 1,000 NSD images of each subject, with the
estimator of tmp_analysis/fig1_data.py (_pr_of, Gram side), for every run in the
fs2 encoding feature cache. Writes results/figures_v2/fs2_pr_ratio.json.

    PYTHONPATH=. python tmp_analysis/fs2_pr_ratio.py
"""
import glob
import json
import os
import re
import zipfile

import numpy as np

CACHE = "/scratch/byrong/mpk/encoding/cache/2026-10-16_fs2"
OUT = "results/figures_v2/fs2_pr_ratio.json"
NIMG = 1000


def _pr_of(X):
    """Participation ratio, Gram side; identical to tmp_analysis/fig1_data.py."""
    X = np.asarray(X[:NIMG], dtype=np.float64)
    X -= X.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh(X @ X.T)
    lam = lam[lam > 0]
    return (lam.sum() ** 2) / (lam ** 2).sum()


def first_rows(path, key):
    """The first NIMG rows of one array of an uncompressed .npz, without reading the rest."""
    with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as f:
        major, _ = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        assert not fortran and len(shape) == 2
        n = min(NIMG, shape[0])
        return np.frombuffer(f.read(n * shape[1] * dtype.itemsize), dtype=dtype).reshape(n, shape[1])


def main():
    out = {}
    for f in sorted(glob.glob(f"{CACHE}/*_subj*.npz")):
        run, subj = re.match(r"(.+)_subj(\d+)\.npz", os.path.basename(f)).groups()
        m, p = _pr_of(first_rows(f, "magno")), _pr_of(first_rows(f, "parvo"))
        out.setdefault(run, {})[f"subj{int(subj):02d}"] = dict(pr_magno=m, pr_parvo=p, ratio=m / p)
        print(f"{run} subj{subj}: magno {m:.2f} parvo {p:.2f} ratio {m / p:.3f}", flush=True)
    json.dump(dict(cache=CACHE, n_images=NIMG, runs=out), open(OUT, "w"), indent=1)
    for run, d in out.items():
        r = np.array([v["ratio"] for v in d.values()])
        print(f"{run:34s} n={len(r)} ratio mean {r.mean():.3f}  above parity {(r > 1).sum()}/{len(r)}")


if __name__ == "__main__":
    main()
