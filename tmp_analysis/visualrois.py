"""A3.3 -- stream preference across retinotopic areas V1-hV4 (n=8).

Prediction: parvocellular input is chromatically opponent and feeds the
colour/form pathway, so parvo-unique variance should rise (SPI should fall)
from V1 toward hV4.

Caveat recorded up front: hV4 is more foveally biased than V1-V3, so this
partly re-expresses the eccentricity gradient rather than testing something
independent of it. The V1v-vs-V1d style ventral/dorsal split within a single
area is the cleaner contrast and is reported alongside.

PROVENANCE (PAPER_PLAN §88). Until 2026-09-07 the model was hardcoded on one
line. §2.2's diet and no-diet rows were therefore produced by editing that line
and re-running, which left no record: the numbers could not be regenerated, and
the search for a configuration that reproduced them could not have succeeded
because the fit they came from had been quarantined and deleted. The model is
now an argument and every run writes a provenance-stamped artifact, so a
published number always has a regenerable source.

The DEFAULT is deliberately `untrained_seed0` on `results/encoding`, because
that is the only row with continuous provenance back to 2026-08-28 and it is
the audit anchor: running this script with no arguments must keep reproducing
`SPI hV4 - V1  mean diff=-0.5441  p=0.0078  0/8 positive`.

Usage (no PYTHONPATH needed, and works from any directory):
    python tmp_analysis/visualrois.py                  # the audit anchor
    python tmp_analysis/visualrois.py --model dkl_bcs_seed0_trained
    python tmp_analysis/visualrois.py --src results/encoding_cc \
        --model dkl_base_seed0_trained
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys

import numpy as np
from scipy import stats as sps

# Make the anchor runnable with a bare `python tmp_analysis/visualrois.py`,
# from any directory. Without this the failure mode for someone following the
# docs is a ModuleNotFoundError rather than a number, which is a bad property
# for the one row a referee may be pointed at. The import below must therefore
# come after this block (E402 is deliberate).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.encoding.stats import load_results  # noqa: E402

VIS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}
NC_MIN = 0.1
MIN_VOX = 20
DENOM_MIN = 1e-3


def git_head() -> dict[str, str]:
    """Commit the numbers were produced at, and whether the tree was dirty."""
    out = {}
    for key, cmd in (("commit", ["git", "rev-parse", "--short", "HEAD"]),
                     ("dirty", ["git", "status", "--porcelain"])):
        try:
            # cwd=REPO: the stamp must describe THIS repo, not whatever
            # directory the script happened to be invoked from.
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=30, cwd=REPO)
            out[key] = r.stdout.strip() if r.returncode == 0 else "unavailable"
        except Exception:
            out[key] = "unavailable"
    out["dirty"] = "yes" if out.get("dirty") else "no"
    return out


def input_stamp(src: str, model: str) -> list[dict]:
    """Exact files read, with size and mtime -- §87/§88 both turned on these."""
    stamp = []
    for f in sorted(glob.glob(f"{src}/{model}_subj*.npz")):
        st = os.stat(f)
        stamp.append({
            "file": os.path.basename(f),
            "bytes": st.st_size,
            "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(),
        })
    return stamp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="untrained_seed0",
                   help="model tag; default is the §88 audit anchor")
    # Defaults are anchored at the repo root so a zero-argument run works from
    # any cwd. A path passed explicitly is used as given, i.e. relative to cwd.
    p.add_argument("--src", default=os.path.join(REPO, "results/encoding"),
                   help="directory of *_subj*.npz fits")
    p.add_argument("--out-dir", default=os.path.join(REPO, "results/visualrois"),
                   help="where the provenance-stamped artifact is written")
    p.add_argument("--no-write", action="store_true",
                   help="print only; do not write an artifact")
    args = p.parse_args()

    res = load_results(args.src, args.model)
    if not res:
        print(f"no fits matched {args.src}/{args.model}_subj*.npz")
        return 1
    subjects = sorted(res)
    labs = sorted(VIS)

    spi = np.full((len(subjects), len(labs)), np.nan)
    um_t = np.full_like(spi, np.nan)
    up_t = np.full_like(spi, np.nan)
    nvox = np.zeros_like(spi, dtype=int)
    for i, s in enumerate(subjects):
        z = res[s]
        roi, nc = z["roi_prf-visualrois"], z["noise_ceiling"]
        um, up = z["unique_magno"], z["unique_parvo"]
        for j, lab in enumerate(labs):
            v = (roi == lab) & (nc > NC_MIN) & ((um + up) > DENOM_MIN)
            nvox[i, j] = int(v.sum())
            if v.sum() >= MIN_VOX:
                r = ((um[v] - up[v]) / (um[v] + up[v]))
                spi[i, j] = np.median(r[np.isfinite(r)])
                um_t[i, j] = np.median(um[v])
                up_t[i, j] = np.median(up[v])

    lines = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"Median voxels per area (nc>{NC_MIN}): " + ", ".join(
        f"{VIS[l]}={int(np.median(nvox[:, j]))}" for j, l in enumerate(labs)))

    emit(f"\nMedian SPI (positive = magno-preferring)")
    emit(f"{'subj':6s}" + "".join(f"{VIS[l]:>9s}" for l in labs))
    for i, s in enumerate(subjects):
        emit(f"{s:<6d}" + "".join(f"{v:>+9.3f}" for v in spi[i]))
    emit(f"{'mean':6s}" + "".join(
        f"{np.nanmean(spi[:, j]):>+9.3f}" for j in range(len(labs))))

    emit(f"\nMedian unique variance / NC")
    emit(f"{'':6s}" + "".join(f"{VIS[l]:>9s}" for l in labs))
    emit(f"{'magno':6s}" + "".join(
        f"{np.nanmean(um_t[:, j]):>9.4f}" for j in range(len(labs))))
    emit(f"{'parvo':6s}" + "".join(
        f"{np.nanmean(up_t[:, j]):>9.4f}" for j in range(len(labs))))

    contrasts: dict[str, dict[str, float]] = {}

    def paired(a, b, name):
        ok = np.isfinite(a) & np.isfinite(b)
        d = (a - b)[ok]
        W, pv = sps.wilcoxon(d)
        emit(f"  {name:<26} mean diff={d.mean():+.4f}  W={W:.0f}  "
             f"p={pv:.4f}  {int((d > 0).sum())}/{ok.sum()} positive")
        contrasts[name] = {
            "mean_diff": float(d.mean()), "W": float(W), "p": float(pv),
            "n_positive": int((d > 0).sum()), "n": int(ok.sum()),
            "per_subject": [float(x) for x in d],
        }

    k = {VIS[l]: j for j, l in enumerate(labs)}
    emit(f"\nPaired contrasts across subjects (n={len(subjects)})")
    v1 = np.nanmean(spi[:, [k["V1v"], k["V1d"]]], axis=1)
    v123 = np.nanmean(spi[:, [k[n] for n in ("V1v", "V1d", "V2v", "V2d",
                                             "V3v", "V3d")]], axis=1)
    paired(spi[:, k["hV4"]], v1, "SPI hV4 - V1")
    paired(spi[:, k["hV4"]], v123, "SPI hV4 - V1/V2/V3")
    # ventral vs dorsal banks of the SAME areas: matched eccentricity, so this
    # is the contrast that is NOT explained by the foveal bias of hV4
    ven = np.nanmean(spi[:, [k[n] for n in ("V1v", "V2v", "V3v")]], axis=1)
    dor = np.nanmean(spi[:, [k[n] for n in ("V1d", "V2d", "V3d")]], axis=1)
    paired(dor, ven, "SPI dorsal - ventral bank")

    if args.no_write:
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = {
        "script": "tmp_analysis/visualrois.py",
        "written": dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_head(),
        "model": args.model,
        "src": args.src,
        "subjects": [int(s) for s in subjects],
        # the convention, recorded so a later reader need not infer it from
        # the code -- §86 cost us a full exchange over exactly this
        "convention": {
            "nc_min": NC_MIN,
            "min_voxels": MIN_VOX,
            "denominator_mask": f"(unique_magno + unique_parvo) > {DENOM_MIN}",
            "voxel_aggregation": "median-of-ratios",
            "spi": "(unique_magno - unique_parvo)/(unique_magno + unique_parvo)"
                   " per voxel, then median over voxels",
            "note": "unique_* are divided by the noise ceiling; the pre-nc"
                    " arrays are unique_*_pre_nc (formerly *_raw, §86.3)",
            "v1_reference": "mean(V1v, V1d)",
            "across_subject_test": "Wilcoxon signed-rank, two-sided",
        },
        "inputs": input_stamp(args.src, args.model),
        "spi_roi_means": {VIS[l]: float(np.nanmean(spi[:, j]))
                          for j, l in enumerate(labs)},
        "contrasts": contrasts,
    }
    tag = dt.date.today().isoformat()
    base = os.path.join(args.out_dir, f"{args.model}_{tag}")
    with open(f"{base}.json", "w") as f:
        json.dump(stamp, f, indent=2)
    with open(f"{base}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nprovenance-stamped -> {base}.json / {base}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
