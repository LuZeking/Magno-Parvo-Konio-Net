"""Time-resolved RSA: TSN stream RDMs vs developmental EEG, 7 sub-age bins.

For every subject in every sub-age bin (infants_6mo, infants_12mo,
youngerChildren_4yo, youngerChildren_6yo, olderChildren_8yo,
olderChildren_12yo, adults_all), correlates (Spearman) each TSN layer RDM
with the subject's time-resolved neural RDM (12 x 12 x T) at every
timepoint, using the subject's own stimulus subset.

Outputs (out_dir):
  per_subject_rsa.csv.gz  long table: source, age_group, subage_bin,
                          subject, subset, model, layer, stream, tp,
                          time_ms, rho
  group_summary.csv       per bin x layer x tp: mean rho, sem, n, and the
                          matching noise ceiling (merged from the campaign
                          noise_ceiling.csv when available)
  stream_contrast.csv     per bin x stage x tp: Wilcoxon of layer-matched
                          magno-minus-parvo rho across subjects — the
                          developmental question: does relative magno vs
                          parvo alignment shift across age?

Run:
    python -m src.analysis.eeg.temporal_rsa_tsn \
        --model-rdm-dir /scratch/byrong/infant/tsn_rdms_orig_input_v1_spearman \
        --model-name tsn_untrained_seed0 --epoch 0 \
        --out-dir results/eeg_rsa/tsn_untrained_seed0
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

INFANT_PROJECT = os.environ.get(
    "INFANT_PROJECT", "/home/byrong/projects/infant"
)
sys.path.insert(0, INFANT_PROJECT)

from analysis.subject_subage_mapping import (  # noqa: E402
    SUBAGE_KEYS_BY_AGE_GROUP,
    subjects_for_bin,
    target_months_for_bin,
)

NEURAL_BASE = os.environ.get(
    "EEG_NEURAL_BASE", "/scratch/byrong/infant/rsa2026/inputs/brain_rdms"
)
NOISE_CEILING_CSV = os.environ.get(
    "EEG_NOISE_CEILING_CSV",
    "/scratch/byrong/infant/noise_ceiling/noise_ceiling.csv",
)
AGE_GROUPS = ["infants", "youngerChildren", "olderChildren", "adults"]
TIME_START_MS, TIME_STEP_MS = -500, 20  # matches campaign noise_ceiling.csv


def condensed(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(rdm.shape[0], k=1)
    return rdm[iu]


def subject_files(source: str, age_group: str) -> list[tuple[int, int, str]]:
    """[(subject_id, subset_number, path)] for one source/age_group."""
    pattern = os.path.join(
        NEURAL_BASE, source, age_group, "mvnn_time", "sub-*_subset*_avg.npy"
    )
    out = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        if "shuffled" in name:
            continue
        m = re.match(r"sub-(\d+)_subset(\d+)_avg\.npy$", name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), path))
    return out


def bin_of_subject(age_group: str, subject_id: int) -> str | None:
    for bin_name in SUBAGE_KEYS_BY_AGE_GROUP[age_group]:
        if subject_id in set(subjects_for_bin(age_group, bin_name).tolist()):
            return bin_name
    return None


def load_model_rdms(
    model_rdm_dir: str, model_name: str, epoch: int, rdm_metric: str
) -> dict[str, dict[str, np.ndarray]]:
    """{subset_name: {layer: condensed+ranked model RDM}}"""
    out: dict[str, dict[str, np.ndarray]] = {}
    for subset_dir in sorted(
        glob.glob(os.path.join(model_rdm_dir, "subset*"))
    ):
        subset = os.path.basename(subset_dir)
        path = os.path.join(
            subset_dir, model_name, f"rdms_epoch_{epoch:03d}.npz"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        z = np.load(path, allow_pickle=True)
        layers = [str(x) for x in z["layers"]]
        out[subset] = {
            layer: sps.rankdata(condensed(z[f"{layer}_rdm_{rdm_metric}"]))
            for layer in layers
        }
    return out


def spearman_vs_ranked_matrix(
    model_ranked: np.ndarray, neural_ranked: np.ndarray
) -> np.ndarray:
    """Spearman rho of one ranked model vector vs (n_pairs, T) pre-ranked
    neural vectors, vectorized over T."""
    a = model_ranked - model_ranked.mean()
    b = neural_ranked - neural_ranked.mean(axis=0, keepdims=True)
    denom = np.sqrt((a @ a) * np.einsum("it,it->t", b, b))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a @ b) / denom, np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-rdm-dir", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--rdm-metric", default="no_zscore",
                   choices=["no_zscore", "zscore"])
    p.add_argument("--sources", default="occipital")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model_rdms = load_model_rdms(
        args.model_rdm_dir, args.model_name, args.epoch, args.rdm_metric
    )
    layers = list(next(iter(model_rdms.values())).keys())

    rows = []
    for source in [s.strip() for s in args.sources.split(",")]:
        for age_group in AGE_GROUPS:
            files = subject_files(source, age_group)
            print(f"[{source}/{age_group}] {len(files)} subjects", flush=True)
            for sid, subset_num, path in files:
                bin_name = bin_of_subject(age_group, sid)
                if bin_name is None:
                    continue
                neural = np.load(path)  # (12, 12, T)
                n_tp = neural.shape[2]
                iu = np.triu_indices(neural.shape[0], k=1)
                neural_vecs = neural[iu[0], iu[1], :]  # (66, T)
                neural_ranked = np.apply_along_axis(
                    sps.rankdata, 0, neural_vecs
                )
                subset = f"subset{subset_num}"
                months = target_months_for_bin(bin_name)
                for layer in layers:
                    mr = model_rdms[subset][layer]
                    stream = layer.split("_")[0]
                    rhos = spearman_vs_ranked_matrix(mr, neural_ranked)
                    for tp in range(n_tp):
                        rows.append(
                            (source, age_group, bin_name, months, sid,
                             subset, args.model_name, layer, stream, tp,
                             TIME_START_MS + tp * TIME_STEP_MS,
                             float(rhos[tp]))
                        )

    per_subject = pd.DataFrame(
        rows,
        columns=["source", "age_group", "subage_bin", "age_months",
                 "subject", "subset", "model", "layer", "stream", "tp",
                 "time_ms", "rho"],
    )
    per_subject.to_csv(
        os.path.join(args.out_dir, "per_subject_rsa.csv.gz"), index=False
    )

    # ---- group summary + noise ceiling merge ----
    grp = (
        per_subject.groupby(
            ["source", "subage_bin", "age_months", "layer", "stream",
             "tp", "time_ms"]
        )["rho"]
        .agg(mean_rho="mean", sem_rho="sem", n_subj="count")
        .reset_index()
    )
    if os.path.exists(NOISE_CEILING_CSV):
        nc = pd.read_csv(NOISE_CEILING_CSV)[
            ["neural_source", "subage_bin", "tp", "NC_lower", "NC_upper"]
        ].rename(columns={"neural_source": "source"})
        grp = grp.merge(nc, on=["source", "subage_bin", "tp"], how="left")
        grp["mean_rho_ncnorm"] = grp["mean_rho"] / grp["NC_lower"]
    grp.to_csv(os.path.join(args.out_dir, "group_summary.csv"), index=False)

    # ---- magno-vs-parvo contrast per bin x stage x tp ----
    stages = sorted(
        {l.split("_", 1)[1] for l in layers if l.startswith("magno_")}
    )
    contrast_rows = []
    for (source, bin_name, tp), g in per_subject.groupby(
        ["source", "subage_bin", "tp"]
    ):
        pivot = g.pivot_table(
            index="subject", columns="layer", values="rho"
        )
        for stage in stages:
            mcol, pcol = f"magno_{stage}", f"parvo_{stage}"
            if mcol not in pivot or pcol not in pivot:
                continue
            d = (pivot[mcol] - pivot[pcol]).dropna().to_numpy()
            if len(d) < 5:
                continue
            if np.allclose(d, 0):
                w_p = 1.0
            else:
                w_p = float(sps.wilcoxon(d).pvalue)
            contrast_rows.append(
                {
                    "source": source, "subage_bin": bin_name,
                    "age_months": target_months_for_bin(bin_name),
                    "stage": stage, "tp": tp,
                    "time_ms": TIME_START_MS + tp * TIME_STEP_MS,
                    "mean_magno_minus_parvo": float(d.mean()),
                    "wilcoxon_p": w_p, "n_subj": len(d),
                }
            )
    pd.DataFrame(contrast_rows).to_csv(
        os.path.join(args.out_dir, "stream_contrast.csv"), index=False
    )

    print(f"\nSaved -> {args.out_dir}")
    peak = (
        grp.loc[grp.groupby(["subage_bin", "stream"])["mean_rho"].idxmax()]
        [["subage_bin", "age_months", "stream", "layer", "time_ms",
          "mean_rho", "n_subj"]]
        .sort_values(["age_months", "stream"])
    )
    print("\nPeak correlation per sub-age bin x stream:")
    print(peak.to_string(index=False))


if __name__ == "__main__":
    main()
