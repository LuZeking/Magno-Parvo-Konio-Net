"""Repaired RSA on the precomputed thesis RDMs (no retraining needed).

Fixes over the thesis analysis:
  * per-subject inference (never average subject RDMs before correlating)
  * lower AND upper noise ceilings per ROI; ceiling-normalized scores
  * pre-declared layer choice: final conv layer of each stream
  * stream x ROI-class interaction via subject-level sign-flip permutation
    (difference of differences), plus both simple effects (Wilcoxon)

Inputs: RDMs/*.pkl  (model RDMs: {layer: condensed 515-RDM};
        *_all_neural_rdms_correlation_special515.pkl:
        {subjXX: {roi: condensed 515-RDM}})

Run:  python -m src.analysis.encoding.repaired_rsa [--rdm-dir RDMs]
Outputs: results/repaired_rsa/{per_layer_correlations.csv,
         final_layer_summary.csv, interaction_tests.csv, noise_ceilings.csv}
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats as sps

STREAM_ROIS = [
    "early", "midventral", "midlateral", "midparietal",
    "ventral", "lateral", "parietal",
]
# pre-declared ROI classes for the interaction contrast
VENTRAL_SET = ["midventral", "ventral"]
PARIETAL_SET = ["midparietal", "parietal"]


def load_neural_rdms(rdm_dir: str) -> dict[str, dict[str, np.ndarray]]:
    """{subj: {roi: condensed RDM}} from the streams pickle plus the
    probmap early-visual pickles (each contributing one ROI)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    streams_path = os.path.join(
        rdm_dir, "streams_all_neural_rdms_correlation_special515.pkl"
    )
    with open(streams_path, "rb") as f:
        streams = pickle.load(f)
    for subj, rois in streams.items():
        out.setdefault(subj, {}).update(rois)

    for path in glob.glob(
        os.path.join(
            rdm_dir, "probmap_*_all_neural_rdms_correlation_special515.pkl"
        )
    ):
        roi_name = os.path.basename(path).split("_all_neural")[0]
        with open(path, "rb") as f:
            d = pickle.load(f)
        for subj, val in d.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    key = roi_name if k == roi_name else f"{roi_name}_{k}"
                    out.setdefault(subj, {})[key] = np.asarray(v)
            else:
                out.setdefault(subj, {})[roi_name] = np.asarray(val)
    return out


def final_conv_key(keys: list[str], stream_prefix: str) -> str | None:
    """Last recorded layer of a stream (dict insertion order = forward
    order in the extraction hooks)."""
    stream_keys = [k for k in keys if k.startswith(stream_prefix)]
    return stream_keys[-1] if stream_keys else None


def rank(v: np.ndarray) -> np.ndarray:
    return sps.rankdata(v).astype(np.float64)


def safe_wilcoxon_p(d: np.ndarray) -> float:
    """Wilcoxon p, defined as 1.0 when all differences are exactly zero
    (e.g. RGB/DVD TSN, whose two streams are identical by construction)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0.0):
        return 1.0
    return float(sps.wilcoxon(d).pvalue)


def pearson_on_ranks(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def noise_ceilings(
    neural: dict[str, dict[str, np.ndarray]],
    ranked_neural: dict[str, dict[str, np.ndarray]],
    roi: str,
) -> pd.DataFrame:
    """Per-subject lower/upper ceilings (Spearman-scale, on ranked RDMs)."""
    subs = sorted(s for s in neural if roi in neural[s])
    rows = []
    for s in subs:
        others = [ranked_neural[o][roi] for o in subs if o != s]
        lower = pearson_on_ranks(ranked_neural[s][roi], np.mean(others, axis=0))
        upper = pearson_on_ranks(
            ranked_neural[s][roi],
            np.mean([ranked_neural[o][roi] for o in subs], axis=0),
        )
        rows.append(
            {"roi": roi, "subject": s, "lower_nc": lower, "upper_nc": upper}
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdm-dir", default="RDMs")
    ap.add_argument("--out-dir", default="results/repaired_rsa")
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    neural = load_neural_rdms(args.rdm_dir)
    ranked_neural = {
        s: {r: rank(v) for r, v in rois.items()}
        for s, rois in neural.items()
    }
    all_rois = sorted({r for s in neural.values() for r in s})

    nc = pd.concat(
        [noise_ceilings(neural, ranked_neural, r) for r in all_rois]
    )
    nc.to_csv(os.path.join(args.out_dir, "noise_ceilings.csv"), index=False)

    model_paths = sorted(
        p
        for p in glob.glob(os.path.join(args.rdm_dir, "*_RDMs.pkl"))
        if "neural" not in os.path.basename(p)
    )

    per_layer_rows = []
    for path in model_paths:
        model = os.path.basename(path).replace("_RDMs.pkl", "")
        with open(path, "rb") as f:
            rdms = pickle.load(f)
        keys = list(rdms.keys())
        finals = {
            st: final_conv_key(keys, f"{st}_block")
            for st in ("stream1", "stream2")
        }
        print(f"[{model}] {len(keys)} layers; finals: {finals}", flush=True)

        for k in keys:
            rk = rank(np.asarray(rdms[k]))
            stream = "stream1" if k.startswith("stream1") else (
                "stream2" if k.startswith("stream2") else "other"
            )
            for subj, rois in ranked_neural.items():
                for roi_name, rvec in rois.items():
                    per_layer_rows.append(
                        {
                            "model": model, "layer": k, "stream": stream,
                            "subject": subj, "roi": roi_name,
                            "spearman": pearson_on_ranks(rk, rvec),
                            "is_final": k in finals.values(),
                        }
                    )

    per_layer = pd.DataFrame(per_layer_rows)
    per_layer = per_layer.merge(nc, on=["roi", "subject"], how="left")
    per_layer["spearman_ncnorm"] = (
        per_layer["spearman"] / per_layer["lower_nc"]
    )
    per_layer.to_csv(
        os.path.join(args.out_dir, "per_layer_correlations.csv"), index=False
    )
    finals_df = per_layer[per_layer["is_final"]]
    finals_df.to_csv(
        os.path.join(args.out_dir, "final_layer_summary.csv"), index=False
    )

    # ---- interaction: (stream1 - stream2) in parietal vs ventral sets ----
    rng = np.random.default_rng(0)
    tests = []
    for model, g in finals_df.groupby("model"):
        if set(g["stream"].unique()) != {"stream1", "stream2"}:
            continue  # single-stream model
        pivot = g.pivot_table(
            index="subject", columns=["stream", "roi"],
            values="spearman_ncnorm",
        )
        try:
            dv = (
                pivot["stream1"][VENTRAL_SET].mean(axis=1)
                - pivot["stream2"][VENTRAL_SET].mean(axis=1)
            )
            dp = (
                pivot["stream1"][PARIETAL_SET].mean(axis=1)
                - pivot["stream2"][PARIETAL_SET].mean(axis=1)
            )
        except KeyError:
            continue
        d = (dp - dv).to_numpy()  # >0: stream1 relatively more parietal
        d = d[np.isfinite(d)]
        obs = d.mean()
        signs = rng.choice([-1.0, 1.0], size=(args.n_perm, len(d)))
        p_perm = (
            np.sum(np.abs((signs * d).mean(axis=1)) >= abs(obs)) + 1
        ) / (args.n_perm + 1)
        tests.append(
            {
                "model": model,
                "interaction_mean_dod": obs,
                "interaction_p_perm": p_perm,
                "simple_ventral_median": float(np.median(dv.dropna())),
                "simple_ventral_p": safe_wilcoxon_p(dv.to_numpy()),
                "simple_parietal_median": float(np.median(dp.dropna())),
                "simple_parietal_p": safe_wilcoxon_p(dp.to_numpy()),
                "n_subjects": len(d),
            }
        )
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(
        os.path.join(args.out_dir, "interaction_tests.csv"), index=False
    )
    print("\n=== Noise ceilings (lower, mean over subjects) ===")
    print(nc.groupby("roi")["lower_nc"].mean().round(3).to_string())
    print("\n=== Stream x ROI-class interaction (final layers) ===")
    if len(tests_df):
        print(tests_df.round(4).to_string(index=False))
    else:
        print("no two-stream models found")


if __name__ == "__main__":
    main()
