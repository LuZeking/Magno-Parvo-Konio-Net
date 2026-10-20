"""Run names and seed discovery for the PAPER_v3 figures (model of record dvd300).

Seeds are discovered from the committed results rather than listed, so control
seeds 1-2 and the channel swap (fs2_dvd300swap) enter every figure as soon as
their phase-B outputs are committed; a run still training has no result file
and no finished history, and is left out.
"""
from __future__ import annotations

import json
import os

import numpy as np

FS2_RUNS = "/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-12_stage2"
BA_RUNS = "/scratch/byrong/VST_data/mpk_runs"
HIST_SNAPSHOT = "docs/slides_2026-10-12_ba_vs_fusion/figures_ba/{}__history.json"   # /home copies, used if /scratch is absent
CORTEX = "docs/slides_2026-10-12_ba_vs_fusion/figures/fs2_cortex_summary.json"

MODEL = "fs2_dvd300_avg_w050_ls_seed{}"
CONTROL = "fs2_dvdfix300_avg_w050_ls_seed{}"
SWAP = "fs2_dvd300swap_avg_w050_ls_seed{}"
FIXEDREC = "fs2_bcs_avg_w050_ls_seed{}"
DVD48 = "fs2_dvd48_avg_w050_ls_seed{}"
BA = "dkl_bcs_seed{}"
BANODIET = "dkl_base_seed{}"
EPOCHS = {"fs2": 50, "dkl": 150}


def runs(tmpl: str, pattern: str = "results/cue_conflict/{}.json") -> list[str]:
    """Runs of a family (seeds 0-2) whose result file exists; default: the cue-conflict JSON."""
    return [tmpl.format(s) for s in range(3) if os.path.exists(pattern.format(tmpl.format(s)))]


def history(run: str) -> list[dict] | None:
    """The run's per-epoch history if training finished, else None."""
    root = BA_RUNS if run.startswith("dkl_") else FS2_RUNS
    f = f"{root}/{run}/history.json"
    if not os.path.exists(f):
        f = HIST_SNAPSHOT.format(run)
        if not os.path.exists(f):
            return None
    h = json.load(open(f))
    return h if len(h) >= EPOCHS["dkl" if run.startswith("dkl_") else "fs2"] else None


def finished(tmpl: str) -> list[str]:
    """Runs of a family (seeds 0-2) whose training finished."""
    return [tmpl.format(s) for s in range(3) if history(tmpl.format(s)) is not None]


def final_top1(run: str) -> float:
    return history(run)[-1]["val_top1"]


def cc(run: str, cond: str = "full", key: str = "shape_bias", root: str = "results/cue_conflict") -> float:
    c = json.load(open(f"{root}/{run}.json"))["conditions"][cond]
    return c["n_shape"] + c["n_texture"] if key == "decided" else c[key]


def sweep(run_ckpt: str) -> tuple[np.ndarray, np.ndarray]:
    """(w, shape bias) of results/fusion_sweep/<run>_<policy>_wsweep.json, sorted by w."""
    pts = sorted(json.load(open(f"results/fusion_sweep/{run_ckpt}_wsweep.json"))["sweep"], key=lambda r: r["w"])
    return np.array([r["w"] for r in pts]), np.array([r["shape_bias"] for r in pts])


def cortex() -> dict:
    """tmp_analysis/fs2_cortex_summary.py's per-arm cortex results (arms: dvd300, control, A, BA diet, BA no diet)."""
    return json.load(open(CORTEX))


def fmt_seeds(n: int) -> str:
    return f"{n} seed" + ("s" if n != 1 else "")
