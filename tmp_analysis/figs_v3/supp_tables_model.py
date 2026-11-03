"""Supplementary Tables S1-S10 for the model of record (docs/PLAN_SUPP_MODEL_2026-11-03.md, sections 1-2).

Every row is computed with the BA table's own statistic from outputs that already exist; nothing is trained,
refit or re-encoded, and no sealed pixels are read (behaviour comes from the stored per-image logits).
The figs_v2 generators are imported where they can be imported (hierarchy_permutation_null,
forced_choice, ext_gradient, fs2_cortex_controls) and copied verbatim where they execute at import
(ed_tables.py: profile/grad/cc_streams/auc; floor_pairing.py, diet_floor_paired.py: the S7 variants).
None of those files is edited.

Arms (section 0 of the plan)
  model          fs2_dvd300_avg_w050_ls_seed{0,1,2}
  control        fs2_dvdfix300_avg_w050_ls_seed{0,1,2}
  channel swap   fs2_dvd300swap_avg_w050_ls_seed{0,1,2}
  untrained      untrained_dvd300_seed0 (cortical tables; untrained_dvd300 for the behavioural probe)

Writes <out>/supp_tables_model.json (every number, per seed / per subject), <out>/supp_tables_model.md
(the rows in the manuscript's table syntax, one block per table) and <out>/provenance.json (inputs
with size, mtime and sha256; commit; environment).

Usage: PYTHONPATH=.:tmp_analysis:tmp_analysis/figs_v2 python tmp_analysis/figs_v3/supp_tables_model.py --out <dir>
       [--only S1,S2,...]   (default: all ten)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

import numpy as np
from scipy import stats as sps

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "tmp_analysis"), os.path.join(ROOT, "tmp_analysis", "figs_v2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ENC = os.environ.get("FS2_ENCODING_DIR", "/scratch/byrong/mpk/encoding/results/2026-10-16_fs2")
VPREC = os.environ.get("FS2_VPREC_DIR", os.path.join(ROOT, "results", "vp_recovery_realnc_v3"))
CKPT = os.environ.get("FS2_CKPT_DIR", "/scratch/byrong/TSN/fusion_stage1/checkpoints/2026-10-12_stage2")
HIST = os.path.join(ROOT, "docs", "slides_2026-10-12_ba_vs_fusion", "figures_ba", "{}__history.json")
EXEMPLAR = os.path.join(ROOT, "results", "figures_v2", "exemplar_streams.json")
CORTEX_CONTROLS = os.path.join(ROOT, "results", "figures_v3", "cortex_controls.json")
EXT_GRADIENT = os.path.join(ROOT, "results", "figures_v3", "ext_gradient.json")
RES = os.path.join(ROOT, "results")

MODEL = [f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)]
CONTROL = [f"fs2_dvdfix300_avg_w050_ls_seed{s}" for s in range(3)]
SWAP = [f"fs2_dvd300swap_avg_w050_ls_seed{s}" for s in range(3)]
ARMS = {"model": MODEL, "control": CONTROL, "channel swap": SWAP}
UNTRAINED_ENC, UNTRAINED_BEH = "untrained_dvd300_seed0", "untrained_dvd300"
SUBJECTS = list(range(1, 9))
NC_MIN, MIN_VOX, DENOM_MIN = 0.1, 20, 1e-3
VIS = [1, 2, 3, 4, 5, 6, 7]
ROI = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]
FAM = ["uniform-noise", "high-pass", "false-colour", "rotation", "phase-scrambling", "contrast", "low-pass"]
TRAINED_IN = ("contrast", "low-pass")          # declared trained-in before the BA run; the mark is kept
READOUTS = [("magno stream alone (w = 1)", "magno_only", 1.0), ("both streams (w = 0.5)", "full", 0.5),
            ("parvo stream alone (w = 0)", "parvo_only", 0.0)]
STAT_LABEL = {"raw": "per-voxel index", "norm": "ratio of medians"}   # hierarchy_permutation_null's labels, as ed_tables.py reads them

_INPUTS: dict[str, dict] = {}


def track(path: str) -> str:
    """Record an input file for provenance.json (size, mtime, sha256)."""
    path = os.path.abspath(path)
    if path not in _INPUTS and os.path.isfile(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        st = os.stat(path)
        _INPUTS[path] = {"bytes": st.st_size, "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                         "sha256": h.hexdigest()}
    return path


def jload(path: str):
    with open(track(path), encoding="utf8") as fh:
        return json.load(fh)


def npz(path: str):
    return np.load(track(path), allow_pickle=True)


def fits(run: str) -> dict[int, dict]:
    """Per-subject encoding fits of one run, from ENC (src.analysis.encoding.stats.load_results, tracked)."""
    from src.analysis.encoding.stats import load_results
    for p in sorted(glob.glob(os.path.join(ENC, f"{run}_subj*.npz"))):
        track(p)
    res = load_results(ENC, run)
    assert sorted(res) == SUBJECTS, (run, sorted(res))
    return res


# ----------------------------------------------------------------------------------------------- formatting
def f(x, nd=3, sign=True) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "−")


def rng(v, nd=3, sign=False) -> str:
    v = np.asarray(v, float)
    if v.size == 1:
        return f(v[0], nd, sign)
    return f"{f(v.mean(), nd, sign)} [{f(v.min(), nd, sign)}, {f(v.max(), nd, sign)}]"


def mptp(v, sign=False, nd=3) -> str:
    """mean [seed range width], the BA S3 format."""
    v = np.asarray(v, float)
    return f"{f(v.mean(), nd, sign)} [{np.ptp(v):.{nd}f}]"


def pfmt(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return f"{p:.4f}".lstrip("0") if p < 1 else "1.00"


def wil(x) -> dict:
    """mean, n < 0, n > 0, n, exact two-sided Wilcoxon p over finite entries (floor p = .0078 at n = 8)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    p = float(sps.wilcoxon(x, method="exact").pvalue) if len(x) >= 5 and np.any(x != 0) else float("nan")
    return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()), "n_neg": int((x < 0).sum()),
            "n_pos": int((x > 0).sum()), "n": int(len(x)), "p": p, "per_subject": x.tolist()}


def seed_of(run: str) -> str:
    m = re.search(r"seed(\d)", run)
    return m.group(1) if m else "—"


# ----------------------------------------------------------------------------------------------- S1
def s1_row(prof: np.ndarray, hier: np.ndarray, perms) -> dict:
    """Exact 630-labelling null of the mean Spearman rho (hierarchy_permutation_null.run, with the profile given)."""
    import hierarchy_permutation_null as hpn
    obs = hpn.mean_rho(prof, hier)
    null = np.array([hpn.mean_rho(prof, p) for p in perms])
    n_le = int((null <= obs + 1e-12).sum())
    return {"observed_mean_rho": float(obs), "p_exact_one_sided": n_le / len(perms), "n_permutations": len(perms),
            "rank_of_observed": n_le, "is_most_extreme": bool(n_le == 1), "null_q025": float(np.quantile(null, 0.025)),
            "null_min": float(null.min()), "null_max": float(null.max()),
            "per_subject_rho": [float(sps.spearmanr(hier, p)[0]) for p in prof]}


def s1() -> tuple[dict, list[str]]:
    import hierarchy_permutation_null as hpn
    perms = hpn.distinct_labelings(hpn.HIER)
    assert len(perms) == 630
    out, rows = [], []
    for arm, runs in list(ARMS.items()) + [("untrained", [UNTRAINED_ENC])]:
        for run in runs:
            fits(run)                      # tracked
            for stat in ("raw", "norm"):
                subs, prof = hpn.spi_profile(run, stat, src=ENC)
                assert list(subs) == SUBJECTS and np.isfinite(prof).all(), (run, stat)
                r = s1_row(prof, hpn.HIER, perms)
                grad = prof[:, 6] - prof[:, :2].mean(1)
                r.update(arm=arm, run=run, seed=seed_of(run), stat=stat, statistic=STAT_LABEL[stat],
                         profile_per_subject=prof.tolist(), profile_mean=prof.mean(0).tolist(),
                         gradient_per_subject=grad.tolist(), gradient_mean=float(grad.mean()))
                out.append(r)
                rows.append(f"| {arm} | {r['seed']} | {r['statistic']} | {f(r['observed_mean_rho'])} | {r['p_exact_one_sided']:.4f} | "
                            f"{'yes' if r['is_most_extreme'] else 'no'} | {f(r['null_q025'])} |")
    head = ["| arm | seed | statistic | observed mean ρ | exact p | most extreme of 630 | null 2.5th pct |", "|---|---|---|---|---|---|---|"]
    return {"rows": out}, head + rows


# ----------------------------------------------------------------------------------------------- S2
def cc_streams(run: str) -> tuple[np.ndarray, np.ndarray]:
    """ed_tables.py:cc_streams on ENC: median over nc > 0.1 voxels of the per-voxel in-silico shape-bias index per stream."""
    res = fits(run)
    mg, pv = [], []
    for s in SUBJECTS:
        z = res[s]
        ok = z["noise_ceiling"] > NC_MIN
        med = lambda v: float(np.median(v[ok & np.isfinite(v)]))
        mg.append(med(z["cc_magno_vsb"])); pv.append(med(z["cc_parvo_vsb"]))
    return np.array(mg), np.array(pv)


def s2() -> tuple[dict, list[str]]:
    per = {run: cc_streams(run) for runs in ARMS.values() for run in runs}
    per[UNTRAINED_ENC] = cc_streams(UNTRAINED_ENC)
    out, rows = {"per_run": {}, "pairs": {}}, []
    for run, (m, p) in per.items():
        out["per_run"][run] = {"magno": m.tolist(), "parvo": p.tolist(), "magno_minus_parvo": (m - p).tolist(),
                               "magno_mean": float(m.mean()), "parvo_mean": float(p.mean()), "diff_mean": float((m - p).mean())}
    for sd in range(3):
        dm = per[MODEL[sd]][0] - per[MODEL[sd]][1]
        for arm, runs in ARMS.items():
            run = runs[sd]
            m, p = per[run]; d = m - p
            cells = [str(sd), arm, f(m.mean()), f(p.mean()), f(d.mean())]
            if arm == "model":
                for other in ("control", "channel swap"):
                    do = per[ARMS[other][sd]][0] - per[ARMS[other][sd]][1]
                    w = wil(dm - do)
                    out["pairs"][f"model_minus_{other.replace(' ', '_')}_seed{sd}"] = w
                    cells += [f(w["mean"]), f"{int((dm > do).sum())}/8", pfmt(w["p"])]
            else:
                cells += ["", "", "", "", "", ""]
            rows.append("| " + " | ".join(cells) + " |")
    m, p = per[UNTRAINED_ENC]
    rows.append(f"| — | untrained | {f(m.mean())} | {f(p.mean())} | {f((m - p).mean())} | | | | | | |")
    head = ["| seed | arm | magno stream | parvo stream | magno − parvo | model − control | sep. | p | model − swap | sep. | p |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    return out, head + rows


# ----------------------------------------------------------------------------------------------- S3
S3_CELLS = [("chromatic (L−M)", "none", "control, L−M stream", "chromatic_none", "control", "parvo"),
            ("chromatic (L−M)", "yes", "channel swap, L−M stream", "chromatic_transform", "channel swap", "parvo"),
            ("achromatic (luminance)", "none", "control, luminance stream", "achromatic_none", "control", "magno"),
            ("achromatic (luminance)", "yes", "model, luminance stream", "achromatic_transform", "model", "magno")]


def shared_bias(run: str, stream: str) -> float:
    return jload(os.path.join(RES, "shape_texture", f"{run}.json"))["streams"][stream]["bias"]


def cc(run: str, cond: str, key: str = "shape_bias"):
    c = jload(os.path.join(RES, "cue_conflict", f"{run}.json"))["conditions"][cond]
    return c["n_shape"] + c["n_texture"] if key == "decided" else c[key]


def s3_rows(two: dict, ex_streams: dict, shared, conflict) -> tuple[dict, list[str]]:
    """two = exemplar_streams.json['two_by_two_v3'] (Q1 held out, Q0 shared); shared(run, stream), conflict(run, cond, key)."""
    Q1, Q0 = two["Q1"], two["Q0"]
    out, rows = {"cells": {}, "interaction_held_out": float(Q1["interaction"]), "interaction_shared": float(Q0["interaction"]),
                 "means_held_out": {k: float(np.mean(v)) for k, v in Q1["cells"].items()},
                 "means_shared": {k: float(np.mean(v)) for k, v in Q0["cells"].items()}}, []
    mm = out["means_held_out"]
    out["ramp_effect_achromatic"] = mm["achromatic_transform"] - mm["achromatic_none"]
    out["ramp_effect_chromatic"] = mm["chromatic_transform"] - mm["chromatic_none"]
    for carrier, ramp, cond, key, arm, stream in S3_CELLS:
        runs = ARMS[arm]
        held = np.array(Q1["cells"][key], float); sh0 = np.array(Q0["cells"][key], float)
        sh = np.array([shared(r, stream) for r in runs]); be = np.array([conflict(r, f"{stream}_only", "shape_bias") for r in runs])
        de = np.array([conflict(r, f"{stream}_only", "decided") for r in runs])
        assert len(held) == len(runs) == 3 and np.allclose(np.sort(sh), np.sort(sh0), atol=5e-4), (key, sh, sh0)
        out["cells"][key] = {"condition": cond, "runs": runs, "stream": stream, "probe_shared": sh.tolist(), "probe_held_out": held.tolist(),
                             "cue_conflict": be.tolist(), "decided": de.tolist()}
        rows.append(f"| {carrier} | {ramp} | {cond} | {mptp(sh, True)} | {mptp(held, True)} | {mptp(be)} | {de.mean():.0f} | {len(runs)} |")
    # the same chromatic cell read in the model (Results 2.2 quotes it beside the control's); not a cell of the interaction
    held = np.array([ex_streams[r]["parvo"]["Q1"]["bias"] for r in MODEL]); sh = np.array([shared(r, "parvo") for r in MODEL])
    be = np.array([conflict(r, "parvo_only", "shape_bias") for r in MODEL]); de = np.array([conflict(r, "parvo_only", "decided") for r in MODEL])
    out["cells"]["chromatic_none_model"] = {"condition": "model, L−M stream (not a cell of the interaction)", "runs": MODEL, "stream": "parvo",
                                            "probe_shared": sh.tolist(), "probe_held_out": held.tolist(), "cue_conflict": be.tolist(), "decided": de.tolist()}
    rows.append(f"| chromatic (L−M) | none | model, L−M stream (same cell read in the model; not in the interaction) | {mptp(sh, True)} | "
                f"{mptp(held, True)} | {mptp(be)} | {de.mean():.0f} | 3 |")
    rows.append("")
    rows.append(f"Carrier × ramp interaction (achromatic ramp effect − chromatic ramp effect): {f(out['interaction_held_out'], 2)} with exemplars held out, "
                f"{f(out['interaction_shared'], 2)} with exemplars shared; the ramp moves the achromatic stream by {f(out['ramp_effect_achromatic'], 2)} "
                f"and the chromatic stream by {f(out['ramp_effect_chromatic'], 2)} (held out).")
    head = ["| carrier | ramp | condition, stream | probe bias | probe bias, exemplars held out | cue-conflict shape bias | decided / 1,200 | seeds |",
            "|---|---|---|---|---|---|---|---|"]
    return out, head + rows


def s3() -> tuple[dict, list[str]]:
    E = jload(EXEMPLAR)
    return s3_rows(E["two_by_two_v3"], E["streams"], shared_bias, cc)


# ----------------------------------------------------------------------------------------------- S4
def n_params(run: str) -> dict:
    """Parameters of the stored final checkpoint (EMA weights), BatchNorm running statistics excluded; by module prefix."""
    import torch
    ck = torch.load(track(os.path.join(CKPT, run, "checkpoint_final.pt")), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    by = {}
    for k, v in sd.items():
        if k.endswith(("running_mean", "running_var", "num_batches_tracked")) or not torch.is_tensor(v):
            continue
        by[k.split(".")[0]] = by.get(k.split(".")[0], 0) + int(v.numel())
    return {"total": int(sum(by.values())), "by_module": by, "weights": str(ck.get("weights", ""))}


def sweep_at(run: str, w: float) -> float:
    sw = jload(os.path.join(RES, "fusion_sweep", f"{run}_final_wsweep.json"))["sweep"]
    return [q for q in sw if abs(q["w"] - w) < 1e-6][0]["shape_bias"]


def s4() -> tuple[dict, list[str]]:
    out, rows = {"arms": {}, "single_stream_controls": "not trained on the dvd300 schedule; rows not regenerated (plan section 1, S4)"}, []
    for arm, runs in ARMS.items():
        P = n_params(runs[0])
        acc = np.array([jload(HIST.format(r))[-1]["val_top1"] for r in runs])
        out["arms"][arm] = {"runs": runs, "parameters": P, "accuracy_final": acc.tolist(), "readouts": {}}
        for lab, cond, w in READOUTS:
            sb = np.array([cc(r, cond, "shape_bias") for r in runs]); de = np.array([cc(r, cond, "decided") for r in runs])
            swv = np.array([sweep_at(r, w) for r in runs])
            out["arms"][arm]["readouts"][cond] = {"label": lab, "shape_bias": sb.tolist(), "decided": de.tolist(), "sweep_shape_bias_at_w": swv.tolist(),
                                                  "max_abs_diff_vs_sweep": float(np.max(np.abs(sb - swv)))}
            rows.append(f"| {arm}, {lab} | {P['total']:,} | {rng(sb)} | {de.mean():.0f} | {rng(acc)} |")
    rows.append("| single-stream controls on the dvd300 schedule | — | not trained | — | — |")
    head = ["| model / readout | parameters | shape bias | decided / 1,200 | accuracy |", "|---|---|---|---|---|"]
    return out, head + rows


# ----------------------------------------------------------------------------------------------- S5
def rel_auc(d: dict) -> dict:
    """ed_tables.py:auc for one run: mean over severities of accuracy / accuracy at severity 0, per family."""
    return {fam: float((np.array(d["families"][fam]["accuracy"], float) / d["families"][fam]["accuracy"][0]).mean()) for fam in FAM}


def s5() -> tuple[dict, list[str]]:
    A = {arm: {fam: np.array([rel_auc(jload(os.path.join(RES, "ood", f"{r}.json")))[fam] for r in runs]) for fam in FAM} for arm, runs in ARMS.items()}
    out, rows = {"per_family": {}}, []
    cell = lambda v: f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]"
    for fam in FAM:
        m, c, s = A["model"][fam], A["control"][fam], A["channel swap"][fam]
        sep = bool((m.max() < c.min()) or (m.min() > c.max()))
        out["per_family"][fam] = {"model": m.tolist(), "control": c.tolist(), "channel swap": s.tolist(), "model_minus_control": float(m.mean() - c.mean()),
                                  "separated_3v3": sep, "trained_in": fam in TRAINED_IN}
        rows.append(f"| {fam}{' ‡' if fam in TRAINED_IN else ''} | {cell(m)} | {cell(c)} | {cell(s)} | {f(m.mean() - c.mean())} | {'yes' if sep else 'no'} |")
    head = ["| family | model | control | channel swap | model − control | sep. |", "|---|---|---|---|---|---|"]
    return out, head + rows


# ----------------------------------------------------------------------------------------------- S6
def s6() -> tuple[dict, list[str]]:
    out, rows = {"runs": {}}, []
    for arm, runs in ARMS.items():
        for run in runs:
            r = jload(os.path.join(RES, "sweep_error_consistency", f"{run}_final_kappa_sweep.json"))
            ceil = r["human_human_ceiling"]
            sw = sorted(r["sweep"], key=lambda q: q["w"])
            mid = [q for q in sw if abs(q["w"] - 0.5) < 1e-6][0]; top = [q for q in sw if abs(q["w"] - 1.0) < 1e-6][0]
            best = max(sw, key=lambda q: q["kappa_mean"])
            out["runs"][run] = {"arm": arm, "ceiling": ceil, "w05": mid, "w1": top, "best": best}
            rows.append(f"| {r['model_name']} | {mid['shape_bias']:.3f} | {f(mid['kappa_mean'], 4)} | {mid['kappa_pct_ceiling']:.1f} | {f(top['kappa_mean'], 4)} | "
                        f"{top['kappa_pct_ceiling']:.1f} | {top['shape_bias']:.3f} | {f(best['kappa_mean'], 4)} | {best['kappa_pct_ceiling']:.1f} | {best['w']:.2f} | "
                        f"{best['shape_bias']:.3f} |")
    head = ["| run | shape bias at w = 0.5 | κ at w = 0.5 | % ceiling | κ at w = 1 (magno alone) | % ceiling | rate at w = 1 | best κ over w | % ceiling | w at best | rate at best |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    return out, head + rows


# ----------------------------------------------------------------------------------------------- S7
def ratio(m, p):
    return (m - p) / (m + p)


def con(v):
    return v[6] - v[[0, 1]].mean()


def real_contrast(run: str, enc_dir: str) -> np.ndarray:
    """floor_pairing.py's `real`: per subject, hV4 - mean(V1v, V1d) of the per-ROI median per-voxel ratio index."""
    out = []
    for p in sorted(glob.glob(os.path.join(enc_dir, f"{run}_subj*.npz"))):
        z = npz(p)
        roi, nc, um, up = z["roi_prf-visualrois"], z["noise_ceiling"], z["unique_magno"], z["unique_parvo"]
        p7 = []
        for c in range(1, 8):
            v = (roi == c) & (nc > NC_MIN) & ((um + up) > DENOM_MIN)
            rr = (um[v] - up[v]) / (um[v] + up[v]) if v.sum() >= MIN_VOX else np.array([np.nan])
            p7.append(np.nanmedian(rr))
        out.append(con(np.array(p7)))
    return np.array(out)


def floor_variants(run: str, vprec_dir: str, enc_dir: str) -> dict:
    """tmp_analysis/floor_pairing.py, with the two source directories passed in (that file hard-codes PAPER_v2's)."""
    files = sorted(glob.glob(os.path.join(vprec_dir, f"vp_recovery_{run}_subj*.json")))
    assert len(files) == 8, (run, vprec_dir, len(files))
    ds = [jload(p) for p in files]
    assert [int(d["subject"]) for d in ds] == SUBJECTS
    Bs = np.array([[ratio(r["est_unique_m"], r["est_unique_p"]) for r in d["conditions"]["B_manufacture"]["rows"]] for d in ds])
    for d in ds:
        assert max(abs(r["true_diff"]) for r in d["conditions"]["B_manufacture"]["rows"]) < 1e-9
    floor_group = float(con(Bs.mean(0)))
    floor_subj = np.array([con(Bs[i]) for i in range(len(ds))])
    T = np.array([[ratio(r["true_unique_m"], r["true_unique_p"]) for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    E = np.array([[ratio(r["est_unique_m"], r["est_unique_p"]) for r in d["conditions"]["A_recovery"]["rows"]] for d in ds])
    Y = E - Bs.mean(0)
    a = float(np.sum(T * Y) / np.sum(T * T))
    rc = float(np.mean([con((Y - a * T)[i]) for i in range(len(ds))]))
    real = real_contrast(run, enc_dir)
    assert len(real) == 8
    out = {"run": run, "a_slope_intercept_fixed_at_group_floor": a, "floor_group": floor_group, "floor_per_subject": floor_subj.tolist(),
           "residual_contrast": rc, "real_per_subject": real.tolist(), "per_roi_floor_mean": Bs.mean(0).tolist(), "variants": {}}
    # diet_floor_paired.py: mean over subjects of the per-subject Spearman rho of the floor profile against ROI rank;
    # the p is the exact Wilcoxon of the eight per-subject rho against zero (PAPER_v2's row carried a hand-entered p)
    rhos = np.array([sps.spearmanr(np.arange(7), Bs[i])[0] for i in range(len(ds))])
    out["per_roi_floor_rho_vs_rank"] = {"rho": float(rhos.mean()), "per_subject": rhos.tolist(), "p": wil(rhos)["p"]}
    for lab, fl in (("group floor", floor_group), ("per-subject floor", floor_subj)):
        for lab2, vv in (("minimal", (real - fl) / a), ("+ residual before scaling", (real - fl - rc) / a), ("+ residual after scaling", (real - fl) / a - rc)):
            out["variants"][f"{lab}, {lab2}"] = wil(vv)
    return out


def s7() -> tuple[dict, list[str]]:
    os.environ.setdefault("FS2_VPREC_DIR", VPREC); os.environ.setdefault("FS2_ENCODING_DIR", ENC)
    import fs2_cortex_controls as FCC          # reads the two directories from the environment at import
    assert FCC.SIM == VPREC and FCC.ENC == ENC, (FCC.SIM, FCC.ENC)
    for p in sorted(glob.glob(os.path.join(VPREC, "vp_recovery_*_subj*.json"))):
        track(p)
    sb = FCC.sim_block()
    assert "skipped" not in sb, sb
    V = {arm: floor_variants(run, VPREC, ENC) for arm, run in (("model", MODEL[0]), ("control", CONTROL[0]))}
    out = {"sim_block": sb, "variants": V}
    rows = []
    cell = lambda w, pos=False: f"{f(w['mean'], 4)} ({w['n_pos'] if pos else w['n_neg']}/{w['n']}{' > 0' if pos else ''}, p = {pfmt(w['p'])})"
    rows.append(f"| floor, normalised contrast (true = 0) | {cell(sb['model']['floor_norm'], True)} | {cell(sb['control']['floor_norm'], True)} |")
    rows.append(f"| floor, raw contrast | {f(sb['model']['floor_raw']['mean'], 4)} (p = {pfmt(sb['model']['floor_raw']['p'])}) | "
                f"{f(sb['control']['floor_raw']['mean'], 4)} (p = {pfmt(sb['control']['floor_raw']['p'])}) |")
    rows.append(f"| real contrast, normalised (seed 0) | {f(np.mean(sb['model']['real_norm_seed0']), 4)} | {f(np.mean(sb['control']['real_norm_seed0']), 4)} |")
    rows.append(f"| displacement real − floor | {cell(sb['model']['displacement_norm'])} | {cell(sb['control']['displacement_norm'])} |")
    rows.append(f"| recovered slope of estimate vs truth (intercept fixed at the group floor) | {V['model']['a_slope_intercept_fixed_at_group_floor']:.3f} | "
                f"{V['control']['a_slope_intercept_fixed_at_group_floor']:.3f} |")
    for k in V["model"]["variants"]:
        rows.append(f"| variant: {k} | {cell(V['model']['variants'][k])} | {cell(V['control']['variants'][k])} |")
    for arm in ("model", "control"):
        pr = V[arm]["per_roi_floor_mean"]; rh = V[arm]["per_roi_floor_rho_vs_rank"]
        txt = ", ".join(f(x) for x in pr) + f" (ρ vs rank {f(rh['rho'], 2)}, p = {pfmt(rh['p'])})"
        rows.append(f"| per-ROI normalised floor, V1v … hV4 ({arm} geometry) | {txt if arm == 'model' else ''} | {txt if arm == 'control' else ''} |")
    head = ["| quantity | model | control |", "|---|---|---|"]
    return out, head + rows


# ----------------------------------------------------------------------------------------------- S8
def univariate(run: str, stat: str) -> dict[int, dict]:
    """z-scored univariate slopes of the per-voxel index on log(ecc + 0.5) alone and on log(size) alone, with
    fs2_cortex_controls.py's voxel selection (pRF R2 > 5%, finite ecc/size, size > 0, >= 200 voxels)."""
    import fs2_cortex_controls as FCC
    res = fits(run)
    out = {}
    for s in SUBJECTS:
        z = res[s]
        ecc, size, r2 = FCC.prf(s)
        val, ok = FCC.index(z, stat == "raw")
        sel = ok & (r2 > FCC.R2_MIN) & np.isfinite(ecc) & np.isfinite(size) & (size > 0)
        if sel.sum() < FCC.MIN_REG:
            out[s] = None; continue
        y = FCC.zs(val[sel]); le, ls = FCC.zs(np.log(ecc[sel] + 0.5)), FCC.zs(np.log(size[sel]))
        out[s] = {"ecc_alone": float(np.linalg.lstsq(np.column_stack([np.ones(len(y)), le]), y, rcond=None)[0][1]),
                  "size_alone": float(np.linalg.lstsq(np.column_stack([np.ones(len(y)), ls]), y, rcond=None)[0][1]), "n_vox": int(sel.sum())}
    return out


def s8() -> tuple[dict, list[str]]:
    C = jload(CORTEX_CONTROLS)
    R = C["per_seed_regression"]
    runs = [(arm, r) for arm, rs in ARMS.items() for r in rs] + [("untrained", UNTRAINED_ENC)]
    out, rows = {"runs": {}, "paired": {}, "interaction": {}}, []
    cellf = lambda w: f"{f(w['mean'])} ({w['n_neg']}/{w['n']} neg, {pfmt(w['p'])})"
    for arm, run in runs:
        for stat in ("norm", "raw"):
            if stat == "raw" and not (run.endswith("seed0") or arm == "untrained"):
                continue
            reg = R[run][stat]
            ecc = np.array([reg[str(s)]["ecc_partial"] for s in SUBJECTS]); size = np.array([reg[str(s)]["size_partial"] for s in SUBJECTS])
            vif = float(np.mean([reg[str(s)]["vif"] for s in SUBJECTS]))
            U = univariate(run, stat)
            ea = np.array([U[s]["ecc_alone"] for s in SUBJECTS]); sa = np.array([U[s]["size_alone"] for s in SUBJECTS])
            w = {"ecc_partial": wil(ecc), "size_partial": wil(size), "ecc_alone": wil(ea), "size_alone": wil(sa), "vif_mean": vif}
            out["runs"].setdefault(run, {})[stat] = w
            lab = f"{arm}, seed {seed_of(run)}" if arm != "untrained" else "untrained network, shared front end"
            rows.append(f"| {lab} | {'normalised' if stat == 'norm' else 'raw'} | {cellf(w['ecc_alone'])} | {cellf(w['size_alone'])} | "
                        f"{cellf(w['ecc_partial'])} | {cellf(w['size_partial'])} |")
    cellp = lambda w: f"{f(w['mean'])} ({w['n_pos']}/{w['n']} pos, {pfmt(w['p'])})"
    for arm in ("model", "control"):
        for stat in ("norm", "raw"):
            a, u = R[ARMS[arm][0]][stat], R[UNTRAINED_ENC][stat]
            d_e = np.array([a[str(s)]["ecc_partial"] - u[str(s)]["ecc_partial"] for s in SUBJECTS])
            d_s = np.array([a[str(s)]["size_partial"] - u[str(s)]["size_partial"] for s in SUBJECTS])
            out["paired"][f"{arm}_seed0_minus_untrained_{stat}"] = {"ecc_partial": wil(d_e), "size_partial": wil(d_s)}
            rows.append(f"| trained {arm} (seed 0) − untrained, paired | {'normalised' if stat == 'norm' else 'raw'} | | | {cellp(wil(d_e))} | {cellp(wil(d_s))} |")
    for stat in ("norm", "raw"):
        def seed_mean(arm, key):
            return np.mean([[R[r][stat][str(s)][key] for s in SUBJECTS] for r in ARMS[arm]], 0)
        d_e = seed_mean("model", "ecc_partial") - seed_mean("control", "ecc_partial")
        d_s = seed_mean("model", "size_partial") - seed_mean("control", "size_partial")
        out["interaction"][stat] = {"ecc_partial": wil(d_e), "size_partial": wil(d_s)}
        rows.append(f"| arm interaction, model − control (seed-averaged) | {'normalised' if stat == 'norm' else 'raw'} | | | {cellp(wil(d_e))} | {cellp(wil(d_s))} |")
    head = ["| model | statistic | ecc alone | size alone | ecc \\| size | size \\| ecc |", "|---|---|---|---|---|---|"]
    return out, head + rows


# ----------------------------------------------------------------------------------------------- S9
def s9() -> tuple[dict, list[str]]:
    import forced_choice as FC
    out, rows = {}, []
    for arm, runs in ARMS.items():
        pat = os.path.join(RES, "cue_conflict", runs[0].replace("seed0", "seed?") + "_logits.npz")
        for p in sorted(glob.glob(pat)):
            track(p)
        for lab, cond, _ in READOUTS:
            rr = FC.arm(pat, cond)
            assert len(rr) == 3, (pat, cond, rr)
            fc = np.array([r["forced"] for r in rr]); rs = np.array([r["restricted"] for r in rr]); de = np.array([r["decided"] for r in rr])
            out[f"{arm}, {lab}"] = {"runs": runs, "forced": fc.tolist(), "restricted": rs.tolist(), "decided": de.tolist()}
            rows.append(f"| {arm}, {lab} (final ckpt) | {rng(rs)} | {de.mean():.0f} | {rng(fc)} | {len(rr)} |")
    head = ["| model / readout | restricted (primary) | decided / 1,200 | forced two-way | seeds |", "|---|---|---|---|---|"]
    return out, head + rows


# ----------------------------------------------------------------------------------------------- S10
S10_ORDER = [("Kastner2015", ["VO1", "VO2", "PHC1", "PHC2", "LO1", "LO2", "TO1", "TO2", "V3A", "V3B", "IPS0", "IPS1", "IPS2"]),
             ("streams", ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]),
             ("floc-faces", ["OFA", "FFA-1", "FFA-2"]), ("floc-places", ["OPA", "PPA"]),
             ("floc-bodies", ["EBA", "FBA-1", "FBA-2"]), ("floc-words", ["OWFA", "VWFA-1", "VWFA-2", "mfs-words"])]


def s10() -> tuple[dict, list[str]]:
    import ext_gradient as EG
    arms = {"untrained": (ENC, [UNTRAINED_ENC]), "control": (ENC, CONTROL), "model": (ENC, MODEL), "channel swap": (ENC, SWAP)}
    for _, runs in arms.values():
        for r in runs:
            fits(r)
    stored = jload(EXT_GRADIENT)
    profs = {arm: EG.profile(arm, arms) for arm in arms}
    atlases = [("Kastner2015", EG.KASTNER), ("streams", EG.STREAMS)] + list(EG.FLOC.items())
    tab = {atlas: {arm: EG.table(profs[arm], atlas, names) for arm in arms} for atlas, names in atlases}
    # the recomputed fs2 arms must reproduce the stored record exactly
    check = {}
    for atlas, names in atlases:
        for arm in ("untrained", "control", "model"):
            for name in names.values():
                a, b = tab[atlas][arm][name]["mean"], stored[atlas][arm][name]["mean"]
                if a is not None and b is not None:
                    check[f"{atlas}/{arm}/{name}"] = abs(a - b)
    out = {"recompute_max_abs_diff_vs_stored": float(max(check.values())), "n_compared": len(check), "tables": tab,
           "stored_ba": {atlas: {arm: stored[atlas][arm] for arm in ("BA no diet", "BA")} for atlas, _ in atlases}}
    cols = [("untrained", tab), ("control", tab), ("model", tab), ("channel swap", tab), ("BA no diet", out["stored_ba"]), ("BA", out["stored_ba"])]
    ff = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:+.2f}".replace("-", "−")
    rows = []
    for atlas, names in S10_ORDER:
        for r in names:
            cells = []
            for arm, src in cols:
                t = src[atlas][arm][r]
                cells.append("—" if t["mean"] is None else f"{ff(t['mean'])} (n={t['n_subjects']})")
            rows.append(f"| {atlas} | {r} | " + " | ".join(cells) + f" | {tab[atlas]['model'][r]['median_vox']:.0f} |")
    head = ["| atlas | ROI | untrained (shared front end) | control (3 seeds) | model (3 seeds) | channel swap (3 seeds) | BA no diet (3 seeds) | BA (3 seeds) | voxels (median, model) |",
            "|---|---|---|---|---|---|---|---|---|"]
    return out, head + rows


# ----------------------------------------------------------------------------------------------- driver
TABLES = {"S1": (s1, "Monotonicity of the profile and the exact permutation null — model of record.",
                 "Spearman ρ between hierarchical rank [1,1,2,2,3,3,4] and stream preference (per-voxel index: median over reliable voxels of (U_m − U_p)/(U_m + U_p); "
                 "ratio of medians: median(U_m − U_p)/median(U_m + U_p)), mean over 8 subjects; the null enumerates all 630 distinct labellings of the seven ROIs "
                 "(one-sided in the pre-stated direction, floor 1/630 = .0016). Source: the stored voxelwise fits of the model, its control, the channel swap and the untrained network."),
          "S2": (s2, "In-silico cue conflict through the fitted encoding models, per seed — model of record.",
                 "Median over reliable voxels (noise ceiling > 0.1) of the per-voxel predicted shape-bias index per stream, mean over 8 subjects; pairings are per subject "
                 "against the seed of the same index (exact Wilcoxon, floor p = .0078). This is PAPER_v2's in-silico index on the NSD-image fits; the cortex-to-decision "
                 "rate of Results 2.5 refits the encoders on cue-conflict features and is a different instrument."),
          "S3": (s3, "The carrier × ramp 2×2 on the probe and on behaviour — model of record.",
                 "Mean over training seeds with the seed range in brackets. “probe bias” is the stratified-fold probe (Methods 4.3), “exemplars held out” the exemplar-balanced "
                 "probe (results/figures_v2/exemplar_streams.json, two_by_two_v3, Q1), “cue-conflict shape bias” the restricted estimator of the named stream read through "
                 "the shared head, “decided” its denominator. Cells are the four of Results 2.2."),
          "S4": (s4, "The readout ladder — model of record.",
                 "Final checkpoint of every run; mean [range] over three seeds; accuracy is validation top-1 at that checkpoint; parameters are counted from the stored "
                 "state dict with BatchNorm running statistics excluded. No single-trunk network was trained on the dvd300 schedule, so the single-stream rows of the BA "
                 "table have no counterpart (Supplementary Note S1)."),
          "S5": (s5, "Out-of-distribution robustness: relative-accuracy AUC per corruption family — model of record.",
                 "Mean over three seeds [range]; “sep.” marks complete seed separation between the model and the control (exact 3 v 3 floor, p = .100); families marked ‡ "
                 "were declared trained-in before the BA run and keep the mark. Source: results/ood/fs2_*.json."),
          "S6": (s6, "Error consistency (Cohen's κ) with the ten human observers — model of record.",
                 "Final checkpoints; the trained readout (w = 0.5), the magno stream alone (w = 1) and the best readout over the 21 values of w, per run; κ as a percentage "
                 "of the human–human ceiling (κ = 0.234, 45 pairs). Source: results/sweep_error_consistency/fs2_*_final_kappa_sweep.json."),
          "S7": (s7, "The estimator's own bias, and the residual-misfit variants — model of record.",
                 "Seed 0 of the model and the control, as the BA table (seeds 1–2 and the channel swap were not simulated). Definitions as the BA rows: the floor is the "
                 "recovered hV4 − mean(V1v, V1d) contrast under a true difference of zero; the six variants differ in how the floor is paired and whether the residual misfit "
                 "of the recovery simulation is removed before or after rescaling by the recovered slope. Source: results/vp_recovery_realnc_v3/."),
          "S8": (s8, "Stream preference regressed on pRF eccentricity and pRF size within subject — model of record.",
                 "Per-voxel stream-preference index regressed on log pRF eccentricity and log pRF size (NSD pRF estimates; voxels with pRF R² > 5%, noise ceiling > 0.1); "
                 "“ecc | size” is the partial slope for eccentricity controlling for size and vice versa; mean over 8 subjects, with the number of subjects on the stated side "
                 "and the exact Wilcoxon p. Partial slopes are the stored record of Extended Data Fig. 3 (results/figures_v3/cortex_controls.json); the univariate slopes "
                 "are computed here with the same voxel selection."),
          "S9": (s9, "Cue-conflict shape bias under the forced two-way estimator — model of record.",
                 "The restricted (primary) estimator and the forced two-way estimator over all 1,200 images, from the stored per-image logits of the final checkpoints "
                 "(results/cue_conflict/fs2_*_logits.npz); mean over seeds [range]."),
          "S10": (s10, "Stream preference beyond hV4, on every NSD atlas — model of record.",
                  "Median over reliable voxels of the per-voxel index (noise ceiling > 0.1, U_m + U_p > 10⁻³, ≥ 20 voxels per ROI per subject), mean over the eight subjects, "
                  "each trained arm the mean over three seeds per subject. Positive = magno band, negative = parvo band. The BA columns are PAPER_v2's stored values; "
                  "the fs2 arms are recomputed from the stored voxelwise fits (no refitting) and reproduce results/figures_v3/ext_gradient.json.")}


def commit_of(root: str) -> str:
    cf = os.path.join(root, "COMMIT")
    if os.path.isfile(cf):
        return open(cf).read().strip()
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = [t.strip() for t in args.only.split(",") if t.strip()] or list(TABLES)
    os.makedirs(args.out, exist_ok=True)
    J, md = {"plan": "docs/PLAN_SUPP_MODEL_2026-11-03.md", "arms": {**ARMS, "untrained": [UNTRAINED_ENC]}, "tables": {}}, []
    for key in only:
        fn, title, caption = TABLES[key]
        print(f"[supp] {key} ...", flush=True)
        data, lines = fn()
        J["tables"][key] = data
        md += [f"**Supplementary Table {key} | {title}** {caption}", ""] + lines + [""]
        print(f"[supp] {key} done: {sum(1 for l in lines if l.startswith('|') and not l.startswith('|---'))-1} rows", flush=True)
    with open(os.path.join(args.out, "supp_tables_model.json"), "w", encoding="utf8") as fh:
        json.dump(J, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(args.out, "supp_tables_model.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(md).rstrip() + "\n")
    import scipy
    prov = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "commit": commit_of(ROOT), "root": ROOT, "tables": only,
            "env": {"ENC": ENC, "VPREC": VPREC, "CKPT": CKPT, "NSD_ROOT": os.environ.get("NSD_ROOT", "(default)")},
            "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "python": sys.version.split()[0],
            "numpy": np.__version__, "scipy": scipy.__version__, "inputs": dict(sorted(_INPUTS.items()))}
    with open(os.path.join(args.out, "provenance.json"), "w", encoding="utf8") as fh:
        json.dump(prov, fh, indent=1)
    print(f"[supp] wrote {args.out}: {len(only)} tables, {len(_INPUTS)} inputs tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
