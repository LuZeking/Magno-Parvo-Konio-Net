#!/usr/bin/env python
"""Re-derive every number docs/PAPER_v3.md states about its model of record and that model's controls, straight
from the result files, then run tmp_analysis/verify_new_claims.py as the comparison block for the BA network and
the model zoo, whose numbers PAPER_v3 carries over from PAPER_v2.

Model of record: fs2_dvd300_avg_w050_ls, seeds 0-2. Control: fs2_dvdfix300_avg_w050_ls, the adult diet from the
first step, seeds 0-2. Channel
swap: fs2_dvd300swap_avg_w050_ls, the ramp on L-M with luminance untransformed, seeds 0-2. Fixed BA diet under the
model's recipe: fs2_bcs_avg_w050_ls.

Every check quotes a phrase of PAPER_v3 and the number printed in it, and fails if the phrase is no longer in the
paper (the text changed: update the check) or if the recomputed value does not round to the printed one (half a
unit of its last decimal unless a tolerance is given). Blocks that need inputs outside the repository (the
per-voxel encoding and readout-weight fits on /scratch) are skipped and listed, never passed. Results the text
still marks pending are listed when they land. Exits non-zero if any check fails, here or in the BA block.

    PYTHONPATH=. python tmp_analysis/verify_v3_claims.py          # v3 checks, then the BA comparison block
    PYTHONPATH=. python tmp_analysis/verify_v3_claims.py --no-ba  # v3 checks only
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import linregress, mannwhitneyu, t as tdist, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verify_new_claims as ba  # noqa: E402  (tmp_analysis/ is on sys.path when run as a script)

PAPER = open("docs/PAPER_v3.md", encoding="utf8").read()
MODEL = [f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)]
CONTROL_SEEDS = (0, 1, 2)  # behaviour: the text reports all three seeds (2026-10-23)
CONTROL = [f"fs2_dvdfix300_avg_w050_ls_seed{s}" for s in CONTROL_SEEDS]
# cortex: the control's fits cover seeds 0-2 since 2026-10-24 (seed 0 alone before)
CORTEX_SEEDS = (0, 1, 2)
CORTEX_CONTROL = [f"fs2_dvdfix300_avg_w050_ls_seed{s}" for s in CORTEX_SEEDS]
SWAP = [f"fs2_dvd300swap_avg_w050_ls_seed{s}" for s in range(3)]
FIXED = [f"fs2_bcs_avg_w050_ls_seed{s}" for s in range(3)]
UNTRAINED = "untrained_dvd300"
HIST = "docs/slides_2026-10-12_ba_vs_fusion/figures_ba/{}__history.json"
CORTEX = "docs/slides_2026-10-12_ba_vs_fusion/figures/fs2_cortex_summary.json"
EXEMPLAR = "results/figures_v2/exemplar_streams.json"
ENC = os.environ.get("FS2_ENCODING_DIR", "/scratch/byrong/mpk/encoding/results/2026-10-16_fs2")
RW = os.environ.get("FS2_READOUT_W_DIR", "/scratch/byrong/mpk/readout_w/results/2026-10-16_fs2_w21")
EXTERNAL = {"encoding": (ENC, "per-voxel banded-ridge fits of the model and control (FS2_ENCODING_DIR)"),
            "readout_w": (RW, "per-voxel readout-weight fits of the model and control (FS2_READOUT_W_DIR)")}
# Numbers PAPER_v3 states but neither this file nor the BA block re-derives.
CARRIED = ["0.556 / 0.445: the BA network's top-1 at its best saved / final checkpoint, thesis recipe (PAPER_v2)",
           "1.60–2.51 against 0.91–0.99: the BA networks' participation-ratio seed means (PAPER_v2, Extended Data Fig. 2)",
           "AlexNet, GoogLeNet, VGG-16 rates and κ in the seven-model fit (Geirhos et al. 2018a; PAPER_v2 Supplementary Table S6)",
           "about 4.3 h / 18.3–21.3 h per run, 99.9 GPU-hours: Slurm accounting of the training arrays (sacct 26637895, "
           "26641053 on the B200; 26671132, 26671793 on RTX A5000s)"]
# Rate and κ of the three published CNNs in the seven-model fit, from the Geirhos et al. (2018a) response files.
PUBLISHED_CNNS = {"AlexNet": (0.429, 0.032), "GoogLeNet": (0.312, 0.025), "VGG-16": (0.172, 0.017)}

FAILS: list[str] = []
SKIPS: list[tuple[str, str]] = []
LANDED: list[str] = []
NUM = re.compile(r"[−+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


def _val(lit: str) -> float:
    return float(lit.replace("−", "-").replace(",", ""))


def said(name: str, got: float, phrase: str, lit: str | None = None, tol: float | None = None, doc: str | None = None) -> None:
    """`phrase` is copied from PAPER_v3 (or from `doc`, a figure caption) and must still be in it; `lit` (default: the
    phrase's last number) is the printed value, and `got` must round to it."""
    lit = NUM.findall(phrase)[-1] if lit is None else lit
    dec = len(lit.split(".")[1]) if "." in lit else 0
    tol = 0.5 * 10 ** -dec + 1e-9 if tol is None else tol
    found = phrase in (PAPER if doc is None else doc) and lit in phrase
    ok = found and abs(float(got) - _val(lit)) <= tol
    print(f"  {'OK ' if ok else 'BAD'} {name:50s} got {float(got):+.4f}  text {lit}" + ("" if found else "  <- phrase not in PAPER_v3"))
    if not ok:
        FAILS.append(name)


def holds(name: str, ok: bool, phrase: str, doc: str | None = None) -> None:
    found = phrase in (PAPER if doc is None else doc)
    ok = bool(ok) and found
    print(f"  {'OK ' if ok else 'BAD'} {name}" + ("" if found else "  <- phrase not in PAPER_v3"))
    if not ok:
        FAILS.append(name)


def pending(name: str, landed: bool, marker: str) -> None:
    """A result the text marks ⟦PENDING⟧. Listed when its input has landed and the text still waits for it."""
    if marker not in PAPER:
        print(f"  BAD {name}: the marker is gone from PAPER_v3; check the number instead")
        FAILS.append(f"{name}: marker gone")
    elif landed:
        print(f"  NEW {name}: landed, the text still marks it pending")
        LANDED.append(name)
    else:
        print(f"  ... {name}: pending")


def p_exact(x) -> float:
    return float(wilcoxon(np.asarray(x, float), method="exact").pvalue)


def hist(run: str) -> list[dict]:
    return json.load(open(HIST.format(run)))


def cc(run: str, cond: str = "full", key: str = "shape_bias") -> float:
    c = json.load(open(f"results/cue_conflict/{run}.json"))["conditions"][cond]
    return c["n_shape"] + c["n_texture"] if key == "decided" else c[key]


def shared(run: str, stream: str) -> float:
    return json.load(open(f"results/shape_texture/{run}.json"))["streams"][stream]["bias"]


def sweep(runs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(w grid, per-seed shape bias over w) of the frozen affine sweep, final checkpoints."""
    rows = [sorted(json.load(open(f"results/fusion_sweep/{r}_final_wsweep.json"))["sweep"], key=lambda q: q["w"]) for r in runs]
    return np.array([q["w"] for q in rows[0]]), np.array([[q["shape_bias"] for q in r] for r in rows])


def kappa(run: str) -> dict:
    return json.load(open(f"results/sweep_error_consistency/{run}_final_kappa_sweep.json"))


def kappa_at(run: str, w: float, pct: bool = True) -> float:
    K = kappa(run)
    k = [q for q in K["sweep"] if abs(q["w"] - w) < 1e-6][0]["kappa_mean"]
    return 100 * k / K["human_human_ceiling"] if pct else k


def check_accuracy() -> None:
    """Results 2.1, Discussion, Limitations: top-1 and training-time shape bias, from the training histories."""
    print("\n[2.1] accuracy and what the trajectory costs (training histories)")
    top = lambda runs: np.array([hist(r)[-1]["val_top1"] for r in runs])
    sb = lambda runs: np.array([hist(r)[-1]["cc"]["joint"]["sb"] for r in runs])
    m, c, f = top(MODEL), top(CONTROL), top(FIXED)
    gap = 100 * max(max(q["val_top1"] for q in hist(r)) - hist(r)[-1]["val_top1"] for r in MODEL)
    p = "the model reaches 0.683 ± 0.002 validation top-1 at its final checkpoint"
    said("model top-1, mean of 3 seeds", m.mean(), p, "0.683")
    said("model top-1, s.d. (ddof=1)", m.std(ddof=1), p, "0.002")
    holds(f"final within 0.9 points of the best epoch, every seed (max {gap:.2f})", gap <= 0.9,
          "within 0.9 points of its best epoch in every seed")
    p = "against 0.764 ± 0.003 for the control on the identical test input"
    said("control top-1", c.mean(), p, "0.764"); said("control top-1, s.d. (ddof=1)", c.std(ddof=1), p, "0.003")
    p = "The fixed BA diet reaches 0.665 ± 0.003 under the same recipe"
    said("fixed diet, model's recipe, top-1", f.mean(), p, "0.665")
    said("fixed diet, model's recipe, top-1 s.d.", f.std(ddof=1), p, "0.003")
    p = ("The trajectory therefore adds 1.8 points and 0.067 of shape bias (training-time evaluation) over the fixed diet "
         "under one recipe, and gives up 8.1 points")
    said("model minus fixed diet, top-1 points", 100 * (m.mean() - f.mean()), p, "1.8")
    said("model minus fixed diet, training-time shape bias", sb(MODEL).mean() - sb(FIXED).mean(), p, "0.067")
    said("control minus model, top-1 points", 100 * (c.mean() - m.mean()), p, "8.1")
    said("same, Discussion", 100 * (c.mean() - m.mean()), "The trajectory costs less: 8.1 points of top-1 against its control", "8.1")
    h = hist("fs2_dvd48_avg_w050_ls_seed0")[-1]
    said("48-month trajectory, top-1", h["val_top1"], "(0.664 top-1, shape bias 0.637)", "0.664")
    said("48-month trajectory, training-time shape bias", h["cc"]["joint"]["sb"], "(0.664 top-1, shape bias 0.637)", "0.637")
    p = "(0.665 top-1, shape bias 0.620, gradient −0.735)"
    said("fixed diet, top-1 (Limitations)", f.mean(), p, "0.665")
    said("fixed diet, training-time shape bias (Limitations)", sb(FIXED).mean(), p, "0.620")


def check_streams() -> None:
    """Results 2.2, Table 1, Discussion: the stream probes (exemplars held out: exemplar_streams.json, from
    tmp_analysis/figs_v2/exemplar_aggregate.py; shared: results/shape_texture/), cue-conflict behaviour, decided
    counts, the forced estimator, the training-time stream rates, and the carrier x ramp 2x2."""
    print("\n[2.2] the two streams (Table 1)")
    S = json.load(open(EXEMPLAR))
    EX = S["streams"]
    missing = [r for r in MODEL + CONTROL + SWAP + [UNTRAINED] if r not in EX]
    if missing:
        print(f"  BAD {EXEMPLAR} lacks {missing}: run slurm/exemplar_aggregate.sh")
        FAILS.append("exemplar_streams.json lacks the model")
        return
    q = lambda r, s, k="bias": EX[r][s]["Q1"][k]
    mean = lambda f, runs: float(np.mean([f(r) for r in runs]))
    hm, hp = np.array([q(r, "magno") for r in MODEL]), np.array([q(r, "parvo") for r in MODEL])
    sm, sp = mean(lambda r: shared(r, "magno"), MODEL), mean(lambda r: shared(r, "parvo"), MODEL)
    um, up = q(UNTRAINED, "magno"), q(UNTRAINED, "parvo")
    usm, usp = shared(UNTRAINED, "magno"), shared(UNTRAINED, "parvo")
    cm, cp, cf = (np.array([cc(r, k) for r in MODEL]) for k in ("magno_only", "parvo_only", "full"))
    xm, xp = mean(lambda r: q(r, "magno"), CONTROL), mean(lambda r: q(r, "parvo"), CONTROL)
    xsm, xsp = mean(lambda r: shared(r, "magno"), CONTROL), mean(lambda r: shared(r, "parvo"), CONTROL)
    xcm, xcp, xcf = (mean(lambda r: cc(r, k), CONTROL) for k in ("magno_only", "parvo_only", "full"))
    wm, wp = mean(lambda r: q(r, "magno"), SWAP), mean(lambda r: q(r, "parvo"), SWAP)
    wsm, wsp = mean(lambda r: shared(r, "magno"), SWAP), mean(lambda r: shared(r, "parvo"), SWAP)
    wcm, wcp, wcf = (mean(lambda r: cc(r, k), SWAP) for k in ("magno_only", "parvo_only", "full"))

    p = "shape − texture accuracy is +0.316 for the magno stream and −0.156 for the parvo stream"
    said("model magno probe, held out", hm.mean(), p, "+0.316")
    said("model parvo probe, held out", hp.mean(), p, "−0.156")
    holds("every seed on its side", (hm > 0).all() and (hp < 0).all(), "every seed on its side")
    holds("every seed's 5-95% interval over draws excludes zero",
          all(q(r, "magno", "bias_p05") > 0 and q(r, "parvo", "bias_p95") < 0 for r in MODEL),
          "every seed's 5–95% interval over draws excluding zero")
    said("model magno probe, shared", sm, "+0.578 and −0.454 with exemplars shared", "+0.578")
    said("model parvo probe, shared", sp, "+0.578 and −0.454 with exemplars shared", "−0.454")
    p = "the magno stream's cue-conflict shape bias is 0.850 and the parvo stream's 0.477, against 0.682 for the integrated readout"
    said("model magno cue-conflict", cm.mean(), p, "0.850")
    said("model parvo cue-conflict", cp.mean(), p, "0.477")
    said("model integrated cue-conflict", cf.mean(), p, "0.682")

    rows = {
        "| linear probe, shape − texture (0 = sign flip), 3 seeds: exemplars held out / shared | **+0.316** / +0.578 | **−0.156** / −0.454 | — |":
            (("magno held out", hm.mean(), "+0.316"), ("magno shared", sm, "+0.578"), ("parvo held out", hp.mean(), "−0.156"), ("parvo shared", sp, "−0.454")),
        "| cue-conflict shape bias (0.5 = parity), 3 seeds | **0.850** | **0.477** | **0.682** |":
            (("cue-conflict magno", cm.mean(), "0.850"), ("cue-conflict parvo", cp.mean(), "0.477"), ("cue-conflict integrated", cf.mean(), "0.682")),
        "| linear probe, untrained | +0.087 / +0.429 | −0.033 / −0.087 | — |":
            (("untrained magno held out", um, "+0.087"), ("untrained magno shared", usm, "+0.429"), ("untrained parvo held out", up, "−0.033"), ("untrained parvo shared", usp, "−0.087")),
        "| *contribution of learning on the maturing input* (trained minus untrained) | **+0.229** / +0.149 | **−0.123** / −0.367 | — |":
            (("learning, magno held out", hm.mean() - um, "+0.229"), ("learning, magno shared", sm - usm, "+0.149"),
             ("learning, parvo held out", hp.mean() - up, "−0.123"), ("learning, parvo shared", sp - usp, "−0.367")),
        "| control, adult input from the first step, 3 seeds: linear probe | **−0.177** / −0.291 | −0.069 / −0.413 | — |":
            (("control magno held out", xm, "−0.177"), ("control magno shared", xsm, "−0.291"), ("control parvo held out", xp, "−0.069"), ("control parvo shared", xsp, "−0.413")),
        "| control: cue-conflict shape bias | 0.384 | 0.426 | 0.374 |":
            (("control cue-conflict magno", xcm, "0.384"), ("control cue-conflict parvo", xcp, "0.426"), ("control cue-conflict integrated", xcf, "0.374")),
        "| channel swap, ramp on L−M, 3 seeds: linear probe | −0.159 / −0.276 | −0.016 / −0.276 | — |":
            (("swap magno held out", wm, "−0.159"), ("swap magno shared", wsm, "−0.276"), ("swap parvo held out", wp, "−0.016"), ("swap parvo shared", wsp, "−0.276")),
        "| channel swap: cue-conflict shape bias | 0.397 | 0.390 | 0.389 |":
            (("swap cue-conflict magno", wcm, "0.397"), ("swap cue-conflict parvo", wcp, "0.390"), ("swap cue-conflict integrated", wcf, "0.389")),
    }
    for row, cells in rows.items():
        for lab, v, lit in cells:
            said(f"Table 1: {lab}", v, row, lit)

    said("untrained magno, held out", um, "(+0.087 against +0.090 on held-out exemplars)", "+0.087")
    said("untrained no-diet luminance, held out", q("untrained_noblur", "magno"), "(+0.087 against +0.090 on held-out exemplars)", "+0.090")
    said("untrained parvo, held out", up, "and the parvo stream begins nearly unbiased (−0.033)")
    said("learning on the maturing input, magno", hm.mean() - um, "Training on the maturing input adds +0.229 to the magno stream")
    said("learning on the maturing input, parvo", hp.mean() - up, "— and −0.123 to the parvo stream")
    p = "across the sign flip to −0.177, below its own parvo stream (−0.069), and the integrated readout to 0.374"
    said("control magno, held out", xm, p, "−0.177"); said("control parvo, held out", xp, p, "−0.069")
    said("control integrated cue-conflict", xcf, p, "0.374")
    p = "The same network holds components at +0.32 and −0.16 on held-out exemplars and behaves at 0.682"
    said("Discussion: magno held out", hm.mean(), p, "+0.32"); said("Discussion: parvo held out", hp.mean(), p, "−0.16")
    said("Discussion: integrated", cf.mean(), p, "0.682")

    p = "decodes texture at 0.148 (chance 0.0625; 0.078 for the BA network's blurred stream) and shape at 0.464"
    said("model magno texture accuracy, held out", mean(lambda r: q(r, "magno", "texture_acc"), MODEL), p, "0.148")
    said("BA magno texture accuracy, held out", mean(lambda r: q(r, "magno", "texture_acc"), [f"dkl_bcs_seed{s}_trained" for s in range(3)]), p, "0.078")
    said("model magno shape accuracy, held out", mean(lambda r: q(r, "magno", "shape_acc"), MODEL), p, "0.464")

    conds = ("magno_only", "parvo_only", "full")
    # decided images per readout, seed means
    dm = [np.mean([cc(r, k, "decided") for r in MODEL]) for k in conds]; dc = [np.mean([cc(r, k, "decided") for r in CONTROL]) for k in conds]
    p = "(426 to 580 of 1,200 in the model, 209 to 673 in the control; seed means)"
    said("decided images, model, fewest (readout, seed mean)", min(dm), p, "426"); said("decided images, model, most", max(dm), p, "580")
    said("decided images, control, fewest", min(dc), p, "209"); said("decided images, control, most", max(dc), p, "673")
    pat_m = "results/cue_conflict/fs2_dvd300_avg_w050_ls_seed[012]_logits.npz"
    pat_c = f"results/cue_conflict/fs2_dvdfix300_avg_w050_ls_seed[{''.join(map(str, CONTROL_SEEDS))}]_logits.npz"
    fc = {(arm, k): ba._logit_estimators(pat, k)[0].mean() for arm, pat in (("model", pat_m), ("control", pat_c)) for k in conds}
    p = "(forced estimator: magno 0.730, integrated 0.648, parvo 0.520; control 0.393, 0.383, 0.471)"
    for (arm, k), lit in zip([("model", "magno_only"), ("model", "full"), ("model", "parvo_only"),
                              ("control", "magno_only"), ("control", "full"), ("control", "parvo_only")],
                             ["0.730", "0.648", "0.520", "0.393", "0.383", "0.471"]):
        said(f"forced estimator, {arm} {k}", fc[(arm, k)], p, lit)
    a, b = cp.mean() - 0.5, fc[("model", "parvo_only")] - 0.5
    holds(f"parvo within 0.03 of parity on both estimators, opposite sides ({a:+.3f}, {b:+.3f})",
          max(abs(a), abs(b)) <= 0.03 and a * b < 0,
          "but the parvo stream sits within 0.03 of parity on both estimators and on opposite sides of it")

    # training-time evaluation, magno stream through the shared head, every epoch of every seed
    HM, HC = [hist(r) for r in MODEL], [hist(r) for r in CONTROL]
    R = HM[0]
    ms = [e["cc"]["tied_magno"]["sb"] for h in HM for e in h]
    xs = [e["cc"]["tied_magno"]["sb"] for h in HC for e in h]
    p = "(shape bias 0.63–0.92 against 0.32–0.495 over three seeds each; training-time evaluation)"
    said("model magno, training-time, lowest epoch", min(ms), p, "0.63"); said("model magno, training-time, highest epoch", max(ms), p, "0.92")
    said("control magno, training-time, lowest epoch", min(xs), p, "0.32"); said("control magno, training-time, highest epoch", max(xs), p, "0.495")
    holds(f"the two magno streams on opposite sides of parity at every epoch of every seed", min(ms) > 0.5 > max(xs)
          and all(len(h) == 50 for h in HM + HC), "The two magno streams sit on opposite sides of parity at every epoch evaluated during training")
    p = "so the bias is present from the first evaluation, after one epoch at ages up to 6 months (top-1 0.09)"
    said("diet age at the end of the first epoch (months)", R[0]["dvd_age"], p, "6")
    said("model top-1 after the first epoch", R[0]["val_top1"], p, "0.09")
    holds(f"model magno above parity after the first epoch ({ms[0]:.3f})", ms[0] > 0.5, p)
    m_top = float(np.mean([hist(r)[-1]["val_top1"] for r in MODEL]))
    hit = [min(h, key=lambda e: abs(e["val_top1"] - m_top)) for h in HC]
    p = ("(epoch 7 of 50 in each of the three seeds), its magno stream's shape bias is 0.41–0.43, and at no epoch of any "
         "seed does it reach parity (highest 0.495; training-time evaluation)")
    # history epochs are 0-based; the text counts from 1
    holds(f"control epoch whose top-1 matches the model's: {[e['epoch'] + 1 for e in hit]}", all(e["epoch"] + 1 == 7 for e in hit), p)
    sbm = [e["cc"]["tied_magno"]["sb"] for e in hit]
    said("control magno shape bias at that epoch, lowest seed", min(sbm), p, "0.41")
    said("control magno shape bias at that epoch, highest seed", max(sbm), p, "0.43")
    said("control magno shape bias, highest epoch of any seed", max(xs), p, "0.495")
    holds("control magno below parity at every epoch of every seed", max(xs) < 0.5, p)

    t = S.get("two_by_two_v3", {}).get("Q1", {})
    if t:
        mm = t["means"]
        p = ("They are achromatic with the ramp (+0.316), achromatic with the adult input from the start (−0.177), chromatic "
             "without the ramp (−0.156 in the model, −0.069 in the control) and chromatic with the ramp: the channel swap, "
             "which applies the same ramp to L−M with luminance untransformed (three seeds; −0.016)")
        said("2x2 cell: achromatic, ramp", mm["achromatic_transform"], p, "+0.316")
        said("2x2 cell: achromatic, adult input", mm["achromatic_none"], p, "−0.177")
        said("2x2 cell: chromatic, no ramp (control)", mm["chromatic_none"], p, "−0.069")
        said("2x2 cell: chromatic, ramp (swap)", mm["chromatic_transform"], p, "−0.016")
        cl, q0 = t["cells"], S["two_by_two_v3"]["Q0"]
        others = [v for k, vs in cl.items() if k != "achromatic_transform" for v in vs]
        p = ("The ramp moves the achromatic stream by +0.49 and the chromatic stream by +0.05, which leaves it below parity; "
             "the carrier × ramp interaction is +0.44 on held-out exemplars (+0.73 with exemplars shared), and no seed of the "
             "conjunction cell (+0.30 to +0.33) overlaps a seed of any other cell (at most −0.011)")
        said("2x2: ramp effect, achromatic", mm["achromatic_transform"] - mm["achromatic_none"], p, "+0.49")
        said("2x2: ramp effect, chromatic", mm["chromatic_transform"] - mm["chromatic_none"], p, "+0.05")
        holds("2x2: chromatic ramp cell below parity in every seed", max(cl["chromatic_transform"]) < 0, p)
        said("2x2: interaction, held out", t["interaction"], p, "+0.44"); said("2x2: interaction, shared", q0["interaction"], p, "+0.73")
        said("2x2: conjunction cell, lowest seed", min(cl["achromatic_transform"]), p, "+0.30")
        said("2x2: conjunction cell, highest seed", max(cl["achromatic_transform"]), p, "+0.33")
        said("2x2: other cells, highest seed", max(others), p, "−0.011")
        p = "The swap reproduces its control on every other measure as well: 0.760 top-1, an integrated cue-conflict rate of 0.389"
        said("swap top-1", np.mean([hist(r)[-1]["val_top1"] for r in SWAP]), p, "0.760"); said("swap integrated cue-conflict", wcf, p, "0.389")
        holds("swap: both streams below parity on the probe and on cue-conflict, every seed",
              all(q(r, s) < 0 and shared(r, s) < 0 and cc(r, k) < 0.5 for r in SWAP for s, k in (("magno", "magno_only"), ("parvo", "parvo_only"))),
              "and both streams below parity (Table 1)")
    else:
        print("  BAD exemplar_streams.json has no two_by_two_v3: run slurm/exemplar_aggregate.sh")
        FAILS.append("exemplar_streams.json lacks two_by_two_v3")


def check_cortex() -> None:
    """Results 2.3, Extended Data Table 1 (normalised), Discussion, Limitations, from the committed cortex summary
    (tmp_analysis/fs2_cortex_summary.py): profiles, gradient, rho, the cue-conflict prediction, atlas groups,
    unique variance beyond V3, noise-ceiling-normalised r²."""
    print("\n[2.3] cortex (fs2_cortex_summary.json)")
    C = json.load(open(CORTEX))
    A, P = C["arms"], C["pairs"]
    m, c, b, nb, fx = A["dvd300"], A["control"], A["BA diet"], A["BA no diet"], A["A"]
    pr = m["profile"]
    p = "In the model the index runs from +0.55 in V1v, magno-explained, through zero at V2v to −0.76 in hV4"
    said("model profile, V1v", pr[0], p, "+0.55"); said("model profile, hV4", pr[6], p, "−0.76")
    holds(f"model profile crosses zero at V2v (V1d {pr[1]:+.3f}, V2v {pr[2]:+.3f}, V2d {pr[3]:+.3f})", abs(pr[2]) < 0.05, p)
    said("control profile, V1v", c["profile"][0], "(+0.81 in V1v, +0.73 in hV4)", "+0.81")
    said("control profile, hV4", c["profile"][6], "(+0.81 in V1v, +0.73 in hV4)", "+0.73")
    holds("control magno-explained in every ROI", min(c["profile"]) > 0, "is magno-explained everywhere and nearly flat")
    sw = A["swap"]
    said("swap profile, V1v", sw["profile"][0], "as are the channel swap (+0.85, +0.83)", "+0.85")
    said("swap profile, hV4", sw["profile"][6], "as are the channel swap (+0.85, +0.83)", "+0.83")
    holds("swap magno-explained in every ROI", min(sw["profile"]) > 0, "as are the channel swap (+0.85, +0.83)")
    said("BA no-diet profile, V1v", nb["profile"][0], "and the BA no-diet network (+0.91, +0.83)", "+0.91")
    said("BA no-diet profile, hV4", nb["profile"][6], "and the BA no-diet network (+0.91, +0.83)", "+0.83")
    said("BA profile, V1v", b["profile"][0], "descends without crossing (−0.00 to −0.84)", "−0.00")
    said("BA profile, hV4", b["profile"][6], "descends without crossing (−0.00 to −0.84)", "−0.84")
    holds("BA profile never above zero", max(b["profile"]) <= 0.005, "descends without crossing (−0.00 to −0.84)")

    g, gb = np.array(m["grad_per_seed"]), np.array(b["grad_per_seed"])
    p = "the gradient is −1.259 to −1.291 across the three seeds against −0.062 to −0.086 for the control's"
    said("model gradient, shallowest seed", g.max(), p, "−1.259"); said("model gradient, steepest seed", g.min(), p, "−1.291")
    said("control gradient, shallowest seed", max(c["grad_per_seed"]), p, "−0.062")
    said("control gradient, steepest seed", min(c["grad_per_seed"]), p, "−0.086")
    holds(f"control: {len(c['grad_per_seed'])} seeds", len(c["grad_per_seed"]) == 3, p)
    d = np.array(P["dvd300 - control"]["grad"])
    holds(f"model below control (seed means), {(d < 0).sum()}/{len(d)} subjects, p {p_exact(d):.4f}",
          (d < 0).all() and len(d) == 8 and p_exact(d) < 0.0079,
          "each seed against the control seed of the same index at the exact Wilcoxon floor for eight subjects (p = .0078, 8/8)")
    p = "It is steeper than the BA network's −0.745 to −0.833 in 8/8 subjects (mean Δ −0.48, p = .0078)"
    said("BA gradient, shallowest seed", gb.max(), p, "−0.745"); said("BA gradient, steepest seed", gb.min(), p, "−0.833")
    d = np.array(P["dvd300 - BA diet"]["grad"])
    said("model minus BA, mean over subjects", d.mean(), p, "−0.48")
    holds(f"model steeper than BA, {(d < 0).sum()}/{len(d)}, p {p_exact(d):.4f}", (d < 0).all() and len(d) == 8 and p_exact(d) < 0.0079, p)
    rho = np.array(m["rho_per_seed"])
    p = "is −0.964 to −0.973 in every seed, against −0.29 to −0.41 for the control"
    said("model rho, weakest seed", rho.max(), p, "−0.964"); said("model rho, strongest seed", rho.min(), p, "−0.973")
    said("control rho, weakest seed", max(c["rho_per_seed"]), p, "−0.29"); said("control rho, strongest seed", min(c["rho_per_seed"]), p, "−0.41")
    p = "has no gradient either (−0.019 normalised, +0.002 raw, seed means; ρ = −0.22), and the model's lies below it in 8/8 subjects (p = .0078)"
    said("swap gradient, seed mean", np.mean(sw["grad_per_seed"]), p, "−0.019"); said("swap rho, seed mean", np.mean(sw["rho_per_seed"]), p, "−0.22")
    d = np.array(P["dvd300 - swap"]["grad"])
    holds(f"model below the swap, {(d < 0).sum()}/{len(d)} subjects, p {p_exact(d):.4f}", (d < 0).all() and len(d) == 8 and p_exact(d) < 0.0079, p)
    said("model gradient, mean of seeds", g.mean(), "steepens it on the normalised index (to −1.27)")
    said("control gradient (Results 2.3)", np.mean(c["grad_per_seed"]), "erases it on both (−0.070; −0.004, seed means)", "−0.070")
    ed1 = ["| normalised | 0 | −1.259 | −0.062 | −0.833 | 8/8 | 0.0078 |", "|  | 1 | −1.291 | −0.062 | −0.785 | 8/8 | 0.0078 |",
           "|  | 2 | −1.262 | −0.086 | −0.745 | 8/8 | 0.0078 |"]
    p = ("The channel swap (the ramp on L−M, luminance untransformed) gives −0.015, −0.016 and −0.025 normalised (seeds 0–2)")
    for s, lit in enumerate(("−0.015", "−0.016", "−0.025")):
        said(f"ED T1 caption: swap normalised, seed {s}", sw["grad_per_seed"][s], p, lit)
    for s, row in enumerate(ed1):
        cells = [x.strip() for x in row.strip("|").split("|")]
        said(f"ED T1 normalised, seed {s}: model", g[s], row, cells[2])
        said(f"ED T1 normalised, seed {s}: BA", gb[s], row, cells[4])
        if s in CORTEX_SEEDS:
            said(f"ED T1 normalised, seed {s}: control", c["grad_per_seed"][CORTEX_SEEDS.index(s)], row, cells[3])

    v, vc, vb = np.array(m["cc_per_seed"]), np.mean(c["cc_per_seed"]), np.array(b["cc_per_seed"])
    p = ("give a predicted magno − parvo shape-bias difference of +1.32 to +1.34 in the model against +0.07 for the control and "
         "−0.01 for the channel swap (8/8 subjects, seed means")
    said("cue-conflict prediction, model, lowest seed", v.min(), p, "+1.32"); said("cue-conflict prediction, model, highest seed", v.max(), p, "+1.34")
    said("cue-conflict prediction, control", vc, p, "+0.07"); said("cue-conflict prediction, swap", np.mean(sw["cc_per_seed"]), p, "−0.01")
    d = np.array(P["dvd300 - swap"]["cc"])
    holds(f"cue-conflict prediction above the swap in {(d > 0).sum()}/{len(d)} subjects", (d > 0).all() and len(d) == 8, p)
    d = np.array(P["dvd300 - control"]["cc"])
    holds(f"cue-conflict prediction above the control in {(d > 0).sum()}/{len(d)} subjects", (d > 0).all() and len(d) == 8, p)
    p = "It is smaller than the BA network's +1.53 to +1.57 (8/8)"
    said("cue-conflict prediction, BA, lowest seed", vb.min(), p, "+1.53"); said("cue-conflict prediction, BA, highest seed", vb.max(), p, "+1.57")
    d = np.array(P["dvd300 - BA diet"]["cc"])
    holds(f"cue-conflict prediction below the BA network's in {(d < 0).sum()}/{len(d)} subjects", (d < 0).all() and len(d) == 8, p)

    at = {k.split("(")[0].split(" ")[0]: np.array(x) for k, x in m["atlas"].items()}
    p = ("relative to V1, hV4 sits at −1.12 and every extrastriate group lower — ventral (VO, PHC) −1.25, lateral (LO, TO) −1.27, "
         "dorsal (V3A/B, IPS0–2) −1.22 — each in 8/8 subjects (p = .0078)")
    for k, lit in (("hV4", "−1.12"), ("ventral", "−1.25"), ("lateral", "−1.27"), ("dorsal", "−1.22")):
        said(f"atlas, {k} minus V1", at[k].mean(), p, lit)
        holds(f"atlas, {k} below V1 in {(at[k] < 0).sum()}/8, p {p_exact(at[k]):.4f}", (at[k] < 0).all() and p_exact(at[k]) < 0.0079, p)
    holds("every extrastriate group below hV4", all(at[k].mean() < at["hV4"].mean() for k in ("ventral", "lateral", "dorsal")), p)
    dv, dl = at["ventral"] - at["hV4"], at["lateral"] - at["hV4"]
    p = "The ventral and lateral groups sit a further 0.14–0.15 below hV4 (p = .039 and .016, uncorrected)"
    said("ventral below hV4", -dv.mean(), p, "0.14"); said("lateral below hV4", -dl.mean(), p, "0.15")
    said("ventral minus hV4, p", p_exact(dv), p, ".039"); said("lateral minus hV4, p", p_exact(dl), p, ".016")
    ps = [p_exact(at[x] - at[y]) for x, y in combinations(("ventral", "lateral", "dorsal"), 2)]
    holds(f"no extrastriate contrast departs from zero (min p {min(ps):.3f})", min(ps) >= 0.075,
          "no contrast among the three extrastriate groups departs from zero (all p ≥ .08)")
    p = "the magno band contributes about 0.013 unique variance against 0.149 for the parvo band"
    said("U_m beyond V3", m["um_ext"], p, "0.013"); said("U_p beyond V3", m["up_ext"], p, "0.149")

    rm, rc, rb = np.mean(m["r2nc_per_seed"]), np.mean(c["r2nc_per_seed"]), np.mean(b["r2nc_per_seed"])
    p = "(noise-ceiling-normalised median r² 0.471 against 0.494, 8/8 subjects)"
    said("r²/NC, model", rm, p, "0.471"); said("r²/NC, control", rc, p, "0.494")
    d = np.array(P["dvd300 - control"]["r2nc"])
    holds(f"r²/NC below the control in {(d < 0).sum()}/{len(d)} subjects", (d < 0).all() and len(d) == 8, p)
    p = "although the model predicts BOLD better than the BA network does (0.395, 8/8; Fig. 3e)"
    said("r²/NC, BA", rb, p, "0.395")
    d = np.array(P["dvd300 - BA diet"]["r2nc"])
    holds(f"r²/NC above the BA network in {(d > 0).sum()}/{len(d)} subjects", (d > 0).all() and len(d) == 8, p)
    said("control minus model, r²/NC", rc - rm, "and 0.024 of noise-ceiling-normalised r²")
    said("fixed diet, model's recipe, gradient", np.mean(fx["grad_per_seed"]), "(0.665 top-1, shape bias 0.620, gradient −0.735)", "−0.735")


NUISANCE = "results/figures_v3/cortex_controls.json"


def check_nuisance() -> None:
    """Results 2.3, Supplementary Note S3 and Extended Data Fig. 3: the gradient within eccentricity bands and pRF-size
    tertiles, and the partial slopes on continuous pRF eccentricity and size (tmp_analysis/fs2_cortex_controls.py)."""
    print(f"\n[2.3, S3, ED Fig. 3] nuisance axes ({NUISANCE})")
    J = json.load(open(NUISANCE))
    T, PS = J["tests"], J["per_subject"]
    band = lambda arm, stat="norm": np.array(PS[arm][stat]["ecc_bands"], float)
    tert = lambda arm, stat="norm": np.array(PS[arm][stat]["size_tertiles"], float)
    fin = lambda v: v[np.isfinite(v)]
    # the seed-mean gradients agree with the cortex summary behind ED Table 1
    p = "| control alone (seed mean), normalised / raw | — | | −0.070 / −0.004 |"
    said("control gradient (nuisance file), normalised", T["norm"]["control"]["gradient"]["mean"], p, "−0.070")
    said("control gradient (nuisance file), raw", T["raw"]["control"]["gradient"]["mean"], p, "−0.004")

    p = ("Among voxels of matched eccentricity it is negative in each of NSD's five eccentricity bands (−0.58 in the foveal "
         "band to −1.33 beyond 4°), and among voxels of matched pRF size in each size tertile (−1.26 to −1.32), below the "
         "control's in every band and tertile (7/7 or 8/8 subjects; 6/6 in the smallest tertile), while the control and "
         "the channel swap stay near zero in each")
    m = [T["norm"]["model"]["ecc_bands"][b]["mean"] for b in range(5)]
    said("model, foveal band", m[0], p, "−0.58"); said("model, beyond 4°", m[4], p, "−1.33")
    holds("model's band gradient grows with eccentricity", all(np.diff(m) < 0), p)
    mt = [T["norm"]["model"]["size_tertiles"][b]["mean"] for b in range(3)]
    said("model, size tertiles, least negative", max(mt), p, "−1.26"); said("model, size tertiles, most negative", min(mt), p, "−1.32")
    for stat in ("norm", "raw"):
        mc = T[stat]["model_minus_control"]
        cells = [(f"band {b}", w) for b, w in enumerate(mc["ecc_bands"])] + [(f"tertile {b}", w) for b, w in enumerate(mc["size_tertiles"])]
        want = [(7, 7), (8, 8), (8, 8), (8, 8), (8, 8), (6, 6), (8, 8), (8, 8)]
        got = [(w["n_neg"], w["n"]) for _, w in cells]
        holds(f"model below control per band/tertile, {stat}: {got}", got == want, p)
    for b, lit in ((0, ".016"), (1, ".0078")):
        said(f"model - control, band {b}, p", T["norm"]["model_minus_control"]["ecc_bands"][b]["p"],
             "the model lies below the control in every band (7/7, p = .016, in the foveal band; 8/8, p = .0078, in the other four)", lit)
    said("model - control, small tertile, p", T["norm"]["model_minus_control"]["size_tertiles"][0]["p"],
         "(6/6, p = .031, in the smallest tertile, where subjects 03 and 08 have too few voxels in the model; 8/8, p = .0078, in the other two)", ".031")
    holds("same counts on the raw statistic",
          [(w["n_neg"], w["n"]) for w in T["raw"]["model_minus_control"]["ecc_bands"] + T["raw"]["model_minus_control"]["size_tertiles"]]
          == [(w["n_neg"], w["n"]) for w in T["norm"]["model_minus_control"]["ecc_bands"] + T["norm"]["model_minus_control"]["size_tertiles"]],
          "On the raw statistic the model-minus-control contrasts have the same counts in every band and tertile.")

    p = ("The control stays between −0.07 and +0.16 in every band and the channel swap between −0.06 and +0.05, and the "
         "model lies below the control in every band")
    cb, sb = [w["mean"] for w in T["norm"]["control"]["ecc_bands"]], [w["mean"] for w in T["norm"]["channel swap"]["ecc_bands"]]
    said("control, band min", min(cb), p, "−0.07"); said("control, band max", max(cb), p, "+0.16")
    said("swap, band min", min(sb), p, "−0.06"); said("swap, band max", max(sb), p, "+0.05")
    p = ("(6/7 subjects in the foveal band, where subject 02 has too few hV4 voxels in every arm and subject 05 is positive; "
         "8/8 in the other four)")
    holds("subject 02 missing from the foveal band in every arm", all(np.isnan(band(a)[1, 0]) and np.isfinite(band(a)[[0, 2, 3, 4, 5, 6, 7], 0]).all()
                                                                       for a in ("model", "control", "channel swap", "untrained")), p)
    holds("model foveal band: subject 05 the only positive", list(np.where(band("model")[:, 0] > 0)[0] + 1) == [5], p)
    holds("model negative in 8/8 beyond the foveal band", (band("model")[:, 1:] < 0).all(), p)
    p = "Within pRF-size tertiles the model sits at −1.28, −1.26 and −1.32 against −0.13, −0.13 and −0.09 for the control"
    for b, (lm, lc) in enumerate((("−1.28", "−0.13"), ("−1.26", "−0.13"), ("−1.32", "−0.09"))):
        said(f"model, tertile {b}", T["norm"]["model"]["size_tertiles"][b]["mean"], p, lm)
        said(f"control, tertile {b}", T["norm"]["control"]["size_tertiles"][b]["mean"], p, lc)
    holds("subjects 03 and 08 missing from the model's small tertile",
          list(np.where(~np.isfinite(tert("model")[:, 0]))[0] + 1) == [3, 8], p)

    # continuous predictors
    def part(arm, stat, term):
        return T[stat][arm][term]
    p = "mean VIF 1.64"
    said("mean VIF, model seed 0", J["vif_mean_model_seed0"], p)
    p = ("the model's stream preference moves toward parvo with increasing pRF size at fixed eccentricity (partial slope "
         "−0.118, 8/8 subjects, p = .0078) and toward magno with increasing eccentricity at fixed size (+0.163, 8/8, p = .0078), "
         "on the raw statistic too (−0.391 and +0.444, 8/8 each)")
    for stat, term, lit, side in (("norm", "size_partial", "−0.118", "n_neg"), ("norm", "ecc_partial", "+0.163", "n_pos"),
                                  ("raw", "size_partial", "−0.391", "n_neg"), ("raw", "ecc_partial", "+0.444", "n_pos")):
        w = part("model", stat, term)
        said(f"model {term} {stat}", w["mean"], p, lit)
        holds(f"model {term} {stat}: {w[side]}/8, p {w['p']:.4f}", w[side] == 8 and w["p"] < 0.0079, p)
    p = ("The control shows no eccentricity effect (−0.030, 6/8 negative, p = .25) and a size effect of the opposite sign "
         "(+0.047, 8/8 positive, p = .0078), and so does the channel swap (−0.025, p = .31; +0.048, 7/8 positive, p = .016)")
    w = part("control", "norm", "ecc_partial"); said("control ecc|size", w["mean"], p, "−0.030"); said("control ecc|size p", w["p"], p, ".25")
    holds(f"control ecc|size {w['n_neg']}/8 negative", w["n_neg"] == 6, p)
    w = part("control", "norm", "size_partial"); said("control size|ecc", w["mean"], p, "+0.047")
    holds(f"control size|ecc {w['n_pos']}/8 positive, p {w['p']:.4f}", w["n_pos"] == 8 and w["p"] < 0.0079, p)
    w = part("channel swap", "norm", "ecc_partial"); said("swap ecc|size", w["mean"], p, "−0.025"); said("swap ecc|size p", w["p"], p, ".31")
    w = part("channel swap", "norm", "size_partial"); said("swap size|ecc", w["mean"], p, "+0.048"); said("swap size|ecc p", w["p"], p, ".016")
    holds(f"swap size|ecc {w['n_pos']}/8 positive", w["n_pos"] == 7, p)
    p = "the model minus the control is +0.193 on the eccentricity partial and −0.165 on the size partial (8/8 each, p = .0078)"
    for term, lit, side in (("ecc_partial", "+0.193", "n_pos"), ("size_partial", "−0.165", "n_neg")):
        w = T["norm"]["model_minus_control"][term]
        said(f"model - control {term}", w["mean"], p, lit)
        holds(f"model - control {term}: {w[side]}/8, p {w['p']:.4f}", w[side] == 8 and w["p"] < 0.0079, p)
    p = "The untrained network on the shared front end already carries the dissociation (+0.083, 7/8, p = .023; −0.040, 8/8, p = .0078)"
    w = part("untrained", "norm", "ecc_partial"); said("untrained ecc|size", w["mean"], p, "+0.083"); said("untrained ecc|size p", w["p"], p, ".023")
    holds(f"untrained ecc|size {w['n_pos']}/8 positive", w["n_pos"] == 7, p)
    w = part("untrained", "norm", "size_partial"); said("untrained size|ecc", w["mean"], p, "−0.040")
    holds(f"untrained size|ecc {w['n_neg']}/8, p {w['p']:.4f}", w["n_neg"] == 8 and w["p"] < 0.0079, p)
    p = ("Training on the maturing input keeps it and strengthens its size term on both statistics (model minus untrained "
         "−0.078 normalised, −0.102 raw, 8/8 each, p = .0078), while its eccentricity term grows on the normalised statistic "
         "only (+0.080, 7/8, p = .023; raw −0.027, 4/8, p = .64)")
    for stat, lit in (("norm", "−0.078"), ("raw", "−0.102")):
        w = T[stat]["model_minus_untrained"]["size_partial"]
        said(f"model - untrained size|ecc {stat}", w["mean"], p, lit)
        holds(f"model - untrained size|ecc {stat}: {w['n_neg']}/8, p {w['p']:.4f}", w["n_neg"] == 8 and w["p"] < 0.0079, p)
    w = T["norm"]["model_minus_untrained"]["ecc_partial"]; said("model - untrained ecc|size norm", w["mean"], p, "+0.080")
    said("model - untrained ecc|size norm p", w["p"], p, ".023"); holds(f"... {w['n_pos']}/8 positive", w["n_pos"] == 7, p)
    w = T["raw"]["model_minus_untrained"]["ecc_partial"]; said("model - untrained ecc|size raw", w["mean"], p, "−0.027")
    said("model - untrained ecc|size raw p", w["p"], p, ".64"); holds(f"... {w['n_neg']}/8 negative", w["n_neg"] == 4, p)
    p = "removes the eccentricity term and reverses the size term (raw +0.265 and +0.270, 8/8 each)"
    for arm, lit in (("control", "+0.265"), ("channel swap", "+0.270")):
        w = part(arm, "raw", "size_partial"); said(f"{arm} size|ecc raw", w["mean"], p, lit)
        holds(f"{arm} size|ecc raw {w['n_pos']}/8 positive", w["n_pos"] == 8, p)


VPREC = "results/vp_recovery_realnc_v3"


def check_estimator_bias() -> None:
    """Results 2.3 and Supplementary Note S3: the zero-truth floor of vp_recovery_realnc.py on seed 0 of the model and
    the control, recomputed from the committed simulation files with the BA block's own statistic (ba.stats_from);
    the real seed-0 contrasts come from the nuisance file (recomputed from the per-voxel fits in check_nuisance_recompute)."""
    print(f"\n[2.3, S3] estimator bias on the model ({VPREC})")
    J = json.load(open(NUISANCE))["estimator_bias"]
    runs = {"model": MODEL[0], "control": CORTEX_CONTROL[0]}
    fl, flr, fid, rho = {}, {}, {}, {}
    for arm, run in runs.items():
        files = sorted(glob.glob(f"{VPREC}/vp_recovery_{run}_subj*.json"))
        assert len(files) == 8, (arm, len(files))
        D = [json.load(open(f)) for f in files]
        assert [d["subject"] for d in D] == list(range(1, 9))
        B = [ba.stats_from(d["conditions"]["B_manufacture"]["rows"], "est_") for d in D]
        fl[arm], flr[arm] = np.array([b[0] for b in B]), np.array([b[1] for b in B])
        fid[arm] = np.mean([ba.stats_from(d["conditions"]["A_recovery"]["rows"], "est_")[0]
                            / ba.stats_from(d["conditions"]["A_recovery"]["rows"], "true_")[0] for d in D])
        rho[arm] = np.array([d["conditions"]["A_recovery"]["rho_est_vs_rank"] for d in D])
        holds(f"{arm} floor matches the nuisance file", np.allclose(fl[arm], J[arm]["floor_norm"]["per_subject"], atol=1e-12),
              "Simulated with the model's fitted feature geometry")
    real = {a: np.array(J[a]["real_norm_seed0"]) for a in runs}
    disp = {a: real[a] - fl[a] for a in runs}

    p = ("the recovered contrast is small and of the wrong sign (+0.018, p = .11; +0.066 on the control's geometry). Corrected "
         "by each arm's own floor, the model's gradient is −1.277 against −0.127 for the control, both non-zero in 8/8 "
         "subjects, and the model's is the steeper in every subject (p = .0078; Supplementary Note S3)")
    said("floor, model geometry", fl["model"].mean(), p, "+0.018"); said("floor, model geometry, p", p_exact(fl["model"]), p, ".11")
    said("floor, control geometry", fl["control"].mean(), p, "+0.066")
    said("displacement, model", disp["model"].mean(), p, "−1.277"); said("displacement, control", disp["control"].mean(), p, "−0.127")
    for a in runs:
        holds(f"displacement, {a}: {(disp[a] < 0).sum()}/8 negative, p {p_exact(disp[a]):.4f}",
              (disp[a] < 0).all() and p_exact(disp[a]) < 0.0079, p)
    dd = disp["model"] - disp["control"]
    holds(f"model's displacement below the control's in {(dd < 0).sum()}/8, p {p_exact(dd):.4f}", (dd < 0).all() and p_exact(dd) < 0.0079, p)

    p = "On the model's geometry the recovered contrast is +0.018 (5/8 subjects positive, p = .11) and on the raw statistic −0.001 (p = .31)"
    holds(f"model floor positive in {(fl['model'] > 0).sum()}/8", (fl["model"] > 0).sum() == 5, p)
    said("raw floor, model", flr["model"].mean(), p, "−0.001"); said("raw floor, model, p", p_exact(flr["model"]), p, ".31")
    p = "On the control's geometry it is +0.066 (8/8, p = .0078), and +0.004 on the raw statistic (8/8, p = .0078)"
    holds(f"control floor positive in {(fl['control'] > 0).sum()}/8, p {p_exact(fl['control']):.4f}",
          (fl["control"] > 0).all() and p_exact(fl["control"]) < 0.0079, p)
    said("raw floor, control", flr["control"].mean(), p, "+0.004")
    holds(f"control raw floor positive in {(flr['control'] > 0).sum()}/8, p {p_exact(flr['control']):.4f}",
          (flr["control"] > 0).all() and p_exact(flr["control"]) < 0.0079, p)
    p = "the two floors differ (Mann–Whitney p = .0003)"
    said("floors differ between geometries", mannwhitneyu(fl["model"], fl["control"], method="exact").pvalue, p, ".0003")
    p = ("The real seed-0 contrast minus each subject's own floor is −1.277 for the model and −0.127 for the control, each "
         "non-zero in 8/8 subjects (p = .0078). The model's displacement is 10.1 times the control's and below it in every "
         "subject (−1.150, p = .0078)")
    said("displacement ratio", disp["model"].mean() / disp["control"].mean(), p, "10.1")
    said("model minus control displacement", dd.mean(), p, "−1.150")
    p = "Uncorrected, its seed-0 gradient is −0.062 (5/8 negative, p = .25)"
    said("control seed-0 gradient, uncorrected", real["control"].mean(), p, "−0.062")
    holds(f"control seed 0 uncorrected: {(real['control'] < 0).sum()}/8 negative", (real["control"] < 0).sum() == 5, p)
    said("control seed-0 gradient, uncorrected, p", p_exact(real["control"]), p, ".25")
    p = ("The simulation recovers a known across-ROI split in the right order (ρ = +1.000 in 8/8 subjects, both geometries) "
         "but overstates its size by a factor of 1.23 on the model's geometry and 1.25 on the control's")
    holds("recovery rho = +1 in every subject, both geometries", all(np.allclose(rho[a], 1.0) for a in runs), p)
    said("recovery est/true, model", fid["model"], p, "1.23"); said("recovery est/true, control", fid["control"], p, "1.25")


def check_nuisance_recompute() -> None:
    """The nuisance file against the per-voxel fits: the model's seed-0 partial slopes recomputed from the encoding outputs."""
    import fs2_cortex_controls as cc_
    print(f"\n[S3] nuisance file recomputed from the per-voxel fits ({ENC}), model seed 0")
    J = json.load(open(NUISANCE))
    got = cc_.per_model(MODEL[0])
    for stat in ("norm", "raw"):
        for s in range(1, 9):
            a, b = got[s][stat]["regression"], J["per_seed_regression"][MODEL[0]][stat][str(s)]
            ok = all(abs(a[k] - b[k]) < 1e-9 for k in ("ecc_partial", "size_partial", "vif"))
            if not ok:
                print(f"  BAD subj{s:02d} {stat} regression differs from {NUISANCE}"); FAILS.append(f"nuisance recompute subj{s} {stat}")
    print("  OK  model seed 0 partial slopes and VIF, 8 subjects x 2 statistics, match the committed file" if not any("nuisance recompute" in f for f in FAILS) else "")
    cc_.SIM = VPREC
    eb = cc_.sim_block()
    for arm in ("model", "control"):
        a, b = np.array(eb[arm]["real_norm_seed0"]), np.array(J["estimator_bias"][arm]["real_norm_seed0"])
        ok = np.allclose(a, b, atol=1e-12)
        print(f"  {'OK ' if ok else 'BAD'} {arm} real seed-0 contrasts behind the displacement match the committed file")
        if not ok:
            FAILS.append(f"nuisance recompute real seed0 {arm}")


def check_encoding() -> None:
    """Results 2.3 and Extended Data Table 1 from the per-voxel fits: the gradient per seed on both statistics, the
    paired separation from the control, the untrained network, and the V1 / hV4 unique-variance levels."""
    from src.analysis.encoding.stats import load_results
    print(f"\n[2.3, ED Table 1] per-voxel fits ({ENC})")
    grad = {}
    for stat, ratio in (("normalised", True), ("raw", False)):
        for tag in MODEL + CORTEX_CONTROL + SWAP + [f"{UNTRAINED}_seed0"]:
            subs, prof = ba.roi_profile(ENC, tag, ratio)
            grad[(stat, tag)] = dict(zip(subs, [ba.hv4_minus_v1(r) for r in prof]))

    def seed_mean(stat, runs):   # per-subject mean over the runs' gradients (subjects every run has)
        subs = sorted(set.intersection(*[set(grad[(stat, t)]) for t in runs]))
        return {k: np.mean([grad[(stat, t)][k] for t in runs]) for k in subs}

    ctl = seed_mean("normalised", CORTEX_CONTROL), seed_mean("raw", CORTEX_CONTROL)
    ed1 = {("normalised", 0): "| normalised | 0 | −1.259 | −0.062 | −0.833 | 8/8 | 0.0078 |",
           ("normalised", 1): "|  | 1 | −1.291 | −0.062 | −0.785 | 8/8 | 0.0078 |",
           ("normalised", 2): "|  | 2 | −1.262 | −0.086 | −0.745 | 8/8 | 0.0078 |",
           ("raw", 0): "| raw | 0 | −0.186 | −0.003 | −0.166 | 8/8 | 0.0078 |",
           ("raw", 1): "|  | 1 | −0.193 | −0.003 | −0.145 | 8/8 | 0.0078 |",
           ("raw", 2): "|  | 2 | −0.190 | −0.007 | −0.153 | 8/8 | 0.0078 |"}
    cap = "(against the control's seed mean it is also 8/8, p = .0078, for every seed and both statistics)"
    for (stat, s), row in ed1.items():
        cells = [x.strip() for x in row.strip("|").split("|")]
        d, x = grad[(stat, MODEL[s])], grad[(stat, CORTEX_CONTROL[s])]   # paired seed by seed
        diff = np.array([d[k] - x[k] for k in sorted(set(d) & set(x))])
        said(f"ED T1 {stat}, seed {s}: model", np.mean(list(d.values())), row, cells[2])
        said(f"ED T1 {stat}, seed {s}: control", np.mean(list(x.values())), row, cells[3])
        said(f"ED T1 {stat}, seed {s}: subjects below the control seed", (diff < 0).sum(), row, "8")
        said(f"ED T1 {stat}, seed {s}: p", p_exact(diff), row, cells[6])
        xm = ctl[stat == "raw"]
        dm = np.array([d[k] - xm[k] for k in sorted(set(d) & set(xm))])
        holds(f"ED T1 {stat}, seed {s}: below the control's seed mean {(dm < 0).sum()}/{len(dm)}, p {p_exact(dm):.4f}",
              (dm < 0).all() and len(dm) == 8 and p_exact(dm) < 0.0079, cap)
    raw = np.array([np.mean(list(grad[("raw", t)].values())) for t in MODEL])
    rawc = np.array([np.mean(list(grad[("raw", t)].values())) for t in CORTEX_CONTROL])
    p = "with the same separation on the raw statistic U_m − U_p (−0.186 to −0.193 against −0.003 to −0.007)"
    said("raw gradient, model, shallowest seed", raw.max(), p, "−0.186"); said("raw gradient, model, steepest seed", raw.min(), p, "−0.193")
    said("raw gradient, control, shallowest seed", rawc.max(), p, "−0.003"); said("raw gradient, control, steepest seed", rawc.min(), p, "−0.007")
    said("raw gradient, model, mean of seeds", raw.mean(), "and holds it on the raw difference (−0.19)")
    said("raw gradient, control (Results 2.3)", np.mean(list(ctl[1].values())), "erases it on both (−0.070; −0.004, seed means)", "−0.004")
    row = "| control alone (seed mean), normalised / raw | — | | −0.070 / −0.004 | | 6/8 / 4/8 < 0 | 0.11 / 0.64 |"
    for stat, lit_m, lit_n, lit_p in (("normalised", "−0.070", "6", "0.11"), ("raw", "−0.004", "4", "0.64")):
        x = np.array(list(ctl[stat == "raw"].values()))
        said(f"control alone, {stat}: mean", x.mean(), row, lit_m)
        said(f"control alone, {stat}: subjects below zero", (x < 0).sum(), row, lit_n)
        said(f"control alone, {stat}: p", p_exact(x), row, lit_p)
    swn, swr = (np.array(list(seed_mean(stat, SWAP).values())) for stat in ("normalised", "raw"))
    p = "and +0.002 raw on the seed mean; alone, on the per-subject seed mean, 6/8 and 4/8 subjects below zero (p = .31 and .95)"
    said("ED T1 caption: swap raw, seed mean", swr.mean(), p, "+0.002")
    said("ED T1 caption: swap alone, normalised, below zero", (swn < 0).sum(), p, "6"); said("ED T1 caption: swap alone, raw, below zero", (swr < 0).sum(), p, "4")
    said("ED T1 caption: swap alone, normalised, p", p_exact(swn), p, ".31"); said("ED T1 caption: swap alone, raw, p", p_exact(swr), p, ".95")
    said("swap raw gradient, seed mean (Results 2.3)", swr.mean(), "has no gradient either (−0.019 normalised, +0.002 raw", "+0.002")
    u = {stat: np.array(list(grad[(stat, f"{UNTRAINED}_seed0")].values())) for stat in ("normalised", "raw")}
    row = "| untrained, the model's front end, normalised / raw | — | −0.480 / −0.195 | | | 8/8 < 0 | 0.0078 |"
    p = "An untrained network on the same test input already descends (−0.480 normalised, −0.195 raw, 8/8 subjects, p = .0078 on both"
    for stat, lit in (("normalised", "−0.480"), ("raw", "−0.195")):
        said(f"untrained, {stat}", u[stat].mean(), row, lit)
        said(f"untrained, {stat} (Results 2.3)", u[stat].mean(), p, lit)
        holds(f"untrained, {stat}: {(u[stat] < 0).sum()}/{len(u[stat])} below zero, p {p_exact(u[stat]):.4f}",
              (u[stat] < 0).all() and len(u[stat]) == 8 and p_exact(u[stat]) < 0.0079, p)

    lev = defaultdict(list)
    for tag in MODEL:
        for s, z in sorted(load_results(ENC, tag).items()):
            roi, nc, um, up = z["roi_prf-visualrois"], z["noise_ceiling"], z["unique_magno"], z["unique_parvo"]
            for name, labs in (("V1", [1, 2]), ("hV4", [7])):
                v = np.isin(roi, labs) & (nc > 0.1) & ((um + up) > 1e-3)
                lev[(tag, name)].append((np.median(um[v]), np.median(up[v]), np.mean(um[v] > up[v])))
    L = {k: np.array(v) for k, v in lev.items()}
    per = lambda name, i: np.array([L[(t, name)][:, i].mean() for t in MODEL])
    p = ("(in V1, U_m 0.10–0.11 against U_p 0.03–0.04, U_m larger in 77–82% of voxels and in 8/8 subjects in every seed; "
         "in hV4, 0.015–0.020 against 0.13)")
    for name, i, lo, hi, scale in (("V1 U_m", 0, "0.10", "0.11", 1), ("V1 U_p", 1, "0.03", "0.04", 1), ("V1 voxels U_m > U_p (%)", 2, "77", "82", 100),
                                   ("hV4 U_m", 0, "0.015", "0.020", 1), ("hV4 U_p", 1, "0.13", "0.13", 1)):
        x = scale * per(name.split()[0], i)
        said(f"{name}, lowest seed", x.min(), p, lo); said(f"{name}, highest seed", x.max(), p, hi)
    holds("V1 U_m > U_p in 8/8 subjects in every seed", all((L[(t, "V1")][:, 0] > L[(t, "V1")][:, 1]).sum() == 8 for t in MODEL), p)


def check_behaviour() -> None:
    """Results 2.4, Abstract, Discussion: the integrated rate, the frozen sweep, the refitted heads, the observer fits
    (results/cue_conflict, fusion_sweep, nonlinear_readout_summary.json, fit_w_observers)."""
    print("\n[2.4] behaviour and the readout")
    cf = np.array([cc(r) for r in MODEL])
    said("integrated cue-conflict, mean", cf.mean(), "The integrated readout's cue-conflict shape bias is 0.682 ± 0.020", "0.682")
    said("integrated cue-conflict, s.d. (ddof=1)", cf.std(ddof=1), "The integrated readout's cue-conflict shape bias is 0.682 ± 0.020", "0.020")
    said("control integrated cue-conflict", np.mean([cc(r) for r in CONTROL]), "against 0.374 for the control, 0.958 for the ten human observers", "0.374")
    said("integration cost", np.mean([cc(r, "magno_only") for r in MODEL]) - cf.mean(), "integration costs 0.168 of the magno stream's 0.850", "0.168")
    w, S = sweep(MODEL)
    r = S.mean(0)
    p = "runs cue-conflict shape bias from 0.48 to 0.85 over 21 values of w and three seeds, and w = 0.5 reproduces each trained model exactly"
    said("sweep, w = 0 (seed mean)", r[0], p, "0.48"); said("sweep, w = 1 (seed mean)", r[-1], p, "0.85")
    said("sweep, number of w", len(w), p, "21")
    holds(f"sweep monotone ends: w = 0 lowest, w = 1 highest; {len(S)} seeds", r[0] == r.min() and r[-1] == r.max() and len(S) == 3, p)
    i05 = int(np.argmin(np.abs(w - 0.5)))
    dev = float(np.abs(S[:, i05] - cf).max())
    holds(f"w = 0.5 reproduces each trained model (max |d| {dev:.1e})", dev < 1e-6, p)
    said("Abstract: sweep, w = 0", r[0], "one readout weight moves the rate from 0.48 to 0.85", "0.48")
    said("Abstract: sweep, w = 1", r[-1], "one readout weight moves the rate from 0.48 to 0.85", "0.85")
    rc = sweep(CONTROL)[1].mean(0)
    rs = sweep(SWAP)[1].mean(0)
    p = "The control's curve is flat (0.37–0.42), and so is the channel swap's (0.38–0.40)"
    said("control sweep, lowest", rc.min(), p, "0.37"); said("control sweep, highest", rc.max(), p, "0.42")
    said("swap sweep, lowest", rs.min(), p, "0.38"); said("swap sweep, highest", rs.max(), p, "0.40")

    NL = json.load(open("results/nonlinear_readout_summary.json"))
    sp = {h: np.array([NL[f"{x}_final"][h]["span"] for x in MODEL]) for h in ("affine", "mlp")}
    gated = np.array([NL[f"{x}_final"]["mlp"]["gated"] for x in MODEL])
    p = ("its median span is 0.395 against 0.371 for the frozen affine head (106%; weakest seed 96%; threshold 80%), and 0.241 "
         "when restricted to readouts within 5 points of peak validation accuracy")
    said("refit: two-layer median span", np.median(sp["mlp"]), p, "0.395")
    said("refit: frozen affine median span", np.median(sp["affine"]), p, "0.371")
    said("refit: two-layer / affine (%)", 100 * np.median(sp["mlp"]) / np.median(sp["affine"]), p, "106")
    said("refit: weakest seed (%)", 100 * (sp["mlp"] / sp["affine"]).min(), p, "96")
    said("refit: gated two-layer median span", np.median(gated), p, "0.241")

    W = [json.load(open(f"results/fit_w_observers/{x}_final_wfit.json")) for x in MODEL]
    obs = [o for d in W for o in d["observers"].values()]
    p = "no observer's rate is reachable in any run (0 of 30 observer-runs)"
    said("fit-w: attainable observer-runs", sum(o["rate_fit"]["attainable"] for o in obs), p, "0"); said("fit-w: observer-runs", len(obs), p, "30")
    p = "(held-out agreement 0.039 against 0.031 for an arbitrary weight, above it in 17 of 30 observer-runs)"
    said("fit-w: held-out agreement", np.mean([o["cross_validation"]["held_out_mean"] for o in obs]), p, "0.039")
    said("fit-w: arbitrary-weight floor", np.mean([o["cross_validation"]["no_fitting_floor_mean"] for o in obs]), p, "0.031")
    said("fit-w: held-out above the floor", sum(o["cross_validation"]["held_out_above_floor"] > 0 for o in obs), p, "17")
    per = defaultdict(list)
    for d in W:
        for k, o in d["observers"].items():
            per[k].append(o["trial_fit"]["w_hat"])
    bw = np.array([np.mean(v) for v in per.values()])
    widths = [o["trial_fit"]["boot_ci"][1] - o["trial_fit"]["boot_ci"][0] for o in obs]
    p = ("(median 0.79 over observers, but per-fit bootstrap intervals average 0.70 wide on [0, 1] and only 10 of 30 fits beat a "
         "permutation null)")
    said("behaviour w*: median over observers", np.median(bw), p, "0.79")
    said("behaviour w*: mean bootstrap interval width", np.mean(widths), p, "0.70")
    said("behaviour w*: fits beating the permutation null", sum(o["permutation_null"]["p_perm"] < 0.05 for o in obs), p, "10")
    row = "| behaviour (model) | 10 human observers, trial fit | 0.79 [0.33, 0.97] | — | — | — | — | — | 0.80 | 10 |"
    said("ED T2 behaviour: w* median", np.median(bw), row, "0.79"); said("ED T2 behaviour: w* lowest", bw.min(), row, "0.33")
    said("ED T2 behaviour: w* highest", bw.max(), row, "0.97"); said("ED T2 behaviour: rate at w*", np.interp(np.median(bw), w, r), row, "0.80")
    said("ED T2 behaviour: observers", len(bw), row, "10")

    span = r.max() - r.min()
    Z = {n["tag"]: n for n in json.load(open("results/figures_v2/exemplar_zoo.json"))["networks"]}
    g6 = [Z[t]["spans"]["E1"]["gated"] for t in ("tv_resnet50", "tv_resnet152", "tv_vgg16", "tv_vit_b_16", "tv_convnext_small", "tv_convnext_base")]
    p = "by 0.14–0.20, 37–55% of the model's two-stream span (0.37)"
    said("model's two-stream span", span, p, "0.37")
    said("six networks' smallest span, % of the model's", 100 * min(g6) / span, p, "37")
    said("six networks' largest span, % of the model's", 100 * max(g6) / span, p, "55")
    T = json.load(open(EXEMPLAR))["trunks"]
    e1 = np.mean([T[f"concat_{x}"]["spans"]["E1"]["span"] for x in MODEL]) if all(f"concat_{x}" in T for x in MODEL) else np.nan
    p = "(a span of 0.37 between the streams against 0.20 for the same network's concatenated features with exemplars held out)"
    said("Discussion: two-stream span", span, p, "0.37"); said("Discussion: concatenation, exemplars held out", e1, p, "0.20")


def check_readout_w_summary() -> None:
    """Results 2.4, Abstract, Extended Data Table 2 caption: the cortex-optimal readout weight, V1 against
    extrastriate, from the committed cortex summary."""
    print("\n[2.4] cortex-optimal readout weight (fs2_cortex_summary.json)")
    C = json.load(open(CORTEX))["readout_w"]
    w, S = sweep(MODEL)
    r = S.mean(0)
    for arm, pre in (("dvd300", "model"), ("control", "control"), ("swap", "channel swap")):
        R = C[arm]
        d = np.array(R["delta"])
        rng = np.random.default_rng(0)
        lo, hi = np.percentile([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)], [2.5, 97.5])
        if arm == "dvd300":
            p = ("In the model that weight falls from 0.65 in V1 to 0.30–0.35 in every extrastriate group, below V1 in 8/8 subjects "
                 "(Δ = −0.323 [−0.371, −0.269], exact p = .0078); at the extrastriate weight the network's own rate would be 0.60, at V1's 0.75")
            said("model w*: V1 median", np.median(R["w_v1"]), p, "0.65")
            said("model w*: extrastriate minus V1", d.mean(), p, "−0.323")
            said("model w*: bootstrap 2.5%", lo, p, "−0.371"); said("model w*: bootstrap 97.5%", hi, p, "−0.269")
            holds(f"model w*: extrastriate below V1 in {(d < 0).sum()}/{len(d)}, p {p_exact(d):.4f}", (d < 0).all() and len(d) == 8 and p_exact(d) < 0.0079, p)
            said("rate at the extrastriate weight", np.interp(np.median(R["w_ext"]), w, r), p, "0.60")
            said("rate at V1's weight", np.interp(np.median(R["w_v1"]), w, r), p, "0.75")
            p = "the weight best predicting extrastriate cortex (0.30–0.35) lies below the one best predicting V1 (0.65)"
            said("Abstract: V1 w*", np.median(R["w_v1"]), p, "0.65")
        elif arm == "control":
            holds(f"control w*: {R['n_seeds']} seeds", R["n_seeds"] == 3, "reversed in the control")
            p = "reversed in the control (+0.308 [+0.194, +0.425], positive in 8/8 subjects, p = .0078)"
            said("control w*: extrastriate minus V1", d.mean(), p, "+0.308")
            said("control w*: bootstrap 2.5%", lo, p, "+0.194"); said("control w*: bootstrap 97.5%", hi, p, "+0.425")
            said("control w*: subjects positive", (d > 1e-9).sum(), p, "8")
            said("control w*: p (exact)", p_exact(d), p, ".0078")
            p = "In the control the gradient reverses (0.43 in V1 to 0.62–0.78 beyond hV4; higher in 8/8 subjects, p = .0078)"
            said("control w*: V1 median", np.median(R["w_v1"]), p, "0.43")
            said("control w*: subjects higher (Results 2.4)", (d > 1e-9).sum(), p, "8")
        else:
            nz = d[np.abs(d) > 1e-9]   # one subject's difference is zero to floating point: a tie, dropped
            p = "and in the channel swap (+0.358 [+0.221, +0.479], positive in 7/8 subjects and tied in one, p = .016, the tie dropped)"
            said("swap w*: extrastriate minus V1", d.mean(), p, "+0.358")
            said("swap w*: bootstrap 2.5%", lo, p, "+0.221"); said("swap w*: bootstrap 97.5%", hi, p, "+0.479")
            said("swap w*: subjects positive", (nz > 0).sum(), p, "7")
            holds(f"swap w*: one subject tied ({(np.abs(d) < 1e-9).sum()})", (np.abs(d) < 1e-9).sum() == 1, p)
            said("swap w*: p (exact, tie dropped)", p_exact(nz), p, ".016")
            p = "and so it does in the channel swap (0.38 in V1 to 0.62–1.00; higher in 7/8 subjects and tied in one, p = .016)"
            said("swap w*: V1 median", np.median(R["w_v1"]), p, "0.38")
            said("swap w*: subjects higher (Results 2.4)", (nz > 0).sum(), p, "7")


def check_readout_w() -> None:
    """Extended Data Table 2 and Results 2.4 from the per-voxel readout-weight fits, reduced by the rule of
    tmp_analysis/figs_v2/readout_w_cortex.py; every cell of the model and control rows."""
    print(f"\n[2.4, ED Table 2] readout-weight fits ({RW})")
    VIS = {"V1": [1, 2], "V2": [3, 4], "V3": [5, 6], "hV4": [7]}
    KAST = {"ventral (VO, PHC)": [8, 9, 10, 11], "lateral (LO, TO)": [12, 13, 14, 15], "dorsal (V3A/B, IPS0–2)": [16, 17, 18, 19, 20]}
    arms = {"model": MODEL, "control": CORTEX_CONTROL, "channel swap": SWAP}
    curves, conc, best = defaultdict(list), defaultdict(list), defaultdict(list)
    grid = None
    for f in sorted(glob.glob(f"{RW}/*.npz")):
        z = np.load(f, allow_pickle=True)
        name = str(z["model_name"])
        arm = next((a for a, runs in arms.items() if any(name.startswith(x) for x in runs)), None)
        if arm is None:
            continue
        grid = z["grid"]
        ok = (z["noise_ceiling"] > 0.1) & np.isfinite(z["r2"]).all(0)
        for g, labs in {**VIS, **KAST}.items():
            v = ok & np.isin(z["roi_prf-visualrois"] if g in VIS else z["roi_Kastner2015"], labs)
            if v.sum() >= 20:
                key = (arm, int(z["subject"]), g)
                curves[key].append(np.median(z["r2"][:, v], axis=1))
                conc[key].append(np.median(z["r2_concat"][v]))
                best[key].append(grid[np.argmax(z["r2"][:, v], axis=0)])
    rate = {a: sweep(runs)[1].mean(0) for a, runs in arms.items()}
    i05 = int(np.argmin(np.abs(grid - 0.5)))
    half_model, ext_w = [], defaultdict(list)
    for arm, runs in arms.items():
        for g in {**VIS, **KAST}:
            row = next(x for x in PAPER.splitlines() if x.startswith(f"| {arm} | {g} |"))
            cells = [x.strip() for x in row.strip("|").split("|")]
            ws, half, mx, cc_, fm, fp, nseed = [], [], [], [], [], [], set()
            for s in range(1, 9):
                key = (arm, s, g)
                if key not in curves:
                    continue
                c = np.mean(curves[key], 0)
                ws.append(grid[int(np.argmax(c))]); half.append(c[i05] / c.max()); mx.append(c.max()); cc_.append(np.mean(conc[key]))
                bw = np.concatenate(best[key]); fm.append((bw >= 0.75).mean()); fp.append((bw <= 0.25).mean()); nseed.add(len(curves[key]))
            ws = np.array(ws)
            wm, wlo, whi = NUM.findall(cells[2])
            for lab, v, lit in (("w* median", np.median(ws), wm), ("w* lowest", ws.min(), wlo), ("w* highest", ws.max(), whi),
                                ("R² at 0.5 / max", np.median(half), cells[3]), ("max R²", np.median(mx), cells[4]),
                                ("concat R²", np.median(cc_), cells[5]), ("voxels w ≥ 0.75", np.mean(fm), cells[6]),
                                ("voxels w ≤ 0.25", np.mean(fp), cells[7]), ("rate at w*", np.interp(np.median(ws), grid, rate[arm]), cells[8]),
                                ("subjects", len(ws), cells[9])):
                said(f"ED T2 {arm}, {g}: {lab}", v, row, lit)
            holds(f"ED T2 {arm}, {g}: {sorted(nseed)} seed(s) per subject", nseed == {len(runs)}, row)
            if arm == "model":
                half_model.append(np.median(half))
            if g in KAST:
                ext_w[arm].append(np.median(ws))
    for arm, lo_, hi_, p in (("control", "0.62", "0.78", "In the control the gradient reverses (0.43 in V1 to 0.62–0.78 beyond hV4"),
                             ("channel swap", "0.62", "1.00", "and so it does in the channel swap (0.38 in V1 to 0.62–1.00")):
        said(f"{arm}: lowest extrastriate w* (Results 2.4)", min(ext_w[arm]), p, lo_)
        said(f"{arm}: highest extrastriate w* (Results 2.4)", max(ext_w[arm]), p, hi_)
    # at the table's printed precision
    holds(f"every model ROI within 2% of its best r² at w = 0.5 (lowest {min(half_model):.4f})", round(min(half_model), 3) >= 0.98,
          "every ROI is within 2% of its best r² at w = 0.5")


def check_dissociations() -> None:
    """Results 2.5 and Discussion: OOD robustness against the control, error consistency, participation ratio."""
    print("\n[2.5] robustness, agreement with observers, dimensionality")

    def rel_auc(runs, fam):
        acc = np.array([json.load(open(f"results/ood/{x}.json"))["families"][fam]["accuracy"] for x in runs], float)
        return (acc / acc[:, :1]).mean(1)

    four = ("uniform noise (−0.135), high-pass (−0.136), false colour (−0.084) and rotation (−0.040), every seed of the model "
            "below every seed of the control (exact Mann–Whitney p = 0.10, the floor for three seeds against three)")
    for fam, lit, p in (("uniform-noise", "−0.135", four), ("high-pass", "−0.136", four), ("false-colour", "−0.084", four),
                        ("rotation", "−0.040", four), ("phase-scrambling", "+0.009", "within 0.01 of it on phase scrambling (+0.009)"),
                        ("low-pass", "+0.078", "and more robust only to low-pass (+0.078)"), ("contrast", "−0.004", "is level (−0.004, seeds interleaved)")):
        said(f"OOD relative AUC, {fam}: model minus control", rel_auc(MODEL, fam).mean() - rel_auc(CONTROL, fam).mean(), p, lit)
    holds("every model seed below every control seed on the four",
          all(rel_auc(MODEL, f).max() < rel_auc(CONTROL, f).min() for f in ("uniform-noise", "high-pass", "false-colour", "rotation")), four)
    said("OOD: exact Mann–Whitney p, complete separation", mannwhitneyu(rel_auc(MODEL, "rotation"), rel_auc(CONTROL, "rotation"),
                                                                     method="exact").pvalue, four, "0.10")
    a, b = rel_auc(MODEL, "contrast"), rel_auc(CONTROL, "contrast")
    holds("contrast: model and control seeds interleaved", a.max() > b.min() and b.max() > a.min(), "is level (−0.004, seeds interleaved)")

    k = lambda runs, w: np.array([kappa_at(x, w) for x in runs])
    ceil = kappa(MODEL[0])["human_human_ceiling"]
    said("human-human κ ceiling", ceil, "the human–human ceiling is κ = 0.234")
    p = ("at 17.3% of the ceiling, above its integrated readout (13.2%), its parvo stream (10.3%) and the control "
         "(11.9% integrated, 12.6% magno)")
    said("κ, model magno (% ceiling)", k(MODEL, 1.0).mean(), p, "17.3"); said("κ, model integrated", k(MODEL, 0.5).mean(), p, "13.2")
    said("κ, model parvo", k(MODEL, 0.0).mean(), p, "10.3"); said("κ, control integrated", k(CONTROL, 0.5).mean(), p, "11.9")
    said("κ, control magno", k(CONTROL, 1.0).mean(), p, "12.6")
    allk = [q["kappa_pct_ceiling"] for x in MODEL for q in kappa(x)["sweep"]]
    p = "the best of its 63 sweep readouts reaches 29.0%"
    said("κ sweep readouts, model", len(allk), p, "63"); said("κ, best sweep readout", max(allk), p, "29.0")
    ours = [q["kappa_pct_ceiling"] for f in glob.glob("results/sweep_error_consistency/*_kappa_sweep.json")
            if os.path.basename(f).startswith(("dkl_", "fs2_")) for q in json.load(open(f))["sweep"]]
    holds(f"κ never above 29% of the ceiling in our networks (max {max(ours):.1f})", max(ours) < 29.5,
          "never exceeds 29% of the human–human ceiling in our networks")
    sbr = lambda runs, cond: float(np.mean([cc(x, cond) for x in runs]))
    pts = [(sbr(MODEL, "full"), k(MODEL, 0.5).mean() * ceil / 100), (sbr(CONTROL, "full"), k(CONTROL, 0.5).mean() * ceil / 100),
           (sbr(MODEL, "magno_only"), k(MODEL, 1.0).mean() * ceil / 100), (sbr(CONTROL, "magno_only"), k(CONTROL, 1.0).mean() * ceil / 100),
           *PUBLISHED_CNNS.values()]
    sb, kap = np.array(pts).T
    res, span, tq = linregress(sb, kap), sb.max() - sb.min(), tdist.ppf(0.95, len(sb) - 2)
    p = "κ rises with the rate by 0.019 [0.011, 0.027] (90% interval) over the 0.68 span of shape bias they cover, 8.1% of the ceiling"
    said("seven models: κ change over the span", res.slope * span, p, "0.019")
    said("seven models: 90% interval, low", (res.slope - tq * res.stderr) * span, p, "0.011")
    said("seven models: 90% interval, high", (res.slope + tq * res.stderr) * span, p, "0.027")
    said("seven models: shape-bias span", span, p, "0.68"); said("seven models: change, % of ceiling", 100 * res.slope * span / ceil, p, "8.1")
    p = "the three magno streams sit at 0.871, 0.842 and 0.838 and reach 5.1%, 25.6% and 21.3% of the ceiling"
    for x, lsb, lk in zip(MODEL, ("0.871", "0.842", "0.838"), ("5.1", "25.6", "21.3")):
        said(f"{x[-5:]} magno: cue-conflict", cc(x, "magno_only"), p, lsb); said(f"{x[-5:]} magno: κ (% ceiling)", kappa_at(x, 1.0), p, lk)
    p = ("is not supported: the model's integrated readout reaches 13.2% of the ceiling (seeds 9.4–17.7%) and the control's "
         "11.9% (10.3–13.2%), with the seeds interleaved (exact Mann–Whitney p = 1.0; magno streams 17.3% against 12.6%, p = 0.70)")
    km, kc = k(MODEL, 0.5), k(CONTROL, 0.5)
    said("κ integrated, model (prediction)", km.mean(), p, "13.2"); said("κ integrated, control (prediction)", kc.mean(), p, "11.9")
    said("κ integrated, lowest model seed", km.min(), p, "9.4"); said("κ integrated, highest model seed", km.max(), p, "17.7")
    said("κ integrated, lowest control seed", kc.min(), p, "10.3"); said("κ integrated, highest control seed", kc.max(), p, "13.2")
    said("κ integrated, exact Mann–Whitney p", mannwhitneyu(km, kc, method="exact").pvalue, p, "1.0")
    said("κ magno, exact Mann–Whitney p", mannwhitneyu(k(MODEL, 1.0), k(CONTROL, 1.0), method="exact").pvalue, p, "0.70")

    # per-image correspondence (tmp_analysis/perimage_correspondence.py over the BA and fs2 logit files, 2026-10-24)
    PI = json.load(open("results/perimage_behavior/perimage_correspondence_v3.json"))["models"]
    pi = lambda runs, key, f="oos_pct_ceiling": np.array([PI[f"{x}_final"][key][f] for x in runs])
    dif, cue = "valid_rate_resid__pearson", "shape_rate_resid__pearson"
    p = ("the model's readout recovers 22.3% of the split-half ceiling on held-out images (seeds 19.6–24.3%), against 9.0% for the "
         "control (6.6–10.4%) and 8.2% for the channel swap (7.1–10.4%), every model seed above every seed of both (exact Mann–Whitney "
         "p = 0.10, the floor); on which cue is picked the three are interleaved (13.3%, 10.4% and 9.5%; p ≥ 0.70; Fig. 5c)")
    for lab, runs, lits in (("model", MODEL, ("22.3", "19.6", "24.3")), ("control", CONTROL, ("9.0", "6.6", "10.4")),
                            ("swap", SWAP, ("8.2", "7.1", "10.4"))):
        v = pi(runs, dif)
        said(f"per-image difficulty, {lab}: mean", v.mean(), p, lits[0])
        said(f"per-image difficulty, {lab}: lowest seed", v.min(), p, lits[1]); said(f"per-image difficulty, {lab}: highest seed", v.max(), p, lits[2])
    m_, c_, s_ = pi(MODEL, dif), pi(CONTROL, dif), pi(SWAP, dif)
    holds("per-image difficulty: every model seed above every control and swap seed", m_.min() > max(c_.max(), s_.max()), p)
    said("per-image difficulty: exact Mann–Whitney p, model vs control", mannwhitneyu(m_, c_, method="exact").pvalue, p, "0.10")
    said("per-image difficulty: exact Mann–Whitney p, model vs swap", mannwhitneyu(m_, s_, method="exact").pvalue, p, "0.10")
    for lab, runs, lit in (("model", MODEL, "13.3"), ("control", CONTROL, "10.4"), ("swap", SWAP, "9.5")):
        said(f"per-image cue picked, {lab}", pi(runs, cue).mean(), p, lit)
    pc = [mannwhitneyu(pi(MODEL, cue), pi(x, cue), method="exact").pvalue for x in (CONTROL, SWAP)]
    holds(f"per-image cue picked: model vs control and swap p ≥ 0.70 ({min(pc):.2f})", min(pc) >= 0.695, p)
    holds("per-image cue picked: model seeds interleaved with the control's",
          pi(MODEL, cue).max() > pi(CONTROL, cue).min() and pi(CONTROL, cue).max() > pi(MODEL, cue).min(), p)
    w = pi(MODEL, dif, "oos_w_mean")
    p = "The readout weight that recovers it is magno-heavy (0.91–0.98, chosen on held-out images)"
    said("per-image difficulty: model w, lowest seed", w.min(), p, "0.91"); said("per-image difficulty: model w, highest seed", w.max(), p, "0.98")
    ba_ = lambda t: np.mean([PI[t.format(s)][dif]["oos_pct_ceiling"] for s in range(3)])
    p = ("The BA network showed the same contrast against its no-diet arm (24.1% against 9.2%), but crossing weights and test input "
         "removed it in both directions (−0.6% and −3.7%; Fig. 5d)")
    for lab, t, lit in (("BA", "dkl_bcs_seed{}_final", "24.1"), ("BA no diet", "dkl_base_seed{}_final", "9.2"),
                        ("BA weights, front end off", "dkl_bcs_seed{}_final_nofe", "−0.6"), ("no-diet weights, front end on", "dkl_base_seed{}_final_fe", "−3.7")):
        said(f"per-image difficulty, {lab}", ba_(t), p, lit)

    PR = json.load(open("results/figures_v2/fs2_pr_ratio.json"))["runs"]
    rat = lambda run: np.array([v["ratio"] for v in PR[run].values()])
    allm = np.concatenate([rat(x) for x in MODEL])
    seedm = [rat(x).mean() for x in MODEL]
    ctl = np.concatenate([rat(x) for x in CORTEX_CONTROL])
    p = ("ends above parity in every subject and seed of the model (24/24; seed means 1.88–1.98) and below it in 23 of its "
         "control's 24 (seed means 0.89–0.95), from the same untrained starting point (1.63)")
    said("PR ratio: model subject-seeds above parity", (allm > 1).sum(), p, "24")
    holds(f"PR ratio: {len(allm)} model subject-seeds", len(allm) == 24, p)
    said("PR ratio: model seed mean, lowest", min(seedm), p, "1.88"); said("PR ratio: model seed mean, highest", max(seedm), p, "1.98")
    seedc = [rat(x).mean() for x in CORTEX_CONTROL]
    said("PR ratio: control subject-seeds below parity", (ctl < 1).sum(), p, "23"); holds(f"PR ratio: {len(ctl)} control subject-seeds", len(ctl) == 24, p)
    said("PR ratio: control seed mean, lowest", min(seedc), p, "0.89"); said("PR ratio: control seed mean, highest", max(seedc), p, "0.95")
    sw = np.concatenate([rat(x) for x in SWAP]); seeds = [rat(x).mean() for x in SWAP]
    q = "The channel swap also ends above parity (24/24; seed means 1.27–2.15)"
    said("PR ratio: swap subject-seeds above parity", (sw > 1).sum(), q, "24"); holds(f"PR ratio: {len(sw)} swap subject-seeds", len(sw) == 24, q)
    said("PR ratio: swap seed mean, lowest", min(seeds), q, "1.27"); said("PR ratio: swap seed mean, highest", max(seeds), q, "2.15")
    pr = lambda run, k: np.mean([v[k] for v in PR[run].values()])
    q = ("its ramped chromatic stream contracts (participation ratio 52–86 against 116–124 in the control, seed means) while its "
         "luminance stream stays where the control's is (109–113 against 108–110), whereas in the model the ramped luminance stream expands (272–281)")
    for lab, runs, k, lo_, hi_ in (("swap parvo", SWAP, "pr_parvo", "52", "86"), ("control parvo", CORTEX_CONTROL, "pr_parvo", "116", "124"),
                                   ("swap magno", SWAP, "pr_magno", "109", "113"), ("control magno", CORTEX_CONTROL, "pr_magno", "108", "110"),
                                   ("model magno", MODEL, "pr_magno", "272", "281")):
        v = [pr(x, k) for x in runs]
        said(f"PR: {lab}, lowest seed", min(v), q, lo_); said(f"PR: {lab}, highest seed", max(v), q, hi_)
    said("PR ratio: untrained mean", rat(f"{UNTRAINED}_seed0").mean(), p, "1.63")


def check_ed_table3() -> None:
    """Extended Data Table 3, the model's row: the subspace gain on the concatenated streams, from the exemplar
    pipeline (exemplar_streams.json trunks; E0 exemplar-shared, E1 exemplar-held-out)."""
    print("\n[ED Table 3] the model's concatenated streams")
    T = json.load(open(EXEMPLAR))["trunks"]
    tags = [f"concat_{x}" for x in MODEL]
    if not all(t in T and "ends" in T[t]["spans"]["E0"] for t in tags):
        print(f"  BAD {EXEMPLAR} lacks the model's concatenation trunks (or their curve ends): run slurm/exemplar_aggregate.sh")
        FAILS.append("exemplar_streams.json lacks concat_fs2 trunks")
        return
    e0 = np.array([T[t]["spans"]["E0"]["span"] for t in tags]); e1 = np.array([T[t]["spans"]["E1"]["span"] for t in tags])
    ends = np.array([T[t]["spans"]["E0"]["ends"] for t in tags])
    row = ("| two-stream, model (developmental order), streams concatenated (validation) | 1024 | 0.682 | 0.515 | 0.800 | "
           "0.286 [0.25, 0.32] (seed range) | 0.202 [0.17, 0.23] |")
    for lab, v, lit in (("trained rate", np.mean([T[t]["trained"] for t in tags]), "0.682"), ("w = 0", ends[:, 0].mean(), "0.515"),
                        ("w = 1", ends[:, 1].mean(), "0.800"), ("span, exemplars shared", e0.mean(), "0.286"),
                        ("span, shared, lowest seed", e0.min(), "0.25"), ("span, shared, highest seed", e0.max(), "0.32"),
                        ("span, exemplars held out", e1.mean(), "0.202"), ("span, held out, lowest seed", e1.min(), "0.17"),
                        ("span, held out, highest seed", e1.max(), "0.23")):
        said(f"ED T3 model: {lab}", v, row, lit)
    p = ("| 0.202 [0.17, 0.23] | 0.99 | 0.25 (0.07–0.93; 7.0 pts; shape basis on stream 1: 0.58–0.62) | 0.181 | — | 0.11 / 0.28 | "
         "0.28 / 0.23 | 0.26 | 3 |")
    said("ED T3 model: forced-estimator span", np.mean([T[t]["spans"]["E0"]["forced_span"] for t in tags]), p, "0.181")
    # the subspace aggregate of the v3 set (tmp_analysis/figs_v2/subspace_behaviour.py --set v3, v3 unit-split base)
    fam = {f["key"]: f for f in json.load(open("results/figures_v3/subspace_behaviour.json"))["families"]}["concat_dvd300"]
    sd = fam["seeds"]
    holds("ED T3 model: the subspace aggregate covers the three seeds", [s["tag"] for s in sd] == tags, p)
    dtop = max(max(abs(np.array(s["gain_accuracy"]["top1"]) - s["gain_accuracy"]["trained_top1"])) for s in sd) * 100
    e1 = [s["descriptives"]["energy_S_on_stream1_full"] for s in sd]
    mean = lambda f: float(np.mean([f(s) for s in sd]))
    for lab, v, lit in (("ρ(w, rate)", fam["rho"], "0.99"), ("gated span", fam["gated_span"], "0.25"),
                        ("gate, lowest w", fam["gate"][0], "0.07"), ("gate, highest w", fam["gate"][1], "0.93"),
                        ("largest change of top-1 (pts)", dtop, "7.0"),
                        ("shape basis on stream 1, lowest seed", min(e1), "0.58"), ("shape basis on stream 1, highest seed", max(e1), "0.62"),
                        ("variance in S", mean(lambda s: s["descriptives"]["var_frac_S"]), "0.11"),
                        ("variance in T", mean(lambda s: s["descriptives"]["var_frac_T"]), "0.28"),
                        ("order sensitivity, S first", mean(lambda s: s["sensitivity"]["S_first"]["span"]), "0.28"),
                        ("order sensitivity, T first", mean(lambda s: s["sensitivity"]["T_first"]["span"]), "0.23"),
                        ("median-split gated span", fam["unit_gated_span"], "0.26")):
        said(f"ED T3 model: {lab}", v, p, lit)
    holds("ED T3 model: gate measured on MiniEcoset validation", all(s["gain_accuracy"]["set"] == "miniecoset_val" for s in sd),
          "for the model's, gated on MiniEcoset validation, it is given after the gate")


CAPTION = "results/figures_v3/{}_caption.md"


def check_review_v9() -> None:
    """The referee simulation v9 consistency pass (2026-10-24): what it rewrote or added in the Abstract, Results 2.4,
    Methods 4.5 and 4.8, the Extended Data Table 4 caption and the Fig. 6 caption, from the committed result files; and
    the wordings it removed, which must not come back."""
    print("\n[v9 pass] the consistency fixes")
    from scipy.stats import spearmanr
    # Methods 4.5: the concatenation is a reference, not a bound (the BA block's statistic, on the model and its control)
    W = json.load(open("results/figures_v3/readout_w_cortex.json"))["arms"]
    cells = [R for a in ("model", "control") for R in W[a].values()]
    mx = np.concatenate([np.array(R["max_r2"], float) for R in cells])
    exc = mx - np.concatenate([np.array(R["concat"], float) for R in cells])
    buys = float(np.median((1.0 - np.concatenate([np.array(R["r2_at_half_over_max"], float) for R in cells])) * mx))
    pos = exc > 0
    p = ("in the model and its control the best mixture exceeds the fitted concatenation in 15 of the 112 subject × ROI × "
         "arm cells by a median 0.0023 r² — more than the 0.0012 the sweep itself buys from w = ½ to w*")
    said("4.5: cells where the best mixture beats the concatenation", pos.sum(), p, "15"); said("4.5: cells", len(mx), p, "112")
    said("4.5: median excess over the concatenation", np.median(exc[pos]), p, "0.0023"); said("4.5: r² the sweep buys", buys, p, "0.0012")
    holds("4.5: the excess exceeds what the sweep buys", np.median(exc[pos]) > buys, p)
    # Abstract: the extrastriate optimum is the range of Extended Data Table 2's three group rows
    g = [R["w_star_median"] for k, R in W["model"].items() if k.split(" ")[0] in ("ventral", "lateral", "dorsal")]
    p = "the weight best predicting extrastriate cortex (0.30–0.35) lies below the one best predicting V1 (0.65)"
    holds(f"Abstract: the three extrastriate groups ({len(g)})", len(g) == 3, p)
    said("Abstract: extrastriate w*, lowest group", min(g), p, "0.30"); said("Abstract: extrastriate w*, highest group", max(g), p, "0.35")
    # Abstract: of the pairs the trained rate resolves, the share the gain reverses, against the resampling floors (Results 2.4)
    F = json.load(open("results/figures_v2/exemplar_floors.json"))["units"]
    frac = {u: F[u]["observed"]["heldout_gate"]["n_resolved_reversed"] / F[u]["n_resolved"] for u in ("image", "texture_exemplar")}
    p = "re-ordering 3–8% of the network pairs the trained rate resolves, against resampling floors below 0.5%"
    said("Abstract: resolved pairs reversed, texture exemplars (%)", 100 * frac["texture_exemplar"], p, "3")
    said("Abstract: resolved pairs reversed, images (%)", 100 * frac["image"], p, "8")
    holds("Abstract: both resolved-pair floors below 0.5%", max(F[u]["floor_resolved"] for u in frac) < 0.005, p)
    # Methods 4.8 and the Extended Data Table 4 caption: the zoo's median span and its ρ with the trained rate, both gates
    G = json.load(open("results/figures_v2/exemplar_gate.json"))
    im = [r for r in G["networks"] if r["group"] == "imagenet"]
    Z = {n["tag"]: n for n in json.load(open("results/figures_v2/exemplar_zoo.json"))["networks"]}
    tr = [r["trained"] for r in im]
    fold = [r["heldout"]["gated_span"] for r in im]
    allim = [Z[r["tag"]]["spans"]["E1"]["gated"] for r in im]
    holds(f"zoo: 38 ImageNet networks ({len(im)})", len(im) == 38, "the zoo summaries barely change")
    p = ("median 0.18 against 0.19, 11 of 38 ≥ 0.20; 19% of pairs reversed, τ = 0.63; Spearman ρ with the trained rate +0.17 "
         "against +0.10)")
    said("4.8: median gated span, fold-basis gate", np.median(fold), p, "0.18")
    said("4.8: median gated span, all-image gate", np.median(allim), p, "0.19")
    said("4.8: ρ(span, trained rate), fold-basis gate", spearmanr(fold, tr)[0], p, "+0.17")
    said("4.8: ρ(span, trained rate), all-image gate", spearmanr(allim, tr)[0], p, "+0.10")
    p = ("gated span 0.08–0.29 (median 0.19; 0.18 with the gate recomputed on the held-out fold bases, as in Results 2.4 and "
         "Methods 4.8), ≥ 0.20 in 11; Spearman ρ with the trained rate +0.10 (+0.17 on the fold-basis gate)")
    said("ED T4 caption: median, all-image gate", np.median(allim), p, "0.19")
    said("ED T4 caption: median, fold-basis gate", np.median(fold), p, "0.18")
    said("ED T4 caption: networks ≥ 0.20, all-image gate", sum(v >= 0.20 for v in allim), p, "11")
    said("ED T4 caption: ρ, all-image gate", spearmanr(allim, tr)[0], p, "+0.10")
    said("ED T4 caption: ρ, fold-basis gate", spearmanr(fold, tr)[0], p, "+0.17")
    # Results 2.4, closing: the within-budget span is the refitted head's (check_behaviour checks 0.241)
    NL = json.load(open("results/nonlinear_readout_summary.json"))
    said("2.4 closing: refitted head within five points of peak accuracy", np.median([NL[f"{x}_final"]["mlp"]["gated"] for x in MODEL]),
         "from 0.48 to 0.85 (by 0.24 for a refitted head held within five points of peak accuracy)", "0.24")
    # Supplementary Note S4 is scoped to the BA network and no longer contradicts the Discussion
    holds("S4 is scoped to the BA network", True, "These passages are PAPER_v2's and describe the BA network")
    holds("S4's 'do not support it' is the BA network's fixed transform", True,
          "for the BA network's fixed transform our results do not support it")
    holds("the Discussion's reading of the trajectory stands", True, "Our results support a version of it in which the input's history matters")

    # Fig. 6 caption: the two-stream row and panel c's reference are the model
    cap = open(CAPTION.format("fig6_generality"), encoding="utf8").read()
    trc = json.load(open("results/figures_v2/exemplar_trunk_curves.json"))["families"]["concat_dvd300"]
    fam = {f["key"]: f for f in json.load(open("results/figures_v3/subspace_behaviour.json"))["families"]}["concat_dvd300"]
    gen = {f["key"]: f for f in json.load(open("results/figures_v3/generality_behaviour.json"))["families"]}["concat_dvd300"]
    holds("Fig. 6: held-out curves are the model's three seeds", [x["tag"] for x in trc] == [f"concat_{x}" for x in MODEL],
          "In the two-stream model the discriminant subspaces are not the streams", doc=cap)
    c1 = np.mean([x["E1"]["mean_curve"] for x in trc], 0)
    c0 = np.mean([x["mean_curve"] for x in fam["seeds"]], 0)
    ws = np.round(np.linspace(0, 1, 21), 10)
    m = (ws >= fam["gate"][0] - 1e-9) & (ws <= fam["gate"][1] + 1e-9)
    en = [x["descriptives"]["energy_S_on_stream1_full"] for x in fam["seeds"]]
    w, S = sweep(MODEL)
    r = S.mean(0)
    p = ("In the two-stream model the discriminant subspaces are not the streams (58–62% of the shape basis's energy lies on "
         "stream 1) and the subspace gain moves the rate by 0.20 with exemplars held out (mean over repetitions and seeds; the "
         "seed-mean curve runs 0.57 → 0.77, 0.16 within the gate) and by 0.29 with exemplars shared (0.52 → 0.80; 0.25 within "
         "the gate), where the streams themselves, which the unit-level rule recovers, give 0.38 and the readout weight of "
         "Fig. 4a 0.37.")
    for lab, v, lit in (("shape basis on stream 1, lowest seed (%)", 100 * min(en), "58"),
                        ("shape basis on stream 1, highest seed (%)", 100 * max(en), "62"),
                        ("held out, mean span", np.mean([x["E1"]["span_mean"] for x in trc]), "0.20"),
                        ("held out, seed-mean curve at w = 0", c1[0], "0.57"), ("held out, seed-mean curve at w = 1", c1[-1], "0.77"),
                        ("held out, within the gate", c1[m].max() - c1[m].min(), "0.16"),
                        ("shared, span", fam["span"], "0.29"), ("shared, w = 0", c0[0], "0.52"), ("shared, w = 1", c0[-1], "0.80"),
                        ("shared, within the gate", fam["gated_span"], "0.25"), ("unit-level rule", gen["span"], "0.38"),
                        ("readout weight of Fig. 4a", r.max() - r.min(), "0.37")):
        said(f"Fig. 6 caption: {lab}", v, p, lit, doc=cap)
    R = {k: float(np.nanmean(v)) for k, v in json.load(open("results/figures_v3/fig6_model_reference.json"))["families"]["two_stream_model"]["index"].items()}
    ext = [R[k] for k in ("ventral", "lateral", "dorsal")]
    p = "voxels pooled per region: +0.51 in V1, −0.76 in hV4 and −0.80 to −0.85 beyond it."
    said("Fig. 6 caption: reference, V1", R["V1"], p, "+0.51", doc=cap); said("Fig. 6 caption: reference, hV4", R["hV4"], p, "−0.76", doc=cap)
    said("Fig. 6 caption: reference beyond hV4, highest", max(ext), p, "−0.80", doc=cap)
    said("Fig. 6 caption: reference beyond hV4, lowest", min(ext), p, "−0.85", doc=cap)

    # the wordings the pass removed
    for where, doc, gone in (("PAPER_v3", PAPER, ("twice the resampling noise", "at no cost in accuracy", "IPS0–3", "Table 11",
                                                  "mpk-rewrite", "zoo summaries do not change", "best predicting extrastriate cortex (0.34)")),
                             ("Fig. 6 caption", cap, ("Table 14", "63–70%", "two-stream diet model's"))):
        for w_ in gone:
            ok = w_ not in doc
            print(f"  {'OK ' if ok else 'BAD'} {where}: “{w_}” stays out")
            if not ok:
                FAILS.append(f"{where}: “{w_}” is back")


def check_fig6_reference_recompute() -> None:
    """Fig. 6c's reference: the model's magno − parvo profile recomputed from the per-voxel fits by
    tmp_analysis/figs_v2/fig6_model_reference.py, against the committed file."""
    print("\n[Fig. 6c] the model's reference profile, recomputed (FS2_ENCODING_DIR)")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_v2"))
    import fig6_model_reference as f6  # noqa: E402
    got = f6.profile(ENC)
    ref = json.load(open("results/figures_v3/fig6_model_reference.json"))["families"]["two_stream_model"]
    ok = got["subjects"] == ref["subjects"] and all(np.allclose(got["index"][r], ref["index"][r], atol=1e-12, equal_nan=True)
                                                    for r in ref["index"])
    print(f"  {'OK ' if ok else 'BAD'} per-subject index in 7 regions x {len(ref['subjects'])} subjects matches the committed file")
    if not ok:
        FAILS.append("Fig. 6c reference recompute")


def _external(fn, key: str) -> None:
    path, what = EXTERNAL[key]
    if not os.path.isdir(path):
        print(f"\n[SKIP] {fn.__name__}: needs {what}")
        SKIPS.append((fn.__name__, what))
        return
    fn()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ba", action="store_true", help="skip the BA comparison block (tmp_analysis/verify_new_claims.py)")
    args = ap.parse_args()
    check_accuracy()
    check_streams()
    check_cortex()
    _external(check_encoding, "encoding")
    check_nuisance()
    check_estimator_bias()
    _external(check_nuisance_recompute, "encoding")
    check_behaviour()
    check_readout_w_summary()
    _external(check_readout_w, "readout_w")
    check_dissociations()
    check_ed_table3()
    check_review_v9()
    _external(check_fig6_reference_recompute, "encoding")
    if SKIPS:
        print(f"\nSKIPPED {len(SKIPS)} blocks for inputs not in the repository:")
        for fn, need in SKIPS:
            print(f"  {fn}: {need}")
    print("\nCarried from PAPER_v2, not re-derived here or in the BA block:")
    for c in CARRIED:
        print(f"  {c}")
    if LANDED:
        print(f"\nLANDED, the text still marks pending: {LANDED}")
    print("\nv3 FAILURES:", FAILS if FAILS else "none")
    rc = 1 if FAILS else 0
    if not args.no_ba:
        print("\n==== comparison block: the BA network and the zoo (tmp_analysis/verify_new_claims.py) ====")
        rc_ba = ba.main()
        print(f"\nv3: {len(FAILS)} failures; BA comparison block: {'pass' if rc_ba == 0 else 'FAIL'}")
        rc |= rc_ba
    return rc


if __name__ == "__main__":
    sys.exit(main())
