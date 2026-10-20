#!/usr/bin/env python
"""Re-derive every number docs/PAPER_v3.md states about its model of record and that model's controls, straight
from the result files, then run tmp_analysis/verify_new_claims.py as the comparison block for the BA network and
the model zoo, whose numbers PAPER_v3 carries over from PAPER_v2.

Model of record: fs2_dvd300_avg_w050_ls, seeds 0-2. Control: fs2_dvdfix300_avg_w050_ls, the adult diet from the
first step (the text reports seed 0 while seeds 1-2 train; CONTROL_SEEDS). Fixed BA diet under the model's recipe:
fs2_bcs_avg_w050_ls.

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
from scipy.stats import linregress, t as tdist, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verify_new_claims as ba  # noqa: E402  (tmp_analysis/ is on sys.path when run as a script)

PAPER = open("docs/PAPER_v3.md", encoding="utf8").read()
MODEL = [f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)]
CONTROL_SEEDS = (0,)  # the text reports the control's seed 0 (⟦PENDING seeds 1–2⟧); widen together with the text
CONTROL = [f"fs2_dvdfix300_avg_w050_ls_seed{s}" for s in CONTROL_SEEDS]
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
           "AlexNet, GoogLeNet, VGG-16 rates and κ in the seven-model fit (Geirhos et al. 2018a; PAPER_v2 Supplementary Table S6)"]
# Rate and κ of the three published CNNs in the seven-model fit, from the Geirhos et al. (2018a) response files.
PUBLISHED_CNNS = {"AlexNet": (0.429, 0.032), "GoogLeNet": (0.312, 0.025), "VGG-16": (0.172, 0.017)}

FAILS: list[str] = []
SKIPS: list[tuple[str, str]] = []
LANDED: list[str] = []
NUM = re.compile(r"[−+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


def _val(lit: str) -> float:
    return float(lit.replace("−", "-").replace(",", ""))


def said(name: str, got: float, phrase: str, lit: str | None = None, tol: float | None = None) -> None:
    """`phrase` is copied from PAPER_v3 and must still be in it; `lit` (default: the phrase's last number) is the
    printed value, and `got` must round to it."""
    lit = NUM.findall(phrase)[-1] if lit is None else lit
    dec = len(lit.split(".")[1]) if "." in lit else 0
    tol = 0.5 * 10 ** -dec + 1e-9 if tol is None else tol
    found = phrase in PAPER and lit in phrase
    ok = found and abs(float(got) - _val(lit)) <= tol
    print(f"  {'OK ' if ok else 'BAD'} {name:50s} got {float(got):+.4f}  text {lit}" + ("" if found else "  <- phrase not in PAPER_v3"))
    if not ok:
        FAILS.append(name)


def holds(name: str, ok: bool, phrase: str) -> None:
    found = phrase in PAPER
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
    said("control top-1", c.mean(), "against 0.761 for the control on the identical test input")
    p = "The fixed BA diet reaches 0.665 ± 0.003 under the same recipe"
    said("fixed diet, model's recipe, top-1", f.mean(), p, "0.665")
    said("fixed diet, model's recipe, top-1 s.d.", f.std(ddof=1), p, "0.003")
    p = ("The trajectory therefore adds 1.8 points and 0.067 of shape bias (training-time evaluation) over the fixed diet "
         "under one recipe, and gives up 7.8 points")
    said("model minus fixed diet, top-1 points", 100 * (m.mean() - f.mean()), p, "1.8")
    said("model minus fixed diet, training-time shape bias", sb(MODEL).mean() - sb(FIXED).mean(), p, "0.067")
    said("control minus model, top-1 points", 100 * (c.mean() - m.mean()), p, "7.8")
    said("same, Discussion", 100 * (c.mean() - m.mean()), "The trajectory costs less: 7.8 points of top-1 against its control", "7.8")
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
    missing = [r for r in MODEL + CONTROL + [UNTRAINED] if r not in EX]
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
        "| control, adult input from the first step: linear probe ⟦PENDING seeds 1–2; seed 0 shown⟧ | **−0.180** / −0.288 | −0.079 / −0.410 | — |":
            (("control magno held out", xm, "−0.180"), ("control magno shared", xsm, "−0.288"), ("control parvo held out", xp, "−0.079"), ("control parvo shared", xsp, "−0.410")),
        "| control: cue-conflict shape bias ⟦PENDING seeds 1–2; seed 0 shown⟧ | 0.378 | 0.388 | 0.372 |":
            (("control cue-conflict magno", xcm, "0.378"), ("control cue-conflict parvo", xcp, "0.388"), ("control cue-conflict integrated", xcf, "0.372")),
    }
    for row, cells in rows.items():
        for lab, v, lit in cells:
            said(f"Table 1: {lab}", v, row, lit)

    said("untrained magno, held out", um, "(+0.087 against +0.090 on held-out exemplars)", "+0.087")
    said("untrained no-diet luminance, held out", q("untrained_noblur", "magno"), "(+0.087 against +0.090 on held-out exemplars)", "+0.090")
    said("untrained parvo, held out", up, "and the parvo stream begins nearly unbiased (−0.033)")
    said("learning on the maturing input, magno", hm.mean() - um, "Training on the maturing input adds +0.229 to the magno stream")
    said("learning on the maturing input, parvo", hp.mean() - up, "— and −0.123 to the parvo stream")
    p = "across the sign flip to −0.180, below its own parvo stream (−0.079), and the integrated readout to 0.372"
    said("control magno, held out", xm, p, "−0.180"); said("control parvo, held out", xp, p, "−0.079")
    said("control integrated cue-conflict", xcf, p, "0.372")
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
    p = "(426 to 580 of 1,200 in the model, 224 to 675 in the control)"
    said("decided images, model, fewest (readout, seed mean)", min(dm), p, "426"); said("decided images, model, most", max(dm), p, "580")
    said("decided images, control, fewest", min(dc), p, "224"); said("decided images, control, most", max(dc), p, "675")
    pat_m = "results/cue_conflict/fs2_dvd300_avg_w050_ls_seed[012]_logits.npz"
    pat_c = f"results/cue_conflict/fs2_dvdfix300_avg_w050_ls_seed[{''.join(map(str, CONTROL_SEEDS))}]_logits.npz"
    fc = {(arm, k): ba._logit_estimators(pat, k)[0].mean() for arm, pat in (("model", pat_m), ("control", pat_c)) for k in conds}
    p = "(forced estimator: magno 0.730, integrated 0.648, parvo 0.520; control 0.394, 0.393, 0.478)"
    for (arm, k), lit in zip([("model", "magno_only"), ("model", "full"), ("model", "parvo_only"),
                              ("control", "magno_only"), ("control", "full"), ("control", "parvo_only")],
                             ["0.730", "0.648", "0.520", "0.394", "0.393", "0.478"]):
        said(f"forced estimator, {arm} {k}", fc[(arm, k)], p, lit)
    a, b = cp.mean() - 0.5, fc[("model", "parvo_only")] - 0.5
    holds(f"parvo within 0.03 of parity on both estimators, opposite sides ({a:+.3f}, {b:+.3f})",
          max(abs(a), abs(b)) <= 0.03 and a * b < 0,
          "but the parvo stream sits within 0.03 of parity on both estimators and on opposite sides of it")

    # training-time evaluation, magno stream through the shared head, seed 0 of each
    R, C = hist(MODEL[0]), hist(CONTROL[0])
    ms, xs = [r["cc"]["tied_magno"]["sb"] for r in R], [r["cc"]["tied_magno"]["sb"] for r in C]
    p = "(shape bias 0.72–0.92 against 0.32–0.45; training-time evaluation, seed 0)"
    said("model magno, training-time, lowest epoch", min(ms), p, "0.72"); said("model magno, training-time, highest epoch", max(ms), p, "0.92")
    said("control magno, training-time, lowest epoch", min(xs), p, "0.32"); said("control magno, training-time, highest epoch", max(xs), p, "0.45")
    holds(f"the two magno streams on opposite sides of parity at all {len(R)} epochs", min(ms) > 0.5 > max(xs) and len(R) == len(C) == 50,
          "The two magno streams sit on opposite sides of parity at every epoch evaluated during training")
    p = "so the bias is present from the first evaluation, after one epoch at ages up to 6 months (top-1 0.09)"
    said("diet age at the end of the first epoch (months)", R[0]["dvd_age"], p, "6")
    said("model top-1 after the first epoch", R[0]["val_top1"], p, "0.09")
    holds(f"model magno above parity after the first epoch ({ms[0]:.3f})", ms[0] > 0.5, p)
    m_top = float(np.mean([hist(r)[-1]["val_top1"] for r in MODEL]))
    r = min(C, key=lambda r: abs(r["val_top1"] - m_top))
    # history epochs are 0-based; the text counts from 1
    said("control epoch whose top-1 matches the model's", r["epoch"] + 1, "matches the model's final 0.683 (epoch 7 of 50)", "7")
    said("control magno shape bias at that epoch", r["cc"]["tied_magno"]["sb"], "its magno stream's shape bias is 0.41")
    said("control magno shape bias, highest epoch", max(xs), "and at no epoch does it exceed 0.45")

    t = S.get("two_by_two_v3", {}).get("Q1", {})
    if t:
        mm = t["means"]
        p = ("achromatic with the ramp (+0.316), achromatic with the adult input from the start (−0.180) and chromatic "
             "without the ramp (−0.156 in the model, −0.079 in the control)")
        said("2x2 cell: achromatic, ramp", mm["achromatic_transform"], p, "+0.316")
        said("2x2 cell: achromatic, adult input", mm["achromatic_none"], p, "−0.180")
        said("2x2 cell: chromatic, no ramp (control)", mm["chromatic_none"], p, "−0.079")
    else:
        print("  BAD exemplar_streams.json has no two_by_two_v3: run slurm/exemplar_aggregate.sh")
        FAILS.append("exemplar_streams.json lacks two_by_two_v3")
    pending("carrier x ramp 2x2 (swap seeds 0-2)", t.get("interaction") is not None,
            "⟦PENDING `fs2_dvd300swap`, 3 seeds: the cell and the carrier × ramp interaction⟧")


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
    said("control profile, V1v", c["profile"][0], "(+0.83 in V1v, +0.76 in hV4)", "+0.83")
    said("control profile, hV4", c["profile"][6], "(+0.83 in V1v, +0.76 in hV4)", "+0.76")
    holds("control magno-explained in every ROI", min(c["profile"]) > 0, "is magno-explained everywhere and nearly flat")
    said("BA no-diet profile, V1v", nb["profile"][0], "as is the BA no-diet network (+0.91, +0.83)", "+0.91")
    said("BA no-diet profile, hV4", nb["profile"][6], "as is the BA no-diet network (+0.91, +0.83)", "+0.83")
    said("BA profile, V1v", b["profile"][0], "descends without crossing (−0.00 to −0.84)", "−0.00")
    said("BA profile, hV4", b["profile"][6], "descends without crossing (−0.00 to −0.84)", "−0.84")
    holds("BA profile never above zero", max(b["profile"]) <= 0.005, "descends without crossing (−0.00 to −0.84)")

    g, gb = np.array(m["grad_per_seed"]), np.array(b["grad_per_seed"])
    p = "the gradient is −1.259 to −1.291 across the three seeds against −0.062 for the control"
    said("model gradient, shallowest seed", g.max(), p, "−1.259"); said("model gradient, steepest seed", g.min(), p, "−1.291")
    said("control gradient", np.mean(c["grad_per_seed"]), p, "−0.062")
    d = np.array(P["dvd300 - control"]["grad"])
    holds(f"model below control, {(d < 0).sum()}/{len(d)} subjects, p {p_exact(d):.4f}", (d < 0).all() and len(d) == 8 and p_exact(d) < 0.0079,
          "each comparison at the exact Wilcoxon floor for eight subjects (p = .0078, 8/8)")
    p = "It is steeper than the BA network's −0.745 to −0.833 in 8/8 subjects (mean Δ −0.48, p = .0078)"
    said("BA gradient, shallowest seed", gb.max(), p, "−0.745"); said("BA gradient, steepest seed", gb.min(), p, "−0.833")
    d = np.array(P["dvd300 - BA diet"]["grad"])
    said("model minus BA, mean over subjects", d.mean(), p, "−0.48")
    holds(f"model steeper than BA, {(d < 0).sum()}/{len(d)}, p {p_exact(d):.4f}", (d < 0).all() and len(d) == 8 and p_exact(d) < 0.0079, p)
    rho = np.array(m["rho_per_seed"])
    p = "is −0.964 to −0.973 in every seed, against −0.29 for the control"
    said("model rho, weakest seed", rho.max(), p, "−0.964"); said("model rho, strongest seed", rho.min(), p, "−0.973")
    said("control rho", np.mean(c["rho_per_seed"]), p, "−0.29")
    said("model gradient, mean of seeds", g.mean(), "steepens it on the normalised index (to −1.27)")
    said("control gradient (Results 2.3)", np.mean(c["grad_per_seed"]), "erases it on both (−0.062; −0.003)", "−0.062")
    ed1 = ["| normalised | 0 | −1.259 | −0.062 | −0.833 | 8/8 | 0.0078 |", "|  | 1 | −1.291 | ⟦PENDING⟧ | −0.785 | 8/8 | 0.0078 |",
           "|  | 2 | −1.262 | ⟦PENDING⟧ | −0.745 | 8/8 | 0.0078 |"]
    for s, row in enumerate(ed1):
        cells = [x.strip() for x in row.strip("|").split("|")]
        said(f"ED T1 normalised, seed {s}: model", g[s], row, cells[2])
        said(f"ED T1 normalised, seed {s}: BA", gb[s], row, cells[4])
        if s in CONTROL_SEEDS:
            said(f"ED T1 normalised, seed {s}: control", c["grad_per_seed"][CONTROL_SEEDS.index(s)], row, cells[3])

    v, vc, vb = np.array(m["cc_per_seed"]), np.mean(c["cc_per_seed"]), np.array(b["cc_per_seed"])
    p = "give a predicted magno − parvo shape-bias difference of +1.32 to +1.34 in the model against +0.08 for the control (8/8 subjects, seed means"
    said("cue-conflict prediction, model, lowest seed", v.min(), p, "+1.32"); said("cue-conflict prediction, model, highest seed", v.max(), p, "+1.34")
    said("cue-conflict prediction, control", vc, p, "+0.08")
    d = np.array(P["dvd300 - control"]["cc"])
    holds(f"cue-conflict prediction above the control in {(d > 0).sum()}/{len(d)} subjects", (d > 0).all() and len(d) == 8, p)
    p = "It is smaller than the BA network's +1.53 to +1.57 (8/8)"
    said("cue-conflict prediction, BA, lowest seed", vb.min(), p, "+1.53"); said("cue-conflict prediction, BA, highest seed", vb.max(), p, "+1.57")
    d = np.array(P["dvd300 - BA diet"]["cc"])
    holds(f"cue-conflict prediction below the BA network's in {(d < 0).sum()}/{len(d)} subjects", (d < 0).all() and len(d) == 8, p)

    at = {k.split("(")[0].split(" ")[0]: np.array(x) for k, x in m["atlas"].items()}
    p = ("relative to V1, hV4 sits at −1.12 and every extrastriate group lower — ventral (VO, PHC) −1.25, lateral (LO, TO) −1.27, "
         "dorsal (V3A/B, IPS0–3) −1.22 — each in 8/8 subjects (p = .0078)")
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
    p = "(noise-ceiling-normalised median r² 0.471 against 0.496, 8/8 subjects)"
    said("r²/NC, model", rm, p, "0.471"); said("r²/NC, control", rc, p, "0.496")
    d = np.array(P["dvd300 - control"]["r2nc"])
    holds(f"r²/NC below the control in {(d < 0).sum()}/{len(d)} subjects", (d < 0).all() and len(d) == 8, p)
    p = "although the model predicts BOLD better than the BA network does (0.395, 8/8; Fig. 3e)"
    said("r²/NC, BA", rb, p, "0.395")
    d = np.array(P["dvd300 - BA diet"]["r2nc"])
    holds(f"r²/NC above the BA network in {(d > 0).sum()}/{len(d)} subjects", (d > 0).all() and len(d) == 8, p)
    said("control minus model, r²/NC", rc - rm, "and 0.026 of noise-ceiling-normalised r²")
    said("fixed diet, model's recipe, gradient", np.mean(fx["grad_per_seed"]), "(0.665 top-1, shape bias 0.620, gradient −0.735)", "−0.735")
    pending("nuisance-axis controls on the model (ED Fig. 3)", False, "these controls have not yet been rerun on the model ⟦PENDING⟧")


def check_encoding() -> None:
    """Results 2.3 and Extended Data Table 1 from the per-voxel fits: the gradient per seed on both statistics, the
    paired separation from the control, the untrained network, and the V1 / hV4 unique-variance levels."""
    from src.analysis.encoding.stats import load_results
    print(f"\n[2.3, ED Table 1] per-voxel fits ({ENC})")
    grad = {}
    for stat, ratio in (("normalised", True), ("raw", False)):
        for tag in MODEL + CONTROL + [f"{UNTRAINED}_seed0"]:
            subs, prof = ba.roi_profile(ENC, tag, ratio)
            grad[(stat, tag)] = dict(zip(subs, [ba.hv4_minus_v1(r) for r in prof]))
    ctl = grad[("normalised", CONTROL[0])], grad[("raw", CONTROL[0])]
    ed1 = {("normalised", 0): "| normalised | 0 | −1.259 | −0.062 | −0.833 | 8/8 | 0.0078 |",
           ("normalised", 1): "|  | 1 | −1.291 | ⟦PENDING⟧ | −0.785 | 8/8 | 0.0078 |",
           ("normalised", 2): "|  | 2 | −1.262 | ⟦PENDING⟧ | −0.745 | 8/8 | 0.0078 |",
           ("raw", 0): "| raw | 0 | −0.186 | −0.003 | −0.166 | 8/8 | 0.0078 |",
           ("raw", 1): "|  | 1 | −0.193 | ⟦PENDING⟧ | −0.145 | 8/8 | 0.0078 |",
           ("raw", 2): "|  | 2 | −0.190 | ⟦PENDING⟧ | −0.153 | 8/8 | 0.0078 |"}
    for (stat, s), row in ed1.items():
        cells = [x.strip() for x in row.strip("|").split("|")]
        d, x = grad[(stat, MODEL[s])], ctl[stat == "raw"]
        diff = np.array([d[k] - x[k] for k in sorted(set(d) & set(x))])
        said(f"ED T1 {stat}, seed {s}: model", np.mean(list(d.values())), row, cells[2])
        if s in CONTROL_SEEDS:
            said(f"ED T1 {stat}, seed {s}: control", np.mean(list(x.values())), row, cells[3])
        said(f"ED T1 {stat}, seed {s}: subjects below the control", (diff < 0).sum(), row, "8")
        said(f"ED T1 {stat}, seed {s}: p", p_exact(diff), row, cells[6])
    raw = np.array([np.mean(list(grad[("raw", t)].values())) for t in MODEL])
    p = "with the same separation on the raw statistic U_m − U_p (−0.186 to −0.193 against −0.003)"
    said("raw gradient, model, shallowest seed", raw.max(), p, "−0.186"); said("raw gradient, model, steepest seed", raw.min(), p, "−0.193")
    said("raw gradient, control", np.mean(list(ctl[1].values())), p, "−0.003")
    said("raw gradient, model, mean of seeds", raw.mean(), "and holds it on the raw difference (−0.19)")
    said("raw gradient, control (Results 2.3)", np.mean(list(ctl[1].values())), "erases it on both (−0.062; −0.003)", "−0.003")
    row = "| control alone, normalised / raw | 0 | | −0.062 / −0.003 | | 5/8 / 4/8 < 0 | 0.25 / 0.84 |"
    for stat, lit_m, lit_n, lit_p in (("normalised", "−0.062", "5", "0.25"), ("raw", "−0.003", "4", "0.84")):
        x = np.array(list(ctl[stat == "raw"].values()))
        said(f"control alone, {stat}: mean", x.mean(), row, lit_m)
        said(f"control alone, {stat}: subjects below zero", (x < 0).sum(), row, lit_n)
        said(f"control alone, {stat}: p", p_exact(x), row, lit_p)
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
    said("control integrated cue-conflict", np.mean([cc(r) for r in CONTROL]), "against 0.372 for the control, 0.958 for the ten human observers", "0.372")
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
    said("control sweep, lowest", rc.min(), "The control's curve is flat (0.37–0.39)", "0.37")
    said("control sweep, highest", rc.max(), "The control's curve is flat (0.37–0.39)", "0.39")

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
    for arm, pre in (("dvd300", "model"), ("control", "control")):
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
            p = "the weight best predicting extrastriate cortex (0.34) lies below the one best predicting V1 (0.65)"
            said("Abstract: extrastriate w*", np.median(R["w_ext"]), p, "0.34"); said("Abstract: V1 w*", np.median(R["w_v1"]), p, "0.65")
        else:
            p = "reversed in the control (+0.248 [+0.052, +0.440], positive in 6/8 subjects and tied in one, p = .078"
            said("control w*: extrastriate minus V1", d.mean(), p, "+0.248")
            said("control w*: bootstrap 2.5%", lo, p, "+0.052"); said("control w*: bootstrap 97.5%", hi, p, "+0.440")
            said("control w*: subjects positive", (d > 1e-9).sum(), p, "6")
            holds(f"control w*: one subject tied ({(np.abs(d) < 1e-9).sum()})", (np.abs(d) < 1e-9).sum() == 1, p)
            said("control w*: p (exact, tie dropped)", p_exact(d), p, ".078")
            p = "In the control the gradient reverses (0.43 in V1 to 0.60–0.95 beyond hV4; higher in 6/8 subjects and tied in one, p = .078"
            said("control w*: V1 median", np.median(R["w_v1"]), p, "0.43")
            said("control w*: subjects higher (Results 2.4)", (d > 1e-9).sum(), p, "6")


def check_readout_w() -> None:
    """Extended Data Table 2 and Results 2.4 from the per-voxel readout-weight fits, reduced by the rule of
    tmp_analysis/figs_v2/readout_w_cortex.py; every cell of the model and control rows."""
    print(f"\n[2.4, ED Table 2] readout-weight fits ({RW})")
    VIS = {"V1": [1, 2], "V2": [3, 4], "V3": [5, 6], "hV4": [7]}
    KAST = {"ventral (VO, PHC)": [8, 9, 10, 11], "lateral (LO, TO)": [12, 13, 14, 15], "dorsal (V3A/B, IPS0–2)": [16, 17, 18, 19, 20]}
    arms = {"model": MODEL, "control (seed 0)": CONTROL}
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
    half_model = []
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
    # at the table's printed precision
    holds(f"every model ROI within 2% of its best r² at w = 0.5 (lowest {min(half_model):.4f})", round(min(half_model), 3) >= 0.98,
          "every ROI is within 2% of its best r² at w = 0.5")


def check_dissociations() -> None:
    """Results 2.5 and Discussion: OOD robustness against the control, error consistency, participation ratio."""
    print("\n[2.5] robustness, agreement with observers, dimensionality")

    def rel_auc(runs, fam):
        acc = np.array([json.load(open(f"results/ood/{x}.json"))["families"][fam]["accuracy"] for x in runs], float)
        return (acc / acc[:, :1]).mean(1)

    four = "uniform noise (−0.138), high-pass (−0.143), false colour (−0.084) and rotation (−0.041), every seed below the control"
    for fam, lit, p in (("uniform-noise", "−0.138", four), ("high-pass", "−0.143", four), ("false-colour", "−0.084", four),
                        ("rotation", "−0.041", four), ("phase-scrambling", "+0.010", "within 0.01 of it on phase scrambling (+0.010)"),
                        ("low-pass", "+0.078", "and more robust only to low-pass (+0.078)"), ("contrast", "−0.002", "is level (−0.002)")):
        said(f"OOD relative AUC, {fam}: model minus control", rel_auc(MODEL, fam).mean() - rel_auc(CONTROL, fam).mean(), p, lit)
    holds("every model seed below the control on the four",
          all(rel_auc(MODEL, f).max() < rel_auc(CONTROL, f).min() for f in ("uniform-noise", "high-pass", "false-colour", "rotation")), four)

    k = lambda runs, w: np.array([kappa_at(x, w) for x in runs])
    ceil = kappa(MODEL[0])["human_human_ceiling"]
    said("human-human κ ceiling", ceil, "the human–human ceiling is κ = 0.234")
    p = ("at 17.3% of the ceiling, above its integrated readout (13.2%), its parvo stream (10.3%) and the control "
         "(10.3% integrated, 10.8% magno)")
    said("κ, model magno (% ceiling)", k(MODEL, 1.0).mean(), p, "17.3"); said("κ, model integrated", k(MODEL, 0.5).mean(), p, "13.2")
    said("κ, model parvo", k(MODEL, 0.0).mean(), p, "10.3"); said("κ, control integrated", k(CONTROL, 0.5).mean(), p, "10.3")
    said("κ, control magno", k(CONTROL, 1.0).mean(), p, "10.8")
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
    p = "κ rises with the rate by 0.020 [0.013, 0.028] (90% interval) over the 0.68 span of shape bias they cover, 8.6% of the ceiling"
    said("seven models: κ change over the span", res.slope * span, p, "0.020")
    said("seven models: 90% interval, low", (res.slope - tq * res.stderr) * span, p, "0.013")
    said("seven models: 90% interval, high", (res.slope + tq * res.stderr) * span, p, "0.028")
    said("seven models: shape-bias span", span, p, "0.68"); said("seven models: change, % of ceiling", 100 * res.slope * span / ceil, p, "8.6")
    p = "the three magno streams sit at 0.871, 0.842 and 0.838 and reach 5.1%, 25.6% and 21.3% of the ceiling"
    for x, lsb, lk in zip(MODEL, ("0.871", "0.842", "0.838"), ("5.1", "25.6", "21.3")):
        said(f"{x[-5:]} magno: cue-conflict", cc(x, "magno_only"), p, lsb); said(f"{x[-5:]} magno: κ (% ceiling)", kappa_at(x, 1.0), p, lk)
    said("κ integrated, model (prediction)", k(MODEL, 0.5).mean(), "now points the right way (13.2% against 10.3%)", "13.2")
    said("κ integrated, control (prediction)", k(CONTROL, 0.5).mean(), "now points the right way (13.2% against 10.3%)", "10.3")
    said("κ integrated, lowest model seed", k(MODEL, 0.5).min(), "and a seed spread of 9.4–17.7%", "9.4")
    said("κ integrated, highest model seed", k(MODEL, 0.5).max(), "and a seed spread of 9.4–17.7%", "17.7")
    pending("per-image correspondence on the model", False, "⟦PENDING per-image correspondence on the model⟧")

    PR = json.load(open("results/figures_v2/fs2_pr_ratio.json"))["runs"]
    rat = lambda run: np.array([v["ratio"] for v in PR[run].values()])
    allm = np.concatenate([rat(x) for x in MODEL])
    seedm = [rat(x).mean() for x in MODEL]
    ctl = np.concatenate([rat(x) for x in CONTROL])
    p = ("ends above parity in every subject and seed of the model (24/24; seed means 1.88–1.98) and below it in every subject "
         "of its control (0.89; 0/8), from the same untrained starting point (1.63)")
    said("PR ratio: model subject-seeds above parity", (allm > 1).sum(), p, "24")
    holds(f"PR ratio: {len(allm)} model subject-seeds", len(allm) == 24, p)
    said("PR ratio: model seed mean, lowest", min(seedm), p, "1.88"); said("PR ratio: model seed mean, highest", max(seedm), p, "1.98")
    said("PR ratio: control mean", ctl.mean(), p, "0.89"); said("PR ratio: control subjects above parity", (ctl > 1).sum(), p, "0")
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
    p = "| 0.202 [0.17, 0.23] | ⟦PENDING⟧ | — (shape basis on stream 1: ⟦PENDING⟧) | 0.181 |"
    said("ED T3 model: forced-estimator span", np.mean([T[t]["spans"]["E0"]["forced_span"] for t in tags]), p, "0.181")
    landed = "concat_fs2_dvd300" in open("results/figures_v2/subspace_behaviour.json").read()
    pending("ED T3 model: ρ(w, rate), stream-1 energy, variance in S / T, order sensitivity", landed, p)


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
    check_behaviour()
    check_readout_w_summary()
    _external(check_readout_w, "readout_w")
    check_dissociations()
    check_ed_table3()
    pending("control seeds 1-2", any(os.path.exists(f"results/cue_conflict/fs2_dvdfix300_avg_w050_ls_seed{s}.json") for s in (1, 2)),
            "⟦PENDING control seeds 1–2⟧")
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
