"""Figure 2 — the two streams specialise in opposite directions, by learning, and the
magno stream's shape bias only with a maturing input.

  (a) linear-probe shape−texture bias per stream before and after training, on
      exemplars the probe was not trained on (Q1 of docs/PLAN_EXEMPLAR_2026-09-21.md;
      the exemplar-shared probe is in Table 1): the model and its control start from
      one untrained network on one test input; the BA network from its own fixed diet.
  (b, c) carrier × ramp on the probe (b) and on cue-conflict behaviour (c): achromatic
      with the ramp = the model's magno stream, without = the control's; chromatic
      without = the control's parvo stream (the model's, beside a ramped magno stream,
      as an open marker); chromatic with the ramp = the channel swap fs2_dvd300swap,
      drawn when its results exist.
  (d) how each readout decides the cue-conflict images: shape / texture / neither.
  (e) training-time evaluation (the shared head on the magno stream alone, seed 0):
      the two magno streams' shape bias by epoch, and (f) against the network's
      validation top-1 — the control at the model's final accuracy is the
      accuracy-matched comparison.

Sources: results/figures_v2/exemplar_streams.json (probe), results/cue_conflict/
(behaviour), run histories (e, f; tmp_analysis/figs_v2/v3runs.py).
Usage: PYTHONPATH=.:tmp_analysis/figs_v2 python tmp_analysis/figs_v2/fig2_streams.py
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import style as S
import v3runs as R

EX = json.load(open("results/figures_v2/exemplar_streams.json"))
EST = "Q1"   # exemplar-balanced, leave-one-exemplar-out probe (Methods 4.3)
UNTRAINED = {"model": "untrained_dvd300", "control": "untrained_dvd300", "BA": "untrained_blur_cs"}
NETS = [("model", R.MODEL, S.MODEL), ("control", R.CONTROL, S.CONTROL), ("BA", R.BA + "_trained", S.BA)]
STREAM_LS = {"magno": "-", "parvo": (0, (3, 2))}


def q1(name, stream):
    """Held-out probe bias, or None while the network is not in exemplar_streams.json."""
    s = EX["streams"].get(name)
    return None if s is None else s[stream][EST]["bias"]


def q1_arm(tmpl, stream):
    v = [q1(tmpl.format(i), stream) for i in range(3)]
    return np.array([x for x in v if x is not None])


def cc_arm(tmpl, cond, key="shape_bias", root="results/cue_conflict"):
    return np.array([R.cc(r, cond, key, root) for r in R.runs(tmpl, f"{root}/{{}}.json")])


def training_eval(run):
    h = R.history(run)
    return (np.array([e["epoch"] + 1 for e in h]), np.array([e["val_top1"] for e in h]),
            np.array([e["cc"]["tied_magno"]["sb"] for e in h]))


def panel_routes(ax):
    S.refline(ax, y=0)
    ends = []
    for net, tmpl, c in NETS:
        for st in ("magno", "parvo"):
            u, t = q1(UNTRAINED[net], st), q1_arm(tmpl, st)
            if u is None or not len(t):
                print(f"  (a) {net} {st}: not in exemplar_streams.json yet")
                continue
            ax.plot([0, 1], [u, t.mean()], ls=STREAM_LS[st], color=c, lw=1.8, zorder=3)
            ax.plot([0], [u], "o", color=c, ms=4.2, mec="white", mew=0.8, zorder=4)
            ax.plot([1, 1], [t.min(), t.max()], "-", color=c, lw=3.5, alpha=0.30, zorder=2, solid_capstyle="butt")
            ax.plot([1], [t.mean()], "o", color=c, ms=4.5, mec="white", mew=0.8, zorder=4)
            ends.append([t.mean(), f"{net} {st}  {t.mean():+.2f}".replace("-", "−"), c])
            print(f"  (a) {net:8s} {st}: untrained {u:+.3f} -> trained {t.mean():+.3f} (n={len(t)}, {t.min():+.3f}..{t.max():+.3f})")
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):                       # push labels apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06)
    for y, txt, c in ends:
        ax.annotate(txt, (1, y), xytext=(7, 0), textcoords="offset points", fontsize=5.9, color=S.INK, va="center")
        ax.plot([1.04], [y], "s", color=c, ms=2.6, mew=0, transform=ax.transData, clip_on=False)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["untrained", "trained"])
    ax.set_xlim(-0.25, 1.95); ax.set_ylim(-0.32, 0.58)
    ax.set_ylabel("probe bias, exemplars held out\nshape − texture accuracy")
    ax.text(-0.22, 0.012, "shape", fontsize=6, color=S.MUTED, va="bottom")
    ax.text(-0.22, -0.30, "texture", fontsize=6, color=S.MUTED, va="bottom")
    h = [plt.Line2D([], [], color=S.MUTED, lw=1.6, ls=STREAM_LS[s], label=f"{s} stream") for s in ("magno", "parvo")]
    ax.legend(handles=h, loc="upper left", fontsize=6.0, handlelength=1.8, bbox_to_anchor=(-0.02, 1.04), ncol=2, columnspacing=1.0)


def cells(measure):
    """carrier × ramp cells: (values, label) per (carrier, ramp); None where the run is pending."""
    if measure == "probe":     # exemplar_aggregate.py's two_by_two_v3, the checker's source
        cell = {k: np.array(v) for k, v in EX["two_by_two_v3"][EST]["cells"].items()}
        return {("achromatic", "ramp"): cell["achromatic_transform"], ("achromatic", "none"): cell["achromatic_none"],
                ("chromatic", "none"): cell["chromatic_none"], ("chromatic", "ramp"): cell["chromatic_transform"],
                ("chromatic", "beside ramp"): q1_arm(R.MODEL, "parvo")}
    get = lambda tmpl, st: cc_arm(tmpl, f"{st}_only")
    return {("achromatic", "ramp"): get(R.MODEL, "magno"), ("achromatic", "none"): get(R.CONTROL, "magno"),
            ("chromatic", "none"): get(R.CONTROL, "parvo"), ("chromatic", "ramp"): get(R.SWAP, "parvo"),
            ("chromatic", "beside ramp"): get(R.MODEL, "parvo")}


def panel_two_by_two(ax, measure, neutral, ylim, ylab):
    S.refline(ax, y=neutral)
    C = cells(measure)
    for carrier, c, ls, lab in [("chromatic", S.MUTED, (0, (3, 2)), "chromatic (L−M)"),
                                ("achromatic", S.INK, "-", "achromatic (luminance)")]:
        ys = [C[(carrier, "none")], C[(carrier, "ramp")]]
        for xi, v in zip((0, 1), ys):
            ax.plot([xi] * len(v), v, "o", color=c, ms=2.8, alpha=0.6, mec="none", zorder=3)
        m = [v.mean() if len(v) else np.nan for v in ys]
        if np.isfinite(m).all():
            ax.plot([0, 1], m, ls=ls, color=c, lw=1.8, zorder=4)
        ax.plot([x for x, v in zip((0, 1), m) if np.isfinite(v)], [v for v in m if np.isfinite(v)], "o", color=c, ms=5,
                mec="white", mew=0.9, zorder=5, label=lab)
        print(f"  ({measure}) {carrier:10s} no ramp {m[0]:+.3f}  ramp {m[1]:+.3f}")
    beside = C[("chromatic", "beside ramp")]
    if len(beside):
        ax.plot([0.12] * len(beside), beside, "o", mfc="none", mec=S.MUTED, ms=3.6, mew=0.8, zorder=4)
        ax.annotate("model's\nparvo", (0.12, beside.mean()), xytext=(5, 0), textcoords="offset points", fontsize=5.2,
                    color=S.MUTED, va="center")
    if not len(C[("chromatic", "ramp")]):
        ax.text(1, neutral, "ramp on L−M\n(swap): training", fontsize=5.4, color=S.MUTED, ha="center", va="center",
                style="italic", bbox=dict(fc="white", ec="none", pad=0.5))
    if len(C[("achromatic", "ramp")]):
        conj = C[("achromatic", "ramp")].mean()
        ax.annotate(f"{conj:+.2f}" if measure == "probe" else f"{conj:.2f}", (1, conj), xytext=(0, 7),
                    textcoords="offset points", color=S.INK, fontsize=7.0, fontweight="bold", ha="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["adult input\nfrom step 0", "maturing\n(ramp)"])
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(*ylim)
    ax.set_ylabel(ylab)


def panel_decisions(ax):
    conds = [("magno_only", "magno\nonly"), ("full", "both\nw = 0.5"), ("parvo_only", "parvo\nonly")]
    arms = [("model", R.MODEL, S.MODEL), ("control", R.CONTROL, S.CONTROL)]
    for ci, (cond, clab) in enumerate(conds):
        for ai, (alab, tmpl, c) in enumerate(arms):
            ns, nt, no = (cc_arm(tmpl, cond, k).mean() for k in ("n_shape", "n_texture", "n_other"))
            x = ci * 2.4 + ai * 1.0
            ax.bar(x, ns, width=0.85, color=c, alpha=0.95, zorder=3)
            ax.bar(x, nt, bottom=ns, width=0.85, color=c, alpha=0.40, zorder=3)
            ax.bar(x, no, bottom=ns + nt, width=0.85, color="#E4E4E4", zorder=3)
            sb = ns / (ns + nt)
            ax.text(x, 1230, f"{sb:.2f}", ha="center", va="bottom", fontsize=6.2, color=S.INK, fontweight="bold")
            print(f"  (d) {cond:10s} {alab:8s} shape {ns:5.0f} texture {nt:5.0f} neither {no:5.0f}  bias {sb:.3f}  decided {ns + nt:.0f}")
        ax.text(ci * 2.4 + 0.5, -50, clab, ha="center", va="top", fontsize=6.2, color=S.INK, linespacing=1.0)
    ax.set_xticks([])
    ax.set_ylim(0, 1950); ax.set_yticks([0, 400, 800, 1200]); ax.set_ylabel("cue-conflict images (of 1,200)")
    ax.text(-0.95, 1345, "shape bias", fontsize=5.8, color=S.MUTED, va="bottom", ha="left", style="italic")
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=lab) for lab, _t, c in arms]
    h += [plt.Rectangle((0, 0), 1, 1, color=S.MUTED, alpha=0.95, label="decided as shape"),
          plt.Rectangle((0, 0), 1, 1, color=S.MUTED, alpha=0.40, label="decided as texture"),
          plt.Rectangle((0, 0), 1, 1, color="#E4E4E4", label="neither category")]
    ax.legend(handles=h, loc="upper right", fontsize=5.5, ncol=2, handlelength=1.0, bbox_to_anchor=(1.03, 1.05),
              labelspacing=0.25, columnspacing=0.8)
    ax.set_xlim(-1.0, 6.3)


def main():
    fig = plt.figure(figsize=(S.FULL, 4.9))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.48, left=0.075, right=0.985,
                          top=0.90, bottom=0.12, width_ratios=[1.15, 1, 1])

    ax = fig.add_subplot(gs[0, 0]); panel_routes(ax)
    S.letter(ax, "a", dx=-0.06, dy=0.03, title="Both biases are learned")

    ax = fig.add_subplot(gs[0, 1])
    panel_two_by_two(ax, "probe", 0.0, (-0.30, 0.45), "probe bias, exemplars held out")
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.6, bbox_to_anchor=(-0.02, 1.02))
    S.letter(ax, "b", dx=-0.055, dy=0.03, title="Carrier × ramp: probe")
    ax = fig.add_subplot(gs[0, 2])
    panel_two_by_two(ax, "beh", 0.5, (0.25, 1.0), "cue-conflict shape bias")
    S.letter(ax, "c", dx=-0.055, dy=0.03, title="…and behaviour")

    ax = fig.add_subplot(gs[1, 0]); panel_decisions(ax)
    S.letter(ax, "d", dx=-0.06, dy=0.03, title="How each readout decides the images")

    # (e, f) training-time evaluation, seed 0 (the runs the text quotes)
    T = {net: training_eval(tmpl.format(0)) for net, tmpl, _ in NETS[:2]}
    for net, (ep, top1, sb) in T.items():
        print(f"  (e) {net:8s} seed 0 magno SB over epochs {sb.min():.2f}–{sb.max():.2f}; final top-1 {top1[-1]:.3f}")
    ax = fig.add_subplot(gs[1, 1])
    S.refline(ax, y=0.5)
    for net, _t, c in NETS[:2]:
        ep, top1, sb = T[net]
        ax.plot(ep, sb, "-", color=c, lw=1.6, zorder=3, label=net)
    sec = ax.secondary_xaxis("top", functions=(lambda e: 6 * e, lambda a: a / 6))
    sec.set_xticks([0, 150, 300]); sec.tick_params(labelsize=5.6)
    sec.set_xlabel("model's diet age (months)", fontsize=5.8, color=S.MUTED, labelpad=1.0)
    ax.set_xlim(0, 51); ax.set_ylim(0.25, 1.0)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("magno-stream shape bias\n(training-time, seed 0)")
    ax.legend(loc="center right", fontsize=6.0, handlelength=1.4)
    S.letter(ax, "e", dx=-0.055, dy=0.075, title="Opposite sides at every epoch")

    ax = fig.add_subplot(gs[1, 2])
    S.refline(ax, y=0.5)
    target = T["model"][1][-1]
    for net, _t, c in NETS[:2]:
        ep, top1, sb = T[net]
        ax.plot(top1, sb, "-", color=c, lw=1.0, alpha=0.6, zorder=2)
        ax.plot(top1, sb, "o", color=c, ms=2.2, mew=0, zorder=3)
    ep, top1, sb = T["control"]
    i = int(np.argmin(np.abs(top1 - target)))
    ax.axvline(target, color="#B8B8B8", lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.plot(top1[i], sb[i], "o", mfc="white", mec=S.CONTROL, ms=6, mew=1.2, zorder=5)
    ax.annotate(f"control at the model's top-1\n(epoch {ep[i]}, {top1[i]:.3f}): {sb[i]:.2f}", (top1[i], sb[i]), xytext=(-8, 22),
                textcoords="offset points", fontsize=5.6, ha="right", va="bottom", color=S.INK,
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.5))
    ax.plot(target, T["model"][2][-1], "o", mfc="white", mec=S.MODEL, ms=6, mew=1.2, zorder=5)
    ax.annotate(f"model, final\ntop-1 {target:.3f}: {T['model'][2][-1]:.2f}", (target, T["model"][2][-1]), xytext=(-6, 4),
                textcoords="offset points", fontsize=5.6, ha="right", va="bottom", color=S.INK)
    print(f"  (f) control at the model's final top-1 {target:.3f}: epoch {ep[i]} top-1 {top1[i]:.3f} magno SB {sb[i]:.3f}; "
          f"control max magno SB {sb.max():.3f}")
    ax.set_xlim(0.0, 0.8); ax.set_ylim(0.25, 1.0)
    ax.set_xlabel("validation top-1 (training-time)")
    ax.set_ylabel("magno-stream shape bias\n(training-time, seed 0)")
    S.letter(ax, "f", dx=-0.055, dy=0.03, title="At matched accuracy")

    S.save(fig, "fig2_streams")


if __name__ == "__main__":
    main()
