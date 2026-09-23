"""PLAN_KAPPA outcomes: agreement with the ten observers at a matched cue-conflict rate.

  PYTHONPATH=. python tmp_analysis/figs_v2/kappa_rate.py [--run-id 2026-09-15_zoo]

Reads the per-network sweeps written by `src/analysis/generality/kappa_sweep.py` (rate and
Cohen's kappa at each of the 21 gains) and the committed `zoo_behaviour.json` (probe bias,
labels, groups), evaluates predictions K-1 to K-6 of `docs/PLAN_KAPPA_2026-09-17.md`, and
writes `results/figures_v2/kappa_rate.json` plus a markdown table.

The zoo is a fixed set of networks, so across-network statistics are descriptive and carry no
p-values, as in Extended Data Table 4. The only sampled unit is the ten observers; kappa
intervals are percentile bootstraps over them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

BEH = "/scratch/byrong/mpk/generality/behaviour"
OUT_JSON = "results/figures_v2/kappa_rate.json"
OUT_MD = "results/figures_v2/kappa_rate.md"
R_STAR, TOL = 0.40, 0.05
R_STAR_2 = 0.30


def rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = x.argsort()
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    # average ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def r2_rank(y, X):
    """R^2 of an OLS fit on rank-transformed columns (descriptive; readouts are not independent)."""
    y = rank(y)
    A = np.column_stack([np.ones(len(y))] + [rank(c) for c in X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1 - (resid ** 2).sum() / ss_tot) if ss_tot else float("nan")


def kendall(a, b):
    """Kendall tau-b over a fixed set (no p-value), with the fraction of pairs that reverse."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    con = dis = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                ta += 1; tb += 1
            elif da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da * db > 0:
                con += 1
            else:
                dis += 1
    tot = con + dis
    denom = np.sqrt((con + dis + ta) * (con + dis + tb))
    return (float((con - dis) / denom) if denom else float("nan"),
            float(dis / tot) if tot else float("nan"), int(dis), int(n * (n - 1) // 2))


def boot_ci(per_human, n=1000, seed=0):
    v = np.array([x for x in per_human.values() if x is not None], dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def boot_norm_ci(row, net, n=1000, seed=0):
    """Percentile bootstrap over the ten observers of kappa/kappa_max, as % of the human-pair value."""
    c = net.get("ceiling_over_max")
    kh, mh = row.get("kappa_per_human"), row.get("kappa_max_per_human")
    if not (c and kh and mh):
        return None
    v = np.array([kh[h] / mh[h] for h in kh
                  if kh[h] is not None and mh.get(h) not in (None, 0)], dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return None
    rng = np.random.default_rng(seed)
    d = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1) * 100 / c
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def at_rate(sweep, target, tol, gated=True):
    """The gated readout whose rate is closest to `target`, or None if none is within tol."""
    rows = [r for r in sweep if (r["in_gate"] or not gated) and np.isfinite(r["rate"])
            and np.isfinite(r["kappa_pct_ceiling"])]
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(r["rate"] - target))
    return best if abs(best["rate"] - target) <= tol else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="2026-09-15_zoo")
    args = ap.parse_args()

    zoo = json.load(open("results/figures_v2/zoo_behaviour.json"))
    meta = {n["tag"]: n for n in zoo["networks"]}

    nets = {}
    for f in sorted(glob.glob(f"{BEH}/{args.run_id}/*_kappa_sweep.json")):
        d = json.load(open(f))
        tag = d["tag"]
        m = meta.get(tag)
        if m is None:
            print(f"  ! {tag}: not in zoo_behaviour.json, skipped")
            continue
        sw = d["sweep"]
        gated = [r for r in sw if r["in_gate"]]
        nets[tag] = dict(
            tag=tag, label=m["label"], group=m["group"], family=m.get("family"),
            probe_bias=m["probe"]["bias"], trained_rate=m["trained"], trained_top1=m["trained_top1"],
            gate=d["gate"], ceiling=d["ceiling"], ceiling_over_max=d.get("ceiling_over_max"), sweep=sw,
            gated_rate_range=float(max(r["rate"] for r in gated) - min(r["rate"] for r in gated)),
            kappa_max_gated=float(max(r["kappa_pct_ceiling"] for r in gated)),
            kappa_max_all=float(max(r["kappa_pct_ceiling"] for r in sw)),
            kappa_sd_gated=float(np.std([r["kappa_pct_ceiling"] for r in gated], ddof=1)),
            n_nan=int(sum(r["n_human_nan"] for r in sw)),
        )
    print(f"[kappa_rate] {len(nets)} networks")

    IN = [t for t, n in nets.items() if n["group"] == "imagenet"]
    SIN = [t for t, n in nets.items() if n["group"] != "imagenet"]
    print(f"  {len(IN)} ImageNet-trained, {len(SIN)} other ({', '.join(nets[t]['label'] for t in SIN)})")

    # ---- K-1: the readout moves the rate, not agreement
    k1 = [t for t in IN if nets[t]["gated_rate_range"] >= 0.20 and nets[t]["kappa_max_gated"] < 40]
    K1 = dict(met=len(k1) >= 36, n=len(k1), of=len(IN),
              n_range_ok=sum(nets[t]["gated_rate_range"] >= 0.20 for t in IN),
              n_kappa_ok=sum(nets[t]["kappa_max_gated"] < 40 for t in IN),
              worst=sorted(((nets[t]["kappa_max_gated"], nets[t]["label"]) for t in IN), reverse=True)[:3])

    # ---- matched rate
    def matched(target):
        rows = {}
        for t in list(IN) + list(SIN):
            r = at_rate(nets[t]["sweep"], target, TOL)
            if r is not None:
                rows[t] = r
        return rows

    m1, target = matched(R_STAR), R_STAR
    switched = False
    if sum(1 for t in m1 if nets[t]["group"] == "imagenet") < 20:      # decision rule 1
        m1, target, switched = matched(R_STAR_2), R_STAR_2, True
    m2 = matched(R_STAR_2 if not switched else R_STAR)
    in_m = [t for t in m1 if nets[t]["group"] == "imagenet"]
    kap = np.array([m1[t]["kappa_pct_ceiling"] for t in in_m])
    pro = np.array([nets[t]["probe_bias"] for t in in_m])
    tra = np.array([nets[t]["trained_rate"] for t in in_m])

    K2 = dict(met=bool(kap.max() / kap.min() >= 2) if kap.min() > 0 else False,
              ratio=float(kap.max() / kap.min()) if kap.min() > 0 else float("inf"),
              lo=float(kap.min()), hi=float(kap.max()), n=len(in_m), target=target, switched=switched,
              excluded=[nets[t]["label"] for t in IN if t not in m1],
              hi_net=nets[in_m[int(kap.argmax())]]["label"], lo_net=nets[in_m[int(kap.argmin())]]["label"],
              hi_ci=boot_ci(m1[in_m[int(kap.argmax())]]["kappa_per_human"]),
              lo_ci=boot_ci(m1[in_m[int(kap.argmin())]]["kappa_per_human"]))
    K3 = dict(met=None, rho=spearman(kap, pro), rho_trained_rate=spearman(kap, tra),
              rho_probe_vs_trained=spearman(pro, tra), n=len(in_m))
    K3["met"] = bool(K3["rho"] >= 0.30)

    within = np.array([nets[t]["kappa_sd_gated"] for t in in_m])
    K4a = dict(met=bool(kap.std(ddof=1) >= np.median(within)),
               across_sd=float(kap.std(ddof=1)), within_sd_median=float(np.median(within)),
               ratio=float(kap.std(ddof=1) / np.median(within)))

    ys, rs, ps = [], [], []
    for t in IN:
        for r in nets[t]["sweep"]:
            if r["in_gate"] and np.isfinite(r["kappa_pct_ceiling"]):
                ys.append(r["kappa_pct_ceiling"]); rs.append(r["rate"]); ps.append(nets[t]["probe_bias"])
    r2_rate = r2_rank(ys, [rs])
    r2_both = r2_rank(ys, [rs, ps])
    K4b = dict(met=bool(r2_both - r2_rate >= 0.05), r2_rate=r2_rate, r2_rate_probe=r2_both,
               delta=float(r2_both - r2_rate), n_readouts=len(ys))

    sin_tag = next((t for t in SIN if "sin" in t and "_in" not in t.replace("_sin", "")), None)
    sin_tag = sin_tag or next((t for t in SIN if t.endswith("_sin")), None)
    K5 = dict(met=None, network=nets[sin_tag]["label"] if sin_tag else None)
    if sin_tag and sin_tag in m1:
        K5.update(kappa=float(m1[sin_tag]["kappa_pct_ceiling"]), rate=float(m1[sin_tag]["rate"]),
                  best_imagenet=float(kap.max()), best_imagenet_net=K2["hi_net"],
                  ci=boot_ci(m1[sin_tag]["kappa_per_human"]))
        K5["met"] = bool(K5["kappa"] > K5["best_imagenet"])
    elif sin_tag:
        r = min((r for r in nets[sin_tag]["sweep"] if r["in_gate"]), key=lambda r: abs(r["rate"] - target))
        K5.update(kappa=float(r["kappa_pct_ceiling"]), rate=float(r["rate"]), closest_only=True,
                  best_imagenet=float(kap.max()), best_imagenet_net=K2["hi_net"],
                  ci=boot_ci(r["kappa_per_human"]))
        K5["met"] = bool(K5["kappa"] > K5["best_imagenet"])

    worst = sorted(((nets[t]["kappa_max_all"], nets[t]["label"]) for t in IN), reverse=True)
    K6 = dict(met=bool(worst[0][0] < 50), max=float(worst[0][0]), net=worst[0][1], top3=worst[:3])

    # ---------------- post hoc, not pre-registered: what the failures of K-2 … K-4 imply ---------
    # (a) does the kappa ranking survive the readout, as the rate ranking did not (tau = 0.47)?
    kt = np.array([next(r["kappa_pct_ceiling"] for r in nets[t]["sweep"] if abs(r["w"] - 0.5) < 1e-9)
                   for t in in_m])
    tau, rev, n_rev, n_pairs = kendall(kt, kap)
    # (b) one curve for the zoo: kappa as a quadratic in the rate over all gated readouts,
    #     and how much each network departs from it
    X = np.column_stack([np.ones(len(rs)), np.asarray(rs), np.asarray(rs) ** 2])
    beta, *_ = np.linalg.lstsq(X, np.asarray(ys), rcond=None)
    pred = X @ beta
    ss = float(((np.asarray(ys) - np.mean(ys)) ** 2).sum())
    r2_curve = float(1 - ((np.asarray(ys) - pred) ** 2).sum() / ss)
    off, k = {}, 0
    for t in IN:
        g = [r for r in nets[t]["sweep"] if r["in_gate"] and np.isfinite(r["kappa_pct_ceiling"])]
        off[t] = float(np.mean([ys[k + i] - pred[k + i] for i in range(len(g))]))
        k += len(g)
    ot = sorted(off, key=lambda t: -off[t])
    resid = np.asarray(ys) - pred
    hi_mask = np.asarray(rs) > 0.5
    curve = dict(beta=[float(x) for x in beta], r2=r2_curve, n_readouts=len(ys),
                 rmse=float(np.sqrt((resid ** 2).mean())),
                 max_abs_resid=float(np.abs(resid).max()),
                 rmse_rate_above_0_5=float(np.sqrt((resid[hi_mask] ** 2).mean())) if hi_mask.any() else None,
                 n_rate_above_0_5=int(hi_mask.sum()),
                 offset_sd=float(np.std(list(off.values()), ddof=1)),
                 offset_range=[float(min(off.values())), float(max(off.values()))],
                 highest=[(nets[t]["label"], round(off[t], 1)) for t in ot[:3]],
                 lowest=[(nets[t]["label"], round(off[t], 1)) for t in ot[-3:]],
                 rho_offset_probe=spearman([off[t] for t in IN], [nets[t]["probe_bias"] for t in IN]),
                 rho_offset_top1=spearman([off[t] for t in IN], [nets[t]["trained_top1"] for t in IN]),
                 rho_offset_trained_rate=spearman([off[t] for t in IN], [nets[t]["trained_rate"] for t in IN]),
                 at_human_rate=float(beta[0] + beta[1] * 0.958 + beta[2] * 0.958 ** 2),
                 max_reachable_rate=float(max(r["rate"] for t in IN for r in nets[t]["sweep"] if r["in_gate"])))
    # (c) the Stylized-ImageNet network at a rate it can actually reach
    hi_target = 0.65
    hi = {t: at_rate(nets[t]["sweep"], hi_target, TOL) for t in nets}
    hi = {t: r for t, r in hi.items() if r is not None}
    hi_rows = sorted(({"label": nets[t]["label"], "group": nets[t]["group"], "w": hi[t]["w"],
                       "rate": hi[t]["rate"], "kappa_pct": hi[t]["kappa_pct_ceiling"],
                       "probe_bias": nets[t]["probe_bias"]} for t in hi),
                     key=lambda r: -r["kappa_pct"])
    # (d) the attainable bound: Cohen's kappa cannot exceed (1 - |p_m - p_h| - p_e)/(1 - p_e), and
    #     the observers sit at 0.958, so any manipulation that raises the model's rate raises the
    #     bound. A = kappa / kappa_max, as a percentage of the same quantity for human pairs.
    def A(row, net):
        c = net["ceiling_over_max"]
        return 100 * row["kappa_over_max"] / c if row.get("kappa_over_max") is not None and c else np.nan

    have_bound = all("kappa_over_max" in r for t in IN for r in nets[t]["sweep"])
    bound = None
    if have_bound:
        ys_a, rs_a, ps_a = [], [], []
        for t in IN:
            for r in nets[t]["sweep"]:
                if r["in_gate"] and np.isfinite(A(r, nets[t])):
                    ys_a.append(A(r, nets[t])); rs_a.append(r["rate"]); ps_a.append(nets[t]["probe_bias"])
        a_m = np.array([A(m1[t], nets[t]) for t in in_m])
        ends = []
        for t in IN:
            g = [r for r in nets[t]["sweep"] if r["in_gate"]]
            lo, hi_ = min(g, key=lambda r: r["w"]), max(g, key=lambda r: r["w"])
            ends.append((hi_["kappa_pct_ceiling"] - lo["kappa_pct_ceiling"],
                         A(hi_, nets[t]) - A(lo, nets[t])))
        wi_raw, wi_norm, pos = [], [], 0
        for t in IN:
            g = [r for r in nets[t]["sweep"] if r["in_gate"] and np.isfinite(A(r, nets[t]))]
            if len(g) < 5:
                continue
            wi_raw.append(spearman([r["rate"] for r in g], [r["kappa_pct_ceiling"] for r in g]))
            wi_norm.append(spearman([r["rate"] for r in g], [A(r, nets[t]) for r in g]))
            lo, hi_ = min(g, key=lambda r: r["w"]), max(g, key=lambda r: r["w"])
            pos += int(A(hi_, nets[t]) > A(lo, nets[t]))
        bound = dict(
            metric="kappa / kappa_max, as % of the human-pair value",
            within_rho_raw_median=float(np.median(wi_raw)), within_rho_norm_median=float(np.median(wi_norm)),
            within_n=len(wi_raw), n_positive_norm_change=pos,
            kappa_max_range=[float(min(r["kappa_max_mean"] for t in IN for r in nets[t]["sweep"] if r["in_gate"])),
                             float(max(r["kappa_max_mean"] for t in IN for r in nets[t]["sweep"] if r["in_gate"]))],
            ceiling_over_max=float(np.mean([nets[t]["ceiling_over_max"] for t in IN])),
            kappa_max_at_trained=float(np.mean([next(r["kappa_max_mean"] for r in nets[t]["sweep"]
                                                     if abs(r["w"] - 0.5) < 1e-9) for t in IN])),
            r2_rate_raw=r2_rate, r2_rate_norm=r2_rank(ys_a, [rs_a]),
            r2_rate_probe_norm=r2_rank(ys_a, [rs_a, ps_a]),
            texture_to_shape_end_raw=float(np.median([e[0] for e in ends])),
            texture_to_shape_end_norm=float(np.median([e[1] for e in ends])),
            matched_lo=float(a_m.min()), matched_hi=float(a_m.max()),
            matched_ratio=float(a_m.max() / a_m.min()) if a_m.min() > 0 else float("inf"),
            matched_sd=float(a_m.std(ddof=1)),
            rho_matched_probe=spearman(a_m, pro), rho_matched_trained_rate=spearman(a_m, tra),
            max_any_readout=float(max(A(r, nets[t]) for t in IN for r in nets[t]["sweep"]
                                      if np.isfinite(A(r, nets[t])))),
            high_rate=[dict(label=nets[t]["label"], group=nets[t]["group"], rate=hi[t]["rate"],
                            norm=float(A(hi[t], nets[t])), raw=hi[t]["kappa_pct_ceiling"],
                            norm_ci=boot_norm_ci(hi[t], nets[t]))
                       for t in sorted(hi, key=lambda t: -A(hi[t], nets[t]))],
        )

    post = dict(tau_kappa_trained_vs_matched=tau, kappa_pairs_reversing=rev,
                n_pairs_reversing=n_rev, n_pairs=n_pairs, n_networks=len(in_m),
                curve=curve, high_rate_target=hi_target, high_rate=hi_rows, bound=bound)

    out = dict(
        run_id=args.run_id, plan="docs/PLAN_KAPPA_2026-09-17.md", target_rate=target, tol=TOL,
        post_hoc=post,
        n_networks=len(nets), n_imagenet=len(IN),
        predictions=dict(K1=K1, K2=K2, K3=K3, K4a=K4a, K4b=K4b, K5=K5, K6=K6),
        matched=[dict(tag=t, label=nets[t]["label"], group=nets[t]["group"], w=m1[t]["w"],
                      rate=m1[t]["rate"], kappa_pct=m1[t]["kappa_pct_ceiling"],
                      kappa=m1[t]["kappa_mean"], decided=m1[t]["decided"],
                      probe_bias=nets[t]["probe_bias"], trained_rate=nets[t]["trained_rate"],
                      kappa_trained_pct=next(r["kappa_pct_ceiling"] for r in nets[t]["sweep"]
                                             if abs(r["w"] - 0.5) < 1e-9),
                      kappa_norm_pct=(100 * m1[t]["kappa_over_max"] / nets[t]["ceiling_over_max"]
                                      if m1[t].get("kappa_over_max") is not None
                                      and nets[t].get("ceiling_over_max") else None))
                 for t in sorted(m1, key=lambda t: -m1[t]["kappa_pct_ceiling"])],
        matched_secondary=dict(target=R_STAR_2 if not switched else R_STAR,
                               n=sum(1 for t in m2 if nets[t]["group"] == "imagenet"),
                               rho_probe=spearman([m2[t]["kappa_pct_ceiling"] for t in m2 if nets[t]["group"] == "imagenet"],
                                                  [nets[t]["probe_bias"] for t in m2 if nets[t]["group"] == "imagenet"]),
                               ratio=float(max(m2[t]["kappa_pct_ceiling"] for t in m2 if nets[t]["group"] == "imagenet")
                                           / min(m2[t]["kappa_pct_ceiling"] for t in m2 if nets[t]["group"] == "imagenet"))),
        trajectories={t: dict(label=nets[t]["label"], group=nets[t]["group"], gate=nets[t]["gate"],
                              probe_bias=nets[t]["probe_bias"],
                              w=[r["w"] for r in nets[t]["sweep"]],
                              rate=[r["rate"] for r in nets[t]["sweep"]],
                              kappa_pct=[r["kappa_pct_ceiling"] for r in nets[t]["sweep"]],
                              in_gate=[r["in_gate"] for r in nets[t]["sweep"]])
                      for t in nets},
    )
    os.makedirs("results/figures_v2", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)

    L = []
    P = L.append
    P(f"# Agreement at a matched rate (PLAN_KAPPA, run {args.run_id})\n")
    P(f"Target rate {target:.2f} ± {TOL:.2f}"
      f"{' (switched by decision rule 1)' if switched else ''}; "
      f"{K2['n']} of {len(IN)} ImageNet-trained networks reach it inside their gate.\n")
    P("| prediction | outcome | value |")
    P("|---|---|---|")
    P(f"| K-1 rate moves ≥ 0.20, κ stays < 40% of ceiling | {'MET' if K1['met'] else 'FAILED'} | {K1['n']}/{K1['of']} networks (rate range ok in {K1['n_range_ok']}, κ ok in {K1['n_kappa_ok']}) |")
    P(f"| K-2 κ varies ≥ 2× at matched rate | {'MET' if K2['met'] else 'FAILED'} | {K2['lo']:.1f}–{K2['hi']:.1f}% ({K2['ratio']:.2f}×; {K2['lo_net']} → {K2['hi_net']}) |")
    P(f"| K-3 ρ(κ at matched rate, probe bias) ≥ +0.30 | {'MET' if K3['met'] else 'FAILED'} | ρ = {K3['rho']:+.2f} (with trained rate {K3['rho_trained_rate']:+.2f}; probe v trained rate {K3['rho_probe_vs_trained']:+.2f}) |")
    P(f"| K-4a across-network sd ≥ within-network sd | {'MET' if K4a['met'] else 'FAILED'} | {K4a['across_sd']:.1f} v {K4a['within_sd_median']:.1f} points ({K4a['ratio']:.2f}×) |")
    P(f"| K-4b probe adds ≥ 0.05 R² over rate | {'MET' if K4b['met'] else 'FAILED'} | {K4b['r2_rate']:.3f} → {K4b['r2_rate_probe']:.3f} (Δ {K4b['delta']:+.3f}, {K4b['n_readouts']} gated readouts) |")
    P(f"| K-5 SIN highest at matched rate | {'MET' if K5.get('met') else 'FAILED'} | {K5.get('kappa', float('nan')):.1f}% at rate {K5.get('rate', float('nan')):.2f} v {K5.get('best_imagenet', float('nan')):.1f}% ({K5.get('best_imagenet_net')}) |")
    P(f"| K-6 no ImageNet readout above 50% anywhere | {'MET' if K6['met'] else 'FAILED'} | max {K6['max']:.1f}% ({K6['net']}) |")
    P("")
    P("| network | w | rate | κ (% of ceiling) | κ at trained readout | probe bias | trained rate |")
    P("|---|---|---|---|---|---|---|")
    for r in out["matched"]:
        star = " *" if r["group"] != "imagenet" else ""
        P(f"| {r['label']}{star} | {r['w']:.2f} | {r['rate']:.3f} | {r['kappa_pct']:.1f} | "
          f"{r['kappa_trained_pct']:.1f} | {r['probe_bias']:+.2f} | {r['trained_rate']:.3f} |")
    P("\n`*` not ImageNet-trained (Stylized-ImageNet control or recipe variant).\n")
    P("## Post hoc (not pre-registered): what the failures imply\n")
    P(f"- Ranking the {post['n_networks']} networks by κ at their trained readouts against by κ at the "
      f"matched rate: Kendall τ = {post['tau_kappa_trained_vs_matched']:.2f}, "
      f"{post['n_pairs_reversing']} of {post['n_pairs']} pairs ({100 * post['kappa_pairs_reversing']:.0f}%) reverse.")
    P(f"- One curve fits the zoo: κ% = {curve['beta'][0]:.1f} {curve['beta'][1]:+.1f}·rate "
      f"{curve['beta'][2]:+.1f}·rate² over {curve['n_readouts']} gated readouts, R² = {curve['r2']:.3f}; "
      f"RMSE {curve['rmse']:.1f} points of ceiling (largest residual {curve['max_abs_resid']:.1f}; RMSE "
      f"{curve['rmse_rate_above_0_5']:.1f} over the {curve['n_rate_above_0_5']} readouts above rate 0.5); "
      f"per-network mean offsets span {curve['offset_range'][0]:+.1f} to {curve['offset_range'][1]:+.1f} points "
      f"(s.d. {curve['offset_sd']:.1f}), and are unrelated to the probe bias "
      f"(ρ = {curve['rho_offset_probe']:+.2f}), top-1 (ρ = {curve['rho_offset_top1']:+.2f}) or the trained "
      f"rate (ρ = {curve['rho_offset_trained_rate']:+.2f}).")
    P(f"- Highest offsets: {curve['highest']}; lowest: {curve['lowest']}.")
    P(f"- Extrapolating that curve to the human rate 0.958 gives {curve['at_human_rate']:.0f}% of the "
      f"ceiling (extrapolation: the largest gated rate observed is {curve['max_reachable_rate']:.2f}).")
    if post.get("bound"):
        B = post["bound"]
        P("\n### The attainable bound\n")
        P(f"- Cohen's κ cannot exceed (1 − |p_model − p_human| − p_e)/(1 − p_e). Against observers at "
          f"0.958 the bound at the trained readouts averages {B['kappa_max_at_trained']:.3f}, so the "
          f"rise of κ with the readout is in part the bound moving. Human pairs sit at "
          f"{100 * B['ceiling_over_max']:.0f}% of their own bound.")
        P(f"- Normalised (κ / κ_max, as % of the human-pair value): from the texture end to the shape "
          f"end of the gate the median network gains {B['texture_to_shape_end_norm']:+.1f} points, "
          f"against {B['texture_to_shape_end_raw']:+.1f} points on the raw scale.")
        P(f"- Rank R² on the rate: {B['r2_rate_norm']:.3f} normalised against {B['r2_rate_raw']:.3f} raw "
          f"(adding the probe bias: {B['r2_rate_probe_norm']:.3f}). Within network, the median Spearman ρ "
          f"between rate and agreement across the gated sweep is {B['within_rho_raw_median']:+.2f} raw and "
          f"{B['within_rho_norm_median']:+.2f} normalised ({B['n_positive_norm_change']} of {B['within_n']} "
          f"networks gain from the texture to the shape end on the normalised scale). The bound itself runs "
          f"from {B['kappa_max_range'][0]:.3f} to {B['kappa_max_range'][1]:.3f} over the gated readouts, "
          f"against the human-pair κ of 0.234.")
        P(f"- At the matched rate the normalised agreement spans {B['matched_lo']:.1f}–{B['matched_hi']:.1f}% "
          f"({B['matched_ratio']:.2f}×, s.d. {B['matched_sd']:.1f}); ρ with probe bias "
          f"{B['rho_matched_probe']:+.2f}, with the trained rate {B['rho_matched_trained_rate']:+.2f}. "
          f"No readout of any ImageNet-trained network exceeds {B['max_any_readout']:.0f}%.")
        P(f"- At rate {hi_target:.2f}: " + "; ".join(
            f"{r['label']} {r['norm']:.0f}%"
            + (f" [{r['norm_ci'][0]:.0f}, {r['norm_ci'][1]:.0f}]" if r.get("norm_ci") else "")
            + f" (raw {r['raw']:.0f}%)" for r in B["high_rate"]) + " (95% bootstrap over the ten observers).")
    P(f"\n| at rate {hi_target:.2f} ± {TOL:.2f} | w | rate | κ (% of ceiling) | probe bias |")
    P("|---|---|---|---|---|")
    for r in hi_rows:
        P(f"| {r['label']}{' *' if r['group'] != 'imagenet' else ''} | {r['w']:.2f} | {r['rate']:.3f} | "
          f"{r['kappa_pct']:.1f} | {r['probe_bias']:+.2f} |")
    open(OUT_MD, "w").write("\n".join(L) + "\n")

    for k, v in out["predictions"].items():
        print(f"  {k}: {'MET' if v.get('met') else 'FAILED'}  {json.dumps({x: y for x, y in v.items() if x != 'met'})[:150]}")
    print("wrote", OUT_JSON, "and", OUT_MD)


if __name__ == "__main__":
    main()
