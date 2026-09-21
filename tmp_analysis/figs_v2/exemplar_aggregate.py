"""Aggregate PLAN_EXEMPLAR (docs/PLAN_EXEMPLAR_2026-09-21.md) and score its predictions.

  python tmp_analysis/figs_v2/exemplar_aggregate.py

Reads the per-network outputs of slurm/exemplar_zoo.sh and slurm/exemplar_streams.sh from /scratch and the
committed results/figures_v2/{zoo_behaviour,span_model}.json; writes results/figures_v2/exemplar_zoo.json
and exemplar_streams.json. Predictions are scored exactly as written in the plan, on point estimates;
across-network statistics carry 95% percentile bootstrap intervals over networks (10,000 resamples, seed 0).
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
from scipy.stats import kendalltau, spearmanr

ZOO_OUT = "/scratch/byrong/mpk/exemplar/results/2026-09-21_exemplar"
STR_OUT = "/scratch/byrong/mpk/exemplar/results/2026-09-21_streams"
N_BOOT_NET, N_BOOT_IMG = 10_000, 2_000


def rho_ci(a, b, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = float(spearmanr(a, b).correlation)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(N_BOOT_NET):
        i = rng.integers(0, len(a), len(a))
        if np.ptp(a[i]) > 0 and np.ptp(b[i]) > 0:
            bs.append(spearmanr(a[i], b[i]).correlation)
    return dict(rho=r, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(a))


def pair_mask_reversed(a, b):
    """Per pair (i < j, itertools.combinations order): does the order differ between a and b (ties not reversed)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    i, j = np.triu_indices(len(a), 1)
    return (a[i] - a[j]) * (b[i] - b[j]) < 0


def reversals(a, b):
    """As tmp_analysis/figs_v2/zoo_behaviour.py: pairs whose order differs (ties not reversed)."""
    m = pair_mask_reversed(a, b)
    return int(m.sum()), int(m.size)


def test_retest(Hs, Ht, Fo, seed=0):
    """Two independent image bootstraps (shared across networks); reversal fraction between the trained
    rankings; and the per-pair 95% interval of the trained-rate difference (resolved pairs)."""
    rng = np.random.default_rng(seed)
    n_img = Hs.shape[1]
    pairs = list(itertools.combinations(range(Hs.shape[0]), 2))
    out = {}
    for name, rate_fn in (("primary", lambda c: (Hs @ c) / ((Hs @ c) + (Ht @ c))), ("forced", lambda c: (Fo @ c) / c.sum())):
        rev, boot = [], []
        for _ in range(N_BOOT_IMG):
            c1 = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
            c2 = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
            r1, r2 = rate_fn(c1), rate_fn(c2)
            rev.append(reversals(r1, r2)[0] / len(pairs))
            boot.append(r1)
        B = np.array(boot)
        ii, jj = np.triu_indices(Hs.shape[0], 1)
        D = B[:, ii] - B[:, jj]
        lo, hi = np.percentile(D, 2.5, 0), np.percentile(D, 97.5, 0)
        out[name] = dict(mean=float(np.mean(rev)), ci=[float(np.percentile(rev, 2.5)), float(np.percentile(rev, 97.5))],
                         resolved=((lo > 0) | (hi < 0)).tolist())
    return out


def load_zoo():
    zb = {n["tag"]: n for n in json.load(open("results/figures_v2/zoo_behaviour.json"))["networks"]}
    sm = {n["tag"]: n for n in json.load(open("results/figures_v2/span_model.json"))["networks"]}
    rows = []
    for tag, n in zb.items():
        f = os.path.join(ZOO_OUT, f"{tag}_exemplar.json")
        if not os.path.exists(f):
            continue
        e = json.load(open(f)); z = np.load(os.path.join(ZOO_OUT, f"{tag}_exemplar.npz"))
        s = sm.get(tag, {})
        rows.append(dict(tag=tag, label=n["label"], group=n["group"], trained=e["trained"], trained_forced=e["trained_forced"],
                         E0_gated_reported=n["gated_span"], E0_shape_end_reported=n["shape_end"],
                         rand_max_abs=s.get("rand_max_abs"), texture_share=s.get("texture_share"),
                         spans={k: dict(span=v["primary"]["span_mean"], gated=v["primary"]["gated_span"],
                                        shape_end=v["primary"]["shape_end"], texture_end=v["primary"]["texture_end"],
                                        rho_mean_curve=v["primary"]["rho_mean_curve"], rho_min=v["primary"]["rho_min"],
                                        forced_span=v["forced"]["span_mean"], forced_gated=v["forced"]["gated_span"],
                                        forced_shape_end=v["forced"]["shape_end"], mean_curve=v["primary"]["mean_curve"])
                                for k, v in e["spans"].items()},
                         probes={k: {kk: vv for kk, vv in v.items() if kk != "bias_draws"} for k, v in e["probes"].items()},
                         _Hs=z["hit_shape"].astype(float), _Ht=z["hit_texture"].astype(float), _Fo=z["forced_shape"].astype(float)))
    return rows


def zoo():
    rows = load_zoo()
    im = [r for r in rows if r["group"] == "imagenet"]
    by = {r["tag"]: r for r in rows}
    print(f"zoo: {len(rows)} networks read, {len(im)} ImageNet (V1)")
    E1 = np.array([r["spans"]["E1"]["span"] for r in im])
    st = dict(n_all=len(rows), n_imagenet=len(im))
    for k in ("E0", "E1", "E2", "E3"):
        v = np.array([r["spans"][k]["span"] for r in im]); g = np.array([r["spans"][k]["gated"] for r in im])
        st[k] = dict(span_median=float(np.median(v)), span_min=float(v.min()), span_max=float(v.max()),
                     gated_median=float(np.median(g)), gated_min=float(g.min()), gated_max=float(g.max()),
                     n_ge_0_20=int((v >= 0.20).sum()), forced_median=float(np.median([r["spans"][k]["forced_span"] for r in im])))
    # ranking
    tr = np.array([r["trained"] for r in im]); se = np.array([r["spans"]["E1"]["shape_end"] for r in im])
    trf = np.array([r["trained_forced"] for r in im]); sef = np.array([r["spans"]["E1"]["forced_shape_end"] for r in im])
    se0 = np.array([r["spans"]["E0"]["shape_end"] for r in im])
    nr, npairs = reversals(tr, se); nrf, _ = reversals(trf, sef); nr0, _ = reversals(tr, se0)
    tt = test_retest(np.stack([r["_Hs"] for r in im]), np.stack([r["_Ht"] for r in im]), np.stack([r["_Fo"] for r in im]))
    res = np.array(tt["primary"]["resolved"]); revm = pair_mask_reversed(tr, se)
    resf = np.array(tt["forced"]["resolved"]); revmf = pair_mask_reversed(trf, sef)
    st["ranking"] = dict(n_pairs=npairs, E0_frac_reversed=nr0 / npairs, E1_frac_reversed=nr / npairs,
                         E1_forced_frac_reversed=nrf / npairs, tau_trained_vs_E1_shape_end=float(kendalltau(tr, se).statistic),
                         floor_primary=dict(mean=tt["primary"]["mean"], ci=tt["primary"]["ci"]),
                         floor_forced=dict(mean=tt["forced"]["mean"], ci=tt["forced"]["ci"]),
                         n_resolved=int(res.sum()), E1_frac_reversed_resolved=float(revm[res].mean()) if res.any() else None,
                         n_resolved_forced=int(resf.sum()), E1_forced_frac_reversed_resolved=float(revmf[resf].mean()) if resf.any() else None)
    q1 = np.array([r["probes"]["Q1"]["bias"] for r in im])
    st["probes"] = {k: dict(median=float(np.median([r["probes"][k]["bias"] for r in im])),
                            min=float(np.min([r["probes"][k]["bias"] for r in im])), max=float(np.max([r["probes"][k]["bias"] for r in im])))
                    for k in ("Q0", "Q1", "Q1b", "Q2", "S64", "S256n") if k in im[0]["probes"]}
    st["probes"]["Q1_n_interval_below_0"] = int(sum(r["probes"]["Q1"]["bias_p95"] < 0 for r in im))
    st["probes"]["Q1_n_interval_above_0"] = int(sum(r["probes"]["Q1"]["bias_p05"] > 0 for r in im))
    allq1 = {r["tag"]: r["probes"]["Q1"]["bias"] for r in rows}
    top = sorted(allq1, key=lambda t: -allq1[t])
    st["rho_Q1_Q0_all"] = rho_ci([r["probes"]["Q1"]["bias"] for r in rows], [r["probes"]["Q0"]["bias"] for r in rows])
    st["rho_trained_E1span"] = rho_ci(tr, E1)
    st["rho_texture_share_Q1"] = rho_ci([r["texture_share"] for r in im], q1)
    st["rho_texture_share_Q0"] = rho_ci([r["texture_share"] for r in im], [r["probes"]["Q0"]["bias"] for r in im])
    st["tau_trained_Q1"] = float(kendalltau(tr, q1).statistic)
    st["rho_Q1_S256n_all"] = rho_ci([r["probes"]["Q1"]["bias"] for r in rows], [r["probes"]["S256n"]["bias"] for r in rows]) if "S256n" in rows[0]["probes"] else None
    st["tau_trained_Q0"] = float(kendalltau(tr, [r["probes"]["Q0"]["bias"] for r in im]).statistic)
    st["Q1_top3"] = [(t, allq1[t]) for t in top[:3]]
    n_null = sum(r["spans"]["E1"]["span"] > r["rand_max_abs"] for r in rows if r["rand_max_abs"] is not None)
    n_null_den = sum(r["rand_max_abs"] is not None for r in rows)
    n_mono = sum(r["spans"]["E1"]["rho_mean_curve"] >= 0.9 for r in rows)
    pred = {
        "P-X1": dict(value=f"{n_null}/{n_null_den} above the stored random max", met=bool(n_null >= 40 and n_null_den == 42)),
        "P-X2": dict(value=f"{n_mono}/{len(rows)} with rho(w, E1 mean curve) >= 0.9", met=bool(n_mono == len(rows) == 42)),
        "P-X3a": dict(value=st["E1"]["span_median"], met=bool(st["E1"]["span_median"] >= 0.15)),
        "P-X3b": dict(value=f"{st['E1']['n_ge_0_20']}/{len(im)} reach 0.20", met=bool(st["E1"]["n_ge_0_20"] < 19)),
        "P-X4": dict(value=f"E2 median {st['E2']['span_median']:.3f} v E3 median {st['E3']['span_median']:.3f}",
                     met=bool(st["E2"]["span_median"] < st["E3"]["span_median"])),
        "P-X5": dict(value=f"E1 {st['ranking']['E1_frac_reversed']:.3f} v floor upper {st['ranking']['floor_primary']['ci'][1]:.3f}",
                     met=bool(st["ranking"]["E1_frac_reversed"] > st["ranking"]["floor_primary"]["ci"][1])),
        "P-X6": dict(value=st["rho_trained_E1span"]["rho"], met=bool(abs(st["rho_trained_E1span"]["rho"]) <= 0.3)),
        "P-X7": dict(value=f"median {st['probes']['Q1']['median']:+.3f}; {st['probes']['Q1_n_interval_below_0']} intervals below 0",
                     met=bool(-0.10 <= st["probes"]["Q1"]["median"] <= 0.10 and st["probes"]["Q1_n_interval_below_0"] < 10)),
        "P-X8": dict(value=st["rho_Q1_Q0_all"]["rho"], met=bool(st["rho_Q1_Q0_all"]["rho"] >= 0.7)),
        "P-X9": dict(value=f"top: {top[0]}", met=bool(top[0] == "hub_resnet50_sin"), note="confirmatory (K1)"),
        "P-X10": dict(value=st["rho_texture_share_Q1"]["rho"], met=bool(st["rho_texture_share_Q1"]["rho"] <= -0.5)),
    }
    for k, v in pred.items():
        print(f"  {k}: {'MET ' if v['met'] else 'FAIL'}  {v['value']}")
    for r in rows:
        for k in ("_Hs", "_Ht", "_Fo"):
            r.pop(k)
    return dict(plan="docs/PLAN_EXEMPLAR_2026-09-21.md", source=ZOO_OUT, stats=st, predictions=pred, networks=rows)


def streams():
    S = {}
    for f in sorted(os.listdir(STR_OUT)) if os.path.isdir(STR_OUT) else []:
        if f.endswith("_stream_exemplar.json"):
            e = json.load(open(os.path.join(STR_OUT, f)))
            S[e["model_name"]] = {s: {k: {kk: vv for kk, vv in v.items() if kk != "bias_draws"} for k, v in e["streams"][s].items()}
                                  for s in ("magno", "parvo")}
    T = {}
    for f in sorted(os.listdir(ZOO_OUT)):
        if f.endswith("_exemplar.json") and f.startswith(("concat_", "run_")):
            e = json.load(open(os.path.join(ZOO_OUT, f)))
            T[e["tag"]] = dict(probes={k: {kk: vv for kk, vv in v.items() if kk != "bias_draws"} for k, v in e["probes"].items()},
                               stream_probes={s: {k: {kk: vv for kk, vv in v.items() if kk != "bias_draws"} for k, v in sp.items()}
                                              for s, sp in e.get("stream_probes", {}).items()},
                               spans={k: dict(span=v["primary"]["span_mean"], gated=v["primary"]["gated_span"],
                                              forced_span=v["forced"]["span_mean"]) for k, v in e["spans"].items()},
                               gate=e["gate"], trained=e["trained"])
    q = lambda name, s, k="Q1": S[name][s][k]["bias"]
    diet = [f"dkl_bcs_seed{i}_trained" for i in range(3)]
    pred, cells = {}, {}
    if all(d in S for d in diet):
        pred["P-X11"] = dict(value=[q(d, "magno") - q(d, "parvo") for d in diet], met=all(q(d, "magno") > q(d, "parvo") for d in diet))
        pred["P-X12"] = dict(value=[q(d, "parvo") for d in diet], met=all(q(d, "parvo") < 0 for d in diet))
    if "untrained_blur_cs" in S:
        pred["P-X13"] = dict(value=q("untrained_blur_cs", "magno"), met=q("untrained_blur_cs", "magno") > 0)
        if all(d in S for d in diet):
            pred["P-X14"] = dict(value=[q(d, "parvo") - q("untrained_blur_cs", "parvo") for d in diet],
                                 met=all(q(d, "parvo") < q("untrained_blur_cs", "parvo") for d in diet))
    single = [f"run_dkl_bcs_single_seed{i}" for i in range(3)]; conc = [f"concat_dkl_bcs_seed{i}" for i in range(3)]
    if all(t in T for t in single + conc):
        sv = [T[s]["probes"]["Q1"]["bias"] for s in single]; mv = [T[c]["stream_probes"]["stream1"]["Q1"]["bias"] for c in conc]
        pred["P-X15"] = dict(value=dict(single=sv, magno_penultimate=mv), met=all(a < b for a, b in zip(sv, mv)))
    spec = [("chromatic_none", "dkl_base_seed{}_trained", "parvo"), ("chromatic_transform", "dkl_bcs_swap_seed{}_trained", "parvo"),
            ("achromatic_none", "dkl_bcs_parvo_highsf_seed{}_trained", "parvo"), ("achromatic_transform", "dkl_bcs_seed{}_trained", "magno")]
    if all(p.format(i) in S for _, p, _ in spec for i in range(3)):
        for est in [e for e in ("Q0", "Q1", "Q1b", "Q2", "S64", "S256n") if e in S[spec[0][1].format(0)][spec[0][2]]]:
            c = {name: [q(p.format(i), s, est) for i in range(3)] for name, p, s in spec}
            m = {k: float(np.mean(v)) for k, v in c.items()}
            add = m["chromatic_transform"] + m["achromatic_none"] - m["chromatic_none"]
            cells[est] = dict(cells=c, means=m, additive=add, interaction=m["achromatic_transform"] - add)
        pred["P-X16"] = dict(value=cells["Q1"]["interaction"], met=cells["Q1"]["interaction"] >= 0.20)
    for k, v in pred.items():
        print(f"  {k}: {'MET ' if v['met'] else 'FAIL'}  {v['value']}")
    return dict(plan="docs/PLAN_EXEMPLAR_2026-09-21.md", source=[STR_OUT, ZOO_OUT], streams=S, trunks=T,
                two_by_two=cells, predictions={k: dict(value=v["value"], met=bool(v["met"])) for k, v in pred.items()})


if __name__ == "__main__":
    z = zoo()
    json.dump(z, open("results/figures_v2/exemplar_zoo.json", "w"), indent=1)
    s = streams()
    json.dump(s, open("results/figures_v2/exemplar_streams.json", "w"), indent=1)
    print("wrote results/figures_v2/exemplar_zoo.json and exemplar_streams.json")
