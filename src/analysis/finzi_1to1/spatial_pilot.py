"""Spatial-loss pilot: probe, gates, topography and verdict
(docs/PLAN_TSN_SPATIAL_PILOT_2026-09-25.md).

Networks: fused_a0 (alpha 0) and fused_a2p5 (alpha 2.5), `extract.network_registry`.
Features and assignments come from the idea-1 pipeline (`extract`, `mapping`, all-units
path); this module adds, per network, on the NSD shared-515 probe stimuli of
`magno_probe` and the float DKL path (plan §3):

  sd       (5 conds x 32,768) layer4.1 response SD per condition -> ACH, LSF
  sd_quant same on the uint8 DKL path the network was trained on (descriptive)
  mr_sd    (2 x 32,768) SD over orig images with the parvo / magno input replaced by
           its response to a uniform mid-grey image -> MR = sd_M / (sd_M + sd_P)
  trunk_*  (2 x 131,072) magno / parvo trunk layer2.1 SD for lum and iso (G-probe)
  l31_sd   (5 x 65,536) layer3.1 SD per condition (descriptive)
  sl_gate  mean SL over 1,000 random neighbourhoods per shared sheet, orig images (G-SL)

    python -m src.analysis.finzi_1to1.spatial_pilot probe
    python -m src.analysis.finzi_1to1.spatial_pilot stats --out results/finzi_1to1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src.analysis.finzi_1to1 import betas_cache_path
from src.analysis.finzi_1to1.magno_probe import (CONDS, DELTA, PROBE_ROOT, SUBJECTS,
                                                  build_stimuli, combine, float_dkl,
                                                  label_index, probe_path, standardise,
                                                  stimulus_checks, subject_deltas)
from src.analysis.finzi_1to1.mapping import out_stem

A0, A1 = "fused_a0", "fused_a2p5"
PILOT_NETS = (A0, A1)
SL_NEIGHBORHOODS = 1000
T1_NEIGHBORHOODS, T1_SHUFFLES = 50, 200
T2_K, T2_SHUFFLES = 10, 1000
ACC_FLOOR, ACC_RATIO, SL_MARGIN = 0.35, 0.8, 0.05


class Moments:
    """Running per-unit mean / SD over images (float64 sums)."""

    def __init__(self):
        self.n, self.s, self.ss = 0, None, None

    def add(self, x):
        x = x.double()
        if self.s is None:
            self.s, self.ss = x.sum(0), (x * x).sum(0)
        else:
            self.s += x.sum(0)
            self.ss += (x * x).sum(0)
        self.n += len(x)

    def sd_mean(self):
        mu = self.s / self.n
        var = (self.ss / self.n - mu * mu).clamp_min(0)
        return var.sqrt().float().cpu().numpy(), mu.float().cpu().numpy()


def probe(networks, chunk: int = 64) -> None:
    import torch

    from src.analysis.encoding import dkl_frontend
    from src.analysis.finzi_1to1.extract import (FRONT_ENDS, PILOT_RUNS, build_fused,
                                                  fused_unit_table, network_registry)
    from src.models import topographic

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    todo = [n for n in networks if not os.path.exists(probe_path(n))]
    print(f"[pilot-probe] device={device} todo={todo}", flush=True)
    if not todo:
        return
    ids, stim, meta = build_stimuli()
    meta["checks"] = stimulus_checks(stim)
    reg = network_registry()
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    units = fused_unit_table()
    grey = np.full((1, 425, 425, 3), 128, np.uint8)

    for n in todo:
        spec = reg[n]
        m, ckpt = build_fused(spec["run"], device)
        fe = FRONT_ENDS[spec["fe"]]

        def inputs(rgb):
            return dkl_frontend.magno_parvo_inputs(rgb, ch_min, ch_max, device=device, **fe)

        @torch.no_grad()
        def forward(x_m, x_p):
            ym, yp = m.magno(x_m), m.parvo(x_p)
            l31 = m.layer3(torch.cat([ym, yp], 1))
            return ym, yp, l31, m.layer4(l31)

        def per_condition():
            """(sd, mean) of layer4.1 per condition, plus layer3.1 and trunk layer2.1."""
            sd4 = np.zeros((len(CONDS), len(units["flat"])), np.float32)
            mu4 = np.zeros_like(sd4)
            sd3 = np.zeros((len(CONDS), 256 * 16 * 16), np.float32)
            trunk = {}
            for ci, c in enumerate(CONDS):
                acc = {k: Moments() for k in ("l4", "l3", "m2", "p2")}
                for i in range(0, len(ids), chunk):
                    ym, yp, l31, l41 = forward(*inputs(stim[c][i:i + chunk]))
                    for k, v in (("l4", l41), ("l3", l31), ("m2", ym), ("p2", yp)):
                        if not torch.isfinite(v).all():
                            raise RuntimeError(f"{n} {c} {k}: non-finite")
                        acc[k].add(v.flatten(1).float())
                sd4[ci], mu4[ci] = acc["l4"].sd_mean()
                sd3[ci] = acc["l3"].sd_mean()[0]
                if c in ("lum", "iso"):
                    trunk[f"magno_{c}"] = acc["m2"].sd_mean()[0]
                    trunk[f"parvo_{c}"] = acc["p2"].sd_mean()[0]
            return sd4, mu4, sd3, trunk

        sd_q, _, _, _ = per_condition()          # uint8 DKL, as trained
        with float_dkl():                         # float DKL (plan §3, amendment 2)
            sd, mu, sd3, trunk = per_condition()
            x_m0, x_p0 = inputs(grey)
            acc = {"M": Moments(), "P": Moments()}
            l41_orig = []
            for i in range(0, len(ids), chunk):
                x_m, x_p = inputs(stim["orig"][i:i + chunk])
                b = len(x_m)
                acc["M"].add(forward(x_m, x_p0.expand(b, -1, -1, -1))[3].flatten(1).float())
                acc["P"].add(forward(x_m0.expand(b, -1, -1, -1), x_p)[3].flatten(1).float())
                l41_orig.append(forward(x_m, x_p)[3].float().cpu())
            mr_sd = np.stack([acc["M"].sd_mean()[0], acc["P"].sd_mean()[0]])
            l41_orig = torch.cat(l41_orig)

        # G-SL: mean layer4.1 SL over 1,000 neighbourhoods of the run's own sheet,
        # all 515 orig images as one batch; layer4.0 descriptive.
        rng = np.random.default_rng(0)
        sl_gate = {}
        sheet = topographic.LayerSheet.from_npz(
            os.path.join(PILOT_RUNS, spec["run"], "sheets", "layer4.1.npz"))
        coords = torch.from_numpy(sheet.coordinates)
        nbs = torch.from_numpy(sheet.neighborhoods)
        pick = rng.choice(len(nbs), SL_NEIGHBORHOODS, replace=False)
        vals = [float(topographic.spatial_correlation_loss(l41_orig, coords, nbs[k]))
                for k in pick]
        sl_gate["layer4.1"] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)),
                                   n=len(vals))

        out = probe_path(n)
        np.savez(out + ".tmp.npz", sd=sd, mean=mu, sd_quant=sd_q, mr_sd=mr_sd, l31_sd=sd3,
                 conds=np.array(CONDS), image_ids=ids,
                 unit_stream=units["stream"], unit_pos=units["pos"], unit_chan=units["chan"],
                 **{f"trunk_{k}": v for k, v in trunk.items()},
                 sl_gate=json.dumps(sl_gate),
                 provenance=json.dumps(dict(spec, checkpoint=ckpt, stimuli=meta,
                                            path="float DKL (sd, mr, trunk, l31); "
                                                 "uint8 DKL (sd_quant)")))
        os.replace(out + ".tmp.npz", out)
        print(f"[pilot-probe] wrote {out} sl_gate={sl_gate} {time.time() - t0:.0f}s",
              flush=True)


# ----------------------------------------------------------------------------- statistics
def indices(network: str) -> dict:
    p = np.load(probe_path(network))
    c = [str(x) for x in p["conds"]]

    def ratio(num, other):
        den = num + other
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    sd = p["sd"].astype(np.float64)
    out = {"ACH": ratio(sd[c.index("lum")], sd[c.index("iso")]),
           "LSF": ratio(sd[c.index("low")], sd[c.index("high")]),
           "MR": ratio(p["mr_sd"][0].astype(np.float64), p["mr_sd"][1].astype(np.float64)),
           "pos": p["unit_pos"], "sl_gate": json.loads(str(p["sl_gate"]))}
    for s in ("magno", "parvo"):
        out[f"trunk_ACH_{s}"] = ratio(p[f"trunk_{s}_lum"].astype(np.float64),
                                      p[f"trunk_{s}_iso"].astype(np.float64))
    return out


def subject_table(network: str, key: str, residualise: bool = True) -> dict:
    ix = indices(network)
    z = standardise(ix[key], ix["pos"], residualise)
    out = {"DV": np.full(len(SUBJECTS), np.nan), "LV": np.full(len(SUBJECTS), np.nan)}
    for i, s in enumerate(SUBJECTS):
        if os.path.exists(out_stem(network, s) + "_assign.npz"):
            r = subject_deltas(z, network, s)
            out["DV"][i], out["LV"][i] = r["DV"], r["LV"]
    return out


def lab(x: np.ndarray) -> dict:
    """Probe labels with one seed (no seed rule)."""
    d = label_index(x, np.array([np.nanmean(x)]))
    d.pop("seeds_same_sign", None)
    d.pop("seed_means", None)
    return d


def history(network: str) -> list:
    from src.analysis.finzi_1to1.extract import PILOT_RUNS, network_registry

    run = network_registry()[network]["run"]
    with open(os.path.join(PILOT_RUNS, run, "history.json")) as f:
        return json.load(f)


def gates(ix: dict) -> dict:
    g = {}
    h = {n: history(n) for n in PILOT_NETS}
    acc = {n: h[n][-1]["val_top1"] for n in PILOT_NETS}
    done = all(len(h[n]) == 150 for n in PILOT_NETS)
    ok = done and acc[A0] >= ACC_FLOOR and acc[A1] >= ACC_RATIO * acc[A0]
    g["G_train"] = dict(epochs={n: len(h[n]) for n in PILOT_NETS}, final_val_top1=acc,
                        best_val_top1={n: max(r["val_top1"] for r in h[n]) for n in PILOT_NETS},
                        gate="PASS" if ok else "FAIL")
    sl = {n: ix[n]["sl_gate"]["layer4.1"]["mean"] for n in PILOT_NETS}
    g["G_SL"] = dict(layer41_sl=sl, margin=SL_MARGIN,
                     gate="PASS" if sl[A1] <= sl[A0] - SL_MARGIN else "FAIL")
    tr = {n: dict(magno_median=float(np.nanmedian(ix[n]["trunk_ACH_magno"])),
                  parvo_median=float(np.nanmedian(ix[n]["trunk_ACH_parvo"])))
          for n in PILOT_NETS}
    g["G_probe"] = dict(trunk_layer21_ACH=tr, gate="PASS" if all(
        v["magno_median"] >= 0.95 and v["parvo_median"] <= 0.05 for v in tr.values()) else "FAIL")
    g["all_pass"] = all(v["gate"] == "PASS" for k, v in g.items() if k.startswith("G_"))
    return g


def t1_mr_map(network: str, rng) -> dict:
    """Pearson corr(|MR_i - MR_j|, sheet distance) over pairs in 50 fixed layer4.1
    neighbourhoods, averaged; null = MR shuffled among units of the same grid cell."""
    from src.analysis.finzi_1to1.extract import PILOT_RUNS, network_registry
    from src.models.topographic import LayerSheet

    run = network_registry()[network]["run"]
    sh = LayerSheet.from_npz(os.path.join(PILOT_RUNS, run, "sheets", "layer4.1.npz"))
    ix = indices(network)
    mr, pos = ix["MR"], ix["pos"]
    nbs = sh.neighborhoods[rng.choice(len(sh.neighborhoods), T1_NEIGHBORHOODS, replace=False)]
    iu = np.triu_indices(nbs.shape[1], 1)
    dist = [np.linalg.norm(sh.coordinates[nb][:, None] - sh.coordinates[nb][None], axis=-1)[iu]
            for nb in nbs]

    def stat(v):
        rs = []
        for nb, d in zip(nbs, dist):
            x = v[nb]
            ad = np.abs(x[:, None] - x[None])[iu]
            ok = np.isfinite(ad)
            rs.append(np.corrcoef(ad[ok], d[ok])[0, 1])
        return float(np.nanmean(rs))

    obs = stat(mr)
    cells = [np.flatnonzero(pos == q) for q in np.unique(pos)]
    null = []
    for _ in range(T1_SHUFFLES):
        v = mr.copy()
        for c in cells:
            v[c] = v[rng.permutation(c)]
        null.append(stat(v))
    null = np.array(null)
    return dict(obs=obs, null_mean=float(null.mean()), null_sd=float(null.std()),
                p_greater=float((1 + np.sum(null >= obs)) / (1 + len(null))),
                n_neighbourhoods=T1_NEIGHBORHOODS, n_shuffles=T1_SHUFFLES)


def t2_stream_clustering(network: str, subject: int, rng) -> dict:
    """Fraction of each assigned unit's 10 nearest assigned sheet neighbours that won the
    same stream, minus its label-shuffle null mean; per hemisphere, then averaged."""
    from scipy.spatial import cKDTree

    from src.analysis.finzi_1to1.extract import PILOT_RUNS, network_registry
    from src.models.topographic import LayerSheet

    run = network_registry()[network]["run"]
    coords = LayerSheet.from_npz(os.path.join(PILOT_RUNS, run, "sheets",
                                              "layer4.1.npz")).coordinates
    a = np.load(out_stem(network, subject) + "_assign.npz")
    b = np.load(betas_cache_path(subject))
    res = []
    for hemi in ("lh", "rh"):
        unit, vox = a[f"unit_{hemi}"], a[f"vox_{hemi}"]
        keep = unit >= 0
        unit, labels = unit[keep], b["label"][vox[keep]]
        _, nn = cKDTree(coords[unit]).query(coords[unit], k=T2_K + 1)
        nn = nn[:, 1:]
        obs = float(np.mean(labels[nn] == labels[:, None]))
        null = np.array([np.mean(p[nn] == p[:, None])
                         for p in (rng.permutation(labels) for _ in range(T2_SHUFFLES))])
        res.append(dict(obs=obs, null_mean=float(null.mean()), excess=obs - float(null.mean()),
                        z=(obs - null.mean()) / null.std(), n_units=int(len(unit))))
    return dict(excess=float(np.mean([r["excess"] for r in res])),
                z=float(np.mean([r["z"] for r in res])), hemis=res)


def stats(out_dir: str) -> dict:
    from scipy.stats import wilcoxon

    ix = {n: indices(n) for n in PILOT_NETS}
    res = {"plan": "docs/PLAN_TSN_SPATIAL_PILOT_2026-09-25.md", "delta_sd": DELTA}
    res["gates"] = gates(ix)
    tabs = {n: {k: subject_table(n, k) for k in ("ACH", "LSF", "MR")} for n in PILOT_NETS}
    for n in PILOT_NETS:
        block = {}
        for c in ("DV", "LV"):
            per = {k: lab(tabs[n][k][c]) for k in ("ACH", "LSF")}
            verdict = combine({k: per[k]["label"] for k in ("ACH", "LSF")})
            if not res["gates"]["all_pass"]:
                verdict = "UNRESOLVED (gate failed)"
            block[c] = dict(verdict=verdict, MR=lab(tabs[n]["MR"][c]), **per)
        res[n] = block
    diff = {}
    for c in ("DV", "LV"):
        per = {k: lab(tabs[A1][k][c] - tabs[A0][k][c]) for k in ("ACH", "LSF", "MR")}
        diff[c] = dict(sl_amplified=all(per[k]["label"] == "MAGNO-LIKE" for k in ("ACH", "LSF")),
                       **per)
    res["A1_minus_A0"] = diff
    raw = {n: {k: subject_table(n, k, residualise=False) for k in ("ACH", "LSF")}
           for n in PILOT_NETS}
    res["raw_index_descriptive"] = {n: {c: {k: lab(raw[n][k][c]) for k in ("ACH", "LSF")}
                                        for c in ("DV", "LV")} for n in PILOT_NETS}
    rng = np.random.default_rng(0)
    res["T1_mr_map"] = {n: t1_mr_map(n, rng) for n in PILOT_NETS}
    t2 = {n: [t2_stream_clustering(n, s, rng) for s in SUBJECTS
              if os.path.exists(out_stem(n, s) + "_assign.npz")] for n in PILOT_NETS}
    ex = {n: np.array([r["excess"] for r in t2[n]]) for n in PILOT_NETS}
    res["T2_stream_clustering"] = dict(
        per_network={n: dict(excess_mean=float(ex[n].mean()), z_mean=float(
            np.mean([r["z"] for r in t2[n]])), per_subject=t2[n]) for n in PILOT_NETS},
        A1_vs_A0_wilcoxon_p=float(wilcoxon(ex[A1] - ex[A0], method="exact").pvalue)
        if len(ex[A1]) == len(ex[A0]) and len(ex[A1]) > 1 else None)
    res["descriptive"] = {n: dict(
        mr_frac_above_0p8=float(np.nanmean(ix[n]["MR"] > 0.8)),
        mr_frac_below_0p2=float(np.nanmean(ix[n]["MR"] < 0.2)),
        mr_median=float(np.nanmedian(ix[n]["MR"])),
        ach_median=float(np.nanmedian(ix[n]["ACH"])),
        lsf_median=float(np.nanmedian(ix[n]["LSF"])),
        sl_gate=ix[n]["sl_gate"]) for n in PILOT_NETS}
    res["next_step"] = next_step(res)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "spatial_pilot_stats.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_dir, "SPATIAL_PILOT_REPORT.md"), "w") as f:
        f.write(render_md(res))
    return res


def next_step(r: dict) -> str:
    """Plan §5 outcome table."""
    v = r[A1]["DV"]["verdict"]
    if not r["gates"]["all_pass"]:
        return "UNRESOLVED (gate failed): diagnose the failing gate before any decision"
    if v == "MAGNO-LIKE":
        if r["A1_minus_A0"]["DV"]["sl_amplified"]:
            return "GO: M/P inputs and the spatial constraint together (SL-AMPLIFIED)"
        return "GO: input pathways make dorsal magno-like; topography on top (not SL-AMPLIFIED)"
    if v.startswith("UNRESOLVED"):
        return "UNRESOLVED: run seed 1 of both arms once, pool, re-decide"
    return "STOP: the hybrid does not give a magno-like dorsal stream"


def render_md(r: dict) -> str:
    def row(name: str, d: dict) -> str:
        if "mean" not in d:
            return f"| {name} | {d['label']} | | | | |"
        lo, hi = d["ci90"]
        return (f"| {name} | **{d['label']}** | {d['mean']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                f"{d['wilcoxon_p']:.4f} | {d['n_positive']}/{d['n']} |")

    g = r["gates"]
    out = ["# TSN x TDANN spatial-loss pilot: result", "",
           f"Plan: `{r['plan']}`. Indices in SD over layer4.1 units (SESOI {r['delta_sd']}); "
           "positive = the stream's units are more magno-like than ventral-stream units.", "",
           f"**Primary (A1 = fused_a2p5, dorsal - ventral): {r[A1]['DV']['verdict']}**", "",
           f"**Next step (plan §5): {r['next_step']}**", "", "## Gates", "",
           f"- G-train: final val top-1 {json.dumps(g['G_train']['final_val_top1'])}, epochs "
           f"{json.dumps(g['G_train']['epochs'])} -> **{g['G_train']['gate']}**",
           f"- G-SL: layer4.1 SL {json.dumps({k: round(v, 4) for k, v in g['G_SL']['layer41_sl'].items()})}"
           f" (A1 must be <= A0 - {g['G_SL']['margin']}) -> **{g['G_SL']['gate']}**",
           f"- G-probe: trunk layer2.1 median ACH {json.dumps(g['G_probe']['trunk_layer21_ACH'])}"
           f" -> **{g['G_probe']['gate']}**", ""]
    for n, title in ((A1, "A1: alpha = 2.5"), (A0, "A0: alpha = 0")):
        out += [f"## {title}", ""]
        for c, cname in (("DV", "dorsal - ventral"), ("LV", "lateral - ventral")):
            out += [f"**{cname}: {r[n][c]['verdict']}**", "",
                    "| index | label | mean | 90% CI | Wilcoxon p | subjects > 0 |",
                    "|---|---|---|---|---|---|",
                    row("ACH", r[n][c]["ACH"]), row("LSF", r[n][c]["LSF"]),
                    row("MR (secondary)", r[n][c]["MR"]), ""]
    out += ["## A1 - A0 (paired by subject)", ""]
    for c in ("DV", "LV"):
        d = r["A1_minus_A0"][c]
        out += [f"**{c}: SL-AMPLIFIED = {d['sl_amplified']}**", "",
                "| index | label | mean | 90% CI | Wilcoxon p | subjects > 0 |",
                "|---|---|---|---|---|---|",
                row("ACH", d["ACH"]), row("LSF", d["LSF"]), row("MR", d["MR"]), ""]
    t2 = r["T2_stream_clustering"]
    out += ["## Topography", "",
            "| network | T1 MR map r (obs) | null mean | p | T2 stream clustering excess | T2 z |",
            "|---|---|---|---|---|---|"]
    for n in PILOT_NETS:
        t1 = r["T1_mr_map"][n]
        p2 = t2["per_network"][n]
        out.append(f"| {n} | {t1['obs']:+.4f} | {t1['null_mean']:+.4f} | {t1['p_greater']:.3f} | "
                   f"{p2['excess_mean']:+.4f} | {p2['z_mean']:+.2f} |")
    out += ["", f"T2 A1 vs A0 paired Wilcoxon p = {t2['A1_vs_A0_wilcoxon_p']}", "",
            "## Descriptive", "", "```", json.dumps(r["descriptive"], indent=1), "```", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("probe", "stats"))
    p.add_argument("--networks", default=",".join(PILOT_NETS))
    p.add_argument("--out", default="results/finzi_1to1")
    a = p.parse_args()
    os.makedirs(PROBE_ROOT, exist_ok=True)
    if a.cmd == "probe":
        probe(a.networks.split(","))
    else:
        r = stats(a.out)
        print(json.dumps({k: r[k] for k in ("gates", "next_step")}, indent=1))
        for n in PILOT_NETS:
            print(n, {c: r[n][c]["verdict"] for c in ("DV", "LV")})


if __name__ == "__main__":
    main()
