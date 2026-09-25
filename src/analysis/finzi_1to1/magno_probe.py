"""Are the TDANN units that win dorsal / lateral voxels magno-like?
(docs/PLAN_TDANN_MAGNO_PROBE_2026-09-25.md)

Two per-unit indices from the NSD shared-515 images, each rendered five ways
(original, luminance-only, isoluminant, low-SF, high-SF):

  ACH = sd_lum / (sd_lum + sd_iso)     achromatic preference   (magno -> 1)
  LSF = sd_low / (sd_low + sd_high)    low-spatial-frequency preference (magno -> 1)

sd_c is the standard deviation over the 515 images of a unit's layer-4 response
in condition c, so each index is scale-free. Colour is decomposed in the TSN
front end's own cone-contrast model (sRGB -> XYZ -> LMS(HPE), contrast against
mid-grey; src.utils.helpers.convert_rgb_to_dkl_numpy): the luminance-only image
keeps each pixel's L+M contrast with equal L, M, S contrast (achromatic); the
isoluminant image is the exact remainder (L-M and S-opponent contrast, zero L+M),
so lum + iso = original in cone contrast. The SF split is a Gaussian low-pass
with its half-amplitude at 12 cycles/image, the TSN magno blur (sigma 4 px at
256 px); high = original - low + image mean.

    python -m src.analysis.finzi_1to1.magno_probe stimuli --limit 16   # checks only
    python -m src.analysis.finzi_1to1.magno_probe extract              # per-unit sd
    python -m src.analysis.finzi_1to1.magno_probe stats --out results/finzi_1to1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src.analysis.finzi_1to1 import OUT_ROOT, betas_cache_path
from src.analysis.finzi_1to1.mapping import out_stem, strat_delta

PROBE_ROOT = os.path.join(OUT_ROOT, "magno_probe")
CONDS = ("orig", "lum", "iso", "low", "high")
SF_HALF_CPI = 12.0  # half-amplitude cut, cycles/image
CHROMA_SCALES = (1.0, 0.75, 0.5, 0.35, 0.25)
MAX_CLIP_FRAC = 0.01  # largest chroma scale whose isoluminant pixels clip <= 1 %
TDANN_PRIMARY = [f"tdann_a025_s{k}" for k in range(5)]
TDANN_DESCRIPTIVE = ["tdann_a0_s0"]
TSN_INSTRUMENT = ["diet_s0", "nodiet_s0", "swap_s0"]
PROBE_NETS = TDANN_PRIMARY + TDANN_DESCRIPTIVE + TSN_INSTRUMENT
DELTA = 0.2  # SESOI, in SDs of the position-residualised index over units
GATE_D = 0.8  # instrument gate: Cohen's d of a known SF manipulation
SUBJECTS = list(range(1, 9))

M_XYZ2LMS = np.array([[0.4002, 0.7075, -0.0807],
                      [-0.2263, 1.1653, 0.0457],
                      [0.0, 0.0, 0.9182]])


# ----------------------------------------------------------------------------- stimuli
def _lms_bg() -> np.ndarray:
    import colour

    return colour.sRGB_to_XYZ(np.array([0.5, 0.5, 0.5])) @ M_XYZ2LMS.T


def rgb_to_cc(rgb01: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> (dL, dM, dS) cone contrast, exactly as the TSN front end."""
    import colour

    bg = _lms_bg()
    return (np.tensordot(colour.sRGB_to_XYZ(rgb01), M_XYZ2LMS.T, axes=1) - bg) / (bg + 1e-6)


def cc_to_rgb(cc: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_cc (unclipped sRGB)."""
    import colour

    bg = _lms_bg()
    lms = cc * (bg + 1e-6) + bg
    return colour.XYZ_to_sRGB(np.tensordot(lms, np.linalg.inv(M_XYZ2LMS).T, axes=1))


def decompose(cc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cone contrast -> (achromatic part, isoluminant part); they sum to cc exactly."""
    a = (cc[..., 0] + cc[..., 1]) / 2.0
    h = (cc[..., 0] - cc[..., 1]) / 2.0
    lum = np.stack([a, a, a], -1)
    iso = np.stack([h, -h, cc[..., 2] - a], -1)
    return lum, iso


def sf_sigma(width: int) -> float:
    """Gaussian sigma (px) whose amplitude response is 0.5 at SF_HALF_CPI."""
    return float(np.sqrt(np.log(2) / (2 * np.pi ** 2)) * width / SF_HALF_CPI)


def _clip_frac(x: np.ndarray) -> float:
    return float(np.mean(np.any((x < 0) | (x > 1), axis=-1)))


def _u8(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def render(rgb_u8: np.ndarray, chroma_scale: float) -> tuple[dict, dict]:
    """(N, H, W, 3) uint8 -> {cond: uint8 images}, {cond: clipped-pixel fraction}."""
    from scipy.ndimage import gaussian_filter

    x = rgb_u8.astype(np.float64) / 255.0
    lum_cc, iso_cc = decompose(rgb_to_cc(x))
    s = sf_sigma(x.shape[2])
    low = gaussian_filter(x, sigma=(0, s, s, 0), mode="reflect")
    # achromatic in cone contrast means R = G = B; take the channel mean so the rendered
    # image is exactly grey (the inverse leaves ~3e-4 channel differences; amendment 2)
    lum = cc_to_rgb(lum_cc).mean(-1, keepdims=True).repeat(3, -1)
    imgs = {"orig": x, "lum": lum, "iso": cc_to_rgb(iso_cc * chroma_scale),
            "low": low, "high": x - low + x.mean(axis=(1, 2), keepdims=True)}
    return ({k: _u8(v) for k, v in imgs.items()}, {k: _clip_frac(v) for k, v in imgs.items()})


def choose_chroma_scale(rgb_u8: np.ndarray, chunk: int = 32) -> tuple[float, dict]:
    clipped = {s: 0.0 for s in CHROMA_SCALES}
    for i in range(0, len(rgb_u8), chunk):
        _, iso_cc = decompose(rgb_to_cc(rgb_u8[i:i + chunk].astype(np.float64) / 255.0))
        for s in CHROMA_SCALES:
            clipped[s] += _clip_frac(cc_to_rgb(iso_cc * s)) * len(iso_cc)
    frac = {s: v / len(rgb_u8) for s, v in clipped.items()}
    ok = [s for s in CHROMA_SCALES if frac[s] <= MAX_CLIP_FRAC]
    return (max(ok) if ok else min(CHROMA_SCALES)), {str(s): round(v, 5) for s, v in frac.items()}


def build_stimuli(limit: int = 0, chunk: int = 32) -> tuple[np.ndarray, dict, dict]:
    from src.analysis.encoding import dkl_frontend

    ids = np.load(betas_cache_path(1))["test_image_ids"]
    if limit:
        ids = ids[:limit]
    rgb = dkl_frontend.load_nsd_images(ids)
    scale, frac = choose_chroma_scale(rgb, chunk)
    stim = {c: np.empty_like(rgb) for c in CONDS}
    clip = {c: 0.0 for c in CONDS}
    for i in range(0, len(rgb), chunk):
        out, cf = render(rgb[i:i + chunk], scale)
        for c in CONDS:
            stim[c][i:i + chunk] = out[c]
            clip[c] += cf[c] * len(out[c])
    meta = dict(n_images=int(len(ids)), chroma_scale=scale, iso_clip_by_scale=frac,
                clip_frac={c: round(v / len(rgb), 5) for c, v in clip.items()},
                sf_half_cpi=SF_HALF_CPI, sf_sigma_px=sf_sigma(rgb.shape[2]))
    return ids, stim, meta


def stimulus_checks(stim: dict) -> dict:
    """Measure what the manipulations did, through TSN's own front end."""
    from src.analysis.encoding import dkl_frontend
    from src.utils.helpers import convert_rgb_to_dkl_numpy

    x = stim["orig"][:8].astype(np.float64)
    cc = rgb_to_cc(x / 255.0)
    ref = convert_rgb_to_dkl_numpy(x)
    mine = np.stack([(cc[..., 0] + cc[..., 1]) / np.sqrt(2), (cc[..., 0] - cc[..., 1]) / np.sqrt(2),
                     (cc[..., 2] - (cc[..., 0] + cc[..., 1])) / np.sqrt(2)], -1)
    out = {"max_abs_dkl_vs_helpers": float(np.abs(mine - ref).max()),
           "max_abs_roundtrip_rgb": float(np.abs(cc_to_rgb(cc) - x / 255.0).max())}

    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    sd = {}
    for c in CONDS:
        t = dkl_frontend.rgb_to_dkl3(stim[c], ch_min, ch_max)  # (N, 3, 256, 256)
        sd[c] = t.flatten(2).std(-1).numpy()  # per image x channel
    o = np.maximum(sd["orig"], 1e-12)
    for c in ("lum", "iso"):
        for j, ch in enumerate(("lum", "rg", "by")):
            out[f"{c}_{ch}_contrast_vs_orig_median"] = float(np.median(sd[c][:, j] / o[:, j]))

    def hi_frac(imgs: np.ndarray) -> np.ndarray:
        g = imgs.astype(np.float64).mean(-1)
        g -= g.mean(axis=(1, 2), keepdims=True)
        p = np.abs(np.fft.fft2(g)) ** 2
        f = np.fft.fftfreq(g.shape[1]) * g.shape[1]
        r = np.hypot(f[:, None], f[None, :])
        return p[:, r > SF_HALF_CPI].sum(1) / np.maximum(p.sum(axis=(1, 2)), 1e-12)

    for c in ("orig", "low", "high"):
        out[f"{c}_power_above_cut_median"] = float(np.median(hi_frac(stim[c])))
    return out


# ----------------------------------------------------------------------------- extraction
def probe_path(network: str) -> str:
    return os.path.join(PROBE_ROOT, f"{network}.npz")


def _rgb_to_dkl3_float(rgb_uint8, ch_min, ch_max, image_size=256):
    """dkl_frontend.rgb_to_dkl3 without its final uint8 quantisation of the DKL channels.

    Amendment 2: mid-grey lands at exactly 127.5 on the uint8 rg scale, and the 1e-6
    term in the cone-contrast denominator makes grey pixels round to 128 when brighter
    than mid-grey and 127 when darker, so the quantised parvo input carries a 1-bit
    luminance map at achromatic pixels. G1 checks the stimuli, so it reads TSN through
    this float path; everything else about the front end is unchanged.
    """
    import torch
    import torch.nn.functional as F

    from src.utils.helpers import convert_rgb_to_dkl_numpy

    t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(0, 3, 1, 2).to(torch.float32)
    if t.shape[-1] != image_size:
        t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", antialias=True,
                          align_corners=False)
    rgb = t.clamp(0, 255).round().permute(0, 2, 3, 1).numpy().astype(np.uint8)
    dkl = (convert_rgb_to_dkl_numpy(rgb) - ch_min) / ((ch_max - ch_min) + 1e-12)
    return torch.from_numpy(np.clip(dkl, 0.0, 1.0).astype(np.float32)).permute(0, 3, 1, 2)


class float_dkl:
    """Context manager: TSN front end with float (unquantised) DKL channels."""

    def __enter__(self):
        from src.analysis.encoding import dkl_frontend

        self._mod, self._orig = dkl_frontend, dkl_frontend.rgb_to_dkl3
        dkl_frontend.rgb_to_dkl3 = _rgb_to_dkl3_float

    def __exit__(self, *exc):
        self._mod.rgb_to_dkl3 = self._orig


def extract(networks: list[str], chunk: int = 64) -> None:
    import torch

    from src.analysis.encoding import dkl_frontend, features
    from src.analysis.finzi_1to1.extract import (FRONT_ENDS, RUNS, TDANN_DIR, build_tdann,
                                                  network_registry, tdann_layer4,
                                                  tdann_unit_table, tsn_layer4, tsn_unit_table)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    todo = [n for n in networks if not os.path.exists(probe_path(n))]
    print(f"[probe] device={device} todo={todo}", flush=True)
    if not todo:
        return
    ids, stim, meta = build_stimuli()
    meta["checks"] = stimulus_checks(stim)
    print(f"[probe] stimuli built {time.time() - t0:.0f}s {json.dumps(meta)}", flush=True)
    with open(os.path.join(PROBE_ROOT, "stimuli_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    reg = network_registry()
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    tsn_units = tsn_unit_table()
    gather = torch.from_numpy(tsn_units["flat"]).to(device)
    s1 = torch.from_numpy(tsn_units["stream"] == 0).to(device)
    for n in todo:
        spec = reg[n]
        if spec["kind"] == "tdann":
            ckpt = os.path.join(TDANN_DIR, spec["ckpt"])
            m = build_tdann(ckpt, device)
            units = tdann_unit_table()

            def run(rgb):
                return tdann_layer4(m, rgb, device).flatten(1)
        else:
            ckpt = os.path.join(RUNS, spec["run"], "checkpoint_last.pt")
            features.assert_converged(ckpt)
            m = features.build_model(checkpoint=ckpt, seed=0, device=device)
            units = tsn_units
            fe = FRONT_ENDS[spec["fe"]]

            @torch.no_grad()
            def run(rgb):
                x_m, x_p = dkl_frontend.magno_parvo_inputs(rgb, ch_min, ch_max, device=device, **fe)
                y1 = tsn_layer4(m.stream1_block1, x_m).flatten(1)
                y2 = tsn_layer4(m.stream2_block1, x_p).flatten(1)
                return torch.where(s1[None, :], y1[:, gather], y2[:, gather])
        def sd_mean():
            sd = np.zeros((len(CONDS), len(units["flat"])), np.float32)
            mu = np.zeros_like(sd)
            for ci, c in enumerate(CONDS):
                f = np.concatenate([run(stim[c][i:i + chunk]).float().cpu().numpy()
                                    for i in range(0, len(ids), chunk)])
                if not np.isfinite(f).all():
                    raise RuntimeError(f"{n} {c}: non-finite features")
                sd[ci], mu[ci] = f.std(0), f.mean(0)
            return sd, mu

        sd, mu = sd_mean()
        extra = {}
        if spec["kind"] == "tsn":  # amendment 2: G1 reads TSN through float DKL channels
            with float_dkl():
                extra["sd_float"], _ = sd_mean()
        out = probe_path(n)
        np.savez(out + ".tmp.npz", sd=sd, mean=mu, conds=np.array(CONDS), image_ids=ids, **extra,
                 unit_stream=units["stream"], unit_pos=units["pos"], unit_chan=units["chan"],
                 provenance=json.dumps(dict(spec, checkpoint=ckpt, stimuli=meta)))
        os.replace(out + ".tmp.npz", out)
        print(f"[probe] wrote {out} {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- statistics
def load_indices(network: str, field: str = "sd") -> dict:
    p = np.load(probe_path(network))
    c = [str(x) for x in p["conds"]]
    sd = p[field].astype(np.float64)

    def ratio(a: str, b: str) -> np.ndarray:
        num, den = sd[c.index(a)], sd[c.index(a)] + sd[c.index(b)]
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    return {"ACH": ratio("lum", "iso"), "LSF": ratio("low", "high"),
            "stream": p["unit_stream"], "pos": p["unit_pos"]}


def standardise(x: np.ndarray, pos: np.ndarray, residualise: bool) -> np.ndarray:
    """Subtract each grid position's mean (residualise), then divide by the SD over units."""
    y = x.copy()
    if residualise:
        for q in np.unique(pos):
            m = pos == q
            y[m] = x[m] - np.nanmean(x[m])
    else:
        y = y - np.nanmean(y)
    return y / np.nanstd(y)


def subject_deltas(z: np.ndarray, network: str, subject: int) -> dict:
    a = np.load(out_stem(network, subject) + "_assign.npz")
    b = np.load(betas_cache_path(subject))
    unit = np.r_[a["unit_lh"], a["unit_rh"]]
    vox = np.r_[a["vox_lh"], a["vox_rh"]]
    keep = unit >= 0
    x = z[unit[keep]]
    lab, nc = b["label"][vox[keep]], b["nc"][vox[keep]]
    ok = np.isfinite(x)
    dv, _ = strat_delta(x[ok], lab[ok], nc[ok], "dorsal", "ventral")
    lv, _ = strat_delta(x[ok], lab[ok], nc[ok], "lateral", "ventral")
    return {"DV": dv, "LV": lv, "n_voxels": int(keep.sum()), "n_nan_index": int((~ok).sum())}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def instrument_gates() -> dict:
    ix = {n: load_indices(n) for n in TSN_INSTRUMENT}
    mag = ix["diet_s0"]["stream"] == 0
    g = {}
    # amendment 2: G1 reads TSN through float DKL channels (the uint8 step injects a
    # luminance bit into rg at grey pixels); the quantised value is reported alongside
    fl = load_indices("diet_s0", "sd_float")
    ach_m = float(np.nanmedian(fl["ACH"][mag]))
    ach_p = float(np.nanmedian(fl["ACH"][~mag]))
    g["G1_ACH_streams"] = dict(magno_median=ach_m, parvo_median=ach_p, path="float DKL",
                               quantised_magno_median=float(np.nanmedian(ix["diet_s0"]["ACH"][mag])),
                               quantised_parvo_median=float(np.nanmedian(ix["diet_s0"]["ACH"][~mag])),
                               gate="PASS" if ach_m >= 0.95 and ach_p <= 0.05 else "FAIL")
    d_m = cohens_d(ix["diet_s0"]["LSF"][mag], ix["nodiet_s0"]["LSF"][mag])
    d_p = cohens_d(ix["swap_s0"]["LSF"][~mag], ix["nodiet_s0"]["LSF"][~mag])
    g["G2_LSF_blur_magno"] = dict(d=d_m, gate="PASS" if d_m >= GATE_D else "FAIL")
    g["G2_LSF_blur_parvo"] = dict(d=d_p, gate="PASS" if d_p >= GATE_D else "FAIL")
    g["all_pass"] = all(v["gate"] == "PASS" for k, v in g.items() if k.startswith("G"))
    return g


def label_index(x_subj: np.ndarray, seed_means: np.ndarray) -> dict:
    from src.analysis.finzi_1to1.stats import decide

    d = decide(x_subj, delta=DELTA)
    base = d["label"]
    d["label"] = {"POSITIVE (A -> first stream)": "MAGNO-LIKE",
                  "NEGATIVE (A -> second stream)": "PARVO-LIKE",
                  "NO PREFERENCE": "NO DIFFERENCE"}.get(base, base)
    if "mean" in d:
        agree = int(np.sum(np.sign(seed_means) == np.sign(d["mean"])))
        d["seeds_same_sign"] = f"{agree}/{len(seed_means)}"
        d["seed_means"] = np.round(seed_means, 4).tolist()
        if d["label"] in ("MAGNO-LIKE", "PARVO-LIKE") and agree < len(seed_means) - 1:
            d["label"] = "SEED-DEPENDENT"
    return d


def combine(labels: dict) -> str:
    good = [k for k, v in labels.items() if v == "MAGNO-LIKE"]
    if len(good) == len(labels):
        return "MAGNO-LIKE"
    if good:
        return "PARTLY MAGNO-LIKE (" + ", ".join(good) + ")"
    if all(v in ("NO DIFFERENCE", "PARVO-LIKE", "SMALL") for v in labels.values()):
        return "NOT MAGNO-LIKE"
    return "UNRESOLVED"


def network_table(nets: list[str], residualise: bool) -> dict:
    """{index: {contrast: (subjects x nets) array}}."""
    out = {k: {c: np.full((len(SUBJECTS), len(nets)), np.nan) for c in ("DV", "LV")}
           for k in ("ACH", "LSF")}
    for j, n in enumerate(nets):
        ix = load_indices(n)
        for k in ("ACH", "LSF"):
            z = standardise(ix[k], ix["pos"], residualise)
            for i, s in enumerate(SUBJECTS):
                if os.path.exists(out_stem(n, s) + "_assign.npz"):
                    r = subject_deltas(z, n, s)
                    out[k]["DV"][i, j], out[k]["LV"][i, j] = r["DV"], r["LV"]
    return out


def stats(out_dir: str) -> dict:
    from scipy.stats import spearmanr

    res = {"plan": "docs/PLAN_TDANN_MAGNO_PROBE_2026-09-25.md", "delta_sd": DELTA,
           "stimuli": json.load(open(os.path.join(PROBE_ROOT, "stimuli_meta.json")))}
    res["instrument"] = instrument_gates()
    for tag, resid in (("primary", True), ("raw_index", False)):
        tab = network_table(TDANN_PRIMARY, resid)
        block = {}
        for c in ("DV", "LV"):
            labels, per = {}, {}
            for k in ("ACH", "LSF"):
                x = tab[k][c]
                per[k] = label_index(np.nanmean(x, 1), np.nanmean(x, 0))
                labels[k] = per[k]["label"]
            verdict = combine(labels)
            if not res["instrument"]["all_pass"]:
                verdict = "UNRESOLVED (instrument gate failed)"
            block[c] = dict(verdict=verdict, **per)
        res[tag] = block
    desc = network_table(TDANN_DESCRIPTIVE, True)
    res["alpha0_descriptive"] = {c: {k: label_index(desc[k][c][:, 0], np.nanmean(desc[k][c], 0))
                                     for k in ("ACH", "LSF")} for c in ("DV", "LV")}
    res["unit_level"] = {}
    for n in TDANN_PRIMARY + TDANN_DESCRIPTIVE:
        ix = load_indices(n)
        ok = np.isfinite(ix["ACH"]) & np.isfinite(ix["LSF"])
        rho = spearmanr(standardise(ix["ACH"], ix["pos"], True)[ok],
                        standardise(ix["LSF"], ix["pos"], True)[ok]).statistic
        res["unit_level"][n] = dict(ach_median=float(np.nanmedian(ix["ACH"])),
                                    lsf_median=float(np.nanmedian(ix["LSF"])),
                                    rho_ach_lsf_residualised=float(rho), n_units=int(ok.sum()))
    dv, lv = res["primary"]["DV"]["verdict"], res["primary"]["LV"]["verdict"]
    res["spatial_loss_trigger"] = "GO" if "MAGNO-LIKE" in (dv, lv) else "NO GO (from this check)"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "magno_probe_stats.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_dir, "MAGNO_PROBE_REPORT.md"), "w") as f:
        f.write(render_md(res))
    return res


def render_md(r: dict) -> str:
    def row(name: str, d: dict) -> str:
        if "mean" not in d:
            return f"| {name} | {d['label']} | | | | |"
        lo, hi = d["ci90"]
        return (f"| {name} | **{d['label']}** | {d['mean']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                f"{d['wilcoxon_p']:.4f} | {d['n_positive']}/{d['n']} subj, "
                f"{d.get('seeds_same_sign', '-')} seeds |")

    g = r["instrument"]
    s = r["stimuli"]
    out = ["# TDANN magno-likeness probe: result", "",
           f"Plan: `{r['plan']}`. Units of the index: SD over TDANN units "
           f"(SESOI {r['delta_sd']}). Positive = the stream's units are more magno-like "
           "than ventral-stream units.", "",
           f"**Spatial-loss trigger (this check): {r['spatial_loss_trigger']}**", "",
           "## Instrument gates (TSN)", "",
           f"- G1 ACH isolates the channels (float DKL, amendment 2): magno median "
           f"{g['G1_ACH_streams']['magno_median']:.3f}, parvo median "
           f"{g['G1_ACH_streams']['parvo_median']:.3f} -> **{g['G1_ACH_streams']['gate']}** "
           f"(as trained, uint8 DKL: magno {g['G1_ACH_streams'].get('quantised_magno_median', float('nan')):.3f}, "
           f"parvo {g['G1_ACH_streams'].get('quantised_parvo_median', float('nan')):.3f})",
           f"- G2 blur raises magno LSF (diet vs nodiet): d = {g['G2_LSF_blur_magno']['d']:+.2f} "
           f"-> **{g['G2_LSF_blur_magno']['gate']}**",
           f"- G2 blur raises parvo LSF (swap vs nodiet): d = {g['G2_LSF_blur_parvo']['d']:+.2f} "
           f"-> **{g['G2_LSF_blur_parvo']['gate']}**", ""]
    for tag, title in (("primary", "Primary (position-residualised), TDANN alpha = 0.25, 5 seeds"),
                       ("raw_index", "Secondary (raw index), same networks")):
        out += [f"## {title}", ""]
        for c, cname in (("DV", "dorsal - ventral"), ("LV", "lateral - ventral")):
            out += [f"**{cname}: {r[tag][c]['verdict']}**", "",
                    "| index | label | mean | 90% CI | Wilcoxon p | consistency |",
                    "|---|---|---|---|---|---|",
                    row("ACH (luminance > isoluminant)", r[tag][c]["ACH"]),
                    row("LSF (low > high SF)", r[tag][c]["LSF"]), ""]
    out += ["## Descriptive: alpha = 0 (no spatial loss), seed 0", "",
            "| contrast | index | label | mean | 90% CI | Wilcoxon p | consistency |",
            "|---|---|---|---|---|---|---|"]
    for c in ("DV", "LV"):
        for k in ("ACH", "LSF"):
            out.append(f"| {c} " + row(k, r["alpha0_descriptive"][c][k]))
    out += ["", "## Unit level (no brain data)", "",
            "| network | median ACH | median LSF | Spearman rho(ACH, LSF), residualised | units |",
            "|---|---|---|---|---|"]
    for n, u in r["unit_level"].items():
        out.append(f"| {n} | {u['ach_median']:.3f} | {u['lsf_median']:.3f} | "
                   f"{u['rho_ach_lsf_residualised']:+.3f} | {u['n_units']} |")
    out += ["", "## Stimuli", "",
            f"- 515 shared images; isoluminant chroma scale {s.get('chroma_scale')} "
            f"(clipped pixels {s.get('clip_frac', {}).get('iso')})",
            f"- checks: `{json.dumps(s.get('checks', {}))}`", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("stimuli", "extract", "stats"))
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--networks", default=",".join(PROBE_NETS))
    p.add_argument("--out", default="results/finzi_1to1")
    a = p.parse_args()
    if a.cmd == "stimuli":
        ids, stim, meta = build_stimuli(limit=a.limit)
        meta["checks"] = stimulus_checks(stim)
        print(json.dumps(meta, indent=1))
    elif a.cmd == "extract":
        extract(a.networks.split(","))
    else:
        r = stats(a.out)
        print(json.dumps({k: r[k] for k in ("instrument", "spatial_loss_trigger")}, indent=1))
        for c in ("DV", "LV"):
            print(c, r["primary"][c]["verdict"],
                  {k: (r["primary"][c][k]["label"], round(r["primary"][c][k].get("mean", np.nan), 3))
                   for k in ("ACH", "LSF")})


if __name__ == "__main__":
    main()
