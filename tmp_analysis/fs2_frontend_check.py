"""Front-end check for every paper pipeline on the Stage-2 TSNs (2026-10-19).

The paper's model of record moves to fs2_dvd300_avg_w050_ls_seed{0,1,2}, whose
magno input at test time is dkl_frontend.apply_dvd_diet(lum, 300). Their
checkpoint config still says ``blur: true, cs: true`` beside
``dvd_end_months: 300``, so a pipeline that builds its front end from blur/cs
flags feeds them the thesis diet without any error. The pipelines now take
their front end through dkl_frontend.frontend_from_args / checkpoint_frontend
and refuse a front end that is not the checkpoint's own (check_frontend).

On the first 16 MiniEcoset val images (ood_robustness: its own first 16 test
images; fit_w_observers / sweep_error_consistency: those 16 tiled over their
first 128-image chunk) this runs each entry point's real code path up to its
first dkl_frontend.magno_parvo_inputs call, captures that call's input
images and output, and checks:

  A  per run: checkpoint_frontend(checkpoint) == checkpoint_frontend(run name
     / run dir), and magno_parvo_inputs(**it) == StreamInputBuilder(cfg) bit
     for bit; check_frontend refuses the other diet on fs2 runs and keeps a
     crossed front end on the BA run; a checkpoint without config gives None
  B  fs2 runs, run given BY NAME with the run's flags: every entry point's
     magno and parvo input == StreamInputBuilder(the checkpoint's config) on
     the very images it was given, bit for bit
  C  fs2 runs with the other diet's flags (blur + CS on a DVD run; the DVD
     diet on fs2_bcs): every CLI entry point raises ValueError before any
     front-end call
  D  BA run (dkl_bcs_seed0) with its usual flags and with the crossed
     no-front-end flags: every entry point's input == the committed HEAD
     dkl_frontend.magno_parvo_inputs with the flags the pre-change code passed
  E  untrained anchors: no checkpoint + --dvd-age-months 300 == the dvd300
     builder; no checkpoint, no flags == HEAD's default (blur + CS48)

Exits 1 on any failure.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tmp_analysis"))
sys.path.insert(0, os.path.join(ROOT, "tmp_analysis", "figs_v2"))

from src.analysis.encoding import dkl_frontend  # noqa: E402
from src.training.standalone_trainer import StreamInputBuilder  # noqa: E402

N = 16
RGB_H5 = "/scratch/byrong/VST_data/VST_datasets/texture2shape_miniecoset_256px.h5"
FS2 = dkl_frontend.FS2_RUNS
BA_DIR = "/scratch/byrong/VST_data/mpk_runs/dkl_bcs_seed0"
THESIS = ("/scratch/byrong/tsn_checkpoints/dkl_blur_cs_dvd_trans/"
          "epoch=147-step=152292.ckpt")
WORK = os.environ.get("TMPDIR") or "/scratch/byrong/mpk/encoding/logs/2026-10-19_frontend/work"
fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


# ------------------------------------------------------------------ fixtures
def load_head_frontend():
    """dkl_frontend.py as committed at HEAD: the pre-change code path."""
    sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    src = subprocess.run(
        ["git", "-C", ROOT, "show", "HEAD:src/analysis/encoding/dkl_frontend.py"],
        capture_output=True, text=True, check=True).stdout
    path = os.path.join(WORK, "dkl_frontend_HEAD.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("dkl_frontend_head", path)
    head = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(head)
    check(not hasattr(head, "checkpoint_frontend"),
          f"HEAD {sha[:10]} dkl_frontend is the pre-change module")
    return head


class Stop(Exception):
    """Raised by the spy at an entry point's first front-end call."""


REAL = dkl_frontend.magno_parvo_inputs
CALLS: list[dict] = []
STOP = {"on": True}


def spy(rgb, ch_min, ch_max, **kw):
    out = REAL(rgb, ch_min, ch_max, **kw)
    CALLS.append(dict(rgb=np.asarray(rgb).copy(), ch_min=np.asarray(ch_min),
                      ch_max=np.asarray(ch_max), kw=kw, out=out))
    if STOP["on"]:
        raise Stop
    return out


@contextlib.contextmanager
def patched(*triples):
    """Set (object, attribute, value) for the duration, then restore."""
    old = [(o, a, getattr(o, a)) for o, a, _ in triples]
    for o, a, v in triples:
        setattr(o, a, v)
    try:
        yield
    finally:
        for o, a, v in old:
            setattr(o, a, v)


def run_main(mod, argv: list[str], extra=()):
    """mod.main() under argv up to its first front-end call.
    Returns ('call', record) | ('raise', exception) | ('done', None)."""
    CALLS.clear()
    old_argv = sys.argv
    sys.argv = [mod.__name__] + argv
    try:
        with patched(*extra):
            mod.main()
        return ("done", None)
    except Stop:
        return ("call", CALLS[-1])
    except (Exception, SystemExit) as e:  # noqa: BLE001 -- reported per check
        return ("raise", e)
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------- references
def builder_ref(cfg: dict, age: int, rec: dict):
    """StreamInputBuilder on the DKL image of exactly the RGB the call got."""
    x3 = dkl_frontend.rgb_to_dkl3(rec["rgb"], rec["ch_min"], rec["ch_max"])
    return StreamInputBuilder(cfg, age_months=age)(x3)


def same(a, b) -> bool:
    return (a[0].shape == b[0].shape and a[1].shape == b[1].shape
            and torch.equal(a[0].cpu(), b[0].cpu())
            and torch.equal(a[1].cpu(), b[1].cpu()))


def maxdiff(a, b) -> str:
    try:
        return (f"max|d| magno {(a[0].cpu() - b[0].cpu()).abs().max():.2e} "
                f"parvo {(a[1].cpu() - b[1].cpu()).abs().max():.2e}")
    except Exception:  # shape mismatch
        return f"shapes {tuple(a[0].shape)}/{tuple(a[1].shape)} vs {tuple(b[0].shape)}/{tuple(b[1].shape)}"


# --------------------------------------------------------------- entry points
def entry_points(rgb16, items16, out_dir):
    """name -> function(ckpt_arg | None, flags) -> run_main result."""
    from src.analysis.behavior import cue_conflict, stream_features, stream_shape_texture
    import fit_w_observers
    import fusion_sweep
    import ood_robustness
    import perimage_logits
    import sweep_error_consistency

    def imgs(paths):
        return rgb16[np.arange(len(paths)) % N]

    def cli(mod, base, items):
        def f(ck, flags, name):
            argv = list(base(name)) + (["--checkpoint", ck] if ck else []) + flags
            extra = [(mod, "load_images", imgs)]
            if items is not None:
                extra.append((mod, "list_cue_conflict_images", lambda: items))
            return run_main(mod, argv, extra)
        return f

    return {
        "stream_features": cli(stream_features, lambda n: [
            "--model-name", f"chk_{n}", "--feat-dir", f"{out_dir}/feat",
            "--out-dir", f"{out_dir}/sf"], items16),
        "stream_shape_texture": cli(stream_shape_texture, lambda n: [
            "--model-name", f"chk_{n}", "--out", f"{out_dir}/st"], items16),
        "cue_conflict": cli(cue_conflict, lambda n: [
            "--model-name", f"chk_{n}", "--out-dir", f"{out_dir}/cc"], items16),
        "fusion_sweep": cli(fusion_sweep, lambda n: [
            "--model-name", f"chk_{n}", "--out-dir", f"{out_dir}/fs"], items16),
        "fit_w_observers": cli(fit_w_observers, lambda n: [
            "--model-name", f"chk_{n}", "--out-dir", f"{out_dir}/fw"], None),
        "sweep_error_consistency": cli(sweep_error_consistency, lambda n: [
            "--model-name", f"chk_{n}", "--out-dir", f"{out_dir}/se"], None),
        "perimage_logits": cli(perimage_logits, lambda n: [
            "--model-name", f"chk_{n}", "--out-dir", f"{out_dir}/pl"], items16),
        "ood_robustness": lambda ck, flags, n: run_main(ood_robustness, [
            "--model-name", f"chk_{n}", "--out-dir", f"{out_dir}/ood",
            "--n-images", str(N)] + (["--checkpoint", ck] if ck else []) + flags),
    }


def config_entry_points(rgb16, items16, out_dir):
    """Entry points that read the front end from the run's config (no flags):
    name -> function(run spec) -> (record | exception, note)."""
    import fig1_model
    import nonlinear_readout_features as nl
    from src.analysis.generality.models import FrozenNet

    def generality(run, ckpt_name):
        CALLS.clear()
        spec = f"concat:{run}" + (f":{ckpt_name}" if ckpt_name else "")
        try:
            net = FrozenNet(spec)
            note = f"dataset {net.cfg['dataset']} exists={os.path.exists(net.cfg['dataset'])}"
            net.features_from_rgb(rgb16)
        except Stop:
            return CALLS[-1], note, os.path.exists(net.cfg["dataset"])
        except Exception as e:  # noqa: BLE001
            return None, repr(e), False
        return None, note, False

    def nonlinear(run_dir, ckpt_name):
        seen = {}

        def stub(model, builder, h5_path, split, device, batch=256, limit=None):
            seen[split] = h5_path
            z = torch.zeros(0, 512)
            return z, z, torch.zeros(0, dtype=torch.long)

        res = run_main(nl, ["--run-dir", run_dir, "--checkpoint-name", ckpt_name,
                            "--tag", "chk", "--out-dir", f"{out_dir}/nl"],
                       [(nl, "extract_h5", stub), (nl, "load_images", lambda p: rgb16[np.arange(len(p)) % N]),
                        (nl, "list_cue_conflict_images", lambda: items16)])
        ok = bool(seen) and all(os.path.exists(p) for p in seen.values())
        return res, f"h5 {sorted(set(seen.values()))}", ok

    def fig1(run_spec):
        """(rgb, lum, rg, lum_t, rg_t) of panel (b) and the front-end calls;
        None tiles (and the error in place of the calls) if it raised."""
        CALLS.clear()
        STOP["on"] = False
        try:
            with patched((fig1_model, "FRONTEND_RUN", run_spec),
                         (dkl_frontend, "load_nsd_images", lambda ids: rgb16[:1])):
                tiles = fig1_model.stream_inputs(0)
        except Exception as e:  # noqa: BLE001
            return None, repr(e)
        finally:
            STOP["on"] = True
        return tiles, list(CALLS)

    return generality, nonlinear, fig1


# ------------------------------------------------------------------------ main
def main() -> None:
    torch.set_num_threads(8)
    os.makedirs(WORK, exist_ok=True)
    out_dir = os.path.join(WORK, "out")
    head = load_head_frontend()
    dkl_frontend.magno_parvo_inputs = spy   # every entry point calls it through the module

    with h5py.File(RGB_H5, "r") as f:
        rgb16 = f["val"]["data"][:N]
    from src.analysis.behavior.cue_conflict import list_cue_conflict_images
    cc_items = list_cue_conflict_images()
    items16 = cc_items[:N]
    ch_min, ch_max = dkl_frontend.compute_dkl_channel_minmax()
    print(f"{N} val images {rgb16.shape}; {len(cc_items)} cue-conflict items "
          f"(labels only; images replaced by the val images)", flush=True)

    eps = entry_points(rgb16, items16, out_dir)
    generality, nonlinear, fig1 = config_entry_points(rgb16, items16, out_dir)

    runs = ([f"fs2_dvd300_avg_w050_ls_seed{s}" for s in range(3)]
            + [f"fs2_dvdfix300_avg_w050_ls_seed{s}" for s in range(3)]
            + ["fs2_bcs_avg_w050_ls_seed0"])
    for run in runs:
        ck = os.path.join(FS2, run, "checkpoint_final.pt")
        if not os.path.exists(ck):
            if run.startswith("fs2_dvdfix300") and not run.endswith("seed0"):
                print(f"SKIP {run}: no checkpoint_final.pt yet (training)", flush=True)
                continue
        cfg = torch.load(ck, map_location="cpu", weights_only=False)["config"]
        age = json.load(open(os.path.join(FS2, run, "config.json")))["age_months"]
        dvd = cfg.get("dvd_end_months")
        right = ["--dvd-age-months", f"{dvd:g}"] if dvd is not None else []
        wrong = [] if dvd is not None else ["--dvd-age-months", "300"]
        print(f"\n=== {run}: blur={cfg['blur']} cs={cfg['cs']} "
              f"dvd_end_months={dvd} age={age}; flags {right or '(none)'}", flush=True)

        # A: the helper itself
        fe = dkl_frontend.checkpoint_frontend(ck)
        fe_name = dkl_frontend.checkpoint_frontend(run)
        check(fe == fe_name, f"A {run} checkpoint_frontend(ckpt) == (run name): {fe}")
        check(dkl_frontend.resolve_checkpoint(run) == ck, f"A {run} resolve_checkpoint(name) -> {ck}")
        x3 = dkl_frontend.rgb_to_dkl3(rgb16, ch_min, ch_max)
        got = REAL(rgb16, ch_min, ch_max, **fe)
        ref = StreamInputBuilder(cfg, age_months=age)(x3)
        check(same(got, ref), f"A {run} magno_parvo_inputs(**checkpoint_frontend) == "
                              f"StreamInputBuilder ({maxdiff(got, ref)})")
        bad = (dict(lum_blur=True, lum_cs=True) if dvd is not None
               else dict(lum_blur=False, lum_cs=False, dvd_age=300.0))
        try:
            dkl_frontend.check_frontend(ck, **bad)
            check(False, f"A {run} check_frontend accepted the other diet {bad}")
        except ValueError as e:
            check(True, f"A {run} check_frontend refuses {bad}: {str(e)[:150]}")
        check(dkl_frontend.check_frontend(ck, **fe) == fe, f"A {run} check_frontend accepts its own kwargs")

        # B / C: CLI entry points, run given by name
        for name, ep in eps.items():
            kind, rec = ep(run, right, f"{run}_right")
            if kind != "call":
                check(False, f"B {run} {name}: no front-end call ({kind}: {rec!r})")
            else:
                r = builder_ref(cfg, age, rec)
                check(same(rec["out"], r), f"B {run} {name} input == StreamInputBuilder "
                                          f"on its {len(rec['rgb'])} images ({maxdiff(rec['out'], r)})")
            kind, rec = ep(run, wrong, f"{run}_wrong")
            check(kind == "raise" and isinstance(rec, ValueError) and not CALLS,
                  f"C {run} {name} with {wrong or 'blur+CS defaults'}: "
                  f"{kind} {type(rec).__name__ if rec is not None else ''} "
                  f"{str(rec)[:90] if kind == 'raise' else ''}")

        # B: entry points that read the run's config
        rec, note, ok = generality(run, None)
        if rec is None:
            check(False, f"B {run} generality concat: no front-end call ({note})")
        else:
            r = builder_ref(cfg, age, rec)
            check(same(rec["out"], r) and ok, f"B {run} generality concat input == "
                                              f"StreamInputBuilder ({maxdiff(rec['out'], r)}); {note}")
        (kind, rec), note, ok = nonlinear(run, "checkpoint_final.pt")
        if kind != "call":
            check(False, f"B {run} nonlinear_readout_features: no front-end call ({kind}: {rec!r})")
        else:
            r = builder_ref(cfg, age, rec)
            check(same(rec["out"], r) and ok, f"B {run} nonlinear_readout_features cue-conflict "
                                              f"input == StreamInputBuilder ({maxdiff(rec['out'], r)}); {note}")
        if run.endswith("seed0"):
            tiles, calls = fig1(run)
            x3_1 = dkl_frontend.rgb_to_dkl3(rgb16[:1], ch_min, ch_max)
            r_m = StreamInputBuilder(cfg, age_months=age)(x3_1)[0][0, 0].numpy()
            r_p = StreamInputBuilder(dict(cfg, transform_target="parvo"), age_months=age)(x3_1)[1][0, 0].numpy()
            check(tiles is not None and len(calls) == 2 and np.array_equal(tiles[3], r_m)
                  and np.array_equal(tiles[4], r_p),
                  f"B {run} fig1_model panel (b) tiles == StreamInputBuilder (magno, and the "
                  f"same diet on L-M)" + (f" ({calls})" if tiles is None else ""))

    # D: the BA run, usual and crossed flags, against HEAD
    ba_ck = os.path.join(BA_DIR, "checkpoint_last.pt")
    ba_cfg = torch.load(ba_ck, map_location="cpu", weights_only=False).get("config")
    print(f"\n=== dkl_bcs_seed0 (BA): checkpoint config {ba_cfg}", flush=True)
    for label, flags, old in (("usual", [], {}),
                              ("crossed no-front-end", ["--no-lum-blur", "--no-lum-cs"],
                               dict(lum_blur=False, lum_cs=False))):
        for name, ep in eps.items():
            kind, rec = ep(ba_ck, flags, f"ba_{label[:5]}")
            if kind != "call":
                check(False, f"D BA {label} {name}: no front-end call ({kind}: {rec!r})")
                continue
            h = head.magno_parvo_inputs(rec["rgb"], rec["ch_min"], rec["ch_max"], **old)
            check(same(rec["out"], h), f"D BA {label} {name} input == HEAD magno_parvo_inputs"
                                       f"({old or 'defaults'}) ({maxdiff(rec['out'], h)})")
    rec, note, ok = generality("dkl_bcs_seed0", None)
    h = head.magno_parvo_inputs(rec["rgb"], rec["ch_min"], rec["ch_max"]) if rec else None
    check(rec is not None and same(rec["out"], h) and ok,
          f"D BA generality concat input == HEAD defaults; {note}")
    (kind, rec), note, ok = nonlinear(BA_DIR, "checkpoint_last.pt")
    h = head.magno_parvo_inputs(rec["rgb"], rec["ch_min"], rec["ch_max"]) if kind == "call" else None
    check(kind == "call" and same(rec["out"], h) and ok,
          f"D BA nonlinear_readout_features input == HEAD defaults; {note}")
    t_run, e_run = fig1(BA_DIR)
    t_def, e_def = fig1(None)
    check(t_run is not None and t_def is not None and np.array_equal(t_run[3], t_def[3])
          and np.array_equal(t_run[4], t_def[4]),
          "D BA fig1_model FIG1_RUN=<dkl_bcs_seed0 dir> tiles == the unchanged default tiles"
          + ("" if t_run is not None and t_def is not None else f" ({e_run} {e_def})"))
    if ba_cfg is not None and "input_mode" in ba_cfg:
        r = dkl_frontend.check_frontend(ba_ck, lum_blur=False, lum_cs=False)
        check(r is not None and r["lum_blur"], "A BA check_frontend keeps a crossed front end (reported, not refused)")
    if os.path.exists(THESIS):
        check(dkl_frontend.checkpoint_frontend(THESIS) is None,
              "A thesis Lightning checkpoint has no condition config -> None (caller's flags)")

    # E: untrained anchors
    dvd_cfg = torch.load(os.path.join(FS2, "fs2_dvd300_avg_w050_ls_seed0", "checkpoint_final.pt"),
                         map_location="cpu", weights_only=False)["config"]
    print("\n=== untrained anchors", flush=True)
    for name in ("stream_features", "stream_shape_texture", "cue_conflict"):
        kind, rec = eps[name](None, ["--dvd-age-months", "300"], "untrained_dvd300")
        r = builder_ref(dvd_cfg, 48, rec) if kind == "call" else None
        check(kind == "call" and same(rec["out"], r),
              f"E untrained {name} --dvd-age-months 300 == dvd300 StreamInputBuilder"
              + (f" ({maxdiff(rec['out'], r)})" if r is not None else f" ({kind}: {rec!r})"))
        kind, rec = eps[name](None, [], "untrained_bcs")
        h = head.magno_parvo_inputs(rec["rgb"], rec["ch_min"], rec["ch_max"]) if kind == "call" else None
        check(kind == "call" and same(rec["out"], h), f"E untrained {name} default == HEAD default")

    print(f"\n{len(fails)} failure(s)", flush=True)
    for m in fails:
        print("  " + m)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
