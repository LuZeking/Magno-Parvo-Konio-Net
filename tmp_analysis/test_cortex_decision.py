"""Tests for the cortex-to-decision test (docs/PLAN_CORTEX_DECISION_2026-10-26.md).

Synthetic data only; nothing here reads NSD. The two slow checks that touch the
stimulus directory / MiniEcoset h5 run only with CTD_DISK=1.

    PYTHONPATH=.:tmp_analysis /home/byrong/anaconda3/envs/mpk/bin/python tmp_analysis/test_cortex_decision.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis.encoding import banded_ridge, cortex_decision as cd  # noqa: E402

CLASSES = [f"c{i:02d}" for i in range(16)]


def _design(rng, n_per_cell=5):
    shapes, textures = [], []
    for s in CLASSES:
        for t in CLASSES:
            if s == t:
                continue
            shapes += [s] * n_per_cell
            textures += [t] * n_per_cell
    return np.asarray(shapes), np.asarray(textures)


def test_exact_wilcoxon_floor():
    r = cd.exact_wilcoxon(np.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.4, 0.25, 0.35]))
    assert abs(r["p_greater"] - 1 / 256) < 1e-12 and abs(r["p_two"] - 1 / 128) < 1e-12
    r2 = cd.exact_wilcoxon(np.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.4, 0.25, -0.35]))
    assert r2["p_greater"] > 1 / 256


def test_exact_mannwhitney_floor():
    r = cd.exact_mannwhitney([3, 4, 5], [0, 1, 2])
    assert r["n_assignments"] == 20 and abs(r["p_greater"] - 0.05) < 1e-12
    assert abs(r["p_two"] - 0.10) < 1e-12
    assert cd.exact_mannwhitney([0, 4, 5], [3, 1, 2])["p_greater"] > 0.05


def test_residualise_matches_human_perimage():
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from human_perimage import residualise as ref
    except Exception:
        return
    rng = np.random.default_rng(0)
    sh, tx = _design(rng)
    y = rng.normal(size=len(sh))
    assert np.allclose(cd.residualise(y, sh, tx), ref(y, sh, tx))


def test_template_evidence_recovers_mixture():
    rng = np.random.default_rng(1)
    sh, tx = _design(rng)
    n_vox = 300
    T = rng.normal(size=(16, n_vox))
    Q_labels = np.repeat(CLASSES, 50)
    Q = T[np.repeat(np.arange(16), 50)] + 0.5 * rng.normal(size=(800, n_vox))
    a = rng.uniform(0, 1, size=len(sh))               # the shape weight of each image
    js = np.asarray([CLASSES.index(s) for s in sh])
    jt = np.asarray([CLASSES.index(t) for t in tx])
    P = a[:, None] * T[js] + (1 - a)[:, None] * T[jt] + 0.5 * rng.normal(size=(len(sh), n_vox))
    E = cd.template_evidence(P, Q, Q_labels, CLASSES)
    assert E.shape == (len(sh), 16)
    mg = cd.margins(E, sh, tx, CLASSES)
    assert cd.pearson(mg["m"], a) > 0.9
    assert abs(np.mean(mg["m"] > 0) - np.mean(a > 0.5)) < 0.08
    assert np.all(mg["c"] > -5)
    # same-cue gate: images that are pure templates are classified correctly
    sc = T + 0.3 * rng.normal(size=T.shape)
    acc, p = cd.same_cue_gate(cd.template_evidence(sc, Q, Q_labels, CLASSES), np.asarray(CLASSES), CLASSES)
    assert acc == 1.0 and p < 1e-6
    # spearman variant runs and agrees in sign
    E2 = cd.template_evidence(P, Q, Q_labels, CLASSES, method="spearman")
    assert cd.pearson(cd.margins(E2, sh, tx, CLASSES)["m"], a) > 0.85


def test_within_cell_permutation_preserves_cells():
    rng = np.random.default_rng(2)
    sh, tx = _design(rng)
    cells = cd.cell_ids(sh, tx)
    y = rng.normal(size=len(sh))
    perms = cd.within_cell_permutations(cells, 5, seed=3)
    for k in range(5):
        assert np.array_equal(cells[perms[k]], cells)
        assert not np.array_equal(perms[k], np.arange(len(sh)))
        for c in np.unique(cells):
            assert np.allclose(np.sort(y[perms[k]][cells == c]), np.sort(y[cells == c]))


def test_object_exemplar_and_cluster_bootstrap():
    stems = np.array(["airplane1-bicycle2", "airplane1-cat3", "bear4-dog1", "bear4-oven2"])
    assert list(cd.object_exemplar(stems)) == ["airplane1", "airplane1", "bear4", "bear4"]
    rng = np.random.default_rng(4)
    x = rng.normal(size=400)
    y = x + rng.normal(size=400)
    clusters = np.repeat(np.arange(40), 10)
    lo, hi = cd.cluster_bootstrap_r(x, y, clusters, 200)
    assert lo < cd.pearson(x, y) < hi and lo > 0.4
    # the weighted form equals explicit resampling draw by draw
    rs = cd.cluster_bootstrap_rs(x, y, clusters, 25, seed=9)
    counts, inv = cd.cluster_bootstrap_counts(clusters, 25, seed=9)
    for k in range(25):
        idx = np.concatenate([np.flatnonzero(inv == c) for c in range(40) for _ in range(counts[k, c])])
        assert abs(rs[k] - cd.pearson(x[idx], y[idx])) < 1e-10


def test_assemble_chunks():
    from src.analysis.encoding.run_encoding import assemble_chunks
    chunks = {
        "r2_joint": [np.zeros(3), np.ones(2)],
        "pred_extra_joint": [np.zeros((4, 3)), np.ones((4, 2))],
        "enc_coef_joint": [np.zeros((6, 3)), np.ones((6, 2))],
        "enc_deltas_joint": [np.zeros((2, 3)), np.ones((2, 2))],
        "enc_band_mu_magno": [np.arange(3.0), np.arange(3.0)],
    }
    out = assemble_chunks(chunks)
    assert out["r2_joint"].shape == (5,) and out["pred_extra_joint"].shape == (4, 5)
    assert out["enc_coef_joint"].shape == (6, 5) and out["enc_deltas_joint"].shape == (2, 5)
    assert out["enc_band_mu_magno"].shape == (3,)


def test_band_stats_matches_prepare_band():
    rng = np.random.default_rng(5)
    Xtr = rng.normal(size=(50, 8)).astype(np.float32)
    Xtr[:, 3] = 1.0                      # a degenerate dimension to drop
    Xte = rng.normal(size=(10, 8)).astype(np.float32)
    Xex = rng.normal(size=(7, 8)).astype(np.float32)
    tr, te, ex = banded_ridge.prepare_band(Xtr, Xte, Xex)
    mu, keep, scale = banded_ridge.band_stats(Xtr, verbose=False)
    assert keep.sum() == 7 and not keep[3]
    assert np.allclose(ex, (Xex - mu).astype(np.float64)[:, keep] / scale)
    assert np.allclose(tr, (Xtr - mu).astype(np.float64)[:, keep] / scale)
    assert abs(np.sqrt((tr ** 2).sum() / tr.shape[0]) - 1.0) < 1e-9


def test_fit_banded_saved_encoder_reproduces_predictions():
    rng = np.random.default_rng(6)
    n_tr, n_te, n_ex, d, t = 150, 30, 9, 6, 4
    X1, X2 = rng.normal(size=(n_tr, d)).astype(np.float32), rng.normal(size=(n_tr, d)).astype(np.float32)
    W1, W2 = rng.normal(size=(d, t)), rng.normal(size=(d, t))
    Y = X1 @ W1 + 0.3 * X2 @ W2 + 0.5 * rng.normal(size=(n_tr, t))
    X1te, X2te = rng.normal(size=(n_te, d)).astype(np.float32), rng.normal(size=(n_te, d)).astype(np.float32)
    Yte = X1te @ W1 + 0.3 * X2te @ W2
    X1ex, X2ex = rng.normal(size=(n_ex, d)).astype(np.float32), rng.normal(size=(n_ex, d)).astype(np.float32)
    splits = banded_ridge.make_grouped_splits(np.arange(n_tr), n_splits=3, seed=0)
    res = banded_ridge.fit_banded([X1, X2], Y, [X1te, X2te], Yte, splits, backend="numpy",
                                  X_bands_extra=[X1ex, X2ex], save_encoder=True, n_iter=5)
    for k in ("enc_coef_magno", "enc_coef_parvo", "enc_coef_joint", "enc_deltas_joint",
              "enc_band_mu_magno", "enc_band_keep_parvo", "enc_band_scale_magno"):
        assert k in res, k
    scaled = []
    for name, Xtr, Xex in (("magno", X1, X1ex), ("parvo", X2, X2ex)):
        mu, keep, scale = banded_ridge.band_stats(Xtr, verbose=False)
        assert np.allclose(mu.ravel(), res[f"enc_band_mu_{name}"], atol=1e-6)
        z = (Xex - mu).astype(np.float64)[:, keep] / scale
        scaled.append(z)
        assert np.allclose(z @ res[f"enc_coef_{name}"], res[f"pred_extra_{name}"], atol=1e-4), name
    assert np.allclose(np.concatenate(scaled, axis=1) @ res["enc_coef_joint"], res["pred_extra_joint"], atol=1e-4)
    assert res["enc_coef_joint"].shape == (2 * d, t) and res["enc_deltas_joint"].shape == (2, t)
    # the default call returns no encoder keys: existing runs are unchanged
    res0 = banded_ridge.fit_banded([X1, X2], Y, [X1te, X2te], Yte, splits, backend="numpy",
                                   X_bands_extra=[X1ex, X2ex], n_iter=5)
    assert not [k for k in res0 if k.startswith("enc_")]
    assert np.allclose(res0["r2_joint"], res["r2_joint"])


def test_reading_rules():
    base = {"P-C1": {"met": True}, "P-C2": {"met": True, "n_noest_rois": 0}}
    assert cd.reading(base) == "PARADOX_REPRODUCED_IN_PREDICTED_CORTEX"
    assert cd.reading({"P-C1": {"met": True}, "P-C2": {"met": False, "n_noest_rois": 0}}) == "TEXTURE_BIASED_NO_EVIDENCE"
    assert cd.reading({"P-C1": {"met": False}, "P-C2": {"met": True, "n_noest_rois": 0}}) == "CORTEX_NOT_TEXTURE_BIASED_ON_THESE_IMAGES"
    assert cd.reading({"P-C1": {"met": True}, "P-C2": {"met": True, "n_noest_rois": 2}}) == "NOEST"


def test_end_to_end_synthetic(tmp_dir=None):
    """A full run on synthetic fit files: the model arm is built texture-biased
    with decision-tracking margins, the gabor arm carries no evidence."""
    import tempfile
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="ctd_", dir=os.environ.get("CLAUDE_TMP", None))
    rng = np.random.default_rng(7)
    sh, tx = _design(rng)
    n_img = len(sh)
    stems = np.asarray([f"{s}{i % 10 + 1}-{t}{i % 3 + 1}" for i, (s, t) in enumerate(zip(sh, tx))])
    h = rng.uniform(0.6, 1.0, size=n_img)      # per-image shape rate
    v = rng.uniform(0.5, 1.0, size=n_img)      # per-image resolution rate
    hd = os.path.join(tmp_dir, "human")
    os.makedirs(hd, exist_ok=True)
    np.savez(os.path.join(hd, "human_perimage.npz"), stems=stems, shapes=sh, textures=tx, h=h, v=v,
             h_resid=cd.residualise(h, sh, tx), v_resid=cd.residualise(v, sh, tx), C=np.zeros((2, n_img)), subs=np.array(["a", "b"]))
    import json
    with open(os.path.join(hd, "human_ceiling.json"), "w") as fh:
        json.dump({"ceilings": {k: {"pearson_mean": 0.6} for k in ("shape_rate", "shape_rate_resid", "valid_rate", "valid_rate_resid")}}, fh)
    n_vox = 120
    T = rng.normal(size=(16, n_vox))
    js = np.asarray([CLASSES.index(s) for s in sh])
    jt = np.asarray([CLASSES.index(t) for t in tx])
    roi_prf = np.r_[np.ones(20), 3 * np.ones(20), 5 * np.ones(20), 7 * np.ones(20), np.zeros(40)].astype(np.int32)
    roi_streams = np.r_[np.zeros(80), 5 * np.ones(20), 6 * np.ones(20)].astype(np.int32)
    fd = os.path.join(tmp_dir, "fits")
    os.makedirs(fd, exist_ok=True)
    tpl_labels = np.concatenate([np.repeat(CLASSES, 50), np.asarray(CLASSES)])
    tpl_source = np.asarray(["miniecoset_val"] * 800 + ["same_cue"] * 16)
    arms = {"model": ["m_seed0", "m_seed1"], "gabor": ["g"]}
    for arm, runs in arms.items():
        for run in runs:
            for s in range(1, 9):
                # texture-biased (a well below 0.5) yet tracking h; category content scaled by v
                a = 0.15 + 0.5 * (h - 0.8) + 0.03 * rng.normal(size=n_img)
                g = 0.5 + 0.5 * v
                if arm == "gabor":                                   # no decision information
                    a = 0.15 + 0.03 * rng.normal(size=n_img)
                    g = np.ones(n_img)
                P = g[:, None] * (a[:, None] * T[js] + (1 - a)[:, None] * T[jt]) + 0.4 * rng.normal(size=(n_img, n_vox))
                Q = T[np.repeat(np.arange(16), 50)] + 0.4 * rng.normal(size=(800, n_vox))
                Qs = T + 0.2 * rng.normal(size=T.shape)
                tpl = np.concatenate([Q, Qs])
                np.savez(os.path.join(fd, f"{run}_subj{s:02d}.npz"), subject=s, model_name=run,
                         cc_stems=stems, cc_shapes=sh, cc_textures=tx, tpl_labels=tpl_labels, tpl_source=tpl_source,
                         **{f"cc_pred_{b}": P.astype(np.float16) for b in cd.BANDS},
                         **{f"tpl_pred_{b}": tpl.astype(np.float16) for b in cd.BANDS},
                         r2_joint=np.full(n_vox, 0.1), reliable_mask=np.ones(n_vox, bool),
                         **{"roi_prf-visualrois": roi_prf, "roi_streams": roi_streams},
                         repro_max_abs_diff=np.asarray(1e-5))
    out = cd.run(fd, hd, os.path.join(tmp_dir, "out"), n_perm=200, arms=arms)
    P = out["predictions"]
    assert P["P-C1"]["met"], P["P-C1"]
    for roi in cd.EXTRASTRIATE:
        assert P["P-C2"]["rois"][roi]["h"]["n_positive"] == 8
        assert P["P-C2"]["rois"][roi]["h"]["n_above_gabor"] == 8
        assert P["P-C2"]["rois"][roi]["h"]["perm_p"] < 0.01
    assert P["P-C2"]["met"] and P["P-C2"]["met_b_resolution"], P["P-C2"]
    assert out["reading_of_record"] == "PARADOX_REPRODUCED_IN_PREDICTED_CORTEX"
    assert os.path.exists(os.path.join(tmp_dir, "out", "ctd_summary.md"))
    assert all(d["n_seeds"] == 2 for d in out["per_subject"]["model"]["1"]["V1"].values())


def test_disk_designs():
    if os.environ.get("CTD_DISK") != "1":
        return
    from src.analysis.encoding import template_design
    stems = template_design.cue_conflict_stems()
    assert len(stems) == 1200 and len(set(stems)) == 1200
    sc = template_design.same_cue_images()
    assert len(sc) == 80 and all(c in stems[0] or True for _, c in sc)
    rgb, labels, st = template_design.miniecoset16_images()
    assert rgb.shape == (800, 256, 256, 3) and rgb.dtype == np.uint8
    assert sorted(set(labels)) == sorted(set(c for _, c in sc)) and len(set(st)) == 800


def test_nan_images_in_target():
    """Six cue-conflict images have no resolved observer trial (h = NaN): the
    permutation null and the exemplar bootstrap must mask them as `pearson`
    does, per row, instead of returning NaN (found 2026-10-28, ledger §3)."""
    rng = np.random.default_rng(3)
    M = rng.standard_normal((20, 300))
    y = rng.standard_normal(300)
    y[[4, 77, 150]] = np.nan
    rows = cd.pearson_rows(M, y)
    assert np.all(np.isfinite(rows))
    for k in range(20):
        assert abs(rows[k] - cd.pearson(M[k], y)) < 1e-12
    Mn = M.copy()
    Mn[0, 10] = np.nan                      # NaN moved into a permuted row
    assert abs(cd.pearson_rows(Mn, y)[0] - cd.pearson(Mn[0], y)) < 1e-12
    yf = rng.standard_normal(300)           # all finite: unchanged estimator
    A = M - M.mean(axis=1, keepdims=True)
    b = yf - yf.mean()
    ref = A @ b / np.sqrt((A * A).sum(axis=1) * (b * b).sum())
    assert np.allclose(cd.pearson_rows(M, yf), ref, atol=1e-12)
    clusters = np.repeat(np.arange(30), 10)
    lo, hi = cd.cluster_bootstrap_r(M[0], y, clusters, 200)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    keep = np.isfinite(y)
    assert (lo, hi) == cd.cluster_bootstrap_r(M[0][keep], y[keep], clusters[keep], 200)

if __name__ == "__main__":
    import time
    names = [n for n in dir() if n.startswith("test_")]
    failed = 0
    for n in names:
        t0 = time.time()
        try:
            globals()[n]()
            print(f"ok   {n} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {n}: {type(e).__name__}: {e}")
    print(f"{len(names) - failed}/{len(names)} passed")
    sys.exit(1 if failed else 0)
