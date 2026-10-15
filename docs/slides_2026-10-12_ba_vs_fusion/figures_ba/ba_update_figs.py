"""The BA's result figures (Manocchio thesis Figs 4-8) redrawn with the new main
TSN next to the BA TSN.

New main TSN = fs2_bcs_avg_w050_ls, seeds 0-2 (2026-10-15; replaces the
Stage-1 arm fs1_bcs_avg, which trained each stream with its own loss): the
BA's architecture, diet, averaging readout and single objective (one
cross-entropy on the network output; the per-stream heads are readouts on
detached features), trained with AdamW + cosine + EMA and label smoothing
0.1 for 50 epochs. Values are seed means; whiskers / error bars show the
seed range / SD.
BA TSN = thesis DKL B&CS, checkpoint dkl_blur_cs_dvd_trans (epoch 147).

Sources, all in the repo or next to this script:
- thesis per-model values: Fig. 4 top-1 (thesis text), Fig. 7 per-stream
  readouts (thesis text), Fig. 6 shape bias / shape accuracy
  (src/analysis/neuroai/raw-data/performances/*.csv, the thesis pipeline);
- BA TSN and new TSN on one instrument: results/cue_conflict/thesis_dvd_trans.json
  (our 1,200-image pipeline) and the Stage-2 histories (same front end, parity 0);
- BA-recipe trajectories: the paper's 150-epoch dkl_bcs re-runs, seeds 0 and 2
  (seed 1's history could not be read off Lustre on 2026-10-12);
- stream probes: ../figures/summarize_all20.txt (BA TSN) and
  ../figures/summarize_stage2.txt (new TSN, seed mean); same grouped probe.
"""
import csv
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
PERF = os.path.join(REPO, 'src/analysis/neuroai/raw-data/performances')

INK, MUTED, GRID, BG = '#1B2333', '#4A5568', '#E4E2DC', '#FAF9F5'
NEW, BA, CTX = '#2A5C9A', '#C66A1B', '#A9AFB8'
MAGNO, PARVO = '#5B4B8A', '#C9A227'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
})


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{HERE}/{name}.{ext}', dpi=250, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print('wrote', name)


# thesis label, performance-CSV name, Fig. 4 top-1
THESIS = [('RGB Baseline', 'ts_final_ss_resnet_50', 0.757),
          ('RGB TSN', 'ts_final_resnet_18', 0.721),
          ('DVD Baseline', 'ts_baseline_resnet50_DVD_Scale_Free_DVD_trans', 0.672),
          ('DVD TSN', 'ts_resnet18_dvd_final', 0.672),
          ('DKL Base', 'ts_resnet18_dkl_base', 0.673),
          ('DKL B', 'ts_resnet18_dkl_blur_final', 0.532),
          ('DKL CS', 'ts_resnet18_dkl_contrast_final', 0.585),
          ('DKL B&CS (BA TSN)', 'ts_resnet18_dkl_blur_cs_dvd_trans', 0.469),
          ('Inter. DKL B&CS', 'ts_resnet18_dkl_blur_cs_concat', 0.611),
          ('M-heavy DKL B&CS', 'ts_resnet18_dkl_blur_cs_s1_90', 0.524),
          ('Inter. M-heavy', 'ts_resnet18_dkl_blur_cs_concat_s1_90', 0.582),
          ('M-driven SSN', 'ts_resnet18_dkl_lum_blur_cs_final', 0.354),
          ('P-driven SSN', 'ts_resnet18_dkl_rg_dvd_trans', 0.445)]
BA_NAME = 'DKL B&CS (BA TSN)'


def perf(name):
    out = {}
    for r in csv.DictReader(open(f'{PERF}/{name}.csv')):
        out[f"{r['dataset_name']}|{r['metric_name']}"] = float(r['performance'])
    return out


NEW_RUNS = [f'fs2_bcs_avg_w050_ls_seed{i}' for i in range(3)]
new_hs = [json.load(open(f'{HERE}/{r}__history.json')) for r in NEW_RUNS]
finals = [h[-1] for h in new_hs]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


new_top1 = [f['val_top1'] for f in finals]
new = {'epoch': finals[0]['epoch'], 'val_top1': mean(new_top1),
       'val': {k: mean([f['val'][k] for f in finals]) for k in finals[0]['val']}}
ba_cc = json.load(open(f'{REPO}/results/cue_conflict/thesis_dvd_trans.json'))['conditions']
N = finals[0]['cc']['joint']['n']
HEADS = ('joint', 'tied_magno', 'tied_parvo', 'magno', 'parvo')
beh_seeds = [{k: dict(sb=f['cc'][k]['sb'], acc=f['cc'][k]['n_shape'] / N) for k in HEADS}
             for f in finals]
new_beh = {k: {m: mean([s[k][m] for s in beh_seeds]) for m in ('sb', 'acc')} for k in HEADS}
new_beh_sd = {k: {m: sd([s[k][m] for s in beh_seeds]) for m in ('sb', 'acc')} for k in HEADS}
print('new TSN per seed: top-1 ' + ' '.join(f'{v:.3f}' for v in new_top1)
      + '  SB ' + ' '.join(f"{s['joint']['sb']:.3f}" for s in beh_seeds))
ba_beh = {'joint': ba_cc['full'], 'magno': ba_cc['magno_only'], 'parvo': ba_cc['parvo_only']}
print(f"new TSN epoch {new['epoch']}: top-1 {new['val_top1']:.3f}  val heads "
      + ' '.join(f"{k} {v:.3f}" for k, v in new['val'].items()))
for k, v in new_beh.items():
    print(f'  new {k:10s} SB {v["sb"]:.3f} shape-acc {v["acc"]:.3f}')
for k, v in ba_beh.items():
    print(f'  BA  {k:10s} SB {v["shape_bias"]:.3f} shape-acc {v["shape_accuracy"]:.3f}')

# ---------------- Fig 4: top-1 ----------------
fig, ax = plt.subplots(figsize=(6.5, 5.6))
rows = [('New TSN (ours)', new['val_top1'])] + [(n, a) for n, _, a in THESIS]
for i, (n, a) in enumerate(rows):
    col = NEW if i == 0 else BA if n == BA_NAME else CTX
    ax.barh(i, a, height=0.68, color=col, edgecolor=BG, lw=1.5, zorder=3)
    if i == 0:
        ax.plot([min(new_top1), max(new_top1)], [0, 0], color=INK, lw=1.6, zorder=4)
    ax.text(max(a, max(new_top1) if i == 0 else a) + 0.008, i, f'{a:.3f}', va='center',
            fontsize=12.5, color=INK, fontweight='bold' if col != CTX else 'normal')
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([n for n, _ in rows], fontsize=12.5, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 0.86)
ax.grid(True, axis='x', color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_xlabel('Top-1 accuracy (MiniEcoset val)')
save(fig, 'ba_fig4_accuracy')

# ---------------- Fig 5: accuracy over training ----------------
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s, ls in ((0, '-'), (2, (0, (5, 2)))):
    h = json.load(open(f'{HERE}/dkl_bcs_seed{s}__history.json'))
    ax.plot([e['epoch'] + 1 for e in h], [e['val_top1'] for e in h], color=BA, lw=1.6, ls=ls,
            label=f'BA TSN recipe, re-run seed {s}')
ax.scatter([148], [0.469], marker='s', s=80, color=BA, edgecolor=BG, lw=1.5, zorder=4,
           label='BA TSN as published (ep 148)')
for i, h in enumerate(new_hs):
    ax.plot([e['epoch'] + 1 for e in h], [e['val_top1'] for e in h], color=NEW, lw=1.8,
            label='New TSN, seeds 0-2 (50 epochs)' if i == 0 else None)
ax.set_xlim(0, 152)
ax.set_ylim(0, 0.75)
ax.set_xlabel('Epoch')
ax.set_ylabel('Top-1 accuracy (MiniEcoset val)')
ax.legend(loc='lower right', fontsize=12, frameon=False)
save(fig, 'ba_fig5_trajectory')

# ---------------- Fig 6: shape bias vs shape accuracy ----------------
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
LAB = {'RGB Baseline': (0.006, -0.03, 'left'), 'DVD Baseline': (0.0, 0.03, 'center'),
       'DVD TSN': (-0.008, 0.0, 'right'), 'M-driven SSN': (-0.006, -0.035, 'right'),
       'DKL Base': (0.006, -0.03, 'left')}
for n, csvn, _ in THESIS:
    if n == BA_NAME:
        continue
    p = perf(csvn)
    x, y = p['cue-conflict|accuracy (top-1)'], p['cue-conflict|mean_shape_bias']
    ax.scatter(x, y, marker='s', s=60, color=CTX, edgecolor=BG, lw=1.2, zorder=3)
    if n in LAB:
        dx, dy, ha = LAB[n]
        ax.text(x + dx, y + dy, n, ha=ha, va='center', fontsize=12, color=MUTED)
ax.scatter([], [], marker='s', s=60, color=CTX, label='Other BA models (thesis values)')
b = ba_beh['joint']
ax.scatter(b['shape_accuracy'], b['shape_bias'], marker='s', s=420, facecolor='none', edgecolor=BA,
           lw=2.5, zorder=4, label='BA TSN')
ax.errorbar(new_beh['joint']['acc'], new_beh['joint']['sb'], xerr=new_beh_sd['joint']['acc'],
            yerr=new_beh_sd['joint']['sb'], fmt='o', ms=10, color=NEW, mec=BG, mew=1.2,
            elinewidth=1.6, capsize=0, zorder=5, label='New TSN (mean ± SD, 3 seeds)')
ax.set_xlabel('Shape accuracy (cue-conflict)')
ax.set_ylabel('Shape bias (cue-conflict)')
ax.set_xlim(0.18, 0.54)
ax.set_ylim(0.38, 0.98)
ax.legend(loc='upper left', fontsize=12, frameon=False, markerscale=0.8)
save(fig, 'ba_fig6_shape_bias')

# ---------------- Fig 7: per-stream linear readouts ----------------
# thesis Fig. 7 values from its text; ours = each stream's own linear head
F7 = [('RGB TSN', 0.705, 0.702, 0.721, CTX), ('DVD TSN', 0.662, 0.662, 0.672, CTX),
      ('BA TSN', 0.236, 0.456, 0.469, BA),
      ('New TSN', new['val']['magno'], new['val']['parvo'], new['val_top1'], NEW)]
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.grid(True, axis='y', color=GRID, lw=0.8)
ax.set_axisbelow(True)
bw = 0.36
for gi, (n, m, p, tot, col) in enumerate(F7):
    for si, (v, c, lab) in enumerate(((m, MAGNO, 'magno stream'), (p, PARVO, 'parvo stream'))):
        x = gi + (si - 0.5) * (bw + 0.04)
        ax.bar(x, v, width=bw, color=c, edgecolor=BG, lw=2, zorder=3, label=lab if gi == 0 else None)
        ax.text(x, v - 0.02, f'{v:.2f}', ha='center', va='top', fontsize=12.5,
                color='#FAF9F5' if c == MAGNO else INK, zorder=5)
    ax.plot([gi - 0.42, gi + 0.42], [tot, tot], color=INK, lw=2, zorder=4,
            label='whole TSN' if gi == 0 else None)
    ax.text(gi, tot + 0.015, f'{tot:.2f}', ha='center', va='bottom', fontsize=12.5, color=INK,
            fontweight='bold')
ax.set_xticks(range(len(F7)))
ax.set_xticklabels([f[0] for f in F7], fontsize=13, color=INK)
for t, f in zip(ax.get_xticklabels(), F7):
    if f[4] != CTX:
        t.set_fontweight('bold')
ax.set_ylim(0, 0.85)
ax.set_ylabel('Top-1 accuracy (MiniEcoset val)')
ax.legend(loc='upper right', fontsize=12, frameon=False, ncol=3, columnspacing=1.0,
          handlelength=1.4, bbox_to_anchor=(1.0, 1.08))
save(fig, 'ba_fig7_stream_readout')

# ---------------- Fig 8: stream-level biases ----------------
txt = open(os.path.join(HERE, '..', 'figures', 'summarize_all20.txt')).read()


def probe(space, model, split):
    block = txt.split(f'TABLE [{space}]')[1].split('\n\n')[0]
    line = next(ln for ln in block.splitlines() if ln.split()[:1] == [model])
    m = re.search(split + r' m [\d.]+/[\d.]+\(([+-][\d.]+)\) p [\d.]+/[\d.]+\(([+-][\d.]+)\)', line)
    return float(m.group(1)), float(m.group(2))


txt2 = open(os.path.join(HERE, '..', 'figures', 'summarize_stage2.txt')).read()
assert 'MAIN TSN mean over 3 seed(s)' in txt2, 'stage-2 probe summary lacks 3 seeds'


def probe_mean(space, split):
    """New TSN: seed-mean magno / parvo bias from summarize_stage2.txt."""
    main = txt2.split('MAIN TSN mean')[1].split('\n\n')[0]
    m = re.search(rf'^\s+{space}\s+{split}\s+([+-][\d.]+) / ([+-][\d.]+)', main, re.M)
    return float(m.group(1)), float(m.group(2))


P = {'BA TSN': (probe('pca', 'dkl_bcs_thesis_trained', 'strat'), probe('readout', 'dkl_bcs_thesis_trained', 'doubl')),
     'New TSN': (probe_mean('pca', 'stratified'), probe_mean('readout', 'double'))}
STREAM_SB = {'BA TSN': (ba_beh['magno']['shape_bias'], ba_beh['joint']['shape_bias'], ba_beh['parvo']['shape_bias']),
             'New TSN': (new_beh['tied_magno']['sb'], new_beh['joint']['sb'], new_beh['tied_parvo']['sb'])}
print('probe', P)
print('stream SB (magno, whole, parvo)', STREAM_SB)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.5))
# a: behavioural shape bias of each stream read alone through the shared head (thesis Fig. 8 layout)
bw = 0.26
for gi, n in enumerate(('BA TSN', 'New TSN')):
    for si, (v, c, lab) in enumerate(zip(STREAM_SB[n], (MAGNO, INK, PARVO), ('magno alone', 'whole TSN', 'parvo alone'))):
        x = gi + (si - 1) * (bw + 0.03)
        a1.bar(x, v, width=bw, color=c, edgecolor=BG, lw=1.5, zorder=3, label=lab if gi == 0 else None)
        a1.text(x, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=11.5, color=INK)
a1.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
a1.set_xticks([0, 1])
a1.set_xticklabels(['BA TSN', 'New TSN'], fontsize=13, color=INK)
a1.set_ylim(0, 1.18)
a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
a1.set_ylabel('Shape bias (cue-conflict)')
a1.legend(loc='upper center', fontsize=11, frameon=False, ncol=3, columnspacing=0.6,
          handlelength=1.0, handletextpad=0.4, bbox_to_anchor=(0.5, 1.13))
a1.grid(True, axis='y', color=GRID, lw=0.8)
a1.set_axisbelow(True)
# b: linear probe on the stream features, shape - texture accuracy
bw = 0.2
for gi, n in enumerate(('BA TSN', 'New TSN')):
    shared, held = P[n]
    vals = [(shared[0], MAGNO, 1.0), (held[0], MAGNO, 0.45), (shared[1], PARVO, 1.0), (held[1], PARVO, 0.45)]
    for si, (v, c, al) in enumerate(vals):
        x = gi + (si - 1.5) * (bw + 0.02)
        a2.bar(x, v, width=bw, color=c, alpha=al, edgecolor=c, lw=1.2, zorder=3)
        a2.text(x, v + (0.03 if v > 0 else -0.03), f'{v:+.2f}'.replace('-', '−'), ha='center',
                va='bottom' if v > 0 else 'top', fontsize=10.5, color=INK, rotation=90)
a2.axhline(0, color=MUTED, lw=1)
a2.set_xticks([0, 1])
a2.set_xticklabels(['BA TSN', 'New TSN'], fontsize=13, color=INK)
a2.set_ylim(-0.85, 1.25)
a2.set_ylabel('Probe: shape − texture accuracy')
a2.legend(handles=[Patch(facecolor='#6B7280', label='exemplars shared (as in BA)'),
                   Patch(facecolor='#6B7280', alpha=0.45, label='exemplars held out')],
          loc='upper center', fontsize=11, frameon=False, ncol=1, handlelength=1.0,
          bbox_to_anchor=(0.5, 1.13))
a2.grid(True, axis='y', color=GRID, lw=0.8)
a2.set_axisbelow(True)
save(fig, 'ba_fig8_streams')
