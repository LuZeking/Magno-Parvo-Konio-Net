"""Two core figures for the BA-vs-fusion slide.

A: MiniEcoset val top-1 vs cue-conflict shape bias - thesis (BA) models vs
   Stage-1 fusion arms (final epoch 49, seed 0) and the A0 averaging w-sweep.
B: stream probe bias (shape acc - texture acc): the BA's Table 1 (ungrouped
   PCA probe) vs the leak-free probe (readout, photos + textures held out) on
   the BA model, our fusion arms and an untrained net.

Inputs: slide_points.json (slide_extract.py) and summarize_all20.txt (grouped
probe summarize.py output), both in this script's folder.
"""
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# inputs and outputs live next to this script (slide_points.json was copied by
# hand from fs1_decide_all.py output, because history.json reads stalled on
# Lustre on 2026-10-12)
T = OUT = os.path.dirname(os.path.abspath(__file__))

INK, MUTED, GRID, BG = '#1B2333', '#4A5568', '#E4E2DC', '#FAF9F5'
BLUE, ORANGE, TEAL = '#2A5C9A', '#C66A1B', '#2A9D8F'
MAGNO, PARVO = '#5B4B8A', '#C9A227'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
})


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=250, bbox_inches='tight', pad_inches=0.08)
    print('wrote', name)


# ---------------- A: accuracy vs shape bias ----------------
P = json.load(open(f'{T}/slide_points.json'))
# thesis (BA_Manocchio_Benito.pdf): top-1 from Fig. 4 labels, SB decoded from Fig. 6 markers
BA = [('RGB ResNet-50 baseline', 0.757, 0.494), ('RGB TSN', 0.721, 0.534),
      ('DVD baseline', 0.672, 0.833), ('DVD TSN', 0.672, 0.802),
      ('DKL Base', 0.673, 0.424), ('BA TSN (DKL B&CS)', 0.469, 0.678),
      ('Inter. B&CS', 0.611, 0.609), ('M-heavy B&CS', 0.524, 0.756),
      ('M-driven SSN', 0.354, 0.927)]
# label text, xy of the text, alignment (None = no label)
LABEL_BA = {'RGB ResNet-50 baseline': ('RGB ResNet-50', (0.505, 0.783), 'left'),
            'DVD baseline': ('DVD TSN & baseline', (0.818, 0.702), 'center'),
            'BA TSN (DKL B&CS)': ('BA TSN (DKL B&CS)', (0.690, 0.462), 'left'),
            'M-driven SSN': ('M-driven SSN', (0.870, 0.322), 'right')}
DIET = ['fs1_bcs_add_w050', 'fs1_bcs_add_w070', 'fs1_bcs_add_w085']
HSF = ['fs1_hsf_add_w050', 'fs1_hsf_add_w070', 'fs1_hsf_add_w085']

fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
sw = sorted((p for p in P['fs1_bcs_avg']['sweep']['tied'] if p['sb'] is not None),
            key=lambda p: p['w'])
print('A0 tied sweep', [(p['w'], round(p['top1'], 3), round(p['sb'], 3)) for p in sw])
ax.plot([p['sb'] for p in sw], [p['top1'] for p in sw], color=BLUE, lw=2, ls=(0, (4, 3)),
        zorder=2, label='Ours: BA averaging rule, w swept')
for name, acc, sb in BA:
    ax.scatter(sb, acc, marker='s', s=70, color=ORANGE, edgecolor=BG, lw=1.5, zorder=3)
    if name in LABEL_BA:
        text, xy, ha = LABEL_BA[name]
        ax.text(*xy, text, ha=ha, va='center', fontsize=13, color=INK)
        if name == 'M-driven SSN':
            ax.annotate('', xy=(sb - 0.008, acc - 0.006), xytext=(xy[0] + 0.004, xy[1] + 0.006),
                        arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
ax.scatter([], [], marker='s', s=70, color=ORANGE, label='BA models (150 ep)')
a0 = P['fs1_bcs_avg']
ax.scatter(a0['sb'], a0['top1'], marker='o', s=80, color=BLUE, edgecolor=BG, lw=1.5, zorder=4)
for r in DIET:
    ax.scatter(P[r]['sb'], P[r]['top1'], marker='o', s=80, color=BLUE, edgecolor=BG, lw=1.5, zorder=4)
ax.scatter([], [], marker='o', s=80, color=BLUE, label='Ours: own head per stream')
for r in HSF:
    ax.scatter(P[r]['sb'], P[r]['top1'], marker='^', s=95, color=TEAL, edgecolor=BG, lw=1.5, zorder=4)
ax.scatter([], [], marker='^', s=95, color=TEAL, label='Ours: + luminance into parvo')
for r in ['fs1_bcs_avg'] + DIET + HSF:
    print(f"{r:20s} top1 {P[r]['top1']:.3f} SB {P[r]['sb']:.3f} cov {P[r]['cover']:.3f}")
# recipe effect at matched SB: BA TSN -> A0
ax.annotate('', xy=(a0['sb'], a0['top1'] - 0.012), xytext=(0.678, 0.469 + 0.012),
            arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6, mutation_scale=16))
ax.annotate(f"+{100 * (a0['top1'] - 0.469):.0f} pp\nsame shape bias",
            ((0.678 + a0['sb']) / 2 - 0.012, 0.565), ha='right', va='center', fontsize=13, color=INK)
ax.set_xlim(0.38, 0.96)
ax.set_ylim(0.30, 0.80)
ax.set_xlabel('Shape bias (cue-conflict)')
ax.set_ylabel('Top-1 accuracy (MiniEcoset val)')
ax.legend(loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=12, frameon=False,
          handlelength=1.6, columnspacing=1.2, borderaxespad=0.2)
save(fig, 'figA_accuracy_vs_shape_bias')

# ---------------- B: stream probe bias ----------------
txt = open(f'{T}/summarize_all20.txt').read()


def table(space):
    block = txt.split(f'TABLE [{space}]')[1].split('\n\n')[0]
    rows = {}
    for line in block.splitlines()[1:]:
        name = line.split()[0]
        cells = {}
        for split in ('strat', 'per_l', 'doubl'):
            m = re.search(split + r' m [\d.]+/[\d.]+\(([+-][\d.]+)\) p [\d.]+/[\d.]+\(([+-][\d.]+)\)', line)
            cells[split] = (float(m.group(1)), float(m.group(2)))
        rows[name] = cells
    return rows


pca, ro = table('pca'), table('readout')
MOR = ['dkl_bcs_seed0_trained', 'dkl_bcs_seed1_trained', 'dkl_bcs_seed2_trained']
FUS = ['fs1_bcs_avg_seed0', 'fs1_bcs_add_w050_seed0', 'fs1_bcs_add_w070_seed0',
       'fs1_bcs_add_w085_seed0', 'fs1_hsf_add_w050_seed0', 'fs1_hsf_add_w070_seed0',
       'fs1_hsf_add_w085_seed0', 'fs1_hsf_hinge_k3_l2_seed0', 'fs1_hsf_hinge_k3_l4_seed0']


def stat(rows, names, split, i):
    v = [rows[n][split][i] for n in names]
    return sum(v) / len(v), min(v), max(v)


groups = [
    ('BA Table 1\n(published probe)', [stat(pca, MOR, 'strat', i) for i in (0, 1)]),
    ('BA TSN\nleak-free', [stat(ro, MOR, 'doubl', i) for i in (0, 1)]),
    ('Our fusion\nleak-free', [(ro['fs1_bcs_add_w070_seed0']['doubl'][i],) + stat(ro, FUS, 'doubl', i)[1:]
                               for i in (0, 1)]),
    ('Untrained net\nleak-free', [(ro['untrained_blur_cs']['doubl'][i],) * 3 for i in (0, 1)]),
]
for g, s in groups:
    print(g.replace('\n', ' '), 'magno %.3f [%.2f, %.2f]' % s[0], 'parvo %.3f [%.2f, %.2f]' % s[1])

fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.grid(True, axis='y', color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.axhline(0, color=MUTED, lw=1)
bw = 0.36
for gi, (gname, s) in enumerate(groups):
    for si, (col, lab) in enumerate(((MAGNO, 'magno stream'), (PARVO, 'parvo stream'))):
        x = gi + (si - 0.5) * (bw + 0.04)
        mean, lo, hi = s[si]
        ax.bar(x, mean, width=bw, color=col, edgecolor=BG, lw=2, label=lab if gi == 0 else None, zorder=3)
        if hi > lo:
            ax.plot([x, x], [lo, hi], color=INK, lw=1.4, zorder=4)
        off = 0.035 if mean > 0 else -0.035
        edge = hi if mean > 0 else lo
        ax.text(x, edge + off, f'{mean:+.2f}'.replace('-', '−'), ha='center',
                va='bottom' if mean > 0 else 'top',
                fontsize=13, color=INK)
ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g for g, _ in groups], fontsize=12.5, color=INK)
ax.set_ylim(-0.72, 1.02)
ax.set_xlim(-0.55, 3.55)
ax.set_ylabel('Probe bias (shape − texture)')
ax.legend(loc='upper right', fontsize=12.5, frameon=False, ncol=2, borderaxespad=0.1)
save(fig, 'figB_stream_probe_bias')
