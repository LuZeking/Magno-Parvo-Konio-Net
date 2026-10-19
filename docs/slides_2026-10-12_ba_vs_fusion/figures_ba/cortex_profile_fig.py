"""Cortex figure for the main TSN (dvd300) beside the BA TSN (PAPER_v2 Fig. 3 /
ED Fig. 5 measures), from ../figures/fs2_cortex_summary.json, which
tmp_analysis/fs2_cortex_summary.py writes.

a  Stream dominance along V1v -> hV4: per-ROI median (Um - Up)/(Um + Up) of the
   banded encoding (+1 = only the magno stream adds unique variance, -1 = only
   parvo). Points = mean over the 8 NSD subjects of each subject's seed mean;
   bars = SEM over subjects.
b  Cortex-optimal readout weight w* (magno share of z = w h_M + (1 - w) h_P that
   best predicts the voxels) in V1 and in extrastriate cortex (mean of ventral,
   lateral, dorsal), per arm; mean +- SEM over subjects.

Arms: BA TSN = thesis diet (dkl_bcs, 3 seeds) and its no-diet twin (dkl_base);
new TSN = fs2_dvd300_avg_w050_ls (3 seeds); control = fs2_dvdfix300_avg_w050_ls
(the new TSN's adult diet from the first step).

Usage: python3 cortex_profile_fig.py   (writes cortex_profile_dvd300.{png,pdf})
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'figures', 'fs2_cortex_summary.json')

INK, MUTED, GRID, BG = '#1B2333', '#4A5568', '#E4E2DC', '#FAF9F5'
NEW, BA = '#2A5C9A', '#C66A1B'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
})
# arm key in the summary -> label, colour, filled marker?, line style
ARMS = [('BA diet', 'BA TSN', BA, True, '-'),
        ('BA no diet', 'BA TSN without diet', BA, False, '--'),
        ('dvd300', 'New TSN', NEW, True, '-'),
        ('control', 'New TSN, adult diet from the start', NEW, False, '--')]
ROIS = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']

js = json.load(open(SRC))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={'width_ratios': [1.35, 1]})

x = np.arange(len(ROIS))
for key, lab, c, filled, ls in ARMS:
    p = np.array(js['arms'][key]['profile_seedmean_per_subject'], float)
    n = np.isfinite(p).sum(0)
    m, se = np.nanmean(p, 0), np.nanstd(p, 0, ddof=1) / np.sqrt(n)
    a1.errorbar(x, m, yerr=se, color=c, ls=ls, lw=2.4, marker='o', ms=8,
                mfc=c if filled else BG, mec=c, mew=2, capsize=0, elinewidth=1.4,
                label=lab, zorder=3)
a1.axhline(0, color=MUTED, lw=1, zorder=1)
a1.set_xticks(x)
a1.set_xticklabels(ROIS)
a1.set_ylim(-1.05, 1.18)
a1.set_yticks([-1, -0.5, 0, 0.5, 1])
a1.set_ylabel('Stream dominance, (Um − Up) / (Um + Up)')
a1.text(-0.35, 1.17, 'magno adds more', color=MUTED, fontsize=12, va='top')
a1.text(-0.35, -1.0, 'parvo adds more', color=MUTED, fontsize=12, va='bottom')
a1.grid(axis='y', color=GRID, lw=1)
a1.legend(loc='upper right', bbox_to_anchor=(1.0, 0.70), frameon=False, fontsize=12)
a1.text(0.0, -0.13, 'mean ± SEM over 8 NSD subjects (each the mean of its seeds)', transform=a1.transAxes,
        color=MUTED, fontsize=11)
a1.set_title('a  Which stream predicts each visual area', loc='left', fontsize=15, color=INK)

R = js['readout_w']
for i, (key, lab, c, filled, ls) in enumerate(ARMS):
    r = R[key]
    v1, ext = np.array(r['w_v1']), np.array(r['w_ext'])
    for k, (v, dx) in enumerate(((v1, -0.14), (ext, 0.14))):
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        a2.errorbar(i + dx, m, yerr=se, color=c, marker='o' if k == 0 else 's', ms=9,
                    mfc=c if filled else BG, mec=c, mew=2, capsize=0, elinewidth=1.4, zorder=3)
    a2.annotate('', xy=(i + 0.14, ext.mean()), xytext=(i - 0.14, v1.mean()),
                arrowprops=dict(arrowstyle='->', color=c, lw=2, ls=ls, shrinkA=9, shrinkB=9), zorder=2)
a2.axhline(0.5, color=MUTED, lw=1, ls=':', zorder=1)
a2.set_xticks(range(len(ARMS)))
a2.set_xticklabels(['BA TSN', 'BA TSN\nno diet', 'New TSN', 'New TSN\nadult diet\nfrom start'], fontsize=12)
a2.set_ylim(0, 1)
a2.set_ylabel('Cortex-optimal magno weight w*')
a2.grid(axis='y', color=GRID, lw=1)
a2.plot([], [], 'o', color=MUTED, ms=8, label='V1')
a2.plot([], [], 's', color=MUTED, ms=8, label='extrastriate')
a2.legend(loc='upper left', frameon=False, fontsize=12, ncol=2)
a2.set_title('b  Readout weight that best predicts cortex', loc='left', fontsize=15, color=INK)

fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{HERE}/cortex_profile_dvd300.{ext}', dpi=250, bbox_inches='tight', pad_inches=0.08)
print('wrote cortex_profile_dvd300')
