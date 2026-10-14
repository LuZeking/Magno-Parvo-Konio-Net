"""Grouped stream probe for the Stage-2 runs (main TSN seeds, DVD-diet arms),
in the row format of the 2026-10-11 summary (summarize.py there), with the BA
TSN row from the 2026-10-11 JSONs as the reference. numpy-free.

usage: python3 -I summarize_grouped_probe_stage2.py <stage2 json dir> <2026-10-11 json dir> <out.txt>
"""
import json
import os
import sys

J2, J1, OUT = sys.argv[1:4]
SPLITS = ('stratified', 'per_label', 'double')
MAIN = [f'fs2_bcs_avg_w050_ls_seed{i}' for i in range(3)]
DVD = ['fs2_dvd48_avg_w050_ls_seed0', 'fs2_dvd300_avg_w050_ls_seed0']
REF = 'dkl_bcs_thesis_trained'

R = {}
for n in MAIN + DVD:
    p = os.path.join(J2, n + '.json')
    if os.path.exists(p):
        R[n] = json.load(open(p))
R[REF] = json.load(open(os.path.join(J1, REF + '.json')))
lines = [f'{len(R) - 1}/{len(MAIN + DVD)} stage-2 results; missing: '
         f'{[n for n in MAIN + DVD if n not in R] or "none"}', '']

lines.append('GATE 2 (16-way decision agreement with logged logits; 10-11: fails by bf16 precision)')
for n in MAIN + DVD:
    if n in R:
        lc = R[n]['logit_check']
        lines.append(f'  {n:32s} magno {lc["magno"]["decision_agreement_16way"]:.4f}  '
                     f'parvo {lc["parvo"]["decision_agreement_16way"]:.4f}')


def row(n, space):
    s = R[n]['splits'][space]
    return '  '.join(
        f'{sp[:5]} ' + ' '.join(f'{st[0]} {s[sp][st]["shape_acc"]:.2f}/{s[sp][st]["texture_acc"]:.2f}({s[sp][st]["bias"]:+.2f})'
                                for st in ('magno', 'parvo', 'both'))
        for sp in SPLITS)


for space in ('pca', 'readout'):
    lines += ['', f'TABLE [{space}]  shape/texture (bias) per stream: m=magno p=parvo b=both']
    for n in [REF] + MAIN + DVD:
        if n in R:
            lines.append(f'  {n:36s} {row(n, space)}')

have = [n for n in MAIN if n in R]
if have:
    lines += ['', f'MAIN TSN mean over {len(have)} seed(s): bias magno / parvo / magno - parvo']
    for space in ('pca', 'readout'):
        for sp in SPLITS:
            b = {st: sum(R[n]['splits'][space][sp][st]['bias'] for n in have) / len(have)
                 for st in ('magno', 'parvo')}
            lines.append(f'  {space:8s} {sp:10s} {b["magno"]:+.3f} / {b["parvo"]:+.3f} / '
                         f'{b["magno"] - b["parvo"]:+.3f}')

lines += ['', 'READING 5 (pca/both: retains texture = double texture >= 0.1875; texture-dominant = t > s under both grouped splits)']
for n in [REF] + MAIN + DVD:
    if n in R:
        s = R[n]['splits']['pca']
        t_dbl = s['double']['both']['texture_acc']
        dom = all(s[sp]['both']['texture_acc'] > s[sp]['both']['shape_acc'] for sp in ('per_label', 'double'))
        lines.append(f'  {n:36s} double texture {t_dbl:.3f} '
                     f'{"retains" if t_dbl >= 0.1875 else "below 3x chance"}  '
                     f'{"TEXTURE-DOMINANT" if dom else "not texture-dominant"}')

open(OUT, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('wrote', OUT)
