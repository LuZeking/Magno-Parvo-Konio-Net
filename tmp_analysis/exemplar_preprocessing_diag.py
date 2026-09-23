"""Exploratory eight-preprocessing test of the exemplar-balanced probe (docs/PLAN_EXEMPLAR_2026-09-21.md,
"Note after the one-network test runs": "A preprocessing sensitivity, found post hoc").

Deposited 2026-09-23 as run on 2026-09-21 on tv_resnet18 and hub_resnet50_sin (three draws each); the body
below is unchanged from the run. The plan named /scratch/byrong/mpk/review_v7/scripts/ as its location, but it
was kept in a session scratch directory; this file replaces that pointer.

  python tmp_analysis/exemplar_preprocessing_diag.py tv_resnet18 hub_resnet50_sin

Variants: PCA on all images (k = 16, 64, 256, standardised; 256 unstandardised), raw standardised features,
PCA inside the folds (k = 16, 64, n - 1). Q1b is 'all256'; Q1 is 'in-n1' with k capped at 256.
"""
import sys, re, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
F = '/scratch/byrong/mpk/generality/features/2026-09-15_zoo/'
def folds(lab, ex, rng):
    f = np.empty(len(lab), int)
    for c in np.unique(lab):
        e = rng.permutation(sorted(set(ex[lab == c])))
        for i, v in enumerate(e): f[ex == v] = i % 3
    return f
def acc(X, lab, f, mode, k, std=True):
    out = []
    for kk in range(3):
        tr, te = f != kk, f == kk
        Xtr, Xte = X[tr], X[te]
        if mode == 'inside':
            kc = min(k, Xtr.shape[0]-1, Xtr.shape[1]); p = PCA(kc, random_state=0).fit(Xtr); Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        if std:
            s = StandardScaler().fit(Xtr); Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        c = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, lab[tr])
        out.append(c.score(Xte, lab[te]))
    return np.mean(out)
for tag in sys.argv[1:]:
    t0 = time.time()
    z = np.load(F + tag + '_cue_conflict.npz', allow_pickle=True)
    y = z['features'].astype(np.float64); sh = np.array([str(s) for s in z['shapes']]); tx = np.array([str(s) for s in z['textures']])
    m = [re.search(r'([a-z]+)(\d+)-([a-z]+)(\d+)', str(s)) for s in z['stems']]
    obj = np.array([a.group(1)+a.group(2) for a in m]); tex = np.array([a.group(3)+a.group(4) for a in m])
    P = {k: PCA(k, random_state=0).fit_transform(y) for k in (16, 64, 256)}
    res = {}
    for d in range(3):
        rng = np.random.default_rng(d)
        keep = np.zeros(len(sh), bool)
        for c in np.unique(sh):
            keep |= np.isin(obj, rng.choice(sorted(set(obj[sh == c])), 3, replace=False))
        fs = folds(sh[keep], obj[keep], rng)
        tk = np.zeros(len(tx), bool)
        for c in np.unique(tx):
            n = int((keep & (sh == c)).sum()); tk[rng.choice(np.flatnonzero(tx == c), n, replace=False)] = True
        ft = folds(tx[tk], tex[tk], rng)
        V = [('all16', P[16], 'none', 0, True), ('all64', P[64], 'none', 0, True), ('all256', P[256], 'none', 0, True),
             ('all256-nostd', P[256], 'none', 0, False), ('raw-std', y, 'none', 0, True),
             ('in16', y, 'inside', 16, True), ('in64', y, 'inside', 64, True), ('in-n1', y, 'inside', 256, True)]
        for name, X, mode, k, std in V:
            a_s = acc(X[keep], sh[keep], fs, mode, k, std); a_t = acc(X[tk], tx[tk], ft, mode, k, std)
            res.setdefault(name, []).append((a_s, a_t))
    print(f'{tag}  ({time.time()-t0:.0f}s)')
    for name, v in res.items():
        v = np.array(v); print(f'   {name:13s} shape {v[:,0].mean():.3f}  texture {v[:,1].mean():.3f}  bias {np.mean(v[:,0]-v[:,1]):+.3f}', flush=True)
