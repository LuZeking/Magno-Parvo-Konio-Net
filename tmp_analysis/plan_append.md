
**2026-08-28 — GROUP RESULT, ALL 8 SUBJECTS. Track A core analysis complete.**
Array 26368287 finished the remaining five subjects overnight; all 8
`untrained_seed0_subj*.npz` exist. Coverage is uniform (12.7k–17.9k voxels,
8.5k–13.5k with NC>0.1, median R²_joint/NC 0.11–0.17), so no subject is
carried by a degenerate fit. Analyses in `tmp_analysis/{group_interaction,
ecc_gradient,visualrois}.py`; unit of inference is the subject throughout.

**A2 — pre-declared stream × cortical-axis interaction (n=8).**
Difference-of-differences, dorsal{midparietal,parietal} minus
ventral{midventral,ventral}: **+0.0266**, sign-flip permutation
**p = .0079**, **8/8 subjects positive**, **dz = +1.64**. With 8 subjects
there are only 2⁸ = 256 sign patterns, so p = .0079 is the second-smallest
attainable value — the test is at its resolution limit, not weakly
significant. Per-ROI simple effects (FDR q=.05): early **+0.052** (p=.008,
survives), midparietal **+0.030** (p=.008, survives), midventral **−0.023**
(p=.016, survives), lateral +0.021 (p=.039, does not survive FDR);
midlateral, ventral, parietal null.

*The double dissociation is incomplete and must be reported as such.* On the
pooled ROI sets the dorsal simple effect is significant (median +0.018,
p=.016) but the ventral one is not (median −0.012, **p=.078**). Per
Nieuwenhuis et al. (2011) the interaction is the valid claim; "magno→dorsal
AND parvo→ventral" as a double dissociation is not yet licensed.

**A3.1 — eccentricity gradient (n=8). This is the strongest panel, as §1
predicted, and it is now the headline candidate.**
Median SPI by `prf-eccrois` band, averaged over subjects:

| ecc band | 0.5° | 1° | 2° | 4° |
|---|---|---|---|---|
| mean SPI | **−0.449** | −0.065 | +0.211 | **+0.326** |

A full **sign reversal**: parvo-dominant at the fovea, magno-dominant in the
periphery. Per-subject slope of SPI on log-eccentricity: mean **+0.322**,
sign-flip **p = .0079**, Wilcoxon p=.0078, **dz = +1.77**, **8/8 subjects
positive**, and monotonic in every subject. Companion contrast
SPI(4°)−SPI(0.5°) = **+0.775**, 8/8 positive. This is the textbook M/P
retinotopic signature recovered from a model that was never fitted to
retinotopy.

**A3.3 — retinotopic areas V1→hV4 (n=8).**
hV4 is the **only** area where parvo-unique variance exceeds magno-unique
(mean SPI **−0.223**); every other area is magno-preferring. SPI(hV4) −
SPI(V1) = **−0.538**, **0/8 subjects positive**, p=.0078.
Mechanism, stated precisely because it is not what the naive prediction says:
magno-unique variance **falls monotonically** along the hierarchy (V1v 0.207
→ hV4 0.055, a 3.8× drop) while parvo-unique is roughly **flat** (0.111 →
0.087). The crossover is magno *losing* explanatory power, not parvo gaining
it. Matched-eccentricity control — dorsal vs ventral bank of V1/V2/V3 —
is **null** (+0.091, p=.109, 6/8), so the hV4 effect is not a generic
ventral-stream effect.

**THE CONTROL THAT DECIDES WHETHER ANY OF THIS IS PUBLISHABLE.**
All three results are consistent with a deflationary account: the magno
stream is low-pass by construction (blur σ=4 + 48-month CSF cut), so it
should predict low-SF, peripheral, non-chromatic voxels better *for trivial
reasons*, with no biology involved. Job **26373186** (array 0-7%3) reruns
the identical pipeline and the identical untrained network with
`--no-lum-blur --no-lum-cs`, i.e. magno = raw luminance, parvo = raw L−M.
- If the eccentricity gradient **survives**, it is carried by
  achromatic-vs-chromatic channel content — the real biological claim.
- If it **collapses**, the gradient is a restatement of the blur and must be
  demoted from a finding to a front-end validation.
Nothing in this section should be written up before that array returns.

Further caveats carried forward: the network is **untrained**, so these are
input-statistics-plus-architecture effects, not learned representations —
the trained comparison is what licenses any claim about learning; and
`prf-eccrois` bands are coarse (4 rings), so the gradient is a trend, not a
continuous pRF fit.

**2026-08-28 — GPU throughput is a blocker for Track B.**
Timing probe 26368283 on an A5000: 60 steps at bs=256 in 270.8 s ≈ **57
img/s** → ~1.3 h/epoch → **~8 days for one 150-epoch run**, which makes a
seed × condition array impossible as designed. Two candidate causes, being
separated by profiler 26373188 (h5 dataloader vs the per-batch DVD front end
vs model fwd/bwd, at two worker counts): the probe ran on g017 alongside
three of my other jobs, and `StreamInputBuilder` calls
`apply_scale_free_contrast` on **every batch** inside the training loop.
If the DVD front end dominates, the fix is to precompute the transform —
the schedule is fixed at 48 months, so the transform is a constant function
of the image and can be baked into the h5 exactly as the DKL conversion was.
