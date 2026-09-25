# TDANN magno-likeness probe: pre-registration (2026-09-25)

**Question.** In Finzi et al.'s topographic network (TDANN, SimCLR, alpha = 0.25), are the
units that win **dorsal** (and **lateral**) stream voxels in the 1-to-1 assignment more
magno-like than the units that win **ventral** voxels?

**Why.** This is the no-training gate for training TSN with the TDANN spatial loss (a
Paper-2 idea). If streams that emerge in a topographic network already carry M-like
input tuning, a TSN + spatial-loss hybrid has a positive claim to test (explicit M/P
inputs vs. emergent streams). If they do not, the hybrid has no claim to lead with and is
not trained.

**Registered before any data exist.** At commit time no probe responses have been computed
and no voxel-to-unit assignment file has been written. The 16-image stimulus check below
touches images only.

Code: `src/analysis/finzi_1to1/magno_probe.py`; assignment output added to `mapping.py`.

## 1. Networks
- **Primary:** TDANN SimCLR alpha = 0.25, seeds 0-4.
  - Seed 0 is Finzi OSF qy32x `simclr_a025_seed0`.
  - Seeds 1-4 are Margalit et al. OSF 64qv3 `isoswap_3_seed_{1..4}`.
  - All five load fully into a torchvision ResNet-18, and their weights differ pairwise.
- **Descriptive:** alpha = 0, seed 0 (no spatial loss).
- **Units:** the output of `layer4.1`, 512 x 7 x 7 = 25,088 units. Images are resized to 224 px and ImageNet-normalised, as in `extract.py`.

## 2. Assignment
This is Finzi's all-units rectangular Hungarian assignment (cost = 1 - train Pearson r), run
per hemisphere per subject, exactly as in the idea-1 pipeline (`run_main`,
`finzi_all_units`). Streams are the NSD `streams` ROIs, restricted to voxels with NC >= 10%.

The only change to `mapping.py` is that it now also saves the unit that won each voxel
(`<net>__subjXX_assign.npz`); `grp` and `r_test` are computed exactly as before.

**Synthetic end-to-end rerun** (home env), compared with Amendment 1 (scratch env):

| check | this rerun | Amendment 1 |
|---|---|---|
| planted primary | +0.410, p = .008 | +0.42 |
| null network | NO PREFERENCE [-0.000, +0.026] | [+0.004, +0.031] |
| PC-TDANN | PASS | PASS |
| PC-TSN | PASS | PASS |
| NEG | PASS [-0.019, +0.008] | PASS |

The small shifts come from the environment change, not from the edit, which only adds an
output.

**Synthetic check of this probe's stats path** (tmp `synth_probe.py`): PASS.
- A signal planted on dorsal-winning units is MAGNO-LIKE: +1.0 SD on both indices, 5/5 seeds agree, GO.
- A null network is NOT MAGNO-LIKE (NO GO).
- A broken instrument forces UNRESOLVED.
- An effect in one seed only is labelled SEED-DEPENDENT (UNRESOLVED).

## 3. Stimuli
The NSD shared-515 images, each rendered five ways:

- **orig**: the original image.
- **lum** (luminance-only): each pixel's L+M cone contrast is kept, with equal L, M and S
  contrast. The colour model is the TSN front end's own sRGB -> XYZ -> LMS(HPE) conversion,
  with contrast taken against mid-grey.
- **iso** (isoluminant): the exact remainder, i.e. the L-M and S-opponent contrast at
  zero L+M contrast. In cone contrast, lum + iso = orig.
  - Chroma rule: iso uses the largest scale in {1, 0.75, 0.5, 0.35, 0.25} whose rendered
    pixels go out of gamut in at most 1% of cases across all 515 images. The same scale
    applies to every image.
- **low**: a Gaussian low-pass filter with its half-amplitude at 12 cycles/image. This
  equals the TSN magno blur (sigma 4 px at 256 px); at 425 px, sigma = 6.64 px.
- **high**: orig - low + the image mean.

Stimulus check on the first 16 images:
- The DKL values match `convert_rgb_to_dkl_numpy` exactly, and the inverse round-trips to
  within 3e-4.
- Through TSN's front end, iso keeps 0.9% of the original luminance contrast and lum keeps
  2.3% of the original L-M contrast.
- The chroma rule picked 0.35 on these 16 images; the full set is re-decided by the rule.
- Share of power above the cut: low 2% and high 94%, versus 19% in the original.

## 4. Indices (per unit)
`sd_c` is the SD over the 515 images of a unit's response in condition c.

- ACH = sd_lum / (sd_lum + sd_iso): achromatic preference (magno -> 1).
- LSF = sd_low / (sd_low + sd_high): low-spatial-frequency preference (magno -> 1).
- If both SDs are 0, the index is NaN.

**Primary form:** position-residualised, i.e. each of the 7 x 7 grid positions has its own
mean over units subtracted. The result is then divided by its SD over units. This removes
any effect of *where* in the image a unit looks (for example, periphery vs centre), and
keeps what the unit is tuned to. **Secondary form:** globally mean-centred and scaled the
same way.

## 5. Per-subject statistic
For network k and subject s:

- Delta^DV_{s,k} = the NC-quintile-stratified mean index of the units assigned to dorsal
  voxels, minus the same for ventral voxels.
  - Stratification uses `mapping.strat_delta`: quintiles of NC over the assigned voxels,
    and only bins with at least 20 voxels per stream count.
  - Hemispheres are pooled.
  - Voxels whose unit has a NaN index are dropped (the count is reported).
- Delta^LV is the same for lateral minus ventral.

**Aggregation:**
- Delta_s = the mean over the 5 seeds, per subject.
- Seed means m_k = the mean over subjects, per seed.

## 6. Decision rule per index
Inference is on n = 8 subjects, with SESOI delta = 0.2 SD. Each index gets an exact
two-sided Wilcoxon test and a 90% t-interval.

| label | condition |
|---|---|
| MAGNO-LIKE | p < .05, mean >= +0.2, and >= 4/5 seed means share the sign (else SEED-DEPENDENT) |
| PARVO-LIKE | p < .05, mean <= -0.2, same seed rule |
| SMALL | p < .05, abs(mean) < 0.2 |
| NO DIFFERENCE | 90% CI inside (-0.2, +0.2) |
| UNRESOLVED | otherwise |

**Combined verdict per contrast:**
- **MAGNO-LIKE** if ACH and LSF are both MAGNO-LIKE.
- **PARTLY MAGNO-LIKE (index)** if only one is.
- **NOT MAGNO-LIKE** if neither is, and both are in {NO DIFFERENCE, PARVO-LIKE, SMALL}.
- **UNRESOLVED** otherwise.

The primary contrast is **DV**. **LV** is secondary under the same rule; it matters because
hMT+ sits in NSD's lateral stream.

## 7. Instrument gates (TSN, same stimuli)
All gates must pass. If any fails, every verdict becomes UNRESOLVED.

- **G1 (stimuli isolate the channels).** In diet_s0, the median ACH of magno units is
  >= 0.95 and the median ACH of parvo units is <= 0.05. This passes by construction only
  if lum and iso really are luminance-only and isoluminant in TSN's colour space. It
  validates the stimuli, not the index's sensitivity.
- **G2 (graded known positive for LSF).** Blurring a stream's input must raise that
  stream's LSF, with Cohen's d >= 0.8 across units in both cases:
  - magno units, diet_s0 vs nodiet_s0;
  - parvo units, swap_s0 vs nodiet_s0.

## 8. What triggers training
**The spatial-loss pilot is GO if the DV or LV verdict is MAGNO-LIKE (both indices).**

- A PARTLY verdict is not a trigger; it is reported and discussed.
- The other trigger, registered earlier, is a POSITIVE idea-1 verdict (TSN magno units win
  dorsal/lateral voxels).
- If both checks come back null, the hybrid is not trained.

## 9. Descriptive only
None of these enter the decision:

- the alpha = 0 network (does any effect need topography?);
- the raw-index form;
- the per-seed values;
- per network, the Spearman rho between residualised ACH and LSF across units, i.e.
  whether TDANN has an M/P-like axis of its own, independent of the brain.

## 10. Limitations, stated up front
1. **Network-level replication rests on the 5 seeds.** The 8 subjects share each network,
   so they replicate only the brain side.
2. **G1 cannot fail on a correct pipeline.** G2 is the real known positive for the index.
3. **Images are static.** The defining M property (transient, high temporal-frequency
   response) is not probed.
4. **Isoluminant chroma is scaled down by the gamut rule.** This moves every unit's ACH the
   same way.
5. **Per-unit shape bias, floated earlier, is dropped.** It has no unit-level definition
   without a readout.

## Amendments

### Amendment 2 (2026-09-25, before any decision statistic)
Registered while no assignment file existed: `results/` was empty and the mapping job was
still waiting on extraction. The dorsal/lateral-vs-ventral statistics that G1 gates
therefore did not exist and could not have been seen. (Amendment 1 is the idea-1 plan's; the
numbering follows the branch.)

**What fired.** G1 as registered FAILED on the first probe run:

| unit set in diet_s0 | median ACH | registered threshold | result |
|---|---|---|---|
| magno | 0.981 | >= 0.95 | pass |
| parvo | 0.199 | <= 0.05 | FAIL |

G2 passed: d = +2.87 (magno) and +2.48 (parvo).

**Diagnosis.** It uses TSN and the stimuli only; no brain data and no TDANN result were read.

- **Where grey lands.** In TSN's uint8-quantised rg channel, mid-grey falls at exactly 127.5,
  the rounding boundary. The 1e-6 term in the cone-contrast denominator
  (`convert_rgb_to_dkl_numpy`) rounds grey pixels up to 128 when they are brighter than
  mid-grey (mean lum 0.69) and down to 127 when darker (0.18).
- **What that does to the parvo input.** On luminance-only images, the quantised parvo input
  therefore carries a 1-bit luminance map.
- **Causal test.** Same 128 images, same network, with only the uint8 step removed:

  | TSN front end | parvo median ACH | magno median ACH |
  |---|---|---|
  | quantised, as trained | 0.194 | 0.981 |
  | float | 0.000 | 0.989 |

- **Conclusion.** The stimuli isolate luminance and L-M as intended. G1 fired on TSN's
  quantiser.

**Repair.**
1. G1 is scored on TSN responses computed with float DKL channels (`magno_probe.float_dkl`).
   The thresholds are unchanged. The quantised (as-trained) medians are reported next to it.
   G2 stays as registered (as trained).
2. Luminance-only images are made exactly grey: the channel mean after rendering. Before,
   ~3e-4 channel differences left some pixels 1/255 off grey.

The probe extraction is re-run for every network. The first run's files are kept in
`magno_probe/superseded_amend2/`.

**Why this is not gate-shopping.**
- G1's registered purpose (section 7) is to validate the stimuli. The float path measures
  exactly that.
- The repair cannot see or move the TDANN decision statistic: G1 carries no sign, and no
  assignment existed.
- The change to the luminance-only images applies to every network equally.

**Side finding, for the TSN manuscript rather than this probe.** As trained, TSN's parvo
("L-M only") input carries a 1-bit luminance-sign signal at achromatic pixels, the same way
here as in the uint8 DKL training set. Claims that parvo is luminance-blind should be scoped
accordingly.
