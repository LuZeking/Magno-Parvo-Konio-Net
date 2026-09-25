# TDANN magno-likeness probe: result

Plan: `docs/PLAN_TDANN_MAGNO_PROBE_2026-09-25.md`. Units of the index: SD over TDANN units (SESOI 0.2). Positive = the stream's units are more magno-like than ventral-stream units.

**Spatial-loss trigger (this check): NO GO (from this check)**

## Instrument gates (TSN)

- G1 ACH isolates the channels (float DKL, amendment 2): magno median 0.989, parvo median 0.000 -> **PASS** (as trained, uint8 DKL: magno 0.981, parvo 0.200)
- G2 blur raises magno LSF (diet vs nodiet): d = +2.87 -> **PASS**
- G2 blur raises parvo LSF (swap vs nodiet): d = +2.48 -> **PASS**

## Primary (position-residualised), TDANN alpha = 0.25, 5 seeds

**dorsal - ventral: NOT MAGNO-LIKE**

| index | label | mean | 90% CI | Wilcoxon p | consistency |
|---|---|---|---|---|---|
| ACH (luminance > isoluminant) | **SMALL** | -0.146 | [-0.222, -0.070] | 0.0234 | 1/8 subj, 5/5 seeds |
| LSF (low > high SF) | **SMALL** | -0.107 | [-0.165, -0.048] | 0.0078 | 0/8 subj, 5/5 seeds |

**lateral - ventral: NOT MAGNO-LIKE**

| index | label | mean | 90% CI | Wilcoxon p | consistency |
|---|---|---|---|---|---|
| ACH (luminance > isoluminant) | **SMALL** | +0.138 | [+0.056, +0.219] | 0.0391 | 6/8 subj, 4/5 seeds |
| LSF (low > high SF) | **NO DIFFERENCE** | -0.008 | [-0.047, +0.032] | 0.7422 | 4/8 subj, 3/5 seeds |

## Secondary (raw index), same networks

**dorsal - ventral: NOT MAGNO-LIKE**

| index | label | mean | 90% CI | Wilcoxon p | consistency |
|---|---|---|---|---|---|
| ACH (luminance > isoluminant) | **SMALL** | -0.171 | [-0.250, -0.092] | 0.0156 | 1/8 subj, 5/5 seeds |
| LSF (low > high SF) | **SMALL** | -0.103 | [-0.166, -0.040] | 0.0156 | 1/8 subj, 5/5 seeds |

**lateral - ventral: NOT MAGNO-LIKE**

| index | label | mean | 90% CI | Wilcoxon p | consistency |
|---|---|---|---|---|---|
| ACH (luminance > isoluminant) | **SMALL** | +0.127 | [+0.048, +0.207] | 0.0234 | 7/8 subj, 4/5 seeds |
| LSF (low > high SF) | **NO DIFFERENCE** | +0.008 | [-0.032, +0.047] | 1.0000 | 5/8 subj, 2/5 seeds |

## Descriptive: alpha = 0 (no spatial loss), seed 0

| contrast | index | label | mean | 90% CI | Wilcoxon p | consistency |
|---|---|---|---|---|---|---|
| DV | ACH | **NO DIFFERENCE** | -0.080 | [-0.142, -0.018] | 0.0781 | 1/8 subj, 1/1 seeds |
| DV | LSF | **NO DIFFERENCE** | -0.022 | [-0.049, +0.004] | 0.1953 | 3/8 subj, 1/1 seeds |
| LV | ACH | **MAGNO-LIKE** | +0.267 | [+0.223, +0.312] | 0.0078 | 8/8 subj, 1/1 seeds |
| LV | LSF | **SMALL** | +0.057 | [+0.025, +0.089] | 0.0156 | 7/8 subj, 1/1 seeds |

## Unit level (no brain data)

| network | median ACH | median LSF | Spearman rho(ACH, LSF), residualised | units |
|---|---|---|---|---|
| tdann_a025_s0 | 0.633 | 0.349 | -0.509 | 25087 |
| tdann_a025_s1 | 0.641 | 0.425 | -0.343 | 25087 |
| tdann_a025_s2 | 0.663 | 0.349 | -0.228 | 25088 |
| tdann_a025_s3 | 0.618 | 0.419 | -0.160 | 25088 |
| tdann_a025_s4 | 0.645 | 0.375 | -0.332 | 25088 |
| tdann_a0_s0 | 0.677 | 0.388 | +0.220 | 25020 |

## Stimuli

- 515 shared images; isoluminant chroma scale 0.5 (clipped pixels 0.00715)
- checks: `{"max_abs_dkl_vs_helpers": 0.0, "max_abs_roundtrip_rgb": 0.00029922719999973314, "lum_lum_contrast_vs_orig_median": 1.0004843473434448, "lum_rg_contrast_vs_orig_median": 0.02897017076611519, "lum_by_contrast_vs_orig_median": 0.9207358956336975, "iso_lum_contrast_vs_orig_median": 0.009176029823720455, "iso_rg_contrast_vs_orig_median": 0.5008208751678467, "iso_by_contrast_vs_orig_median": 0.1514819711446762, "orig_power_above_cut_median": 0.2198056914705418, "low_power_above_cut_median": 0.02800964408490296, "high_power_above_cut_median": 0.9412056059144265}`
