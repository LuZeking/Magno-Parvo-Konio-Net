# TSN x TDANN spatial-loss pilot: result

Plan: `docs/PLAN_TSN_SPATIAL_PILOT_2026-09-25.md`. Indices in SD over layer4.1 units (SESOI 0.2); positive = the stream's units are more magno-like than ventral-stream units.

**Primary (A1 = fused_a2p5, dorsal - ventral): NOT MAGNO-LIKE**

**Next step (plan §5): STOP: the hybrid does not give a magno-like dorsal stream**

## Gates

- G-train: final val top-1 {"fused_a0": 0.5239285714285714, "fused_a2p5": 0.48232142857142857}, epochs {"fused_a0": 150, "fused_a2p5": 150} -> **PASS**
- G-SL: layer4.1 SL {"fused_a0": 0.4869, "fused_a2p5": 0.1938} (A1 must be <= A0 - 0.05) -> **PASS**
- G-probe: trunk layer2.1 median ACH {"fused_a0": {"magno_median": 0.9856329050156705, "parvo_median": 0.0}, "fused_a2p5": {"magno_median": 0.9807639433903277, "parvo_median": 0.0}} -> **PASS**

## A1: alpha = 2.5

**dorsal - ventral: NOT MAGNO-LIKE**

| index | label | mean | 90% CI | Wilcoxon p | subjects > 0 |
|---|---|---|---|---|---|
| ACH | **SMALL** | -0.072 | [-0.101, -0.042] | 0.0156 | 1/8 |
| LSF | **SMALL** | -0.165 | [-0.187, -0.142] | 0.0078 | 0/8 |
| MR (secondary) | **NO DIFFERENCE** | -0.044 | [-0.075, -0.013] | 0.0781 | 2/8 |

**lateral - ventral: NOT MAGNO-LIKE**

| index | label | mean | 90% CI | Wilcoxon p | subjects > 0 |
|---|---|---|---|---|---|
| ACH | **NO DIFFERENCE** | -0.032 | [-0.059, -0.005] | 0.0547 | 1/8 |
| LSF | **SMALL** | -0.149 | [-0.177, -0.121] | 0.0078 | 0/8 |
| MR (secondary) | **NO DIFFERENCE** | -0.005 | [-0.029, +0.019] | 0.9453 | 5/8 |

## A0: alpha = 0

**dorsal - ventral: NOT MAGNO-LIKE**

| index | label | mean | 90% CI | Wilcoxon p | subjects > 0 |
|---|---|---|---|---|---|
| ACH | **SMALL** | +0.075 | [+0.027, +0.123] | 0.0078 | 8/8 |
| LSF | **SMALL** | -0.133 | [-0.182, -0.084] | 0.0078 | 0/8 |
| MR (secondary) | **SMALL** | +0.083 | [+0.033, +0.134] | 0.0156 | 7/8 |

**lateral - ventral: NOT MAGNO-LIKE**

| index | label | mean | 90% CI | Wilcoxon p | subjects > 0 |
|---|---|---|---|---|---|
| ACH | **SMALL** | -0.034 | [-0.057, -0.012] | 0.0391 | 1/8 |
| LSF | **SMALL** | +0.062 | [+0.041, +0.083] | 0.0078 | 8/8 |
| MR (secondary) | **NO DIFFERENCE** | -0.033 | [-0.057, -0.009] | 0.0547 | 1/8 |

## A1 - A0 (paired by subject)

**DV: SL-AMPLIFIED = False**

| index | label | mean | 90% CI | Wilcoxon p | subjects > 0 |
|---|---|---|---|---|---|
| ACH | **SMALL** | -0.147 | [-0.204, -0.089] | 0.0078 | 0/8 |
| LSF | **NO DIFFERENCE** | -0.032 | [-0.071, +0.007] | 0.1484 | 2/8 |
| MR | **SMALL** | -0.127 | [-0.185, -0.069] | 0.0078 | 0/8 |

**LV: SL-AMPLIFIED = False**

| index | label | mean | 90% CI | Wilcoxon p | subjects > 0 |
|---|---|---|---|---|---|
| ACH | **NO DIFFERENCE** | +0.002 | [-0.023, +0.027] | 0.9453 | 3/8 |
| LSF | **PARVO-LIKE** | -0.211 | [-0.254, -0.167] | 0.0078 | 0/8 |
| MR | **NO DIFFERENCE** | +0.028 | [+0.004, +0.053] | 0.0547 | 6/8 |

## Topography

| network | T1 MR map r (obs) | null mean | p | T2 stream clustering excess | T2 z |
|---|---|---|---|---|---|
| fused_a0 | +0.0224 | +0.0191 | 0.090 | +0.0071 | +2.34 |
| fused_a2p5 | +0.1416 | +0.0521 | 0.005 | +0.0619 | +20.51 |

T2 A1 vs A0 paired Wilcoxon p = 0.0078125

## Descriptive

```
{
 "fused_a0": {
  "mr_frac_above_0p8": 0.0,
  "mr_frac_below_0p2": 0.157989501953125,
  "mr_median": 0.3725572144720956,
  "ach_median": 0.3845183179355605,
  "lsf_median": 0.4970957266571424,
  "sl_gate": {
   "layer4.1": {
    "mean": 0.486905200779438,
    "sd": 0.004011190864031207,
    "n": 1000
   }
  }
 },
 "fused_a2p5": {
  "mr_frac_above_0p8": 0.12646484375,
  "mr_frac_below_0p2": 0.037322998046875,
  "mr_median": 0.5271373131189981,
  "ach_median": 0.5413701504506114,
  "lsf_median": 0.5510036398517344,
  "sl_gate": {
   "layer4.1": {
    "mean": 0.1938269979953766,
    "sd": 0.02664893582390671,
    "n": 1000
   }
  }
 }
}
```
