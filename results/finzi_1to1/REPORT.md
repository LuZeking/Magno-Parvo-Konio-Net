# Finzi 1-to-1 stream-preference test: results

Plan: `docs/PLAN_FINZI_1TO1_2026-09-24.md` (with amendment 1). δ = 0.05. Values are per-subject means (n = 8); 90 % t-interval; exact two-sided Wilcoxon.

**Verdict: primary label: POSITIVE (A -> first stream)**

## Gates

| gate | mean Δ | 90 % CI | p | n>0 | outcome |
|---|---|---|---|---|---|
| PC-TDANN | +0.272 | [+0.232, +0.311] | 0.0078 | 8/8 | PASS |
| PC-TDANN (registered mean-r groups, descriptive) | +0.229 | [+0.180, +0.278] | 0.0078 | 8/8 | PASS |
| PC-TSN | +0.318 | [+0.276, +0.361] | 0.0078 | 8/8 | PASS |
| PC-TSN (registered mean-r groups, descriptive) | +0.284 | [+0.243, +0.324] | 0.0078 | 8/8 | PASS |
| NEG | -0.003 | [-0.007, +0.001] | 0.3828 | 4/8 | PASS |
| NEG-TDANN (descriptive) | +0.000 | [-0.003, +0.004] | 1.0000 | 4/8 | NO PREFERENCE |

## Primary

| test | mean Δ | 90 % CI | p | n>0 | label |
|---|---|---|---|---|---|
| diet dDV_strat | +0.125 | [+0.089, +0.161] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |

Δ_DV = share of dorsal voxels won by magno units − share of ventral voxels won by magno units (NC-quintile stratified). POSITIVE = magno → dorsal.

## Secondary (labels do not change the primary)

| test | mean Δ | 90 % CI | p | n>0 | label |
|---|---|---|---|---|---|
| diet dLV_strat | +0.161 | [+0.147, +0.176] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| diet dDV_raw | +0.172 | [+0.146, +0.199] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| diet dLV_raw | +0.159 | [+0.142, +0.177] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| nodiet dDV_strat | -0.035 | [-0.075, +0.004] | 0.1094 | 1/8 | UNRESOLVED |
| nodiet dLV_strat | -0.061 | [-0.096, -0.026] | 0.0391 | 2/8 | NEGATIVE (A -> second stream) |
| nodiet dDV_raw | -0.087 | [-0.126, -0.049] | 0.0156 | 1/8 | NEGATIVE (A -> second stream) |
| nodiet dLV_raw | -0.057 | [-0.092, -0.023] | 0.0234 | 1/8 | NEGATIVE (A -> second stream) |
| swap dDV_strat | -0.006 | [-0.046, +0.034] | 0.8438 | 4/8 | NO PREFERENCE |
| swap dLV_strat | +0.007 | [-0.028, +0.042] | 0.7422 | 4/8 | NO PREFERENCE |
| swap dDV_raw | -0.064 | [-0.097, -0.031] | 0.0078 | 0/8 | NEGATIVE (A -> second stream) |
| swap dLV_raw | +0.014 | [-0.019, +0.047] | 0.3125 | 6/8 | NO PREFERENCE |
| untrained_diet dDV_strat | +0.064 | [+0.029, +0.100] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| untrained_diet dLV_strat | +0.268 | [+0.244, +0.292] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| untrained_diet dDV_raw | +0.061 | [+0.036, +0.085] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| untrained_diet dLV_raw | +0.275 | [+0.249, +0.301] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| untrained_nodiet dDV_strat | +0.035 | [-0.004, +0.074] | 0.1484 | 6/8 | UNRESOLVED |
| untrained_nodiet dLV_strat | +0.267 | [+0.238, +0.296] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| untrained_nodiet dDV_raw | +0.030 | [+0.003, +0.057] | 0.0781 | 6/8 | UNRESOLVED |
| untrained_nodiet dLV_raw | +0.274 | [+0.241, +0.307] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| diet - untrained_diet dDV_strat | +0.061 | [-0.001, +0.122] | 0.1953 | 7/8 | UNRESOLVED |
| diet - nodiet dDV_strat | +0.160 | [+0.112, +0.209] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |
| diet - untrained_diet dLV_strat | -0.106 | [-0.132, -0.081] | 0.0078 | 0/8 | NEGATIVE (A -> second stream) |
| diet - nodiet dLV_strat | +0.223 | [+0.180, +0.265] | 0.0078 | 8/8 | POSITIVE (A -> first stream) |

## Descriptive

Magno share per stream and NC-normalised test r (r/√NC) of assigned pairs, fixed-quota draws (unforced variant in brackets).

| arm | share ventral | share lateral | share dorsal | r/√NC ventral | lateral | dorsal |
|---|---|---|---|---|---|---|
| diet | 0.395 (0.181) | 0.554 (0.323) | 0.568 (0.311) | 0.122 | 0.088 | 0.083 |
| nodiet | 0.547 (0.738) | 0.490 (0.704) | 0.460 (0.719) | 0.144 | 0.111 | 0.096 |
| swap | 0.516 (0.712) | 0.530 (0.773) | 0.452 (0.730) | 0.143 | 0.106 | 0.096 |
| untrained_diet | 0.377 (0.325) | 0.652 (0.640) | 0.438 (0.348) | 0.119 | 0.060 | 0.073 |
| untrained_nodiet | 0.384 (0.328) | 0.658 (0.640) | 0.414 (0.321) | 0.115 | 0.052 | 0.067 |

TDANN with Finzi's own design (all units, rectangular assignment), r/√NC; published (fsaverage, α = 0.25): dorsal 0.22, lateral 0.25, ventral 0.34.

| network | ventral | lateral | dorsal |
|---|---|---|---|
| tdann_a025_s0 (Finzi all-units) | 0.374 | 0.310 | 0.319 |
| tdann_a0_s0 (Finzi all-units) | 0.282 | 0.233 | 0.224 |
