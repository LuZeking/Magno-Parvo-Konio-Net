# Agreement at a matched rate (PLAN_KAPPA, run 2026-09-15_zoo)

Target rate 0.40 ± 0.05; 33 of 38 ImageNet-trained networks reach it inside their gate.

| prediction | outcome | value |
|---|---|---|
| K-1 rate moves ≥ 0.20, κ stays < 40% of ceiling | FAILED | 34/38 networks (rate range ok in 34, κ ok in 38) |
| K-2 κ varies ≥ 2× at matched rate | FAILED | 10.0–18.7% (1.88×; ResNeXt-101 32x8d → GoogLeNet) |
| K-3 ρ(κ at matched rate, probe bias) ≥ +0.30 | FAILED | ρ = +0.01 (with trained rate +0.07; probe v trained rate +0.41) |
| K-4a across-network sd ≥ within-network sd | FAILED | 2.3 v 4.1 points (0.55×) |
| K-4b probe adds ≥ 0.05 R² over rate | FAILED | 0.891 → 0.895 (Δ +0.004, 798 gated readouts) |
| K-5 SIN highest at matched rate | MET | 28.7% at rate 0.65 v 18.7% (GoogLeNet) |
| K-6 no ImageNet readout above 50% anywhere | MET | max 36.7% (ViT-B/32) |

| network | w | rate | κ (% of ceiling) | κ at trained readout | probe bias | trained rate |
|---|---|---|---|---|---|---|
| GoogLeNet | 0.90 | 0.397 | 18.7 | 9.5 | -0.52 | 0.251 |
| ShuffleNet-V2 x1.0 | 0.80 | 0.401 | 18.1 | 9.2 | -0.43 | 0.234 |
| Inception-v3 | 0.80 | 0.406 | 17.8 | 11.0 | -0.56 | 0.305 |
| Swin-S | 0.95 | 0.413 | 17.5 | 8.9 | -0.55 | 0.236 |
| MobileNet-V3-L | 0.80 | 0.407 | 17.0 | 10.8 | -0.47 | 0.265 |
| Swin-B | 1.00 | 0.418 | 17.0 | 7.7 | -0.57 | 0.233 |
| MaxViT-T | 0.85 | 0.401 | 16.0 | 9.1 | -0.57 | 0.295 |
| Swin-T | 0.90 | 0.407 | 15.4 | 8.4 | -0.52 | 0.225 |
| ViT-L/16 | 0.75 | 0.408 | 15.4 | 11.5 | -0.43 | 0.319 |
| EfficientNet-B3 | 0.75 | 0.396 | 15.1 | 12.5 | -0.50 | 0.291 |
| DenseNet-201 | 0.85 | 0.403 | 14.9 | 8.5 | -0.49 | 0.273 |
| MobileNet-V2 | 0.95 | 0.389 | 14.4 | 5.6 | -0.52 | 0.184 |
| Wide-ResNet-101-2 | 0.90 | 0.402 | 13.9 | 7.9 | -0.60 | 0.279 |
| EfficientNet-B0 | 0.85 | 0.407 | 13.9 | 7.1 | -0.54 | 0.271 |
| ResNet-101 | 0.95 | 0.405 | 13.8 | 8.7 | -0.52 | 0.284 |
| ResNet-152 | 0.90 | 0.402 | 13.7 | 9.1 | -0.49 | 0.293 |
| ResNet-50 SIN+IN, ft IN * | 0.95 | 0.397 | 13.5 | 5.9 | -0.58 | 0.227 |
| ResNet-50 SIN+IN * | 0.60 | 0.399 | 13.3 | 11.7 | -0.40 | 0.363 |
| RegNet-Y-16GF | 0.90 | 0.409 | 13.0 | 7.7 | -0.50 | 0.261 |
| ConvNeXt-B | 0.80 | 0.398 | 13.0 | 8.0 | -0.47 | 0.269 |
| ViT-B/32 | 0.00 | 0.418 | 12.9 | 23.0 | -0.30 | 0.617 |
| ResNeXt-50 32x4d | 0.90 | 0.383 | 12.9 | 5.6 | -0.58 | 0.219 |
| ResNet-18 | 0.90 | 0.407 | 12.7 | 6.3 | -0.53 | 0.233 |
| ResNet-34 | 0.85 | 0.403 | 12.5 | 6.1 | -0.53 | 0.259 |
| DenseNet-121 | 0.90 | 0.402 | 12.5 | 5.7 | -0.55 | 0.224 |
| MNASNet-1.0 | 0.85 | 0.406 | 12.5 | 8.7 | -0.53 | 0.226 |
| ConvNeXt-S | 0.85 | 0.399 | 12.4 | 8.1 | -0.53 | 0.248 |
| ConvNeXt-T | 0.85 | 0.395 | 12.3 | 7.0 | -0.50 | 0.241 |
| ResNet-50 | 1.00 | 0.394 | 12.2 | 5.1 | -0.56 | 0.222 |
| AlexNet | 0.75 | 0.393 | 12.0 | 7.1 | -0.45 | 0.261 |
| ConvNeXt-L | 0.80 | 0.401 | 11.8 | 9.2 | -0.49 | 0.293 |
| ViT-B/16 | 0.50 | 0.399 | 11.6 | 11.6 | -0.43 | 0.399 |
| RegNet-Y-8GF | 0.90 | 0.407 | 11.5 | 6.7 | -0.50 | 0.244 |
| Wide-ResNet-50-2 | 0.95 | 0.405 | 10.5 | 5.0 | -0.62 | 0.230 |
| ResNeXt-101 32x8d | 0.80 | 0.400 | 10.0 | 7.9 | -0.53 | 0.294 |
| ResNet-50 (V2 recipe) * | 1.00 | 0.356 | 7.8 | 0.5 | -0.62 | 0.185 |

`*` not ImageNet-trained (Stylized-ImageNet control or recipe variant).

## Post hoc (not pre-registered): what the failures imply

- Ranking the 33 networks by κ at their trained readouts against by κ at the matched rate: Kendall τ = 0.33, 178 of 528 pairs (34%) reverse.
- One curve fits the zoo: κ% = 0.1 +22.3·rate +28.3·rate² over 798 gated readouts, R² = 0.897; RMSE 1.7 points of ceiling (largest residual 10.1; RMSE 3.5 over the 40 readouts above rate 0.5); per-network mean offsets span -2.2 to +2.8 points (s.d. 1.3), and are unrelated to the probe bias (ρ = +0.04), top-1 (ρ = -0.02) or the trained rate (ρ = -0.13).
- Highest offsets: [('EfficientNet-B3', 2.8), ('ShuffleNet-V2 x1.0', 2.8), ('GoogLeNet', 2.7)]; lowest: [('ResNet-34', -1.6), ('ResNet-50', -2.0), ('Wide-ResNet-50-2', -2.2)].
- Extrapolating that curve to the human rate 0.958 gives 48% of the ceiling (extrapolation: the largest gated rate observed is 0.77).

### The attainable bound

- Cohen's κ cannot exceed (1 − |p_model − p_human| − p_e)/(1 − p_e). Against observers at 0.958 the bound at the trained readouts averages 0.033, so the rise of κ with the readout is in part the bound moving. Human pairs sit at 35% of their own bound.
- Normalised (κ / κ_max, as % of the human-pair value): from the texture end to the shape end of the gate the median network gains +6.6 points, against +13.5 points on the raw scale.
- Rank R² on the rate: 0.001 normalised against 0.891 raw (adding the probe bias: 0.003). Within network, the median Spearman ρ between rate and agreement across the gated sweep is +0.99 raw and +0.20 normalised (22 of 38 networks gain from the texture to the shape end on the normalised scale). The bound itself runs from 0.006 to 0.219 over the gated readouts, against the human-pair κ of 0.234.
- At the matched rate the normalised agreement spans 100.5–185.9% (1.85×, s.d. 23.2); ρ with probe bias +0.08, with the trained rate +0.19. No readout of any ImageNet-trained network exceeds 248%.
- At rate 0.65: ResNet-50 SIN 135% [104, 165] (raw 29%); ViT-B/32 120% [64, 174] (raw 26%); AlexNet 79% [61, 96] (raw 14%) (95% bootstrap over the ten observers).

| at rate 0.65 ± 0.05 | w | rate | κ (% of ceiling) | probe bias |
|---|---|---|---|---|
| ResNet-50 SIN * | 0.00 | 0.650 | 28.7 | -0.11 |
| ViT-B/32 | 0.60 | 0.653 | 25.9 | -0.30 |
| AlexNet | 1.00 | 0.611 | 14.3 | -0.45 |
