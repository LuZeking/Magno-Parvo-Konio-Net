"""Generality of the readout account beyond the two-stream architecture.

Pre-specified in docs/PLAN_GENERALITY_2026-09-14.md. Modules:

  models        one interface to a frozen network's penultimate features and
                its own linear head (our runs, the two-stream concatenation,
                torchvision ImageNet networks)
  extract       feature dumps for the cue-conflict set, MiniEcoset and NSD
  unit_factor   shape/texture eta^2 per unit, median split, fixed-head gain sweep
  refit_sum     lossy-sum mixture with refitted linear and two-layer heads
  cortex_bands  NSD banded ridge with the shape- and texture-leaning halves
                as bands, plus the cortex-optimal mixing weight
"""
