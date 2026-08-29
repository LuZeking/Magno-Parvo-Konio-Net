# Quarantined: probed mid-training

These two swap-condition results were produced from `checkpoint_last.pt` while
the run was at **epoch 33 of 150**, then compared against `dkl_base` results
from fully converged 150-epoch runs. The condition is therefore confounded with
training length and the numbers must not be used. See PAPER_PLAN §10.1.

They are kept rather than deleted so the re-run can be compared against them.
`features.assert_converged()` now blocks this class of mistake at the source.

Re-run after 26373625 tasks 9-11 finish:
    sbatch slurm/shape_texture.sh   # swap entries only
