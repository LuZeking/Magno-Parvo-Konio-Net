#!/bin/bash
# Does this torch build actually produce kernels for the GPUs we intend to
# use, and how fast is the TSN step on each?
#
# This exists because the `gpu` partition turned out to be GTX 1080 Ti
# (sm_61), which torch 2.13+cu130 no longer ships kernels for -- every job
# there died instantly with "no kernel image is available for execution on
# the device". g017's B200 is sm_100, new enough to have the mirror-image
# risk, so it gets verified before any array is pointed at it.
#
# Submit once per target:
#   sbatch --partition=agcichy  --gres=gpu:nvidia_b200:1 slurm/gpu_capability_probe.sh
#   sbatch --partition=begendiv --gres=gpu:h100:1        slurm/gpu_capability_probe.sh
#   sbatch --partition=agcichy  --gres=gpu:a5000:1       slurm/gpu_capability_probe.sh
#SBATCH --job-name=mpk_gpucap
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/gpucap_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 PYTHONPATH=.
PY=/scratch/byrong/anaconda3/envs/mpk/bin/python

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY - <<'PYEOF'
import time
import torch
from src.models.pytorch_models import TwoStreamNetwork
from src.training.standalone_trainer import (
    CONDITIONS, in_channels_for, patch_stream_input_channels)

print("torch", torch.__version__, "| cuda", torch.version.cuda)
cap = torch.cuda.get_device_capability(0)
print("device", torch.cuda.get_device_name(0), "sm_%d%d" % cap)
print("arch list", torch.cuda.get_arch_list())

# The 1080 Ti failure surfaced only when a kernel actually launched, so
# force real work rather than trusting the capability string.
try:
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print("bf16 matmul ok:", float((a @ a).float().abs().mean()))
    c = torch.nn.Conv2d(3, 64, 7, 2, 3).cuda()
    x = torch.randn(8, 3, 256, 256, device="cuda")
    torch.cuda.synchronize()
    print("conv2d ok:", tuple(c(x).shape))
except Exception as e:
    print("KERNEL FAILURE:", type(e).__name__, e)
    raise SystemExit(1)

cfg = CONDITIONS["dkl_bcs"]
c1, c2 = in_channels_for(cfg)
lossf = torch.nn.CrossEntropyLoss()
for bs in (256, 512):
    for amp in (False, True):
        try:
            m = TwoStreamNetwork(num_classes=112,
                                 legacy_asymmetric_mixing=False)
            patch_stream_input_channels(m, c1, c2)
            m = m.cuda()
            if amp:
                m = m.to(memory_format=torch.channels_last)
            opt = torch.optim.Adam(m.parameters(), lr=1e-4)
            x1 = torch.randn(bs, c1, 256, 256, device="cuda")
            x2 = torch.randn(bs, c2, 256, 256, device="cuda")
            if amp:
                x1 = x1.contiguous(memory_format=torch.channels_last)
                x2 = x2.contiguous(memory_format=torch.channels_last)
            y = torch.randint(0, 112, (bs,), device="cuda")

            def step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = lossf(m(x1, x2), y)
                loss.backward()
                opt.step()

            for _ in range(5):
                step()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                step()
            torch.cuda.synchronize()
            r = 20 * bs / (time.time() - t0)
            print(f"  bs={bs:<4} amp={str(amp):<5} {r:7.0f} img/s  "
                  f"-> {263200 * 150 / r / 3600:5.1f} h per 150-epoch run")
            del m, opt, x1, x2
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"  bs={bs:<4} amp={str(amp):<5} OOM")
            torch.cuda.empty_cache()
PYEOF
