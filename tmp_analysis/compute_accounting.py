"""Compute and energy accounting for Methods 4.2 (docs/PAPER_v2.md).

  python tmp_analysis/compute_accounting.py [--since 2026-08-01] [--dump FILE] [--out results/figures_v2/compute_accounting.json]

Reads the scheduler's accounting for every job of this project (job names prefixed `mpk_`,
the convention of every script under slurm/) via `sacct`, or from a saved `sacct -P` dump
(--dump; columns JobID,JobName,Partition,AllocCPUS,ElapsedRaw,AllocTRES,State,Start), and
writes allocated core-hours, GPU-hours and memory GB-hours per job name and in total, with the
energy and carbon estimate of Lannelongue, Grealey & Inouye (2021, Adv. Sci.; "Green
Algorithms"): E = t x (n_cores x P_core x u + mem_GB x P_mem) x PUE, GPUs at their TDP.

Assumptions (all stated in Methods 4.2; every one errs upward):
  P_core = 12 W per allocated core (the Green Algorithms default for an unlisted Xeon; the
           Skylake Xeons of the `main` partition sit at 6-10 W per core by TDP / core count),
  u      = 1 (allocated cores counted as fully used),
  P_mem  = 0.3725 W per GB of allocated memory,
  P_gpu  = 230 W per GPU (NVIDIA RTX A5000 TDP),
  PUE    = 1.67 (Green Algorithms' world-average default),
  CI     = 0.35 kg CO2e per kWh (Germany's grid average, 2023; Umweltbundesamt 2024 gives
           0.380 for 2023 and a provisional 0.363 for 2024; Green Algorithms lists 0.338).
Not counted: the pretraining of the public networks, the acquisition of the two human
datasets, interactive sessions and jobs under other names.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess

P_CORE, USAGE, P_MEM, P_GPU, PUE, CI = 12.0, 1.0, 0.3725, 230.0, 1.67, 0.35

GROUPS = {  # job-name prefixes -> reporting group (Methods 4.2)
    "training (21 runs)": ("mpk_train", "mpk_single", "mpk_single34"),
    "matched-control retraining": ("mpk_matched",),
    "two-band encoding fits and their controls": ("mpk_encode", "mpk_enc_cc", "mpk_cc_seeds", "mpk_cc_ctrl", "mpk_konio",
                                                  "mpk_konio_seeds", "mpk_fe2x2", "mpk_cc_gab", "mpk_cc_pca", "mpk_vprec",
                                                  "mpk_vprec_diet", "mpk_vprec_realnc", "mpk_enc_probe", "mpk_equiv"),
    "single-trunk generality (unit rule)": ("mpk_gen_ctx", "mpk_gen_ctx2", "mpk_gen_ctx_test", "mpk_gen_mini", "mpk_gen_nsd",
                                            "mpk_gen_cc", "mpk_gen_inval", "mpk_gen_rules"),
    "cortex-optimal readout weight": ("mpk_readout_w",),
    "subspace-rule cortex fits": ("mpk_sub_ctx", "mpk_sub_refit", "mpk_sub_nsd", "mpk_sub_cc"),
    "model zoo": ("mpk_zoo", "mpk_zoo_kappa"),
}


def parse_mem_gb(v: str) -> float:
    try:
        if v.endswith("G"): return float(v[:-1])
        if v.endswith("M"): return float(v[:-1]) / 1024
        if v.endswith("T"): return float(v[:-1]) * 1024
        return float(v) / 1024
    except ValueError:
        return 0.0


def load_rows(since: str, dump: str | None):
    if dump:
        text = open(dump, encoding="utf8").read()
    else:
        text = subprocess.run(["sacct", "-S", f"{since}T00:00:00", "-E", "now", "-X", "-n", "-P",
                               "--format=JobID,JobName,Partition,AllocCPUS,ElapsedRaw,AllocTRES,State,Start"],
                              check=True, capture_output=True, text=True).stdout
    return [l.split("|") for l in text.splitlines() if l.count("|") >= 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-01")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--prefix", default="mpk_")
    ap.add_argument("--out", default="results/figures_v2/compute_accounting.json")
    args = ap.parse_args()
    per = collections.defaultdict(lambda: dict(n=0, core_h=0.0, gpu_h=0.0, mem_gbh=0.0, states=collections.Counter()))
    first, last = None, None
    for jid, name, part, cpus, el, tres, state, start in (r[:8] for r in load_rows(args.since, args.dump)):
        if not name.startswith(args.prefix):
            continue
        try:
            el, cpus = float(el), float(cpus or 0)
        except ValueError:
            continue
        gpu = mem = 0.0
        for kv in tres.split(","):
            if kv.startswith("gres/gpu"):
                try: gpu = float(kv.split("=")[-1])
                except ValueError: pass
            elif kv.startswith("mem="):
                mem = parse_mem_gb(kv[4:])
        a = per[name]
        a["n"] += 1; a["core_h"] += cpus * el / 3600; a["gpu_h"] += gpu * el / 3600; a["mem_gbh"] += mem * el / 3600
        a["states"][state.split()[0]] += 1
        if start and start[:4].isdigit():
            first = min(first or start, start); last = max(last or start, start)
    tot = dict(n=sum(a["n"] for a in per.values()), core_h=sum(a["core_h"] for a in per.values()),
               gpu_h=sum(a["gpu_h"] for a in per.values()), mem_gbh=sum(a["mem_gbh"] for a in per.values()))
    groups = {}
    for g, prefixes in GROUPS.items():
        names = [n for n in per if n in prefixes]
        groups[g] = dict(n=sum(per[n]["n"] for n in names), core_h=sum(per[n]["core_h"] for n in names),
                         gpu_h=sum(per[n]["gpu_h"] for n in names), job_names=sorted(names))
    grouped = {n for ps in GROUPS.values() for n in ps}
    groups["other (smoke tests, figures, verifier, caches, robustness, seeds)"] = dict(
        n=sum(a["n"] for n, a in per.items() if n not in grouped), core_h=sum(a["core_h"] for n, a in per.items() if n not in grouped),
        gpu_h=sum(a["gpu_h"] for n, a in per.items() if n not in grouped), job_names=sorted(n for n in per if n not in grouped))
    e_core = tot["core_h"] * P_CORE * USAGE / 1000
    e_mem = tot["mem_gbh"] * P_MEM / 1000
    e_gpu = tot["gpu_h"] * P_GPU / 1000
    e_total = (e_core + e_mem + e_gpu) * PUE
    energy = dict(kwh_cores=e_core, kwh_memory=e_mem, kwh_gpus=e_gpu, pue=PUE, kwh_total=e_total,
                  carbon_intensity_kg_per_kwh=CI, kg_co2e=e_total * CI,
                  parameters=dict(P_core_W=P_CORE, usage=USAGE, P_mem_W_per_GB=P_MEM, P_gpu_W=P_GPU))
    out = dict(since=args.since, first_start=first, last_start=last, prefix=args.prefix, totals=tot, groups=groups,
               energy=energy, per_job_name={n: dict(n=a["n"], core_h=a["core_h"], gpu_h=a["gpu_h"], mem_gbh=a["mem_gbh"],
                                                   states=dict(a["states"])) for n, a in sorted(per.items())})
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"jobs {tot['n']}  core-h {tot['core_h']:,.0f}  GPU-h {tot['gpu_h']:,.1f}  mem GB-h {tot['mem_gbh']:,.0f}  ({first[:10]} .. {last[:10]})")
    for g, v in groups.items():
        print(f"  {g:60s} n={v['n']:4d}  core-h {v['core_h']:8,.0f}  GPU-h {v['gpu_h']:6.1f}")
    print(f"energy: cores {e_core:,.0f} + memory {e_mem:,.0f} + GPUs {e_gpu:,.0f} kWh, x PUE {PUE} = {e_total:,.0f} kWh; "
          f"{e_total * CI:,.0f} kg CO2e at {CI} kg/kWh")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
