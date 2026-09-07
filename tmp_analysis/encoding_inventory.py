"""Which models are fitted in which encoding directory, and with how many subjects?

Answers the E2 control-table question: can the six front-end controls be
regenerated from the declared source (`encoding_cc`), or only from `encoding`?
Enumerated rather than eyeballed, because "looked to hold" is how §88 happened.
"""
import glob
import os
import re
from collections import defaultdict

R = "/home/byrong/projects/TSN/Magno-Parvo-Konio-Net/.claude/worktrees/mpk-dev"
DIRS = ["encoding", "encoding_cc", "encoding/quarantine"]

inv: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
for d in DIRS:
    for f in sorted(glob.glob(f"{R}/results/{d}/*.npz")):
        b = os.path.basename(f)
        m = re.match(r"^(.*)_subj(\d+)\.npz$", b)
        if m:
            inv[m.group(1)][d].append(int(m.group(2)))
        else:
            inv[b.replace(".npz", "") + "  [NOT subj-tagged]"][d].append(-1)

models = sorted(inv)
print(f"{'model':38s} {'encoding':>10s} {'encoding_cc':>12s} {'quar':>5s}")
print("-" * 68)
for mo in models:
    cells = []
    for d in DIRS:
        v = inv[mo].get(d, [])
        cells.append(f"n={len(v)}" if v else "-")
    print(f"{mo:38s} {cells[0]:>10s} {cells[1]:>12s} {cells[2]:>5s}")

# The six front-end controls E2's table needs
CONTROLS = {
    "swap": "dkl_bcs_swap",
    "parvo_highsf": "dkl_bcs_parvo_highsf",
    "konio substitution": "untrained_konio",
    "Gabor bank": "gabor",
    "untrained-on-diet": "untrained_bcs",
    "no-blur": "noblur",
}
print("\n--- the six controls E2's table needs ---")
for label, pat in CONTROLS.items():
    hits = [mo for mo in models if pat in mo]
    if not hits:
        print(f"{label:22s} NO FIT FOUND matching '{pat}'")
        continue
    for mo in hits:
        e = len(inv[mo].get("encoding", []))
        c = len(inv[mo].get("encoding_cc", []))
        verdict = "in encoding_cc" if c else "encoding ONLY"
        print(f"{label:22s} {mo:34s} enc n={e:<2d} cc n={c:<2d} -> {verdict}")

only_enc = [mo for mo in models
            if inv[mo].get("encoding") and not inv[mo].get("encoding_cc")]
print(f"\n{len(only_enc)} model(s) exist in encoding/ but NOT encoding_cc/:")
for mo in only_enc:
    print(f"  {mo}  (n={len(inv[mo]['encoding'])})")
