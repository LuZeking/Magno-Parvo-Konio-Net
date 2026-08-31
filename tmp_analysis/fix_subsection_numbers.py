"""One-off: undo a mis-targeted renumbering in docs/PAPER_PLAN.md.

A str.replace(old, new, 1) intended for my own section hit the FIRST match in
the file instead, which was in the other session's section. This restores their
subsection headers and applies the renumbering to mine, addressed by line number
so there is no ambiguity about which occurrence is which.
"""

from __future__ import annotations

P = "docs/PAPER_PLAN.md"

# 1-indexed line -> the header that line should carry
WANT = {
    1996: "### 22.1 What happened",
    2016: "### 22.2 Verified directly, not inferred",
    2030: "### 22.3 Scope: exactly one file",
    2172: "### 25.1 The prediction holds — and the input factors now form a clean 2×2",
    2189: "### 25.2 The capacity confound is not merely argued away — it is measured",
    2206: "### 25.3 It independently confirms §20.2's averaging account",
}

lines = open(P).read().split("\n")
for ln, want in sorted(WANT.items()):
    cur = lines[ln - 1]
    assert cur.startswith("### 2"), f"line {ln} is not a subsection header: {cur!r}"
    print(f"  {ln}: {cur}\n     -> {want}")
    lines[ln - 1] = want
open(P, "w").write("\n".join(lines))
print("fixed")
