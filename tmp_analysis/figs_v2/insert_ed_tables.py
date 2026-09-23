"""Insert results/figures_v2/ed_tables.md into docs/PAPER_v2.md (idempotent): the Extended Data tables between
the ED_TABLES markers and the Supplementary tables between the SUPP_TABLES markers (the 14 tables moved to the
Supplementary for length, 2026-09-23)."""
import re

p = "docs/PAPER_v2.md"
s = open(p, encoding="utf8").read()
tables = open("results/figures_v2/ed_tables.md", encoding="utf8").read().strip()
blocks = re.split(r"(?m)^(?=\*\*(?:Extended Data|Supplementary) Table S?\d+ \|)", tables)
pre, blocks = blocks[0], blocks[1:]
ed = [b.strip() for b in blocks if b.startswith("**Extended Data")]
su = [b.strip() for b in blocks if b.startswith("**Supplementary")]
key = lambda b: int(re.match(r"\*\*\D+(\d+)", b).group(1))
ed.sort(key=key); su.sort(key=key)
for (a, b), body in ((("<!-- ED_TABLES_START -->", "<!-- ED_TABLES_END -->"), "\n\n".join(([pre.strip()] if pre.strip() else []) + ed)),
                     (("<!-- SUPP_TABLES_START -->", "<!-- SUPP_TABLES_END -->"), "\n\n".join(su))):
    i, j = s.index(a) + len(a), s.index(b)
    s = s[:i] + "\n\n" + body + "\n\n" + s[j:]
open(p, "w", encoding="utf8").write(s)
print("inserted", len(ed), "Extended Data and", len(su), "Supplementary tables")
