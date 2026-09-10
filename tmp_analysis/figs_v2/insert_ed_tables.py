"""Insert results/figures_v2/ed_tables.md into docs/PAPER_v2.md between the
ED_TABLES markers (idempotent)."""
p = "docs/PAPER_v2.md"
s = open(p, encoding="utf8").read()
tables = open("results/figures_v2/ed_tables.md", encoding="utf8").read().strip()
a, b = "<!-- ED_TABLES_START -->", "<!-- ED_TABLES_END -->"
i, j = s.index(a) + len(a), s.index(b)
s = s[:i] + "\n\n" + tables + "\n\n" + s[j:]
open(p, "w", encoding="utf8").write(s)
print("inserted", tables.count("\n") + 1, "lines of tables")
