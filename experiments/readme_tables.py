"""Splice the rendered in-engine tables into README.md, between their markers.

    python3 experiments/readme_tables.py

`moeplot.py` owns the numbers; this only keeps the README from drifting from them.
"""
import pathlib
import re

p = pathlib.Path("README.md")
s = p.read_text()
for stem in ("disptime", "moetime"):
    tbl = pathlib.Path(f"runs/{stem}.md").read_text().split("\n\n", 1)[1].strip()
    s, n = re.subn(rf"<!-- {stem} -->\n.*?\n<!-- /{stem} -->",
                   lambda _m, t=tbl, k=stem: f"<!-- {k} -->\n{t}\n<!-- /{k} -->", s, flags=re.S)
    assert n == 1, f"{stem}: {n} marker pairs in README.md"
p.write_text(s)
print("README.md tables updated")
