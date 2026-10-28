#!/usr/bin/env python3
"""
Fix collapse: add tags:["hide_input"] to every code cell that should be
collapsed. Modern Colab hides input via this tag, not via metadata.collapsed.
"""
import json
from pathlib import Path

NB = Path("/home/z/my-project/train_llama_from_scratch.ipynb")
with open(NB) as f:
    nb = json.load(f)

# Cells that should STAY expanded (visible by default):
# - git clone (run once)
# - all 5 Config cells (forms)
# - training loop (progress logs)
# - verification cell
KEEP_EXPANDED = {5, 8, 9, 10, 11, 12, 13, 28, 33}

n_collapsed = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    c.setdefault("metadata", {})
    if i in KEEP_EXPANDED:
        c["metadata"]["collapsed"] = False
        c["metadata"]["tags"] = []   # no hide_input tag
    else:
        c["metadata"]["collapsed"] = True
        c["metadata"]["tags"] = ["hide_input"]   # ← modern Colab way to hide input
        n_collapsed += 1

with open(NB, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {NB}")
print(f"  cells with hide_input tag: {n_collapsed}")
print(f"  cells expanded           : {len(KEEP_EXPANDED)}")

# Verify
print("\nVerification:")
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    tags = c.get("metadata", {}).get("tags", [])
    hidden = "hide_input" in tags
    first = "".join(c["source"]).split("\n")[0][:60]
    mark = "▶" if hidden else "▼"
    print(f"  {i:2d} {mark} {first}")
