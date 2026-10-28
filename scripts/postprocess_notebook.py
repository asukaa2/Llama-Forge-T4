#!/usr/bin/env python3
"""
postprocess_notebook.py — Two fixes to train_llama_from_scratch.ipynb:

1. Insert a "git clone the repo" cell at the top of Section 1 (Setup), so when
   the notebook is opened via the Colab badge it clones Llama-Forge-T4 into
   /content and train.py is available alongside the notebook.

2. Set `collapsed: true` on ALL code cells (except Config / training loop /
   verification) so the notebook opens clean — only markdown section headers
   are visible. Users click to expand any cell they want to inspect or edit.
"""
import json
from pathlib import Path

NB = Path("/home/z/my-project/train_llama_from_scratch.ipynb")

with open(NB) as f:
    nb = json.load(f)

# ---------------------------------------------------------------------------
# 1. Build the git clone cell — insert right after the existing Drive mount cell
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/asukaa2/Llama-Forge-T4"
REPO_DIR = "/content/Llama-Forge-T4"

clone_cell = {
    "cell_type": "code",
    "metadata": {"collapsed": False},
    "source": [
        "# Clone the repo so train.py is available alongside the notebook.\n",
        "# (Useful if you want to run `python train.py ...` instead of cells, or\n",
        "#  if you forked the repo and want your changes picked up.)\n",
        f"!rm -rf {REPO_DIR}\n",
        f"!git clone --depth 1 {REPO_URL} {REPO_DIR}\n",
        f"!cp -n {REPO_DIR}/train.py /content/train.py 2>/dev/null || true\n",
        "!ls -la /content/train.py\n",
    ],
    "outputs": [],
    "execution_count": None,
}

# Find Drive mount cell
mount_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if "drive.mount" in src:
            mount_idx = i
            break

if mount_idx is None:
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            if "torch.cuda.is_available" in src:
                mount_idx = i
                break

assert mount_idx is not None, "Couldn't find Drive mount cell"
print(f"Inserting git clone cell after cell index {mount_idx}")
nb["cells"].insert(mount_idx + 1, clone_cell)

# ---------------------------------------------------------------------------
# 2. Collapse all code cells except Config / training loop / verification / clone
# ---------------------------------------------------------------------------
clone_idx = mount_idx + 1
config_idx = loop_idx = verify_idx = None

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if config_idx is None and "DRIVE_PATH" in src and "TOKEN_BUDGET" in src and "#@param" in src:
        config_idx = i
    if loop_idx is None and "while True:" in src and "Gradient accumulation" in src:
        loop_idx = i
    if verify_idx is None and "VERIFICATION CELL" in src:
        verify_idx = i

keep_open = {clone_idx, config_idx, loop_idx, verify_idx}
print(f"Keeping expanded: clone={clone_idx}  config={config_idx}  "
      f"loop={loop_idx}  verify={verify_idx}")

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        c.setdefault("metadata", {})
        c["metadata"]["collapsed"] = (i not in keep_open)

# ---------------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------------
with open(NB, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
n_collapsed = sum(1 for c in nb["cells"]
                  if c["cell_type"] == "code" and c.get("metadata", {}).get("collapsed"))
print(f"\nWrote {NB}")
print(f"  total cells: {len(nb['cells'])}")
print(f"  code cells : {n_code}")
print(f"  collapsed  : {n_collapsed}")
print(f"  expanded   : {n_code - n_collapsed}")
