import ast
import json
from pathlib import Path

path = Path(__file__).with_name("Capstone_Full_Demo.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4 and notebook["cells"]
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if any(line.startswith("%pip ") for line in source.splitlines()):
        source = "\n".join(line for line in source.splitlines() if not line.startswith("%pip "))
    ast.parse(source, filename=f"notebook-cell-{index}")
print(f"Notebook JSON and {len(notebook['cells'])} cells validated; Python syntax passes.")
