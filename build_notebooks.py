"""Build self-contained Colab notebooks from the source modules.

part*/part*_*.ipynb : standalone, modules embedded with %%writefile (no clone needed).
part*/run_colab.ipynb: git-based runners; outputs saved back from Colab are kept for
                       every cell whose code did not change. Re-run this script after editing a module:
    python3 build_notebooks.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def writefile(folder, name):
    with open(os.path.join(ROOT, folder, name)) as f:
        return code(f"%%writefile {name}\n" + f.read())


def _src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def save(cells, path, keep_outputs=False):
    """Write a notebook. keep_outputs=True (runner notebooks saved back from Colab):
    cells whose code is unchanged keep their outputs, and the Colab badge cell is kept."""
    full = os.path.join(ROOT, path)
    meta = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
    if keep_outputs and os.path.exists(full):
        old = json.load(open(full))
        meta = old.get("metadata", meta)
        by_src = {_src(c): c for c in old["cells"]}
        badge = [c for c in old["cells"][:1] if "colab-badge" in _src(c)]
        merged, kept = [], 0
        for c in cells:
            prev = by_src.get(_src(c))
            if prev is not None and c["cell_type"] == "code":
                c = dict(c, outputs=prev.get("outputs", []), execution_count=prev.get("execution_count"))
                kept += bool(c["outputs"])
            merged.append(c)
        cells = badge + merged
        print(f"  kept outputs of {kept} unchanged cells")
    nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}
    with open(full, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


