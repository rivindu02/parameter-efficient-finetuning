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


