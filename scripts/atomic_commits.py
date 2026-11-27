#!/usr/bin/env python3
"""Split pending changes into many small, focused commits.

How changes are split:
  * new Python file   -> built up one top-level block at a time: a "skeleton"
                         commit (docstring, imports, constants), then one commit
                         per function / class, then the `if __name__` entry point
  * new other file    -> one commit
  * modified file     -> one commit per diff hunk
  * deleted file      -> one commit

Commits are built in the index only (the working tree is never touched), so the
final tree is exactly your working tree. Messages are "<scope>: <action>", e.g.
"part2_agent: Add parse_tool_calls() to agent.py".

Usage:
  python3 scripts/atomic_commits.py --dry-run     # show the plan
  python3 scripts/atomic_commits.py               # commit everything pending
  python3 scripts/atomic_commits.py part2_agent   # only paths under part2_agent/
"""
import argparse
import ast
import os
import re
import subprocess
import sys

# commit order: earlier patterns first; ties broken by path
ORDER = [r"^\.gitignore$", r"^README\.md$", r"^requirements", r"tools\.py$", r"make_data\.py$", r"/data/",
         r"agent\.py$", r"evaluate\.py$", r"train\.py$", r"run_eval\.py$", r"\.py$", r"\.sh$", r"\.ipynb$",
         r"\.md$", r"\.pdf$"]


def git(*args, input=None, check=True):
    r = subprocess.run(["git", *args], input=input, capture_output=True, text=isinstance(input, str) or input is None)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def scope_of(path):
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "repo"


def order_key(path):
    for i, pat in enumerate(ORDER):
        if re.search(pat, path):
            return (i, path)
    return (len(ORDER), path)


def file_mode(path):
    return "100755" if os.access(path, os.X_OK) else "100644"


def stage_blob(path, content: bytes):
    sha = git("hash-object", "-w", "--stdin", input=content).strip()
    sha = sha.decode() if isinstance(sha, bytes) else sha
    git("update-index", "--add", "--cacheinfo", f"{file_mode(path)},{sha},{path}")


def commit(msg, dry):
    if dry:
        print("  " + msg)
        return
    git("commit", "-q", "--no-verify", "-m", msg)
    print("  " + msg)


# ---------------------------------------------------------------------------
# New Python file: one commit per top-level definition
# ---------------------------------------------------------------------------
def py_chunks(path, text):
    """Return [(end_line_exclusive, message)] cumulative cut points."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines(keepends=True)
    name = os.path.basename(path)
    cuts = []  # (start_line_index, label)
    for node in tree.body:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        # pull preceding comment banners into this chunk
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cuts.append((start, f"Add {node.name}() to {name}"))
        elif isinstance(node, ast.ClassDef):
            cuts.append((start, f"Add {node.name} class to {name}"))
        elif isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            cuts.append((start, f"Add command-line entry point to {name}"))
        elif isinstance(node, ast.Assign) and cuts and cuts[-1][1].startswith("Add") and len(lines[start:node.end_lineno]) > 8:
            targets = ", ".join(t.id for t in node.targets if isinstance(t, ast.Name)) or "constants"
            cuts.append((start, f"Add {targets} to {name}"))
    if not cuts:
        return []
    chunks = []
    if cuts[0][0] > 0:
        chunks.append((cuts[0][0], f"Add {name} skeleton (docstring, imports, constants)"))
    for i, (start, label) in enumerate(cuts):
        end = cuts[i + 1][0] if i + 1 < len(cuts) else len(lines)
        if end > start:
            chunks.append((end, label))
    # merge chunks that end at the same line
    out = []
    for end, label in chunks:
        if out and out[-1][0] == end:
            continue
        out.append((end, label))
    return [(end, label) for end, label in out]


