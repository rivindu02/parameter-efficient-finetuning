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


def commit_new_file(path, dry):
    with open(path, "rb") as f:
        data = f.read()
    scope = scope_of(path)
    chunks = []
    if path.endswith(".py"):
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        chunks = [("".join(lines[:end]).encode(), label) for end, label in py_chunks(path, text)]
    if len(chunks) <= 1:
        stage_blob(path, data) if not dry else None
        commit(f"{scope}: Add {path.split('/')[-1] if scope != 'repo' else path}", dry)
        return 1
    for content, label in chunks:
        if not dry:
            stage_blob(path, content)
        commit(f"{scope}: {label}", dry)
    if chunks[-1][0] != data and not dry:  # trailing text after the last block
        stage_blob(path, data)
        commit(f"{scope}: Finish {os.path.basename(path)}", dry)
    return len(chunks)


# ---------------------------------------------------------------------------
# Modified file: one commit per hunk
# ---------------------------------------------------------------------------
def commit_hunks(path, dry):
    diff = git("diff", "--no-color", "--", path)
    if not diff.strip():
        return 0
    header, _, body = diff.partition("\n@@")
    hunks = ["@@" + h for h in ("\n@@" + body).split("\n@@")[1:]]
    name = os.path.basename(path)
    if path.endswith(".ipynb") or "Binary files" in diff or len(hunks) == 1:
        if not dry:
            git("add", "--", path)
        commit(f"{scope_of(path)}: Update {name}", dry)
        return 1
    n = 0
    for h in hunks:
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@ ?(.*)", h)
        ctx = m.group(2).strip() if m else ""
        fn = re.search(r"(?:def|class)\s+(\w+)", ctx)
        where = f"{fn.group(1)}()" if fn else f"line {m.group(1) if m else '?'}"
        if not dry:
            r = subprocess.run(["git", "apply", "--cached", "--recount", "-"], input=header + "\n" + h + "\n",
                               capture_output=True, text=True)
            if r.returncode != 0:  # hunk no longer applies on its own: stage the rest of the file
                git("add", "--", path)
                commit(f"{scope_of(path)}: Update {name}", dry)
                return n + 1
        commit(f"{scope_of(path)}: Update {where} in {name}", dry)
        n += 1
    if not dry and git("diff", "--name-only", "--", path).strip():
        git("add", "--", path)
        commit(f"{scope_of(path)}: Update {name}", dry)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="limit to these paths / folders")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.chdir(git("rev-parse", "--show-toplevel").strip())
    if git("diff", "--cached", "--name-only").strip():
        sys.exit("index has staged changes; commit or unstage them first (git restore --staged .)")
    spec = a.paths or ["."]
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *spec).split()
    modified = git("diff", "--name-only", "--diff-filter=M", "--", *spec).split()
    deleted = git("diff", "--name-only", "--diff-filter=D", "--", *spec).split()
    total = 0
    for p in sorted(deleted):
        if not a.dry_run:
            git("rm", "-q", "--cached", "--", p)
        commit(f"{scope_of(p)}: Remove {os.path.basename(p)}", a.dry_run)
        total += 1
    for p in sorted(modified, key=order_key):
        total += commit_hunks(p, a.dry_run)
    for p in sorted(untracked, key=order_key):
        total += commit_new_file(p, a.dry_run)
    print(f"{'planned' if a.dry_run else 'made'} {total} commits")


if __name__ == "__main__":
    main()
