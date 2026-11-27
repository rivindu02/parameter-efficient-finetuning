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


