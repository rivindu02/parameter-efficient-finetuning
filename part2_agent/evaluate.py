"""Metrics and failure analysis for agent traces.

Metrics (computed per test example, then averaged over the relevant subset):
  call_validity : tool-needed examples whose FIRST model turn parses as a tool call
                  in the expected <tool_call>{"name": ..., "arguments": {...}}</tool_call>
                  format (format only; calling a non-existent tool counts as wrong_tool).
  call_accuracy : tool-needed examples whose full sequence of calls matches the gold
                  sequence (right tools, right arguments, right order).
  abstention    : no-tool examples where the model answered without calling a tool.
  task_success  : all examples; final plain-text answer satisfies every gold check
                  (and, for no-tool examples, no tool was called).
"""
import argparse
import json
import math
import re
from collections import Counter

from tools import TOOLS, canon_unit, eval_expression, parse_date

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


