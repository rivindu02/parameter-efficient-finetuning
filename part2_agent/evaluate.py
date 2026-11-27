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


# ----------------------------------------------------------------------------
# Argument normalisation / call matching
# ----------------------------------------------------------------------------
def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _close(a, b, rel=2e-3, abs_=0.5):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def args_match(name, pred, gold):
    if not isinstance(pred, dict):
        return False
    if name == "calculator":
        try:
            return _close(eval_expression(pred.get("expression")), eval_expression(gold["expression"]), rel=1e-4)
        except Exception:  # noqa: BLE001
            return False
    if name == "convert_units":
        return (_close(_num(pred.get("value")), _num(gold["value"]), rel=1e-4)
                and canon_unit(pred.get("from_unit")) == canon_unit(gold["from_unit"])
                and canon_unit(pred.get("to_unit")) == canon_unit(gold["to_unit"]))
    if name == "get_weather":
        return str(pred.get("city", "")).strip().lower() == gold["city"].lower()
    if name == "get_stock_price":
        return str(pred.get("ticker", "")).strip().upper() == gold["ticker"].upper()
    if name == "days_between":
        return (parse_date(pred.get("start_date", "")) == parse_date(gold["start_date"])
                and parse_date(pred.get("end_date", "")) == parse_date(gold["end_date"]))
    return pred == gold


