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


def pred_calls(trace):
    return [c for st in trace["steps"] if st["valid"] for c in st["calls"]]


def format_ok(st):
    """Turn parses as <tool_call>{json with name + object arguments}</tool_call>, whatever the tool name."""
    return bool(st["attempted"] and (st["valid"] or any(
        isinstance(c.get("arguments"), dict) for c in st["calls"])))


def hallucinated_tool(st):
    return bool(st["attempted"] and not st["valid"] and any(c.get("name") not in TOOLS for c in st["calls"]))


def attempted_calls(trace):
    """Every parsed call in order, including calls to tools that do not exist."""
    return [c for st in trace["steps"] if st["valid"] or hallucinated_tool(st) for c in st["calls"]]


def _stem(w):
    return w if len(w) <= 4 else w[:max(4, len(w) - 3)]


def str_ok(value, text):
    """Keyword check robust to inflection: 'drizzle' matches 'drizzling', 'snow' matches 'snowing'."""
    value, text = value.lower(), text.lower()
    if value in text:
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", value) if w not in ("and", "the", "of")]
    return bool(words) and all(re.search(r"\b" + re.escape(_stem(w)), text) for w in words)


def calls_match(pred, gold):
    return len(pred) == len(gold) and all(
        p["name"] == g["name"] and args_match(g["name"], p["arguments"], g["arguments"])
        for p, g in zip(pred, gold))


def answer_ok(final, checks):
    if final is None:
        return False
    low = final.lower()
    nums = [_num(m) for m in _NUM_RE.findall(final)]
    for c in checks:
        if c["type"] == "str" and not str_ok(c["value"], low):
            return False
        if c["type"] == "num" and not any(_close(n, float(c["value"])) for n in nums):
            return False
    return True


# ----------------------------------------------------------------------------
# Per-example scoring + failure category
# ----------------------------------------------------------------------------
def score_example(ex, tr):
    gold = ex["gold_calls"]
    needs_tool = bool(gold)
    first = tr["steps"][0] if tr["steps"] else {"attempted": False, "valid": False, "calls": []}
    any_attempt = any(st["attempted"] for st in tr["steps"])
    pcalls = pred_calls(tr)
    r = {"id": ex["id"], "category": ex["category"], "needs_tool": needs_tool}
    if needs_tool:
        r["valid"] = format_ok(first)
        r["call_acc"] = calls_match(pcalls, gold)
        r["success"] = answer_ok(tr["final"], ex["checks"])
    else:
        r["abstain"] = not any_attempt
        r["success"] = r["abstain"] and answer_ok(tr["final"], ex["checks"])
    r["failure"] = None if r["success"] else categorize(ex, tr, pcalls)
    return r


FAILURE_CATEGORIES = [
    "missed_call",        # tool needed but model answered directly (often hallucinated a value)
    "wrong_format",       # tried to call a tool but output does not parse in the expected format
    "wrong_tool",         # parsable call to the wrong tool (or to a tool that does not exist)
    "wrong_argument",     # right tool, wrong arguments
    "incomplete_chain",   # multi-step task: stopped after the first call
    "extra_calls",        # repeated / unnecessary extra calls, often hitting the step limit
    "ignored_result",     # calls correct but final answer does not use the tool result
    "unnecessary_call",   # no tool needed but model called one
    "wrong_answer",       # no tool needed, no call, but answer is wrong
]


def categorize(ex, tr, pcalls):
    gold = ex["gold_calls"]
    any_attempt = any(st["attempted"] for st in tr["steps"])
    any_invalid = any(st["attempted"] and not st["valid"] for st in tr["steps"])
    if not gold:
        return "unnecessary_call" if any_attempt else "wrong_answer"
    if not any_attempt:
        return "missed_call"
    pcalls = attempted_calls(tr)
    if any_invalid and not pcalls:
        return "wrong_format"
    # compare call by call
    for i, g in enumerate(gold):
        if i >= len(pcalls):
            return "wrong_format" if any_invalid else "incomplete_chain"
        p = pcalls[i]
        if p["name"] != g["name"]:
            return "wrong_tool"
        if not args_match(g["name"], p["arguments"], g["arguments"]):
            return "wrong_argument"
    if len(pcalls) > len(gold):
        return "extra_calls"
    if any(st["attempted"] and not format_ok(st) for st in tr["steps"]):
        return "wrong_format"
    return "ignored_result"


def summarize(rows):
    tool = [r for r in rows if r["needs_tool"]]
    notool = [r for r in rows if not r["needs_tool"]]
    pct = lambda xs, k: 100.0 * sum(x[k] for x in xs) / max(len(xs), 1)  # noqa: E731
    return {
        "n_total": len(rows), "n_tool": len(tool), "n_no_tool": len(notool),
        "call_validity": pct(tool, "valid"),
        "call_accuracy": pct(tool, "call_acc"),
        "abstention": pct(notool, "abstain"),
        "task_success": pct(rows, "success"),
        "task_success_tool": pct(tool, "success"),
        "task_success_multi": pct([r for r in tool if r["category"] in ("weather_f", "stock_shares")], "success"),
        "failures": dict(Counter(r["failure"] for r in rows if r["failure"])),
    }


def evaluate(test, traces):
    rows = [score_example(ex, tr) for ex, tr in zip(test, traces)]
    return rows, summarize(rows)


def compare(rows_base, rows_lora):
    """Transitions per example between baseline and LoRA (fixed / persisted / introduced)."""
    out = {"fixed": Counter(), "persisted": Counter(), "introduced": Counter()}
    for b, l in zip(rows_base, rows_lora):
        if b["failure"] and not l["failure"]:
            out["fixed"][b["failure"]] += 1
        elif b["failure"] and l["failure"]:
            out["persisted"][f"{b['failure']} -> {l['failure']}"] += 1
        elif l["failure"] and not b["failure"]:
            out["introduced"][l["failure"]] += 1
    return {k: dict(v) for k, v in out.items()}


