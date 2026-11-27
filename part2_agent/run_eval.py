"""Run the agent on the test set (baseline or with a LoRA adapter) and save traces + metrics."""
import argparse
import json
import os
import time

import torch

from agent import MODEL_ID, SYSTEM_PROMPT, load_model, run_agent_batch
from evaluate import evaluate, load_jsonl


def run(name, test_path, out_dir, adapter=None, batch_size=16, limit=None, system_prompt=SYSTEM_PROMPT):
    torch.manual_seed(42)
    test = load_jsonl(test_path)[:limit]
    model, tok = load_model(MODEL_ID, adapter_path=adapter)
    t0 = time.time()
    traces = run_agent_batch(model, tok, [ex["query"] for ex in test], system_prompt=system_prompt,
                             batch_size=batch_size)
    rows, summary = evaluate(test, traces)
    summary["seconds"] = time.time() - t0
    summary["system_prompt"] = system_prompt
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"traces_{name}.jsonl"), "w") as f:
        for ex, tr, r in zip(test, traces, rows):
            tr = {k: v for k, v in tr.items() if k != "messages"}
            f.write(json.dumps({"id": ex["id"], "category": ex["category"], **tr, "score": r}, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, f"summary_{name}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(name, json.dumps(summary, indent=1))
    return traces, rows, summary


