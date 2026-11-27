"""Minimal tool-calling agent: generate -> parse tool call -> execute -> feed back -> repeat.

Uses the model's own chat template (Qwen2.5 style: tools in the system prompt,
calls emitted as <tool_call>{"name": ..., "arguments": {...}}</tool_call>).
`run_agent_batch` runs many independent conversations in lock-step so the GPU/CPU
generates for all active conversations in one batch.
"""
import json
import re

from tools import TOOL_SCHEMAS, TOOLS, execute_tool, tool_result_str

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Call a tool only when the request needs live data or a calculation the tools provide; "
    "otherwise answer directly. After receiving tool results, answer the user in plain text."
)

MAX_STEPS = 4
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_STRIP_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]


def load_model(model_id=MODEL_ID, adapter_path=None, device=None, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


