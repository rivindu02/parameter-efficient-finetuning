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


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
def parse_tool_calls(text):
    """Return (attempted, valid, calls).

    attempted: the output looks like an attempt to call a tool (tagged or raw JSON).
    valid:     every call parses as JSON in the expected <tool_call> format, names a
               known tool, and has an object of arguments.
    calls:     list of {"name", "arguments"} (only when valid).
    """
    blocks = _TOOL_CALL_RE.findall(text)
    stripped = text.strip()
    raw_json_attempt = (stripped.startswith("{") and '"name"' in stripped) or "<tool_call>" in text
    if not blocks:
        return (raw_json_attempt, False, []) if raw_json_attempt else (False, False, [])
    calls = []
    for b in blocks:
        try:
            obj = json.loads(b)
        except json.JSONDecodeError:
            return True, False, []
        if not isinstance(obj, dict):
            return True, False, []
        name, args = obj.get("name"), obj.get("arguments", obj.get("parameters"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return True, False, []
        if name not in TOOLS or not isinstance(args, dict):
            return True, False, [{"name": name, "arguments": args}]
        calls.append({"name": name, "arguments": args})
    return True, True, calls


def clean(text):
    for t in _STRIP_TOKENS:
        text = text.replace(t, "")
    return text.strip()


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def generate_batch(model, tok, conversations, max_new_tokens=200):
    import torch
    prompts = [tok.apply_chat_template(c, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)
               for c in conversations]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
    new = out[:, enc["input_ids"].shape[1]:]
    return [clean(tok.decode(row, skip_special_tokens=False)) for row in new]


