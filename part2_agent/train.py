"""LoRA finetuning with loss on assistant turns only.

The full gold conversation (system + tools, user, assistant tool-call turns, tool
results, final assistant answer) is rendered with the model's chat template.
Labels are -100 everywhere except tokens inside assistant turns
(`<|im_start|>assistant\n ... <|im_end|>`), found via the tokenizer offset mapping.
"""
import argparse
import json
import math
import os
import random
import re
import time

import torch

from agent import MODEL_ID, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS

_ASSIST_RE = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tok, messages, max_len):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    text = tok.apply_chat_template(msgs, tools=TOOL_SCHEMAS, tokenize=False)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=max_len)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    spans = [m.span(1) for m in _ASSIST_RE.finditer(text)]
    labels = [-100] * len(ids)
    for i, (a, b) in enumerate(offs):
        if b > a and any(s <= a and b <= e for s, e in spans):
            labels[i] = ids[i]
    return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    L = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, x in enumerate(batch):  # right padding for training
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        lab[i, :n] = torch.tensor(x["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


