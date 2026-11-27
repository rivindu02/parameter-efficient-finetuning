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


