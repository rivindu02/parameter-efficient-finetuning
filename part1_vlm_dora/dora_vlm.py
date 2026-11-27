"""Part 1: does DoRA's "learning pattern" claim transfer to a vision-language model?

DoRA (Liu et al., 2024) Sec. 2 / Fig. 2: decompose each weight W into magnitude
m = ||W||_c and direction V = W/||W||_c. Over training checkpoints and layers,
full fine-tuning (FT) shows a NEGATIVE correlation between the magnitude change
dM and the direction change dD, LoRA shows a POSITIVE one, and DoRA behaves like
FT (negative). We repeat the analysis on SmolVLM-256M finetuned on DocVQA, and
split it by tower (vision encoder / connector / language model).

Pipeline
  prepare_data()  -> cache a DocVQA subset (images as JPEG on disk + jsonl)
  train_method()  -> train "full" | "lora" | "dora", recording dM/dD per target
                     layer at several checkpoints (stats only, no weight dumps)
  evaluate_anls() -> ANLS on the held-out subset (zero-shot or finetuned)
  analyze()       -> correlations / slopes + Fig.2-style scatter plots

Definitions (per weight matrix, rows = output neurons, matching peft's DoRA):
  dM_t = mean_n | ||W_t[n]|| - ||W_0[n]|| |
  dD_t = mean_n ( 1 - cos(W_t[n], W_0[n]) )
For LoRA/DoRA W_t is the merged weight (DoRA: m * (W0+sBA)/||W0+sBA||).
"""
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field

import torch

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
PROMPT_SUFFIX = "\nAnswer with a single word or short phrase."


