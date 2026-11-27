"""Run the agent on the test set (baseline or with a LoRA adapter) and save traces + metrics."""
import argparse
import json
import os
import time

import torch

from agent import MODEL_ID, SYSTEM_PROMPT, load_model, run_agent_batch
from evaluate import evaluate, load_jsonl


