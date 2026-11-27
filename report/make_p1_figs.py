"""Figures and numbers for the Part 1 report, from the synced run outputs.

    python report/make_p1_figs.py [results_dir]     (default: colab_results/part1_vlm_dora)
"""
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "colab_results/part1_vlm_dora"
HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = [("full", "Full FT"), ("lora", "LoRA"), ("dora", "DoRA")]
COL = {"full": "#444444", "lora": "#2b7bba", "dora": "#d1495b"}


