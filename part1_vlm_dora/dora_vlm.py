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


@dataclass
class Config:
    model_id: str = MODEL_ID
    dataset: str = "HuggingFaceM4/DocumentVQA"
    train_split: str = "train"
    eval_split: str = "validation"
    n_train: int = 2000
    n_eval: int = 300
    data_dir: str = "data_docvqa"
    out_dir: str = "outputs"
    image_longest_edge: int = 1024     # SmolVLM splits into 512px tiles: <= 2x2 tiles + 1 global
    seed: int = 42
    epochs: int = 1
    batch_size: int = 8
    micro_batch: int = 4
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    lr_adapter: float = 1e-4           # LoRA and DoRA
    lr_full: float = 1e-5              # full finetuning
    targets: str = "all"               # "all" | "vision" | "text"
    n_snapshots: int = 8               # checkpoints at which dM/dD are recorded
    max_new_tokens: int = 24
    eval_batch: int = 16
    extra: dict = field(default_factory=dict)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def prepare_data(cfg: Config):
    """Stream a subset of DocVQA, resize images, cache them as JPEG + jsonl. Idempotent."""
    from datasets import load_dataset
    for split, n in [(cfg.train_split, cfg.n_train), (cfg.eval_split, cfg.n_eval)]:
        meta_path = os.path.join(cfg.data_dir, f"{split}.jsonl")
        if os.path.exists(meta_path) and sum(1 for _ in open(meta_path)) >= n:
            print(f"{split}: cached at {meta_path}")
            continue
        img_dir = os.path.join(cfg.data_dir, split)
        os.makedirs(img_dir, exist_ok=True)
        # shuffle buffer scales with the subset size, so tiny runs do not download thousands of images
        ds = load_dataset(cfg.dataset, split=split, streaming=True).shuffle(
            seed=cfg.seed, buffer_size=min(2000, 4 * n))
        recs = []
        for ex in ds:
            answers = [a for a in (ex.get("answers") or []) if a]
            if not answers:
                continue
            img = ex["image"].convert("RGB")
            img.thumbnail((cfg.image_longest_edge, cfg.image_longest_edge))
            path = os.path.join(img_dir, f"{len(recs):05d}.jpg")
            img.save(path, quality=90)
            recs.append({"image": path, "question": ex["question"], "answers": answers})
            if len(recs) >= n:
                break
        with open(meta_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: wrote {len(recs)} examples to {meta_path}")


def load_split(cfg, split):
    with open(os.path.join(cfg.data_dir, f"{split}.jsonl")) as f:
        return [json.loads(l) for l in f]


def _load_image(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


