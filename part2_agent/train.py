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


def show_masking(tok, example):
    """Print which part of one example carries loss (sanity check)."""
    ids, lab = example["input_ids"], example["labels"]
    kept = tok.decode([t for t, l in zip(ids, lab) if l != -100])
    print(f"tokens: {len(ids)}, supervised: {sum(l != -100 for l in lab)}")
    print("supervised text:\n" + kept)


def train(train_path, out_dir, model_id=MODEL_ID, epochs=2, lr=2e-4, batch_size=8, micro_batch=4,
          r=16, alpha=32, dropout=0.05, max_len=1024, seed=42, max_steps=None, log_every=10):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"  # fp16 autocast on T4 (no bf16), fp32 on CPU
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    with open(train_path) as f:
        data = [encode(tok, json.loads(l)["messages"], max_len) for l in f]
    show_masking(tok, data[0])

    accum = max(batch_size // micro_batch, 1)
    steps_per_epoch = math.ceil(len(data) / batch_size)
    total = steps_per_epoch * epochs if max_steps is None else max_steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    step, t0, rng, log = 0, time.time(), random.Random(seed), []
    for ep in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for s in range(0, len(order), batch_size):
            if step >= total:
                break
            idx = order[s:s + batch_size]
            loss_sum = 0.0
            for m in range(0, len(idx), micro_batch):
                b = collate([data[i] for i in idx[m:m + micro_batch]], tok.pad_token_id)
                b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(**b).loss / accum
                scaler.scale(loss).backward()
                loss_sum += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            log.append({"step": step, "epoch": ep, "loss": loss_sum, "lr": sched.get_last_lr()[0]})
            if step % log_every == 0 or step == 1:
                print(f"step {step}/{total} epoch {ep} loss {loss_sum:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump({"config": dict(model_id=model_id, epochs=epochs, lr=lr, batch_size=batch_size, r=r,
                                  alpha=alpha, dropout=dropout, seed=seed, n_train=len(data), steps=step,
                                  seconds=time.time() - t0), "log": log}, f, indent=1)
    print(f"saved adapter to {out_dir} after {step} steps, {time.time() - t0:.0f}s")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(os.path.dirname(__file__), "data", "train.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outputs", "lora_adapter"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train(a.train, a.out, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, micro_batch=a.micro_batch,
          r=a.r, max_steps=a.max_steps, seed=a.seed)
