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


def _messages(question, answer=None):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + PROMPT_SUFFIX}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model_and_processor(cfg):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as AutoVLM
    proc = AutoProcessor.from_pretrained(cfg.model_id)
    proc.image_processor.size = {"longest_edge": cfg.image_longest_edge}
    model = AutoVLM.from_pretrained(cfg.model_id, dtype=torch.float32)
    return model, proc


def target_regex(targets):
    vis = r".*vision_model.*\.(q_proj|k_proj|v_proj|out_proj)$"
    txt = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
    con = r".*connector\.modality_projection\.proj$"
    parts = {"all": [vis, con, txt], "vision": [vis], "text": [txt]}[targets]
    return "|".join(f"(?:{p})" for p in parts)


def tower_of(name):
    if "vision_model" in name:
        return "vision"
    if "connector" in name:
        return "connector"
    return "text"


def proj_of(name):
    leaf = name.split(".")[-1]
    return {"out_proj": "o_proj"}.get(leaf, leaf)


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


# ----------------------------------------------------------------------------
# Magnitude / direction tracking
# ----------------------------------------------------------------------------
def _clean_name(n):
    return n.replace("base_model.model.", "")


def find_targets(model, regex):
    """name -> module for every Linear-like module to track (peft-wrapped or plain)."""
    pat = re.compile(regex)
    out = {}
    for n, mod in model.named_modules():
        cn = _clean_name(n)
        if pat.fullmatch(cn) and (hasattr(mod, "lora_A") or isinstance(mod, torch.nn.Linear)):
            out[cn] = mod
    return out


@torch.no_grad()
def effective_weight(mod):
    if hasattr(mod, "lora_A") and len(mod.active_adapters) > 0:
        a = mod.active_adapters[0]
        W0 = mod.base_layer.weight.float()
        A, B = mod.lora_A[a].weight.float(), mod.lora_B[a].weight.float()
        W = W0 + mod.scaling[a] * (B @ A)
        if getattr(mod, "use_dora", {}).get(a, False):
            m = mod.lora_magnitude_vector[a].weight.float().view(-1, 1)
            W = m * W / W.norm(dim=1, keepdim=True)
        return W
    return mod.weight.float()


@torch.no_grad()
def snapshot_stats(targets, W0s, step):
    rows = []
    for name, mod in targets.items():
        W, W0 = effective_weight(mod), W0s[name]
        nW, nW0 = W.norm(dim=1), W0.norm(dim=1)
        dM = (nW - nW0).abs().mean().item()
        dD = (1 - torch.nn.functional.cosine_similarity(W, W0, dim=1)).mean().item()
        rows.append({"step": step, "module": name, "tower": tower_of(name), "proj": proj_of(name),
                     "layer": layer_of(name), "dM": dM, "dD": dD})
    return rows


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _find_sub(seq, sub):
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def make_batch(proc, examples, device):
    texts = [proc.apply_chat_template(_messages(e["question"], e["answers"][0]), add_generation_prompt=False)
             for e in examples]
    images = [[_load_image(e["image"])] for e in examples]
    proc.tokenizer.padding_side = "right"
    enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
    labels = torch.full_like(enc["input_ids"], -100)
    marker = proc.tokenizer.encode("Assistant:", add_special_tokens=False)
    for i, ids in enumerate(enc["input_ids"].tolist()):
        j = _find_sub(ids, marker)
        n = int(enc["attention_mask"][i].sum())
        if j >= 0:  # supervise only the answer (+ end-of-utterance token)
            labels[i, j + len(marker):n] = enc["input_ids"][i, j + len(marker):n]
        else:
            print("WARNING: 'Assistant:' marker not found; example gets no loss")
    enc["labels"] = labels
    return {k: v.to(device) for k, v in enc.items()}


def show_masking(cfg, n=2):
    """Sanity check: decode the supervised tokens of a few training examples."""
    _, proc = load_model_and_processor(cfg)
    b = make_batch(proc, load_split(cfg, cfg.train_split)[:n], "cpu")
    for i in range(n):
        ids = b["input_ids"][i][b["labels"][i] != -100]
        print(f"seq len {int(b['attention_mask'][i].sum())}, supervised {len(ids)} tokens: "
              f"{proc.tokenizer.decode(ids)!r}")


def run_name(cfg: Config, method: str):
    """File tag of a run: "dora" for all targets, "dora_vision" / "dora_text" for a single-tower ablation."""
    return method if cfg.targets == "all" else f"{method}_{cfg.targets}"


def train_method(cfg: Config, method: str):
    """method in {"full", "lora", "dora"}. Saves stats_<run_name>.json + weights."""
    assert method in ("full", "lora", "dora")
    name = run_name(cfg, method)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    model, proc = load_model_and_processor(cfg)
    regex = target_regex(cfg.targets)

    if method == "full":
        for p in model.parameters():
            p.requires_grad = True
        lr = cfg.lr_full
    else:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                          target_modules=regex, use_dora=(method == "dora"), init_lora_weights=True)
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
        lr = cfg.lr_adapter
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    targets = find_targets(model, regex)
    W0s = {n: effective_weight(m).clone() for n, m in targets.items()}
    print(f"tracking {len(targets)} weight matrices "
          f"({sum(tower_of(n) == 'vision' for n in targets)} vision, "
          f"{sum(tower_of(n) == 'connector' for n in targets)} connector, "
          f"{sum(tower_of(n) == 'text' for n in targets)} text)")

    data = load_split(cfg, cfg.train_split)
    accum = max(cfg.batch_size // cfg.micro_batch, 1)
    total = math.ceil(len(data) / cfg.batch_size) * cfg.epochs
    snap_at = sorted({max(1, round(total * (k + 1) / cfg.n_snapshots)) for k in range(cfg.n_snapshots)})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    stats, log, step, t0 = [], [], 0, time.time()
    rng = random.Random(cfg.seed)
    for ep in range(cfg.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            loss_sum = 0.0
            for m in range(0, len(idx), cfg.micro_batch):
                b = make_batch(proc, [data[i] for i in idx[m:m + cfg.micro_batch]], device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(**b).loss / accum
                scaler.scale(loss).backward()
                loss_sum += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            log.append({"step": step, "loss": loss_sum})
            if step % 10 == 0 or step == 1:
                print(f"[{method}] step {step}/{total} loss {loss_sum:.4f} {time.time() - t0:.0f}s", flush=True)
            if step in snap_at:
                stats += snapshot_stats(targets, W0s, step)

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, f"stats_{method}.json"), "w") as f:
        json.dump({"config": asdict(cfg), "method": method, "steps": step, "seconds": time.time() - t0,
                   "snap_at": snap_at, "stats": stats, "log": log}, f)
    save_dir = os.path.join(cfg.out_dir, f"model_{method}")
    if method == "full":
        model.save_pretrained(save_dir, safe_serialization=True)
    else:
        model.save_pretrained(save_dir)
    print(f"[{method}] done in {time.time() - t0:.0f}s; stats + weights in {cfg.out_dir}")
    model.eval()
    return model, proc


def load_trained(cfg, method):
    """Reload a finished run (model_<method> on Drive) for re-evaluation without retraining."""
    path = os.path.join(cfg.out_dir, f"model_{method}")
    model, proc = load_model_and_processor(cfg)
    if method == "full":
        model = type(model).from_pretrained(path, dtype=torch.float32)
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval(), proc


def needs_eval(cfg, name):
    """True if eval_<name>.json is missing or from the old format without all predictions."""
    p = os.path.join(cfg.out_dir, f"eval_{name}.json")
    return not os.path.exists(p) or "predictions" not in json.load(open(p))


# ----------------------------------------------------------------------------
# Evaluation: ANLS (DocVQA metric)
# ----------------------------------------------------------------------------
def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, answers, tau=0.5):
    p = " ".join(pred.strip().lower().split())
    best = 0.0
    for a in answers:
        a = " ".join(a.strip().lower().split())
        nl = _levenshtein(p, a) / max(len(p), len(a), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best


def _norm(s, strip_punct=False):
    s = " ".join(s.strip().lower().split())
    return s.rstrip(" .,;:") if strip_punct else s


def score_predictions(preds, data):
    """Official-style ANLS / exact match, plus variants that ignore trailing punctuation
    (the zero-shot model often ends answers with '.', which finetuning removes)."""
    out = {}
    for suffix, strip in [("", False), ("_norm", True)]:
        sc = [anls(_norm(p, strip), [_norm(a, strip) for a in e["answers"]]) for p, e in zip(preds, data)]
        em = [float(any(_norm(p, strip) == _norm(a, strip) for a in e["answers"])) for p, e in zip(preds, data)]
        out["anls" + suffix] = 100 * sum(sc) / len(sc)
        out["exact_match" + suffix] = 100 * sum(em) / len(em)
    return out


@torch.no_grad()
def evaluate_anls(cfg, model, proc, name):
    device = next(model.parameters()).device
    data = load_split(cfg, cfg.eval_split)[:cfg.n_eval]
    model.eval()
    proc.tokenizer.padding_side = "left"
    preds = []
    for i in range(0, len(data), cfg.eval_batch):
        chunk = data[i:i + cfg.eval_batch]
        texts = [proc.apply_chat_template(_messages(e["question"]), add_generation_prompt=True) for e in chunk]
        enc = proc(text=texts, images=[[_load_image(e["image"])] for e in chunk],
                   return_tensors="pt", padding=True).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=False)
        dec = proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        preds += [d.strip() for d in dec]
    res = {"name": name, "n": len(data), **score_predictions(preds, data),
           "predictions": [{"q": e["question"], "pred": p, "answers": e["answers"]} for p, e in zip(preds, data)]}
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, f"eval_{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{name}] ANLS {res['anls']:.2f}  EM {res['exact_match']:.2f}  "
          f"(ignoring trailing punctuation: ANLS {res['anls_norm']:.2f}  EM {res['exact_match_norm']:.2f})  n={len(data)}")
    return res


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def _pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def _spearman(x, y):
    rank = lambda v: [sorted(v).index(a) for a in v]  # noqa: E731  (ties are rare for floats)
    return _pearson(rank(x), rank(y))


def correlation_table(stats, groups=("tower", "proj")):
    """Correlation of dM vs dD.

    pooled:       all (layer, checkpoint) points of a group together (as in DoRA Fig. 2)
    within_ckpt:  mean over checkpoints of the across-layer correlation, which removes
                  the trivial "both grow with training time" trend.
    """
    from collections import defaultdict
    by = defaultdict(list)
    for r in stats:
        by["ALL"].append(r)
        by[f"{r[groups[0]]}/ALL"].append(r)
        by["/".join(str(r[g]) for g in groups)].append(r)
    out = []
    for key, rows in sorted(by.items()):
        x, y = [r["dD"] for r in rows], [r["dM"] for r in rows]
        steps = sorted({r["step"] for r in rows})
        within = [_pearson([r["dD"] for r in rows if r["step"] == s], [r["dM"] for r in rows if r["step"] == s])
                  for s in steps]
        within = [w for w in within if not math.isnan(w)]
        out.append({"group": key, "n": len(rows),
                    "pearson_pooled": _pearson(x, y), "spearman_pooled": _spearman(x, y),
                    "pearson_within_ckpt": sum(within) / len(within) if within else float("nan")})
    return out


def analyze(out_dir, methods=("full", "lora", "dora"), proj_filter=("q_proj", "v_proj")):
    """Print correlation tables; return {method: table}. proj_filter=None uses all projections."""
    res = {}
    for m in methods:
        p = os.path.join(out_dir, f"stats_{m}.json")
        if not os.path.exists(p):
            continue
        stats = json.load(open(p))["stats"]
        if proj_filter:
            stats = [r for r in stats if r["proj"] in proj_filter or r["tower"] == "connector"]
        res[m] = correlation_table(stats)
        print(f"\n== {m} ==  {'group':<22}{'n':>5}{'r_pooled':>10}{'rho_pooled':>12}{'r_within':>10}")
        for r in res[m]:
            print(f"{'':>10}{r['group']:<22}{r['n']:>5}{r['pearson_pooled']:>10.3f}"
                  f"{r['spearman_pooled']:>12.3f}{r['pearson_within_ckpt']:>10.3f}")
    return res


def plot_scatter(out_dir, methods=("full", "lora", "dora"), tower="text", proj="q_proj", fname=None):
    """DoRA Fig. 2 style: dD (x) vs dM (y), colour = checkpoint, one panel per method."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.6))
    for ax, m in zip(axes, methods):
        p = os.path.join(out_dir, f"stats_{m}.json")
        if not os.path.exists(p):
            continue
        st = [r for r in json.load(open(p))["stats"] if r["tower"] == tower and r["proj"] == proj]
        steps = sorted({r["step"] for r in st})
        cmap = plt.get_cmap("viridis")
        for i, s in enumerate(steps):
            pts = [r for r in st if r["step"] == s]
            ax.scatter([r["dD"] for r in pts], [r["dM"] for r in pts], s=14,
                       color=cmap(i / max(len(steps) - 1, 1)), label=f"step {s}")
        x, y = [r["dD"] for r in st], [r["dM"] for r in st]
        if len(x) > 2:
            mx, my = sum(x) / len(x), sum(y) / len(y)
            sl = sum((a - mx) * (b - my) for a, b in zip(x, y)) / max(sum((a - mx) ** 2 for a in x), 1e-30)
            xs = [min(x), max(x)]
            ax.plot(xs, [my + sl * (v - mx) for v in xs], "k--", lw=1)
            ax.set_title(f"{m.upper()}  r={_pearson(x, y):.2f}")
        ax.set_xlabel("ΔD (direction)")
        ax.set_ylabel("ΔM (magnitude)")
        ax.ticklabel_format(style="sci", scilimits=(-2, 2))
    axes[-1].legend(fontsize=6, loc="best")
    fig.suptitle(f"{tower} tower, {proj}")
    fig.tight_layout()
    fname = fname or os.path.join(out_dir, f"fig_{tower}_{proj}.png")
    fig.savefig(fname, dpi=150)
    return fname
