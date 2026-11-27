"""Build self-contained Colab notebooks from the source modules.

part*/part*_*.ipynb : standalone, modules embedded with %%writefile (no clone needed).
part*/run_colab.ipynb: git-based runners; outputs saved back from Colab are kept for
                       every cell whose code did not change. Re-run this script after editing a module:
    python3 build_notebooks.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def writefile(folder, name):
    with open(os.path.join(ROOT, folder, name)) as f:
        return code(f"%%writefile {name}\n" + f.read())


def _src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def save(cells, path, keep_outputs=False):
    """Write a notebook. keep_outputs=True (runner notebooks saved back from Colab):
    cells whose code is unchanged keep their outputs, and the Colab badge cell is kept."""
    full = os.path.join(ROOT, path)
    meta = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
    if keep_outputs and os.path.exists(full):
        old = json.load(open(full))
        meta = old.get("metadata", meta)
        by_src = {_src(c): c for c in old["cells"]}
        badge = [c for c in old["cells"][:1] if "colab-badge" in _src(c)]
        merged, kept = [], 0
        for c in cells:
            prev = by_src.get(_src(c))
            if prev is not None and c["cell_type"] == "code":
                c = dict(c, outputs=prev.get("outputs", []), execution_count=prev.get("execution_count"))
                kept += bool(c["outputs"])
            merged.append(c)
        cells = badge + merged
        print(f"  kept outputs of {kept} unchanged cells")
    nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}
    with open(full, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


SETUP = """
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install -U "transformers>=4.46" peft accelerate datasets
import os, sys
USE_DRIVE = True   # checkpoints + results go to Google Drive so a disconnect is not fatal
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT = '/content/drive/MyDrive/EN4554/{part}'
else:
    OUT = '/content/{part}_outputs'
os.makedirs(OUT, exist_ok=True)
os.makedirs('/content/{part}', exist_ok=True)
%cd /content/{part}
sys.path.insert(0, '/content/{part}')
SEED = 42
print('outputs ->', OUT)
"""

GIT_SETUP = """
# ---- config: code comes from GitHub, results go to Google Drive ----
REPO_URL = 'https://github.com/rivindu02/parameter-efficient-finetuning.git'   # <- your repo
BRANCH = 'main'
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install -U "transformers>=4.46" peft accelerate datasets
import os, sys, subprocess
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/EN4554/{part}'
os.makedirs(OUT, exist_ok=True)
url = REPO_URL
try:  # private repo: add a Colab secret GH_TOKEN (read-only fine-grained token) with notebook access
    from google.colab import userdata
    url = REPO_URL.replace('https://', f"https://{userdata.get('GH_TOKEN')}@")
except Exception:
    pass
if not os.path.exists('/content/repo/.git'):
    subprocess.run(['git', 'clone', '-q', '-b', BRANCH, url, '/content/repo'], check=True)
else:
    subprocess.run(['git', '-C', '/content/repo', 'fetch', '-q', 'origin', BRANCH], check=True)
    subprocess.run(['git', '-C', '/content/repo', 'reset', '-q', '--hard', f'origin/{BRANCH}'], check=True)
print(subprocess.run(['git', '-C', '/content/repo', 'log', '-1', '--oneline'], capture_output=True, text=True).stdout)
%cd /content/repo/{part}
sys.path.insert(0, '/content/repo/{part}')
SEED = 42
print('outputs ->', OUT)
"""

PULL = code("""
# re-run after pushing code changes: syncs to the latest commit (fetch + reset) and reloads modules
import importlib, subprocess
subprocess.run(['git', '-C', '/content/repo', 'fetch', '-q', 'origin', BRANCH], check=True)
subprocess.run(['git', '-C', '/content/repo', 'reset', '-q', '--hard', f'origin/{BRANCH}'], check=True)
print(subprocess.run(['git', '-C', '/content/repo', 'log', '-1', '--oneline'], capture_output=True, text=True).stdout)
for m in list(sys.modules):
    if m in MODULES: importlib.reload(sys.modules[m])
""")


def runner(cells, part, modules):
    """Turn a standalone notebook into a git-based runner: swap setup, drop %%writefile cells."""
    out = []
    for c in cells:
        src = c["source"]
        if c["cell_type"] == "code" and src.startswith("%%writefile"):
            continue
        if c["cell_type"] == "code" and "drive.mount" in src and "USE_DRIVE" in src:
            extra = "!pip -q install matplotlib" if "matplotlib" in src else ""
            out.append(code(GIT_SETUP.replace("{part}", part) + f"\nMODULES = {modules!r}\n" + extra))
            out.append(PULL)
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------
p2 = [
    md("""
# Part 2 — LLM agent: tool calling with LoRA

Model **Qwen2.5-0.5B-Instruct** · 5 tools with made-up data · template dataset (800 train / 100 test, disjoint phrasings)
· baseline vs LoRA (loss on assistant turns only) · metrics: call validity, call accuracy, abstention, task success · failure analysis.

Runtime: *Runtime → Change runtime type → T4 GPU*. Seed 42 everywhere. Whole notebook ≈ 20–30 min on a T4.
"""),
    code(SETUP.replace("{part}", "part2_agent")),
    md("## 1. Source modules (tools, data generator, agent loop, evaluation, training)"),
    *[writefile("part2_agent", f) for f in ["tools.py", "make_data.py", "agent.py", "evaluate.py", "train.py", "run_eval.py"]],
    md("## 2. Data\nTemplates for train and test never overlap; the date task uses ISO dates in train and written dates (`March 1, 2024`) in test."),
    code("""
!python make_data.py --out data --seed 42
from evaluate import load_jsonl
from collections import Counter
train, test = load_jsonl('data/train.jsonl'), load_jsonl('data/test.jsonl')
print('train', Counter(r['category'] for r in train))
print('test ', Counter(r['category'] for r in test))
for r in test[:5]:
    print('-', r['query'], '=>', [c['name'] for c in r['gold_calls']] or 'NO TOOL')
"""),
    md("## 3. The agent loop on the untouched model (demo)"),
    code("""
import torch, json
from agent import load_model, chat, SYSTEM_PROMPT, MODEL_ID
from tools import TOOL_SCHEMAS
model, tok = load_model(MODEL_ID)
print(tok.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':'weather in Oslo?'}],
                              tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)[-600:])
for q in ['How much would 12 shares of ZORB cost?', 'What is the capital of Japan?',
          'Count the days from March 3, 2024 to May 1, 2024.']:
    print('='*80); print('Q:', q); chat(model, tok, q)
del model; torch.cuda.empty_cache()
"""),
    md("## 4. Baseline evaluation"),
    code("""
from run_eval import run
base_traces, base_rows, base_sum = run('base', 'data/test.jsonl', OUT, adapter=None, batch_size=16)
"""),
    md("## 5. LoRA finetuning (loss on assistant turns only)\nr=16, alpha=32, dropout 0.05, all linear layers, lr 2e-4 cosine, 2 epochs, batch 8, fp16 autocast."),
    code("""
from train import train, encode
from transformers import AutoTokenizer
import os
ADAPTER = os.path.join(OUT, 'lora_adapter')
if os.path.exists(os.path.join(ADAPTER, 'adapter_config.json')):
    print('adapter already trained, skipping (delete the folder to retrain)')
else:
    train('data/train.jsonl', ADAPTER, epochs=2, lr=2e-4, batch_size=8, micro_batch=4, r=16, alpha=32, seed=SEED)
"""),
    code("""
import json, matplotlib.pyplot as plt
log = json.load(open(os.path.join(ADAPTER, 'train_log.json')))
print(log['config'])
plt.figure(figsize=(5,3)); plt.plot([l['step'] for l in log['log']], [l['loss'] for l in log['log']])
plt.xlabel('step'); plt.ylabel('train loss (assistant tokens)'); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'train_loss.png'), dpi=150); plt.show()
"""),
    md("## 6. LoRA evaluation (same test set, same system prompt)"),
    code("""
lora_traces, lora_rows, lora_sum = run('lora', 'data/test.jsonl', OUT, adapter=ADAPTER, batch_size=16)
"""),
    md("## 7. Baseline vs LoRA"),
    code("""
from evaluate import print_table, compare
print_table({'base': base_sum, 'lora': lora_sum})
trans = compare(base_rows, lora_rows)
print(json.dumps(trans, indent=1))
json.dump({'base': base_sum, 'lora': lora_sum, 'transitions': trans}, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
"""),
    md("## 8. Failure analysis\nAll failures of each model, with the raw model turns. Categories are assigned automatically and then checked by hand."),
    code("""
def show_failures(name, traces, rows, k=15):
    fails = [(ex, tr, r) for ex, tr, r in zip(test, traces, rows) if r['failure']]
    print(f'##### {name}: {len(fails)} failures (showing {min(k, len(fails))})')
    for ex, tr, r in fails[:k]:
        print(f"\\n[{r['failure']}] {ex['id']} ({ex['category']}) Q: {ex['query']}")
        print('   gold calls:', [(c['name'], c['arguments']) for c in ex['gold_calls']] or 'none')
        for i, st in enumerate(tr['steps']):
            print(f'   turn {i}:', st['raw'][:300].replace('\\n', ' '))
        print('   final:', (tr['final'] or '')[:200], '| stop:', tr['stop'])
show_failures('BASE', base_traces, base_rows)
"""),
    code("show_failures('LORA', lora_traces, lora_rows)"),
    md("""
### Notes for the write-up
* Which categories did LoRA fix / leave / introduce? → see `transitions` above.
* Could a better prompt alone have been enough? Try the cell below: a stronger system prompt with a format example, on the base model.
"""),
    code("""
STRONG_PROMPT = SYSTEM_PROMPT + (
    ' To call a tool, reply ONLY with <tool_call>{"name": <tool>, "arguments": {...}}</tool_call>. '
    'Never guess weather, stock prices, conversions, date differences or arithmetic: use the tool. '
    'Call one tool at a time; when you have the result, give the final answer with the number from the result. '
    'Dates must be YYYY-MM-DD. For questions that need no tool, answer directly without any tool call.')
pt_traces, pt_rows, pt_sum = run('base_prompt', 'data/test.jsonl', OUT, adapter=None, batch_size=16,
                                 system_prompt=STRONG_PROMPT)
print_table({'base': base_sum, 'base+prompt': pt_sum, 'lora': lora_sum})
json.dump({'base': base_sum, 'base_strong_prompt': pt_sum, 'lora': lora_sum, 'transitions': trans},
          open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
"""),
]

# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------
p1 = [
    md("""
# Part 1 — Does DoRA's learning pattern transfer to a vision-language model?

**Paper claim (DoRA, Liu et al. 2024, Fig. 2):** decompose each weight into magnitude and direction. Across layers and
checkpoints, full fine-tuning shows a **negative** correlation between magnitude change ΔM and direction change ΔD; LoRA
shows a **positive** one; DoRA's pattern resembles full fine-tuning (negative).

**Hypothesis:** the same holds when SmolVLM-256M is fine-tuned on DocVQA — in the language model *and* in the vision
encoder (bonus: per-tower analysis: vision encoder / connector / LM).

Setup: SmolVLM-256M-Instruct · DocVQA subset (2000 train / 300 validation) · LoRA vs DoRA (r=8, α=16) vs full FT on the
same targets (q,k,v,o of both towers + connector) · 1 epoch · ΔM/ΔD recorded at 8 checkpoints · ANLS for accuracy.
Runtime: T4 GPU. Each training run ≈ 15–30 min; every result is saved to Drive, so cells can be re-run after a disconnect.
"""),
    code(SETUP.replace("{part}", "part1_vlm_dora") + "\n!pip -q install matplotlib"),
    writefile("part1_vlm_dora", "dora_vlm.py"),
    md("## 1. Config + data subset (cached as JPEG on local disk)"),
    code("""
import json, os, torch, gc
import dora_vlm as dv
SMOKE = False   # True: tiny run (16 train / 8 eval, ~5 min) to check the pipeline end to end
if SMOKE:
    cfg = dv.Config(out_dir=OUT + '/smoke', data_dir='/content/data_docvqa_smoke', seed=SEED,
                    n_train=16, n_eval=8, n_snapshots=2, eval_batch=4)
else:
    cfg = dv.Config(out_dir=OUT, data_dir='/content/data_docvqa', seed=SEED)
print(cfg)
dv.prepare_data(cfg)
dv.show_masking(cfg)   # only the answer tokens should be supervised
"""),
    md("## 2. Zero-shot baseline"),
    code("""
if not os.path.exists(os.path.join(OUT, 'eval_zeroshot.json')):
    model, proc = dv.load_model_and_processor(cfg)
    dv.evaluate_anls(cfg, model.to('cuda'), proc, 'zeroshot')
    del model; gc.collect(); torch.cuda.empty_cache()
"""),
    md("## 3. Training runs (one cell each, skipped if results already on Drive)"),
    *[code(f"""
if not os.path.exists(os.path.join(OUT, 'eval_{m}.json')):
    model, proc = dv.train_method(cfg, '{m}')
    dv.evaluate_anls(cfg, model, proc, '{m}')
    del model; gc.collect(); torch.cuda.empty_cache()
""") for m in ["lora", "dora", "full"]],
    md("## 4. Accuracy: zero-shot vs adapted"),
    code("""
rows = []
for n in ['zeroshot', 'lora', 'dora', 'full']:
    p = os.path.join(OUT, f'eval_{n}.json')
    if os.path.exists(p):
        r = json.load(open(p)); rows.append((n, r['anls'], r['exact_match']))
print(f"{'model':<10}{'ANLS':>8}{'EM':>8}")
for n, a, e in rows: print(f'{n:<10}{a:>8.2f}{e:>8.2f}')
"""),
    md("""
## 5. Claim test: ΔM vs ΔD correlation
`r_pooled`: all (layer, checkpoint) points, as in the paper's Fig. 2. `r_within`: mean across-layer correlation per checkpoint
(removes the trivial "both grow over time" trend). Paper's prediction: **full < 0, LoRA > 0, DoRA < 0**.
First q/v projections (the modules analysed in the paper), then all projections.
"""),
    code("corr_qv = dv.analyze(OUT, proj_filter=('q_proj', 'v_proj'))"),
    code("corr_all = dv.analyze(OUT, proj_filter=None)\njson.dump({'qv': corr_qv, 'all': corr_all}, open(os.path.join(OUT, 'correlations.json'), 'w'), indent=1)"),
    md("## 6. Fig. 2-style scatter plots, per tower"),
    code("""
from IPython.display import Image, display
for tower in ['text', 'vision']:
    for proj in ['q_proj', 'v_proj']:
        display(Image(dv.plot_scatter(OUT, tower=tower, proj=proj)))
"""),
    md("""
## 7. Bonus: where does the adaptation happen?
Mean ΔD and ΔM per tower at the last checkpoint: does each method move the vision encoder, the connector or the language
model the most, and does DoRA's magnitude/direction split behave differently in the vision encoder?
"""),
    code("""
from collections import defaultdict
for m in ['full', 'lora', 'dora']:
    p = os.path.join(OUT, f'stats_{m}.json')
    if not os.path.exists(p): continue
    st = json.load(open(p))['stats']; last = max(r['step'] for r in st)
    agg = defaultdict(list)
    for r in st:
        if r['step'] == last: agg[r['tower']].append(r)
    print(m, {t: dict(dD=f"{sum(r['dD'] for r in v)/len(v):.2e}", dM=f"{sum(r['dM'] for r in v)/len(v):.2e}", n=len(v))
              for t, v in agg.items()})
"""),
]

if __name__ == "__main__":
    # standalone notebooks (code embedded with %%writefile) -- submission fallback
    save(p2, "part2_agent/part2_agent.ipynb")
    save(p1, "part1_vlm_dora/part1_dora_vlm.ipynb")
    # git-based runners -- daily workflow
    save(runner(p2, "part2_agent", ["tools", "make_data", "agent", "evaluate", "train", "run_eval"]),
         "part2_agent/run_colab.ipynb", keep_outputs=True)
    save(runner(p1, "part1_vlm_dora", ["dora_vlm"]), "part1_vlm_dora/run_colab.ipynb", keep_outputs=True)
