# Parameter-Efficient Fine-Tuning

**EN4554 Deep Learning for Vision, Assignment 1.**

This repo has two small studies of LoRA-style parameter-efficient fine-tuning. Both run inside one free Google Colab session on a T4 GPU.

| Part | Question | Model | Data |
|---|---|---|---|
| **1. DoRA on a vision-language model** | Does DoRA's "learning pattern" claim from NLP also hold for a vision-language model? | SmolVLM-256M-Instruct | DocVQA subset (2,000 train / 300 validation) |
| **2. Tool-calling agent** | How much does LoRA fine-tuning improve a small LLM's tool calls inside an agent loop? | Qwen2.5-0.5B-Instruct | Template-generated requests (800 train / 100 held-out test) |

---

## Part 1: DoRA learning pattern in a VLM

**Paper claim.** DoRA [(Liu et al., 2024)](https://arxiv.org/abs/2402.09353) splits each pretrained weight into a *magnitude* part and a *direction* part. Its Fig. 2 makes this claim:
- In full fine-tuning, the magnitude change (ΔM) and the direction change (ΔD) are **negatively** correlated across layers and checkpoints.
- In LoRA they are **positively** correlated.
- DoRA behaves like full fine-tuning.

**Hypothesis.** The same pattern appears when SmolVLM-256M is fine-tuned for document VQA. We test this separately in the **vision encoder**, the **vision–language connector** and the **language model**.

**Method.**
- Three runs: full fine-tuning, LoRA and DoRA (r = 8, α = 16). All three update the same weights: the q, k, v and o attention projections in both towers, plus the connector.
- ΔM and ΔD are recorded per layer at 8 checkpoints.
- We compare the sign of the ΔM–ΔD correlation across the three methods, both pooled (as in the paper) and within each checkpoint.
- DocVQA ANLS measures accuracy. Results are reported for zero-shot, LoRA, DoRA and full fine-tuning.

**Files** (in `part1_vlm_dora/`):
- `dora_vlm.py`: data caching, training, ΔM/ΔD tracking, ANLS evaluation and analysis plots.
- `run_colab.ipynb`: Colab runner that pulls the code from this repo.
- `part1_dora_vlm.ipynb`: standalone notebook with all code embedded.

## Part 2: LoRA fine-tuning for tool calling

**Agent.** The loop runs Goal → Act → Observe → Done:
1. Generate a reply.
2. Parse any `<tool_call>{json}</tool_call>`.
3. Run the tool and feed its result back to the model.
4. Stop on a plain-text answer, or after 4 steps.

**Tools.** Five tools; the lookup tools return made-up data, so the model cannot answer from memory:
- `calculator`
- `convert_units`
- `get_weather` (made-up data)
- `get_stock_price` (made-up tickers)
- `days_between`

**Data.**
- 800 training and 100 test requests. 20% need no tool, and some need a two-step chain.
- Train and test use separate phrasing templates.
- Test dates are written in words (e.g. "March 1, 2024") instead of ISO format.

**Training.** LoRA with r = 16, α = 32 on all linear layers. 2 epochs, fp16. The loss covers only the assistant turns.

**Metrics.**
- **Call validity:** the output parses as a tool call in the expected format.
- **Call accuracy:** right tool and right arguments.
- **Abstention:** correctly answers without a tool when none is needed.
- **Task success:** the final answer is correct end to end.

A failure analysis sorts errors into categories (missed call, wrong format, wrong tool, wrong argument, incomplete chain, ignored result, unnecessary call). It also compares a stronger prompt alone against LoRA.

**Files** (in `part2_agent/`):

| File | Purpose |
|---|---|
| `tools.py` | Tools, their JSON schemas and the dispatcher |
| `make_data.py` | Dataset generator; `data/` holds the generated files |
| `agent.py` | Batched agent loop and tool-call parser |
| `evaluate.py` | Metrics, failure categories, baseline-vs-LoRA comparison |
| `train.py` | LoRA training with loss on assistant turns only |
| `run_eval.py` | Runs the test set and saves traces and metrics |
| `run_colab.ipynb` / `part2_agent.ipynb` | Git-based Colab runner / standalone notebook |

---

## Running

**Colab (T4 GPU):**
1. Open `part*/run_colab.ipynb` in Colab.
2. Select a T4 runtime.
3. Click **Run all**.

The runner clones this repo and writes every result (metrics JSON, agent traces, adapters, plots) to `MyDrive/EN4554/<part>/`, so a disconnect is not fatal. Training cells skip work that is already finished.

**Pulling results to your machine:** set up an rclone remote named `gdrive` once, then:

```bash
./sync_results.sh        # metrics, traces, logs, plots -> colab_results/
./sync_results.sh --all  # also model weights
```

The sync only downloads from Drive. It never changes anything on Drive.

**Local CPU check.** The Part 2 scripts also run on CPU. For example, `python part2_agent/run_eval.py --name base --limit 16` evaluates the base model on 16 test requests.

**Regenerating the standalone notebooks after editing the modules:**

```bash
python3 build_notebooks.py
```

## Reproducibility

- Seed 42 everywhere.
- fp16 mixed precision on the GPU (the T4 has no bf16).
- Dataset subset sizes are fixed and listed above.
- Each run saves its full configuration next to its results.

## References

- S.-Y. Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation.* arXiv:2402.09353, 2024.
- E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- A. Marafioti et al. *SmolVLM: Redefining small and efficient multimodal models.* 2025.
- Qwen Team. *Qwen2.5 Technical Report.* 2024.
