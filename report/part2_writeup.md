# Part 2: LoRA fine-tuning a small LLM for tool calling

**Setup.**

- **Model:** Qwen2.5-0.5B-Instruct (494M parameters), using its native tool-calling chat template.
- **Tools:** five tools with made-up data: `calculator`, `convert_units`, `get_weather` (12 cities, invented values), `get_stock_price` (8 invented tickers) and `days_between`.
- **Agent loop:** generate a reply (greedy), parse `<tool_call>{json}</tool_call>`, run the tool and append the result as a `tool` message; stop on a plain-text reply or after 4 steps.
- **Data (templates):** 800 train requests (160 need no tool, 100 need a two-step chain: weather→°F, price→total) and 100 test requests (20 no tool, 20 two-step; 91 unique). Test phrasings come from separate templates, and test dates are written in words ("March 1, 2024") while train uses ISO dates.
- **LoRA:** r = 16, α = 32, dropout 0.05 on all linear layers (8.8M trainable parameters, 1.8%); lr 2e-4 cosine, batch 8, 2 epochs (200 steps), fp16 autocast; loss only on assistant turns (tool calls and final answers). Colab T4, seed 42, 8.1 min; train loss 0.60 → 0.011.

**Baseline vs. LoRA** (same 100 test requests, same system prompt):

| Metric | Base | Base + stronger prompt¹ | **LoRA** |
|---|---|---|---|
| Call validity (parses in the expected format) | 86.2 | 93.8 | **96.2** |
| Call accuracy (right tool and arguments, in order) | 52.5 | 56.2 | **93.8** |
| Abstention (no call when none is needed, n = 20) | 60.0 | 25.0 | **100.0** |
| Task success (final answer correct, end to end) | 54.0 | 49.0 | **93.0** |
| — two-step tasks only (n = 20) | 10.0 | 0.0 | **100.0** |

¹ The stronger prompt adds a format example, "never guess live values", "one tool at a time" and "dates as YYYY-MM-DD". This column was scored before a small scorer fix (keyword stemming; calls to non-existent tools counted as wrong tool rather than wrong format). The fix changed base task success by only 2 points.

**Failure analysis.** Each failed request was assigned one category automatically, and every case was then checked by hand. The base model fails 46 of 100 requests; LoRA fails 7. LoRA has fewer than 10 failures, so all 7 are listed.

| Category | Base | LoRA | Typical base behaviour |
|---|---|---|---|
| Incomplete chain | 12 | 0 | Gets the weather in °C, then answers without converting to °F. Gets the price, then never multiplies by the number of shares. |
| Missed call | 11 | 3 | Asks the user for the data ("please tell me the ticker") or makes up a price ("$50 per share"). |
| Unnecessary call | 8 | 0 | Calls `get_weather("New York")` for "Who first walked on the Moon?" |
| Wrong argument | 7 | 0 | Wrong date range, unit direction reversed (kg→lb), `993 + 40 * 9` instead of `(993 + 40) * 9`, typo `NIBM` |
| Wrong tool | 6 | 2 | `get_weather(city="KRYO")` for a stock; invents `split_into_parts` |
| Wrong answer (no tool needed) | 2 | 2 | — |

**What LoRA fixed, kept and introduced.**

- **Fixed (39 of 46):** every incomplete chain, unnecessary call and wrong argument, 8 of 11 missed calls and 4 of 6 wrong-tool cases. LoRA learned to chain a second call after reading the first result, to leave general-knowledge questions alone, and to copy argument values exactly.
- **Kept (7):** three test phrasings unlike anything in training. "Split 180 into 45 equal parts" (2): both models invent a `split_into_parts` tool. "Before I invest, tell me what NIMB costs per share" (3): LoRA says it has no real-time data, or makes up a price. "Which country is home to Kyoto?" (2, a duplicated request): LoRA answers "Kyoto is home to the city of Kyoto".
- **Introduced:** none. No request that the base model solved fails after fine-tuning.

**Could a better prompt alone have been enough? No.**

The stronger prompt raised format validity (86→94%) but made the base model call tools far too often (abstention 60→25%, unnecessary calls 8→15), and it did not teach chaining (two-step success 10→0%, 18 incomplete chains). At 0.5B parameters, instructions shift the overall tendency to call a tool but do not teach *when* to call; showing the full call→result→answer pattern through LoRA does. Caveat: test requests share the tool set, cities and tickers of training, so the 93% measures how well this tool set was learned, not tool use in general.
