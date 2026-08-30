# stem_mcqa

Multilingual STEM multiple-choice QA, the **second environment** of the
Nemotron-3-Nano-style multi-env recipe (after `math_with_judge_lang`).

It is the upstream `mcqa` env forked with the shared **multiplicative language
penalty + format gate** (ported from infinite-rl-nemo, reused via
`nemo_gym.language_penalty`):

```
reward = mcqa_reward * language_multiplier * format_factor
```

- **`mcqa_reward`** — 1.0 iff the extracted option letter equals `expected_answer`.
  Extraction uses each row's `template_metadata.output_regex` (e.g.
  `Selected Option -> X`), falling back to the base grading modes.
- **`language_multiplier`** — enforces the per-row target reasoning language
  (`yue` colloquial Cantonese / `zh-hk` Written Traditional 書面語 / `en`), and
  forbids Simplified Chinese. Graded over the reasoning *before* `</think>`, so the
  English answer boilerplate after `</think>` does not pollute it. Wrong language
  keeps only `wrong_lang_floor` (0.1) of the reward.
- **`format_factor`** — a rollout without a closed `</think>` keeps only `floor`
  (0.2). Pair with `overlong_filtering: false` in the GRPO config.

Config: [`configs/stem_mcqa.yaml`](configs/stem_mcqa.yaml) — resources server
`stem_mcqa` + agent `stem_mcqa_simple_agent`.

## Data

The MCQA blend already ships each row in Gym shape (per-language prompt prebuilt in
`responses_create_params.input`; `options` / `expected_answer` /
`template_metadata` / `language` top-level), so prep only filters + retargets:

```bash
python3 prepare_stem_mcqa_data.py   # -> data/stem_mcqa/{train,validation,validation_smoke}.jsonl
```

Source: `nano_v3_sft_profiled_stem_mcqa.jsonl` (yue/zh-hk/en). Filters: profiled
`pass_rate >= 0.5`, and en rows whose prompt is actually Chinese (`>2` CJK chars).

## Rollout log

Every `verify()` appends a record (language / answer / reward breakdown / full
output) to `language_penalty.rollout_log_dir` as
`stem_mcqa_rollouts_<ts>_<pid>.jsonl` — analyze those offline (reuse
`benchmark/analyze_math_bench.py`, which keys on the same fields).
