# code_gen_lang

The upstream **`code_gen`** env (LiveCodeBench competitive-coding unit-test
execution) forked with the **multiplicative language penalty + format gate** used
across this workspace (`math_with_judge_lang`, `stem_mcqa`; reward module ported
from infinite-rl-nemo).

```
reward = code_reward × language_multiplier × format_factor
```

- **`code_reward`** — `1.0` iff the extracted `python` program passes **all** unit
  tests, else `0.0`. Extraction (`extract_code`, LiveCodeBench `OpenAIChat` style)
  and execution (`lcb_integration/`, Ray, no sandbox) are verbatim from upstream.
- **`language_multiplier`** — enforces the per-row target reasoning language
  (`yue` colloquial Cantonese / `zh-hk` Written Traditional Chinese 書面語 / `en`),
  penalizing Simplified Chinese. Reads only the reasoning **before `</think>`**;
  the `python` code block after `</think>` (full of English keywords) never
  pollutes the check. Correct-code-wrong-language keeps `wrong_lang_floor` (0.1).
- **`format_factor`** — `1.0` if a closed `</think>` is present, else `floor`
  (0.2). Pair with `overlong_filtering: false` so truncated loopers reach the
  optimizer as negative advantage.

## Why not the upstream `reasoning_format_penalty`?

Upstream flags any `<think>`/`</think>` still in `output_text` — correct only when
a reasoning **parser** splits reasoning from the answer. Our policy runs
`uses_reasoning_parser: false`, so `<think>...</think>` stays **inline** in
`output_text`; the upstream check would false-positive on every rollout. We drop
it and use the shared multiplicative `format_gate` instead.

## Data

Blend `nano_v3_sft_profiled_comp_coding_50tests.jsonl` — already Gym-shaped
(per-language prompt prebuilt in `responses_create_params.input`,
`verifier_metadata.unit_tests` in LiveCodeBench stdin/stdout format, `language`,
`pass_rate` top-level). `prepare_code_gen_lang_data.py` filters `pass_rate >= 0.5`
+ en-CJK contamination and retargets `agent_ref -> code_gen_lang_simple_agent`;
it does **not** rebuild prompts.

## Files

- `app.py` — fork of the upstream verify() with the language/format multiplier.
- `configs/code_gen_lang.yaml` — `code_gen_lang` server + `code_gen_lang_simple_agent`.
- `lcb_integration/` — copied verbatim from `code_gen` (the unit-test runner).
- rollouts logged to `logs/rollouts/code_gen_lang_rollouts_*.jsonl`.
- Add `resources_servers/code_gen_lang` to `prebuild_gym_server_venvs.sh` before a run.
