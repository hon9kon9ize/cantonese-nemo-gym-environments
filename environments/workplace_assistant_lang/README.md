# workplace_assistant_lang

The upstream **`workplace_assistant`** env (multi-step tool-using agent — the model
issues tool calls across up to 6 turns; reward = did those calls reach the
ground-truth end state) forked with the **multiplicative language penalty + format
gate** used across this workspace (`math_with_judge_lang`, `stem_mcqa`,
`code_gen_lang`; reward module ported from infinite-rl-nemo).

```
reward = wb_reward × language_multiplier × format_factor
```

- **`wb_reward`** — `is_correct(predicted_tool_calls, ground_truth)`, a **binary**
  state-diff check (execute the predicted calls and the ground-truth calls, compare
  the resulting tool-env state). Grader + tool suite are **reused** from the base
  env via the `resources_servers` namespace package (no copy).
- **`language_multiplier`** — enforces the per-row target reasoning language
  (`yue` / `zh-hk` 書面語 / `en`), penalizing Simplified Chinese.
- **`format_factor`** — `1.0` if the trajectory closes a `</think>`, else `floor`
  (0.2).

## Whole-trajectory language grading (the multi-turn wrinkle)

A tool-use rollout has **one `<think>…</think>` per turn** (the chat template
pre-opens `<think>` each assistant turn). The shared single-block extractor would
therefore score only turn 1. Instead the fork takes the full concatenated model
text (`response.output_text` across all turns), strips the think tags, re-wraps as
one block, and calls `language_multiplier` with `reasoning_template=False` — so
**every turn's reasoning** is judged (the "grade the whole response, not only
`<think>`" lesson). Tool **calls** are separate `function_call` items (not
`output_text`), so they never enter the language check.

## Data

Blend `nano_v3_sft_profiled_workbench.jsonl` — already Gym-shaped
(`responses_create_params.input` = system+user, `tools` = 27 schemas, `ground_truth`
= expected tool calls, `category`, `environment_name`, `language`, `pass_rate`).
`prepare_workplace_assistant_lang_data.py` filters `pass_rate >= 0.5` + en-CJK and
retargets `agent_ref -> workplace_assistant_lang_simple_agent`; it does **not**
rebuild prompts.

## Files

- `app.py` — fork: reuses base `get_tools`/`is_correct` + `seed_session`/tool routing,
  adds the language/format layer in `verify()`.
- `configs/workplace_assistant_lang.yaml` — server + `workplace_assistant_lang_simple_agent`
  (`max_steps: 6`).
- rollouts logged to `logs/rollouts/workplace_assistant_lang_rollouts_*.jsonl`.
- Add `resources_servers/workplace_assistant_lang` to `prebuild_gym_server_venvs.sh`
  before a run (its venv imports the base env's tool modules, so the base
  `requirements.txt` — `-e nemo-gym[dev]` — is enough).
