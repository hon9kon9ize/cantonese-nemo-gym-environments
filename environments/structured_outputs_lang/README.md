# structured_outputs_lang

Multilingual (yue / zh-hk / en) **structured-output generation** — given a JSON Schema
and a Cantonese/Chinese news article, emit a document in the target format
(**JSON / YAML / XML / TOML / CSV**) that conforms to the schema. Forked from the base
`structured_outputs` env with the shared multiplicative language penalty + format gate
(same recipe as `code_gen_lang`, `instruction_following_lang`, `math_with_judge_lang`,
`stem_mcqa`):

```
reward = so_reward * language_multiplier * format_factor
```

- **`so_reward`** — 1.0 iff the ANSWER (text after `</think>`) parses in the target
  format AND satisfies the schema.
- **`language_multiplier`** — enforces the per-row reasoning language and forbids
  Simplified Chinese (`nemo_gym.language_penalty`, `wrong_lang_floor=0.1`,
  `min_target_script_ratio=0.15`).
- **`format_factor`** — closed `</think>` required, else `floor=0.2`.

## Two differences vs the base env

1. **Five formats.** The base env only parses **json / yaml / xml**; this data is
   **27% toml + csv** (schema_type mix: json 605 / toml 286 / yaml 298 / csv 75 /
   xml 70), which the base env always scores 0. We reuse the **vendored**
   `cantonese_rlvr.so_verify` (`cantonese_rlvr/`) — the exact reward the
   `infinite-rl-nemo` training env uses (`nemo_reward._so_reward`), giving reward
   PARITY. It handles all five:
   - **json / yaml / toml** → full JSON-Schema validation (`jsonschema.validate`).
     The nano_v3 schemas are self-strict (`additionalProperties:false` + explicit
     `required`), so plain validation enforces structure — no strictify override.
   - **xml / csv** → well-formed + required top-level keys / columns present
     (structural; these formats aren't type-preserving).
2. **schema_only (structural, no grounding).** Our rows carry **no gold
   `expected_data`**, so grading is purely structural — right fields + types in the
   right format, *not* content grounding. `so_verify`'s title-grounding is a no-op
   here (no expected title). This matches the base env's schema-validation philosophy;
   add `expected_data` later for value grounding.

## `<think>` handling

The base env parses the raw output; with reasoning inline
(`uses_reasoning_parser: false`) that would never parse. We extract the answer AFTER
`</think>` first (`so_verify.parse()` then tolerates a markdown code fence), and use
the shared `format_gate` for the closed-`</think>` check. The language check reads only
the reasoning BEFORE `</think>`, so the JSON/English keys in the answer never pollute it.

## Dependencies

The verifier is vendored and depends only on `jsonschema` + `pyyaml` (already in the
container), vendored `toml` (`cantonese_rlvr/_vendor/`), and stdlib `json/csv/xml`. So
— unlike the base env — it needs **neither** `openapi-schema-validator` **nor**
`xmltodict`; `requirements.txt` adds nothing beyond `nemo_gym`.

## Data

`prepare_structured_outputs_lang_data.py` (workspace root) builds
`data/{train,validation,validation_smoke}.jsonl` from
`nano_v3_sft_profiled_structured_outputs.jsonl`. **No `pass_rate` filter** — that field
is a PLACEHOLDER (not yet profiled with our model). It retargets `agent_ref`,
stratifies by language, and sanity-checks that every `schema_str` parses.

## Benchmark

`benchmark/benchmark_structured_outputs_lang_job.sh` profiles a deployed vLLM policy
against the exact `verify()` GRPO will optimize — per-language AND **per-format**
pass rate (format difficulty varies a lot), plus the language / format multipliers.
