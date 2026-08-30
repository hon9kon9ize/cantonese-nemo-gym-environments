# Cantonese-RL NeMo-Gym environments

Six custom **NeMo-Gym** environments for reinforcement-learning a Cantonese-reasoning
LLM. Each is a fork of a stock NeMo-Gym environment that adds a **multiplicative
language reward** on top of the task verifier:

```
reward = task_reward × language_multiplier × format_factor
```

- **`language_multiplier`** enforces the row's target *reasoning* language — `yue`
  (colloquial Cantonese, judged by a vendored `cantofilter`), `zh` / `zh-hk`
  (Standard Written Chinese, mostly-Han script), or `en` — and penalizes Simplified
  Chinese via a curated character list (`data/SC_list.txt`). A correct answer reasoned
  in the wrong language scores near zero, so language cannot be traded for task success.
- **`format_factor`** gates generation hygiene (a closed `</think>` block).

Laid out like NVIDIA's [`Gym/environments/`](https://github.com/NVIDIA-NeMo/Gym/tree/main/environments):
each folder is self-contained (verifier `app.py`, `config`, `prepare.py`, example
data, tests) and ships a runnable `data/example.jsonl`.

## Environments

| Environment | Task verifier | Language reward | Notes |
|---|---|---|---|
| `math_with_judge_lang` | math-verify on `\boxed{}` | ✓ | |
| `code_gen_lang` | LiveCodeBench unit tests (`lcb_integration/`) | ✓ | |
| `stem_mcqa` | exact-match MCQA | ✓ | |
| `workplace_assistant_lang` | WorkBench multi-turn tool-calling | ✓ | agentic; language judged per turn |
| `instruction_following_lang` | dual registry: NVIDIA `verifiable_instructions` (en) + vendored `cantonese_rlvr` (yue/zh-hk) | ✓ | 52 CJK-aware constraint verifiers |
| `structured_outputs_lang` | 5-format validation (json/yaml/toml/xml/csv) via `cantonese_rlvr.so_verify` | ✓ | broader than the base env's json/yaml/xml |

See each environment's `README.md` for its input schema, reward decomposition, and an
example rollout.

## Layout

```
environments/<name>/
  README.md            # overview, input schema, reward, example rollout
  app.py               # the resources-server verifier (task × language × format)
  config(s)/…          # server config (target language, sc_list, thresholds)
  prepare.py           # build train/validation data from a source blend
  data/
    example.jsonl      # a few runnable rows (ships)
    SC_list.txt        # curated Simplified-Chinese penalty list
  tests/               # unit tests (where present)
  cantonese_rlvr/      # (IF & SO only) vendored CJK-aware verifiers
nemo_gym/
  language_penalty/    # shared language-reward module the envs import
```

## Install & run

These target **NeMo-Gym `0.3.0rc0`** (the resources-server layout).

1. **Shared module.** Every environment imports the language reward as
   `nemo_gym.language_penalty`. Copy `nemo_gym/language_penalty/` into your installed
   `nemo_gym` package (i.e. make it importable as `nemo_gym.language_penalty`).
2. **Environments.** Copy each `environments/<name>/` into your NeMo-Gym
   `resources_servers/` (newer NeMo-Gym: `environments/`), then build the per-server
   venv the usual way.
3. **Run** the resources server / collect rollouts against the shipped
   `data/example.jsonl` exactly as for any NeMo-Gym environment (see each env README).

## Data

Only a small runnable `data/example.jsonl` ships per environment. To build the full
train/validation split, run the env's `prepare.py` against a **source blend you
supply** — the `SRC` default points at a placeholder `${BLEND_SRC}` (the NVIDIA
nano_v3 SFT blends we used are **not** redistributed here). `prepare.py` only reshapes
rows into Gym format and splits by language; it does not filter on the source
`pass_rate` (an unprofiled placeholder). See each env README.

## License & notices

- **License:** MIT — see [`LICENSE`](LICENSE) (fill in the copyright holder before publishing).
- **Vendored components & data provenance:** [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
  All vendored deps (cantofilter, jieba, toml) are MIT-licensed.
