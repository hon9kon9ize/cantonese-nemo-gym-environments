# Third-party notices

These environments build on and vendor third-party components. Their licenses apply
to those files.

## Framework (not included — install separately)
- **NVIDIA NeMo-Gym** (Apache-2.0). These are custom *resources-server environments*
  that run inside NeMo-Gym (target: `0.3.0rc0`). Four of the six fork a stock
  NeMo-Gym environment (`math_with_judge`, `code_gen`, `mcqa`, `workplace_assistant`);
  `instruction_following_lang` / `structured_outputs_lang` fork the instruction /
  structured-output envs and swap in Cantonese-aware verifiers. NVIDIA's
  `verifiable_instructions` package (used for the English constraint path) is part of
  NeMo-Gym and is not redistributed here.

## Vendored
- **cantofilter** (MIT) — Cantonese/Mandarin sentence classifier, vendored at
  `nemo_gym/language_penalty/_vendor/cantofilter/`. Used to judge whether `yue`
  reasoning is genuinely colloquial Cantonese. Upstream:
  https://github.com/CanCLID/canto-filter (MIT License).
- **jieba** (MIT) and **toml** (MIT) — vendored under
  `environments/{instruction_following_lang,structured_outputs_lang}/cantonese_rlvr/_vendor/`
  for Chinese word segmentation and TOML parsing in the verifiers.
- **LiveCodeBench** integration — `environments/code_gen_lang/lcb_integration/` (from
  the upstream `code_gen` environment); see LiveCodeBench for its terms.

## Data
- **No training data is redistributed.** `prepare.py` reads NVIDIA nano_v3 SFT blends
  that you supply (`SRC` / `${BLEND_SRC}`). Only each env's small `data/example.jsonl`
  and the curated `data/SC_list.txt` are original to this project and included here.
- Verify you have redistribution rights before publishing any derived data.
