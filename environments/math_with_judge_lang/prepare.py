#!/usr/bin/env python3
"""Prepare the multilingual math_with_judge_lang training data.

Takes the translated skywork blend (yue / zh-hk / en) and:
  - rewrites each row's user turn into the GSM8K-blend prompt shape (no system
    role, instruction in the user turn) that the GSM8K-GRPOed 30B was RL-trained
    on -- crucially, zh-hk says "Written Traditional Chinese 繁體中文書面語
    (NOT colloquial Cantonese)", which fixes the observed failure where written-
    Chinese rows were answered in Cantonese;
  - retargets agent_ref -> math_with_judge_lang_simple_agent;
  - keeps question / expected_answer / language for the verifier;
  - splits into train / validation / validation_smoke, stratified by language.

Pure stdlib -- run on the login node:  python3 prepare_math_lang_data.py
"""
import argparse
import collections
import copy
import json
import random
import re
from pathlib import Path

# CJK Unified Ideographs (incl. Ext-A). Used to drop en-labelled rows whose
# problem body is actually Chinese: the dapo17k/skywork blends have ~169 en rows
# that are a fully-Chinese problem behind an English instruction wrapper (a failed
# translation), which would train the model to "reason in English" on Chinese
# input. The contamination is cleanly bimodal -- genuine en rows have 0 CJK chars
# (one stray OCR glyph aside), contaminated ones have >=6 -- so a small threshold
# separates them without touching real English data. yue/zh-hk are exempt (their
# problems are legitimately Chinese).
_CJK = re.compile(r"[㐀-鿿]")


def cjk_count(text: str) -> int:
    return len(_CJK.findall(text or ""))

# All translated datasets that route to math_with_judge (both carry question /
# expected_answer / language). Add more math_with_judge files here as they arrive.
SRCS = [
    "${BLEND_SRC}/data/translated/blended/nano_v3_sft_profiled_skywork_no_omni.jsonl",
    "${BLEND_SRC}/data/translated/blended/nano_v3_sft_profiled_dapo17k.jsonl",
]
OUT_DIR = "data"
ENV_DATA_DIR = "./data"
AGENT_NAME = "math_with_judge_lang_simple_agent"

# GSM8K-blend prompt shape — the regime the 30B GSM8K-GRPOed model was RL-trained
# on: NO system role, per-language instruction embedded in the USER turn. With the
# earlier system-prompt design the model never triggered its tagged-reasoning
# behavior (0/593 rollouts with <answer> or a closed </think>); mirroring the
# infinite-rl-nemo gsm8k_blend wording recovers it. Deviations from gsm8k_blend:
# final answer in \boxed{} (what math-verify extracts) instead of <answer> tags,
# and the strengthened zh-hk 書面語 wording (forbid Cantonese colloquial chars +
# Simplified). No literal <think> tokens in prompts (tag-scaffolding lesson).
# {question} is substituted via .replace() — questions are full of LaTeX braces,
# so str.format() would blow up.
USER_TEMPLATES = {
    "yue": (
        "Solve this math problem.\n\n**Problem:**\n{question}\n\n"
        "Reason step by step in Cantonese (粵語口語，用「係」「嘅」「唔」等字，"
        "唔好用書面語或者簡體字), writing naturally with proper spaces and "
        "punctuation. Then give the final answer inside \\boxed{}.\n"
    ),
    "zh-hk": (
        "Solve this math problem.\n\n**Problem:**\n{question}\n\n"
        "Reason step by step in Written Traditional Chinese (繁體中文書面語——"
        "切勿使用廣東話口語字詞如「係」「嘅」「唔」，切勿使用簡體字), "
        "writing naturally with proper spaces and punctuation. "
        "Then give the final answer inside \\boxed{}.\n"
    ),
    "en": (
        "Solve this math problem.\n\n**Problem:**\n{question}\n\n"
        "Reason step by step in English, writing naturally with proper spaces "
        "and punctuation. Then give the final answer inside \\boxed{}.\n"
    ),
}


def transform(row: dict) -> dict:
    lang = (row.get("language") or "en").lower()
    tmpl = USER_TEMPLATES.get(lang, USER_TEMPLATES["en"])
    out = copy.deepcopy(row)
    out["agent_ref"] = {"type": "responses_api_agents", "name": AGENT_NAME}
    rcp = out.setdefault("responses_create_params", {})
    # Rebuild the single user turn from the bare `question` field (the source
    # rows' user content is just a short per-language preamble + the question).
    rcp["input"] = [
        {"role": "user", "content": tmpl.replace("{question}", row["question"])}
    ]
    return out


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-per-lang", type=int, default=20)
    ap.add_argument("--smoke-per-lang", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    # Drop ONLY the utterly-hopeless rows (NVIDIA pass_rate == 0: their strong profiler
    # never solved it, so ours won't either -> reward 0, no signal, wasted generation).
    # Everything NVIDIA solved at least once is kept, INCLUDING the hard-but-doable band
    # that GRPO learns most from -- our own 30B full-set profiling downstream is the real
    # curriculum filter, not this coarse pre-prune. default 0.125 = smallest positive
    # (k/8) bucket, i.e. keep pass_rate > 0.
    ap.add_argument("--min-pass-rate", type=float, default=0.125)
    # Drop en-labelled rows whose question carries more than this many CJK chars
    # (mislabelled Chinese problems). The 1..2 band is only stray OCR glyphs in
    # real English; contamination starts at >=6, so 2 is a safe cutoff.
    ap.add_argument("--en-max-cjk", type=int, default=2)
    args = ap.parse_args()

    by_lang = collections.defaultdict(list)
    kept = dropped = dropped_en_cjk = 0
    for src in SRCS:
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if (r.get("pass_rate") if r.get("pass_rate") is not None else 1.0) < args.min_pass_rate:
                    dropped += 1
                    continue
                lang = (r.get("language") or "en").lower()
                # Filter en rows that are actually Chinese (failed translation).
                if lang == "en" and cjk_count(r.get("question", "")) > args.en_max_cjk:
                    dropped_en_cjk += 1
                    continue
                kept += 1
                by_lang[lang].append(transform(r))
    print(f"pass_rate filter >= {args.min_pass_rate}: kept {kept}, dropped {dropped}")
    print(f"en Chinese-contamination filter (>{args.en_max_cjk} CJK chars): dropped {dropped_en_cjk} en rows")

    rng = random.Random(args.seed)
    train, val, smoke = [], [], []
    for lang, rows in sorted(by_lang.items()):
        rng.shuffle(rows)
        val_rows = rows[: args.val_per_lang]
        val += val_rows
        smoke += val_rows[: args.smoke_per_lang]
        train += rows[args.val_per_lang :]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(smoke)

    out = Path(OUT_DIR)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "validation.jsonl", val)
    write_jsonl(out / "validation_smoke.jsonl", smoke)
    # 5-row example.jsonl in the env dir (NeMo-Gym convention).
    write_jsonl(Path(ENV_DATA_DIR) / "example.jsonl", (smoke + val)[:5])

    counts = {k: len(v) for k, v in by_lang.items()}
    print(f"source language counts: {counts}")
    print(f"train={len(train)}  validation={len(val)}  validation_smoke={len(smoke)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
