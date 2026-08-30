#!/usr/bin/env python3
"""Prepare the multilingual workplace_assistant_lang training data.

The workbench blend already ships each row in Gym shape: the per-language prompt
(system + user) is prebuilt in responses_create_params.input (with the 27 tool
schemas under responses_create_params.tools), and the grading fields (ground_truth
= expected tool calls, id, category, environment_name, language) are top-level. So
this script only:
  - filters by profiled pass_rate (drop the model-can't-touch rows),
  - drops en-labelled rows whose prompt is actually Chinese (failed translation),
  - retargets agent_ref -> workplace_assistant_lang_simple_agent,
  - splits into train / validation / validation_smoke, stratified by language.

Pure stdlib. Run on the login node:  python3 prepare_workplace_assistant_lang_data.py
"""
import argparse
import collections
import copy
import json
import random
import re
from pathlib import Path

SRC = "${BLEND_SRC}/data/translated/blended/nano_v3_sft_profiled_workbench.jsonl"
OUT_DIR = "data"
ENV_DATA_DIR = "./data"
AGENT_NAME = "workplace_assistant_lang_simple_agent"

# CJK Unified Ideographs (incl. Ext-A); used to drop en rows that are really
# Chinese (same filter as the other *_lang preps). yue/zh-hk are exempt.
_CJK = re.compile(r"[㐀-鿿]")


def cjk_count(text: str) -> int:
    return len(_CJK.findall(text or ""))


def prompt_text(row: dict) -> str:
    """Concatenate the message content(s) of the prebuilt prompt (system+user)."""
    parts = []
    for m in row.get("responses_create_params", {}).get("input", []) or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def transform(row: dict) -> dict:
    out = copy.deepcopy(row)
    out["agent_ref"] = {"type": "responses_api_agents", "name": AGENT_NAME}
    return out


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--val-per-lang", type=int, default=20)
    ap.add_argument("--smoke-per-lang", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    # 0.125 = smallest positive (k/8) bucket -> drop only NVIDIA pass_rate==0 (hopeless),
    # keep the hard-but-doable band for GRPO; our own 30B profiling is the real filter.
    ap.add_argument("--min-pass-rate", type=float, default=0.125)
    ap.add_argument("--en-max-cjk", type=int, default=2)
    args = ap.parse_args()

    by_lang = collections.defaultdict(list)
    kept = dropped = dropped_en_cjk = 0
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("pass_rate") if r.get("pass_rate") is not None else 1.0) < args.min_pass_rate:
                dropped += 1
                continue
            lang = (r.get("language") or "en").lower()
            if lang == "en" and cjk_count(prompt_text(r)) > args.en_max_cjk:
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
    write_jsonl(Path(ENV_DATA_DIR) / "example.jsonl", (smoke + val)[:5])

    print(f"source language counts: {dict((k, len(v)) for k, v in by_lang.items())}")
    print(f"train={len(train)}  validation={len(val)}  validation_smoke={len(smoke)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
