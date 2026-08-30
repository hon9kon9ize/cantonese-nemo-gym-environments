#!/usr/bin/env python3
"""Prepare the multilingual structured_outputs_lang training data.

The structured-output blend already ships each row in Gym shape: the per-language
prompt (schema + Cantonese/Chinese news article) is prebuilt in
responses_create_params.input, and the verifier fields (schema_str + schema_type +
language) are top-level. So this script only:
  - retargets agent_ref -> structured_outputs_lang_simple_agent,
  - splits into train / validation / validation_smoke, stratified by language,
  - sanity-checks that every schema_str parses as JSON (else the verifier can't run).

IMPORTANT: it does NOT filter on `pass_rate`. That field in the source file is a
PLACEHOLDER -- the dataset has NOT been profiled with our model yet. Filtering on it
would drop ~81% of rows on noise. Re-profile with our model first, then add a filter.

Reports the schema_type (json/yaml/xml/toml/csv) x language distribution -- the reward
covers all five (json/yaml/toml full JSON-Schema validation; xml/csv required
keys/columns), unlike the base env which only handles json/yaml/xml.

Pure stdlib. Run on the login node:  python3 prepare_structured_outputs_lang_data.py
"""
import argparse
import collections
import copy
import json
import random
from pathlib import Path

SRC = "${BLEND_SRC}/data/instructions_following/blended/nano_v3_sft_profiled_structured_outputs.jsonl"
OUT_DIR = "data"
ENV_DIR = "."
AGENT_NAME = "structured_outputs_lang_simple_agent"


def transform(row: dict) -> dict:
    out = copy.deepcopy(row)
    out["agent_ref"] = {"type": "responses_api_agents", "name": AGENT_NAME}
    for k in ("pass_rate", "pass_rate_total", "pass_rate_passed"):
        out.pop(k, None)
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
    args = ap.parse_args()

    by_lang = collections.defaultdict(list)
    fmt_by_lang = collections.defaultdict(collections.Counter)
    bad_schema = 0
    total = 0
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            try:
                json.loads(r.get("schema_str", ""))
            except Exception:
                bad_schema += 1
                continue  # unparseable schema -> verifier can never grade it, drop
            lang = (r.get("language") or "en").lower()
            by_lang[lang].append(transform(r))
            fmt_by_lang[lang][r.get("schema_type")] += 1

    kept = sum(len(v) for v in by_lang.values())
    print(f"loaded {total} rows (NO pass_rate filter -- it is a placeholder); "
          f"dropped {bad_schema} with unparseable schema_str -> kept {kept}")
    print("schema_type x language (the reward covers all five):")
    for lang in sorted(fmt_by_lang):
        print(f"  {lang:6s} {dict(fmt_by_lang[lang])}")

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
    write_jsonl(Path(ENV_DIR) / "data" / "example.jsonl", (smoke + val)[:5])

    print(f"\nsource language counts: {dict((k, len(v)) for k, v in by_lang.items())}")
    print(f"train={len(train)}  validation={len(val)}  validation_smoke={len(smoke)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
