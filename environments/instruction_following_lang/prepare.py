#!/usr/bin/env python3
"""Prepare the multilingual instruction_following_lang training data.

The IFEval-style blend already ships each row in Gym shape: the per-language
prompt is prebuilt in responses_create_params.input, and the verifier fields
(instruction_id_list + kwargs + language) are top-level. So this script only:
  - retargets agent_ref -> instruction_following_lang_simple_agent,
  - splits into train / validation / validation_smoke, stratified by language.

IMPORTANT: it does NOT filter on `pass_rate`. That field in the source file is a
PLACEHOLDER -- the dataset has NOT been profiled with our model yet (unlike the
code_gen / stem_mcqa blends, which carry a real profiler pass_rate). Filtering on
it would drop ~80% of rows on noise. Once we run a real reward-profile with our
own model, add a pass_rate filter here (see prepare_code_gen_lang_data.py).

Reports registry coverage per language (how many constraints the vendored
cantonese_rlvr.registry vs NVIDIA verifiable_instructions must grade) as a
sanity check, but keeps every row.

Pure stdlib. Run on the login node:  python3 prepare_instruction_following_lang_data.py
"""
import argparse
import collections
import copy
import json
import random
import sys
from pathlib import Path

SRC = "${BLEND_SRC}/data/instructions_following/blended/nano_v3_sft_profiled_instruction_following.jsonl"
OUT_DIR = "data"
ENV_DIR = "."
AGENT_NAME = "instruction_following_lang_simple_agent"


def transform(row: dict) -> dict:
    out = copy.deepcopy(row)
    out["agent_ref"] = {"type": "responses_api_agents", "name": AGENT_NAME}
    # pass_rate is a placeholder; drop it so nothing downstream mistakes it for a
    # profiled difficulty. (Re-profile with our model to add a real one.)
    for k in ("pass_rate", "pass_rate_total", "pass_rate_passed"):
        out.pop(k, None)
    return out


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def coverage_report(rows: list) -> None:
    """Print per-language registry coverage as a sanity check (keeps every row)."""
    try:
        sys.path.insert(0, ENV_DIR)
        from cantonese_rlvr import registry as R

        reg_ids = set(R.REG.keys())
    except Exception as e:
        print(f"(coverage check skipped: {e})")
        return
    zero = collections.Counter()
    tot = collections.Counter()
    nvidia_needed = collections.Counter()
    for d in rows:
        lang = d.get("language")
        tot[lang] += 1
        ids = d.get("instruction_id_list") or []
        known = [i for i in ids if i in reg_ids]
        if not known:
            zero[lang] += 1
        if any(i not in reg_ids for i in ids):
            nvidia_needed[lang] += 1
    print("registry coverage (vendored cantonese_rlvr.REG):")
    for lg in sorted(tot):
        print(
            f"  {lg:6s} rows={tot[lg]:5d}  zero-known-id={zero[lg]:3d}  "
            f"need-NVIDIA-for-some={nvidia_needed[lg]:3d}"
        )
    print("  -> yue/zh-hk should be fully covered by REG; en needs NVIDIA "
          "verifiable_instructions for the 9 English-only letter/case types.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--val-per-lang", type=int, default=20)
    ap.add_argument("--smoke-per-lang", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_lang = collections.defaultdict(list)
    all_rows = []
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            all_rows.append(r)
            lang = (r.get("language") or "en").lower()
            by_lang[lang].append(transform(r))

    print(f"loaded {len(all_rows)} rows (NO pass_rate filter -- it is a placeholder)")
    coverage_report(all_rows)

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
