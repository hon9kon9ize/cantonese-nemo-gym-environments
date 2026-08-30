"""Pure verifiable-reward helper for the NeMo-RL environment integration.

Kept framework-agnostic so it can be unit-tested outside NeMo-RL. The environment
calls ``verifiable_reward`` for the Cantonese RLVR task types.

  verifiable_reward(task_type, response, meta) -> (reward, correct, format_valid)

task types:
  "instruction_following" : meta carries instruction_id_list (list[str]) and
                            instruction_kwargs (list[dict]); reward = fraction of
                            constraints satisfied (non-deterministic verifiers,
                            which return None, are excluded from the denominator).
  "structured_output"     : meta carries `verification` (dict or json str with
                            problem_type/target_format/schema/expected_data);
                            reward = 1.0 if the output passes so_verify else 0.0.

A leading <think>...</think> block is stripped so the *answer* is scored.
"""
from __future__ import annotations
import json
import re

from . import registry as R
from . import so_verify

CANTO_TASK_TYPES = ("instruction_following", "structured_output")
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def _if_reward(response: str, ids, kwargs) -> tuple[float, int]:
    checked = passed = 0
    for iid, kw in zip(ids or [], kwargs or []):
        entry = R.REG.get(iid)
        if entry is None:
            continue
        try:
            v = entry["verify"](response, kw)
        except Exception:
            v = False
        if v is None:            # e.g. rhyme without pycantonese: not scored
            continue
        checked += 1
        passed += 1 if v else 0
    return (passed / checked if checked else 0.0), checked


def _so_reward(response: str, verification: dict) -> float:
    try:
        ok = so_verify.verify(
            verification["problem_type"], verification["target_format"],
            response, verification["schema"], verification["expected_data"],
        )
    except Exception:
        ok = False
    return 1.0 if ok else 0.0


def verifiable_reward(task_type: str, response: str, meta: dict) -> tuple[float, float, bool]:
    resp = strip_think(response)
    if not resp:
        return 0.0, 0.0, False
    if task_type == "instruction_following":
        r, checked = _if_reward(resp, meta.get("instruction_id_list"), meta.get("instruction_kwargs"))
        correct = 1.0 if (checked and r >= 1.0) else 0.0
        return r, correct, True
    if task_type == "structured_output":
        v = meta.get("verification")
        if isinstance(v, str):
            v = json.loads(v)
        r = _so_reward(resp, v or {})
        return r, r, True
    return 0.0, 0.0, False
