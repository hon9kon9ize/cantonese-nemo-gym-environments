# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Light unit tests for structured_outputs_lang reward-path helpers.

Exercise the pieces directly (no server / HTTP). Import works because the entrypoint
dir (this env) is on sys.path, giving `cantonese_rlvr` and `app`.
"""
import json

import app

_SCHEMA = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "n"],
        "properties": {"title": {"type": "string"}, "n": {"type": "integer"}},
    }
)


class _Srv:
    """Bind the unbound _so_reward method without constructing the full server."""

    _so_reward = app.StructuredOutputsLangResourcesServer._so_reward


def test_extract_answer_after_think():
    txt = "諗緊點砌<think>思考</think>\n```json\n{\"title\": \"x\", \"n\": 1}\n```"
    assert "json" in app.StructuredOutputsLangResourcesServer._extract_answer(txt, True)


def test_extract_answer_unclosed_is_empty_under_template():
    assert app.StructuredOutputsLangResourcesServer._extract_answer("一直諗未收", True) == ""


def test_so_reward_valid_json():
    srv = _Srv()
    assert srv._so_reward(_SCHEMA, "json", '{"title": "x", "n": 1}') == 1.0


def test_so_reward_invalid_missing_required():
    srv = _Srv()
    assert srv._so_reward(_SCHEMA, "json", '{"title": "x"}') == 0.0  # n missing


def test_so_reward_wrong_type():
    srv = _Srv()
    assert srv._so_reward(_SCHEMA, "json", '{"title": "x", "n": "not-int"}') == 0.0


def test_so_reward_yaml_and_fence():
    srv = _Srv()
    assert srv._so_reward(_SCHEMA, "yaml", "```yaml\ntitle: x\nn: 1\n```") == 1.0


def test_so_reward_empty_answer():
    srv = _Srv()
    assert srv._so_reward(_SCHEMA, "json", "") == 0.0
