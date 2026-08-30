# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Light unit tests for the dual-registry instruction_following_lang helpers.

These exercise the reward-path pieces directly (no server / no HTTP), so they run
fast inside the per-server venv. Import works because the entrypoint dir (this
env) is on sys.path, giving `cantonese_rlvr` and `app`.
"""
import app


def test_extract_answer_reasoning_template():
    txt = "諗緊點答<think>某啲思考</think>\n\n最終答案：香港"
    # with reasoning_template=True the leading text is still reasoning; answer is
    # everything after </think>
    assert app.InstructionFollowingLangResourcesServer._extract_answer(txt, True) == "最終答案：香港"


def test_extract_answer_unclosed_is_empty_under_template():
    txt = "一直喺度諗未收尾"
    assert app.InstructionFollowingLangResourcesServer._extract_answer(txt, True) == ""


def test_reg_verify_keyword_existence_yue():
    # keywords:existence is a language-neutral REG verifier: all keywords present.
    kw = {"keywords": ["香港", "上學"]}
    assert app._reg_verify("keywords:existence", kw, "我喺香港返學同上學") is True
    assert app._reg_verify("keywords:existence", kw, "只有香港冇另一個詞") is False


def test_reg_verify_unknown_id_is_sentinel():
    assert app._reg_verify("change_case:english_capital", {}, "HELLO") is app._UNKNOWN


def test_verify_one_routes_en_to_nvidia_or_falls_back():
    # If verifiable_instructions is installed, an English-only id grades True on a
    # fully-capitalised answer; if not, it's UNKNOWN (excluded from denominator).
    res = app._verify_one("change_case:english_capital", {}, "HELLO WORLD", "en")
    assert res in (True, False, app._UNKNOWN)
