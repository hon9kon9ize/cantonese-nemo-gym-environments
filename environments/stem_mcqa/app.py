# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# stem_mcqa = the base `mcqa` env (multiple-choice grading via template_metadata
# output_regex) forked with the SAME multiplicative language penalty + format gate
# used by `math_with_judge_lang` (ported from infinite-rl-nemo). The MCQA answer
# ("Selected Option -> X") is graded exactly as upstream; on top of that
#     reward = mcqa_reward * language_multiplier * format_factor
# enforces the per-row reasoning language (yue / zh-hk 書面語 / en, forbidding
# Simplified Chinese) and requires a closed </think> block. The answer boilerplate
# lives AFTER </think>, so the language check (which reads the reasoning before
# </think>) is not polluted by it.
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.language_penalty import language_multiplier, load_sc_set
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


# --------------------------------------------------------------------------- #
# Multiplicative language penalty + format gate (identical design to
# math_with_judge_lang; the reward module itself is shared via nemo_gym).
# --------------------------------------------------------------------------- #
class LanguagePenaltyConfig(BaseModel):
    enabled: bool = True
    sc_list_path: str  # curated simplified-only char list (SC_list.txt)
    think_tag: str = "think"
    reasoning_template: bool = False  # True if the chat template pre-opens <think>
    wrong_lang_floor: float = 0.0
    min_target_script_ratio: float = 0.15  # >= this share of alphabetic chars in the
    # target script counts as target-language (RELAXED for heavy code-mixing).
    sc_strength: float = 1.0
    sc_tolerance: float = 0.05
    sc_full_ratio: float = 0.20
    sc_floor: float = 0.0
    sc_min_hits: int = 2
    rollout_log_dir: Optional[str] = None


class FormatGateConfig(BaseModel):
    enabled: bool = True
    floor: float = 0.2
    think_tag: str = "think"


class StemMCQAResourcesServerConfig(BaseResourcesServerConfig):
    grading_mode: Optional[
        Literal[
            "strict_single_letter_boxed",
            "lenient_boxed",
            "lenient_answer_colon",
            "lenient_answer_colon_md",
        ]
    ] = None
    language_penalty: Optional[LanguagePenaltyConfig] = None
    format_gate: Optional[FormatGateConfig] = None


class StemMCQARunRequest(BaseRunRequest):
    uuid: Optional[str] = None
    options: Optional[list[dict[str, Optional[str]]]] = None
    expected_answer: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    grading_mode: Literal[
        "strict_single_letter_boxed",
        "lenient_boxed",
        "lenient_answer_colon",
    ] = "strict_single_letter_boxed"
    template_metadata: Optional[dict[str, Any]] = None
    # Per-row target language (yue / zh-hk / en) for the language penalty.
    language: Optional[str] = None


class StemMCQAVerifyRequest(StemMCQARunRequest, BaseVerifyRequest):
    pass


class StemMCQAVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    # Reward breakdown (reward = mcqa_reward * lang_multiplier * format_factor).
    mcqa_reward: float = 0.0
    lang_multiplier: float = 1.0
    lang_factor: float = 1.0
    sc_factor: float = 1.0
    format_factor: float = 1.0


# --------------------------------------------------------------------------- #
# MCQA answer parsing (verbatim from the base `mcqa` env).
# --------------------------------------------------------------------------- #
CHOICE_LETTER_PATTERN = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")
STRICT_BOXED_PATTERN = re.compile(r"\\boxed\{\s*[^A-Za-z]*([A-Z])[^A-Za-z]*\s*\}")
ANSWER_COLON_PATTERN = re.compile(r"(?i)answer\s*:\s*(.+)")
ANSWER_COLON_MD_PATTERN = re.compile(r"(?i)[*_]{0,2}Answer[*_]{0,2}\s*:[*_\s]{0,2}\s*([A-Z])(?![a-zA-Z0-9])")
BOXED_CONTENT_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)
LATEX_TEXT_WRAP_PATTERN = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)


def _get_allowed_letters_from_options(options: Optional[list[dict[str, str]]]) -> set[str]:
    letters: set[str] = set()
    if options:
        for entry in options:
            for k, v in entry.items():
                if isinstance(k, str) and len(k) == 1 and k.isalpha() and v is not None:
                    letters.add(k.upper())
    return letters


def _strip_latex_wrappers(s: str) -> str:
    while True:
        m = LATEX_TEXT_WRAP_PATTERN.fullmatch(s)
        if not m:
            break
        s = m.group(1)
    return s


def _normalize_for_match(s: str) -> str:
    return " ".join(s.lower().split())


def _parse_answer_letter_strict_boxed(text: str, allowed_letters: set[str]) -> Optional[str]:
    m = STRICT_BOXED_PATTERN.search(text)
    if not m:
        return None
    letter = m.group(1).upper()
    return letter if letter in allowed_letters else None


def _match_option_text(text: str, options: list[dict[str, str]], allowed_letters: set[str]) -> Optional[str]:
    boxed = BOXED_CONTENT_PATTERN.search(text)
    if not boxed:
        return None
    inner = boxed.group(1)
    normalized_candidates = [_normalize_for_match(t) for t in (inner, _strip_latex_wrappers(inner))]
    normalized_options: list[tuple[str, str]] = []
    for entry in options or []:
        for k, v in entry.items():
            if v is not None and isinstance(k, str) and len(k) == 1 and k.upper() in allowed_letters:
                normalized_options.append((k.upper(), _normalize_for_match(v)))
    matched_letters: set[str] = set()
    for cand in normalized_candidates:
        for letter, opt_norm in normalized_options:
            if opt_norm and opt_norm in cand:
                matched_letters.add(letter)
    return next(iter(matched_letters)) if len(matched_letters) == 1 else None


def _parse_answer_with_custom_regex(
    text: str, regex_pattern: str, allowed_letters: set[str], options: Optional[list[dict[str, str]]]
) -> Optional[str]:
    try:
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        if not matches:
            return None
        captured = matches[-1].strip().upper()
        if len(captured) == 1 and captured.isalpha():
            if allowed_letters and captured in allowed_letters:
                return captured
            elif not allowed_letters:
                return captured
            else:
                return captured
        normalized_captured = _normalize_for_match(captured)
        for entry in options or []:
            for k, v in entry.items():
                if v is not None and k.upper() in allowed_letters and _normalize_for_match(v) == normalized_captured:
                    return k.upper()
        return None
    except re.error:
        return None


def _grade_mcqa(body: StemMCQAVerifyRequest, text: str, grading_mode: str) -> tuple[Optional[str], str]:
    """Return (predicted_letter, gold_letter) using the base mcqa logic."""
    options, expected_answer = body.options, body.expected_answer
    gold = (expected_answer or "").strip().upper()
    allowed_letters = _get_allowed_letters_from_options(options)

    pred: Optional[str] = None
    if body.template_metadata and "output_regex" in body.template_metadata:
        pred = _parse_answer_with_custom_regex(text, body.template_metadata["output_regex"], allowed_letters, options)

    if pred is None:
        if grading_mode == "strict_single_letter_boxed":
            pred = _parse_answer_letter_strict_boxed(text, allowed_letters)
        elif grading_mode == "lenient_boxed":
            pred = _parse_answer_letter_strict_boxed(text, allowed_letters)
            if pred is None:
                pred = _match_option_text(text, options, allowed_letters)
        elif grading_mode == "lenient_answer_colon":
            m = ANSWER_COLON_PATTERN.search(text)
            if m:
                candidate = _strip_latex_wrappers(m.group(1)).strip()
                if len(candidate) == 1 and candidate.isalpha() and candidate.upper() in allowed_letters:
                    pred = candidate.upper()
                if pred is None:
                    cand_norm = _normalize_for_match(candidate)
                    for entry in options or []:
                        for k, v in entry.items():
                            if k.upper() in allowed_letters and _normalize_for_match(v) == cand_norm:
                                pred = k.upper()
                                break
                        if pred is not None:
                            break
        elif grading_mode == "lenient_answer_colon_md":
            md_match = ANSWER_COLON_MD_PATTERN.search(text)
            if md_match:
                letter_up = md_match.group(1).strip().upper()
                if letter_up in allowed_letters:
                    pred = letter_up
    return pred, gold


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
class StemMCQAResourcesServer(SimpleResourcesServer):
    config: StemMCQAResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        self._sc_set = frozenset()
        lp = self.config.language_penalty
        if lp is not None and lp.enabled:
            self._sc_set = load_sc_set(lp.sc_list_path)
            logging.getLogger("nemo_gym").info(
                "Language penalty ON: %d simplified chars, wrong_lang_floor=%s",
                len(self._sc_set),
                lp.wrong_lang_floor,
            )

        self._rollout_log_path: Optional[str] = None
        self._rollout_lock = threading.Lock()
        if lp is not None and lp.enabled and lp.rollout_log_dir:
            os.makedirs(lp.rollout_log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._rollout_log_path = os.path.join(lp.rollout_log_dir, f"stem_mcqa_rollouts_{ts}_{os.getpid()}.jsonl")
            logging.getLogger("nemo_gym").info("Saving rollouts to %s", self._rollout_log_path)

    def _log_rollout(self, record: Dict[str, Any]) -> None:
        if not self._rollout_log_path:
            return
        try:
            line = json.dumps(record, ensure_ascii=False)
            with self._rollout_lock, open(self._rollout_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:  # logging must never break verification
            logging.getLogger("nemo_gym").exception("Failed to write rollout record")

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    def compute_metrics(self, tasks):
        return compute_pass_majority_metrics(
            tasks,
            score_fn=lambda r: {"accuracy": r["reward"]},
            answer_key="extracted_answer",
        )[0]

    def get_key_metrics(self, agent_metrics):
        key = {}
        if "mean/reward" in agent_metrics:
            key["mean/reward"] = agent_metrics["mean/reward"]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy", "no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", score_names=["accuracy"]))
        if "pass@1/accuracy" in agent_metrics:
            key["pass@1/accuracy"] = agent_metrics["pass@1/accuracy"]
        return key

    async def verify(self, body: StemMCQAVerifyRequest) -> StemMCQAVerifyResponse:
        text = body.response.output_text.strip()
        grading_mode = self.config.grading_mode or body.grading_mode
        pred, gold = _grade_mcqa(body, text, grading_mode) if text else (None, (body.expected_answer or "").strip().upper())

        mcqa_reward = 1.0 if (pred is not None and gold and pred == gold) else 0.0
        reward = mcqa_reward

        # Multiplicative language penalty over the reasoning (before </think>).
        lang_mult = lang_factor = sc_factor = 1.0
        lp = self.config.language_penalty
        if lp is not None and lp.enabled:
            lang_mult, lang_factor, sc_factor = language_multiplier(
                text,
                body.language,
                self._sc_set,
                think_tag=lp.think_tag,
                reasoning_template=lp.reasoning_template,
                wrong_lang_floor=lp.wrong_lang_floor,
                min_target_script_ratio=lp.min_target_script_ratio,
                sc_strength=lp.sc_strength,
                sc_tolerance=lp.sc_tolerance,
                sc_full_ratio=lp.sc_full_ratio,
                sc_floor=lp.sc_floor,
                sc_min_hits=lp.sc_min_hits,
            )
            reward = mcqa_reward * lang_mult

        # Format gate: no closed </think> -> keep only `floor` of the reward.
        fg = self.config.format_gate
        reasoning_template = bool(lp is not None and lp.reasoning_template)
        think_open = reasoning_template or (f"<{fg.think_tag if fg else 'think'}>" in text)
        think_closed = "</think>" in text
        format_factor = 1.0
        if fg is not None and fg.enabled:
            format_factor = 1.0 if f"</{fg.think_tag}>" in text else fg.floor
            reward = reward * format_factor

        self._log_rollout(
            {
                "language": body.language,
                "expected_answer": gold,
                "extracted_answer": pred,
                "mcqa_reward": mcqa_reward,
                "lang_multiplier": lang_mult,
                "lang_factor": lang_factor,
                "sc_factor": sc_factor,
                "format_factor": format_factor,
                "reward": reward,
                "think_open": think_open,
                "think_closed": think_closed,
                "output": text,
            }
        )

        return StemMCQAVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            expected_answer=gold,
            extracted_answer=pred,
            mcqa_reward=mcqa_reward,
            lang_multiplier=lang_mult,
            lang_factor=lang_factor,
            sc_factor=sc_factor,
            format_factor=format_factor,
        )


if __name__ == "__main__":
    StemMCQAResourcesServer.run_webserver()
