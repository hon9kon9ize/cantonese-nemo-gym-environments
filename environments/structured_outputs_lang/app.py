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
# structured_outputs_lang = the base `structured_outputs` env (emit JSON/YAML/XML/
# TOML/CSV conforming to a given JSON Schema) forked with the SAME multiplicative
# language penalty + format gate used by code_gen_lang / instruction_following_lang
# / math_with_judge_lang / stem_mcqa:
#
#     reward = so_reward * language_multiplier * format_factor
#
# where so_reward = 1.0 iff the ANSWER (the text after </think>) parses in the
# target format AND satisfies the schema, language_multiplier enforces the per-row
# reasoning language (yue / zh-hk / en) and forbids Simplified Chinese, and
# format_factor requires a closed </think>.
#
# TWO DIFFERENCES vs the base env:
#  1. FIVE FORMATS. The base env only parses json/yaml/xml; this data is 27% toml +
#     csv, which the base env always scores 0. We reuse the VENDORED
#     `cantonese_rlvr.so_verify` (the exact reward the infinite-rl-nemo training env
#     uses) which handles all five: json/yaml/toml -> full JSON-Schema validation;
#     xml/csv -> well-formed + required keys/columns present. This also gives reward
#     PARITY with our existing pipeline. (The nano_v3 schemas are self-strict --
#     additionalProperties:false + explicit `required` -- so plain jsonschema.validate
#     enforces structure without the base env's strictify override.)
#  2. schema_only_generation. Our rows carry NO gold `expected_data`, so grading is
#     purely STRUCTURAL (right fields + types in the right format), not content
#     grounding. so_verify's title-grounding is a no-op here (no expected title).
#
# NOTE on <think>: the base env parses the raw output; with reasoning inline
# (uses_reasoning_parser: false) that raw output starts with the CoT and would never
# parse. We extract the answer AFTER </think> first (so_verify then tolerates a
# markdown code fence around it), and use the shared multiplicative `format_gate`
# for the closed-</think> check. The language check reads only the reasoning BEFORE
# </think>, so the JSON/English keys in the answer never pollute it.

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.language_penalty import language_multiplier, load_sc_set

# Vendored 5-format structured-output verifier (json/yaml/toml/xml/csv). Same reward
# the infinite-rl-nemo training env uses (cantonese_rlvr.nemo_reward._so_reward).
from cantonese_rlvr import so_verify as _so_verify

_SCHEMA_ONLY = "schema_only_generation"  # our rows have no gold expected_data


# --------------------------------------------------------------------------- #
# Multiplicative language penalty + format gate (identical design to the other
# _lang envs; the reward module is shared via nemo_gym).
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


# ----------------------------
# Config
# ----------------------------
class StructuredOutputsLangResourcesServerConfig(BaseResourcesServerConfig):
    language_penalty: Optional[LanguagePenaltyConfig] = None
    format_gate: Optional[FormatGateConfig] = None


# ----------------------------
# Schemas
# ----------------------------
class StructuredOutputsLangRunRequest(BaseRunRequest):
    # string representation of the JSON Schema (a JSON dict) + target format.
    schema_str: str = ""
    schema_type: str = "json"  # json | yaml | xml | toml | csv
    # Per-row target reasoning language (yue / zh-hk / en) for the language penalty.
    language: Optional[str] = None


class StructuredOutputsLangVerifyRequest(StructuredOutputsLangRunRequest, BaseVerifyRequest):
    pass


class StructuredOutputsLangVerifyResponse(BaseVerifyResponse):
    schema_str: str = ""
    schema_type: str = "json"
    extracted_answer: Optional[str] = None
    # Reward breakdown (reward = so_reward * lang_multiplier * format_factor).
    so_reward: float = 0.0  # 1.0 iff the answer parses + conforms to the schema
    lang_multiplier: float = 1.0
    lang_factor: float = 1.0
    sc_factor: float = 1.0
    format_factor: float = 1.0


# ----------------------------
# Server
# ----------------------------
class StructuredOutputsLangResourcesServer(SimpleResourcesServer):
    config: StructuredOutputsLangResourcesServerConfig

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
            self._rollout_log_path = os.path.join(
                lp.rollout_log_dir, f"structured_outputs_lang_rollouts_{ts}_{os.getpid()}.jsonl"
            )
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

    @staticmethod
    def _extract_answer(text: str, reasoning_template: bool) -> str:
        """The answer is everything AFTER the closed </think>. With a reasoning
        template the prompt pre-opens <think>, so an unclosed rollout is pure
        (runaway) reasoning -> empty answer, and the format gate floors it.
        so_verify.parse() tolerates a markdown code fence around the answer."""
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        if reasoning_template:
            return ""
        return text.strip()

    def _so_reward(self, schema_str: str, schema_type: str, answer: str) -> float:
        if not answer or not schema_str:
            return 0.0
        try:
            schema = json.loads(schema_str)
        except Exception:
            return 0.0
        try:
            ok = _so_verify.verify(_SCHEMA_ONLY, (schema_type or "").lower(), answer, schema, {})
        except Exception:  # a malformed schema/output must score 0, never crash
            ok = False
        return 1.0 if ok else 0.0

    async def verify(
        self, body: StructuredOutputsLangVerifyRequest
    ) -> StructuredOutputsLangVerifyResponse:
        model_out = body.response.output_text or ""
        text = model_out.strip()

        lp = self.config.language_penalty
        reasoning_template = bool(lp is not None and lp.reasoning_template)
        answer = self._extract_answer(text, reasoning_template)

        # ---- 1) schema validation in the target format -> so_reward (1.0/0.0) ----
        so_reward = self._so_reward(body.schema_str, body.schema_type, answer)
        reward = so_reward

        # ---- 2) multiplicative language penalty over the reasoning (< </think>) ----
        lang_mult = lang_factor = sc_factor = 1.0
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
            reward = reward * lang_mult

        # ---- 3) format gate: no closed </think> -> keep only `floor` of reward ----
        fg = self.config.format_gate
        think_open = reasoning_template or (f"<{fg.think_tag if fg else 'think'}>" in text)
        think_closed = f"</{fg.think_tag if fg else 'think'}>" in text
        format_factor = 1.0
        if fg is not None and fg.enabled:
            format_factor = 1.0 if f"</{fg.think_tag}>" in text else fg.floor
            reward = reward * format_factor

        self._log_rollout(
            {
                "language": body.language,
                "schema_type": body.schema_type,
                "so_reward": so_reward,
                "lang_multiplier": lang_mult,
                "lang_factor": lang_factor,
                "sc_factor": sc_factor,
                "format_factor": format_factor,
                "reward": reward,
                "think_open": think_open,
                "think_closed": think_closed,
                "answer": answer,
                "output": text,
            }
        )

        return StructuredOutputsLangVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            extracted_answer=answer,
            so_reward=so_reward,
            lang_multiplier=lang_mult,
            lang_factor=lang_factor,
            sc_factor=sc_factor,
            format_factor=format_factor,
        )


if __name__ == "__main__":
    StructuredOutputsLangResourcesServer.run_webserver()
