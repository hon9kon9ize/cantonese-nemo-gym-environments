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
# workplace_assistant_lang = the base `workplace_assistant` env (multi-step
# tool-using agent; reward = did the model's tool calls reach the ground-truth
# end state) forked with the SAME multiplicative language penalty + format gate
# used by math_with_judge_lang / stem_mcqa / code_gen_lang:
#     reward = wb_reward * language_multiplier * format_factor
# where wb_reward = is_correct(predicted_tool_calls, ground_truth) (binary), the
# language multiplier enforces the per-row reasoning language (yue = colloquial
# Cantonese, zh-hk = Written Traditional Chinese 書面語, en = English) and forbids
# Simplified Chinese, and format_factor requires a closed </think>.
#
# The tool suite + grader (get_tools / is_correct) and the seed_session / tool-
# routing machinery are REUSED from the base env via the `resources_servers`
# namespace package -- this fork adds only the language/format layer.
#
# LANGUAGE grading is over the WHOLE trajectory, not just the first <think>. A
# tool-use rollout has one <think>...</think> per turn (the chat template pre-opens
# <think> each assistant turn), so the single-block extractor would score only
# turn 1. Instead we strip the think tags from the full concatenated model text
# and re-wrap it, so language_multiplier judges every turn's reasoning (per the
# "grade the whole response, not only <think>" lesson). The tool CALLS are
# separate function_call items (not output_text), so they never enter the check.

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.language_penalty import language_multiplier, load_sc_set
from nemo_gym.server_utils import SESSION_ID_KEY

# Reuse the base env's tool suite + state-diff grader (namespace package import).
from resources_servers.workplace_assistant.utils import get_tools, is_correct


# --------------------------------------------------------------------------- #
# Multiplicative language penalty + format gate (identical design to the other
# *_lang envs; the reward module itself is shared via nemo_gym).
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
# Config / Schemas
# ----------------------------
class WorkbenchLangResourcesServerConfig(BaseResourcesServerConfig):
    language_penalty: Optional[LanguagePenaltyConfig] = None
    format_gate: Optional[FormatGateConfig] = None


class WorkbenchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchLangVerifyRequest(BaseVerifyRequest):
    ground_truth: list[Dict[str, str]] | str
    id: int
    category: str
    environment_name: str
    # Per-row target language (yue / zh-hk / en) for the language penalty.
    language: Optional[str] = None


class WorkbenchLangVerifyResponse(BaseVerifyResponse):
    # Reward breakdown (reward = wb_reward * lang_multiplier * format_factor).
    wb_reward: float = 0.0
    lang_multiplier: float = 1.0
    lang_factor: float = 1.0
    sc_factor: float = 1.0
    format_factor: float = 1.0


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
class WorkbenchLangResourcesServer(SimpleResourcesServer):
    config: WorkbenchLangResourcesServerConfig
    session_id_to_tool_env: Dict[str, Any] = Field(default_factory=dict)

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
                lp.rollout_log_dir, f"workplace_assistant_lang_rollouts_{ts}_{os.getpid()}.jsonl"
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

    # ---- tool machinery (reused verbatim from the base workplace_assistant) ----
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)
        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        toolkits = [
            "email",
            "calendar",
            "analytics",
            "project_management",
            "customer_relationship_manager",
        ]
        self.session_id_to_tool_env[session_id] = get_tools(toolkits)
        return BaseSeedSessionResponse()

    async def route_to_python_function(self, path: str, body: WorkbenchRequest, request: Request) -> WorkbenchResponse:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_tool_env:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Please call seed_session first.",
            )
        tool_env = self.session_id_to_tool_env[session_id]
        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}
        try:
            function = tool_env["functions"][path]
            result = function(**args)
            return WorkbenchResponse(output=result)
        except Exception as e:
            return WorkbenchResponse(output=f"Error executing tool '{path}': {str(e)}")

    # ---- reward: tool-state match * language multiplier * format gate ----
    def _full_model_text(self, body: WorkbenchLangVerifyRequest) -> str:
        """All natural-language text the model produced across the trajectory.

        Prefer the SDK's concatenated `output_text`; fall back to joining the
        text of every output_text/message item if that property is unavailable.
        """
        text = getattr(body.response, "output_text", None)
        if text:
            return text
        parts = []
        for item in body.response.output or []:
            itype = getattr(item, "type", None)
            if itype == "output_text":
                t = getattr(item, "text", None)
                if isinstance(t, str):
                    parts.append(t)
            elif itype == "message":
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
        return "\n".join(parts)

    async def verify(self, body: WorkbenchLangVerifyRequest) -> WorkbenchLangVerifyResponse:
        ground_truth = body.ground_truth

        # 1) base reward: did the predicted tool calls reach the ground-truth state?
        predicted_function_calls = [
            m.model_dump() for m in body.response.output if getattr(m, "type", None) == "function_call"
        ]
        wb_reward = float(is_correct(predicted_function_calls, ground_truth, None)) * 1.0
        reward = wb_reward

        text = self._full_model_text(body)

        # 2) multiplicative language penalty over the WHOLE trajectory's reasoning.
        # Strip the per-turn think tags and re-wrap as a single block so the shared
        # (single-block) extractor judges every turn, not just turn 1.
        lang_mult = lang_factor = sc_factor = 1.0
        lp = self.config.language_penalty
        if lp is not None and lp.enabled:
            cleaned = text.replace(f"<{lp.think_tag}>", " ").replace(f"</{lp.think_tag}>", " ").strip()
            wrapped = f"<{lp.think_tag}>{cleaned}</{lp.think_tag}>" if cleaned else ""
            lang_mult, lang_factor, sc_factor = language_multiplier(
                wrapped,
                body.language,
                self._sc_set,
                think_tag=lp.think_tag,
                reasoning_template=False,  # we re-wrap manually -> standard <think> body
                wrong_lang_floor=lp.wrong_lang_floor,
                min_target_script_ratio=lp.min_target_script_ratio,
                sc_strength=lp.sc_strength,
                sc_tolerance=lp.sc_tolerance,
                sc_full_ratio=lp.sc_full_ratio,
                sc_floor=lp.sc_floor,
                sc_min_hits=lp.sc_min_hits,
            )
            reward = reward * lang_mult

        # 3) format gate: the model must close at least one </think> across the run.
        fg = self.config.format_gate
        think_closed = f"</{fg.think_tag if fg else 'think'}>" in text
        format_factor = 1.0
        if fg is not None and fg.enabled:
            format_factor = 1.0 if think_closed else fg.floor
            reward = reward * format_factor

        self._log_rollout(
            {
                "id": body.id,
                "category": body.category,
                "language": body.language,
                "n_tool_calls": len(predicted_function_calls),
                "wb_reward": wb_reward,
                "lang_multiplier": lang_mult,
                "lang_factor": lang_factor,
                "sc_factor": sc_factor,
                "format_factor": format_factor,
                "reward": reward,
                "think_closed": think_closed,
                "output": text,
            }
        )

        return WorkbenchLangVerifyResponse(
            **body.model_dump(),
            reward=reward,
            wb_reward=wb_reward,
            lang_multiplier=lang_mult,
            lang_factor=lang_factor,
            sc_factor=sc_factor,
            format_factor=format_factor,
        )


if __name__ == "__main__":
    WorkbenchLangResourcesServer.run_webserver()
