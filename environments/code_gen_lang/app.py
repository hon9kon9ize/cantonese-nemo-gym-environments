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
# code_gen_lang = the base `code_gen` env (LiveCodeBench unit-test execution)
# forked with the SAME multiplicative language penalty + format gate used by
# `math_with_judge_lang` and `stem_mcqa` (ported from infinite-rl-nemo):
#     reward = code_reward * language_multiplier * format_factor
# where code_reward = 1.0 iff the extracted program passes ALL unit tests,
# language_multiplier enforces the per-row reasoning language (yue = colloquial
# Cantonese, zh-hk = Written Traditional Chinese 書面語, en = English) and
# forbids Simplified Chinese, and format_factor requires a closed </think>.
#
# NOTE on the format check: the upstream env uses `reasoning_format_penalty`,
# which flags any `<think>`/`</think>` still present in output_text — correct
# only when a reasoning PARSER splits reasoning from the answer. Our policy runs
# with `uses_reasoning_parser: false`, so `<think>...</think>` stays INLINE in
# output_text; the upstream check would then false-positive on every rollout.
# We drop it and use the shared multiplicative `format_gate` instead (the exact
# piece stem_mcqa/math_with_judge_lang use). The final ```python``` code block
# sits AFTER </think>, so the language check (which reads only the reasoning
# before </think>) is not polluted by the code's English keywords.

import json
import logging
import os
import threading
import time
from asyncio import Semaphore, get_running_loop
from time import time as _time
from typing import Any, Dict, List, Optional, Union

import ray
from lcb_integration.compute_code_generation_metrics import check_correctness_remote
from lcb_integration.extraction_utils import LMStyle, extract_code
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.language_penalty import language_multiplier, load_sc_set
from nemo_gym.reward_profile import (
    add_avg_sample_std_dev,
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)


# --------------------------------------------------------------------------- #
# Multiplicative language penalty + format gate (identical design to
# math_with_judge_lang / stem_mcqa; the reward module is shared via nemo_gym).
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
class CompCodingLangResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int
    unit_test_timeout_secs: int
    debug: bool
    language_penalty: Optional[LanguagePenaltyConfig] = None
    format_gate: Optional[FormatGateConfig] = None


# ----------------------------
# Schemas
# ----------------------------


# This is LiveCodeBench format
class UnitTests(BaseModel):
    inputs: List[str]
    outputs: List[str]
    fn_name: Optional[str] = None


class CompCodingLangRunRequest(BaseRunRequest):
    # Per-row target language (yue / zh-hk / en) for the language penalty.
    language: Optional[str] = None


class CompCodingLangVerifyRequest(CompCodingLangRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class CompCodingLangVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_model_code: Optional[str] = None
    result: Optional[List[Union[int, bool]]] = None
    metadata: Optional[Dict[str, Any]] = None
    unit_tests_time_taken: Optional[float] = None
    difficulty: Optional[str] = None
    # Reward breakdown (reward = code_reward * lang_multiplier * format_factor).
    code_reward: float = 0.0
    lang_multiplier: float = 1.0
    lang_factor: float = 1.0
    sc_factor: float = 1.0
    format_factor: float = 1.0


# ----------------------------
# Server
# ----------------------------
class CompCodingLangResourcesServer(SimpleResourcesServer):
    config: CompCodingLangResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)

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
                lp.rollout_log_dir, f"code_gen_lang_rollouts_{ts}_{os.getpid()}.jsonl"
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
    def _code_score_fn(r: dict) -> Dict[str, float]:
        return {"accuracy": float(r["reward"] > 0)}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Code-generation metrics: pass@k, majority@k, per-sample statistics."""
        metrics, all_score_dicts, score_names, max_k = compute_pass_majority_metrics(
            tasks,
            score_fn=self._code_score_fn,
            answer_key="extracted_model_code",
        )
        add_avg_sample_std_dev(metrics, all_score_dicts, score_names, max_k)
        metrics.update(compute_subset_metrics(tasks, "difficulty", self._code_score_fn, "extracted_model_code"))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens", "mean/reward"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", score_names=["accuracy"]))
        for prefix in {k.split("/pass@")[0] for k in agent_metrics if "/pass@" in k and k[0].islower()}:
            key.update(highest_k_metrics(agent_metrics, f"{prefix}/pass@1[avg-of-{{k}}]", score_names=["accuracy"]))
        return key

    async def verify(self, body: CompCodingLangVerifyRequest) -> CompCodingLangVerifyResponse:
        model_out = body.response.output_text
        text = (model_out or "").strip()
        difficulty = (body.verifier_metadata or {}).get("difficulty")

        # ---- 1) run the unit tests -> code_reward (1.0 all-pass else 0.0) ----
        code: Optional[str] = None
        result = None
        metadata = None
        unit_tests_time_taken = None
        code_reward = 0.0

        if text:
            code = extract_code(model_out, LMStyle.OpenAIChat)
        if code and body.verifier_metadata and body.verifier_metadata.get("unit_tests"):
            tests = UnitTests.model_validate(body.verifier_metadata["unit_tests"])
            async with self._semaphore:
                loop = get_running_loop()
                start_time = _time()
                task_args = (
                    {"input_output": tests.model_dump_json()},  # sample
                    code,  # generation
                    self.config.unit_test_timeout_secs,  # timeout
                    self.config.debug,  # debug
                )
                future = check_correctness_remote.remote(*task_args)
                try:
                    result, metadata = await loop.run_in_executor(None, ray.get, future)
                except Exception as e:
                    # The sandbox Ray task runs untrusted model-generated code. If that
                    # worker dies hard (OOM-kill, segfault, hostile code killing its parent,
                    # or Ray losing the worker log -> FileNotFoundError in ray.get), the
                    # candidate is simply un-evaluable: score it 0 rather than raising a 500,
                    # which NeMo-Gym's raise_for_status turns into a fatal RayTaskError that
                    # aborts the whole GRPO run. A failing solution must not kill training.
                    # Log str(e), not repr(e): a Ray RayTaskError's repr hides the wrapped
                    # exception's message (prints an empty "()"), but str() renders the full
                    # remote traceback with the real file+line. Only failures that exhaust the
                    # task's max_retries reach here, so the volume stays bounded.
                    logging.getLogger("nemo_gym").warning(
                        "check_correctness_remote failed after retries; scoring code_reward=0.0:\n%s", e
                    )
                    result, metadata = None, {"error": repr(e)}
                unit_tests_time_taken = _time() - start_time
            code_reward = 1.0 if (result and all(r == True for r in result)) else 0.0

        reward = code_reward

        # ---- 2) multiplicative language penalty over the reasoning (< </think>) ----
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
            reward = reward * lang_mult

        # ---- 3) format gate: no closed </think> -> keep only `floor` of reward ----
        fg = self.config.format_gate
        reasoning_template = bool(lp is not None and lp.reasoning_template)
        think_open = reasoning_template or (f"<{fg.think_tag if fg else 'think'}>" in text)
        think_closed = f"</{fg.think_tag if fg else 'think'}>" in text
        format_factor = 1.0
        if fg is not None and fg.enabled:
            format_factor = 1.0 if f"</{fg.think_tag}>" in text else fg.floor
            reward = reward * format_factor

        self._log_rollout(
            {
                "language": body.language,
                "difficulty": difficulty,
                "code_reward": code_reward,
                "n_tests": len(result) if result else 0,
                "lang_multiplier": lang_mult,
                "lang_factor": lang_factor,
                "sc_factor": sc_factor,
                "format_factor": format_factor,
                "reward": reward,
                "think_open": think_open,
                "think_closed": think_closed,
                "extracted_code": code,
                "output": text,
            }
        )

        return CompCodingLangVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_model_output=model_out,
            extracted_model_code=code,
            result=result,
            metadata=metadata,
            unit_tests_time_taken=unit_tests_time_taken,
            difficulty=difficulty,
            code_reward=code_reward,
            lang_multiplier=lang_mult,
            lang_factor=lang_factor,
            sc_factor=sc_factor,
            format_factor=format_factor,
        )


if __name__ == "__main__":
    CompCodingLangResourcesServer.run_webserver()
