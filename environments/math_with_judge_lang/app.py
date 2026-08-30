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
import contextlib
import json
import logging
import os
import threading
import time
from io import StringIO
from typing import Any, ClassVar, Dict, List, Optional, Union

from fastapi import FastAPI
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.language_penalty import language_multiplier, load_sc_set
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import get_response_json


class LanguagePenaltyConfig(BaseModel):
    """Multiplicative language-consistency + Simplified-Chinese penalty (ported from
    infinite-rl-nemo). The task reward is multiplied by a factor in [floor, 1]."""

    enabled: bool = True
    sc_list_path: str  # curated simplified-only char list (SC_list.txt)
    think_tag: str = "think"
    reasoning_template: bool = False  # True if the chat template pre-opens <think>
    wrong_lang_floor: float = 0.0  # reward multiplier when reasoning is in the wrong language
    min_target_script_ratio: float = 0.15  # >= this share of alphabetic chars in the
    # target script counts as target-language (RELAXED for heavy code-mixing).
    sc_strength: float = 1.0
    sc_tolerance: float = 0.05
    sc_full_ratio: float = 0.20
    sc_floor: float = 0.0
    sc_min_hits: int = 2
    # If set, every verify() appends a full rollout record (prompt / model output /
    # reward breakdown) as JSONL under this dir -- one timestamped file per server
    # start -- for offline analysis instead of scraping the training log.
    rollout_log_dir: Optional[str] = None


class FormatGateConfig(BaseModel):
    """Multiplicative format gate (the missing piece vs the proven infinite-rl-nemo
    recipe): a rollout without a closed </think> block keeps only `floor` of its
    reward. This makes untagged direct answers and truncated think-loopers strictly
    worse than proper <think>...</think> + answer, so GRPO reinforces the tagged
    reasoning format instead of silently drifting to no-think."""

    enabled: bool = True
    # Non-zero floor for the same reason as wrong_lang_floor: keep a math signal so
    # a group of all-unformatted rollouts still has gradient toward correct math.
    floor: float = 0.2
    think_tag: str = "think"


class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True
    language_penalty: Optional[LanguagePenaltyConfig] = None
    format_gate: Optional[FormatGateConfig] = None


class LibraryJudgeMathRunRequest(BaseRunRequest):
    question: str
    expected_answer: str
    # Per-row target language (yue / zh-hk / en) for the language penalty.
    language: Optional[str] = None


class LibraryJudgeMathVerifyRequest(LibraryJudgeMathRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class LibraryJudgeMathVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    library_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]]
    # Reward breakdown (reward = math_reward * lang_multiplier * format_factor).
    math_reward: float = 0.0
    lang_multiplier: float = 1.0
    lang_factor: float = 1.0
    sc_factor: float = 1.0
    format_factor: float = 1.0


class LibraryJudgeMathResourcesServer(SimpleResourcesServer):
    # These judge messages are adapted from ones used in Arena Hard.
    # https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/utils/judge_utils.py
    # They are intended to serve as example messages for an LLM judge, and have not
    # been customized for a specific judge model.
    JUDGE_SYSTEM_MESSAGE: ClassVar[
        str
    ] = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to the mathematical problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the mathematical equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "<|Problem|>\n{question}\n\n<|Start of Assistant A's Answer|>\n{first_answer}\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n{second_answer}\n<|End of Assistant B's Answer|>"
    )

    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"

    config: LibraryJudgeMathResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self._library_verifier = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

        # Load the curated Simplified-Chinese char set once (empty = penalty off).
        self._sc_set = frozenset()
        lp = self.config.language_penalty
        if lp is not None and lp.enabled:
            self._sc_set = load_sc_set(lp.sc_list_path)
            logging.getLogger("nemo_gym").info(
                "Language penalty ON: %d simplified chars, wrong_lang_floor=%s",
                len(self._sc_set),
                lp.wrong_lang_floor,
            )

        # Rollout dump: one timestamped JSONL per server start, appended under a lock.
        self._rollout_log_path: Optional[str] = None
        self._rollout_lock = threading.Lock()
        if lp is not None and lp.enabled and lp.rollout_log_dir:
            os.makedirs(lp.rollout_log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._rollout_log_path = os.path.join(
                lp.rollout_log_dir, f"math_lang_rollouts_{ts}_{os.getpid()}.jsonl"
            )
            logging.getLogger("nemo_gym").info("Saving rollouts to %s", self._rollout_log_path)

    def _log_rollout(self, record: Dict[str, Any]) -> None:
        """Append one rollout record as a JSON line (best-effort; never raise)."""
        if not self._rollout_log_path:
            return
        try:
            line = json.dumps(record, ensure_ascii=False)
            with self._rollout_lock, open(self._rollout_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:  # logging must never break verification
            logging.getLogger("nemo_gym").exception("Failed to write rollout record")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: LibraryJudgeMathVerifyRequest) -> LibraryJudgeMathVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        (
            reward,
            extracted_answer,
            library_reward,
            judge_evaluations,
        ) = await self._verify_answer(body.question, body.expected_answer, combined_response)

        # Multiplicative language penalty: correct-language reasoning keeps the full
        # math reward; wrong language / colloquial-Cantonese-for-written / Simplified
        # Chinese scales it down (toward wrong_lang_floor).
        math_reward = reward
        lang_mult = lang_factor = sc_factor = 1.0
        lp = self.config.language_penalty
        if lp is not None and lp.enabled:
            lang_mult, lang_factor, sc_factor = language_multiplier(
                combined_response,
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
            reward = math_reward * lang_mult

        # Format gate: no closed </think> block -> keep only `floor` of the reward.
        # Pair with overlong_filtering: false in the GRPO config so truncated
        # think-loopers actually reach the optimizer as negative advantage instead
        # of being hidden from it.
        fg = self.config.format_gate
        # With a pre-opening chat template the literal <think> lives in the PROMPT,
        # not the model output, so treat reasoning as implicitly open in that mode
        # (keeps the logged think_open meaningful for offline rollout analysis).
        reasoning_template = bool(lp is not None and lp.reasoning_template)
        think_open = reasoning_template or ("<think>" in combined_response)
        think_closed = "</think>" in combined_response
        format_factor = 1.0
        if fg is not None and fg.enabled:
            format_factor = 1.0 if f"</{fg.think_tag}>" in combined_response else fg.floor
            reward = reward * format_factor

        # Persist the full rollout for offline analysis (see language_penalty.rollout_log_dir).
        self._log_rollout(
            {
                "language": body.language,
                "question": body.question,
                "expected_answer": body.expected_answer,
                "extracted_answer": extracted_answer,
                "math_reward": math_reward,
                "library_reward": library_reward,
                "lang_multiplier": lang_mult,
                "lang_factor": lang_factor,
                "sc_factor": sc_factor,
                "format_factor": format_factor,
                "reward": reward,
                "think_open": think_open,
                "think_closed": think_closed,
                "output": combined_response,
            }
        )

        return LibraryJudgeMathVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            library_reward=library_reward,
            judge_evaluations=judge_evaluations,
            math_reward=math_reward,
            lang_multiplier=lang_mult,
            lang_factor=lang_factor,
            sc_factor=sc_factor,
            format_factor=format_factor,
        )

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        """Verify the correctness of a generated answer.

        Verify the correctness of the specified model-generated answer to the
        specified question in comparison with the specified expected answer.
        """

        library_reward, extracted_answer = self._verify_answer_with_library(expected_answer, generated_answer)
        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        judge_answer = extracted_answer if extracted_answer else generated_answer
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(question, expected_answer, judge_answer)
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    @staticmethod
    def _strip_math_delimiters(s: str) -> str:
        """Strip outer math delimiters from expected answers.

        Many expected_answer values are wrapped in \\(...\\) or $...$,
        which causes the math_verify parser to fail when we wrap them
        in \\boxed{}.  Removing these outer delimiters fixes parsing.
        """
        s = s.strip()
        if s.startswith("\\(") and s.endswith("\\)"):
            s = s[2:-2].strip()
        if s.startswith("$") and s.endswith("$") and len(s) > 1:
            s = s[1:-1].strip()
        return s

    def _verify_answer_with_library(self, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
        # This functionality is migrated from Nemo RL.
        # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
        try:
            stripped = self._strip_math_delimiters(expected_answer)
            ground_truth_parsable = "\\boxed{" + stripped + "}"
            with self._mute_output():
                ret_score, extracted_answer = self._library_verifier([ground_truth_parsable], [generated_answer])

            reward = float(ret_score)

            if extracted_answer is not None:
                # Make sure the extracted answer has two elements.
                assert len(extracted_answer) == 2

                extracted_gold, extracted_prediction = extracted_answer

                # Get the extracted answer.
                for pred in extracted_prediction:
                    if any(grader.verify(gold, pred) for gold in extracted_gold):
                        extracted_answer = pred
                        break
                else:
                    # If no match is found, that means all the answers are
                    # incorrect.  The first prediction is used as the extracted
                    # answer.
                    extracted_answer = extracted_prediction[0] if extracted_prediction else None

            return reward, extracted_answer

        # It's possible to emit a TimeoutException and that wouldn't be caught since
        # it actually subclasses from BaseException and math-verify itself does not
        # catch it.
        except (Exception, TimeoutException):
            return 0.0, None

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # The judge is asked to evaluate whether the answers are equal using both
        # orders of the answers, in case there is any positional bias in terms of
        # the order in which the answers are presented to the judge model.
        (
            first_order_equal,
            first_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, expected_answer, generated_answer)
        if not first_order_equal:
            return 0.0, [first_judge_evaluation]

        (
            second_order_equal,
            second_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, generated_answer, expected_answer)
        if second_order_equal:
            reward = 1.0
        else:
            reward = 0.0
        return reward, [first_judge_evaluation, second_judge_evaluation]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(
                role="system",
                content=self.JUDGE_SYSTEM_MESSAGE,
            ),
            NeMoGymEasyInputMessage(
                role="user",
                content=judge_prompt,
            ),
        ]

        response = await self.server_client.post(
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Currently, for all the cases in which the response from the LLM judge
        # does not conform to the expected format, the judge's evaluation is
        # treated as if the answers are not equal.  This may not be ideal, but it
        # is intended to minimize the number of failures for verify requests.
        last_output = judge_response.output[-1]
        if last_output.type != "message":
            return False, judge_evaluation

        last_content = last_output.content[-1]
        if last_content.type != "output_text":
            return False, judge_evaluation

        output_text = last_content.text
        equal_choice_position = output_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_choice_position = output_text.find(self.JUDGE_NOT_EQUAL_LABEL)

        # The first label that appears in the text is used for the evaluation.
        if equal_choice_position < 0:
            if not_equal_choice_position < 0:
                return False, judge_evaluation
            else:
                return False, judge_evaluation
        else:
            if not_equal_choice_position < 0:
                return True, judge_evaluation
            elif equal_choice_position < not_equal_choice_position:
                return True, judge_evaluation
            else:
                return False, judge_evaluation

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _math_score_fn(r: dict) -> Dict[str, Union[float, bool]]:
        scores: Dict[str, Union[float, bool]] = {}
        if "library_reward" in r:
            scores["symbolic_accuracy"] = r["library_reward"]
        if "judge_evaluations" in r and r["judge_evaluations"] is not None:
            scores["judge_accuracy"] = r["reward"]
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute math-specific metrics: pass@k, majority@k, per-sample statistics."""
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._math_score_fn,
            answer_key="extracted_answer",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for this math benchmark."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))

        return key


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
