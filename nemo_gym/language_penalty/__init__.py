# Language-consistency + Simplified-Chinese penalty for NeMo-Gym rewards.
#
# Ported from infinite-rl-nemo (infinite_rl_nemo/language_reward.py + parsing.py +
# the vendored cantofilter) so it can be REUSED by any NeMo-Gym resources server:
#
#     from nemo_gym.language_penalty import language_multiplier, load_sc_set
#
# `language_multiplier(output, target_language, sc_set, ...)` returns a factor in
# [0, 1] that a verifier multiplies into its task reward:
#   - language match : the reasoning is in the row's target language
#         yue          -> vendored cantofilter (cantonese / mixed ok, mandarin bad)
#         zh / zh-hk / zh-hant -> mostly-CJK (Han) script
#         en           -> mostly-latin script
#   - simplified cleanliness : fraction of CJK chars in the curated simplified-only
#         list (SC_list.txt); our targets are Cantonese / Traditional / English, so
#         ANY simplified-distinct form (国/书/见, not shared 我/你/的) is penalized.
#
# final multiplier = lang_factor * sc_factor, both in [0, 1].
from __future__ import annotations

from typing import FrozenSet, Optional, Tuple

# ── vendored Cantonese detector (pure-python; optional) ─────────────────────
_yue_detector = None
_lang_import_attempted = False


def _try_import_yue() -> None:
    global _yue_detector, _lang_import_attempted
    if _lang_import_attempted:
        return
    _lang_import_attempted = True
    try:
        from nemo_gym.language_penalty._vendor.cantofilter import judge as yue_detector

        _yue_detector = yue_detector
    except Exception:
        _yue_detector = None


# ── reasoning (think) extraction ────────────────────────────────────────────
import re


def _extract_tag(text: str, tag: str = "think") -> str:
    if not tag:
        return text
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    return ""


def extract_think_content(model_output: str, think_tag: str = "think", reasoning_template: bool = False) -> str:
    """Reasoning content from ``model_output``.

    reasoning_template=True: the chat template pre-opens ``<think>`` in the prompt,
    so the reasoning is everything before the single ``</think>`` in the output.
    Otherwise the standard ``<think>...</think>`` body is returned.
    """
    output = model_output or ""
    if reasoning_template:
        close = f"</{think_tag}>"
        idx = output.find(close)
        if idx > 0:
            content = output[:idx].strip()
            open_tag = f"<{think_tag}>"
            if content.startswith(open_tag):
                content = content[len(open_tag):].strip()
            return content
        return ""
    return _extract_tag(output, tag=think_tag).strip()


# ── script / simplified statistics ──────────────────────────────────────────
def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3400 <= o <= 0x4DBF  # CJK Ext A
        or 0x4E00 <= o <= 0x9FFF  # CJK Unified
        or 0xF900 <= o <= 0xFAFF  # CJK Compatibility
        or 0x20000 <= o <= 0x2A6DF  # CJK Ext B
    )


def load_sc_set(path: str) -> FrozenSet[str]:
    """Load the curated simplified-only character list (one char per line)."""
    chars: set = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                chars.update(s)  # tolerate stray multi-char lines
    return frozenset(chars)


def _cjk_stats(text: str, sc_set: FrozenSet[str]) -> Tuple[int, int, int]:
    n_cjk = n_sc = n_latin = 0
    for c in text:
        if _is_cjk(c):
            n_cjk += 1
            if sc_set and c in sc_set:
                n_sc += 1
        elif "a" <= c <= "z" or "A" <= c <= "Z":
            n_latin += 1
    return n_cjk, n_sc, n_latin


def simplified_factor(
    n_cjk: int,
    n_sc: int,
    *,
    sc_strength: float,
    sc_tolerance: float,
    sc_full_ratio: float,
    sc_floor: float,
    sc_min_hits: int,
) -> float:
    """1.0 when clean; drops toward sc_floor as the simplified share of CJK chars
    rises from sc_tolerance to sc_full_ratio. A dead zone (sc_min_hits absolute +
    sc_tolerance ratio) absorbs curated-list false positives (chars valid in both
    scripts, e.g. 出)."""
    if n_cjk == 0 or n_sc < sc_min_hits:
        return 1.0
    ratio = n_sc / n_cjk
    if ratio <= sc_tolerance:
        return 1.0
    span = max(sc_full_ratio - sc_tolerance, 1e-6)
    frac = min(1.0, (ratio - sc_tolerance) / span) * sc_strength
    return max(sc_floor, 1.0 - frac)


def language_match(
    content: str, target_language: str, n_cjk: int, n_latin: int, min_target_script_ratio: float = 0.15
) -> float:
    """In [0, 1]: is the reasoning written in the target language?

    RELAXED for code-mixing (2026-07-14): the reasoning is accepted as the target
    script as long as that script makes up at least ``min_target_script_ratio`` of
    the alphabetic characters — NOT a strict majority. Our questions are heavily
    code-mixed by design (Cantonese / written-Chinese reasoning sprinkled with
    English technical terms, tool names, identifiers), and a strict Han>=Latin gate
    wrong-language-penalized genuinely-Chinese CoT the moment the English tokens
    outnumbered the Han ones. Now only reasoning that is *essentially all* the wrong
    script (target script < ratio of alphabetic chars) is treated as wrong-language;
    everything else goes to the cantofilter, which judges colloquial-vs-written from
    the Chinese characters alone (English tokens don't affect it). Digits/symbols
    count as neither script.

    Cantonese (yue) and Written Traditional Chinese (zh-hk / 書面語) are DIFFERENT
    targets and must not be conflated: colloquial Cantonese (係/嘅/唔/喺 …) is the
    goal for `yue` but WRONG for `zh-hk`, where we want standard written Chinese.
    The cantofilter judgement drives both directions."""
    lang = (target_language or "en").lower()
    total = n_cjk + n_latin
    if total == 0:
        script = "none"
    else:
        # A script counts as "present enough" at >= min_target_script_ratio of the
        # alphabetic chars. For the target's own script this is what relaxes the gate;
        # the non-target script only wins when the target's is below the ratio.
        han_frac = n_cjk / total
        chinese_target = lang in ("yue", "zh", "zh-hk", "zh-hant", "zh-hans", "zh-cn")
        if chinese_target:
            script = "han" if han_frac >= min_target_script_ratio else "latin"
        else:
            script = "latin" if (1.0 - han_frac) >= min_target_script_ratio else "han"

    chinese_targets = ("yue", "zh", "zh-hk", "zh-hant", "zh-hans", "zh-cn")
    if lang in chinese_targets:
        if script == "latin":
            return 0.0  # English / latin reasoning for a Chinese question
        if script == "none":
            return 0.5  # pure math, can't tell — don't hard-penalize

        _try_import_yue()
        if _yue_detector is None or not content:
            # detector unavailable -> can't tell colloquial vs written; don't penalize
            return 1.0
        judgement = str(_yue_detector(content))

        if lang == "yue":
            # want COLLOQUIAL Cantonese
            if judgement in ("cantonese", "mixed"):
                return 1.0
            if judgement == "neutral":
                return 0.5  # Chinese but not distinctly Cantonese
            return 0.0  # "mandarin"/written when Cantonese is wanted
        else:
            # zh-hk / zh / zh-hant: want WRITTEN Chinese (書面語). Colloquial Cantonese
            # is WRONG here (the exact failure mode observed: written rows answered in
            # Cantonese). Traditional-vs-Simplified is handled by the SC penalty.
            if judgement in ("cantonese", "mixed"):
                return 0.0  # colloquial Cantonese when written Chinese is wanted
            return 1.0  # "mandarin"/"neutral" == standard written Chinese

    # english / default target: Chinese reasoning is wrong language.
    return 0.0 if script == "han" else 1.0


def language_multiplier(
    output: str,
    target_language: str,
    sc_set: FrozenSet[str],
    *,
    think_tag: str = "think",
    reasoning_template: bool = False,
    wrong_lang_floor: float = 0.1,
    min_target_script_ratio: float = 0.15,
    sc_strength: float = 1.0,
    sc_tolerance: float = 0.05,
    sc_full_ratio: float = 0.20,
    sc_floor: float = 0.0,
    sc_min_hits: int = 2,
) -> Tuple[float, float, float]:
    """Reward multiplier for language + script correctness.

    Returns (multiplier, lang_factor, sc_factor), all in [0, 1]. If there is no
    reasoning to judge, returns (1.0, 1.0, 1.0) so the format/task reward decides.
    """
    content = extract_think_content(output or "", think_tag=think_tag, reasoning_template=reasoning_template)
    if not content:
        # No usable <think>...</think> block (model didn't reason, or the block is
        # unclosed). Grade the WHOLE response instead of skipping: we want the entire
        # answer in the target language, not only the CoT. Digits/LaTeX in \boxed{}
        # count as neither script, so the math answer doesn't distort the language call.
        content = (output or "").strip()
    if not content:
        return 1.0, 1.0, 1.0
    n_cjk, n_sc, n_latin = _cjk_stats(content, sc_set)
    lm = language_match(content, target_language, n_cjk, n_latin, min_target_script_ratio=min_target_script_ratio)
    lang_factor = wrong_lang_floor + (1.0 - wrong_lang_floor) * lm
    sc_factor = simplified_factor(
        n_cjk,
        n_sc,
        sc_strength=sc_strength,
        sc_tolerance=sc_tolerance,
        sc_full_ratio=sc_full_ratio,
        sc_floor=sc_floor,
        sc_min_hits=sc_min_hits,
    )
    return lang_factor * sc_factor, lang_factor, sc_factor


__all__ = ["language_multiplier", "load_sc_set", "extract_think_content"]
