"""Cantonese value pools for sampling instruction kwargs.

Kept deliberately small and high-frequency so generated constraints read naturally and
are realistically satisfiable in a Cantonese response.
"""

# Common Cantonese / Chinese content words usable as required/forbidden keywords.
KEYWORDS = [
    "香港", "市民", "政府", "社會", "經濟", "教育", "文化", "科技", "健康", "環境",
    "交通", "天氣", "美食", "旅遊", "歷史", "音樂", "電影", "運動", "新聞", "故事",
    "屋企", "返工", "返學", "朋友", "屋邨", "茶餐廳", "地鐵", "巴士", "颱風", "中秋",
    "開心", "煩惱", "希望", "夢想", "回憶", "自由", "公平", "創新", "傳統", "未來",
]

# Single hanzi for character-frequency constraints (common, easy to place).
COMMON_HANZI = ["我", "你", "佢", "係", "嘅", "好", "唔", "人", "有", "心", "love", "家", "話"]
COMMON_HANZI = [c for c in COMMON_HANZI if len(c) == 1]

# Short phrases for copy:repeat_phrase.
PHRASES = [
    "今日天氣真係好好",
    "香港係我嘅屋企",
    "凡事都有轉機",
    "努力就會有收穫",
    "細水長流先至長久",
    "知足者常樂",
]

# End phrases for startend:end_checker.
END_PHRASES = [
    "多謝晒大家",
    "希望幫到你",
    "祝你好運",
    "下次再傾",
    "就係咁多",
]

# First/last word candidates (single tokens that can naturally start/end a sentence).
EDGE_WORDS = ["其實", "首先", "另外", "總括", "所以", "因此", "我", "你", "佢", "我哋"]

# Postscript markers (Chinese only — avoids clashing with no-English / full-width rules).
POSTSCRIPT_MARKERS = ["附註", "備註", "後記", "補充"]

# Cantonese colloquial marker characters (口語字) — presence signals 粵語口語 register.
CANTONESE_MARKERS = list("嘅喺咗佢啲唔係嚟睇嘢咁喇冇諗畀嗰乜嘥攰")

# Sentence-final particles for cantonese:sentence_final_particle.
FINAL_PARTICLES = list("啦喎㗎呀嘞囉㖞咩")

# Section split markers for detectable_format:multiple_sections.
SECTION_SPLITTERS = ["第", "部分", "SECTION", "段落"]

# 成語 dictionary for chinese:chengyu_count — the verifier counts how many of these appear
# in a response, so recall matters. Loaded from data/chengyu.txt (one idiom per line,
# CJK-only, len>=3); falls back to a small inline seed if the file is missing.
import os as _os
import re as _re

_CHENGYU_SEED = [
    "一帆風順", "心想事成", "萬事如意", "自強不息", "精益求精", "腳踏實地",
    "持之以恆", "一鳴驚人", "守株待兔", "畫蛇添足", "亡羊補牢", "塞翁失馬",
    "井底之蛙", "對牛彈琴", "胸有成竹", "一石二鳥", "因小失大", "捨本逐末",
    "水到渠成", "順其自然", "得心應手", "全力以赴", "繼往開來", "日新月異",
]
_CJK_ONLY = _re.compile(r"^[一-鿿]+$")


def _load_chengyu():
    path = _os.path.join(_os.path.dirname(__file__), "data", "chengyu.txt")
    words = set(_CHENGYU_SEED)
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if len(w) >= 3 and _CJK_ONLY.match(w):
                    words.add(w)
    except FileNotFoundError:
        pass
    return words


CHENGYU_SET = _load_chengyu()
CHENGYU = sorted(CHENGYU_SET)

# Constrained-response options (Cantonese rendering of the yes/no/maybe set).
CONSTRAINED_OPTIONS = ["我嘅答案係：係。", "我嘅答案係：唔係。", "我嘅答案係：可能。"]
