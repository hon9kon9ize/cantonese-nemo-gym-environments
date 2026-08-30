"""Cantonese/Chinese-aware text utilities shared by the verifiers.

These back the verifiable-reward functions, so they must match how a constraint is
*scored*, not just rendered. Chinese has no whitespace word boundaries, so "words" are
jieba tokens and "characters" are CJK ideographs; sentences split on full/half-width
terminators; paragraphs split on blank lines or markdown dividers.
"""
import re
import jieba

# Silence jieba's first-run logging to stderr.
jieba.setLogLevel(60)

# CJK Unified Ideographs (+ Ext-A) — what we count as a "Chinese character".
_CJK = r"㐀-䶿一-鿿"
CJK_RE = re.compile(f"[{_CJK}]")

# Full-width / Chinese punctuation we consider "punctuation" for full-width checks.
FULLWIDTH_PUNCT = "，。、！？；：「」『』（）《》〈〉【】…—～"
# Half-width punctuation that should be absent when full-width is required.
ASCII_PUNCT = ",.!?;:\"'()[]{}<>"

SENT_TERMINATORS = "。！？!?…"
_SENT_SPLIT_RE = re.compile(r"[。！？!?]+[」』）\)]?|…+")

CANTONESE_FINAL_PARTICLES = set("啦喎㗎呀嘞囉㖞咩咧吖喇嘛吧呢喔嗮添")


def count_hanzi(text: str) -> int:
    """Number of CJK ideographs (ignores punctuation, latin, digits, spaces)."""
    return len(CJK_RE.findall(text))


def char_count_of(text: str, ch: str) -> int:
    return text.count(ch)


def tokenize(text: str) -> list[str]:
    """jieba word tokens, dropping pure whitespace/punctuation tokens."""
    toks = []
    for t in jieba.lcut(text):
        t = t.strip()
        if not t:
            continue
        if all(not (c.isalnum() or CJK_RE.match(c)) for c in t):
            continue  # pure punctuation
        toks.append(t)
    return toks


def count_words(text: str) -> int:
    return len(tokenize(text))


def split_sentences(text: str) -> list[str]:
    """Split on Chinese/English sentence terminators; keep terminator with the sentence."""
    out, last = [], 0
    for m in _SENT_SPLIT_RE.finditer(text):
        seg = text[last:m.end()].strip()
        if seg:
            out.append(seg)
        last = m.end()
    tail = text[last:].strip()
    if tail:
        out.append(tail)
    return out


def split_paragraphs(text: str, divider: str | None = None) -> list[str]:
    """Split into paragraphs. If a divider is given (e.g. '***'), split on it;
    otherwise split on blank lines (two newlines)."""
    if divider:
        parts = re.split(re.escape(divider), text)
    else:
        parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def strip_terminal_punct(s: str) -> str:
    return s.rstrip(FULLWIDTH_PUNCT + ASCII_PUNCT + " \t\n").rstrip()


def first_token(text: str) -> str | None:
    toks = tokenize(text)
    return toks[0] if toks else None


def last_token(text: str) -> str | None:
    toks = tokenize(text)
    return toks[-1] if toks else None


def has_ascii_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def has_arabic_digits(text: str) -> bool:
    return bool(re.search(r"[0-9]", text))


def relation_ok(n: int, relation: str, target: int) -> bool:
    """Compare a count against a target under an IFEval-style relation token."""
    r = relation.strip().lower()
    if r in ("at least", "least", ">=", "最少"):
        return n >= target
    if r in ("at most", "most", "<=", "最多"):
        return n <= target
    if r in ("exactly", "==", "剛好"):
        return n == target
    if r in ("less than", "<", "少於"):
        return n < target
    if r in ("more than", ">", "多於"):
        return n > target
    if r in ("around", "about", "~", "大約"):
        return abs(n - target) <= max(1, round(0.1 * target))
    raise ValueError(f"unknown relation: {relation!r}")


# Cantonese display strings for the English relation tokens stored in kwargs.
RELATION_YUE = {
    "at least": "最少",
    "at most": "最多",
    "exactly": "剛好",
    "less than": "少於",
    "more than": "多於",
    "around": "大約",
}
