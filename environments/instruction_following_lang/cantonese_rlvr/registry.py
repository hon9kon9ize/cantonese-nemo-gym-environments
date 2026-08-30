"""Per-instruction_id registry: a kwargs sampler + a verifiable-reward function.

For each active instruction_id we provide:
  sample(rng) -> (kwargs, render_vars)
      kwargs       : the machine-readable params stored in the record (verifier reads these)
      render_vars  : extra/override values used only to fill the Cantonese template text
  verify(response, kwargs) -> bool | None
      the RLVR reward check.  None means "not deterministically verifiable here"
      (e.g. rhyme without pycantonese).

The Cantonese instruction text itself lives in analysis/cantonese_instruction_templates.json
(single source of truth); this module only supplies values + checks.
"""
from __future__ import annotations
import json
import re
import difflib

from . import lexicon as L
from . import text_utils as T

REG: dict[str, dict] = {}

# Keywords jieba keeps as a single token — required for token-position checks.
_SINGLE_TOK_KW = [w for w in L.KEYWORDS if len(T.tokenize(w)) == 1]


def use_lexicon(mod):
    """Point the kwarg samplers at a different value-pool module (build-time only).

    Verifiers are language-neutral; only sample() (and a couple of verify() fallbacks)
    read these pools. build_if/build_so/test_verifiers call this with lexicon_zh under
    --lang zh to generate 書面語 kwargs. Default stays the Cantonese lexicon.
    """
    global L, _SINGLE_TOK_KW
    L = mod
    _SINGLE_TOK_KW = [w for w in L.KEYWORDS if len(T.tokenize(w)) == 1]


def _reg(iid):
    def deco(cls):
        REG[iid] = {"sample": cls.sample, "verify": cls.verify}
        return cls
    return deco


def _join_kw(words):
    return "、".join(f"「{w}」" for w in words)


def _rel_disp(relation):
    return T.RELATION_YUE.get(relation, relation)


# --------------------------------------------------------------------------- A
@_reg("keywords:existence")
class _:
    def sample(rng):
        kws = rng.sample(L.KEYWORDS, 2)
        return {"keywords": kws}, {"keywords": _join_kw(kws)}
    def verify(r, kw): return all(w in r for w in kw["keywords"])

@_reg("keywords:forbidden_words")
class _:
    def sample(rng):
        kws = rng.sample(L.KEYWORDS, 2)
        return {"forbidden_words": kws}, {"forbidden_words": _join_kw(kws)}
    def verify(r, kw): return all(w not in r for w in kw["forbidden_words"])

@_reg("keywords:word_once")
class _:
    def sample(rng):
        w = rng.choice(L.KEYWORDS)
        return {"keyword": w}, {"keyword": w}
    def verify(r, kw): return r.count(kw["keyword"]) >= 1

@_reg("keywords:word_count_different_numbers")
class _:
    def sample(rng):
        w = rng.choice(L.KEYWORDS); n = rng.randint(2, 4)
        return {"keyword": w, "frequency": n}, {"keyword": w, "frequency": n}
    def verify(r, kw): return r.count(kw["keyword"]) == kw["frequency"]

@_reg("keywords:frequency")
class _:
    def sample(rng):
        w = rng.choice(L.KEYWORDS); n = rng.randint(2, 4)
        rel = rng.choice(["at least", "at most", "exactly"])
        return {"keyword": w, "relation": rel, "frequency": n}, {"keyword": w, "relation": _rel_disp(rel), "frequency": n}
    def verify(r, kw): return T.relation_ok(r.count(kw["keyword"]), kw["relation"], kw["frequency"])

@_reg("count:count_increment_word")
class _:
    def sample(rng):
        a, b = rng.sample(L.KEYWORDS, 2)
        return {"keyword1": a, "keyword2": b}, {"keyword1": a, "keyword2": b}
    def verify(r, kw): return r.count(kw["keyword1"]) == 1 and r.count(kw["keyword2"]) == 2

@_reg("detectable_content:number_placeholders")
class _:
    def sample(rng):
        n = rng.randint(2, 4)
        return {"num_placeholders": n}, {"num_placeholders": n}
    def verify(r, kw): return len(re.findall(r"\[[^\]]*\]", r)) >= kw["num_placeholders"]

@_reg("detectable_content:postscript")
class _:
    def sample(rng):
        m = rng.choice(L.POSTSCRIPT_MARKERS)
        return {"postscript_marker": m}, {"postscript_marker": m}
    def verify(r, kw): return kw["postscript_marker"] in r

@_reg("detectable_format:title")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw): return bool(re.search(r"《[^》]+》", r))

@_reg("detectable_format:number_bullet_lists")
class _:
    def sample(rng):
        n = rng.randint(2, 6)
        return {"num_bullets": n}, {"num_bullets": n}
    def verify(r, kw):
        return len(re.findall(r"(?m)^\s*[*\-]\s+\S", r)) == kw["num_bullets"]

@_reg("detectable_format:number_highlighted_sections")
class _:
    def sample(rng):
        n = rng.randint(2, 4)
        return {"num_highlights": n}, {"num_highlights": n}
    def verify(r, kw):
        return len(re.findall(r"\*[^*\n]+\*", r)) >= kw["num_highlights"]

@_reg("detectable_format:multiple_sections")
class _:
    def sample(rng):
        n = rng.randint(2, 4); sp = rng.choice(["第", "部分", "段落"])
        return {"num_sections": n, "section_spliter": sp}, {"num_sections": n, "section_spliter": sp}
    def verify(r, kw): return r.count(kw["section_spliter"]) >= kw["num_sections"]

@_reg("detectable_format:constrained_response")
class _:
    def sample(rng): return {"options": list(L.CONSTRAINED_OPTIONS)}, {}
    def verify(r, kw): return r.strip() in kw.get("options", L.CONSTRAINED_OPTIONS)

@_reg("detectable_format:json_format")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        s = re.sub(r"^```[a-zA-Z]*\n?|```$", "", r.strip()).strip()
        try:
            json.loads(s); return True
        except Exception:
            return False

@_reg("combination:two_responses")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        parts = [p for p in r.split("******") if p.strip()]
        return len(parts) == 2

@_reg("copy:repeat_phrase")
class _:
    def sample(rng):
        p = rng.choice(L.PHRASES); n = rng.randint(2, 4)
        return {"phrase": p, "small_n": n}, {"phrase": p, "small_n": n}
    def verify(r, kw):
        units = re.split(r"[\n，。、!！?？]", r)
        hit = sum(1 for u in units if u.strip() and difflib.SequenceMatcher(None, u.strip(), kw["phrase"]).ratio() >= 0.6)
        return hit >= kw["small_n"]

@_reg("startend:quotation")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        s = r.strip()
        return s.startswith("「") and s.endswith("」")

@_reg("startend:end_checker")
class _:
    def sample(rng):
        p = rng.choice(L.END_PHRASES)
        return {"end_phrase": p}, {"end_phrase": p}
    def verify(r, kw): return r.rstrip().endswith(kw["end_phrase"])

@_reg("length_constraints:number_paragraphs")
class _:
    def sample(rng):
        n = rng.randint(2, 5)
        return {"num_paragraphs": n}, {"num_paragraphs": n}
    def verify(r, kw): return len(T.split_paragraphs(r, "***")) == kw["num_paragraphs"]

@_reg("paragraphs:paragraphs")
class _:
    def sample(rng):
        n = rng.randint(2, 5)
        return {"num_paragraphs": n}, {"num_paragraphs": n}
    def verify(r, kw): return len(T.split_paragraphs(r, "***")) == kw["num_paragraphs"]

@_reg("paragraphs:paragraphs2")
class _:
    def sample(rng):
        n = rng.randint(2, 5)
        return {"num_paragraphs": n}, {"num_paragraphs": n}
    def verify(r, kw): return len(T.split_paragraphs(r)) == kw["num_paragraphs"]

@_reg("language:response_language")
class _:
    def sample(rng): return {"language": getattr(L, "RESPONSE_LANGUAGE", "粵語")}, {}
    def verify(r, kw):
        body = re.sub(r"\s", "", r)
        return bool(body) and T.count_hanzi(r) / len(body) >= 0.5


# --------------------------------------------------------------------------- B
@_reg("length_constraints:number_words")
class _:
    def sample(rng):
        n = rng.choice([50, 80, 100, 150, 200]); rel = rng.choice(["at least", "at most", "around"])
        return {"relation": rel, "num_words": n}, {"relation": _rel_disp(rel), "num_words": n}
    def verify(r, kw): return T.relation_ok(T.count_words(r), kw["relation"], kw["num_words"])

@_reg("length_constraints:number_sentences")
class _:
    def sample(rng):
        n = rng.randint(3, 8); rel = rng.choice(["at least", "at most", "exactly"])
        return {"relation": rel, "num_sentences": n}, {"relation": _rel_disp(rel), "num_sentences": n}
    def verify(r, kw): return T.relation_ok(len(T.split_sentences(r)), kw["relation"], kw["num_sentences"])

@_reg("length_constraints:nth_paragraph_first_word")
class _:
    def sample(rng):
        n = rng.randint(2, 4); nth = rng.randint(1, n); w = rng.choice(L.EDGE_WORDS)
        return {"num_paragraphs": n, "nth_paragraph": nth, "first_word": w}, {"num_paragraphs": n, "nth_paragraph": nth, "first_word": w}
    def verify(r, kw):
        paras = T.split_paragraphs(r)
        if len(paras) != kw["num_paragraphs"]:
            return False
        return paras[kw["nth_paragraph"] - 1].startswith(kw["first_word"])

@_reg("first_word:first_word_answer")
class _:
    def sample(rng):
        w = rng.choice(L.EDGE_WORDS); return {"first_word": w}, {"first_word": w}
    def verify(r, kw): return (T.first_token(r) or "").startswith(kw["first_word"]) or r.lstrip().startswith(kw["first_word"])

@_reg("first_word:first_word_sent")
class _:
    def sample(rng):
        w = rng.choice(L.EDGE_WORDS); return {"first_word": w}, {"first_word": w}
    def verify(r, kw):
        sents = T.split_sentences(r)
        return bool(sents) and all(s.lstrip().startswith(kw["first_word"]) for s in sents)

@_reg("last_word:last_word_answer")
class _:
    def sample(rng):
        w = rng.choice(L.EDGE_WORDS); return {"last_word": w}, {"last_word": w}
    def verify(r, kw): return T.strip_terminal_punct(r).endswith(kw["last_word"])

@_reg("last_word:last_word_sent")
class _:
    def sample(rng):
        w = rng.choice(L.EDGE_WORDS); return {"last_word": w}, {"last_word": w}
    def verify(r, kw):
        sents = T.split_sentences(r)
        return bool(sents) and all(T.strip_terminal_punct(s).endswith(kw["last_word"]) for s in sents)

@_reg("keywords:start_end")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        ft, lt = T.first_token(r), T.last_token(r)
        s = r.strip()
        return bool(ft) and ft == lt and not s[-1:] in (T.FULLWIDTH_PUNCT + T.ASCII_PUNCT)

@_reg("keywords:keyword_specific_position")
class _:
    def sample(rng):
        w = rng.choice(_SINGLE_TOK_KW); n = rng.randint(1, 3); m = rng.randint(1, 3)
        return {"keyword": w, "n": n, "m": m}, {"keyword": w, "n": n, "m": m}
    def verify(r, kw):
        sents = T.split_sentences(r)
        if len(sents) < kw["n"]:
            return False
        toks = T.tokenize(sents[kw["n"] - 1])
        return len(toks) >= kw["m"] and toks[kw["m"] - 1] == kw["keyword"]

@_reg("count:count_unique")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        toks = T.tokenize(r)
        # require a non-trivial response — an empty/near-empty answer is vacuously "all unique"
        # (that degenerate pass is a reward-hacking hole under GRPO).
        return len(toks) >= 8 and len(toks) == len(set(toks))

@_reg("count:counting_composition")
class _:
    def sample(rng):
        n = rng.randint(2, 4); return {"n_sent": n}, {"n_sent": n}
    def verify(r, kw):
        n = kw["n_sent"]
        paras = T.split_paragraphs(r, "* * *")
        if not paras:
            return False
        for p in paras:
            sents = T.split_sentences(p)
            if len(sents) != n:
                return False
            for s in sents:
                if T.count_words(s) != n:
                    return False
        return True

@_reg("detectable_format:square_brackets")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        return T.count_hanzi(re.sub(r"\[[^\]]*\]", "", r)) == 0 and "[" in r

@_reg("detectable_format:bigram_wrapping")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        return "<<" in r and T.count_hanzi(re.sub(r"<<[^>]*>>", "", r)) == 0

@_reg("detectable_format:sentence_hyphens")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        return "-" in r and " " not in r.strip()

@_reg("punctuation:no_comma")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw): return not any(c in r for c in "，、,")

@_reg("punctuation:punctuation_dot")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw): return "。" not in r and "." not in r

@_reg("punctuation:punctuation_exclamation")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw): return "！" not in r and "!" not in r


# --------------------------------------------------------------------------- D
@_reg("chinese:char_count_hanzi")
class _:
    def sample(rng):
        n = rng.choice([50, 100, 150, 200]); rel = rng.choice(["at least", "at most", "around"])
        return {"relation": rel, "num_chars": n}, {"relation": _rel_disp(rel), "num_chars": n}
    def verify(r, kw): return T.relation_ok(T.count_hanzi(r), kw["relation"], kw["num_chars"])

@_reg("chinese:traditional_only")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        try:
            from opencc import OpenCC
            return OpenCC("s2t").convert(r) == r
        except Exception:
            return None

@_reg("chinese:no_english_letters")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw): return not T.has_ascii_letters(r)

@_reg("chinese:numbers_in_chinese")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw): return not T.has_arabic_digits(r)

@_reg("chinese:full_width_punctuation")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw): return not any(c in r for c in T.ASCII_PUNCT)

@_reg("chinese:specific_char_frequency")
class _:
    def sample(rng):
        c = rng.choice(L.COMMON_HANZI); n = rng.randint(2, 4); rel = rng.choice(["at least", "at most", "exactly"])
        return {"char": c, "relation": rel, "frequency": n}, {"char": c, "relation": _rel_disp(rel), "frequency": n}
    def verify(r, kw): return T.relation_ok(r.count(kw["char"]), kw["relation"], kw["frequency"])

@_reg("chinese:chengyu_count")
class _:
    def sample(rng):
        n = rng.randint(1, 3); return {"num_idioms": n}, {"num_idioms": n}
    def verify(r, kw):
        return len({c for c in L.CHENGYU_SET if c in r}) >= kw["num_idioms"]

@_reg("chinese:sentence_end_particle_punct")
class _:
    def sample(rng):
        p = rng.choice(["。", "！", "？"]); return {"punctuation": p}, {"punctuation": p}
    def verify(r, kw):
        sents = T.split_sentences(r)
        return bool(sents) and all(s.rstrip().endswith(kw["punctuation"]) for s in sents)

@_reg("cantonese:colloquial_required")
class _:
    def sample(rng):
        n = rng.randint(3, 8); return {"num_markers": n}, {"num_markers": n}
    def verify(r, kw):
        return sum(r.count(m) for m in L.CANTONESE_MARKERS) >= kw["num_markers"]

@_reg("cantonese:no_colloquial")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw): return not any(m in r for m in L.CANTONESE_MARKERS)

@_reg("cantonese:sentence_final_particle")
class _:
    def sample(rng): return {}, {}
    def verify(r, kw):
        sents = T.split_sentences(r)
        if not sents:
            return False
        for s in sents:
            body = T.strip_terminal_punct(s)
            if not body or body[-1] not in T.CANTONESE_FINAL_PARTICLES:
                return False
        return True

@_reg("cantonese:jyutping_rhyme")
class _:
    def sample(rng):
        n = rng.randint(2, 4); return {"num_rhyme_lines": n}, {"num_rhyme_lines": n}
    def verify(r, kw):
        try:
            import pycantonese
        except Exception:
            return None
        import collections
        finals = collections.Counter()
        for line in [l for l in r.splitlines() if l.strip()]:
            ch = T.strip_terminal_punct(line)[-1:]
            if not ch:
                continue
            jp = pycantonese.characters_to_jyutping(ch)
            if jp and jp[0][1]:
                syl = jp[0][1]
                m = re.match(r"[bpmfdtnlgknghwjcsz]*(.*?)[1-6]$", syl)
                if m:
                    finals[m.group(1)] += 1
        return bool(finals) and max(finals.values()) >= kw["num_rhyme_lines"]


_JYUTPING_QUOTE_RE = re.compile(r'「([^」]+)」')
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
_JYUTPING_PHONE_RE = re.compile(r'\b[a-z]+[1-6]\b')


@_reg("cantonese:jyutping_phone_count")
class _:
    def sample(rng, ctx=None):
        """ctx = {"base_prompt": str} when called from build_if for 粵拼 prompts."""
        if ctx:
            m = _JYUTPING_QUOTE_RE.search(ctx.get("base_prompt", ""))
            if m:
                n = len(_CJK_RE.findall(m.group(1)))
                if n > 0:
                    return {"num_phones": n}, {"num_phones": n}
        n = rng.randint(2, 8)
        return {"num_phones": n}, {"num_phones": n}

    def verify(r, kw):
        phones = _JYUTPING_PHONE_RE.findall(r.lower())
        return len(phones) == kw["num_phones"]
