"""Verifiable-reward functions for the structured-output tasks.

- format_translation / direct_extraction : exact match to the given data + JSON-Schema
      validity (both present the data in the prompt; type-preserving json/yaml/toml).
- schema_only (json/yaml/toml)            : JSON-Schema validity (generation — no single answer).
- schema_only (xml/csv)                   : well-formed + required keys/columns present.
"""
from __future__ import annotations

import jsonschema

from . import so_formats as F


def _deep_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def _required_keys(schema) -> set:
    return set(schema.get("required", []))


def _find_title(obj):
    """First string `title` value anywhere in the (possibly nested) object, else None."""
    if isinstance(obj, dict):
        if isinstance(obj.get("title"), str):
            return obj["title"]
        for v in obj.values():
            t = _find_title(v)
            if t is not None:
                return t
    return None


def verify(problem_type: str, target_fmt: str, output: str, schema: dict, expected: dict) -> bool:
    try:
        parsed = F.parse(output, target_fmt)
    except Exception:
        return False

    if problem_type in ("format_translation", "direct_extraction"):
        if not _deep_equal(parsed, expected):
            return False
        try:
            jsonschema.validate(parsed, schema)
        except Exception:
            return False
        return True

    # schema_only_generation
    if target_fmt in F.TYPE_PRESERVING:
        try:
            jsonschema.validate(parsed, schema)
        except Exception:
            return False
        # ground the generation: the prompt names the article title, so if the data carries a
        # title the output must reproduce it (kills the "any schema-valid junk stub" hack).
        exp_title = _find_title(expected)
        if exp_title is not None and _find_title(parsed) != exp_title:
            return False
        return True
    if target_fmt == "xml":
        return _required_keys(schema) <= parsed  # parsed is a set of top-level tags
    if target_fmt == "csv":
        return _required_keys(schema) <= parsed  # parsed is a set of columns
    return False
