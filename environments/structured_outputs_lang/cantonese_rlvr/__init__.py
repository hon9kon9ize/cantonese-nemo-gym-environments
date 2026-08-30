"""Cantonese RLVR verifiers (vendored from the sibling nemo-instruction-following
project) for the infinite_rl NeMo-RL environment.

Pure-python third-party deps that are NOT in the read-only NeMo-RL container
(`jieba`, `toml`) are vendored under ``_vendor/`` and put on ``sys.path`` here, so
the eager ``import jieba`` / ``import toml`` in text_utils/so_formats resolve to
the vendored copies with nothing to pip-install — the same approach aux_rewards
uses for the vendored ``cantofilter`` (see README "dependency-light pure python").
Both are ``py3-none-any`` (no compiled extension), so there is no ABI/Python-version
coupling. ``jsonschema``/``yaml`` are already present in the container; the optional
``pycantonese``/``opencc`` (rhyme / 簡繁 verifiers) stay lazy ``try/except`` in
registry.py and simply skip those checks when absent (like ``pycld2``).
"""
import os as _os
import sys as _sys

_VENDOR = _os.path.join(_os.path.dirname(__file__), "_vendor")
if _VENDOR not in _sys.path:
    _sys.path.insert(0, _VENDOR)
