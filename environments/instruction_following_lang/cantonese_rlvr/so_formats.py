"""Serialize / parse a python dict across the 5 output formats used by the dataset:
JSON, YAML, XML, TOML, CSV.

Type-preserving formats (json/yaml/toml) support exact-match verification; xml/csv are
used only where a structural (well-formed + keys present) check is sufficient.
"""
from __future__ import annotations
import json
import io
import csv
import xml.etree.ElementTree as ET

import yaml
import toml

DISPLAY = {"json": "JSON", "yaml": "YAML", "xml": "XML", "toml": "TOML", "csv": "CSV"}
TYPE_PRESERVING = ("json", "yaml", "toml")


# --------------------------------------------------------------------------- serialize
def serialize(data: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(data, ensure_ascii=False, indent=2)
    if fmt == "yaml":
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False).strip()
    if fmt == "toml":
        return toml.dumps(data).strip()
    if fmt == "xml":
        return _to_xml(data)
    if fmt == "csv":
        return _to_csv(data)
    raise ValueError(fmt)


def _to_xml(data: dict, root="output") -> str:
    el = ET.Element(root)
    _build_xml(el, data)
    return ET.tostring(el, encoding="unicode")


def _build_xml(parent, obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = ET.SubElement(parent, str(k))
            _build_xml(child, v)
    elif isinstance(obj, list):
        for item in obj:
            child = ET.SubElement(parent, "item")
            _build_xml(child, item)
    else:
        parent.text = "" if obj is None else str(obj)


def _to_csv(data: dict) -> str:
    # flat scalar dict -> header + one row
    buf = io.StringIO()
    w = csv.writer(buf)
    keys = list(data.keys())
    w.writerow(keys)
    w.writerow(["" if data[k] is None else data[k] for k in keys])
    return buf.getvalue().strip()


# --------------------------------------------------------------------------- parse
def parse(text: str, fmt: str):
    text = text.strip()
    # tolerate markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    if fmt == "json":
        return json.loads(text)
    if fmt == "yaml":
        return yaml.safe_load(text)
    if fmt == "toml":
        return toml.loads(text)
    if fmt == "xml":
        return _xml_keys(text)
    if fmt == "csv":
        return _csv_cols(text)
    raise ValueError(fmt)


def _xml_keys(text: str) -> set:
    root = ET.fromstring(text)
    return {child.tag for child in root}


def _csv_cols(text: str) -> set:
    reader = csv.reader(io.StringIO(text))
    header = next(reader, [])
    return set(header)
