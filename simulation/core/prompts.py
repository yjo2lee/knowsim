"""Prompt loading and formatting helpers."""

from __future__ import annotations

from typing import Dict


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(template: str, values: Dict[str, str]) -> str:
    return template.format(**values)

