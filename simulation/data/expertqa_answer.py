"""Pick reference answer text from ExpertQA JSONL `answers` object."""

from __future__ import annotations

from typing import Dict


def select_expertqa_answer(
    answers: Dict[str, Dict[str, object]],
    preferred_variant: str,
) -> tuple[str, str, str]:
    """Return (answer_text, variant_key, source_field)."""
    if not answers:
        return "", "", ""

    variants: list[str] = []
    if preferred_variant and preferred_variant in answers:
        variants = [preferred_variant]
    if not variants:
        variants = list(answers.keys())

    def _pick(variant: str) -> tuple[str, str]:
        entry = answers.get(variant, {}) or {}
        revised = entry.get("revised_answer_string") or ""
        if revised:
            return str(revised), "revised_answer_string"
        answer = entry.get("answer_string") or ""
        if answer:
            return str(answer), "answer_string"
        return "", ""

    for variant in variants:
        text, source = _pick(variant)
        if text:
            return text, variant, source

    for variant in answers.keys():
        if variant in variants:
            continue
        text, source = _pick(variant)
        if text:
            return text, variant, source

    return "", "", ""
