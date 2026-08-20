"""Knowledge state definitions and helpers."""

from __future__ import annotations

from typing import List


STATE_ORDER: List[str] = [
    "unaware",
    "struggling",
    "partial_understanding",
    "knows_well",
]


def is_higher_state(state_a: str, state_b: str) -> bool:
    """Return True if state_a ranks higher than state_b."""
    return STATE_ORDER.index(state_a) > STATE_ORDER.index(state_b)

