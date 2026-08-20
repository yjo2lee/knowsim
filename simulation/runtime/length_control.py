"""Length control helpers (ported from user_simulation_math_tutoring)."""

from __future__ import annotations


def round_down_to_nearest_5(n: int) -> int:
    return max(1, (n // 5) * 5)


def round_up_to_nearest_5(n: int) -> int:
    if n <= 0:
        return 1
    return ((n + 4) // 5) * 5


def count_words(text: str) -> int:
    return len([word for word in text.split() if word])

