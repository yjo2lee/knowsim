"""Interaction-style user profile loader and formatter."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def load_interaction_profiles(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_interaction_profile(features: List[Dict[str, Any]], length_text: str) -> str:
    lines = ["## Interaction Style", f"- Length of User Message: The user's query/response is always {length_text}."]
    for feature in features:
        lines.append(f"- {feature['Feature Name']}: {feature['Feature Question Answer']}")
    return "\n".join(lines)

