"""LLM call logging utilities (ported from SimulatorArena utils).

The ``Settings`` dataclass below used to live in ``simulation/core/config.py``
but had only this module as a consumer, so it was inlined during the Tier-1
cleanup. ``Settings.from_config()`` reads ``simulation/config.json``.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


def _load_settings_dict(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from ``simulation/config.json``."""

    log_llm_calls: bool = False
    log_llm_path: str = "logs/llm_calls.jsonl"
    print_llm_calls: bool = False

    @staticmethod
    def from_config(path: str = "simulation/config.json") -> "Settings":
        data = _load_settings_dict(path)
        return Settings(
            log_llm_calls=bool(data.get("log_llm_calls", False)),
            log_llm_path=str(data.get("log_llm_path", "logs/llm_calls.jsonl")),
            print_llm_calls=bool(data.get("print_llm_calls", False)),
        )


_LOG_LOCK = asyncio.Lock()
_LOG_PATH_CACHED: Optional[str] = None
_LOG_FH: Optional[Any] = None


def get_log_path(settings: Optional[Settings] = None) -> Optional[str]:
    settings = settings or Settings.from_config()
    if not settings.log_llm_calls:
        return None
    global _LOG_PATH_CACHED
    if _LOG_PATH_CACHED:
        return _LOG_PATH_CACHED
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = settings.log_llm_path
    if "{timestamp}" in path:
        _LOG_PATH_CACHED = path.replace("{timestamp}", ts)
        return _LOG_PATH_CACHED
    root, ext = os.path.splitext(path)
    _LOG_PATH_CACHED = f"{root}_{ts}{ext or '.jsonl'}"
    return _LOG_PATH_CACHED


def should_print_calls(settings: Optional[Settings] = None) -> bool:
    settings = settings or Settings.from_config()
    return settings.print_llm_calls


async def log_llm_calls(entries: List[Dict[str, Any]], settings: Optional[Settings] = None) -> None:
    """Append LLM call logs as JSONL entries."""
    log_path = get_log_path(settings)
    if not log_path or not entries:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    async with _LOG_LOCK:
        global _LOG_FH
        if _LOG_FH is None:
            _LOG_FH = open(log_path, "a", encoding="utf-8")
        for entry in entries:
            _LOG_FH.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _LOG_FH.flush()


def print_llm_calls(entries: List[Dict[str, Any]], settings: Optional[Settings] = None) -> None:
    if not entries or not should_print_calls(settings):
        return
    for entry in entries:
        print("\n=== LLM CALL ===")
        print(f"model: {entry.get('model_name')}")
        print(f"temperature: {entry.get('temperature')}, max_tokens: {entry.get('max_tokens')}, n: {entry.get('n')}")
        print("--- system_prompt ---")
        print(entry.get("system_prompt", ""))
        print("--- user_prompt ---")
        print(entry.get("user_prompt", ""))
        print("--- output ---")
        print(entry.get("output"))


def build_log_entry(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    messages: List[Dict[str, str]],
    output: Any,
    temperature: float,
    max_tokens: int,
    n: int,
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": model_name,
        "user_level": os.environ.get("SIM_USER_LEVEL", ""),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": messages,
        "output": output,
    }

