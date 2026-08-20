"""IU graph extraction from question + reference answer."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

from ..core.prompts import load_prompt


def _slice_json_array(s: str, start: int) -> Optional[str]:
    """Return s[start : end+1] for a JSON array starting at start (must be '[')."""
    if start >= len(s) or s[start] != "[":
        return None
    stack: List[str] = ["["]
    in_str = False
    esc = False
    for j in range(start + 1, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "[":
            stack.append("[")
        elif c == "{":
            stack.append("{")
        elif c == "]":
            if stack and stack[-1] == "[":
                stack.pop()
                if not stack:
                    return s[start : j + 1]
        elif c == "}":
            if stack and stack[-1] == "{":
                stack.pop()
    return None


def _recover_truncated_iu_graph(text: str) -> Dict[str, Any]:
    """
    If the model hits max_tokens mid-JSON, we may have valid `nodes` but broken `edges`.
    Extract a balanced `nodes` array and default missing fields.
    """
    text = _coerce_model_text_to_json_string(text)
    if not text:
        return {}
    idx = text.find('"nodes"')
    if idx < 0:
        return {}
    bracket = text.find("[", idx)
    if bracket < 0:
        return {}
    nodes_json = _slice_json_array(text, bracket)
    if not nodes_json:
        return {}
    try:
        nodes: List[Any] = json.loads(nodes_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(nodes, list) or not nodes:
        return {}

    knowledge_areas: List[str] = []
    ka_key = '"knowledge_areas"'
    ka_pos = text.find(ka_key)
    if ka_pos >= 0 and ka_pos < idx:
        ka_bracket = text.find("[", ka_pos)
        if ka_bracket >= 0:
            ka_slice = _slice_json_array(text, ka_bracket)
            if ka_slice:
                try:
                    loaded = json.loads(ka_slice)
                    if isinstance(loaded, list):
                        knowledge_areas = [str(x) for x in loaded if isinstance(x, str)]
                except json.JSONDecodeError:
                    pass

    return {
        "knowledge_areas": knowledge_areas,
        "nodes": nodes,
        "edges": [],
    }


def _coerce_model_text_to_json_string(text: str) -> str:
    """Normalize odd model outputs (e.g. Python repr of a one-element list of JSON string)."""
    text = (text or "").strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            val = ast.literal_eval(text)
            if isinstance(val, list) and len(val) == 1:
                el = val[0]
                if isinstance(el, str) and el.strip().startswith("{"):
                    return el.strip()
        except (ValueError, SyntaxError, TypeError):
            pass
    return text


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = _coerce_model_text_to_json_string(text)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                return first
            if isinstance(first, str):
                try:
                    return json.loads(first)
                except json.JSONDecodeError:
                    pass
    except json.JSONDecodeError:
        recovered = _recover_truncated_iu_graph(text)
        if recovered.get("nodes"):
            return recovered
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                return first
            if isinstance(first, str):
                try:
                    return json.loads(first)
                except json.JSONDecodeError:
                    pass
    except (ValueError, SyntaxError):
        pass
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        recovered = _recover_truncated_iu_graph(text)
        return recovered if recovered.get("nodes") else {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        recovered = _recover_truncated_iu_graph(text)
        return recovered if recovered.get("nodes") else {}


def _fill_iu_extraction_prompt(
    template: str,
    question: str,
    answer: str,
    field: str = "",
    specific_field: str = "",
) -> str:
    """Substitute question/answer/domain without str.format (answers may contain { } braces)."""
    q = question or ""
    a = (answer or "").strip()
    if not a:
        a = (
            "(No reference answer was provided for this row. Extract IUs from the question and "
            "stated goal only; prefer slightly broader nodes if the target answer is ambiguous.)"
        )
    return (
        template
        .replace("{question}", q)
        .replace("{answer}", a)
        .replace("{field}", field or "Not specified")
        .replace("{specific_field}", specific_field or "Not specified")
    )


async def extract_iu_graph(
    *,
    question: str,
    answer: str,
    model_client: Any,
    field: str = "",
    specific_field: str = "",
    max_tokens: int = 1200,
    show_progress: bool = False,
    prompt_path: str = "simulation/prompts/knowledge_state/iu_graph_extraction.txt",
) -> Dict[str, Any]:
    template = load_prompt(prompt_path)
    prompt = _fill_iu_extraction_prompt(template, question, answer, field, specific_field)

    async def _call_once(prompt_text: str, temperature: float, mt: int) -> str:
        responses = await model_client.generate_responses(
            [[{"role": "system", "content": "You are an expert knowledge graph extractor."},
              {"role": "user", "content": prompt_text}]],
            temperature=temperature,
            max_tokens=mt,
            n=1,
            show_progress=show_progress,
            json_mode=True,
        )
        return responses[0][0] if responses and responses[0] else ""

    def _token_escalation(mt: int) -> list[int]:
        tiers = [mt, min(max(mt * 2, 8192), 32768), 32768]
        out: list[int] = []
        for x in tiers:
            if x > 0 and x not in out:
                out.append(x)
        return out

    token_budgets = _token_escalation(max_tokens)

    raw = await _call_once(prompt, 0.2, max_tokens)
    parsed = _extract_json_object(raw)
    if parsed and parsed.get("nodes"):
        return parsed

    # Retry once with a stricter JSON-only instruction and lower temperature.
    retry_prompt = prompt + "\n\nReturn only valid JSON. Do not include any extra text."
    for mt in token_budgets:
        raw_retry = await _call_once(retry_prompt, 0.0, mt)
        parsed_retry = _extract_json_object(raw_retry)
        if parsed_retry and parsed_retry.get("nodes"):
            return parsed_retry

    # Final retry with tighter output constraints to avoid truncation.
    tight_prompt = (
        prompt
        + "\n\nReturn only valid JSON. Do not include any extra text."
        + "\nLimit to 12 nodes. Keep descriptions to 1 sentence each."
        + "\nUse only essential prerequisite edges."
    )
    last_raw = raw
    for mt in token_budgets:
        raw_tight = await _call_once(tight_prompt, 0.0, mt)
        last_raw = raw_tight or last_raw
        parsed_tight = _extract_json_object(raw_tight)
        if parsed_tight and parsed_tight.get("nodes"):
            return parsed_tight

    snippet = repr((last_raw or "")[:240])
    raise RuntimeError(
        "IU graph extraction failed: empty or unparsable response "
        f"(snippet: {snippet}). "
        "Try --iu_max_tokens 32768 or a smaller IU graph prompt."
    )

