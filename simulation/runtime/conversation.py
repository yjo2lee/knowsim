"""Conversation loop orchestration (math tutoring only)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

from .assistant_strategies import build_assistant_system_prompt
from ..core.feature_flags import get_flags
from ..knowledge.update_v2 import compute_overload_threshold, get_absorption_preset


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _safe_format(template: str, **kwargs: Any) -> str:
    return template.format_map(_SafeFormatDict(kwargs))


def _format_knowledge_state(knowledge_state: Optional[Dict[str, Any]]) -> str:
    if not knowledge_state:
        return "{}"
    compact = {
        concept: {"state": info.get("state")}
        for concept, info in knowledge_state.items()
    }
    return json.dumps(compact, indent=2)


def _normalize_state(state: str) -> str:
    if not state:
        return "unaware"
    lowered = state.strip()
    if lowered in {"unknown_unknown", "not_introduced"}:
        return "unaware"
    return lowered


def _get_unaware_concepts(knowledge_state: Optional[Dict[str, Any]]) -> List[str]:
    if not knowledge_state:
        return []
    return [
        concept for concept, info in knowledge_state.items()
        if _normalize_state(info.get("state", "")) == "unaware"
    ]


def _get_concepts_by_state(knowledge_state: Optional[Dict[str, Any]], target_state: str) -> List[str]:
    if not knowledge_state:
        return []
    return [
        concept for concept, info in knowledge_state.items()
        if _normalize_state(info.get("state", "")) == target_state
    ]


def _get_askable_concepts(
    knowledge_state: Optional[Dict[str, Any]],
    explained_concepts: Optional[List[str]] = None,
) -> List[str]:
    """Return concepts the user simulator can meaningfully ask about.

    - struggling: always askable (user is confused, needs help)
    - partial_understanding: askable only if not yet explained this session
    - unaware / knows_well: never askable
    """
    if not knowledge_state:
        return []
    explained_set = set(explained_concepts or [])
    askable = []
    for concept, info in knowledge_state.items():
        state = _normalize_state(info.get("state", ""))
        if state == "struggling":
            askable.append(concept)
        elif state == "partial_understanding" and concept not in explained_set:
            askable.append(concept)
    return askable


def _get_last_assistant_message(assistant_messages: List[Dict[str, str]]) -> str:
    for message in reversed(assistant_messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def _get_assistant_history_label(task: str) -> str:
    return "AI Tutor" if task == "math" else "AI Assistant"


def _get_articulation_mode(data: Dict[str, Any]) -> str:
    return data.get("articulation_mode") or "Vague"


def _get_overall_density(data: Dict[str, Any]) -> float:
    return data.get("overall_density", 0.5) if data.get("overall_density") is not None else 0.5


_UNIVERSAL_ENGAGEMENT_GUIDANCE = (
    "Match your engagement to your knowledge state honestly. Be direct and "
    "confident on concepts you know well, hedge on concepts you partially "
    "understand, and express specific confusion on concepts you're struggling "
    "with. Don't pretend knowledge you don't have, and don't refuse to attempt "
    "concepts where you have enough background."
)


def _get_articulation_guidance(data: Dict[str, Any]) -> str:
    # pilot2_20260514: when universal_engagement_guidance flag is set, return a
    # single fixed text regardless of density. Removes the level-conditioned
    # behavior cue so engagement is driven entirely by IU state + Rule 1-5.
    if get_flags().user_sim.universal_engagement_guidance:
        return _UNIVERSAL_ENGAGEMENT_GUIDANCE
    density = _get_overall_density(data)
    if density >= 0.6:
        return "\n".join(
            [
                "- Use direct, explicit questions.",
                "- Ask for precise definitions or steps.",
                "- Be confident and specific about what you know.",
            ]
        )
    if density >= 0.3:
        return "\n".join(
            [
                "- Use broad, clarifying questions.",
                "- Summarize uncertainty briefly.",
                "- Keep a neutral, exploratory tone.",
            ]
        )
    return "\n".join(
        [
            "- Keep messages short and compliant.",
            "- Avoid over-asking or challenging the tutor.",
            "- Use brief acknowledgments and simple questions only.",
            "- Do not argue; accept guidance and move on.",
            "- If you try to work through a concept you're struggling with, it's natural to get it wrong — take a genuine but imperfect guess rather than just asking the tutor to do it for you.",
        ]
    )


def _get_density_label(data: Dict[str, Any]) -> str:
    density = _get_overall_density(data)
    if density >= 0.6:
        return "HIGH"
    if density >= 0.3:
        return "MEDIUM"
    return "LOW"


def _get_density_percent(data: Dict[str, Any]) -> int:
    density = _get_overall_density(data)
    return int(round(density * 100))


def _get_gap_sum(data: Dict[str, Any]) -> float:
    return data.get("gap_sum", 0.0) if data.get("gap_sum") is not None else 0.0


def _get_query_tone(data: Dict[str, Any]) -> str:
    return data.get("query_tone") or "Vague"


def _state_score(state: str) -> int:
    mapping = {
        "unaware": 0,
        "struggling": 1,
        "partial_understanding": 2,
        "knows_well": 3,
    }
    return mapping.get(_normalize_state(state), 0)


def _compute_structured_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    initial_state = data.get("initial_knowledge_state") or {}
    final_state = data.get("knowledge_state") or {}

    # Max possible gain: denominator unchanged (based on initial state headroom)
    gain_den = 0.0
    for concept, init_info in initial_state.items():
        init_state = _normalize_state((init_info or {}).get("state", "unaware"))
        if _state_score(init_state) >= 3:
            continue
        gain_den += max(0, 3 - _state_score(init_state))

    # Absorbed-only knowledge gain: accumulated per-turn, only for IUs not cut off by overload
    absorbed_gain_total = float(data.get("absorbed_gain_total", 0.0))
    knowledge_gain = absorbed_gain_total
    normalized_knowledge_gain = (absorbed_gain_total / gain_den) if gain_den > 0 else 0.0

    # Alternative NKG: raw state progress / (3 × N).
    # Σ over IUs of max(0, final_idx - init_idx), normalized by 3 × |IUs|.
    # Cross-session comparable (denom depends only on graph size, not init KS),
    # symmetric (no W_STATE weighting in numerator or denominator).
    n_ius_total = len(initial_state)
    raw_state_progress = 0
    for concept, init_info in initial_state.items():
        init_idx = _state_score(_normalize_state((init_info or {}).get("state", "unaware")))
        final_info = final_state.get(concept) or {}
        final_idx = _state_score(_normalize_state((final_info or {}).get("state", "unaware")))
        d = final_idx - init_idx
        if d > 0:
            raw_state_progress += d
    normalized_knowledge_gain_3N = (
        float(raw_state_progress) / (3.0 * n_ius_total)
    ) if n_ius_total > 0 else 0.0

    ic_num = float(data.get("ic_numerator_total", 0.0))
    ic_den = float(data.get("ic_denominator_total", 0.0))
    information_calibration = (ic_num / ic_den) if ic_den > 0 else 0.0

    # Volume-based PO: average of per-turn max(0, load-threshold)/load values
    turn_metrics_history = data.get("turn_metrics_history", []) or []
    po_vals = [
        float(t["perceived_overload_turn"])
        for t in turn_metrics_history
        if t.get("perceived_overload_turn") is not None
    ]
    perceived_overload = (sum(po_vals) / len(po_vals)) if po_vals else 0.0
    new_ius_total = float(data.get("new_ius_total", 0.0))
    disallowed_total = float(data.get("disallowed_total", 0.0))
    zpd_size_total = float(data.get("zpd_size_total", 0.0))

    # F1-based IC: precision = current IC, recall = coverage of ZPD
    ic_precision = information_calibration  # |explained ∩ ZPD| / |explained|
    ic_recall = (ic_num / zpd_size_total) if zpd_size_total > 0 else 1.0
    ic_f1 = (
        (2 * ic_precision * ic_recall / (ic_precision + ic_recall))
        if (ic_precision + ic_recall) > 0 else 0.0
    )

    # Capacity-style PO (additional axis): min(1, eff/threshold). Computed
    # from the per-turn field if present; otherwise derived on the fly from
    # effective_load and overload_threshold so legacy run JSONs (no stored
    # capacity term) can still report this metric.
    po_cap_vals: List[float] = []
    for t in turn_metrics_history:
        if not isinstance(t, dict):
            continue
        v = t.get("perceived_overload_capacity_turn")
        if v is None:
            eff = t.get("effective_load")
            thr = t.get("overload_threshold")
            if eff is None or thr is None or float(thr) <= 0:
                continue
            v = min(1.0, float(eff) / float(thr))
        po_cap_vals.append(float(v))
    perceived_overload_capacity = (
        sum(po_cap_vals) / len(po_cap_vals) if po_cap_vals else 0.0
    )

    # Absorbed-IC (additional axis): IC numerator restricted to mentioned ∧
    # in-ZPD ∧ upward-transitioned. Per-turn field absorbed_ic_numerator is
    # emitted by fresh runs of update_v2.py; legacy runs use the proxy
    # min(ic_numerator, n_upward_transitions) which slightly overcounts (a
    # non-ZPD-but-upward IU could not be in absorbed_ic by definition, but
    # the proxy can't tell those apart from in-ZPD-and-upward IUs).
    absorbed_ic_num_total = 0
    legacy_proxy_used = False
    for t in turn_metrics_history:
        if not isinstance(t, dict):
            continue
        v = t.get("absorbed_ic_numerator")
        if v is None:
            ic_n = t.get("ic_numerator")
            n_up = t.get("n_upward_transitions")
            if ic_n is None or n_up is None:
                continue
            v = min(int(ic_n or 0), int(n_up or 0))
            legacy_proxy_used = True
        absorbed_ic_num_total += int(v or 0)
    absorbed_ic_precision = (
        absorbed_ic_num_total / ic_den if ic_den > 0 else 0.0
    )
    absorbed_ic_recall = (
        absorbed_ic_num_total / zpd_size_total if zpd_size_total > 0 else 1.0
    )
    absorbed_ic_f1 = (
        (2 * absorbed_ic_precision * absorbed_ic_recall
         / (absorbed_ic_precision + absorbed_ic_recall))
        if (absorbed_ic_precision + absorbed_ic_recall) > 0 else 0.0
    )

    engagement_history = data.get("engagement_score_history", []) or []
    if engagement_history:
        engagement = sum((entry.get("engagement", 0.0) for entry in engagement_history)) / len(engagement_history)
    else:
        engagement = 0.0

    series_map = {
        "engagement": [],
        "information_calibration_turn": [],
        "perceived_overload_turn": [],
        "familiarity_term": [],
        "novelty_term": [],
    }
    for entry in turn_metrics_history:
        if not isinstance(entry, dict):
            continue
        for key in series_map.keys():
            value = entry.get(key)
            if value is None:
                continue
            try:
                series_map[key].append(float(value))
            except (TypeError, ValueError):
                continue

    def _mean_var(values: List[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return mean, var

    turn_stats: Dict[str, Any] = {}
    for key, values in series_map.items():
        mean, var = _mean_var(values)
        turn_stats[f"{key}_mean"] = mean
        turn_stats[f"{key}_var"] = var

    overload_inverse = max(0.0, 1.0 - perceived_overload)
    # "Learning experience" is a proxy aggregate until subjective ratings are available.
    learning_experience = (overload_inverse + information_calibration + engagement) / 3.0

    metrics = {
        "knowledge_gain": knowledge_gain,
        "normalized_knowledge_gain": normalized_knowledge_gain,
        "normalized_knowledge_gain_3N": normalized_knowledge_gain_3N,
        "raw_state_progress": raw_state_progress,
        "information_calibration": information_calibration,
        "perceived_overload": perceived_overload,
        "engagement": engagement,
        "learning_experience": learning_experience,
        "overload_inverse": overload_inverse,
        "ic_numerator_total": ic_num,
        "ic_denominator_total": ic_den,
        "zpd_size_total": zpd_size_total,
        "ic_precision": ic_precision,
        "ic_recall": ic_recall,
        "ic_f1": ic_f1,
        "perceived_overload_capacity": perceived_overload_capacity,
        "absorbed_ic_numerator_total": absorbed_ic_num_total,
        "absorbed_ic_precision": absorbed_ic_precision,
        "absorbed_ic_recall": absorbed_ic_recall,
        "absorbed_ic_f1": absorbed_ic_f1,
        "absorbed_ic_uses_legacy_proxy": legacy_proxy_used,
        "new_ius_total": new_ius_total,
        "disallowed_total": disallowed_total,
    }
    metrics.update(turn_stats)
    return metrics


def _is_suspicious_user_output(raw: str) -> Optional[str]:
    """Heuristic for malformed user-simulator output.

    With JSON-schema enforcement these conditions should be unreachable, but
    we keep this as a defense in depth — older SDK versions, partial schema
    support, or content-policy escapes can still slip through. Returns a
    short reason string when output is suspicious, ``None`` otherwise.
    """
    if not raw:
        return None
    s = raw
    # 1. Runaway repetition of the same XML opener (the bug we hit pre-migration).
    if s.count("<thought>") > 5:
        return f"<thought> opener repeated {s.count('<thought>')} times"
    # 2. Wildly unbalanced JSON braces, suggesting truncation or a stuck stream.
    if s.count("{") - s.count("}") > 3 or s.count("}") - s.count("{") > 3:
        return f"unbalanced JSON braces (open={s.count('{')}, close={s.count('}')})"
    # 3. Suspiciously long output for a user-simulator turn (>16 KB).
    if len(s) > 16_000:
        return f"output is {len(s)} chars long (expected <2 KB)"
    return None


def _try_parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON-object extraction from a model response.

    Tolerates: bare JSON, ```json fenced blocks, leading/trailing text, and
    minor structural noise. Returns None if no parseable object found.
    """
    s = (raw or "").strip()
    if not s:
        return None
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    else:
        # Look for the first balanced-looking {...} block. This is a best-effort
        # match, not a full JSON parser; sufficient for well-formed model output.
        m = re.search(r"\{.*\}", s, re.DOTALL)
        candidate = m.group(0) if m else None
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_user_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the user simulator's structured JSON output.

    Expected schema (after the JSON-format migration):
        {
          "thought": "...",                   # private reasoning, optional consumer
          "message": "...",                   # visible reply to the assistant
          "attempt_verify": <bool, optional>, # default False
          "terminate":      <bool, optional>, # default False
        }

    Returns a normalized dict with always-present keys ``message``,
    ``attempt_verify``, ``terminate``, ``thought`` — or ``None`` if the input
    is not a JSON object (in which case the legacy XML-tag extractors are
    used as a fallback).
    """
    obj = _try_parse_json_object(raw)
    if obj is None:
        return None
    return {
        "message": str(obj.get("message", "")).strip(),
        "attempt_verify": bool(obj.get("attempt_verify", False)),
        "terminate": bool(obj.get("terminate", False)),
        "thought": obj.get("thought") or None,
    }


def _extract_attempt_verify_marker(text: str) -> tuple[str, bool]:
    """Strip the <attempt_verify>true</attempt_verify> sibling tag and set the flag.

    Legacy XML-tag fallback for prompts that pre-date the JSON migration.

    Sibling of <message>, emitted when the user attempted a derivation and wants
    verification (Rule 3 in the structured simulator prompt).
    """
    if not text:
        return text, False
    pattern = re.compile(r"<attempt_verify>\s*true\s*</attempt_verify>", re.IGNORECASE)
    if not pattern.search(text):
        return text, False
    cleaned = pattern.sub("", text)
    return cleaned.strip(), True


def _extract_terminate_marker(text: str) -> tuple[str, bool]:
    """Strip the <terminate>true</terminate> sibling tag and set the flag."""
    if not text:
        return text, False
    pattern = re.compile(r"<terminate>\s*true\s*</terminate>", re.IGNORECASE)
    if not pattern.search(text):
        return text, False
    cleaned = pattern.sub("", text)
    return cleaned.strip(), True


def _get_stop_hint(stop_reason: str) -> str:
    """Return a directive injected into the user prompt to produce a natural exit message."""
    if stop_reason == "cognitive_overload":
        return (
            "IMPORTANT: You are overwhelmed — the explanations have been too dense "
            "and you can't keep up. In your JSON response, put one short "
            "sentence expressing that you're confused or lost in the "
            "\"message\" field, and set \"terminate\": true."
        )
    return ""


def _check_termination(data: dict, min_warmup: int = 3):
    """Check the two theory-grounded termination triggers (after warm-up).

    Returns (stop_reason, stop_details) or None.
    Mastery is handled separately via the existing full_comprehension check.
    """
    if data.get("turns", 0) < min_warmup:
        return None
    tm = data.get("turn_metrics_history", [])
    recent = tm[-3:]
    if len(recent) < 3:
        return None
    # Combined trigger: overload persists (≥2 of last 3 turns) AND no progress (≤1 upward transition)
    total_up = sum(t.get("n_upward_transitions", 0) for t in recent)
    overload_count = sum(1 for t in recent if t.get("overload", False))
    if overload_count >= 2 and total_up <= 1:
        return ("cognitive_overload", {"type": "cognitive_overload", "turn_number": data["turns"], "overloaded_turns": overload_count, "recent_transitions": total_up})
    return None

def _effective_allow_terminate(pipeline: str) -> bool:
    """Resolve ``allow_terminate`` per pipeline ('baseline' or 'structured'),
    falling back to the global ``allow_terminate`` when the pipeline-specific
    override is unset (``None``).

    See ``simulation/core/feature_flags.py:UserSimulatorFlags`` for the full
    policy. Added 2026-05-17 to let new_canonical adopt different terminate
    policies per pipeline: baselines respect terminate (May-13-style) while
    the structured pipeline ignores it (new_pilot2 / new_pilot5 style).
    """
    flags = get_flags().user_sim
    if pipeline == "baseline":
        override = flags.baseline_allow_terminate
    elif pipeline == "structured":
        override = flags.structured_allow_terminate
    else:
        override = None
    return flags.allow_terminate if override is None else override


def _is_terminate_signal(text: str) -> bool:
    if not text:
        return False
    lowered = text.strip().lower()
    if lowered in {"terminate: true", "terminate:true", "terminate: ture"}:
        return True
    if lowered == "terminate conversation":
        return True
    return "terminate" in lowered and "true" in lowered


def _split_thought_xml_block(raw: str) -> Tuple[Optional[str], str]:
    """If a balanced <thought>...</thought> exists, return (inner, tail_after). Else (None, raw)."""
    lower = raw.lower()
    open_tag = "<thought>"
    close_tag = "</thought>"
    start_open = lower.find(open_tag)
    if start_open < 0:
        return None, raw
    start_inner = start_open + len(open_tag)
    start_close = lower.find(close_tag, start_inner)
    if start_close < 0:
        return None, raw
    inner = raw[start_inner:start_close].strip()
    tail = raw[start_close + len(close_tag) :].strip()
    return inner, tail


def _extract_initial_query_block(raw: str) -> Optional[str]:
    lower = raw.lower()
    if "<initial_query>" not in lower or "</initial_query>" not in lower:
        return None
    try:
        start = lower.index("<initial_query>") + len("<initial_query>")
        end = lower.index("</initial_query>")
        return raw[start:end].strip()
    except Exception:
        return None


def _extract_message_block(raw: str) -> str | None:
    """Extract content from <message>...</message> tags if present."""
    lower = raw.lower()
    if "<message>" not in lower or "</message>" not in lower:
        return None
    try:
        start = lower.index("<message>") + len("<message>")
        end = lower.index("</message>")
        return raw[start:end].strip() or None
    except Exception:
        return None


def _message_for_tutor_from_user_raw(raw: str) -> str:
    """Strip user-simulator scaffolding → the message the tutor/assistant should see."""
    raw = (raw or "").strip()
    if not raw:
        return ""

    # Preferred path (post-migration): the user simulator emits a JSON object
    # whose ``message`` field is the tutor-visible reply.
    parsed = _parse_user_response(raw)
    if parsed is not None and parsed["message"]:
        return parsed["message"]

    # Legacy XML-tag fallbacks below — kept so that mid-experiment data and
    # any prompts that haven't migrated still parse cleanly.
    initial_inner = _extract_initial_query_block(raw)
    if initial_inner is not None:
        return initial_inner

    # Try <message> tag first (new structured format)
    message_inner = _extract_message_block(raw)
    if message_inner is not None:
        return message_inner

    thought_inner, tail = _split_thought_xml_block(raw)
    if thought_inner is not None:
        tail = tail.strip()
        if tail:
            return tail
        # Thought-only output is not tutor-visible user content.
        return ""

    if "Thought:" in raw:
        try:
            if "Response:" in raw:
                return raw.split("Response:", 1)[1].strip()
            if "Query:" in raw:
                return raw.split("Query:", 1)[1].strip()
            if "Message:" in raw:
                return raw.split("Message:", 1)[1].strip()
        except Exception:
            pass

    # Unclosed <thought> tag: LLM started reasoning but never closed the tag or wrote <message>.
    # Strip all stacked opening tags and return the content so it reaches the tutor
    # without the tag, and so conversation_history stays clean.
    lower_raw = raw.lower()
    if "<thought>" in lower_raw:
        idx = lower_raw.index("<thought>") + len("<thought>")
        content = raw[idx:].strip()
        # Strip any additional stacked opening tags (e.g., <thought><thought><thought>...)
        while content.lower().startswith("<thought>"):
            content = content[len("<thought>"):].strip()
        _logger = logging.getLogger("sim_error")
        _logger.warning("_message_for_tutor_from_user_raw: unclosed <thought> tag stripped; "
                        "using content as tutor-visible message")
        return content

    return raw.strip()


def _require_tutor_visible_user_message(parsed: str, raw: str, *, context: str) -> str:
    """Return the tutor-visible message, falling back to thought content when the LLM
    forgot to write a message after </thought>."""
    msg = (parsed or "").strip()
    # Defensive: if parsed result still starts with a <thought> tag (e.g., from an unhandled
    # edge case upstream), log and strip it rather than propagating.
    if msg and msg.lower().startswith("<thought>"):
        _logger = logging.getLogger("sim_error")
        _logger.warning("%s: parsed message starts with <thought> tag — stripping defensively", context)
        inner = msg[len("<thought>"):].strip()
        while inner.lower().startswith("<thought>"):
            inner = inner[len("<thought>"):].strip()
        msg = inner
    if msg:
        return msg
    # Last-resort: LLM only produced a <thought> block — extract its content and use
    # it as the message so the conversation can continue rather than crashing.
    _logger = logging.getLogger("sim_error")
    thought_inner, _ = _split_thought_xml_block((raw or "").strip())
    if thought_inner and thought_inner.strip():
        _logger.warning("%s: LLM produced only <thought> with no visible message; using thought content as fallback", context)
        return thought_inner.strip()
    _logger.warning(
        "%s: no tutor-visible message after parsing (empty <thought> or blank output). "
        "Returning empty string so the conversation loop can handle gracefully.",
        context,
    )
    return ""


def _is_stuck(knowledge_state_history: List[Dict[str, Any]]) -> bool:
    if not knowledge_state_history or len(knowledge_state_history) < 2:
        return False
    last_state = knowledge_state_history[-1] or {}
    prev_state = knowledge_state_history[-2] or {}
    return json.dumps(last_state, sort_keys=True) == json.dumps(prev_state, sort_keys=True)


_EMPTY_RETRY_LIMIT = 2  # number of extra attempts when the user LLM returns empty


async def _retry_empty_user_queries(
    user_model_client: Any,
    user_full_contexts: List[List[Dict[str, str]]],
    user_queries: List[List[str]],
    *,
    temperature: float,
    max_tokens: int,
    show_progress: bool,
    json_mode: bool = False,
) -> List[List[str]]:
    """Re-call the LLM for individual items that returned an empty string.

    Retries up to ``_EMPTY_RETRY_LIMIT`` times per empty item before giving up.
    Returns the (possibly updated) *user_queries* list.
    """
    _logger = logging.getLogger("sim_error")
    empty_indices = [
        i for i, q in enumerate(user_queries)
        if not (q[0] if q else "").strip()
    ]
    if not empty_indices:
        return user_queries

    for attempt in range(1, _EMPTY_RETRY_LIMIT + 1):
        if not empty_indices:
            break
        _logger.warning(
            "Retrying %d empty user query(ies) (attempt %d/%d)",
            len(empty_indices), attempt, _EMPTY_RETRY_LIMIT,
        )
        retry_contexts = [user_full_contexts[i] for i in empty_indices]
        try:
            retry_results = await user_model_client.generate_responses(
                retry_contexts,
                temperature=temperature,
                max_tokens=max_tokens,
                show_progress=show_progress,
                json_mode=json_mode,
            )
        except Exception as exc:
            _logger.warning("Retry batch failed: %s", exc)
            break

        still_empty: List[int] = []
        for list_idx, orig_idx in enumerate(empty_indices):
            new_text = retry_results[list_idx][0] if retry_results[list_idx] else ""
            if (new_text or "").strip():
                user_queries[orig_idx] = retry_results[list_idx]
            else:
                still_empty.append(orig_idx)
        empty_indices = still_empty

    return user_queries


# ── Assistant response length guard ──────────────────────────────────────────

# Sentence limits by strategy (used as the post-processing threshold).
_STRATEGY_SENTENCE_LIMITS = {
    "adaptive": 5,
    "socratic": 3,
    "comprehensive": 10,  # thorough but capped
}

_SHORTEN_INSTRUCTION = (
    "Your previous response was too long. Rewrite it following these rules strictly:\n"
    "- Keep ONLY the single most important concept or reasoning point.\n"
    "- Remove all algebraic derivations — the student should do the computation.\n"
    "- Maximum {limit} sentences. End with a question or a single check for the student.\n"
    "- Do NOT use numbered lists, bullet points, or multi-step explanations.\n\n"
    "Original response to shorten:\n{original}"
)


def _count_sentences(text: str) -> int:
    """Rough sentence count based on sentence-ending punctuation."""
    import re as _re
    # Split on sentence-ending punctuation followed by space or end-of-string
    parts = _re.split(r'[.!?](?:\s|$)', text.strip())
    return len([p for p in parts if p.strip()])


def _has_multiple_display_equations(text: str) -> bool:
    """Check if the response contains more than one display equation (\\[ ... \\] or $$ ... $$)."""
    import re as _re
    display_eqs = _re.findall(r'\\\[.*?\\\]|\$\$.*?\$\$', text, flags=re.DOTALL)
    return len(display_eqs) > 1


async def _enforce_assistant_length(
    assistant_model_client: Any,
    assistant_contexts: List[List[Dict[str, str]]],
    assistant_responses: List[List[str]],
    strategy: str,
    *,
    max_tokens: int,
    show_progress: bool,
) -> List[List[str]]:
    """Re-generate assistant responses that exceed the strategy's length/content limits."""
    _logger = logging.getLogger("sim_error")
    limit = _STRATEGY_SENTENCE_LIMITS.get(strategy, 12)

    indices_to_regen: List[int] = []
    regen_contexts: List[List[Dict[str, str]]] = []

    for i, resp in enumerate(assistant_responses):
        text = resp[0] if resp else ""
        if not text:
            continue
        sentence_count = _count_sentences(text)
        multi_eq = _has_multiple_display_equations(text)
        if sentence_count > limit or multi_eq:
            _logger.info(
                "Assistant response %d exceeds limits (sentences=%d/%d, multi_eq=%s); regenerating",
                i, sentence_count, limit, multi_eq,
            )
            shorten_msg = _SHORTEN_INSTRUCTION.format(limit=limit, original=text)
            # Append a user message asking for shortening, then let the model re-answer
            ctx = list(assistant_contexts[i]) + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": shorten_msg},
            ]
            indices_to_regen.append(i)
            regen_contexts.append(ctx)

    if not regen_contexts:
        return assistant_responses

    _logger.info("Regenerating %d over-long assistant responses for strategy=%s", len(regen_contexts), strategy)
    try:
        regen_results = await assistant_model_client.generate_responses(
            regen_contexts,
            temperature=0.0,
            max_tokens=max_tokens,
            show_progress=show_progress,
        )
        for list_idx, orig_idx in enumerate(indices_to_regen):
            new_text = regen_results[list_idx][0] if regen_results[list_idx] else ""
            if new_text.strip():
                assistant_responses[orig_idx] = regen_results[list_idx]
    except Exception as exc:
        _logger.warning("Assistant regen batch failed: %s", exc)

    return assistant_responses


def _assistant_reference_answer_block(reference_answer: str, *, task: str) -> str:
    """Ground-truth block appended to the assistant system prompt only (not user simulator)."""
    text = (reference_answer or "").strip()
    if not text:
        return ""
    task_label = "open-domain question" if task == "expertqa" else "problem"
    return (
        "\n\n---\n"
        f"## Reference answer (expert / ground truth for this {task_label})\n"
        "Use this only to keep explanations factually aligned with the intended solution. "
        "Do not reveal it verbatim to the student unless they clearly need a factual correction "
        "or explicitly ask for the answer after good-faith effort.\n\n"
        f"{text}\n"
        "---\n"
    )


def _get_misguided_attempt_hint(
    knowledge_state: Optional[Dict[str, Any]],
    knowledge_state_history: List[Dict[str, Any]],
) -> str:
    if not knowledge_state:
        return ""
    unaware_concepts = _get_unaware_concepts(knowledge_state)
    if not unaware_concepts:
        return ""
    if _is_stuck(knowledge_state_history):
        return "If you are stuck, try an approach based on what you currently know, even if it may not be correct, without referencing concepts outside your knowledge."
    return ""


async def run_conversation_batch(
    *,
    problems: List[str],
    user_model_client: Any,
    assistant_model_client: Any,
    prompt_initial_query_template: str,
    prompt_template: str,
    task: str = "math",
    assistant_strategy: str = "adaptive",
    user_profiles: Optional[List[str]] = None,
    knowledge_level_instructions: str = "",
    initial_knowledge_states: Optional[List[str]] = None,
    show_progress: bool = True,
    user_temperature: float = 0.7,
    assistant_temperature: float = 0.0,
    max_tokens: int = 3000,
    max_turns: int = 15,
    length_control_bool: bool = False,
    length_control_list: Optional[List[str]] = None,
    reference_answers: Optional[List[str]] = None,
    disable_length_enforcement: bool = False,
    disable_early_stop: bool = False,
) -> List[Dict[str, Any]]:
    """
    Ported from utils.simulate_conversation_in_batch_math_tutoring (no refinement/user profile).
    """
    length_control_list = length_control_list or []
    user_profiles = user_profiles or ["" for _ in problems]

    conversations_data = []
    assistant_history_label = _get_assistant_history_label(task)
    for i, (problem, user_profile) in enumerate(zip(problems, user_profiles)):
        length_control = length_control_list[i] if length_control_bool else None
        ref_ans = ""
        if reference_answers and i < len(reference_answers):
            ref_ans = reference_answers[i] or ""
        ks_text = (initial_knowledge_states[i] if initial_knowledge_states and i < len(initial_knowledge_states) else "")
        data = {
            "problem": problem,
            "reference_answer": ref_ans,
            "user_profile": user_profile,
            "initial_knowledge_state": ks_text,
            "conversation": [],
            "conversation_history": "",
            "length_control": length_control,
            "assistant_messages": [],
            "first_query": True,
            "turns": 0,
            "finished": False,
            "over_max": False,
            "stop_reason": None,
            "stop_details": {},
            "run_type": "baseline",
        }
        sys_text = build_assistant_system_prompt(task, assistant_strategy)
        assistant_system_prompt = {"role": "system", "content": sys_text}
        data["assistant_messages"].append(assistant_system_prompt)
        conversations_data.append(data)

    for turn in range(max_turns):
        user_full_contexts = []
        active_conversations = []
        for data in conversations_data:
            if data["finished"] or data["over_max"]:
                continue
            if data["first_query"]:
                if length_control_bool:
                    user_message_content = _safe_format(
                        prompt_initial_query_template,
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data["length_control"],
                        user_profile=data.get("user_profile", ""),
                        message_style=data.get("user_profile", ""),
                        knowledge_level_instructions=knowledge_level_instructions,
                        initial_knowledge_state=data.get("initial_knowledge_state", ""),
                    )
                else:
                    user_message_content = _safe_format(
                        prompt_initial_query_template,
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        user_profile=data.get("user_profile", ""),
                        message_style=data.get("user_profile", ""),
                        knowledge_level_instructions=knowledge_level_instructions,
                        initial_knowledge_state=data.get("initial_knowledge_state", ""),
                    )
                data["first_query"] = False
            else:
                if length_control_bool:
                    user_message_content = _safe_format(
                        prompt_template,
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data["length_control"],
                        user_profile=data.get("user_profile", ""),
                        message_style=data.get("user_profile", ""),
                        knowledge_level_instructions=knowledge_level_instructions,
                        initial_knowledge_state=data.get("initial_knowledge_state", ""),
                    )
                else:
                    user_message_content = _safe_format(
                        prompt_template,
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        user_profile=data.get("user_profile", ""),
                        message_style=data.get("user_profile", ""),
                        knowledge_level_instructions=knowledge_level_instructions,
                        initial_knowledge_state=data.get("initial_knowledge_state", ""),
                    )

            user_messages = [{"role": "user", "content": user_message_content}]
            data["user_messages"] = user_messages
            user_full_contexts.append(user_messages)
            active_conversations.append(data)

        if not active_conversations:
            break

        try:
            user_queries = await user_model_client.generate_responses(
                user_full_contexts,
                temperature=user_temperature,
                max_tokens=max_tokens,
                show_progress=show_progress,
            )
        except Exception as exc:
            _logger = logging.getLogger("sim_error")
            _logger.error("user generate_responses batch failed (vanilla): %s\n%s", exc, traceback.format_exc())
            for data in active_conversations:
                data["finished"] = True
                data["stop_reason"] = "api_error"
                data["stop_details"] = {"phase": "user_generation", "error": str(exc), "turn": data.get("turns", 0)}
            break

        user_queries = await _retry_empty_user_queries(
            user_model_client, user_full_contexts, user_queries,
            temperature=user_temperature, max_tokens=max_tokens,
            show_progress=show_progress,
        )

        for data, user_query in zip(active_conversations, user_queries):
            user_query_text = user_query[0] if user_query else ""
            if not (user_query_text or "").strip():
                data["stop_reason"] = "empty_user_message"
                data["stop_details"] = {"user_message": user_query_text}
                data["finished"] = True
                continue

            query = _require_tutor_visible_user_message(
                _message_for_tutor_from_user_raw(user_query_text),
                user_query_text,
                context="User simulator (vanilla batch)",
            )
            if not (query or "").strip():
                data["stop_reason"] = "empty_user_message"
                data["stop_details"] = {"user_message": user_query_text, "parse_error": "thought_only_or_empty"}
                data["finished"] = True
                continue
            data["conversation"].append(("user", query))

            if _is_terminate_signal(query) and _effective_allow_terminate("baseline"):
                if not disable_early_stop:
                    data["stop_reason"] = "user_requested_terminate"
                    data["stop_details"] = {"user_message": query}
                    data["finished"] = True
                    continue
                else:
                    # Record when user-requested termination would have fired,
                    # but keep the conversation running to max_turns. Mirrors
                    # run_conversation_with_interaction_profile (structured path).
                    if data.get("virtual_early_stop_turn") is None:
                        data["virtual_early_stop_turn"] = data["turns"]
                        data["virtual_early_stop_reason"] = "user_requested_terminate"

            data["assistant_messages"].append({"role": "user", "content": query})
            if len(data["assistant_messages"]) == 2:
                data["first_query_content"] = query

        active_conversations = [data for data in active_conversations if not data["finished"]]
        if not active_conversations:
            break

        assistant_full_contexts = [data["assistant_messages"] for data in active_conversations]
        try:
            assistant_responses = await assistant_model_client.generate_responses(
                assistant_full_contexts,
                temperature=assistant_temperature,
                max_tokens=max_tokens,
                show_progress=show_progress,
            )
        except Exception as exc:
            _logger = logging.getLogger("sim_error")
            _logger.error("assistant generate_responses batch failed (vanilla): %s\n%s", exc, traceback.format_exc())
            for data in active_conversations:
                data["finished"] = True
                data["stop_reason"] = "api_error"
                data["stop_details"] = {"phase": "assistant_generation", "error": str(exc), "turn": data.get("turns", 0)}
            break

        if not disable_length_enforcement:
            assistant_responses = await _enforce_assistant_length(
                assistant_model_client, assistant_full_contexts, assistant_responses,
                assistant_strategy,
                max_tokens=max_tokens, show_progress=show_progress,
            )

        for data, assistant_response in zip(active_conversations, assistant_responses):
            assistant_text = assistant_response[0] if assistant_response else ""
            data["conversation"].append(("assistant", assistant_text))
            last_user_message = data["assistant_messages"][-1]["content"]
            if len(data["assistant_messages"]) == 2:
                last_user_message = data["first_query_content"]
            data["conversation_history"] += (
                f"- You: {last_user_message}\n- {assistant_history_label}: {assistant_text}\n"
            )
            data["assistant_messages"].append({"role": "assistant", "content": assistant_text})

            data["turns"] += 1
            if not assistant_text:
                data["stop_reason"] = "empty_assistant"
                data["stop_details"] = {"turn_number": data["turns"]}
                data["finished"] = True
            if data["turns"] >= max_turns:
                data["over_max"] = True
                data["stop_reason"] = "max_turns"
                data["stop_details"] = {"turn_number": data["turns"], "max_turns": max_turns}

    for data in conversations_data:
        # Keep baseline output schema compatible with structured runs.
        data.setdefault("turn_metrics_history", [])
        data.setdefault("engagement_score_history", [])
        data.setdefault("iu_analysis_history", [])
        data.setdefault(
            "metrics",
            {
                "knowledge_gain": None,
                "normalized_knowledge_gain": None,
                "information_calibration": None,
                "perceived_overload": None,
                "engagement": None,
                "learning_experience": None,
                "overload_inverse": None,
                "turn_metrics_history": [],
            },
        )

    return conversations_data


async def run_conversation_with_interaction_profile(
    *,
    problems: List[str],
    problem_ids: List[str],
    user_profiles: List[str],
    user_model_client: Any,
    assistant_model_client: Any,
    prompt_initial_query_template: str,
    prompt_template: str,
    concept_graph: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    iu_graphs: Optional[Dict[str, Dict[str, Any]]] = None,
    id_maps: Optional[Dict[str, Dict[str, str]]] = None,
    knowledge_states: Optional[List[Dict[str, Any]]] = None,
    knowledge_levels: Optional[List[str]] = None,
    task: str = "math",
    assistant_strategy: str = "adaptive",
    knowledge_level_instructions: str = "",
    user_temperature: float = 0.7,
    assistant_temperature: float = 0.0,
    max_tokens: int = 3000,
    max_turns: int = 15,
    length_control_bool: bool = False,
    length_control_list: Optional[List[str]] = None,
    show_progress: bool = True,
    disable_early_stop: bool = False,
    reference_answers: Optional[List[str]] = None,
    absorption_mode: str = "default",
    temporal_decay: bool = False,
    disable_length_enforcement: bool = False,
) -> List[Dict[str, Any]]:
    """
    Ported from utils.simulate_conversation_with_user_profile_in_batch_math_tutoring,
    simplified to interaction-style profiles only.
    """
    length_control_list = length_control_list or []
    absorption_preset = get_absorption_preset(absorption_mode)

    conversations_data = []
    for i, problem in enumerate(problems):
        ref_ans = ""
        if reference_answers and i < len(reference_answers):
            ref_ans = reference_answers[i] or ""
        length_control = length_control_list[i] if length_control_bool else None
        knowledge_state = knowledge_states[i] if knowledge_states else None
        knowledge_level = knowledge_levels[i] if knowledge_levels else "intermediate"
        problem_id = problem_ids[i] if problem_ids else None
        articulation_mode = "Vague"
        overall_density = 0.5
        gap_sum = 0.0
        query_tone = "Vague"
        initial_overload_threshold = 5  # fallback default
        if knowledge_state is not None and iu_graphs is not None and problem_id:
            from ..knowledge.update_v2 import compute_overall_density_from_state, compute_gap_sum_from_state

            iu_graph = iu_graphs.get(str(problem_id)) or {}
            id_map = id_maps.get(str(problem_id)) if id_maps else None
            if iu_graph:
                metrics = compute_overall_density_from_state(
                    iu_graph=iu_graph,
                    knowledge_state=knowledge_state,
                    id_map=id_map,
                )
                gap_metrics = compute_gap_sum_from_state(
                    iu_graph=iu_graph,
                    knowledge_state=knowledge_state,
                    id_map=id_map,
                )
                articulation_mode = metrics.get("articulation_mode", articulation_mode)
                overall_density = metrics.get("overall_density", overall_density)
                gap_sum = gap_metrics.get("gap_sum", 0.0)
                query_tone = "Vague" if gap_sum >= 2.0 else "Explicit"
                # Overload threshold is mode-aware: the default preset uses
                # per-level ratios (novice=0.15, intermediate=0.40,
                # advanced=0.60); weightabsorb uses a flat 0.40 for all levels
                # because the level-effect lives in the load weights instead.
                _n_nodes = len(iu_graph.get("nodes", []))
                initial_overload_threshold = compute_overload_threshold(
                    _n_nodes, knowledge_level, absorption_mode,
                )
        data = {
            "problem": problem,
            "reference_answer": ref_ans,
            "problem_id": problem_id,
            "user_profile": user_profiles[i],
            "length_control": length_control,
            "knowledge_state": knowledge_state,
            "knowledge_state_history": [knowledge_state] if knowledge_state else [],
            "articulation_mode": articulation_mode,
            "overall_density": overall_density,
            "gap_sum": gap_sum,
            "query_tone": query_tone,
            "initial_overload_threshold": initial_overload_threshold,
            "absorption_mode": absorption_mode,
            "absorption_config": dict(absorption_preset),
            "explained_concepts": [],
            "engagement_score_history": [],
            "iu_analysis_history": [],
            "turn_metrics_history": [],
            "ic_numerator_total": 0.0,
            "ic_denominator_total": 0.0,
            "new_ius_total": 0.0,
            "disallowed_total": 0.0,
            "zpd_size_total": 0.0,
            "absorbed_gain_total": 0.0,
            "attempt_verify": False,
            "pending_stop_hint": "",
            "conversation": [],
            "conversation_history": "",
            "assistant_messages": [],
            "first_query": True,
            "turns": 0,
            "finished": False,
            "over_max": False,
            "stop_reason": None,
            "stop_details": {},
            "run_type": "KS-based",
        }
        if knowledge_state:
            data["initial_knowledge_state"] = json.loads(json.dumps(knowledge_state))
        system_content = build_assistant_system_prompt(task, assistant_strategy)
        assistant_system_prompt = {"role": "system", "content": system_content}
        data["assistant_messages"].append(assistant_system_prompt)
        conversations_data.append(data)

    for turn in range(max_turns):
        user_full_contexts = []
        active_conversations = []
        for data in conversations_data:
            if data["finished"] or data["over_max"]:
                continue
            if data["first_query"]:
                ks = data.get("knowledge_state")
                _knows_well = _get_concepts_by_state(ks, "knows_well")
                _partial = _get_concepts_by_state(ks, "partial_understanding")
                _struggling = _get_concepts_by_state(ks, "struggling")
                _unaware = _get_unaware_concepts(ks)
                if length_control_bool:
                    user_message_content = _safe_format(
                        prompt_initial_query_template,
                        user_profile=data["user_profile"],
                        message_style=data["user_profile"],
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data["length_control"],
                        knowledge_state_formatted=_format_knowledge_state(ks),
                        knows_well_concepts=_knows_well,
                        partial_understanding_concepts=_partial,
                        struggling_concepts=_struggling,
                        unaware_concepts=_unaware,
                        askable_concepts=_get_askable_concepts(ks, data.get("explained_concepts", [])),
                        assistant_message=_get_last_assistant_message(data.get("assistant_messages", [])),
                        articulation_mode=_get_articulation_mode(data),
                        articulation_guidance=_get_articulation_guidance(data),
                        density_label=_get_density_label(data),
                        density_percent=_get_density_percent(data),
                        explained_concepts=data.get("explained_concepts", []),
                        asked_concepts=data.get("explained_concepts", []),
                        knowledge_level_instructions=knowledge_level_instructions,
                    )
                else:
                    user_message_content = _safe_format(
                        prompt_initial_query_template,
                        user_profile=data["user_profile"],
                        message_style=data["user_profile"],
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data.get("length_control") or "",
                        knowledge_state_formatted=_format_knowledge_state(ks),
                        knows_well_concepts=_knows_well,
                        partial_understanding_concepts=_partial,
                        struggling_concepts=_struggling,
                        unaware_concepts=_unaware,
                        askable_concepts=_get_askable_concepts(ks, data.get("explained_concepts", [])),
                        assistant_message=_get_last_assistant_message(data.get("assistant_messages", [])),
                        articulation_mode=_get_articulation_mode(data),
                        articulation_guidance=_get_articulation_guidance(data),
                        density_label=_get_density_label(data),
                        density_percent=_get_density_percent(data),
                        explained_concepts=data.get("explained_concepts", []),
                        asked_concepts=data.get("explained_concepts", []),
                        knowledge_level_instructions=knowledge_level_instructions,
                    )
                data["first_query"] = False
            else:
                pending_hint = data.get("pending_stop_hint", "")
                data["pending_stop_hint"] = ""
                if length_control_bool:
                    user_message_content = _safe_format(
                        prompt_template,
                        user_profile=data["user_profile"],
                        message_style=data["user_profile"],
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data["length_control"],
                        knowledge_state_formatted=_format_knowledge_state(data.get("knowledge_state")),
                        askable_concepts=_get_askable_concepts(
                            data.get("knowledge_state"),
                            data.get("explained_concepts", []),
                        ),
                        unaware_concepts=_get_unaware_concepts(data.get("knowledge_state")),
                        assistant_message=_get_last_assistant_message(data.get("assistant_messages", [])),
                        articulation_mode=_get_articulation_mode(data),
                        articulation_guidance=_get_articulation_guidance(data),
                        density_label=_get_density_label(data),
                        density_percent=_get_density_percent(data),
                        explained_concepts=data.get("explained_concepts", []),
                        asked_concepts=data.get("explained_concepts", []),
                        misguided_attempt_hint=_get_misguided_attempt_hint(
                            data.get("knowledge_state"),
                            data.get("knowledge_state_history", []),
                        ),
                        stop_hint=pending_hint,
                        knowledge_level_instructions=knowledge_level_instructions,
                    )
                else:
                    user_message_content = _safe_format(
                        prompt_template,
                        user_profile=data["user_profile"],
                        message_style=data["user_profile"],
                        math_problem=data["problem"],
                        conversation_history=data["conversation_history"].strip(),
                        length_control=data.get("length_control") or "",
                        knowledge_state_formatted=_format_knowledge_state(data.get("knowledge_state")),
                        askable_concepts=_get_askable_concepts(
                            data.get("knowledge_state"),
                            data.get("explained_concepts", []),
                        ),
                        unaware_concepts=_get_unaware_concepts(data.get("knowledge_state")),
                        assistant_message=_get_last_assistant_message(data.get("assistant_messages", [])),
                        articulation_mode=_get_articulation_mode(data),
                        articulation_guidance=_get_articulation_guidance(data),
                        density_label=_get_density_label(data),
                        density_percent=_get_density_percent(data),
                        explained_concepts=data.get("explained_concepts", []),
                        asked_concepts=data.get("explained_concepts", []),
                        misguided_attempt_hint=_get_misguided_attempt_hint(
                            data.get("knowledge_state"),
                            data.get("knowledge_state_history", []),
                        ),
                        stop_hint=pending_hint,
                        knowledge_level_instructions=knowledge_level_instructions,
                    )

            user_messages = [{"role": "user", "content": user_message_content}]
            data["user_messages"] = user_messages
            user_full_contexts.append(user_messages)
            active_conversations.append(data)

        if not active_conversations:
            break

        try:
            # The structured user simulator emits JSON ({thought, message,
            # attempt_verify, terminate}). Constraining the provider to JSON
            # mode prevents the runaway-tag failure mode we hit pre-migration.
            user_queries = await user_model_client.generate_responses(
                user_full_contexts,
                temperature=user_temperature,
                max_tokens=max_tokens,
                show_progress=show_progress,
                json_mode=True,
            )
        except Exception as exc:
            _logger = logging.getLogger("sim_error")
            _logger.error("user generate_responses batch failed (structured): %s\n%s", exc, traceback.format_exc())
            for data in active_conversations:
                data["finished"] = True
                data["stop_reason"] = "api_error"
                data["stop_details"] = {"phase": "user_generation", "error": str(exc), "turn": data.get("turns", 0)}
            break

        user_queries = await _retry_empty_user_queries(
            user_model_client, user_full_contexts, user_queries,
            temperature=user_temperature, max_tokens=max_tokens,
            show_progress=show_progress,
            json_mode=True,
        )

        # Defense in depth: even with provider-level JSON enforcement, log a
        # warning if any user-simulator output looks malformed (runaway tag
        # repetition, unbalanced braces, suspiciously long body). The warning
        # surfaces in the per-condition errors_*.log so we'd notice if the
        # provider's JSON-mode guarantee ever fails in production.
        for _idx, _q in enumerate(user_queries):
            _raw = _q[0] if _q else ""
            _why = _is_suspicious_user_output(_raw)
            if _why:
                logging.getLogger("sim_error").warning(
                    "Suspicious user-simulator output detected for conv #%d: %s. "
                    "Output head: %r", _idx, _why, _raw[:300]
                )

        for data, user_query in zip(active_conversations, user_queries):
            user_query_text = user_query[0] if user_query else ""
            # Preferred path (post-migration): structured JSON. Pull all fields
            # from the parsed dict in one shot. Falls through to legacy XML-tag
            # extractors when the output isn't JSON (older prompts or older
            # mid-experiment data).
            _parsed = _parse_user_response(user_query_text)
            if _parsed is not None:
                attempt_verify_flag = _parsed["attempt_verify"]
                terminate_flag = _parsed["terminate"]
            else:
                user_query_text, attempt_verify_flag = _extract_attempt_verify_marker(user_query_text)
                user_query_text, terminate_flag = _extract_terminate_marker(user_query_text)
            data["attempt_verify"] = attempt_verify_flag
            if not (user_query_text or "").strip():
                data["stop_reason"] = "empty_user_message"
                data["stop_details"] = {"user_message": user_query_text}
                data["finished"] = True
                continue

            query = _require_tutor_visible_user_message(
                _message_for_tutor_from_user_raw(user_query_text),
                user_query_text,
                context="User simulator (structured)",
            )
            if not (query or "").strip():
                data["stop_reason"] = "empty_user_message"
                data["stop_details"] = {"user_message": user_query_text, "parse_error": "thought_only_or_empty"}
                data["finished"] = True
                continue
            data["conversation"].append(("user", query))

            if terminate_flag and _effective_allow_terminate("structured"):
                if not disable_early_stop:
                    details = data.get("stop_details") or {}
                    data["stop_reason"] = details.get("type") or "user_requested_terminate"
                    data["stop_details"] = {**details, "user_message": query}
                    data["finished"] = True
                    continue
                else:
                    # Record when user-requested termination would have fired,
                    # but keep the conversation running to max_turns.
                    if data.get("virtual_early_stop_turn") is None:
                        data["virtual_early_stop_turn"] = data["turns"]  # current turn (0-indexed: no assistant yet this turn)
                        data["virtual_early_stop_reason"] = "user_requested_terminate"

            data["assistant_messages"].append({"role": "user", "content": query})
            if len(data["assistant_messages"]) == 2:
                data["first_query_content"] = query

        active_conversations = [data for data in active_conversations if not data["finished"]]
        if not active_conversations:
            break

        assistant_full_contexts = [data["assistant_messages"] for data in active_conversations]
        try:
            assistant_responses = await assistant_model_client.generate_responses(
                assistant_full_contexts,
                temperature=assistant_temperature,
                max_tokens=max_tokens,
                show_progress=show_progress,
            )
        except Exception as exc:
            _logger = logging.getLogger("sim_error")
            _logger.error("assistant generate_responses batch failed: %s\n%s", exc, traceback.format_exc())
            for data in active_conversations:
                data["finished"] = True
                data["stop_reason"] = "api_error"
                data["stop_details"] = {"phase": "assistant_generation", "error": str(exc), "turn": data.get("turns", 0)}
            break

        if not disable_length_enforcement:
            assistant_responses = await _enforce_assistant_length(
                assistant_model_client, assistant_full_contexts, assistant_responses,
                assistant_strategy,
                max_tokens=max_tokens, show_progress=show_progress,
            )

        # ── Phase 1: per-conversation sync bookkeeping (assistant text, turn counter, over_max). ──
        post_turn_info: List[Tuple[Dict[str, Any], str, str]] = []  # (data, last_user_message, assistant_text)
        for data, assistant_response in zip(active_conversations, assistant_responses):
            assistant_text = assistant_response[0] if assistant_response else ""
            data["conversation"].append(("assistant", assistant_text))
            last_user_message = data["assistant_messages"][-1]["content"]
            if len(data["assistant_messages"]) == 2:
                last_user_message = data["first_query_content"]
            data["conversation_history"] += f"- You: {last_user_message}\n- AI Tutor: {assistant_text}\n"
            data["assistant_messages"].append({"role": "assistant", "content": assistant_text})
            data["turns"] += 1
            if not assistant_text:
                data["stop_reason"] = "empty_assistant"
                data["stop_details"] = {"turn_number": data["turns"]}
                data["finished"] = True
            if data["turns"] >= max_turns:
                data["over_max"] = True
                pending_reason = (data.get("stop_details") or {}).get("type")
                data["stop_reason"] = pending_reason if pending_reason else "max_turns"
                data["stop_details"] = {
                    **(data.get("stop_details") or {}),
                    "turn_number": data["turns"],
                    "max_turns": max_turns,
                }
            post_turn_info.append((data, last_user_message, assistant_text))

        # ── Phase 2: collect KS-update coroutines for qualifying conversations. ──
        from ..knowledge.update_v2 import update_dynamic_knowledge_state

        ks_coros: List[Optional[Any]] = []
        for data, last_user_message, assistant_text in post_turn_info:
            if data.get("knowledge_state") is not None and data.get("problem_id"):
                problem_id = str(data.get("problem_id"))
                iu_graph = iu_graphs.get(problem_id) if iu_graphs else None
                id_map = id_maps.get(problem_id) if id_maps else None
                # Prior conversation = everything in data["conversation"] except the last user/assistant pair
                # (which is the current turn we're already passing as user_message + assistant_message).
                # Phase B uses this to distinguish user reasoning from articulation/parroting.
                prior_turns = data.get("conversation", [])[:-2]
                prior_history_text = "\n".join(
                    f"- {'User' if role == 'user' else 'Assistant'}: {text}"
                    for role, text in prior_turns
                )
                ks_coros.append(update_dynamic_knowledge_state(
                    assistant_message=assistant_text,
                    user_message=last_user_message,
                    attempt_verify=bool(data.get("attempt_verify")),
                    knowledge_state=data["knowledge_state"],
                    model_client=user_model_client,
                    iu_graph=iu_graph,
                    id_map=id_map,
                    concept_graph=concept_graph,
                    problem_id=problem_id,
                    reference_answer=str(data.get("reference_answer", "") or ""),
                    max_tokens=absorption_preset["phase_b_max_tokens"],
                    show_progress=False,
                    return_metrics=True,
                    overload_threshold=data.get("initial_overload_threshold", 5),
                    turn_number=data["turns"],
                    absorption_mode=absorption_mode,
                    conversation_history=prior_history_text,
                    temporal_decay=temporal_decay,
                    max_turns=max_turns,
                ))
            else:
                ks_coros.append(None)

        # ── Phase 3: fan out Phase B LLM calls concurrently. ──
        runnable_pairs = [(i, c) for i, c in enumerate(ks_coros) if c is not None]
        ks_results: List[Any] = [None] * len(ks_coros)
        if runnable_pairs:
            gathered = await asyncio.gather(
                *[c for _, c in runnable_pairs], return_exceptions=True
            )
            for (i, _), result in zip(runnable_pairs, gathered):
                ks_results[i] = result

        # ── Phase 4: apply KS results + mastery/termination checks (sync per-data). ──
        for (data, last_user_message, assistant_text), update_result in zip(post_turn_info, ks_results):
            if update_result is None:
                continue
            if isinstance(update_result, BaseException):
                exc = update_result
                _logger = logging.getLogger("sim_error")
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                _logger.error(
                    "Knowledge state update failed | problem_id=%s turn=%d | %s\n%s",
                    data.get("problem_id"), data.get("turns", 0), exc, tb,
                )
                data.setdefault("errors", []).append({
                    "turn": data.get("turns", 0),
                    "phase": "knowledge_state_update",
                    "error": str(exc),
                    "traceback": tb,
                })
                data["finished"] = True
                data["stop_reason"] = "error"
                data["stop_details"] = {"type": "knowledge_state_update_error", "turn_number": data.get("turns", 0), "error": str(exc)}
                continue
            try:
                updated_state, metrics = update_result
                data["knowledge_state"] = updated_state
                data["knowledge_state_history"].append(updated_state)
                data["articulation_mode"] = metrics.get("articulation_mode", data.get("articulation_mode"))
                data["overall_density"] = metrics.get("overall_density", data.get("overall_density"))
                data["gap_sum"] = metrics.get("gap_sum", data.get("gap_sum"))
                data["query_tone"] = "Vague" if (data.get("gap_sum") or 0.0) >= 2.0 else "Explicit"
                data["engagement_score_history"].append({
                    "engagement": metrics.get("engagement_score", 0.0),
                    "familiarity": metrics.get("familiarity_term", 0.0),
                    "novelty": metrics.get("novelty_term", 0.0),
                })
                data["iu_analysis_history"].append(metrics.get("iu_analysis", []))

                ic_num_turn = float(metrics.get("information_calibration_numerator", 0.0))
                ic_den_turn = float(metrics.get("information_calibration_denominator", 0.0))
                new_ius_turn = float(metrics.get("new_ius", 0.0))
                disallowed_turn = float(metrics.get("disallowed", 0.0))
                zpd_size_turn = float(metrics.get("zpd_size", 0.0))
                ic_turn = (ic_num_turn / ic_den_turn) if ic_den_turn > 0 else 0.0
                # Volume-based PO comes directly from update_v2 (max(0, load-threshold)/load)
                po_turn = float(metrics.get("perceived_overload_turn", 0.0))
                data["turn_metrics_history"].append(
                    {
                        "turn": data["turns"],
                        "engagement": float(metrics.get("engagement_score", 0.0)),
                        "information_calibration_turn": ic_turn,
                        "ic_numerator": ic_num_turn,
                        "ic_denominator": ic_den_turn,
                        "zpd_size": zpd_size_turn,
                        "perceived_overload_turn": po_turn,
                        "overload_threshold": float(metrics.get("overload_threshold", 0.0)),
                        "familiarity_term": float(metrics.get("familiarity_term", 0.0)),
                        "novelty_term": float(metrics.get("novelty_term", 0.0)),
                        "overload": bool(metrics.get("overload", False)),
                        "n_upward_transitions": int(metrics.get("n_upward_transitions", 0)),
                        # |A_t| for Delivery Calibration. Left as None when the
                        # updater did not report it, so the aggregate keeps its
                        # documented fallback for histories written earlier.
                        "absorbed_ic_numerator": metrics.get("absorbed_ic_numerator"),
                        "absorbed_gain": float(metrics.get("absorbed_gain", 0.0)),
                        # Decomposition of effective_load. attempt_load is 0.0
                        # unless absorption_mode opted into include_attempt_load
                        # (e.g. weightabsorb_attempt). Lets the dashboard split
                        # PO into comprehension vs attempt contributions.
                        "comprehension_load": float(metrics.get("comprehension_load", 0.0)),
                        "attempt_load": float(metrics.get("attempt_load", 0.0)),
                        "effective_load": float(metrics.get("effective_load", 0.0)),
                    }
                )
                data["ic_numerator_total"] += ic_num_turn
                data["ic_denominator_total"] += ic_den_turn
                data["new_ius_total"] += new_ius_turn
                data["disallowed_total"] += disallowed_turn
                data["zpd_size_total"] += zpd_size_turn
                data["absorbed_gain_total"] += float(metrics.get("absorbed_gain", 0.0))
                data["attempt_verify"] = False

                # Accumulate explained_concepts from Phase B (concept labels directly explained this turn)
                newly_explained = metrics.get("explained_concepts", [])
                merged_explained = set(data.get("explained_concepts", []))
                merged_explained.update(newly_explained)
                data["explained_concepts"] = sorted(merged_explained)

                # Phase B empty after retry: KS is frozen and unreliable.
                # Stop the conversation to avoid generating turns with stale state.
                if metrics.get("phase_b_empty", False):
                    _pb_logger = logging.getLogger("sim_error")
                    _pb_logger.warning(
                        "Stopping conversation (problem_id=%s, turn=%d): "
                        "Phase B returned empty iu_analysis even after retry.",
                        data.get("problem_id"), data.get("turns", 0),
                    )
                    data["finished"] = True
                    data["stop_reason"] = "phase_b_empty"
                    data["stop_details"] = {
                        "type": "phase_b_empty",
                        "turn_number": data.get("turns", 0),
                    }
                    continue

                # Mastery: all concepts reached knows_well.
                if updated_state and all(
                    _normalize_state((info.get("state", "") if isinstance(info, dict) else info)) == "knows_well"
                    for info in updated_state.values()
                ):
                    if not disable_early_stop:
                        data["finished"] = True
                        data["stop_reason"] = "mastery"
                        data["stop_details"] = {"type": "mastery", "turn_number": data["turns"]}
                        continue
                    else:
                        # Record when mastery would have triggered for ablation analysis,
                        # but keep the conversation running to max_turns.
                        if data.get("virtual_mastery_turn") is None:
                            data["virtual_mastery_turn"] = data["turns"]  # 1-indexed for traceability
                            # Consolidate into virtual_early_stop_turn (0-indexed).
                            # Overwrite only if mastery fires earlier than any previously recorded trigger.
                            mastery_0idx = data["turns"] - 1
                            if data.get("virtual_early_stop_turn") is None or mastery_0idx < data["virtual_early_stop_turn"]:
                                data["virtual_early_stop_turn"] = mastery_0idx
                                data["virtual_early_stop_reason"] = "mastery"

                # Stagnation / overload-persistence: defer to next user turn via hint.
                termination = _check_termination(data, min_warmup=3)
                if termination:
                    stop_reason, details = termination
                    if not disable_early_stop:
                        data["pending_stop_hint"] = _get_stop_hint(stop_reason)
                        data["stop_details"] = details
                    else:
                        # Record when stagnation/overload would have triggered for ablation analysis.
                        # Only write if no earlier trigger (mastery or prior overload) already recorded.
                        # Value is 0-indexed to match compute_early_stop_metrics.py convention.
                        if data.get("virtual_early_stop_turn") is None:
                            data["virtual_early_stop_turn"] = data["turns"] - 1
                            data["virtual_early_stop_reason"] = stop_reason
            except Exception as exc:
                _logger = logging.getLogger("sim_error")
                tb = traceback.format_exc()
                _logger.error(
                    "Knowledge state update failed | problem_id=%s turn=%d | %s\n%s",
                    data.get("problem_id"), data.get("turns", 0), exc, tb,
                )
                data.setdefault("errors", []).append({
                    "turn": data.get("turns", 0),
                    "phase": "knowledge_state_update",
                    "error": str(exc),
                    "traceback": tb,
                })
                data["finished"] = True
                data["stop_reason"] = "error"
                data["stop_details"] = {"type": "knowledge_state_update_error", "turn_number": data.get("turns", 0), "error": str(exc)}

    for data in conversations_data:
        if data.get("knowledge_state") is not None:
            data["metrics"] = _compute_structured_metrics(data)
        else:
            data["metrics"] = {
                "knowledge_gain": None,
                "normalized_knowledge_gain": None,
                "information_calibration": None,
                "perceived_overload": None,
                "engagement": None,
                "learning_experience": None,
                "overload_inverse": None,
            }

    return conversations_data

