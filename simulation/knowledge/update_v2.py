"""Update dynamic knowledge states (Kt) using simplified 5-rule IU pipeline."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional

from ..core.feature_flags import get_flags
from ..core.prompts import load_prompt
from .metrics import absorbed_ic_numerator_per_turn
from .state import STATE_ORDER


# Absorption-mode presets.
#
# "default" reproduces the original count-based hard-cutoff absorption
# behavior. "weightabsorb" introduces state-weighted load and continuous
# per-IU dropoff under overload, with the level-effect moved from the
# threshold into the load weights. See KNOWLEDGE_STATE_UPDATE.md for the
# semantic difference and PROJECT_OVERVIEW.md section 14 for the metric
# implications.
ABSORPTION_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "weighting": "uniform_review",
        "review_weight": 0.5,
        "dropoff": False,
        "ratio_table": {"novice": 0.15, "intermediate": 0.40, "advanced": 0.60},
        "rule2_cap": True,
        # Was 1200 — too small. Phase B emits one JSON object per IU with
        # concept / teaching_quality / reasoning / attempted / role fields;
        # for graphs with 10+ IUs the response routinely passes ~4000 chars
        # (~1100+ tokens) and gets truncated mid-JSON, leaving the parser
        # with no closing brace → empty iu_analysis → no KS updates → NKG=0.
        # 4000 leaves headroom for 20+ IU graphs.
        "phase_b_max_tokens": 128000,
        "include_attempt_load": False,
    },
    "weightabsorb": {
        "weighting": "state_weighted",
        # Retune (2026-04-27): flatter W_STATE curve and higher ratio.
        # Prior canonical tuning was {2.5, 1.75, 0.75, 0.1} with ratio 0.40;
        # earlier retune {2.0, 1.5, 1.0, 0.3} with ratio 0.40 was reverted in
        # commit 2a5c983. Current values raise the absorption ceiling
        # (ratio 0.40 → 0.55) and soften the partial→knows_well dropoff
        # (knows_well 0.1 → 0.5) for validation-set re-evaluation.
        "W_STATE": {
            "unaware": 2.5,
            "struggling": 1.5,
            "partial_understanding": 1.0,
            "knows_well": 0.5,
        },
        "dropoff": True,
        "ratio_table": {"novice": 0.55, "intermediate": 0.55, "advanced": 0.55},
        "rule2_cap": False,
        "phase_b_max_tokens": 128000,
        "include_attempt_load": False,
    },
    # ── weightabsorb + attempt-load ──────────────────────────────────────────
    # Toggleable variant of weightabsorb that adds user-side cognitive load
    # (reasoning + articulation) into effective_load before the overload
    # check. The single dropoff at update_v2.py:_compute_effective_load /
    # absorption_efficiency then scales BOTH absorbed advances and Rule-5
    # advances under one shared overload signal — the unified working-memory
    # design: a single overload signal scales both absorbed advances and
    # Rule-5 advances together (see `_compute_attempt_load` below for the
    # attempt-side load formula). Existing weightabsorb behavior
    # is preserved; switch via --absorption_mode weightabsorb_attempt.
    "weightabsorb_attempt": {
        "weighting": "state_weighted",
        # Retune (2026-04-27, v1): both W_STATE and W_REASONING scaled to
        # 0.7× of the original `weightabsorb` curve. Without this, the
        # unified comprehension+attempt effective_load was 1.4–2.3× the
        # 0.55-ratio threshold, firing overload on 70–100% of turns and
        # collapsing Rule-5 advances under the dropoff. The 0.7× scaling
        # lands typical effective_load near the threshold so the dropoff
        # behaves comparably to plain weightabsorb (overload ~30–90%).
        "W_STATE": {
            "unaware": 1.75,
            "struggling": 1.0,
            "partial_understanding": 0.7,
            "knows_well": 0.35,
        },
        "dropoff": True,
        "ratio_table": {"novice": 0.55, "intermediate": 0.55, "advanced": 0.55},
        "rule2_cap": False,
        "phase_b_max_tokens": 128000,
        "include_attempt_load": True,
        # Reasoning about a shaky concept costs more working memory than
        # reasoning about a solid one. Same 0.7× scale as W_STATE.
        "W_REASONING": {
            "unaware": 1.75,
            "struggling": 1.4,
            "partial_understanding": 0.7,
            "knows_well": 0.35,
        },
        # Articulation costs something (the user must retrieve and restate)
        # but less than novel reasoning. 0.35 = same as W_STATE[knows_well]
        # post-retune, to keep articulation cost proportional to the rest.
        "W_ARTICULATION": 0.35,
    },
    # ── weightabsorb_attempt + saturating dropoff + temporal decay ───────────
    # Same weights as weightabsorb_attempt, but replaces the linear dropoff
    # (efficiency = 1 - overage) with a saturating curve (efficiency =
    # threshold / load) that never reaches zero. Temporal decay is also
    # enabled by default: efficiency *= max_turns / (max_turns + t - 1).
    "weightabsorb_attempt_saturating": {
        "weighting": "state_weighted",
        "W_STATE": {
            "unaware": 1.75,
            "struggling": 1.0,
            "partial_understanding": 0.7,
            "knows_well": 0.35,
        },
        "dropoff": True,
        "dropoff_curve": "saturating",
        "temporal_decay": True,
        "ratio_table": {"novice": 0.55, "intermediate": 0.55, "advanced": 0.55},
        "rule2_cap": False,
        "phase_b_max_tokens": 128000,
        "include_attempt_load": True,
        "W_REASONING": {
            "unaware": 1.75,
            "struggling": 1.4,
            "partial_understanding": 0.7,
            "knows_well": 0.35,
        },
        "W_ARTICULATION": 0.35,
    },
    # ── pilot1_20260514: same as saturating but with sigmoid dropoff ────────
    # Continuous at threshold (efficiency=1.0 at eff=thr, smooth decay above).
    # Used to test whether the discontinuity at threshold is part of the
    # model-arm KG alignment failure (see 20260514_EXPERIMENT_LOG.md §7).
    "weightabsorb_attempt_sigmoid": {
        "weighting": "state_weighted",
        "W_STATE": {
            "unaware": 1.75,
            "struggling": 1.0,
            "partial_understanding": 0.7,
            "knows_well": 0.35,
        },
        "dropoff": True,
        "dropoff_curve": "sigmoid",
        "temporal_decay": True,
        "ratio_table": {"novice": 0.55, "intermediate": 0.55, "advanced": 0.55},
        "rule2_cap": False,
        "phase_b_max_tokens": 128000,
        "include_attempt_load": True,
        "W_REASONING": {
            "unaware": 1.75,
            "struggling": 1.4,
            "partial_understanding": 0.7,
            "knows_well": 0.35,
        },
        "W_ARTICULATION": 0.35,
    },
    # ── unified (pilot1_20260518): single-mechanism replacement ─────────────
    # Empirically validated to reproduce the weightabsorb_attempt_sigmoid +
    # per_iu_hinge stack at 99.6% per-IU state match across 4050 turns of
    # modified_20260517_* (see scripts/audit_unified_absorption.md). Drops
    # the sigmoid_softening, per_iu_hinge, and temporal_decay layers entirely
    # in favor of one uniform saturating curve (efficiency = thr/load).
    #
    # Geometric W:    W[k] = w0 · base^(L-1-k) with w0=0.35, base=1.8
    #                 → {unaware:2.041, struggling:1.134, partial:0.63, kw:0.35}
    # Reasoning cost: equals W[state] (alpha=1.0; no separate W_REASONING curve).
    # Articulation:   flat at w0 (same as W[knows_well]).
    #
    # Must be used with ks_update flags: minimal_rules=true,
    # cap_well_explained_at_one=true, shallow_unaware_only=true,
    # fractional_accumulation=true, per_iu_hinge=false (default).
    "unified": {
        "weighting": "state_weighted",
        "W_STATE": {
            "unaware": 2.041,
            "struggling": 1.134,
            "partial_understanding": 0.63,
            "knows_well": 0.35,
        },
        "dropoff": True,
        "dropoff_curve": "saturating",
        "ratio_table": {"novice": 0.55, "intermediate": 0.55, "advanced": 0.55},
        "rule2_cap": False,
        "phase_b_max_tokens": 128000,
        "include_attempt_load": True,
        "W_REASONING": {
            "unaware": 2.041,
            "struggling": 1.134,
            "partial_understanding": 0.63,
            "knows_well": 0.35,
        },
        "W_ARTICULATION": 0.35,
    },
    # ── unified_clean (pilot2_20260519): clean-hyperparameters refinement ──
    # Same single-mechanism design as `unified`, with three lessons applied:
    #
    #   1. Geometric W cleaned to powers-of-2: w0=0.25, base=2
    #      → W = {2, 1, 0.5, 0.25}   (was {2.041, 1.134, 0.63, 0.35})
    #      Smaller magnitude on partial/knows_well drops intermediate +
    #      advanced loads, addressing the harsher-than-modified overload
    #      that the pilot1 → modified_* comparison revealed.
    #
    #   2. budget_ratio cleaned: 0.55 → 0.50 (= 1/2).
    #
    #   3. articulation cost bound to w0 (=0.25), removing the free constant.
    #
    # Reasoning cost still equals W[state] (alpha=1, "reasoning doubles
    # mention"). Saturating curve eff = thr/load unchanged.
    #
    # Replay vs modified_*: 99.47% per-IU state match (vs pilot1's 99.61%;
    # 0.14pp delta is expected — see scripts/audit_unified_absorption*.md).
    "unified_clean": {
        "weighting": "state_weighted",
        "W_STATE": {
            "unaware": 2.0,
            "struggling": 1.0,
            "partial_understanding": 0.5,
            "knows_well": 0.25,
        },
        "dropoff": True,
        "dropoff_curve": "saturating",
        "ratio_table": {"novice": 0.5, "intermediate": 0.5, "advanced": 0.5},
        "rule2_cap": False,
        "phase_b_max_tokens": 128000,
        "include_attempt_load": True,
        "W_REASONING": {  # = W_STATE (alpha=1.0; reasoning doubles mention cost)
            "unaware": 2.0,
            "struggling": 1.0,
            "partial_understanding": 0.5,
            "knows_well": 0.25,
        },
        "W_ARTICULATION": 0.25,  # = W[knows_well] = w0
    },
    "greedy_fill": {
        "weighting": "state_weighted",
        "W_STATE": {
            "unaware": 2.5,
            "struggling": 1.75,
            "partial_understanding": 0.75,
            "knows_well": 0.1,
        },
        "dropoff": False,
        "greedy_fill": True,
        "ratio_table": {"novice": 0.40, "intermediate": 0.40, "advanced": 0.40},
        "rule2_cap": True,
        "phase_b_max_tokens": 128000,
        "include_attempt_load": False,
    },
}


def get_absorption_preset(mode: str) -> Dict[str, Any]:
    """Return the preset dict for the given absorption mode."""
    if mode not in ABSORPTION_PRESETS:
        raise ValueError(
            f"Unknown absorption mode {mode!r}. "
            f"Valid modes: {sorted(ABSORPTION_PRESETS)}."
        )
    return ABSORPTION_PRESETS[mode]


def compute_overload_threshold(
    n_nodes: int,
    knowledge_level: str = "intermediate",
    mode: str = "default",
) -> int:
    """Per-conversation overload threshold given graph size, level, and mode.

    Floor of 2 keeps small graphs from collapsing the threshold to 0 or 1.
    """
    preset = get_absorption_preset(mode)
    ratio = preset["ratio_table"].get(knowledge_level, 0.40)
    return max(2, round(n_nodes * ratio))


def _compute_effective_load(
    advanceable_items: List[Dict[str, Any]],
    review_items: List[Dict[str, Any]],
    state_by_id: Dict[str, str],
    preset: Dict[str, Any],
) -> float:
    """Effective cognitive load for the turn under the given preset.

    "uniform_review" weighting: ``len(adv) * 1.0 + len(review) * review_weight``.
    "state_weighted" weighting: ``sum(W_STATE[state_of(iu)])`` over all mentioned IUs.
    """
    weighting = preset["weighting"]
    if weighting == "uniform_review":
        return (
            len(advanceable_items) * 1.0
            + len(review_items) * preset["review_weight"]
        )
    if weighting == "state_weighted":
        w_state = preset["W_STATE"]
        return sum(
            w_state.get(state_by_id.get(item.get("id"), "unaware"), 1.0)
            for item in advanceable_items + review_items
        )
    raise ValueError(f"Unknown absorption weighting {weighting!r}.")


def _compute_attempt_load(
    iu_analysis: List[Dict[str, Any]],
    state_by_id: Dict[str, str],
    preset: Dict[str, Any],
) -> float:
    """User-side cognitive load for the turn — sum over reasoning + articulation IUs.

    Returns 0.0 when ``preset["include_attempt_load"]`` is False (existing
    presets keep the old behavior of comprehension-only load).

    The reasoning weight scales with pre-state of the IU: reasoning about a
    shaky concept costs more working memory than reasoning about a solid one.
    Articulation is a flat constant since the cognitive cost of restating
    prior assistant content does not depend much on user mastery.
    """
    if not preset.get("include_attempt_load"):
        return 0.0
    w_reasoning = dict(preset.get("W_REASONING", {}))
    w_articulation = float(preset.get("W_ARTICULATION", 0.0))
    total = 0.0
    for item in iu_analysis:
        if not isinstance(item, dict):
            continue
        attempt_type = _resolve_attempt_type(item)
        iu_id = item.get("id")
        if attempt_type == "reasoning":
            total += w_reasoning.get(state_by_id.get(iu_id, "unaware"), 1.0)
        elif attempt_type == "articulation":
            total += w_articulation
    return total


def _resolve_attempt_type(update_info: Dict[str, Any]) -> str:
    """Return one of {"reasoning", "articulation", "none"} for a Phase B IU dict.

    New Phase B output emits ``user_attempt_type`` directly. Legacy logs
    only have the boolean ``attempted_by_user``; map True→"reasoning"
    (the old behavior — every attempt fired Rule 5) so replay tools keep
    working on old runs.
    """
    raw = update_info.get("user_attempt_type")
    if raw in ("reasoning", "articulation", "none"):
        return raw
    if update_info.get("attempted_by_user"):
        return "reasoning"
    return "none"


_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')

# Phase B schema-drift recovery: gemini-3-flash-preview occasionally typos
# field names or omits the categorical teaching_quality value (while still
# populating the reasoning prose). 2026-05-16 audit of new_pilot2 found 62
# entries with missing teaching_quality, 100 entries with typo'd field names
# (~0.6% of all IU entries combined).
_TQ_TRAILING_LABEL_RE = re.compile(
    r"(?:->|→)\s*(not_mentioned|shallow|well_explained)\b", re.I
)
_PHASE_B_FIELD_ALIASES = {
    "assistant_reasoning_role": "assistant_response_role",
    "assistant_reason_role": "assistant_response_role",
    "assistant_role": "assistant_response_role",
    "assistant_response": "assistant_response_role",
    "user_attempt": "user_attempt_type",
}


def _normalize_phase_b_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    """Defensive normalization of one Phase B IU entry.

    Two recoveries:
    1. Field-name typos: map gemini's drifted names to the canonical schema.
       Only fills the canonical key if it isn't already present, so legitimate
       data isn't overwritten.
    2. Missing teaching_quality: if the field is absent (74% of historical
       null cases match a 'Not mentioned' prose pattern; 26% end with a trailing
       arrow like '→ shallow'), recover via regex on
       ``teaching_quality_reasoning`` when possible; default to
       ``not_mentioned`` otherwise (the safe no-op). Previously the consumer
       fell through to the well_explained branch, silently advancing the IU.
    """
    if not isinstance(item, dict):
        return item
    for src, dst in _PHASE_B_FIELD_ALIASES.items():
        if src in item and dst not in item:
            item[dst] = item[src]
    if "teaching_quality" not in item:
        reasoning = (item.get("teaching_quality_reasoning") or "").lower()
        m = _TQ_TRAILING_LABEL_RE.search(reasoning)
        item["teaching_quality"] = m.group(1).lower() if m else "not_mentioned"
    return item


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Robust JSON-from-LLM extractor for Phase B / Phase A.

    Handles three classes of failure that gemini-3-flash-preview produces ~1% of
    the time, all of which silently corrupted IU classifications under the prior
    greedy-regex extractor:

    1. ``}}`` trailing junk — gemini appends a spurious second closing brace
       after a complete JSON object. Handled by brace-balanced, string-aware
       scanning that stops at the first matching ``}``.
    2. Invalid backslash escapes inside string values — gemini emits LaTeX such
       as ``"e^{2\\pi i k/9}"`` with single backslashes. ``\\p`` is not a valid
       JSON escape and ``json.loads`` rejects the whole document. Handled by
       double-escaping any ``\\X`` where ``X`` is not a valid JSON-escape char
       before re-parsing.
    3. Schema hallucination (gemini emits ``{"knowledge_areas": [...], "nodes":
       [...]}`` instead of ``{"iu_analysis": [...]}``) — NOT recoverable here;
       falls through to ``{}`` and Phase B treats the turn as no-op.
    """
    if not text:
        return {}
    # Strip a fenced ```json ... ``` wrapper if present
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    inner = fenced.group(1) if fenced else text
    # Walk from first '{' to its matching '}' using brace-depth + string-aware
    # scanning (handles the }} trailing-brace case).
    start = inner.find("{")
    if start < 0:
        return {}
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i in range(start, len(inner)):
        ch = inner[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return {}
    candidate = inner[start:end]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        # Salvage: sanitize invalid escape sequences (LaTeX-in-JSON) and retry.
        sanitized = _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", candidate)
        try:
            obj = json.loads(sanitized)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}


def _build_label_graph_from_concept_graph(
    concept_graph: Dict[str, List[Dict[str, Any]]],
    problem_id: str,
) -> Dict[str, Any]:
    items = concept_graph.get(str(problem_id), []) or []
    nodes = []
    edges = []
    for item in items:
        concept_id = item.get("concept_id")
        if not concept_id:
            continue
        nodes.append({"id": concept_id, "concept": concept_id})
    for item in items:
        concept_id = item.get("concept_id")
        prereqs = item.get("prerequisites", []) or []
        if not concept_id:
            continue
        for prereq in prereqs:
            edges.append({"from": prereq, "to": concept_id})
    return {"nodes": nodes, "edges": edges}


def _normalize_state(state: str) -> str:
    if not state:
        return "unaware"
    lowered = state.strip()
    alias_map = {
        "unknown_unknown": "unaware",
        "not_introduced": "unaware",
        "unaware": "unaware",
    }
    normalized = alias_map.get(lowered, lowered)
    return normalized if normalized in STATE_ORDER else "unaware"


def _state_index(state: str) -> int:
    return STATE_ORDER.index(_normalize_state(state))


def _reference_answer_for_phase_b(reference_answer: str) -> str:
    """Gold answer for analysts only; Phase B uses it to spot materially wrong 'well_explained' claims."""
    t = (reference_answer or "").strip()
    if not t:
        return (
            "(Not provided — assess teaching quality using the IU graph and assistant response only; "
            "do not infer a separate gold answer.)"
        )
    return t


def _format_current_state(iu_graph: Dict[str, Any], state_by_id: Dict[str, str]) -> str:
    formatted = {
        node.get("id", ""): {"state": state_by_id.get(node.get("id", ""), "unaware")}
        for node in iu_graph.get("nodes", [])
        if node.get("id")
    }
    return json.dumps(formatted, indent=2, ensure_ascii=True)


def _build_state_map(
    *,
    iu_graph: Dict[str, Any],
    knowledge_state: Dict[str, Any],
    id_map: Optional[Dict[str, str]],
) -> Dict[str, str]:
    id_to_label = id_map or {}
    state_by_id: Dict[str, str] = {}
    for node in iu_graph.get("nodes", []):
        iu_id = node.get("id")
        if not iu_id:
            continue
        label = id_to_label.get(iu_id, iu_id)
        raw = knowledge_state.get(label, {})
        state_str = raw.get("state", "unaware") if isinstance(raw, dict) else raw
        state_by_id[iu_id] = _normalize_state(state_str)
    return state_by_id


def _get_neighbors(edges: List[Dict[str, Any]], iu_id: str) -> List[str]:
    neighbors = set()
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src == iu_id and dst:
            neighbors.add(dst)
        if dst == iu_id and src:
            neighbors.add(src)
    return sorted(neighbors)


def _get_prereqs(edges: List[Dict[str, Any]], iu_id: str) -> List[str]:
    return [edge.get("from") for edge in edges if edge.get("to") == iu_id and edge.get("from")]


def _is_known(state: str) -> bool:
    return _normalize_state(state) in {"partial_understanding", "knows_well"}


def _compute_topo_depth(edges: List[Dict[str, Any]], all_node_ids: List[str]) -> Dict[str, int]:
    """Return topological depth (longest path from any root) for each node.

    Root nodes (no incoming edges) have depth 0. Preserves IU graph insertion
    order for nodes at the same depth (used as tiebreaker in overload cutoff).
    """
    in_edges: Dict[str, List[str]] = {n: [] for n in all_node_ids}
    for edge in edges:
        src, dst = edge.get("from"), edge.get("to")
        if src and dst and dst in in_edges:
            in_edges[dst].append(src)

    depth: Dict[str, int] = {}

    def _depth(node: str, visiting: set) -> int:
        if node in depth:
            return depth[node]
        if node in visiting:
            return 0  # cycle guard
        visiting.add(node)
        parents = in_edges.get(node, [])
        d = 0 if not parents else 1 + max(_depth(p, visiting) for p in parents)
        depth[node] = d
        return d

    for node in all_node_ids:
        _depth(node, set())
    return depth


def _compute_base_ceiling(prereq_states: List[str]) -> str:
    """Rule 1: Prerequisite-based ceiling.

    All prerequisites knows_well           → ceiling = knows_well
    Any prerequisite partial_understanding → ceiling = partial_understanding
    Any prerequisite struggling or unaware → ceiling = struggling
    Root nodes (no prerequisites)          → ceiling = knows_well
    """
    if not prereq_states:
        return "knows_well"
    weakest_idx = min(_state_index(state) for state in prereq_states)
    weakest = STATE_ORDER[weakest_idx]
    if weakest in {"unaware", "struggling"}:
        return "struggling"
    if weakest == "partial_understanding":
        return "partial_understanding"
    return "knows_well"


def _compute_new_state(
    teaching_quality: str,
    previous_state: str,
    ceiling: str,
    *,
    user_attempt_type: str = "none",
    assistant_response_role: str = "not_applicable",
    overload: bool = False,
) -> str:
    """Compute new state using the 5-rule priority order.

    Rule 5 (user attempt) → Rule 4 (teaching_quality) → Rule 1 (ceiling) → Rule 2 (overload cap) → Rule 3 (no downgrade)

    teaching_quality values: not_mentioned | shallow | well_explained
    user_attempt_type values: reasoning | articulation | none
        Only "reasoning" fires Rule 5 — the user must produce new inference
        about the IU, not echo prior assistant content. "articulation" falls
        through to the teaching_quality path so parroted concepts only
        advance when the assistant teaches them in this turn.
    assistant_response_role values:
        confirmed_correct | confirmed_incorrect_no_explanation |
        confirmed_incorrect_with_explanation | ignored | not_applicable
    """
    prev_idx = _state_index(previous_state)
    ceil_idx = _state_index(ceiling)

    # pilot3_20260515: minimal_rules mode — drop Rule 2 (overload cap),
    # Rule 5 (reasoning jump), and ALT-A (unaware → +1 cap). Keep only
    # Rule 1 (ceiling, already computed) + Rule 3 (monotonicity below) +
    # simplified Rule 4 (teaching_quality drives transitions).
    if get_flags().ks_update.minimal_rules:
        if teaching_quality == "not_mentioned":
            return previous_state
        _ks_flags = get_flags().ks_update
        if teaching_quality == "shallow":
            # pilot4: optionally restrict shallow to only unlock the bottom
            # transition (unaware → struggling). Above that, shallow is a no-op.
            if _ks_flags.shallow_unaware_only and previous_state != "unaware":
                return previous_state
            advance = 1
        else:  # well_explained
            # pilot4: optionally cap well_explained at +1 (halves comp's
            # per-mention amplitude advantage). Default minimal_rules is +2.
            advance = 1 if _ks_flags.cap_well_explained_at_one else 2
        new_idx = min(prev_idx + advance, ceil_idx)
        return STATE_ORDER[max(new_idx, prev_idx)]  # Rule 3 monotonicity

    # Rule 5: a user *reasoning* attempt overrides teaching_quality signals.
    # Articulation does not fire Rule 5 — it is handled by the teaching_quality path.
    #
    # Behavior is controlled by ``feature_flags.ks_update.rule5``:
    #   - "off"             — Rule 5 disabled; reasoning falls through to Rule 4.
    #   - "struggling_gate" — Rule 5 fires at struggling+ (legacy GATE5).
    #   - "partial_gate"    — Rule 5 fires only at partial+ (GATE5+, recommended).
    # Reasoning below the gate falls through to Rule 4 — prevents the "Phase B
    # leak" where the simulated user fluently reasons about concepts it
    # shouldn't know yet.
    _rule5_mode = get_flags().ks_update.rule5
    _gate5_threshold_state = (
        "partial_understanding" if _rule5_mode == "partial_gate" else "struggling"
    )
    _rule5_enabled = _rule5_mode != "off"
    if _rule5_enabled and user_attempt_type == "reasoning" and prev_idx >= _state_index(_gate5_threshold_state):
        if assistant_response_role == "ignored":
            return previous_state
        elif assistant_response_role == "confirmed_correct":
            # Jump to knows_well, but still respect prerequisite ceiling
            return STATE_ORDER[min(_state_index("knows_well"), ceil_idx)]
        elif assistant_response_role == "confirmed_incorrect_no_explanation":
            # Realising you're wrong without guidance → struggling
            # Exception: knows_well is treated as a slip, no downgrade
            if previous_state == "knows_well":
                return previous_state
            return "struggling"
        elif assistant_response_role == "confirmed_incorrect_with_explanation":
            # Corrected with explanation — treat as well_explained (+2), subject to overload
            teaching_quality = "well_explained"
        # fall through for confirmed_incorrect_with_explanation using well_explained path

    # Rule 4: teaching_quality determines advance
    if teaching_quality == "not_mentioned":
        return previous_state

    if teaching_quality == "shallow":
        # Extra ceiling cap: if user is at or below struggling, cap at struggling
        effective_ceil_idx = ceil_idx
        if prev_idx <= _state_index("struggling"):
            effective_ceil_idx = min(ceil_idx, _state_index("struggling"))
        new_idx = min(prev_idx + 1, effective_ceil_idx)
    else:  # well_explained
        base_advance = 2
        # ALT-A: encoding ≠ consolidation — a single broadcast explanation
        # produces initial awareness, not understanding. Cap advance from
        # unaware at +1 (to struggling) regardless of overload.
        if prev_idx == 0:  # unaware
            base_advance = 1
        # Rule 2: overload caps well_explained advance to +1
        elif overload:
            base_advance = 1
        new_idx = min(prev_idx + base_advance, ceil_idx)

    # Rule 3: no downgrade (except for confirmed_incorrect_no_explanation handled above)
    return STATE_ORDER[max(new_idx, prev_idx)]


def compute_overall_density_from_state(
    *,
    iu_graph: Dict[str, Any],
    knowledge_state: Dict[str, Any],
    id_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    state_by_id = _build_state_map(
        iu_graph=iu_graph,
        knowledge_state=knowledge_state,
        id_map=id_map,
    )
    edges = iu_graph.get("edges", []) or []
    nodes = [n.get("id") for n in iu_graph.get("nodes", []) if n.get("id")]

    densities: List[float] = []
    for iu_id in nodes:
        neighbors = _get_neighbors(edges, iu_id)
        if not neighbors:
            density = 1.0
        else:
            known_neighbor_count = len(
                [n for n in neighbors if _is_known(state_by_id.get(n, "unaware"))]
            )
            density = known_neighbor_count / len(neighbors)
        densities.append(density)

    overall_density = sum(densities) / len(densities) if densities else 1.0
    if overall_density >= 0.6:
        articulation_mode = "Explicit"
    elif overall_density >= 0.3:
        articulation_mode = "Vague"
    else:
        articulation_mode = "Compliant"

    return {"overall_density": overall_density, "articulation_mode": articulation_mode}


def compute_gap_sum_from_state(
    *,
    iu_graph: Dict[str, Any],
    knowledge_state: Dict[str, Any],
    id_map: Optional[Dict[str, str]] = None,
    response_iu_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Kept for backward compatibility with conversation.py metrics reporting."""
    if response_iu_ids is None:
        return {"gap_sum": 0.0, "avg_gap": 0.0}
    state_by_id = _build_state_map(
        iu_graph=iu_graph,
        knowledge_state=knowledge_state,
        id_map=id_map,
    )
    edges = iu_graph.get("edges", []) or []
    prereq_map: Dict[str, List[str]] = {}
    for edge in edges:
        dst = edge.get("to")
        src = edge.get("from")
        if dst and src:
            prereq_map.setdefault(dst, []).append(src)

    def _state_level(state: str) -> int:
        mapping = {"unaware": 0, "struggling": 1, "partial_understanding": 2, "knows_well": 3}
        return mapping.get(_normalize_state(state), 0)

    def _nearest_known_ancestor(start_id: str) -> tuple[int, int]:
        visited = set()
        queue = [(start_id, 0)]
        while queue:
            node_id, dist = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            if node_id != start_id and _state_level(state_by_id.get(node_id, "unaware")) >= 2:
                return dist, _state_level(state_by_id.get(node_id, "unaware"))
            for parent in prereq_map.get(node_id, []):
                if parent not in visited:
                    queue.append((parent, dist + 1))
        return 0, 0

    response_set = {iu_id for iu_id in response_iu_ids if iu_id}
    nodes = [n_id for n_id in response_set if n_id in state_by_id]
    gaps: List[float] = []
    for iu_id in nodes:
        if _is_known(state_by_id.get(iu_id, "unaware")):
            continue
        hop_dist, anchor_level = _nearest_known_ancestor(iu_id)
        gap = float(hop_dist) + float(3 - anchor_level)
        gaps.append(gap)

    gap_sum = sum(gaps)
    avg_gap = gap_sum / len(gaps) if gaps else 0.0
    return {"gap_sum": gap_sum, "avg_gap": avg_gap}


async def update_dynamic_knowledge_state(
    *,
    assistant_message: str,
    user_message: str,
    attempt_verify: bool = False,
    knowledge_state: Dict[str, Any],
    model_client: Any,
    iu_graph: Optional[Dict[str, Any]] = None,
    id_map: Optional[Dict[str, str]] = None,
    concept_graph: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    problem_id: Optional[str] = None,
    reference_answer: str = "",
    max_tokens: int = 800,
    show_progress: bool = False,
    return_metrics: bool = False,
    overload_threshold: int = 5,
    turn_number: int = 1,
    absorption_mode: str = "default",
    conversation_history: str = "",
    temporal_decay: bool = False,
    max_turns: int = 15,
) -> "Dict[str, Any] | tuple[Dict[str, Any], Dict[str, Any]]":
    if iu_graph is None:
        if concept_graph and problem_id:
            iu_graph = _build_label_graph_from_concept_graph(concept_graph, str(problem_id))
        else:
            return knowledge_state

    preset = get_absorption_preset(absorption_mode)

    # ── Phase A (code only) ──────────────────────────────────────────────────
    # 1. Build state map
    state_by_id = _build_state_map(
        iu_graph=iu_graph,
        knowledge_state=knowledge_state,
        id_map=id_map,
    )

    all_nodes = iu_graph.get("nodes", []) or []
    node_map = {n.get("id"): n.get("concept", "") for n in all_nodes}
    edges = iu_graph.get("edges", []) or []

    # 2. Classify known / struggling / unaware
    # known     = partial_understanding + knows_well → familiarity
    # struggling = user is aware but confused → contributes to novelty (user is curious)
    # unaware   = user doesn't know these exist → excluded from novelty
    #             (Curious U: attention weight = 0 for unaware concepts)
    known_iu_ids: List[str] = []
    struggling_iu_ids: List[str] = []  # aware-but-unknown; drives genuine curiosity
    for node in all_nodes:
        iu_id = node.get("id")
        if not iu_id:
            continue
        state = _normalize_state(state_by_id.get(iu_id, "unaware"))
        if _is_known(state):
            known_iu_ids.append(iu_id)
        elif state == "struggling":
            struggling_iu_ids.append(iu_id)
        # unaware: intentionally excluded from both lists

    # 3. Compute neighbor density for articulation mode (user prompt only, not absorption)
    densities: List[float] = []
    for node in all_nodes:
        iu_id = node.get("id")
        if not iu_id:
            continue
        neighbors = _get_neighbors(edges, iu_id)
        total_neighbors = len(neighbors)
        if total_neighbors == 0:
            density = 1.0
        else:
            density = len([n for n in neighbors if _is_known(state_by_id.get(n, "unaware"))]) / total_neighbors
        densities.append(density)

    overall_density = sum(densities) / len(densities) if densities else 1.0
    if overall_density >= 0.6:
        articulation_mode = "Explicit"
    elif overall_density >= 0.3:
        articulation_mode = "Vague"
    else:
        articulation_mode = "Compliant"

    # 4. Point 1: mutable CURRENT_states for within-turn cascading ceiling updates.
    # Prerequisites updated earlier in the topological loop are immediately visible
    # when computing ceilings for their dependents in the same turn.
    CURRENT_states: Dict[str, str] = dict(state_by_id)

    # ── Phase B (single LLM call) ────────────────────────────────────────────
    # Point 3: unified prompt always receives user_message + full 6-field output schema
    phase_b_template = load_prompt("simulation/prompts/knowledge_state/knowledge_state_update_phase_b_unified.md")
    # Rename top-level IU graph keys before serializing into the prompt so that
    # gemini-3-flash-preview does not echo the input structure (`nodes`/`edges`/
    # `knowledge_areas`) into its output — a ~17%-of-error-cases bug we saw on
    # new_pilot1 where gemini emitted `{"knowledge_areas":[...],"nodes":[...]}`
    # instead of `{"iu_analysis":[...]}`. The prompt body itself never references
    # these keys, so renaming is invisible to the model semantically.
    _iu_graph_for_prompt = {
        ("concept_areas" if k == "knowledge_areas"
         else "concepts" if k == "nodes"
         else "prerequisite_pairs" if k == "edges"
         else k): v
        for k, v in (iu_graph or {}).items()
    }
    phase_b_prompt_values = {
        "iu_graph": json.dumps(_iu_graph_for_prompt, indent=2, ensure_ascii=True),
        "current_state": _format_current_state(iu_graph, state_by_id),
        "conversation_history": (
            conversation_history.strip()
            if conversation_history and conversation_history.strip()
            else "(this is the first turn — no prior conversation)"
        ),
        "user_message": user_message,
        "assistant_response": assistant_message,
        "reference_answer": _reference_answer_for_phase_b(reference_answer),
    }
    phase_b_prompt = phase_b_template
    for _key, _val in phase_b_prompt_values.items():
        phase_b_prompt = phase_b_prompt.replace("{" + _key + "}", str(_val))
    # Phase B retry: a well-formed response always labels every IU, so an empty
    # iu_analysis means the call failed (gemini-3-flash malformed JSON, or a
    # transient empty response). Re-call before falling back to an empty no-op
    # turn — attempt 0 stays deterministic (temp=0); retries use temp>0 so the
    # model produces a genuinely different, hopefully parseable response.
    _PHASE_B_MAX_ATTEMPTS = 3
    phase_b_raw = ""
    iu_analysis = []
    for _phase_b_attempt in range(_PHASE_B_MAX_ATTEMPTS):
        phase_b_responses = await model_client.generate_responses(
            [[{"role": "system", "content": "You are a careful learning assessment analyst."},
              {"role": "user", "content": phase_b_prompt}]],
            temperature=0.0 if _phase_b_attempt == 0 else 0.4,
            max_tokens=max_tokens,
            n=1,
            show_progress=show_progress,
        )
        phase_b_raw = phase_b_responses[0][0] if phase_b_responses and phase_b_responses[0] else ""
        phase_b_parsed = _extract_json_object(phase_b_raw)
        iu_analysis = phase_b_parsed.get("iu_analysis", []) if isinstance(phase_b_parsed, dict) else []
        # Normalize each entry: alias typo'd field names and salvage missing
        # teaching_quality (see _normalize_phase_b_entry docstring).
        iu_analysis = [_normalize_phase_b_entry(item) for item in iu_analysis if isinstance(item, dict)]
        if iu_analysis:
            break  # parse succeeded — use it
    # All attempts exhausted with an empty parse → fall back to a no-op turn.
    # Surface it loudly so we notice in the error log instead of shrugging.
    if not iu_analysis and phase_b_raw and len(phase_b_raw) > 200:
        import logging as _logging
        _logging.getLogger("sim_error").warning(
            "Phase B parse fell back to empty iu_analysis after %d attempts — "
            "last raw response was %d chars long.",
            _PHASE_B_MAX_ATTEMPTS, len(phase_b_raw),
        )
    updates_by_id = {item.get("id"): item for item in iu_analysis if isinstance(item, dict) and item.get("id")}

    # ── Post-processing (code) ───────────────────────────────────────────────
    # 1. Identify IUs in response and compute engagement score
    ius_in_response = [
        item for item in iu_analysis
        if isinstance(item, dict) and item.get("teaching_quality") != "not_mentioned"
    ]
    response_ids = [item.get("id") for item in ius_in_response if item.get("id")]

    known_id_set = set(known_iu_ids)
    total_count = len(ius_in_response)  # kept for IC/PO metrics only
    # familiarity_term and novelty_term deferred to after allowed_ids is computed so
    # both terms use only the absorbed subset (concepts not cut off by overload).

    # 2. Rule 2: information overload — cognitive load scaled by preset.
    # Under the "uniform_review" weighting, advanceable IUs cost 1.0 and
    # review (knows_well) IUs cost preset["review_weight"]. Under
    # "state_weighted" the cost is W_STATE[state] per mentioned IU.
    mentioned_items = [
        item for item in iu_analysis
        if isinstance(item, dict) and item.get("teaching_quality") != "not_mentioned"
        and item.get("id")
    ]
    advanceable_items = [
        item for item in mentioned_items
        if state_by_id.get(item.get("id"), "unaware") != "knows_well"
    ]
    review_items = [
        item for item in mentioned_items
        if state_by_id.get(item.get("id"), "unaware") == "knows_well"
    ]
    comprehension_load = _compute_effective_load(
        advanceable_items, review_items, state_by_id, preset,
    )
    # Attempt-load addend (0.0 unless preset opts in via include_attempt_load).
    # Adds user-side cognitive cost — reasoning + articulation — into the
    # same effective_load that drives the dropoff. This unified-overload
    # design scales BOTH absorbed advances and Rule-5 advances under one
    # shared signal.
    attempt_load = _compute_attempt_load(iu_analysis, state_by_id, preset)
    effective_load = comprehension_load + attempt_load
    overload = effective_load >= overload_threshold

    # When overloaded, priority = fewest unmet prerequisites among advanceable concepts.
    # knows_well concepts pass through unconditionally (they cannot advance further).
    # Unmet prerequisite: a prereq concept the user does not yet know (not partial_understanding
    # or knows_well). Tiebreaker 1: topological depth (shallower first). Tiebreaker 2: IU graph
    # insertion order. Only the first overload_threshold advanceable concepts are allowed to
    # advance; the rest get zero absorption.
    topo_depth = _compute_topo_depth(edges, [n.get("id") for n in all_nodes if n.get("id")])
    graph_order = {n.get("id"): idx for idx, n in enumerate(all_nodes) if n.get("id")}

    def _unmet_prereq_count(iu_id: str) -> int:
        return sum(
            1 for p in _get_prereqs(edges, iu_id)
            if not _is_known(state_by_id.get(p, "unaware"))
        )

    # Absorption efficiency (dropoff mode only). Turn-level constant: 1.0 at
    # or below threshold, decays above. Applied later as a per-IU
    # step-downgrade multiplier on upward state transitions, replacing the
    # binary cutoff that the default mode uses.
    # Two curves: "linear" (default, reaches 0 at 2× threshold) and
    # "saturating" (threshold / load, never reaches 0).
    if preset["dropoff"] and overload_threshold > 0 and effective_load > overload_threshold:
        dropoff_curve = preset.get("dropoff_curve", "linear")
        if dropoff_curve == "saturating":
            absorption_efficiency = overload_threshold / effective_load
        elif dropoff_curve == "sigmoid":
            # Smooth above-threshold dropoff: continuous at threshold (eff=1.0)
            # and decays as 1/(1+x²) with x = (eff-thr)/(thr*k). Default k=1.0
            # (pilot1+): at eff=1.5·thr → 0.8, 2·thr → 0.5, 3·thr → 0.2.
            # pilot5: sigmoid_softening k>1.0 widens the curve; at k=2.0
            # the same loads give 0.94, 0.80, 0.50 respectively.
            softening = max(1e-6, get_flags().ks_update.sigmoid_softening)
            overage = (effective_load - overload_threshold) / (overload_threshold * softening)
            absorption_efficiency = 1.0 / (1.0 + overage ** 2.0)
        else:
            overage_ratio = (effective_load - overload_threshold) / overload_threshold
            absorption_efficiency = max(0.0, 1.0 - overage_ratio)
    else:
        absorption_efficiency = 1.0

    # Temporal decay: models working-memory fatigue over turns.
    # efficiency *= max_turns / (max_turns + t - 1). No new hyperparameters.
    if preset.get("temporal_decay") and temporal_decay and max_turns > 0:
        absorption_efficiency *= max_turns / (max_turns + turn_number - 1)

    # pilot7: per-IU graduated dropoff. Reuses the same sigmoid formula but
    # applied at cumulative-cost positions in a W_STATE-descending priority
    # walk. Stacked multiplicatively with the turn-level absorption_efficiency
    # (which acts as the global fatigue layer).
    # pilot11: per_iu_hinge (takes precedence over per_iu_dropoff) — within-turn
    # hard breadth cap. Top-N priority IUs get full eff; rest get tail_factor.
    # N = max(1, round(hinge_threshold_ratio * graph_size)) scales with problem.
    per_iu_efficiency: Dict[str, float] = {}
    _ks_flags = get_flags().ks_update
    if (_ks_flags.per_iu_hinge
        and preset["dropoff"]
        and overload_threshold > 0):
        w_state_dict = preset.get("W_STATE", {})
        graph_size = len(all_nodes) if all_nodes else 0
        n_keep = max(1, round(_ks_flags.hinge_threshold_ratio * graph_size))
        tail = _ks_flags.hinge_tail_factor
        priority_sorted = sorted(
            mentioned_items,
            key=lambda item: (
                -w_state_dict.get(_normalize_state(state_by_id.get(item.get("id", ""), "unaware")), 1.0),
                topo_depth.get(item.get("id", ""), 0),
                graph_order.get(item.get("id", ""), 0),
            ),
        )
        for idx, item in enumerate(priority_sorted):
            iu_id_local = item.get("id")
            if not iu_id_local:
                continue
            per_iu_eff_value = 1.0 if idx < n_keep else tail
            # Stack with turn-level fatigue (same convention as per_iu_dropoff)
            per_iu_efficiency[iu_id_local] = per_iu_eff_value * absorption_efficiency
    elif (preset["dropoff"]
        and overload_threshold > 0
        and _ks_flags.per_iu_dropoff
        and absorption_efficiency < 1.0):
        w_state_dict = preset.get("W_STATE", {})
        softening = max(1e-6, _ks_flags.sigmoid_softening)
        # Priority sort: high-W IUs (most-needy) first. Tiebreak by topo depth then
        # graph order so the walk is deterministic across IUs with same W.
        priority_sorted = sorted(
            mentioned_items,
            key=lambda item: (
                -w_state_dict.get(_normalize_state(state_by_id.get(item.get("id", ""), "unaware")), 1.0),
                topo_depth.get(item.get("id", ""), 0),
                graph_order.get(item.get("id", ""), 0),
            ),
        )
        cumulative_cost = 0.0
        for item in priority_sorted:
            iu_id_local = item.get("id")
            if not iu_id_local:
                continue
            state = _normalize_state(state_by_id.get(iu_id_local, "unaware"))
            cost = w_state_dict.get(state, 1.0)
            cumulative_cost += cost
            if cumulative_cost > overload_threshold:
                overage_local = (cumulative_cost - overload_threshold) / (overload_threshold * softening)
                per_iu_eff_value = 1.0 / (1.0 + overage_local ** 2.0)
            else:
                per_iu_eff_value = 1.0
            # Stack: per-IU × turn-level. Stored value is the combined
            # efficiency the dropoff loop should use for this IU.
            per_iu_efficiency[iu_id_local] = per_iu_eff_value * absorption_efficiency

    # Helper: per-concept W_STATE weight (used by greedy_fill).
    def _concept_weight(iu_id: str) -> float:
        w_state = preset.get("W_STATE", {})
        return w_state.get(_normalize_state(state_by_id.get(iu_id, "unaware")), 1.0)

    advanceable_sorted = sorted(
        advanceable_items,
        key=lambda item: (
            _unmet_prereq_count(item.get("id", "")),
            topo_depth.get(item.get("id", ""), 0),
            graph_order.get(item.get("id", ""), 0),
        ),
    )

    if preset["dropoff"]:
        # Dropoff path: every mentioned IU absorbs; the efficiency multiplier
        # scales the upward state step in the update loop below. Degradation
        # is graded rather than binary, so no IU is cut off entirely.
        allowed_ids = {item["id"] for item in mentioned_items}
    elif preset.get("greedy_fill") and overload:
        # Greedy fill: walk advanceable concepts in readiness order,
        # accumulate W_STATE weight, stop when cumulative weight exceeds threshold.
        running_weight = 0.0
        absorbed_advanceable_ids: set = set()
        for item in advanceable_sorted:
            w = _concept_weight(item["id"])
            if running_weight + w > overload_threshold:
                break
            absorbed_advanceable_ids.add(item["id"])
            running_weight += w
        allowed_ids = absorbed_advanceable_ids | {item["id"] for item in review_items}
    elif overload:
        # Default cutoff path: count-based decay under overload.
        decay = overload_threshold / effective_load if effective_load > 0 else 1.0
        n_within = min(len(advanceable_sorted), overload_threshold)
        n_absorbing = int(n_within * decay)  # floor
        allowed_ids = (
            {item["id"] for item in advanceable_sorted[:n_absorbing]}
            | {item["id"] for item in review_items}
        )
    else:
        allowed_ids = {item["id"] for item in mentioned_items}

    # Absorbed IUs: concepts not cut off by overload.  When there is no overload,
    # absorbed == all mentioned, so engagement behaviour is unchanged.
    absorbed_ius_in_response = [iu for iu in ius_in_response if iu.get("id") in allowed_ids]
    absorbed_count = len(absorbed_ius_in_response)

    known_ius_absorbed = [iu for iu in absorbed_ius_in_response if iu.get("id") in known_id_set]
    familiarity_term = len(known_ius_absorbed) / absorbed_count if absorbed_count > 0 else 0.0

    # 2.5 Strict IC/PO metric counters (definition-aligned).
    # - explained_ids: concepts mentioned this turn (teaching_quality != not_mentioned)
    # - information_calibration_numerator/denominator:
    #   among explained_ids, count those whose final ceiling reaches at least partial_understanding.
    # - new_ius/disallowed:
    #   among explained_ids with previous state in {unaware, struggling},
    #   disallowed if final ceiling stays below partial_understanding.
    explained_ids: List[str] = [item["id"] for item in mentioned_items if item.get("id")]
    calibration_denominator = len(explained_ids)
    partial_idx = _state_index("partial_understanding")
    # IC numerator: "just right" concepts = new to user (< knows_well) AND prerequisite-reachable.
    # This absorbs the prerequisite-readiness signal: explaining concepts the user can't reach
    # (too_hard) or already knows (too_easy) both reduce IC.
    calibration_numerator = sum(
        1
        for iu_id in explained_ids
        if _state_index(_compute_base_ceiling(
            [CURRENT_states.get(pid, "unaware") for pid in _get_prereqs(edges, iu_id)]
        )) >= partial_idx
        and _normalize_state(CURRENT_states.get(iu_id, "unaware")) != "knows_well"
    )
    # ZPD size: count of all IUs in the graph that are learnable this turn
    # (not yet knows_well AND prerequisite ceiling >= partial_understanding).
    zpd_size = sum(
        1
        for node in all_nodes
        if node.get("id")
        and _normalize_state(CURRENT_states.get(node["id"], "unaware")) != "knows_well"
        and _state_index(_compute_base_ceiling(
            [CURRENT_states.get(pid, "unaware") for pid in _get_prereqs(edges, node["id"])]
        )) >= partial_idx
    )
    new_ius = 0
    disallowed = 0
    for iu_id in explained_ids:
        previous_state = _normalize_state(CURRENT_states.get(iu_id, "unaware"))
        final_ceiling = _compute_base_ceiling(
            [CURRENT_states.get(pid, "unaware") for pid in _get_prereqs(edges, iu_id)]
        )
        if previous_state in {"unaware", "struggling"}:
            new_ius += 1
            if _state_index(final_ceiling) < partial_idx:
                disallowed += 1

    # Volume-based overload: proportion of effective cognitive load that exceeds the threshold.
    # Replaces the old prereq-gap PO (now folded into IC); this directly captures
    # "how much did the assistant exceed working memory capacity this turn?"
    perceived_overload_turn = (
        max(0.0, effective_load - overload_threshold) / effective_load
        if effective_load > 0 else 0.0
    )

    # 3. Apply state updates using 5-rule pipeline
    # Point 1: process IUs in topological order so each IU's ceiling reflects
    # prerequisites already updated earlier in this same turn.
    updated_state = copy.deepcopy(knowledge_state)
    well_explained_ids: List[str] = []
    sorted_nodes_for_update = sorted(
        all_nodes,
        key=lambda n: topo_depth.get(n.get("id", ""), 0),
    )
    for node in sorted_nodes_for_update:
        iu_id = node.get("id")
        if not iu_id:
            continue
        label = id_map.get(iu_id, iu_id) if id_map else iu_id
        previous_state = CURRENT_states.get(iu_id, "unaware")
        # Point 1: live ceiling from CURRENT_states (reflects just-updated prerequisites)
        prereq_ids = _get_prereqs(edges, iu_id)
        ceiling = _compute_base_ceiling([CURRENT_states.get(pid, "unaware") for pid in prereq_ids])
        update_info = updates_by_id.get(iu_id, {})

        teaching_quality = update_info.get("teaching_quality", "not_mentioned")
        # Point 3: always read attempt fields — unified schema always provides them
        user_attempt_type = _resolve_attempt_type(update_info)
        assistant_response_role = update_info.get("assistant_response_role", "not_applicable")

        # Rule 2: advanceable concepts beyond the overload cutoff get zero absorption.
        # knows_well concepts are always in allowed_ids and pass through unchanged.
        if overload and iu_id not in allowed_ids:
            teaching_quality = "not_mentioned"

        proposed_state = _compute_new_state(
            teaching_quality=teaching_quality,
            previous_state=previous_state,
            ceiling=ceiling,
            user_attempt_type=user_attempt_type,
            assistant_response_role=assistant_response_role,
            # Rule-2 cap can be suppressed by the active preset (e.g.
            # weightabsorb suppresses it because the dropoff multiplier
            # already scales the advance — applying the cap would
            # double-penalize).
            overload=overload and preset["rule2_cap"],
        )

        # Dropoff step-downgrade: scale upward-step magnitude by the
        # turn-level absorption_efficiency.
        # - Default (rounding): actual_steps = max(0, round(intended × efficiency))
        #   creates a rounding cliff: single-step intended advances with
        #   efficiency < 0.5 round to 0.
        # - pilot3_20260515 (fractional_accumulation): per-IU progress
        #   accumulator carries the unused fractional part to next turn.
        # Downgrades (intended ≤ 0) are left alone — Rule-3 monotonicity below.
        carried_progress = 0.0  # fractional remainder to persist (only used in fractional mode)
        if preset["dropoff"] and absorption_efficiency < 1.0:
            # pilot7: if per_iu_dropoff is on, look up this IU's combined
            # efficiency (per_iu × turn_eff); else use turn-level uniformly.
            effective_efficiency = per_iu_efficiency.get(iu_id, absorption_efficiency)
            prev_idx_dd = _state_index(previous_state)
            proposed_idx_dd = _state_index(proposed_state)
            if proposed_idx_dd > prev_idx_dd:
                intended_steps = proposed_idx_dd - prev_idx_dd
                if get_flags().ks_update.fractional_accumulation:
                    # Read prior carryover from knowledge_state input
                    prior_info = knowledge_state.get(label, {}) if isinstance(knowledge_state, dict) else {}
                    if not isinstance(prior_info, dict):
                        prior_info = {}
                    prior_progress = float(prior_info.get("fractional_progress", 0.0) or 0.0)
                    total_progress = prior_progress + intended_steps * effective_efficiency
                    actual_steps = int(total_progress)  # floor (Python int truncates)
                    # pilot4: optionally reset the accumulator on upward
                    # transition so leftover progress doesn't seed the next
                    # state's quota; otherwise persist the remainder.
                    if actual_steps > 0 and get_flags().ks_update.reset_fractional_on_transition:
                        carried_progress = 0.0
                    else:
                        carried_progress = max(0.0, total_progress - actual_steps)
                else:
                    actual_steps = max(0, round(intended_steps * effective_efficiency))
                capped_idx = min(prev_idx_dd + actual_steps, len(STATE_ORDER) - 1)
                proposed_state = STATE_ORDER[capped_idx]

        if label not in updated_state:
            updated_state[label] = {}
        updated_state[label]["state"] = proposed_state
        updated_state[label]["evidence"] = update_info.get("teaching_quality_reasoning", "")
        if get_flags().ks_update.fractional_accumulation:
            updated_state[label]["fractional_progress"] = carried_progress

        # Point 1: immediately propagate new state so dependents see the updated ceiling
        CURRENT_states[iu_id] = proposed_state

        if teaching_quality == "well_explained":  # uses post-override value for consistency
            well_explained_ids.append(iu_id)

    # 4. Rule 3: final monotonicity pass — exempt confirmed_incorrect_no_explanation downgrades
    for node in all_nodes:
        iu_id = node.get("id")
        if not iu_id:
            continue
        label = id_map.get(iu_id, iu_id) if id_map else iu_id
        update_info = updates_by_id.get(iu_id, {})
        # Skip monotonicity for deliberate setbacks (user got corrected with no explanation).
        # Only reasoning attempts can produce a setback — articulation falls through to the
        # teaching_quality path which has no downgrade, so the exemption doesn't apply there.
        if (
            _resolve_attempt_type(update_info) == "reasoning"
            and update_info.get("assistant_response_role") == "confirmed_incorrect_no_explanation"
            and state_by_id.get(iu_id, "unaware") != "knows_well"
        ):
            continue
        orig_info = knowledge_state.get(label, {})
        orig_raw = orig_info if isinstance(orig_info, dict) else {"state": orig_info}
        orig_state = _normalize_state(orig_raw.get("state", "unaware"))
        new_info = updated_state.get(label, {})
        new_raw = new_info if isinstance(new_info, dict) else {"state": new_info}
        new_state = _normalize_state(new_raw.get("state", "unaware"))
        if _state_index(new_state) < _state_index(orig_state):
            if not isinstance(updated_state.get(label), dict):
                updated_state[label] = {}
            updated_state[label]["state"] = orig_state

    # 4.5 Absorbed-only knowledge gain: count state advances only for IUs in allowed_ids.
    # Overloaded concepts that were capped (not truly processed) are excluded, consistent
    # with the absorbed-only engagement calculation.
    absorbed_gain_this_turn = 0.0
    n_upward_transitions = 0
    for node in all_nodes:
        iu_id = node.get("id")
        if not iu_id or iu_id not in allowed_ids:
            continue
        label_for_gain = id_map.get(iu_id, iu_id) if id_map else iu_id
        prev_score = _state_index(state_by_id.get(iu_id, "unaware"))
        new_state_str = _normalize_state((updated_state.get(label_for_gain) or {}).get("state", "unaware"))
        new_score = _state_index(new_state_str)
        absorbed_gain_this_turn += max(0.0, float(new_score - prev_score))
        if new_score > prev_score:
            n_upward_transitions += 1

    # 4.5b Delivery-Calibration numerator |A_t|, per the metric's definition:
    #   A_t = { v in E_t : s_{t-1}(v) < knows_well
    #                      and ceil_{t-1}(v) >= partial_understanding
    #                      and s_t(v) > s_{t-1}(v) }
    # Every condition reads the state entering the turn (``state_by_id``), not
    # the mid-turn cascading snapshot ``CURRENT_states`` that the topo-sort
    # update mutates as it goes. Storing the count here means the aggregate no
    # longer falls back to min(ic_numerator, n_upward_transitions) — a bound
    # that cannot tell an in-ZPD advance from a non-ZPD one.
    post_turn_states_by_id = {
        node["id"]: _normalize_state(
            (updated_state.get(id_map.get(node["id"], node["id"]) if id_map else node["id"]) or {})
            .get("state", "unaware")
        )
        for node in all_nodes
        if node.get("id")
    }
    absorbed_ic_numerator = absorbed_ic_numerator_per_turn(
        explained_ids=explained_ids,
        pre_turn_states=state_by_id,
        post_turn_states=post_turn_states_by_id,
        edges=edges,
    )

    # 4.6 Post-update novelty: among absorbed IUs that were unaware before this turn,
    # count those that became struggling or partial_understanding.
    # Also count direct unaware -> knows_well when the user actually reasoned about the concept
    # (parroted/articulation attempts do not qualify — only reasoning fires Rule 5).
    # Using absorbed_id_set (not all response_ids) so overloaded concepts that were
    # never truly processed don't inflate or deflate the novelty signal.
    absorbed_id_set = {iu.get("id") for iu in absorbed_ius_in_response}
    newly_aware_count = 0
    for node in all_nodes:
        iu_id = node.get("id")
        if not iu_id or iu_id not in absorbed_id_set:
            continue
        pre_state = _normalize_state(state_by_id.get(iu_id, "unaware"))
        if pre_state != "unaware":
            continue
        label = id_map.get(iu_id, iu_id) if id_map else iu_id
        post_state = _normalize_state(updated_state.get(label, {}).get("state", "unaware"))
        if post_state in {"struggling", "partial_understanding"}:
            newly_aware_count += 1
        elif _resolve_attempt_type(updates_by_id.get(iu_id, {})) == "reasoning" and post_state == "knows_well":
            newly_aware_count += 1

    if absorbed_count > 0:
        novelty_term = newly_aware_count / absorbed_count
        engagement_score = familiarity_term * novelty_term
    else:
        novelty_term = 0.0
        engagement_score = 0.0

    # 5. Collect concept labels for well_explained IUs this turn
    explained_concept_labels: List[str] = [
        (id_map.get(iu_id, iu_id) if id_map else iu_id)
        for iu_id in well_explained_ids
    ]

    # 6. Enrich iu_analysis with resolved concept labels for the viewer
    enriched_iu_analysis = []
    for item in iu_analysis:
        if not isinstance(item, dict):
            continue
        iu_id = item.get("id", "")
        label = (id_map.get(iu_id, iu_id) if id_map else iu_id)
        enriched_iu_analysis.append({**item, "label": label})

    if return_metrics:
        updated_density_metrics = compute_overall_density_from_state(
            iu_graph=iu_graph,
            knowledge_state=updated_state,
            id_map=id_map,
        )
        return updated_state, {
            "overall_density": updated_density_metrics.get("overall_density", overall_density),
            "articulation_mode": updated_density_metrics.get("articulation_mode", articulation_mode),
            "gap_sum": compute_gap_sum_from_state(
                iu_graph=iu_graph,
                knowledge_state=updated_state,
                id_map=id_map,
                response_iu_ids=response_ids,
            ).get("gap_sum", 0.0),
            "engagement_score": engagement_score,
            "familiarity_term": familiarity_term,
            "novelty_term": novelty_term,
            "overload": overload,
            "n_upward_transitions": n_upward_transitions,
            "absorbed_ic_numerator": absorbed_ic_numerator,
            "attempt_verify": attempt_verify,
            "explained_concepts": explained_concept_labels,
            "explained_count": calibration_denominator,
            "information_calibration_numerator": calibration_numerator,
            "information_calibration_denominator": calibration_denominator,
            "new_ius": new_ius,
            "disallowed": disallowed,
            "zpd_size": zpd_size,
            "perceived_overload_turn": perceived_overload_turn,
            "overload_threshold": overload_threshold,
            "absorbed_gain": absorbed_gain_this_turn,
            "iu_analysis": enriched_iu_analysis,
            # Decomposition of effective_load — populated for every preset, but
            # attempt_load is 0.0 unless the preset opted in via
            # include_attempt_load. Lets the dashboard / post-hoc recompute
            # split PO into comprehension vs attempt contributions without
            # rerunning Phase B.
            "comprehension_load": comprehension_load,
            "attempt_load": attempt_load,
            "effective_load": effective_load,
        }
    return updated_state
