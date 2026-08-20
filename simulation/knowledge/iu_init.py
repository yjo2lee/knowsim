"""IU-based knowledge state initialization (v3)."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple


def _topological_sort(nodes: List[Dict[str, str]], edges: List[Dict[str, str]]) -> List[str]:
    """Return all input node IDs in topological order.

    Robust to cycles: any nodes that are part of a cycle (and would never reach
    in-degree 0 via the standard Kahn's algorithm) are appended to the end of
    the output in graph-input order. This ensures every node passed in is
    represented in the returned order — initialize_knowledge_state and the
    downstream KS dict will then have an entry for every IU in the graph,
    regardless of whether the IU extraction emitted a cyclic prerequisite edge.
    Without this, cyclic nodes were silently dropped from the initial KS while
    Phase B continued to label them, producing NKG > 1.0 edge cases when those
    "missing" IUs were later "discovered" mid-conversation.
    """
    incoming = {n["id"]: 0 for n in nodes}
    children = {n["id"]: [] for n in nodes}
    for e in edges:
        src = e.get("from")
        dst = e.get("to")
        if src in incoming and dst in incoming:
            incoming[dst] += 1
            children[src].append(dst)
    queue = [n_id for n_id, deg in incoming.items() if deg == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in children.get(node, []):
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    if len(order) < len(nodes):
        sorted_ids = set(order)
        # Append remaining (cyclic) nodes in original input order so the
        # function's output is deterministic given the input.
        for n in nodes:
            iu_id = n.get("id")
            if iu_id and iu_id not in sorted_ids:
                order.append(iu_id)
    return order


def _get_prereqs(edges: List[Dict[str, str]], node_id: str) -> List[str]:
    return [e["from"] for e in edges if e.get("to") == node_id]


# Single source of truth for initialization ratios per knowledge level.
# Intermediate updated 2026-05-07 (IntBumpA): {known 0.40 → 0.55, partial 0.20 →
# 0.25, unaware 0.40 → 0.20}. Combined with GATE5+ (env-gated, see
# update_v2.py) this lifts strategy-arm KG alignment vs H-Study from 5/9 to
# 8/9 (τ +0.78). The bump shrinks comp's high-W_STATE pool at intermediate
# so broad-coverage explanations stop self-overloading. See
# the branch git history for ablation results.
_IU_INIT_RATIOS: Dict[str, Dict[str, float]] = {
    "novice":       {"known": 0.10, "partial": 0.10, "unaware": 0.80},
    "intermediate": {"known": 0.55, "partial": 0.25, "unaware": 0.20},
    "advanced":     {"known": 0.80, "partial": 0.20, "unaware": 0.00},
}

# pilot3_20260515: 4-state init with explicit `struggling` band.
# Used when ``iu_init.include_struggling=true`` feature flag is set.
# Rationale: gives dense-teaching models a head start with "concept
# introduced but foundation not solid" IUs that can advance 2 steps in one
# well_explained mention instead of being capped by ALT-A on unaware.
_IU_INIT_RATIOS_4STATE: Dict[str, Dict[str, float]] = {
    "novice":       {"known": 0.10, "partial": 0.10, "struggling": 0.30, "unaware": 0.50},
    "intermediate": {"known": 0.55, "partial": 0.15, "struggling": 0.20, "unaware": 0.10},
    "advanced":     {"known": 0.80, "partial": 0.10, "struggling": 0.10, "unaware": 0.00},
}

# P1 calibrated-linear formula — score → (known_ratio, partial_ratio).
#   known(s)   = 0.10 + 0.035 * s        (0.10 → 0.45 at s=10)
#   partial(s) = 0.20 + 0.015 * s        (0.20 → 0.35; preserves novice partial)
#   unaware(s) = 0.70 - 0.050 * s        (0.70 → 0.20; persistent reservoir)
_SCORE_KNOWN_INTERCEPT = 0.10
_SCORE_KNOWN_SLOPE = 0.035
_SCORE_PARTIAL_INTERCEPT = 0.20
_SCORE_PARTIAL_SLOPE = 0.015

# P2 exp-decay formula — single smooth curve, no piecewise.
#   known(s)   = 0.05 + 0.05 * s                 (0.05 → 0.55; 5% per point)
#   partial(s) = 0.25 + 0.10 * exp(-s / 2)       (0.35 → 0.25; ZPD boost decays with half-life 2 pts)
#   unaware(s) = 1 - known - partial              (0.60 → 0.20; strictly monotone)
# Key properties vs P1:
#   - novice (s=0): partial 0.20→0.35 (+75%), unaware 0.70→0.60 (-14%) — larger ZPD favours adaptive
#   - intermediate (s=3-7): virtually unchanged (Δpartial < 0.03, Δunaware < 0.03)
#   - advanced (s=10): unaware preserved at 0.20; partial tightens 0.35→0.25
_P2_KNOWN_INTERCEPT = 0.05
_P2_KNOWN_SLOPE = 0.05
_P2_PARTIAL_BASE = 0.25
_P2_PARTIAL_BOOST = 0.10
_P2_PARTIAL_DECAY = 0.5   # half-life = ln(2)/0.5 ≈ 2 score points

INIT_FORMULAS = ("p1", "p2_exp", "bucket")


def score_to_ratios(score: int, formula: str = "p1") -> Tuple[float, float]:
    """Convert a prescreening score (0-10) to (known_ratio, partial_ratio).

    formula="p1"     — P1 calibrated-linear (default, original behaviour)
    formula="p2_exp" — P2 exp-decay: higher partial at novice, flat at intermediate+
    """
    s = max(0, min(10, int(score)))
    if formula == "p2_exp":
        known   = _P2_KNOWN_INTERCEPT + _P2_KNOWN_SLOPE * s
        partial = _P2_PARTIAL_BASE + _P2_PARTIAL_BOOST * math.exp(-_P2_PARTIAL_DECAY * s)
    else:  # p1 (default)
        known   = _SCORE_KNOWN_INTERCEPT + _SCORE_KNOWN_SLOPE * s
        partial = _SCORE_PARTIAL_INTERCEPT + _SCORE_PARTIAL_SLOPE * s
    return (known, partial)


def score_to_level(score: int) -> str:
    """Map a prescreening score (1-10) to the nearest categorical level.

    Used for secondary systems (knowledge_level_instructions, overload
    threshold, user profile text) that require a categorical level label.
    Boundaries match the prescreening score group definitions from the
    human validation study: novice=[1-2], intermediate=[3-7], advanced=[8-10].
    """
    if score <= 2:
        return "novice"
    if score <= 7:
        return "intermediate"
    return "advanced"


def initialize_knowledge_state(
    iu_graph: Dict[str, List[Dict[str, str]]],
    level: str,
    rng: random.Random,
    prescreening_score: Optional[int] = None,
    init_formula: str = "p1",
) -> Dict[str, List[str]]:
    nodes = iu_graph.get("nodes", [])
    edges = iu_graph.get("edges", [])
    all_nodes = _topological_sort(nodes, edges)
    total = len(all_nodes)

    if prescreening_score is not None:
        if init_formula == "bucket":
            # Bucket-then-table: score → categorical level → _IU_INIT_RATIOS lookup.
            # Reproduces the 3402af6 (04/29) init behavior, where prescreening_score
            # didn't exist as a CLI flag and per-participant runs used --knowledge_level.
            bucket_level = score_to_level(prescreening_score)
            ratios = _IU_INIT_RATIOS[bucket_level]
            target_known_ratio = ratios["known"]
            target_partial_ratio = ratios["partial"]
        else:
            # Per-score formula path (P1 calibrated-linear, P2 exp-decay).
            target_known_ratio, target_partial_ratio = score_to_ratios(prescreening_score, formula=init_formula)
    else:
        ratios = _IU_INIT_RATIOS.get(level)
        if ratios is None:
            raise ValueError(f"Unsupported level: {level}")
        target_known_ratio = ratios["known"]
        target_partial_ratio = ratios["partial"]

    known = set()
    partially_known = set()
    unknown = set(all_nodes)

    # Guarantee at least 1 known IU so every level starts with some foundation.
    # Without this floor, small graphs (e.g. 6 IUs) can round to 0 known for
    # novice (int(round(0.10 * 6)) = 1, but 0.10 * 5 = 0), producing degenerate
    # states where the user has no anchor knowledge at all.
    target_known_count = max(1, min(total - 1, int(round(target_known_ratio * total))))
    candidates = list(all_nodes)
    rng.shuffle(candidates)

    while len(known) < target_known_count and candidates:
        progressed = False
        for node in list(candidates):
            prereqs = _get_prereqs(edges, node)
            if all(p in known for p in prereqs):
                known.add(node)
                unknown.discard(node)
                candidates.remove(node)
                progressed = True
                if len(known) >= target_known_count:
                    break
        if not progressed:
            break

    # Guarantee at least 1 partial IU so there is always a "frontier" between
    # known and unknown. This prevents advanced users in coarse graphs from
    # having zero partial IUs (all known or all unknown), which collapses
    # engagement dynamics.
    remaining = total - len(known)
    target_partial_count = max(1, min(remaining - 1, int(round(target_partial_ratio * total)))) if remaining > 1 else 0
    # All remaining unknown nodes are candidates, including root nodes (no prerequisites).
    partial_candidates = sorted(unknown)
    rng.shuffle(partial_candidates)
    for node in partial_candidates:
        if len(partially_known) >= target_partial_count:
            break
        prereqs = _get_prereqs(edges, node)
        if not prereqs:
            # Root node: no prerequisites required, always eligible for partial_understanding.
            partially_known.add(node)
            unknown.discard(node)
        else:
            known_prereqs = [p for p in prereqs if p in known]
            if len(known_prereqs) == len(prereqs):
                # All prerequisites known → eligible for knows_well or partial_understanding.
                partially_known.add(node)
                unknown.discard(node)
            elif len(known_prereqs) > 0:
                # Some (not all) prerequisites known → eligible for at most partial_understanding.
                partially_known.add(node)
                unknown.discard(node)
            # else: no prerequisites known → must remain unaware.

    # pilot3_20260515: when iu_init.include_struggling=true, split the
    # unaware pool into an explicit struggling band per _IU_INIT_RATIOS_4STATE.
    # Only fires when prescreening_score is None (level-based path) and the
    # current level has a 4-state ratio defined.
    struggling: set = set()
    try:
        from ..core.feature_flags import get_flags
        _flags = get_flags()
    except Exception:
        _flags = None
    if (_flags is not None and _flags.iu_init.include_struggling
            and prescreening_score is None
            and level in _IU_INIT_RATIOS_4STATE):
        target_struggling_ratio = _IU_INIT_RATIOS_4STATE[level]["struggling"]
        target_struggling_count = int(round(target_struggling_ratio * total))
        # Pull from the current `unknown` pool. Prefer IUs with some
        # prereq exposure (any prereq in known or partial), falling back to
        # any remaining unknown if needed.
        candidates_with_prereqs = []
        candidates_isolated = []
        for n in sorted(unknown):
            prereqs = _get_prereqs(edges, n)
            if any(p in known or p in partially_known for p in prereqs):
                candidates_with_prereqs.append(n)
            else:
                candidates_isolated.append(n)
        rng.shuffle(candidates_with_prereqs)
        rng.shuffle(candidates_isolated)
        ordered_candidates = candidates_with_prereqs + candidates_isolated
        for n in ordered_candidates[:target_struggling_count]:
            struggling.add(n)
            unknown.discard(n)

    return {
        "known": sorted(known),
        "partially_known": sorted(partially_known),
        "struggling": sorted(struggling),
        "unknown": sorted(unknown),
    }

