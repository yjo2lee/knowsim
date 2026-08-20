"""Single source of truth for per-turn and aggregate metric formulas.

Why this module exists
----------------------
Per-turn metric math used to live in three places (`update_v2.py`,
`recompute_metrics.py`, `compute_early_stop_metrics.py`) and had drifted
in subtle ways (see ``metric_audit.md``). This module collects the
canonical formulas so that runtime computation, post-hoc recomputation,
and truncated-at-early-stop computation can all share the same code.

Migration status
----------------
The runtime (`update_v2.py`) and the recompute paths still inline some
of these formulas. Tier-3 cleanup item 22 (merge of `recompute_metrics`
and `compute_early_stop_metrics`) will route every site through this
module. New code should import from here directly.

Decisions encoded here (from ``metric_audit.md`` / ``PROJECT_OVERVIEW.md``)
-----------------------------------------------------------------------
A. **IC ZPD ceiling uses pre-turn snapshot.** Mid-turn cascading state
   (``CURRENT_states``) is an implementation artifact of the topo-sort
   update; the *semantic* claim of IC is "what was teachable entering
   this turn".
B. **IC aggregation is micro-average.** ``sum(num) / sum(den)`` across
   turns, never mean of per-turn ratios.
C. **PO effective_load always counts review weight at 0.5.**
   ``eff = len(advanceable) * 1.0 + len(review) * 0.5``.
D. **Engagement is recomputable from ``iu_analysis_history``.** Phase B
   per-IU outputs are stored, so post-hoc recompute does not need to
   rerun the LLM. (Helper provided here; consumers must adopt it.)
E. **NKG denominator iterates the IU graph nodes.** Single source of
   truth for the concept set; resolves labels via the id_map once.
F. **``ic_recall == 1.0`` is the canonical vacuous-perfect value.**
G. **None-valued per-turn entries are filtered out**, never coerced
   to ``0.0`` before averaging.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .state import STATE_ORDER

PARTIAL_IDX = STATE_ORDER.index("partial_understanding")
KNOWS_WELL_IDX = STATE_ORDER.index("knows_well")


# -----------------------------------------------------------------------
# State helpers (mirror the ones in update_v2.py)
# -----------------------------------------------------------------------

def normalize_state(state: Optional[str]) -> str:
    if not state:
        return "unaware"
    s = state.strip()
    aliases = {"unknown_unknown": "unaware", "not_introduced": "unaware"}
    s = aliases.get(s, s)
    return s if s in STATE_ORDER else "unaware"


def state_index(state: Optional[str]) -> int:
    return STATE_ORDER.index(normalize_state(state))


def compute_base_ceiling(prereq_states: Iterable[str]) -> str:
    """Ceiling for an IU given its prerequisites' states.

    No prereqs → ``knows_well``;
    all ``knows_well`` → ``knows_well``;
    any ``partial_understanding`` → ``partial_understanding``;
    otherwise → ``struggling``.
    """
    states = [normalize_state(s) for s in prereq_states]
    if not states:
        return "knows_well"
    if all(s == "knows_well" for s in states):
        return "knows_well"
    if any(s == "partial_understanding" for s in states):
        return "partial_understanding"
    return "struggling"


def get_prereqs(edges: Sequence[Mapping[str, str]], iu_id: str) -> List[str]:
    return [e["from"] for e in edges if e.get("to") == iu_id and e.get("from")]


# -----------------------------------------------------------------------
# Per-turn formulas
# -----------------------------------------------------------------------

def ic_per_turn(
    *,
    explained_ids: Iterable[str],
    pre_turn_states: Mapping[str, str],
    edges: Sequence[Mapping[str, str]],
) -> Dict[str, float]:
    """Information Calibration numerator/denominator for a single turn.

    Decision A: pre-turn states only — never mid-turn cascaded.
    """
    explained = [iu for iu in explained_ids if iu]
    den = len(explained)
    num = 0
    for iu in explained:
        prereqs = get_prereqs(edges, iu)
        ceiling = compute_base_ceiling(pre_turn_states.get(p, "unaware") for p in prereqs)
        if state_index(ceiling) >= PARTIAL_IDX and normalize_state(pre_turn_states.get(iu, "unaware")) != "knows_well":
            num += 1
    return {"ic_numerator": num, "ic_denominator": den}


def zpd_size_per_turn(
    *,
    all_node_ids: Iterable[str],
    pre_turn_states: Mapping[str, str],
    edges: Sequence[Mapping[str, str]],
) -> int:
    """ZPD size: how many IUs were teachable entering this turn."""
    count = 0
    for iu in all_node_ids:
        if not iu:
            continue
        if normalize_state(pre_turn_states.get(iu, "unaware")) == "knows_well":
            continue
        prereqs = get_prereqs(edges, iu)
        ceiling = compute_base_ceiling(pre_turn_states.get(p, "unaware") for p in prereqs)
        if state_index(ceiling) >= PARTIAL_IDX:
            count += 1
    return count


def perceived_overload_per_turn(
    *,
    n_advanceable: int,
    n_review: int,
    overload_threshold: int,
) -> Dict[str, float]:
    """Decision C: ``eff = adv * 1.0 + review * 0.5`` (always)."""
    eff = n_advanceable * 1.0 + n_review * 0.5
    if eff <= 0:
        return {"effective_load": 0.0, "perceived_overload_turn": 0.0}
    po = max(0.0, eff - overload_threshold) / eff
    return {"effective_load": eff, "perceived_overload_turn": po}


def perceived_overload_capacity_per_turn(
    *,
    effective_load: float,
    overload_threshold: int,
) -> float:
    """Capacity-style PO: ``min(1, eff_load / threshold)``.

    Captures graded sub-threshold load that the standard
    ``max(0, eff-thr)/eff`` formulation zeros out. Saturates at 1.0 once
    eff_load reaches threshold so it does not run away on heavy turns.
    Reported alongside the legacy PO; never replaces it.
    """
    if overload_threshold <= 0 or effective_load <= 0:
        return 0.0
    return min(1.0, float(effective_load) / float(overload_threshold))


def perceived_overload_demand_per_turn(
    *,
    comprehension_load: float,
    attempt_load: float,
    n_reasoning_attempts: int,
    n_upward_transitions: int,
    overload_threshold: int,
) -> Dict[str, float]:
    """Demand-weighted PO with adaptive attempt multiplier.

    ``k = 1 + failure_rate`` where failure_rate is the fraction of reasoning
    attempts that did not produce an upward state transition.  k ranges
    [1.0, 2.0] — successful attempts cost the same as comprehension,
    failed attempts cost double (effort with no learning payoff).

    Uses the capacity formula ``min(1, demand / threshold)`` to preserve
    graded sub-threshold signal.
    """
    if n_reasoning_attempts > 0:
        attempt_successes = min(n_upward_transitions, n_reasoning_attempts)
        failure_rate = 1.0 - (attempt_successes / n_reasoning_attempts)
    else:
        failure_rate = 0.0

    k = 1.0 + failure_rate
    demand_load = comprehension_load + attempt_load * k

    if overload_threshold <= 0 or demand_load <= 0:
        po = 0.0
    else:
        po = min(1.0, demand_load / float(overload_threshold))

    return {
        "perceived_overload_demand_turn": po,
        "demand_load": demand_load,
        "attempt_failure_rate": failure_rate,
        "attempt_weight_k": k,
    }


def perceived_overload_demand_aggregate(per_turn: Sequence[Mapping[str, Any]]) -> float:
    """Mean of per-turn ``perceived_overload_demand_turn``."""
    vals = [
        float(t["perceived_overload_demand_turn"])
        for t in per_turn
        if t is not None and t.get("perceived_overload_demand_turn") is not None
    ]
    return (sum(vals) / len(vals)) if vals else 0.0


def absorbed_ic_numerator_per_turn(
    *,
    explained_ids: Iterable[str],
    pre_turn_states: Mapping[str, str],
    post_turn_states: Mapping[str, str],
    edges: Sequence[Mapping[str, str]],
) -> int:
    """Strict IC numerator: mentioned ∧ in-ZPD ∧ upward-transitioned.

    Standard IC (``ic_per_turn``) counts every mentioned IU whose prereq
    ceiling is ≥ partial_understanding and whose pre-state is not
    ``knows_well``. That number rewards mentions equally regardless of
    whether the user actually absorbed the concept — turns that overflow
    the threshold and get scaled to zero by the saturating dropoff still
    contribute to IC numerator.

    The absorbed variant adds a third gate: the IU's state must have
    actually moved upward this turn. Mentions cut off by overload (or
    blocked by Rule 3 monotonicity, or blocked by ceiling caps) do not
    count.
    """
    total = 0
    for iu in explained_ids:
        if not iu:
            continue
        if normalize_state(pre_turn_states.get(iu, "unaware")) == "knows_well":
            continue
        prereqs = get_prereqs(edges, iu)
        ceiling = compute_base_ceiling(pre_turn_states.get(p, "unaware") for p in prereqs)
        if state_index(ceiling) < PARTIAL_IDX:
            continue
        prev = state_index(pre_turn_states.get(iu, "unaware"))
        new = state_index(post_turn_states.get(iu, "unaware"))
        if new > prev:
            total += 1
    return total


def absorbed_gain_per_turn(
    *,
    allowed_ids: Iterable[str],
    pre_turn_states: Mapping[str, str],
    post_turn_states: Mapping[str, str],
) -> float:
    """Sum of ``max(0, post - pre)`` over IUs that were allowed to absorb."""
    total = 0.0
    for iu in allowed_ids:
        if not iu:
            continue
        prev = state_index(pre_turn_states.get(iu, "unaware"))
        new = state_index(post_turn_states.get(iu, "unaware"))
        total += max(0.0, float(new - prev))
    return total


# -----------------------------------------------------------------------
# Aggregation across turns
# -----------------------------------------------------------------------

def ic_aggregate(per_turn: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Decision B: micro-average. Decision F: vacuous = 1.0 for recall.

    Each entry should provide ``ic_numerator``, ``ic_denominator``, and
    optionally ``zpd_size``.
    """
    num_total = 0
    den_total = 0
    zpd_total = 0
    has_zpd = False
    for t in per_turn:
        if t is None:
            continue
        num_total += int(t.get("ic_numerator", 0) or 0)
        den_total += int(t.get("ic_denominator", 0) or 0)
        if "zpd_size" in t and t["zpd_size"] is not None:
            zpd_total += int(t["zpd_size"])
            has_zpd = True
    precision = (num_total / den_total) if den_total > 0 else 0.0
    recall = (num_total / zpd_total) if (has_zpd and zpd_total > 0) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "ic_numerator_total": num_total,
        "ic_denominator_total": den_total,
        "zpd_size_total": zpd_total if has_zpd else None,
        "ic_precision": precision,
        "ic_recall": recall,
        "ic_f1": f1,
        "information_calibration": precision,  # Legacy alias.
    }


def perceived_overload_aggregate(per_turn: Sequence[Mapping[str, Any]]) -> float:
    """Decision G: filter ``None`` before averaging; never coerce to 0."""
    vals = [
        float(t["perceived_overload_turn"])
        for t in per_turn
        if t is not None and t.get("perceived_overload_turn") is not None
    ]
    return (sum(vals) / len(vals)) if vals else 0.0


def perceived_overload_capacity_aggregate(per_turn: Sequence[Mapping[str, Any]]) -> float:
    """Mean of per-turn ``perceived_overload_capacity_turn``.

    Computes from the stored field if present; otherwise derives on the
    fly from ``effective_load`` and ``overload_threshold`` so legacy run
    JSONs (no stored capacity term) can still report this metric.
    """
    vals: List[float] = []
    for t in per_turn:
        if t is None:
            continue
        v = t.get("perceived_overload_capacity_turn")
        if v is None:
            eff = t.get("effective_load")
            thr = t.get("overload_threshold")
            if eff is None or thr is None or float(thr) <= 0:
                continue
            v = min(1.0, float(eff) / float(thr))
        vals.append(float(v))
    return (sum(vals) / len(vals)) if vals else 0.0


def absorbed_ic_aggregate(per_turn: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Micro-averaged absorbed-IC P/R/F1.

    Numerator: per-turn ``absorbed_ic_numerator`` (mentioned ∧ in-ZPD ∧
    upward-transitioned). Denominator for precision: ``ic_denominator``
    (mentioned-this-turn). Denominator for recall: ``zpd_size``
    (teachable-entering-this-turn).

    Legacy fallback: if ``absorbed_ic_numerator`` is not stored, use
    ``min(ic_numerator, n_upward_transitions)`` as a proxy that bounds
    the count from above (every absorbed-IC IU was in the in-ZPD-mentioned
    set AND was upward-transitioned, so the numerator can't exceed
    either count individually). The proxy slightly overcounts when a
    non-ZPD mention happened to upward-transition, which is rare under
    the absorption priority logic.
    """
    num_total = 0
    den_total = 0
    zpd_total = 0
    has_zpd = False
    for t in per_turn:
        if t is None:
            continue
        v = t.get("absorbed_ic_numerator")
        if v is None:
            ic_n = t.get("ic_numerator")
            n_up = t.get("n_upward_transitions")
            if ic_n is None or n_up is None:
                continue
            v = min(int(ic_n or 0), int(n_up or 0))
        num_total += int(v or 0)
        den_total += int(t.get("ic_denominator", 0) or 0)
        if "zpd_size" in t and t["zpd_size"] is not None:
            zpd_total += int(t["zpd_size"])
            has_zpd = True
    precision = (num_total / den_total) if den_total > 0 else 0.0
    recall = (num_total / zpd_total) if (has_zpd and zpd_total > 0) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "absorbed_ic_numerator_total": num_total,
        "absorbed_ic_denominator_total": den_total,
        "absorbed_ic_zpd_size_total": zpd_total if has_zpd else None,
        "absorbed_ic_precision": precision,
        "absorbed_ic_recall": recall,
        "absorbed_ic_f1": f1,
    }


def engagement_aggregate(per_turn: Sequence[Mapping[str, Any]]) -> float:
    """Decision G: filter ``None`` before averaging."""
    vals = [
        float(t["engagement"])
        for t in per_turn
        if t is not None and t.get("engagement") is not None
    ]
    return (sum(vals) / len(vals)) if vals else 0.0


def nkg_denominator_from_iu_graph(
    *,
    iu_graph_nodes: Iterable[Mapping[str, Any]],
    id_map: Mapping[str, str],
    initial_state_by_label: Mapping[str, Mapping[str, Any]],
) -> float:
    """Decision E: walk the IU graph (single source of truth) for the concept set."""
    total = 0.0
    for node in iu_graph_nodes:
        iu_id = node.get("id")
        if not iu_id:
            continue
        label = id_map.get(iu_id, iu_id)
        init_info = initial_state_by_label.get(label) or {}
        init_score = state_index(init_info.get("state", "unaware"))
        if init_score < KNOWS_WELL_IDX:
            total += float(KNOWS_WELL_IDX - init_score)
    return total


def normalized_knowledge_gain(
    *,
    absorbed_gain_total: float,
    iu_graph_nodes: Iterable[Mapping[str, Any]],
    id_map: Mapping[str, str],
    initial_state_by_label: Mapping[str, Mapping[str, Any]],
) -> float:
    den = nkg_denominator_from_iu_graph(
        iu_graph_nodes=iu_graph_nodes,
        id_map=id_map,
        initial_state_by_label=initial_state_by_label,
    )
    return (float(absorbed_gain_total) / den) if den > 0 else 0.0


def raw_state_progress(
    *,
    iu_graph_nodes: Iterable[Mapping[str, Any]],
    id_map: Mapping[str, str],
    initial_state_by_label: Mapping[str, Mapping[str, Any]],
    final_state_by_label: Mapping[str, Mapping[str, Any]],
) -> int:
    """Sum of integer state advances across all IUs in the graph.

    For each IU: contribution = max(0, final_idx - init_idx). Range per IU
    is [0, 3] given the 4-state model (unaware=0 → knows_well=3). No W_STATE
    weighting; counts pure step transitions.
    """
    total = 0
    for node in iu_graph_nodes:
        iu_id = node.get("id")
        if not iu_id:
            continue
        label = id_map.get(iu_id, iu_id)
        init_info = initial_state_by_label.get(label) or {}
        final_info = final_state_by_label.get(label) or {}
        init_idx = state_index(init_info.get("state", "unaware"))
        final_idx = state_index(final_info.get("state", "unaware"))
        d = final_idx - init_idx
        if d > 0:
            total += d
    return total


def normalized_knowledge_gain_3N(
    *,
    iu_graph_nodes: Iterable[Mapping[str, Any]],
    id_map: Mapping[str, str],
    initial_state_by_label: Mapping[str, Mapping[str, Any]],
    final_state_by_label: Mapping[str, Mapping[str, Any]],
) -> float:
    """Alternative NKG: raw state progress / (3 × N).

    Numerator: Σ over IUs of max(0, final_idx - init_idx). No W_STATE weighting.
    Denominator: 3 × |IUs in graph| (i.e., the theoretical max gain assuming
        every IU could go from unaware to knows_well).

    Properties vs. the W_STATE-weighted normalized_knowledge_gain:
      * Cross-session comparable: denominator depends only on graph size,
        not the user's starting KS, so high-baseline-knowledge sessions
        aren't artificially boosted by a small denominator.
      * Symmetric: numerator and denominator both count integer state-advances.
      * Range [0, 1]: 1.0 only when every IU starts at unaware and ends at
        knows_well; high-init sessions are naturally capped because their
        unreachable territory still counts toward the 3N ceiling.
      * No hyperparameters: no W_STATE values to defend.

    Useful as a parallel NKG measurement alongside the W_STATE-weighted variant
    when reporting hypothesis alignment, similar to how perceived_overload has
    multiple variants (native / capacity / demand / attempt).
    """
    nodes_list = list(iu_graph_nodes)
    n_ius = sum(1 for n in nodes_list if n.get("id"))
    if n_ius == 0:
        return 0.0
    progress = raw_state_progress(
        iu_graph_nodes=nodes_list,
        id_map=id_map,
        initial_state_by_label=initial_state_by_label,
        final_state_by_label=final_state_by_label,
    )
    return float(progress) / (3.0 * n_ius)


def learning_experience(
    *,
    perceived_overload: float,
    information_calibration: float,
    engagement: Optional[float],
) -> Optional[float]:
    """Composite of (1 - PO + IC + Eng) / 3.

    Decision D: if ``engagement`` is None (legacy run with no
    ``iu_analysis_history`` to recompute from), return None rather than
    silently mixing stale and fresh terms.
    """
    if engagement is None:
        return None
    overload_inverse = max(0.0, 1.0 - float(perceived_overload))
    return (overload_inverse + float(information_calibration) + float(engagement)) / 3.0
