"""Recompute native metrics from stored simulation data without re-running LLM.

Reads existing structured-run output JSON files and rewrites the metrics field using
the revised metric definitions:
  - normalized_knowledge_gain : absorbed-only (IU graph required)
  - information_calibration   : Goldilocks — new AND prereq-reachable (IU graph required)
  - perceived_overload        : volume-based max(0, load−threshold)/load (no IU graph needed)

Because all per-turn IU analysis is stored in iu_analysis_history and all per-turn
knowledge states are stored in knowledge_state_history, no LLM calls are needed.
The IU graph is only required for IC and KG (ceiling computation and overload priority).

Usage
-----
    # With IU cache (recomputes all three metrics):
    python -m simulation.tools.recompute_metrics \\
        --input_dir output/competition_math/gpt-4o \\
        --iu_cache_path output/iu_cache/competition_math_gpt-4o-mini_....json

    # Without IU cache (recomputes PO only):
    python -m simulation.tools.recompute_metrics \\
        --input_dir output/competition_math/gpt-4o \\
        --po_only

    # Dry run (print stats without writing):
    python -m simulation.tools.recompute_metrics \\
        --input_dir output/competition_math/gpt-4o \\
        --iu_cache_path output/iu_cache/... \\
        --dry_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse helpers from update_v2 (private by convention, not enforcement)
from ..knowledge.update_v2 import (
    _normalize_state,
    _state_index,
    _get_prereqs,
    _compute_base_ceiling,
    _compute_topo_depth,
    _is_known,
    _compute_effective_load,
    _compute_attempt_load,
    get_absorption_preset,
)
from ..knowledge.iu_graph import build_concept_graph_from_iu


# ── Helpers ────────────────────────────────────────────────────────────────────

def _unmet_prereq_count(iu_id: str, state_by_id: Dict[str, str], edges: List[Dict]) -> int:
    return sum(
        1 for p in _get_prereqs(edges, iu_id)
        if not _is_known(state_by_id.get(p, "unaware"))
    )


def _state_score(state: str) -> int:
    mapping = {"unaware": 0, "struggling": 1, "partial_understanding": 2, "knows_well": 3}
    return mapping.get(_normalize_state(state), 0)


def _lookup_state(ks: Dict[str, Any], label: str) -> str:
    info = ks.get(label, {})
    if isinstance(info, dict):
        return _normalize_state(info.get("state", "unaware"))
    return _normalize_state(str(info))


# ── Per-turn recomputation ─────────────────────────────────────────────────────

def _recompute_turn(
    iu_analysis: List[Dict[str, Any]],
    prev_ks: Dict[str, Any],
    next_ks: Optional[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    id_to_label: Dict[str, str],
    topo_depth: Dict[str, int],
    graph_order: Dict[str, int],
    threshold: int,
    absorption_mode: str = "default",
) -> Dict[str, Any]:
    """Return recomputed per-turn metrics for one assistant turn."""
    preset = get_absorption_preset(absorption_mode)
    all_node_ids = [n["id"] for n in nodes if n.get("id")]

    # Build state_by_id from previous knowledge state
    state_by_id: Dict[str, str] = {}
    for iu_id in all_node_ids:
        label = id_to_label.get(iu_id, iu_id)
        state_by_id[iu_id] = _lookup_state(prev_ks, label)

    # Prerequisite-based ceilings
    ceiling_by_id: Dict[str, str] = {}
    for iu_id in all_node_ids:
        prereq_ids = _get_prereqs(edges, iu_id)
        prereq_states = [state_by_id.get(pid, "unaware") for pid in prereq_ids]
        ceiling_by_id[iu_id] = _compute_base_ceiling(prereq_states)

    # Mentioned IUs (teaching_quality != not_mentioned)
    mentioned_items = [
        item for item in iu_analysis
        if isinstance(item, dict)
        and item.get("teaching_quality") != "not_mentioned"
        and item.get("id")
    ]
    advanceable_items = [
        item for item in mentioned_items
        if state_by_id.get(item["id"], "unaware") != "knows_well"
    ]
    review_items = [
        item for item in mentioned_items
        if state_by_id.get(item["id"], "unaware") == "knows_well"
    ]
    # Mode-aware effective_load: default uses adv*1.0 + review*0.5,
    # weightabsorb uses sum(W_STATE[state]) over mentioned IUs.
    # When the preset opts into attempt_load (weightabsorb_attempt), the
    # user-side cost from reasoning + articulation IUs is added on top.
    comprehension_load = _compute_effective_load(
        advanceable_items, review_items, state_by_id, preset,
    )
    attempt_load = _compute_attempt_load(iu_analysis, state_by_id, preset)
    effective_load = comprehension_load + attempt_load
    overload = effective_load >= threshold

    # ── Volume-based PO ──────────────────────────────────────────────────────
    po_turn = (
        max(0.0, effective_load - threshold) / effective_load
        if effective_load > 0 else 0.0
    )

    # ── Allowed IDs ──────────────────────────────────────────────────────────
    # Default mode: hard cutoff under overload, top N + reviews absorb.
    # weightabsorb: every mentioned IU absorbs (degradation handled at
    # update time via the absorption_efficiency multiplier; reflected in
    # the saved next_ks state advances).
    advanceable_sorted = sorted(
        advanceable_items,
        key=lambda item: (
            _unmet_prereq_count(item["id"], state_by_id, edges),
            topo_depth.get(item["id"], 0),
            graph_order.get(item["id"], 0),
        ),
    )

    if preset["dropoff"]:
        allowed_ids = {item["id"] for item in mentioned_items}
    elif preset.get("greedy_fill") and overload:
        w_state = preset.get("W_STATE", {})
        running_weight = 0.0
        absorbed_advanceable_ids: set = set()
        for item in advanceable_sorted:
            w = w_state.get(state_by_id.get(item["id"], "unaware"), 1.0)
            if running_weight + w > threshold:
                break
            absorbed_advanceable_ids.add(item["id"])
            running_weight += w
        allowed_ids = absorbed_advanceable_ids | {item["id"] for item in review_items}
    elif overload:
        decay = threshold / effective_load if effective_load > 0 else 1.0
        n_within = min(len(advanceable_sorted), threshold)
        n_absorbing = int(n_within * decay)
        allowed_ids = (
            {item["id"] for item in advanceable_sorted[:n_absorbing]}
            | {item["id"] for item in review_items}
        )
    else:
        allowed_ids = {item["id"] for item in mentioned_items}

    # ── Absorbed-only knowledge gain ─────────────────────────────────────────
    absorbed_gain_turn = 0.0
    if next_ks is not None:
        for node in nodes:
            iu_id = node.get("id")
            if not iu_id or iu_id not in allowed_ids:
                continue
            label = id_to_label.get(iu_id, iu_id)
            prev_score = _state_index(state_by_id.get(iu_id, "unaware"))
            new_score = _state_index(_lookup_state(next_ks, label))
            absorbed_gain_turn += max(0.0, float(new_score - prev_score))

    # ── Revised IC (Goldilocks: new AND prereq-reachable) ────────────────────
    explained_ids = [item["id"] for item in mentioned_items]
    partial_idx = _state_index("partial_understanding")
    ic_num_turn = sum(
        1 for iu_id in explained_ids
        if _state_index(ceiling_by_id.get(iu_id, "unaware")) >= partial_idx
        and _normalize_state(state_by_id.get(iu_id, "unaware")) != "knows_well"
    )
    ic_den_turn = len(explained_ids)
    ic_turn = ic_num_turn / ic_den_turn if ic_den_turn > 0 else 0.0

    # ZPD size: all IUs that are learnable this turn
    zpd_size_turn = sum(
        1 for iu_id in all_node_ids
        if _normalize_state(state_by_id.get(iu_id, "unaware")) != "knows_well"
        and _state_index(ceiling_by_id.get(iu_id, "unaware")) >= partial_idx
    )

    return {
        "po_turn": po_turn,
        "absorbed_gain_turn": absorbed_gain_turn,
        "ic_num_turn": float(ic_num_turn),
        "ic_den_turn": float(ic_den_turn),
        "ic_turn": ic_turn,
        "zpd_size_turn": float(zpd_size_turn),
        "comprehension_load": comprehension_load,
        "attempt_load": attempt_load,
        "effective_load": effective_load,
    }


# ── Per-item recomputation ─────────────────────────────────────────────────────

def _recompute_item(
    item: Dict[str, Any],
    iu_cache: Optional[Dict[str, Dict]],
    po_only: bool,
) -> Tuple[bool, str]:
    """Recompute metrics for one conversation item. Returns (changed, reason)."""
    if item.get("simulation_method") != "structured":
        return False, "not structured"

    iu_analysis_history: List[List[Dict]] = item.get("iu_analysis_history") or []
    knowledge_state_history: List[Dict] = item.get("knowledge_state_history") or []

    if not iu_analysis_history:
        return False, "no iu_analysis_history"
    if len(knowledge_state_history) < 1:
        return False, "no knowledge_state_history"

    threshold = int(item.get("initial_overload_threshold", 5))
    problem_id = str(item.get("problem_id", ""))
    # Old logs predate the absorption_mode field; default-mode behavior is
    # preserved when the field is missing.
    absorption_mode = str(item.get("absorption_mode", "default") or "default")

    # Resolve IU graph
    nodes: List[Dict] = []
    edges: List[Dict] = []
    id_to_label: Dict[str, str] = {}
    topo_depth: Dict[str, int] = {}
    graph_order: Dict[str, int] = {}

    if not po_only:
        if not iu_cache:
            return False, "no IU cache (use --po_only to recompute PO only)"
        iu_graph = iu_cache.get(problem_id) or iu_cache.get(problem_id.lstrip("0"))
        if not iu_graph:
            return False, f"problem_id={problem_id} not in IU cache"
        nodes = iu_graph.get("nodes", [])
        edges = iu_graph.get("edges", [])
        all_node_ids = [n["id"] for n in nodes if n.get("id")]
        # Reconstruct id_to_label the same way build_concept_graph_from_iu does
        for node in nodes:
            iu_id = node.get("id", "")
            concept = node.get("concept", "")
            label = f"{iu_id}: {concept}" if iu_id and concept else iu_id or concept
            id_to_label[iu_id] = label
        topo_depth = _compute_topo_depth(edges, all_node_ids)
        graph_order = {n["id"]: i for i, n in enumerate(nodes) if n.get("id")}

    # history[0] = initial state (before any turns)
    # history[i] = state after turn i
    n_turns = len(iu_analysis_history)
    absorbed_gain_total = 0.0
    ic_num_total = 0.0
    ic_den_total = 0.0
    zpd_size_total = 0.0
    po_vals: List[float] = []
    updated_turn_metrics: List[Dict[str, Any]] = []

    for turn_idx in range(n_turns):
        prev_ks = knowledge_state_history[turn_idx] if turn_idx < len(knowledge_state_history) else {}
        next_ks = knowledge_state_history[turn_idx + 1] if (turn_idx + 1) < len(knowledge_state_history) else None
        iu_analysis = iu_analysis_history[turn_idx]

        if po_only:
            # Recompute PO only — no IU graph needed
            mentioned = [
                x for x in iu_analysis
                if isinstance(x, dict) and x.get("teaching_quality") != "not_mentioned" and x.get("id")
            ]
            # Classify advanceable vs review using stored labels in iu_analysis (has "label")
            adv, rev = [], []
            for x in mentioned:
                label = x.get("label", x.get("id", ""))
                state_info = prev_ks.get(label, {})
                state_str = state_info.get("state", "unaware") if isinstance(state_info, dict) else str(state_info)
                if _normalize_state(state_str) == "knows_well":
                    rev.append(x)
                else:
                    adv.append(x)
            eff = len(adv) * 1.0
            po_turn = max(0.0, eff - threshold) / eff if eff > 0 else 0.0
            po_vals.append(po_turn)

            existing = item.get("turn_metrics_history", [{}] * n_turns)
            orig = existing[turn_idx] if turn_idx < len(existing) else {}
            updated_turn_metrics.append({**orig, "perceived_overload_turn": po_turn})
        else:
            result = _recompute_turn(
                iu_analysis=iu_analysis,
                prev_ks=prev_ks,
                next_ks=next_ks,
                nodes=nodes,
                edges=edges,
                id_to_label=id_to_label,
                topo_depth=topo_depth,
                graph_order=graph_order,
                threshold=threshold,
                absorption_mode=absorption_mode,
            )
            absorbed_gain_total += result["absorbed_gain_turn"]
            ic_num_total += result["ic_num_turn"]
            ic_den_total += result["ic_den_turn"]
            zpd_size_total += result["zpd_size_turn"]
            po_vals.append(result["po_turn"])

            existing = item.get("turn_metrics_history", [{}] * n_turns)
            orig = existing[turn_idx] if turn_idx < len(existing) else {}
            updated_turn_metrics.append({
                **orig,
                "information_calibration_turn": result["ic_turn"],
                "ic_numerator": result["ic_num_turn"],
                "ic_denominator": result["ic_den_turn"],
                "perceived_overload_turn": result["po_turn"],
                "zpd_size": result["zpd_size_turn"],
                "absorbed_gain": result["absorbed_gain_turn"],
            })

    # ── Aggregate final metrics ───────────────────────────────────────────────
    perceived_overload = sum(po_vals) / len(po_vals) if po_vals else 0.0

    old_metrics = item.get("metrics", {}) or {}
    new_metrics = dict(old_metrics)
    new_metrics["perceived_overload"] = perceived_overload
    new_metrics["overload_inverse"] = max(0.0, 1.0 - perceived_overload)

    if not po_only:
        # Knowledge gain denominator: max possible (from initial state, unchanged)
        initial_ks = knowledge_state_history[0] if knowledge_state_history else {}
        gain_den = 0.0
        for node in nodes:
            iu_id = node.get("id")
            if not iu_id:
                continue
            label = id_to_label.get(iu_id, iu_id)
            init_score = _state_score(_lookup_state(initial_ks, label))
            if init_score < 3:
                gain_den += max(0, 3 - init_score)

        normalized_kg = absorbed_gain_total / gain_den if gain_den > 0 else 0.0
        information_calibration = ic_num_total / ic_den_total if ic_den_total > 0 else 0.0

        ic_precision = information_calibration
        ic_recall = (ic_num_total / zpd_size_total) if zpd_size_total > 0 else 1.0
        ic_f1 = (
            (2 * ic_precision * ic_recall / (ic_precision + ic_recall))
            if (ic_precision + ic_recall) > 0 else 0.0
        )

        new_metrics["knowledge_gain"] = absorbed_gain_total
        new_metrics["normalized_knowledge_gain"] = normalized_kg
        new_metrics["information_calibration"] = information_calibration
        new_metrics["absorbed_gain_total"] = absorbed_gain_total
        new_metrics["ic_numerator_total"] = ic_num_total
        new_metrics["ic_denominator_total"] = ic_den_total
        new_metrics["zpd_size_total"] = zpd_size_total
        new_metrics["ic_precision"] = ic_precision
        new_metrics["ic_recall"] = ic_recall
        new_metrics["ic_f1"] = ic_f1

        # Recompute learning_experience with new values
        engagement = old_metrics.get("engagement", 0.0) or 0.0
        new_metrics["learning_experience"] = (
            new_metrics["overload_inverse"] + information_calibration + engagement
        ) / 3.0

    item["metrics"] = new_metrics
    item["turn_metrics_history"] = updated_turn_metrics
    return True, "ok"


# ── File-level processing ──────────────────────────────────────────────────────

def recompute_file(
    path: Path,
    iu_cache: Optional[Dict[str, Dict]],
    po_only: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    """Process one JSON file. Returns (n_updated, n_skipped)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [skip] {path.name}: cannot read ({e})")
        return 0, 0

    if not isinstance(data, list):
        return 0, 0

    updated, skipped = 0, 0
    for item in data:
        if not isinstance(item, dict):
            continue
        changed, reason = _recompute_item(item, iu_cache, po_only)
        if changed:
            updated += 1
        else:
            skipped += 1

    if updated > 0 and not dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return updated, skipped


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute native metrics from stored simulation data.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str,
                             help="Directory to search for JSON files recursively.")
    input_group.add_argument("--input_file", type=str,
                             help="Single JSON file to recompute.")
    parser.add_argument("--iu_cache_path", type=str, default="",
                        help="Path to IU graph cache JSON (required for IC and KG; optional for PO).")
    parser.add_argument("--po_only", action="store_true",
                        help="Recompute perceived_overload only (no IU cache needed).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Compute and print stats without writing files.")
    args = parser.parse_args()

    if not args.po_only and not args.iu_cache_path:
        parser.error("Provide --iu_cache_path or use --po_only.")

    # Load IU cache
    iu_cache: Optional[Dict[str, Dict]] = None
    if args.iu_cache_path:
        cache_path = Path(args.iu_cache_path)
        if not cache_path.exists():
            print(f"ERROR: IU cache not found: {cache_path}")
            return
        with cache_path.open(encoding="utf-8") as f:
            iu_cache = json.load(f)
        print(f"Loaded IU cache: {len(iu_cache)} problems from {cache_path.name}")

    skip_parts = {"metrics", "experiment_comparison", "dashboard", "iu_cache"}
    if args.input_file:
        paths = [Path(args.input_file)]
    else:
        paths = [
            p for p in sorted(Path(args.input_dir).rglob("*.json"))
            if not any(part in p.parts for part in skip_parts)
        ]

    print(f"Processing {len(paths)} file(s)  [dry_run={args.dry_run}, po_only={args.po_only}]")

    total_updated = total_skipped = 0
    for path in paths:
        n_up, n_sk = recompute_file(path, iu_cache, args.po_only, args.dry_run)
        if n_up > 0:
            action = "would update" if args.dry_run else "updated"
            print(f"  {path.name}: {action} {n_up} items (skipped {n_sk})")
        total_updated += n_up
        total_skipped += n_sk

    action = "Would update" if args.dry_run else "Updated"
    print(f"\n{action} {total_updated} structured conversations ({total_skipped} skipped).")


if __name__ == "__main__":
    main()
