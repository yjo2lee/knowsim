"""Retroactively detect virtual early stop turns and compute metrics for that truncation point.

For experiments run with --disable_early_stop, conversations run to max_turns even though
early stopping would have triggered at some earlier turn. This script:
  1. Detects the first turn where either termination rule would have fired:
     - Mastery: all IUs reached knows_well
     - Cognitive overload: overload in >= 2 of last 3 turns AND <= 1 upward transition (after 3-turn warmup)
  2. Computes native metrics (NKG, IC, PO, engagement, LExp) up to that turn
  3. Runs LLM judge on the truncated conversation
  4. Stores results as virtual_early_stop_turn, virtual_early_stop_reason,
     metrics_at_early_stop, judged_metrics_at_early_stop

Usage:
    python -m simulation.tools.compute_early_stop_metrics \\
        --input_dir output/competition_math/gpt-4.1 \\
        --judge_model gpt-5.2 \\
        --methods dynamic-knowledge-state \\
        [--force]

    python -m simulation.tools.compute_early_stop_metrics \\
        --input_file output/competition_math/gpt-4.1/dynamic-knowledge-state_structured_novice_adaptive_gpt-4.1.json \\
        --judge_model gpt-5.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import SingleModelClient
from .judge_conversations import (
    _build_judge_prompt,
    _load_judge_prompt,
    _parse_judge_response,
)

COGNITIVE_OVERLOAD_WARMUP = 3

_STRUCTURED_RUN_TYPES: frozenset = frozenset({"KS-based"})
_BASELINE_RUN_TYPES: frozenset = frozenset({"baseline"})


# ---------------------------------------------------------------------------
# State helpers (mirrors conversation.py private functions)
# ---------------------------------------------------------------------------

_STATE_ALIASES: Dict[str, str] = {
    "knows well": "knows_well",
    "partial understanding": "partial_understanding",
    "partial": "partial_understanding",
    "struggling": "struggling",
    "unaware": "unaware",
}

_STATE_SCORE_MAP: Dict[str, int] = {
    "unaware": 0,
    "struggling": 1,
    "partial_understanding": 2,
    "knows_well": 3,
}


def _normalize_state(state: str) -> str:
    s = (state or "").lower().strip().replace("-", "_").replace(" ", "_")
    return _STATE_ALIASES.get(s, s)


def _state_score(state: str) -> int:
    return _STATE_SCORE_MAP.get(_normalize_state(state), 0)


# ---------------------------------------------------------------------------
# Early stop detection
# ---------------------------------------------------------------------------

def _find_virtual_mastery_turn(
    item: Dict[str, Any],
) -> Optional[int]:
    """Return 0-indexed turn where all IUs first reached knows_well, or None.

    Checks knowledge_state_history snapshots (index 0 = initial state,
    index i = state after turn i). Returns the first turn index (1-based in
    the history, converted to 0-based turn index) where every IU is knows_well.
    Also respects the virtual_mastery_turn field if the runner already recorded it.
    """
    # If the runner already recorded virtual mastery, use that (0-indexed).
    vmt = item.get("virtual_mastery_turn")
    if vmt is not None:
        return int(vmt) - 1  # runner stores 1-based turn number

    ks_history = item.get("knowledge_state_history", []) or []
    # Skip index 0 (initial state); check from turn 1 onward.
    for idx in range(1, len(ks_history)):
        ks = ks_history[idx]
        if not ks:
            continue
        if all(
            _normalize_state((info.get("state", "") if isinstance(info, dict) else info)) == "knows_well"
            for info in ks.values()
        ):
            return idx - 1  # convert to 0-based turn index
    return None


def _find_virtual_cognitive_overload_turn(
    turn_metrics_history: List[Dict[str, Any]],
    min_warmup: int = COGNITIVE_OVERLOAD_WARMUP,
) -> Optional[int]:
    """Return 0-indexed turn where cognitive overload termination would fire, or None.

    Mirrors _check_termination() from conversation.py:
      - Requires at least min_warmup turns before checking
      - Fires when: overload in >= 2 of last 3 turns AND <= 1 upward transition
    """
    for i in range(len(turn_metrics_history)):
        if i + 1 < min_warmup:
            continue
        recent = turn_metrics_history[max(0, i - 2):i + 1]
        if len(recent) < 3:
            continue
        overload_count = sum(1 for t in recent if t.get("overload", False))
        total_up = sum(t.get("n_upward_transitions", 0) for t in recent)
        if overload_count >= 2 and total_up <= 1:
            return i
    return None


def find_virtual_early_stop_turn(
    turn_metrics_history: List[Dict[str, Any]],
    item: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Return 0-indexed turn index of the first early-stop trigger, or None.

    Checks both termination rules from conversation.py, returning whichever
    would have fired first:

    1. Mastery: all IUs in knowledge_state_history reached knows_well
    2. Cognitive overload: overload in >= 2 of last 3 turns AND <= 1 upward
       transition in those turns (after 3-turn warmup)
    """
    mastery_turn = _find_virtual_mastery_turn(item) if item else None
    overload_turn = _find_virtual_cognitive_overload_turn(turn_metrics_history)

    candidates = [t for t in (mastery_turn, overload_turn) if t is not None]
    return min(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Baseline early-stop detection
# ---------------------------------------------------------------------------

def _get_baseline_early_stop_turn(item: Dict[str, Any]) -> Optional[int]:
    """Return 0-indexed early-stop turn for a baseline item, or None.

    judged_end_turn (from baseline_termination_judge.py) is 1-based.
    Early stop fires only when judged_end_turn < actual turns.
    Zero-turn conversations (turns == 0) cannot be truncated → None.
    """
    jet = item.get("judged_end_turn")
    actual = item.get("turns", 0)
    if jet is None or actual == 0:
        return None
    jet = int(jet)
    if jet < actual:
        return jet - 1  # convert to 0-indexed
    return None


# ---------------------------------------------------------------------------
# Native metrics at early stop
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_native_metrics_at_early_stop(
    item: Dict[str, Any],
    early_stop_turn: int,
) -> Dict[str, Any]:
    """Compute native metrics up to and including early_stop_turn (0-indexed).

    Aggregation methods match the final-turn computation in _compute_structured_metrics:
    - IC:  cumulative ratio sum(ic_numerator_t) / sum(ic_denominator_t)
    - NKG: sum(absorbed_gain_t) / gain_den  (absorbed-only, matching absorbed_gain_total)
    - PO:  mean of per-turn values           (already a per-turn ratio, mean is correct)
    - Eng: mean of per-turn values           (matches engagement_score_history mean)
    - LExp: (1-PO + IC + Eng) / 3           (derived from the above)

    Falls back to approximations for runs predating the ic_numerator/absorbed_gain fields,
    with a note in the returned dict.
    """
    N = early_stop_turn
    turn_hist = item.get("turn_metrics_history", []) or []
    eng_hist = item.get("engagement_score_history", []) or []
    ks_history = item.get("knowledge_state_history", []) or []
    initial_ks = item.get("initial_knowledge_state") or {}

    slice_turn = turn_hist[: N + 1]
    slice_eng = eng_hist[: N + 1]
    notes = []

    # --- Perceived Overload: mean of per-turn values (matches final computation) ---
    po_vals = [
        float(t["perceived_overload_turn"])
        for t in slice_turn
        if t.get("perceived_overload_turn") is not None
    ]
    perceived_overload = _safe_mean(po_vals) or 0.0

    # --- Engagement: mean of per-turn values (matches final computation) ---
    eng_vals = [
        float(e["engagement"])
        for e in slice_eng
        if e.get("engagement") is not None
    ]
    engagement = _safe_mean(eng_vals) or 0.0

    # --- IC: cumulative ratio (matches final sum(ic_num)/sum(ic_den)) ---
    has_ic_components = any("ic_numerator" in t for t in slice_turn)
    if has_ic_components:
        ic_num = sum(float(t.get("ic_numerator", 0.0)) for t in slice_turn)
        ic_den = sum(float(t.get("ic_denominator", 0.0)) for t in slice_turn)
        information_calibration = (ic_num / ic_den) if ic_den > 0 else 0.0
    else:
        # Legacy runs: fall back to per-turn mean approximation.
        ic_vals = [
            float(t["information_calibration_turn"])
            for t in slice_turn
            if t.get("information_calibration_turn") is not None
        ]
        information_calibration = _safe_mean(ic_vals) or 0.0
        notes.append("IC approximated from per-turn mean (ic_numerator not stored; rerun simulator to fix)")

    # --- F1-based IC (precision/recall against ZPD) ---
    has_zpd = any("zpd_size" in t for t in slice_turn)
    if has_ic_components and has_zpd:
        zpd_total = sum(float(t.get("zpd_size", 0.0)) for t in slice_turn)
        ic_precision = information_calibration
        ic_recall = (ic_num / zpd_total) if zpd_total > 0 else 1.0
        ic_f1 = (
            (2 * ic_precision * ic_recall / (ic_precision + ic_recall))
            if (ic_precision + ic_recall) > 0 else 0.0
        )
    else:
        zpd_total = 0.0
        ic_precision = information_calibration
        ic_recall = None
        ic_f1 = None
        if not has_zpd and has_ic_components:
            notes.append("ic_f1 unavailable (zpd_size not stored; rerun simulator to fix)")

    # --- NKG: accumulated absorbed_gain (matches final absorbed_gain_total / gain_den) ---
    gain_den = sum(
        max(0, 3 - _state_score(_normalize_state((init_info or {}).get("state", "unaware"))))
        for init_info in initial_ks.values()
        if _state_score(_normalize_state((init_info or {}).get("state", "unaware"))) < 3
    )
    has_absorbed_gain = any("absorbed_gain" in t for t in slice_turn)
    if has_absorbed_gain:
        gained = sum(float(t.get("absorbed_gain", 0.0)) for t in slice_turn)
        notes.append("NKG from accumulated absorbed_gain (matches final metric)")
    else:
        # Legacy runs: fall back to KS-snapshot approach.
        # Note: diverges from absorbed_gain_total when confirmed_incorrect caused a downgrade.
        final_ks_idx = N + 1
        final_ks: Dict[str, Any] = {}
        if final_ks_idx < len(ks_history):
            final_ks = ks_history[final_ks_idx] or {}
        elif ks_history:
            final_ks = ks_history[-1] or {}
        gained = sum(
            max(0, _state_score(_normalize_state((final_ks.get(concept) or {}).get("state", "unaware")))
                - _state_score(_normalize_state((init_info or {}).get("state", "unaware"))))
            for concept, init_info in initial_ks.items()
            if _state_score(_normalize_state((init_info or {}).get("state", "unaware"))) < 3
        )
        notes.append("NKG from KS snapshot (absorbed_gain not stored; rerun simulator to fix)")

    normalized_knowledge_gain = (gained / gain_den) if gain_den > 0 else 0.0
    knowledge_gain = gained

    # --- Alternative NKG: raw state progress / (3 × N) ---
    # Σ over IUs of max(0, final_idx - init_idx) at the @es turn boundary,
    # normalized by 3 × |IUs in graph|. Cross-session comparable; symmetric;
    # no W_STATE weighting. See metrics.normalized_knowledge_gain_3N.
    final_ks_idx = N + 1
    final_ks: Dict[str, Any] = {}
    if final_ks_idx < len(ks_history):
        final_ks = ks_history[final_ks_idx] or {}
    elif ks_history:
        final_ks = ks_history[-1] or {}
    n_ius_total = len(initial_ks)
    raw_state_progress = 0
    for concept, init_info in initial_ks.items():
        init_idx = _state_score(_normalize_state((init_info or {}).get("state", "unaware")))
        final_info = final_ks.get(concept) or {}
        final_idx = _state_score(_normalize_state((final_info or {}).get("state", "unaware")))
        d = final_idx - init_idx
        if d > 0:
            raw_state_progress += d
    normalized_knowledge_gain_3N = (
        float(raw_state_progress) / (3.0 * n_ius_total)
    ) if n_ius_total > 0 else 0.0

    # --- Learning experience ---
    overload_inverse = max(0.0, 1.0 - perceived_overload)
    learning_experience = (overload_inverse + information_calibration + engagement) / 3.0

    result = {
        "knowledge_gain": knowledge_gain,
        "normalized_knowledge_gain": normalized_knowledge_gain,
        "normalized_knowledge_gain_3N": normalized_knowledge_gain_3N,
        "raw_state_progress": raw_state_progress,
        "information_calibration": information_calibration,
        "perceived_overload": perceived_overload,
        "engagement": engagement,
        "learning_experience": learning_experience,
        "overload_inverse": overload_inverse,
        "ic_precision": ic_precision,
        "ic_recall": ic_recall,
        "ic_f1": ic_f1,
        "turns_included": N + 1,
        "turn_metrics_history": slice_turn,
        "note": "; ".join(notes) if notes else "all metrics use exact aggregation matching final-turn computation",
    }
    return result


def _copy_final_metrics(item: Dict[str, Any]) -> Dict[str, Any]:
    """Copy final-turn metrics into the same schema as compute_native_metrics_at_early_stop.

    Used as the fallback when virtual early stop never triggered — the early-stop
    policy produces the same result as the max-turns policy for this conversation.
    """
    native = item.get("metrics", {}) or {}
    turns = item.get("turns", 0)

    # Backfill normalized_knowledge_gain_3N if missing from native metrics
    # (older runs predating this field). Compute directly from initial vs final KS.
    nkg_3N = native.get("normalized_knowledge_gain_3N")
    raw_progress = native.get("raw_state_progress")
    if nkg_3N is None or raw_progress is None:
        init_ks_for_3N = item.get("initial_knowledge_state") or {}
        final_ks_for_3N = item.get("knowledge_state") or {}
        n_ius_3N = len(init_ks_for_3N)
        prog_3N = 0
        for concept, init_info in init_ks_for_3N.items():
            init_idx_3N = _state_score(_normalize_state((init_info or {}).get("state", "unaware")))
            final_info_3N = final_ks_for_3N.get(concept) or {}
            final_idx_3N = _state_score(_normalize_state((final_info_3N or {}).get("state", "unaware")))
            d_3N = final_idx_3N - init_idx_3N
            if d_3N > 0:
                prog_3N += d_3N
        nkg_3N = (float(prog_3N) / (3.0 * n_ius_3N)) if n_ius_3N > 0 else 0.0
        raw_progress = prog_3N

    return {
        "knowledge_gain": native.get("knowledge_gain", 0.0),
        "normalized_knowledge_gain": native.get("normalized_knowledge_gain", 0.0),
        "normalized_knowledge_gain_3N": nkg_3N,
        "raw_state_progress": raw_progress,
        "information_calibration": native.get("information_calibration", 0.0),
        "perceived_overload": native.get("perceived_overload", 0.0),
        "engagement": native.get("engagement", 0.0),
        "learning_experience": native.get("learning_experience", 0.0),
        "overload_inverse": native.get("overload_inverse", 0.0),
        "ic_precision": native.get("ic_precision"),
        "ic_recall": native.get("ic_recall"),
        "ic_f1": native.get("ic_f1"),
        "turns_included": turns,
        "turn_metrics_history": item.get("turn_metrics_history", []),
        "note": "No early stop triggered — metrics identical to final-turn values",
    }


# ---------------------------------------------------------------------------
# LLM judge at early stop
# ---------------------------------------------------------------------------

async def judge_items_at_early_stop(
    items: List[Dict[str, Any]],
    prompt_template: str,
    judge_model: str,
    llm_provider: str,
    force: bool,
) -> None:
    """Add judged_metrics_at_early_stop to each item in-place."""
    client = SingleModelClient(judge_model, provider=llm_provider)

    indices_to_judge: List[int] = []
    contexts: List[List[Dict[str, str]]] = []

    for i, item in enumerate(items):
        early_stop_turn = item.get("virtual_early_stop_turn")
        if early_stop_turn is None:
            # No early stop detected — skip LLM judge
            continue
        if not force and item.get("judged_metrics_at_early_stop"):
            continue

        N = early_stop_turn
        full_conv = item.get("conversation", []) or []
        # Truncate: keep first N+1 assistant turns → entries 0 through 2*N+1
        truncated_conv = full_conv[: 2 * (N + 1)]

        # Build a temporary item proxy with truncated conversation
        proxy = {**item, "conversation": truncated_conv}
        prompt_text = _build_judge_prompt(prompt_template, proxy)
        indices_to_judge.append(i)
        contexts.append([{"role": "user", "content": prompt_text}])

    if not contexts:
        return

    print(f"  Judging {len(contexts)} early-stopped conversations...")
    responses_batched = await client.generate_responses(
        full_contexts=contexts,
        temperature=0.0,
        # See judge_conversations.py for the 4096 rationale.
        max_tokens=4096,
        n=1,
        show_progress=True,
        json_mode=True,
    )

    for list_idx, item_idx in enumerate(indices_to_judge):
        raw_responses = responses_batched[list_idx]
        raw = raw_responses[0] if raw_responses else ""
        parsed = _parse_judge_response(raw)
        if parsed is None:
            print(f"  WARNING: Could not parse judge response for item {item_idx}: {raw[:200]}")
            parsed = {
                "information_calibration": None,
                "perceived_overload": None,
                "quality_score": None,
                "perceived_learning": None,
                "ic_reasoning": f"parse_error: {raw[:100]}",
                "po_reasoning": "",
                "quality_reasoning": "",
                "pl_reasoning": "",
            }
        items[item_idx]["judged_metrics_at_early_stop"] = parsed


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

async def process_file(
    path: Path,
    prompt_template: str,
    judge_model: str,
    llm_provider: str,
    target_methods: Optional[set],
    force: bool,
    dry_run: bool,
    skip_judge: bool,
    include_baselines: bool = False,
) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return 0

    allowed_run_types = _STRUCTURED_RUN_TYPES | (_BASELINE_RUN_TYPES if include_baselines else frozenset())
    eligible = [
        item for item in data
        if isinstance(item, dict)
        and (target_methods is None or item.get("simulation_method") in target_methods)
        and item.get("run_type") in allowed_run_types
    ]

    if not eligible:
        return 0

    print(f"[{path.name}] {len(eligible)} eligible items")

    if dry_run:
        for item in eligible:
            is_baseline = item.get("run_type") in _BASELINE_RUN_TYPES
            if is_baseline:
                est = _get_baseline_early_stop_turn(item)
                label = f"judged_end_turn={item.get('judged_end_turn')}"
            else:
                turn_hist = item.get("turn_metrics_history", []) or []
                est = find_virtual_early_stop_turn(turn_hist, item=item)
                label = ""
            print(f"  {item.get('problem_id', '?')} [{item.get('run_type')}]: virtual_early_stop_turn={est} {label}".rstrip())
        return len(eligible)

    processed = 0
    for item in eligible:
        is_baseline = item.get("run_type") in _BASELINE_RUN_TYPES

        if is_baseline:
            # --- Baseline path ---
            # Early-stop criterion: judged_end_turn < actual turns (from baseline_termination_judge.py).
            # No native metrics (turn_metrics_history absent) — only judged metrics apply.
            est = _get_baseline_early_stop_turn(item)

            if force or item.get("virtual_early_stop_turn", "MISSING") == "MISSING":
                item["virtual_early_stop_turn"] = est
                item["virtual_early_stop_reason"] = (
                    item.get("judged_termination_reason") if est is not None else None
                )
            # metrics_at_early_stop is intentionally omitted for baselines —
            # native KS metrics don't exist. judged_metrics_at_early_stop is
            # populated below via judge_items_at_early_stop / fallback copy.

        else:
            # --- Structured path (unchanged) ---
            turn_hist = item.get("turn_metrics_history", []) or []
            est = find_virtual_early_stop_turn(turn_hist, item=item)

            if not force and item.get("virtual_early_stop_turn", "MISSING") != "MISSING":
                # Already computed — only update if force
                pass
            else:
                item["virtual_early_stop_turn"] = est
                # Record which mechanism would have fired first (for ablation analysis).
                if est is not None:
                    mastery_turn = _find_virtual_mastery_turn(item)
                    overload_turn = _find_virtual_cognitive_overload_turn(turn_hist)
                    if mastery_turn is not None and (overload_turn is None or mastery_turn <= overload_turn):
                        item["virtual_early_stop_reason"] = "mastery"
                    else:
                        item["virtual_early_stop_reason"] = "cognitive_overload"
                else:
                    item["virtual_early_stop_reason"] = None

            if force or not item.get("metrics_at_early_stop"):
                if est is not None:
                    item["metrics_at_early_stop"] = compute_native_metrics_at_early_stop(item, est)
                else:
                    # Early stop never triggered — fall back to final-turn metrics
                    # so both columns aggregate over the same full set.
                    item["metrics_at_early_stop"] = _copy_final_metrics(item)

            # Invariant check: when early stop fires at the final turn, metrics_at_early_stop
            # must equal the final metrics exactly (up to float precision).
            if est is not None and item.get("turns") is not None and est == item["turns"] - 1:
                es = item.get("metrics_at_early_stop", {})
                final = item.get("metrics", {})
                _INVARIANT_KEYS = [
                    ("normalized_knowledge_gain", "normalized_knowledge_gain"),
                    ("normalized_knowledge_gain_3N", "normalized_knowledge_gain_3N"),
                    ("information_calibration", "information_calibration"),
                    ("perceived_overload", "perceived_overload"),
                    ("engagement", "engagement"),
                ]
                for es_key, final_key in _INVARIANT_KEYS:
                    es_val = es.get(es_key)
                    final_val = final.get(final_key)
                    if es_val is not None and final_val is not None:
                        if abs(es_val - final_val) > 1e-6:
                            print(
                                f"  INVARIANT FAIL {item.get('problem_id','?')}: "
                                f"{es_key} early_stop={es_val:.6f} != final={final_val:.6f}"
                            )

        processed += 1

    # LLM judge on truncated conversations (only for items with actual early stop).
    # For items without early stop, copy judged_metrics as the fallback.
    if not skip_judge:
        await judge_items_at_early_stop(eligible, prompt_template, judge_model, llm_provider, force)
    for item in eligible:
        if item.get("virtual_early_stop_turn") is None:
            # No early stop — the early-stop window IS the full conversation,
            # so judged_metrics_at_early_stop must mirror judged_metrics.
            # Refresh on --force so that a re-judge of judged_metrics
            # propagates here too. Previously this copy was gated on the
            # field being empty, which left a STALE judgment in
            # judged_metrics_at_early_stop after a re-judge (e.g. the
            # opus-4-7 values that survived a gpt-5.2 force-rejudge).
            jm = item.get("judged_metrics")
            if jm and (force or not item.get("judged_metrics_at_early_stop")):
                item["judged_metrics_at_early_stop"] = dict(jm)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    early_stop_count = sum(1 for item in eligible if item.get("virtual_early_stop_turn") is not None)
    print(f"  Done. {early_stop_count}/{processed} items have a virtual early stop turn.")
    return processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect virtual early stop turns and compute metrics for that truncation point."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str, help="Directory to search for JSON files recursively.")
    input_group.add_argument("--input_file", type=str, help="Single JSON file to process.")
    parser.add_argument("--judge_model", type=str, default="gpt-5.2", help="LLM to use as judge.")
    parser.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["structured"],
        help="Only process conversations with these simulation_method values (default: structured).",
    )
    parser.add_argument(
        "--all_methods",
        action="store_true",
        help="Process all methods (skips method filter).",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if results already present.")
    parser.add_argument("--dry_run", action="store_true", help="Print detected early stop turns without writing.")
    parser.add_argument("--skip_judge", action="store_true", help="Skip LLM judge step (native metrics only).")
    parser.add_argument(
        "--include_baselines",
        action="store_true",
        help=(
            "Also process baseline conversations (zero-shot, zero-shot-cot, "
            "zero-shot-cot-user-profile) using judged_end_turn as early-stop criterion "
            "instead of the KS-based mastery/overload rules. Requires baseline_termination_judge "
            "to have been run first so that judged_end_turn is present in all baseline items."
        ),
    )
    args = parser.parse_args()

    prompt_template = _load_judge_prompt()
    target_methods: Optional[set] = None if args.all_methods else set(args.methods)

    if args.input_file:
        paths = [Path(args.input_file)]
    else:
        paths = sorted(Path(args.input_dir).rglob("*.json"))
        paths = [p for p in paths if "metrics" not in p.parts and ".bak." not in p.name]

    total = 0
    for path in paths:
        n = await process_file(
            path,
            prompt_template,
            args.judge_model,
            args.llm_provider,
            target_methods,
            args.force,
            args.dry_run,
            args.skip_judge,
            include_baselines=args.include_baselines,
        )
        total += n

    action = "Would process" if args.dry_run else "Processed"
    print(f"\n{action} {total} conversations across {len(paths)} file(s).")


if __name__ == "__main__":
    asyncio.run(main())
