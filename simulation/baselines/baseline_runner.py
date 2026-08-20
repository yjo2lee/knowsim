"""CLI entry point for baseline user-simulator experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.logging import get_log_path
from ..core.models import SingleModelClient
from ..core.prompts import load_prompt
from ..data.expertqa_answer import select_expertqa_answer
from ..data.loaders import load_dataset_rows, load_jsonl_rows, row_problem_id
from ..profiles.interaction import format_interaction_profile, load_interaction_profiles
from ..runtime.conversation import run_conversation_batch, _format_knowledge_state
from ..core.paths import conversation_json_name
from ..runtime.knowledge_levels import get_knowledge_level_instructions, get_synthetic_user_profile
from ..knowledge.iu_graph import build_concept_graph_from_iu
from ..knowledge.iu_init import initialize_knowledge_state

# IMPORTANT: Do not hard-code API keys in the repository.
# Provide required keys via environment variables (e.g. OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY).


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline user-simulator runner.")
    parser.add_argument(
        "--baseline_method",
        type=str,
        default="zero-shot",
        choices=["vanilla", "zero-shot", "zero-shot-cot", "zero-shot-cot-user-profile"],
    )
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--task", type=str, default="math", choices=["math", "expertqa"])
    parser.add_argument("--num_conversations", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0,
                        help="Skip this many rows from the start of the CSV. "
                             "problem_id will still reflect the original row index.")
    parser.add_argument(
        "--problem_indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific CSV row indices to run (non-contiguous list). When set, "
             "overrides --start_index/--num_conversations: only the listed rows "
             "are loaded, and problem_id is the literal CSV row index.",
    )
    parser.add_argument("--max_turns", type=int, default=15,
                        help="Maximum number of user-assistant exchanges per conversation (default: 15).")
    from simulation.knowledge.update_v2 import ABSORPTION_PRESETS as _ABSORPTION_PRESETS
    parser.add_argument("--absorption_mode", type=str, default="default",
                        choices=sorted(_ABSORPTION_PRESETS.keys()),
                        help="Tag-only on this runner: baselines do not run the IU "
                             "knowledge-state update. The mode is recorded on each "
                             "output JSON and influences the output filename suffix "
                             "so cross-mode runs are kept separate.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Root output directory. Run files land in <output_dir>/<task>/<assistant_model>/. "
                             "Default 'output' matches the orchestrator and dashboards.")
    parser.add_argument("--user_model", type=str, default="gpt-5.2")
    parser.add_argument("--assistant_model", type=str, default="")
    parser.add_argument(
        "--assistant_strategy",
        type=str,
        default="adaptive",
        choices=["adaptive", "comprehensive", "socratic", "validation_study_minimal"],
        help="Assistant prompt strategy. 'validation_study_minimal' is the verbatim "
             "prompt used by the deployed Next.js validation-study app for the model arm.",
    )
    parser.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "gemini", "anthropic", "together", "groq"])
    parser.add_argument("--assistant_llm_provider", type=str, default=None,
                        help="LLM provider for the assistant model only. If not set, uses --llm_provider.")
    parser.add_argument("--prompts_root", type=str, default="simulation/prompts")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="Data/competition_math/data/train_fixed_level_4_5.csv",
    )
    parser.add_argument(
        "--expertqa_jsonl",
        type=str,
        default="Data/expertqa/r2_compiled_anon.jsonl",
    )
    parser.add_argument("--expertqa_answer_variant", type=str, default="")
    parser.add_argument("--knowledge_level", type=str, default="intermediate", choices=["novice", "intermediate", "advanced"])
    parser.add_argument(
        "--interaction_profile_path",
        type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "profiles", "interaction_style.json")
        ),
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--iu_cache_path",
        type=str,
        default="",
        help="Path to pre-built IU graph cache JSON (required for zero-shot-cot-user-profile KS init).",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing output JSON; merge pass keeps only this run's results (no prior problem_ids).",
    )
    parser.add_argument("--run_id", type=str, default="",
                        help="Run identifier (passed by the orchestrator). "
                             "When set, outputs go into <output_dir>/<task>/<model>/<run_id>/.")
    parser.add_argument("--flat_output_layout", action="store_true",
                        help="Drop the <assistant_model>/ segment from the output path. "
                             "Used by the validation per-participant runners which group by "
                             "(arm, run_id, participant) instead of by model. With this flag, "
                             "output goes to <output_dir>/<task>/<run_id>/ (no model parent).")
    parser.add_argument("--disable_length_enforcement", action="store_true",
                        help="Skip the post-processing assistant-length enforcement. "
                             "Recommended for validation-study comparisons.")
    parser.add_argument("--disable_interaction_style", action="store_true",
                        help="Skip injecting the ## Interaction Style block into the "
                             "user-simulator prompt. Mirrors --disable_interaction_style "
                             "on runner.py. With this flag, zero-shot-cot-user-profile "
                             "uses get_synthetic_user_profile(level) instead of "
                             "looking up real-user profiles.")
    return parser


def _load_prompt_pair(prompts_root: str, version: str, task: str = "math") -> tuple[str, str]:
    """Simulator prompts live under ``<prompts_root>/simulator/``.

    For ExpertQA, task-specific templates (``<version>-expertqa.txt``) are
    preferred when they exist; otherwise the math templates are used as a
    fallback.
    """
    sim_dir = os.path.join(prompts_root, "simulator")
    if task == "expertqa":
        expertqa_prompt = os.path.join(sim_dir, f"{version}-expertqa.txt")
        expertqa_initial = os.path.join(sim_dir, f"{version}-initial-query-expertqa.txt")
        prompt_path = expertqa_prompt if os.path.exists(expertqa_prompt) else os.path.join(sim_dir, f"{version}.txt")
        initial_path = expertqa_initial if os.path.exists(expertqa_initial) else os.path.join(sim_dir, f"{version}-initial-query.txt")
    else:
        prompt_path = os.path.join(sim_dir, f"{version}.txt")
        initial_path = os.path.join(sim_dir, f"{version}-initial-query.txt")
    return load_prompt(prompt_path), load_prompt(initial_path)


def _load_annotations(args: argparse.Namespace) -> List[Dict[str, object]]:
    annotations: List[Dict[str, object]] = []
    if args.task == "expertqa":
        rows = load_jsonl_rows(args.expertqa_jsonl)
        if args.problem_indices:
            indexed = [(i, rows[i]) for i in args.problem_indices if 0 <= i < len(rows)]
        else:
            if args.start_index > 0:
                rows = rows[args.start_index:]
            if args.num_conversations > 0:
                rows = rows[:args.num_conversations]
            indexed = [(idx + args.start_index, row) for idx, row in enumerate(rows)]
        for csv_idx, row in indexed:
            answer_text, answer_variant, answer_source = select_expertqa_answer(
                row.get("answers", {}) or {},
                args.expertqa_answer_variant,
            )
            annotations.append(
                {
                    "problem_id": row_problem_id(row, csv_idx),
                    "question": row.get("question", ""),
                    "solution": answer_text,
                    "metadata": row.get("metadata", {}),
                    "answer_variant": answer_variant,
                    "answer_source": answer_source,
                    "annotator_id": row.get("annotator_id", ""),
                }
            )
    else:
        rows = load_dataset_rows(args.input_csv)
        if args.problem_indices:
            indexed = [(i, rows[i]) for i in args.problem_indices if 0 <= i < len(rows)]
        else:
            if args.start_index > 0:
                rows = rows[args.start_index:]
            if args.num_conversations > 0:
                rows = rows[:args.num_conversations]
            indexed = [(idx + args.start_index, row) for idx, row in enumerate(rows)]
        for csv_idx, row in indexed:
            annotations.append(
                {
                    "problem_id": row_problem_id(row, csv_idx),
                    "question": row.get("problem", ""),
                    "solution": row.get("solution", ""),
                    "level": row.get("level", ""),
                    "type": row.get("type", ""),
                }
            )
    return annotations


def _normalize_problem_id(problem_id: str) -> str:
    try:
        return str(int(problem_id))
    except (TypeError, ValueError):
        return str(problem_id)


def _get_interaction_features(
    interaction_profiles: Dict[str, List[Dict[str, str]]],
    problem_id: str,
    user_id: str,
    model: str,
    rng: random.Random,
) -> List[Dict[str, str]]:
    if not interaction_profiles:
        return []

    normalized_pid = _normalize_problem_id(problem_id)
    key_int = repr((int(normalized_pid) if normalized_pid.isdigit() else normalized_pid, user_id, model))
    key_str = repr((str(problem_id), user_id, model))

    if key_int in interaction_profiles:
        return interaction_profiles[key_int]
    if key_str in interaction_profiles:
        return interaction_profiles[key_str]

    prefix_int = f"({normalized_pid},"
    prefix_str = f"('{str(problem_id)}',"
    for key in interaction_profiles.keys():
        if key.startswith(prefix_int) or key.startswith(prefix_str):
            return interaction_profiles.get(key, [])

    return rng.choice(list(interaction_profiles.values()))


def _resolve_prompt_version(args: argparse.Namespace) -> str:
    if args.version:
        return args.version
    if args.baseline_method == "vanilla":
        return "vanilla"
    return args.baseline_method


def _build_user_profiles(
    annotations: List[Dict[str, object]],
    interaction_profile_path: str,
    seed: int,
) -> List[str]:
    interaction_profiles: Dict[str, List[Dict[str, str]]] = {}
    if interaction_profile_path and os.path.exists(interaction_profile_path):
        interaction_profiles = load_interaction_profiles(interaction_profile_path)

    rng = random.Random(seed)
    user_profiles: List[str] = []
    for ann in annotations:
        features = _get_interaction_features(
            interaction_profiles,
            str(ann.get("problem_id", "")),
            str(ann.get("user_id", "")),
            str(ann.get("model", "")),
            rng,
        )
        user_profiles.append(format_interaction_profile(features, "around 20 words"))
    return user_profiles


def _build_initial_knowledge_states(
    annotations: List[Dict[str, object]],
    iu_cache_path: str,
    knowledge_level: str,
    seed: int,
) -> Optional[List[str]]:
    """Build per-problem initial knowledge states from IU cache (mirrors runner.py logic)."""
    if not iu_cache_path or not os.path.exists(iu_cache_path):
        return None
    with open(iu_cache_path, encoding="utf-8") as f:
        iu_graphs = json.load(f)
    _, id_maps = build_concept_graph_from_iu(iu_graphs)
    rng = random.Random(seed)
    result = []
    for ann in annotations:
        pid = str(ann.get("problem_id", ""))
        iu_graph = iu_graphs.get(pid)
        if not iu_graph:
            result.append("")
            continue
        id_map = id_maps.get(pid, {})
        edges = iu_graph.get("edges", [])
        prereqs_by_id: Dict[str, List[str]] = {}
        for e in edges:
            prereqs_by_id.setdefault(e["to"], []).append(e["from"])
        state = initialize_knowledge_state(iu_graph, knowledge_level, rng)
        known_ids = set(state.get("known", []))
        partial_ids = set(state.get("partially_known", []))
        unknown_ids = set(state.get("unknown", []))

        def _clean_label(raw_label: str) -> str:
            return raw_label.split(": ", 1)[-1] if ": " in raw_label else raw_label

        mapped: Dict[str, Dict[str, str]] = {}
        for iu_id in known_ids:
            mapped[_clean_label(id_map.get(iu_id, iu_id))] = {"state": "knows_well"}
        for iu_id in partial_ids:
            prereqs = prereqs_by_id.get(iu_id, [])
            known_ratio = len([p for p in prereqs if p in known_ids]) / max(len(prereqs), 1)
            mapped[_clean_label(id_map.get(iu_id, iu_id))] = {
                "state": "partial_understanding" if known_ratio >= 0.5 else "struggling"
            }
        for iu_id in unknown_ids:
            mapped[_clean_label(id_map.get(iu_id, iu_id))] = {"state": "unaware"}
        result.append(_format_knowledge_state(mapped))
    return result


async def main() -> None:
    parser = cli_parser()
    args = parser.parse_args()

    version = _resolve_prompt_version(args)
    prompt_template, prompt_initial_query_template = _load_prompt_pair(args.prompts_root, version, task=args.task)
    annotations = _load_annotations(args)
    for ann in annotations:
        ann.setdefault("user_id", "baseline_user" if args.task == "math" else "expertqa_user")
        ann.setdefault("model", args.assistant_model or args.user_model)
    assistant_model_name = args.assistant_model or args.user_model
    assistant_provider = args.assistant_llm_provider or args.llm_provider
    user_model_client = SingleModelClient(args.user_model, provider=args.llm_provider)
    assistant_model_client = SingleModelClient(assistant_model_name, provider=assistant_provider)
    knowledge_level_instructions = get_knowledge_level_instructions(args.knowledge_level)

    # ── Determine output path and load existing results ───────────────────
    output_base = "expertqa" if args.task == "expertqa" else "competition_math"
    if args.flat_output_layout:
        # See runner.py:flat_output_layout. Drop both <task> and <model>; the
        # umbrella (with task baked in) lives in --output_dir.
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, output_base, assistant_model_name)
    if args.run_id:
        output_dir = os.path.join(output_dir, args.run_id)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        conversation_json_name(
            version=version,
            method=args.baseline_method,
            knowledge_level=args.knowledge_level,
            assistant_strategy=args.assistant_strategy,
            assistant_model_name=assistant_model_name,
            absorption_mode=args.absorption_mode,
        ),
    )

    # ── Skip-already-done: load existing results, identify completed vs failed ──
    existing_results: List[Dict[str, Any]] = []
    done_ids: set = set()
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            for item in existing_results:
                pid = item.get("problem_id")
                if not pid:
                    continue
                stop = item.get("stop_reason", "")
                if stop not in ("api_error", "error", "empty_user_message", "empty_assistant"):
                    done_ids.add(pid)
            failed_count = len(existing_results) - len(done_ids)
            print(f"Loaded {len(existing_results)} existing results "
                  f"({len(done_ids)} complete, {failed_count} to retry)")
        except Exception:
            existing_results = []

    # Filter annotations to only run problems that aren't already done
    all_annotations = annotations[:]
    annotations = [ann for ann in annotations if ann.get("problem_id", "") not in done_ids]
    if not annotations:
        print("All conversations already complete — nothing to run.")
        return

    print(f"Running {len(annotations)} conversations ({len(all_annotations) - len(annotations)} skipped as done)")

    # ── Build user profiles (only for annotations we'll actually run) ─────
    user_profiles = None
    initial_knowledge_states = None
    if args.baseline_method == "zero-shot-cot-user-profile":
        if args.disable_interaction_style:
            # Level-fixed synthetic profile; matches runner.py:426 behavior
            # for the zero-shot-cot-user-profile structured version.
            user_profiles = [get_synthetic_user_profile(args.knowledge_level, task=args.task)] * len(annotations)
        else:
            user_profiles = _build_user_profiles(
                annotations,
                args.interaction_profile_path,
                args.seed,
            )
        initial_knowledge_states = _build_initial_knowledge_states(
            annotations, args.iu_cache_path, args.knowledge_level, args.seed
        )
        if initial_knowledge_states is None:
            print("WARNING: --iu_cache_path not set or not found. {initial_knowledge_state} will be empty.")

    results = await run_conversation_batch(
        problems=[ann["question"] for ann in annotations],
        user_model_client=user_model_client,
        assistant_model_client=assistant_model_client,
        prompt_initial_query_template=prompt_initial_query_template,
        prompt_template=prompt_template,
        task=args.task,
        assistant_strategy=args.assistant_strategy,
        user_profiles=user_profiles,
        knowledge_level_instructions=knowledge_level_instructions,
        initial_knowledge_states=initial_knowledge_states,
        user_temperature=0.7,
        assistant_temperature=0.0,
        max_tokens=3000,
        max_turns=args.max_turns,
        show_progress=True,
        reference_answers=[str(ann.get("solution", "") or "") for ann in annotations],
        disable_length_enforcement=args.disable_length_enforcement,
        # Note: we do NOT pass disable_early_stop here. Baselines naturally terminate
        # at the user simulator's <terminate>true</terminate> signal — extending past
        # that point produces degraded "I'm done" / empty assistant exchanges.
        # baseline_termination_judge.py adds judged_end_turn post-hoc for analysis.
    )

    for idx, item in enumerate(results):
        item["simulation_method"] = args.baseline_method
        item["assistant_strategy"] = args.assistant_strategy
        item["task"] = args.task
        item["knowledge_level"] = args.knowledge_level
        item["absorption_mode"] = args.absorption_mode
        if idx < len(annotations):
            item["problem_id"] = annotations[idx].get("problem_id", "")
            item["reference_answer"] = str(annotations[idx].get("solution", "") or "")
            if user_profiles:
                item["user_profile"] = user_profiles[idx]
            if args.task == "expertqa":
                item["metadata"] = annotations[idx].get("metadata", {})
                item["answer_variant"] = annotations[idx].get("answer_variant", "")
                item["answer_source"] = annotations[idx].get("answer_source", "")
                item["annotator_id"] = annotations[idx].get("annotator_id", "")

    output_base = "expertqa" if args.task == "expertqa" else "competition_math"
    if args.flat_output_layout:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, output_base, assistant_model_name)
    if args.run_id:
        output_dir = os.path.join(output_dir, args.run_id)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        conversation_json_name(
            version=version,
            method=args.baseline_method,
            knowledge_level=args.knowledge_level,
            assistant_strategy=args.assistant_strategy,
            assistant_model_name=assistant_model_name,
            absorption_mode=args.absorption_mode,
        ),
    )

    # ── Merge with existing file (same pattern as structured runner) ─────
    # For each problem_id in this run, prefer the new result; carry over any
    # existing entries for problem_ids not in scope this run. Broken old
    # entries (empty_user_message / empty_assistant / error) are dropped in
    # favor of new entries with the same problem_id.
    existing_results_for_merge: List[Dict[str, Any]] = []
    if not args.no_resume and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_results_for_merge = json.load(f)
        except Exception:
            existing_results_for_merge = []

    new_by_pid: Dict[Any, Dict[str, Any]] = {
        r.get("problem_id"): r for r in results if r.get("problem_id")
    }
    current_pids = {ann.get("problem_id") for ann in all_annotations}
    merged: List[Dict[str, Any]] = []
    # 1) For each annotation in this run: prefer new result, fall back to existing
    for ann in all_annotations:
        pid = ann.get("problem_id")
        if pid in new_by_pid:
            merged.append(new_by_pid[pid])
        else:
            existing_item = next(
                (it for it in existing_results_for_merge if it.get("problem_id") == pid),
                None,
            )
            if existing_item:
                merged.append(existing_item)
    # 2) Carry over existing entries for problem_ids outside this run's scope
    for item in existing_results_for_merge:
        pid = item.get("problem_id")
        if pid and pid not in current_pids and pid not in new_by_pid:
            merged.append(item)

    # Back up existing file before overwriting
    if existing_results_for_merge and os.path.exists(out_path):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        bak_path = out_path.replace(".json", f"_{ts}.bak.json")
        os.rename(out_path, bak_path)
        print(f"Backed up existing file to: {bak_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved {len(merged)} results ({len(results)} new) to: {out_path}")

    # Per-run HTML viewers are development tooling and are omitted from this
    # release; the run's JSON output is unaffected.


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
