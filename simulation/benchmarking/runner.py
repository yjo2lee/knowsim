"""CLI entry point for a single benchmarking condition.

Runs one (model x level x seed) cell. Called as a subprocess by
``run_benchmarking.py`` or directly for testing:

    python -m simulation.benchmarking.runner \
        --task math \
        --knowledge_level novice \
        --assistant_model gpt-5.4 \
        --seed 42 \
        --num_conversations 2 \
        --input_csv Data/competition_math/data/train_fixed_level_4_5.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.models import SingleModelClient
from ..core.prompts import load_prompt
from ..core.logging import get_log_path
from ..data.expertqa_answer import select_expertqa_answer
from ..data.loaders import load_dataset_rows, load_jsonl_rows, row_problem_id
from ..knowledge.iu_extraction import extract_iu_graph
from ..knowledge.iu_graph import build_concept_graph_from_iu
from ..knowledge.iu_init import initialize_knowledge_state
from ..profiles.interaction import format_interaction_profile, load_interaction_profiles
from ..runtime.knowledge_levels import get_knowledge_level_instructions
from .conversation import run_benchmarking_conversations


def _benchmarking_output_name(
    knowledge_level: str,
    assistant_model_name: str,
    seed: int,
    absorption_mode: str = "default",
) -> str:
    suffix = "" if absorption_mode == "default" else f"_{absorption_mode}"
    return f"benchmarking_{knowledge_level}_{assistant_model_name}{suffix}_seed{seed}.json"


def _normalize_problem_id(problem_id: str) -> str:
    try:
        return str(int(problem_id))
    except (TypeError, ValueError):
        return str(problem_id)


def _extract_json_object(text: str) -> Dict[str, object]:
    if not text:
        return {}
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


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


def _setup_error_logger(output_dir: str, knowledge_level: str, model: str, seed: int) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"errors_benchmarking_{knowledge_level}_{model}_seed{seed}_{ts}.log")
    logger = logging.getLogger("sim_error")
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.info(f"Benchmarking error log: level={knowledge_level}, model={model}, seed={seed}")
    print(f"Error log: {log_file}")
    return logger


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmarking runner — single condition.")
    parser.add_argument("--task", type=str, default="math", choices=["math", "expertqa"])
    parser.add_argument("--knowledge_level", type=str, default="intermediate",
                        choices=["novice", "intermediate", "advanced"])
    parser.add_argument("--num_conversations", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--user_model", type=str, default="gemini-3.1-flash")
    parser.add_argument("--assistant_model", type=str, required=True)
    parser.add_argument("--assistant_llm_provider", type=str, default=None)
    parser.add_argument("--iu_model", type=str, default="gpt-5.2")
    parser.add_argument("--iu_max_tokens", type=int, default=0)
    parser.add_argument("--llm_provider", type=str, default="openai",
                        choices=["openai", "gemini", "anthropic", "together", "groq"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--absorption_mode", type=str, default="default")
    parser.add_argument("--input_csv", type=str,
                        default="Data/competition_math/data/train_fixed_level_4_5.csv")
    parser.add_argument("--expertqa_jsonl", type=str,
                        default="Data/expertqa/r2_compiled_anon.jsonl")
    parser.add_argument("--interaction_profile_path", type=str,
                        default=os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "..", "profiles", "interaction_style.json")
                        ))
    parser.add_argument("--prompts_root", type=str, default="simulation/prompts")
    parser.add_argument("--output_dir", type=str, default="output_benchmarking")
    parser.add_argument("--iu_cache_path", type=str, default="")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--run_id", type=str, default="")
    # Accepted for compatibility with extra_flags from config; always-on in benchmarking.
    parser.add_argument("--dynamic_knowledge_state_init", action="store_true")
    return parser


async def main() -> None:
    parser = cli_parser()
    args = parser.parse_args()

    knowledge_level = args.knowledge_level
    os.environ["SIM_USER_LEVEL"] = knowledge_level
    knowledge_level_instructions = get_knowledge_level_instructions(knowledge_level)

    # Load prompt templates (same as structured simulator)
    version = "dynamic-knowledge-state"
    prompt_dir = os.path.join(args.prompts_root, "simulator")
    prompt_template = load_prompt(os.path.join(prompt_dir, f"{version}.txt"))
    prompt_initial_query_template = load_prompt(os.path.join(prompt_dir, f"{version}-initial-query.txt"))

    # Load data
    annotations: List[Dict] = []
    if args.task == "expertqa":
        rows = load_jsonl_rows(args.expertqa_jsonl)
        if args.start_index > 0:
            rows = rows[args.start_index:]
        if args.num_conversations > 0:
            rows = rows[:args.num_conversations]
        for idx, row in enumerate(rows):
            answer_text, answer_variant, answer_source = select_expertqa_answer(
                row.get("answers", {}) or {}, "",
            )
            annotations.append({
                "problem_id": row_problem_id(row, idx + args.start_index),
                "question": row.get("question", ""),
                "solution": answer_text,
                "metadata": row.get("metadata", {}),
                "answer_variant": answer_variant,
                "answer_source": answer_source,
                "annotator_id": row.get("annotator_id", ""),
                "user_id": "expertqa_user",
                "model": args.assistant_model,
            })
    else:
        rows = load_dataset_rows(args.input_csv)
        if args.start_index > 0:
            rows = rows[args.start_index:]
        if args.num_conversations > 0:
            rows = rows[:args.num_conversations]
        for idx, row in enumerate(rows):
            annotations.append({
                "problem_id": row_problem_id(row, idx + args.start_index),
                "question": row.get("problem", ""),
                "solution": row.get("solution", ""),
                "level": row.get("level", ""),
                "type": row.get("type", ""),
                "user_id": "csv_user",
                "model": args.assistant_model,
            })

    if args.num_conversations > 0:
        annotations = annotations[:args.num_conversations]

    # Output paths
    assistant_model_name = args.assistant_model
    output_base = "expertqa" if args.task == "expertqa" else "competition_math"
    output_dir = os.path.join(args.output_dir, output_base, assistant_model_name)
    if args.run_id:
        output_dir = os.path.join(output_dir, args.run_id)
    os.makedirs(output_dir, exist_ok=True)

    out_filename = _benchmarking_output_name(
        knowledge_level=knowledge_level,
        assistant_model_name=assistant_model_name,
        seed=args.seed,
        absorption_mode=args.absorption_mode,
    )
    out_path = os.path.join(output_dir, out_filename)

    error_logger = _setup_error_logger(output_dir, knowledge_level, assistant_model_name, args.seed)

    # Resume logic
    existing_results: List[Dict] = []
    completed_pids: set = set()
    if not args.no_resume and os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as _f:
                existing_results = json.load(_f)
            for item in existing_results:
                pid = str(item.get("problem_id", ""))
                stop = item.get("stop_reason", "")
                if pid and stop not in ("error", "api_error", "empty_user_message", "empty_assistant"):
                    completed_pids.add(pid)
            print(f"Loaded {len(existing_results)} existing ({len(completed_pids)} complete)")
        except (json.JSONDecodeError, KeyError) as exc:
            error_logger.warning(f"Could not load existing results: {exc}")
            existing_results = []

    original_annotations = annotations[:]
    annotations = [ann for ann in annotations if str(ann["problem_id"]) not in completed_pids]
    if not annotations:
        print("All conversations already complete — nothing to run.")
        return

    # Create model clients
    assistant_provider = args.assistant_llm_provider or args.llm_provider
    user_model_client = SingleModelClient(args.user_model, provider=args.llm_provider)
    assistant_model_client = SingleModelClient(assistant_model_name, provider=assistant_provider)
    iu_model_client = SingleModelClient(args.iu_model, provider=args.llm_provider)

    # Build IU graphs
    iu_graphs: Dict[str, Dict] = {}
    if args.iu_cache_path and os.path.exists(args.iu_cache_path):
        with open(args.iu_cache_path, encoding="utf-8") as _f:
            iu_graphs = json.load(_f)
        print(f"Loaded IU cache: {args.iu_cache_path} ({len(iu_graphs)} problems)")

    if args.iu_max_tokens > 0:
        iu_max_tokens = args.iu_max_tokens
    elif args.llm_provider == "gemini":
        iu_max_tokens = 6000
    else:
        iu_max_tokens = 16384

    _iu_prompt_file = (
        "iu_graph_prompt_expertqa.md" if args.task == "expertqa"
        else "iu_graph_prompt.md"
    )
    _iu_prompt_path = os.path.join(args.prompts_root, "knowledge_state", _iu_prompt_file)

    missing_annotations = [ann for ann in annotations if str(ann["problem_id"]) not in iu_graphs]
    if missing_annotations:
        print(f"Extracting IU graphs for {len(missing_annotations)} problem(s)...")
        for ann in missing_annotations:
            _meta = ann.get("metadata") or {}
            iu_graph = await extract_iu_graph(
                question=ann["question"],
                answer=ann["solution"],
                model_client=iu_model_client,
                field=str(_meta.get("field", "")),
                specific_field=str(_meta.get("specific_field", "")),
                max_tokens=iu_max_tokens,
                show_progress=False,
                prompt_path=_iu_prompt_path,
            )
            iu_graphs[str(ann["problem_id"])] = iu_graph

    concept_graph, id_maps = build_concept_graph_from_iu(iu_graphs)
    problem_ids = [str(ann["problem_id"]) for ann in annotations]
    problems = [ann["question"] for ann in annotations]
    reference_answers = [str(ann.get("solution", "") or "") for ann in annotations]

    # Interaction profiles
    rng = random.Random(args.seed)
    interaction_profiles: Dict[str, List[Dict[str, str]]] = {}
    if args.interaction_profile_path and os.path.exists(args.interaction_profile_path):
        interaction_profiles = load_interaction_profiles(args.interaction_profile_path)

    user_profiles = []
    for ann in annotations:
        features = _get_interaction_features(
            interaction_profiles,
            ann["problem_id"],
            ann["user_id"],
            ann["model"],
            rng,
        )
        user_profiles.append(format_interaction_profile(features, "around 20 words"))

    # Initialize knowledge states
    knowledge_states = None
    rng = random.Random(args.seed)
    knowledge_states = []
    for pid, ann in zip(problem_ids, annotations):
        iu_graph = iu_graphs.get(str(pid), {})
        id_map = id_maps.get(str(pid), {})
        edges = iu_graph.get("edges", [])
        nodes = iu_graph.get("nodes", [])
        question_text = ann.get("question") or ""

        # Detect mentioned IUs from problem text
        mentioned_ids: set = set()
        if nodes:
            iu_items = [{"id": n.get("id", ""), "concept": n.get("concept", "")} for n in nodes]
            prompt = (
                "You are labeling which IU concepts are explicitly mentioned in the problem text.\n"
                "Return ONLY valid JSON.\n\n"
                "Problem:\n"
                f"{question_text}\n\n"
                "IUs:\n"
                f"{json.dumps(iu_items, ensure_ascii=False)}\n\n"
                "Rules:\n"
                "- Mark an IU as mentioned ONLY if the concept is clearly and directly named in the problem text.\n"
                "- Do NOT infer or paraphrase. Avoid indirect or implied mentions.\n"
                "- Only count it if the wording in the problem explicitly matches the concept name or a very close variant.\n"
                "- Do not infer from the answer; use only the problem text.\n"
                '- Return a JSON object: {"mentioned_ids": ["IU1", ...]}.\n'
            )
            responses = await iu_model_client.generate_responses(
                [[{"role": "user", "content": prompt}]],
                temperature=0.0,
                max_tokens=600,
                n=1,
                show_progress=False,
                json_mode=True,
            )
            raw = responses[0][0] if responses and responses[0] else ""
            parsed = _extract_json_object(raw)
            for iu_id in parsed.get("mentioned_ids", []) if isinstance(parsed, dict) else []:
                if iu_id:
                    mentioned_ids.add(iu_id)

        prereqs_by_id: Dict[str, List[str]] = {}
        for e in edges:
            src = e.get("from")
            dst = e.get("to")
            if not src or not dst:
                continue
            prereqs_by_id.setdefault(dst, []).append(src)

        state = initialize_knowledge_state(iu_graph, knowledge_level, rng)
        known_ids = set(state.get("known", []))
        partially_ids = set(state.get("partially_known", []))
        unknown_ids = set(state.get("unknown", []))

        mapped: Dict[str, Dict[str, str]] = {}
        for iu_id in known_ids:
            mapped[id_map.get(iu_id, iu_id)] = {"state": "knows_well"}

        for iu_id in partially_ids:
            prereqs = prereqs_by_id.get(iu_id, [])
            known_ratio = len([p for p in prereqs if p in known_ids]) / max(len(prereqs), 1)
            state_label = "partial_understanding" if known_ratio >= 0.5 else "struggling"
            mapped[id_map.get(iu_id, iu_id)] = {"state": state_label}

        for iu_id in unknown_ids:
            mapped[id_map.get(iu_id, iu_id)] = {"state": "unaware"}

        if knowledge_level == "novice" and not known_ids and not partially_ids and unknown_ids:
            fallback_id = next(iter(unknown_ids))
            mapped[id_map.get(fallback_id, fallback_id)] = {"state": "partial_understanding"}

        knowledge_states.append(mapped)

    # Run conversations
    results: List[Dict] = []
    try:
        results = await run_benchmarking_conversations(
            problems=problems,
            problem_ids=problem_ids,
            user_profiles=user_profiles,
            user_model_client=user_model_client,
            assistant_model_client=assistant_model_client,
            prompt_initial_query_template=prompt_initial_query_template,
            prompt_template=prompt_template,
            concept_graph=concept_graph,
            iu_graphs=iu_graphs,
            id_maps=id_maps,
            knowledge_states=knowledge_states,
            knowledge_levels=[knowledge_level] * len(problems),
            task=args.task,
            knowledge_level_instructions=knowledge_level_instructions,
            reference_answers=reference_answers,
            absorption_mode=args.absorption_mode,
            max_turns=args.max_turns,
            show_progress=True,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        error_logger.error("FATAL: run_benchmarking_conversations crashed\n%s", tb)
        print(f"ERROR: benchmarking conversation failed — see error log. {exc}")

    # Annotate results
    for idx, item in enumerate(results):
        if idx < len(annotations):
            item["problem_id"] = str(annotations[idx].get("problem_id", idx))
        item["knowledge_level"] = knowledge_level
        item["simulation_method"] = "benchmarking"
        item["assistant_strategy"] = "passthrough"
        item["assistant_model"] = assistant_model_name
        item["seed"] = args.seed
        item["task"] = args.task
        if idx < len(annotations):
            item["reference_answer"] = str(annotations[idx].get("solution", "") or "")
            item["level"] = str(annotations[idx].get("level", ""))
            item["type"] = str(annotations[idx].get("type", ""))
        if args.task == "expertqa" and idx < len(annotations):
            item["metadata"] = annotations[idx].get("metadata", {})
            item["answer_variant"] = annotations[idx].get("answer_variant", "")
            item["answer_source"] = annotations[idx].get("answer_source", "")
            item["annotator_id"] = annotations[idx].get("annotator_id", "")

    # Merge with existing results
    new_by_pid = {}
    for item in results:
        pid = str(item.get("problem_id", ""))
        if pid:
            new_by_pid[pid] = item

    merged_results = []
    for ann in original_annotations:
        pid = str(ann["problem_id"])
        if pid in new_by_pid:
            merged_results.append(new_by_pid[pid])
        else:
            existing_item = next((item for item in existing_results if str(item.get("problem_id", "")) == pid), None)
            if existing_item:
                merged_results.append(existing_item)

    # Carry over existing results for problem_ids outside current scope
    current_pids = {str(ann["problem_id"]) for ann in original_annotations}
    error_stops = {"error", "api_error", "empty_user_message", "empty_assistant"}
    carry_over: dict = {}
    for item in existing_results:
        pid = str(item.get("problem_id", ""))
        if not pid or pid in current_pids:
            continue
        prev = carry_over.get(pid)
        if prev is None or prev.get("stop_reason") in error_stops:
            carry_over[pid] = item
    merged_results.extend(carry_over.values())

    error_count = sum(1 for item in merged_results if item.get("stop_reason") in error_stops)
    total = len(merged_results)
    if error_count:
        print(f"WARNING: {error_count}/{total} conversations have errors — re-run to retry")

    if merged_results:
        if os.path.exists(out_path):
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = out_path.replace(".json", f"_{ts}.bak.json")
            os.rename(out_path, backup_path)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, indent=2)
        print(f"Saved {total} results ({total - error_count} complete) to: {out_path}")
    else:
        print("ERROR: no results produced")

    # The per-run HTML conversation viewer is development tooling and is omitted
    # from this release; the run's JSON output is unaffected.


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
