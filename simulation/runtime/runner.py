"""CLI entry point for the reconstructed simulator."""

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
from ..knowledge.iu_init import initialize_knowledge_state, score_to_level
from ..profiles.interaction import format_interaction_profile, load_interaction_profiles
from .conversation import run_conversation_with_interaction_profile
from ..core.paths import conversation_json_name
from .knowledge_levels import get_knowledge_level_instructions, get_synthetic_user_profile
from .length_control import count_words, round_down_to_nearest_5, round_up_to_nearest_5

def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstructed tutoring simulator.")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument(
        "--simulation_method",
        type=str,
        default="structured",
        choices=["structured"],
        help="Kept for output-naming compatibility; only 'structured' is supported here. Use baseline_runner.py for baselines.",
    )
    parser.add_argument("--task", type=str, default="math", choices=["math", "expertqa"])
    parser.add_argument("--annotation_id", type=str, default="math_tutoring_annotations")
    parser.add_argument("--num_conversations", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0,
                        help="Skip this many rows from the start of the CSV (0-based). "
                             "problem_id will still reflect the original row index.")
    parser.add_argument(
        "--problem_indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific CSV row indices to run (non-contiguous list). When set, "
             "overrides --start_index/--num_conversations: only the listed rows "
             "are loaded, and problem_id is the literal CSV row index so IU-cache "
             "keys keep their original meaning.",
    )
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
    parser.add_argument("--iu_model", type=str, default="gpt-5.2")
    parser.add_argument(
        "--iu_max_tokens",
        type=int,
        default=0,
        help="Max completion tokens for IU graph extraction; 0 = auto (high enough for 10–30 IUs).",
    )
    parser.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "gemini", "anthropic", "together", "groq", "openrouter"])
    parser.add_argument("--assistant_llm_provider", type=str, default=None,
                        help="LLM provider for the assistant model only. If not set, uses --llm_provider.")
    parser.add_argument("--assistant_gemini_thinking_level", type=str, default=None,
                        help="Gemini thinking level for the assistant model only (e.g. LOW, MINIMAL, NONE). "
                             "Overrides the default per-model policy without affecting the user simulator.")
    parser.add_argument("--disable_interaction_style", action="store_true")
    parser.add_argument("--disable_length_enforcement", action="store_true",
                        help="Skip the post-processing assistant-length enforcement "
                             "(_enforce_assistant_length). Recommended for validation-study "
                             "comparisons since the deployed Next.js app has no such layer.")
    parser.add_argument("--length_control", action="store_true")
    parser.add_argument("--length_control_setting", type=str, default="range")
    parser.add_argument("--dynamic_knowledge_state_init", action="store_true")
    parser.add_argument("--data_root", type=str, default="Data")
    parser.add_argument(
        "--interaction_profile_path",
        type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "profiles", "interaction_style.json")
        ),
    )
    parser.add_argument("--prompts_root", type=str, default="simulation/prompts")
    parser.add_argument("--input_csv", type=str, default="Data/competition_math/data/train_fixed_level_4_5.csv")
    parser.add_argument(
        "--expertqa_jsonl",
        type=str,
        default="Data/expertqa/r2_compiled_anon.jsonl",
    )
    parser.add_argument("--expertqa_answer_variant", type=str, default="")
    parser.add_argument("--knowledge_level", type=str, default="intermediate", choices=["novice", "intermediate", "advanced"])
    parser.add_argument("--prescreening_score", type=int, default=None,
                        help="Prescreening test score (1-10). When provided, overrides --knowledge_level "
                             "for KS initialization using piecewise-linear ratio interpolation.")
    parser.add_argument("--seed", type=int, default=2)
    # Conversations always run to max_turns now; --disable_early_stop is the
    # default behavior and only the virtual_* markers are recorded. The flag is
    # kept for backward-compat with existing YAML configs and CLIs but ignored.
    parser.add_argument("--disable_early_stop", action="store_true",
                        help="(Deprecated, no-op) Conversations always run to max_turns; "
                             "score-at-end is computed post-hoc by run_experiment.py.")
    parser.add_argument("--max_turns", type=int, default=15,
                        help="Maximum number of user-assistant exchanges per conversation (default: 15).")
    from simulation.knowledge.iu_init import INIT_FORMULAS as _INIT_FORMULAS
    parser.add_argument("--init_formula", type=str, default="p1",
                        choices=list(_INIT_FORMULAS),
                        help="KS initialisation formula. 'p1' = P1 calibrated-linear (default). "
                             "'p2_exp' = exp-decay ZPD boost (higher partial at novice, flat at intermediate+).")
    from simulation.knowledge.update_v2 import ABSORPTION_PRESETS as _ABSORPTION_PRESETS
    parser.add_argument("--absorption_mode", type=str, default="default",
                        choices=sorted(_ABSORPTION_PRESETS.keys()),
                        help="Cognitive-overhead absorption regime. "
                             "'default' = uniform-load + hard cutoff under overload + per-level threshold. "
                             "'weightabsorb' = state-weighted load + soft per-IU dropoff + universal threshold. "
                             "'weightabsorb_attempt' = weightabsorb + user-side attempt_load (reasoning + articulation). "
                             "See KNOWLEDGE_STATE_UPDATE.md for the semantic difference.")
    parser.add_argument("--temporal_decay", action="store_true",
                        help="Enable turn-fatigue temporal decay on absorption efficiency. "
                             "efficiency *= max_turns / (max_turns + t - 1). "
                             "Only effective when the preset has dropoff=True.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Root output directory. Run files land in <output_dir>/<task>/<assistant_model>/. "
                             "Default 'output' matches the orchestrator and dashboards.")
    parser.add_argument("--iu_cache_path", type=str, default="",
                        help="Path to a pre-built IU graph JSON cache. When provided, skips re-extraction "
                             "for any problem_id already present in the cache.")
    parser.add_argument("--init_ks_cache_path", type=str, default="",
                        help="Path to a pre-computed init KS JSON cache (keyed by task/problem_id/level). "
                             "When provided, uses cached initial knowledge states instead of computing "
                             "dynamically, ensuring identical init KS across all models.")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not skip conversations already present in the output JSON; re-run all requested problem_ids.",
    )
    parser.add_argument("--run_id", type=str, default="",
                        help="Run identifier (passed by the orchestrator). "
                             "When set, outputs go into <output_dir>/<task>/<model>/<run_id>/.")
    parser.add_argument("--flat_output_layout", action="store_true",
                        help="Drop the <assistant_model>/ segment from the output path. "
                             "Used by the validation per-participant runners which group by "
                             "(arm, run_id, participant) instead of by model. With this flag, "
                             "output goes to <output_dir>/<task>/<run_id>/ (no model parent).")
    parser.add_argument("--simulator_flags", type=str, default="",
                        help="JSON-encoded SimulatorFeatureFlags dict (see "
                             "simulation/core/feature_flags.py). Set by the experiment "
                             "orchestrator from the YAML config's simulator_flags: section. "
                             "When omitted, flags are loaded from env vars (back-compat).")
    return parser


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


def _load_prompt_pair(prompts_root: str, version: str) -> tuple[str, str]:
    """Simulator prompts live under ``<prompts_root>/simulator/``.

    The active user-sim ``prompt_variant`` (``"canonical"`` or ``"tightened"``)
    is read from the typed feature-flags config (see
    ``simulation/core/feature_flags.py``). The ``"tightened"`` variant loads
    ``{version}-tightened.txt`` / ``{version}-tightened-initial-query.txt``
    while keeping the surface-level version string unchanged, so run output
    filenames stay clean. Falls back to canonical if the requested variant's
    files don't exist for this version.
    """
    from ..core.feature_flags import get_flags
    flags = get_flags()
    variant = flags.user_sim.prompt_variant
    noterm = not flags.user_sim.allow_terminate

    # Build a list of candidate suffixes in preference order. First match wins.
    # tightened+noterm requires the combined file; falls back to tightened, then
    # noterm, then canonical if intermediate files are missing for this version.
    if variant == "tightened" and noterm:
        suffixes = ["-tightened-noterm", "-tightened", "-noterm", ""]
    elif variant == "tightened":
        suffixes = ["-tightened", ""]
    elif noterm:
        suffixes = ["-noterm", ""]
    else:
        suffixes = [""]

    for suffix in suffixes:
        prompt_path = os.path.join(prompts_root, "simulator", f"{version}{suffix}.txt")
        initial_path = os.path.join(prompts_root, "simulator", f"{version}{suffix}-initial-query.txt")
        if os.path.exists(prompt_path) and os.path.exists(initial_path):
            return load_prompt(prompt_path), load_prompt(initial_path)
    raise FileNotFoundError(f"No prompt files found for version={version!r} under {prompts_root}/simulator/")


def _build_length_control_list(
    annotations: List[dict],
    length_control_setting: str,
) -> List[str]:
    length_control_list = []
    for ann in annotations:
        user_queries = ann["user_queries"]
        problem_turns = ann["problem_1_turns"] if ann["problem_1_turns"] > 0 else len(user_queries)
        user_queries = user_queries[:problem_turns]
        query_length_list = [count_words(query) for query in user_queries]
        if length_control_setting == "range":
            rounded_min = int(round_down_to_nearest_5(min(query_length_list)))
            rounded_max = int(round_up_to_nearest_5(max(query_length_list)))
            length_text = f"between {rounded_min} and {rounded_max} words"
        elif length_control_setting == "average":
            avg_length = sum(query_length_list) / len(query_length_list)
            length_text = f"around {int(round_up_to_nearest_5(avg_length))} words"
        else:
            raise ValueError(f"Unsupported length_control_setting: {length_control_setting}")
        length_control_list.append(length_text)
    return length_control_list


def _setup_error_logger(output_dir: str, version: str, knowledge_level: str, strategy: str, model: str) -> logging.Logger:
    """Set up a file logger that captures errors during conversation generation."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace("/", "_")
    log_file = os.path.join(output_dir, f"errors_{version}_{knowledge_level}_{strategy}_{model_safe}_{ts}.log")
    # Use the shared "sim_error" logger so conversation.py can also write to it
    logger = logging.getLogger("sim_error")
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers across multiple runs in the same process
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
    else:
        # Replace the file handler to point to the new log file
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
    logger.info(f"Error log initialized: version={version}, level={knowledge_level}, strategy={strategy}, model={model}")
    print(f"Error log: {log_file}")
    return logger


async def main() -> None:
    parser = cli_parser()
    args = parser.parse_args()

    # Install feature-flag config:
    #   1. Preferred — ``--simulator_flags`` CLI arg (JSON), set by the
    #      experiment orchestrator from the YAML's ``simulator_flags:`` section.
    #   2. Fallback — env vars (``MYSIMARE_FLAGS`` JSON or legacy individual
    #      ``MYSIMARE_GATE5_PARTIAL=1`` etc.) for ad-hoc CLI invocations.
    #   3. Dataclass defaults if neither is set.
    # All pipeline-stage modules read via ``get_flags()`` after this point.
    import json as _json
    from ..core.feature_flags import SimulatorFeatureFlags, init_flags
    if args.simulator_flags:
        try:
            init_flags(SimulatorFeatureFlags.from_dict(_json.loads(args.simulator_flags)))
        except (ValueError, TypeError) as _err:
            print(f"WARNING: failed to parse --simulator_flags JSON ({_err}); "
                  f"falling back to env vars.")
            init_flags(SimulatorFeatureFlags.from_env())
    else:
        init_flags(SimulatorFeatureFlags.from_env())

    # When prescreening_score is set, derive the categorical level for
    # secondary systems (user profile text, overload threshold, instructions).
    effective_knowledge_level = (
        score_to_level(args.prescreening_score) if args.prescreening_score is not None
        else args.knowledge_level
    )
    os.environ["SIM_USER_LEVEL"] = effective_knowledge_level
    knowledge_level_instructions = get_knowledge_level_instructions(effective_knowledge_level)

    prompt_template, prompt_initial_query_template = _load_prompt_pair(args.prompts_root, args.version)

    annotations: List[Dict[str, str]] = []
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
                    "user_id": "expertqa_user",
                    "model": args.assistant_model or args.user_model,
                }
            )
    else:
        # Load problems from CSV (question + reference answer)
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
                    "user_id": "csv_user",
                    "model": args.assistant_model or args.user_model,
                }
            )

    if args.num_conversations > 0 and not args.problem_indices:
        annotations = annotations[:args.num_conversations]

    assistant_model_name = args.assistant_model or args.user_model
    output_base = "expertqa" if args.task == "expertqa" else "competition_math"
    if args.flat_output_layout:
        # Validation umbrella layout: drop both <task> and <assistant_model>
        # segments. The umbrella prefix (which includes the task) is expected
        # to be baked into --output_dir by the caller, so the final path is
        # <output_dir>/<run_id>/. Matches paths.run_dir(flat=True).
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, output_base, assistant_model_name)
    if args.run_id:
        output_dir = os.path.join(output_dir, args.run_id)
    os.makedirs(output_dir, exist_ok=True)
    # When prescreening_score is set, use "score_N" as the level segment so
    # each score lands in a distinct output file.
    effective_level_label = (
        f"score_{args.prescreening_score}" if args.prescreening_score is not None
        else args.knowledge_level
    )
    out_path = os.path.join(
        output_dir,
        conversation_json_name(
            version=args.version,
            method=args.simulation_method,
            knowledge_level=effective_level_label,
            assistant_strategy=args.assistant_strategy,
            assistant_model_name=assistant_model_name,
            absorption_mode=args.absorption_mode,
        ),
    )
    error_logger = _setup_error_logger(
        output_dir, args.version, args.knowledge_level, args.assistant_strategy, assistant_model_name,
    )

    # ── Skip-already-done: load existing results and only re-run failures ──
    existing_results: List[Dict] = []
    completed_pids: set = set()
    if not args.no_resume and os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as _f:
                existing_results = json.load(_f)
            for item in existing_results:
                pid = str(item.get("problem_id", ""))
                stop = item.get("stop_reason", "")
                if pid and stop not in ("error", "api_error", "empty_user_message", "empty_assistant", "phase_b_empty"):
                    completed_pids.add(pid)
            error_logger.info(
                f"Loaded {len(existing_results)} existing results from {out_path} "
                f"({len(completed_pids)} complete, {len(existing_results) - len(completed_pids)} to retry)"
            )
            print(
                f"Loaded {len(existing_results)} existing results "
                f"({len(completed_pids)} complete, {len(existing_results) - len(completed_pids)} to retry)"
            )
        except (json.JSONDecodeError, KeyError) as exc:
            error_logger.warning(f"Could not load existing results: {exc}")
            existing_results = []

    original_annotations = annotations[:]
    annotations = [ann for ann in annotations if str(ann["problem_id"]) not in completed_pids]
    if not annotations:
        print("All conversations already complete — nothing to run.")
        error_logger.info("All conversations already complete — skipping run.")
        return

    assistant_provider = args.assistant_llm_provider or args.llm_provider
    user_model_client = SingleModelClient(args.user_model, provider=args.llm_provider)
    assistant_kwargs = {"provider": assistant_provider}
    if args.assistant_gemini_thinking_level:
        assistant_kwargs["gemini_thinking_level"] = args.assistant_gemini_thinking_level
    assistant_model_client = SingleModelClient(assistant_model_name, **assistant_kwargs)
    problems = [ann["question"] for ann in annotations]
    reference_answers = [str(ann.get("solution", "") or "") for ann in annotations]
    results = []
    iu_model_client = SingleModelClient(args.iu_model, provider=args.llm_provider)

    # Build IU graphs from question + answer, then convert to concept graph.
    # If a cache file is provided, load from it and only extract missing problems.
    iu_graphs: Dict[str, Dict] = {}
    if args.iu_cache_path and os.path.exists(args.iu_cache_path):
        with open(args.iu_cache_path, encoding="utf-8") as _f:
            iu_graphs = json.load(_f)
        print(f"Loaded IU graph cache from: {args.iu_cache_path} ({len(iu_graphs)} problems)")

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
        if args.iu_cache_path:
            print(f"Cache missing {len(missing_annotations)} problem(s) — extracting now.")
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

    length_control_list = []
    if args.length_control and args.task == "math":
        length_control_list = _build_length_control_list(annotations, args.length_control_setting)
    elif args.length_control:
        logging.warning(
            "--length_control is not supported for task=%s and will be skipped.", args.task
        )

    rng = random.Random(args.seed)
    interaction_profiles: Dict[str, List[Dict[str, str]]] = {}
    if args.interaction_profile_path and os.path.exists(args.interaction_profile_path):
        interaction_profiles = load_interaction_profiles(args.interaction_profile_path)

    # Build interaction-only user profiles (fallback if no profile file is available)
    use_synthetic_profile = args.version.startswith("zero-shot-cot-user-profile")
    user_profiles = []
    for ann, length_text in zip(annotations, length_control_list or ["" for _ in annotations]):
        if use_synthetic_profile:
            user_profiles.append(get_synthetic_user_profile(effective_knowledge_level))
        else:
            features = []
            if not args.disable_interaction_style:
                features = _get_interaction_features(
                    interaction_profiles,
                    ann["problem_id"],
                    ann["user_id"],
                    ann["model"],
                    rng,
                )
            user_profiles.append(format_interaction_profile(features, length_text or "around 20 words"))

    knowledge_states = None
    # --- Init KS from pre-computed cache (canonical) ---
    _init_ks_cache: Optional[Dict] = None
    if args.init_ks_cache_path and os.path.exists(args.init_ks_cache_path):
        with open(args.init_ks_cache_path) as _ikf:
            _init_ks_cache = json.load(_ikf)
        logging.info("Loaded init KS cache from %s", args.init_ks_cache_path)

    if args.dynamic_knowledge_state_init:
        rng = random.Random(args.seed)
        # Resolve task key for init KS cache lookup
        _iks_task_key = {"math": "competition_math", "expertqa": "expertqa"}.get(args.task, args.task)
        knowledge_states = []
        for pid, ann in zip(problem_ids, annotations):
            # --- Try pre-computed init KS cache first ---
            if _init_ks_cache is not None:
                cached_ks = (_init_ks_cache
                             .get(_iks_task_key, {})
                             .get(str(pid), {})
                             .get(effective_knowledge_level))
                if cached_ks:
                    knowledge_states.append(cached_ks)
                    continue

            iu_graph = iu_graphs.get(str(pid), {})
            id_map = id_maps.get(str(pid), {})
            edges = iu_graph.get("edges", [])
            nodes = iu_graph.get("nodes", [])
            question_text = ann.get("question") or ""
            mentioned_ids = set()
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
                    "- Return a JSON object: {\"mentioned_ids\": [\"IU1\", ...]}.\n"
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

            state = initialize_knowledge_state(
                iu_graph, args.knowledge_level, rng,
                prescreening_score=args.prescreening_score,
                init_formula=args.init_formula,
            )
            known_ids = set(state.get("known", []))
            partially_ids = set(state.get("partially_known", []))
            struggling_ids = set(state.get("struggling", []))  # pilot3: explicit struggling band
            unknown_ids = set(state.get("unknown", []))

            mapped: Dict[str, Dict[str, str]] = {}
            for iu_id in known_ids:
                mapped[id_map.get(iu_id, iu_id)] = {"state": "knows_well"}

            for iu_id in partially_ids:
                prereqs = prereqs_by_id.get(iu_id, [])
                known_ratio = (
                    len([p for p in prereqs if p in known_ids]) / max(len(prereqs), 1)
                )
                state_label = "partial_understanding" if known_ratio >= 0.5 else "struggling"
                mapped[id_map.get(iu_id, iu_id)] = {"state": state_label}

            for iu_id in struggling_ids:
                # pilot3: explicitly assigned struggling band (from
                # _IU_INIT_RATIOS_4STATE; see iu_init.py).
                mapped[id_map.get(iu_id, iu_id)] = {"state": "struggling"}

            for iu_id in unknown_ids:
                prereqs = prereqs_by_id.get(iu_id, [])
                if iu_id in mentioned_ids:
                    state_label = "unaware"
                else:
                    state_label = "unaware"
                mapped[id_map.get(iu_id, iu_id)] = {"state": state_label}

            if (
                effective_knowledge_level == "novice"
                and not known_ids
                and not partially_ids
                and unknown_ids
            ):
                fallback_id = next(iter(unknown_ids))
                mapped[id_map.get(fallback_id, fallback_id)] = {"state": "partial_understanding"}

            knowledge_states.append(mapped)

    try:
        results = await run_conversation_with_interaction_profile(
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
            knowledge_levels=[effective_knowledge_level] * len(problems),
            task=args.task,
            assistant_strategy=args.assistant_strategy,
            knowledge_level_instructions=knowledge_level_instructions,
            user_temperature=0.7,
            assistant_temperature=0.0,
            max_tokens=3000,
            max_turns=args.max_turns,
            length_control_bool=args.length_control,
            length_control_list=length_control_list,
            show_progress=True,
            disable_early_stop=True,  # Always-on: orchestrator computes score-at-end post-hoc.
            reference_answers=reference_answers,
            absorption_mode=args.absorption_mode,
            temporal_decay=args.temporal_decay,
            disable_length_enforcement=args.disable_length_enforcement,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        error_logger.error("FATAL: run_conversation_with_interaction_profile crashed\n%s", tb)
        print(f"ERROR: structured conversation failed — see error log. {exc}")
    for idx, item in enumerate(results):
        if idx < len(annotations):
            item["problem_id"] = str(annotations[idx].get("problem_id", idx))
        item["knowledge_level"] = effective_knowledge_level
        if args.prescreening_score is not None:
            item["prescreening_score"] = args.prescreening_score
        item["simulation_method"] = args.simulation_method
        item["assistant_strategy"] = args.assistant_strategy
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

    # ── Merge new results with existing ──
    # Build a map of new results by problem_id
    new_by_pid = {}
    for item in results:
        pid = str(item.get("problem_id", ""))
        if pid:
            new_by_pid[pid] = item

    # Merge: for each problem in the original annotation set, pick the best result
    merged_results = []
    for ann in original_annotations:
        pid = str(ann["problem_id"])
        if pid in new_by_pid:
            merged_results.append(new_by_pid[pid])
        else:
            # Use existing result for this problem_id
            existing_item = next((item for item in existing_results if str(item.get("problem_id", "")) == pid), None)
            if existing_item:
                merged_results.append(existing_item)

    # Also carry over existing results for problem_ids outside the current annotation scope
    # (e.g. when runner is called once per problem via --start_index / --num_conversations 1).
    # Deduplicate by problem_id: prefer non-error; last-seen wins among errors.
    current_pids = {str(ann["problem_id"]) for ann in original_annotations}
    carry_over: dict = {}
    error_stops = {"error", "api_error", "empty_user_message", "empty_assistant"}
    for item in existing_results:
        pid = str(item.get("problem_id", ""))
        if not pid or pid in current_pids:
            continue
        prev = carry_over.get(pid)
        if prev is None or prev.get("stop_reason") in error_stops:
            carry_over[pid] = item
    merged_results.extend(carry_over.values())

    # Count errors in merged results
    error_count = sum(1 for item in merged_results if item.get("stop_reason") in ("error", "api_error", "empty_user_message", "empty_assistant"))
    total = len(merged_results)
    if error_count:
        error_logger.warning(f"{error_count}/{total} conversations have errors after merge")
        print(f"WARNING: {error_count}/{total} conversations have errors — re-run to retry")

    if merged_results:
        # Backup existing file before overwriting
        if os.path.exists(out_path):
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = out_path.replace(".json", f"_{ts}.bak.json")
            os.rename(out_path, backup_path)
            error_logger.info(f"Backed up existing results to {backup_path}")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, indent=2)
        print(f"Saved {total} results ({total - error_count} complete, {error_count} errors) to: {out_path}")
        error_logger.info(f"Saved {total} results ({total - error_count} complete, {error_count} errors) to {out_path}")
    else:
        error_logger.error("No results produced — output file not written")
        print("ERROR: no results produced — check error log")

    # The internal build renders per-run HTML views of the conversations and the
    # LLM-call trace here. Those viewers are development tooling and are not part
    # of this release; the run's JSON output is unaffected.


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
