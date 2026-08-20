"""Benchmarking conversation loop — passthrough (no system prompt) variant.

Mirrors the structured path in ``simulation.runtime.conversation`` but:
- No assistant system prompt injected (passthrough mode).
- No length enforcement (_enforce_assistant_length skipped).
- Always runs to max_turns (disable_early_stop=True).
- Records virtual stop points for ablation analysis.

All helper functions are imported from the existing conversation module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import SingleModelClient
from ..knowledge.update_v2 import (
    compute_overall_density_from_state,
    compute_gap_sum_from_state,
    compute_overload_threshold,
    get_absorption_preset,
    update_dynamic_knowledge_state,
)
from ..runtime.conversation import (
    _safe_format,
    _format_knowledge_state,
    _normalize_state,
    _state_score,
    _get_concepts_by_state,
    _get_unaware_concepts,
    _get_askable_concepts,
    _get_last_assistant_message,
    _get_articulation_mode,
    _get_articulation_guidance,
    _get_density_label,
    _get_density_percent,
    _get_misguided_attempt_hint,
    _get_stop_hint,
    _check_termination,
    _compute_structured_metrics,
    _parse_user_response,
    _message_for_tutor_from_user_raw,
    _require_tutor_visible_user_message,
    _extract_attempt_verify_marker,
    _extract_terminate_marker,
    _is_suspicious_user_output,
    _retry_empty_user_queries,
    _get_assistant_history_label,
)


async def run_benchmarking_conversations(
    *,
    problems: List[str],
    problem_ids: List[str],
    user_profiles: List[str],
    user_model_client: SingleModelClient,
    assistant_model_client: SingleModelClient,
    prompt_initial_query_template: str,
    prompt_template: str,
    concept_graph: Dict,
    iu_graphs: Dict,
    id_maps: Dict,
    knowledge_states: List[Optional[Dict]],
    knowledge_levels: List[str],
    task: str,
    knowledge_level_instructions: str,
    reference_answers: List[str],
    absorption_mode: str = "default",
    max_turns: int = 15,
    user_temperature: float = 0.7,
    assistant_temperature: float = 0.0,
    max_tokens: int = 3000,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Run benchmarking conversations (passthrough — no assistant system prompt).

    Follows the same KS-aware loop as the structured simulator but without
    injecting any pedagogical system prompt. The assistant model responds
    with its default behavior.
    """
    absorption_preset = get_absorption_preset(absorption_mode)

    # ── Initialize per-conversation data ──
    conversations_data: List[Dict[str, Any]] = []
    for i, problem in enumerate(problems):
        ref_ans = reference_answers[i] if i < len(reference_answers) else ""
        knowledge_state = knowledge_states[i] if knowledge_states else None
        knowledge_level = knowledge_levels[i] if knowledge_levels else "intermediate"
        problem_id = problem_ids[i] if problem_ids else None

        articulation_mode = "Vague"
        overall_density = 0.5
        gap_sum = 0.0
        query_tone = "Vague"
        initial_overload_threshold = 5

        if knowledge_state is not None and iu_graphs is not None and problem_id:
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
                _n_nodes = len(iu_graph.get("nodes", []))
                initial_overload_threshold = compute_overload_threshold(
                    _n_nodes, knowledge_level, absorption_mode,
                )

        data: Dict[str, Any] = {
            "problem": problem,
            "reference_answer": ref_ans,
            "problem_id": problem_id,
            "user_profile": user_profiles[i],
            "length_control": None,
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
            "assistant_messages": [],  # NO system prompt — starts empty
            "first_query": True,
            "turns": 0,
            "finished": False,
            "over_max": False,
            "stop_reason": None,
            "stop_details": {},
            "run_type": "benchmarking",
        }
        if knowledge_state:
            data["initial_knowledge_state"] = json.loads(json.dumps(knowledge_state))
        # Passthrough: no system prompt appended to assistant_messages
        conversations_data.append(data)

    assistant_label = _get_assistant_history_label(task)

    # ── Turn loop ──
    for turn in range(max_turns):
        # Build user prompts
        user_full_contexts: List[List[Dict[str, str]]] = []
        active_conversations: List[Dict[str, Any]] = []

        for data in conversations_data:
            if data["finished"] or data["over_max"]:
                continue

            if data["first_query"]:
                ks = data.get("knowledge_state")
                user_message_content = _safe_format(
                    prompt_initial_query_template,
                    user_profile=data["user_profile"],
                    message_style=data["user_profile"],
                    math_problem=data["problem"],
                    conversation_history=data["conversation_history"].strip(),
                    length_control=data.get("length_control") or "",
                    knowledge_state_formatted=_format_knowledge_state(ks),
                    knows_well_concepts=_get_concepts_by_state(ks, "knows_well"),
                    partial_understanding_concepts=_get_concepts_by_state(ks, "partial_understanding"),
                    struggling_concepts=_get_concepts_by_state(ks, "struggling"),
                    unaware_concepts=_get_unaware_concepts(ks),
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

        # ── Generate user queries ──
        try:
            user_queries = await user_model_client.generate_responses(
                user_full_contexts,
                temperature=user_temperature,
                max_tokens=max_tokens,
                show_progress=show_progress,
                json_mode=True,
            )
        except Exception as exc:
            _logger = logging.getLogger("sim_error")
            _logger.error("user generate_responses batch failed (benchmarking): %s\n%s", exc, traceback.format_exc())
            for data in active_conversations:
                data["finished"] = True
                data["stop_reason"] = "api_error"
                data["stop_details"] = {"phase": "user_generation", "error": str(exc), "turn": data.get("turns", 0)}
            break

        user_queries = await _retry_empty_user_queries(
            user_model_client, user_full_contexts, user_queries,
            temperature=user_temperature, max_tokens=max_tokens,
            show_progress=show_progress, json_mode=True,
        )

        # Suspicious output check
        for _idx, _q in enumerate(user_queries):
            _raw = _q[0] if _q else ""
            _why = _is_suspicious_user_output(_raw)
            if _why:
                logging.getLogger("sim_error").warning(
                    "Suspicious user-simulator output (benchmarking) conv #%d: %s", _idx, _why
                )

        # ── Parse user responses ──
        for data, user_query in zip(active_conversations, user_queries):
            user_query_text = user_query[0] if user_query else ""
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
                context="User simulator (benchmarking)",
            )
            if not (query or "").strip():
                data["stop_reason"] = "empty_user_message"
                data["stop_details"] = {"user_message": user_query_text, "parse_error": "thought_only_or_empty"}
                data["finished"] = True
                continue
            data["conversation"].append(("user", query))

            # Always disable_early_stop — record virtual stop points
            if terminate_flag:
                if data.get("virtual_early_stop_turn") is None:
                    data["virtual_early_stop_turn"] = data["turns"]
                    data["virtual_early_stop_reason"] = "user_requested_terminate"

            data["assistant_messages"].append({"role": "user", "content": query})
            # No system prompt → first user message is at index 0
            if data.get("first_query_content") is None and len(data["assistant_messages"]) == 1:
                data["first_query_content"] = query

        active_conversations = [data for data in active_conversations if not data["finished"]]
        if not active_conversations:
            break

        # ── Generate assistant responses (no system prompt, no length enforcement) ──
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
            _logger.error("assistant generate_responses batch failed (benchmarking): %s\n%s", exc, traceback.format_exc())
            for data in active_conversations:
                data["finished"] = True
                data["stop_reason"] = "api_error"
                data["stop_details"] = {"phase": "assistant_generation", "error": str(exc), "turn": data.get("turns", 0)}
            break

        # No _enforce_assistant_length — passthrough mode lets models respond naturally

        # ── Phase 1: sync bookkeeping ──
        post_turn_info: List[Tuple[Dict[str, Any], str, str]] = []
        for data, assistant_response in zip(active_conversations, assistant_responses):
            assistant_text = assistant_response[0] if assistant_response else ""
            data["conversation"].append(("assistant", assistant_text))
            # With no system prompt, first user message is at index 0
            last_user_message = data["assistant_messages"][-1]["content"]
            if data.get("first_query_content") and data["turns"] == 0:
                last_user_message = data["first_query_content"]
            data["conversation_history"] += f"- You: {last_user_message}\n- {assistant_label}: {assistant_text}\n"
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

        # ── Phase 2: collect KS-update coroutines ──
        ks_coros: List[Optional[Any]] = []
        for data, last_user_message, assistant_text in post_turn_info:
            if data.get("knowledge_state") is not None and data.get("problem_id"):
                problem_id = str(data["problem_id"])
                iu_graph = iu_graphs.get(problem_id) if iu_graphs else None
                id_map = id_maps.get(problem_id) if id_maps else None
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
                    temporal_decay=False,
                    max_turns=max_turns,
                ))
            else:
                ks_coros.append(None)

        # ── Phase 3: fan out Phase B LLM calls ──
        runnable_pairs = [(i, c) for i, c in enumerate(ks_coros) if c is not None]
        ks_results: List[Any] = [None] * len(ks_coros)
        if runnable_pairs:
            gathered = await asyncio.gather(
                *[c for _, c in runnable_pairs], return_exceptions=True
            )
            for (i, _), result in zip(runnable_pairs, gathered):
                ks_results[i] = result

        # ── Phase 4: apply KS results + virtual stop checks ──
        for (data, last_user_message, assistant_text), update_result in zip(post_turn_info, ks_results):
            if update_result is None:
                continue
            if isinstance(update_result, BaseException):
                exc = update_result
                _logger = logging.getLogger("sim_error")
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                _logger.error(
                    "KS update failed (benchmarking) | problem_id=%s turn=%d | %s\n%s",
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
                po_turn = float(metrics.get("perceived_overload_turn", 0.0))
                absorbed_gain = float(metrics.get("absorbed_gain", 0.0))

                data["turn_metrics_history"].append({
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
                    "absorbed_gain": absorbed_gain,
                    "comprehension_load": float(metrics.get("comprehension_load", 0.0)),
                    "attempt_load": float(metrics.get("attempt_load", 0.0)),
                    "effective_load": float(metrics.get("effective_load", 0.0)),
                })
                data["ic_numerator_total"] += ic_num_turn
                data["ic_denominator_total"] += ic_den_turn
                data["new_ius_total"] += new_ius_turn
                data["disallowed_total"] += disallowed_turn
                data["zpd_size_total"] += zpd_size_turn
                data["absorbed_gain_total"] += absorbed_gain
                data["attempt_verify"] = False

                newly_explained = metrics.get("explained_concepts", [])
                merged_explained = set(data.get("explained_concepts", []))
                merged_explained.update(newly_explained)
                data["explained_concepts"] = sorted(merged_explained)

                # Virtual mastery check
                if updated_state and all(
                    _normalize_state((info.get("state", "") if isinstance(info, dict) else info)) == "knows_well"
                    for info in updated_state.values()
                ):
                    if data.get("virtual_mastery_turn") is None:
                        data["virtual_mastery_turn"] = data["turns"]
                        mastery_0idx = data["turns"] - 1
                        if data.get("virtual_early_stop_turn") is None or mastery_0idx < data["virtual_early_stop_turn"]:
                            data["virtual_early_stop_turn"] = mastery_0idx
                            data["virtual_early_stop_reason"] = "mastery"

                # Virtual stagnation / overload-persistence check
                termination = _check_termination(data, min_warmup=3)
                if termination:
                    stop_reason, details = termination
                    if data.get("virtual_early_stop_turn") is None:
                        data["virtual_early_stop_turn"] = data["turns"] - 1
                        data["virtual_early_stop_reason"] = stop_reason

            except Exception as exc:
                _logger = logging.getLogger("sim_error")
                tb = traceback.format_exc()
                _logger.error(
                    "KS update processing failed (benchmarking) | problem_id=%s turn=%d | %s\n%s",
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

    # ── Compute final metrics ──
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
