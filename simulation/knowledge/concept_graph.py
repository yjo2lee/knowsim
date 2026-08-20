"""Concept graph utilities (ported from utils.build_concept_graph_from_extracted)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def format_concept_list_with_prerequisites(concepts: List[Dict[str, Any]]) -> str:
    lines = []
    for concept in concepts:
        concept_id = concept.get("concept_id", "")
        description = concept.get("description", "")
        prerequisites = concept.get("prerequisites", [])
        prerequisites_text = ", ".join(prerequisites) if prerequisites else "none listed"
        lines.append(f"- {concept_id}: {description} (prerequisites: {prerequisites_text})")
    return "\n".join(lines)


async def build_concept_graph_from_extracted(
    extracted_concepts: Dict[str, Dict[str, Any]],
    *,
    model_client: Any,
    max_tokens: int = 800,
    show_progress: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a concept graph from extracted concepts (two-step LLM process).
    """
    def _normalize(name: str) -> str:
        return re.sub(r"\s+", " ", (name or "").strip().lower())

    def _append_prereq(item: Dict[str, Any], prereq_name: str) -> None:
        if not prereq_name:
            return
        if prereq_name not in item["prerequisites"]:
            item["prerequisites"].append(prereq_name)

    def _extract_json_object(text: str) -> Dict[str, Any]:
        if not text:
            return {}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    concept_graph: Dict[str, List[Dict[str, Any]]] = {}
    original_concepts_by_problem: Dict[str, List[Dict[str, Any]]] = {}

    for problem_id, payload in extracted_concepts.items():
        concepts = payload.get("extracted_concepts", [])
        items = []
        for concept in concepts:
            name = concept.get("Concept Name", "").strip()
            if not name:
                continue
            items.append(
                {
                    "concept_id": name,
                    "description": concept.get("Concept Explanation", "").strip(),
                    "prerequisites": [],
                }
            )
        concept_graph[problem_id] = items
        original_concepts_by_problem[problem_id] = list(items)

    # Step 1: Check prerequisite relations among existing concepts
    relation_contexts: List[List[Dict[str, str]]] = []
    relation_problem_ids: List[str] = []
    for problem_id, items in concept_graph.items():
        if len(items) <= 1:
            continue
        concept_list = [
            {"concept_id": item["concept_id"], "description": item["description"]}
            for item in items
        ]
        prompt = (
            "You are given a list of concepts for a single math problem.\n"
            "Identify prerequisite relationships among ONLY these concepts.\n"
            "Return JSON in this format:\n"
            "{\n"
            '  "prerequisites": {\n'
            '    "Concept A": ["Concept B", "Concept C"],\n'
            '    "Concept B": []\n'
            "  }\n"
            "}\n"
            "Rules:\n"
            "- Only include prerequisite concepts that appear in the provided list.\n"
            "- If no prerequisites, use an empty list.\n"
            "- Do not invent new concepts.\n"
            f"Concept list:\n{json.dumps(concept_list, indent=2)}"
        )
        relation_contexts.append([{"role": "user", "content": prompt}])
        relation_problem_ids.append(problem_id)

    if relation_contexts:
        relation_responses = await model_client.generate_responses(
            relation_contexts,
            temperature=0.2,
            max_tokens=max_tokens,
            show_progress=show_progress,
        )
        for problem_id, response in zip(relation_problem_ids, relation_responses):
            parsed = _extract_json_object(response[0] if response else "")
            prereq_map = parsed.get("prerequisites", {}) if isinstance(parsed, dict) else {}
            if not isinstance(prereq_map, dict):
                continue
            item_by_norm = {
                _normalize(item["concept_id"]): item
                for item in concept_graph.get(problem_id, [])
            }
            for concept_name, prereqs in prereq_map.items():
                concept_item = item_by_norm.get(_normalize(concept_name))
                if not concept_item or not isinstance(prereqs, list):
                    continue
                for prereq in prereqs:
                    prereq_name = str(prereq).strip()
                    if not prereq_name:
                        continue
                    if _normalize(prereq_name) in item_by_norm and _normalize(prereq_name) != _normalize(concept_name):
                        _append_prereq(concept_item, prereq_name)

    # Step 2: Generate 1-3 prerequisite concepts for each original concept
    gen_contexts: List[List[Dict[str, str]]] = []
    gen_meta: List[Dict[str, str]] = []
    for problem_id, items in original_concepts_by_problem.items():
        existing_names = [item["concept_id"] for item in concept_graph.get(problem_id, [])]
        for item in items:
            prompt = (
                "You are given a target concept from a math problem.\n"
                "Generate 1-3 prerequisite concepts that are more foundational.\n"
                "Return JSON in this format:\n"
                "{\n"
                '  "prerequisites": [\n'
                '    {"concept_id": "Prereq 1", "description": "Short description"},\n'
                '    {"concept_id": "Prereq 2", "description": "Short description"}\n'
                "  ]\n"
                "}\n"
                "Rules:\n"
                "- Do not include the target concept itself.\n"
                "- Keep descriptions short and factual.\n"
                "- Avoid duplicates with the existing concept list unless truly necessary.\n"
                f"Target concept: {item['concept_id']}\n"
                f"Target description: {item['description']}\n"
                f"Existing concepts: {json.dumps(existing_names, ensure_ascii=False)}"
            )
            gen_contexts.append([{"role": "user", "content": prompt}])
            gen_meta.append({"problem_id": problem_id, "concept_id": item["concept_id"]})

    if gen_contexts:
        gen_responses = await model_client.generate_responses(
            gen_contexts,
            temperature=0.6,
            max_tokens=max_tokens,
            show_progress=show_progress,
        )
        for meta, response in zip(gen_meta, gen_responses):
            problem_id = meta["problem_id"]
            target_name = meta["concept_id"]
            parsed = _extract_json_object(response[0] if response else "")
            prereq_items = parsed.get("prerequisites", []) if isinstance(parsed, dict) else []
            if not isinstance(prereq_items, list):
                continue
            concept_items = concept_graph.get(problem_id, [])
            item_by_norm = {_normalize(item["concept_id"]): item for item in concept_items}
            target_item = item_by_norm.get(_normalize(target_name))
            if not target_item:
                continue
            for prereq in prereq_items:
                if not isinstance(prereq, dict):
                    continue
                prereq_name = str(prereq.get("concept_id", "")).strip()
                if not prereq_name or _normalize(prereq_name) == _normalize(target_name):
                    continue
                prereq_desc = str(prereq.get("description", "")).strip()
                existing_item = item_by_norm.get(_normalize(prereq_name))
                if existing_item is None:
                    new_item = {
                        "concept_id": prereq_name,
                        "description": prereq_desc,
                        "prerequisites": [],
                    }
                    concept_items.append(new_item)
                    item_by_norm[_normalize(prereq_name)] = new_item
                    existing_item = new_item
                _append_prereq(target_item, existing_item["concept_id"])

    return concept_graph

