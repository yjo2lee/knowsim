"""IU graph helpers and conversion to concept graph."""

from __future__ import annotations

from typing import Dict, List, Tuple


def build_concept_graph_from_iu(iu_graphs: Dict[str, Dict]) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, str]]]:
    """
    Convert IU graphs into the concept_graph structure used by the tutor pipeline.
    concept_id is a readable label that includes the IU id and concept name.
    """
    concept_graph: Dict[str, List[Dict]] = {}
    id_maps: Dict[str, Dict[str, str]] = {}
    for problem_id, graph in iu_graphs.items():
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        id_to_label: Dict[str, str] = {}
        for node in nodes:
            iu_id = node.get("id", "")
            concept = node.get("concept", "")
            label = f"{iu_id}: {concept}" if iu_id and concept else iu_id or concept
            id_to_label[iu_id] = label
        id_maps[problem_id] = id_to_label
        prereq_map: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
        for e in edges:
            src = e.get("from")
            dst = e.get("to")
            if src in prereq_map and dst in prereq_map:
                prereq_map[dst].append(src)
        items = []
        for node in nodes:
            iu_id = node.get("id", "")
            items.append(
                {
                    "concept_id": id_to_label.get(iu_id, iu_id),
                    "description": node.get("description", ""),
                    "prerequisites": [id_to_label.get(pid, pid) for pid in prereq_map.get(iu_id, [])],
                }
            )
        concept_graph[problem_id] = items
    return concept_graph, id_maps

