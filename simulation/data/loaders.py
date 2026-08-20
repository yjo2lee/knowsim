"""Dataset loaders for annotations and profiles."""

from __future__ import annotations

import json
import csv
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotations(path: str) -> List[Dict[str, Any]]:
    return load_json(path)


def load_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def load_dataset_rows(path: str) -> List[Dict[str, Any]]:
    """Load dataset rows, dispatching on file extension.

    JSON Lines preserves the item text exactly as written; CSV is still
    accepted so existing inputs keep working.
    """
    return load_jsonl_rows(path) if str(path).endswith(".jsonl") else load_csv_rows(path)


def row_problem_id(row: Dict[str, Any], index: int) -> str:
    """The row's own ``id`` when it carries one, else its position in the file.

    Bundled item sets ship a stable public id (``math_001``), which is what the
    IU-graph and initial-knowledge-state caches are keyed by. Inputs without an
    ``id`` field fall back to the row index, the identifier those inputs have
    always used.
    """
    rid = row.get("id")
    return str(rid) if rid not in (None, "") else str(index)
