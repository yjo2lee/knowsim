"""Aggregate simulator metrics from output JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


METRIC_KEYS = [
    "knowledge_gain",
    "normalized_knowledge_gain",
    "information_calibration",
    "perceived_overload",
    "overload_inverse",
    "engagement",
    "learning_experience",
    "behavioral_realism_score",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Granularity variant tags embedded in filenames (e.g. dynamic-knowledge-state_coarse_…).
_GRANULARITY_TAGS = {"_coarse_", "_medium_", "_fine_"}


def _iter_result_files(
    input_dir: Path,
    *,
    include_granularity: bool = False,
) -> Iterable[Path]:
    skip_parts = {"metrics", "experiment_comparison", "dashboard"}
    for path in sorted(input_dir.rglob("*.json")):
        if any(p in path.parts for p in skip_parts):
            continue
        if ".bak." in path.name:
            continue
        if not include_granularity and any(tag in path.name for tag in _GRANULARITY_TAGS):
            continue
        yield path


def _row_from_item(item: Dict[str, Any], source_file: Path, idx: int) -> Dict[str, Any]:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
    behavioral = item.get("behavioral_realism", {}) if isinstance(item.get("behavioral_realism"), dict) else {}
    row: Dict[str, Any] = {
        "source_file": str(source_file),
        "conversation_index": idx,
        "task": item.get("task", ""),
        "simulation_method": item.get("simulation_method", ""),
        "run_type": item.get("run_type", ""),
        "assistant_strategy": item.get("assistant_strategy", ""),
        "knowledge_level": item.get("knowledge_level", ""),
        "absorption_mode": str(item.get("absorption_mode") or "default"),
        "problem_id": item.get("problem_id", ""),
    }
    for key in METRIC_KEYS:
        val = _safe_float(metrics.get(key))
        if val is None and key == "behavioral_realism_score":
            val = _safe_float(behavioral.get("behavioral_realism_score"))
        row[key] = val
    return row


def _group_key(row: Dict[str, Any], group_by: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(str(row.get(col, "")) for col in group_by)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(rows: List[Dict[str, Any]], group_by: Tuple[str, ...]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, group_by)].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        summary: Dict[str, Any] = {col: key[idx] for idx, col in enumerate(group_by)}
        summary["n"] = len(items)
        for metric in METRIC_KEYS:
            values = [float(v) for v in (item.get(metric) for item in items) if v is not None]
            summary[f"{metric}_mean"] = (sum(values) / len(values)) if values else None
        summary_rows.append(summary)
    return summary_rows


def _to_html_table(rows: List[Dict[str, Any]], title: str) -> str:
    if not rows:
        return f"<h2>{title}</h2><p>No data</p>"
    headers = list(rows[0].keys())
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = []
        for h in headers:
            val = row.get(h)
            if isinstance(val, float):
                tds.append(f"<td>{val:.4f}</td>")
            else:
                tds.append(f"<td>{'' if val is None else val}</td>")
        trs.append(f"<tr>{''.join(tds)}</tr>")
    return (
        f"<h2>{title}</h2>"
        "<table border='1' cellspacing='0' cellpadding='4' style='border-collapse:collapse;'>"
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    )


def aggregate_from_dir(input_dir: Path, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for path in _iter_result_files(input_dir):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                rows.append(_row_from_item(item, path, idx))

    conversation_csv = output_dir / "conversation_metrics.csv"
    summary_csv = output_dir / "summary_metrics.csv"
    summary_by_strategy_csv = output_dir / "summary_metrics_by_strategy.csv"
    summary_by_level_strategy_csv = output_dir / "summary_metrics_by_level_strategy.csv"
    html_report = output_dir / "metrics_report.html"

    conversation_fields = [
        "source_file",
        "conversation_index",
        "task",
        "run_type",
        "simulation_method",
        "assistant_strategy",
        "knowledge_level",
        "absorption_mode",
        "problem_id",
        *METRIC_KEYS,
    ]
    _write_csv(conversation_csv, rows, conversation_fields)

    # absorption_mode is part of every grouping key so cross-mode runs are
    # never blended in the summary CSVs.
    base_cols = ("task", "run_type", "simulation_method", "absorption_mode")
    by_level_cols = (*base_cols, "knowledge_level")
    by_strategy_cols = (*base_cols, "assistant_strategy")
    by_level_strategy_cols = (*base_cols, "knowledge_level", "assistant_strategy")

    summary_rows = _build_summary(rows, by_level_cols)
    summary_by_strategy_rows = _build_summary(rows, by_strategy_cols)
    summary_by_level_strategy_rows = _build_summary(rows, by_level_strategy_cols)

    def _summary_fields(cols: Tuple[str, ...]) -> List[str]:
        return [*cols, "n", *[f"{k}_mean" for k in METRIC_KEYS]]

    _write_csv(summary_csv, summary_rows, _summary_fields(by_level_cols))
    _write_csv(summary_by_strategy_csv, summary_by_strategy_rows, _summary_fields(by_strategy_cols))
    _write_csv(
        summary_by_level_strategy_csv,
        summary_by_level_strategy_rows,
        _summary_fields(by_level_strategy_cols),
    )

    html_content = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>Simulator Metrics Report</title></head><body>"
        "<h1>Simulator Metrics Report</h1>"
        f"<p>Input: {input_dir}</p>"
        f"{_to_html_table(summary_rows, 'Summary by Knowledge Level')}"
        f"{_to_html_table(summary_by_strategy_rows, 'Summary by Assistant Strategy')}"
        f"{_to_html_table(summary_by_level_strategy_rows, 'Summary by Knowledge Level × Strategy')}"
        f"{_to_html_table(rows[:200], 'Conversation-level rows (first 200)')}"
        "</body></html>"
    )
    html_report.write_text(html_content, encoding="utf-8")

    return {
        "conversation_csv": conversation_csv,
        "summary_csv": summary_csv,
        "summary_by_strategy_csv": summary_by_strategy_csv,
        "summary_by_level_strategy_csv": summary_by_level_strategy_csv,
        "html_report": html_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate conversation-level metrics from output JSON files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing output JSON files.")
    parser.add_argument("--output_dir", type=str, default="output/metrics", help="Directory to write reports.")
    args = parser.parse_args()
    outputs = aggregate_from_dir(Path(args.input_dir), Path(args.output_dir))
    print(f"Wrote: {outputs['conversation_csv']}")
    print(f"Wrote: {outputs['summary_csv']}")
    print(f"Wrote: {outputs['summary_by_strategy_csv']}")
    print(f"Wrote: {outputs['summary_by_level_strategy_csv']}")
    print(f"Wrote: {outputs['html_report']}")


if __name__ == "__main__":
    main()
