"""Benchmarking analysis — produces Tables 1-3 from assistant_benchmarking.md.

Usage:
    python -m simulation.benchmarking.analysis \
        --input_dirs output_benchmarking/competition_math output_benchmarking/expertqa \
        --output_dir output_benchmarking/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)


METRIC_COLS = {
    "NKG": ("normalized_knowledge_gain", True),   # higher is better
    "IC":  ("ic_f1", True),
    "PO":  ("perceived_overload", False),          # lower is better
    "ENG": ("engagement", True),
    "LE":  ("learning_experience", True),
    "IQ":  ("interaction_quality_score", True),
}


def _extract_records(input_dirs: List[str]) -> List[Dict[str, Any]]:
    """Walk output JSONs and extract per-conversation metric records."""
    records: List[Dict[str, Any]] = []
    for input_dir in input_dirs:
        root = Path(input_dir)
        if not root.exists():
            print(f"WARNING: {root} does not exist, skipping.")
            continue
        for json_path in sorted(root.rglob("*.json")):
            # Skip non-conversation files
            if any(skip in json_path.parts for skip in ("metrics", "dashboard", "iu_cache", "condition_logs")):
                continue
            if "manifest" in json_path.name:
                continue
            try:
                with open(json_path, encoding="utf-8") as f:
                    conversations = json.load(f)
            except (json.JSONDecodeError, IsADirectoryError):
                continue
            if not isinstance(conversations, list):
                continue

            # Infer model name from directory structure: .../<task>/<model>/<run_id>/file.json
            parts = json_path.parts
            model_name = None
            for i, part in enumerate(parts):
                if part in ("competition_math", "expertqa") and i + 1 < len(parts):
                    model_name = parts[i + 1]
                    break

            # Infer task from directory
            task_label = "math"
            if "expertqa" in str(json_path):
                task_label = "expertqa"

            for conv in conversations:
                if not isinstance(conv, dict):
                    continue
                metrics = conv.get("metrics") or {}
                # Skip errored conversations
                if conv.get("stop_reason") in ("error", "api_error", "empty_user_message", "empty_assistant"):
                    continue

                record = {
                    "model": conv.get("assistant_model") or model_name or "unknown",
                    "knowledge_level": conv.get("knowledge_level", "unknown"),
                    "seed": conv.get("seed"),
                    "problem_id": conv.get("problem_id"),
                    "task": conv.get("task") or task_label,
                    "NKG": metrics.get("normalized_knowledge_gain"),
                    "IC": metrics.get("ic_f1"),
                    "PO": metrics.get("perceived_overload"),
                    "ENG": metrics.get("engagement"),
                    "LE": metrics.get("learning_experience"),
                    "IQ": conv.get("interaction_quality_score"),
                }
                records.append(record)
    return records


def _rank_column(series: pd.Series, ascending: bool) -> pd.Series:
    """Rank a series (higher rank = 1 for better). For PO, ascending=True means lower is better."""
    return series.rank(ascending=ascending, method="min")


def _build_ranked_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Build a metric table with per-column ranks and average rank."""
    metric_names = list(METRIC_COLS.keys())
    agg = df.groupby(group_cols)[metric_names].mean()

    # Compute ranks
    rank_df = pd.DataFrame(index=agg.index)
    for metric, (_, higher_is_better) in METRIC_COLS.items():
        if metric not in agg.columns:
            continue
        rank_df[f"{metric}_rank"] = _rank_column(agg[metric], ascending=not higher_is_better)

    rank_cols = [c for c in rank_df.columns if c.endswith("_rank")]
    agg["Avg_Rank"] = rank_df[rank_cols].mean(axis=1)

    # Merge values + ranks
    result = agg.copy()
    for metric in metric_names:
        if f"{metric}_rank" in rank_df.columns:
            result[f"{metric}_rank"] = rank_df[f"{metric}_rank"]

    return result.sort_values("Avg_Rank")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmarking analysis — Tables 1-3.")
    parser.add_argument("--input_dirs", nargs="+", required=True,
                        help="One or more directories containing benchmarking output JSONs.")
    parser.add_argument("--output_dir", type=str, default="output_benchmarking/tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting records...")
    records = _extract_records(args.input_dirs)
    if not records:
        print("ERROR: no valid conversation records found.")
        sys.exit(1)

    df = pd.DataFrame(records)
    print(f"  {len(df)} conversations from {df['model'].nunique()} models")

    # Drop rows with missing key metrics
    for metric in ["NKG", "IC", "PO", "ENG", "LE"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["IQ"] = pd.to_numeric(df["IQ"], errors="coerce")

    # ── Table 1: Model × Metric, level-marginalized ──
    print("\nTable 1: Level-marginalized model ranking")
    table1 = _build_ranked_table(df, ["model"])
    table1_path = output_dir / "table1_level_marginalized.csv"
    table1.to_csv(table1_path)
    print(table1[["NKG", "IC", "PO", "ENG", "LE", "IQ", "Avg_Rank"]].to_string())
    print(f"  Saved: {table1_path}")

    # ── Table 2: Model × Metric, per knowledge level ──
    print("\nTable 2: Per-level model ranking")
    for level in ["novice", "intermediate", "advanced"]:
        subset = df[df["knowledge_level"] == level]
        if subset.empty:
            print(f"  [{level}] No data, skipping.")
            continue
        table2 = _build_ranked_table(subset, ["model"])
        table2_path = output_dir / f"table2_{level}.csv"
        table2.to_csv(table2_path)
        print(f"\n  [{level}] ({len(subset)} conversations)")
        print(table2[["NKG", "IC", "PO", "ENG", "LE", "IQ", "Avg_Rank"]].to_string())
        print(f"  Saved: {table2_path}")

    # ── Table 3: Dataset split (MATH vs ExpertQA) ──
    if df["task"].nunique() > 1:
        print("\nTable 3: Dataset split (LE, IQ only)")
        table3 = df.groupby(["model", "task"])[["LE", "IQ"]].mean()
        table3_path = output_dir / "table3_dataset_split.csv"
        table3.to_csv(table3_path)
        print(table3.to_string())
        print(f"  Saved: {table3_path}")
    else:
        print(f"\nTable 3: Skipped (only one task: {df['task'].iloc[0]})")

    print(f"\nAll tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
