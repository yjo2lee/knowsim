"""Pre-extract IU graphs for a problem set and save to a cache file.

Run this once before an experiment to produce a canonical IU decomposition
shared across all strategy/level conditions, eliminating the IU-count confound
in cross-strategy comparisons.

Usage
-----
    python -m simulation.tools.extract_iu_graphs \\
        --input_csv  Data/competition_math/data/train_fixed_level_4_5.csv \\
        --output     output/iu_cache/competition_math_gpt4o_mini.json \\
        --iu_model   gpt-5.2 \\
        --num_problems 15

    # ExpertQA
    python -m simulation.tools.extract_iu_graphs \\
        --expertqa_jsonl Data/expertqa/r2_compiled.jsonl \\
        --output output/iu_cache/expertqa_gpt4o_mini.json \\
        --iu_model gpt-5.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Allow running as __main__ from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simulation.core.models import SingleModelClient
from simulation.data.expertqa_answer import select_expertqa_answer
from simulation.data.loaders import load_dataset_rows, load_jsonl_rows, row_problem_id
from simulation.knowledge.iu_extraction import extract_iu_graph


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-extract and cache IU graphs for a problem set."
    )
    # Input
    parser.add_argument("--input_csv", type=str, default="",
                        help="CSV file with 'problem' and 'solution' columns (math task).")
    parser.add_argument("--expertqa_jsonl", type=str, default="",
                        help="JSONL file for ExpertQA task.")
    parser.add_argument("--expertqa_answer_variant", type=str, default="")
    parser.add_argument("--num_problems", type=int, default=-1,
                        help="Number of problems to process (-1 = all remaining after start_index).")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Skip this many rows from the start (0-based). Items without their own "
                             "id are keyed by start_index + row offset.")
    parser.add_argument("--indexes_file", type=str, default="",
                        help="Path to a text file with one 0-based row index per line. "
                             "When set, --start_index and --num_problems are ignored "
                             "and only the listed rows are extracted. "
                             "Each item is keyed by its own id when it has one, else by row index.")
    # Model
    parser.add_argument("--iu_model", type=str, default="gpt-5.2")
    parser.add_argument("--iu_max_tokens", type=int, default=0,
                        help="Max completion tokens (0 = auto).")
    parser.add_argument("--llm_provider", type=str, default="openai",
                        choices=["openai", "gemini"])
    parser.add_argument("--prompts_root", type=str, default="simulation/prompts")
    # Output
    parser.add_argument("--output", type=str, default="",
                        help="Path to write the cache JSON. "
                             "Defaults to output/iu_cache/<task>_<model>_<ts>.json")
    # Incremental
    parser.add_argument("--existing_cache", type=str, default="",
                        help="Path to an existing cache to extend (skips already-extracted problems).")
    args = parser.parse_args()

    if not args.input_csv and not args.expertqa_jsonl:
        parser.error("Provide --input_csv or --expertqa_jsonl.")

    # ── Load problems ───────────────────────────────────────────────────────
    annotations: List[Dict] = []
    task = "math"

    si = max(0, args.start_index)

    # Load explicit row indexes when --indexes_file is set; otherwise fall
    # back to the contiguous start_index/num_problems slice.
    selected_indexes: List[int] = []
    if args.indexes_file:
        with open(args.indexes_file, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                selected_indexes.append(int(ln))
        # Dedupe + sort for determinism while preserving the user's set.
        selected_indexes = sorted(set(selected_indexes))
        if not selected_indexes:
            parser.error(f"--indexes_file {args.indexes_file} is empty.")

    if args.expertqa_jsonl:
        task = "expertqa"
        all_rows = load_jsonl_rows(args.expertqa_jsonl)
        if selected_indexes:
            picked = [(i, all_rows[i]) for i in selected_indexes if 0 <= i < len(all_rows)]
        else:
            rows = all_rows[si:] if si else all_rows
            if args.num_problems > 0:
                rows = rows[: args.num_problems]
            picked = [(si + idx, row) for idx, row in enumerate(rows)]
        for actual_idx, row in picked:
            answer_text, _variant, _source = select_expertqa_answer(
                row.get("answers", {}) or {},
                args.expertqa_answer_variant,
            )
            _meta = row.get("metadata") or {}
            annotations.append({
                "problem_id": row_problem_id(row, actual_idx),
                "question": row.get("question", ""),
                "solution": answer_text,
                "field": str(_meta.get("field", "")),
                "specific_field": str(_meta.get("specific_field", "")),
            })
    else:
        all_rows = load_dataset_rows(args.input_csv)
        if selected_indexes:
            picked = [(i, all_rows[i]) for i in selected_indexes if 0 <= i < len(all_rows)]
        else:
            rows = all_rows[si:] if si else all_rows
            if args.num_problems > 0:
                rows = rows[: args.num_problems]
            picked = [(si + idx, row) for idx, row in enumerate(rows)]
        for actual_idx, row in picked:
            annotations.append({
                "problem_id": row_problem_id(row, actual_idx),
                "question": row.get("problem", ""),
                "solution": row.get("solution", ""),
            })

    print(f"Loaded {len(annotations)} problem(s) for IU extraction.")

    # ── Load existing cache (incremental mode) ──────────────────────────────
    iu_graphs: Dict[str, Dict] = {}
    if args.existing_cache and os.path.exists(args.existing_cache):
        with open(args.existing_cache, encoding="utf-8") as f:
            iu_graphs = json.load(f)
        print(f"Loaded existing cache: {len(iu_graphs)} problem(s) already extracted.")

    to_extract = [ann for ann in annotations if str(ann["problem_id"]) not in iu_graphs]
    if not to_extract:
        print("All problems already in cache — nothing to do.")
    else:
        print(f"Extracting IU graphs for {len(to_extract)} problem(s)...")

    # ── Extract ─────────────────────────────────────────────────────────────
    if to_extract:
        model_client = SingleModelClient(args.iu_model, provider=args.llm_provider)

        if args.iu_max_tokens > 0:
            iu_max_tokens = args.iu_max_tokens
        elif args.llm_provider == "gemini":
            iu_max_tokens = 6000
        else:
            iu_max_tokens = 16384

        _iu_prompt_file = (
            "iu_graph_prompt_expertqa.md" if task == "expertqa"
            else "iu_graph_prompt.md"
        )
        prompt_path = os.path.join(args.prompts_root, "knowledge_state", _iu_prompt_file)

        # Fan out IU extraction across all problems in parallel.
        # The model client's internal aiolimiter caps the actual concurrency.
        async def _extract_one(ann: Dict) -> tuple[str, Dict]:
            pid = str(ann["problem_id"])
            iu_graph = await extract_iu_graph(
                question=ann["question"],
                answer=ann["solution"],
                model_client=model_client,
                field=ann.get("field", ""),
                specific_field=ann.get("specific_field", ""),
                max_tokens=iu_max_tokens,
                show_progress=False,
                prompt_path=prompt_path,
            )
            return pid, iu_graph

        print(f"  Extracting {len(to_extract)} graph(s) concurrently...")
        results = await asyncio.gather(
            *[_extract_one(ann) for ann in to_extract],
            return_exceptions=True,
        )
        for ann, result in zip(to_extract, results):
            pid = str(ann["problem_id"])
            if isinstance(result, Exception):
                print(f"  [{pid}] FAILED: {result}")
                continue
            _, iu_graph = result
            iu_graphs[pid] = iu_graph
            n_nodes = len(iu_graph.get("nodes", []))
            n_edges = len(iu_graph.get("edges", []))
            print(f"  [{pid}] {n_nodes} nodes, {n_edges} edges")

    # ── Save ────────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_slug = args.iu_model.replace("/", "-")
        out_path = Path("output") / "iu_cache" / f"{task}_{model_slug}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(iu_graphs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(iu_graphs)} IU graph(s) to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
