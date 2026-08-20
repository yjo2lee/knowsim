"""Fetch Hendrycks MATH (train split, levels 4 & 5) and write the CSV the
simulator expects at data/competition_math/train_fixed_level_4_5.csv.

Source: HuggingFace dataset `qwedsacf/competition_math`, which mirrors the
Hendrycks et al. 2021 "Measuring Mathematical Problem Solving" dataset with
the same schema (`problem`, `level`, `type`, `solution`). The official
`hendrycks/competition_math` HF repo was pulled by the authors.

On the "_fixed" suffix in the expected filename:
  The repo inherited the filename `train_fixed_level_4_5.csv` but no
  preprocessing script ships with it. The pipeline consumes `problem` and
  `solution` as raw strings (sent to LLMs; no boxed-answer extraction, no
  LaTeX normalization — see simulation/runtime/conversation.py and
  simulation/knowledge/update_v2.py). That leaves only a small menu of
  plausible "fixes": (1) filter to levels 4-5, (2) drop rows with empty
  problem or solution, (3) dedupe, (4) strip Asymptote [asy] diagram
  blocks, (5) LaTeX cleanup. We apply (1) and (2) here — the minimum that
  guarantees every row is a valid LLM input. We skip (3)-(5): the raw
  Hendrycks corpus has very few near-duplicates in train, Asymptote blocks
  are legitimate problem content that modern LLMs can reason about, and
  LaTeX normalization is risky without a known target format.

Usage:
    python scripts/fetch_math.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from datasets import load_dataset


HF_NAME = "qwedsacf/competition_math"
SPLIT = "train"
KEEP_LEVELS = {"Level 4", "Level 5"}
OUT_PATH = Path("data/competition_math/train_fixed_level_4_5.csv")
COLUMNS = ["problem", "level", "type", "solution"]


def main() -> None:
    print(f"Loading {HF_NAME} split={SPLIT} from HuggingFace...")
    ds = load_dataset(HF_NAME, split=SPLIT)
    total = len(ds)

    rows_kept = []
    dropped_level = 0
    dropped_empty = 0
    for ex in ds:
        level = (ex.get("level") or "").strip()
        if level not in KEEP_LEVELS:
            dropped_level += 1
            continue
        problem = (ex.get("problem") or "").strip()
        solution = (ex.get("solution") or "").strip()
        if not problem or not solution:
            dropped_empty += 1
            continue
        rows_kept.append({
            "problem": problem,
            "level": level,
            "type": (ex.get("type") or "").strip(),
            "solution": solution,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows_kept)

    print(f"\nTotal rows in source split : {total}")
    print(f"Dropped (not level 4 or 5) : {dropped_level}")
    print(f"Dropped (empty problem/sol): {dropped_empty}")
    print(f"Written to {OUT_PATH}       : {len(rows_kept)}")

    # Sanity preview
    if rows_kept:
        sample = rows_kept[0]
        print(f"\nSample row:")
        print(f"  level={sample['level']!r}  type={sample['type']!r}")
        print(f"  problem={sample['problem'][:100]!r}...")
        print(f"  solution={sample['solution'][:100]!r}...")


if __name__ == "__main__":
    main()
