"""Post-hoc termination judge for baseline conversations.

Replicates SimulatorArena's post-hoc LLM judge approach: reads completed baseline
conversation JSONs, passes the numbered user queries to an LLM judge, and identifies
the natural ending turn for each conversation.

Criteria for math (SimulatorArena math tutoring):
  - problem_completion : student has no more relevant questions about the original problem
  - problem_shift      : student begins asking about a different math problem
  - circular_queries   : student repeats similar responses without showing progress

Criteria for expertqa (adapted from SimulatorArena document-creation):
  - comprehension_satisfied : user has no remaining questions; needs are addressed
  - unproductive_stalling   : circular discussion or assistant fails to advance understanding

Results are written to each conversation as:
  judged_end_turn        : int  — natural end turn identified by the judge
  judged_termination_reason : str — one of the criteria labels above, or "max_turns"

Usage:
    python -m simulation.tools.baseline_termination_judge \\
        --input_dir output/expertqa/gpt-4.1 \\
        --output_dir output/expertqa/gpt-4.1/baseline_termination \\
        --judge_model gpt-5.2 \\
        --task expertqa

    python -m simulation.tools.baseline_termination_judge \\
        --input_dir output/competition_math/gpt-4.1 \\
        --task math
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import SingleModelClient

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "judges"
_PROMPT_MATH = _PROMPTS_DIR / "baseline_termination_judge_math.txt"
_PROMPT_EXPERTQA = _PROMPTS_DIR / "baseline_termination_judge_expertqa.txt"

BASELINE_METHODS = ["zero-shot", "zero-shot-cot", "zero-shot-cot-user-profile", "vanilla"]

def _glob_patterns(methods: List[str]) -> List[str]:
    return [f"{m}_*.json" for m in methods]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _strip_thought_tags(text: str) -> str:
    cleaned = re.sub(r"<thought>.*?</thought>\s*", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"</?message>", "", cleaned).strip()
    return cleaned if cleaned else text


def _extract_user_queries(conversation: List) -> List[Tuple[int, str]]:
    """Return (1-based turn number, user text) for each user turn."""
    queries: List[Tuple[int, str]] = []
    turn = 0
    for entry in conversation:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            role, text = str(entry[0]).strip().lower(), str(entry[1]).strip()
        elif isinstance(entry, dict):
            role = entry.get("role", "").strip().lower()
            text = entry.get("content", "").strip()
        else:
            continue
        if role in ("user", "human", "student"):
            turn += 1
            text = _strip_thought_tags(text)
            # Strip SimulatorArena "terminate conversation" signals so the judge
            # evaluates content, not the termination keyword itself.
            text = re.sub(r"\bterminate conversation\b", "", text, flags=re.I).strip()
            if text:
                queries.append((turn, text))
    return queries


def _format_user_queries(queries: List[Tuple[int, str]]) -> str:
    lines = []
    for turn, text in queries:
        lines.append(f"[Turn {turn}] {text}")
    return "\n\n".join(lines)


def _build_prompt(template: str, item: Dict[str, Any]) -> str:
    conversation = item.get("conversation") or []
    queries = _extract_user_queries(conversation)
    user_queries_text = _format_user_queries(queries)

    problem = str(
        item.get("problem") or item.get("question") or item.get("math_problem", "")
    ).strip()

    return (
        template
        .replace("{user_queries}", user_queries_text)
        .replace("{problem}", problem)
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(raw: str, n_turns: int) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"judged_end_turn": None, "judged_termination_reason": "parse_error"}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {"judged_end_turn": None, "judged_termination_reason": "parse_error"}

    try:
        ending_turn = int(parsed.get("ending_turn", n_turns))
        ending_turn = max(1, min(ending_turn, n_turns))
    except (TypeError, ValueError):
        ending_turn = n_turns

    reason = str(parsed.get("termination_reason", "max_turns"))
    return {
        "judged_end_turn": ending_turn,
        "judged_termination_reason": reason,
        "judged_turn_analysis": parsed.get("turn_analysis", []),
    }


# ---------------------------------------------------------------------------
# Batch judging
# ---------------------------------------------------------------------------

async def judge_items(
    items: List[Dict[str, Any]],
    template: str,
    judge_model: str,
    llm_provider: str,
    force: bool,
) -> List[Dict[str, Any]]:
    client = SingleModelClient(judge_model, provider=llm_provider)

    to_judge: List[int] = []
    contexts: List[List[Dict[str, str]]] = []

    for i, item in enumerate(items):
        if not force and item.get("judged_end_turn") is not None:
            continue
        prompt_text = _build_prompt(template, item)
        to_judge.append(i)
        contexts.append([{"role": "user", "content": prompt_text}])

    if not contexts:
        print("  All items already judged — skipping.")
        return items

    print(f"  Judging {len(contexts)} conversations...")
    responses_batched = await client.generate_responses(
        full_contexts=contexts,
        temperature=0.0,
        max_tokens=2048,
        n=1,
        show_progress=True,
        json_mode=True,
    )

    for list_idx, item_idx in enumerate(to_judge):
        raw_responses = responses_batched[list_idx]
        raw = raw_responses[0] if raw_responses else ""
        if not raw:
            print(f"  WARNING: Empty judge response for item {item_idx} "
                  f"(problem_id={items[item_idx].get('problem_id', '?')})")
        n_turns = items[item_idx].get("turns", 15)
        result = _parse_response(raw, n_turns)
        if result.get("judged_termination_reason") == "parse_error" and raw:
            print(f"  WARNING: Could not parse judge response for item {item_idx} "
                  f"(problem_id={items[item_idx].get('problem_id', '?')}): {raw[:200]}")
        items[item_idx].update(result)

    return items


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

async def judge_file(
    path: Path,
    template: str,
    judge_model: str,
    llm_provider: str,
    force: bool,
) -> Optional[List[Dict[str, Any]]]:
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  ERROR loading {path.name}: {exc}")
        return None

    if not isinstance(items, list) or not items:
        return None

    items = await judge_items(items, template, judge_model, llm_provider, force)
    return items


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def _print_summary(all_rows: List[Dict[str, Any]]) -> None:
    # Group by (method, level, strategy)
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in all_rows:
        key = (r.get("method", "?"), r.get("level", "?"), r.get("strategy", "?"))
        groups[key].append(r)

    METHOD_ORDER = ["zero-shot", "zero-shot-cot", "zero-shot-cot-user-profile", "vanilla"]
    LEVEL_ORDER = ["novice", "intermediate", "advanced"]
    STRAT_ORDER = ["adaptive", "comprehensive", "socratic"]

    header = (
        f"{'method':<30} {'level':<14} {'strategy':<14} {'n':>4} "
        f"{'actual_turns':>13} {'judged_end':>11} {'delta':>7} "
        f"{'completion%':>12} {'stalling%':>10}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for method in METHOD_ORDER:
        for level in LEVEL_ORDER:
            for strat in STRAT_ORDER:
                grp = groups.get((method, level, strat), [])
                if not grp:
                    continue
                n = len(grp)
                actual = [r["actual_turns"] for r in grp]
                judged = [r["judged_end_turn"] for r in grp if r.get("judged_end_turn") is not None]
                deltas = [r["actual_turns"] - r["judged_end_turn"]
                          for r in grp if r.get("judged_end_turn") is not None]
                completion_pct = sum(
                    1 for r in grp
                    if r.get("judged_termination_reason") in ("problem_completion", "comprehension_satisfied")
                ) / n * 100
                stalling_pct = sum(
                    1 for r in grp
                    if r.get("judged_termination_reason") in ("circular_queries", "unproductive_stalling")
                ) / n * 100

                def _ms(vals: List) -> str:
                    if not vals:
                        return "  —  "
                    return f"{statistics.mean(vals):.2f}"

                lines.append(
                    f"{method:<30} {level:<14} {strat:<14} {n:>4} "
                    f"{_ms(actual):>13} {_ms(judged):>11} {_ms(deltas):>7} "
                    f"{completion_pct:>11.0f}% {stalling_pct:>9.0f}%"
                )
        lines.append(sep)

    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir) if args.output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    template = (
        _PROMPT_EXPERTQA if args.task == "expertqa" else _PROMPT_MATH
    ).read_text(encoding="utf-8")

    # Collect all matching baseline files (recursive — per-participant runs
    # nest each session's output under <pid>_s<i>_<strategy>/ subdirs).
    methods = args.methods if args.methods else BASELINE_METHODS
    files: List[Path] = []
    for pattern in _glob_patterns(methods):
        files.extend(p for p in input_path.rglob(pattern) if ".bak" not in p.name)
    files = sorted(set(files))

    if not files:
        print(f"No baseline JSON files found in {input_path}")
        return

    print(f"Found {len(files)} baseline file(s) to judge.")

    all_rows: List[Dict[str, Any]] = []

    for fpath in files:
        print(f"\n{fpath.name}")
        stem = fpath.stem
        parts = stem.split("_")

        # Parse method, level, strategy from filename
        # e.g. zero-shot_zero-shot_novice_adaptive_gpt-4.1
        method = parts[0]
        level, strategy = "unknown", "unknown"
        for i, p in enumerate(parts):
            if p in ("novice", "intermediate", "advanced"):
                level = p
                if i + 1 < len(parts):
                    strategy = parts[i + 1]
                break

        items = await judge_file(
            fpath, template, args.judge_model, args.llm_provider, args.force
        )
        if items is None:
            continue

        # Save augmented JSON. When --output_dir is not specified, write
        # back to the original file path (in-place, preserves per-session
        # subdir structure for per-participant runs). When --output_dir is
        # specified, mirror the relative subdir under it.
        if args.output_dir:
            rel = fpath.relative_to(input_path)
            out_file = output_path / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_file = fpath
        out_file.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved: {out_file}")

        for item in items:
            all_rows.append({
                "method": method,
                "level": level,
                "strategy": strategy,
                "actual_turns": item.get("turns", 0),
                "judged_end_turn": item.get("judged_end_turn"),
                "judged_termination_reason": item.get("judged_termination_reason", ""),
            })

    if all_rows:
        _print_summary(all_rows)

        # Save summary CSV
        csv_path = output_path / "baseline_termination_summary.csv"
        cols = ["method", "level", "strategy", "actual_turns", "judged_end_turn", "judged_termination_reason"]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in all_rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        print(f"\nSaved: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc termination judge for baseline conversations.")
    parser.add_argument("--input_dir", required=True, help="Directory with baseline output JSONs.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write augmented JSONs (default: same as input_dir).",
    )
    parser.add_argument(
        "--task",
        default="math",
        choices=["math", "expertqa"],
        help="Task type — selects the judge prompt.",
    )
    parser.add_argument("--judge_model", default="gpt-5.2", help="LLM model for judging.")
    parser.add_argument(
        "--llm_provider",
        default="openai",
        choices=["openai", "gemini", "anthropic", "together", "groq"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=BASELINE_METHODS,
        default=None,
        help="Which baseline methods to judge (default: all). E.g. --methods zero-shot-cot",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-judge items that already have judged_end_turn set.",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
