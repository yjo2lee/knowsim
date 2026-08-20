"""Post-hoc LLM judge for Information Calibration, Perceived Overload, and Quality Score.

Reads output JSON files (typically from baseline runs) and adds `judged_metrics` to each
conversation item. Safe to re-run: skips items that already have `judged_metrics` unless
--force is passed.

Usage:
    python -m simulation.tools.judge_conversations \
        --input_dir output/competition_math/gpt-4o \
        --judge_model gpt-5.2 \
        --methods zero-shot zero-shot-cot zero-shot-cot-user-profile vanilla

    # Judge everything in a directory (including structured, for cross-validation):
    python -m simulation.tools.judge_conversations \
        --input_dir output/competition_math/gpt-4o \
        --judge_model gpt-5.2 \
        --all_methods
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import SingleModelClient

# Methods that lack native IC/PO and must be judged.
BASELINE_METHODS = {"vanilla", "zero-shot", "zero-shot-cot", "zero-shot-cot-user-profile"}

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "judges" / "conversation_quality_judge.md"


def _load_judge_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _strip_thought_tags(text: str) -> str:
    """Remove <thought>...</thought> blocks and <message> wrapper tags from user messages."""
    cleaned = re.sub(r"<thought>.*?</thought>\s*", "", text, flags=re.DOTALL).strip()
    # Strip <message>...</message> wrapper if present (keep inner content)
    cleaned = re.sub(r"</?message>", "", cleaned).strip()
    return cleaned if cleaned else text


def _format_conversation(conversation: Any) -> str:
    """Convert conversation list of [role, text] pairs to a readable string."""
    if not conversation:
        return "(empty conversation)"
    lines: List[str] = []
    turn_num = 0
    buffer: List[str] = []

    for entry in conversation:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            role, text = str(entry[0]).strip().lower(), str(entry[1]).strip()
        elif isinstance(entry, dict):
            role = entry.get("role", "").strip().lower()
            text = entry.get("content", "").strip()
        else:
            continue

        if role in ("user", "human", "student"):
            if buffer:
                # Flush previous assistant turn
                lines.append("\n".join(buffer))
                buffer = []
            turn_num += 1
            text = _strip_thought_tags(text)
            buffer.append(f"[Turn {turn_num}]")
            buffer.append(f"Student: {text}")
        elif role in ("assistant", "tutor"):
            buffer.append(f"Tutor: {text}")

    if buffer:
        lines.append("\n".join(buffer))

    return "\n\n".join(lines) if lines else "(empty conversation)"


def _build_judge_prompt(prompt_template: str, item: Dict[str, Any]) -> str:
    conversation = item.get("conversation") or item.get("assistant_messages") or []
    # assistant_messages is [{role, content}, ...]
    if conversation and isinstance(conversation[0], dict) and "role" in conversation[0]:
        conversation_text = _format_conversation(conversation)
    else:
        conversation_text = _format_conversation(conversation)

    knowledge_level = item.get("knowledge_level", "unknown")
    ref_raw = item.get("reference_answer") or item.get("solution") or ""
    ref_text = str(ref_raw).strip()
    if not ref_text:
        reference_answer = "(Not provided — evaluate based on the conversation only.)"
    else:
        reference_answer = ref_text

    filled = (
        prompt_template.replace("{conversation_text}", conversation_text)
        .replace("{knowledge_level}", knowledge_level)
        .replace("{reference_answer}", reference_answer)
    )
    return filled


def _parse_judge_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from the LLM response, tolerating markdown code fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
    # Find first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    def _clamp(val: Any, lo: float, hi: float) -> Optional[float]:
        try:
            return max(lo, min(hi, float(val)))
        except (TypeError, ValueError):
            return None

    return {
        "information_calibration": _clamp(parsed.get("information_calibration"), 0.0, 1.0),
        "perceived_overload": _clamp(parsed.get("perceived_overload"), 0.0, 1.0),
        "quality_score": _clamp(parsed.get("quality_score"), 1.0, 10.0),
        "perceived_learning": _clamp(parsed.get("perceived_learning"), 1.0, 10.0),
        "ic_reasoning": str(parsed.get("ic_reasoning", "")),
        "po_reasoning": str(parsed.get("po_reasoning", "")),
        "quality_reasoning": str(parsed.get("quality_reasoning", "")),
        "pl_reasoning": str(parsed.get("pl_reasoning", "")),
    }


async def judge_items(
    items: List[Dict[str, Any]],
    prompt_template: str,
    judge_model: str,
    llm_provider: str,
    force: bool,
) -> List[Dict[str, Any]]:
    """Add `judged_metrics` to each item in-place and return the list."""
    client = SingleModelClient(judge_model, provider=llm_provider)

    # Build contexts only for items that need judging
    indices_to_judge: List[int] = []
    contexts: List[List[Dict[str, str]]] = []

    for i, item in enumerate(items):
        jm = item.get("judged_metrics")
        if not force and jm and jm.get("quality_score") is not None:
            continue
        prompt_text = _build_judge_prompt(prompt_template, item)
        indices_to_judge.append(i)
        contexts.append([{"role": "user", "content": prompt_text}])

    if not contexts:
        return items

    print(f"  Judging {len(contexts)} conversations...")
    responses_batched = await client.generate_responses(
        full_contexts=contexts,
        temperature=0.0,
        # 4096: Claude Sonnet 4.6 sometimes writes a long reasoning preamble
        # before the JSON. We started at 512, raised to 2048 (~1% of long
        # MathQA conversations still truncated mid-preamble), and finally to
        # 4096 (empirically clears all observed cases). Actual output is
        # typically ≤1k tok; we only pay for what's used.
        max_tokens=4096,
        n=1,
        show_progress=True,
        json_mode=True,
    )

    for list_idx, item_idx in enumerate(indices_to_judge):
        raw_responses = responses_batched[list_idx]
        raw = raw_responses[0] if raw_responses else ""
        parsed = _parse_judge_response(raw)
        if parsed is None:
            print(f"  WARNING: Could not parse judge response for item {item_idx}: {raw[:200]}")
            parsed = {
                "information_calibration": None,
                "perceived_overload": None,
                "quality_score": None,
                "perceived_learning": None,
                "ic_reasoning": f"parse_error: {raw[:100]}",
                "po_reasoning": "",
                "quality_reasoning": "",
                "pl_reasoning": "",
            }
        items[item_idx]["judged_metrics"] = parsed

    return items


async def judge_file(
    path: Path,
    prompt_template: str,
    judge_model: str,
    llm_provider: str,
    target_methods: Optional[set],
    force: bool,
    dry_run: bool,
) -> int:
    """Judge all eligible items in a JSON file. Returns number of items judged."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return 0

    eligible = [
        item for item in data
        if isinstance(item, dict)
        and (target_methods is None or item.get("simulation_method") in target_methods)
    ]

    if not eligible:
        return 0

    print(f"[{path.name}] {len(eligible)} eligible items (target methods: {target_methods})")

    if dry_run:
        return len(eligible)

    judged = await judge_items(eligible, prompt_template, judge_model, llm_provider, force)

    # Merge judged results back (they were modified in-place, but we reassure)
    for item, judged_item in zip(eligible, judged):
        item["judged_metrics"] = judged_item.get("judged_metrics")

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return len(eligible)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc LLM judge for baseline conversations.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str, help="Directory to search for JSON files recursively.")
    input_group.add_argument("--input_file", type=str, help="Single JSON file to judge.")
    parser.add_argument("--judge_model", type=str, default="gpt-5.2", help="Model to use as judge.")
    parser.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(BASELINE_METHODS),
        help="Only judge conversations with these simulation_method values.",
    )
    parser.add_argument(
        "--all_methods",
        action="store_true",
        help="Judge all methods (including structured), e.g. for cross-validation.",
    )
    parser.add_argument("--force", action="store_true", help="Re-judge even if judged_metrics already present.")
    parser.add_argument("--dry_run", action="store_true", help="Count eligible items without calling the LLM.")
    args = parser.parse_args()

    prompt_template = _load_judge_prompt()
    target_methods: Optional[set] = None if args.all_methods else set(args.methods)

    if args.input_file:
        paths = [Path(args.input_file)]
    else:
        paths = sorted(Path(args.input_dir).rglob("*.json"))
        # Skip backup files, metrics/ subdirectories, and manifest
        paths = [p for p in paths if "metrics" not in p.parts
                 and not p.name.endswith(".bak.json")
                 and p.name != "experiment_manifest.json"]

    # Fan out per-file judging concurrently.
    counts = await asyncio.gather(
        *[
            judge_file(path, prompt_template, args.judge_model, args.llm_provider,
                       target_methods, args.force, args.dry_run)
            for path in paths
        ],
        return_exceptions=True,
    )
    total = 0
    for path, n in zip(paths, counts):
        if isinstance(n, Exception):
            print(f"  ERROR judging {path.name}: {n}")
        else:
            total += int(n or 0)

    action = "Would judge" if args.dry_run else "Judged"
    print(f"\n{action} {total} conversations across {len(paths)} file(s).")


if __name__ == "__main__":
    asyncio.run(main())
