"""Orchestrator for assistant benchmarking — multi-model, multi-seed.

Reads a YAML config, pre-extracts IU graphs once, then spawns one subprocess
per (model x level x seed) condition via ``simulation.benchmarking.runner``.

Usage:
    python -m simulation.benchmarking.run_benchmarking \
        --config simulation/benchmarking/config_math.yaml --dry_run

    python -m simulation.benchmarking.run_benchmarking \
        --config simulation/benchmarking/config_math.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _generate_run_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"bench_{ts}"


def _get_git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _resolve_seeds(cfg: Dict[str, Any]) -> List[int]:
    exp = cfg.get("experiment", {})
    seeds = exp.get("seeds")
    if seeds and isinstance(seeds, list):
        return [int(s) for s in seeds]
    return [int(exp.get("seed", 42))]


def _resolve_assistant_models(cfg: Dict[str, Any], cli_model: Optional[str], cli_provider: Optional[str]) -> List[Dict[str, str]]:
    if cli_model:
        return [{"model": cli_model, "provider": cli_provider or ""}]
    multi = cfg.get("models", {}).get("assistant_models")
    if multi:
        out: List[Dict[str, str]] = []
        for entry in multi:
            if isinstance(entry, str):
                out.append({"model": entry, "provider": ""})
            else:
                out.append({"model": entry["model"], "provider": entry.get("provider", "")})
        return out
    return [{"model": cfg["models"]["assistant_model"], "provider": ""}]


def _dataset_row_count(cfg: Dict[str, Any]) -> int:
    from ..data.loaders import load_csv_rows, load_jsonl_rows
    exp = cfg.get("experiment", {})
    data = cfg.get("data", {})
    task = exp.get("task", "math")
    if task == "expertqa":
        path = (data.get("expertqa_jsonl") or "").strip()
        if not path or not Path(path).is_file():
            return 0
        return len(load_jsonl_rows(path))
    path = (data.get("input_csv") or "").strip()
    if not path or not Path(path).is_file():
        return 0
    return len(load_csv_rows(path))


def _run_iu_extraction(cfg: Dict[str, Any], output_dir: str, dry_run: bool, start_index: int = 0) -> str:
    """Pre-extract IU graphs once and return the cache file path."""
    conds = cfg.get("conditions", {})
    existing = conds.get("iu_cache_path", "").strip()
    if existing:
        print(f"\n[IU cache] Using pre-built cache: {existing}")
        return existing

    exp = cfg["experiment"]
    models = cfg["models"]
    data = cfg["data"]
    task = exp["task"]
    task_dir = "expertqa" if task == "expertqa" else "competition_math"
    cache_dir = Path(output_dir) / "iu_cache"
    cache_path = cache_dir / f"{task_dir}_{models['iu_model'].replace('/', '-')}_{exp.get('name', 'bench')}.json"

    print(f"\n{'='*60}")
    print("Pre-extracting IU graphs...")
    print(f"  Output: {cache_path}")
    print(f"{'='*60}")

    if dry_run:
        print("  (dry_run — skipping)")
        return str(cache_path)

    cmd = [
        sys.executable, "-m", "simulation.tools.extract_iu_graphs",
        "--output", str(cache_path),
        "--iu_model", models["iu_model"],
        "--iu_max_tokens", str(models.get("iu_max_tokens", 0)),
        "--llm_provider", models["llm_provider"],
        "--prompts_root", "simulation/prompts",
        "--num_problems", str(exp["n_per_condition"]),
    ]
    if task == "expertqa":
        cmd.extend(["--expertqa_jsonl", data.get("expertqa_jsonl", "")])
    else:
        cmd.extend(["--input_csv", data["input_csv"]])
    if start_index > 0:
        cmd.extend(["--start_index", str(start_index)])
    if cache_path.exists():
        cmd.extend(["--existing_cache", str(cache_path)])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  WARNING: IU extraction failed — conditions will extract independently.")
        return ""
    return str(cache_path)


def _build_commands(
    cfg: Dict[str, Any],
    model_entry: Dict[str, str],
    seeds: List[int],
    iu_cache_path: str,
    start_index: int,
    output_dir: str,
    run_id: str,
) -> List[Dict[str, Any]]:
    """Build all runner commands for one assistant model."""
    exp = cfg["experiment"]
    models = cfg["models"]
    data = cfg["data"]
    conds = cfg.get("conditions", {})
    knowledge_levels = conds.get("knowledge_levels", ["novice", "intermediate", "advanced"])
    extra_flags = conds.get("extra_flags", [])

    commands = []
    for seed in seeds:
        for level in knowledge_levels:
            label = f"bench_{model_entry['model']}_{level}_seed{seed}"
            cmd = [
                sys.executable, "-m", "simulation.benchmarking.runner",
                "--task", exp["task"],
                "--knowledge_level", level,
                "--user_model", models["user_model"],
                "--assistant_model", model_entry["model"],
                "--iu_model", models["iu_model"],
                "--iu_max_tokens", str(models.get("iu_max_tokens", 0)),
                "--llm_provider", models["llm_provider"],
                "--seed", str(seed),
                "--max_turns", str(exp.get("max_turns", 15)),
                "--output_dir", output_dir,
                "--num_conversations", str(exp["n_per_condition"]),
            ]
            if model_entry.get("provider"):
                cmd.extend(["--assistant_llm_provider", model_entry["provider"]])
            if exp["task"] == "expertqa":
                cmd.extend(["--expertqa_jsonl", data.get("expertqa_jsonl", "")])
            else:
                cmd.extend(["--input_csv", data.get("input_csv", "")])
            if data.get("interaction_profile_path"):
                cmd.extend(["--interaction_profile_path", data["interaction_profile_path"]])
            if iu_cache_path:
                cmd.extend(["--iu_cache_path", iu_cache_path])
            if run_id:
                cmd.extend(["--run_id", run_id])
            if start_index > 0:
                cmd.extend(["--start_index", str(start_index)])
            if exp.get("absorption_mode"):
                cmd.extend(["--absorption_mode", exp["absorption_mode"]])
            for flag in extra_flags:
                cmd.extend(flag.strip().split(None, 1))
            commands.append({"label": label, "cmd": cmd})
    return commands


def _run_commands_pool(entries: List[Dict[str, Any]], max_parallel: int,
                       log_root: Path) -> Tuple[int, List[str]]:
    """Run commands in a pool of up to ``max_parallel`` concurrent subprocesses."""
    if max_parallel <= 1 or len(entries) <= 1:
        succeeded = 0
        failed: List[str] = []
        for entry in entries:
            print(f"\n{'='*60}")
            print(f"Running: {entry['label']}")
            print(f"  {' '.join(entry['cmd'])}")
            print(f"{'='*60}")
            result = subprocess.run(entry["cmd"], capture_output=False)
            if result.returncode != 0:
                print(f"  ERROR: {entry['label']} exited with code {result.returncode}")
                failed.append(entry["label"])
            else:
                succeeded += 1
        return succeeded, failed

    log_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[parallel] running {len(entries)} conditions with pool size {max_parallel}")

    queue = list(entries)
    in_flight: Dict[Any, Dict[str, Any]] = {}
    succeeded = 0
    failed: List[str] = []

    while queue or in_flight:
        while queue and len(in_flight) < max_parallel:
            entry = queue.pop(0)
            log_path = log_root / f"{entry['label']}.log"
            log_handle = open(log_path, "w", buffering=1)
            print(f"[start] {entry['label']}  → {log_path}")
            popen = subprocess.Popen(
                entry["cmd"], stdout=log_handle, stderr=subprocess.STDOUT,
            )
            in_flight[popen] = {"entry": entry, "log_handle": log_handle, "log_path": log_path}
        time.sleep(0.5)
        finished = [p for p in in_flight if p.poll() is not None]
        for popen in finished:
            ctx = in_flight.pop(popen)
            ctx["log_handle"].close()
            entry = ctx["entry"]
            print(f"[done]  {entry['label']} (exit={popen.returncode})")
            if popen.returncode == 0:
                succeeded += 1
            else:
                failed.append(entry["label"])
                try:
                    with open(ctx["log_path"]) as f:
                        tail = f.readlines()[-10:]
                    if tail:
                        print(f"  [tail of {ctx['log_path']}]:")
                        for line in tail:
                            print(f"    {line.rstrip()}")
                except Exception:
                    pass

    return succeeded, failed


def _output_run_dir(cfg: Dict[str, Any], model: str, output_dir: str, run_id: Optional[str] = None) -> Path:
    task = cfg["experiment"]["task"]
    task_dir = "expertqa" if task == "expertqa" else "competition_math"
    base = Path(output_dir) / task_dir / model
    if run_id:
        return base / run_id
    return base


async def _run_judge(cfg: Dict[str, Any], model: str, output_dir: str, run_id: str, dry_run: bool) -> None:
    run_dir = _output_run_dir(cfg, model, output_dir, run_id)
    if not run_dir.exists():
        return
    print(f"\n{'='*60}")
    print(f"Running LLM judge on {run_dir}...")
    print(f"{'='*60}")
    if dry_run:
        print("  (dry_run — skipping)")
        return

    from ..tools.judge_conversations import judge_file, _load_judge_prompt
    prompt_template = _load_judge_prompt()
    paths = sorted(run_dir.rglob("*.json"))
    paths = [p for p in paths if "metrics" not in p.parts and "dashboard" not in p.parts]
    coros = [
        judge_file(
            path=path,
            prompt_template=prompt_template,
            judge_model=cfg["models"]["judge_model"],
            llm_provider=cfg["models"]["llm_provider"],
            target_methods=None,
            force=False,
            dry_run=dry_run,
        )
        for path in paths
    ]
    counts = await asyncio.gather(*coros, return_exceptions=True)
    total = sum(int(n or 0) for n in counts if not isinstance(n, Exception))
    print(f"  Judged {total} conversations.")


def _write_manifest(cfg: Dict[str, Any], config_path: str, run_id: str,
                    output_run_dir: Path) -> None:
    config_text = Path(config_path).read_text(encoding="utf-8")
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()[:16]
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "config_hash": config_hash,
        "config": cfg,
        "type": "benchmarking",
        **_get_git_info(),
    }
    output_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_run_dir / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Manifest: {manifest_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmarking orchestrator — multi-model, multi-seed.")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmarking YAML config.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--assistant_model", type=str, default=None,
                        help="Override: run a single model instead of the full list.")
    parser.add_argument("--assistant_llm_provider", type=str, default=None)
    parser.add_argument("--n_per_condition", type=int, default=None)
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--start_index", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    if args.output_dir is None:
        args.output_dir = cfg.get("experiment", {}).get("output_dir", "output_benchmarking")
    if args.n_per_condition is not None:
        cfg["experiment"]["n_per_condition"] = args.n_per_condition

    start_index = max(0, int(args.start_index or cfg.get("experiment", {}).get("start_index", 0)))
    seeds = _resolve_seeds(cfg)
    model_list = _resolve_assistant_models(cfg, args.assistant_model, args.assistant_llm_provider)
    run_id = _generate_run_id()

    exp_name = cfg["experiment"].get("name", "benchmarking")
    print(f"Benchmarking: {exp_name}")
    print(f"Run ID: {run_id}")
    print(f"Task: {cfg['experiment']['task']}, N: {cfg['experiment'].get('n_per_condition', '?')}")
    print(f"Seeds: {seeds}")
    print(f"Models: {[m['model'] for m in model_list]}")

    # Pre-extract IU graphs (shared across all models)
    iu_cache_path = _run_iu_extraction(cfg, args.output_dir, args.dry_run, start_index=start_index)

    log_root = Path(args.output_dir) / "condition_logs"

    for model_entry in model_list:
        model_name = model_entry["model"]
        print(f"\n{'#'*60}")
        print(f"# Model: {model_name} (provider: {model_entry.get('provider') or 'auto'})")
        print(f"{'#'*60}")

        out_dir = _output_run_dir(cfg, model_name, args.output_dir, run_id)
        if not args.dry_run:
            _write_manifest(cfg, args.config, run_id, out_dir)

        commands = _build_commands(
            cfg, model_entry, seeds, iu_cache_path,
            start_index, args.output_dir, run_id,
        )
        print(f"  Conditions: {len(commands)}")
        for entry in commands:
            print(f"  [{entry['label']}]  {' '.join(entry['cmd'])}")

        if args.dry_run:
            print(f"  (dry_run — no commands executed for {model_name})")
            continue

        succeeded, failed = _run_commands_pool(commands, args.max_parallel, log_root)
        print(f"\n  Completed {succeeded}/{len(commands)} conditions for {model_name}.")
        if failed:
            print(f"  Failed: {failed}")

        if not args.skip_judge:
            await _run_judge(cfg, model_name, args.output_dir, run_id, args.dry_run)

    print(f"\nBenchmarking complete. Run ID: {run_id}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
