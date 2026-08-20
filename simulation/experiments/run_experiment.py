"""Orchestrates all conditions for a per-level validation experiment.

Reads an experiment YAML config (one of the four canonical configs under
``simulation/experiments/``), runs each condition as a subprocess, then
runs the post-hoc LLM judge on outputs and generates the comparison report.

Simulator behavioural flags are declared under ``simulator_flags:`` in the
YAML and propagated to each runner subprocess via the ``MYSIMARE_FLAGS``
env var (see ``simulation/core/feature_flags.py``).

Canonical configs:
    simulation/experiments/experiment_config_strategy_arm.yaml
    simulation/experiments/experiment_config_model_arm.yaml
    simulation/experiments/experiment_config_strategy_arm_expertqa.yaml
    simulation/experiments/experiment_config_model_arm_expertqa.yaml

Usage:
    python -m simulation.experiments.run_experiment \\
        --config simulation/experiments/experiment_config_strategy_arm.yaml

    # Dry-run: print all commands without executing
    python -m simulation.experiments.run_experiment \\
        --config <path> --dry_run

    # Run only structured conditions (skip baselines)
    python -m simulation.experiments.run_experiment \\
        --config <path> --only structured

    # Skip judging and comparison (just run simulations)
    python -m simulation.experiments.run_experiment \\
        --config <path> --skip_judge --skip_compare
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import time
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

from ..data.loaders import load_csv_rows, load_jsonl_rows


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _generate_run_id(absorption_mode: str) -> str:
    """Generate a run ID: ``run_YYYYMMDD_HHMMSS_<absorption_mode>``."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{absorption_mode}"


def _get_git_info() -> Dict[str, Any]:
    """Capture current git commit and dirty status. Best-effort."""
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
        diff = subprocess.check_output(
            ["git", "diff", "--stat"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        info["git_dirty"] = bool(diff)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _resolve_extra_flags(extra_flags: List[str]) -> Dict[str, Any]:
    """Parse extra_flags list into a dict for the manifest."""
    resolved: Dict[str, Any] = {}
    for flag in extra_flags:
        parts = flag.strip().split(None, 1)
        if not parts:
            continue
        key = parts[0].lstrip("-").replace("-", "_")
        if len(parts) == 1:
            resolved[key] = True
        else:
            # Try to parse as int/float/bool, fall back to string
            val = parts[1]
            try:
                resolved[key] = int(val)
            except ValueError:
                try:
                    resolved[key] = float(val)
                except ValueError:
                    resolved[key] = val
    return resolved


def _write_manifest(
    cfg: Dict[str, Any],
    config_path: str,
    run_id: str,
    output_run_dir: Path,
    cli_args: argparse.Namespace,
) -> Path:
    """Write experiment_manifest.json into the run directory."""
    from ..knowledge.iu_init import _IU_INIT_RATIOS

    config_text = Path(config_path).read_text(encoding="utf-8")
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()[:16]

    extra_flags = cfg.get("structured_conditions", {}).get("extra_flags", [])

    # Build typed feature flags from YAML's optional ``simulator_flags:`` section,
    # falling back to env vars / dataclass defaults. Recorded in manifest and
    # propagated to subprocess runners via the MYSIMARE_FLAGS env var (see
    # ``_run_commands_pool`` and ``_run_one``).
    from simulation.core.feature_flags import SimulatorFeatureFlags
    sim_flags = (
        SimulatorFeatureFlags.from_dict(cfg["simulator_flags"])
        if "simulator_flags" in cfg
        else SimulatorFeatureFlags.from_env()
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "config_hash": config_hash,
        "config": cfg,
        "resolved_flags": _resolve_extra_flags(extra_flags),
        "simulator_flags": sim_flags.to_dict(),
        "initial_knowledge_state_ratios": _IU_INIT_RATIOS,
        **_get_git_info(),
        "cli_args": {
            k: v for k, v in vars(cli_args).items()
            if k not in ("config",)
        },
    }

    output_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_run_dir / "experiment_manifest.json"
    if manifest_path.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        only = getattr(cli_args, "only", "all")
        manifest_path = output_run_dir / f"experiment_manifest_{only}_{ts}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Manifest written: {manifest_path}")
    return manifest_path


def _dataset_row_count(cfg: Dict[str, Any]) -> int:
    """Return number of items in the configured dataset (expertqa JSONL or math CSV)."""
    exp = cfg.get("experiment", {})
    data = cfg.get("data", {})
    task = exp.get("task", "math")
    if task == "expertqa":
        path = (data.get("expertqa_jsonl") or "").strip()
        if not path:
            return 0
        p = Path(path)
        if not p.is_file():
            return 0
        return len(load_jsonl_rows(str(p)))
    path = (data.get("input_csv") or "").strip()
    if not path:
        return 0
    p = Path(path)
    if not p.is_file():
        return 0
    return len(load_csv_rows(str(p)))


def _resolve_experiment_start_index(cfg: Dict[str, Any], cli_start: Optional[int]) -> int:
    """Effective 0-based start row: CLI overrides config; use_last_n uses tail slice."""
    exp = cfg.get("experiment", {})
    if cli_start is not None:
        return max(0, int(cli_start))

    if exp.get("use_last_n_problems"):
        n = int(exp.get("n_per_condition", 0) or 0)
        total = _dataset_row_count(cfg)
        if n <= 0 or total <= 0:
            print(
                "WARNING: use_last_n_problems needs n_per_condition > 0 and a readable dataset; "
                "using start_index=0."
            )
            return max(0, int(exp.get("start_index") or 0))
        return max(0, total - n)

    return max(0, int(exp.get("start_index") or 0))


def _resolve_assistant_models(
    cfg: Dict[str, Any],
    cli_model: Optional[str],
    cli_provider: Optional[str],
) -> List[Dict[str, str]]:
    """Resolve the list of assistant models to iterate over.

    Priority:
      1. CLI --assistant_model flag → single-element list
      2. cfg["models"]["assistant_models"] → multi-model list
      3. cfg["models"]["assistant_model"] → single-element fallback
    """
    if cli_model:
        return [{"model": cli_model, "provider": cli_provider or ""}]

    multi = cfg.get("models", {}).get("assistant_models")
    if multi:
        out: List[Dict[str, str]] = []
        for entry in multi:
            if isinstance(entry, str):
                out.append({"model": entry, "provider": ""})
            else:
                d = {
                    "model": entry["model"],
                    "provider": entry.get("provider", ""),
                }
                if entry.get("gemini_thinking_level"):
                    d["gemini_thinking_level"] = entry["gemini_thinking_level"]
                out.append(d)
        return out

    return [{
        "model": cfg["models"]["assistant_model"],
        "provider": cfg["models"].get("assistant_llm_provider", ""),
    }]


def _build_structured_commands(
    cfg: Dict[str, Any],
    iu_cache_path: str = "",
    start_index: int = 0,
    output_dir: str = "output",
    run_id: str = "",
) -> List[Dict[str, Any]]:
    """Build all structured-runner commands from config."""
    exp = cfg["experiment"]
    models = cfg["models"]
    data = cfg["data"]
    sc = cfg["structured_conditions"]

    problem_indices: List[int] = exp.get("problem_indices", [])

    absorption_mode = exp.get("absorption_mode", "default")
    temporal_decay = exp.get("temporal_decay", False)

    def _base_cmd(knowledge_level: str, assistant_strategy: str,
                  prescreening_score: Optional[int] = None) -> List[str]:
        cmd = [
            sys.executable, "-m", "simulation.runtime.runner",
            "--task", exp["task"],
            "--simulation_method", "structured",
            "--knowledge_level", knowledge_level,
            "--assistant_strategy", assistant_strategy,
            "--user_model", models["user_model"],
            "--assistant_model", models["assistant_model"],
            "--iu_model", models["iu_model"],
            "--iu_max_tokens", str(models.get("iu_max_tokens", 0)),
            "--llm_provider", models["llm_provider"],
            "--seed", str(exp["seed"]),
            "--input_csv", data.get("input_csv", ""),
            "--interaction_profile_path", data["interaction_profile_path"],
            "--output_dir", output_dir,
            "--absorption_mode", absorption_mode,
        ]
        if run_id:
            cmd.extend(["--run_id", run_id])
        if temporal_decay:
            cmd.append("--temporal_decay")
        if models.get("assistant_llm_provider"):
            cmd.extend(["--assistant_llm_provider", models["assistant_llm_provider"]])
        if models.get("assistant_gemini_thinking_level"):
            cmd.extend(["--assistant_gemini_thinking_level", models["assistant_gemini_thinking_level"]])
        if exp["task"] == "expertqa":
            cmd.extend(["--expertqa_jsonl", data.get("expertqa_jsonl", "")])
        if iu_cache_path:
            cmd.extend(["--iu_cache_path", iu_cache_path])
        init_ks_cache = sc.get("init_ks_cache_path", "").strip()
        if init_ks_cache:
            cmd.extend(["--init_ks_cache_path", init_ks_cache])
        if exp.get("no_resume"):
            cmd.append("--no_resume")
        if exp.get("flat_output_layout"):
            cmd.append("--flat_output_layout")
        if prescreening_score is not None:
            cmd.extend(["--prescreening_score", str(prescreening_score)])
        for flag in sc.get("extra_flags", []):
            cmd.extend(flag.strip().split(None, 1))
        return cmd

    # Determine iteration axis: prescreening_scores (fine-grained) or
    # knowledge_levels (categorical). When prescreening_scores is set,
    # knowledge_levels is ignored for command generation — the runner
    # derives the categorical level from the score via score_to_level().
    prescreening_scores: List[int] = sc.get("prescreening_scores", [])

    commands = []
    if prescreening_scores:
        from ..knowledge.iu_init import score_to_level
        for score in prescreening_scores:
            level_label = score_to_level(score)
            score_label = f"score_{score}"
            for assistant_strategy in sc["assistant_strategies"]:
                group = f"structured_{score_label}_{assistant_strategy}"
                cmd = _base_cmd(level_label, assistant_strategy, prescreening_score=score)
                if problem_indices:
                    cmd.extend(["--problem_indices", *[str(i) for i in problem_indices]])
                else:
                    cmd.extend(["--num_conversations", str(exp["n_per_condition"])])
                    if start_index > 0:
                        cmd.extend(["--start_index", str(start_index)])
                commands.append({"label": group, "cmd": cmd, "type": "structured", "group": group})
    else:
        for knowledge_level in sc["knowledge_levels"]:
            for assistant_strategy in sc["assistant_strategies"]:
                group = f"structured_{knowledge_level}_{assistant_strategy}"
                cmd = _base_cmd(knowledge_level, assistant_strategy)
                if problem_indices:
                    cmd.extend(["--problem_indices", *[str(i) for i in problem_indices]])
                else:
                    cmd.extend(["--num_conversations", str(exp["n_per_condition"])])
                    if start_index > 0:
                        cmd.extend(["--start_index", str(start_index)])
                commands.append({"label": group, "cmd": cmd, "type": "structured", "group": group})
    return commands


def _resolve_umbrella_cache_path(
    output_dir: str, task_dir: str, iu_model: str, source_csv: str
) -> Optional[Path]:
    """Compute the convention path for an umbrella IU cache.

    Convention: ``<task_root>/iu_cache/<iu_model>_<source_csv_basename>.json``
    where ``<task_root>`` is the part of ``output_dir`` up to and including
    the task segment (e.g., ``output/competition_math``).

    Returns ``None`` if the output_dir doesn't contain the task segment
    (i.e., ad-hoc / non-umbrella runs — they fall back to the per-output
    auto-build location).
    """
    parts = Path(output_dir).parts
    if task_dir not in parts:
        return None
    idx = parts.index(task_dir)
    task_root = Path(*parts[: idx + 1])
    if not source_csv:
        return None
    csv_basename = Path(source_csv).stem  # drop extension
    iu_model_safe = iu_model.replace("/", "-")
    return task_root / "iu_cache" / f"{iu_model_safe}_{csv_basename}.json"


def _run_iu_extraction(cfg: Dict[str, Any], output_dir: str, dry_run: bool, start_index: int = 0) -> str:
    """Pre-extract IU graphs once and return the cache file path.

    Returns the path to the cache file, or "" on failure / dry_run.

    Resolution order:
      1. If ``structured_conditions.iu_cache_path`` is set, use it directly.
      2. If the run's output_dir is under a task umbrella (e.g.,
         ``output/competition_math/...``), use the
         convention path ``<task_root>/iu_cache/<iu_model>_<csv_stem>.json``.
         The cache lives next to ``per_level/`` and ``per_participant/`` so
         all experiments under the same task share it.
      3. Otherwise (ad-hoc / non-umbrella runs), fall back to
         ``<output_dir>/iu_cache/<task>_<iu_model>_<experiment.name>.json``.
    """
    sc = cfg.get("structured_conditions", {})
    existing = sc.get("iu_cache_path", "").strip()
    if existing:
        print(f"\n[IU cache] Using pre-built cache from config: {existing}")
        return existing

    exp = cfg["experiment"]
    models = cfg["models"]
    data = cfg["data"]
    task = exp["task"]
    task_dir = "expertqa" if task == "expertqa" else "competition_math"

    # Try the umbrella convention first.
    source_csv = (data.get("expertqa_jsonl") if task == "expertqa" else data.get("input_csv")) or ""
    umbrella = _resolve_umbrella_cache_path(output_dir, task_dir, models["iu_model"], source_csv)
    if umbrella is not None:
        cache_path = umbrella
        print(f"\n[IU cache] Using umbrella convention path: {cache_path}")
    else:
        cache_dir = Path(output_dir) / "iu_cache"
        cache_path = cache_dir / f"{task_dir}_{models['iu_model'].replace('/', '-')}_{exp.get('name', 'exp')}.json"

    print(f"\n{'='*60}")
    print("Pre-extracting IU graphs (shared across all conditions)...")
    print(f"  Output: {cache_path}")
    print(f"{'='*60}")

    if dry_run:
        print("  (dry_run — skipping IU extraction)")
        return str(cache_path)

    def _extraction_cmd(prob_start: int, num_problems: int) -> List[str]:
        cmd = [
            sys.executable, "-m", "simulation.tools.extract_iu_graphs",
            "--output", str(cache_path),
            "--iu_model", models["iu_model"],
            "--iu_max_tokens", str(models.get("iu_max_tokens", 0)),
            "--llm_provider", models["llm_provider"],
            "--prompts_root", "simulation/prompts",
            "--num_problems", str(num_problems),
        ]
        if task == "expertqa":
            cmd.extend(["--expertqa_jsonl", data.get("expertqa_jsonl", "")])
        else:
            cmd.extend(["--input_csv", data["input_csv"]])
        if prob_start > 0:
            cmd.extend(["--start_index", str(prob_start)])
        if cache_path.exists():
            cmd.extend(["--existing_cache", str(cache_path)])
        return cmd

    problem_indices: List[int] = exp.get("problem_indices", [])
    if problem_indices:
        # Extract each problem separately, accumulating into the shared cache.
        for prob_idx in problem_indices:
            print(f"  Extracting IU graph for problem index {prob_idx}...")
            if not dry_run:
                result = subprocess.run(_extraction_cmd(prob_idx, 1))
                if result.returncode != 0:
                    print(f"  WARNING: IU extraction failed for index {prob_idx} — conditions will extract independently.")
                    return ""
    else:
        if not dry_run:
            result = subprocess.run(_extraction_cmd(start_index, exp["n_per_condition"]))
            if result.returncode != 0:
                print("  WARNING: IU extraction failed — conditions will extract independently.")
                return ""

    return str(cache_path)


def _build_baseline_commands(
    cfg: Dict[str, Any],
    start_index: int = 0,
    output_dir: str = "output",
    run_id: str = "",
) -> List[Dict[str, Any]]:
    """Build all baseline-runner commands from config."""
    exp = cfg["experiment"]
    models = cfg["models"]
    data = cfg["data"]
    bc = cfg["baseline_conditions"]

    # Support both old scalar key and new list key for backwards compatibility.
    strategies = bc.get("assistant_strategies") or [bc.get("assistant_strategy", "adaptive")]
    problem_indices: List[int] = exp.get("problem_indices", [])
    absorption_mode = exp.get("absorption_mode", "default")

    def _base_cmd(method: str, knowledge_level: str, assistant_strategy: str) -> List[str]:
        cmd = [
            sys.executable, "-m", "simulation.baselines.baseline_runner",
            "--baseline_method", method,
            "--task", exp["task"],
            "--knowledge_level", knowledge_level,
            "--assistant_strategy", assistant_strategy,
            "--user_model", models.get("baseline_user_model", models["user_model"]),
            "--assistant_model", models["assistant_model"],
            "--llm_provider", models["llm_provider"],
            "--seed", str(exp["seed"]),
            "--input_csv", data.get("input_csv", ""),
            "--interaction_profile_path", data["interaction_profile_path"],
            "--output_dir", output_dir,
            "--absorption_mode", absorption_mode,
        ]
        if run_id:
            cmd.extend(["--run_id", run_id])
        if models.get("assistant_llm_provider"):
            cmd.extend(["--assistant_llm_provider", models["assistant_llm_provider"]])
        if exp["task"] == "expertqa":
            cmd.extend(["--expertqa_jsonl", data.get("expertqa_jsonl", "")])
        if exp.get("no_resume"):
            cmd.append("--no_resume")
        if exp.get("flat_output_layout"):
            cmd.append("--flat_output_layout")
        if method == "zero-shot-cot-user-profile":
            bc_cache = bc.get("iu_cache_path", "").strip()
            if bc_cache:
                cmd.extend(["--iu_cache_path", bc_cache])
        for flag in bc.get("extra_flags", []):
            cmd.extend(flag.strip().split(None, 1))
        return cmd

    commands = []
    for method in bc["methods"]:
        for knowledge_level in bc["knowledge_levels"]:
            for assistant_strategy in strategies:
                group = f"baseline_{method}_{knowledge_level}_{assistant_strategy}"
                cmd = _base_cmd(method, knowledge_level, assistant_strategy)
                if problem_indices:
                    cmd.extend(["--problem_indices", *[str(i) for i in problem_indices]])
                else:
                    cmd.extend(["--num_conversations", str(exp["n_per_condition"])])
                    if start_index > 0:
                        cmd.extend(["--start_index", str(start_index)])
                commands.append({"label": group, "cmd": cmd, "type": "baseline", "group": group})
    return commands


def _print_command(entry: Dict[str, Any]) -> None:
    print(f"\n[{entry['label']}]")
    print("  " + " ".join(entry["cmd"]))


def _run_command(entry: Dict[str, Any]) -> bool:
    """Run a single command synchronously. Returns True on success.

    Simulator feature flags are passed through the command itself
    (``--simulator_flags <json>`` appended in the per-model loop), so no
    per-entry env-var injection is needed.
    """
    print(f"\n{'='*60}")
    print(f"Running: {entry['label']}")
    print(f"  {' '.join(entry['cmd'])}")
    print(f"{'='*60}")
    result = subprocess.run(entry["cmd"], capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: {entry['label']} exited with code {result.returncode}")
        return False
    print(f"  OK: {entry['label']} completed.")
    return True


def _run_commands_pool(entries: List[Dict[str, Any]], max_parallel: int,
                        log_root: Path = Path("output/condition_logs")) -> tuple[int, List[str]]:
    """Run commands in a pool of up to ``max_parallel`` concurrent subprocesses.

    Each worker's stdout+stderr is redirected to its own file under
    ``log_root/<label>.log``. Doing this through PIPE would deadlock when the
    OS pipe buffer (≈64KB) fills before the parent reads — workers can
    legitimately produce many MB of output per condition.

    The orchestrator dumps the tail of each log when the subprocess exits.
    """
    if max_parallel <= 1 or len(entries) <= 1:
        # Sequential path: live-streaming output goes straight to the terminal.
        succeeded = 0
        failed: List[str] = []
        for entry in entries:
            if _run_command(entry):
                succeeded += 1
            else:
                failed.append(entry["label"])
        return succeeded, failed

    log_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[parallel] running {len(entries)} conditions with pool size {max_parallel}")
    print(f"[parallel] per-condition logs: {log_root}/<label>.log")

    queue = list(entries)
    in_flight: Dict[Any, Dict[str, Any]] = {}  # popen → {entry, log_handle, log_path}
    in_flight_groups: set = set()  # groups with a running subprocess
    succeeded = 0
    failed: List[str] = []

    while queue or in_flight:
        # Fill the pool — skip entries whose group already has an in-flight subprocess.
        i = 0
        while i < len(queue) and len(in_flight) < max_parallel:
            entry = queue[i]
            group = entry.get("group")
            if group is not None and group in in_flight_groups:
                i += 1
                continue
            queue.pop(i)
            log_path = log_root / f"{entry['label']}.log"
            log_handle = open(log_path, "w", buffering=1)
            print(f"\n[start] {entry['label']}  → {log_path}")
            popen = subprocess.Popen(
                entry["cmd"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            in_flight[popen] = {
                "entry": entry,
                "log_handle": log_handle,
                "log_path": log_path,
            }
            if group is not None:
                in_flight_groups.add(group)
        # Wait for any subprocess to finish.
        time.sleep(0.5)
        finished = [p for p in in_flight if p.poll() is not None]
        for popen in finished:
            ctx = in_flight.pop(popen)
            entry = ctx["entry"]
            group = entry.get("group")
            if group is not None:
                in_flight_groups.discard(group)
            ctx["log_handle"].close()
            print(f"[done]  {entry['label']} (exit={popen.returncode}, log={ctx['log_path']})")
            if popen.returncode == 0:
                succeeded += 1
            else:
                failed.append(entry["label"])
                # Print last few lines of the log for quick context.
                try:
                    with open(ctx["log_path"]) as f:
                        tail_lines = f.readlines()[-10:]
                    if tail_lines:
                        print(f"  [last lines of {ctx['log_path']}]:")
                        for line in tail_lines:
                            print(f"    {line.rstrip()}")
                except Exception:
                    pass

    return succeeded, failed


async def _run_judge(cfg: Dict[str, Any], output_dir: str, dry_run: bool,
                     run_id: Optional[str] = None) -> None:
    """Run the post-hoc LLM judge on all output files (baselines + structured)."""
    models = cfg["models"]
    exp = cfg["experiment"]
    judge_input_dir = _output_run_dir(cfg, output_dir, run_id=run_id)

    print(f"\n{'='*60}")
    print("Running post-hoc LLM judge on all outputs...")
    print(f"  Input dir: {judge_input_dir}")
    print(f"{'='*60}")

    if not judge_input_dir.exists():
        print(f"  WARNING: Output directory not found: {judge_input_dir}")
        return

    from ..tools.judge_conversations import judge_file, _load_judge_prompt

    prompt_template = _load_judge_prompt()
    paths = sorted(judge_input_dir.rglob("*.json"))
    paths = [p for p in paths if "metrics" not in p.parts]

    # Judge every output file concurrently. Each judge_file batches its own
    # conversations into a single generate_responses call internally, so
    # gathering at this level multiplies the concurrency by the file count.
    coros = [
        judge_file(
            path=path,
            prompt_template=prompt_template,
            judge_model=models["judge_model"],
            llm_provider=models["llm_provider"],
            target_methods=None,  # judge all methods including structured
            force=False,
            dry_run=dry_run,
        )
        for path in paths
    ]
    counts = await asyncio.gather(*coros, return_exceptions=True)
    total = 0
    for path, n in zip(paths, counts):
        if isinstance(n, Exception):
            print(f"  ERROR judging {path.name}: {n}")
        else:
            total += int(n or 0)

    action = "Would judge" if dry_run else "Judged"
    print(f"\n{action} {total} conversations.")


def _output_run_dir(cfg: Dict[str, Any], output_dir: str,
                    run_id: Optional[str] = None) -> Path:
    """Per-experiment output directory shared by all post-hoc steps."""
    from ..core.paths import run_dir as _run_dir
    return _run_dir(
        task=cfg["experiment"]["task"],
        assistant_model=cfg["models"]["assistant_model"],
        output_dir=output_dir,
        run_id=run_id,
        flat=bool(cfg.get("experiment", {}).get("flat_output_layout", False)),
    )


def _run_compare(cfg: Dict[str, Any], output_dir: str, dry_run: bool,
                  run_id: Optional[str] = None) -> None:
    """Run the comparison report generator."""
    compare_input_dir = _output_run_dir(cfg, output_dir, run_id=run_id)
    compare_output_dir = compare_input_dir / "experiment_comparison"

    print(f"\n{'='*60}")
    print("Generating comparison report...")
    print(f"  Input dir:  {compare_input_dir}")
    print(f"  Output dir: {compare_output_dir}")
    print(f"{'='*60}")

    if dry_run:
        print("  (dry_run — skipping)")
        return

    # The internal build renders a structured-vs-baseline comparison report here.
    # That reporting tool is development infrastructure and is omitted from this
    # release; per-condition metrics are still written by the aggregate step.
    print("  (comparison report omitted in the public release)")


async def _run_early_stop(cfg: Dict[str, Any], output_dir: str, dry_run: bool,
                          run_id: Optional[str] = None) -> None:
    """Two-step early-stop pipeline:
      1. baseline_termination_judge: writes ``judged_end_turn`` to baseline files.
      2. compute_early_stop_metrics: writes ``metrics_at_early_stop`` and
         ``judged_metrics_at_early_stop`` to every file (structured + baselines).
    """
    run_dir = _output_run_dir(cfg, output_dir, run_id=run_id)
    if not run_dir.exists():
        print(f"  WARNING: Output directory not found, skipping early-stop: {run_dir}")
        return

    print(f"\n{'='*60}")
    print("Computing score-at-end variants (default-on)...")
    print(f"  Input dir: {run_dir}")
    print(f"{'='*60}")

    if dry_run:
        print("  (dry_run — skipping)")
        return

    # Step 1: baseline termination judge.
    bt_cmd = [
        sys.executable, "-m", "simulation.tools.baseline_termination_judge",
        "--input_dir", str(run_dir),
        "--task", cfg["experiment"]["task"],
        "--judge_model", cfg["models"]["judge_model"],
    ]
    print(f"  $ {' '.join(bt_cmd)}")
    bt_result = subprocess.run(bt_cmd)
    if bt_result.returncode != 0:
        print(f"  WARNING: baseline_termination_judge exited with code {bt_result.returncode}")

    # Step 2: compute early-stop metrics for every JSON in the run dir.
    es_cmd = [
        sys.executable, "-m", "simulation.tools.compute_early_stop_metrics",
        "--input_dir", str(run_dir),
        "--judge_model", cfg["models"]["judge_model"],
        "--include_baselines",
        "--all_methods",
    ]
    print(f"  $ {' '.join(es_cmd)}")
    es_result = subprocess.run(es_cmd)
    if es_result.returncode != 0:
        print(f"  WARNING: compute_early_stop_metrics exited with code {es_result.returncode}")


def _run_aggregate_and_dashboard(cfg: Dict[str, Any], output_dir: str, dry_run: bool,
                                  run_id: Optional[str] = None) -> None:
    """Build aggregate-metric CSVs and the multi-tab dashboard once, after
    all conditions complete. Replaces the per-condition rebuild that used to
    run inside runner.py and baseline_runner.py.
    """
    run_dir = _output_run_dir(cfg, output_dir, run_id=run_id)
    if not run_dir.exists():
        print(f"  WARNING: Output directory not found, skipping dashboard: {run_dir}")
        return

    print(f"\n{'='*60}")
    print("Refreshing aggregate metrics + dashboard...")
    print(f"  Input dir: {run_dir}")
    print(f"{'='*60}")

    if dry_run:
        print("  (dry_run — skipping)")
        return

    # Lazy import to keep the orchestrator's startup light.
    from ..tools.aggregate_metrics import aggregate_from_dir

    metrics_dir = run_dir / "metrics"
    aggregate_from_dir(run_dir, metrics_dir)
    print(f"  Metrics report: {metrics_dir}")


async def _run_single_model_pipeline(
    cfg: Dict[str, Any],
    model_entry: Dict[str, str],
    args: argparse.Namespace,
    iu_cache_path: str,
    start_index: int,
    run_id: str = "",
) -> Tuple[str, Path]:
    """Run all conditions (structured + baselines) + post-hoc steps for one assistant model.

    Returns (model_name, output_run_dir).
    """
    cfg = copy.deepcopy(cfg)
    cfg["models"]["assistant_model"] = model_entry["model"]
    cfg["models"]["assistant_llm_provider"] = model_entry["provider"]
    if model_entry.get("gemini_thinking_level"):
        cfg["models"]["assistant_gemini_thinking_level"] = model_entry["gemini_thinking_level"]
    else:
        cfg["models"].pop("assistant_gemini_thinking_level", None)

    model_name = model_entry["model"]
    provider = model_entry["provider"] or cfg["models"]["llm_provider"]

    # When flat_output_layout is set AND there are multiple assistant models
    # (e.g. the model arm grid), inject the model name as a subdir of run_id so
    # each model's outputs land at <output_dir>/<run_id>/<model>/. Without this,
    # all models would write to the same flat <output_dir>/<run_id>/ and
    # clobber each other.
    flat = bool(cfg.get("experiment", {}).get("flat_output_layout", False))
    multi_model = len(cfg.get("models", {}).get("assistant_models", [])) > 1
    effective_run_id = run_id
    if flat and multi_model and run_id:
        effective_run_id = f"{run_id}/{model_name}"

    print(f"\n{'#'*60}")
    print(f"# Model: {model_name} (provider: {provider})")
    if effective_run_id:
        print(f"# Run ID: {effective_run_id}")
    print(f"{'#'*60}")

    # Replace run_id with the model-suffixed version for all subprocess calls
    # and path computations downstream.
    run_id = effective_run_id

    out_dir = _output_run_dir(cfg, args.output_dir, run_id=run_id)

    # Write experiment manifest before running any conditions.
    if run_id and not args.dry_run:
        _write_manifest(cfg, args.config, run_id, out_dir, args)

    commands: List[Dict[str, Any]] = []
    if args.only in ("structured", "all"):
        commands.extend(_build_structured_commands(
            cfg, iu_cache_path=iu_cache_path, start_index=start_index,
            output_dir=args.output_dir, run_id=run_id,
        ))
    if args.only in ("baselines", "all"):
        commands.extend(_build_baseline_commands(
            cfg, start_index=start_index, output_dir=args.output_dir,
            run_id=run_id,
        ))

    # Propagate simulator feature flags from the YAML config into each
    # runner subprocess as a CLI arg (JSON-encoded). Each runner reads
    # ``--simulator_flags`` and installs via ``init_flags()``.
    from simulation.core.feature_flags import SimulatorFeatureFlags
    sim_flags = (
        SimulatorFeatureFlags.from_dict(cfg["simulator_flags"])
        if "simulator_flags" in cfg
        else SimulatorFeatureFlags.from_env()
    )
    sim_flags_json = sim_flags.to_json()
    for entry in commands:
        # Baseline runner CLI doesn't accept --simulator_flags (baselines are
        # pure LLM outputs and don't use cognitive-model flags). Only attach
        # to structured runners.
        if entry.get("type") == "baseline":
            continue
        entry["cmd"].extend(["--simulator_flags", sim_flags_json])

    total = len(commands)
    print(f"\nConditions for {model_name}: {total}")
    for entry in commands:
        _print_command(entry)

    if args.dry_run:
        print(f"\n[dry_run] No commands executed for {model_name}.")
    else:
        succeeded, failed = _run_commands_pool(commands, args.max_parallel)
        print(f"\nCompleted {succeeded}/{total} conditions for {model_name}.")
        if failed:
            print(f"Failed conditions: {failed}")

    if not args.skip_judge:
        await _run_judge(cfg, args.output_dir, args.dry_run, run_id=run_id)

    if not args.skip_early_stop:
        await _run_early_stop(cfg, args.output_dir, args.dry_run, run_id=run_id)

    if not args.skip_compare:
        _run_compare(cfg, args.output_dir, args.dry_run, run_id=run_id)

    if not args.skip_dashboard:
        _run_aggregate_and_dashboard(cfg, args.output_dir, args.dry_run, run_id=run_id)

    return model_name, out_dir


def _run_cross_model_dashboard(
    cfg: Dict[str, Any],
    model_output_dirs: List[Tuple[str, Path]],
    output_dir: str,
    dry_run: bool,
    run_id: Optional[str] = None,
) -> None:
    """Omitted in the public release.

    The internal build renders an interactive cross-model HTML dashboard here.
    Per-model aggregate metrics are still written by
    ``_run_aggregate_and_dashboard`` above.
    """
    print("\n  (cross-model dashboard omitted in the public release)")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiment conditions.")
    parser.add_argument(
        "--config",
        type=str,
        default="simulation/experiments/experiment_config_strategy_arm_new_canonical.yaml",
        help="Path to an experiment YAML config (one of experiment_config_{strategy,model}_arm{,_expertqa}.yaml).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root output directory. Overrides experiment.output_dir from YAML (default: 'output').",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["structured", "baselines", "all"],
        default="all",
        help="Run only structured, only baselines, or all conditions.",
    )
    parser.add_argument(
        "--n_per_condition",
        type=int,
        default=None,
        help="Override n_per_condition from config (e.g. 1 for a quick single-sample test).",
    )
    parser.add_argument(
        "--assistant_model", type=str, default=None,
        help="Override assistant model from config (for multi-model comparison runs).",
    )
    parser.add_argument(
        "--assistant_llm_provider", type=str, default=None,
        help="Override LLM provider for the assistant model (e.g., 'anthropic', 'together'). "
             "User simulator and judge always use the provider from config.",
    )
    parser.add_argument("--skip_judge", action="store_true", help="Skip the post-hoc judge step.")
    parser.add_argument("--skip_early_stop", action="store_true",
                        help="Skip baseline_termination_judge + compute_early_stop_metrics.")
    parser.add_argument("--skip_compare", action="store_true", help="Skip the comparison report step.")
    parser.add_argument("--skip_dashboard", action="store_true",
                        help="Skip aggregate-metrics + dashboard build.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--max_parallel", type=int, default=4,
                        help="Number of conditions to run as concurrent subprocesses. Default 4. "
                             "Set to 1 for the legacy sequential behavior (live-streaming output).")
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Override dataset start row (0-based). If omitted, uses experiment.start_index "
        "or experiment.use_last_n_problems from the YAML.",
    )
    from simulation.knowledge.update_v2 import ABSORPTION_PRESETS as _ABSORPTION_PRESETS
    parser.add_argument(
        "--absorption_mode",
        type=str,
        default=None,
        choices=sorted(_ABSORPTION_PRESETS.keys()),
        help="Override the absorption mode from config. Valid values: "
             "'default' (uniform load + hard cutoff + per-level threshold), "
             "'weightabsorb' (state-weighted load + soft dropoff + universal threshold), "
             "'weightabsorb_attempt' (weightabsorb + user-side attempt_load).",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Reuse an existing run_id (e.g. to add baselines to a prior structured run). "
             "Outputs land in <output_dir>/<task>/<model>/<run_id>/ alongside any existing files.",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    # Resolve output_dir: CLI flag > YAML experiment.output_dir > "output"
    if args.output_dir is None:
        args.output_dir = cfg.get("experiment", {}).get("output_dir", "output")
    if args.n_per_condition is not None:
        cfg["experiment"]["n_per_condition"] = args.n_per_condition
    if args.absorption_mode is not None:
        cfg["experiment"]["absorption_mode"] = args.absorption_mode
    # Store assistant-specific provider separately; llm_provider stays as-is for user/IU models
    cfg["models"]["assistant_llm_provider"] = args.assistant_llm_provider or ""
    start_index = _resolve_experiment_start_index(cfg, args.start_index)

    # Resolve the list of assistant models to iterate over.
    model_list = _resolve_assistant_models(cfg, args.assistant_model, args.assistant_llm_provider)

    # Generate a unique run ID for this experiment invocation, or reuse one passed in.
    absorption_mode = cfg["experiment"].get("absorption_mode", "default")
    run_id = args.run_id or _generate_run_id(absorption_mode)

    exp_name = cfg["experiment"].get("name", "experiment")
    n_info = cfg['experiment'].get('problem_indices') or cfg['experiment'].get('n_per_condition', '?')
    print(f"Experiment: {exp_name}")
    print(f"Run ID: {run_id}")
    print(f"Task: {cfg['experiment']['task']}, N per condition: {n_info}")
    print(f"Assistant model(s): {[m['model'] for m in model_list]}")
    print(f"Dataset start_index: {start_index}")

    # Pre-extract IU graphs once (shared across all models — keyed by iu_model, not assistant_model).
    iu_cache_path = ""
    if args.only in ("structured", "all"):
        iu_cache_path = _run_iu_extraction(cfg, args.output_dir, args.dry_run, start_index=start_index)

    # Run each assistant model sequentially (conditions within each model run in parallel).
    model_output_dirs: List[Tuple[str, Path]] = []
    for model_entry in model_list:
        name, out_dir = await _run_single_model_pipeline(
            cfg, model_entry, args, iu_cache_path, start_index, run_id=run_id,
        )
        model_output_dirs.append((name, out_dir))

    # Cross-model dashboard when multiple models were run.
    if len(model_output_dirs) > 1 and not args.skip_dashboard:
        _run_cross_model_dashboard(
            cfg, model_output_dirs, args.output_dir, args.dry_run, run_id=run_id,
        )


if __name__ == "__main__":
    asyncio.run(main())
