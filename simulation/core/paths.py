"""Canonical output-path resolution.

Centralizes the directory layout under ``output/`` so that tools and
the orchestrator do not have to duplicate the same defaults. New tools
should call into these helpers rather than hard-coding paths.

Layout:
    output/
      <task>/
        <assistant_model>/
          <run JSON files>
          dashboard/
          metrics/
          experiment_comparison/
      iu_cache/
        <cache_key>.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


_TASK_DIR = {"math": "competition_math", "expertqa": "expertqa"}


def task_dir_name(task: str) -> str:
    """Map ``--task`` value (math/expertqa) to its output subdirectory name."""
    return _TASK_DIR.get(task, task)


def output_root(output_dir: str = "output") -> Path:
    """Repo-root-relative output directory; pass-through for already-absolute paths."""
    p = Path(output_dir)
    return p


def run_dir(
    *,
    task: str,
    assistant_model: str,
    output_dir: str = "output",
    run_id: Optional[str] = None,
    flat: bool = False,
) -> Path:
    """Per-experiment output directory.

    Default (legacy) layout: ``<output>/<task_dir>/<assistant_model>/[<run_id>]``.

    With ``flat=True``: ``<output>/[<run_id>]`` — drops both the task and
    assistant-model segments. Used by the per-level / per-participant layouts,
    where the prefix is baked into ``output_dir`` (e.g.
    ``output/competition_math/per_level/strategy_arm``) and the model name is no
    longer load-bearing on the path (the analysis groups by arm and run_id, not
    by model).
    """
    if flat:
        base = output_root(output_dir)
    else:
        base = output_root(output_dir) / task_dir_name(task) / assistant_model
    if run_id:
        return base / run_id
    return base


def dashboard_dir(*, task: str, assistant_model: str, output_dir: str = "output",
                  run_id: Optional[str] = None, flat: bool = False) -> Path:
    return run_dir(task=task, assistant_model=assistant_model, output_dir=output_dir,
                   run_id=run_id, flat=flat) / "dashboard"


def metrics_dir(*, task: str, assistant_model: str, output_dir: str = "output",
                run_id: Optional[str] = None, flat: bool = False) -> Path:
    return run_dir(task=task, assistant_model=assistant_model, output_dir=output_dir,
                   run_id=run_id, flat=flat) / "metrics"


def comparison_dir(*, task: str, assistant_model: str, output_dir: str = "output",
                   run_id: Optional[str] = None, flat: bool = False) -> Path:
    return run_dir(task=task, assistant_model=assistant_model, output_dir=output_dir,
                   run_id=run_id, flat=flat) / "experiment_comparison"


def iu_cache_path(
    *,
    task: str,
    iu_model: str,
    experiment_name: Optional[str] = None,
    output_dir: str = "output",
) -> Path:
    """Pre-extracted IU graph cache path.

    Used by the orchestrator's pre-extraction step and shared across all
    structured conditions. ``experiment_name`` is appended only when
    provided, to allow per-experiment caches.
    """
    cache_root = output_root(output_dir) / "iu_cache"
    parts = [task_dir_name(task), iu_model]
    if experiment_name:
        parts.append(experiment_name)
    return cache_root / ("_".join(parts) + ".json")


def conversation_json_name(
    *,
    version: str,
    method: str,
    knowledge_level: str,
    assistant_strategy: str,
    assistant_model_name: str,
    absorption_mode: str = "default",
) -> str:
    """Standard per-condition output filename.

    The default absorption mode produces no filename suffix, preserving the
    existing layout. Non-default modes append ``_<mode>`` before ``.json`` so
    cross-mode runs land in distinct files even when version / method /
    level / strategy / model are identical.
    """
    suffix = "" if absorption_mode == "default" else f"_{absorption_mode}"
    model_safe = assistant_model_name.replace("/", "_")
    return (
        f"{version}_{method}_{knowledge_level}_{assistant_strategy}"
        f"_{model_safe}{suffix}.json"
    )
