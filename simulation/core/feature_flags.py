"""Typed feature flags for the simulator pipeline.

Single source of truth for behavioral knobs the user can configure per run.
Grouped by pipeline stage so flags are discoverable and the relationships
between them are explicit.

Three ways to set flags, in order of preference for new code:

1. **YAML config** — under a top-level ``simulator_flags:`` section of an
   experiment YAML. Read by the experiment orchestrator
   (``simulation/experiments/run_experiment.py``) and converted to env vars
   passed to subprocess runners.

2. **Environment variable** — ``MYSIMARE_FLAGS`` set to a JSON dict matching
   the dataclass shape. Read by ``SimulatorFeatureFlags.from_env()``.
   Useful for one-off CLI runs without editing the YAML.

3. **Legacy individual env vars** (back-compat) — ``MYSIMARE_GATE5_PARTIAL=1``,
   ``MYSIMARE_RULE5_OFF=1``, ``MYSIMARE_TIGHTENED_USER_SIM=1``. Still read
   by ``from_env()`` as a fallback. Existing scripts continue to work.

Adding a new feature flag:

1. Add a typed field to the appropriate ``*Flags`` dataclass below.
2. Read it via ``get_flags().<stage>.<field>`` at the call site.
3. Document the new field's enum/bool semantics in its docstring.
4. The YAML config schema picks it up automatically.

Removing a feature flag:

1. Delete the field from the dataclass and all call-site reads.
2. Remove any legacy env-var fallback in ``from_env()``.
3. Search the repo (configs, docs) for stale references.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Literal, Optional




# ---------------------------------------------------------------------------
# Per-stage flag groups
# ---------------------------------------------------------------------------

Rule5Mode = Literal["off", "struggling_gate", "partial_gate"]


@dataclass
class KnowledgeStateUpdateFlags:
    """Flags controlling ``simulation/knowledge/update_v2.py``'s per-turn
    5-rule pipeline (the state-machine that decides how each IU advances
    given assistant teaching + user attempt + overload state).
    """

    rule5: Rule5Mode = "struggling_gate"
    """When Rule 5 (user reasoning attempt → ``knows_well`` jump) is allowed
    to fire:

    - ``"off"`` — Rule 5 is disabled entirely; reasoning attempts fall
      through to Rule 4 (teaching_quality path).
    - ``"struggling_gate"`` — legacy GATE5. Rule 5 fires when the user's
      prior state on this IU is at least ``struggling`` (i.e. they've been
      introduced to the concept).
    - ``"partial_gate"`` — GATE5+ (recommended branch default). Rule 5
      fires only at ``partial_understanding+``. Pedagogically rationalized
      as: a user can't fluently reason about a concept they're still merely
      struggling with.

    Has no effect when ``minimal_rules=true`` (Rule 5 disabled in that mode).
    """

    fractional_accumulation: bool = False
    """Whether to use per-IU fractional progress accumulation instead of
    per-turn integer rounding in the dropoff step.

    - ``False`` (default) — current behavior:
      ``actual_steps = max(0, round(intended_steps × efficiency))``.
      Single-step intended advances with efficiency < 0.5 round to 0,
      creating the "T1 stall" rounding cliff.
    - ``True`` — per-IU progress accumulator carries fractional progress
      across turns:
      ``progress[iu] += intended × efficiency; actual_steps = int(progress[iu]);
       progress[iu] -= actual_steps``.
      Used by ``pilot3_20260515``.
    """

    minimal_rules: bool = False
    """Whether to use a simplified Phase-A rule set in ``_compute_new_state``.

    - ``False`` (default) — full Rule 1-5 pipeline (ceiling + overload-cap +
      monotonicity + teaching_quality + reasoning_jump + ALT-A cap).
    - ``True`` — keep only Rule 1 (ceiling) + Rule 3 (monotonicity) +
      simplified Rule 4 (well_explained → +2, shallow → +1, not_mentioned →
      no change). Drop Rule 2 (overload cap; redundant with dropoff
      efficiency), Rule 5 (user reasoning jump; reasoning falls through to
      teaching_quality path), and ALT-A (unaware → +1 cap removed; the
      ALT-A interaction with rounding created the T1-stall pathology).
      Used by ``pilot3_20260515``.
    """

    cap_well_explained_at_one: bool = False
    """pilot4: in the ``minimal_rules`` branch, cap well_explained advance
    at +1 instead of +2. Halves comp's per-mention amplitude advantage.
    Only active when ``minimal_rules=true``.
    """

    shallow_unaware_only: bool = False
    """pilot4: in the ``minimal_rules`` branch, restrict shallow mentions to
    only advance IUs from ``unaware → struggling`` (no advance on
    struggling/partial_understanding). Motivated by the §9.11 novice
    diagnostic where adaptive's many shallow mentions on already-struggling
    IUs produced 0 transitions due to prereq ceiling locks; under this rule
    shallow becomes a "prereq-unlock" signal rather than a generic
    progress signal. Only active when ``minimal_rules=true``.
    """

    reset_fractional_on_transition: bool = False
    """pilot4: when ``fractional_accumulation=true``, reset an IU's carried
    fractional progress to 0 whenever it actually advances state this turn
    (instead of persisting the remainder into the next state's quota).
    Only active when ``fractional_accumulation=true``.
    """

    sigmoid_softening: float = 1.0
    """pilot5: widening factor for the sigmoid dropoff curve. The curve is
    ``efficiency = 1 / (1 + ((eff_load - thr) / (thr * k))²)`` where k is
    this factor.

    - ``1.0`` (default, pilot1+ behaviour): standard sigmoid. At
      ``eff_load = 2·thr`` → efficiency 0.50.
    - ``1.5``: at ``2·thr`` → ~0.69.
    - ``2.0``: at ``2·thr`` → ~0.80, at ``3·thr`` → ~0.50. Softer falloff
      means high-coverage turns lose less of their per-IU intended steps
      to dropoff, allowing them to compound more reliably across turns.
      Only takes effect with ``absorption_mode=weightabsorb_attempt_sigmoid``.
    """

    per_iu_dropoff: bool = False
    """pilot7: replace turn-uniform sigmoid efficiency with per-IU graduated
    dropoff, stacked with the existing turn-level sigmoid as a global
    fatigue layer.

    - ``False`` (default): existing behaviour — all mentioned IUs in an
      overloaded turn get the same ``absorption_efficiency`` from a single
      sigmoid call on total effective_load.
    - ``True``: each IU gets ``per_iu_eff × turn_eff`` where both are the
      *same* sigmoid formula (with existing ``sigmoid_softening`` and
      ``overload_threshold``):
        * Mentioned IUs are sorted by W_STATE descending (priority queue:
          high-W IUs absorb first).
        * Walking that priority order, each IU's ``per_iu_eff`` is the
          sigmoid applied to the cumulative comp_load up to and including
          that IU. Early IUs (cum ≤ thr) get per_iu_eff = 1.0; later IUs
          get the sigmoid penalty proportional to how far past threshold
          their cumulative position lands.
        * ``turn_eff`` is the existing sigmoid on total effective_load
          (global fatigue layer).
        * Combined as multiplicative stack ``per_iu_eff × turn_eff``.

    Effect: in non-overloaded turns nothing changes (both factors = 1.0).
    In modestly overloaded turns the early IUs are unaffected and only
    the tail degrades. In heavily overloaded turns the global fatigue
    multiplier reduces every IU and the tail degrades doubly.

    Reuses existing ``overload_threshold`` and ``sigmoid_softening``
    parameters — no new hyperparameters. Only effective when
    ``absorption_mode`` uses dropoff (e.g.
    ``weightabsorb_attempt_sigmoid``).
    """

    per_iu_hinge: bool = False
    """pilot11: within-turn hinge mechanism. Of the mentioned IUs in a
    turn, sorted by W_STATE descending (priority order — most-needy
    first), the top-N get ``per_iu_eff = 1.0`` and the rest get
    ``per_iu_eff = hinge_tail_factor``. Stacks multiplicatively with
    ``turn_eff`` like ``per_iu_dropoff``.

    Distinct from ``per_iu_dropoff`` which applies a smooth sigmoid
    along the cumulative-cost walk; ``per_iu_hinge`` is a hard cutoff
    after N IUs.

    N is computed as ``max(1, round(hinge_threshold_ratio * graph_size))``
    so the breadth cap scales with the IU graph size of the problem.

    The 20260515 lever audit identified ``hinge_threshold_ratio=0.143``
    (≈ N=2 at the canonical 14-IU graph size) and
    ``hinge_tail_factor=0.5`` as the strongest single-mechanism
    candidate for KG novice (replay predicts strat KG novice 5/9 → 8/9
    while preserving model KG).

    Mutually exclusive with ``per_iu_dropoff`` in practice; if both are
    True, hinge takes precedence.
    """

    hinge_threshold_ratio: float = 0.143
    """pilot11 hinge: scaling factor for the breadth cap. N (top IUs that
    get full credit each turn) = ``max(1, round(hinge_threshold_ratio *
    n_graph_nodes))``. Default 0.143 yields N=2 at the typical 14-IU
    graph size of our pilot problems; smaller graphs get N=1 and
    larger graphs (20+ IUs) scale up proportionally.
    """

    hinge_tail_factor: float = 0.5
    """pilot11 hinge: efficiency for IUs past the top-N priority cutoff.
    0.0 = full block; 1.0 = no penalty (mechanism off). 0.5 = the
    20260515 audit's recommended midpoint.
    """


UserSimPromptVariant = Literal["canonical", "tightened"]


@dataclass
class UserSimulatorFlags:
    """Flags controlling the user-simulator behaviour (which prompt file
    set the runner loads in ``_load_prompt_pair``).
    """

    prompt_variant: UserSimPromptVariant = "canonical"
    """Which user-simulator prompt template family to load:

    - ``"canonical"`` — the default ``dynamic-knowledge-state.txt`` and
      ``dynamic-knowledge-state-initial-query.txt``.
    - ``"tightened"`` — adds three HARD CONSTRAINTS above the canonical
      rule text: HC-1 forbids using unaware concepts' mechanics, HC-2
      requires visible errors/hesitation on struggling concepts, HC-3
      requires an explicit confusion-lexicon marker. Files are loaded as
      ``{version}-tightened.txt`` and ``{version}-tightened-initial-query.txt``.
    """

    allow_terminate: bool = True
    """Global fallback: whether the user simulator's ``terminate=true``
    signal is honored as a virtual_early_stop trigger. Used as the default
    for BOTH the baseline and structured pipelines when the more specific
    ``baseline_allow_terminate`` / ``structured_allow_terminate`` overrides
    are unset (``None``).

    - ``True`` (default) — user-sim's ``terminate=true`` ends the
      conversation; runner records
      ``virtual_early_stop_reason='user_requested_terminate'``.
    - ``False`` — terminate signals ignored (used by ``pilot1_20260514``
      and the new_pilot* family).

    The split into pipeline-specific overrides (added 2026-05-17) lets
    new_canonical adopt different terminate policies per pipeline:
    baselines respect terminate (mimics May-13-style behaviour) while the
    structured pipeline ignores it (keeps new_pilot2 / new_pilot5 behaviour
    where conversations run to ``max_turns`` for cleaner ES-window
    measurement).
    """

    baseline_allow_terminate: Optional[bool] = None
    """Pipeline-specific override for the baseline runner
    (``run_conversation_batch``). ``None`` means inherit from
    ``allow_terminate``. Set ``True`` in new_canonical so baseline
    user-sim's ``terminate=true`` signal is honored (matches May 13
    canonical baseline behaviour).
    """

    structured_allow_terminate: Optional[bool] = None
    """Pipeline-specific override for the structured runner
    (``run_conversation_with_interaction_profile``). ``None`` means
    inherit from ``allow_terminate``. Set ``False`` in new_canonical so
    the structured pipeline ignores ``terminate=true`` and conversations
    run to ``max_turns`` (matches new_pilot2 / new_pilot5 behaviour where
    we want a clean ES-window measurement).
    """

    universal_engagement_guidance: bool = False
    """Whether to replace the density-conditioned ``{articulation_guidance}``
    block (3-branch LOW/MEDIUM/HIGH text) with a single universal text that
    reinforces IU-state-honest engagement without level conditioning.

    - ``False`` (default) — density-branched guidance from
      ``_get_articulation_guidance`` in ``conversation.py``.
    - ``True`` — fixed universal text regardless of density. Used by
      ``pilot2_20260514``. ``{density_label}`` and ``{density_percent}``
      (informational status line) are unaffected.
    """


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class IUInitFlags:
    """Flags controlling ``simulation/knowledge/iu_init.py``'s knowledge-state
    initialization at the start of a conversation.
    """

    include_struggling: bool = False
    """Whether to assign ``struggling`` as an explicit initial state for a
    fraction of IUs.

    - ``False`` (default) — 3-state init (known / partial / unaware). The
      ``runner.py`` post-processing maps some ``partial_known`` IUs to
      ``struggling`` based on a prereq-ratio heuristic, but no IUs are
      explicitly assigned to a struggling pool.
    - ``True`` — 4-state init with explicit ratios per level (see
      ``_IU_INIT_RATIOS_4STATE`` in ``iu_init.py``). Used by
      ``pilot3_20260515`` to give dense-teaching models a head start with
      "concept introduced but foundation not solid" IUs.
    """


@dataclass
class SimulatorFeatureFlags:
    """All run-configurable behavioural flags for the simulator pipeline."""

    ks_update: KnowledgeStateUpdateFlags = field(default_factory=KnowledgeStateUpdateFlags)
    user_sim: UserSimulatorFlags = field(default_factory=UserSimulatorFlags)
    iu_init: IUInitFlags = field(default_factory=IUInitFlags)

    # ------------------------------------------------------------------ loaders

    @classmethod
    def from_env(cls) -> "SimulatorFeatureFlags":
        """Build from environment variables.

        Priority order:
        1. ``MYSIMARE_FLAGS`` (JSON dict) — preferred for new code.
        2. Legacy individual env vars (``MYSIMARE_GATE5_PARTIAL``,
           ``MYSIMARE_RULE5_OFF``, ``MYSIMARE_TIGHTENED_USER_SIM``).
        3. Defaults from the dataclass.
        """
        # Preferred path: JSON-encoded MYSIMARE_FLAGS
        raw = os.environ.get("MYSIMARE_FLAGS")
        if raw:
            try:
                return cls.from_dict(json.loads(raw))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # fall through to legacy path

        # Legacy path: individual env vars
        ks = KnowledgeStateUpdateFlags()
        if os.environ.get("MYSIMARE_RULE5_OFF") == "1":
            ks.rule5 = "off"
        elif os.environ.get("MYSIMARE_GATE5_PARTIAL") == "1":
            ks.rule5 = "partial_gate"
        us = UserSimulatorFlags()
        if os.environ.get("MYSIMARE_TIGHTENED_USER_SIM") == "1":
            us.prompt_variant = "tightened"
        return cls(ks_update=ks, user_sim=us)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SimulatorFeatureFlags":
        """Build from a plain dict (e.g. parsed YAML section).

        Unknown keys in either sub-dict are silently ignored so that JSON
        encoded by a different version of the code (e.g. with a flag that
        has since been removed) still loads cleanly.
        """
        if not data:
            return cls()
        ks_known = {f.name for f in fields(KnowledgeStateUpdateFlags)}
        us_known = {f.name for f in fields(UserSimulatorFlags)}
        ii_known = {f.name for f in fields(IUInitFlags)}
        ks_data = {k: v for k, v in (data.get("ks_update") or {}).items() if k in ks_known}
        us_data = {k: v for k, v in (data.get("user_sim") or {}).items() if k in us_known}
        ii_data = {k: v for k, v in (data.get("iu_init") or {}).items() if k in ii_known}
        return cls(
            ks_update=KnowledgeStateUpdateFlags(**ks_data),
            user_sim=UserSimulatorFlags(**us_data),
            iu_init=IUInitFlags(**ii_data),
        )

    # ------------------------------------------------------------------ writers

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe; for manifest recording)."""
        return asdict(self)

    def to_json(self) -> str:
        """Encode as a sorted JSON string. Used by the experiment
        orchestrator to pass the resolved flag config to each runner
        subprocess via the ``--simulator_flags`` CLI arg."""
        return json.dumps(self.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------
#
# Use ``get_flags()`` from any pipeline-stage module to read the active flag
# configuration. ``init_flags()`` is called once at runner startup; if it's
# not called, ``get_flags()`` returns ``SimulatorFeatureFlags.from_env()``
# so that ad-hoc CLI invocations still work without orchestrator setup.

_active_flags: Optional[SimulatorFeatureFlags] = None


def init_flags(flags: SimulatorFeatureFlags) -> None:
    """Install the active flag configuration. Call once at runner startup
    after parsing yaml / env / CLI."""
    global _active_flags
    _active_flags = flags


def get_flags() -> SimulatorFeatureFlags:
    """Return the active flag configuration. Falls back to ``from_env()`` if
    ``init_flags`` was never called (so direct module-level imports of the
    pipeline still work)."""
    if _active_flags is None:
        return SimulatorFeatureFlags.from_env()
    return _active_flags
