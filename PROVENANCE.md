# Provenance of the paper-final experiment configurations

The experiment configs under `simulation/experiments/` that correspond to the
runs reported in the paper are **not** hand-written. Each was recovered from the
`experiment_manifest.json` the orchestrator wrote at run time, so the published
config is the one the run actually used.

Which runs are "paper-final" is fixed by the canonical-results command in the
analysis runbook; the eight source manifests below are the runs behind it.

Three classes of edit are applied, none of which changes runtime behaviour:

1. **Runtime-injected model fields** — `models.assistant_model`,
   `models.assistant_llm_provider`, `models.assistant_gemini_thinking_level`.
   The orchestrator writes these per assistant model from
   `models.assistant_models`; they are not authored in the config.

2. **Flags for features the code no longer defines** —
   `simulator_flags.ks_update.{paq_enabled, priming_discount, priming_gate}`.
   All were set to their disabled values in the runs, and
   `SimulatorFeatureFlags.from_dict()` ignores keys it does not define, so the
   parsed flag object is identical with or without them.

3. **Path rewrites.** Inputs this repository does not redistribute (the two
   datasets, and the participant-derived interaction-style profiles) point at
   `data/` or are left empty; `experiment.output_dir` is dropped so the runner's
   default `output/` applies. The IU graph caches used by the paper runs *are*
   included, under `data/iu_cache/` — re-extracting them would produce different
   graphs and therefore different KG and DC denominators.

The IU graph caches under `data/iu_cache/` are kept as two separate pairs. The
benchmarking study covers 30 items per task; the validation study covers 15
(MathQA) and 20 (ExpertQA). The benchmarking cache is a superset and the graphs
for the shared items are identical, but the two must not be merged or swapped:
pointing the benchmarking configs at the validation cache would leave half the
items without a graph. `data/init_ks_cache.json` is likewise an experimental
control rather than a speed cache — it is what makes every benchmarked
assistant start from the same initial knowledge state.

The hash below is taken over the source record *before* edits (2) and (3), and
after edit (1) for the recovered configs. It is reproducible from this
repository's configs and identifies which record each file came from.

| Config | Used by | Source | source sha256[:12] | dropped flags |
|---|---|---|---|---|
| `experiment_config_benchmarking_expertqa.yaml` | Assistant benchmarking | authored config | `5321f7797f7c` | — |
| `experiment_config_benchmarking_math.yaml` | Assistant benchmarking | authored config | `a224eebd3c27` | — |
| `experiment_config_model_arm_expertqa.yaml` | ExpertQA-Model (baselines) | run manifest | `89e094ef5a40` | — |
| `experiment_config_model_arm_expertqa_modified_beta.yaml` | ExpertQA-Model (structured) | run manifest | `07ecd83c270c` | — |
| `experiment_config_model_arm_new_canonical_claude_only.yaml` | MathQA-Model (baselines)<br>MathQA-Model (structured) | run manifest | `c83e8db367be` | paq_enabled, priming_discount, priming_gate |
| `experiment_config_model_arm_new_canonical_gemini_only.yaml` | MathQA-Model (baselines)<br>MathQA-Model (structured) | run manifest | `c11a0717c444` | paq_enabled, priming_discount, priming_gate |
| `experiment_config_model_arm_new_canonical_gpt54_only.yaml` | MathQA-Model (baselines)<br>MathQA-Model (structured) | run manifest | `668c7fc380ae` | paq_enabled, priming_discount, priming_gate |
| `experiment_config_strategy_arm_expertqa_baseline_regen.yaml` | ExpertQA-Strategy (baselines) | run manifest | `0a2d8296b6ac` | — |
| `experiment_config_strategy_arm_expertqa_modified_beta.yaml` | ExpertQA-Strategy (structured) | run manifest | `5eb638c3f763` | — |
| `experiment_config_strategy_arm_new_canonical.yaml` | MathQA-Strategy (baselines)<br>MathQA-Strategy (structured) | run manifest | `db471993cfc1` | paq_enabled, priming_discount, priming_gate |
