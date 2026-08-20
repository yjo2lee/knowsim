# KnowSim

[![Project page](https://img.shields.io/badge/project-page-1f6feb.svg)](https://yoonjoolee.com/knowsim/)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2608.17150)
[![HuggingFace](https://img.shields.io/badge/🤗-KnowChat_dataset-yellow.svg)](https://huggingface.co/datasets/yjlee36/knowchat-multi-turn-dialogues)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

KnowSim simulates a user whose comprehension is tracked as explicit state, so
you can evaluate an assistant on what the user *learned*, not just on what the
assistant *said*.

Each problem is decomposed into a graph of Information Units (IUs) with
prerequisite edges. The simulated user starts with a knowledge state over that
graph, and every assistant turn updates it. That trajectory yields deterministic
metrics — knowledge gain, delivery calibration, cognitive overload — alongside an
LLM-judged interaction quality score, and it decides when the conversation ends.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # add the API keys for the providers you use
```

## Quick start

The paper's items and their IU graphs ship with the repository, so these run as
written. Pass `--llm_provider` / `--user_model` / `--assistant_model` to use a
backend other than the default.

One conversation on a math problem:

```bash
python -m simulation.runtime.runner \
  --version dynamic-knowledge-state \
  --input_csv data/paper_subset/math.jsonl \
  --iu_cache_path data/iu_cache/math.json \
  --knowledge_level intermediate \
  --dynamic_knowledge_state_init \
  --num_conversations 1
```

On an open-ended expert question:

```bash
python -m simulation.runtime.runner \
  --task expertqa --version dynamic-knowledge-state \
  --expertqa_jsonl data/paper_subset/expertqa.jsonl \
  --iu_cache_path data/iu_cache/expertqa.json \
  --knowledge_level intermediate \
  --dynamic_knowledge_state_init \
  --num_conversations 1
```

A baseline simulator with no knowledge state (`zero-shot`, `zero-shot-cot`,
`zero-shot-cot-user-profile`):

```bash
python -m simulation.baselines.baseline_runner \
  --task math --baseline_method zero-shot-cot \
  --input_csv data/paper_subset/math.jsonl \
  --num_conversations 1
```

A full experiment — every knowledge level × condition, then judging and metric
aggregation. `--dry_run` prints the commands without running them:

```bash
python -m simulation.experiments.run_experiment \
  --config simulation/experiments/experiment_config_strategy_arm_new_canonical.yaml
```

Run experiments one at a time — the simulator and the state update share a
rate-limited backend, and concurrent jobs can exhaust the quota and corrupt a run
silently.

## Data

```
data/paper_subset/{math,expertqa}.jsonl   30 items each
data/iu_cache/{math,expertqa}.json        their IU graphs
data/init_ks_cache.json                   canonical initial knowledge states
data/examples/                            sample outputs
```

Items carry a public id (`math_001`) and a `split`: the benchmarking study used
all 30 per task, the validation study 15 (MathQA) and 20 (ExpertQA) of them.

Reproducing the paper means using these files as they are. The items are the
exact text the runs saw, and both caches are experimental controls — IU
extraction is nondeterministic and its graph sets the denominators for knowledge
gain and delivery calibration, while `init_ks_cache.json` is what makes every
benchmarked assistant start from the same state.

Source datasets, both MIT licensed and redistributed under those terms:
**MATH** ([Hendrycks et al., 2021](https://github.com/hendrycks/math)) and
**ExpertQA** ([Malaviya et al., 2024](https://github.com/chaitanyamalaviya/ExpertQA)).
Please cite them alongside this work.

### Other items

An item file is JSON Lines with `problem` and `solution` (or `question` for
ExpertQA). Give each item an `id` and that id keys its IU graph and its output
records; without one, items are keyed by row position.

```bash
python -m simulation.tools.extract_iu_graphs \
  --input_csv my_items.jsonl --output my_iu_cache.json --iu_model gpt-5.2
```

Pre-extracting is optional — an item with no cached graph is extracted on the
fly and added to the cache. `scripts/fetch_math.py` downloads the full MATH
release in the input format.

### Human study

The conversations and survey responses the simulator was validated against are
released separately as
[KnowChat](https://huggingface.co/datasets/yjlee36/knowchat-multi-turn-dialogues)
(CC BY-NC 4.0).

## Output

One JSON file per condition, holding a list of conversation records. Each record
carries the conversation plus the state trajectory behind it:
`knowledge_state_history`, `iu_analysis_history` (the per-IU signals each update
came from), `turn_metrics_history`, aggregate `metrics`, and `stop_reason`. See
`data/examples/` for the shape.

`--disable_early_stop` runs to `max_turns` and records where early stopping
*would* have fired, for evaluating termination rules after the fact.

## Knowledge state

Concepts move through `unaware` → `struggling` → `partial_understanding` →
`knows_well`, constrained by the graph's prerequisite edges.
`simulation/knowledge/update_v2.py` applies the transitions after each assistant
turn: a rule-based phase, then one LLM call that extracts per-IU comprehension
signals. Per-level starting distributions are in
`simulation/knowledge/iu_init.py`.

## Repository layout

```
simulation/
├── core/           multi-provider async LLM client, prompts, feature flags
├── knowledge/      IU extraction, state initialization, update rules, metrics
├── runtime/        CLI entry point, conversation loop, assistant strategies
├── baselines/      zero-shot / CoT / CoT+profile simulators
├── benchmarking/   assistant benchmarking runner and tables
├── experiments/    YAML orchestration and the configs behind the reported runs
├── prompts/        simulator, knowledge-state and judge prompts
└── tools/          IU pre-extraction, judging, metric aggregation
scripts/
└── fetch_math.py   optional: the full MATH release, for other items
```

[PROVENANCE.md](PROVENANCE.md) records where each experiment config came from.

## Citation

```bibtex
@article{lee2026knowsim,
  title         = {KnowSim: Evaluating Information Calibration in LLM Assistants
                   with User Simulators that Learn},
  author        = {Lee, Yoonjoo and Jin, Hyoungwook and Kim, Tae Soo and
                   Zhang, Shaoyang and Laban, Philippe and Liao, Q. Vera},
  year          = {2026},
  eprint        = {2608.17150},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
}
```

## License

MIT — see [LICENSE](LICENSE). KnowChat is CC BY-NC 4.0; the source datasets keep
their own MIT licenses.
