# Quickstart

This guide takes you from a fresh clone to a first **bilevel fishery experiment** using MetaMARL's YAML command-line interface. It also explains how to inspect results and modify a configuration.

The fishery benchmark is a theoretical common-pool resource model, not a calibrated real-world fishery. **Freshwater and hydrological benchmarks are not included in this release**; they are under development on `dev`.

## 1. Install and validate

Install [uv](https://docs.astral.sh/uv/) and Python 3.12, then run:

```bash
git clone https://github.com/nadinemgh/MetaMARL.git
cd MetaMARL
uv sync --group dev
WANDB_MODE=offline uv run metamarl check examples/bilevel_fishery/config.yaml
```

The project requires Python `>=3.12,<3.13`. `uv sync --group dev` installs its runtime dependencies and the developer tools, including pytest and Ruff.

The `check` command **constructs and validates** the YAML experiment without training. It should report successful configuration validation. If a YAML object or import is invalid, the configuration loader reports the affected path.

You can also invoke the CLI as a Python module:

```bash
WANDB_MODE=offline uv run python -m core.config.cli check examples/bilevel_fishery/config.yaml
```

The `PYTHONHASHSEED` initialization used by the CLI helps standardize Python hash behavior across runs. It is only one part of reproducibility; identical runs are not guaranteed across distributed configurations or different hardware.

## 2. Run the fishery experiment

```bash
WANDB_MODE=offline uv run metamarl run examples/bilevel_fishery/config.yaml
```

The command runs the experiment described by the YAML file. **The checked-in configuration may specify a substantial training budget**, so for a short local smoke run, make a copy and reduce the relevant settings before launching it:

```bash
cp examples/bilevel_fishery/config.yaml examples/bilevel_fishery/config.local.yaml
```

Edit `config.local.yaml` to reduce the outer ES generations, inner PPO/APPO training iterations, agent population, and episode horizon as appropriate for the actual keys and builder calls in that file. Keep the number of sampled candidates, seeds, and environment slots consistent with the configured optimizer. Validate the copy first:

```bash
WANDB_MODE=offline uv run metamarl check examples/bilevel_fishery/config.local.yaml
WANDB_MODE=offline uv run metamarl run examples/bilevel_fishery/config.local.yaml
```

Do not commit your local experiment configuration or generated outputs unless they are intended as reproducible examples.

### What happens during a run?

In the bilevel fishery example:

1. The outer **Evolution Strategies (ES)** optimizer proposes candidate mechanisms.
2. The regulator publishes these candidates to the shared `World`.
3. Fishers learn with the configured RLlib optimizer (PPO or APPO) under their assigned mechanisms.
4. The inner optimizer evaluates the trained policies and collects episode metrics.
5. The regulator aggregates fishery outcomes, scores the candidates, and returns fitness to ES.
6. ES updates its search distribution and repeats according to the configured budget or stopping criterion.

A candidate is installed in a follower environment for its assigned slot; after a published mechanism is assigned, ordinary episode resets **do not replace it**. This is important when comparing repeated training episodes under a fixed candidate.

The fishery objective combines **realized harvest relative to maximum sustainable yield** with a weighted measure of **normalized fish biomass**. The weight, sustainability threshold, and metric aggregation window are configurable. See `examples/bilevel_fishery/regulator_env.py` for the actual scoring implementation.

## 3. Read the results

MetaMARL supports three reporter backends:

| Reporter | Use |
| --- | --- |
| CSV | Export structured observations and metrics for plotting or post-processing |
| TensorBoard | Inspect scalar learning and optimization curves locally |
| Weights & Biases | Track runs, dashboards, and comparisons; supports offline mode |

The selected reporter and its output directory are defined **in YAML**. The old `--reporter` option from `examples/bilevel_fishery/debug.py` is no longer part of the execution workflow.

For CSV reporting, the default output directory is `results/`, organized by the configured project, world, reporter label, and query title. Exported data typically includes a query label, x-coordinate, series, value, and optional error or color fields. The exact files depend on the configured queries.

For offline W&B runs, `WANDB_MODE=offline` writes local W&B data instead of requiring online synchronization.

Metrics and visualizations are defined separately from optimization in `core/metrics/` and `core/reporting/`. The fishery's reporting queries are defined with its example configuration and reporting code. Not every query can be rendered identically by every backend; in particular, a scatter plot may not map directly to a TensorBoard scalar.

## 4. Change the experiment

The YAML file declares the optimizer, environment, agent population, mechanisms, evaluation procedure, and reporting configuration. MetaMARL's configuration loader recognizes:

| YAML key | Meaning |
| --- | --- |
| `_target_` | Construct an object from an importable Python target |
| `_args_` | Supply positional arguments to that target |
| `_calls_` | Apply configured builder methods in order |
| `_symbol_` | Import a symbol without constructing it |
| `_tuple_` | Construct a tuple |

The experiment definition and optional `run` section determine what is built and which methods are invoked during execution.

For example, the fishery can include a quota mechanism that targets the fishers' harvest actions:

```yaml
_target_: core.mechanism.algorithms.quota.Quota
id: quota
acts_on:
  _tuple_: [fisherman, harvest]
obs_map:
  resource_level: fish
```

This fragment illustrates the quota's role; its full construction also depends on the surrounding agent and mechanism configuration. Consult the checked-in YAML for a complete runnable definition.

The CLI also accepts **configuration overrides** after the YAML path:

```text
uv run metamarl run examples/bilevel_fishery/config.yaml path.to.setting=value
```

`path.to.setting=value` is a syntax illustration, **not a literal key in the example configuration**. Use an actual path present in your YAML and validate it with `metamarl check` before training.

## 5. Validate your changes

Run the test suite and lint checks from the repository root:

```bash
uv run python -m pytest
uv run ruff check --no-fix .
uv run ruff format --check .
```

For a faster test selection that excludes the integration and notebook markers:

```bash
uv run python -m pytest -m "not integration and not notebook"
```

Some tests are skipped for tracked limitations, with stable IDs such as `OBS-001`, `MET-001`, `MET-002`, and `APPO-001`. These are documented test gaps, not evidence that every corresponding edge case works.

## 6. Next steps

- Explore the CartPole example under `examples/cartpole/` for a simpler end-to-end learning and reporting pipeline. Use its available YAML configuration, rather than the removed `debug.py` entry point.
- Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to learn how the optimizer, `World`, RLlib environment adapter, mechanisms, and metric schemas interact.
- Open the notebooks in [`tutorials/`](tutorials/) to study mechanism construction, custom environments, and reporting.
- Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a new benchmark or algorithm adapter.

The project is released under [Apache License 2.0](LICENSE). Please use [CITATION.cff](CITATION.cff) when citing MetaMARL in research.
