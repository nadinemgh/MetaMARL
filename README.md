# MetaMARL

**MetaMARL** is an extensible, algorithm-agnostic research framework for **multi-agent learning, mechanism design, and the evaluation of interventions in interacting dynamical systems**.

It supports experiments in which multiple adaptive agents interact through a shared environment, with or without a separate regulator or outer optimization loop. Researchers can define agent behaviors, mechanisms, environment dynamics, and evaluation objectives independently, then combine them to study how local incentives and interventions affect collective outcomes.

MetaMARL is designed for **both multi-agent experimentation and bilevel optimization**; optimizing a mechanism is an optional experimental setup, not a requirement of the framework.

## What MetaMARL provides

- **Multi-agent environments:** Define heterogeneous agents, observations, actions, rewards, and shared state transitions. Agents can interact within the same Markov decision process (MDP), with their actions jointly influencing the next state.
- **Composable mechanisms:** Define and combine regulatory or behavioral mechanisms independently of the environment and learning algorithm. Mechanisms can affect what agents observe, how their actions change the environment, or the incentives they receive.
- **Optional bilevel optimization:** Evaluate candidate mechanisms against populations of learning agents. A regulator can optimize a mechanism based on the behavior and system-level outcomes that emerge after agent adaptation.
- **Interchangeable optimization components:** Keep the environment, agent-learning algorithm, and mechanism optimizer separate. The current release includes Evolution Strategies (ES) for outer-loop optimization and RLlib-based PPO and APPO for agent learning; other solvers can be integrated through the framework's abstractions.
- **Experiment coordination:** Use the shared `World` abstraction to coordinate mechanism candidates, training and evaluation contexts, and feedback between optimization levels.
- **Reproducible configuration and reporting:** Specify experiments in YAML, override configuration values at the command line, collect structured metrics, and export results through CSV, TensorBoard, or Weights & Biases (W&B).

A common bilevel experiment consists of a **regulator (leader)** proposing a rule, such as a fishing quota or incentive, and a population of **agents (followers)** adapting their policies under that rule. The regulator scores the resulting outcomes and uses them to evaluate or improve the mechanism. The same environment and agent abstractions can also be used without an outer regulator, for example to study multi-agent learning under a fixed mechanism.

The core framework lives in `core/` and is independent of any particular environmental model.

## Included examples

This release includes **two examples**:

| Example | Purpose | What it demonstrates |
| --- | --- | --- |
| [`examples/bilevel_fishery/`](examples/bilevel_fishery/) | Common-pool resource management | Learning fishers interacting with a renewable resource under configurable mechanisms, including harvesting constraints and incentives; optional optimization of regulatory parameters |
| [`examples/cartpole/`](examples/cartpole/) | End-to-end pipeline validation | PPO/APPO training and the bilevel execution and reporting pipeline in a simple environment with a constant outer fitness objective |

### Fishery benchmark

The fishery example models interacting agents harvesting from a renewable fish stock. Individual agents seek rewards from resource use, while their combined actions affect the stock available in subsequent steps. Mechanisms can modify their incentives and constraints, making the environment useful for investigating strategic adaptation, resource sustainability, and the effects of regulatory interventions.

The benchmark is a **theoretical proof of concept**, not a calibrated representation of a particular real-world fishery. It is intended to exercise the framework and enable controlled experiments.

### CartPole example

The CartPole example provides a compact way to exercise the learning, optimization, and reporting infrastructure. Its outer objective is deliberately constant; it is a pipeline-validation example rather than a substantive mechanism-design benchmark.

### Freshwater development

**Freshwater and hydrological applications are not included in this release.** A water-management benchmark is under active development on the `dev` branch, including work on hydrological simulation and interacting water users. Its interfaces and experimental setup are not part of the released API or supported examples described here.

## Architecture

MetaMARL separates domain-specific dynamics from learning and optimization infrastructure:

| Component | Responsibility |
| --- | --- |
| `core/agents/` | Agent abstractions and configurations |
| `core/mechanism/` | Composable mechanisms and their effects on observations, actions, and incentives |
| `core/envs/` | Environment interfaces, shared MDP state, transition and reset hooks, and regulator/follower environments |
| `core/optimizers/` | Agent-learning and outer-optimization configurations, including bilevel composition and ES |
| `core/adaptors/ray/` | Ray/RLlib integration, policy and environment adapters, and distributed execution |
| `core/world/` | Coordination of mechanism candidates, runs, and evaluation contexts |
| `core/metrics/` | Structured metric schemas, collection, and aggregation |
| `core/reporting/` | CSV, TensorBoard, and W&B reporting backends |
| `core/config/` | YAML configuration loading, command-line execution, and reproducibility utilities |

The framework supports **shared or independent policies**, depending on the experimental configuration. Mechanism design and agent training are intentionally decoupled: a mechanism can be evaluated under a fixed population, while bilevel experiments can retrain or evaluate agents under candidate rules.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for more detail on the execution model.

## Installation

MetaMARL uses **Python 3.12** (`>=3.12,<3.13`) and [uv](https://docs.astral.sh/uv/) for dependency management.

From the repository root:

```bash
uv sync --group dev
```

This installs the project's runtime dependencies and developer tools, including pytest, coverage, Ruff, and notebook tooling. The runtime includes Ray/RLlib, PyTorch, and scientific Python dependencies. Reporter availability depends on the selected dependencies and configuration.

## Running an experiment

Experiments are described in YAML and launched through the `metamarl` CLI. The old per-example `debug.py` entry points are **no longer part of the workflow**.

### Validate a configuration

```bash
WANDB_MODE=offline uv run metamarl check examples/bilevel_fishery/config.yaml
```

`check` constructs and validates the configuration without starting training.

### Run the fishery example

```bash
WANDB_MODE=offline uv run metamarl run examples/bilevel_fishery/config.yaml
```

The CLI is also available through its Python module:

```bash
WANDB_MODE=offline uv run python -m core.config.cli run examples/bilevel_fishery/config.yaml
```

### Run the CartPole example

```bash
WANDB_MODE=offline uv run metamarl run examples/cartpole/config.yaml
```

### Override YAML settings

The CLI accepts command-line configuration overrides in `path.to.parameter=value` form, after the YAML filename. For example, to change an experiment setting, use:

```bash
uv run metamarl run examples/bilevel_fishery/config.yaml path.to.parameter=value
```

Here `path.to.parameter` is a placeholder: replace it with an actual key path from the selected YAML configuration. This avoids changing the configuration file for every experiment variation.

Setting `WANDB_MODE=offline` prevents W&B from requiring an online account and stores W&B run data locally. Select a different reporter in the experiment's YAML configuration if desired.

### Reproducibility

MetaMARL provides explicit seeds and utilities for deterministic configuration. Its entry-point tooling also initializes `PYTHONHASHSEED` when needed. These measures support reproducible comparisons, but **distributed execution, floating-point operations, and the order in which random numbers are consumed can still affect exact repeatability**. In particular, matching policy seeds alone does not guarantee matched action samples when policies consume different numbers of random draws.

## Reporting and visualization

Metrics are defined through schemas and can be collected at multiple levels: individual agents and policies, shared environment dynamics, and mechanism evaluation or outer-loop fitness.

| Backend | Purpose |
| --- | --- |
| **CSV** | Structured outputs for independent analysis, plotting, and post-processing |
| **TensorBoard** | Local visualization of learning and optimization curves |
| **Weights & Biases** | Run tracking, comparisons, dashboards, and experiment monitoring |

Configure the reporter in YAML rather than passing the legacy `--reporter` flag to a deleted debug script. Output locations and available visualizations depend on the configured reporter and experiment.

The [`tutorials/`](tutorials/) directory contains notebooks illustrating framework abstractions, mechanisms, experiment construction, and result visualization. For an introductory walkthrough, see [QUICKSTART.md](QUICKSTART.md).

## Research applications and roadmap

MetaMARL aims to support research at the intersection of multi-agent reinforcement learning, mechanism design, and dynamical systems. Relevant questions include how agents adapt to shared constraints, how interventions affect collective outcomes, and whether a mechanism remains effective when the agents subject to it learn and respond strategically.

Areas of interest include:

- **Mechanism design and evaluation:** Compare fixed or optimized interventions in interacting populations.
- **Common-pool resource dilemmas:** Investigate sustainability, incentives, and strategic resource use.
- **Multi-agent coordination and alignment:** Study how local incentives and coupled dynamics lead to cooperation, competition, or undesirable collective behavior.
- **Algorithm development:** Use modular environments and objective definitions to evaluate new optimizers, learning methods, and control approaches.

The long-term direction is an extensible collection of algorithms and realistic benchmarks, rather than a system tied to a single learning algorithm or environmental domain. Work in progress on the `dev` branch should not be interpreted as functionality available in this release.

## Repository layout

| Path | Contents |
| --- | --- |
| `core/` | Framework abstractions, optimizers, Ray adapters, world coordination, configuration, metrics, and reporting |
| `examples/bilevel_fishery/` | Fishery environment and YAML experiment configuration |
| `examples/cartpole/` | CartPole pipeline-validation example |
| `tests/` | Unit, integration, and example tests |
| `tutorials/` | Research and instructional notebooks |
| `docs/` | Architecture and project documentation |
| `paper/` | Research-software manuscript and publication materials |

See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before contributing, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the principal components and data flow.

## Tests and development

Install the development dependencies as described above, then run:

```bash
uv run python -m pytest
uv run python -m pytest -m "not integration and not notebook"
uv run ruff check --no-fix .
uv run ruff format --check .
```

Pytest also checks executable docstring examples (doctests) and reports coverage for `core/`. Integration tests can launch Ray and example processes and may take longer than unit tests.

Some tests are intentionally skipped for tracked limitations, including observation-size composition and certain reporting or missing-metric edge cases. Their pytest skip reasons use stable issue IDs so they can be addressed without hiding the rest of the test suite. A high line-coverage percentage does not imply that these outstanding behaviors are verified.

`ruff check --no-fix` is intentional: repository configuration may otherwise allow Ruff to modify files during lint checks.

## Authorship, affiliations, and funding

**Principal author and developer:** **Nadine (Mohamed) Abdelghani**, publishing as **Nadine Mohamed**.

**Scientific contributors and publication authors:**

- Nadine Mohamed
- Elise Devoie
- Benjamin Rosman
- Christof Brandtner
- Guillaume Dumas

The project was developed in affiliation with the **PPSP Lab (PI: Guillaume Dumas)**, **Mila – Quebec AI Institute**, and **Université de Montréal**, with funding from **CIFAR**.

## Citation

If you use MetaMARL in scientific work, please **cite the software and its associated research publication**, when available, and acknowledge the project authors. Software citation metadata can be maintained in a root-level `CITATION.cff` file so GitHub can provide formatted citations.

## License

MetaMARL is released under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for the full terms and any accompanying attribution notices.

Redistributions and derivative works must preserve applicable copyright, license, and attribution notices as required by that license. Academic citation is requested separately; it is not a universal legal requirement of Apache 2.0.
