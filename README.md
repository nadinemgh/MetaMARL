# MetaMARL

MetaMARL is an extensible research framework for **multi-agent reinforcement learning, bilevel mechanism design, and agent alignment**. It provides a modular environment for studying how adaptive agents interact, how their individual incentives shape collective outcomes, and how mechanisms or additional learning agents can influence the resulting dynamics.

In a typical bilevel experiment, a regulator (leader) searches over the rules governing an environment, such as fishing quotas, monitoring penalties, or water-allocation policies, while a society of learning agents (followers) adapts its policies under each candidate mechanism. The regulator evaluates mechanisms based on the collective behavior that emerges from the trained society, allowing it to search for interventions that improve system-level outcomes while accounting for self-interested adaptation.

**The framework is optimizer-agnostic by design.** Its architecture separates the outer mechanism optimizer, the inner agent-learning algorithms, and the environment dynamics. The current release implements Evolution Strategies (ES) for outer-loop optimization and RLlib-based APPO and PPO for inner-loop learning. The interfaces are designed to accommodate alternative optimization and control algorithms, including reinforcement learning, genetic and evolutionary algorithms, model predictive control (MPC), and other numerical or learning-based solvers in future releases.

Beyond conventional bilevel optimization, MetaMARL is designed to accommodate different modes of multi-agent adaptation:

- **Bilevel optimization and meta-learning:** Leader agents or regulators evaluate candidate mechanisms through a shared `World` abstraction that coordinates environments, policy training, evaluation, and feedback across optimization levels.
- **Simultaneous multi-agent learning:** Multiple agents interact within a shared Markov decision process (MDP), selecting actions from the same initial state at each discrete time step. Their combined effects determine the subsequent transition, supporting the study of coupled learning dynamics and strategic interactions.
- **Composable mechanisms and agent behaviors:** Agent and mechanism objects can be independently defined, configured, and combined. Actions and interventions are expressed through a vectorized decomposition, enabling multiple mechanisms to contribute to the evolution of a shared state and supporting heterogeneous agent roles.
- **Leader–follower experimentation:** Environments can incorporate regulators, meta-learning agents, and followers, providing building blocks for investigating incentive design, adaptive regulation, and alignment in non-stationary multi-agent systems.

The framework's core abstractions live in `core/` and remain independent of domain-specific models. Users can construct custom multi-agent environments by defining agent objects, mechanism objects, shared state dynamics, observations, rewards, and evaluation objectives without rewriting the optimization and reporting infrastructure.

MetaMARL currently provides three example environments:

| Example | What the regulator searches | What the society learns |
| --- | --- | --- |
| `examples/bilevel_fishery` | Fishing regulations, including harvest quotas and configurable mechanisms | How much each fisher harvests and how much it restores the stock |
| `examples/fresh_water` | Water-allocation policies for a reservoir | How much water each farm requests each day |
| `examples/cartpole` | An inert mechanism parameter | How to balance the pole; a pipeline check whose fitness is the constant 1.0 |

The **fresh-water benchmark is a work in progress**, under development as a richer multi-agent alignment testbed. It aims to investigate adaptive resource management, heterogeneous agents, competing objectives, and cooperation under environmental constraints. The benchmark is intended to extend beyond simple mechanism optimization toward more complex settings involving dynamically coupled decisions and collective resource sustainability.

## Research scope and long-term vision

MetaMARL is currently oriented toward **policy makers and researchers working on mechanism design**, providing flexible tools for constructing environments, defining regulatory interventions, and evaluating how adaptive populations respond to different incentive structures.

Its longer-term objective is to evolve into an open, extensible research ecosystem bringing together:

- **Mechanism-design algorithms:** A collection of interchangeable optimization, control, and learning approaches for designing and evaluating interventions in multi-agent systems.
- **Agent-alignment testbeds:** Configurable environments for studying coordination, cooperation, competition, strategic adaptation, and the effects of regulatory or alignment agents.
- **Multi-agent learning benchmarks:** Reproducible tasks and evaluation protocols for comparing algorithms across different environments, agent populations, mechanism classes, and objectives.
- **Multi-agent AI safety research:** Infrastructure for developing and evaluating approaches to aligning adaptive agents when their objectives, policies, and interactions evolve over time.

Of particular interest is the investigation of **multi-agent alignment as a dynamical problem**: how the introduction of a mechanism, incentive, or additional agent changes the behavior of an interacting population, and under what conditions these interventions can steer the system toward more desirable or stable outcomes.

The broader aim is to support a collaborative research ecosystem of algorithms and benchmarks rather than a framework tied to one optimization method, environmental model, or definition of alignment.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12, which `pyproject.toml` requires (`>=3.12,<3.13`). From the repository root:

```bash
uv sync
```

This installs the runtime dependencies (Ray 2.53 with RLlib, PyTorch, Weights & Biases and the scientific stack) together with the `dev` group, which holds pytest, ruff, nbconvert, the Jupyter kernel and TensorBoard. An installation without the `dev` group gets the TensorBoard reporter through the optional `tensorboard` extra.

The fresh-water example runs on a built-in lake model by default. Its `--hydrology raven` option drives the external Raven hydrological model instead; Raven is not a Python dependency, and its executable and model directory are given on the command line (see the docstring of `examples/fresh_water/debug.py`).

## Running an experiment

Every entry point fixes the string-hash seed before it does any work: when `PYTHONHASHSEED` is unset, the process restarts itself with `PYTHONHASHSEED=0`, which the first two log lines report. This standardizes Python's hash behavior across runs and supports reproducibility when used alongside fixed random seeds and deterministic settings. Other sources of nondeterminism, including distributed execution and numerical operations, may still affect results.

Set `WANDB_MODE=offline` to run without a Weights & Biases account; the runs are then stored under `wandb/`.

An experiment described in YAML runs through the `metamarl` command, which `uv sync` installs:

```bash
WANDB_MODE=offline uv run metamarl check examples/bilevel_fishery/config.yaml
WANDB_MODE=offline uv run metamarl run examples/bilevel_fishery/config.yaml
```

`check` builds the experiment without running it and prints `Config OK: core.optimizers.bilevel.BilevelConfig`; `run` builds it and trains it. The same command is available as `uv run python -m core.config.cli`.

Each example also has a script that builds the same kind of experiment in Python and exposes its main sizes as options. Run the scripts as modules from the repository root:

```bash
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \
    --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 --reporter csv

WANDB_MODE=offline uv run python -m examples.cartpole.debug --algo appo \
    --outer-iters 2 --train-iters 2 --horizon 20 --reporter csv

WANDB_MODE=offline uv run python -m examples.fresh_water.debug \
    --outer-iters 2 --train-iters 2 --horizon 30 --num-agents 20 --reporter csv
```

These smoke configurations are intended for quick validation on a development machine. Without options, the scripts run substantially larger experiments: the fishery, for instance, runs 1000 generations of 50 APPO iterations with ten fishers.

### Reporting and visualization

MetaMARL includes a modular metrics, reporting, and visualization infrastructure designed to separate experiment execution from the tools used to inspect and analyze results.

The reporting layer supports multiple backends:

| Backend | Purpose |
| --- | --- |
| **Weights & Biases (W&B)** | Experiment tracking, metrics visualization, comparison across runs, and monitoring optimization |
| **TensorBoard** | Local visualization of training curves, optimization metrics, and learning progress |
| **CSV** | Structured export of experiment metrics for independent statistical analysis, plotting, and post-processing |

The reporting abstractions are designed to allow additional backends without modifying the environment or optimization logic.

Metrics can be collected at different levels of the experiment, including individual agents and policies, environment dynamics, and outer-loop mechanism evaluation. Depending on the benchmark, these can include agent rewards, collective welfare, resource sustainability, mechanism parameters, and optimization fitness.

With `--reporter csv`, experiment metrics are written under `results/bilevel/<world>-<optimizer>/`. With the default `--reporter wandb`, metrics are sent to Weights & Biases.

The `tutorials/` directory also provides notebooks for exploring framework abstractions and visualizing experiments.

[QUICKSTART.md](QUICKSTART.md) walks through a first run and its outputs.

## Repository layout

| Path | Content |
| --- | --- |
| `core/optimizers/` | Bilevel optimization composition (`bilevel.py`), the ES optimizer (`es/`), and configurations for RLlib optimizers (`appo/`, `ppo/`); structured for interchangeable outer and inner solvers |
| `core/adaptors/ray/` | Ray and RLlib integration: the inner optimizer, RLlib algorithm actors, environment adapter, and distributed runtime |
| `core/envs/` | Single-agent and multi-agent environment abstractions, regulated environments, and the environments connecting agent learning to mechanism optimization |
| `core/agents/` | Modular agent objects and interfaces for composing heterogeneous agents and learning roles |
| `core/mechanism/` | Configurable, composable mechanisms and state-dependent interventions; supports vectorized mechanism and action decomposition |
| `core/world/` | Shared `World` abstraction coordinating mechanism candidates and environment instances |
| `core/metrics/` | Structured metric definitions and collection across agents, mechanisms, and optimization levels |
| `core/reporting/` | Reporting and visualization backends, including CSV, TensorBoard, and Weights & Biases |
| `core/config/` | YAML configuration loading, the `metamarl` CLI, and reproducibility utilities |
| `examples/` | Fishery and CartPole benchmarks, alongside the developing fresh-water alignment benchmark |
| `tests/` | Unit, integration, and benchmark tests covering framework components and examples |
| `tutorials/` | Notebooks on framework abstractions, environment construction, mechanisms, and visualization |
| `docs/` | Architecture documentation and maintainer notes |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how these pieces work together during a run, and [AGENTS.md](AGENTS.md) is the guide for anyone, human or coding agent, about to change the code. [docs/MERGE_NOTES.md](docs/MERGE_NOTES.md) records what the latest cleanup pass changed and what it left open, and [TODO.md](TODO.md) holds the maintainer's plan and the archived in-code notes.

## Tests and lint

```bash
uv run python -m pytest                                         # whole suite
uv run python -m pytest -m "not integration and not notebook"   # without the child-process runs
uv run ruff check --no-fix .
uv run ruff format --check .
```

The suite runs the doctests of `core/` and `examples/` as well, and measures the coverage of `core/`.

Tests marked `integration` start the example scripts and the `metamarl` command in child processes, each with its own Ray runtime. Use the `-m pytest` form, which binds pytest to the project's environment.

Always pass `--no-fix` to `ruff check`: `ruff.toml` sets `fix = true`, so a plain `ruff check` rewrites files.

Continuous integration (`.github/workflows/ci.yml`) runs the lint, the unit tests and the integration tests as three jobs.

## License

BSD 3-Clause; see [LICENSE](LICENSE).