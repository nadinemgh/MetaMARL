# MetaMARL

MetaMARL is a research framework for bilevel mechanism design with multi-agent
reinforcement learning. A regulator searches the rules of an environment, such as
a fishing quota or a water-allocation policy, while a society of learning agents
is trained under each candidate rule. The outer level is an Evolution Strategies
(ES) optimizer; the inner level is an RLlib optimizer (APPO or PPO) that trains
the agents and reports how the society fared. The fitness of a rule is computed
from what the trained society does under it, so the regulator learns which rules
lead self-interested learners to a good collective outcome.

The framework lives in `core/`, which knows nothing about fisheries or water. The
three benchmarks in `examples/` show how a problem is plugged in:

| Example | What the regulator searches | What the society learns |
| --- | --- | --- |
| `examples/bilevel_fishery` | one quota on the harvest, as a fraction of the stock | how much each fisher harvests and how much it restores the stock |
| `examples/fresh_water` | eight rules of a water policy for a reservoir | how much water each farm requests every day |
| `examples/cartpole` | an inert dial | how to balance the pole; a pipeline check whose fitness is the constant 1.0 |

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12, which
`pyproject.toml` requires (`>=3.12,<3.13`). From the repository root:

```bash
uv sync
```

This installs the runtime dependencies (Ray 2.53 with RLlib, PyTorch, Weights &
Biases and the scientific stack) together with the `dev` group, which holds
pytest, ruff, nbconvert, the Jupyter kernel and TensorBoard. An installation
without the `dev` group gets the TensorBoard reporter through the optional
`tensorboard` extra.

The fresh-water example runs on a built-in lake model by default. Its
`--hydrology raven` option drives the external Raven hydrological model instead;
Raven is not a Python dependency, and its executable and model directory are given
on the command line (see the docstring of `examples/fresh_water/debug.py`).

## Running an experiment

Every entry point fixes the string-hash seed before it does any work: when
`PYTHONHASHSEED` is unset, the process restarts itself with `PYTHONHASHSEED=0`,
which the first two log lines report. This is what makes two runs with the same
seeds give identical results. Set `WANDB_MODE=offline` to run without a Weights &
Biases account; the runs are then stored under `wandb/`.

An experiment described in YAML runs through the `metamarl` command, which `uv
sync` installs:

```bash
WANDB_MODE=offline uv run metamarl check examples/bilevel_fishery/config.yaml
WANDB_MODE=offline uv run metamarl run examples/bilevel_fishery/config.yaml
```

`check` builds the experiment without running it and prints `Config OK:
core.optimizers.bilevel.BilevelConfig`; `run` builds it and trains it. The same
command is available as `uv run python -m core.config.cli`.

Each example also has a script that builds the same kind of experiment in Python
and exposes its main sizes as options. Run the scripts as modules from the
repository root:

```bash
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \
    --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 --reporter csv
WANDB_MODE=offline uv run python -m examples.cartpole.debug --algo appo \
    --outer-iters 2 --train-iters 2 --horizon 20 --reporter csv
WANDB_MODE=offline uv run python -m examples.fresh_water.debug \
    --outer-iters 2 --train-iters 2 --horizon 30 --num-agents 20 --reporter csv
```

These smoke configurations take about fifteen seconds each on a laptop. Without
options the scripts run the full configurations, which are much larger: the
fishery, for instance, runs 1000 generations of 50 APPO iterations with ten
fishers. With `--reporter csv` every
plot of the run is written as a CSV file under
`results/bilevel/<world>-<optimizer>/`; with the default `--reporter wandb` the
plots go to Weights & Biases. [QUICKSTART.md](QUICKSTART.md) walks through a first
run and its outputs.

## Repository layout

| Path | Content |
| --- | --- |
| `core/optimizers/` | the bilevel composition root (`bilevel.py`), the ES optimizer (`es/`) and the configurations of the RLlib optimizers (`appo/`, `ppo/`) |
| `core/adaptors/ray/` | everything that touches Ray and RLlib: the inner optimizer, the actor that owns the RLlib algorithm, the environment adapter and the runtime |
| `core/envs/` | the multi-agent environment the society trains in and the regulator environment the ES steps |
| `core/agents/`, `core/mechanism/` | agents, the mechanisms they apply and the shared state the mechanisms act on |
| `core/world/` | the `World` actor through which the regulator hands each candidate to the environments |
| `core/metrics/`, `core/reporting/` | typed metric logging and the CSV, TensorBoard and Weights & Biases reporters |
| `core/config/` | the YAML loader, the `metamarl` command and the hash-seed guard |
| `examples/` | the three benchmarks |
| `tests/` | the test suite, one directory per layer of `core/` and one for the examples |
| `tutorials/` | notebooks on the abstractions, custom benchmarks, mechanisms and visualisation |
| `docs/` | the architecture guide and the notes for the maintainer |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how these pieces work
together during a run, and [AGENTS.md](AGENTS.md) is the guide for anyone, human
or coding agent, about to change the code. [docs/MERGE_NOTES.md](docs/MERGE_NOTES.md)
records what the latest cleanup pass changed and what it left open, and
[TODO.md](TODO.md) holds the maintainer's plan and the archived in-code notes.

## Tests and lint

```bash
uv run python -m pytest                                         # whole suite
uv run python -m pytest -m "not integration and not notebook"   # without the child-process runs
uv run ruff check --no-fix .
uv run ruff format --check .
```

The suite runs the doctests of `core/` and `examples/` as well, and measures the
coverage of `core/`. Tests marked `integration` start the example scripts and the
`metamarl` command in child processes, each with its own Ray runtime; they take
about a minute and a half, the rest of the suite about forty seconds. Use the `-m pytest` form, which binds pytest to the project's
environment. Always pass `--no-fix` to `ruff check`: `ruff.toml` sets `fix = true`,
so a plain `ruff check` rewrites files. Continuous integration
(`.github/workflows/ci.yml`) runs the lint, the unit tests and the integration
tests as three jobs.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
