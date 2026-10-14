# AGENTS.md

This repository is MetaMARL, a research framework for bilevel mechanism design. An
outer Evolution Strategies optimizer searches the parameters of a regulatory mechanism
(a quota, an incentive, a penalty or a social-influence rule). For every candidate, an
inner RLlib society of learning agents is trained and evaluated under that mechanism,
and the society's outcome becomes the candidate's fitness. The reference benchmark is
the fishery in `examples/bilevel_fishery/`. This file is for a coding agent or a new
contributor; the design rationale lives in `docs/ARCHITECTURE.md`.

## Where to read first

Start with `README.md` for the scientific picture and `QUICKSTART.md` for a first
run that finishes in about fifteen seconds. Then read `docs/ARCHITECTURE.md` for how
the two levels, the `World` actor and the metric pipeline fit together.
`TODO.md` holds the maintainer's open plan, two handoff checklists ranked P0 to P2,
and in its last section the in-code TODO notes that the cleanup pass archived, some
marked as resolved; check an item against the code before acting on it. The
maintainer's `# TODO (nadine)` comments that are still in the code are her open
questions: leave them where they are.

## Module map

| Path | What it holds |
| --- | --- |
| `core/optimizers/` | `Optimizer` base class, `BilevelConfig` (the composition root), and the `es/`, `appo/` and `ppo/` configurations |
| `core/adaptors/ray/` | `RayOptimizer`, `PolicyActor`, `RayRuntime`, the seed and policy layout in `optimizer_config.py`, the environment adapter in `marl_env.py` (which decodes the followers' actions), the common random numbers of `common_random.py`, and `learner_drain.py` |
| `core/envs/` | `MultiAgentEnv` (inner society), `RegulatorEnv` (outer step), and the typed state in `schema.py` |
| `core/mechanism/` | `Mechanism`, `MechanismConfig`, and the shipped rules in `algorithms/` |
| `core/world/` | The `World` Ray actor and the `MechanismContext` records it stores |
| `core/metrics/` | `MetricSchema`, `MetricLogger` and the accumulators in `metric/` |
| `core/reporting/` | `Query`, `Reporter` and the CSV, TensorBoard and W&B backends |
| `core/config/` | The `metamarl` command line, the YAML loader and `ensure_hash_seed` |
| `examples/` | The `bilevel_fishery`, `cartpole` and `fresh_water` benchmarks, each with a `debug.py` |
| `tests/` | One directory per layer of `core/`, plus `unit/` for the cross-cutting modules (`core/callbacks.py`, `core/annotations.py`, `core/utils.py`), `examples/` and `integration/` |
| `tutorials/` | Five notebooks that walk through the framework |

## Invariants

Candidate mechanisms travel from the outer to the inner level only through the `World`
actor, as `MechanismContext` records. Nothing else carries them, so a change that
bypasses the `World` breaks the link between a candidate and its trained society.

A mechanism returns a residual of the state, reward or observation of an agent and
never overwrites what an earlier mechanism or the environment produced, because
`MDPState.add` composes the residuals of several mechanisms. Actions are decoded
before they reach a mechanism: the adapter decodes the followers' actions in `step`,
and `MultiAgentEnv.reset` decodes the leaders' candidates, so `MDPState.actions` holds
decoded values and the state keeps no raw action.

The number of environments per runner must be a multiple of the number of training
seeds, because environment `k` runs mechanism `k % M` with seed index `k // M`; the
check is in `core/adaptors/ray/optimizer_config.py`. Separately, the ES population
must be even, because antithetic sampling builds it from mirrored noise pairs. The
exceptions are a population of one, which runs a (1+1)-ES, a mechanism with no
parameter to search, and `break_symmetry`; the check is in the `batch_capacity`
setter of `core/optimizers/es/optimizer.py`.

Ray runs in `local_mode=True`, hard-coded in `core/adaptors/ray/runtime.py`. Library
code must not call `ensure_hash_seed`; only entry points do. When `PYTHONHASHSEED` is
unset it restarts the process with `PYTHONHASHSEED=0` and never returns, so a test
calls it only with `os.execv` replaced or in a child process.

`ruff.toml` sets `fix = true`, so a bare `ruff check` rewrites files. Always pass
`--no-fix` when you only want a report.

## Commands

Run each command from any directory by pointing `uv` at the repository. Set
`WANDB_MODE=offline` so no run tries to reach the network. The tests and the lint need
the `dev` group, so install with `--group dev`, as the continuous integration does.
This table describes what each command is for and what a successful run looks like; it
quotes no test, file or coverage counts, which change with every commit. Run the
command to see the present figures.

| Command | Purpose | Success looks like |
| --- | --- | --- |
| `uv sync --directory <repo> --locked --group dev` | Install the locked environment with the test and lint tools | Exit 0 and no change to `uv.lock` |
| `uv run --directory <repo> ruff check --no-fix .` | Lint without rewriting | Exit 0 and `All checks passed!` |
| `uv run --directory <repo> ruff format --check .` | Formatting check | Exit 0 and every file reported as already formatted |
| `uv run --directory <repo> python -m pytest -m "not integration and not notebook"` | Unit suite and doctests, as in CI | Exit 0; the summary line reports no failure, and the coverage report covers `core` |
| `uv run --directory <repo> python -m pytest -m integration --no-cov` | Integration suite, which starts the example scripts and the `metamarl` command in child processes | Exit 0; it takes minutes, not seconds |
| `uv run --directory <repo> python -m pytest -m notebook --no-cov` | The tutorial notebooks, executed end to end | Exit 0 |
| `uv run --directory <repo> python -m pytest tests/world --no-cov` | One layer only | Exit 0 in about a second |
| `uv run --directory <repo> metamarl check examples/bilevel_fishery/config.yaml` | Validate a YAML config | `Config OK: core.optimizers.bilevel.BilevelConfig`, exit 0 |
| `uv run --directory <repo> python -m examples.bilevel_fishery.debug --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 --reporter csv` | Smoke run of the full bilevel loop | Exit 0 after about fifteen seconds, ending with `[Bilevel] Run finished` |
| `uv run --directory <repo> python -m examples.cartpole.debug --algo appo --outer-iters 2 --train-iters 2 --horizon 20 --reporter csv` | Smoke run of the cart-pole pipeline check (`--algo ppo` for the PPO variant) | Exit 0, ending with `[Bilevel] Run finished` and `best_fitness=1.0000` |
| `uv run --directory <repo> python -m examples.fresh_water.debug --outer-iters 2 --train-iters 2 --horizon 30 --num-agents 20 --reporter csv` | Smoke run of the fresh-water benchmark on the built-in lake model | Exit 0, ending with `[Bilevel] Run finished` |

`metamarl run|check` exits 0 on success, 2 on a configuration error and 130 on an
interrupt. The default `pytest` options collect the doctests of `core/` and of the three
example packages, so the collected total is larger than the number of test
functions. The fresh-water example can also drive the external Raven model with
`--hydrology raven --raven-cwd <dir> --raven-cmd <exe>`; Raven is not in the
repository and this path is untested. `examples/fresh_water/debug_raven.py` is a Raven
variant that does not run yet, because it still uses the builders that `.regulator` and
`.society` replaced.

## Conventions

Every public function and class carries a NumPy-style docstring that states array shapes
and units, a "When to use" paragraph, a runnable example and, for a method taken from a
paper, the reference with its DOI. Because `--doctest-modules` is on, the example is
executed by the test run and must stay correct.

Code, comments, docstrings, documentation, notebooks and commit messages are written in
English. Commits are imperative and scoped, for example
`fix(ray): label the Ray optimizer's messages [Ray] instead of [PPO]`.

An invalid value of a public parameter raises `ValueError` and an unknown argument
raises `TypeError`; `assert` is not used in `core/`. Comments explain why, not what.
Do not add commented-out code or new `TODO` comments, although `ruff.toml` selects
the `ERA`, `FIX` and `TD` rules and then ignores them; the maintainer's existing
`# TODO (nadine)` comments stay as they are. Overriding methods use
`core.annotations.override`, naming a direct base of the class; a test in
`tests/unit/test_annotations.py` checks every use. Formatting is `ruff format` at 88
columns with double quotes and no magic trailing comma.

Tests mirror the layer they cover and use hand-written fakes, such as `FakeWorld`
and `ScriptedRegulatorEnv` from `tests/optimizers/conftest.py`, rather than starting
Ray. Only `tests/integration/` starts Ray, and the hook in `tests/conftest.py` runs
those items last.

## Traps

The patterns `*.csv` and `*.txt` are gitignored, so a CSV written by the smoke run does
not show up in `git status`. It lands in `results/<project>/<world>-<label>/`, one file
per query title.

Weights & Biases is the default reporter of the three `debug.py` scripts and of
`examples/bilevel_fishery/config.yaml`. Pass `--reporter csv` for a local run, and keep
`WANDB_MODE=offline` otherwise.

`ESConfig` and its builders reject an unknown argument with `TypeError`, and
`ESConfig.debugging` rejects `num_seeds`: the regulator uses the training seeds of
the society configuration, so set the count there.

## Debugging common symptoms

If building the inner optimizer raises `num_envs_per_env_runner=N must be divisible by
num_seeds=M`, the layout rule above is violated. The check is in
`core/adaptors/ray/optimizer_config.py`. `.env_runners(num_envs_per_env_runner=N)`
takes N as the number of mechanism candidates and multiplies it by the number of
seeds itself, so the usual cause is that `.env_runners(...)` was never called on the
society configuration, or that the count was set on the RLlib configuration
directly. Call `.env_runners` with the number of candidates.

If `ESOptimizer` raises `Antithetic ES requires an even batch size`, the population is
odd. It is the inner optimizer's `batch_capacity`, the `num_envs_per_env_runner`
given to the society's `.env_runners()`, which `core/optimizers/bilevel.py` copies to
the ES optimizer. Pass an even number there or enable `break_symmetry`.

If `RayOptimizer` raises `needs an evaluation setup`, `.evaluation(evaluation_config=
{'rollout_fragment_length': ...})` was never called on the society configuration.

If a callback raises `Env has no seed. It must be assigned at construction`, the
environment was built without a seed; the raise is in `core/callbacks.py`.

If `PolicyActor` fails with `RLlib's IMPALA/APPO learner no longer exposes the learner
thread`, the installed RLlib renamed the private attributes read by
`core/adaptors/ray/learner_drain.py`. Pin the locked version with `uv sync --locked`.

If two runs with the same seeds disagree, first confirm the process ran with
`PYTHONHASHSEED=0`. The entry points restart themselves with that value when the
variable is unset, keep a value already set, and log it at INFO level, or at WARNING
for `random`.
