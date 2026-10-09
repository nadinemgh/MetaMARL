# AGENTS.md

This repository is MetaMARL, a research framework for bilevel mechanism design. An
outer Evolution Strategies optimizer searches the parameters of a regulatory mechanism
(a quota, a subsidy, a penalty or a social-influence rule). For every candidate, an
inner RLlib society of learning agents is trained and evaluated under that mechanism,
and the society's outcome becomes the candidate's fitness. The reference benchmark is
the fishery in `examples/bilevel_fishery/`. This file is for a coding agent or a new
contributor; the design rationale lives in `docs/ARCHITECTURE.md`.

## Where to read first

Start with `README.md` for the scientific picture and `QUICKSTART.md` for a first
run that finishes in about fifteen seconds. Then read `docs/ARCHITECTURE.md` for how
the two levels, the `World` actor and the metric pipeline fit together, and
`docs/MERGE_NOTES.md` for what the latest cleanup pass changed and what it left open.
`TODO.md` holds the maintainer's open plan, two handoff checklists ranked P0 to P2,
and in its last section the in-code TODO notes that the cleanup pass archived, some
marked as resolved; check an item against the code before acting on it.
`docs/REPRISE.md` is the status board of the cleanup pass and is only needed to
resume that work.

## Module map

| Path | What it holds |
| --- | --- |
| `core/optimizers/` | `Optimizer` base class, `BilevelConfig` (the composition root), and the `es/`, `appo/` and `ppo/` configurations |
| `core/adaptors/ray/` | `RayOptimizer`, `PolicyActor`, `RayRuntime`, the seed and policy layout in `optimizer_config.py`, and `learner_drain.py` |
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
`MDPState.add` composes the residuals of several mechanisms. The only in-place write
is the decoded action, which a mechanism stores in `actions` at the current step.

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
`WANDB_MODE=offline` so no run tries to reach the network. The results below were
obtained on this tree at the time of writing.

| Command | Purpose | Result observed |
| --- | --- | --- |
| `uv sync --directory <repo> --locked` | Install the locked environment | 169 packages resolved, 147 checked, no change |
| `uv run --directory <repo> ruff check --no-fix .` | Lint without rewriting | `All checks passed!` |
| `uv run --directory <repo> ruff format --check .` | Formatting check | `191 files already formatted` |
| `uv run --directory <repo> python -m pytest -m "not integration and not notebook"` | Unit suite, as in CI | 1925 passed, 1 skipped, 19 deselected in 39 s, 99 % coverage of `core` |
| `uv run --directory <repo> python -m pytest -m integration --no-cov -q` | Integration suite | 14 passed in 105 s |
| `uv run --directory <repo> python -m pytest tests/world --no-cov -q` | One layer only | 40 passed in under a second |
| `uv run --directory <repo> metamarl check examples/bilevel_fishery/config.yaml` | Validate a YAML config | `Config OK: core.optimizers.bilevel.BilevelConfig`, exit 0 |
| `uv run --directory <repo> python -m examples.bilevel_fishery.debug --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 --reporter csv` | Smoke run of the full bilevel loop | Exit 0 in 16 s, ending with `[Bilevel] Run finished` and `best_fitness=2.6060` |

`metamarl run|check` exits 0 on success, 2 on a configuration error and 130 on an
interrupt. The default `pytest` options collect the doctests of `core/` and of the three
example packages, which is why the collected total (1945) is larger than the number
of test functions.

## Conventions

Every public function and class carries a NumPy-style docstring that states array shapes
and units, a "When to use" paragraph, a runnable example and, for a method taken from a
paper, the reference with its DOI. Because `--doctest-modules` is on, the example is
executed by the test run and must stay correct.

Code, comments, docstrings, documentation, notebooks and commit messages are written in
English. Commits are imperative and scoped, for example
`fix(ray): label the Ray optimizer's messages [Ray] instead of [PPO]`.

An invalid value of a public parameter raises `ValueError` and an unknown argument
raises `TypeError`; `assert` is not used in `core/`. Comments explain why, not what,
and neither `TODO` comments nor commented-out code are kept, because the lint
configuration selects the `ERA`, `FIX` and `TD` rules. Overriding methods use
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
