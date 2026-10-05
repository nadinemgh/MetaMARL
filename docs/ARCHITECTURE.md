# Architecture

This guide explains how a MetaMARL run is organised, so that you can extend the
framework with a new benchmark, a new mechanism or a new reporting backend.
[QUICKSTART.md](../QUICKSTART.md) shows how to start a first run; this page explains
what happens inside it.

## What a bilevel run is

A bilevel run searches for the rules under which a society of learning agents
behaves well. The rules are called a mechanism; in the fishery benchmark the
mechanism is a fishing quota, and the society is a group of fishers that learn with
reinforcement learning. The outer level is an Evolution Strategies optimizer that
proposes a population of candidate mechanisms. For each candidate the inner level
trains a fresh society with RLlib, the reinforcement learning library on Ray, and
evaluates it. The outer level scores each candidate with a fitness computed from the
evaluation and moves its search distribution towards the better candidates. One
round of this exchange is a generation, and a run lasts a fixed number of
generations unless the optional convergence stop ends it earlier.

The two levels are not isolated. The outer environment holds the inner optimizer and
calls it directly; only the candidates travel through a shared Ray actor, the
`World`, because they must reach environments that live in other processes.

## Objects and who owns what

The composition root is `BilevelConfig` in `core/optimizers/bilevel.py`, a fluent
builder whose `build_optimizer` method creates everything else and returns a
`BilevelOptimizer`. The diagram shows the objects that exist after that call.

```text
BilevelOptimizer ------------------------------ reporter (primary, label "bilevel")
 |-- outer: ESOptimizer
 |      `-- env: RegulatorEnv subclass ---- inner (the RayOptimizer below)
 `-- inner: RayOptimizer
        `-- policy_actor: PolicyActor (Ray actor)
               `-- RLlib Algorithm
                      `-- env runners
                             `-- RLlibMultiAgentEnvAdapter
                                    `-- MultiAgentEnv subclass (leaders + followers)

World (Ray actor): RegulatorEnv publishes, MultiAgentEnv fetches, RayOptimizer flushes
```

`build_optimizer` starts Ray through `RayRuntime.ensure_initialized`
(`core/adaptors/ray/runtime.py`), creates the `World` actor, copies the two level
configurations, and copies the inner training seeds into the outer environment. It
hands the regulator's agent configurations to the inner environment as its leaders,
builds the inner `RayOptimizer` and then the outer `ESOptimizer`, and finally sets
the outer population size to the inner optimizer's `batch_capacity`, which is the
number of environments per env runner divided by the number of training seeds, that
is, the number of mechanism candidates. Every optimizer
derives from `Optimizer` (`core/optimizers/base.py`), which fixes `train`, `evaluate`,
`reset`, `stop` and `save`, and every configuration from `OptimizerConfig`
(`core/optimizers/config.py`), a builder whose `build_optimizer` works on a frozen
copy of itself, so the original stays editable.

## One generation, step by step

`ESOptimizer.train` in `core/optimizers/es/optimizer.py` runs the generations. In
each one it samples a population in logit space, which keeps every candidate inside
`(0, 1)`. The population is antithetic, made of mirrored noise pairs, unless it holds
a single candidate, which turns the search into a (1+1)-ES, or `break_symmetry`
replaces one mirrored sample by an independent one. Each sample becomes a dictionary
from mechanism identifier to action array. The optimizer then calls `reset` on the
regulator environment, which restarts the inner policy from its initial weights so
that every generation starts from the same learner, and calls `step` until the
episode ends.

`RegulatorEnv.step` in `core/envs/regulator.py` publishes every candidate to the
`World`, once per training seed, as a `Context` that carries a `MechanismContext` in
the `published` state. It then calls `RayOptimizer.train`, which runs the configured
number of RLlib iterations through the `PolicyActor`, runs one evaluation pass, and
returns the metrics it accumulated. During those iterations the inner environments
fetch their candidate from the `World` at every reset. The step passes the metrics
to `reward`, which a benchmark overrides to return one fitness per candidate. The
fishery version, `FisheryRegulatorEnv` in `examples/bilevel_fishery/regulator_env.py`,
averages the last `fitness_tail_steps` steps of each evaluation episode and appends
a `done` context for each candidate. The fishery regulator uses `horizon=1`, so its
episode ends after this one step.

Back in `ESOptimizer.train`, a fitness that is not finite, or whose length differs
from the population size, raises a `RuntimeError`. Otherwise the optimizer updates
the search mean and standard deviation, pushes one `ESSchema` record whose `inner`
field holds the inner metrics, and renders its queries. `BilevelOptimizer.train`
stops both levels and closes the reporter in a `finally` block, even after a failure.

## The World and the life of a candidate

The `World` (`core/world/base.py`) is a Ray actor that acts as a blackboard. It keeps
three registries in step: all contexts, the mechanism payloads, and the context
identifiers that each optimizer owns. Every change goes through a `.remote()` call.

A candidate passes through the four states of `MechanismStatus` in
`core/world/context.py`. It is `published` when the regulator appends it. The first
training environment that asks for it with `get_mechanism_by_id(mode=train)` moves it
to `train`, and every later training fetch of the same candidate and seed returns
`None`, which means "nothing new"; `MultiAgentEnv.reset` keeps the last candidate it
received. An evaluation fetch (`mode=eval`) accepts `train` and `eval` entries and can
be repeated, so several evaluation seeds share one trained candidate.
`RayOptimizer.evaluate` ends with `World.flush(status=eval)`, which removes the
evaluated candidates. Only a benchmark's regulator environment writes `done`, with
a fitness, from its `reward` override; the base `RegulatorEnv` only publishes.

Candidates are matched to environments through indexes. Inside one env runner,
environment `k` runs mechanism `k % M` with training seed number `k // M`, where `M`
is the number of candidates, and each pair owns one RLModule named
`<policy>_m<index>_s<seed>`. The callback `tag_episode_with_env_idx`
(`core/callbacks.py`) rewrites each episode identifier to
`env=<i>|m=<mechanism>|ps=<policy seed>|ss=<env seed>|raw=<id>`, and the
`policy_mapping_fn` built in `core/adaptors/ray/optimizer_config.py` parses it to
route each agent to its module.

## The inner environment step

`MultiAgentEnv` in `core/envs/marl_regulated.py` is a plain class that works on one
shared state object. RLlib never sees it: `RLlibMultiAgentEnvAdapter` in
`core/adaptors/ray/marl_env.py` wraps it, builds a fresh state at every `reset`,
writes the followers' actions into the current state at every `step`, and translates
the answer into RLlib's dictionaries. Only the followers are
RLlib agents; the leaders, which hold the regulator's mechanisms, never appear in
the returned spaces.

At `reset` the environment fetches its candidate from the `World` and writes it as
the leaders' action; before any candidate has arrived, each leader plays the
`default` action of its mechanisms. A `step` then runs in a fixed order: every
agent applies its mechanisms and adds its reward, leaders first, and the method marked
`@transition` advances the state, the termination flags are set (all agents are
truncated once `horizon` is reached), the step is logged, and the observations are
rebuilt, with the leaders' mechanisms adding their own contribution. The decorators
`@reset` and `@transition` live in `core/envs/hooks.py`.

## Mechanisms and the state model

An `MDPState` (`core/mechanism/base.py`) holds the time `t`, the fixed `params`, and
five time-indexed trees: `state`, `obs`, `actions`, `rewards` and `raw_actions`. The
trees are `Trajectory` objects (`core/mechanism/types.py`) whose leaves are lists
with one entry per step. The state and the actions are stocks, so an unwritten step
keeps the previous value. The rewards and observations are flows, held by
`FlowTrajectory`, so an unwritten step starts from zero. `raw_actions` keeps each
action as it arrived, before any mechanism decoded it.

A mechanism does not write its effect into the shared state. It returns a residual
`MDPState` that holds only the fields it changes, and `MDPState.add` composes the
residuals into a new state: additive quantities are summed and constrained spaces
are intersected. The one exception is the decoded action, which a mechanism writes
in place into `actions` at the current step.

A `Mechanism` implements three methods. `decode` maps the raw action to the
mechanism's own coordinates, `apply` returns the residual of the transition, and
`observe` returns the contribution to the observations. The environment calls
`observe` separately, after the transition, because what `apply` writes into `obs`
would be read too late. A `MechanismConfig` is the frozen recipe that names the
mechanism class and builds it for an agent. The shipped rules (`Quota`, `Subsidy`,
`ThresholdPenalty`, `SocialInfluence`) are in `core/mechanism/algorithms/`.

An `Agent` (`core/agents/base.py`) owns mechanisms, and its `action` method passes
the raw entry of the current step to each mechanism it holds an action for. This is
why the quota works on the followers' raw policy output: the regulator acts first, so
the request it reads has not been decoded yet. The fishers decode their output with
`sigmoid(z / ACTION_TEMPERATURE)`, where `ACTION_TEMPERATURE` is 4.0 in `core/utils.py`.

## Metrics and reporting

A `MetricSchema` (`core/metrics/schemas.py`) declares what is logged. Every leaf
field names its reduction: mean, sum, last, min, max or series, from `ReduceProtocol`
in `core/metrics/enums.py`, which also declares an `EMA` that is not implemented. A
nested schema is a static branch, and a `dict[ID, MetricSchema]` is a dynamic branch
whose children are created on first use, one per agent, mechanism or seed. A
`MetricLogger` (`core/metrics/logger.py`) accumulates values with `push` and
`push_data`, returns raw histories with `peek`, and returns reduced values, clearing
the logger, with `reduce`. A dynamic key absent from a push leaves a gap, shown as
`None` in a peeked history.

A `Query` (`core/reporting/query.py`) selects series by path. At a dynamic branch the
path continues with a reduction token, so the fishery's training biomass is
`("train", "rollout", "by_mechanism", ReduceProtocol.SERIES, "by_seed",
ReduceProtocol.MEAN, "by_episode", ReduceProtocol.MEAN, "fish_norm_next_mean")`: one
curve per mechanism, averaged over seeds and episodes. A `Reporter`
(`core/reporting/base.py`) resolves each query against a populated schema and calls
the backend hook `_report`; each query is rendered on its own, so one malformed query
is logged and does not stop the run. The backends are `CSVReporter`,
`TensorBoardReporter` and `WandbReporter`, each built by a `ReporterConfig`
subclass through `build(label=...)`. The label keeps the owners apart: the CSV
reporter writes `<output_dir>/<project>/<world>-<label>/<title>.csv`, where
`output_dir` defaults to `results`, the title is the query title with every run of
characters other than letters, digits, `_` and `-` replaced by `_`, and the label is
`bilevel` for the run's primary reporter, an optimizer class name, or an environment
identifier.

Three places report: `ESOptimizer` at the end of a generation, `RayOptimizer` at the
end of an inner training call, and each inner environment at the end of every episode
through `log_and_report_episode_metrics` (`core/callbacks.py`), which also hands the
reduced episode metrics to RLlib. `RaySchema` groups those episodes by mechanism,
seed and episode, the shape that `FisheryRegulatorEnv.reward` walks.

## Configuration

A run can be configured in Python or in YAML, and both describe the same objects.
The Python form chains builders on `BilevelConfig` (`world`, `reporter`, `ray`,
`regulator`, `society`) and on the two level configurations, `ESConfig`
(`core/optimizers/es/config.py`) and a subclass of `RayOptimizerConfig` such as
`APPOptimizerConfig`. The inner configuration mirrors the builders of RLlib's
`AlgorithmConfig` but only records them, and replays them when `build_optimizer`
runs, so that `debugging` can adjust the number of environments per runner before
RLlib validates it. `examples/bilevel_fishery/debug.py` is the reference.

The YAML form is read by `core/config/yaml.py`. A mapping with `_target_` is built by
calling that class, `_args_` gives its positional arguments, `_calls_` lists builder
methods applied in order, `_symbol_` imports an object without calling it, and
`_tuple_` builds a tuple. The `experiment` section describes the object and the
optional `run` section lists the calls that `metamarl run` applies to it. Every
failure is a `ConfigError`. An error raised while building a node or applying its
calls names the YAML path of that node; an import error names the import path, and a
missing or unreadable file names the file. `metamarl check <file>` builds the
experiment without running it, so these errors appear before any training.

## Reproducibility

Equal seeds give equal runs, through several mechanisms. The training seeds come
from `.debugging(seed=..., num_seeds=...)` on the inner configuration, the evaluation
seeds from `.evaluation(base_seed=..., num_seeds=...)`, and each RLModule gets a
seeded Xavier initializer. The outer search draws its noise from a generator seeded
with the regulator configuration's `debugging(seed=...)`. The outer level has no
training seeds of its own: the regulator publishes one context per inner training
seed, which `BilevelConfig` passes to it through its `env_config`, so
`ESConfig.debugging` rejects a `num_seeds` argument.

Python randomises string hashes per process, and RLlib iterates over sets of agent
identifiers, so the order of agents in an inference batch would otherwise change
between runs. `ensure_hash_seed` (`core/config/hash_seed.py`) restarts the process
once with `PYTHONHASHSEED=0` when the variable is unset; the debug scripts and the
`metamarl` command call it first. APPO applies gradient updates on a background
thread, so `PolicyActor.evaluate` waits for that thread through
`wait_for_learner_thread` (`core/adaptors/ray/learner_drain.py`) before evaluating.
That function relies on private RLlib attributes and raises when they are missing.
Ray runs in `local_mode`, so every actor lives in the driver process.

## Extending the framework

To add a benchmark, write an `Agent` subclass that overrides `observation` and
`reward`, and an `AgentConfig` subclass whose class variable `agent_cls` points at
it. Write a `MultiAgentEnv` subclass with a `@reset` and a `@transition` method, and
subclasses of `AgentEnvStepSchema` and `EpisodeRolloutSchema` (`core/envs/schema.py`)
for what it logs. Write a `RegulatorEnv` subclass whose `reward` returns one fitness
per candidate, plus the `Query` tuples to plot, and assemble them in a script like
`debug.py`. The notebook `tutorials/custom_benchmark_creation.ipynb` walks through
the steps.

To add a mechanism, subclass `Mechanism` and implement `apply`, overriding `decode` or
`observe` when needed. Add a `MechanismConfig` subclass with the parameters as
keyword-only fields and `mechanism_cls` set to the new class. A rule with nothing to
search uses `empty_action_space()` from `core/mechanism/config.py`, as the penalty does.

To add a reporting backend, subclass `Reporter` and implement `_report(query, x, ys,
errors, colors)` and `close`, then subclass `ReporterConfig` and implement
`build(*, label=None)`. `core/reporting/csv.py` is a compact model.

## Testing layout

The tests live in `tests/`, in one directory per layer of `core/`, plus `unit` for
the top-level helpers, `examples` and `integration`. The unit tests never start Ray:
the `conftest.py` files of the test directories provide fakes, such as `FakeWorld` and
`ScriptedRegulatorEnv` in `tests/optimizers/conftest.py`, the toy benchmark `ToyEnv`
in `tests/envs/conftest.py`, and the undecorated `World` class reached through
`World.__ray_metadata__.modified_class` in `tests/world/conftest.py`. The markers
`unit`, `integration`, `system` and `notebook` are declared in `pytest.ini` with
`--strict-markers`. Integration tests launch the example scripts as child processes
with a time limit, and `tests/conftest.py` runs them last. `pytest.ini` also collects
the doctests of `core/` and the examples, and `.github/workflows/ci.yml` has three
jobs: `lint`, `unit` and `integration`.
