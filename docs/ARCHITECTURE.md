# Architecture

This guide describes the principal components of **MetaMARL**, how they interact, and where to extend them. Start with [QUICKSTART.md](../QUICKSTART.md) to run an experiment. This document focuses on the architecture rather than a single example script.

MetaMARL supports multi-agent learning **with or without an outer mechanism optimizer**. The current release includes the fishery and CartPole examples. Freshwater and hydrological applications are under development on `dev` and are **not part of this release**.

## Design principles

- **Separate dynamics from algorithms.** Domain environments own state transitions and reward definitions. Optimizers own policy training or mechanism search.
- **Keep mechanisms composable.** Interventions can modify observations, incentives, actions, or shared state subject to their declared composition semantics.
- **Make bilevel optimization optional.** A multi-agent environment can be studied with fixed mechanisms; an outer optimizer is needed only when candidate mechanisms are being searched.
- **Make experiments inspectable.** Configurations, context objects, metric schemas, queries, and reporters expose the structure of a run.

## Core components

| Package | Responsibility |
| --- | --- |
| `core/agents/` | Agents and their mechanism configurations |
| `core/mechanism/` | `MDPState`, mechanism contracts, and composable effects |
| `core/envs/` | Shared multi-agent dynamics, lifecycle hooks, and regulator environments |
| `core/optimizers/` | Optimizer contracts, ES, and optional bilevel composition |
| `core/adaptors/ray/` | Ray/RLlib algorithm, policy, and environment adapters |
| `core/world/` | Shared context registry for mechanism publication and evaluation |
| `core/metrics/` | Structured metrics and reductions |
| `core/reporting/` | Queries and CSV, TensorBoard, and W&B reporters |
| `core/config/` | YAML object construction, CLI, and seed utilities |

The benchmark-specific agents, ecology, regulator fitness, and experiment configurations live under `examples/`.

## Single-level and bilevel experiments

A **single-level experiment** trains or evaluates agents in an environment under a specified mechanism. The environment, agents, and mechanisms can be reused independently of the algorithm driving policy updates.

A **bilevel experiment** adds an outer search:

1. A regulator or outer optimizer proposes candidate mechanisms.
2. An inner optimizer trains and evaluates agents against those candidates.
3. A regulator environment converts the observed outcomes into a fitness for each candidate.
4. The outer optimizer updates its search distribution.

In the released fishery example, the outer optimizer is **Evolution Strategies (ES)** and the inner optimizer is RLlib-based **PPO or APPO**, depending on configuration. This is an implemented combination, not a constraint on the abstractions: other optimization or control algorithms would require their own adapters.

### Bilevel object graph

`BilevelConfig` (`core/optimizers/bilevel.py`) is the composition entry point for this setup. A simplified view is:

```text
BilevelOptimizer
├── outer: ESOptimizer
│   └── env: RegulatorEnv subclass
├── inner: RayOptimizer
│   └── policy actor / RLlib Algorithm
│       └── environment runners
│           └── RLlibMultiAgentEnvAdapter
│               └── MultiAgentEnv subclass
└── reporter and shared run configuration

World (Ray actor)
├── published mechanism contexts
├── candidate status and identity
└── coordination between regulator and inner environments
```

`RayRuntime` manages Ray initialization for the configured run. The outer environment invokes the inner optimizer to obtain fitness evidence. The shared `World` makes candidate contexts available to environments in other Ray processes; it is **not** itself the learner.

`Optimizer` and `OptimizerConfig` provide the common optimizer and configuration contracts. ES, PPO, and APPO remain distinct implementations behind their respective configuration and adapter layers.

## One fishery ES generation

The following is the conceptual sequence, not a promise about the exact number of policies, candidates, or training iterations in any YAML configuration:

1. **Sample candidates.** `ESOptimizer` draws a population of candidate parameters; its search configuration determines the population, sampling behavior, and update parameters.
2. **Publish candidates.** `RegulatorEnv` creates contexts for the relevant candidate and seed combinations and publishes them through `World`.
3. **Train followers.** `RayOptimizer` drives the configured RLlib training iterations. Environment runners evaluate their assigned candidates through the adapter.
4. **Evaluate.** The inner optimization/evaluation flow records episode metrics for candidates and seeds.
5. **Compute fitness.** The benchmark-specific regulator reduces the selected inner metrics and returns one score per mechanism candidate.
6. **Update ES.** The optimizer uses candidate fitness to update its search parameters, report metrics, and advance to the next generation.

The `BilevelOptimizer` coordinates training and cleanup across both levels. The underlying learning budget and population layout are chosen by configuration, not hardcoded by the fishery's domain dynamics.

## World and candidate lifecycle

`World` (`core/world/base.py`) coordinates `Context` and `MechanismContext` records (`core/world/context.py`). The mechanism lifecycle uses statuses such as `published`, `train`, `eval`, and `done` to distinguish availability, training/evaluation, and completed fitness records.

A regulator publishes candidates; the corresponding inner environments retrieve the candidate assigned to their slot and seed. A key invariant of the **current environment lifecycle** is:

> A published mechanism, once successfully assigned to an environment, is immutable for that environment. Subsequent episode resets do not fetch a replacement mechanism.

In `MultiAgentEnv.reset`, retrieval is conditional on leaders being present **and** no published mechanism having been assigned. Default mechanisms may be used before assignment. The world and optimizer manage broader generation-level publication and cleanup; that does not mean a running environment dynamically switches mechanism during ordinary episode resets.

Policy and environment slots are identified by mechanism, policy seed, and other run-specific identifiers. `core/adaptors/ray/optimizer_config.py` constructs policy mapping, while episode callbacks retain identifying information so that metrics can be associated with the proper candidate.

## Inner environments and the RLlib adapter

The framework's `MultiAgentEnv` (`core/envs/marl_regulated.py`) models the shared environment. `RLlibMultiAgentEnvAdapter` (`core/adaptors/ray/marl_env.py`) translates it into the observation, action, reward, and termination dictionaries that RLlib expects.

- **Followers** are the RLlib-controlled agents and appear in the adapter's agent IDs and spaces.
- **Leaders or regulators** provide mechanisms and candidate interventions; they are not ordinary follower-policy entries in the adapter's returned RLlib spaces.
- Each follower can control several mechanisms; its RLlib action space is organized by mechanism identifier.
- The adapter creates a fresh `MDPState` for an episode, passes it to the framework environment, and translates observations and results back to RLlib.

`@reset` and `@transition` (`core/envs/hooks.py`) declare domain-specific lifecycle operations. A benchmark defines the relevant state initialization and dynamics; the framework drives them in conjunction with agent and mechanism effects.

An environment step conceptually:

1. Receives and decodes the followers' actions into mechanism coordinates.
2. Applies participating agents' mechanisms to the shared state.
3. Accumulates the agents' rewards and environment effects.
4. Executes the domain transition hook.
5. Advances termination/truncation and metric reporting.
6. Constructs observations for the next step.

The precise order is defined by `MultiAgentEnv.reset()` and `MultiAgentEnv.step()`; extensions should test interactions with that order, particularly when observation effects depend on a transition.

## MDPState, trajectories, and mechanisms

`MDPState` (`core/mechanism/base.py`) holds a timestep, parameters, and time-indexed `state`, `obs`, `actions`, and `rewards`. Trajectory types in `core/mechanism/types.py` control how entries are retained and composed over time.

Mechanisms implement the relevant portions of a small contract:

- `decode`: interpret a raw policy or regulator action in mechanism coordinates.
- `apply`: produce effects of a mechanism action on the current step.
- `on_reset`: contribute initialization or next-observation effects where applicable.

Mechanism effects are expressed through residual `MDPState` values and composed by the shared state operations. **Not all effects are simple addition:** a constrained space, an additive reward, and an observation contribution have different semantics. A new mechanism must document what it writes, which agents it targets, and how its results combine with other mechanisms.

`MechanismConfig` defines construction and action-space requirements. Current mechanism implementations include quota, incentives, penalties, and social influence in `core/mechanism/algorithms/`.

### Observation composition limitation

The framework has an open observation-size contract issue (`OBS-001`): leader or mechanism contributions may not match declared observation dimensions, and dynamically sized compositions are not fully supported. New mechanisms should **not hardcode** dimensions or silently pad observations in production without defining a consistent contract. Tests covering currently unsupported combinations remain tracked separately.

## Fishery-specific dynamics and fitness

The fishery example (`examples/bilevel_fishery/`) models a renewable stock, harvesting, restoration, and regulatory mechanisms. Its ecological equations and agent-specific choices are domain code, **not part of the generic `MultiAgentEnv` contract**.

`FisheryRegulatedEnv.reset_fishery` initializes the current stock in `state["fish"]` and harvest in `state["harvest"]`. The former `state["fish_start"]` field is not part of this reset result; the fishery transition reconstructs its required starting stock from the updated state and flows.

`FisheryRegulatorEnv.reward` selects the configured `train` or `eval` metrics and walks them by mechanism, seed, and episode. In the current implementation it reads these **reduced** episode fields:

```text
reward_mean
fish_norm_next_mean
H_realized
MSY
```

It converts harvest to a fraction of maximum sustainable yield, evaluates normalized fish biomass against the sustainability threshold, and summarizes the trailing `fitness_tail_steps` entries of the provided arrays. Episode summaries are aggregated into a `FitnessContext`. Its main objective is:

```text
harvest_score + sustainability_weight * mean_fish
```

The resulting fitness is an assessment of the learned population under a candidate mechanism; it should not be confused with an individual fisher's reward. The fishery is a **theoretical proof of concept**, not a calibrated real-world fisheries model.

## Metrics, queries, and reporting

Metric schemas (`core/metrics/`) define what an experiment logs and how individual values or histories are reduced. `MetricLogger` accumulates measurements and provides operations for inspecting or reducing them. Dynamic branches support separate records for agent, candidate, seed, and episode identifiers.

`Query` (`core/reporting/query.py`) selects metrics from a schema and specifies how series are organized. `Reporter` implementations resolve and render those queries. Included backends are:

| Backend | Module | Typical use |
| --- | --- | --- |
| CSV | `core/reporting/csv.py` | Structured files for independent analysis |
| TensorBoard | `core/reporting/tensor_board.py` | Local scalar curves |
| Weights & Biases | `core/reporting/wandb.py` | Experiment tracking and comparison |

CSV outputs are organized by configured output directory, project, world and reporting label. The reporter configuration—not obsolete script flags—determines destination and format.

For the fishery regulator, metrics used in fitness are **reduced episode fields**, even if additional step-series metrics also exist for reporting. Documentation and doctests should refer to the fields the implementation actually reads.

Missing or NaN episode-metric handling is separately tracked (`MET-002`). A test skip for this limitation is not a guarantee that gaps are universally supported throughout reduction.

## Declarative YAML configuration and CLI

The supported execution workflow uses `core/config/yaml.py` and `core/config/cli.py`:

```bash
uv run metamarl check examples/bilevel_fishery/config.yaml
uv run metamarl run examples/bilevel_fishery/config.yaml
```

`check` constructs and validates the experiment without starting training; `run` executes the configured workflow. The equivalent module entry point is `uv run python -m core.config.cli`.

YAML object construction supports `_target_`, `_args_`, `_calls_`, `_symbol_`, and `_tuple_`. Builder calls recorded in YAML configure the objects and their execution. CLI overrides use the form `path.to.setting=value` after the configuration path, with paths taken from the actual YAML file.

The former `examples/.../debug.py` commands have been removed from the release workflow. Python classes and builders are still available for programmatic extension, but the released examples are launched through YAML configurations.

## Reproducibility and RNG boundaries

MetaMARL provides seeded initialization and configurations, and the CLI initializes `PYTHONHASHSEED` when necessary. For mechanism comparisons, policy initialization and action-sampling RNG behavior matter independently of the environment's own randomness.

`IidGetActions` (`core/adaptors/ray/iid_get_actions.py`) maintains a **persistent CPU PyTorch RNG state per module ID, per connector instance**, initialized from the seed encoded in the policy/module identifier. This can align exploration draws for policies with matching seeds **when they consume random numbers in the same order and quantities**. It does not create a separate RNG stream for every environment or episode, and its `fork_rng(devices=[])` block does not isolate GPU random streams.

Therefore, do not claim bit-for-bit identical results merely from matching seeds. RLlib sampling schedules, batch sizes, asynchronous updates, numerical hardware, and differences in random-number consumption can break exact alignment. Reproducibility-sensitive contributions should test these assumptions under the intended execution configuration.

An APPO inner CSV-query/reporting limitation is currently tracked as `APPO-001`; a synchronization or threading cause is suspected but not established. Avoid treating that suspected cause as a verified implementation detail.

## Extending MetaMARL

### Add an environment

Create the domain-specific agent and configuration objects, define the shared state and `@reset`/`@transition` behavior, and supply metric schemas and reporting queries. Test the observation/action space contracts and transitions independently before adding distributed training. Add a regulator environment only if the experiment needs outer-loop mechanism optimization.

### Add a mechanism

Implement the appropriate `Mechanism` operations and declare its action space, targets, and composition semantics. Include tests for decoding, application, reset/observation effects, and interactions with other mechanisms. Be mindful of `OBS-001` when changing observation contributions.

### Add an optimizer

Implement or adapt the `Optimizer`/`OptimizerConfig` contract without putting algorithm-specific assumptions into domain code. For RLlib-based algorithms, isolate version-sensitive behavior in `core/adaptors/ray/` and provide unit and integration tests.

### Add a reporter

Implement the `Reporter` contract and its `ReporterConfig` construction, and test both individual query resolution and output behavior. `core/reporting/csv.py` provides a compact reference implementation.

## Testing and known limitations

The tests are grouped under `tests/` by framework component, alongside example and integration tests. Pytest also executes doctest examples in `core/` and `examples/`. Run:

```bash
uv run python -m pytest
uv run python -m pytest -m "not integration and not notebook"
uv run ruff check --no-fix .
uv run ruff format --check .
```

CI separates lint, unit, integration, and notebook checks. Known skipped tests use stable issue IDs: `OBS-001` (observation composition), `MET-001` (logger-optional fishery transition), `MET-002` (missing reduced metrics), and `APPO-001` (inner APPO reporting). These limitations should be fixed and the tests re-enabled rather than silently deleted.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution expectations and [README.md](../README.md) for project scope, authorship, and release information.
