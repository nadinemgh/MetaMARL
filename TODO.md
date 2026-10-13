# Visualization feature branch — handoff TODO

Reconciliation of 2026-10-05: every box of the visualization plan and of the mechanism plan below was compared with the code of the branch `feature/social-influence-testing-v2`, and only markers were appended. A ticked box ends with `done in <hash>` and the symbol that exists now ("realised differently" means the goal was met another way, for example with reduction tokens instead of "*"); an unticked box ending with `partly done in <hash>` says what is missing, `obsolete` gives the reason and the commit, and no marker means the item is still open. "Already checked" qualifies a box the plan author had ticked, and "answered" closes a question asked in the text.

This document is the implementation handoff for the visualization/metrics
feature branch.

The target architecture is:

```text
WHAT DATA EXISTS        -> MetricSchema
HOW DATA ACCUMULATES    -> Metric / MetricLogger
WHAT DATA TO SELECT     -> Query
HOW IT IS DISPLAYED     -> Reporter
```

The branch is not complete until W&B reproduces the dev visualizations,
dynamic runtime keys are queryable, ES advanced plots are supported, tests are
in place, and CSV/TensorBoard reporters are complete.

# Mechanism abstraction branch — research engineer handoff TODO

## Scope

This branch redesigns regulation as explicit mechanism algorithms that intervene
on the agent/environment loop through three channels:

\[
\mathcal M_\theta
=
\left(
\mathcal M_\theta^O,
\mathcal M_\theta^A,
\mathcal M_\theta^R
\right)
\]

with:

\[
o_t^\* = \mathcal M_\theta^O(s_t, o_t),
\qquad
a_t^\* = \mathcal M_\theta^A(s_t, a_t),
\qquad
r_t^\* = \mathcal M_\theta^R(r_t, s_t, a_t^\*, s_{t+1}).
\]

The implementation has been designed and mostly written, but **has not yet been
validated end-to-end against `dev`**. Treat the current code as an integration
branch, not as a validated replacement.

Primary concerned files:

```text
core/envs/hooks.py
core/envs/marl_regulated.py
core/envs/regulated.py
core/mechanism/algorithms/penalty.py
core/mechanism/algorithms/quota.py
core/mechanism/algorithms/social_influence.py
core/mechanism/algorithms/subsidy.py
core/mechanism/base.py
core/mechanism/composition/chained_mechanism.py
core/mechanism/composition/parallel_mechanism.py
core/mechanism/space.py
core/types.py
```

The first goal is **not new features**. The first goal is to make this complete
abstraction run end-to-end, cover the concerned modules with tests, and verify
that the new abstraction preserves benchmark behavior.

---

# 0. Definition of done

- [ ] Environment plots on the feature branch reproduce the dev plots/data.
- [ ] Inner optimizer plots reproduce the dev plots/data.
- [ ] ES plots reproduce the dev plots/data.
- [ ] Mechanism IDs, seed IDs, episode IDs, policy IDs, agent IDs, and ES
      parameter names do not need to be hard-coded into user queries. — partly done in `bd4728b`: the example queries reach mechanism, seed, episode and policy IDs through the `ReduceProtocol.SERIES` and `ReduceProtocol.MEAN` tokens; the searched ES parameter `quota` is still named in its query, and no per-agent query exists
- [x] `Query` supports runtime dict keys with `"*"`. — done in `fdd2567`: realised with reduction tokens instead of "*": `ReduceProtocol.SERIES` expands a dynamic dict node into one group per key and `ReduceProtocol.MEAN` averages over it, both in `Reporter._resolve_path`
- [x] Mean ±1 std across seeds works per mechanism. — done in `bd4728b`: `Query.error="std"` with an `error_path` ending at `by_seed` gives the per-mechanism std band, as used by `INNER_QUERIES`; resolution is tested by `test_std_per_mechanism_over_seeds` in `70c5244`
- [ ] Train-vs-eval shaded plots work per mechanism. — partly done in `bd4728b`: the ES-level query "Final train vs eval episode return" draws train and eval in one figure with one curve per mechanism, but `error_path` names one branch, so only the eval path carries the std band
- [x] ES cumulative parameter scatter plots work across every candidate and
      generation. — done in `cd28820`: the query "Candidate fitness vs quota" (final form in `34ae6f6`) plots every candidate of every generation in one scatter, coloured by outer iteration through `Query.color`
- [ ] ES parallel-coordinates plot is supported.
- [ ] Unit tests cover MetricLogger, schema polymorphism, Query resolution,
      reporters, environment/Ray/ES integration, and dynamic wildcards. — partly done in `7e4b2de`: unit suites exist for the metric logger (`7e4b2de`), `Query` and the reporters (`70c5244`), the ES payload (`1cdf19f`) and the Ray payload (`9c22690`), plus a debug-script smoke test (`7dccabd`); there is no dev-parity test and no test of a real Ray run that checks the IDs
- [x] CSV export is implemented and tested. — done in `f863b4e`: `CSVReporter` writes one long-form file per query; tested in `tests/reporting/test_csv_reporter.py` (`70c5244`)
- [x] TensorBoard reporting is implemented and tested. — done in `63bce35`: `TensorBoardReporter` writes one scalar tag per series; tested in `tests/reporting/test_tensorboard_reporter.py` (`70c5244`)
- [ ] Legacy dev W&B plotting utilities can be removed only after parity is
      proven. — partly done in `c02db47`: the stale W&B plotting utilities were removed, but no dev-versus-branch parity comparison has been run

---

# 1. P0 — validate exact dev plot parity

## 1.1 Environment-level plots

The environment reporter is attached to `FisheryMetricSchema` and should
render the environment horizon before the episode logger is destructively
reduced.

Required current queries:

```python
FISHERY_ENV_QUERIES = [
    Query(
        title="Fish biomass",
        x=("iter",),
        y=("fish_norm",),
    ),
    Query(
        title="Next fish biomass",
        x=("iter",),
        y=("fish_norm_next",),
    ),
    Query(
        title="Fish stock",
        x=("iter",),
        y=("fish_stock",),
    ),
    Query(
        title="Next fish stock",
        x=("iter",),
        y=("fish_stock_next",),
    ),
    Query(
        title="Biological growth",
        x=("iter",),
        y=("growth",),
    ),
    Query(
        title="Growth noise",
        x=("iter",),
        y=("growth_noise",),
    ),
    Query(
        title="Attempted harvest",
        x=("iter",),
        y=("H_attempted",),
    ),
    Query(
        title="Realized harvest",
        x=("iter",),
        y=("H_realized",),
    ),
    Query(
        title="Allowed harvest",
        x=("iter",),
        y=("allowed_harvest",),
    ),
    Query(
        title="Total usage normalized",
        x=("iter",),
        y=("total_usage_norm",),
    ),
    Query(
        title="Quota stress",
        x=("iter",),
        y=("quota_stress",),
    ),
    Query(
        title="Biomass at MSY",
        x=("iter",),
        y=("B_msy",),
    ),
    Query(
        title="Maximum sustainable yield",
        x=("iter",),
        y=("MSY",),
    ),
    Query(
        title="Fishing mortality at MSY",
        x=("iter",),
        y=("F_msy",),
    ),
]
```

Current per-agent helper:

```python
def fishery_agent_queries(agent_id: str) -> list[Query]:
    base = ("by_agent", agent_id)

    return [
        Query(
            title=f"Reward — {agent_id}",
            x=("iter",),
            y=base + ("reward",),
        ),
        Query(
            title=f"Action — {agent_id}",
            x=("iter",),
            y=base + ("action",),
        ),
        Query(
            title=f"Observation — {agent_id}",
            x=("iter",),
            y=base + ("observation",),
        ),
        Query(
            title=f"Intrinsic utility — {agent_id}",
            x=("iter",),
            y=base + ("intrinsic_utility",),
        ),
        Query(
            title=f"Violation signal — {agent_id}",
            x=("iter",),
            y=base + ("violation_signal",),
        ),
        Query(
            title=f"Requested harvest — {agent_id}",
            x=("iter",),
            y=base + ("requested_harvest",),
        ),
        Query(
            title=f"Delivered harvest — {agent_id}",
            x=("iter",),
            y=base + ("delivered_harvest",),
        ),
        Query(
            title=f"Requested harvest fraction — {agent_id}",
            x=("iter",),
            y=base + ("requested_frac",),
        ),
        Query(
            title=f"Quota violation — {agent_id}",
            x=("iter",),
            y=base + ("quota_violation",),
        ),
        Query(
            title=f"Quota penalty — {agent_id}",
            x=("iter",),
            y=base + ("quota_penalty",),
        ),
        Query(
            title=f"Risk penalty — {agent_id}",
            x=("iter",),
            y=base + ("risk_penalty",),
        ),
    ]
```

### Environment acceptance checks

- [x] `env.logger.peek()` produces a horizon-length x series and aligned y
      series before `reduce()`. — done in `9c74d85`: `test_every_step_logs_one_value_per_dynamics_field` checks that `FisheryRegulatedEnv` leaves one value per step in every stock-level and per-agent series of `env.logger.peek()`
- [x] `env.reporter.report(env.logger.peek())` does not mutate or clear the
      logger. — done in `654dc20`: `log_and_report_episode_metrics` peeks, reports, then reduces; `test_report_does_not_mutate_the_metrics` (`70c5244`) shows that `Reporter.report` leaves the schema untouched
- [x] After reporting, `env.logger.reduce()` still returns the correct compiled
      episode metrics. — done in `a89fb77`: `test_episode_end_hook_reports_then_reduces` checks that the reduced episode metrics are still handed to RLlib after the report
- [ ] Agent query traces match the agent values shown on dev.
- [x] No environment-specific field silently disappears when the runtime
      `FisheryMetricSchema` subtype is materialized. — done in `e2ce117`: `TestSpecialisationKeepsPushedValues` shows that specialising a node to a subclass keeps what was pushed, and `test_every_step_logs_one_value_per_dynamics_field` (`9c74d85`) lists the `FisheryMetricSchema` fields the environment logs
- [x] No `FisheryAgentMetricSchema` field disappears at the deeper `by_agent`
      runtime subtype. — done in `e2ce117`: `test_values_pushed_under_a_dynamic_id_survive` covers the dynamic level, and the environment test (`9c74d85`) checks `requested_harvest` and `delivered_harvest` for every agent in `by_agent`

### Data/schema gaps versus old dev environment plotting

The dev environment plotting code referenced some fields not present in the
provided current `FisheryMetricSchema`, including:

- `full_required_harvest`
- `min_demand_frac`
- `max_demand_frac`
- some environment-level `intrinsic_utility` usage
- named observation/info components

For exact parity:

- [ ] Decide whether each missing metric is environment-level or agent-level.
- [ ] Add shared values to `FisheryMetricSchema`.
- [ ] Add agent-specific values to `FisheryAgentMetricSchema`.
- [ ] Add named observation fields if exact old observation plots are required.
- [x] Add queries only after the schema field exists. — done in `3205b73`: `test_every_episode_statistic_named_by_a_query_is_filled_by_the_environment` and `test_every_environment_query_resolves_against_the_environment_logger` fail when a query names a field the environment does not fill

---

## 1.2 Inner optimizer: raw RLlib rollout plots

Required queries:

```python
RAY_ROLLOUT_QUERIES = [
    Query(
        title="Train reward",
        x=("iter",),
        y=(
            ("train", "rollout", "aggregate", "reward_mean"),
            ("train", "rollout", "aggregate", "reward_min"),
            ("train", "rollout", "aggregate", "reward_max"),
        ),
    ),
    Query(
        title="Train episode length",
        x=("iter",),
        y=(
            ("train", "rollout", "aggregate", "episode_len_mean"),
            ("train", "rollout", "aggregate", "episode_len_min"),
            ("train", "rollout", "aggregate", "episode_len_max"),
        ),
    ),
    Query(
        title="Train episodes",
        x=("iter",),
        y=("train", "rollout", "aggregate", "num_episodes"),
    ),
    Query(
        title="Train episodes lifetime",
        x=("iter",),
        y=("train", "rollout", "aggregate", "num_episodes_lifetime"),
    ),
    Query(
        title="Eval reward",
        x=("iter",),
        y=(
            ("eval", "rollout", "aggregate", "reward_mean"),
            ("eval", "rollout", "aggregate", "reward_min"),
            ("eval", "rollout", "aggregate", "reward_max"),
        ),
    ),
    Query(
        title="Eval episode length",
        x=("iter",),
        y=(
            ("eval", "rollout", "aggregate", "episode_len_mean"),
            ("eval", "rollout", "aggregate", "episode_len_min"),
            ("eval", "rollout", "aggregate", "episode_len_max"),
        ),
    ),
    Query(
        title="Eval episodes",
        x=("iter",),
        y=("eval", "rollout", "aggregate", "num_episodes"),
    ),
]
```

Acceptance:

- [ ] Train reward mean/min/max are numerically equivalent to dev.
- [ ] Eval reward mean/min/max are numerically equivalent to dev.
- [ ] Episode lengths and episode counts are equivalent.
- [ ] x-axis uses the intended RLlib iteration and is monotonic. — partly done in `04e9e7f`: the x axis is `iter`, which `RayOptimizer.train` fills with its own zero-based monotonic inner counter; RLlib's lifetime `training_iteration` is only printed in the log line

---

## 1.3 Inner optimizer: performance plots

Required queries:

```python
RAY_PERFORMANCE_QUERIES = [
    Query(
        title="Train environment steps",
        x=("iter",),
        y=(
            ("train", "performance", "env_steps_this_iter"),
            ("train", "performance", "env_steps_lifetime"),
        ),
    ),
    Query(
        title="Train agent steps",
        x=("iter",),
        y=(
            ("train", "performance", "agent_steps_this_iter_sum"),
            ("train", "performance", "agent_steps_lifetime_sum"),
        ),
    ),
    Query(
        title="Environment throughput",
        x=("iter",),
        y=("train", "performance", "env_steps_throughput"),
    ),
    Query(
        title="Training timing",
        x=("iter",),
        y=(
            ("train", "performance", "training_iteration_s"),
            ("train", "performance", "training_step_s"),
            ("train", "performance", "sample_s"),
            ("train", "performance", "learner_update_s"),
        ),
    ),
    Query(
        title="Weights sequence number",
        x=("iter",),
        y=("train", "performance", "weights_seq_no"),
    ),
    Query(
        title="Eval environment steps",
        x=("iter",),
        y=(
            ("eval", "performance", "env_steps_this_iter"),
            ("eval", "performance", "env_steps_lifetime"),
        ),
    ),
    Query(
        title="Eval agent steps",
        x=("iter",),
        y=(
            ("eval", "performance", "agent_steps_this_iter_sum"),
            ("eval", "performance", "agent_steps_lifetime_sum"),
        ),
    ),
    Query(
        title="Eval weights sequence number",
        x=("iter",),
        y=("eval", "performance", "weights_seq_no"),
    ),
]
```

Old dev also had timing/throughput fields such as sync-weight time and
learn-throughput depending on the RLlib result version.

- [ ] Compare current `PerformanceSchema` to the exact dev fields.
- [ ] Add missing fields only if exact dev parity requires them.
- [x] Do not read raw RLlib dictionaries directly from the reporter once the
      schema adaptor owns those mappings. — done in `04e9e7f`: `build_rollout`, `build_learner` and `build_performance` in `core/adaptors/ray/utils.py` own the RLlib mapping, and the reporters read only `MetricSchema` paths

---

## 1.4 Inner optimizer: per-policy learner plots

Current helper:

```python
def ray_policy_queries(policy_id: str) -> list[Query]:
    base = ("train", "learner", "by_policy", policy_id)

    return [
        Query(
            title=f"Batch size — {policy_id}",
            x=("iter",),
            y=base + ("batch_size",),
        ),
        Query(
            title=f"Total loss — {policy_id}",
            x=("iter",),
            y=base + ("total_loss",),
        ),
        Query(
            title=f"Residual variance — {policy_id}",
            x=("iter",),
            y=base + ("residual_variance",),
        ),
        Query(
            title=f"Sample staleness — {policy_id}",
            x=("iter",),
            y=base + ("sample_staleness",),
        ),
        Query(
            title=f"Policy loss — {policy_id}",
            x=("iter",),
            y=base + ("policy_loss",),
        ),
        Query(
            title=f"Policy entropy — {policy_id}",
            x=("iter",),
            y=base + ("policy_entropy",),
        ),
        Query(
            title=f"Policy entropy coefficient — {policy_id}",
            x=("iter",),
            y=base + ("policy_entropy_coeff",),
        ),
        Query(
            title=f"Policy relative entropy — {policy_id}",
            x=("iter",),
            y=base + ("policy_relative_entropy",),
        ),
        Query(
            title=f"Entropy pressure — {policy_id}",
            x=("iter",),
            y=base + ("entropy_pressure",),
        ),
        Query(
            title=f"Policy KL — {policy_id}",
            x=("iter",),
            y=base + ("policy_kl",),
        ),
        Query(
            title=f"Policy KL coefficient — {policy_id}",
            x=("iter",),
            y=base + ("policy_kl_coeff",),
        ),
        Query(
            title=f"Value loss — {policy_id}",
            x=("iter",),
            y=base + ("value_loss",),
        ),
        Query(
            title=f"Value mean — {policy_id}",
            x=("iter",),
            y=base + ("value_mean",),
        ),
        Query(
            title=f"Value target — {policy_id}",
            x=("iter",),
            y=base + ("value_target",),
        ),
        Query(
            title=f"Gradient norm — {policy_id}",
            x=("iter",),
            y=base + ("gradient_norm",),
        ),
        Query(
            title=f"Gradient noise — {policy_id}",
            x=("iter",),
            y=base + ("gradient_noise",),
        ),
    ]
```

Acceptance:

- [x] Every policy in `LearnerSchema.by_policy` can be plotted. — done in `bd4728b`: the learner queries of `INNER_QUERIES` reach every policy of `LearnerSchema.by_policy` through `ReduceProtocol.SERIES`
- [x] Policies do not have to be manually hard-coded after wildcard support. — done in `bd4728b`: policies are reached with the `ReduceProtocol.SERIES` token instead of a "*" wildcard, so none is hard-coded
- [x] `__all_modules__` / aggregate learner entries are handled intentionally:
      either include with a clear label or exclude explicitly. — done in `04e9e7f`: `build_learner` skips the `__all_modules__` entry explicitly and keeps one entry per policy module
- [ ] Loss/entropy/KL/gradient values match dev for the same deterministic run.

Old dev learner fields included names such as `kl`, `entropy`, `vf_loss`,
`policy_loss`, `total_loss`, `vf_explained_var`, `grad_gnorm`, `cur_lr`,
and `cur_kl_coeff`.

Current schema uses normalized names such as `policy_kl`, `policy_entropy`,
`value_loss`, and `gradient_norm`.

- [ ] Make a documented mapping table from dev metric names to new schema names.
- [ ] Add genuinely missing metrics if exact parity requires them.
- [ ] Do not duplicate equivalent metrics under two names without a reason.

---

# 2. P0 — dynamic dictionary key support in Query

Current limitation:

```python
Query(
    ...,
    y=(
        "train",
        "rollout",
        "by_mechanism",
        "0",       # hard-coded
        "by_seed",
        "100",     # hard-coded
        ...
    ),
)
```

Target:

```python
Query(
    ...,
    y=(
        "train",
        "rollout",
        "by_mechanism",
        "*",
        "by_seed",
        "*",
        ...
    ),
)
```

## 2.1 Required wildcard semantics

- [x] `"*"` matches keys only at dynamic dict nodes. — done in `fdd2567`: realised with tokens: `ReduceProtocol.SERIES` and `MEAN` apply only at dynamic dict nodes, and a token on a static field raises (`test_reduction_token_on_a_schema_is_a_type_error`, `70c5244`)
- [x] Static schema fields are not accidentally wildcarded. — done in `fdd2567`: a token on a static schema field raises a `TypeError` instead of expanding (`test_reduction_token_on_a_schema_is_a_type_error`, `70c5244`)
- [x] One wildcard expands to one trace per matched runtime key. — done in `fdd2567`: `ReduceProtocol.SERIES` gives one group, hence one trace, per runtime key (`test_series_expands_one_group_per_dynamic_id`, `70c5244`)
- [x] Multiple wildcards retain their bindings. — done in `fdd2567`: nested tokens chain their groups (`test_two_levels_of_series_chain_the_groups`, `70c5244`)
- [x] Expansion order is deterministic (sort keys or preserve a documented
      insertion order). — done in `fdd2567`: groups are sorted by key whatever the insertion order (`test_groups_come_out_sorted_whatever_the_insertion_order`, `70c5244`)
- [x] Missing dynamic branches produce a useful error or an empty match
      according to an explicit policy. — done in `fdd2567`: an empty dynamic node resolves to no series (`test_series_over_an_empty_dynamic_node_is_empty`), `Reporter.report` logs and skips a query that fails (`7e6bd39`), and an unknown path raises a `KeyError` that names it
- [x] An exact concrete key continues to work unchanged. — done in `fdd2567`: `test_concrete_path_resolves_to_the_empty_group` (`70c5244`) and the ES query path through `by_parameter` and the key `quota` show that concrete keys still work
- [x] Wildcard expansion does not mutate the schema/MetricLogger. — done in `70c5244`: `test_report_does_not_mutate_the_metrics`
- [x] Wildcard resolution works on runtime subtype nodes. — done in `19125d3`: `test_every_query_of_the_debug_script_renders` runs the debug script and fails if any query, including those read through the runtime-specialised `inner` slot, cannot be rendered

## 2.2 x/y binding

This is critical for ES scatter plots.

Given:

```python
x=(
    "by_mechanism",
    "*",
    "by_parameter",
    "fixed_quota",
    "value",
)

y=(
    "by_mechanism",
    "*",
    "fitness",
)
```

the same wildcard binding must be used on both sides:

```text
x m0 <-> y m0
x m1 <-> y m1
x m2 <-> y m2
x m3 <-> y m3
```

Never form a Cartesian product of candidate x values and candidate y values.

Tests must make candidate values obviously distinct so a binding error cannot
pass accidentally.

---

# 3. P0 — mechanism mean ±std across seeds

Desired user-facing query:

```python
Query(
    title="Train fish biomass by mechanism ±1 std across seeds",
    x=("iter",),
    y=(
        "train",
        "rollout",
        "by_mechanism",
        "*",
        "by_seed",
        "*",
        "by_episode",
        "*",
        "fish_norm",
    ),
    reduce="mean",
    error="std",
)
```

Expected output:

```text
m0 mean + shaded ±1 std
m1 mean + shaded ±1 std
m2 mean + shaded ±1 std
...
```

The implementation must distinguish:

- grouping dimension: mechanism;
- replicate/reduction dimensions: seed and, if needed, episode.

A naïve wildcard implementation that averages every mechanism and every seed
into one line is incorrect.

## 3.1 Resolve the `SeedRolloutSchema.aggregate` question

The requested older-style path was:

```python
(
    "by_mechanism",
    "0",
    "by_seed",
    "100",
    "aggregate",
    "fish_norm",
)
```

But the supplied schema is:

```python
class SeedRolloutSchema(MetricSchema):
    by_episode: dict[EpisodeID, EpisodeRolloutSchema]
```

There is no `aggregate`.

Choose one: — answered: Option B was taken, `SeedRolloutSchema` holds only `by_episode` and has no `aggregate` (`04e9e7f`)

### Option A — add seed aggregate

```python
class SeedRolloutSchema(MetricSchema):
    aggregate: EpisodeRolloutSchema
    by_episode: dict[EpisodeID, EpisodeRolloutSchema] = Field(
        default_factory=dict
    )
```

Then update the Ray adaptor to populate it.

### Option B — no schema change

Keep `by_episode` only and let Query resolution aggregate wildcard-matched
episode leaves.

Acceptance:

- [x] The choice is documented. — done in `a831019`: the `SeedRolloutSchema` docstring documents Option B, a level that holds only `by_episode`
- [x] Tests use the final supported path only. — done in `9c22690`: `test_payloads_accumulate_in_the_logger_and_reduce_to_the_last_iteration` reads `by_seed[...].by_episode[...]`, the only supported path
- [x] No tutorial/example advertises a nonexistent `aggregate` field. — done in `bd4728b`: the example queries average `by_episode` with `ReduceProtocol.MEAN`, and none reads a seed-level `aggregate` field

---

# 4. P0 — train-vs-eval shaded mechanism plots

Target query:

```python
Query(
    title="Fish biomass: train vs eval by mechanism ±1 std across seeds",
    x=("iter",),
    y=(
        (
            "train",
            "rollout",
            "by_mechanism",
            "*",
            "by_seed",
            "*",
            "by_episode",
            "*",
            "fish_norm",
        ),
        (
            "eval",
            "rollout",
            "by_mechanism",
            "*",
            "by_seed",
            "*",
            "by_episode",
            "*",
            "fish_norm",
        ),
    ),
    reduce="mean",
    error="std",
)
```

Dev behavior to reproduce:

- [x] train and eval appear in the same figure; — done in `bd4728b`: "Final train vs eval episode return" and its siblings in `ES_QUERIES` put the train and eval paths in one figure
- [x] one mean curve per phase × mechanism; — done in `bd4728b`: those queries give one curve per mechanism and phase (`by_mechanism` with `ReduceProtocol.SERIES`, seeds and episodes averaged with `ReduceProtocol.MEAN`)
- [ ] ±1 std shaded band across seeds; — partly done in `bd4728b`: `error_path` names one branch, so in the combined train/eval queries only the eval path gets the std band
- [x] mechanism identity is distinguishable; — done in `7f06f47`: `Reporter._series_label` appends the mechanism ID to the legend name
- [x] train/eval identity is distinguishable; — done in `bd4728b`: `legend_labels=("train", "eval")` names the two phases, and the reporters give each y path its own colour
- [x] deterministic legend order; — done in `fdd2567`: groups come out sorted and y paths keep their declared order
- [ ] horizon version uses environment step;
- [ ] over-training version uses RLlib training iteration; — partly done in `04e9e7f`: x is `iter`, the optimizer's own iteration counter, not RLlib's lifetime `training_iteration`
- [x] no W&B-specific grouping logic is required in the optimizer. — done in `41d9abf`: the ES optimizer pushes an `ESSchema` and renders through the `Reporter`; `core/optimizers` makes no W&B call

Required target queries for the primary fishery metrics:

```python
TARGET_TRAIN_EVAL_QUERIES = [
    Query(
        title="Fish biomass: train vs eval by mechanism ±1 std",
        x=("iter",),
        y=(
            (
                "train", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "fish_norm",
            ),
            (
                "eval", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "fish_norm",
            ),
        ),
        reduce="mean",
        error="std",
    ),
    Query(
        title="Realized harvest: train vs eval by mechanism ±1 std",
        x=("iter",),
        y=(
            (
                "train", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "H_realized",
            ),
            (
                "eval", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "H_realized",
            ),
        ),
        reduce="mean",
        error="std",
    ),
    Query(
        title="Reward: train vs eval by mechanism ±1 std",
        x=("iter",),
        y=(
            (
                "train", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "reward_mean",
            ),
            (
                "eval", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "reward_mean",
            ),
        ),
        reduce="mean",
        error="std",
    ),
    Query(
        title="Quota stress: train vs eval by mechanism ±1 std",
        x=("iter",),
        y=(
            (
                "train", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "quota_stress",
            ),
            (
                "eval", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "quota_stress",
            ),
        ),
        reduce="mean",
        error="std",
    ),
]
```

Extend the same pattern to the remaining fishery environment fields only when
they are useful and present in the schema.

---

# 5. P0 — ES exact dev plot parity

The old dev ES visualization had one accumulated history row per evaluated
candidate:

```text
generation
mechanism_idx
fitness
sigma
parameter_0
parameter_1
...
```

It produced:

1. fitness over outer generations;
2. one cumulative fitness-vs-parameter scatter per optimized parameter;
3. cumulative parallel coordinates;
4. scalar series for sigma;
5. search mean per parameter;
6. global-best fitness;
7. global-best candidate parameter values;
8. generation-best parameter values;
9. all-generations table.

The new logger should remain the source of truth; do not reintroduce a separate
global `_ES_HISTORY_TABLES` history cache.

---

## 5.1 ES schema verification

The supplied `ESSchema` includes:

- `sigma`
- `population_size`
- `fitness_mean`
- `fitness_best`
- `best_mechanism_idx`
- `best_fitness_global`
- `by_mechanism`
- `search_mean`
- `global_best`
- `inner`

### Required checks

- [x] `generation` is explicitly present and is `ReduceProtocol.SERIES`, or the
      entire implementation consistently uses inherited `iter`. — done in `f52e7c6`: `ESSchema.generation` is a `ReduceProtocol.SERIES` field filled by `ESOptimizer._to_logger_payload` next to the inherited `iter`
- [x] The optimizer and Query use the same x field. — done in `f52e7c6`: `_to_logger_payload` fills `iter` and `generation` with the same value, and `ES_QUERIES` read `iter`
- [x] Add `generation_best` for exact dev parity. — done in `5796732`: `ESSchema.generation_best` holds the parameters of the best candidate of each generation, filled in `_to_logger_payload`
- [ ] Consider renaming the type alias used for `search_mean` and
      `global_best` keys from `MechanismID` to `ParameterName`; those dict keys
      are parameter names, not mechanism IDs. — partly done in `5796732`: `generation_best` is keyed by the new `ParameterName` alias, while `search_mean` and `global_best` still use `MechanismID`

Recommended additions if not already present:

```python
ParameterName: TypeAlias = str

class ESSchema(MetricSchema):
    generation: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )

    generation_best: dict[ParameterName, ESParameterSchema] = Field(
        default_factory=dict
    )
```

Populate:

```python
generation_best={
    parameter_name: ESParameterSchema(
        value=float(population[best_idx, parameter_idx])
    )
    for parameter_idx, parameter_name in enumerate(parameter_names)
}
```

---

## 5.2 ES fitness-over-generations queries

Fishery acceptance fixture:

```python
ES_CANDIDATE_IDS = ("0", "1", "2", "3")
```

Required query data:

```python
Query(
    title="Fitness over outer optimization iterations",
    x=("generation",),
    y=(
        ("by_mechanism", "0", "fitness"),
        ("by_mechanism", "1", "fitness"),
        ("by_mechanism", "2", "fitness"),
        ("by_mechanism", "3", "fitness"),
        ("fitness_mean",),
        ("fitness_best",),
    ),
)

Query(
    title="Global best fitness",
    x=("generation",),
    y=("best_fitness_global",),
)

Query(
    title="ES sigma",
    x=("generation",),
    y=("sigma",),
)
```

Exact visual parity:

- [x] candidate fitness = marker traces; — done in `cd28820`: "Fitness over outer optimization iterations" draws the candidates with `plot_modes` "markers"
- [x] generation mean = line + markers; — done in `cd28820`: the generation mean of the same query is a `ReduceProtocol.MEAN` path drawn in "lines+markers" with a std band
- [x] generation best = line + markers; — done in `cd28820`: `fitness_best` is drawn in "lines+markers" under the label "Generation best"
- [ ] candidate hover contains outer generation, candidate/mechanism index, and
      fitness; — partly done in `cd28820`: x is the outer iteration and y the fitness, but the reporter sets no custom hover text, so the candidate index is not shown
- [x] figure is cumulative over all completed generations. — done in `1cdf19f`: the ES loop reports the accumulated series after every generation (`test_one_report_per_generation_with_the_accumulated_series`)

This likely requires trace style metadata or a specialized plot query. Data
selection alone is not enough to reproduce marker-vs-line semantics.

---

## 5.3 ES search mean and global best

Required:

```python
Query(
    title="ES search mean",
    x=("generation",),
    y=(
        ("search_mean", "fixed_quota", "value"),
        ("search_mean", "restoration_subsidy", "value"),
    ),
)

Query(
    title="Global-best mechanism parameters",
    x=("generation",),
    y=(
        ("global_best", "fixed_quota", "value"),
        ("global_best", "restoration_subsidy", "value"),
    ),
)
```

After `generation_best` is added:

```python
Query(
    title="Generation-best mechanism parameters",
    x=("generation",),
    y=(
        ("generation_best", "fixed_quota", "value"),
        ("generation_best", "restoration_subsidy", "value"),
    ),
)
```

Parameter names are runtime-defined by `MechanismSpace`.

Target dynamic form:

```python
Query(
    title="ES search mean",
    x=("generation",),
    y=("search_mean", "*", "value"),
)

Query(
    title="Global-best mechanism parameters",
    x=("generation",),
    y=("global_best", "*", "value"),
)

Query(
    title="Generation-best mechanism parameters",
    x=("generation",),
    y=("generation_best", "*", "value"),
)
```

- [ ] No optimized parameter name must be hard-coded after wildcard support. — partly done in `34ae6f6`: `ReduceProtocol.SERIES` can expand `global_best` and `generation_best` without naming a parameter, but no fishery query reads them, and the scatter names `quota`, guarded by `test_every_parameter_named_by_a_query_is_searched_by_the_es`

---

## 5.4 ES fitness-vs-parameter scatter

Current concrete smoke-test generator:

```python
def es_parameter_fitness_queries(
    candidate_ids=("0", "1", "2", "3"),
    parameter_names=("fixed_quota", "restoration_subsidy"),
):
    queries = []

    for parameter_name in parameter_names:
        for candidate_id in candidate_ids:
            queries.append(
                Query(
                    title=f"Fitness vs {parameter_name} — candidate {candidate_id}",
                    x=(
                        "by_mechanism",
                        candidate_id,
                        "by_parameter",
                        parameter_name,
                        "value",
                    ),
                    y=(
                        "by_mechanism",
                        candidate_id,
                        "fitness",
                    ),
                )
            )

    return queries
```

This verifies that the data paths work, but it does **not** reproduce the dev
figure exactly.

Exact target:

```python
Query(
    title="Fitness vs fixed_quota",
    x=(
        "by_mechanism",
        "*",
        "by_parameter",
        "fixed_quota",
        "value",
    ),
    y=("by_mechanism", "*", "fitness"),
)
```

Required semantics:

- [x] all candidates in one cumulative scatter; — done in `cd28820`: one scatter query holds all candidates (final form in `34ae6f6`)
- [x] all generations included; — done in `cd28820`: the query reads whole series, so every generation reported so far is included
- [x] x/y wildcard bindings aligned by candidate ID; — done in `70c5244`: `_resolve_query` requires the x and y groups to match (`test_dynamic_x_and_y_groups_must_match`)
- [x] each point carries generation metadata; — done in `cd28820`: `Query.color` resolves one colour value per point; the scatter uses `color=("iter",)`
- [x] point color represents outer generation, as on dev; — done in `cd28820`: the scatter query sets `colorscale="Viridis"` on the outer iteration
- [x] colorbar title identifies outer iteration; — done in `cd28820`: `color_label="Outer iteration"` titles the colour bar (`test_color_query_widens_legend_margin_and_adds_a_color_bar`, `70c5244`)
- [ ] hover includes outer iteration, candidate/mechanism index, parameter
      value, and fitness; — partly done in `cd28820`: the default hover shows the parameter value and the fitness, but the reporter sets no custom text with the iteration or the candidate index
- [ ] one figure per runtime optimized parameter. — partly done in `34ae6f6`: one scatter exists, for the searched parameter `quota`; no figure is generated per runtime parameter

The current `Query` has no z/color metadata. Implement one of:

- [x] optional query metadata path for color/group; — done in `cd28820`: `Query.color`, `Query.color_label` and `Query.colorscale` carry the colour metadata
- [ ] `ScatterQuery`; — obsolete: no `ScatterQuery` class was added; the colour path of `Query` (`cd28820`) covers the need
- [ ] generic named-dimension query consumed by the W&B reporter. — obsolete: no separate named-dimension query was added; the W&B reporter consumes `Query.color` (`cd28820`)

Do not bury generation lookup inside a W&B-only helper if the same semantic
plot should be portable to another backend.

---

## 5.5 ES parallel coordinates

Dev behavior:

- every evaluated candidate is one line;
- each optimized mechanism parameter is one axis;
- fitness is the final axis;
- fitness also controls line color;
- axis ranges are dynamically padded;
- all generations accumulate.

The current `Query(x, y)` API cannot represent this.

Implement one of:

### Option A

```python
ParallelCoordinatesQuery(
    title="Parallel coordinates of evaluated mechanisms",
    dimensions=(
        ("by_mechanism", "*", "by_parameter", "*", "value"),
        ("by_mechanism", "*", "fitness"),
    ),
    color=("by_mechanism", "*", "fitness"),
)
```

### Option B

A generic table/multidimensional query that resolves a row per evaluated
candidate and lets reporters choose a parallel-coordinate renderer.

Acceptance:

- [ ] all optimized parameters appear exactly once;
- [ ] fitness appears exactly once as final axis;
- [ ] line color = fitness;
- [ ] rows remain aligned across parameter dimensions;
- [ ] cumulative data across generations;
- [ ] fixed-mode ES still uses the full default mechanism vector; — obsolete: the full-default-vector plotting arrays were removed from `ESOptimizer` in `41d9abf`; a fixed-mode generation now logs fitness only, with an empty `by_parameter`
- [ ] empty/constant ranges do not crash.

---

# 6. P0 — schema extension rules must be documented and tested

## Add shared environment metric

Subclass `EpisodeRolloutSchema`:

```python
class FisheryMetricSchema(EpisodeRolloutSchema):
    new_env_metric: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
```

## Add per-agent metric

Subclass `AgentEnvStepSchema` and override `by_agent`:

```python
class FisheryAgentMetricSchema(AgentEnvStepSchema):
    new_agent_metric: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

class FisheryMetricSchema(EpisodeRolloutSchema):
    by_agent: dict[str, FisheryAgentMetricSchema] = Field(
        default_factory=dict
    )
```

## Add learner metric

Add to `PolicyLearnerSchema`.

## Add runtime policy dimension

Use:

```python
by_policy: dict[PolicyID, PolicyLearnerSchema]
```

Do not add one static field per policy.

## Add performance metric

Add to `PerformanceSchema`.

## Add ES generation metric

Add to `ESSchema`, generally as `SERIES` if the plot needs the full outer
history.

## Add ES candidate metric

Add to `ESCandidateSchema`.

## Add ES parameter metric

Add to `ESParameterSchema`.

## Add inner optimizer-specific schema under ES

Keep:

```python
inner: Optional[MetricSchema] = None
```

and rely on runtime subtype binding.

Tests must prove:

```text
ESSchema.inner declared MetricSchema
              runtime RaySchema
                     -> train/eval
                     -> by_mechanism
                     -> by_seed
                     -> by_episode runtime FisheryMetricSchema
                     -> by_agent runtime FisheryAgentMetricSchema
```

---

# 7. P1 — MetricLogger unit tests

Create focused unit tests, not only integration tests.

Suggested files:

```text
tests/metrics/test_metric_logger_schema_build.py
tests/metrics/test_metric_logger_push_data.py
tests/metrics/test_metric_logger_dynamic.py
tests/metrics/test_metric_logger_peek_reduce.py
```

Required tests:

## 7.1 Schema build

- [x] static nested `MetricSchema` builds the correct node tree; — done in `7e4b2de`: `test_static_tree_and_refs`
- [x] `dict[ID, MetricSchema]` becomes a dynamic node; — done in `7e4b2de`: `test_push_materializes_dynamic_ids_independently`
- [x] leaf reducer metadata creates the correct Metric subclass; — done in `7e4b2de`: `test_static_tree_and_refs` checks `MeanMetric` and `SeriesMetric` against the declared reductions
- [x] `_refs` contains every materialized leaf path. — done in `7e4b2de`: `test_static_tree_and_refs` checks that `_refs` holds each leaf as the same object as `_tree`

## 7.2 Push leaf values

- [x] push scalar into existing leaf; — done in `7e4b2de`: `test_push_static_leaf_and_unknown_path`
- [x] skip `None`; — done in `7e4b2de`: `test_push_data_static_and_skips_none`
- [x] reject unknown field; — done in `7e4b2de`: `test_push_static_leaf_and_unknown_path` and `test_unknown_field_when_node_lacks_it`
- [x] reject incompatible value/schema. — done in `7e4b2de`: `test_schema_value_on_a_metric_leaf` and `test_dict_value_on_a_non_dynamic_node`

## 7.3 Dynamic dict materialization

- [x] first dynamic ID materializes its subtree; — done in `7e4b2de`: `test_push_materializes_dynamic_ids_independently`
- [x] second dynamic ID materializes independently; — done in `7e4b2de`: `test_push_materializes_dynamic_ids_independently`
- [x] runtime subclass of declared schema is accepted; — done in `7e4b2de`: `test_push_data_dynamic_and_runtime_subtype`
- [x] unrelated schema is rejected; — done in `7e4b2de`: `test_push_data_dynamic_and_runtime_subtype` expects a `TypeError` "not a subclass"
- [x] runtime schema cannot silently change for an already-bound ID. — done in `7e4b2de`: `test_push_data_dynamic_and_runtime_subtype` expects a `TypeError` "Runtime schema changed"

## 7.4 Static nested runtime subtype binding

This is the ES inner-optimizer regression test.

- [x] `ESSchema.inner` starts declared as `MetricSchema`; — done in `7e4b2de`: `test_static_nested_runtime_subtype_binding` asserts the `inner` slot starts as `MetricSchema`
- [x] first `RaySchema` push replaces/materializes the inner subtree; — done in `1cdf19f`: `test_the_inner_metrics_of_the_environment_are_forwarded` checks that `ESSchema.inner` takes the concrete schema of the first push
- [x] `train` and `eval` fields exist; — done in `93b8d34`: `RaySchema.train` and `RaySchema.eval` are optional `TrainSchema` and `EvalSchema` fields
- [x] second `RaySchema` push reuses the same subtree; — done in `1cdf19f`: the second generation keeps filling the same `inner` (`logged.inner.value == [7.0, 7.0]`)
- [x] second push accumulates metrics instead of resetting to length 1; — done in `1cdf19f`: the same test sees `[7.0, 7.0]`, so the second push accumulates
- [x] switching to an incompatible concrete subtype in the same logger raises. — done in `7e4b2de`: `test_static_nested_runtime_subtype_binding` expects a `TypeError` for an incompatible subtype

## 7.5 Deep runtime polymorphism

Push:

```text
ESSchema
 -> inner RaySchema
 -> train rollout
 -> by_mechanism["0"]
 -> by_seed["seed"]
 -> by_episode["episode"]
 -> FisheryMetricSchema
 -> by_agent["utilizer:0"]
 -> FisheryAgentMetricSchema
```

Assert deep fields such as:

```text
fish_norm
quota_stress
requested_harvest
quota_penalty
```

are registered and receive values.

## 7.6 `peek()`

- [x] non-destructive; — done in `7e4b2de`: `test_peek_is_non_destructive_and_reduce_is_destructive`
- [x] two consecutive peeks are equal; — done in `7e4b2de`: the same test asserts `logger.peek() == peeked`
- [x] SERIES history remains intact; — done in `7e4b2de`: `test_peeked_history_is_a_copy` and `test_peek_does_not_end_the_episode`
- [x] calling reporter after peek does not change logger contents. — done in `70c5244`: `test_report_does_not_mutate_the_metrics`

## 7.7 `reduce()`

- [x] destructive according to current Metric semantics; — done in `7e4b2de`: `test_peek_is_non_destructive_and_reduce_is_destructive`
- [x] resulting typed schema is correct; — done in `7e4b2de`: the same test checks `isinstance(peeked, Root)`, and `test_compile_serialises_a_runtime_subtype_with_its_extra_fields` covers runtime subtypes
- [x] empty reducer semantics are correct:
      Series `[]`, Mean `None`, Min `None`, Max `None`, Last `None`,
      Sum `0`, Count `0`. — done in `7e4b2de`: `test_reducing_an_untouched_logger_gives_the_empty_value_of_each_protocol`; no Count protocol exists

## 7.8 `_refs`

- [x] `_refs[path]` is the same leaf object as the corresponding `_tree` leaf; — done in `7e4b2de`: `test_static_tree_and_refs`
- [x] dynamic materialization updates `_refs`; — done in `7e4b2de`: `push` and `peek_value` read through `_refs`, so `test_push_materializes_dynamic_ids_independently` exercises it
- [x] reduction does not leave stale aliases. — done in `7e4b2de`: `test_a_second_reduce_without_new_pushes_is_empty` and `test_consecutive_episodes_do_not_leak_into_each_other`

---

# 8. P1 — Query unit tests

Suggested files:

```text
tests/reporting/test_query.py
tests/reporting/test_query_resolution.py
tests/reporting/test_query_wildcards.py
```

## 8.1 Constructor / validation

- [x] one-element path tuple is supported; — done in `70c5244`: `test_a_single_path_is_wrapped`
- [x] one y path; — done in `70c5244`: `test_a_single_path_is_wrapped`
- [x] multiple y paths; — done in `70c5244`: `test_a_tuple_of_paths_is_kept`
- [x] `error="std"` with `reduce="none"` raises; — done in `70c5244`, realised differently: there is no `reduce` argument; `Query.__post_init__` rejects `error="std"` without an `error_path` followed by `ReduceProtocol.MEAN` (`test_invalid_combination_is_rejected`)
- [x] malformed empty path raises or has documented behavior. — done in `70c5244`: `Query.__post_init__` raises a `ValueError` for an empty x or y path (`test_invalid_combination_is_rejected`)

## 8.2 Static resolution

- [x] root leaf; — done in `70c5244`: `test_concrete_path_resolves_to_the_empty_group`
- [x] nested leaf; — done in `70c5244`: `test_concrete_path_resolves_to_the_empty_group`
- [x] multiple y paths; — done in `70c5244`: `test_static_x_with_several_y_paths`
- [x] x/y length mismatch produces a clear error. — done in `70c5244`: `test_static_x_length_mismatch_names_x_y_and_group`

## 8.3 Mean/std

Synthetic series:

```python
seed_1 = [1.0, 2.0, 3.0]
seed_2 = [3.0, 4.0, 5.0]
```

Expected:

```python
mean = [2.0, 3.0, 4.0]
std = [1.0, 1.0, 1.0]
```

Assert exact output before testing W&B rendering.

## 8.4 Wildcards

- [x] one wildcard; — done in `70c5244`, with tokens instead of "*": `test_series_expands_one_group_per_dynamic_id`
- [x] two nested wildcards; — done in `70c5244`, with tokens: `test_two_levels_of_series_chain_the_groups`
- [x] concrete key + wildcard; — done in `34ae6f6`: `test_every_es_level_query_resolves` resolves a path that mixes `ReduceProtocol.SERIES` with the concrete key `quota`
- [x] deterministic match order; — done in `70c5244`: `test_groups_come_out_sorted_whatever_the_insertion_order`
- [x] no matches; — done in `70c5244`: `test_series_over_an_empty_dynamic_node_is_empty`
- [x] wildcard only applies to dynamic dict nodes. — done in `70c5244`: `test_reduction_token_on_a_schema_is_a_type_error` and `test_unsupported_dictionary_reduction`

## 8.5 Wildcard grouping

For:

```text
mechanism -> seed -> value
```

assert:

- [x] one group per mechanism; — done in `70c5244`: `test_mean_over_seeds_keeps_one_group_per_mechanism`
- [x] reduction across seeds only; — done in `70c5244`: `test_mean_over_seeds_keeps_one_group_per_mechanism`
- [x] mechanism groups do not get averaged together. — done in `70c5244`: `test_std_per_mechanism_over_seeds`

## 8.6 Wildcard x/y alignment

Synthetic candidate data:

```text
m0 parameter=0.1 fitness=10
m1 parameter=0.2 fitness=20
m2 parameter=0.3 fitness=30
```

Resolved scatter rows must be:

```text
(0.1, 10)
(0.2, 20)
(0.3, 30)
```

Never:

```text
(0.1, 20)
...
```

---

# 9. P1 — Reporter base tests

Suggested:

```text
tests/reporting/test_reporter_base.py
```

- [x] Reporter resolves every registered query against the configured schema. — done in `70c5244`: `test_every_query_is_resolved_and_forwarded_in_order`; `test_every_es_level_query_resolves` (`34ae6f6`) does it for the fishery ES queries
- [x] Empty query list is a no-op. — done in `70c5244`: `test_no_query_means_no_backend_call`
- [x] Missing path includes the full path in the error. — done in `70c5244`: `test_path_errors_name_the_path`
- [x] Multiple y series preserve labels/identity. — done in `70c5244`: `test_static_x_with_several_y_paths` and `test_each_y_path_has_its_own_color_label_and_mode`
- [x] Reduced mean/std data has expected shape. — done in `70c5244`: `test_mean_over_the_dynamic_node_is_pointwise` and `test_std_per_mechanism_over_seeds`
- [x] Wildcard metadata/bindings survive until backend `_report`. — done in `70c5244`: `test_every_query_is_resolved_and_forwarded_in_order` shows the groups reach `_report`
- [x] Reporter does not mutate input `MetricSchema`. — done in `70c5244`: `test_report_does_not_mutate_the_metrics`
- [x] Same accumulated SERIES may be reported repeatedly as it grows. — done in `1cdf19f`: `test_one_report_per_generation_with_the_accumulated_series`

---

# 10. P1 — W&B reporter tests

Use mocks/fakes; unit tests should not require a network connection.

Suggested:

```text
tests/reporting/test_wandb_reporter.py
```

Required:

- [x] simple line query logs under stable key; — done in `70c5244`: `test_report_logs_one_figure_under_the_sanitised_title`
- [x] multiple raw y series produce expected trace count; — done in `70c5244`: `test_one_trace_per_group_with_group_in_the_name`
- [x] mean/std creates mean + band; — done in `70c5244`: `test_error_band_is_a_closed_polygon_before_the_line`
- [x] dynamic wildcard trace labels contain mechanism/seed/policy/agent ID; — done in `70c5244`: `test_one_trace_per_group_with_group_in_the_name`
- [ ] repeated report with growing SERIES updates using the complete current
      history; — partly done in `a094e05`: the CSV and TensorBoard tests (`test_reporting_again_rewrites_the_file`, `test_a_growing_history_writes_only_the_new_points`) report a growing history; no W&B test does
- [ ] ES fitness plot trace count and types match dev;
- [ ] ES parameter scatter point count =
      generations × population size;
- [ ] ES scatter x/y candidate correspondence is exact; — partly done in `70c5244`: `_resolve_query` rejects mismatched x and y groups (`test_dynamic_x_and_y_groups_must_match`), but no ES-specific test checks the candidate pairing
- [ ] parallel coordinates dimensions are exact;
- [ ] constant fitness/parameter values do not crash range calculation;
- [x] no global W&B history table is required as source of truth. — done in `c02db47`: the global W&B history tables were removed, and the reporter draws from the logger's accumulated series

For dev parity, assert figure structure where practical:

```text
trace names
trace count
trace mode
x arrays
y arrays
band upper/lower
hover metadata fields
```

A screenshot comparison can be a secondary integration check, but numeric
trace assertions should be primary.

---

# 11. P1 — environment integration tests

Suggested:

```text
tests/integration/test_fishery_visualization.py
```

Small deterministic fixture:

```text
2 mechanisms
2 seeds
2 agents
short horizon
few training iterations
```

- [x] environment logger contains expected horizon length; — done in `9c74d85`: `test_every_step_logs_one_value_per_dynamics_field`
- [x] `FisheryMetricSchema` values match direct env values; — done in `9c74d85`: the dynamics tests compare the logged series with the values computed by hand (for example `test_realised_harvest_is_the_sum_of_the_delivered_catches`)
- [x] `by_agent` contains both agents; — done in `9c74d85`: `test_every_step_logs_one_value_per_dynamics_field` loops over every agent of `by_agent`
- [x] horizon reporter receives data before episode reduction; — done in `a89fb77`: `test_episode_end_hook_reports_then_reduces` shows that the reporter receives the raw history
- [x] reduction afterward still works. — done in `3205b73`: `test_the_series_fields_keep_every_step_of_the_episode` reduces after peeking

---

# 12. P1 — Ray/inner optimizer integration tests

Use a short deterministic run or a synthetic adaptor payload when full RLlib
would be too expensive.

- [ ] `RaySchema.train` populated; — partly done in `9c22690`: `RayOptimizer._to_logger_payload` is unit-tested on synthetic RLlib results, which cover the train branch; no test checks them on a real Ray run
- [ ] `RaySchema.eval` populated after explicit evaluation; — partly done in `9c22690`: `RayOptimizer._to_logger_payload` is unit-tested on synthetic RLlib results, which cover the eval branch; no test checks them on a real Ray run
- [ ] all mechanism IDs present; — partly done in `9c22690`: `RayOptimizer._to_logger_payload` is unit-tested on synthetic RLlib results, which cover the mechanism IDs; no test checks them on a real Ray run
- [ ] all seed IDs present; — partly done in `9c22690`: `RayOptimizer._to_logger_payload` is unit-tested on synthetic RLlib results, which cover the seed IDs; no test checks them on a real Ray run
- [ ] stable episode IDs present; — partly done in `9c22690`: `RayOptimizer._to_logger_payload` is unit-tested on synthetic RLlib results, which cover the episode IDs; no test checks them on a real Ray run
- [ ] per-policy learner IDs present; — partly done in `9c22690`: `RayOptimizer._to_logger_payload` is unit-tested on synthetic RLlib results, which cover the per-policy learner IDs; no test checks them on a real Ray run
- [ ] performance fields present; — partly done in `9c22690`: `RayOptimizer._to_logger_payload` is unit-tested on synthetic RLlib results, which cover the performance fields; no test checks them on a real Ray run
- [ ] train/eval query outputs match manually computed values;
- [ ] mechanism mean/std across seeds matches NumPy calculation. — partly done in `70c5244`: mean and std across seeds are tested on a synthetic schema (`test_std_per_mechanism_over_seeds`), not against a NumPy computation on a Ray payload

---

# 13. P1 — ES integration tests

Use a deterministic synthetic environment if possible.

Fixture:

```text
population size = 4
dimension = 2
3 generations
parameter names = fixed_quota, restoration_subsidy
```

- [x] one ES payload per generation; — done in `1cdf19f`: `test_one_report_per_generation_with_the_accumulated_series`
- [x] generation series length grows 1 -> 2 -> 3; — done in `1cdf19f`: `test_series_grow_by_one_value_per_generation`
- [x] candidate fitness series length grows 1 -> 2 -> 3; — done in `1cdf19f`: `test_series_grow_by_one_value_per_generation`
- [x] search mean series grows; — done in `1cdf19f`: `test_series_grow_by_one_value_per_generation`
- [x] global best is updated after current population evaluation; — done in `1cdf19f`: `test_mean_global_best_and_generation_best_come_from_different_sources`
- [x] logged mean/sigma correspond to the pre-update distribution that sampled
      the population; — done in `1cdf19f`: `test_the_logged_sigma_and_mean_are_those_the_population_was_drawn_from`
- [x] `inner` contains the concrete inner schema; — done in `1cdf19f`: `test_the_inner_metrics_of_the_environment_are_forwarded`
- [x] second generation does not reconstruct/reset `inner`; — done in `1cdf19f`: the same test sees `[7.0, 7.0]` over two generations, so `inner` is not rebuilt
- [ ] fitness plot has `3 * 4 = 12` candidate points;
- [ ] each parameter scatter has 12 points;
- [ ] parallel coordinates has 12 lines;
- [ ] global-best trajectory is monotonic non-decreasing for maximization.

Fixed-mode regression:

- [x] ES dimension 0 does not crash reporting; — done in `1cdf19f`: `test_fixed_mode_generation_completes`
- [ ] plotting payload uses the full default mechanism vector; — obsolete: the code that padded the plotting payload with the default mechanism vector was removed from `ESOptimizer` in `41d9abf`
- [x] parameter names match the default mechanism vector. — done in `1cdf19f`: `test_every_candidate_is_keyed_by_its_index_and_parameter_name`; `test_every_parameter_named_by_a_query_is_searched_by_the_es` (`34ae6f6`) ties the query names to `parameter_names`

---

# 14. P1 — CSV reporter implementation

CSV export is part of the feature definition and must be completed.

Suggested file:

```text
core/reporting/csv.py
```

The CSV reporter must consume the same `Query` contract.

## 14.1 Required behavior

- [x] implement Reporter subclass; — done in `f863b4e`: `CSVReporter`
- [x] configure output directory/path through `ReporterConfig`; — done in `f863b4e`: `CSVConfig(output_dir=...)` builds the reporter
- [x] create directories safely; — done in `70c5244`: `test_building_creates_the_directory`
- [x] stable file naming from query title/key; — done in `70c5244`: `test_title_is_sanitised_into_the_file_name`
- [x] append/update semantics documented; — done in `f863b4e`: the module docstring states that each report rewrites the whole file (`test_reporting_again_rewrites_the_file`, `70c5244`)
- [x] no W&B dependency; — done in `f863b4e`: `core/reporting/csv.py` imports only the standard library and the reporting base
- [x] scalar series export; — done in `70c5244`: `test_static_series_rows`
- [x] multiple raw series export; — done in `70c5244`: `test_static_series_rows` writes two series into one file
- [x] mean/std export; — done in `70c5244`: `test_error_column_holds_the_pointwise_std`
- [x] dynamic wildcard labels exported; — done in `70c5244`: `test_dynamic_groups_become_labelled_series`
- [ ] train/eval/mechanism/seed dimensions preserved as columns; — partly done in `f863b4e`: the dimensions appear inside the `series` label, not as separate columns
- [ ] ES candidate/parameter metadata preserved; — partly done in `f863b4e`: the candidate and parameter sit in the `series` label and the outer iteration in the `color` column, not in dedicated columns
- [x] flush/close lifecycle; — done in `f863b4e`: every file is closed after each report and `CSVReporter.close` has nothing left to do (`test_close_leaves_the_files_in_place`)
- [ ] safe behavior if process exits after partial run. — partly done in `f863b4e`: each report rewrites the whole file, so a partial run keeps the last complete report, but the write is in place rather than atomic

Recommended long-form representation:

```text
query
x
series
value
error_std
phase
mechanism
seed
episode
policy
agent
parameter
generation
```

Only populate dimensions relevant to the query.

For simple queries, a wide CSV may also be convenient:

```text
iter, reward_mean, reward_min, reward_max
```

Do not throw away dynamic identity just to force everything into wide format.

## 14.2 CSV tests

- [x] temp directory fixture; — done in `70c5244`: `tests/reporting/test_csv_reporter.py` writes into a temporary directory
- [x] single series; — done in `70c5244`: `test_static_series_rows`
- [x] multi-series; — done in `70c5244`: `test_static_series_rows`
- [x] mean/std; — done in `70c5244`: `test_error_column_holds_the_pointwise_std`
- [x] wildcard series labels; — done in `70c5244`: `test_dynamic_groups_become_labelled_series`
- [x] repeated report appends/updates correctly; — done in `70c5244`: `test_reporting_again_rewrites_the_file`
- [x] no duplicate header; — done in `70c5244`: `test_reporting_again_rewrites_the_file` expects a single header row
- [ ] NaN/None policy documented;
- [ ] output can be loaded by pandas and reconstruct expected series. — partly done in `70c5244`: the tests read the files back with the `csv` module, not with pandas

---

# 15. P1 — TensorBoard reporter implementation

TensorBoard support must be completed.

Suggested file:

```text
core/reporting/tensorboard.py
```

Use the same resolved Query result, not raw optimizer dictionaries.

## 15.1 Required behavior

- [x] Reporter subclass; — done in `63bce35`: `TensorBoardReporter`
- [x] `SummaryWriter` lifecycle; — done in `70c5244`: `test_building_is_lazy`, `test_writer_is_created_once_and_reused` and `test_close_closes_and_forgets_the_writer`
- [x] stable tag naming; — done in `63bce35`: the tag is the sanitised query title followed by the series label
- [x] single scalar/series using `add_scalar`; — done in `63bce35`: `TensorBoardReporter` writes each point with `add_scalar`
- [x] multiple related series using `add_scalars` where appropriate; — done in `63bce35`, realised differently: every series gets its own tag through `add_scalar`, and `add_scalars` is not used
- [x] dynamic mechanism/policy/agent labels represented in tags; — done in `70c5244`: `test_dynamic_groups_get_one_tag_each`
- [x] mean/std behavior documented; — done in `63bce35`: the module docstring states that the std goes under a `/std` suffix
- [x] train/eval grouping represented consistently; — done in `63bce35`: train and eval share the query title and differ by series label, so their tags follow one scheme
- [x] flush and close; — done in `70c5244`: `test_close_closes_and_forgets_the_writer`
- [x] no W&B imports. — done in `63bce35`: `core/reporting/tensor_board.py` imports no W&B module

Complex figures:

- line/scatter/shaded plots can be logged either as:
  - scalar families that TensorBoard renders natively; or
  - a rendered figure/image where native scalar APIs are insufficient.
- parallel coordinates likely requires image/figure rendering because
  TensorBoard does not have a native parallel-coordinate primitive.

- [ ] choose and document the complex-figure representation; — partly done in `63bce35`: the module docstring says that the colour path is ignored; no representation exists for scatter or parallel-coordinate figures
- [x] avoid adding a heavy conversion dependency unless justified. — done in `63bce35`: the `tensorboard` package is an optional extra imported lazily, and no figure conversion was added

## 15.2 TensorBoard tests

- [x] temporary logdir; — done in `70c5244`: `TestRealEventFiles` writes into a temporary log directory
- [x] event file created; — done in `70c5244`: `test_scalars_are_written_and_readable`
- [x] expected scalar tags exist; — done in `70c5244`: `test_scalars_are_written_and_readable`
- [x] repeated iterations produce multiple steps; — done in `a094e05`: `test_event_files_hold_each_step_once`
- [x] dynamic tags are stable; — done in `70c5244`: `test_dynamic_groups_get_one_tag_each`
- [x] writer flush/close works; — done in `70c5244`: `test_close_closes_and_forgets_the_writer`
- [ ] complex figure path has a test if supported. — obsolete: `TensorBoardReporter` has no complex-figure path to test (`63bce35` ignores the colour path)

---

# 16. P2 — exact dev environment tables / post-hoc analysis

The old dev environment module also emitted:

- raw long-form timestep table;
- raw wide timestep table;
- derived wide table;
- correlation matrix;
- distribution summary;
- training metrics table.

Decide whether these belong in the generic visualization API or in CSV/post-hoc
analysis.

- [ ] Do not silently drop them if they are still required.
- [ ] Prefer CSV/table export over forcing them into line `Query`.
- [x] Keep generic reporting backend-agnostic. — done in `146108c`: `Reporter` in `core/reporting/base.py` holds the resolution and imports no backend; the W&B, CSV and TensorBoard reporters subclass it
- [ ] Water-specific observed-vs-simulated plots should remain domain-specific
      unless generalized deliberately.

---

# 17. P2 — clean up legacy dev plotting only after parity

Legacy modules currently hold W&B-specific history/state such as accumulated
tables.

Once the feature branch passes parity tests:

- [x] remove obsolete direct W&B calls from optimizers; — done in `c02db47`: the direct W&B calls were removed from `RayOptimizer` (`c02db47`) and `ESOptimizer` (`41d9abf`), before any parity comparison
- [ ] remove duplicate ES history caches; — partly done in `41d9abf`: the W&B history caches are gone; `ESOptimizer.population_history` remains, documented and used by the results and tests
- [x] remove dead `plot_population`, `plot_parameter_names`, `plot_mean`, and
      `plot_best_candidate` preparation if no longer used; — done in `41d9abf`: `plot_population`, `plot_parameter_names`, `plot_mean` and `plot_best_candidate` no longer exist in `ESOptimizer`
- [ ] remove old plotting entry points only after screenshots/data are compared; — partly done in `c02db47`: the old plotting modules were removed, without comparing screenshots or data first
- [ ] leave migration notes or deprecation stubs if other examples import them.

Do not delete dev reference code before the parity test is complete.

---

# 18. P2 — optional-schema presence semantics

Current schema design has optional nested branches such as:

```python
class RaySchema(MetricSchema):
    train: Optional[TrainSchema] = None
    eval: Optional[EvalSchema] = None
```

Verify that an absent optional branch remains `None` instead of being
constructed as an empty schema during `peek()`/`reduce()`. — answered: `RayOptimizer._to_logger_payload` keeps an absent `train` or `eval` as `None` (`9c22690`), but `MetricLogger.peek()` and `reduce()` return an empty schema for a branch that was never pushed, measured on 2026-10-05

Do not infer absence from leaf reduced values because legitimate empty reducer
values include:

```text
Sum   -> 0
Count -> 0
Series -> []
```

If needed, add explicit node presence tracking.

- [x] test train-only payload; — done in `9c22690`: `test_training_result_fills_every_train_block_and_leaves_eval_empty`
- [x] test eval-only payload; — done in `9c22690`: `test_eval_only_payload_has_no_train_branch_and_no_learner_block`
- [x] test train + eval payload; — done in `9c22690`: `test_evaluation_block_nested_in_a_training_result_fills_the_eval_branch`
- [x] test destructive reduce/reset behavior. — done in `9c22690`: `test_payloads_accumulate_in_the_logger_and_reduce_to_the_last_iteration`

---

# 19. P2 — Ray serialization boundary regression

The optimizer-local `MetricLogger` should not have to cross into the `World`
actor merely to register an optimizer ID.

- [x] base optimizer config sends only optimizer ID/registry data to World; — done in `41d9abf`: `OptimizerConfig.build_optimizer` registers only a generated ID with the World (`_set_new_opt_id.remote`)
- [x] Ray optimizer config follows the same ownership model; — done in `088c164`: `RayOptimizerConfig.build_optimizer` registers only a generated ID the same way
- [ ] regression test proves an optimizer containing an unpicklable local
      object can still register if only its ID crosses the actor boundary.

This keeps logger/reporter runtime state local to the optimizer that owns it.

---

# 20. Recommended implementation order

1. [x] Verify/add `ESSchema.generation`. — done in `f52e7c6`: `ESSchema.generation` exists and is filled
2. [x] Add `ESSchema.generation_best`. — done in `5796732`: `ESSchema.generation_best`
3. [x] Lock the current non-wildcard environment, Ray, and ES query smoke tests. — done in `34ae6f6`: `test_every_es_level_query_resolves` and the environment and debug-script query tests (`3205b73`, `19125d3`)
4. [x] Implement wildcard path expansion. — done in `fdd2567`, with tokens instead of "*": `ReduceProtocol.SERIES` and `MEAN` in `Reporter._resolve_path`
5. [x] Implement wildcard x/y binding. — done in `fdd2567`: x and y groups are bound by `Reporter._resolve_query`
6. [x] Implement mechanism grouping + seed mean/std. — done in `bd4728b`: `error_path` gives the per-mechanism std across seeds
7. [ ] Implement train-vs-eval grouped shaded rendering. — partly done in `bd4728b`: train and eval share a figure, but only the eval path carries the std band
8. [x] Reproduce ES fitness-over-generations trace modes. — done in `cd28820`: `Query.plot_modes` and `legend_labels` style the ES fitness query
9. [x] Reproduce cumulative ES parameter scatter with generation color. — done in `cd28820`: the "Candidate fitness vs quota" scatter is coloured by outer iteration, for the searched parameter only
10. [ ] Implement parallel-coordinate query/renderer.
11. [ ] Run deterministic dev-vs-feature parity validation.
12. [ ] Complete unit/integration tests. — partly done in `70c5244`: unit suites cover the logger, `Query`, the reporters, the ES and Ray payloads, but there is no dev-parity test and no test of a real Ray run that checks the IDs
13. [x] Complete CSV reporter. — done in `f863b4e`: `CSVReporter`
14. [x] Complete TensorBoard reporter. — done in `63bce35`: `TensorBoardReporter`
15. [x] Remove legacy W&B-specific plotting/cache code. — done in `c02db47`: the W&B-specific plotting modules and caches were removed (see also `41d9abf`)
16. [ ] Finalize docs/tutorial examples. — partly done in `268b500`: the visualization tutorial was rewritten against the reporting API, but nothing documents parallel coordinates or parity

---

# 21. Final acceptance run

Use one small deterministic fishery configuration and preserve its config/seed
in the test documentation.

Recommended:

```text
4 ES candidates
2 optimized parameters:
    fixed_quota
    restoration_subsidy
2-3 environment seeds
>= 3 outer generations
short horizon for CI
explicit evaluation
```

Capture both dev and feature outputs.

Compare:

## Environment

- [ ] same horizon x values;
- [ ] same fish biomass;
- [ ] same harvest;
- [ ] same quota stress;
- [ ] same agent reward/action/harvest/penalty values.

## Inner optimizer

- [ ] same rollout aggregates;
- [ ] same performance values;
- [ ] same per-policy learner values;
- [ ] same mechanism × seed values;
- [ ] same train/eval grouping;
- [ ] same mean/std calculation.

## ES

- [ ] same candidate-to-fitness correspondence;
- [ ] same generation mean/best;
- [ ] same global best;
- [ ] same search mean/sigma semantics;
- [ ] same parameter scatter point set;
- [ ] same parallel-coordinate row set.

Plot styling can differ only where intentionally documented. The goal for this
branch request is to reproduce the dev plots exactly unless a difference is
explicitly approved.

---

# 22. Feature status summary

Already supported / expected to work now:

- [x] Environment horizon plots. — already checked; confirmed in `bd4728b`: `FISHERY_ENV_QUERIES` holds one environment query, which neither `config.yaml` nor `debug.py` wires
- [x] Raw RLlib rollout plots. — already checked; confirmed in `bd4728b`: the rollout queries of `INNER_QUERIES` plot the return and the fish biomass over the training episodes
- [x] Performance plots. — already checked; qualified: `build_performance` fills `PerformanceSchema` (`04e9e7f`), but no example query plots it
- [x] Per-policy learner data/plots with concrete IDs. — already checked; confirmed in `bd4728b`, with tokens instead of concrete IDs: the learner queries of `INNER_QUERIES` reach each policy through `ReduceProtocol.SERIES`
- [x] Per-agent environment data/plots with concrete IDs. — already checked; qualified: the per-agent data is logged (`requested_harvest` and `delivered_harvest` in `by_agent`, `9c74d85`), but `queries.py` has no per-agent query since `3205b73`
- [x] ES SERIES accumulation through `push_data -> peek -> report`. — already checked; confirmed in `1cdf19f`: `test_one_report_per_generation_with_the_accumulated_series`
- [x] Nested ES `inner: MetricSchema` runtime specialization to `RaySchema`. — already checked; confirmed in `7e4b2de` for the logger (`test_static_nested_runtime_subtype_binding`) and in `1cdf19f` for `ESSchema.inner`
- [x] Deep runtime specialization to fishery episode and agent schemas. — already checked; confirmed in `19125d3`: the debug-script smoke test renders the queries that read the fishery episode schema

Still required:

- [x] Dynamic mechanism IDs. — done in `fdd2567`: `ReduceProtocol.SERIES` at a dynamic `by_mechanism` node
- [x] Dynamic seed IDs. — done in `fdd2567`: `ReduceProtocol.SERIES` or `MEAN` at `by_seed`
- [x] Dynamic episode IDs. — done in `fdd2567`: `ReduceProtocol.SERIES` or `MEAN` at `by_episode`
- [x] Dynamic policy IDs. — done in `fdd2567`: `ReduceProtocol.SERIES` at `by_policy`
- [x] Dynamic agent IDs. — done in `fdd2567`: the tokens work at `by_agent`; no fishery query reads it yet
- [ ] Dynamic ES parameter keys. — partly done in `fdd2567`: the tokens can expand the parameter keys, but the fishery scatter names `quota`
- [x] Mechanism mean ±std across seeds through Query API. — done in `bd4728b`: `error_path` ending at `by_seed` in `INNER_QUERIES`
- [ ] Train-vs-eval shaded mechanism plots through Query API. — partly done in `bd4728b`: train and eval share a figure, but only the eval path carries the std band
- [x] ES exact candidate/mean/best mixed trace styling. — done in `cd28820`: "Fitness over outer optimization iterations" mixes markers, a mean line and a best line (not compared with dev)
- [x] ES all-candidate parameter scatter in one plot. — done in `cd28820`: one scatter holds all candidates of all generations (final form in `34ae6f6`)
- [x] ES generation color metadata. — done in `cd28820`: `Query.color` with `color=("iter",)`
- [ ] ES parallel coordinates.
- [ ] Generation-best ES parameter schema/queries. — partly done in `5796732`: `ESSchema.generation_best` exists, but no query reads it
- [ ] Full unit tests. — partly done in `70c5244`: unit suites exist, with the gaps listed under the unit-test items
- [ ] Full integration parity tests.
- [x] CSV reporter. — done in `f863b4e`: `CSVReporter`
- [x] TensorBoard reporter. — done in `63bce35`: `TensorBoardReporter`
- [ ] Legacy visualization cleanup after parity. — partly done in `c02db47`: the legacy plotting code was removed, but parity was never measured
- [x] The fishery benchmark builds and runs end-to-end with the new mechanism abstraction. — done in `7dccabd`: `examples/bilevel_fishery/debug.py` builds the benchmark on the present abstraction (`FisheryRegulatedEnv`, `FisheryRegulatorEnv`, `Quota`) and runs it through ES and the inner APPO; `tests/integration/test_debug_script_csv_smoke.py` runs it as a child process and checks that it exits 0.
- [x] A quota-only run completes training and evaluation. — done in `7dccabd`: `debug.py` holds a single `Quota` mechanism and evaluates after every inner iteration (`evaluation_interval=1`); `tests/integration/test_debug_script_csv_smoke.py` checks that this run completes (two generations, two fishermen, twenty steps).
- [ ] A quota + subsidy run completes. — partly done in `7dccabd`: the real `Quota`, `Subsidy` and `ThresholdPenalty` are stepped together through the real fishery behind a scripted World (`tests/integration/test_fishery_regulator_composition.py`) and ES searches their two dimensions, but no full bilevel run with training and evaluation holds a subsidy.
- [ ] A quota + subsidy + social-observation run completes. — partly done in `dbe2338`: `SocialInfluence` writes the peers' delivered actions into the fishermen's observation and `tests/integration/test_fishery_regulator_composition.py` steps quota, subsidy, penalty and social mechanisms together, but this is not a full bilevel run with training and evaluation.
- [x] Chained composition works for action, observation, and reward channels. — done in `22142f4`: realised differently: nothing is chained any more; a leader's mechanisms all read the same input state and their residuals are summed by `MDPState.add` through `Agent.action`, `Agent.reward` and `Agent.mechanism_observations`, and `tests/integration/test_fishery_regulator_composition.py` checks the three channels (corrections summed, reward equal to harvest plus subsidy plus penalty, social entries in the observation).
- [x] Parallel composition has one coherent API and tests. — done in `aaa0820`: realised differently: there is no `ParallelMechanism`; `Agent.action` applies every mechanism of an agent to the same state and composes the residuals with `MDPState.add`, tested in `tests/agents/test_agent_base.py` and `tests/mechanism/test_trajectory_edges.py`.
- [x] All concrete `Mechanism` implementations satisfy the abstract base class. — done in `dbe2338`: `Mechanism.apply` is the only abstract method; after the port of `SocialInfluence` every class of `core/mechanism/algorithms/` implements it, and `tests/mechanism/test_mechanism_base.py` checks that the base class stays abstract.
- [x] Mechanism optimizer vectors encode/decode correctly. — done in `a65b543`: ES builds its search space with `flatten_space`/`flatdim` over the dictionary of the regulator agents' mechanism action spaces, names the parameters `id` or `id[i]`, and hands each candidate to the environment as a dictionary through `unflatten`; see `tests/optimizers/test_es_construction.py` and `tests/optimizers/test_es_generation_loop.py`.
- [ ] Action and observation spaces agree with transformed values. — partly done in `7dccabd`: `tests/integration/test_fishery_regulator_composition.py` asserts the observation shape at reset and at every step, and `tests/adaptors/test_marl_env_adapter_episode.py` checks that the adapter's spaces equal the agents' spaces; no test checks every transformed observation and action against `space.contains`.
- [x] Unit tests cover every concerned mechanism/env/composition module. — done in `7dccabd`: measured on HEAD: 1893 tests pass and `core/` is at 99 % line coverage with every file of `core/mechanism/` and `core/envs/` at 100 %; the composition modules no longer exist.
- [x] Integration tests cover the benchmark + mechanism lifecycle. — done in `7dccabd`: `tests/integration/test_fishery_regulator_composition.py` (candidate hand-off, reward, observation, truncation) and `tests/integration/test_debug_script_csv_smoke.py` (full run) cover the benchmark with its mechanisms.
- [ ] Reproducibility against `dev` is checked where practical.
- [ ] Quota behavior is numerically compared against the dev fishery benchmark if time permits.
- [x] The tutorial notebooks run after the P0 integration fixes are merged. — done in `6aaee13`: `tests/notebooks/test_tutorial_notebooks.py` executes every tutorial under the `notebook` marker, after the tutorials were rewritten against the present API (`97af7e4`, `13aff1b`, `279fba0`, `268b500`).

---

# 1. P0 — make the current implementation internally consistent

There are several API mismatches in the supplied branch that should be fixed
before trying to compare results.

## 1.1 Reconcile `BilevelConfig.mechanism(...)`

The supplied config method currently has the shape:

```python
def mechanism(
    self,
    *,
    space: MechanismSpace,
    default: Mechanism = None,
    **kwargs,
) -> Self:
    ...
```

but the proposed benchmark config uses:

```python
.mechanism(
    mechanism=ChainedMechanism(...)
)
```

Choose and document one public API.

Recommended target:

```python
.mechanism(
    mechanism=mechanism,
    space=optional_space,
)
```

or, if `MechanismSpace` remains the owner of default construction:

```python
.mechanism(
    space=space,
    default=mechanism,
)
```

Acceptance:

- [x] exactly one supported builder signature; — done in `ed6051c`: realised differently: `BilevelConfig.mechanism(...)` and `MechanismSpace` were removed; mechanisms are declared once, as `MechanismConfig` objects in `AgentConfig(mechanisms=...)` on the regulator's `.agents(...)`.
- [x] examples and tutorials use that signature; — done in `97af7e4`: the tutorials and examples declare their mechanisms through `AgentConfig(mechanisms=...)` (`examples/bilevel_fishery/debug.py`, `tutorials/`), and no file still calls `.mechanism(`.
- [x] `BilevelConfig.build_optimizer()` injects the same mechanism/space into inner and outer components; — done in `03d77fd`: `BilevelConfig.build_optimizer` passes `outer_cfg.agents_cfgs` to the inner environment as `leaders_cfg_dict` and ES builds its action space from the same agents; `tests/optimizers/test_bilevel_config_build.py` checks the hand-off.
- [x] fixed mechanisms work without an unnecessary optimizer space; — done in `dbe2338`: a fixed mechanism declares `empty_action_space()` (a `Box` of shape `(0,)`), so ES searches no dimension for it; `tests/optimizers/test_es_construction.py` checks that only empty spaces give the fixed mode with dimension 0.
- [x] optimized mechanisms expose an optimizer dimension unambiguously. — done in `a65b543`: ES `dimension` is `flatdim` of the union of every regulator agent's mechanism spaces and `parameter_names` lists `id` or `id[i]` in sorted order, with a duplicate id rejected; see `TestSearchSpace` in `tests/optimizers/test_es_construction.py`.

---

## 1.2 Make every concrete mechanism instantiable

`Mechanism` currently declares these abstract methods/properties:

```text
dimension
encode
decode
clip
param_names
to_vector
```

plus identity implementations for:

```text
action
observation
reward
```

Audit every concrete class.

### `QuotaMechanism`

Currently shown:

```text
to_vector       YES
param_names     YES
action          YES
observation     YES

dimension       MISSING in supplied code
encode          MISSING
decode          MISSING
clip            MISSING
```

- [x] implement missing abstract API or move common parameterized behavior into a reusable base. — done in `22142f4`: realised differently: the abstract vector API (`dimension`, `encode`, `decode`, `clip`, `param_names`, `to_vector`) is gone; `QuotaMechanism` implements `decode` and `apply`, and its search dimension comes from the `action_space` of its `Quota` configuration.

### `SubsidyMechanism`

Currently shown:

```text
to_vector       YES
param_names     YES
reward          YES

dimension       MISSING
encode          MISSING
decode          MISSING
clip            MISSING
```

- [x] implement missing abstract API. — done in `d317b9f`: `SubsidyMechanism` implements `decode` and `apply` and its `Subsidy` configuration carries the `action_space`; `tests/mechanism/algorithms/test_subsidy.py` covers it.

### `SocialInfluenceMechanism`

Currently shown only with `observation(...)`.

- [x] implement fixed/optimized parameter API; — done in `dbe2338`: `SocialInfluenceMechanism` implements `observe` and has an empty `action_space` by default, so it is fixed; `influence_weight` is a constructor parameter (`tests/mechanism/algorithms/test_social_influence.py`).
- [x] decide whether `influence_weight` is optimized or fixed; — done in `dbe2338`: `influence_weight` is fixed: it is a constructor parameter, validated as non-negative, reserved for the reward bonus of Jaques et al. and without effect.
- [x] if fixed, `dimension == 0`; — done in `dbe2338`: the default `action_space` of `SocialInfluence` is `empty_action_space()`, so its dimension is 0 (`test_default_action_space_is_empty` in `tests/mechanism/algorithms/test_social_influence.py`).
- [ ] if optimized, define normalized encode/decode bounds. — obsolete: `influence_weight` is fixed, so no optimizer bounds are needed; the vector API `encode`/`decode` they would have used was removed in `22142f4`.

### `ThresholdPenaltyMechanism`

Currently has `dimension`, `encode`, `decode`, `param_names`, `reward`.

- [ ] verify/implement `clip`; — obsolete: `clip` was part of the abstract vector API removed in `22142f4`; `ThresholdPenalty` has no searched parameter, so nothing is clipped.
- [ ] verify/implement `to_vector`; — obsolete: `to_vector` was part of the abstract vector API removed in `22142f4`, and `20f878e` removed it from `ThresholdPenalty`.
- [x] decide whether threshold/penalty are fixed or optimizer-controlled. — done in `20f878e`: threshold and penalty are fixed in the `ThresholdPenalty` configuration, and its `action_space` is empty (`test_default_action_space_is_empty` in `tests/mechanism/algorithms/test_penalty.py`).

### `ChainedMechanism`

- [ ] verify/implement `clip`; — obsolete: `ChainedMechanism` was removed in `3639bda` and `clip` with the abstract vector API in `22142f4`.
- [ ] define `to_vector` for the semantic vector exposed to agents; — obsolete: `ChainedMechanism` was removed in `3639bda`; the vector exposed to the agents no longer exists (`to_vector` left the base class in `22142f4`).
- [x] test concatenation/slicing of child optimizer vectors. — done in `9d8ced6`: realised differently: ES concatenates the mechanisms' spaces with `flatten_space` and slices each candidate back with `unflatten`; `test_each_candidate_is_a_dict_of_mechanism_actions` in `tests/optimizers/test_es_generation_loop.py` checks the slicing.

### `ParallelMechanism`

- [ ] same abstract-method audit; — obsolete: `ParallelMechanism` was removed in `3639bda`; every concrete mechanism now implements `apply`.
- [ ] same vector semantics audit. — obsolete: `ParallelMechanism` was removed in `3639bda`; there is no composite vector left to audit.

No concrete mechanism should remain abstract accidentally.

---

## 1.3 Remove stale constructor arguments from examples

The proposed example still uses old `MechanismSpace`-style arguments:

```python
QuotaMechanism(
    optimize_params=["fixed_quota"],
    default_fixed_quota=0.56224,
    default_max_demand_frac=1.0,
)
```

but the shown new dataclass is:

```python
QuotaMechanism(
    fixed_quota: float,
    bindings: ...,
    action_component: int = 0,
    ...
)
```

Likewise `SubsidyMechanism` currently declares:

```python
SubsidyMechanism(
    subsidy: float,
    cost: float,
    action_component: int = 1,
)
```

not old `optimize_params/default_*` arguments.

- [x] update all examples to the new object model; — done in `827dff6`: the fishery example was rebuilt on `MechanismConfig` objects (`Quota`, `FishingConfig`, `RestoreConfig`), and the cart-pole and fresh-water examples were ported the same way (`09033f7`, `2010b10`); no example passes `optimize_params` or `default_*` any more.
- [x] keep optimization selection in one place only; — done in `22142f4`: what is searched is decided by the `action_space` of each `MechanismConfig` alone; an empty `Box` means fixed.
- [x] do not duplicate defaults in both mechanism objects and spaces. — done in `22142f4`: the default action lives only in the `default` field of `MechanismConfig`; no space object exists to repeat it.

---

# 2. P0 — fix environment mechanism dispatch

The regulated env should have one explicit lifecycle:

```text
policy output
    ↓
normalize action
    ↓
benchmark action hook (optional)
    ↓
mechanism action transform
    ↓
benchmark intrinsic/base reward
    ↓
benchmark transition
    ↓
mechanism reward transform
    ↓
benchmark observation
    ↓
mechanism observation transform
    ↓
publish context
```

Audit `MultiAgentRegulatedEnv` against this.

## 2.1 Wrong mechanism method calls

The supplied `reward(...)` method returns:

```python
return self.mechanism.action(
    reward_dict,
    env=self,
)
```

- [x] call `self.mechanism.reward(...)`. — done in `4fcf8e4`: realised differently: `MultiAgentEnv.step` calls `Agent.action` then `Agent.reward`, and a mechanism's reward residual is the `rewards` field returned by its `apply`; `tests/envs/test_marl_env_step.py` covers the step.

The supplied `observation(...)` method returns:

```python
return self.mechanism.action(
    obs_with_theta,
    env=self,
)
```

- [x] call `self.mechanism.observation(...)`. — done in `5c2c52f`: realised differently: `MultiAgentEnv` builds observations with `Agent.observation` for the followers and `Agent.mechanism_observations` for the leaders, which call `Mechanism.observe`; `tests/envs/test_marl_regulated_observation.py` covers it.

---

## 2.2 Observation concatenation is currently dict-unsafe

The supplied code contains:

```python
theta = self.mechanism.to_vector()
obs_with_theta = np.concatenate(
    [observation_dict, theta],
    axis=0,
)
```

`observation_dict` is a multi-agent dictionary, not an array.

Target:

```python
obs_with_theta = {
    agent_id: np.concatenate(
        [
            np.asarray(observation, dtype=np.float32).reshape(-1),
            theta,
        ]
    ).astype(np.float32, copy=False)
    for agent_id, observation in observation_dict.items()
}
```

Then pass the dict through `mechanism.observation(...)`.

- [ ] add a regression test for this exact failure mode. — obsolete: the concatenation of a mechanism vector onto the observation was removed with `to_vector` in `22142f4` and the rewrite of the environment in `4fcf8e4`; observations are now built per agent, and `tests/envs/test_marl_regulated_observation.py` covers that path.

---

## 2.3 Avoid bypassing the public reward/observation pipeline

`step()` currently directly calls mechanism reward and public observation.

Decide whether public `reward(...)` is the pipeline method or remove it.
There should not be two overlapping reward paths.

Recommended:

```python
intrinsic_rewards = benchmark_reward(delivered_actions)
...
rewards = self.reward(
    intrinsic_rewards,
    action_after=delivered_actions,
)
obs = self.observation({})
```

- [x] one reward path only; — done in `4fcf8e4`: `MultiAgentEnv.step` computes the reward through `Agent.reward` only; `tests/envs/test_marl_env_step.py`.
- [x] one observation path only; — done in `5c2c52f`: `MultiAgentEnv.reset` and `step` build the observation through `Agent.observation` and `Agent.mechanism_observations` only; `tests/envs/test_marl_regulated_observation.py`.
- [x] one action path only. — done in `4fcf8e4`: the action goes through `Agent.action` only, which applies each mechanism to the raw action of the step (`56dde42` made it the raw one); `tests/agents/test_agent_base.py`.

---

## 2.4 Fix "no published mechanism" fallback path

The supplied branch calls:

```python
self.observation(agent_id, self.S_t)
```

even though `observation(...)` accepts one `observation_dict`.

- [x] make fallback reset/step behavior use the same observation pipeline; — done in `472feb1`: before any candidate reaches the world, `MultiAgentEnv.reset` gives each leader the `default` action of its mechanisms and builds the observation through the same pipeline as later steps.
- [x] add a test where the world has not published a non-default mechanism. — done in `472feb1`: `tests/envs/test_marl_env_reset.py` covers the reset before any publication, and `test_no_candidate_before_the_first_publication` in `tests/envs/test_marl_regulated_mechanism.py` checks that the default never counts as a published candidate.

---

# 3. P0 — fix the fishery benchmark under the new action semantics

The benchmark currently has a 2-component action:

```text
component 0 = harvest fraction
component 1 = restoration effort
```

but the transition currently treats the full vector as harvest:

```python
delivered_harvest = {
    agent_id: action * full_required_harvest
    for agent_id, action in A_t.items()
}
```

Target:

```python
harvest_fraction = float(action[0])
restoration_effort = float(action[1])
```

Then:

```python
requested_harvest_i = harvest_fraction_i * full_required_harvest
```

- [x] extract action components deliberately; — done in `827dff6`: the fisherman holds two separate mechanisms, `Fishing` (action `harvest`) and `Restore` (action `restore`), each with its own one-component action, so no code indexes into a two-component vector.
- [x] document the semantic component map; — done in `827dff6`: the component map is now the set of mechanism ids `harvest` and `restore` documented in `examples/bilevel_fishery/regulated_env.py` (`Fishing`, `Restore`).
- [x] avoid implicit whole-vector arithmetic. — done in `827dff6`: `Fishing.apply` and `Restore.apply` each read their own scalar action; `tests/examples/test_fishery_env_dynamics.py` checks the catch and the restoration by hand.

## 3.1 Restoration dynamics are currently disconnected

The supplied transition uses:

```python
growth = biological_growth + noise + kwargs["restoration"]
```

but the shown env step does not pass `restoration`.

Recommended:

```python
restoration = restoration_effectiveness * sum(
    action[1] for action in A_t.values()
)
```

and either pass it explicitly to the transition hook or derive it there.

The subsidy mechanism should modify reward; ecological restoration belongs in
the benchmark transition.

- [x] connect restoration action to fish dynamics; — done in `827dff6`: `Restore.apply` records the restored biomass and `pella_tomlinson` adds it to the next stock; `test_restoration_enters_the_equation_next_to_the_production` in `tests/examples/test_fishery_env_dynamics.py`.
- [x] keep ecology and incentive shaping separate. — done in `827dff6`: restoration is ecology in the transition, while the subsidy is a reward residual of `Subsidy` that never touches the stock.

## 3.2 Fix `K` reference

The transition contains:

```python
fish_next = float(np.clip(fish_next, 0.0, K))
```

- [x] use `self.K` or deliberately remove the upper clipping; — done in `53e5ddd`: the upper clipping was removed on purpose and the growth term uses `self.K`; `test_stock_at_capacity_without_fishing_stays_there` and `test_stochastic_reset_is_reproducible_and_clipped_to_capacity` check the capacity.
- [x] add boundary tests. — done in `53e5ddd`: `tests/examples/test_fishery_env_dynamics.py` has boundary tests: stock at capacity, stock below capacity, stock never negative, and the catch never above the stock.

## 3.3 Define the base reward

The shown `FisheryRegulatedEnv` does not include a `@reward` hook.

- [x] add or verify the benchmark base reward; — done in `827dff6`: `Fisherman.reward` pays the harvest fraction of the step, and `FisheryRegulatedEnv` needs no reward hook.
- [x] test reward before any mechanism; — done in `9c74d85`: `test_reward_is_the_harvest_fraction_of_the_step` in `tests/examples/test_fishery_env_dynamics.py`.
- [x] test reward after subsidy/penalty. — done in `7dccabd`: `test_reward_is_harvest_plus_subsidy_plus_penalty` in `tests/integration/test_fishery_regulator_composition.py` checks the reward after subsidy and penalty.

---

# 4. P0 — fix `SubsidyMechanism`

Intended reward:

\[
r_{i,t}^{*}
=
r_{i,t}
-
c e_{i,t}^{2}
+
\sigma_\theta e_{i,t}.
\]

The supplied implementation contains:

```python
actions[agent_id[self.action_component]]
```

Fix to:

```python
actions[agent_id][self.action_component]
```

Recommended implementation:

```python
effort = float(
    actions[agent_id][self.action_component]
)

reward
+ self.subsidy * effort
- self.cost * effort**2
```

Acceptance:

- [x] zero effort -> no subsidy/cost; — done in `d317b9f`: `test_no_effort_leaves_the_reward_unchanged` in `tests/mechanism/algorithms/test_subsidy.py`.
- [x] positive effort -> exact analytical reward; — done in `d317b9f`: `test_analytical_value` checks `rate * MAX_SUBSIDY * e - cost * e**2`.
- [x] component selection tested; — done in `d317b9f`: realised differently: the effort comes from the action of the targeted mechanism named in `acts_on`, not from a component index; `test_effort_is_read_from_the_targeted_mechanism` and `tests/mechanism/algorithms/test_subsidy_targeting.py`.
- [ ] reward type remains `float`; — partly done in `d317b9f`: `SubsidyMechanism.apply` returns a plain Python float per targeted agent (checked on HEAD), but no test asserts the type.
- [x] public bounds use `ValueError`, not only `assert`. — done in `d317b9f`: `SubsidyMechanism` raises `ValueError` for a cost outside `[0, 1]` (`test_cost_must_be_in_unit_interval`) and when `acts_on` is missing.

---

# 5. P0 — finish `SocialInfluenceMechanism`

Full social influence from the project slides:

\[
c_t^i
=
\sum_{j\neq i}
D_{KL}
\left[
\pi_j(a_t^j \mid a_t^i, s_t^j)
\|
\pi_j(a_t^j \mid s_t^j)
\right]
\]

and:

\[
r_t^i = r_{i,t} + \beta c_t^i.
\]

The supplied implementation currently implements only observation shaping:

\[
o_{i,t}^{*}
=
[
o_{i,t},
a_{1,t-1},
\dots,
a_{j,t-1},
\dots
].
\]

- [x] document that this is observation augmentation, not the full Jacques et al. KL bonus; — done in `dbe2338`: the docstring of `SocialInfluenceMechanism` says it exposes the peers' delivered actions and that `influence_weight` is reserved for the bonus of Jaques et al.
- [x] `influence_weight` is currently unused in the shown implementation; — done in `dbe2338`: `influence_weight` is documented as reserved with no effect and validated non-negative (`test_influence_weight_is_reserved`).
- [x] either implement the KL reward term or scope/rename the class; — done in `dbe2338`: the class is scoped as an observation mechanism: the reward term is not implemented and the weight is kept for it, and the name was not changed.
- [ ] add `bindings` to the dataclass if constructor-injected bindings are intended; — obsolete: the `bindings` field was removed with `Mechanism.resolve` in `22142f4`; the configuration now carries `acts_on` and `obs_offset`.
- [x] test peer-action ordering; — done in `14557d7`: `test_peer_ordering_and_self_exclusion` and the numeric-order tests in `tests/mechanism/algorithms/test_social_influence.py`.
- [x] test self-action exclusion; — done in `dbe2338`: `test_peer_ordering_and_self_exclusion` shows that each agent receives only its peers' entries.
- [x] test observation dimensionality. — done in `dbe2338`: `test_reserved_entries_must_fit_in_the_observation` in `tests/mechanism/algorithms/test_social_influence.py` and the shape checks of `tests/integration/test_fishery_regulator_composition.py`.

---

# 6. P0 — quota mechanism numerical tests

The quota computes:

\[
L = \sigma((0-q)/w_q),
\quad
U = \sigma((1-q)/w_q),
\quad
C_t = \sigma((b_t-q)/w_q)
\]

and:

\[
\alpha_t = \frac{C_t-L}{U-L}.
\]

Requested fraction \(u_{i,t}\) becomes:

\[
u_{i,t}^{*}
=
u_{i,t}
-
\operatorname{smooth}_{+}
\left(
u_{i,t}-\alpha_t;
w_u
\right).
\]

Tests:

- [x] resource close to 0 -> allowed fraction near lower end; — done in `7dccabd`: `test_nothing_is_allowed_when_the_stock_is_empty` in `tests/mechanism/algorithms/test_quota.py`.
- [x] resource close to 1 -> allowed fraction near 1; — done in `7dccabd`: `test_everything_is_allowed_when_the_stock_is_full` in `tests/mechanism/algorithms/test_quota.py`.
- [x] resource near `fixed_quota` -> expected sigmoid transition; — done in `7dccabd`: `test_half_the_stock_at_the_quota` in `tests/mechanism/algorithms/test_quota.py`.
- [x] request below allowed fraction remains approximately unchanged; — done in `7dccabd`: `test_request_below_the_allowance_is_left_unchanged` in `tests/mechanism/algorithms/test_quota.py`.
- [x] request above allowed fraction is smoothly capped; — done in `7dccabd`: `test_request_above_the_allowance_is_capped_and_lowered` and `test_cap_is_continuous_around_the_allowance` in `tests/mechanism/algorithms/test_quota.py`.
- [x] non-target action components are unchanged; — done in `7dccabd`: `test_residual_keeps_the_shape_and_leaves_other_components_alone` in `tests/mechanism/algorithms/test_quota.py`.
- [x] input arrays are not mutated in place; — done in `7dccabd`: `test_request_is_not_mutated_in_place` in `tests/mechanism/algorithms/test_quota.py`.
- [x] per-agent mapping preserved; — done in `7dccabd`: `test_each_agent_is_capped_on_its_own_request` in `tests/mechanism/algorithms/test_quota.py`.
- [x] `allowed_frac` is available to the quota observation transform. — done in `6bf771c`: realised differently: the allowed fraction is published as the state entry `allowed_frac:<id>` rather than to an observation transform; see the `allowed_frac` tests in `tests/mechanism/algorithms/test_quota.py`.

---

# 7. P0/P1 — optional quota reproducibility against `dev`

Preferred but not blocking if time is limited.

Create a deterministic quota-only fixture using the same:

```text
r
K
p
fish_init/B0
sigma
policy seed
environment seed
action trajectory
fixed quota
transition widths
```

Compare old dev and new quota transforms before involving RLlib.

At each step compare:

```text
resource_level
allowed_frac/effective quota
requested harvest fraction
delivered harvest fraction
fish stock
fish_norm
H_attempted
H_realized
reward if reward semantics are unchanged
```

Suggested tolerance:

```python
np.testing.assert_allclose(
    new,
    dev,
    rtol=1e-6,
    atol=1e-7,
)
```

If a difference is intentional, document whether it comes from normalization,
smoothing, reward semantics, or dynamics.

---

# 8. P1 — hook discovery tests

`core/envs/hooks.py` attaches markers and
`MultiAgentRegulatedEnv.__init_subclass__` discovers them.

Tests:

- [x] `@reset` registers reset hook; — done in `9c74d85`: `test_subclass_records_the_name_of_the_marked_method` in `tests/envs/test_env_hooks.py`.
- [ ] `@action` registers action hook; — obsolete: the `action` decorator of `core.envs.hooks` was deleted in `be2d354`; an agent's action is now the `apply` of its mechanisms.
- [ ] `@reward` registers reward hook; — obsolete: the `reward` decorator of `core.envs.hooks` was deleted in `be2d354`; the reward is now `Agent.reward` and the mechanisms' residuals.
- [ ] `@observation` registers observation hook; — obsolete: the `observation` decorator of `core.envs.hooks` was deleted in `be2d354`; the observation is now `Agent.observation` and `Mechanism.observe`.
- [x] `@transition` registers transition hook; — done in `9c74d85`: `test_subclass_records_the_name_of_the_marked_method` in `tests/envs/test_env_hooks.py` covers `transition` as well as `reset`.
- [x] inherited hooks behave intentionally; — done in `9c74d85`: `test_inherited_hooks_are_kept_and_can_be_redeclared` in `tests/envs/test_env_hooks.py`.
- [x] multiple hooks of one type either raise or have documented deterministic behavior. — done in `05dfc50`: `MultiAgentEnv` raises `TypeError` for two methods that carry the same hook mark (`test_two_methods_with_the_same_hook_mark_are_rejected` in `tests/envs/test_env_hooks.py`).

Recommendation: fail fast rather than silently letting the last same-type hook
win.

---

# 9. P1 — mechanism binding tests

A binding is:

```python
binding: env -> runtime context value
```

Example:

```python
bindings={
    "resource_level": lambda env: (
        env.S_t["fish"] / max(env.K, EPS)
    ),
}
```

Tests:

- [ ] `resolve(env)` returns configured keys; — obsolete: `Mechanism.resolve` and the `bindings` field were removed in `22142f4`; a mechanism reads the state through `obs_map` and `acts_on` of its configuration.
- [ ] missing required binding raises at construction; — partly done in `7dccabd`: a missing `obs_map` entry or `acts_on` raises `ValueError`, but at the first `apply` or `observe`, not at construction (`test_apply_requires_the_resource_level` in `tests/mechanism/algorithms/test_quota.py`).
- [x] quota receives normalized resource level; — done in `7dccabd`: `QuotaMechanism.apply` divides the stock named by `obs_map['resource_level']` by `K` (`test_resource_level_is_the_state_over_the_capacity`).
- [x] social observation receives `previous_actions` and `agent_ids`; — done in `dbe2338`: realised differently: `SocialInfluenceMechanism.observe` reads the delivered actions of step `t - 1` from the state it receives, and the ids come from the state's `aids` (`tests/mechanism/algorithms/test_social_influence.py`).
- [ ] child bindings in compositions resolve against the correct env; — obsolete: mechanisms are no longer composed into children with their own bindings (`3639bda`, `22142f4`); every mechanism of an agent reads the same state.
- [ ] bindings remain serializable in Ray/cloudpickle integration. — obsolete: the `bindings` callables were removed in `22142f4`; mechanism configurations are frozen dataclasses of plain values.

---

# 10. P1 — chained composition tests

For:

```python
ChainedMechanism(
    children=(m1, m2, m3)
)
```

contract:

\[
x^\*
=
M_3(M_2(M_1(x))).
\]

Tests:

- [ ] action order exactly follows child tuple order; — obsolete: `ChainedMechanism` was removed in `3639bda`; the mechanisms of an agent are applied to the same state and their residuals summed, so order does not matter.
- [ ] reward order exactly follows child tuple order; — obsolete: `ChainedMechanism` was removed in `3639bda`; rewards are residuals summed by `MDPState.add`.
- [ ] observation order exactly follows child tuple order; — obsolete: `ChainedMechanism` was removed in `3639bda`; observation contributions are summed by `MDPState.add`.
- [ ] each child receives previous child's transformed output; — obsolete: `ChainedMechanism` was removed in `3639bda`; all mechanisms of one agent now read the same input state, as the `Subsidy` docstring states.
- [ ] each child resolves its own env bindings; — obsolete: per-child bindings were removed in `22142f4`; each configuration carries its own `obs_map`.
- [x] dimension is sum of child dimensions; — done in `a65b543`: realised differently: the search dimension is `flatdim` of the dictionary of mechanism spaces, so it is the sum of their sizes (`test_parameter_names_follow_the_sorted_mechanism_ids` in `tests/optimizers/test_es_construction.py`).
- [x] encode is concatenation; — done in `9d8ced6`: realised differently: `flatten_space` of the dictionary of mechanism spaces is the concatenation, in sorted mechanism id order.
- [x] decode slices correctly; — done in `e767ed4`: realised differently: ES slices each candidate with `unflatten` (`test_each_candidate_is_a_dict_of_mechanism_actions` in `tests/optimizers/test_es_generation_loop.py`).
- [x] parameter names preserve child identity/order; — done in `a65b543`: `parameter_names` is `id` or `id[i]` for each mechanism in sorted order (`test_parameter_names_follow_the_sorted_mechanism_ids`).
- [x] zero-dimension children do not break slicing. — done in `a65b543`: the penalty's empty space sits next to the quota and subsidy spaces in `test_parameter_names_follow_the_sorted_mechanism_ids`, and `test_only_empty_action_spaces_make_the_fixed_mode` covers the all-empty case.

Interaction test:

```text
child 1: multiply by 2
child 2: add 1

expected chain:
2x + 1
```

---

# 11. P1 — parallel composition API repair and tests

The supplied `ParallelMechanism` calls:

```python
child.apply_action(...)
child.apply_reward(...)
child.apply_observation(...)
```

while `Mechanism` exposes:

```python
action(...)
reward(...)
observation(...)
```

- [ ] reconcile this before use. — obsolete: `ParallelMechanism` was removed in `3639bda`.

Recommended target:

```python
context = child.resolve(env)
child.action(copy, **context)
```

Parallel contract:

\[
M_{\parallel}^{A}(x)
=
\Gamma_A
\left(
x,
M_1^A(x),
\dots,
M_k^A(x)
\right).
\]

Each child receives the same original input.

Tests:

- [x] every child sees the same original input; — done in `aaa0820`: realised differently: `Agent.action` applies every mechanism to the same state (`test_action_applies_every_mechanism_it_holds_an_action_for` in `tests/agents/test_agent_base.py`).
- [x] no child sees another child's output; — done in `aaa0820`: realised differently: each mechanism returns a residual and the state is only changed when `MDPState.add` composes them, so no mechanism sees another's output.
- [ ] merge receives original + tuple of outputs; — obsolete: there is no merge function since `3639bda`; `MDPState.add` sums the residuals.
- [ ] merge ordering is documented; — obsolete: there is no merge step since `3639bda`; `MDPState.add` sums additive fields, so ordering is irrelevant.
- [ ] deep copies prevent cross-child mutation; — partly done in `aaa0820`: there are no deep copies; mechanisms return residuals instead of editing the shared state, and only the quota has a test that its input is not mutated (`test_request_is_not_mutated_in_place`).
- [x] action/reward/observation merge functions tested separately; — done in `27fbd29`: realised differently: `MDPState.add` is tested for rewards and observations summed at one step (`tests/mechanism/test_trajectory.py`, `0bb367c`) and for its edge cases in `tests/mechanism/test_trajectory_edges.py`.
- [ ] dimensions/encode/decode tested. — obsolete: `ParallelMechanism` was removed in `3639bda`; the dimensions now come from the ES search space.

---

# 12. P1 — action and observation spaces

The current example hard-codes:

```python
action_space = Box(shape=(2,))
```

and an observation shape based on an older mechanism-space abstraction.

Expected fishery features include:

```text
base observation:
    fish_norm
    total_usage_norm

quota augmentation:
    allowed_frac/effective quota

social augmentation:
    previous peer actions

optional mechanism parameter vector:
    theta
```

For 10 agents and 2-D actions, social influence adds:

```text
(10 - 1) * 2 = 18
```

features per agent.

- [x] compute/validate final observation dimension; — done in `dbe2338`: `SocialInfluenceMechanism.observe` raises `ValueError` when the reserved entries do not fit in the agent's observation (`test_reserved_entries_must_fit_in_the_observation`).
- [x] decide whether mechanisms expose `observation_dimension_delta`; — done in `dbe2338`: decided against it: a mechanism writes into entries of the agent's declared `observation_space` that the agent leaves at zero (`obs_offset`), so no dimension delta exists.
- [ ] decide whether `to_vector()` is always appended; — obsolete: `to_vector` left the base class in `22142f4`; the mechanism parameters are not appended to the observation.
- [x] remove dependencies on obsolete `FisheryMechanismSpace().full_dimension` where inappropriate; — done in `827dff6`: the observation space of the fishery no longer depends on `FisheryMechanismSpace().full_dimension`.
- [ ] assert actual observation shape matches declared space; — partly done in `7dccabd`: `tests/integration/test_fishery_regulator_composition.py` asserts the observation shape (5,) at every step against the size declared in `observation_space`, but not through `space.contains`.
- [ ] assert normalized action shape matches declared action space. — partly done in `a8db5a4`: `tests/adaptors/test_marl_env_adapter_episode.py` checks that the adapter's action spaces equal those of the mechanisms; no test checks a normalised action against its space.

---

# 13. P1 — context publishing tests

Each step should preserve:

```text
env_id
seed
policy_seed
mode/status
mechanism_id
observation
reward
action
info
```

- [ ] values correspond to regulated action/reward/observation actually used; — partly done in `733be6c`: the environment logs the mean follower reward of each step and `test_the_env_logs_the_mean_follower_reward_of_each_step` checks it against the reward returned; the regulator now publishes candidates, not per-step actions or observations.
- [x] seeds remain immutable for an env instance; — done in `9c74d85`: `test_reset_does_not_reseed_the_environment` in `tests/envs/test_marl_env_reset.py`.
- [x] mechanism ID matches the published mechanism; — done in `9c74d85`: `MultiAgentEnv.reset` fetches the candidate by its `mechanism_id` (`tests/envs/test_marl_env_reset.py`), and `tests/envs/test_regulator_publish.py` checks that every candidate is published for every seed (`dea1919`).
- [ ] publication does not modify behavior.

---

# 14. P1 — recommended test layout

```text
tests/envs/test_hooks.py
tests/envs/test_marl_regulated.py

tests/mechanism/test_base.py
tests/mechanism/algorithms/test_quota.py
tests/mechanism/algorithms/test_subsidy.py
tests/mechanism/algorithms/test_penalty.py
tests/mechanism/algorithms/test_social_influence.py

tests/mechanism/composition/test_chained_mechanism.py
tests/mechanism/composition/test_parallel_mechanism.py
tests/mechanism/test_space.py

tests/examples/test_fishery_regulated_env.py
tests/integration/test_fishery_mechanisms.py
```

Coverage target:

- [ ] meaningful branch coverage for all concerned files; — partly done in `3db1e46`: measured on HEAD: `core/` is at 99 % line coverage (5 lines missed, in `core/optimizers/es/optimizer.py` and `core/adaptors/ray/learner_drain.py`); branch coverage is not measured.
- [x] aim for >=90% line coverage on pure mechanism/composition modules; — done in `3db1e46`: measured on HEAD: 99 % line coverage on `core/`, with `core/mechanism/` and `core/envs/` at 100 %.
- [x] every mechanism dispatch path covered even if distributed integration coverage is lower. — done in `7dccabd`: `Agent.action`, `Agent.reward`, `Agent.mechanism_observations` and `MultiAgentEnv.step` are covered at 100 % by `tests/agents/test_agent_base.py`, `tests/envs/` and `tests/integration/test_fishery_regulator_composition.py`.

Suggested command:

```bash
pytest \
  tests/envs \
  tests/mechanism \
  tests/examples/test_fishery_regulated_env.py \
  tests/integration/test_fishery_mechanisms.py \
  --cov=core.envs \
  --cov=core.mechanism \
  --cov-report=term-missing
```

---

# 15. P1 — staged end-to-end smoke runs

Do not begin with 1000 outer iterations.

## Stage A — environment only

```text
2 agents
deterministic seed
fixed mechanism
horizon 5
```

## Stage B — inner optimizer only

```text
2 agents
1 mechanism
1 seed
horizon 10
1-2 train iterations
```

## Stage C — quota-only bilevel

```text
2 candidates
1 seed
2 outer generations
```

## Stage D — quota + subsidy

Verify restoration dynamics and reward incentive.

## Stage E — quota + subsidy + social observation

Verify observation-space growth.

## Stage F — evaluation

```text
2-3 explicit eval seeds
deterministic evaluation policy
```

Only then restore the larger benchmark configuration.

---

# 16. P2 — clarify stateful mechanism context

`QuotaMechanism` stores transient values in `_context`.

A frozen dataclass can still mutate a contained dict, but this makes the
mechanism stateful.

Decide:

- [x] Is one mechanism object shared across multiple env instances? — done in `9c74d85`: no: each environment builds its own agents and mechanisms from the frozen configurations (`test_every_build_creates_independent_agents` in `tests/agents/test_agent_base.py`).
- [x] Could vectorized envs overwrite one another's `_context`? — done in `282d125`: no: `_context` was removed from `QuotaMechanism`, which now publishes its allowed fraction in the state, and instances are not shared between environments.
- [x] Should mechanism state reset per episode? — done in `b330a82`: decided: the fetched candidate is kept for every episode of a generation and replaced only by a newly published one (`tests/envs/test_marl_regulated_mechanism.py`); the allowed fraction is a state entry, reset with the state.
- [x] Should step context live on the env instead? — done in `282d125`: yes: the step context is a state entry (`allowed_frac:<id>`), and the mechanism definition is a frozen configuration.

Recommended default:

```text
Mechanism definition = immutable
Per-step mechanism context = environment-local
```

If mechanism-local context remains, tests must prove instances are not shared
across concurrently stepping envs.

---

# 17. P2 — public validation should not rely on `assert`

Examples use:

```python
assert 0.0 <= self.fixed_quota <= 1.0
```

- [x] use explicit `ValueError` for public configuration; — done in `282d125`: `QuotaMechanism`, `SubsidyMechanism`, `ThresholdPenalty` and `SocialInfluence` raise `ValueError` for bad parameters (`d317b9f`, `20f878e`, `dbe2338`), tested in `tests/mechanism/algorithms/`.
- [x] keep assertions for internal invariants only. — done in `282d125`: no `assert` statement is left under `core/` or `examples/`.

---

# 18. P2 — documentation acceptance

Ship:

```text
tutorials/mechanism_benchmarks_tutorial.py
tutorials/custom_benchmark_mechanism_tutorial.py
MECHANISM_ABSTRACTION_TODO.md
```

Tutorial examples must reflect the final merged public API.

---

# 19. Recommended implementation order

1. [x] Reconcile `BilevelConfig.mechanism` public API. — done in `ed6051c`: see the items of 1.1: mechanisms are declared by `AgentConfig(mechanisms=...)`.
2. [x] Make all mechanism classes concretely instantiable. — done in `dbe2338`: every mechanism of `core/mechanism/algorithms/` implements `apply`.
3. [x] Fix reward/observation dispatch in `MultiAgentRegulatedEnv`. — done in `4fcf8e4`: `MultiAgentEnv.step` calls `Agent.reward` and `Agent.observation`.
4. [ ] Fix per-agent observation concatenation. — obsolete: the concatenation was removed with `to_vector` in `22142f4`; observations are built per agent.
5. [x] Fix fishery 2-component action decomposition. — done in `827dff6`: the fisherman holds the `Fishing` and `Restore` mechanisms, each with a one-component action.
6. [x] Connect restoration action to transition dynamics. — done in `827dff6`: `Restore.apply` feeds the restoration into `pella_tomlinson`.
7. [x] Fix subsidy indexing bug. — done in `d317b9f`: `SubsidyMechanism.apply` reads the effort from the targeted mechanism's action.
8. [x] Scope/finish social influence behavior. — done in `dbe2338`: `SocialInfluence` is scoped as an observation mechanism with a reserved `influence_weight`.
9. [ ] Repair `ParallelMechanism` method API. — obsolete: `ParallelMechanism` was removed in `3639bda`.
10. [x] Add unit tests for hooks and transforms. — done in `9c74d85`: `tests/envs/test_env_hooks.py` and the mechanism tests under `tests/mechanism/algorithms/`.
11. [x] Add composition tests. — done in `7dccabd`: `tests/integration/test_fishery_regulator_composition.py` and `tests/agents/test_agent_base.py`.
12. [x] Add deterministic fishery tests. — done in `9c74d85`: `tests/examples/test_fishery_env_dynamics.py` checks the fishery transitions by hand.
13. [x] Run quota-only smoke benchmark. — done in `7dccabd`: `tests/integration/test_debug_script_csv_smoke.py`.
14. [ ] Run quota + subsidy smoke benchmark. — partly done in `7dccabd`: the composition is stepped in `tests/integration/test_fishery_regulator_composition.py`, but not through a full bilevel run.
15. [ ] Run social observation smoke benchmark. — partly done in `dbe2338`: the social entries are stepped in `tests/integration/test_fishery_regulator_composition.py`, but not through a full bilevel run.
16. [ ] Add evaluation smoke test. — partly done in `7dccabd`: evaluation runs inside the end-to-end smoke test (`evaluation_interval=1` in `debug.py`) and the lifecycle is unit-tested (`tests/adaptors/test_ray_optimizer_lifecycle.py`), but no assertion targets the evaluation result.
17. [ ] Optional/preferred: numerical quota parity against `dev`.
18. [x] Update tutorials to final API. — done in `97af7e4`: the tutorials were rewritten on the present API (`13aff1b`, `279fba0`, `268b500`) and executed by `tests/notebooks/test_tutorial_notebooks.py`.
19. [ ] Run coverage and close remaining untested branches. — partly done in `3db1e46`: coverage was run (99 % of `core/` on HEAD) and the untested branches closed except 5 lines in `core/optimizers/es/optimizer.py` and `core/adaptors/ray/learner_drain.py`.

# In-code TODO notes moved out of the source (October 2026)

The project's `ruff.toml` forbids TODO comments in the code (rules `FIX` and `TD`),
so the cleanup pass on `feature/social-influence-testing-v2` moved every such comment
here before removing it from the source. There are 211 notes. Their wording is kept
verbatim, including author tags. Each note names the symbol it was attached to and its
line in commit `96294f6`, the last commit before the pass, so
`git show 96294f6:<path>` shows it in its original context. Notes marked
"inside commented-out code" belonged to code that was already disabled; that code was
deleted in the same pass and remains in the git history.

One finding of the pass belongs with these notes although it was not written as a
TODO. In `core/mechanism/algorithms/quota.py`, `QuotaMechanism.apply` built a
per-agent observation dictionary holding the allowed fraction and the regulator's
action, `obs = {aid: np.asarray([0.0, 0.0, 0.0, 0.0, allowed_frac, action]) for aid
in self.acts_on}`, and then never used it: the returned `MDPState` carries no
observation, so agents do not observe the quota. The comprehension also iterated over
`self.acts_on`, which is the `(agent, mechanism)` pair rather than the agent IDs. The
unused assignment was removed (ruff rule `F841`); whether the quota should reach the
agents' observations is an open design question for the mechanism port.

## `core/adaptors/ray/marl_env.py`

- `module level` (line 1): (nadine) wraps ray multiagentenv with MetaMARL abstraction
- `module level` (line 31): remove ray and gymnasium dependency
- `RLlibMultiAgentEnvAdapter` (line 47): future enhancement. decouple gym inheritance to support MPC, trajectory optimization etc.
- `RLlibMultiAgentEnvAdapter.__init__` (line 85): (nadine) env serialization with numpy to avoid using MDPState and faster computation

## `core/adaptors/ray/optimizer.py`

- `module level` (line 22): temporary
- `module level` (line 45): perhaps we would first want an adaptor for core ray algorithm and then the PPO inherits it
- `RayOptimizer.__init__` (line 84): maybe this either needs to be an actor. or atleast have method to serialize data
- `RayOptimizer.__init__` (line 88): fallback if rollout_fragment_length not in eval_cfg
- `RayOptimizer._build_agent_policy_map` (line 134): move to utils — resolved in `82aaac2`: the helper was never called and is deleted.
- `RayOptimizer._to_logger_payload` (line 176): (nadine) : in the future this could be separated into a different class if justified
- `RayOptimizer.train` (line 229): temporary to be moved to a logger Extract metrics
- `RayOptimizer.evaluate` (line 258): (nadine) future support for async eval, otherwise must publish eval mechanism obj
- `RayOptimizer.save` (line 307): (empty TODO, no text)

## `core/adaptors/ray/optimizer_config.py`

- `module level` (line 56): override environment to attach docstrings
- `RayOptimizerConfig.algo_class` (line 112): review this
- `RayOptimizerConfig.__init__` (line 122): termporary setting until find out how to share world context accross runners
- `RayOptimizerConfig.rllib_config_mutator` (line 132): let mutator accept an explicit ID
- `RayOptimizerConfig` (line 249) (inside commented-out code, a commented-out `evaluation` method): to infer from horizon
- `RayOptimizerConfig._apply_agents_to_rllib` (lines 546-554): (nadinemgh) this does not guarantee tht different mechanism's policy will be initiated with the same seed ! what we want : run mechanism 0, seed 101; run mechanism 1, seed 101; run mechanism 2, seed 101; run mechanism 0, seed 202; run mechanism 1, seed 202; run mechanism 2, seed 202
- `RayOptimizerConfig._apply_agents_to_rllib` (line 556): verify case when null seed
- `RayOptimizerConfig._apply_agents_to_rllib` (line 590): not needed ?
- `RayOptimizerConfig._apply_agents_to_rllib.policy_mapping_fn` (line 600): this is depregated !
- `RayOptimizerConfig.build_optimizer` (line 707): verify this

## `core/adaptors/ray/policy_actor.py`

- `PolicyActor.train` (line 62): config ability to debug remote actors (the note was followed by commented-out code, now removed: `import debugpy, os`, `debugpy.listen(("127.0.0.1", 5678))`, `print(f"[debugpy] worker pid={os.getpid()} listening on 5678")`, `debugpy.wait_for_client()`, `debugpy.breakpoint()`)
- `PolicyActor.train` (line 68): mapping result
- `PolicyActor.reset` (line 142): verify reset is using the same seed (followed by commented-out code, now removed: `self.algo.set_weights(self._init_weights)`)
- `PolicyActor.reset` (line 146): tie the init_weights with seeding

## `core/adaptors/ray/schema.py`

- `PolicyLearnerSchema` (line 52): this is a callable (note on the `sample_staleness` field)
- `PolicyLearnerSchema` (line 83): what is total loss ?
- `PolicyLearnerSchema` (line 84): kl vs kl loss
- `PolicyLearnerSchema` (line 85): curr_kl_coeff
- `PolicyLearnerSchema` (line 86): entropy vs entropy coeff
- `PolicyLearnerSchema` (line 101): Q-statistics
- `PolicyLearnerSchema` (line 102): Advantage statistics
- `PolicyLearnerSchema` (line 103): advantage statistics
- `PolicyLearnerSchema` (line 105): Reward (R) debugging (the following comment line `# Reward metrics. N.B. episode == trajectory` was kept in the code)
- `module level` (lines 250-256): (bare TODO at the end of the file, followed by the continuation comment lines) num_env_steps_sampled_lifetime_throughput; timers; throughput_since_last_restore; num_agent_steps_sampled; num_agent_steps_sampled_lifetime; prevent non terminal leaves to have reduce objects

## `core/adaptors/ray/utils.py`

- `build_episode_aggregate` (line 198): remove finite
- `build_performance` (line 242): refactor this to get data from env

## `core/agents/base.py`

- `Agent.__init_subclass__` (line 37): (nadine) only reset, transition and state space belong to env
- `Agent._normalize_action` (line 43): move this to Agent
- `Agent._normalize_action` (line 44): make this configurable in future
- `Agent` (line 54): (nadine) MDPState should be EnvState payload to replace MDPState

## `core/callbacks.py`

- `tag_episode_with_env_idx` (line 114): inject policy_id to env for traceability and debugging
- `module level` (line 117): to be moved to a separate actor in the future for extensibility (note placed above `log_and_report_episode_metrics`)
- `log_and_report_episode_metrics` (line 163): could we just have the EnvRolloutSchema here ?
- `_evaluate_with_fixed_duration_once` (lines 366-367): (kourosh) This approach will cause an OOM issue when the dataset gets huge (should be ok for now). (comment inherited from RLlib's `Algorithm._evaluate_with_fixed_duration`)

## `core/envs/marl_regulated.py`

- `module level` (line 55): create a reward type
- `module level` (line 56): separate reported vs type mdp from agent to principal
- `module level` (line 57): wrapper for MultiAgentEnv adaptor to work with ray
- `MultiAgentEnv` (line 59): future enhancement. decouple gym inheritance to support MPC, trajectory optimization etc.
- `MultiAgentEnv.__init__` (line 109): (nadine) later replace with planner_id
- `MultiAgentEnv.__init__` (line 130): change name to just Status
- `MultiAgentEnv.__init__` (line 132): (nadine) later replace with planner id
- `MultiAgentEnv.reset` (line 231): raising error if training started and default mechanism is still on - leads to silent error — partly addressed in `472feb1`: leaders now play their mechanism defaults until a candidate is fetched, and each default is logged at INFO; no error is raised, which stays open.

## `core/envs/regulator.py`

- `RegulatorEnv.__init__` (line 68): (nadine) not clear and avoid providing instantiated obj
- `RegulatorEnv.step` (line 125): input should only be one action. regulator env parallelized instead
- `RegulatorEnv.step` (line 147): (nadine) alternative way to pass mechanism to agents
- `RegulatorEnv.step` (line 169): (nadine) return reduced results to avoid L:223
- `RegulatorEnv.step` (line 172): (nadine) self.reward func should not be taking metrics

## `core/envs/schema.py`

- `AgentEnvStepSchema` (line 41): mean to support bool
- `AgentEnvStepSchema` (line 45): mean to supprot bool
- `EpisodeRolloutSchema` (line 69): add recducer metadata attachment.
- `EpisodeRolloutSchema` (line 130): models such as PILCO, Dyna, Qyna-Q

## `core/mechanism/algorithms/quota.py`

- `QuotaMechanism` (line 18): what if two mechanisms interfere by requiring context from each other ? for example a penalty based on how much quota is violated ?

## `core/mechanism/algorithms/subsidy.py`

- `Subsidy.__post_init__` (line 19): (no text after the TODO token, on the line `assert 0.0 <= self.cost <= 1.0`) — resolved in `d317b9f`: `SubsidyMechanism` raises a `ValueError` for a cost outside `[0, 1]`.
- `Subsidy.reward` (line 32): fix this, passing action after and before — resolved in `d317b9f`: the reward channel is gone; `SubsidyMechanism.apply` reads the decoded effort of the step and returns a reward residual.

## `core/mechanism/base.py`

- `MDPState` (line 33): fix type annotations
- `Mechanism.__call__` (line 147): (nadine) enforce shape 1 action

## `core/metrics/logger.py`

- `MetricLogger.from_schema` (line 130): immutability
- `MetricLogger._build_from_schema` (line 157): guardrails when Metric isnt well formatted
- `MetricLogger.push_data` (line 280): refactor into one push function
- `MetricLogger.push_data` (line 355): only verify it is a metric schema not that its env
- `MetricLogger.push` (line 408): narrow down Any to stricter type annotation
- `MetricLogger.peek_value` (line 419): narrow down Any to stricter type annotation
- `MetricLogger.reduce` (line 462): custom exceptions
- `MetricLogger.reset` (line 491): custom exceptions

## `core/metrics/metric/base.py`

- `module level` (line 11): what is an ABCMeta

## `core/metrics/metric/last.py`

- `LastMetric.reduce` (line 32): move to base cls

## `core/metrics/metric/min.py`

- `MinMetric.reduce` (line 44): move to base cls

## `core/metrics/metric/sum.py`

- `SumMetric.reduce` (line 42): move to base cls

## `core/optimizers/base.py`

- `module level` (line 14): move ray dependencies out of ray optimizer
- `Optimizer` (line 39): ability to save data offline
- `Optimizer.__init__` (line 55): replace by envFactory
- `Optimizer.__init__` (line 63): review
- `Optimizer` (line 73): setup accessors and mutators
- `Optimizer.get_default_config` (line 181): default config logic
- `Optimizer.train` (line 234): change this to training step

## `core/optimizers/bilevel.py`

- `BilevelConfig.reporter` (line 128): remote actor access to credentials
- `BilevelOptimizer.train` (line 246): fig reporter with the new wandb reporter actor

## `core/optimizers/config.py`

- `OptimizerConfig.__init__` (line 87): registry to allow opt_class str
- `OptimizerConfig.__init__` (line 88): runtime checking of opt_class
- `OptimizerConfig.__init__` (line 106): default value
- `OptimizerConfig.__init__` (line 107): default
- `OptimizerConfig._merge_env_config` (line 150): generalize this function
- `OptimizerConfig.freeze` (line 170): freezing for nested configs
- `OptimizerConfig.from_dict` (line 206): review this
- `OptimizerConfig.from_yaml` (line 219): review this
- `OptimizerConfig.build_optimizer` (line 254): deep copy allows on may be toggled later with use_copy
- `OptimizerConfig.build_optimizer` (line 255): build_optimizer() to accept logger_creator: Optional[Callable[[], Logger]] = None,
- `OptimizerConfig.build_optimizer` (line 256): move optimizer registration to executor in future
- `OptimizerConfig.build_optimizer` (line 257): enable multiple world registration
- `OptimizerConfig.environment` (line 306): EnvConfigDict
- `OptimizerConfig` (line 454): Docstring explanation (above the commented-out `ressources` stub)
- `OptimizerConfig` (line 459): Docstring explanation (inside commented-out code, `evaluation` stub)
- `OptimizerConfig` (line 464): Docstring explanation (inside commented-out code, `reporting` stub)
- `OptimizerConfig` (line 469): Docstring explanation (inside commented-out code, `checkpointing` stub)
- `OptimizerConfig` (line 474): Docstring explanation (inside commented-out code, `fault_tolerance` stub)
- `OptimizerConfig` (line 479): Docstring explanation (inside commented-out code, `experimental` stub)

## `core/optimizers/es/config.py`

- `ESConfig` (line 124): this is where the random seed goes (inside commented-out code, `fault_tolerance` override)
- `ESConfig` (line 125): do we put rng here ? (inside commented-out code, `fault_tolerance` override)

## `core/optimizers/es/optimizer.py`

- `ESOptimizer.__init__` (line 89): (nadine) we support only one planning agent in the config. to extend in future
- `ESOptimizer.__init__` (line 103): (nadine) built-in normalization not supported yet
- `ESOptimizer._has_converged` (line 666): (nadine) implement early stopping criteria — resolved in `54a70a1` and `f25f223`: a convergence stop on the displacement of the search mean is implemented as a flagged heuristic and is off unless `convergence_eps` is set.
- `ESOptimizer.train` (line 692): either move this t reset or mutation requires this is always true

## `core/reporting/base.py`

- `Reporter` (line 33): how to store data in the results reporter ?
- `Reporter._resolve_path` (line 65): remove reduction logic from reporting

## `core/reporting/wandb.py`

- `WandbReporter._series_label` (line 120): when by_agent followed by add, then skip

## `core/types.py`

- `ContextID` (line 13): what if we want the contextID to be a unique UUID and we keep a registry of already existing contextID in the world
- `ContextID` (line 14): registry object for the world.
- `OptimizerID` (line 25): again what if we want a way to register the Optimizer in a memory object and generate a unique uuid for it ?
- `module level` (line 39): create the WorldEnv
- `module level` (line 40): in ray there are different types of envs : BaseEnv, ExternalEnv, ExternalMultiAgentEnv
- `module level` (line 41): i really dont like any because it is not restricting enough. but I want ability to accomodate other environments in the future
- `module level` (line 42): WorldEnv should be also a gymnasium Env with the added feature to have sub envs

## `core/utils.py`

- `module level` (line 20): restrict Any type annotation.

## `core/world/base.py`

- `World.__init__` (line 51): the reporting type annotation to add
- `World.__init__` (line 54): replace with registry
- `World.get_mechanism_by_index` (line 315): fix this function. now the primary key is ctx_id — resolved in `f48cabb`: the lookup now scans the registry for the candidate's batch index and skips `done` entries.
- `World.flush` (line 515): fix this function. now the primary key is ctx_id — resolved in `caf369e`: flushing now removes each mechanism from the three registries together.

## `core/world/context.py`

- `module level` (line 60): some world contexts are singletons (mutable) others are simply mutable.
- `module level` (line 61): for now singleton/or no is deffered to world
- `module level` (line 62): Enums for Context to access different Context Schemas.
- `EnvStepContext` (line 110): strict type annotations rm Any

## `examples/bilevel_fishery/contexts.py`

- `FitnessContext.from_metrics` (line 60): (empty TODO marker, followed by commented-out alternative objective formulas that were deleted; the weighted log-utility formula was kept as a prose comment)

## `examples/bilevel_fishery/debug.py`

- `module level` (line 106): maybe constrain this to always be (1,)
- `module level` (line 109): (nadine) better name ?
- `module level` (line 110): (nadine) what if acts on another type of object such as observation ?
- `module level` (line 247): (nadine) enforce strict shape
- `module level` (line 256): (nadine) enforce strict shape
- `module level` (line 265): (nadine) maybe an observation object needed to avoid hardcoding this
- `module level` (line 283): test queries agg over mechanisms (or other dynamic fields)
- `module level` (line 284): test queries with y keys from reduced (env)

## `examples/bilevel_fishery/metric_schema.py`

- `FisheryMetricSchema` (line 73): move this into logging for mechanism (inside commented-out code: it annotated the commented-out `max_demand_frac` field)

## `examples/bilevel_fishery/queries.py`

- `INNER_QUERIES` (line 351): two ways over junction : either plot them in separete line or mean over
- `INNER_QUERIES` (line 352): what if you wanna avergae over specific type of agent ?
- `INNER_QUERIES` (line 353): seeding : is error bar, by_episode is mean, by_agent is mean -> for that we leave the separation int he mapping. ID should be by type strictly
- `INNER_QUERIES` (line 526): Again seeding over policy ? error bars ?
- `module level` (line 641): x and y axis labels
- `module level` (line 642): eval vs training from different logger instances
- `FISHERY_ENV_QUERIES` (line 644): since risk penality is null, violation singal == quota_penalty
- `FISHERY_ENV_QUERIES` (line 651): plot none when the data is not pushed to prevent experiemnt breaking

## `examples/bilevel_fishery/regulated_env.py`

- `Fisherman.observation` (line 64): (nadine) replace usage with inidividual harvest observation
- `FisheryRegulatedEnv.reset_fishery` (line 141): (nadine) reset should not take mdp and init params should not be stateful
- `FisheryRegulatedEnv.reset_fishery` (line 148): (nadine) change to scipy truncnorm rather than clip to avoid flat signal
- `FisheryRegulatedEnv.pella_tomlinson` (line 180): remove clipping (inside commented-out code: the trailing note of the commented-out `fish_next = float(np.clip(fish_next, 0.0, self.K))` line) — resolved in `53e5ddd`: the new transition has no upper bound at `K` and keeps only the lower bound at 0.
- `FisheryRegulatedEnv.pella_tomlinson` (line 196): (nadine) add observation

## `examples/bilevel_fishery/regulator_env.py`

- `FisheryRegulatorEnv.reward` (line 88): move num_steps here (followed by commented-out `num_steps = getattr(metrics, "iter")`, deleted)
- `FisheryRegulatorEnv.reward` (line 94): when running parallel eval, async may duplicate runs ! should not statistically change the result
- `FisheryRegulatorEnv.reward` (line 99): ensure aggregation by policy seed
- `FisheryRegulatorEnv.reward` (line 107): this is the mean however this is not good representation for late learning mechanisms

## `examples/cartpole/main_appo.py`

- `module level` (line 18): the default mechanism config and fisherman, and observation spaces and action spaces part of config
- `module level` (line 19): where to do ray initialization ? gpu vs cpu - needs to happen when we build optimizer
- `module level` (line 20): num_fisherman
- `module level` (line 21): wire up the evaluation cfg
- `module level` (line 22): seeding API
- `module level` (line 23): experimentation helpers
- `module level` (line 24): review ray configz
- `module level` (line 28): move this to the config ! (refers to the `ModelCatalog.register_custom_model("mps_fcnet", ...)` registration that follows) — resolved in `09033f7`: the cart-pole port removed the custom model and its registration.
- `module level` (line 57): dimension inferred from mechanism ? (trailing comment on `.training(` of the outer `ESConfig`)
- `module level` (line 68): implement early stop for plateau (trailing comment on `train_iters=200` of the outer `ESConfig`)
- `module level` (line 79): fix this its using old api stack (above the commented-out `.model(custom_model="mps_fcnet")`) — resolved in `09033f7`: the commented-out old-API model call is gone and the scripts use the new API stack.
- `module level` (line 81): use the new api stack and better custom model integration (above `.api_stack(`)
- `module level` (line 107): review (trailing comment on `circular_buffer_num_batches=2`)
- `module level` (line 108): review (trailing comment on `circular_buffer_iterations_per_batch=1`)
- `module level` (line 166): (no text on the TODO line; the next comment line read `custom_evaluation_function`, i.e. a custom evaluation function is still to be done)
- `module level` (line 174): add this after run done (above the final `ray.shutdown()`)

## `examples/cartpole/main_ppo.py`

- `module level` (line 18): the default mechanism config and fisherman, and observation spaces and action spaces part of config
- `module level` (line 19): where to do ray initialization ? gpu vs cpu - needs to happen when we build optimizer
- `module level` (line 20): num_fisherman
- `module level` (line 21): wire up the evaluation cfg
- `module level` (line 22): seeding API
- `module level` (line 23): experimentation helpers
- `module level` (line 24): review ray configz
- `module level` (line 28): move this to the config ! (refers to the `ModelCatalog.register_custom_model("mps_fcnet", ...)` registration that follows) — resolved in `09033f7`: the cart-pole port removed the custom model and its registration.
- `module level` (line 57): dimension inferred from mechanism ? (trailing comment on `.training(` of the outer `ESConfig`)
- `module level` (line 68): implement early stop for plateau (trailing comment on `train_iters=100` of the outer `ESConfig`)
- `module level` (line 79): fix this its using old api stack (above the commented-out `.model(custom_model="mps_fcnet")`) — resolved in `09033f7`: the commented-out old-API model call is gone and the scripts use the new API stack.
- `module level` (line 81): use the new api stack and better custom model integration (above `.api_stack(`)
- `module level` (line 105): review (inside commented-out code, trailing comment on `# circular_buffer_num_batches=2,`)
- `module level` (line 106): review (inside commented-out code, trailing comment on `# circular_buffer_iterations_per_batch=1,`)
- `module level` (line 164): (no text on the TODO line; the next comment line read `custom_evaluation_function`, i.e. a custom evaluation function is still to be done)
- `module level` (line 172): add this after run done (above the final `ray.shutdown()`)

## `examples/cartpole/regulated_env.py`

- `module level` (line 26): add multiagent state in types
- `module level` (line 27): ban proportional to violation severity
- `module level` (line 30): number of agents spawned dynamically as a byproduct of config stating number of agents
- `CartpoleRegulatedEnv.transition_kernel` (line 138): (no text; bare trailing `# TODO` on the `pass` body of the method)
- `CartpoleRegulatedEnv._observation` (line 146): canonical observation in base multiagent env

## `examples/fresh_water/debug.py`

- `module level` (line 30): adding defaults (above `space=WaterMechanismSpace()` in the `.mechanism(...)` call)
- `module level` (line 93): move this to Raven helper (above `"full_stage_m": 420.41` in the `ecology_cfg` of the inner environment config)

## `examples/fresh_water/regulated_env_ed_hs_v4.py`

- `WaterRegulatedEdHsEnv._estimate_temp_c` (line 194): temporary since raven does not output temperature
- `WaterRegulatedEdHsEnv.intrinsic_utility` (line 251): must also retreive the time of the day to water in order to normalize per seconds (the note continues on the next comment line: `per seconds`; it sits above `full_required_m3_day = (deficit_mm_day / 1000.0 * self.max_farm_area_m2 # / 86400.0)`)
- `WaterRegulatedEdHsEnv.violation_signal` (line 319): review this (above `quota_violation_m3_day = max(0.0, requested_m3_day - allowed_m3_day)`)
- `WaterRegulatedEdHsEnv.transition_kernel` (line 457): review this (above the comment `compute flow penalty`, before `release_pressure = min(...)`)
- `WaterRegulatedEdHsEnv.transition_kernel` (line 459): this may need to be capped or may explode (above `release_pressure = min(...)`)
- `WaterRegulatedEdHsEnv.transition_kernel` (line 487): this updates every time step - find way to update once at reset (above `self._update_infos(key="baseline_ref", ...)`)
- `WaterRegulatedEdHsEnv._observation` (line 527): Q : what can we normalize streamflow_m3s with ? (two commented-out lines that followed were removed with it: `# streamflow_m3s = float(S_t.get("streamflow_m3s", 0.0))` and `# streamflow_norm = streamflow_m3s / max(EPS, self.streamflow_ref_m3s)`)

## `examples/fresh_water/regulator_env_raven.py`

- `WaterRegulatorRavenEnv.aggregate_rewards` (line 81): group by seed as well (the note continues on the next comment line: `for now we assume aggregation over seed`; it sits above the loop filling `by_run`)
- `WaterRegulatorRavenEnv.aggregate_rewards` (line 99): for now we average accross agents but later another solution required! (above `if isinstance(r, dict):` in the reward loop)
- `WaterRegulatorRavenEnv.aggregate_rewards` (line 114): take into consideration economic vs susteinability reward (above `max_idx = max(economic_by_m.keys())`)
- `WaterRegulatorRavenEnv.aggregate_rewards` (line 134): add the mechanism object (trailing comment on `mechanism=None,` in the `MechanismContext(...)` call)

