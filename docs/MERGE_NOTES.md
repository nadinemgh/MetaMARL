# Notes for Nadine: the cleanup pass on `feature/social-influence`

This branch, `feature/social-influence-testing-v2`, starts from your branch at
`96294f6` (3 October 2026) and leaves your branch untouched. Rémy asked for the same
pass as in August on your two earlier branches: make every entry point run end to
end, bring the unit coverage of `core/` above 90 %, document every public symbol,
fix every defect found with a test, rewrite the guides against the code, and make
the notebooks run. These notes tell you what changed under your code and what is
left for you to decide. Every change is its own commit with its own test, so any of
them can be reverted on its own; the hashes below are those of the branch when it is
pushed.

The state at the end of the pass is the following. The suite has about 1890 tests,
including the doctests of `core/` and `examples/`, and covers 99 % of `core/`. The
three examples run end to end, and two runs with the same seeds give bit-identical
fitness. `ruff check --no-fix .` and `ruff format --check .` pass on the whole tree
with your configuration, and continuous integration runs the lint, the unit tests,
the integration tests and the five tutorial notebooks. The README, QUICKSTART, AGENTS and
`docs/ARCHITECTURE.md` describe the present code; the commands they quote were
executed.

## How to look at it

```bash
git switch feature/social-influence-testing-v2
uv sync
uv run python -m pytest -m "not integration and not notebook" -q
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \
    --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 --reporter csv
```

The fishery run takes about fifteen seconds and writes its plots as CSV files under
`results/bilevel/`. QUICKSTART explains what to read in its log.

## Changes that alter results

These are the commits after which a run of your experiments gives different
numbers. Where a measurement was taken, it is quoted.

**Rewards and observations are per step.** The fishery's reward was a running sum
over the episode and the observation a running sum of the stock; both are now the
values of the step (`27fbd29`, `0bb367c`). In the shrunk configuration an episode
return fell from 205.9 to 19.4 for the same harvests.

**Every environment and every episode plays the candidate.** Before the pass, an
environment could reset without the candidate it was meant to evaluate and play
the following episodes without any mechanism, and only the first seed received
the candidate. Each environment now keeps its candidate for every episode
(`b330a82`), every seed receives it (`dea1919`) and the evaluation builds one
environment per mechanism (`03330ef`). Until the first candidate arrives, leaders
play the default action of their mechanisms (`472feb1`); in the shrunk fishery
this changed no training or evaluation episode, because the 48 resets that play
the default all happen while RLlib builds its runners. A mechanism default that is
falsy, such as 0, is no longer dropped (`5bc72ab`).

**Runs are reproducible.** Four sources of run-to-run variation were removed: the
environment's metric logger is reset at every episode (`439249a`), the
`disable_env_checking` setting reaches RLlib (`be8d83e`), every entry point
restarts itself with `PYTHONHASHSEED=0` when the variable is unset (`2dd94c9`),
and evaluation waits for APPO's learner thread to apply every queued batch
(`ac83a64`). Five shrunk and three full-size runs then gave bit-identical fitness.

**The fishery dynamics are the standard discrete surplus-production model.** The
fishers used to harvest one after the other, each from what the previous one
left, with the growth computed on the previous step's stock. The transition is now
`B(t+1) = max(B(t) + g(B(t)) + restoration(t) + noise(t) - C(t), 0)` with the
Pella-Tomlinson production on the current stock; every fisher requests from
`B(t)`, and a total request above the stock is shared pro rata (`53e5ddd`, on
Rémy's decision). The constants `FISH_NORM`, `USAGE_NORM` and `HARVEST` went with
the old transition. No before-and-after fitness was recorded for this change.

**The fitness reads the real tail window.** The fishery fitness now uses the last
`fitness_tail_steps` (50) steps of each evaluation episode, as its docstring
always said (`3205b73`, on Rémy's decision).

**Smaller corrections.** The quota publishes the allowed fraction of the step
instead of accumulating it (it read 1.0, 2.0, 3.0 over three steps; `ee062e9`). A
horizon of N gives N steps (`4b8449b`). `decode` receives the raw policy output at
every step through the new `MDPState.raw_actions`, so it no longer has to be
idempotent; the four mechanisms are bit-identical under this change (`56dde42`).
The bilevel optimizer forwards its number of episodes to the ES and takes
`converged` from the ES summary (`0cda396`, `95d3c30`). `debugging` scales the same
whatever the order of its calls (`d9d29fc`).

**Training curves average every episode.** An environment that ends several
episodes in one iteration used to report only the last one. Every episode is now
kept under a key ending in `|n=<index>` (`ed09a16`), and the metric logger records
a gap for a dynamic key a push leaves out, which the reporter's mean skips
(`b9998d4`). The fishery fitness is unchanged, since evaluation ends one episode
per environment; in a cart-pole APPO iteration one environment ended 12 episodes,
of which the old code kept 1.

**The logged metrics are the right ones.** The step reward is logged, so
`mean_reward` is no longer NaN (`733be6c`); terminal values use the last value
instead of the maximum (`bc5a02d`); the policy loss is read from the new API stack
(`3ec65bc`), the learner's batch size may be fractional (`92d93d9`), and a
statistic the learner does not report is NaN (`6e512e9`); the reporter reaches the
optimizer, a configuration without one runs, and every query of the fishery
renders on its own (`19125d3`, `d3b1dc5`, `7e6bd39`), which gives 16 CSV files with
unchanged fitness. Five query titles spelled "episdoes"; the CSV files now say
"episodes" (`c23531d`), so a script that reads them by name needs the new names.

**The mechanisms you started are usable.** `Subsidy`, `ThresholdPenalty` and
`SocialInfluence` are ported to the `Mechanism.apply` interface on the model of
`Quota` (`d317b9f`, `20f878e`, `dbe2338`), and `Mechanism.observe` lets a
mechanism add to the observation (`5c2c52f`). On a shrunk fishery whose
regulator holds a quota, a subsidy and a penalty, the reward equals the harvest
plus the subsidy and penalty residuals to 1.1e-16 over 4256 agent-steps.

**The fresh-water example was ported, then changed on Rémy's decision.** The port
(`2010b10`) fixed two defects: the no-withdrawal baseline was copied once and never
re-run, so the deviation was zero by construction, and the crop stage rebuilt the
planting date from the current year, so about 37 % of episodes had a free
off-season reward. Fitness values from before the port are therefore not
comparable. Rémy then decided four changes. The sustainability term measures how
far the farms lower the lake level (`ad73d49`, which renames
`streamflow_deviation` to `level_deviation` and removes `deviation_series`). The
delivered-water observation is divided by the largest daily need of all farms
(`5fe4446`). The residence time is the stored volume over the outflow, in days,
where it read about 1e-6 (`2591cbf`). The Raven baseline runs once per episode
instead of once per day (`e6a3015`). With 20 farms, 30-day episodes and three
generations, the fitness moved from 1.2533, 1.2020 and 1.2250 to 1.2816, 1.2319 and
1.2595, and the residence time now ranges from 42 to 158 days.

**Three behaviour changes decided on 5 October.** The regulator environment
hands the same candidate to every leader and each leader applies the mechanisms
it owns, yet the ES built its search space from the first regulator agent only, so
a second leader never received a candidate. The ES now searches the union of the
mechanisms of every regulator agent (`a65b543`). Every quota wrote its allowance
to the same state entry, so a second quota overwrote the first; the entry is now
`allowed_frac:<mechanism id>` (`6bf771c`). Two quotas on one target still sum
their corrections, so the fisher is cut twice; the docstring and a test now state
it. Social influence sorted the peers by their identifier as a string, which puts
`fisherman:10` before `fisherman:2` beyond ten fishers; the numeric index now
decides, and identifiers without one follow in string order (`14557d7`). The
single-regulator fishery with fewer than eleven fishers is unchanged by all three.

**Averaged curves keep their order.** A `MEAN` reduction over a `SERIES` level
iterated a set, whose order depends on `PYTHONHASHSEED`, so the order of the
averaged curves, and with it their colours and their place in the legend, could
change from one run to the next. It now keeps the sorted order of the groups
(`e00ec36`).

## Interface changes

The following were deleted because nothing called them or because they could not
work. The `World` lost its claim and update methods (`get_mechanism`,
`try_get_mechanism`, `update_context`, the statuses `assigned` and `init`;
`c1f091f`) and its unused context helpers (`get_ctx_registry`, the
`get_*env_step_contexts` family, `set_new_context`, `remove_context`, `flush_ctx`,
and `core.utils.add`; `ae0e56f`). The agent hook decorators for actions, rewards
and observations went (`be2d354`), with `Agent._normalize_action` (`da2fb4a`) and
`violation_transition_width` (`31d8e30`). On the reporting side, `ReporterType`,
`Resolution`, `Reporter.schema` and the `reporting(schema=...)` argument are gone
(`8608dfb`, `9b6a9d9`, `bbba0f9`, `c0a24d3`). In the optimizers, `ESConfig` lost
`generation` and `dimension` (`1510447`); `BilevelConfig` lost
`default_mechanism` and `output_dir`, and `BilevelOptimizer` its history
attributes (`3206a8f`); the Ray layer lost the MPS model, `eval_episodes`,
`rollout_fragment_length`, `RayRuntime._initialized` (`c0446c8`),
`_build_agent_policy_map`, `_get_policy_handle` (`82aaac2`) and `_inner_iter`
(`5a8c05a`). `RegulatorEnv` lost its `mode` parameter, `env_id` and `_t_agents`,
and `MultiAgentEnv` its `_t` counter (`c180199`). In the fishery, the regulator no
longer has `K` or `raw_sustainability_threshold` (`4c416a4`), `total_fines` and
`trajectories` are gone (`4b1b39d`, `7c49367`), the metric schema lost the fields
the new dynamics no longer produce (`requested_frac`, `quota_violation`,
`quota_penalty`, `risk_penalty`, `quota_stress`, `allowed_harvest`, `fish_norm`;
`3205b73`), and the "Restoration subsidy vs fixed quota" plot was removed with the
subsidy it plotted (`34ae6f6`). Thirteen dead files went at the start of the pass,
among them `core/registry.py` (`9ca215b`).

A few things were added. `MDPState.raw_actions` carries the raw policy output
(`56dde42`); `Mechanism.observe` shapes observations (`5c2c52f`); one
`ACTION_TEMPERATURE` in `core.utils` replaces four copies (`e5148b2`);
`OptimizerConfig.training` returns the config so it chains (`1d0d0f4`); the ES
summary carries `episodes` and `converged` (`6eea234`); `ActType` comes from
gymnasium and the `gym` dependency is gone (`ae31d93`, `4c168d8`). The project is
now called `metamarl`, after the repository, and installs a `metamarl` command
that runs `metamarl run config.yaml` and `metamarl check config.yaml` (`5a95efa`,
`bd5254f`); `python -m core.config.cli` still works.

Two names changed with the behaviour changes above. Code that reads the quota's
allowance must read `allowed_frac:<mechanism id>`, for example
`allowed_frac:quota`, instead of `allowed_frac` (`6bf771c`). `ESOptimizer` no
longer has the `agents_cfgs` attribute that held the first regulator agent
(`a65b543`).

## New errors

Several silent failures now raise. The ES rejects bounds other than `[0, 1]`
(`c99a214`). The Ray optimizer raises without an evaluation setup (`9cfe798`),
without a config or a batch capacity (`16d1c6d`), without a World and agents
(`9f765da`) or without training seeds (`0477d20`). A regulator without a horizon is
rejected before the ES loop, which would otherwise never end (`4256d1d`), and so is
a conflict between the bilevel and ES episode counts (`0cda396`). Two methods
carrying the same hook mark (`05dfc50`, `bfcc5c5`), a trajectory that skips a step
(`29f351d`), a context that fails validation (`a18c5b8`), a fetch mode other than
train or eval (`5a2f552`), an unknown `WandbConfig` argument (`86b0c80`) and a
negative `influence_weight` (`dbe2338`) raise as well. A failed mechanism fetch
names the mechanism (`5f2b295`, `4efe1de`). The ES raises a `ValueError` that names
both agents when two regulator agents declare the same mechanism id, since the
candidate is keyed by mechanism id, and a `ValueError` instead of an
`AttributeError` when no regulator agent was declared (`a65b543`).

`ESConfig` and its `training` builder accepted any keyword and ignored it, so a
misspelled option such as `mean_rl` kept the default without a word; an unknown
argument now raises `TypeError` (`bbd97a8`). On Rémy's decision,
`ESConfig.debugging` also rejects `num_seeds` (`2ff20a1`). The outer level never
read its own seeds: the regulator publishes one context per training seed of the
society, which `BilevelConfig` passes to it, so an outer seed count had no effect.
The debug scripts, the fishery YAML and two tutorial cells dropped their outer
`num_seeds=1`; set the count on the society configuration.

## Kept on Rémy's decision

These choices were discussed and kept as they are; they are listed so you can
revisit them. The console silencing stays: W&B runs with `quiet`, the Ray loggers
are at `WARNING` and `log_to_driver=False`. The fisher's reward is its harvest
fraction after the regulator's quota and before the stock's pro-rata rationing, so a
fisher is paid for what the quota lets it request even when the stock cannot
deliver it; the composition test of the fishery measures it. In the fresh-water
example, `under_irrigation_penalty_scale` and `max_farm_area_m2` are searched and
observed but read by no dynamics, so the ES spends two of its eight dimensions on
them. The cart-pole example is a pipeline check: every step of `CartPole-v1` pays
1.0, so its fitness is the constant 1.0 and its dial is inert. The ES convergence
stop is implemented but off unless `convergence_eps` is set, because in
single-candidate mode it fired after 12 to 41 generations far from the optimum.

## Questions for you

`SocialInfluence` only shapes the observation, and its `influence_weight` has no
effect. Either the influence term of Jaques et al. (2019, PMLR 97) is implemented,
which needs the counterfactual policies, or the class is renamed for what it does.

`RayOptimizer.stop` returns the metrics reduced over the run, while the base
`Optimizer.stop` returns `None` and is annotated `-> Any`. Nothing reads the
returned metrics; whether `stop` should return a result at all is yours to choose.

Sections 4 and 5 of your notes in `TODO.md` still describe the earlier constructors
of the subsidy and the social influence. Their acceptance lists are covered by
tests. The quota's observation question in the archived notes is still open:
`Mechanism.observe` now exists, but `QuotaMechanism` does not implement it.

A fresh-water configuration that still passes `deviation_series` is silently
ignored, like any unknown key of `ecology_cfg`. Whether `ecology_cfg` should reject
unknown keys is a design choice for all three examples.

The wait for the learner thread before evaluation reads private RLlib attributes.
It raises rather than skipping when they are missing, and it does not cover GPU
learners or IMPALA's deque queue.

## To do together

The Raven path of the fresh-water example is tested only against a stand-in
executable, since the Raven model is not in the repository, and a run with the 500
farms of the default configuration did not finish within ten minutes. Both need
your model.

On 4 October, in the shrunk fishery configuration, the four fitness values of a
generation were bit-identical to those of the previous one although the candidates
differed. Each population slot trains its own policy module with its own initial
weights, so the ES gradient may follow the differences between slot initialisations
rather than the mechanism. The measurement predates the new dynamics and the tail
window; it is repeated on the shrunk and full configurations before the push, and
its result will be added here.

The plans at the top of `TODO.md` were reconciled with the branch box by box: each
box now ends with a marker that names the commit doing it, partly doing it or making
it obsolete (`245a3aa`). The boxes about parity with `dev` stay open, because no run
compared `dev` with this branch; only the reproducibility of this branch from one
run to the next was measured. That comparison needs your judgment on which numbers
should match, now that the dynamics and the per-step rewards changed on purpose.

## The tutorials

Most of the code of the tutorials sat in fenced markdown blocks that never ran,
and it had drifted from the API without any test failing. Every code example is
now an executable cell, the sections that taught removed concepts were rewritten
to their present equivalent, and a test runs each notebook end to end under the
`notebook` marker, in its own CI job (`6aaee13`). The custom benchmark tutorial
builds a common-pool environment, an extraction mechanism and a custom scarcity
tax, tests them analytically and runs a tiny bilevel experiment (`97af7e4`). The
mechanism tutorial steps the real fishery under each mechanism and closes with a
tiny training run (`13aff1b`). The visualization tutorial draws every query from
made-up metrics against the present reporting API (`268b500`). The continuous-time
notebook is mathematics only and did not change. Together the five run in about
two minutes.

Your fishery tutorial now runs at a reduced scale by default, so that it executes
in about a minute: 12 ES generations, 8 APPO iterations per generation, a horizon
of 50, 4 fishers and one evaluation seed (`279fba0`). Its Section 10 gives the
values of the original experiment (1000 generations, 50 iterations, a horizon of
100, 10 fishers, 3 evaluation seeds) and says to set them to launch the real run.
It writes an offline W&B run under `tutorials/wandb/`. The line `!tutorials/*` of
`.gitignore` un-ignored that directory and the CSV output of the other tutorials,
so it was removed (`f77ea7b`).

The notebook pass found four documentation defects in `core/` and the fishery,
each fixed in its own commit: `raw_actions` also receives the leaders' residuals
(`45cc14f`), the Ray schema is never passed to `reporting` (`e6dc300`), the
rollout schema is keyed by the environment seed and the learner schema by the
policy seed (`17f48b6`), and the society environment reads no regulator `K`
(`acd529b`). One limit is worth knowing: the fishery observation has five entries
and the fisher fills two of them, so social influence on a scalar action fits at
most three fishers, and a fourth raises an error that names the missing entries.

## Tooling

The development tools are a `dev` dependency group (pytest, pytest-cov, ruff,
nbconvert, ipykernel, nbformat, spacing, TensorBoard), `spacing` is no longer a
runtime dependency, and TensorBoard is also an optional extra (`9ca215b`,
`fd7ba11`). `pytest.ini` runs the doctests of `core/` and `examples/`. The whole
tree complies with your ruff configuration (`e9ea8fc`); the 211 TODO comments of
the code were moved verbatim to the last section of `TODO.md`, and those that later
commits resolved say so. Notebooks are no longer git-ignored (`ff67552`). No library
module configures the root logger at import any more; the entry points do it, before
the hash-seed check so that its notice is printed (`da92de8`, `00e40ec`). The
messages of the Ray optimizer start with `[Ray]` instead of `[PPO]`, since it
trains APPO as well (`f172c1d`). `MultiAgentEnv.reset` no longer carries
`@override(gym.Env)`, since the class does not derive from `gym.Env`, and a test
checks that every `@override` names a base of its class (`a0dc6cc`). The
`# Deprecated` comment above the `to_float` import of the Ray optimizer was removed
on Rémy's decision, because the function is used and nothing marks it as deprecated;
tell us if you meant to replace it (`fa3085e`). The example of the `Query` module
docstring used a path that no schema resolves; it is now one of the fishery's
queries, checked by a test (`d3e3b0b`). Every
public symbol of `core/` and of the examples has a NumPy docstring with a runnable
example, and the public classes and functions say when to use them.
