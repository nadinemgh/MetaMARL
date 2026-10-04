# Resume file — cleanup, tests and documentation pass on `feature/social-influence` (October 2026)

**Read this first in every session.** It is the status board of the pass: what is done,
what is in progress, what remains, what waits on a decision, and how to resume from disk.
Update it before every `/clear`, phase change or break.

## Context

Nadine asked on 2026-10-04 to redo, on the branch she now uses, the pass that was done
between 2026-08-27 and 2026-09-01 on her two earlier branches. That earlier pass is
recorded in `docs/REPRISE.md` of `feature/social-influence-testing` (read it with
`git show feature/social-influence-testing:docs/REPRISE.md`) and of
`feature/integration-trial` (worktree `../bilevel-fishery-integration`). Its method was: make the branch run end to end, bring
unit coverage of `core/` above 90 %, document every public symbol, rewrite the guides
against the tree, execute every notebook, then audit everything by measurement.

## Where things live

| Branch | Worktree | Role |
| --- | --- | --- |
| `feature/social-influence` | none (remote only) | Nadine's branch; never modified by this pass |
| `feature/social-influence-testing-v2` | `../bilevel-fishery` (main directory) | this pass; created from `origin/feature/social-influence` at `96294f6` (2026-10-03); local only so far |
| `feature/social-influence-testing` | none (branch kept, pushed) | the August pass on the earlier mechanism branch |
| `feature/integration-trial` | `../bilevel-fishery-integration` | source of the earlier tests and guides to port |

`dev` is never modified.

## Status board

- [x] Earlier pass reconstructed from the resume files and git history (2026-10-04).
- [x] Working branch and worktree created (2026-10-04).
- [x] Baseline measured on `96294f6` (2026-10-04); see the next section.
- [x] Plan and scope agreed with Rémy (2026-10-04).
- [x] Phase 0 — test and packaging base (2026-10-04):
  - [x] development dependency group (`pytest`, `pytest-cov`, `ruff`, `nbconvert`, `ipykernel`, `nbformat`, `tensorboard`) and a `tensorboard` extra, which the code already pointed to; the unused `dev` extra was removed; the lock changed no existing version;
  - [x] `pytest.ini` no longer covers the deleted `src` package and declares a `notebook` marker;
  - [x] `tests/conftest.py` with the shared Ray session and the Ray-last ordering hook (the fakes of the earlier suite come in phase 2, once checked against the new `World`);
  - [x] continuous integration in `.github/workflows/ci.yml` (lint with Nadine's full ruff configuration, unit tests, integration tests);
  - [x] whole tree formatted with `ruff format` and imports sorted; program structure checked identical file by file;
  - [x] full compliance with Nadine's ruff configuration on every live file (`ruff check --no-fix` and `ruff format --check` pass; the 269 remaining errors are all in the thirteen dead files). Commented-out code was deleted, the 211 TODO comments were moved verbatim to a new last section of `TODO.md` (every line number checked against `96294f6`), long lines were rewrapped and implicit concatenations made explicit. A comparison of syntax trees against the pre-edit snapshot shows that, apart from comments, docstring rewraps and import changes, the only code change is the removal of an unused `obs` assignment in `QuotaMechanism.apply` (recorded in `TODO.md` as an open design question). All 57 modules of `core/` import and `core.config.cli check` accepts the fishery configuration;
  - [x] tutorials: the commented-out worked examples of `custom_benchmark_creation` and `mechanism_algorithms` were turned into fenced code blocks in markdown cells instead of being deleted, because deleting them would empty whole tutorial sections; phase 5 turns them back into runnable cells once the API they show works;
  - [x] phase 0 committed in two commits (tooling, then lint compliance).
  - [x] removal of the thirteen dead files (`core/registry.py`, five uncollected scripts, five test files that fail at import, two unread YAML files), run by Rémy on 10-04 after the safety guard refused it to Claude; `ruff check --no-fix .` and `ruff format --check .` now pass on the whole tree;
- [ ] Phase 1 — every entry point of the fishery example runs end to end (in progress):
  - [x] Ray could not start under `uv run`: the August override of `RAY_ENABLE_UV_RUN_RUNTIME_ENV` was lost in a merge on Nadine's branch while its docstring survived; restored with its test (`bfb3bed`);
  - [x] the command line now logs the traceback of a configuration error instead of only the failing YAML path (`a882c1b`);
  - [x] `RayOptimizer` raises a `ValueError` naming the missing `.evaluation(...)` call instead of `'NoneType' object has no attribute 'get'` (`9cfe798`);
  - [x] `examples/bilevel_fishery/config.yaml` lacked the society's env runners, learners, evaluation, fishermen, fault tolerance, debugging and reporting; completed with the values of `debug.py` on Rémy's decision. A field-by-field comparison of both built configs differs only in the random world-name suffix and in the quota default (float32 in the YAML, float64 in `debug.py`) (`2f9021d`);
  - [x] `ESOptimizer.train` returns `episodes` (generations run) and `converged`, the keys `BilevelOptimizer.train` reads; this also removes the `UnboundLocalError` at zero generations (`6eea234`);
  - [x] two `Reporter` error messages lacked the `f` prefix (`9adb6ca`);
  - [x] the ES bounds check lacked parentheses and let spaces such as [0, 2] through (`c99a214`);
  - [x] measured: `python -m core.config.cli run` completes on a shrunk copy of the YAML (2 generations × 2 inner iterations, 2 fishermen, horizon 20, about 13 s, exit 0); the `mechanism_algorithms` tutorial still executes;
  - [x] fitness tail window: Rémy decided on 10-04 not to change the objective and to record the finding for Nadine (see "Findings for Nadine");
  - [x] `fish_norm_next_last` reduced with `MAX` instead of `LAST` (`bc5a02d`);
  - [x] **rewards were cumulative**: `MDPState.rewards` used the carry-forward `Trajectory`, so the reward RLlib received at step t was the sum of all rewards since the start of the episode (measured: 0.89463 at t=2 for harvests 0.45785 and 0.43679, 9.47 at t=20; behaviour introduced by `32ef1b5` on 09-30). Fixed with a `FlowTrajectory` whose unwritten steps start at 0 while same-step deltas still sum, on Rémy's decision (`27fbd29`). Re-measured: reward equals harvest at every step; episode return of the shrunk config 205.9 → 19.4; the three tutorials that executed at baseline still execute;
  - [x] the step reward is logged again, as the env docstring states, into the five reward fields (`733be6c`); `mean_reward` is no longer NaN;
  - [x] `policy_loss=NA` on every training line: the reader only knew the classic `info/learner` layout; it now reads `learners/<module>/policy_loss` (`3ec65bc`). Measured on 45 inner iterations: APPO's learner reports its stats only once every 20 gradient updates (`IMPALALearner.update` in Ray 2.53), so real losses appear at iterations 21 and 42 and the other lines correctly print NA;
  - [x] the `Mean of empty slice` warnings were traced to RLlib's own EMA timers (`env_reset_timer` and connector timers of the env runners) on iterations where they measured nothing; they do not touch learning and are left visible;
  - [x] `Optimizer.__init__` read `config.episodes` before its `None` guard, and an unset batch capacity raised a bare `AttributeError`; it now raises a `RuntimeError` naming the optimizer (`16d1c6d`);
  - [x] `RayOptimizer.stop` raised `TypeError` on a misspelled keyword and nothing called it; `BilevelOptimizer.train` now stops both levels in its `finally` block on Rémy's decision, and the stop is logged (`691e08e`); whether `stop` should return the reduced metrics is left to Nadine;
  - [x] the two unused helpers `_build_agent_policy_map` and `_get_policy_handle` (the latter broken) were removed on Rémy's decision (`82aaac2`); `TODO.md` still mentions the first one;
  - [x] `RayOptimizerConfig.build_optimizer` now requires a World and declared agents, with an error naming what to add, on Rémy's decision (`9f765da`);
  - [x] **the regulator's candidate reached training only in the first episode of each environment**: the environment now keeps the last candidate it fetched and gives it to the leaders at every reset, on Rémy's decision (`b330a82`). Measured on the shrunk config: every sampled training episode (16 of 16) and every evaluation episode (24 of 24) carries the candidate; the resets without one all happen while RLlib builds its env runners, before the generation's candidates are published. In that config the quota never binds (the stock stays between 0.64 and 0.85 of capacity, untrained fishermen request about half their capacity, the quota allows at least 93 %), so training returns are unchanged by the fix there;
  - [ ] **with more than one training seed, only the last seed's environments receive the candidate** (measured 10-04 with two seeds): `FisheryRegulatorEnv.step` builds one context per seed but calls `append_context` after the seed loop instead of inside it, while its docstring says one context is published per (candidate, seed). Fix waiting on Rémy;
  - [ ] **with more than one training seed, evaluation crashes** ("fewer units than requested: requested=24, completed=48"): `debugging` multiplies `num_envs_per_env_runner` by the number of seeds for the training runners, and the evaluation runners inherit that count, while the evaluation layout and `evaluation_duration` expect one environment per mechanism on each evaluation runner. The error message also says "fewer" when the count is higher. Fix waiting on Rémy;
  - [ ] **runs are not reproducible**: two runs of the same config give different ES trajectories (see "Findings for Nadine"); investigation in progress on Rémy's decision, to resume now that the mechanism fix is in;
  - [ ] `debug.py` documents `--outer-iters`, `--train-iters`, `--num-agents`, `--horizon` and `--reporter` but parses no option and runs 1000 generations;
  - [ ] port of `Subsidy`, `SocialInfluence` and `ThresholdPenalty` (own session, see next step).
- [ ] Phase 2 — tests ported from `feature/integration-trial` and written for the new code; coverage target above 90 % on `core/`.
- [ ] Phase 3 — docstrings and type hints on every public symbol of `core/` and `examples/bilevel_fishery`.
- [ ] Phase 4 — README, QUICKSTART, AGENTS, ARCHITECTURE and notes for Nadine rewritten against the tree.
- [ ] Phase 5 — the five notebooks execute and are exercised by the test suite.
- [ ] Phase 6 — validation audit by measurement, then push.

## Baseline measured on `96294f6` (2026-10-04)

The raw logs of these measurements were written to a session scratch directory and are
not kept; every number below can be reproduced with the commands in the last section.

**Install.** `uv sync` succeeds from the lock file. There is no development dependency
group: `pytest`, `pytest-cov` and `ruff` are runtime dependencies, and the `dev` extra
lists tools the repository does not use. `nbconvert` and `ipykernel` are not installed.

**Lint.** `ruff check . --no-fix` reports 1855 errors, of which about 1035 are the rules
that flag every TODO comment and 448 flag commented-out code. `ruff format --check`
would reformat 63 of 90 files. The configuration sets `fix = true`, so a plain
`ruff check .` rewrites files.

**Tests.** The suite collects zero tests: the five test files fail at import, four on a
`src` package that no longer exists and one on `core.envs.base`. Four underscore-prefixed
scripts and a `main.py` in `tests/integration` are never collected and import deleted
modules. Coverage is therefore unmeasured.

**End-to-end runs.**
`python -m core.config.cli check` accepts the fishery YAML, but `run` fails in
`core/adaptors/ray/optimizer.py:91`, where `evaluation_config` is `None`; the command line
hides the cause behind a generic configuration error.
`examples/bilevel_fishery/debug.py` has no command-line options although its docstring
documents some; with its hard-coded sizes it trains correctly but would run 1000 outer
iterations. A shrunk copy completes two generations in about 15 s and then crashes at
`core/optimizers/bilevel.py:252` on `result["episodes"]`, a key the ES optimizer does not
return. The cartpole and fresh-water examples fail at import. Ray starts correctly under
`uv run`.

**Mechanisms.** Only `Quota` implements the new `Mechanism.apply` interface. `Subsidy`,
`SocialInfluence` and `ThresholdPenalty` still have the earlier shape and cannot be
instantiated. This was read from the code, not exercised by a run.

**Notebooks.** Three of five execute. `metamarl_fishery_tutorial` fails at its import cell
because `examples` is not importable from `tutorials/`. `visualization` fails on
`Query(..., reduce=...)`, an argument that was removed.

**Docstrings.** 71 % of public symbols in `core/` and `examples/bilevel_fishery` have a
docstring (304 of 429) and 92 % of public callables are fully annotated. The gap is
concentrated in `core/agents`, `core/mechanism`, `core/envs` and the CSV and TensorBoard
reporters. Several existing docstrings describe classes that were removed.

**Earlier test suite against the new code.** Of 57 files exported from
`feature/integration-trial`, 13 pass unchanged; in total 254 tests pass, 123 fail and 35
files cannot be imported. The metrics, callbacks, utilities and Ray runtime layers are
stable. The breakage follows the refactor: mechanism classes renamed and re-shaped,
`MultiAgentRegulatedEnv` replaced by `MultiAgentEnv` plus an RLlib adapter, composition
classes removed, `run()` renamed `train()`, `BilevelConfig.society` and `.regulator`
replacing `.mechanism`, `.inner` and `.outer`, and `Series` and `ParallelCoordinatesQuery`
removed from the reporting queries. `integration/test_fishery_reporting.py` hangs and must
be run with a timeout.

**Other findings for Nadine.** `core/registry.py` imports ten missing modules and is used
nowhere. The `spacing` formatter is a GPL-3.0 runtime dependency of a BSD-3-Clause
project. `logging.basicConfig` is called at import time in two library modules. The
README describes directories and commands that do not exist and ends with leftover
assistant text.

## Decisions log

| Date | Decision | By |
| --- | --- | --- |
| 10-04 | Work on a new branch `feature/social-influence-testing-v2` created from Nadine's branch; her branch is not modified | Rémy |
| 10-04 | The six-phase plan of the status board is approved | Rémy |
| 10-04 | Port `Subsidy`, `SocialInfluence` and `ThresholdPenalty` to the new `Mechanism.apply` interface on the model of `Quota`, with tests | Rémy, on Claude's recommendation |
| 10-04 | Leave the cartpole and fresh-water examples unported and record them for Nadine | Rémy, on Claude's recommendation |
| 10-04 | ~~Keep Nadine's `ruff.toml` untouched; fix genuine errors only and report that the check cannot pass as configured~~ (superseded the same day) | Rémy, on Claude's recommendation |
| 10-04 | The branch is ours: make the whole tree pass Nadine's `ruff.toml` and formatter rather than leaving half the work to her; the configuration itself stays untouched | Rémy |
| 10-04 | TODO comments are moved verbatim into `TODO.md`, grouped by file, then removed from the code; commented-out code is deleted (it stays in git history) | Rémy, on Claude's recommendation |
| 10-04 | In the tutorials, commented-out worked examples become fenced code blocks in markdown cells rather than being deleted | Rémy, on Claude's recommendation |
| 10-04 | Delete `core/registry.py` and the five uncollected scripts in `tests/integration` | Rémy, on Claude's recommendation |
| 10-04 | The pass lives in the main directory, which was switched to the new branch; the temporary audit worktree was removed | Rémy |
| 10-04 | Complete `config.yaml` with exactly the values of `debug.py` and add a clear error when the evaluation setup is missing | Rémy, on Claude's recommendation |
| 10-04 | Leave the ES objective and its no-op tail window unchanged; record the finding for Nadine | Rémy, on Claude's recommendation |
| 10-04 | Log the step reward and reduce `fish_norm_next_last` with `LAST`; record the negative net harvest for Nadine | Rémy, on Claude's recommendation |
| 10-04 | Fix cumulative rewards at the source (`FlowTrajectory` for `MDPState.rewards`) | Rémy, on Claude's recommendation |
| 10-04 | Whether `RayOptimizer.stop` returns metrics is Nadine's choice; it keeps its intended return with the typo fixed | Rémy |
| 10-04 | `BilevelOptimizer.train` stops both levels at the end of the run | Rémy, on Claude's recommendation |
| 10-04 | Remove the two unused policy helpers of `RayOptimizer` | Rémy, on Claude's recommendation |
| 10-04 | `build_optimizer` of the society requires a World and declared agents | Rémy, on Claude's recommendation |
| 10-04 | Investigate the non-reproducibility now and come back with a fix proposal before changing code | Rémy, on Claude's recommendation |
| 10-04 | `debug.py` parses the options its docstring documents, with today's values as defaults | Rémy, on Claude's recommendation |

## Waiting on

Rémy: how to fix the two multi-seed bugs (the publish loop and the evaluation env count).
Decided and queued: the reproducibility investigation, and `debug.py` gets the options it
documents.

## Findings for Nadine (to go into the phase 4 notes)

**Fitness tail window (left unchanged on Rémy's decision).** Traced by a subagent and verified by
Claude on 10-04. `FisheryRegulatorEnv.reward` averages the last `fitness_tail_steps` (50)
entries of per-episode arrays to measure the steady state, but since commits `654dc20`,
`04e9e7f` and `02eef11` (13–20 August) those arrays hold one value per episode: the
episode callback stores `env.logger.reduce()`, i.e. the mean over every step of the
episode, under `reduce="item"`. In evaluation each array therefore has length 1 and the
tail window does nothing; in training it slices across episodes. The ES objective
(`harvest_score + sustainability_weight * mean_fish`) is thus computed on whole-episode
means, not on the last 50 steps. Separately, nothing pushes `reward_mean`, although the
docstring of `core/envs/marl_regulated.py` says rewards are logged each step, so
`mean_reward` was NaN in every report (it does not enter the objective; fixed in
`733be6c`). `H_realized = B_{t-1} - fish_t` is negative whenever restoration adds fish,
so the harvest score is a net harvest: a modelling question, left unchanged.

**Cumulative rewards (fixed in `27fbd29`).** Every run since `32ef1b5` (09-30) trained
the fishermen on running sums of their harvests; results from that period are not
comparable with runs after the fix.

**`RayOptimizer.stop` return value (her choice).** The base `Optimizer.stop` returns
nothing, while `RayOptimizer.stop` returns the metrics reduced over the run. The typo
that made it crash was fixed without settling which contract is wanted.

**Policy loss cadence.** With APPO on the new API stack the learner reduces its
statistics once every 20 gradient updates, so the loss appears in the training log only
on those iterations. This is RLlib's behaviour, not a bug of the framework.

**The candidate mechanism is missing from almost every training episode (measured on
10-04).** `MultiAgentEnv.reset` fetches the mechanism from the World, but
`World.get_mechanism_by_id` returns it only on the first training fetch of a candidate
(it moves the entry from `published` to `train`, and `train` is not a valid predecessor
of `train`). The environment never stores what it fetched (`self.m` and `self.m_ctx` are
never assigned), and the RLlib adapter builds a fresh `MDPState` at every reset, so from
the second episode on the leaders hold no action and `AgentConfig.action` returns the
state unchanged: no quota is applied. A probe logging every reset of the shrunk config
showed the candidate quota in the first training episode of each environment and in
every evaluation episode (`eval` is a valid predecessor of `eval`), and no mechanism in
every later training episode. The fishermen therefore train almost entirely without
regulation and are then scored under the candidate, so the ES does not optimise the
bilevel objective it is meant to. The code dates from `08dbeb9` (08-16); Nadine's own
TODO at that spot warned that a missing mechanism after training starts "leads to silent
error". The comment above the fetch ("keep the current mechanism for subsequent
episodes") and the unused `m` and `m_ctx` fields suggest the intended design was to keep
the fetched mechanism. Fixed in `b330a82` on Rémy's decision: the environment still asks
the World at every reset, so a newly published candidate replaces the kept one, and
otherwise reuses the one it kept. Every result produced before this fix trained the
fishermen without regulation after their first episode.

**Multi-seed runs are broken in two places (measured on 10-04).** With two training seeds
the regulator publishes the candidates only for the last seed, because the
`append_context` call sits after the seed loop, so the first seed's fishermen never see a
candidate. The same run then crashes at evaluation: the evaluation runners inherit the
training runners' environment count, which `debugging` multiplied by the number of seeds,
so each one runs twice the episodes `evaluation_duration` expects. Both paths are unused
by the fishery configuration, which trains on a single seed.

**Runs are not reproducible (measured on 10-04).** Two runs of the shrunk fishery config
with the same seeds give identical training returns over the first two inner iterations
(19.3691 then 20.0886) but different evaluation fitness from the first generation
(1.4876 against 1.4803, then 1.4877, 1.4868 and 1.4886 in later runs), so the ES
trajectories diverge (best mechanism 0.668, 0.561, 0.495). The spread between runs is
of the order of 0.001 to 0.007, while the spread between candidates of one generation is
about 0.03. Weight fingerprints taken inside the evaluation function show that the
evaluated policies are the trained ones (learner, training runner and evaluation runners
agree in the second generation), so the fitness is not computed on untrained policies.
Tracing the circular buffer and every learner update then showed that the sampled data
and the weights after each update are identical across runs, so the update itself is
deterministic, but APPO's learner thread finishes its updates while evaluation is
starting: RLlib copies the learner's weights to the evaluation runners at a moment that
depends on timing. A prototype that waits for the thread to process every queued batch
before evaluating made the evaluated weights identical in two of three runs, yet the
fitness still differed slightly (1.4781 against 1.4779), and the third run had different
weights, so at least one more source remains. Evaluation does not explore
(`explore: false`). APPO's `CircularBuffer` draws batches with an unseeded
`np.random.default_rng()`, which matters only when more than one batch is waiting.
Separately, `PolicyActor.reset` builds a new `Algorithm` each generation without
stopping the previous one, whose learner thread keeps polling its empty buffer every
0.1 ms; the cost over many generations is not measured yet.

The other findings for Nadine are to be collected in the notes written in phase 4. Already
known: `ruff` warns that `isort.split-on-trailing-comma` conflicts with
`format.skip-magic-trailing-comma = true` in her configuration (no oscillation was
observed on this tree); `.gitignore` ignores `*.ipynb` although the tutorials are
tracked.

## Next step

Finish phase 1: Rémy's decision on the two multi-seed bugs, then the reproducibility
investigation (separate the remaining source, then propose fixes before changing code),
then `debug.py` (its documented options, on Rémy's decision). Then the port of the three mechanisms in its own
session. Still open from the reading of the code, for phase 2 or 3:
`RayOptimizerConfig.build_optimizer` calls `self._reporter_cfg.build` without the `None`
check its docstring describes, and `RayOptimizer.train` returns `self.logger.peek()`
although it is annotated `-> None`.

**Suite conseillée :** modèle opus, effort high — phases 0 to 2 are orchestration and
test porting against an API delta that is already mapped. The port of the three mechanisms
is the one part of phase 1 to run as its own session on fable, since it touches the
scientific core; do the base and the two crashes first.

## Probable bugs found while reading (not fixed yet)

The lint agents listed these while reading the code; none was changed. They feed phases
1 and 2, where each gets a failing test before its fix.

The ES optimizer's `train` returns the key `"episode"` while `BilevelOptimizer.train`
reads `result["episodes"]` and `result["converged"]`; this is the known crash at
`core/optimizers/bilevel.py:252`. In `ESOptimizer.__init__` the bounds check
`not np.allclose(low, 0.0) and np.allclose(high, 1.0)` lacks parentheses, so a space
whose upper bound is not 1 passes silently. `ESOptimizer.train` leaves `generation`
undefined when `episodes` is 0, and `batch_capacity` reads `_batch_capacity` before any
assignment. `Optimizer.__init__` reads `config.episodes` before its own `None` guard.
`RayOptimizer` refers to `self.algo` and `self._inner_iter`, which are never defined,
calls `self.logger.reduce(complie=True)`, and returns values from methods annotated
`-> None`. `RayOptimizerConfig.build_optimizer` leaves `opt_id` and `agents` unbound when
there is no world or no agent configuration, and builds the reporter without checking
for `None`. Two error messages in `core/reporting/base.py` lack the `f` prefix and print
their placeholders literally. The cartpole and fresh-water examples import
`MultiAgentRegulatedEnv`, which no longer exists; the fresh-water `debug.py` hard-codes
paths under `/Users/nadine`. The docstring of `examples/bilevel_fishery/__init__.py`
names two modules that do not exist.

## Known traps

Run scripts as modules from the repository root. Set `WANDB_MODE=offline` for every run.
Clear `__pycache__` after switching branches. The main directory holds ignored leftovers of the earlier branch (`htmlcov`, caches); they are harmless. Ray-backed tests must run last (the earlier
`tests/conftest.py` has the ordering hook). `uv run jupyter` resolves to a system Jupyter
because the virtual environment has none; use `uv run --with nbconvert --with ipykernel`
and an explicit kernel name. Three notebooks carry no kernel specification. Never run
`ruff check .` without `--no-fix` while `fix = true` is in `ruff.toml`.

## Resume commands

```bash
# from the main directory, on feature/social-influence-testing-v2
uv sync
uv run ruff check . --no-fix
uv run ruff format --check .
WANDB_MODE=offline uv run python -m pytest
WANDB_MODE=offline uv run python -m core.config.cli check examples/bilevel_fishery/config.yaml
WANDB_MODE=offline uv run python -m core.config.cli run examples/bilevel_fishery/config.yaml
```
