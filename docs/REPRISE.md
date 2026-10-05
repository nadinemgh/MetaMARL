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
- [x] Phase 1 — every entry point of the fishery example runs end to end (complete on 10-04):
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
  - [x] **with more than one training seed, only the last seed's environments received the candidate**: `RegulatorEnv.step` built one context per seed but called `append_context` after the seed loop; the call is now inside it, on Rémy's decision (`dea1919`);
  - [x] **with more than one training seed, evaluation crashed** ("fewer units than requested: requested=24, completed=48"): the evaluation runners inherited the training runners' environment count, which `debugging` multiplies by the number of seeds. `build_optimizer` now gives each evaluation runner one environment per mechanism, and the message says "fewer" or "more" as the case is, on Rémy's decision (`03330ef`). Measured with two seeds on the shrunk config: the run completes and all 80 sampled training and evaluation resets carry the candidate for both seeds;
  - [x] **runs are not reproducible**: investigation finished on 10-04; three sources found and measured (see "Findings for Nadine"), each removed by a scratch prototype, with bit-identical fitness over repeated runs once all are removed;
  - [x] **reproducibility fixed** on Rémy's decision, one commit per source, each with its test: the environment's metric logger is reset at every episode (`439249a`); `disable_env_checking` reaches RLlib (`be8d83e`); `core.config.cli` and `debug.py` restart themselves with `PYTHONHASHSEED=0` when the variable is unset (`2dd94c9`); evaluation waits for APPO's learner thread to apply every queued update (`ac83a64`). Measured from a shell without `PYTHONHASHSEED`, so every run went through the restart: five runs of the shrunk config with 10 inner iterations and three full-size runs (10 fishermen, horizon 100, 2 generations) each gave bit-identical fitness vectors, equal to those of the scratch prototypes. The test suite has 59 tests, all passing;
  - [x] `debug.py` now parses `--outer-iters`, `--train-iters`, `--num-agents`, `--horizon` and `--reporter` (`wandb` or `csv`), defaults being the former hard-coded values; measured: the smoke configuration of its docstring with `--reporter csv` exits 0. **Correction of 10-04 (phase 2):** this line used to say that the run "writes the CSV files under `results/`". That was wrong: the run creates the directories under `results/` and no file, because the optimizers never received their reporter. The reporting fix that closed phase 2 (`7e6bd39` to `0477d20`) corrected that, and the run now writes 16 CSV files (see "Reports were never written" in the findings);
  - [x] **observations were cumulative**: the observation a fisherman received at step t was the sum of every observation since the reset (measured: first entry 0.766, 1.511, 2.261, then 14.569 at step 20, while the normalised stock went from 0.766 to 0.683). `MDPState.obs` is now a `FlowTrajectory`, like the rewards, on Rémy's decision (`0bb367c`, 3 tests). Re-measured on a shrunk run: the 2280 observations handed to RLlib equal the state they encode;
  - [x] `Subsidy` ported to `Mechanism.apply` with 12 tests (`d317b9f`): the regulator's action is the subsidy rate normalised by `MAX_SUBSIDY`, and the residual `rate * e - cost * e**2` is added to each targeted fisherman's reward;
  - [x] `ThresholdPenalty` ported with 12 tests (`20f878e`): a fixed rule with an empty action space, which subtracts the logistic penalty from every targeted fisherman's reward;
  - [x] measured on a shrunk fishery whose regulator holds quota, subsidy and penalty: the ES searches two dimensions, the empty penalty action goes through the ES and the candidate hand-off, and over 4256 agent-steps the reward equals harvest plus subsidy residual plus penalty residual to 1.1e-16;
  - [x] a mechanism can now contribute to the observations: `Mechanism.observe` returns nothing by default, and the environment adds the leaders' contributions where it already adds the followers' observations, at reset and at the end of each step, on Rémy's decision (`5c2c52f`, 5 tests);
  - [x] `SocialInfluence` ported with 14 tests (`dbe2338`): it implements `observe` and writes each peer's delivered action of the step just finished into reserved entries of the observation. Measured on a shrunk fishery whose regulator holds all five mechanisms: over 4160 observations, entries 3 and 4 of each fisherman equal its peer's last delivered harvest and restoration exactly, and they stay zero while no candidate is published;
  - [x] end-of-phase check on `dbe2338`: `ruff check --no-fix` and `ruff format --check` pass on the whole tree, the suite has 105 tests, all passing, `core.config.cli check` accepts the fishery configuration, two runs of the smoke configuration of `debug.py` give bit-identical fitness vectors, and the three tutorials that executed at baseline still execute.
- [x] Phase 2 — tests ported from `feature/integration-trial` and written for the new code; coverage target above 90 % on `core/` (complete on 10-04, at 99 %):
  - [x] honest baseline: most directories of `core/` are namespace packages, so coverage silently left out every file that no test imported and reported 63 % on 52 of 59 files. With `include_namespace_packages = true` in `pyproject.toml` (`3db1e46`) the true starting point was 57 % (3691 statements, 1597 missed) for the 105 tests of phase 1;
  - [x] triage of the August suite against the tree after phase 1: of 54 files, 11 pass unchanged, 2 are skipped entirely, 1 runs no test and 40 fail in whole or in part, 17 of them at import;
  - [x] the work was split by layer between eight subagents sharing one brief (test files only, no change under `core/` or `examples/`, a defect is demonstrated by a strict `xfail` test and never fixed or hidden). Claude re-ran every layer, re-measured its coverage, linted it and read the code behind every reported defect before committing;
  - [x] metrics: 174 tests, `core/metrics` at 100 % (`7e4b2de`);
  - [x] reporting: 157 tests, `core/reporting` at 100 % (`70c5244`);
  - [x] Ray adaptor: 252 tests and 4 recorded defects, `core/adaptors/ray` at 100 % (`a8db5a4`, `9c22690`);
  - [x] World, callbacks, utilities and annotations: 157 tests and 4 recorded defects, all at 100 % (`a89fb77`);
  - [x] environments, agents, YAML loader, command line and fishery ecology: 194 tests and 9 recorded defects; `marl_regulated.py` at 99 % and `regulator.py` at 95 %, the missing lines being unreachable until two of those defects are fixed (`9c74d85`);
  - [x] optimizers: 253 tests and 4 recorded defects; `es/optimizer.py` at 97 %, the missing lines being unreachable or behind one of those defects (`1cdf19f`);
  - [x] mechanisms: 154 tests and 3 recorded defects, `core/mechanism` at 100 %, quota included (36 % before) (`7dccabd`);
  - [x] the two composition checks of phase 1 are now repeatable tests in `tests/integration/test_fishery_regulator_composition.py`: they step the real shrunk fishery with the real mechanisms behind a scripted World, without Ray, in about one second. The debug script is run as a child process with the CSV reporter and a 180 s limit in `test_debug_script_csv_smoke.py`;
  - [x] end-of-phase measurement on `7dccabd` with the default command (`WANDB_MODE=offline uv run python -m pytest`): 1341 passed, 24 xfailed, no failure, 27 s; `core/` at 99 % (3691 statements, 13 missed, all in the three files named above); `ruff check . --no-fix` and `ruff format --check .` pass on the whole tree;
  - [x] the 24 strict `xfail` tests recorded 14 distinct defects. On Rémy's decision the three that stopped reports from being written were fixed, together with the problems that the fix uncovered, in seven commits, each with its test: every query is rendered on its own and a failing query is logged instead of aborting the optimizer (`7e6bd39`); the ES query on the quota names the parameter the regulator searches and the subsidy query is removed (`34ae6f6`); the learner batch size is declared as a float (`92d93d9`); learner statistics that RLlib did not report on an iteration are logged as NaN, so every series keeps one value per iteration (`6e512e9`); both `build_optimizer` methods hand the reporter under the keyword `reporting` (`19125d3`); a configuration without reporter runs without reporting (`d3b1dc5`); a society configuration without training seeds is rejected with an error naming the missing call (`0477d20`). Seven `xfail` markers were removed with the code they described;
  - [x] reporting fix measured on 10-04: the smoke configuration of `debug.py` with `--reporter csv` exits 0 and writes 16 CSV files (7 for the ES, 9 for the inner optimizer), no query fails, and its fitness vectors are bit-identical to those of the run before the fix, which wrote none. A run of 23 inner iterations also exits 0 with 16 files; its policy-loss series is NaN on iterations 0 to 19, holds -0.904 at iteration 20 (the 21st, where APPO reports) and is NaN again on 21 and 22. The 11 `Mean of empty slice` warnings of that run all come from RLlib's EMA statistics (`ray/rllib/utils/metrics/stats/ema.py:124`), as in phase 1, and the run before the fix shows the same count on the smoke configuration;
  - [x] end-of-phase measurement on `0477d20`: 1363 passed, 17 xfailed, no failure, 28 s; `ruff check --no-fix .` and `ruff format --check .` pass on the whole tree. The 17 remaining strict `xfail` tests record 11 defects, left unchanged on Rémy's decision and handed to Nadine.
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
| 10-04 | Apply all four reproducibility fixes (logger scoped to the episode, `disable_env_checking` forwarded, fixed string-hash seed, wait for the learner thread before evaluation), one commit each with a test, then verify by repeated runs | Rémy, on Claude's recommendation |
| 10-04 | The string-hash seed is the constant 0 when `PYTHONHASHSEED` is unset; an explicit value is respected | Rémy, on Claude's recommendation |
| 10-04 | Fix cumulative observations at the source (`FlowTrajectory` for `MDPState.obs`), as was done for the rewards | Rémy ("go" to the brief that recommended it), on Claude's recommendation |
| 10-04 | Add an optional `observe` method to `Mechanism`, called by the environment for the leaders' mechanisms, and port `SocialInfluence` onto it | Rémy ("go" to the brief that recommended it), on Claude's recommendation |
| 10-04 | Rémy's "go" approved both the observation fix and the `observe` method (confirmed explicitly) | Rémy |
| 10-04 | Quota `allowed_frac`: leave the code unchanged and record it for Nadine | Rémy, on Claude's recommendation |
| 10-04 | Fix the lost reporter now: keyword at both `build_optimizer` call sites, stale ES queries, per-query error isolation, `None` reporter check, empty seed list error; one commit each with its test | Rémy, on Claude's recommendation |
| 10-04 | Learner statistics missing on an iteration are logged as NaN, so every series keeps one value per iteration | Rémy, on Claude's recommendation |
| 10-04 | The other recorded defects are left unchanged and handed to Nadine in the phase 4 notes, each with its test (eleven defects after the reporting fix, which also covered the `None` reporter and the empty seed list) | Rémy, on Claude's recommendation |
| 10-04 | Remove the ES query "Restoration subsidy vs fixed quota", whose parameter the fishery regulator does not search, and record it for Nadine | Rémy, on Claude's recommendation |
| 10-04 | Declare the learner batch size as a float, since RLlib reports a mean | Rémy, on Claude's recommendation |

## Waiting on

Nothing. Phase 2 is closed and phase 3 depends on no decision.

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
`733be6c`). The earlier note here said that `H_realized` is a net harvest that turns
negative when restoration adds fish. Phase 2 measured more than that: the logged value is
not the harvest at all (see "The logged harvest is not the harvest" below).

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
by the fishery configuration, which trains on a single seed. Both fixed on Rémy's decision
(`dea1919`, `03330ef`).

**Runs are not reproducible (measured on 10-04).** Two runs of the shrunk fishery config
with the same seeds give different evaluation fitness from the first generation, by
0.001 to 0.007, while the spread between candidates of one generation is about 0.03, so
the ES trajectories diverge. The investigation ran on scratch copies of the config (2
generations, 2 fishermen, horizon 20, then 10 inner iterations, then the full size with 10
fishermen and horizon 100) with a probe environment that logged the random generator
state, the mechanism, the actions and the stock at every reset and step. It found three
independent sources.

The largest is a leak of RLlib's environment check into the fitness. When RLlib builds
an env runner it calls `check_multiagent_environments`, which resets every
sub-environment and steps it once with `action_space.sample()`, an unseeded random action.
The step pushes its stock, harvest and reward into the environment's `MetricLogger`, and
`MultiAgentEnv.reset` then flushes only the `iter` series, so the first real episode of
every evaluation environment is reduced together with that random step. The probe showed
the leftover value in the logger at the real reset, and showed that the real evaluation
episodes themselves are bit-identical across runs. This check runs even with
`disable_env_checking: true`, because `OptimizerConfig.environment` stores that flag and
nothing forwards it to RLlib. Even with no training at all (zero inner iterations), three
runs gave three different fitness vectors; clearing the logger at reset made them
bit-identical.

The second source is Python's randomised string hashing. RLlib's env-to-module connector
iterates `MultiAgentEpisode.get_agents_that_stepped()`, which returns a `set` of agent ids,
so the order of the fishermen in the inference batch depends on the per-process hash seed.
The exploration noise is seeded and identical across runs, but it is handed to the
fishermen in a different order: the probe showed the two fishermen's training actions
exactly swapped between runs. Training returns then differ in the fourth decimal and the
evaluated policies differ. With the logger fix alone, 10 inner iterations gave two
distinct outcomes over five runs; with `PYTHONHASHSEED=0` in addition, five runs were
bit-identical, and three full-size runs too. Because Ray runs in `local_mode`, everything
executes in the driver process, so the hash seed must be set before the interpreter
starts.

The third is a race between APPO's learner thread and evaluation. `Algorithm.train`
returns while the learner thread is still applying the last update, and evaluation copies
the learner's weights immediately. A counter around the thread showed 9 of 10 updates done
at every evaluation, in the shrunk and the full-size config, and waiting for the thread
took 13 ms and 31 ms respectively. In these measurements the race always resolved the same
way, so it did not break reproducibility, but the evaluated policy is not the trained one:
at full size, waiting moved the best fitness from 1.4788 to 1.4651 and the ES mean after
the first generation from 0.612 to 0.596. A slower update or a loaded machine could flip
the outcome. Seeding APPO's `CircularBuffer` (`np.random.default_rng()` without a seed)
changed nothing, because no more than one batch was ever waiting.

The earlier reading of this section, that the learner thread's timing explained the
divergence, was wrong: the swapped agent order and the leaked check step produced the
differences attributed to it. Separately, `PolicyActor.reset` builds a new `Algorithm`
each generation without stopping the previous one, whose learner thread keeps polling its
empty buffer every 0.1 ms; the cost over many generations is not measured yet.

All three sources were fixed on Rémy's decision. The environment's logger is now reset
at every episode, and `disable_env_checking` is forwarded to RLlib, so the fishery config
can also skip the check entirely. The command line and `debug.py` restart themselves with
`PYTHONHASHSEED=0` when the variable is unset and log the value in use; an explicit value
is respected, and `random` is logged as a warning. `PolicyActor.evaluate` waits until the
learner thread has itself found its input buffer empty, which it only does once its
previous update is complete. That wait reads private RLlib attributes and raises rather
than skipping when they are missing; it also raises for GPU learners and for IMPALA's
deque queue, which it does not cover. Scores produced before these fixes are not
comparable with later ones: the leaked check step changed every fitness, and at full size
the wait alone moved the best fitness of the first generation from 1.4788 to 1.4651.

**In the shrunk config the fitness ignores the candidate (measured on 10-04).** With zero
inner iterations and the logger fix, the four fitness values of the second generation are
bit-identical to those of the first, although the candidates differ (ES mean 0.5, then
0.60). The quota never binds there, and each population slot keeps its rank across
generations, because each slot trains and evaluates its own policy module
(`fisher_policy_m<idx>_s<seed>`), whose initial weights differ. The ES gradient in that
config therefore follows the differences between slot initialisations, not the
mechanism. Whether the full-size config shows the same pattern is not measured yet.

**Observations were cumulative (measured on 10-04, fixed in `0bb367c`).** The fishermen
did not observe the current stock but the sum of every observation since the reset. A scratch
probe wrapped `RLlibMultiAgentEnvAdapter.reset` and `step` during a shrunk run of
`debug.py` (1 generation, 1 inner iteration, 2 fishermen, horizon 20) and logged the
observation handed to RLlib next to the state it should encode. In one training episode
the normalised stock went from 0.766 to 0.683 while the first entry of the observation
went 0.766, 1.511, 2.261 and reached 14.569 at step 20; the usage entry accumulates in
the same way. At the full horizon of 100 the first entry would be near 70. The cause is
the one behind the cumulative rewards: `MultiAgentEnv.step` adds each follower's
observation with `MDPState.add`, and `obs` is a carry-forward `Trajectory`, so the new
observation is added on top of the previous one. The code dates from `32ef1b5` (09-30).
The declared observation space is unbounded, so nothing failed. Every run between that
commit and the fix trained the fishermen on these running sums, so results from that
period are not comparable with later ones. The fix makes `obs` a `FlowTrajectory`, as
`27fbd29` did for the rewards; the same probe then showed 2280 observations equal to the
state they encode.

**The quota's `allowed_frac` state entry accumulates (measured on 10-04, left unchanged on Rémy's decision).**
`QuotaMechanism.apply` returns `state={"allowed_frac": allowed_frac}`. State entries are
stocks, so the returned value is added to the previous one instead of replacing it: the
probe read 1.0, 1.9999, 2.9999 and 3.9998 over four steps. Nothing in `core/` or in the
fishery example reads that entry, so no result is affected today, but any code that reads
it later will get a running sum. Whether the entry should be a per-step value, a
replacement, or should not exist is her choice.

**A decoded action is decoded again at every step (measured on 10-04, not fixed).**
`Mechanism.__call__` writes the decoded action back into `mdp.actions`, and the leader's
action is set once at reset and carried forward, so at the next step `decode` receives
its own previous output. A `decode` that is not idempotent therefore compounds: a first
version of the subsidy that scaled the rate by 0.5 inside `decode` gave 0.15, 0.075,
0.0375 and 0.01875 over four steps. `Quota` and the ported `Subsidy` only clip in
`decode`, which is idempotent, and `Subsidy` applies its scaling in `apply`; its
docstring says why. The interface does not state this constraint anywhere else, and the
next mechanism written against it can fall into the same trap.

**The action temperature is hard-coded in five places.** A follower's raw policy output
`z` becomes an effort through `sigmoid(z / 4.0)`. The regulator acts before the
followers, so a regulator mechanism that reads a follower's action must apply the same
map itself. The constant 4.0 is now written in `QuotaMechanism.apply`,
`Agent._normalize_action`, `Fishing`, `Restore` and `ACTION_TEMPERATURE` of
`subsidy.py`. Changing it in one place silently breaks the others.

**Differences between the ported mechanisms and their August versions.** The subsidy
reads the effort the fisherman requested, not one modified by another mechanism of the
same regulator, because all mechanisms of one agent read the same input state. The
threshold penalty uses only the agent part of `acts_on`. `SocialInfluence` differs most.
The new interface composes by addition, so it cannot append entries to an observation;
the port fills entries the benchmark reserves, starting at `obs_offset`, and raises a
`ValueError` naming the agent when they do not fit. One instance exposes the action of
one mechanism, so the fishery needs two instances to expose harvest and restoration.
Peers are ordered by their identifier sorted as a string, so `fisherman:10` comes before
`fisherman:2`. The value exposed is the delivered action, after the quota, not the
request. In the fishery the `Fisherman` observation has five entries, of which indices
1, 3 and 4 are left at zero. One instance fills a contiguous range, so the two entries
at indices 3 and 4 carry both actions of one peer (two fishermen) or one action of two
peers (three fishermen); beyond that the benchmark's observation must be enlarged. The
`influence_weight` parameter is kept and validated but has no effect: the causal
influence reward of Jaques et al. (2019, PMLR 97) is not implemented.

**Her own acceptance lists for these mechanisms are in `TODO.md`.** Sections 4 and 5 of
the notes she committed on 08-21 ("fix `SubsidyMechanism`", "finish
`SocialInfluenceMechanism`") give the intended subsidy formula and a list of checks. The
port keeps that formula and its tests cover each check: the analytical value, the
selection of the targeted action, the float residual, `ValueError` on the bounds, the
peer ordering, the exclusion of the agent's own action and the size of the observation.
One item stays open and is hers to decide: implement the KL influence term or rename
the class, since it only shapes the observation. Those sections describe the earlier
constructor arguments and were not edited.

**The `mechanism_algorithms` tutorial shows the earlier constructors.** It still
executes, but its worked examples of the three mechanisms are the fenced code blocks of
phase 0 and build `SubsidyMechanism`, `SocialInfluenceMechanism` and
`ThresholdPenaltyMechanism` directly with the earlier arguments. Phase 5 has to rewrite
them against the ported configuration classes. The last section of `TODO.md` also
carries two comments moved from the earlier `subsidy.py`, whose line numbers no longer
match the file.

**Reports were never written (measured on 10-04 in phase 2, fixed the same day on
Rémy's decision).** `OptimizerConfig.build_optimizer` and
`RayOptimizerConfig.build_optimizer` handed the reporter to the optimizer as
`reporter=...`, while `Optimizer.__init__` names that argument `reporting`. The keyword
fell into `**kwargs`, `self.reporting` stayed `None`, and `report_metrics` returned at
once. Only `BilevelOptimizer` kept its reporter. A run of `debug.py --reporter csv`
therefore exited 0, created the directories under `results/` and wrote no file; the same
held for W&B and TensorBoard, since the loss happened before the reporter was chosen.
Both call sites now pass `reporting=` (`19125d3`).

Fixing the keyword alone would have turned a run that wrote nothing into a run that
crashed, because `Reporter.report` resolved the queries one after the other and the
first error aborted the optimizer that called it. `Reporter.report` now renders each
query on its own and logs the traceback of a query that fails, naming it, while the
others are still rendered (`7e6bd39`). Three further problems sat behind the keyword.

The first was two stale ES queries, which asked for the parameters `fixed_quota` and
`restoration_subsidy` while the ES logs the regulator's parameters under their mechanism
ids. "Candidate fitness vs fixed quota" now reads the parameter `quota`; its title and
its axis label still say "fixed quota", which Nadine may want to reword. "Restoration
subsidy vs fixed quota" was removed on Rémy's decision, because the fishery regulator
searches no subsidy parameter; Nadine should restore it if she adds a subsidy to the
regulator (`34ae6f6`). A test builds the ES from the fishery YAML and resolves every ES
query, so a renamed mechanism now fails in the suite instead of at the end of a run.

The second was the four learner queries of the inner optimizer. APPO reports its learner
statistics only once every 20 gradient updates, and an iteration without statistics
added nothing to the series, so a run of 22 iterations stopped with "Query series must
have equal length". On Rémy's decision, every module declared in the RLlib
configuration now gets a NaN for each statistic that RLlib did not report on an
iteration, so each series keeps one value per iteration (`6e512e9`). The batch size of
those statistics is a mean, which RLlib reports as a fraction, and is now declared as a
float instead of an integer (`92d93d9`).

The third is four queries that can never produce anything, left unchanged. They ask for
`quota_penalty` and `intrinsic_utility`, which the metric schema declares and which no
code in `core/` or in the fishery example writes any more. They resolve to zero values
and the reporter skips them without a word.

Two defects recorded by the phase 2 tests were fixed with the keyword, because the
keyword fix made them reachable from every entry point. A configuration without a
reporter, documented as "no reporting", made the society optimizer and both environments
call `build` on `None`; they now build no reporter, and the episode callback reports only
when the environment has one (`d3b1dc5`). A society configuration with an empty list of
training seeds reached a division by zero; `build_optimizer` now rejects it with an
error that names the `.debugging(seed=...)` call (`0477d20`).

**The logged harvest is not the harvest (measured on 10-04 in phase 2, not changed).**
A scratch probe stepped the real `FisheryRegulatedEnv` without noise or restoration, two
fishermen each asking for half of their capacity from a stock of 800, and recomputed
every quantity by hand. Four things came out, all in the `pella_tomlinson` transition of
`examples/bilevel_fishery/regulated_env.py`.

First, the value logged as `H_realized` is the true harvest minus the growth of the
previous step. From step 1 on the transition recovers the harvest as the difference
between the stock of the previous step and the current one, and that difference already
contains the previous growth. The probe read 0.0 at step 0 for a true harvest of 115.5,
then 43.4 for 108.2, 37.1 for 101.9, and 17.3 for 90.6 at step 5: the logged value falls
toward zero as the stock settles, whatever the fishermen catch. `FisheryRegulatorEnv`
builds the harvest term of the ES objective from this value (`realized_harvest /
msy`), so that term measures the net decline of the stock, not the catch. This
consequence is read from the code; it has not been measured on a full run.

Second, the growth term is computed on the stock of the previous step, not on the
current one. At step 1 the code gives a stock of 705.897 where growth on the current
post-harvest stock would give 710.136.

Third, the code harvests and then grows, while the module docstring states the equation
with growth on the stock before the harvest. From 800 the first step gives 749.288 in
the code and 728.000 with the documented equation, and the gap persists (610.9 against
576.8 at step 5).

Fourth, the fishermen harvest one after the other, each from what the previous one left,
so two identical requests of 60 deliver 115.5 in total and the order of the agents
matters.

`tests/examples/test_fishery_env_dynamics.py` pins only quantities that do not depend on
these choices and lists the three order questions in its docstring. Rémy's decision of
10-04 not to change the ES objective stands; these are modelling questions for Nadine.

**Defects recorded by the phase 2 tests (left unchanged on Rémy's decision).** Phase 2
recorded 14 defects in 24 strict `xfail` tests. Three of them (the reporter keyword, the
`None` reporter and the empty seed list) were fixed with the reporting fix described
above, and their seven markers were removed. Each of the 11 defects below is
demonstrated by one or more of the 17 remaining strict `xfail` tests, whose reason
string names the defect; `uv run python -m pytest -rx` lists them. Claude read the code
behind each one.

- The configuration snapshot handed to the Ray optimizer is not frozen, because
  `RayOptimizerConfig.freeze` is a deferred RLlib mutator and `copy(copy_frozen=True)`
  never applies it (1 test).
- `Mechanism.__init__` keeps its default with `default if default else None`
  (`core/mechanism/base.py:151`): an array of several entries raises "truth value of an
  array is ambiguous" and a default equal to zero is silently replaced by `None`
  (3 tests).
- `MultiAgentEnv` cannot be built without leaders although `leaders_cfg_dict` is
  optional (1 test), and cannot reset without a metric schema although the docstring
  says `schema=None` disables logging (1 test).
- The `opt_id` getter of `MultiAgentEnv` and of `RegulatorEnv` returns `self.opt_id` and
  so calls itself until the recursion limit (2 tests).
- When the fetch of a mechanism fails, `MultiAgentEnv.reset` calls `self._debug_remote`,
  which no class defines, so an `AttributeError` hides the error meant to name the
  mechanism (1 test).
- `RegulatorEnv.reset(seed=...)` reads `self.seed`, which is never assigned (1 test).
- `OptimizerConfig.training` is annotated `-> Self` and returns `None`, which breaks
  chained calls (1 test).
- `OptimizerConfig.build_optimizer` reads `opt.id` before checking for a World, so
  `world=None` raises although the signature allows it (1 test).
- The ES "fixed" mode, a regulator with no parameter to optimise, crashes when it logs:
  `_to_logger_payload` reads `env.m_space`, which no environment defines
  (`core/optimizers/es/optimizer.py:618`) (1 test).
- In `World`, `set_new_context` and `update_context` store the context before rejecting
  one that lacks `env_id`, so a failed call still mutates the registries (2 tests). The
  cursors of `get_new_env_step_contexts` count positions in lists that `flush_ctx` and
  `remove_context` shrink, so contexts appended afterwards are skipped (2 tests). No
  live code calls these four methods today.

**Points the tests do not settle (open questions for Nadine).** The subagents listed
these and no test asserts either way, because the intended behaviour is hers to state.
The ES convergence test `_has_converged` always answers "not converged": the two
convergence settings are stored and never read, although the docstring says they stop
the loop. `RegulatorEnv.step` ends the episode one step early for a horizon of two or
more. The `Max` and `Min` metrics depend on the order of their inputs when one is NaN,
and the numeric metrics reject `np.float32` and `np.int64` values. `WandbConfig`
swallows unknown keyword arguments. Three `or` chains in `core/adaptors/ray/utils.py`
(lines 63, 242 and 411) treat a value of 0.0 as missing. Calling `debugging` twice
compounds its scaling, and it must be called after `env_runners`. `PolicyActor.reset`
does not stop the previous algorithm. `RayOptimizer.train` returns `logger.peek()`
although it is annotated `-> None`, and `core/envs/regulator.py:164` uses that return
value; `_inner_iter` is set only by `reset`. When two hooks carry the same name the last
one wins silently. `core/agents/base.py` imports `ActType` from the unmaintained `gym`
package, which prints a deprecation notice at every start. `debug.py` parses its command
line at import time, so no test can import it with its own arguments. Several docstrings
describe removed behaviour; phase 3 rewrites them.

**What the August suite tested that no longer exists.** The mechanism tests of August
checked `dimension`, `encode`, `to_vector`, `param_names` and `clip` on every mechanism;
none of these is defined in `core/` any more, since a mechanism's parameters now live in
its action space. The two composition test files were dropped with the composition
classes they tested; composition is now covered through `Agent.action` in the quota
tests and in the integration test of the composed fishery.

The other findings for Nadine are to be collected in the notes written in phase 4. Already
known: `ruff` warns that `isort.split-on-trailing-comma` conflicts with
`format.skip-magic-trailing-comma = true` in her configuration (no oscillation was
observed on this tree); `.gitignore` ignores `*.ipynb` although the tutorials are
tracked.

## Next step

Phase 2 is closed. The suite has 1363 passing tests and 17 strict `xfail` tests, `core/`
is covered at 99 % as measured at the end of the test work, and the whole tree passes the
lint and format checks. The phase holds seventeen commits (`3db1e46` to `7dccabd` for the
tests, `d412fd0` for this file, `7e6bd39` to `0477d20` for the reporting fix), all
local; the branch has never been pushed. The close criterion is met: `debug.py
--reporter csv` writes its 16 CSV files on the smoke configuration and on a run of 23
training iterations.

Phase 3 comes next and waits on nothing: a docstring and type hints on every public
symbol of `core/` and `examples/bilevel_fishery`. The baseline measured 71 % of public
symbols documented (304 of 429) and 92 % of public callables fully annotated, with the
gap in `core/agents`, `core/mechanism`, `core/envs` and the CSV and TensorBoard
reporters; the first action is to measure both figures again on the present tree, since
phases 0 to 2 changed them. The docstrings follow the NumPy style with shapes, units, a
"When to use" note, a runnable example and the source papers. Each one must describe what
the code does today, which the phase 2 tests now pin: where a docstring promises
behaviour that an `xfail` test shows missing (a `None` schema, convergence settings that
stop the loop), the docstring states the present behaviour and the defect stays in the
notes for Nadine. The work splits by layer exactly as phase 2 did, so the writing goes to
subagents with one shared brief and the main session verifies each layer by running the
suite, the lint and a docstring coverage count.

The scratch files of phase 2 (the shared brief, the triage logs, the reporting and
dynamics probes, the run logs of the reporting fix and the worktree at `d412fd0` used
for the before-and-after comparison) lived in a session directory and are not kept.
Everything they measured is recorded above.

**Suite conseillée :** modèle opus, effort high — phase 3 is careful reading and writing against behaviour that the tests now pin, spread over subagents; it needs accuracy more than the bug-hunting depth of phase 2. Start it in a fresh session after `/clear`.

## Probable bugs found while reading (not fixed yet)

The lint agents listed these while reading the code during phase 0. Status after phase
2: the ES return keys, the bounds check, the undefined `generation`, the capacity read
before assignment, the `None` guard of `Optimizer.__init__`, the misspelled `complie`
keyword, the unbound `opt_id` and `agents`, and the two missing `f` prefixes were fixed
in phase 1. The reporter built without a `None` check was fixed at the end of phase
2 (`d3b1dc5`), and the values returned from methods annotated `-> None` are listed in the open
questions for Nadine. The two unported examples and the stale docstring of
`examples/bilevel_fishery/__init__.py` are unchanged. The paragraph below is kept as it
was written.

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

The test directories have no `__init__.py`, so every test file needs a basename that is
unique in the whole of `tests/`; two files of the same name in different directories
fail at collection. A strict `xfail` test fails the suite the day its defect is fixed:
remove the marker in the commit that fixes the code. When several processes measure
coverage at the same time, give each its own `COVERAGE_FILE`. The smoke test of the
debug script starts a child process that runs Ray for about 15 s. In zsh an unquoted
variable holding several paths is not split into words; use an array.

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

The entry points restart once with `PYTHONHASHSEED=0` when the variable is unset; the
first two log lines say so. Bit-identical repeat runs are the check for reproducibility:
compare the `fitness=[...]` part of the `[ES] BEFORE UPDATE` lines.
