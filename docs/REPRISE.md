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
  - [x] fitness tail window: Rémy decided on 10-04 not to change the objective and to record the finding for Nadine. **Superseded the same day:** Rémy then decided to make the tail window real, which the fix pass of phase 3 did (`3205b73`);
  - [x] `fish_norm_next_last` reduced with `MAX` instead of `LAST` (`bc5a02d`);
  - [x] **rewards were cumulative**: `MDPState.rewards` used the carry-forward `Trajectory`, so the reward RLlib received at step t was the sum of all rewards since the start of the episode (measured: 0.89463 at t=2 for harvests 0.45785 and 0.43679, 9.47 at t=20; behaviour introduced by `32ef1b5` on 09-30). Fixed with a `FlowTrajectory` whose unwritten steps start at 0 while same-step deltas still sum, on Rémy's decision (`27fbd29`). Re-measured: reward equals harvest at every step; episode return of the shrunk config 205.9 → 19.4; the three tutorials that executed at baseline still execute;
  - [x] the step reward is logged again, as the env docstring states, into the five reward fields (`733be6c`); `mean_reward` is no longer NaN;
  - [x] `policy_loss=NA` on every training line: the reader only knew the classic `info/learner` layout; it now reads `learners/<module>/policy_loss` (`3ec65bc`). Measured on 45 inner iterations: APPO's learner reports its stats only once every 20 gradient updates (`IMPALALearner.update` in Ray 2.53), so real losses appear at iterations 21 and 42 and the other lines correctly print NA;
  - [x] the `Mean of empty slice` warnings were traced to RLlib's own EMA timers (`env_reset_timer` and connector timers of the env runners) on iterations where they measured nothing; they do not touch learning and are left visible;
  - [x] `Optimizer.__init__` read `config.episodes` before its `None` guard, and an unset batch capacity raised a bare `AttributeError`; it now raises a `RuntimeError` naming the optimizer (`16d1c6d`);
  - [x] `RayOptimizer.stop` raised `TypeError` on a misspelled keyword and nothing called it; `BilevelOptimizer.train` now stops both levels in its `finally` block on Rémy's decision, and the stop is logged (`691e08e`); whether `stop` should return the reduced metrics is left to Nadine;
  - [x] the two unused helpers `_build_agent_policy_map` and `_get_policy_handle` (the latter broken) were removed on Rémy's decision (`82aaac2`); `TODO.md` still mentions the first one;
  - [x] `RayOptimizerConfig.build_optimizer` now requires a World and declared agents, with an error naming what to add, on Rémy's decision (`9f765da`);
  - [x] **the regulator's candidate reached training only in the first episode of each environment**: the environment now keeps the last candidate it fetched and gives it to the leaders at every reset, on Rémy's decision (`b330a82`). Measured on the shrunk config: every sampled training episode (16 of 16) and every evaluation episode (24 of 24) carries the candidate; the resets without one all happen while RLlib builds its env runners, before the generation's candidates are published. In that config the quota never binds (the stock stays between 0.64 and 0.85 of capacity, untrained fishermen request about half their capacity, the quota allows at least 93 %), so training returns are unchanged by the fix there. Since `472feb1` those resets play the leaders' mechanism defaults, and the binding measurement predates the new dynamics of `53e5ddd`;
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
  - [x] end-of-phase measurement on `0477d20`: 1363 passed, 17 xfailed, no failure, 28 s; `ruff check --no-fix .` and `ruff format --check .` pass on the whole tree. The 17 remaining strict `xfail` tests record 11 defects, left unchanged on Rémy's decision and handed to Nadine. **Superseded on 10-04:** Rémy decided that every defect found is fixed; the fix pass of phase 3 fixed all eleven, and no `xfail` marker remains in the tree.
- [x] Phase 3 — docstrings and type hints on every public symbol of `core/` and `examples/bilevel_fishery`.
  - [x] measured again on `436dc0f` with an explicit counting rule (a public symbol is a module-level class or function without a leading underscore and the public methods and properties of such a class; `__init__` counts for annotations only): 78 % of public symbols documented (266 of 342), 93 % of public callables fully annotated (262 of 283), 57 of 66 modules with a module docstring. The figures of the baseline used an unrecorded rule and are not comparable. Several existing docstrings are false (the optimizer base class still names `run`), so the phase also corrects existing docstrings;
  - [x] Rémy decided that docstring examples run in the suite and that classes and module-level functions get the full template while methods get a short one. `pytest.ini` now runs `--doctest-modules` on `core/` and `examples/bilevel_fishery`, excludes `debug.py` (importing it re-executes the process) and sets `consider_namespace_packages = true`, without which `core/reporting/csv.py` and `core/mechanism/types.py` collide with the standard library modules of the same name. With that change the suite gives 1373 passed and 3 failed doctests, all three in `core/optimizers`. Since `183caa8` the exclusion is gone, because nothing runs at the import of a debug script any more, and the cart-pole and fresh-water examples are test paths as well;
  - [x] seven subagents, one per layer (optimizers, Ray adaptor, environments and agents, mechanisms, reporting and metrics, World with configuration and top-level utilities, fishery example), shared one brief: docstrings and annotations only, no behaviour change, annotation contradictions and suspected bugs reported rather than fixed, every reference verified online. For each layer the lead checked with an AST comparison that the code is identical to the previous commit once docstrings, annotations and imports are removed, re-ran the layer's tests and doctests, the lint and the docstring count, and read the claims behind the reported defects. One commit per layer: `5786596` mechanisms, `e4e3adb` reporting and metrics, `e24306e` optimizers, `ce079e9` the doctest configuration, `a831019` Ray adaptor, `572e08e` environments and agents, `730f17e` fishery example, `6a17ffa` World, configuration and utilities, `ffbfdc0` one stale sentence left in `RayOptimizer.evaluate`;
  - [x] end-of-writing measurement on `ffbfdc0`: every one of the 66 modules has a module docstring, 342 of 342 public symbols have a docstring and 283 of 283 public callables are fully annotated (annotation contradictions counted as annotated, see below); all 134 public classes and module-level functions carry a "When to use" paragraph and an example. The suite gives 1505 passed (1363 tests and 142 doctests), 1 skipped doctest (`RayRuntime`, which calls `ray.init`) and 17 xfailed, in 27 s; `core/` stays at 99 %; `ruff check --no-fix .` and `ruff format --check .` pass on the whole tree. References added and verified online: Salimans et al. 2017 (arXiv:1703.03864), Espeholt et al. 2018 (arXiv:1802.01561), Liang et al. 2018 (arXiv:1712.09381), Pella and Tomlinson 1969 (page range corrected to 421-458), Jaques et al. 2019 kept;
  - [x] fix pass decided on 10-04 (complete on 10-05): every annotation contradiction, every defect found in phases 1 to 3 (the eleven phase 2 defects included) and the dead code are fixed or deleted, each defect in its own commit with a test. Six subagents worked one layer each under one brief (`fix_brief` in the session scratchpad, not kept); the lead re-ran each layer, read every change and made the cross-layer commits. Everything below is committed:
    - [x] World and utilities: the uncalled methods and helper are deleted while the read accessors stay, on Rémy's decision (`ae0e56f`, `c1f091f`); an updated context is validated before it is stored (`a18c5b8`); flushing keeps the three registries in sync (`caf369e`); the index lookup scans the candidates' batch index and skips scored entries (`f48cabb`); a fetch mode other than train or eval is rejected up front (`5a2f552`); annotations corrected (`a25f6b8`);
    - [x] optimizers: `OptimizerConfig.training` returns the config, and `build_optimizer` no longer reads the optimizer identifier before checking for a World, a fix its commit message does not mention (`1d0d0f4`); ES fixed mode logs its generation (`eabd85c`); the ES convergence stop is implemented as a flagged heuristic and made opt-in on Rémy's decision (`54a70a1`, `f25f223`); `ESSchema.generation` is filled (`f52e7c6`); the bilevel optimizer takes `converged` from the ES summary (`95d3c30`) and forwards its episode count, rejecting a conflict (`0cda396`); a regulator without horizon is rejected before the ES loop, which would otherwise never end (`4256d1d`); the inert reporting schema parameter is removed everywhere on Rémy's decision (`bbba0f9`, `c0a24d3`); dead settings deleted (`1510447`, `3206a8f`); annotations corrected (`3f64c7b`);
    - [x] environments: `MultiAgentEnv` runs without leaders (`14bc038`) and without a metric schema (`08aa583`, `95fb945` for the fishery); the identifier getters no longer recurse (`c77815f`); a failed mechanism fetch raises the error that names it (`5f2b295`, `4efe1de`); no library module configures the root logger at import (`da92de8`, `3feb8f6`, `2795092`); `RegulatorEnv.reset` accepts and ignores a seed (`0df8a81`) and has no mutable default (`1fc646a`); a horizon of N gives N steps (`4b8449b`); two methods carrying the same hook mark are rejected (`05dfc50`, `bfcc5c5`); leaders apply their mechanism defaults before the first candidate, on Rémy's decision (`472feb1`); dead counters and hook decorators deleted (`c180199`, `be2d354`); annotations corrected (`b4de98f`);
    - [x] metrics and reporting: `Max` and `Min` ignore NaN whatever the push order (`a3b0759`); NumPy scalars are accepted (`9e4cb3a`); a specialised node keeps its earlier values (`e2ce117`); the abstract reporter config declares `label` (`60a0f4b`); `WandbConfig` rejects unknown arguments (`86b0c80`) and starts no run for an empty query (`4baa02f`); TensorBoard writes each point once (`a094e05`); the dynamic children of the metric logger receive gaps so every series stays aligned with its pushes, and the reporter's mean skips them, on Rémy's decision (`b9998d4`); dead enums, copied helpers and the write-only `Reporter.schema` deleted (`8608dfb`, `7f06f47`, `9b6a9d9`); annotations corrected (`a6e599b`, `64738ea`);
    - [x] Ray adaptor and callbacks: the society config can be frozen (`5426913`); `debugging` is order-independent and idempotent (`d9d29fc`); a measured 0.0 is kept (`a9446f1`); non-finite losses are filtered in the classic layout (`1bc65f8`); the config mutator keeps the builder's name and docstring (`121f006`); `PolicyActor.reset` stops the previous algorithm and its learner thread (`b2fe39c`); every episode an environment finishes in an iteration is kept, under keys ending in `|n=<index>` (`ed09a16`); dead state and the unused MPS model deleted (`5a8c05a`, `c0446c8`, `d93e3e7`); annotations corrected (`334fe48`);
    - [x] mechanisms and agents: a mechanism default is kept unchanged instead of being tested for truth (`5bc72ab`); the quota publishes the current allowed fraction instead of accumulating it (`ee062e9`); `decode` receives the raw action at every step through a new `MDPState.raw_actions` field, bit-identical for the four mechanisms (`56dde42`, `3fca5d9`); `Trajectory.add` rejects a skipped step (`29f351d`); `ActType` comes from gymnasium (`ae31d93`) and the unused `gym` dependency is gone (`4c168d8`); one shared `ACTION_TEMPERATURE` replaces five copies (`e5148b2`, `da2fb4a`); `violation_transition_width` deleted (`31d8e30`); annotations corrected (`2f7a13b`);
    - [x] fishery: the standard discrete surplus-production dynamics with pro-rata rationing, on Rémy's decision (`53e5ddd`); the fitness uses the last 50 steps of each evaluation episode, on Rémy's decision (`3205b73`); `total_fines` removed (`4b1b39d`); clearer errors for a missing capacity and an empty split (`79bebf1`, `0b1e627`, `6e66155`); the debug script runs from `main` (`f9ed276`); the never-filled trajectories deleted (`7c49367`), and so are the regulator's write-only denormalised threshold and the `K` it alone required (`4c416a4`); two long docstring lines reflowed (`86e52b2`);
    - [x] cart-pole: ported to the present interface (`09033f7`); its fitness pools every logged episode (`b9998d4`); its documentation states that the fitness is the constant 1.0 and that the dial is inert (`c6fc0ce`);
    - [x] fresh-water: ported to the present interface (`2010b10`), then changed as Rémy decided on 10-05, by one subagent whose work the lead verified in the code. The sustainability deviation compares the lake's filled level with and without withdrawal, `deviation_series` is gone and `streamflow_deviation` is renamed `level_deviation` (`ad73d49`). The delivered-water entry of the farm observation is divided by `max_daily_need_m3_day`, the number of farms times the farm area times the peak crop need of 7.26478 mm/day (maximum crop coefficient 1.15 times the July reference evapotranspiration of 6.3172 mm/day); for the 500 farms of `debug.py` that is 3 632 390 m³/day, an upper bound the entry does not reach because the two peaks never fall on the same day (`5fe4446`). The residence time is the stored volume over the outflow, in days, with the capacity `lake_area_m2 × max_depth_m` now exposed for both lake models (`2591cbf`). The Raven baseline runs once per episode up to its last day and is read daily; the readings equal those of the per-day re-run, checked against the stand-in only (`e6a3015`). The two inert rules are documented as such (`dff5828`). The Raven integration test was vacuous, because the stand-in's rain covered the whole crop need; it now runs without rain and checks a real withdrawal (`ba11d82`). Measured with `debug.py --outer-iters 3 --train-iters 2 --horizon 30 --num-agents 20 --reporter csv` and `PYTHONHASHSEED=0`, deterministic over two runs: the level deviation is 0.0014, 0.0010 and 0.0014 over the three generations where the outflow deviation was 0 (20 farms barely lower a lake of 6.3e7 m³), the fitness moves from 1.2533, 1.2020 and 1.2250 to 1.2816, 1.2319 and 1.2595 after the deviation and normalisation changes, the other changes leave it unchanged, the residence time ranges from 42 to 158 days (mean about 80) where it was about 1e-6, and in the stand-in run the baseline ran 17 times against 295 main runs. A run with the 500 farms of `debug.py` did not finish within 10 minutes;
    - [x] repository: notebooks are no longer git-ignored (`ff67552`); the doctests of every example run by default (`183caa8`); isort agrees with the formatter and `spacing` is a dev tool (`fd7ba11`); the archived in-code notes that later commits resolved say so in `TODO.md` (`6ee25c5`);
    - [x] measured: applying the mechanism defaults changed no training or evaluation episode in the reduced fishery configuration, because the 48 resets that play the default all happen while RLlib builds its runners; fitness, returns and initial weights are identical. Keeping every episode leaves the fishery fitness and returns identical, since evaluation ends one episode per environment, while the training curves now average the five episodes each environment ends per iteration instead of showing the last one; in a cart-pole APPO iteration one environment ended 12 episodes, of which the old code kept 1. On `ba11d82`, the end of the fix pass, the suite gives 1855 passed and 1 skipped in 110 s, and `ruff check --no-fix .` and `ruff format --check .` pass on the whole tree with no warning.
- [x] Phase 4 — README, QUICKSTART, AGENTS, ARCHITECTURE and notes for Nadine rewritten against the tree (closed on 10-05).
  - [x] inventory on 10-05 by three read-only subagents, checked by the lead where it touched a decision. Only `README.md` exists on this branch; `QUICKSTART.md`, `AGENTS.md`, `docs/ARCHITECTURE.md` and `docs/MERGE_NOTES.md` exist on `feature/social-influence-testing` (August) and `feature/integration-trial`, and describe an earlier architecture, so they are rewritten from the tree rather than corrected. The README has 21 of 26 statements wrong or stale (entry points `main.py`, top-level `mechanism/`, `legacy_code/`, Python 3.13, pip and conda, RavenPy, `water_usage`). The guides' architecture is wrong on these points: fitness comes from the inner optimizer's metrics through `RegulatorEnv.reward`, not from published `EnvStepContext`s; `RegulatorEnv` holds the inner optimizer; the inner environment is `MultiAgentEnv` behind `RLlibMultiAgentEnvAdapter`; an environment without a candidate plays its leaders' defaults, not an inert step; `run` became `train`, `aggregate_rewards` became `reward`; the ES unflattens each vector into a dictionary of mechanism actions; mechanisms are `decode`/`apply`/`observe` with a `MechanismConfig`; reporting is a `Reporter` base with CSV, TensorBoard and W&B backends; only `reset` and `transition` hooks remain. Never described anywhere: the agents layer, `MDPState` and trajectories, the metrics layer, the YAML loader and `core.config.cli`, `ensure_hash_seed`, `learner_drain`, the cart-pole and fresh-water examples, the CI jobs and the conventions decided in this pass;
  - [x] the PPO crash that `examples/cartpole/main_ppo.py` documents no longer happens: `debug --algo ppo --outer-iters 1 --train-iters 2 --horizon 20 --reporter csv` exits 0 on `0a5c251` (the episode-keeping fix `ed09a16` came after the note). The note is stale, not the code;
  - [x] the findings for Nadine were checked against the 167 commits since `96294f6`: 14 fixed, 5 decided by Rémy, 8 open, 2 superseded. Stale against the tree: the removed-API list omits about fifteen deletions (`ae0e56f`, `82aaac2`, `5a8c05a`, `c0446c8`, `3206a8f`, `c180199`, `3205b73`, `34ae6f6`, `ad73d49`, `7c49367`, `53e5ddd`); the removal of the "Restoration subsidy vs fixed quota" query (`34ae6f6`) was to be recorded for Nadine and is not; `ACTION_TEMPERATURE` replaced four copies, not five; in `TODO.md` five archived notes are resolved but unmarked (`_build_agent_policy_map` by `82aaac2`, `_has_converged` by `54a70a1`, the clipping note by `53e5ddd`, the `Subsidy` notes by `d317b9f`, possibly the default-mechanism note by `472feb1`) and §4/§5 still describe the earlier constructors;
  - [x] small changes before the guides, done on 10-05 (`5a95efa`, `bd5254f`, `38680f8`, `63394e9`, `1959417`, `b5c7365`, `c23531d`, `00e40ec`): package renamed `metamarl` with the MetaMARL URLs and `include = ["core*"]` (the namespace discovery also lists `docs`, `htmlcov`, `results`, `src`, `tutorials` and `wandb` as top-level names), `legacy_code` dropped from both excludes; the `metamarl` console command with its test; the stale PPO note; the fisher reward docstring; the unused `ray_session` fixture; the `TODO.md` marks;
  - [x] README, QUICKSTART, AGENTS, `docs/ARCHITECTURE.md` and `docs/MERGE_NOTES.md` written from the tree, every command in them executed.
    - [x] README rewritten and QUICKSTART written, every command executed (`eb684c8`);
    - [x] `docs/MERGE_NOTES.md` written for Nadine (`5065a9f`): its 86 quoted hashes resolve on the branch and its figures were checked against the commit messages and this file; the check found that the Ray optimizer's messages said `[PPO]` under APPO, fixed with its test (`f172c1d`). Full suite on that commit: 1861 passed, 1 skipped, `core/` at 99 %;
    - [x] `AGENTS.md` and `docs/ARCHITECTURE.md` (`d3bbbd3`): drafted by a subagent, then checked claim by claim by two independent read-only subagents (about 125 claims for the architecture guide, 12 wrong or imprecise; about 50 for AGENTS, 8 wrong), all corrected. The drafting turned up code defects, fixed with their tests: `@override(gym.Env)` on `MultiAgentEnv.reset`, whose class does not derive from `gym.Env`, with a static test over every `@override` (`a0dc6cc`); `ESConfig` silently ignoring unknown arguments (`bbd97a8`); the `Query` module example using a path no schema resolves (`d3e3b0b`). On Rémy's decision, `ESConfig.debugging` rejects `num_seeds`, which nothing read (`2ff20a1`), and the stray `# Deprecated` comment above the `to_float` import is gone (`fa3085e`). Dismissed after checking: the two `test_annotations.py` files collect without conflict, and `RayOptimizer.save` doing nothing matches the base class. Measured on `d3bbbd3`'s tree: 1857 unit tests passed, 1 skipped; 13 integration tests passed; 1871 collected; `core/` at 99 %; the smoke run gives the same fitness vectors as before.
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
assistant text. (Since then `core/registry.py` was deleted, `spacing` became a dev
dependency in `fd7ba11` and the library modules stopped configuring logging; the README
is rewritten in phase 4.)

## Decisions log

| Date | Decision | By |
| --- | --- | --- |
| 10-04 | Work on a new branch `feature/social-influence-testing-v2` created from Nadine's branch; her branch is not modified | Rémy |
| 10-04 | The six-phase plan of the status board is approved | Rémy |
| 10-04 | Port `Subsidy`, `SocialInfluence` and `ThresholdPenalty` to the new `Mechanism.apply` interface on the model of `Quota`, with tests | Rémy, on Claude's recommendation |
| 10-04 | ~~Leave the cartpole and fresh-water examples unported and record them for Nadine~~ (superseded the same day: both are ported) | Rémy, on Claude's recommendation |
| 10-04 | ~~Keep Nadine's `ruff.toml` untouched; fix genuine errors only and report that the check cannot pass as configured~~ (superseded the same day) | Rémy, on Claude's recommendation |
| 10-04 | The branch is ours: make the whole tree pass Nadine's `ruff.toml` and formatter rather than leaving half the work to her; the configuration itself stays untouched | Rémy |
| 10-04 | TODO comments are moved verbatim into `TODO.md`, grouped by file, then removed from the code; commented-out code is deleted (it stays in git history) | Rémy, on Claude's recommendation |
| 10-04 | In the tutorials, commented-out worked examples become fenced code blocks in markdown cells rather than being deleted | Rémy, on Claude's recommendation |
| 10-04 | Delete `core/registry.py` and the five uncollected scripts in `tests/integration` | Rémy, on Claude's recommendation |
| 10-04 | The pass lives in the main directory, which was switched to the new branch; the temporary audit worktree was removed | Rémy |
| 10-04 | Complete `config.yaml` with exactly the values of `debug.py` and add a clear error when the evaluation setup is missing | Rémy, on Claude's recommendation |
| 10-04 | ~~Leave the ES objective and its no-op tail window unchanged; record the finding for Nadine~~ (superseded the same day: the tail window is made real) | Rémy, on Claude's recommendation |
| 10-04 | Log the step reward and reduce `fish_norm_next_last` with `LAST`; record the negative net harvest for Nadine (the last part is moot since the new dynamics) | Rémy, on Claude's recommendation |
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
| 10-04 | ~~Quota `allowed_frac`: leave the code unchanged and record it for Nadine~~ (superseded the same day: fixed in `ee062e9`) | Rémy, on Claude's recommendation |
| 10-04 | Fix the lost reporter now: keyword at both `build_optimizer` call sites, stale ES queries, per-query error isolation, `None` reporter check, empty seed list error; one commit each with its test | Rémy, on Claude's recommendation |
| 10-04 | Learner statistics missing on an iteration are logged as NaN, so every series keeps one value per iteration | Rémy, on Claude's recommendation |
| 10-04 | ~~The other recorded defects are left unchanged and handed to Nadine in the phase 4 notes, each with its test (eleven defects after the reporting fix, which also covered the `None` reporter and the empty seed list)~~ (superseded the same day: every defect is fixed) | Rémy, on Claude's recommendation |
| 10-04 | Remove the ES query "Restoration subsidy vs fixed quota", whose parameter the fishery regulator does not search, and record it for Nadine | Rémy, on Claude's recommendation |
| 10-04 | Declare the learner batch size as a float, since RLlib reports a mean | Rémy, on Claude's recommendation |
| 10-04 | Docstring examples run in the suite (`--doctest-modules`); classes and module-level functions get the full template, methods a short one | Rémy, on Claude's recommendation |
| 10-04 | Every annotation that contradicts the code is corrected, design cases included, aligned on what the code does | Rémy |
| 10-04 | Every defect found is recorded and fixed, including the eleven defects of phase 2 that had been left for Nadine: leaving them was a mistake | Rémy |
| 10-04 | Dead code (attributes and settings never read, unused enums and constants, functions without caller, helpers copied in three reporters) is recorded and deleted | Rémy |
| 10-04 | The ES convergence settings are implemented: stop when the displacement of the mean stays under the threshold for `patience` generations (a heuristic, flagged as such) | Rémy, on Claude's recommendation |
| 10-04 | The fishery `total_fines` field, which carried the mean biomass, is removed | Rémy, on Claude's recommendation |
| 10-04 | The fishery dynamics become the standard discrete surplus-production model: B(t+1) = B(t) + g(B(t)) − C(t), Pella-Tomlinson growth on the current stock, simultaneous requests rationed pro rata, logged harvest equal to the actual catch | Rémy, on Claude's recommendation |
| 10-04 | The fitness tail window becomes real: evaluation episodes log step-by-step series and the objective uses their last steps | Rémy, on Claude's recommendation |
| 10-04 | The cartpole and fresh-water examples are ported to the present interface, each with a reduced run in the suite | Rémy, on Claude's recommendation |
| 10-04 | `World` keeps its read accessors; the claim and update methods that nothing calls are deleted | Rémy, on Claude's recommendation |
| 10-04 | The ES convergence stop is opt-in: `convergence_eps` defaults to 0, which disables it, because the measured stop fired far from the optimum in single-candidate mode | Rémy, on Claude's recommendation |
| 10-04 | Leaders apply their mechanism defaults before the first candidate is published | Rémy, on Claude's recommendation |
| 10-04 | The optimizer-level `reporting(schema=...)` parameter, which nothing read, is removed everywhere | Rémy, on Claude's recommendation |
| 10-04 | The console silencing (W&B `quiet`, Ray loggers at `WARNING`, `log_to_driver=False`) is left unchanged and flagged for Nadine | Rémy |
| 10-04 | The fisher's reward stays the requested fraction rather than the delivered catch; it is flagged for Nadine and in the documentation | Rémy |
| 10-05 | Every episode an environment finishes in an iteration is kept: the episode callback logs with `item_series`, the metric logger records a gap (`None`) for a dynamic id that a push leaves out, curves average over the episodes present and the fitness skips the gaps | Rémy, on Claude's recommendation |
| 10-05 | Fresh-water sustainability deviation is measured on the lake level with and without withdrawal, replacing the inflow/outflow choice | Rémy, on Claude's recommendation |
| 10-05 | The two fresh-water rules that no dynamics read (`under_irrigation_penalty_scale`, `max_farm_area_m2`) are kept, documented as inert and flagged for Nadine | Rémy |
| 10-05 | The fresh-water observation of the water delivered the previous day is normalised by the maximal daily need of all farms | Rémy, on Claude's recommendation |
| 10-05 | The fresh-water residence time becomes storage volume over outflow, in days | Rémy, on Claude's recommendation |
| 10-05 | The Raven no-withdrawal baseline is run once per episode and read daily instead of being re-run every day | Rémy, on Claude's recommendation |
| 10-05 | The cart-pole example stays a pipeline check; its documentation states that the fitness is the constant 1.0 and the dial inert | Rémy, on Claude's recommendation |
| 10-05 | The six commit trailers that name Sonnet 5.5 are rewritten once, just before the push, and every cited hash is updated afterwards | Rémy, on Claude's recommendation |
| 10-05 | The fisher's reward stays as it is. Measured by the composition test of the fishery regulator, it is the harvest fraction after the regulator's quota and before the stock's pro-rata rationing; the notes and the docstring say so | Rémy, on Claude's recommendation |
| 10-05 | The project is called MetaMARL everywhere: guides, package name and project URLs in `pyproject.toml` | Rémy, on Claude's recommendation |
| 10-05 | A `metamarl` console command is added for `run` and `check`; the module form keeps working | Rémy, on Claude's recommendation |

## Waiting on

Nothing. Phase 4 is in progress.

## Findings for Nadine (to go into the phase 4 notes)

Every finding recorded in this section up to 10-04 that described a defect was fixed,
either in phase 1, by the reporting fix that closed phase 2, or by the fix pass of
phase 3; the status board names the commit of each. The earlier text of this section,
with the full measurements behind each finding, is in the git history
(`git show dd59fb7:docs/REPRISE.md`). What follows is what Nadine still needs to know:
the questions left to her, the behaviour that changed under her code, and the choices
kept on Rémy's decision.

**Questions left to her.** Whether `RayOptimizer.stop` should return the metrics reduced
over the run is her choice; the base class is now annotated `-> Any` and the Ray
optimizer returns them. Her acceptance lists for the subsidy and social influence in
`TODO.md` (sections 4 and 5 of her notes of 08-21) are covered by tests, but one item is
hers to decide: implement the KL influence term of Jaques et al. (2019, PMLR 97) or
rename `SocialInfluence`, which only shapes the observation and whose `influence_weight`
has no effect. Those sections still describe the earlier constructor arguments.

**A methodological doubt to re-measure.** On 10-04, in the shrunk fishery configuration,
the four fitness values of the second generation were bit-identical to those of the
first although the candidates differed. Each population slot trains and evaluates its
own policy module (`fisher_policy_m<idx>_s<seed>`) with its own initial weights, so the
ES gradient there followed the differences between slot initialisations rather than the
mechanism. That measurement predates the new dynamics, the real tail window and the
mechanism defaults, so it must be repeated in phase 6, on the shrunk and on the full
configuration, before anything is claimed either way.

**Behaviour that changed under her code.** `decode` now receives the raw action at
every step through `MDPState.raw_actions`, so it no longer has to be idempotent.
`Trajectory.add` rejects a skipped step, as `update` already did. The ES convergence
stop exists but is opt-in: `convergence_eps` defaults to 0, because in single-candidate
mode the measured stop fired after 12 to 41 generations far from the optimum (a
rejected step has zero displacement); it remains a heuristic. `get_mechanism_by_index`
returns the first copy that is not scored among the copies the regulator publishes for
each seed. Leaders play their mechanism defaults until the first candidate arrives. A
horizon of N gives N steps. The metric logger now records a gap (`None`) for a dynamic id
that a push leaves out, compiled values and the reporter's mean skip the gaps, and the
episodes of one environment in one iteration are stored under keys ending in
`|n=<index>`; the train curves therefore average every episode instead of showing the
last one. The removed APIs are the World's claim and update methods (`get_mechanism`,
`try_get_mechanism`, `update_context`, the `assigned` and `init` statuses), the agent
hook decorators, `ReporterType`, `Resolution`, `Reporter.schema` and the optimizer-level
`reporting(schema=...)`, `violation_transition_width`, `total_fines`,
`Agent._normalize_action`, the ESConfig `generation` and `dimension`, the MPS model and
the fishery regulator's `K` and `raw_sustainability_threshold`.

**Kept on Rémy's decision and flagged.** The console silencing stays: W&B runs with
`quiet`, the Ray loggers are set to `WARNING` and `log_to_driver=False`. The fisher's
reward is its harvest fraction after the regulator's quota and before the stock's
pro-rata rationing, so a fisher is paid for what the quota lets it take even when the
stock cannot deliver it (measured by `tests/integration/test_fishery_regulator_composition.py`). In the fresh-water
example the rules `under_irrigation_penalty_scale` and `max_farm_area_m2` are searched
and observed but read by no dynamics, as before the port, so the ES spends two
dimensions on them.

**Cart-pole.** The example is a pipeline check, not an optimisation problem: every step
of `CartPole-v1` pays 1.0, so the mean per-step reward that serves as fitness is the
constant 1.0, and the searched dial is inert.

**Fresh-water.** The port fixed two defects of the earlier code. The no-withdrawal
baseline was copied once at reset and never re-run, so the deviation was zero by
construction; it now runs in lockstep with the regulated lake. The crop stage rebuilt
the planting date from the current year, so about 37 % of episodes crossed 1 January
into an "offseason" with no water need and a free reward of 1.0; days after planting
are now counted from the true planting date. Fitness values from before the port are
therefore not comparable. On 10-05 Rémy decided four further changes, recorded with their
measurements in the status board: the sustainability term now measures how far the farms
lower the lake level (a new choice; neither the inflow nor the outflow measured it), the
delivered-water observation is normalised, the residence time is in days and the Raven
baseline runs once per episode. A configuration that still passes `deviation_series` is
silently ignored, like any unknown key of `ecology_cfg`. The Raven path is tested only
against a stand-in executable, because the Raven model is not in the repository; a run
on the real model with Nadine remains to be done.

**Tutorials (phase 5).** `mechanism_algorithms` builds the three mechanisms with their
earlier constructors in fenced code blocks and still mentions
`violation_transition_width`; `custom_benchmark_creation` uses the earlier
environment-level hook API; `visualization` uses stale queries; several tutorials pass
`reporting(schema=...)`, and `metamarl_fishery_tutorial` passes `K` to the regulator.
Three notebooks carry no kernel specification. The last section of `TODO.md` carries two
comments moved from the earlier `subsidy.py` whose line numbers no longer match.

**What the August suite tested that no longer exists.** The mechanism tests of August
checked `dimension`, `encode`, `to_vector`, `param_names` and `clip` on every mechanism;
none of these is defined in `core/` any more, since a mechanism's parameters now live in
its action space. The two composition test files were dropped with the composition
classes they tested; composition is now covered through `Agent.action` in the quota tests
and in the integration test of the composed fishery.

**Policy loss cadence.** With APPO on the new API stack the learner reduces its
statistics once every 20 gradient updates, so the loss appears in the training log only
on those iterations. This is RLlib's behaviour, not a bug of the framework.

**Learner wait before evaluation.** `PolicyActor.evaluate` waits until the learner thread
has found its input buffer empty. That wait reads private RLlib attributes and raises
rather than skipping when they are missing; it also raises for GPU learners and for
IMPALA's deque queue, which it does not cover.

## Next step

Phase 4 is closed. Phase 5 makes the five notebooks in `tutorials/` execute against
the present API and exercises them in the suite under the `notebook` marker, with
the tutorial items listed in the findings. The notebooks still call removed
interfaces (for example `reporting(schema=...)` and an `.inner(...)` builder), so the
first action is to execute each one and list its failures, delegating that reading
to a subagent; the fixes and decisions stay in the main session. The notes for Nadine
then gain a paragraph on the notebook changes.

Phase 6 audits by measurement:
- re-measure the slot-initialisation doubt above;
- time a fresh-water run at full size;
- just before the push, rewrite the six commit trailers that name Sonnet 5.5 and update
  every cited hash;
- after the rewrite, remap every hash quoted in `docs/MERGE_NOTES.md`, whose preface promises
  the pushed hashes, and add to it the slot measurement;
- push with Rémy's go-ahead.

**Suite conseillée :** modèle opus, effort high — phase 5 repairs five notebooks against
an API that changed under them, with judgment on each teaching cell; start it in a
fresh session after `/clear`.

## Known traps

Run scripts as modules from the repository root. Set `WANDB_MODE=offline` for every run.
Clear `__pycache__` after switching branches. The main directory holds ignored leftovers of the earlier branch (`htmlcov`, caches); they are harmless. Ray-backed tests must run last (`tests/conftest.py` has the ordering hook).
`nbconvert`, `ipykernel` and `jupyter` are in the dev group, so `uv run jupyter nbconvert`
uses the project's environment; give an explicit kernel name, since three notebooks carry
no kernel specification. Never run
`ruff check .` without `--no-fix` while `fix = true` is in `ruff.toml`.

The test directories have no `__init__.py`, so every test file needs a basename that is
unique in the whole of `tests/`; two files of the same name in different directories
fail at collection. No `xfail` marker remains; if one is ever used to demonstrate a defect, a strict `xfail`
fails the suite the day the defect is fixed, so remove the marker in the fixing commit. When several processes measure
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
