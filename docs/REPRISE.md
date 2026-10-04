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
  - [ ] two `Mean of empty slice` warnings remain inside RLlib's EMA stats at the second training step, and the training log shows `policy_loss=NA`: not investigated yet;
  - [ ] remaining small bugs: `RayOptimizer.stop` calls `reduce(complie=True)` (a `TypeError` if ever called; nothing calls it today); `_get_policy_handle` references `self.algo` (dead); `RayOptimizerConfig.build_optimizer` leaves `opt_id` and `agents` unbound without a world or agents; `Optimizer.__init__` reads `config.episodes` before its `None` guard; `ESOptimizer.batch_capacity` raises `AttributeError` before it is set;
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

## Waiting on

Nothing waits on Rémy.

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

The other findings for Nadine are to be collected in the notes written in phase 4. Already
known: `ruff` warns that `isort.split-on-trailing-comma` conflicts with
`format.skip-magic-trailing-comma = true` in her configuration (no oscillation was
observed on this tree); `.gitignore` ignores `*.ipynb` although the tutorials are
tracked.

## Next step

Finish phase 1: the RLlib `policy_loss=NA` and empty-slice warnings, the remaining small
bugs listed in the status board, then `debug.py` (its documented options, on Rémy's
decision). Then the port of the three mechanisms in its own session.

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
