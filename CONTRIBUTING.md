# Contributing to MetaMARL

MetaMARL welcomes contributions to reusable agent and mechanism abstractions, optimization and RLlib adapters, benchmark environments, testing, reproducibility, reporting, and documentation.

The framework supports **multi-agent experiments with or without bilevel optimization**. Contributions should preserve that separation: a domain-specific model should not require an outer regulator, and an optimizer should not be coupled to one benchmark.

## Getting started

1. Fork or clone the [MetaMARL repository](https://github.com/nadinemgh/MetaMARL).
2. Install Python 3.12 and [uv](https://docs.astral.sh/uv/).
3. Install the project and development dependencies from the repository root:

   ```bash
   uv sync --group dev
   ```

4. Read [README.md](README.md), [QUICKSTART.md](QUICKSTART.md), and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), particularly if changing public interfaces.

The **release includes the fishery and CartPole examples**. The freshwater/hydrology benchmark is in development on `dev` and is **not part of the current release**. Do not describe development-only work as released functionality.

## Proposing changes

Use [GitHub Issues](https://github.com/nadinemgh/MetaMARL/issues) for bug reports, feature proposals, benchmark suggestions, and architectural discussions. For substantial changes, discuss the intended contract and a minimal reproducible example before implementing a new abstraction.

Create a focused branch and open a pull request against `master` for release-targeted changes, unless a maintainer asks you to target `dev` for work in progress. A pull request should explain:

- The problem being solved and the proposed behavior.
- The affected interfaces, benchmarks, and supported algorithms.
- The tests run and any relevant configuration or reproducibility implications.
- Any compatibility changes, limitations, and documentation updates.

Keep changes scoped. Avoid unrelated formatting or refactoring when fixing a focused bug.

## Checks before a pull request

Run from the repository root after `uv sync --group dev`:

```bash
uv run python -m pytest
uv run ruff check --no-fix .
uv run ruff format --check .
```

To run the faster tests without the integration and notebook markers:

```bash
uv run python -m pytest -m "not integration and not notebook"
```

When changing the agent–environment interface, mechanism composition, an optimizer, or reporting, also validate a relevant YAML experiment:

```bash
uv run metamarl check examples/bilevel_fishery/config.yaml
```

For changes affecting execution, run a suitably small experiment using the YAML CLI, as described in [QUICKSTART.md](QUICKSTART.md). The obsolete per-example `debug.py` commands are not supported entry points.

CI separates lint, unit, integration, and notebook checks. The test suite also collects doctests from modules, so update executable docstring examples whenever changing an interface. Always use `ruff check --no-fix`: repository Ruff settings may otherwise modify files.

### Testing and tracked limitations

- Prefer small, deterministic unit tests for individual contracts, and integration tests for Ray/RLlib execution and reporter outputs.
- Assert actual behavior, not just that a function was called or a process exited.
- Preserve tests that expose bugs; do not weaken assertions merely to obtain a green build.
- If a test must be skipped for a known limitation, link it to a tracked issue using the project's stable skip-reason IDs (for example, `OBS-001`, `MET-001`, `MET-002`, or `APPO-001`). Remove the skip when the behavior is fixed and verified.
- Explain the limits of deterministic execution. Equal random seeds do not guarantee equal trajectories when sampling order, batch sizes, random-number consumption, or asynchronous execution differ.

Do not commit generated run outputs, local virtual environments, credentials, private datasets, or experiment logs.

## Software design expectations

**Separation of concerns.** Keep application-specific dynamics and objectives in `examples/` where practical. Put generic agent, environment, mechanism, optimizer, and reporting contracts in `core/`.

**Explicit state contracts.** Document the state variables, observations, actions, rewards, reset behavior, and transition semantics of new environments. Preserve simultaneous multi-agent interactions with one shared state per step.

**Composable mechanisms.** Explain whether a mechanism contributes additively, constrains an admissible space, changes observations, or otherwise affects the transition. Do not assume every intervention is additive. Mechanisms should be testable independently and in composition.

**Algorithm independence.** Keep environment logic independent of the optimizer used to train agents or search mechanisms. Distinguish implemented algorithms (currently ES, PPO, and APPO in their respective roles) from proposed integrations.

**Reproducibility and measurement.** Document relevant seeds, evaluation protocols, training budgets, and random-stream assumptions. Include metrics that permit meaningful comparison, not only a final score. When modifying reporting schemas, update the corresponding queries and tests.

**Documentation.** Update README, quickstart, architecture notes, and API docstrings whenever you change workflows or publicly described behavior. Keep runnable examples consistent with the current YAML configuration interface.

## Reporting bugs and requesting support

[Open an issue](https://github.com/nadinemgh/MetaMARL/issues) with the smallest reproducible example you can provide. Include the relevant YAML configuration or override values, Python and Ray versions, command, traceback, and expected versus observed behavior. Remove API tokens, personal information, and private data.

## Authorship, attribution, and license

MetaMARL is principally developed by **Nadine (Mohamed) Abdelghani** (publishing as **Nadine Mohamed**), with scientific collaborators and institutional acknowledgements listed in the [README](README.md), [NOTICE](NOTICE), and [CITATION.cff](CITATION.cff).

Contributions are submitted for distribution under the repository's **[Apache License 2.0](LICENSE)**. Do not contribute third-party code or data unless its license permits inclusion and its required notices are retained. By submitting a contribution, you represent that you have the right to submit it under those terms; this does not automatically assign your copyright to the maintainers.

If you use MetaMARL in academic work, please cite it using [CITATION.cff](CITATION.cff). Academic citation is requested separately from the legal attribution obligations of Apache 2.0.
