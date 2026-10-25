# AGENTS.md

This file guides coding agents and contributors working on **MetaMARL**. It records practical project conventions and boundaries; the architecture and scientific motivation are documented in [README.md](README.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

MetaMARL is an algorithm-agnostic framework for multi-agent learning, mechanism design, and experiments in interacting dynamical systems. **Bilevel optimization is optional.** The current release includes Evolution Strategies (ES) for mechanism optimization and RLlib-based PPO/APPO for agent learning. The released examples are `examples/bilevel_fishery/` and `examples/cartpole/`. **Freshwater is under development on `dev` and is not included in this release.** Do not add freshwater commands or functionality to release-facing documentation unless explicitly requested.

## Before making changes

1. Read [README.md](README.md), [QUICKSTART.md](QUICKSTART.md), and the relevant portion of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
2. Inspect the current implementation, nearby tests, and YAML configuration before proposing an interface change. Do not assume old examples or comments are accurate.
3. Consult [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and review expectations. `TODO.md` is a planning document, not proof that a feature is implemented or still broken; check the code.
4. Preserve existing maintainer `# TODO (nadine)` notes unless the requested change directly resolves them. Do not silently delete, rewrite, or create broad speculative TODO lists.

## Repository map

| Path | Responsibility |
| --- | --- |
| `core/agents/` | Agent configurations and action, observation, and reward interfaces |
| `core/mechanism/` | Mechanisms, configuration, residual effects, and composition |
| `core/envs/` | Multi-agent environments, regulator environments, reset/transition hooks, and schemas |
| `core/optimizers/` | Optimizer interfaces, ES, PPO/APPO configuration, and optional bilevel composition |
| `core/adaptors/ray/` | RLlib environment adapter, Ray optimization runtime, policy actors, policy mapping, and action sampling |
| `core/world/` | Shared `World` and mechanism lifecycle/context records |
| `core/metrics/` | Metric schemas, loggers, histories, and reductions |
| `core/reporting/` | Queries and CSV, TensorBoard, and W&B reporters |
| `core/config/` | YAML configuration, CLI, and hash-seed utilities |
| `examples/bilevel_fishery/` | Released fishery proof-of-concept benchmark |
| `examples/cartpole/` | Released RLlib/bilevel pipeline-validation example |
| `tests/` | Unit, integration, example, and notebook tests |
| `tutorials/` | Tutorial notebooks |

## Design invariants

- **Keep domain-specific logic outside `core/` where possible.** Reusable interfaces must not depend on fishery-specific ecology or regulatory objectives.
- **Do not assume a regulator exists.** Support fixed mechanisms and non-bilevel multi-agent experiments as well as leader–follower experiments.
- **Preserve the `World` coordination contract for bilevel runs.** Published candidates are conveyed to inner environments using `MechanismContext` records. Do not bypass this lifecycle with ad hoc state sharing.
- **Mechanism assignment is stable after publication.** An environment may initially request its candidate, but after a published mechanism has been installed it must not fetch or replace it during later resets. Distinguish this from use of a default mechanism before assignment.
- **Preserve simultaneous multi-agent semantics.** Agents act on the same current shared state; mechanisms and actions contribute to the resulting state transition according to explicit composition rules.
- **Respect state composition.** Mechanisms can contribute residual state, reward, or observation changes. Do not assume every effect is additive: constrained components may be intersected. Read `MDPState` and `Trajectory` implementations before changing composition. Decoded action values belong in `MDPState.actions`.
- **Preserve observation/action-space contracts.** Do not silently pad or resize observations in production as a workaround. Observation composition and dynamic sizing remain tracked under `OBS-001`.
- **Keep optimizer and environment concerns separate.** An algorithm-agnostic interface is not evidence that an unimplemented optimizer or benchmark is already supported.
- **Avoid claiming full determinism.** Matching initial weights and RNG seeds does not guarantee identical distributed RL training trajectories. `IidGetActions` maintains persistent CPU RNG state per policy/module ID **within each connector instance**, not per environment or episode; alignment requires matching random-number consumption.

## Configuration and commands

Use YAML configurations and the `metamarl` CLI. **Do not introduce or document the deleted per-example `debug.py` entry points.** Run from the repository root:

```bash
uv sync --locked --group dev
WANDB_MODE=offline uv run metamarl check examples/bilevel_fishery/config.yaml
WANDB_MODE=offline uv run metamarl run examples/bilevel_fishery/config.yaml
```

The module entry point is also available:

```bash
WANDB_MODE=offline uv run python -m core.config.cli run examples/bilevel_fishery/config.yaml
```

The CLI accepts overrides after the configuration path in `path.to.key=value` form. Use keys verified in the actual YAML file; don't invent flags or example override paths. `check` builds a configuration without training. `run` may start a long experiment, so choose a validated small configuration for smoke testing rather than assuming defaults are lightweight.

Configure reporters in YAML rather than using old `--reporter` script flags. `WANDB_MODE=offline` affects W&B runs; choosing CSV or TensorBoard requires the corresponding configuration. Do not assume fixed output paths, file counts, or exact fitness numbers without inspecting the configured queries and reporter.

## Tests and quality gates

```bash
uv run python -m pytest
uv run python -m pytest -m "not integration and not notebook"
uv run python -m pytest -m integration --no-cov
uv run python -m pytest -m notebook --no-cov
uv run ruff check --no-fix .
uv run ruff format --check .
```

- Run relevant targeted tests while iterating; run the full suite for changes affecting shared interfaces, optimization, or reporting. Integration tests may launch Ray and subprocesses.
- Tests and doctests must reflect the **current implementation**. Fix stale documented APIs instead of introducing deprecated methods merely to satisfy examples.
- Add assertions for behavior, not only execution or coverage. Test edge cases, failure paths, seeds, and cross-module interactions where they matter.
- Don't weaken or delete failing tests to obtain a green build without explaining why. A known limitation may be skipped only with a tracked and consistent reason. Use stable identifiers such as `OBS-001`, `MET-001`, `MET-002`, and `APPO-001` from `tests/known_issues.py` where that registry exists. Remove skips when an issue is resolved.
- The APPO inner CSV-reporting problem is tracked as `APPO-001`; threading or synchronization is a **suspected**, not confirmed, root cause. Keep other APPO coverage active where possible.
- `ruff.toml` may enable automatic fixes: use `ruff check --no-fix` when only inspecting lint results.

## Coding and documentation conventions

- Prefer focused changes to sweeping refactors. Maintain compatibility where reasonable; explain deliberate public API changes.
- Use explicit types, units, shapes, and reset/step semantics for new mechanisms or environments.
- Use NumPy-style docstrings for public interfaces, with executable `>>>` examples when they are short, stable, and appropriate. Complex distributed behavior belongs in pytest tests rather than fragile doctests.
- Use clear validation errors for invalid public arguments. Do not use `assert` as a substitute for runtime parameter validation.
- Explain *why* a non-obvious choice is necessary; avoid redundant comments and commented-out code.
- Write source, tests, comments, and documentation in English. Keep commits focused and descriptive.
- Do not commit credentials, `.venv/`, experimental output directories, or generated reporting artifacts.

## Debugging guide

- **Config fails:** Run `metamarl check` first and inspect the failing YAML target, builder call, or override path.
- **Candidate assignment fails:** Check mechanism ID, policy/environment seed, train/eval status, leader configuration, and whether the environment already holds a published mechanism. Do not add repeated fetches after assignment.
- **Observation dimension mismatch:** Inspect declared spaces, mechanism contributions, and composition order; consult tracked issue `OBS-001` before adding padding.
- **Inconsistent random streams:** Inspect policy ID seed suffixes, connector instances, module iteration order, batch sizes, and action-sampling draw counts. Do not assume environment-level RNG isolation.
- **Missing fishery regulator metrics:** The current `reward()` reads reduced episode fields including `reward_mean`, `fish_norm_next_mean`, `H_realized`, and `MSY`; old `*_series`-only mocks may be stale.
- **APPO reporting failure:** Preserve diagnostic logs, isolate inner CSV query/reporting from training behavior, and reference `APPO-001` instead of attributing the failure to a proven threading bug.
- **Reproducibility difference:** Check configuration, seeds, `PYTHONHASHSEED`, Ray/RLlib version, policy initialization, sampling order, and nondeterministic operations before concluding that a change is behaviorally equivalent.

## Scientific attribution and release scope

MetaMARL is distributed under **Apache License 2.0**. Follow [LICENSE](LICENSE), any [NOTICE](NOTICE) attribution requirements, and [CITATION.cff](CITATION.cff) for scholarly citation information. Retain applicable copyright notices; do not treat project authorship, institutional affiliation, and ownership of individual code contributions as interchangeable.

The principal developer is **Nadine (Mohamed) Abdelghani** (publishing as **Nadine Mohamed**). The research is affiliated with PPSP Lab (PI Guillaume Dumas), Mila – Quebec AI Institute, and Université de Montréal, with CIFAR funding. Consult the README and citation metadata for the full scientific author list.

Only **Fishery** and **CartPole** belong to this release. Describe freshwater/hydrological experiments as **development work on `dev`**, not installed, verified, or supported release functionality.
