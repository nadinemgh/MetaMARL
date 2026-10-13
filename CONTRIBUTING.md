# Contributing to MetaMARL

MetaMARL welcomes contributions to reusable agent and mechanism abstractions, optimization adapters, benchmark environments, tests, reproducibility, and documentation.

## Getting started

1. Fork or clone the [repository](https://github.com/nadinemgh/MetaMARL).
2. Install Python 3.12 and [uv](https://docs.astral.sh/uv/).
3. From the repository root, run `uv sync`.
4. Read [README.md](README.md), [QUICKSTART.md](QUICKSTART.md), and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing core abstractions.

## Proposing changes

Open a [GitHub issue](https://github.com/nadinemgh/MetaMARL/issues) for bugs, feature requests, research benchmark proposals, or architectural discussion. For substantial changes, discuss the interface and expected behavior before implementation. Please distinguish currently implemented features from planned research directions.

Create a focused branch, make your changes, and open a pull request against `master`. Explain the motivation, affected components, testing performed, and any changes to default behavior or experiment reproducibility. Add or update documentation and tests alongside code changes.

## Checks before a pull request

Run these commands from the repository root:

```bash
uv run python -m pytest
uv run ruff check --no-fix .
uv run ruff format --check .
```

Run a relevant example or a small smoke experiment when changing the agent–environment interface, mechanism composition, optimizer, or reporting. See [QUICKSTART.md](QUICKSTART.md) for a starting point. Avoid committing generated run outputs, credentials, `.venv/`, or local experiment logs.

## Software design expectations

- Keep domain-specific benchmark logic out of reusable `core/` abstractions when possible.
- Document the state, action, reward, and mechanism interfaces for any new agent or environment.
- Explain whether a state-update component is additive, constraining, or otherwise composed; do not assume all mechanism effects are additive.
- Preserve simultaneous-action semantics when updating shared environment state.
- Provide small, deterministic tests where feasible and disclose sources of nondeterminism in distributed experiments.
- Document solver support accurately: an extensible interface does not imply that every proposed solver has been implemented.

## Community support

Use [GitHub Issues](https://github.com/nadinemgh/MetaMARL/issues) for questions and bug reports. Include a minimal example, relevant configuration, software versions, and an error trace when reporting a problem. Avoid including tokens or private data.

## License

By contributing, you agree that your contributions will be distributed under the repository's [BSD 3-Clause License](LICENSE).
