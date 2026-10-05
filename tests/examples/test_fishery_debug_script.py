"""``examples.bilevel_fishery.debug`` does nothing at import and parses its own options.

The script used to parse ``sys.argv`` and build and train the whole experiment
when it was imported, which also made importing it under pytest (or from a
tool that collects doctests) exit on the first unknown option or restart the
process. Everything now lives in ``main``. The end-to-end run of the script is
covered by ``tests/integration/test_debug_script_csv_smoke.py``.
"""

import importlib
import sys

import pytest
import ray


@pytest.mark.unit
def test_importing_the_script_parses_nothing_and_starts_nothing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest", "--not-an-option-of-the-script"])
    monkeypatch.delitem(sys.modules, "examples.bilevel_fishery.debug", raising=False)

    module = importlib.import_module("examples.bilevel_fishery.debug")

    assert not ray.is_initialized()
    assert callable(module.main)


@pytest.mark.unit
def test_the_defaults_are_the_full_run_and_the_options_are_read_from_argv():
    from examples.bilevel_fishery.debug import parse_args

    defaults = parse_args([])
    smoke = parse_args(
        ["--outer-iters", "2", "--train-iters", "2", "--num-agents", "2"]
        + ["--horizon", "20", "--reporter", "csv"]
    )

    assert (defaults.outer_iters, defaults.train_iters) == (1000, 50)
    assert (defaults.num_agents, defaults.horizon) == (10, 100)
    assert defaults.reporter == "wandb"
    assert (smoke.outer_iters, smoke.train_iters, smoke.num_agents) == (2, 2, 2)
    assert (smoke.horizon, smoke.reporter) == (20, "csv")


@pytest.mark.unit
def test_an_unknown_reporter_is_rejected_by_the_parser():
    from examples.bilevel_fishery.debug import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--reporter", "tensorboard"])


@pytest.mark.unit
def test_the_configuration_is_built_from_the_options_without_starting_ray():
    from examples.bilevel_fishery.debug import build_config, parse_args

    config = build_config(
        parse_args(["--num-agents", "3", "--horizon", "20", "--reporter", "csv"])
    )

    assert type(config).__name__ == "BilevelConfig"
    assert not ray.is_initialized()
