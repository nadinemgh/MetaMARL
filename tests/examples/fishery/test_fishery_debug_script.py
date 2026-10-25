"""The standalone fishery ``run`` module does not train at import.

CLI overrides and reporter selection belong to ``core.config.cli`` and the YAML
configuration, rather than an experiment-specific ``argparse`` implementation.
"""

import importlib
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import ray

from core.config.yaml import load_experiment, load_yaml

RUN_MODULE = "examples.bilevel_fishery.run"
YAML_CONFIG = (
    Path(__file__).resolve().parents[3] / "examples" / "bilevel_fishery" / "config.yaml"
)


@pytest.mark.unit
def test_importing_the_script_parses_nothing_and_starts_nothing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest", "--not-an-option-of-the-script"])
    monkeypatch.delitem(sys.modules, RUN_MODULE, raising=False)

    module = importlib.import_module(RUN_MODULE)

    assert not ray.is_initialized()
    assert callable(module.main)
    assert type(module.bilevel_opt_cfg).__name__ == "BilevelConfig"


@pytest.mark.unit
def test_the_defaults_are_the_full_run_and_the_options_are_read_from_yaml():
    data = load_yaml(YAML_CONFIG)

    assert data["params"]["outer_iters"] == 1000
    assert data["params"]["train_iters"] == 50
    assert data["params"]["num_agents"] == 10
    assert data["params"]["horizon"] == 100

    smoke = load_yaml(
        YAML_CONFIG,
        overrides=[
            "params.outer_iters=2",
            "params.train_iters=2",
            "params.num_agents=2",
            "params.horizon=20",
        ],
    )

    assert (
        smoke["params"]["outer_iters"],
        smoke["params"]["train_iters"],
        smoke["params"]["num_agents"],
        smoke["params"]["horizon"],
    ) == (2, 2, 2, 20)


@pytest.mark.unit
def test_an_unknown_override_is_rejected_by_hydra():
    from core.config.yaml import ConfigError

    with pytest.raises(ConfigError):
        load_yaml(YAML_CONFIG, overrides=["params.unknown_setting=1"])


@pytest.mark.unit
def test_yaml_config_uses_wandb_reporter():
    data = load_yaml(YAML_CONFIG)
    calls = data["experiment"]["_calls_"]
    reporter = next(call["reporter"]["config"] for call in calls if "reporter" in call)

    assert reporter["_target_"] == "core.reporting.wandb.WandbConfig"
    assert reporter["project"] == "bilevel"


@pytest.mark.unit
def test_both_entry_points_construct_the_bilevel_configuration():
    module = importlib.import_module(RUN_MODULE)

    assert type(module.bilevel_opt_cfg).__name__ == "BilevelConfig"
    assert type(load_experiment(YAML_CONFIG)).__name__ == "BilevelConfig"
    assert not ray.is_initialized()


@pytest.mark.unit
def test_python_script_main_builds_and_trains(monkeypatch):
    module = importlib.import_module(RUN_MODULE)
    calls = []

    class DummyOptimizer:
        def train(self):
            calls.append("train")

    def build_optimizer():
        calls.append("build")

        return DummyOptimizer()

    monkeypatch.setattr(module.bilevel_opt_cfg, "build_optimizer", build_optimizer)
    monkeypatch.setattr(module.ray, "shutdown", lambda: calls.append("shutdown"))
    module.main()
    assert calls.count("build") == 1
    assert calls.count("train") == 1
    assert calls.index("build") < calls.index("train")
    assert calls[-1] == "shutdown"


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "examples.bilevel_fishery.regulated_env",
        "examples.bilevel_fishery.regulator_env",
    ],
)
def test_importing_an_environment_leaves_the_root_logger_alone(module):
    code = "; ".join(
        [
            "import logging",
            f"import {module}",
            "root = logging.getLogger()",
            "print(root.handlers, root.level)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip().splitlines()[-1] == f"[] {logging.WARNING}"
