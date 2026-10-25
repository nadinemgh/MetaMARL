"""``core.config.cli``: run/check, Hydra overrides and both fishery entry points.

The YAML entry point is tested without starting training; its full build/train
path is exercised with a stub optimizer. The independent Python ``run.py`` is
also exercised with its optimizer patched, so neither test starts Ray workers.
"""

from __future__ import annotations

import logging
import os
import runpy
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import yaml

from core.config import cli, hash_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
FISHERY_CONFIG = REPO_ROOT / "examples" / "bilevel_fishery" / "config.yaml"
CONSOLE_SCRIPT = Path(sys.executable).parent / "metamarl"
VALID_CONFIG = """\
params:
  denominator: 4
experiment:
  _target_: fractions.Fraction
  _args_:
    - 6
    - ${params.denominator}
run:
  - limit_denominator:
      _args_: [1]
"""


def _main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["metamarl", *argv])

    return cli.main()


@pytest.fixture
def config_path(tmp_path) -> Path:
    path = tmp_path / "config.yaml"

    path.write_text(VALID_CONFIG)

    return path


@pytest.mark.unit
def test_run_executes_the_run_section_and_exits_zero(monkeypatch, config_path):
    results = []

    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda path, *, overrides=None: results.append((path, overrides)),
    )
    assert _main(monkeypatch, "run", str(config_path)) == 0
    assert results == [(str(config_path), [])]


@pytest.mark.unit
def test_run_passes_hydra_overrides_to_loader(monkeypatch, config_path):
    results = []

    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda path, *, overrides=None: results.append((path, overrides)),
    )

    assert _main(monkeypatch, "run", str(config_path), "params.denominator=3") == 0

    assert results == [(str(config_path), ["params.denominator=3"])]


@pytest.mark.unit
def test_check_passes_hydra_overrides_to_loader(monkeypatch, config_path, capsys):
    results = []

    def load(path, *, overrides=None):
        results.append((path, overrides))

        return object()

    monkeypatch.setattr(cli, "load_experiment", load)

    assert _main(monkeypatch, "check", str(config_path), "params.denominator=3") == 0

    assert results == [(str(config_path), ["params.denominator=3"])]
    assert "Config OK: builtins.object" in capsys.readouterr().out


@pytest.mark.unit
def test_run_with_the_real_loader_succeeds_on_a_valid_config(monkeypatch, config_path):
    assert _main(monkeypatch, "run", str(config_path)) == 0


@pytest.mark.unit
def test_check_does_not_execute_the_run_section(monkeypatch, config_path, capsys):
    def fail(path, *, overrides=None):
        raise AssertionError("check must not run the experiment")

    monkeypatch.setattr(cli, "run_experiment", fail)
    assert _main(monkeypatch, "check", str(config_path)) == 0
    assert "Config OK: fractions.Fraction" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize("command", ["check", "run"])
def test_a_missing_file_is_a_configuration_error(
    tmp_path, monkeypatch, caplog, command
):
    with caplog.at_level(logging.ERROR):
        code = _main(monkeypatch, command, str(tmp_path / "absent.yaml"))

    assert code == 2
    assert "Configuration error: Config file not found" in caplog.text


@pytest.mark.unit
def test_a_config_without_an_experiment_is_a_configuration_error(
    tmp_path, monkeypatch, caplog
):
    path = tmp_path / "empty.yaml"

    path.write_text("other: 1\n")

    with caplog.at_level(logging.ERROR):
        assert _main(monkeypatch, "check", str(path)) == 2

    assert "requires a top-level 'experiment:'" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("command", ["check", "run"])
def test_keyboard_interrupt_exits_with_code_130(monkeypatch, config_path, command):
    def interrupt(path, *, overrides=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "load_experiment", interrupt)
    monkeypatch.setattr(cli, "run_experiment", interrupt)
    assert _main(monkeypatch, command, str(config_path)) == 130


@pytest.mark.unit
@pytest.mark.parametrize("argv", [[], ["train", "x.yaml"], ["check"]])
def test_invalid_arguments_exit_with_the_argparse_code(monkeypatch, argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _main(monkeypatch, *argv)

    assert exit_info.value.code == 2
    assert "usage: metamarl" in capsys.readouterr().err


@pytest.mark.unit
def test_invalid_override_exits_with_config_error(monkeypatch, config_path, caplog):
    # Additional positional arguments now represent Hydra overrides; the parser
    # accepts them, but composition rejects anything that is not a valid override.
    with caplog.at_level(logging.ERROR):
        status = _main(monkeypatch, "run", str(config_path), "b.yaml")

    assert status == 2
    assert "Configuration error:" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("command", ["check", "run"])
def test_print_config_does_not_build_or_run(monkeypatch, config_path, capsys, command):
    def fail(*args, **kwargs):
        raise AssertionError("--print-config must not construct an experiment")

    monkeypatch.setattr(cli, "load_experiment", fail)
    monkeypatch.setattr(cli, "run_experiment", fail)
    assert (
        _main(
            monkeypatch,
            command,
            str(config_path),
            "params.denominator=3",
            "--print-config",
        )
        == 0
    )

    composed = yaml.safe_load(capsys.readouterr().out)

    assert composed["params"]["denominator"] == 3
    assert composed["experiment"]["_args_"] == [6, 3]


@pytest.mark.unit
def test_script_entry_restarts_first_then_exits_with_the_command_status(
    monkeypatch, config_path, capsys
):
    order = []

    monkeypatch.setattr(hash_seed, "ensure_hash_seed", lambda: order.append("seed"))
    monkeypatch.setattr(sys, "argv", ["metamarl", "check", str(config_path)])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*found in sys.modules.*")

        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("core.config.cli", run_name="__main__")

    assert order == ["seed"]
    assert exit_info.value.code == 0
    assert "Config OK: fractions.Fraction" in capsys.readouterr().out


@pytest.mark.unit
def test_script_entry_exits_with_two_on_a_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setattr(hash_seed, "ensure_hash_seed", lambda: None)
    monkeypatch.setattr(sys, "argv", ["metamarl", "check", str(tmp_path / "no.yaml")])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*found in sys.modules.*")

        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("core.config.cli", run_name="__main__")

    assert exit_info.value.code == 2


@pytest.mark.unit
def test_fishery_configuration_passes_the_check(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert _main(monkeypatch, "check", str(FISHERY_CONFIG)) == 0
    assert "Config OK: core.optimizers.bilevel.BilevelConfig" in capsys.readouterr().out


@pytest.mark.unit
def test_fishery_yaml_run_builds_and_trains_without_starting_ray(monkeypatch):
    """Exercise the actual YAML path, replacing only costly optimizer execution."""

    from core.optimizers.bilevel import BilevelConfig

    calls = []

    class DummyOptimizer:
        def train(self):
            calls.append("train")

    def build(self):
        calls.append("build")

        return DummyOptimizer()

    monkeypatch.setattr(BilevelConfig, "build_optimizer", build)
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert (
        _main(
            monkeypatch,
            "run",
            str(FISHERY_CONFIG),
            "params.outer_iters=2",
            "params.train_iters=2",
            "params.num_agents=2",
            "params.horizon=20",
        )
        == 0
    )
    assert calls == ["build", "train"]


@pytest.mark.unit
def test_python_run_module_builds_and_trains_without_starting_ray(monkeypatch):
    """The independent run.py must still run without delegating to the YAML CLI."""

    import ray
    from core.optimizers.bilevel import BilevelConfig

    calls = []

    class DummyOptimizer:
        def train(self):
            calls.append("train")

    def build(self):
        calls.append("build")

        return DummyOptimizer()

    monkeypatch.setattr(BilevelConfig, "build_optimizer", build)
    monkeypatch.setattr(ray, "shutdown", lambda: calls.append("shutdown"))
    monkeypatch.setenv("WANDB_MODE", "offline")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*found in sys.modules.*")
        runpy.run_module("examples.bilevel_fishery.run", run_name="__main__")

    assert calls.count("build") == 1
    assert calls.count("train") == 1
    assert calls.index("build") < calls.index("train")
    assert calls[-1] == "shutdown"


LOCAL_MODULE_CONFIG = """\
experiment:
  _target_: local_experiment.Experiment
"""


@pytest.mark.unit
def test_entry_resolves_modules_of_the_working_directory(monkeypatch, tmp_path, capsys):
    (tmp_path / "local_experiment.py").write_text("class Experiment:\n    pass\n")
    (tmp_path / "config.yaml").write_text(LOCAL_MODULE_CONFIG)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "path", [p for p in sys.path if p not in ("", os.getcwd())]
    )
    monkeypatch.delitem(sys.modules, "local_experiment", raising=False)

    order = []

    monkeypatch.setattr(cli, "ensure_hash_seed", lambda: order.append("seed"))
    monkeypatch.setattr(sys, "argv", ["metamarl", "check", "config.yaml"])

    with pytest.raises(SystemExit) as exit_info:
        cli.entry()

    assert order == ["seed"]
    assert sys.path[0] == os.getcwd()
    assert exit_info.value.code == 0
    assert "Config OK: local_experiment.Experiment" in capsys.readouterr().out


@pytest.mark.unit
def test_entry_does_not_duplicate_the_working_directory(monkeypatch, config_path):
    monkeypatch.chdir(config_path.parent)
    monkeypatch.setattr(sys, "path", [os.getcwd(), *sys.path])
    monkeypatch.setattr(cli, "ensure_hash_seed", lambda: None)
    monkeypatch.setattr(sys, "argv", ["metamarl", "check", str(config_path)])

    with pytest.raises(SystemExit):
        cli.entry()

    assert sys.path.count(os.getcwd()) == 1


@pytest.mark.integration
@pytest.mark.skipif(
    not CONSOLE_SCRIPT.exists(), reason="metamarl console script not installed"
)
def test_installed_command_restarts_and_checks_the_fishery_config():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["WANDB_MODE"] = "offline"
    result = subprocess.run(
        [str(CONSOLE_SCRIPT), "check", str(FISHERY_CONFIG.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr

    assert "restarting with PYTHONHASHSEED=0" in result.stderr

    assert "Config OK: core.optimizers.bilevel.BilevelConfig" in result.stdout
