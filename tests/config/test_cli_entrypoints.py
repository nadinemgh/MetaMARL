"""``core.config.cli``: the ``run`` success path, interrupts and the script entry.

``test_cli.py`` covers ``check`` on a valid config and the exit code 2 of a
configuration error. This module adds the other exits of ``main`` (``run``
success, Ctrl-C), the ``python -m core.config.cli`` entry point and the
``entry`` function behind the ``metamarl`` console script with the hash-seed
restart replaced by a recorder, argument errors, and the real fishery
configuration through the ``check`` path only: it builds the ``BilevelConfig``
and starts neither Ray nor a training run. One integration test runs the
installed ``metamarl`` command in a child process, restart included.
"""

import logging
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from core.config import cli, hash_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
FISHERY_CONFIG = REPO_ROOT / "examples" / "bilevel_fishery" / "config.yaml"
CONSOLE_SCRIPT = Path(sys.executable).parent / "metamarl"

VALID_CONFIG = """
experiment:
  _target_: fractions.Fraction
  _args_: [6, 4]
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
    monkeypatch.setattr(cli, "run_experiment", lambda path: results.append(path))

    assert _main(monkeypatch, "run", str(config_path)) == 0
    assert results == [str(config_path)]


@pytest.mark.unit
def test_run_with_the_real_loader_succeeds_on_a_valid_config(monkeypatch, config_path):
    assert _main(monkeypatch, "run", str(config_path)) == 0


@pytest.mark.unit
def test_check_does_not_execute_the_run_section(monkeypatch, config_path):
    def fail(path):
        raise AssertionError("check must not run the experiment")

    monkeypatch.setattr(cli, "run_experiment", fail)

    assert _main(monkeypatch, "check", str(config_path)) == 0


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
    def interrupt(path):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "load_experiment", interrupt)
    monkeypatch.setattr(cli, "run_experiment", interrupt)

    assert _main(monkeypatch, command, str(config_path)) == 130


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv", [[], ["train", "x.yaml"], ["check"], ["run", "a.yaml", "b.yaml"]]
)
def test_invalid_arguments_exit_with_the_argparse_code(monkeypatch, argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _main(monkeypatch, *argv)

    assert exit_info.value.code == 2
    assert "usage: metamarl" in capsys.readouterr().err


@pytest.mark.unit
def test_script_entry_restarts_first_then_exits_with_the_command_status(
    monkeypatch, config_path, capsys
):
    order = []
    monkeypatch.setattr(hash_seed, "ensure_hash_seed", lambda: order.append("seed"))
    monkeypatch.setattr(sys, "argv", ["metamarl", "check", str(config_path)])

    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("core.config.cli", run_name="__main__")

    assert order == ["seed"]
    assert exit_info.value.code == 0
    assert "Config OK: fractions.Fraction" in capsys.readouterr().out


@pytest.mark.unit
def test_script_entry_exits_with_two_on_a_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setattr(hash_seed, "ensure_hash_seed", lambda: None)
    monkeypatch.setattr(sys, "argv", ["metamarl", "check", str(tmp_path / "no.yaml")])

    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("core.config.cli", run_name="__main__")

    assert exit_info.value.code == 2


@pytest.mark.unit
def test_fishery_configuration_passes_the_check(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "offline")

    assert _main(monkeypatch, "check", str(FISHERY_CONFIG)) == 0
    assert "Config OK: core.optimizers.bilevel.BilevelConfig" in capsys.readouterr().out


LOCAL_MODULE_CONFIG = """
experiment:
  _target_: local_experiment.Experiment
"""


@pytest.mark.unit
def test_entry_resolves_modules_of_the_working_directory(monkeypatch, tmp_path, capsys):
    # A console script starts with its bin directory, not the working
    # directory, first on sys.path; entry must add the latter as python -m does.
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
def test_installed_command_restarts_and_checks_the_fishery_config():
    # The real console script, run from the repository root with the hash seed
    # unset: it must restart once and then resolve the example's modules.
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
