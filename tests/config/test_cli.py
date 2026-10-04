"""``core.config.cli``: exit codes and error reporting of ``check`` and ``run``."""

import logging
import sys

import pytest

from core.config import cli

# ``Fraction(1, 0)`` raises ``ZeroDivisionError``: a construction failure with a
# known root cause that the command line must surface.
FAILING_CONFIG = """
experiment:
  _target_: fractions.Fraction
  _args_: [1, 0]
"""

VALID_CONFIG = """
experiment:
  _target_: fractions.Fraction
  _args_: [1, 2]
"""


def _main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["metamarl", *argv])
    return cli.main()


@pytest.mark.unit
def test_check_accepts_a_valid_config(tmp_path, monkeypatch, capsys):
    path = tmp_path / "ok.yaml"
    path.write_text(VALID_CONFIG)

    assert _main(monkeypatch, "check", str(path)) == 0
    assert "Config OK: fractions.Fraction" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize("command", ["check", "run"])
def test_configuration_error_reports_its_root_cause(
    tmp_path, monkeypatch, caplog, command
):
    path = tmp_path / "bad.yaml"
    path.write_text(FAILING_CONFIG)

    with caplog.at_level(logging.ERROR):
        assert _main(monkeypatch, command, str(path)) == 2

    assert "failed constructing 'fractions.Fraction'" in caplog.text
    assert "ZeroDivisionError" in caplog.text
