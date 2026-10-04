"""``ensure_hash_seed``: entry points restart with a fixed string-hash seed."""

import os
import subprocess
import sys

import pytest

from core.config import hash_seed


class ExecCalled(Exception):
    """Raised by the ``os.execv`` stand-in so the test regains control."""


@pytest.fixture
def fake_execv(monkeypatch):
    calls = []

    def execv(path, argv):
        calls.append((path, argv))
        raise ExecCalled

    monkeypatch.setattr(hash_seed.os, "execv", execv)
    return calls


@pytest.mark.unit
def test_unset_seed_restarts_with_the_same_command(monkeypatch, fake_execv):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(sys, "orig_argv", ["python", "-m", "pkg.mod", "run", "x.yaml"])

    with pytest.raises(ExecCalled):
        hash_seed.ensure_hash_seed()

    assert os.environ["PYTHONHASHSEED"] == "0"
    assert fake_execv == [
        (sys.executable, [sys.executable, "-m", "pkg.mod", "run", "x.yaml"])
    ]


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "123", "random"])
def test_a_set_seed_is_respected(monkeypatch, fake_execv, value):
    monkeypatch.setenv("PYTHONHASHSEED", value)

    hash_seed.ensure_hash_seed()

    assert os.environ["PYTHONHASHSEED"] == value
    assert fake_execv == []


@pytest.mark.unit
def test_the_restarted_process_has_a_fixed_hash_order(tmp_path):
    # Two fresh interpreters without PYTHONHASHSEED must agree on the order of
    # a set of strings once ensure_hash_seed has restarted them.
    script = tmp_path / "order.py"
    script.write_text(
        "from core.config.hash_seed import ensure_hash_seed\n"
        + "ensure_hash_seed()\n"
        + "print(list({f'fisherman:{i}' for i in range(10)}))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["PYTHONPATH"] = os.getcwd()

    orders = [
        subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for _ in range(3)
    ]

    assert orders[0] == orders[1] == orders[2]
