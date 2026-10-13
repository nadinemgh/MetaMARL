"""The debug scripts configure logging before the hash-seed check.

``ensure_hash_seed`` logs at INFO that it restarts the process. Before the root
logger is configured only WARNING and above reach the console, so a script that
checked the seed first lost that notice. Each script's ``main`` is run with the
check and ``logging.basicConfig`` replaced by recorders, and stopped by an
invalid option before anything heavy starts.
"""

import importlib
import logging

import pytest

SCRIPTS = (
    "examples.bilevel_fishery.debug",
    "examples.cartpole.debug",
    "examples.fresh_water.debug",
)


@pytest.mark.unit
@pytest.mark.parametrize("module_name", SCRIPTS)
def test_logging_is_configured_before_the_hash_seed_check(monkeypatch, module_name):
    script = importlib.import_module(module_name)
    calls = []
    monkeypatch.setattr(script, "ensure_hash_seed", lambda: calls.append("seed"))
    monkeypatch.setattr(logging, "basicConfig", lambda **_: calls.append("logging"))

    with pytest.raises(SystemExit):
        script.main(["--no-such-option"])

    assert calls == ["logging", "seed"]
