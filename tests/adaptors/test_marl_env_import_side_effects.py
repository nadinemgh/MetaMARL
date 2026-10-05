"""Importing the Ray adapter modules leaves the root logger alone.

Entry points (``core.config.cli`` and the example scripts) configure logging;
a library module that called ``logging.basicConfig`` at import would install a
handler and a level on the root logger of every program that imports it.
"""

import subprocess
import sys

import pytest

PROBE = """
import logging
import {module}
root = logging.getLogger()
print(len(root.handlers), root.level)
"""


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "core.adaptors.ray.marl_env",
        "core.adaptors.ray.optimizer",
        "core.adaptors.ray.policy_actor",
    ],
)
def test_importing_an_adapter_module_does_not_configure_logging(module):
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module)],
        capture_output=True,
        text=True,
        check=True,
    )

    # An untouched root logger has no handler and the WARNING level (30).
    assert result.stdout.split()[-2:] == ["0", "30"]
