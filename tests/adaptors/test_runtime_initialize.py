"""``RayRuntimeConfig.initialize`` and ``RayRuntime.ensure_initialized``.

``ray.init`` is replaced by a recorder, so no Ray runtime starts. The tests
check what ``initialize`` hands to ``ray.init`` (resources, runtime environment,
extra keyword arguments and the fixed local-mode settings), the order of the
environment export and the Ray call, the level of the library loggers it
silences, and the process-wide guard of ``ensure_initialized``.

The existing ``test_runtime.py`` already covers the
``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` override, and ``test_runtime_env_vars.py``
covers the per-device environment variables; neither is repeated here.

Ported from the August ``test_runtime.py``. Its resource test asserted that
``num_gpus`` is absent from the ``ray.init`` arguments when unset, but the code
now always passes ``num_gpus=None``; the rewritten test asserts that.
"""

from __future__ import annotations

import logging
import os

import pytest
import ray
import torch
from ray._private import ray_constants

from core.adaptors.ray.runtime import RayRuntime, RayRuntimeConfig

_MANAGED_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "RLLIB_NUM_GPUS",
    "USE_CUDA",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "RAY_USE_MPS",
    "OMP_NUM_THREADS",
    "RAY_DEBUG",
    "RAY_LOG_TO_STDERR",
    "RAY_BACKEND_LOG_LEVEL",
    "TUNE_DISABLE_AUTO_CALLBACK_LOGGERS",
)
_SILENCED_LOGGERS = ("ray", "ray.rllib", "ray.tune", "tensorboardX", "asyncio")


@pytest.fixture
def isolated_runtime(monkeypatch):
    """Record ``ray.init`` calls and keep the process state of the test local.

    Environment variables, the torch default device, the Ray constant and the
    levels of the silenced loggers are all restored at teardown.
    """

    events: list = []

    for name in _MANAGED_VARS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)

    monkeypatch.setattr(
        torch, "set_default_device", lambda dev: events.append(("device", dev))
    )
    monkeypatch.setattr(ray, "init", lambda **kwargs: events.append(("init", kwargs)))
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", True)
    monkeypatch.setattr(RayRuntime, "_initialized", False)

    levels = {name: logging.getLogger(name).level for name in _SILENCED_LOGGERS}
    yield events

    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def _init_kwargs(events: list) -> dict:
    inits = [payload for kind, payload in events if kind == "init"]
    assert len(inits) == 1
    return inits[0]


@pytest.mark.unit
def test_initialize_forwards_resources_runtime_env_and_extra_kwargs(isolated_runtime):
    cfg = RayRuntimeConfig(
        device="cpu",
        num_cpus=3,
        num_gpus=1,
        logging_level="WARNING",
        runtime_env={"excludes": ["x"]},
        init_kwargs={"namespace": "fishery"},
    )

    cfg.initialize()

    kwargs = _init_kwargs(isolated_runtime)
    assert kwargs["num_cpus"] == 3
    assert kwargs["num_gpus"] == 1
    assert kwargs["logging_level"] == "WARNING"
    assert kwargs["runtime_env"] == {"excludes": ["x"]}
    assert kwargs["namespace"] == "fishery"


@pytest.mark.unit
def test_initialize_without_resources_leaves_them_to_ray(isolated_runtime):
    RayRuntimeConfig(device="cpu").initialize()

    kwargs = _init_kwargs(isolated_runtime)
    assert kwargs["num_cpus"] is None
    assert kwargs["num_gpus"] is None
    assert kwargs["runtime_env"] is None
    assert kwargs["logging_level"] == "ERROR"


@pytest.mark.unit
def test_initialize_fixes_the_local_mode_settings(isolated_runtime):
    RayRuntimeConfig(device="cpu").initialize()

    kwargs = _init_kwargs(isolated_runtime)
    assert kwargs["local_mode"] is True
    assert kwargs["ignore_reinit_error"] is True
    assert kwargs["log_to_driver"] is False
    assert kwargs["include_dashboard"] is False
    assert kwargs["_system_config"] == {
        "metrics_report_interval_ms": 0,
        "enable_metrics_collection": False,
    }


@pytest.mark.unit
def test_initialize_exports_the_environment_before_ray_starts(isolated_runtime):
    # Child processes inherit the environment at fork time, so the device
    # must be set before ``ray.init`` is reached.
    RayRuntimeConfig(device="cpu", num_gpus=2, omp_threads=4).initialize()

    assert [kind for kind, _ in isolated_runtime] == ["device", "init"]
    assert isolated_runtime[0] == ("device", "cpu")

    assert os.environ["RLLIB_NUM_GPUS"] == "2"
    assert os.environ["OMP_NUM_THREADS"] == "4"


@pytest.mark.unit
def test_initialize_silences_the_library_loggers(isolated_runtime):
    for name in _SILENCED_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)

    RayRuntimeConfig(device="cpu").initialize()

    assert logging.getLogger("ray").level == logging.WARNING
    assert logging.getLogger("ray.rllib").level == logging.WARNING
    assert logging.getLogger("ray.tune").level == logging.WARNING
    assert logging.getLogger("tensorboardX").level == logging.ERROR
    assert logging.getLogger("asyncio").level == logging.ERROR


@pytest.mark.unit
def test_ensure_initialized_starts_ray_once_with_the_given_config(isolated_runtime):
    cfg = RayRuntimeConfig(device="cpu", num_cpus=5)

    RayRuntime.ensure_initialized(cfg)

    assert _init_kwargs(isolated_runtime)["num_cpus"] == 5
    assert RayRuntime._initialized is True


@pytest.mark.unit
def test_ensure_initialized_ignores_a_new_config_while_ray_runs(
    isolated_runtime, monkeypatch
):
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    RayRuntime.ensure_initialized(RayRuntimeConfig(device="cpu", num_cpus=9))

    assert isolated_runtime == []
