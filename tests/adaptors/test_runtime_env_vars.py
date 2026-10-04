"""``RayRuntimeConfig._apply_env_vars``: per-device environment exports.

Each device branch (``cpu``, ``mps``, ``cuda``) sets a different combination
of ``CUDA_VISIBLE_DEVICES``, ``RLLIB_NUM_GPUS``, ``USE_CUDA``,
``PYTORCH_ENABLE_MPS_FALLBACK`` and ``RAY_USE_MPS``; the optional knobs
(``num_gpus``, ``omp_threads``, ``ray_debug``) add or skip further exports.
``torch.set_default_device`` is replaced by a recorder so the tests never
touch the real torch default device, and every variable is cleared from
``os.environ`` beforehand so absence can be asserted.

Copied from the August suite. The ``clean_env`` fixture was changed so that the
variables the code under test exports are removed again at teardown: before,
a variable absent at the start (``CUDA_VISIBLE_DEVICES=""``, ``RLLIB_NUM_GPUS``,
``OMP_NUM_THREADS``) stayed in ``os.environ`` for the rest of the session and
leaked into every later test and child process.
"""

from __future__ import annotations

import os

import pytest
import torch

from core.adaptors.ray.runtime import RayRuntimeConfig

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


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every managed variable and record ``torch.set_default_device`` calls."""
    for name in _MANAGED_VARS:
        # ``setenv`` first, so that teardown removes what the test exports even
        # when the variable was absent to begin with.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    calls: list = []
    monkeypatch.setattr(torch, "set_default_device", lambda dev: calls.append(dev))
    return calls


def _env(name: str) -> str | None:
    return os.environ.get(name)


@pytest.mark.unit
@pytest.mark.parametrize("disable_mps", [True, False])
def test_cpu_branch(clean_env, disable_mps):
    RayRuntimeConfig(device="cpu", disable_mps=disable_mps)._apply_env_vars()

    assert _env("CUDA_VISIBLE_DEVICES") == ""
    assert _env("RLLIB_NUM_GPUS") == "0"
    assert _env("USE_CUDA") == "0"
    assert _env("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    assert _env("RAY_USE_MPS") == ("0" if disable_mps else None)
    assert clean_env == ["cpu"]


@pytest.mark.unit
@pytest.mark.parametrize("disable_mps", [True, False])
@pytest.mark.parametrize("disable_cuda", [True, False])
def test_mps_branch_ignores_disable_flags(clean_env, disable_mps, disable_cuda):
    cfg = RayRuntimeConfig(
        device="mps", disable_mps=disable_mps, disable_cuda=disable_cuda
    )
    cfg._apply_env_vars()

    assert _env("CUDA_VISIBLE_DEVICES") == ""
    assert _env("RLLIB_NUM_GPUS") == "0"
    assert _env("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    assert _env("USE_CUDA") is None
    assert _env("RAY_USE_MPS") is None
    assert clean_env == ["mps"]


@pytest.mark.unit
@pytest.mark.parametrize("disable_cuda", [True, False])
def test_cuda_branch(clean_env, disable_cuda):
    RayRuntimeConfig(device="cuda", disable_cuda=disable_cuda)._apply_env_vars()

    assert _env("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    assert _env("CUDA_VISIBLE_DEVICES") == ("" if disable_cuda else None)
    assert _env("RLLIB_NUM_GPUS") is None
    assert _env("USE_CUDA") is None
    assert clean_env == ["cuda"]


@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("num_gpus", [None, 0, 2])
def test_num_gpus_overrides_rllib_num_gpus(clean_env, device, num_gpus):
    RayRuntimeConfig(device=device, num_gpus=num_gpus)._apply_env_vars()

    if num_gpus is None:
        expected = "0" if device in ("cpu", "mps") else None
    else:
        expected = str(num_gpus)
    assert _env("RLLIB_NUM_GPUS") == expected


@pytest.mark.unit
@pytest.mark.parametrize("omp_threads", [None, 1, 8])
def test_omp_threads_export(clean_env, omp_threads):
    RayRuntimeConfig(omp_threads=omp_threads)._apply_env_vars()
    assert _env("OMP_NUM_THREADS") == (
        None if omp_threads is None else str(omp_threads)
    )


@pytest.mark.unit
@pytest.mark.parametrize("ray_debug", [True, False])
def test_ray_debug_export(clean_env, ray_debug):
    RayRuntimeConfig(ray_debug=ray_debug)._apply_env_vars()
    assert _env("RAY_DEBUG") == ("1" if ray_debug else None)


@pytest.mark.unit
def test_logging_variables_always_set(clean_env):
    RayRuntimeConfig()._apply_env_vars()
    assert _env("RAY_LOG_TO_STDERR") == "0"
    assert _env("RAY_BACKEND_LOG_LEVEL") == "error"
    assert _env("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS") == "1"


@pytest.mark.unit
def test_unknown_device_sets_only_generic_variables(clean_env):
    RayRuntimeConfig(device="tpu")._apply_env_vars()  # type: ignore[arg-type]
    assert _env("CUDA_VISIBLE_DEVICES") is None
    assert _env("PYTORCH_ENABLE_MPS_FALLBACK") is None
    assert _env("OMP_NUM_THREADS") == "1"
    assert clean_env == ["tpu"]
