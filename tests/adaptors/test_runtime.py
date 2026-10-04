"""``RayRuntimeConfig``: environment variables and the ``ray.init`` arguments."""

import pytest
import ray
from ray._private import ray_constants

from core.adaptors.ray.runtime import RayRuntime, RayRuntimeConfig


@pytest.fixture
def captured_init(monkeypatch):
    """Replace ``ray.init`` with a recorder and report Ray as not running."""
    calls = []
    monkeypatch.setattr(ray, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    return calls


@pytest.mark.unit
def test_initialize_disables_uv_run_runtime_env_hook(captured_init, monkeypatch):
    # Under ``uv run`` Ray injects ``working_dir=<cwd>`` into the runtime env,
    # which ``local_mode`` rejects as "not a valid URI".
    monkeypatch.setattr(ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", True)

    RayRuntimeConfig(device="cpu").initialize()

    assert len(captured_init) == 1
    assert captured_init[0]["local_mode"] is True
    assert ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is False


@pytest.mark.unit
def test_ensure_initialized_is_idempotent(captured_init, monkeypatch):
    RayRuntime.ensure_initialized(RayRuntimeConfig())
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    RayRuntime.ensure_initialized(RayRuntimeConfig())

    assert len(captured_init) == 1
