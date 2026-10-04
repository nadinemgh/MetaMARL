"""Fixtures for the ``World`` tests.

``World`` is decorated with ``@ray.remote``. The tests bypass the actor
machinery and instantiate the undecorated class stored in
``World.__ray_metadata__.modified_class``, so no Ray runtime is started and
every method is called directly instead of through ``.remote``.
"""

from __future__ import annotations

from typing import Callable

import pytest

from core.world.base import World
from core.world.context import (
    Context,
    ContextSchema,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)

WORLD_CLASS = World.__ray_metadata__.modified_class


class OtherSchema(ContextSchema):
    """A payload type unrelated to mechanisms and environment steps."""

    value: int = 0


@pytest.fixture
def make_other() -> Callable[..., OtherSchema]:
    """Factory of payloads that are neither mechanisms nor environment steps."""

    def factory(value: int = 0) -> OtherSchema:
        return OtherSchema(value=value)

    return factory


@pytest.fixture
def world():
    """A fresh, empty ``World`` instance (the plain class, not an actor)."""
    return WORLD_CLASS()


@pytest.fixture
def make_mechanism() -> Callable[..., MechanismContext]:
    """Factory of ``MechanismContext`` payloads, published by default."""

    def factory(
        index: int = 0,
        seed: int | None = 0,
        status: MechanismStatus = MechanismStatus.published,
        env_id: str | None = None,
    ) -> MechanismContext:
        return MechanismContext(
            index=index,
            env_id=env_id,
            seed=seed,
            status=status,
            mechanism={"tax": 0.1},
            metrics=None,
        )

    return factory


@pytest.fixture
def make_step() -> Callable[..., EnvStepContext]:
    """Factory of ``EnvStepContext`` payloads whose observation encodes ``step``."""

    def factory(step: int = 0) -> EnvStepContext:
        return EnvStepContext(
            env_id=0,
            seed=0,
            policy_seed=0,
            status=MechanismStatus.train,
            mechanism=0,
            observation=[float(step)],
            observation_map=None,
            reward=1.0,
            action=None,
            info={},
        )

    return factory


@pytest.fixture
def make_context() -> Callable[..., Context]:
    """Factory of runtime ``Context`` wrappers around a payload."""

    def factory(
        payload: ContextSchema,
        opt_id: str | None = "opt",
        step: int = 0,
        ctx_id: str | None = None,
    ) -> Context:
        return Context(id=ctx_id, opt_id=opt_id, step=step, env="env", payload=payload)

    return factory
