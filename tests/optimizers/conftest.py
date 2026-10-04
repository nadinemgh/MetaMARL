"""Fixtures and fakes shared by the optimizer tests.

Nothing here starts Ray. ``FakeWorld`` stands in for the ``World`` actor and
``ScriptedRegulatorEnv`` stands in for the regulator environment the
evolution strategy steps: it receives the candidates, returns a fitness
computed by a plain Python function, and records every call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import ray
from gymnasium import spaces

from core.agents.base import AgentConfig
from core.mechanism.config import MechanismConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.es.optimizer import ESOptimizer


class FakeWorld:
    """In-memory replacement for the ``World`` Ray actor.

    It exposes the two calls ``OptimizerConfig.build_optimizer`` makes through
    ``<method>.remote(...)`` and records the identifiers it hands out.
    """

    def __init__(self) -> None:
        self.opt_ids: list[str] = []
        self.get_opt_registry = SimpleNamespace(remote=lambda: set(self.opt_ids))
        self._set_new_opt_id = SimpleNamespace(remote=self._register)

    def _register(self, opt_id: str) -> str:
        self.opt_ids.append(opt_id)
        return opt_id


@pytest.fixture
def fake_world(monkeypatch) -> FakeWorld:
    """A ``FakeWorld`` with ``ray.get`` patched to return its argument."""
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)
    return FakeWorld()


def unit_box(size: int = 1) -> spaces.Box:
    """Action space of one mechanism with ``size`` parameters in ``[0, 1]``."""
    return spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)


def regulator_config(mechanism_spaces: dict[str, spaces.Space]) -> AgentConfig:
    """Regulator agent holding one mechanism per entry of ``mechanism_spaces``."""
    return AgentConfig(
        id="regulator",
        policy_id="regulator_policy",
        mechanisms=tuple(
            MechanismConfig(id=mechanism_id, action_space=space)
            for mechanism_id, space in mechanism_spaces.items()
        ),
    )


@pytest.fixture
def es_config_factory() -> Callable[..., ESConfig]:
    """Build an ``ESConfig`` for a regulator with the given mechanisms.

    The default regulator holds one mechanism, ``quota``, with a single
    parameter. Keyword arguments go to ``ESConfig.training``.
    """

    def factory(
        mechanism_spaces: dict[str, spaces.Space] | None = None,
        *,
        seed: int | None = 0,
        episodes: int = 1,
        **training: Any,
    ) -> ESConfig:
        cfg = ESConfig().training(episodes=episodes, **training)
        cfg.agents(regulator_config(mechanism_spaces or {"quota": unit_box()}))

        return cfg.debugging(seed=seed)

    return factory


@pytest.fixture
def es_factory(es_config_factory) -> Callable[..., ESOptimizer]:
    """Build an ``ESOptimizer`` with its population size already set."""

    def factory(
        mechanism_spaces: dict[str, spaces.Space] | None = None,
        *,
        population: int | None = 4,
        seed: int | None = 0,
        episodes: int = 1,
        **training: Any,
    ) -> ESOptimizer:
        cfg = es_config_factory(
            mechanism_spaces, seed=seed, episodes=episodes, **training
        )
        opt = ESOptimizer(cfg)

        if population is not None:
            opt.batch_capacity = population

        return opt

    return factory


class ScriptedRegulatorEnv:
    """Regulator environment whose fitness is a plain function of the candidates.

    Parameters
    ----------
    fitness_fn : callable
        Receives the list of candidate actions (one dict per candidate, keyed
        by mechanism id) and returns one fitness per candidate.
    steps : int
        Number of ``step`` calls before the episode ends.
    end : {"terminated", "truncated"}
        How the last step ends the episode.
    info : dict or None
        Info dictionary returned by every step.
    """

    def __init__(
        self,
        fitness_fn: Callable[[list[dict]], Any],
        *,
        steps: int = 1,
        end: str = "terminated",
        info: dict | None = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.steps = steps
        self.end = end
        self.info = info if info is not None else {}
        self.resets = 0
        self.received: list[list[dict]] = []
        self._taken = 0

    def reset(self):
        self.resets += 1
        self._taken = 0

        return None, {}

    def step(self, actions):
        self.received.append(actions)
        self._taken += 1
        last = self._taken >= self.steps

        return (
            None,
            self.fitness_fn(actions),
            last and self.end == "terminated",
            last and self.end == "truncated",
            self.info,
        )


@pytest.fixture
def scripted_env() -> type[ScriptedRegulatorEnv]:
    """The ``ScriptedRegulatorEnv`` class, for tests that build several."""
    return ScriptedRegulatorEnv


class RecordingReporting:
    """Reporter stand-in recording the metrics it is asked to render."""

    def __init__(self) -> None:
        self.reports: list[Any] = []

    def report(self, metrics) -> None:
        self.reports.append(metrics)


@pytest.fixture
def recording_reporting() -> RecordingReporting:
    return RecordingReporting()
