"""``RegulatorEnv.step``: one published context per (candidate, policy seed).

Every inner environment fetches its mechanism from the ``World`` by candidate
index and policy seed, so a candidate published for only some seeds leaves the
other seeds' fishermen training without it.
"""

from types import SimpleNamespace

import pytest

from core.envs import regulator
from core.envs.regulator import RegulatorEnv
from core.world.context import MechanismStatus


class RecordingWorld:
    """World stand-in that keeps every appended context."""

    def __init__(self) -> None:
        self.contexts = []
        self.append_context = SimpleNamespace(remote=self.contexts.append)


def make_env(world: RecordingWorld, seeds: list[int]) -> RegulatorEnv:
    """``RegulatorEnv`` with only the attributes ``step`` touches."""
    env = RegulatorEnv.__new__(RegulatorEnv)
    env.world = world
    env._opt_id = None
    env._t = 0
    env.horizon = None
    env.seeds = seeds
    env.logger = None
    env.inner = SimpleNamespace(train=lambda: {}, reduce_metrics=lambda: {})
    return env


@pytest.fixture(autouse=True)
def ray_get_is_identity(monkeypatch):
    monkeypatch.setattr(regulator.ray, "get", lambda ref: ref)


@pytest.mark.unit
def test_every_candidate_is_published_for_every_seed():
    world = RecordingWorld()
    env = make_env(world, seeds=[11, 22])

    env.step([{"quota": 0.3}, {"quota": 0.7}])

    published = sorted(
        (c.payload.index, c.payload.seed, c.payload.mechanism["quota"])
        for c in world.contexts
    )
    assert published == [(0, 11, 0.3), (0, 22, 0.3), (1, 11, 0.7), (1, 22, 0.7)]
    assert {c.payload.status for c in world.contexts} == {MechanismStatus.published}
