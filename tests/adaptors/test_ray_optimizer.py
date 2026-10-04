"""``RayOptimizer``: validation of the RLlib config and the stop lifecycle."""

from types import SimpleNamespace

import pytest

from core.adaptors.ray import optimizer as optimizer_module
from core.adaptors.ray.optimizer import RayOptimizer


def _config(evaluation_config):
    rllib_cfg = SimpleNamespace(
        evaluation_duration=4, evaluation_config=evaluation_config
    )
    return SimpleNamespace(episodes=1, env=None, rllib_cfg=rllib_cfg)


@pytest.mark.unit
@pytest.mark.parametrize(
    "evaluation_config",
    [None, {}, {"explore": False}],
    ids=["evaluation-not-called", "empty", "no-fragment-length"],
)
def test_missing_evaluation_setup_is_a_clear_error(evaluation_config):
    # The inner loop is scored by evaluation episodes, whose count is derived
    # from ``evaluation_config["rollout_fragment_length"]``.
    with pytest.raises(ValueError, match=r"\.evaluation\(.*rollout_fragment_length"):
        RayOptimizer(_config(evaluation_config))


class RecordingActorMethod:
    """Stand-in for a Ray actor method: ``remote()`` records the call."""

    def __init__(self) -> None:
        self.calls = 0

    def remote(self) -> str:
        self.calls += 1
        return "object-ref"


@pytest.mark.unit
def test_stop_stops_the_policy_actor_and_returns_the_reduced_metrics(monkeypatch):
    monkeypatch.setattr(optimizer_module.ray, "get", lambda ref: ref)
    stop = RecordingActorMethod()
    opt = SimpleNamespace(
        policy_actor=SimpleNamespace(stop=stop),
        logger=SimpleNamespace(reduce=lambda: "reduced"),
    )

    assert RayOptimizer.stop(opt) == "reduced"
    assert stop.calls == 1
