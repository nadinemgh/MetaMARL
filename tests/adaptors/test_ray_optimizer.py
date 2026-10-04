"""``RayOptimizer``: construction-time validation of the RLlib config."""

from types import SimpleNamespace

import pytest

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
