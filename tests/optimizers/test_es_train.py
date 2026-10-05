"""``ESOptimizer.train``: the summary dict consumed by ``BilevelOptimizer.train``."""

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import AgentConfig
from core.mechanism.algorithms.quota import Quota
from core.optimizers.bilevel import BilevelOptimizer
from core.optimizers.es.config import ESConfig
from core.optimizers.es.optimizer import ESOptimizer


class ConstantFitnessEnv:
    """One-step regulator env returning the same fitness for every candidate."""

    def __init__(self) -> None:
        self.steps = 0

    def reset(self):
        return None, {}

    def step(self, actions):
        self.steps += 1
        return None, np.ones(len(actions), dtype=np.float32), True, False, {}


def make_es(
    episodes: int, population: int = 2, low: float = 0.0, high: float = 1.0
) -> ESOptimizer:
    regulator = AgentConfig(
        id="regulator",
        policy_id="quota_policy",
        mechanisms=(
            Quota(
                id="quota",
                action_space=spaces.Box(
                    low=low, high=high, shape=(1,), dtype=np.float32
                ),
                acts_on=("fisherman", "harvest"),
                obs_map={"resource_level": "fish"},
                default=np.asarray(0.5, dtype=np.float32),
            ),
        ),
    )
    cfg = (
        ESConfig()
        .training(episodes=episodes, sigma=0.1, mean_lr=0.1)
        .agents(regulator)
        .debugging(seed=0)
    )
    opt = ESOptimizer(cfg)
    opt.batch_capacity = population
    opt.env = ConstantFitnessEnv()
    return opt


@pytest.mark.unit
@pytest.mark.parametrize("episodes", [0, 1, 3])
def test_train_reports_generations_run_and_convergence(episodes):
    opt = make_es(episodes)

    result = opt.train()

    assert result["episodes"] == episodes == opt.env.steps
    assert result["converged"] is False
    assert {"best_fitness", "best_mechanism", "population_history"} <= set(result)


@pytest.mark.unit
def test_bilevel_train_accepts_the_es_summary():
    es = make_es(episodes=2)
    config = SimpleNamespace(episodes=None, env=None, world_name="w")
    bilevel = BilevelOptimizer(config, outer=es, inner=None, reporter=None)

    result = bilevel.train()

    assert result["episodes"] == 2 and result["converged"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "low, high", [(0.0, 2.0), (0.5, 2.0), (0.2, 1.0)], ids=["high", "both", "low"]
)
def test_action_bounds_outside_unit_interval_are_rejected(low, high):
    # The ES samples in logit space and maps candidates back to (0, 1).
    with pytest.raises(ValueError, match=r"bounds in \[0, 1\]"):
        make_es(episodes=1, low=low, high=high)


@pytest.mark.unit
def test_train_before_population_size_is_set_raises_a_clear_error():
    opt = make_es(episodes=1)
    opt._batch_capacity = None

    with pytest.raises(RuntimeError, match="batch_capacity not set"):
        opt.train()
