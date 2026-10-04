"""The evolution strategy drives a regulator of ``Quota`` mechanisms to an optimum.

Ported from the August ``test_es_regulator_loop``, which ran the optimizer
against a ``RegulatorEnv`` and a real ``World`` actor. The regulator
environment is now replaced by a scripted one whose fitness is a closed form,
so the test needs no Ray runtime and keeps what it was written to show: the
real ``ESOptimizer``, built from the real ``ESConfig`` and the real mechanism
configurations, moves its search mean towards the maximum of a known landscape.
"""

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import AgentConfig
from core.mechanism.algorithms.quota import Quota
from core.optimizers.es.config import ESConfig
from core.optimizers.es.optimizer import ESOptimizer

OPTIMUM = np.array([0.8, 0.2, 0.6], dtype=np.float32)
MECHANISM_IDS = ("quota_0", "quota_1", "quota_2")
GENERATIONS = 30
POPULATION = 16


class AnalyticRegulatorEnv:
    """One-step regulator environment with fitness ``-||x - OPTIMUM||**2``."""

    def __init__(self) -> None:
        self.fitness_calls: list[np.ndarray] = []

    def reset(self):
        return None, {}

    def step(self, actions):
        candidates = np.array(
            [[a[mid][0] for mid in MECHANISM_IDS] for a in actions], dtype=np.float32
        )
        fitness = -np.sum((candidates - OPTIMUM) ** 2, axis=1)
        self.fitness_calls.append(fitness)
        return None, fitness, True, False, {}


def make_optimizer() -> ESOptimizer:
    regulator = AgentConfig(
        id="regulator",
        policy_id="quota_policy",
        mechanisms=tuple(
            Quota(
                id=mid,
                action_space=spaces.Box(0.0, 1.0, (1,), np.float32),
                acts_on=("fisherman", "harvest"),
                obs_map={"resource_level": "fish"},
            )
            for mid in MECHANISM_IDS
        ),
    )
    cfg = (
        ESConfig()
        .training(
            episodes=GENERATIONS,
            sigma=0.2,
            mean_lr=0.2,
            sigma_lr=0.0,
            break_symmetry=True,
        )
        .agents(regulator)
        .debugging(seed=0)
    )
    optimizer = ESOptimizer(cfg)
    optimizer.batch_capacity = POPULATION
    optimizer.env = AnalyticRegulatorEnv()
    return optimizer


@pytest.mark.unit
def test_the_search_mean_moves_towards_the_optimum():
    optimizer = make_optimizer()
    initial_distance = np.linalg.norm(optimizer.mean - OPTIMUM)

    result = optimizer.train()

    final_distance = np.linalg.norm(optimizer.mean - OPTIMUM)
    assert result["episodes"] == GENERATIONS
    assert final_distance < 0.5 * initial_distance


@pytest.mark.unit
def test_every_generation_evaluates_the_whole_population():
    optimizer = make_optimizer()

    optimizer.train()

    assert len(optimizer.env.fitness_calls) == GENERATIONS
    assert len(optimizer.population_history) == GENERATIONS
    for population, fitness in optimizer.population_history:
        assert population.shape == (POPULATION, 3)
        assert fitness.shape == (POPULATION,)


@pytest.mark.unit
def test_the_best_candidate_found_beats_the_first_generation():
    optimizer = make_optimizer()

    result = optimizer.train()

    first_generation = optimizer.population_history[0][1]
    assert result["best_fitness"] >= first_generation.max()
    assert result["best_fitness"] > -0.05
    assert np.linalg.norm(result["best_mechanism"] - OPTIMUM) < 0.25
