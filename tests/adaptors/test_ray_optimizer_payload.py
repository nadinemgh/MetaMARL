"""``RayOptimizer._to_logger_payload``: RLlib results become ``RaySchema`` payloads.

The result builders themselves (``build_rollout``, ``build_learner`` ...) are
tested in ``test_ray_utils_builders.py``. Here the unit is the method that
assembles them into the train and eval branches, and the hand-over of those
payloads to a ``MetricLogger``. The method does not read ``self``, so it is
called through the class with ``None`` as instance, which keeps the tests free
of any actor or config.
"""

from __future__ import annotations

import pytest

from core.adaptors.ray.optimizer import RayOptimizer
from core.adaptors.ray.schema import RaySchema
from core.envs.schema import EpisodeRolloutSchema
from core.metrics.logger import MetricLogger


def episode(mechanism_id, seed, reward):
    return EpisodeRolloutSchema(
        mechanism_id=mechanism_id, seed=seed, reward_mean=reward
    )


def result_dict(*, return_mean=1.5, steps=80, evaluation=None):
    """New-API-stack ``train()`` result with two modules and three episodes."""
    result = {
        "env_runners": {
            "episode_return_mean": return_mean,
            "episode_return_min": 1.0,
            "episode_return_max": 2.0,
            "num_env_steps_sampled": steps,
            "num_env_steps_sampled_lifetime": 2 * steps,
            "by_episode": {
                "e1": episode(0, 100, 1.0),
                "e2": episode(1, 100, 3.0),
                "e3": episode(1, 200, 4.0),
            },
        },
        "timers": {"training_iteration": 1.0},
        "learners": {
            "__all_modules__": {"learner_thread_in_queue_wait_timer": 0.1},
            "fisher_m0_s100": {"policy_loss": 0.2, "total_loss": 0.5},
            "fisher_m1_s100": {"policy_loss": 0.4, "total_loss": 0.7},
        },
    }

    if evaluation is not None:
        result["evaluation"] = evaluation

    return result


@pytest.mark.unit
def test_training_result_fills_every_train_block_and_leaves_eval_empty():
    payload = RayOptimizer._to_logger_payload(None, result_dict())

    assert isinstance(payload, RaySchema)
    assert payload.eval is None
    assert payload.train.rollout.aggregate.reward_mean == 1.5
    assert set(payload.train.rollout.by_mechanism) == {"0", "1"}
    assert set(payload.train.rollout.by_mechanism["1"].by_seed) == {"100", "200"}
    by_mechanism = payload.train.learner.by_mechanism
    assert by_mechanism["0"].by_seed["100"].by_policy["fisher"].policy_loss == 0.2
    assert by_mechanism["1"].by_seed["100"].by_policy["fisher"].total_loss == 0.7
    assert payload.train.performance.env_steps_this_iter == 80.0
    assert payload.train.performance.training_iteration_s == 1.0


@pytest.mark.unit
def test_evaluation_block_nested_in_a_training_result_fills_the_eval_branch():
    nested = {"env_runners": {"episode_return_mean": 9.0, "num_env_steps_sampled": 12}}

    payload = RayOptimizer._to_logger_payload(None, result_dict(evaluation=nested))

    assert payload.train.rollout.aggregate.reward_mean == 1.5
    assert payload.eval.rollout.aggregate.reward_mean == 9.0
    assert payload.eval.performance.env_steps_this_iter == 12.0
    assert payload.eval.rollout.by_mechanism == {}


@pytest.mark.unit
@pytest.mark.parametrize("evaluation", ["pending", None, 3.0, ["x"]])
def test_non_dict_evaluation_entry_is_ignored(evaluation):
    payload = RayOptimizer._to_logger_payload(None, result_dict(evaluation=evaluation))

    assert payload.eval is None
    assert payload.train.rollout.aggregate.reward_mean == 1.5


@pytest.mark.unit
def test_eval_only_payload_has_no_train_branch_and_no_learner_block():
    payload = RayOptimizer._to_logger_payload(None, result_dict(), is_eval=True)

    assert payload.train is None
    assert payload.eval.rollout.aggregate.reward_min == 1.0
    assert payload.eval.performance.env_steps_this_iter == 80.0
    assert set(payload.eval.rollout.by_mechanism) == {"0", "1"}
    assert not hasattr(payload.eval, "learner")


@pytest.mark.unit
def test_payloads_accumulate_in_the_logger_and_reduce_to_the_last_iteration():
    logger = MetricLogger.from_schema(RaySchema)
    payloads = [
        RayOptimizer._to_logger_payload(
            None, result_dict(return_mean=mean, steps=steps)
        )
        for mean, steps in [(1.5, 80), (2.5, 90)]
    ]

    for iteration, payload in enumerate(payloads, start=1):
        logger.push(("iter",), iteration)
        logger.push_data(payload)

    peeked = logger.peek()
    assert peeked.iter == [1, 2]
    assert peeked.train.rollout.aggregate.reward_mean == [1.5, 2.5]
    assert peeked.train.rollout.by_mechanism["1"].by_seed["200"].by_episode[
        "e3"
    ].reward_mean == [4.0, 4.0]

    reduced = logger.reduce()
    assert reduced.iter == 2
    assert reduced.train.rollout.aggregate.reward_mean == 2.5
    assert reduced.train.performance.env_steps_this_iter == 90.0
