"""``PolicyActor``: forwarding to the owned ``Algorithm`` without a Ray runtime.

The actor is a ``@ray.remote`` class; its plain Python class is reached through
``PolicyActor.__ray_metadata__.modified_class`` and instantiated here with a
mocked ``AlgorithmConfig`` whose ``build_algo`` returns mocked ``Algorithm``
objects. Nothing is trained and Ray is never started. The learner-thread wait
that ``evaluate`` performs has its own tests in ``test_learner_drain.py``; here
only the way ``evaluate`` calls it and reacts to its outcome is checked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.adaptors.ray.learner_drain import stop_learner_thread, wait_for_learner_thread
from core.adaptors.ray.policy_actor import PolicyActor
from core.adaptors.ray.utils import hash_weights

# The plain Python class behind the ``@ray.remote`` decorator.
PolicyActorClass = PolicyActor.__ray_metadata__.modified_class

INIT_WEIGHTS = {"fisher_m0_s1": {"layer": np.arange(3, dtype=np.float32)}}


def make_actor(*algos: MagicMock):
    """Build an actor whose config hands out ``algos`` one per ``build_algo`` call.

    Returns ``(actor, algo_config, algos)``. With no ``algos`` given, a single
    mocked algorithm that reports ``INIT_WEIGHTS`` is used.
    """
    if not algos:
        algo = MagicMock(name="algo")
        algo.get_weights.return_value = INIT_WEIGHTS
        algos = (algo,)

    algo_config = MagicMock(name="algo_config")
    algo_config.build_algo.side_effect = list(algos)

    return PolicyActorClass(algo_config), algo_config, algos


@pytest.mark.unit
def test_init_builds_the_algorithm_inside_the_actor_and_keeps_initial_weights():
    actor, algo_config, (algo,) = make_actor()

    algo_config.build_algo.assert_called_once_with()
    assert actor.algo is algo
    assert actor.algo_config is algo_config
    assert actor._init_weights is INIT_WEIGHTS


@pytest.mark.unit
def test_train_returns_the_raw_rllib_result():
    actor, _, (algo,) = make_actor()
    algo.train.return_value = {"training_iteration": 1}

    assert actor.train() == {"training_iteration": 1}
    algo.train.assert_called_once_with()


@pytest.mark.unit
def test_evaluate_waits_for_every_learner_before_evaluating():
    actor, _, (algo,) = make_actor()
    order = MagicMock()
    ok_results = [SimpleNamespace(ok=True), SimpleNamespace(ok=True)]
    order.attach_mock(algo.learner_group.foreach_learner, "foreach_learner")
    order.attach_mock(algo.evaluate, "evaluate")
    algo.learner_group.foreach_learner.return_value = ok_results
    algo.evaluate.return_value = {"evaluation": "done"}

    assert actor.evaluate() == {"evaluation": "done"}

    # The wait is a function shipped to every learner, and it comes first.
    assert [call[0] for call in order.mock_calls] == ["foreach_learner", "evaluate"]
    algo.learner_group.foreach_learner.assert_called_once_with(wait_for_learner_thread)


@pytest.mark.unit
def test_evaluate_reraises_a_learner_failure_and_skips_the_evaluation():
    actor, _, (algo,) = make_actor()
    failure = RuntimeError("learner thread died")
    failed = MagicMock(ok=False)
    failed.get.side_effect = failure
    algo.learner_group.foreach_learner.return_value = [SimpleNamespace(ok=True), failed]

    with pytest.raises(RuntimeError, match="learner thread died"):
        actor.evaluate()

    algo.evaluate.assert_not_called()


@pytest.mark.unit
def test_compute_actions_samples_from_the_rl_module_distribution():
    actor, _, (algo,) = make_actor()
    obs_batch = np.ones((2, 4), dtype=np.float32)
    expected = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    module = algo.get_module.return_value
    module.forward_inference.return_value = {"action_dist_inputs": "logits"}
    dist_cls = module.get_inference_action_dist_cls.return_value
    sampled = dist_cls.from_logits.return_value.sample.return_value
    sampled.cpu.return_value.numpy.return_value = expected

    actions = actor.compute_actions("fisher_m0_s1", obs_batch)

    np.testing.assert_array_equal(actions, expected)
    algo.get_module.assert_called_once_with("fisher_m0_s1")
    module.forward_inference.assert_called_once_with({"obs": obs_batch})
    dist_cls.from_logits.assert_called_once_with("logits")
    algo.get_policy.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "failing_step",
    ["get_module", "forward_inference", "sample"],
    ids=lambda step: f"{step}-fails",
)
def test_compute_actions_falls_back_to_the_policy_api_without_exploration(failing_step):
    actor, _, (algo,) = make_actor()
    module = algo.get_module.return_value
    dist = module.get_inference_action_dist_cls.return_value.from_logits.return_value

    failure = AttributeError("not on this API stack")
    if failing_step == "get_module":
        algo.get_module.side_effect = failure
    elif failing_step == "forward_inference":
        module.forward_inference.side_effect = failure
    else:
        dist.sample.side_effect = failure

    policy = algo.get_policy.return_value
    policy.compute_single_action.side_effect = [
        (np.array([1.0]), None, {}),
        (np.array([2.0]), None, {}),
    ]
    obs_batch = np.zeros((2, 3), dtype=np.float32)

    actions = actor.compute_actions("fisher_m0_s1", obs_batch)

    np.testing.assert_array_equal(actions, np.array([[1.0], [2.0]]))
    algo.get_policy.assert_called_once_with("fisher_m0_s1")
    assert policy.compute_single_action.call_count == 2
    for call in policy.compute_single_action.call_args_list:
        assert call.kwargs == {"explore": False}


@pytest.mark.unit
def test_reset_stops_the_previous_algorithm_then_replaces_it(caplog):
    first, second = MagicMock(name="first"), MagicMock(name="second")
    first.get_weights.return_value = INIT_WEIGHTS
    second.get_weights.return_value = INIT_WEIGHTS
    actor, algo_config, _ = make_actor(first, second)
    order = MagicMock()
    order.attach_mock(first.stop, "stop_first")
    order.attach_mock(algo_config.build_algo, "build_algo")

    with caplog.at_level("INFO", logger="core.adaptors.ray.policy_actor"):
        actor.reset()

    assert algo_config.build_algo.call_count == 2
    assert actor.algo is second
    second.set_weights.assert_called_once_with(INIT_WEIGHTS)
    first.set_weights.assert_not_called()
    assert f"Initial policy weight hash: {hash_weights(INIT_WEIGHTS)}" in caplog.text
    # The previous algorithm releases its workers before the new one asks for
    # the same resources.
    assert [call[0] for call in order.mock_calls] == ["stop_first", "build_algo"]
    first.stop.assert_called_once_with()
    second.stop.assert_not_called()


@pytest.mark.unit
def test_reset_ends_the_learner_threads_of_the_previous_algorithm():
    first, second = MagicMock(name="first"), MagicMock(name="second")
    first.get_weights.return_value = INIT_WEIGHTS
    second.get_weights.return_value = INIT_WEIGHTS
    actor, _, _ = make_actor(first, second)

    actor.reset()

    first.learner_group.foreach_learner.assert_called_once_with(stop_learner_thread)
    second.learner_group.foreach_learner.assert_not_called()


@pytest.mark.unit
def test_stop_ends_the_learner_threads_then_stops_the_algorithm():
    actor, _, (algo,) = make_actor()
    order = MagicMock()
    algo.learner_group.foreach_learner.return_value = [SimpleNamespace(ok=True)]
    order.attach_mock(algo.learner_group.foreach_learner, "foreach_learner")
    order.attach_mock(algo.stop, "stop")

    actor.stop()

    assert [call[0] for call in order.mock_calls] == ["foreach_learner", "stop"]
    algo.learner_group.foreach_learner.assert_called_once_with(stop_learner_thread)
    algo.stop.assert_called_once_with()


@pytest.mark.unit
def test_stop_still_stops_the_algorithm_when_a_learner_thread_cannot_be_ended():
    actor, _, (algo,) = make_actor()
    failed = MagicMock(ok=False)
    failed.get.side_effect = TimeoutError("learner thread did not stop")
    algo.learner_group.foreach_learner.return_value = [failed]

    with pytest.raises(TimeoutError, match="did not stop"):
        actor.stop()

    algo.stop.assert_called_once_with()


@pytest.mark.unit
def test_reset_builds_no_new_algorithm_when_the_previous_one_cannot_be_ended():
    first, second = MagicMock(name="first"), MagicMock(name="second")
    first.get_weights.return_value = INIT_WEIGHTS
    failed = MagicMock(ok=False)
    failed.get.side_effect = TimeoutError("learner thread did not stop")
    first.learner_group.foreach_learner.return_value = [failed]
    actor, algo_config, _ = make_actor(first, second)

    with pytest.raises(TimeoutError):
        actor.reset()

    first.stop.assert_called_once_with()
    assert algo_config.build_algo.call_count == 1
