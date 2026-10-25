"""Tests for PolicyActor without learner-thread synchronization."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from core.adaptors.ray.policy_actor import PolicyActor

# Access the underlying Python class without starting a Ray actor.
ActorClass = PolicyActor.__ray_metadata__.modified_class


def make_actor(algo=None):
    """Construct an actor without invoking its initialization."""

    actor = object.__new__(ActorClass)
    actor.algo = algo if algo is not None else MagicMock()

    return actor


@pytest.mark.unit
def test_train_forwards_to_algorithm():
    actor = make_actor()
    expected = {"training_iteration": 3}
    actor.algo.train.return_value = expected

    assert actor.train() is expected
    actor.algo.train.assert_called_once_with()


@pytest.mark.unit
def test_evaluate_forwards_directly_to_algorithm():
    actor = make_actor()
    expected = {"evaluation": {"episode_return_mean": 12.0}}
    actor.algo.evaluate.return_value = expected

    assert actor.evaluate() is expected
    actor.algo.evaluate.assert_called_once_with()


@pytest.mark.unit
def test_stop_forwards_directly_to_algorithm():
    actor = make_actor()

    actor.stop()
    actor.algo.stop.assert_called_once_with()


@pytest.mark.unit
def test_initialization_builds_algorithm_and_saves_initial_weights():
    weights = {
        "fisher_m0_s11": {"w": np.array([1.0, 2.0])},
        "fisher_m1_s11": {"w": np.array([8.0, 9.0])},
        "fisher_m0_s22": {"w": np.array([3.0, 4.0])},
    }
    algo = MagicMock()
    algo.get_weights.return_value = deepcopy(weights)
    config = SimpleNamespace(build_algo=MagicMock(return_value=algo))
    actor = ActorClass(config)

    config.build_algo.assert_called_once_with()
    algo.set_weights.assert_called_once()

    initialized = algo.set_weights.call_args.args[0]

    # Policies sharing a training seed receive identical initial weights.
    np.testing.assert_array_equal(
        initialized["fisher_m0_s11"]["w"], initialized["fisher_m1_s11"]["w"]
    )

    # Different seeds retain their respective initial weights.
    np.testing.assert_array_equal(
        initialized["fisher_m0_s22"]["w"], weights["fisher_m0_s22"]["w"]
    )

    # Initial weights are retained for subsequent resets.
    np.testing.assert_array_equal(
        actor._init_weights["fisher_m0_s11"]["w"], initialized["fisher_m0_s11"]["w"]
    )


@pytest.mark.unit
def test_reset_rebuilds_algorithm_and_restores_initial_weights():
    initial_weights = {"fisher_m0_s11": {"w": np.array([1.0, 2.0])}}
    first_algo = MagicMock()
    first_algo.get_weights.return_value = deepcopy(initial_weights)
    second_algo = MagicMock()
    second_algo.get_weights.return_value = deepcopy(initial_weights)
    config = SimpleNamespace(
        build_algo=MagicMock(side_effect=[first_algo, second_algo])
    )
    actor = ActorClass(config)
    saved_weights = deepcopy(actor._init_weights)

    with patch("core.adaptors.ray.policy_actor.hash_weights", return_value="test-hash"):
        actor.reset()

    assert config.build_algo.call_count == 2
    assert actor.algo is second_algo
    second_algo.set_weights.assert_called_once()

    restored = second_algo.set_weights.call_args.args[0]

    np.testing.assert_array_equal(
        restored["fisher_m0_s11"]["w"], saved_weights["fisher_m0_s11"]["w"]
    )


@pytest.mark.unit
def test_compute_actions_uses_rlmodule_inference():
    actor = make_actor()
    obs = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    expected = torch.tensor([[0.5], [0.7]], dtype=torch.float32)
    module = MagicMock()
    module.forward_inference.return_value = {"action_dist_inputs": torch.zeros(2, 2)}
    distribution = MagicMock()
    distribution.sample.return_value = expected
    distribution_cls = MagicMock()
    distribution_cls.from_logits.return_value = distribution
    module.get_inference_action_dist_cls.return_value = distribution_cls
    actor.algo.get_module.return_value = module
    actual = actor.compute_actions("fisher_m0_s11", obs)

    np.testing.assert_allclose(actual, expected.numpy())
    actor.algo.get_module.assert_called_once_with("fisher_m0_s11")
    distribution.sample.assert_called_once_with()


@pytest.mark.unit
def test_compute_actions_falls_back_to_legacy_policy():
    actor = make_actor()
    actor.algo.get_module.side_effect = RuntimeError("RLModule inference unavailable")
    policy = MagicMock()
    policy.compute_single_action.side_effect = [
        (np.array([0.2]), [], {}),
        (np.array([0.8]), [], {}),
    ]
    actor.algo.get_policy.return_value = policy
    obs = np.array([[1.0], [2.0]], dtype=np.float32)
    actual = actor.compute_actions("fisher_m0_s11", obs)

    np.testing.assert_allclose(actual, [[0.2], [0.8]])
    assert policy.compute_single_action.call_count == 2

    for call in policy.compute_single_action.call_args_list:
        assert call.kwargs["explore"] is False
