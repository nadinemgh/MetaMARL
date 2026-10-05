"""``core.adaptors.ray.utils``: scalar getters, weight hashing and learner IDs.

The getters are fed hand-built RLlib result dictionaries in both the new API
stack layout (``env_runners/...``, ``learners/<module>/...``) and the classic
one (top-level ``episode_reward_mean``, ``timesteps_total``,
``info/learner/...``). ``hash_weights`` is checked for determinism, key-order
independence and sensitivity to values across tensors, arrays and plain
leaves. No Ray runtime is involved.

Ported from the August suite. The August test that expected the new-stack
``learners/<module>/policy_loss`` entry to be ignored was rewritten, because
``get_policy_loss_if_present`` reads that layout now.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from core.adaptors.ray.utils import (
    _get_env,
    get_env_steps,
    get_episode_return_mean,
    get_policy_loss_if_present,
    hash_weights,
    parse_learner_id,
)

# ---------------------------------------------------------------------------
# _get_env
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "result, expected",
    [
        ({}, {}),
        ({"env_runners": None}, {}),
        ({"env_runners": {}}, {}),
        ({"env_runners": {"a": 1}}, {"a": 1}),
    ],
)
def test_get_env(result, expected):
    assert _get_env(result) == expected


# ---------------------------------------------------------------------------
# Mean episode return
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_episode_return_prefers_new_stack_key():
    result = {
        "env_runners": {"episode_return_mean": np.float32(1.5)},
        "episode_reward_mean": 99.0,
    }
    assert get_episode_return_mean(result) == 1.5


@pytest.mark.unit
def test_episode_return_new_stack_zero_is_kept():
    # A genuine zero return must not fall through to the legacy key.
    result = {"env_runners": {"episode_return_mean": 0.0}, "episode_reward_mean": 9.0}
    assert get_episode_return_mean(result) == 0.0


@pytest.mark.unit
def test_episode_return_legacy_top_level():
    assert get_episode_return_mean({"episode_reward_mean": 2.0}) == 2.0


@pytest.mark.unit
def test_episode_return_legacy_top_level_zero_is_kept():
    # A genuine zero at the top level must not fall through to the nested key.
    result = {"episode_reward_mean": 0.0, "env_runners": {"episode_reward_mean": 9.0}}
    assert get_episode_return_mean(result) == 0.0


@pytest.mark.unit
def test_episode_return_legacy_inside_env_runners():
    assert get_episode_return_mean({"env_runners": {"episode_reward_mean": 3}}) == 3.0


@pytest.mark.unit
def test_episode_return_defaults_to_zero():
    assert get_episode_return_mean({}) == 0.0
    result = {"env_runners": {"episode_return_mean": "not a number"}}
    assert get_episode_return_mean(result) == 0.0


@pytest.mark.unit
def test_episode_return_negative_value_is_kept():
    assert get_episode_return_mean({"episode_reward_mean": -3.5}) == -3.5


# ---------------------------------------------------------------------------
# get_env_steps
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_env_steps_new_stack():
    result = {
        "env_runners": {
            "num_env_steps_sampled": 200,
            "num_env_steps_sampled_lifetime": np.int64(4000),
        }
    }
    assert get_env_steps(result) == (200, 4000)


@pytest.mark.unit
def test_env_steps_legacy_fallback():
    result = {"timesteps_this_iter": 10.0, "timesteps_total": "30"}
    assert get_env_steps(result) == (10, 30)


@pytest.mark.unit
def test_env_steps_mixed_and_missing():
    result = {"env_runners": {"num_env_steps_sampled": 5}, "timesteps_total": 50}
    assert get_env_steps(result) == (5, 50)
    assert get_env_steps({}) == (0, 0)
    assert get_env_steps({"env_runners": {"num_env_steps_sampled": "x"}}) == (0, 0)


@pytest.mark.unit
def test_env_steps_new_stack_zero_is_kept():
    # Zero steps sampled is a measurement; the legacy counters must not
    # replace it.
    result = {
        "env_runners": {
            "num_env_steps_sampled": 0,
            "num_env_steps_sampled_lifetime": 0,
        },
        "timesteps_this_iter": 8,
        "timesteps_total": 80,
    }
    assert get_env_steps(result) == (0, 0)


@pytest.mark.unit
def test_env_steps_truncates_floats():
    result = {"timesteps_this_iter": 7.9, "timesteps_total": 12.1}
    assert get_env_steps(result) == (7, 12)


# ---------------------------------------------------------------------------
# get_policy_loss_if_present
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_policy_loss_new_stack_averages_modules_and_skips_the_aggregate():
    result = {
        "learners": {
            "p0_m0_s1": {"policy_loss": 1.0},
            "p1_m0_s1": {"policy_loss": np.float32(3.0)},
            "__all_modules__": {"policy_loss": 100.0},
        }
    }
    assert get_policy_loss_if_present(result) == 2.0


@pytest.mark.unit
def test_policy_loss_new_stack_skips_non_finite_and_malformed_modules():
    # IMPALA/APPO learners reduce their stats only every 20 updates; the
    # iterations in between hold NaN, which must not poison the mean.
    result = {
        "learners": {
            "a": {"policy_loss": float("nan")},
            "b": {"policy_loss": float("inf")},
            "c": {"policy_loss": 0.5},
            "d": {"other": 1.0},
            "e": "not a dict",
        }
    }
    assert get_policy_loss_if_present(result) == 0.5


@pytest.mark.unit
def test_policy_loss_new_stack_wins_over_classic_layout():
    result = {
        "learners": {"m": {"policy_loss": 0.25}},
        "info": {"learner": {"p": {"learner_stats": {"policy_loss": 9.0}}}},
    }
    assert get_policy_loss_if_present(result) == 0.25


@pytest.mark.unit
def test_policy_loss_all_nan_new_stack_falls_back_to_classic_layout():
    result = {
        "learners": {"m": {"policy_loss": float("nan")}},
        "info": {"learner": {"p": {"learner_stats": {"policy_loss": 4.0}}}},
    }
    assert get_policy_loss_if_present(result) == 4.0


@pytest.mark.unit
def test_policy_loss_classic_layout_averages_across_policies():
    result = {
        "info": {
            "learner": {
                "p0": {"learner_stats": {"policy_loss": 1.0}},
                "p1": {"learner_stats": {"policy_loss": np.float32(3.0)}},
                "p2": {"learner_stats": {}},
                "p3": None,
                "p4": {"learner_stats": None},
                "p5": {"learner_stats": {"policy_loss": "bad"}},
            }
        }
    }
    assert get_policy_loss_if_present(result) == 2.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "result",
    [
        {},
        {"info": None},
        {"info": {}},
        {"info": {"learner": None}},
        {"info": {"learner": {}}},
        {"info": {"learner": ["not", "a", "dict"]}},
        {"info": {"learner": {"p0": {"learner_stats": {"other": 1}}}}},
        {"learners": {"__all_modules__": {"policy_loss": 0.5}}},
        {"learners": {"m0": {"policy_loss": float("nan")}}},
    ],
)
def test_policy_loss_nan_when_absent(result):
    assert math.isnan(get_policy_loss_if_present(result))


# ---------------------------------------------------------------------------
# hash_weights
# ---------------------------------------------------------------------------


def _weights(scale: float = 1.0) -> dict:
    return {
        "policy": {
            "fc.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) * scale,
            "fc.bias": np.array([0.1, 0.2]) * scale,
        },
        "meta": {"lr": 0.01 * scale, "name": "adam"},
    }


@pytest.mark.unit
def test_hash_weights_is_deterministic_hex_sha256():
    h1 = hash_weights(_weights())
    h2 = hash_weights(_weights())
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)


@pytest.mark.unit
def test_hash_weights_independent_of_dict_insertion_order():
    a = {"x": np.ones(2), "y": np.zeros(2)}
    b = {"y": np.zeros(2), "x": np.ones(2)}
    assert hash_weights(a) == hash_weights(b)


@pytest.mark.unit
def test_hash_weights_changes_with_values_and_keys():
    base = hash_weights(_weights())
    assert hash_weights(_weights(scale=2.0)) != base
    renamed = _weights()
    renamed["policy"]["fc.w"] = renamed["policy"].pop("fc.weight")
    assert hash_weights(renamed) != base


@pytest.mark.unit
def test_hash_weights_tensor_and_equivalent_array_hash_alike():
    t = torch.tensor([1.0, 2.0], dtype=torch.float64)
    a = np.array([1.0, 2.0], dtype=np.float64)
    assert hash_weights(t) == hash_weights(a)
    # Non-contiguous inputs are made contiguous before hashing.
    t2 = torch.arange(4, dtype=torch.float64).reshape(2, 2).t()
    a2 = np.arange(4, dtype=np.float64).reshape(2, 2).T
    assert hash_weights(t2) == hash_weights(a2)
    assert hash_weights(t2) != hash_weights(
        torch.arange(4, dtype=torch.float64).reshape(2, 2)
    )


@pytest.mark.unit
def test_hash_weights_tensor_requiring_grad_hashes_like_its_values():
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    assert hash_weights(t) == hash_weights(np.array([1.0, 2.0], dtype=np.float32))


@pytest.mark.unit
def test_hash_weights_plain_leaves_use_repr():
    assert hash_weights(1) != hash_weights(1.0)
    assert hash_weights("a") == hash_weights("a")
    assert hash_weights(None) != hash_weights({})
    assert hash_weights({"k": 1}) != hash_weights({"j": 1})


# ---------------------------------------------------------------------------
# parse_learner_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "learner_id, expected",
    [
        ("fisher_m0_s11", ("fisher", "0", "11")),
        ("fisher:0_mquota_s3", ("fisher:0", "quota", "3")),
        # Only the last "_m" and the last "_s" are separators.
        ("a_m1_m2_s3", ("a_m1", "2", "3")),
        ("x_s_m0_s1", ("x_s", "0", "1")),
    ],
)
def test_parse_learner_id_splits_policy_mechanism_and_seed(learner_id, expected):
    assert parse_learner_id(learner_id) == expected


@pytest.mark.unit
@pytest.mark.parametrize("learner_id", ["p", "p_s1", "p_m0", "", "p_m0s1"])
def test_parse_learner_id_rejects_malformed_ids(learner_id):
    with pytest.raises(ValueError, match="<policy>_m<mechanism>_s<seed>") as info:
        parse_learner_id(learner_id)

    assert repr(learner_id) in str(info.value)
    # The unpacking error is replaced, not chained.
    assert info.value.__cause__ is None
