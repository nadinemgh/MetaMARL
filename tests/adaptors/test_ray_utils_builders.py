"""``core.adaptors.ray.utils``: builders from an RLlib result to typed schemas.

``build_episode_aggregate``, ``build_performance``, ``build_rollout`` and
``build_learner`` are fed hand-built result dictionaries and their output
(``EpisodeRolloutSchema``, ``PerformanceSchema``, ``RolloutSchema``,
``LearnerSchema``) is compared field by field, including the derived learner
quantities.

Ported from the August suite. ``LearnerSchema`` is now grouped as
``by_mechanism -> by_seed -> by_policy`` and keyed by the learner ID
``<policy>_m<mechanism>_s<seed>``, so the August learner tests were rewritten
against that layout. The August test that expected the ``__all_modules__``
block to become a pseudo-policy was inverted: the block is now skipped as a
policy and only feeds the queue-wait lag indicator. The August test that
expected a fractional ``module_train_batch_size_mean`` to raise is not kept,
because it is unclear whether that is intended (see the report).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.adaptors.ray.schema import (
    LearnerSchema,
    PerformanceSchema,
    PolicyLearnerSchema,
    RolloutSchema,
)
from core.adaptors.ray.utils import (
    build_episode_aggregate,
    build_learner,
    build_performance,
    build_rollout,
)
from core.envs.schema import EpisodeRolloutSchema

# ---------------------------------------------------------------------------
# build_episode_aggregate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_episode_aggregate_reads_env_runner_stats():
    result = {
        "env_runners": {
            "episode_return_mean": np.float32(1.5),
            "episode_return_min": -2.0,
            "episode_return_max": 4,
            "episode_len_mean": 10.5,
            "episode_len_min": 8,
            "episode_len_max": 12,
            "num_episodes": 3,
            "num_episodes_lifetime": 30,
        }
    }
    agg = build_episode_aggregate(result)

    assert isinstance(agg, EpisodeRolloutSchema)
    assert agg.reward_mean == 1.5
    assert agg.reward_min == -2.0
    assert agg.reward_max == 4.0
    assert (agg.episode_len_mean, agg.episode_len_min, agg.episode_len_max) == (
        10.5,
        8.0,
        12.0,
    )
    assert (agg.num_episodes, agg.num_episodes_lifetime) == (3, 30)
    # RLlib provides no aggregate-level totals or terminal statistics.
    assert agg.reward_total is None and agg.reward_terminal is None
    assert agg.value_terminal is None and agg.value_penultimate is None


@pytest.mark.unit
@pytest.mark.parametrize("result", [{}, {"env_runners": None}, {"env_runners": {}}])
def test_build_episode_aggregate_missing_block_gives_none_fields(result):
    agg = build_episode_aggregate(result)
    assert agg.reward_mean is None and agg.num_episodes is None


@pytest.mark.unit
def test_build_episode_aggregate_rejects_non_finite_values():
    result = {
        "env_runners": {"episode_return_mean": float("nan"), "episode_len_mean": "x"}
    }
    agg = build_episode_aggregate(result)
    assert agg.reward_mean is None and agg.episode_len_mean is None


# ---------------------------------------------------------------------------
# build_performance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_performance_sums_agent_steps_and_reads_timers():
    result = {
        "env_runners": {
            "num_env_steps_sampled": 100,
            "num_env_steps_sampled_lifetime": np.int64(700),
            "num_agent_steps_sampled": {"a:0": 60, "a:1": np.float32(40)},
            "num_agent_steps_sampled_lifetime": {"a:0": 400, "a:1": "bad"},
            "num_env_steps_sampled_lifetime_throughput": {
                "throughput_since_last_reduce": 55.5,
                "throughput_since_last_restore": 11.0,
            },
            "weights_seq_no": 4,
        },
        "timers": {
            "training_iteration": 2.5,
            "training_step": 2.0,
            "sample": 1.0,
            "learner_update_timer": 0.5,
        },
    }
    perf = build_performance(result)

    assert isinstance(perf, PerformanceSchema)
    assert perf.env_steps_this_iter == 100.0
    assert perf.env_steps_lifetime == 700.0
    assert perf.agent_steps_this_iter_sum == 100.0
    # Unconvertible per-agent entries count as zero.
    assert perf.agent_steps_lifetime_sum == 400.0
    assert perf.env_steps_throughput == 55.5
    assert perf.training_iteration_s == 2.5
    assert perf.training_step_s == 2.0
    assert perf.sample_s == 1.0
    assert perf.learner_update_s == 0.5
    assert perf.weights_seq_no == 4.0


@pytest.mark.unit
def test_build_performance_throughput_falls_back_to_since_restore():
    env = {
        "num_env_steps_sampled_lifetime_throughput": {
            "throughput_since_last_reduce": None,
            "throughput_since_last_restore": 11.0,
        }
    }
    assert build_performance({"env_runners": env}).env_steps_throughput == 11.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "env",
    [
        {},
        {
            "num_env_steps_sampled_lifetime_throughput": 3.0,
            "num_agent_steps_sampled": 12,
            "num_agent_steps_sampled_lifetime": [1, 2],
        },
    ],
)
def test_build_performance_non_dict_blocks_give_none(env):
    perf = build_performance({"env_runners": env, "timers": None})
    assert perf.env_steps_throughput is None
    assert perf.agent_steps_this_iter_sum is None
    assert perf.agent_steps_lifetime_sum is None
    assert perf.training_iteration_s is None


# ---------------------------------------------------------------------------
# build_rollout
# ---------------------------------------------------------------------------


def _episode(mechanism_id, seed, reward_mean=0.0) -> EpisodeRolloutSchema:
    return EpisodeRolloutSchema(
        mechanism_id=mechanism_id, seed=seed, reward_mean=reward_mean
    )


@pytest.mark.unit
def test_build_rollout_groups_episodes_by_mechanism_then_seed():
    e00 = _episode(0, 11, 1.0)
    e01 = _episode(0, 22, 2.0)
    e10 = _episode(1, 11, 3.0)
    e00b = _episode(0, 11, 4.0)
    result = {
        "env_runners": {
            "episode_return_mean": 2.5,
            "by_episode": {
                "env=0|m=0|ps=11|ss=11": e00,
                "env=1|m=0|ps=22|ss=22": e01,
                "env=2|m=1|ps=11|ss=11": e10,
                "env=0|m=0|ps=11|ss=11#2": e00b,
            },
        }
    }
    rollout = build_rollout(result)

    assert isinstance(rollout, RolloutSchema)
    assert rollout.aggregate.reward_mean == 2.5
    assert set(rollout.by_mechanism) == {"0", "1"}
    mech0 = rollout.by_mechanism["0"]
    assert set(mech0.by_seed) == {"11", "22"}
    assert mech0.by_seed["11"].by_episode == {
        "env=0|m=0|ps=11|ss=11": e00,
        "env=0|m=0|ps=11|ss=11#2": e00b,
    }
    assert mech0.by_seed["22"].by_episode == {"env=1|m=0|ps=22|ss=22": e01}
    assert rollout.by_mechanism["1"].by_seed["11"].by_episode == {
        "env=2|m=1|ps=11|ss=11": e10
    }
    # The episode objects are filed as-is, not copied.
    assert mech0.by_seed["11"].by_episode["env=0|m=0|ps=11|ss=11"] is e00


@pytest.mark.unit
@pytest.mark.parametrize(
    "result",
    [{}, {"env_runners": {"by_episode": None}}, {"env_runners": {"by_episode": {}}}],
)
def test_build_rollout_without_episodes_is_empty(result):
    rollout = build_rollout(result)
    assert rollout.by_mechanism == {}
    assert rollout.aggregate.reward_mean is None


@pytest.mark.unit
def test_build_rollout_stringifies_missing_identity_as_none_key():
    # Episodes without ``mechanism_id`` / ``seed`` are not rejected; they land
    # under the literal "None" keys.
    result = {"env_runners": {"by_episode": {"ep": _episode(None, None)}}}
    rollout = build_rollout(result)
    assert list(rollout.by_mechanism) == ["None"]
    assert list(rollout.by_mechanism["None"].by_seed) == ["None"]


# ---------------------------------------------------------------------------
# build_learner
# ---------------------------------------------------------------------------

LEARNER_ID = "fisher_m0_s11"


def _learner_result(**overrides):
    stats = {
        "module_train_batch_size_mean": 128,
        "total_loss": 0.9,
        "policy_loss": 0.25,
        "entropy": 0.8,
        "curr_entropy_coeff": 0.01,
        "kl": 0.02,
        "curr_kl_coeff": 0.2,
        "vf_loss": 0.4,
        "value_mean": 1.1,
        "value_target": 1.2,
        "gradients_default_optimizer_global_norm": 3.0,
        "grad_gnorm": 2.0,
        "gradient_noise": 0.05,
        "diff_num_grad_updates_vs_sampler_policy": 1.0,
        "num_module_steps_trained_lifetime_throughput": {
            "throughput_since_last_reduce": 10.0,
            "throughput_since_last_restore": 5.0,
        },
        "ignored_nan": float("nan"),
        "ignored_text": "n/a",
    }
    stats.update(overrides)
    return {
        "learners": {
            LEARNER_ID: stats,
            "__all_modules__": {"learner_thread_in_queue_wait_timer": 0.5},
        },
        "learner_group": {"actor_manager_num_outstanding_async_reqs": 2},
        "mean_num_training_step_calls_since_last_synch_worker_weights": 3,
    }


def _only_policy(learner: LearnerSchema, policy_id: str = "fisher"):
    return learner.by_mechanism["0"].by_seed["11"].by_policy[policy_id]


@pytest.mark.unit
def test_build_learner_copies_stats_and_derives_quantities():
    learner = build_learner(_learner_result())

    assert isinstance(learner, LearnerSchema)
    policy = _only_policy(learner)
    assert isinstance(policy, PolicyLearnerSchema)
    assert policy.batch_size == 128
    assert policy.total_loss == 0.9
    assert policy.policy_loss == 0.25
    assert policy.policy_entropy == 0.8
    assert policy.policy_entropy_coeff == 0.01
    assert policy.policy_relative_entropy == pytest.approx(0.8 / 0.01)
    assert policy.entropy_pressure == pytest.approx(0.8 * 0.01)
    assert policy.policy_kl == 0.02 and policy.policy_kl_coeff == 0.2
    assert policy.value_loss == 0.4
    assert policy.value_mean == 1.1 and policy.value_target == 1.2
    assert policy.gradient_norm == 3.0
    assert policy.gradient_noise == 0.05
    # lag1 + training calls since sync + outstanding reqs + queue wait.
    assert policy.sample_staleness == pytest.approx(1.0 + 3 + 2 + 0.5)
    assert policy.residual_variance is None


@pytest.mark.unit
def test_build_learner_files_modules_by_mechanism_seed_and_policy():
    result = {
        "learners": {
            "fisher_m0_s11": {"policy_loss": 1.0},
            "fisher_m0_s22": {"policy_loss": 2.0},
            "fisher_m1_s11": {"policy_loss": 3.0},
            "regulator_m1_s11": {"policy_loss": 4.0},
        }
    }
    learner = build_learner(result)

    assert set(learner.by_mechanism) == {"0", "1"}
    assert set(learner.by_mechanism["0"].by_seed) == {"11", "22"}
    losses = {
        (mechanism_id, seed, policy_id): policy.policy_loss
        for mechanism_id, mechanism in learner.by_mechanism.items()
        for seed, seed_learner in mechanism.by_seed.items()
        for policy_id, policy in seed_learner.by_policy.items()
    }
    assert losses == {
        ("0", "11", "fisher"): 1.0,
        ("0", "22", "fisher"): 2.0,
        ("1", "11", "fisher"): 3.0,
        ("1", "11", "regulator"): 4.0,
    }


@pytest.mark.unit
def test_build_learner_skips_the_all_modules_block_as_a_policy():
    learner = build_learner(_learner_result())

    assert list(learner.by_mechanism) == ["0"]
    assert list(learner.by_mechanism["0"].by_seed["11"].by_policy) == ["fisher"]


@pytest.mark.unit
def test_build_learner_keeps_the_throughput_dict_out_of_the_policy_schema():
    # The throughput dict is not a scalar stat; it must not reach the schema
    # (which has no such field) and must not break the build.
    policy = _only_policy(build_learner(_learner_result()))

    assert "num_module_steps_trained_lifetime_throughput" not in policy.model_dump()
    assert policy.policy_loss == 0.25


@pytest.mark.unit
def test_build_learner_gradient_norm_falls_back_to_grad_gnorm():
    learner = build_learner(
        _learner_result(gradients_default_optimizer_global_norm=None)
    )
    assert _only_policy(learner).gradient_norm == 2.0

    learner = build_learner(_learner_result(grad_gnorm=None))
    assert _only_policy(learner).gradient_norm == 3.0


@pytest.mark.unit
def test_build_learner_staleness_uses_only_the_available_indicators():
    result = _learner_result(diff_num_grad_updates_vs_sampler_policy=None)
    del result["learner_group"]
    del result["learners"]["__all_modules__"]

    policy = _only_policy(build_learner(result))
    # Only the training calls since the last weight sync remain.
    assert policy.sample_staleness == pytest.approx(3.0)


@pytest.mark.unit
def test_build_learner_without_entropy_or_lags_leaves_derived_none():
    result = {
        "learners": {"p_m0_s11": {"policy_loss": 0.1, "curr_entropy_coeff": 0.01}}
    }
    policy = _only_policy(build_learner(result), "p")
    assert policy.policy_relative_entropy is None
    assert policy.entropy_pressure is None
    assert policy.sample_staleness is None
    assert policy.gradient_norm is None

    # Zero entropy coefficient: the ratio is undefined, the product still defined.
    result = {"learners": {"p_m0_s11": {"entropy": 0.5, "curr_entropy_coeff": 0.0}}}
    policy = _only_policy(build_learner(result), "p")
    assert policy.policy_relative_entropy is None
    assert policy.entropy_pressure == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("result", [{}, {"learners": None}, {"learners": {}}])
def test_build_learner_without_learners_is_empty(result):
    assert build_learner(result).by_mechanism == {}


@pytest.mark.unit
def test_build_learner_rejects_a_module_id_without_mechanism_and_seed():
    with pytest.raises(ValueError, match="<policy>_m<mechanism>_s<seed>"):
        build_learner({"learners": {"p": {"policy_loss": 0.1}}})
