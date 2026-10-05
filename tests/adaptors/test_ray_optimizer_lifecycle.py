"""``RayOptimizer``: construction, training loop, evaluation, reset and stop.

``RayOptimizer`` forwards its calls to a ``PolicyActor`` and feeds every RLlib
result into a ``MetricLogger`` built from ``RaySchema``. Here the actor is an
in-memory stub exposing the ``<method>.remote()`` call shape, the ``World`` is a
fake with the same shape, ``ray.get`` is the identity and the frozen
``RayOptimizerConfig`` is a namespace carrying only the attributes the
optimizer reads. Nothing starts Ray. The logger content is read back through
``peek`` (after ``train``) or through ``stop``, which reduces it.

``RayOptimizer.train`` returns the peeked ``RaySchema`` although the base class
annotates it ``-> None``: the regulator environment reads that return value as
the inner optimizer's metrics, so the tests pin the return value and not the
annotation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import ray

import core.adaptors.ray.policy_actor as policy_actor_module
from core.adaptors.ray.optimizer import RayOptimizer
from core.adaptors.ray.schema import RaySchema
from core.envs.schema import EpisodeRolloutSchema
from core.world.context import MechanismStatus

EPISODE_ID = "env=0|m=0|ps=1|ss=1|raw=abc"


class StubPolicyActor:
    """Stand-in for a ``PolicyActor`` handle.

    ``train`` pops the next entry of ``train_results`` (the last one repeats),
    ``evaluate`` returns ``eval_result``; every remote call is appended to
    ``calls``.
    """

    def __init__(self, algo_config, train_results, eval_result):
        self.algo_config = algo_config
        self.calls: list[str] = []
        self.train_results = list(train_results)
        self.eval_result = eval_result
        self.train = SimpleNamespace(remote=lambda: self._call("train"))
        self.evaluate = SimpleNamespace(remote=lambda: self._call("evaluate"))
        self.reset = SimpleNamespace(remote=lambda: self._call("reset"))
        self.stop = SimpleNamespace(remote=lambda: self._call("stop"))

    def _call(self, name):
        self.calls.append(name)

        if name == "train":
            if len(self.train_results) > 1:
                return self.train_results.pop(0)

            return self.train_results[0]

        if name == "evaluate":
            return self.eval_result

        return None


class StubPolicyActorClass:
    """Mimics the ``PolicyActor.remote(...)`` factory of a Ray actor class."""

    def __init__(self):
        self.train_results = [{}]
        self.eval_result = {}
        self.instances: list[StubPolicyActor] = []

    def remote(self, algo_config):
        actor = StubPolicyActor(algo_config, self.train_results, self.eval_result)
        self.instances.append(actor)

        return actor


class FakeWorld:
    """``World`` stand-in recording the ``flush(status=...)`` calls."""

    def __init__(self):
        self.flushed: list[MechanismStatus | None] = []
        self.flush = SimpleNamespace(remote=self._flush)

    def _flush(self, status=None):
        self.flushed.append(status)


def make_config(
    *,
    episodes=1,
    evaluation_duration=10,
    rollout_fragment_length=5,
    num_envs_per_env_runner=6,
    seeds=(1, 2),
    policies=(),
):
    """Namespace carrying the attributes ``RayOptimizer`` reads from its config."""
    rllib_cfg = SimpleNamespace(
        evaluation_duration=evaluation_duration,
        evaluation_config={"rollout_fragment_length": rollout_fragment_length},
        num_envs_per_env_runner=num_envs_per_env_runner,
        policies={policy_id: None for policy_id in policies},
    )

    return SimpleNamespace(
        episodes=episodes, env=None, rllib_cfg=rllib_cfg, seeds=list(seeds)
    )


def train_result(*, return_mean=1.5, iteration=7, steps=100, lifetime=700, loss=0.25):
    """Minimal new-API-stack ``Algorithm.train()`` result with one episode."""
    return {
        "training_iteration": iteration,
        "env_runners": {
            "episode_return_mean": return_mean,
            "num_env_steps_sampled": steps,
            "num_env_steps_sampled_lifetime": lifetime,
            "by_episode": {
                EPISODE_ID: EpisodeRolloutSchema(
                    mechanism_id=0, seed=1, reward_mean=return_mean
                )
            },
        },
        "learners": {"fisher_m0_s1": {"policy_loss": loss, "entropy": 0.5}},
        "timers": {"training_iteration": 2.0},
    }


@pytest.fixture
def actors(monkeypatch) -> StubPolicyActorClass:
    """Patch ``PolicyActor`` (imported lazily in ``__init__``) and ``ray.get``."""
    stub = StubPolicyActorClass()
    monkeypatch.setattr(policy_actor_module, "PolicyActor", stub)
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)

    return stub


def make_optimizer(**config_kwargs) -> tuple[RayOptimizer, FakeWorld]:
    world = FakeWorld()

    return RayOptimizer(config=make_config(**config_kwargs), world=world), world


@pytest.mark.unit
def test_init_spawns_the_actor_from_the_rllib_config(actors):
    cfg = make_config(evaluation_duration=10, rollout_fragment_length=5)

    opt = RayOptimizer(config=cfg)

    assert actors.instances == [opt.policy_actor]
    assert opt.policy_actor.algo_config is cfg.rllib_cfg
    assert opt._es_round == 0
    assert opt.logger._schema is RaySchema
    assert isinstance(opt.logger.peek(), RaySchema)


@pytest.mark.unit
def test_training_logs_learner_statistics_for_every_declared_module(actors):
    # The module IDs come from the RLlib config, so a module that RLlib leaves
    # out of a result is still logged, with NaN statistics.
    opt, _ = make_optimizer(policies=("fisher_m0_s1", "fisher_m1_s1"))

    assert opt._module_ids == ("fisher_m0_s1", "fisher_m1_s1")

    by_mechanism = opt._to_logger_payload(train_result()).train.learner.by_mechanism
    assert by_mechanism["0"].by_seed["1"].by_policy["fisher"].policy_loss == 0.25
    assert np.isnan(by_mechanism["1"].by_seed["1"].by_policy["fisher"].policy_loss)


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_envs, seeds, expected",
    [(6, (1, 2), 3), (8, (1, 2, 3, 4), 2), (5, (1,), 5)],
    ids=["6-envs-2-seeds", "8-envs-4-seeds", "single-seed"],
)
def test_batch_capacity_is_the_mechanism_count_per_runner(
    actors, num_envs, seeds, expected
):
    opt, _ = make_optimizer(num_envs_per_env_runner=num_envs, seeds=seeds)

    assert opt.batch_capacity == expected


@pytest.mark.unit
def test_batch_capacity_without_training_seeds_divides_by_zero(actors):
    opt, _ = make_optimizer(seeds=())

    with pytest.raises(ZeroDivisionError):
        _ = opt.batch_capacity


@pytest.mark.unit
def test_train_runs_every_inner_iteration_then_evaluates(actors, caplog):
    actors.train_results = [
        train_result(return_mean=1.0, iteration=7, steps=100, lifetime=100),
        train_result(return_mean=3.0, iteration=8, steps=100, lifetime=200),
    ]
    actors.eval_result = {"env_runners": {"episode_return_mean": 9.0}}
    opt, world = make_optimizer(episodes=2)

    with caplog.at_level("INFO", logger="core.adaptors.ray.optimizer"):
        returned = opt.train()

    assert opt.policy_actor.calls == ["train", "train", "evaluate"]
    assert world.flushed == [MechanismStatus.eval]

    lines = [
        r.getMessage()
        for r in caplog.records
        if "Training step completed" in r.getMessage()
    ]
    assert len(lines) == 2
    assert "inner_iter=0" in lines[0] and "rllib_iter_lifetime=7" in lines[0]
    assert "inner_iter=1" in lines[1] and "rllib_iter_lifetime=8" in lines[1]
    assert "ep_return=3.0000" in lines[1] and "env_steps_lifetime=200" in lines[1]
    assert "policy_loss=0.250000" in lines[0]

    # The return value is the logger content, with training and evaluation.
    assert isinstance(returned, RaySchema)
    assert returned.iter == [0, 1]
    assert returned.train.rollout.aggregate.reward_mean == [1.0, 3.0]
    assert returned.train.learner.by_mechanism["0"].by_seed["1"].by_policy[
        "fisher"
    ].policy_loss == [0.25, 0.25]
    assert returned.eval.rollout.aggregate.reward_mean == [9.0]


@pytest.mark.unit
def test_train_reports_the_metrics_once_after_the_evaluation(actors):
    actors.train_results = [train_result()]
    actors.eval_result = {"env_runners": {"episode_return_mean": 9.0}}
    reporter = MagicMock(name="reporter")
    opt = RayOptimizer(config=make_config(), world=FakeWorld(), reporting=reporter)

    returned = opt.train()

    reporter.report.assert_called_once()
    (reported,) = reporter.report.call_args.args
    assert isinstance(reported, RaySchema)
    assert reported.iter == returned.iter == [0]
    assert reported.train.rollout.aggregate.reward_mean == [1.5]
    assert reported.eval.rollout.aggregate.reward_mean == [9.0]


@pytest.mark.unit
def test_train_logs_na_when_the_result_has_no_policy_loss(actors, caplog):
    # APPO's learner reports its stats once every 20 updates: the iterations in
    # between carry no loss, which is printed as ``NA``.
    actors.train_results = [{"env_runners": {"episode_return_mean": 0.0}}]
    opt, _ = make_optimizer(episodes=2)

    with caplog.at_level("INFO", logger="core.adaptors.ray.optimizer"):
        opt.train()

    assert caplog.text.count("policy_loss=NA") == 2
    assert "rllib_iter_lifetime=0" in caplog.text


@pytest.mark.unit
def test_train_logs_the_nested_evaluation_block_of_a_result(actors):
    result = train_result()
    result["evaluation"] = {"env_runners": {"episode_return_mean": 4.0}}
    actors.train_results = [result]
    actors.eval_result = {"env_runners": {"episode_return_mean": 9.0}}
    opt, _ = make_optimizer()

    returned = opt.train()

    # First the block nested in the training result, then the explicit pass.
    assert returned.eval.rollout.aggregate.reward_mean == [4.0, 9.0]


@pytest.mark.unit
def test_evaluate_forwards_to_the_actor_logs_the_eval_branch_and_flushes(
    actors, caplog
):
    actors.eval_result = {
        "env_runners": {"episode_return_mean": 4.0, "num_env_steps_sampled": 12}
    }
    opt, world = make_optimizer()

    with caplog.at_level("INFO", logger="core.adaptors.ray.optimizer"):
        opt.evaluate()

    assert opt.policy_actor.calls == ["evaluate"]
    assert "Evaluation started" in caplog.text
    assert "Evaluation completed" in caplog.text
    assert world.flushed == [MechanismStatus.eval]

    reduced = opt.logger.reduce()
    assert reduced.eval.rollout.aggregate.reward_mean == 4.0
    assert reduced.eval.performance.env_steps_this_iter == 12.0
    assert reduced.train.rollout.aggregate.reward_mean is None


@pytest.mark.unit
def test_reset_clears_the_logger_and_counts_the_round(actors):
    actors.train_results = [train_result(return_mean=2.0)]
    opt, _ = make_optimizer()
    opt.train()
    assert opt.logger.peek().train is not None

    opt.reset()

    assert opt.policy_actor.calls[-1] == "reset"
    assert opt._es_round == 1
    reduced = opt.logger.reduce()
    assert reduced.iter is None
    assert reduced.train.rollout.aggregate.reward_mean is None

    opt.reset()
    assert opt._es_round == 2


@pytest.mark.unit
def test_stop_stops_the_actor_and_returns_the_metrics_reduced_over_the_run(actors):
    actors.train_results = [
        train_result(return_mean=1.0),
        train_result(return_mean=3.0),
    ]
    opt, _ = make_optimizer()
    opt.train()
    opt.train()

    reduced = opt.stop()

    assert opt.policy_actor.calls[-1] == "stop"
    assert isinstance(reduced, RaySchema)
    # ``RaySchema.train`` is reduced with ``subtree_reduce=LAST``: the run's
    # summary is the last iteration, not the mean over iterations.
    assert reduced.train.rollout.aggregate.reward_mean == 3.0


@pytest.mark.unit
def test_save_is_a_stub_returning_none(actors):
    opt, _ = make_optimizer()

    assert opt.save() is None
    assert opt.save(checkpoint_dir="/nowhere") is None
