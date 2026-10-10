"""``MultiAgentEnv`` construction: agents, seeds, logger, reporter and properties.

The August suite built the environment from keyword arguments such as
``agents=`` and ``mechanism=``; it is now built from agent configurations
(``agents_cfg_dict`` for followers, ``leaders_cfg_dict`` for leaders) that each
build an ``Agent`` holding its mechanisms. The toy benchmark of
``tests/envs/conftest.py`` provides them.
"""

import numpy as np
import pytest

from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState
from core.metrics.logger import MetricLogger
from core.reporting.query import Query
from core.world.context import MechanismStatus


@pytest.mark.unit
def test_agents_are_built_from_their_configurations_leaders_first(toy):
    env = toy.make_env(toy.ScriptedWorld(), followers=2)

    assert list(env.followers) == ["f0", "f1"]
    assert list(env.leaders) == ["regulator"]
    assert env.lids == {"regulator"}
    assert list(env.agents) == ["regulator", "f0", "f1"]
    assert all(isinstance(a, toy.Fisher) for a in env.followers.values())
    # Each agent owns the mechanisms of its configuration, stamped with its id.
    harvest = env.followers["f1"].mechanisms["harvest"]
    assert isinstance(harvest, toy.Harvest) and harvest.aid == "f1"
    fee = env.leaders["regulator"].mechanisms["fee"]
    assert isinstance(fee, toy.Fee) and fee.aid == "regulator"


@pytest.mark.unit
def test_identity_and_seeding_attributes(toy):
    env = toy.make_env(
        toy.ScriptedWorld(), mode="eval", seed=11, policy_seed=4, horizon=7
    )

    assert (env.seed, env.policy_seed, env.horizon) == (11, 4, 7)
    assert env.mode is MechanismStatus.eval
    assert env.mechanism_id == 0
    assert env.m is None and env.m_ctx is None
    assert env.rng.random() == np.random.default_rng(11).random()


@pytest.mark.unit
def test_an_unknown_mode_is_rejected(toy):
    with pytest.raises(ValueError, match="'training' is not a valid MechanismStatus"):
        toy.make_env(toy.ScriptedWorld(), mode="training")


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, label",
    [
        ({}, "toy|mode=train|m=0|ps=3|ss=5"),
        ({"mode": "eval", "mechanism_id": 2}, "toy|mode=eval|m=2|ps=3|ss=5"),
        ({"mechanism_id": None}, "toy|mode=train|ps=3|ss=5"),
        ({"seed": None, "policy_seed": None}, "toy|mode=train|m=0|ps=None|ss=None"),
    ],
)
def test_reporter_label_names_the_environment_instance(toy, overrides, label):
    env = toy.make_env(toy.ScriptedWorld(), **overrides)

    assert env.reporter.label == label


@pytest.mark.unit
def test_reporter_receives_the_schema_and_the_queries(toy):
    queries = (
        Query(title="reward", x=("iter",), y=("reward_mean",)),
        Query(title="total", x=("iter",), y=("reward_total",)),
    )
    env = toy.make_env(toy.ScriptedWorld(), queries=queries)

    assert env.reporter.queries == queries


@pytest.mark.unit
def test_without_queries_the_reporter_has_none(toy):
    env = toy.make_env(toy.ScriptedWorld())

    assert env.reporter.queries == ()


@pytest.mark.unit
def test_logger_is_built_from_the_schema(toy):
    env = toy.make_env(toy.ScriptedWorld())

    assert isinstance(env.logger, MetricLogger)
    assert env.logger.peek().iter == []


@pytest.mark.unit
def test_environment_can_be_built_without_a_reporter(toy):
    env = toy.make_env(toy.ScriptedWorld(), reporter_cfg=None, schema=None)

    assert env.reporter is None


@pytest.mark.unit
def test_environment_can_be_built_without_leaders(toy):
    env = toy.make_env(toy.ScriptedWorld(), leaders_cfg_dict=None)

    assert env.leaders == {} and env.lids == set()


@pytest.mark.unit
def test_environment_without_a_schema_can_reset(toy, identity_ray_get):
    env = toy.make_env(toy.ScriptedWorld(), schema=None)

    assert env.logger is None
    env.reset(MDPState(aids={"f0", "f1"}))


@pytest.mark.unit
def test_mechanism_is_unset_until_a_candidate_is_fetched(toy, identity_ray_get):
    world = toy.ScriptedWorld([toy.candidate(0.25)])
    env = toy.make_env(world)

    assert env.mechanism is None
    assert not env.published_mechanism_assigned

    env.reset(MDPState(aids={"f0", "f1"}))

    assert env.published_mechanism_assigned
    assert env.mechanism is env.m is env.m_ctx.mechanism
    np.testing.assert_allclose(env.mechanism["fee"], [0.25])


@pytest.mark.unit
def test_opt_id_setter_stores_the_optimizer_identifier(toy):
    env = toy.make_env(toy.ScriptedWorld())

    env.opt_id = "opt_3"

    assert env._opt_id == "opt_3"


@pytest.mark.unit
def test_opt_id_getter_returns_the_optimizer_identifier(toy):
    env = toy.make_env(toy.ScriptedWorld(), opt_id="opt_9")

    assert env.opt_id == "opt_9"


@pytest.mark.unit
def test_follower_configurations_are_required():
    with pytest.raises(TypeError, match="agents_cfg_dict"):
        MultiAgentEnv(world=None, mechanism_id=0)
