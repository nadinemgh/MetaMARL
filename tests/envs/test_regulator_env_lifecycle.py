"""``RegulatorEnv``: construction, reset, step bookkeeping and known defects.

The publication of one context per (candidate, policy seed) is pinned by
``test_regulator_publish.py``. The August tests of this class decoded
optimizer vectors through a mechanism template, ran ``inner.run()`` for
``train_iters`` iterations and flushed evaluation contexts; the refactored class
publishes the candidates the optimizer already decoded, calls ``inner.train()``
once and returns ``reward(results)``, so those tests were dropped and the new
contract is tested here with a recording inner optimizer.

Three tests are expected failures (see their reasons): ``reset(seed=...)``,
the ``reporter_cfg=None`` default and the ``opt_id`` getter.
"""

import pytest

from core.envs import regulator
from core.envs.regulator import RegulatorEnv
from core.envs.schema import EpisodeRolloutSchema
from core.reporting.query import Query
from core.world.context import MechanismStatus


class RecordingInner:
    """Inner optimizer stub with the calls ``RegulatorEnv`` makes."""

    def __init__(self) -> None:
        self.resets = 0
        self.trains = 0

    def reset(self) -> None:
        self.resets += 1

    def train(self) -> dict:
        self.trains += 1
        return {"trained": self.trains}

    def reduce_metrics(self) -> str:
        return f"reduced-{self.trains}"


class FitnessEnv(RegulatorEnv):
    """Regulator environment whose reward is the training results themselves."""

    def reward(self, results=None, **kwargs):
        return results


@pytest.fixture(autouse=True)
def ray_get_is_identity(monkeypatch):
    monkeypatch.setattr(regulator.ray, "get", lambda ref: ref)


@pytest.fixture
def inner() -> RecordingInner:
    return RecordingInner()


def make_env(toy, inner, *, env_cls=FitnessEnv, **overrides):
    world = toy.ScriptedWorld()
    kwargs = {
        "world": world,
        "optimizer": inner,
        "horizon": 5,
        "agents_cfgs": toy.fisher_configs(2),
        "seeds": [7],
        "reporter_cfg": toy.RecordingReporterConfig(project="p"),
        "schema": EpisodeRolloutSchema,
        "opt_id": "opt_1",
    }
    kwargs.update(overrides)
    return env_cls(**kwargs)


@pytest.mark.unit
def test_construction_builds_agents_logger_and_reporter(toy, inner):
    env = make_env(toy, inner)

    assert list(env.agents) == ["f0", "f1"]
    assert all(isinstance(a, toy.Fisher) for a in env.agents.values())
    assert env.inner is inner
    assert env.seeds == [7] and env.horizon == 5 and env._t == 0
    assert env.logger.peek().iter == []
    assert env.reporter.label == "FitnessEnv"
    assert env.reporter.schema is EpisodeRolloutSchema


@pytest.mark.unit
def test_without_a_schema_there_is_no_logger(toy, inner):
    assert make_env(toy, inner, schema=None).logger is None


@pytest.mark.unit
@pytest.mark.parametrize("seeds, expected", [(None, []), ([], []), ([1, 2], [1, 2])])
def test_missing_seeds_become_an_empty_list(toy, inner, seeds, expected):
    assert make_env(toy, inner, seeds=seeds).seeds == expected


@pytest.mark.unit
def test_queries_are_added_to_the_reporter(toy, inner):
    query = Query(title="iter", x=("iter",), y=("reward_mean",))

    env = make_env(toy, inner, queries=(query,))

    assert env.reporter.queries == (query,)


@pytest.mark.unit
def test_reset_resets_the_inner_policy_unless_asked_to_persist(toy, inner):
    env = make_env(toy, inner)
    env._t = 4

    assert env.reset() == (None, {})
    assert (env._t, inner.resets) == (0, 1)

    env.reset(options={"persist_agents_policy": True})
    assert inner.resets == 1

    env.reset(options={"persist_agents_policy": False})
    assert inner.resets == 2


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason=(
        "reset(seed=...) reads self.seed, which RegulatorEnv never assigns, so "
        + "the Gymnasium reset signature raises AttributeError"
    ),
)
def test_reset_accepts_a_seed(toy, inner):
    env = make_env(toy, inner)

    assert env.reset(seed=3) == (None, {})


@pytest.mark.unit
def test_step_returns_the_reward_and_the_reduced_inner_metrics(toy, inner):
    env = make_env(toy, inner)

    obs, reward, terminated, truncated, info = env.step([{"fee": 0.1}])

    assert obs is None
    assert reward == {"trained": 1}
    assert info == {"metrics": "reduced-1"}
    assert (terminated, truncated) == (False, False)
    assert inner.trains == 1


@pytest.mark.unit
def test_step_publishes_before_it_trains(toy, inner):
    env = make_env(toy, inner)
    seen_at_train = []
    inner.train = lambda: seen_at_train.append(len(env.world.contexts)) or {}

    env.step([{"fee": 0.1}, {"fee": 0.2}])

    # Two candidates and one seed: both contexts are in the World by then.
    assert seen_at_train == [2]


@pytest.mark.unit
def test_published_contexts_carry_the_optimizer_step_and_env(toy, inner):
    env = make_env(toy, inner, seeds=[7, 8])

    env.step([{"fee": 0.1}])
    env.step([{"fee": 0.2}])

    contexts = env.world.contexts
    assert [c.step for c in contexts] == [0, 0, 1, 1]
    assert {c.opt_id for c in contexts} == {"opt_1"}
    assert {c.env for c in contexts} == {"FitnessEnv"}
    assert {c.payload.env_id for c in contexts} == {None}
    assert {c.payload.status for c in contexts} == {MechanismStatus.published}
    assert [c.payload.mechanism for c in contexts[2:]] == [{"fee": 0.2}] * 2


@pytest.mark.unit
def test_opt_id_setter_is_stamped_on_the_published_contexts(toy, inner):
    env = make_env(toy, inner, opt_id=None)

    env.opt_id = "opt_9"
    env.step([{"fee": 0.1}])

    assert env.world.contexts[0].opt_id == "opt_9"


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    raises=RecursionError,
    reason="the opt_id getter returns self.opt_id, i.e. it calls itself forever",
)
def test_opt_id_getter_returns_the_optimizer_identifier(toy, inner):
    assert make_env(toy, inner).opt_id == "opt_1"


@pytest.mark.unit
def test_step_counts_and_logs_the_outer_iteration(toy, inner):
    env = make_env(toy, inner)

    for _ in range(3):
        env.step([{"fee": 0.1}])

    assert env._t == 3
    assert env.logger.peek().iter == [1, 2, 3]


@pytest.mark.unit
def test_step_without_a_logger_still_counts(toy, inner):
    env = make_env(toy, inner, schema=None)

    env.step([{"fee": 0.1}])

    assert env._t == 1


@pytest.mark.unit
def test_horizon_none_never_terminates(toy, inner):
    env = make_env(toy, inner, horizon=None)

    terminated = [env.step([{"fee": 0.1}])[2] for _ in range(4)]

    assert terminated == [False] * 4


@pytest.mark.unit
def test_a_horizon_of_one_terminates_after_the_first_step(toy, inner):
    env = make_env(toy, inner, horizon=1)

    assert env.step([{"fee": 0.1}])[2] is True


@pytest.mark.unit
def test_default_transforms_return_nothing(toy, inner):
    env = make_env(toy, inner, env_cls=RegulatorEnv)

    assert env.action([1]) is None
    assert env.observation(None) is None
    assert env.reward({}) is None


@pytest.mark.unit
def test_environment_can_be_built_without_a_reporter(toy, inner):
    env = make_env(toy, inner, reporter_cfg=None)

    assert env.reporter is None
