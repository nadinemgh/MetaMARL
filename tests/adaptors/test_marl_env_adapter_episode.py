"""``RLlibMultiAgentEnvAdapter``: the RLlib-facing view of ``MultiAgentEnv``.

The adapter turns the framework environment, which works on an ``MDPState``,
into RLlib's ``reset() -> (obs, infos)`` and ``step(actions) -> (obs, rewards,
terminateds, truncateds, infos)`` contract. The tests run it on top of the
real ``MultiAgentEnv.reset`` and ``MultiAgentEnv.step``, with a ``World``
stand-in that answers ``get_mechanism_by_id`` and ``ray.get`` replaced by the
identity, as in ``tests/envs/test_marl_regulated_mechanism.py``. Only the
agents are fakes: followers that record the actions they receive and earn
``10 * quota + t``, and a leader whose mechanism writes ``t + 1`` into the
second observation entry.

No Ray runtime is started. The adapter does not call the benchmark hooks
itself, so the environment subclass below declares the one transition hook
that advances time.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from core.adaptors.ray.marl_env import RLlibMultiAgentEnvAdapter
from core.envs import marl_regulated
from core.envs.hooks import transition
from core.envs.marl_regulated import MultiAgentEnv
from core.envs.schema import EpisodeRolloutSchema
from core.mechanism.base import MDPState
from core.metrics.logger import MetricLogger
from core.world.context import MechanismStatus

HORIZON = 3


class Follower:
    """Follower that earns ``10 * quota + t`` and observes ``[t, 0]``."""

    def __init__(self, aid: str) -> None:
        self.aid = aid
        self.observation_space = spaces.Box(0.0, 100.0, shape=(2,), dtype=np.float32)
        self.mechanisms = {
            "quota": SimpleNamespace(
                action_space=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
            )
        }
        self.quota_seen: list[float] = []

    def action(self, mdp: MDPState) -> MDPState:
        quota = mdp.actions.data.get(self.aid, {}).get("quota")
        self.quota_seen.append(float(quota[mdp.t][0]) if quota else float("nan"))
        return mdp

    def reward(self, mdp: MDPState) -> MDPState:
        quota = self.quota_seen[-1]
        return MDPState(rewards={self.aid: 10.0 * quota + mdp.t})

    def observation(self, mdp: MDPState) -> MDPState:
        return MDPState(obs={self.aid: np.asarray([mdp.t, 0.0], dtype=np.float32)})


class Regulator:
    """Leader whose mechanism writes ``t + 1`` into the second entry."""

    def action(self, mdp: MDPState) -> MDPState:
        return mdp

    def reward(self, mdp: MDPState) -> MDPState:
        return MDPState()

    def mechanism_observations(self, mdp: MDPState) -> list[MDPState]:
        if "regulator" not in mdp.actions.data:
            return []
        signal = np.asarray([0.0, float(mdp.t + 1)], dtype=np.float32)
        return [MDPState(obs={aid: signal for aid in ("fisher:0", "fisher:1")})]


class ScriptedWorld:
    """``World`` stand-in whose ``get_mechanism_by_id`` replays fixed answers."""

    def __init__(self, answers: list) -> None:
        self.answers = list(answers)
        self.fetches: list[dict] = []
        self.get_mechanism_by_id = SimpleNamespace(remote=self._fetch)

    def _fetch(self, **kwargs):
        self.fetches.append(kwargs)
        return self.answers.pop(0) if self.answers else None


class StepEnv(MultiAgentEnv):
    """``MultiAgentEnv`` with a fixed cast, built without a configuration."""

    def __init__(self, world: ScriptedWorld, *, with_leader: bool = False) -> None:
        self.world = world
        self.followers = {aid: Follower(aid) for aid in ("fisher:0", "fisher:1")}
        self.leaders = {"regulator": Regulator()} if with_leader else {}
        self.lids = set(self.leaders)
        self.agents = {**self.leaders, **self.followers}
        self.horizon = HORIZON
        self._t = 0
        self.m = self.m_ctx = None
        self._using_default_mechanism = True
        self.mechanism_id = "7"
        self.seed = 11
        self.policy_seed = 22
        self.mode = MechanismStatus.train
        self.logger = MetricLogger.from_schema(EpisodeRolloutSchema)
        self.reporter = SimpleNamespace(label="reporter")

    @transition
    def advance_time(self, mdp: MDPState) -> MDPState:
        return mdp.advance(state={"clock": mdp.t + 1})


def candidate(quota: float):
    return SimpleNamespace(mechanism={"quota": np.array([quota], dtype=np.float32)})


def quota_action(value: float) -> dict:
    return {"quota": np.array([value], dtype=np.float32)}


@pytest.fixture(autouse=True)
def ray_get_is_identity(monkeypatch):
    monkeypatch.setattr(marl_regulated.ray, "get", lambda ref: ref)


@pytest.fixture
def adapter() -> RLlibMultiAgentEnvAdapter:
    return RLlibMultiAgentEnvAdapter(StepEnv(ScriptedWorld([])))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_agents_are_the_followers_in_declaration_order(adapter):
    assert adapter.agents == ["fisher:0", "fisher:1"]
    assert adapter.possible_agents == ["fisher:0", "fisher:1"]


@pytest.mark.unit
def test_leaders_are_not_rllib_agents():
    adapter = RLlibMultiAgentEnvAdapter(StepEnv(ScriptedWorld([]), with_leader=True))

    assert adapter.possible_agents == ["fisher:0", "fisher:1"]
    assert "regulator" not in adapter.observation_spaces.spaces
    assert "regulator" not in adapter.action_spaces.spaces


@pytest.mark.unit
def test_spaces_are_dicts_keyed_by_agent_and_mechanism(adapter):
    assert set(adapter.observation_spaces.spaces) == {"fisher:0", "fisher:1"}
    for aid, box in adapter.observation_spaces.spaces.items():
        assert box == adapter.env.followers[aid].observation_space

    quota_space = adapter.env.followers["fisher:0"].mechanisms["quota"].action_space
    assert isinstance(adapter.action_spaces, spaces.Dict)
    assert adapter.action_spaces["fisher:0"]["quota"] == quota_space
    assert set(adapter.action_spaces["fisher:1"].spaces) == {"quota"}


@pytest.mark.unit
def test_identity_logger_and_reporter_come_from_the_wrapped_env(adapter):
    env = adapter.env

    assert adapter.mechanism_id == "7"
    assert adapter.seed == 11
    assert adapter.policy_seed == 22
    assert adapter.mode is MechanismStatus.train
    assert adapter.logger is env.logger
    assert adapter.reporter is env.reporter


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reset_returns_the_first_observation_of_every_follower_and_no_infos(adapter):
    obs, infos = adapter.reset()

    assert infos == {}
    assert set(obs) == {"fisher:0", "fisher:1"}
    for value in obs.values():
        np.testing.assert_array_equal(value, [0.0, 0.0])
        assert value.dtype == np.float32


@pytest.mark.unit
def test_reset_gives_the_env_a_fresh_state_that_carries_the_spaces(adapter):
    seen: list[MDPState] = []
    original = adapter.env.reset
    adapter.env.reset = lambda mdp: (seen.append(mdp), original(mdp))[1]

    adapter.reset()

    (mdp,) = seen
    assert mdp.t == 0
    assert mdp.aids == {"fisher:0", "fisher:1"}
    assert mdp.obs_space is adapter.observation_spaces
    assert mdp.action_spaces is adapter.action_spaces


@pytest.mark.unit
def test_reset_ignores_seed_and_options(adapter):
    obs, _ = adapter.reset(seed=999, options={"anything": True})

    assert adapter.env.seed == 11
    assert adapter.seed == 11
    assert set(obs) == {"fisher:0", "fisher:1"}


@pytest.mark.unit
def test_reset_after_steps_starts_a_new_episode_at_time_zero(adapter):
    adapter.reset()
    adapter.step({"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.5)})
    adapter.step({"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.5)})

    obs, _ = adapter.reset()

    np.testing.assert_array_equal(obs["fisher:0"], [0.0, 0.0])
    assert adapter._mdp.t == 0


@pytest.mark.unit
def test_reset_fetches_the_candidate_by_mechanism_id_policy_seed_and_mode():
    world = ScriptedWorld([None])
    RLlibMultiAgentEnvAdapter(StepEnv(world)).reset()

    assert world.fetches == [
        {"mechanism_id": "7", "seed": 22, "mode": MechanismStatus.train}
    ]


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_returns_the_five_rllib_dicts(adapter):
    adapter.reset()

    obs, rewards, terminateds, truncateds, infos = adapter.step(
        {"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.25)}
    )

    assert set(obs) == set(rewards) == {"fisher:0", "fisher:1"}
    assert set(terminateds) == set(truncateds) == {"fisher:0", "fisher:1", "__all__"}
    assert infos == {}


@pytest.mark.unit
def test_step_delivers_each_agents_action_to_the_env(adapter):
    adapter.reset()

    adapter.step({"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.25)})

    assert adapter.env.followers["fisher:0"].quota_seen == [0.5]
    assert adapter.env.followers["fisher:1"].quota_seen == [0.25]


@pytest.mark.unit
def test_step_rewards_are_those_earned_during_the_step_just_finished(adapter):
    adapter.reset()

    first = adapter.step(
        {"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.25)}
    )
    second = adapter.step(
        {"fisher:0": quota_action(1.0), "fisher:1": quota_action(0.0)}
    )

    # reward = 10 * quota + t, with t the time at which the action was taken.
    assert first[1] == pytest.approx({"fisher:0": 5.0, "fisher:1": 2.5})
    assert second[1] == pytest.approx({"fisher:0": 11.0, "fisher:1": 1.0})


@pytest.mark.unit
def test_step_observations_belong_to_the_new_time_step(adapter):
    adapter.reset()

    obs, *_ = adapter.step(
        {"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.5)}
    )
    obs2, *_ = adapter.step(
        {"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.5)}
    )

    np.testing.assert_array_equal(obs["fisher:0"], [1.0, 0.0])
    np.testing.assert_array_equal(obs2["fisher:1"], [2.0, 0.0])


@pytest.mark.unit
def test_episode_is_truncated_at_the_horizon_and_never_terminated(adapter):
    adapter.reset()
    flags = []

    for _ in range(HORIZON):
        _, _, terminateds, truncateds, _ = adapter.step(
            {"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.5)}
        )
        flags.append((terminateds["__all__"], truncateds["__all__"]))

    assert flags == [(False, False), (False, False), (False, True)]
    assert truncateds == {"fisher:0": True, "fisher:1": True, "__all__": True}
    assert terminateds == {"fisher:0": False, "fisher:1": False, "__all__": False}


@pytest.mark.unit
def test_step_state_persists_between_calls(adapter):
    adapter.reset()
    adapter.step({"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.5)})
    state_after_one = adapter._mdp

    adapter.step({"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.5)})

    assert state_after_one.t == 1
    assert adapter._mdp.t == 2
    assert state_after_one.state["clock"][-1] == 1
    assert adapter._mdp.state["clock"][-1] == 2


# ---------------------------------------------------------------------------
# With a published mechanism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_leader_contribution_reaches_the_observations_once_a_candidate_is_published():
    world = ScriptedWorld([candidate(0.3)])
    adapter = RLlibMultiAgentEnvAdapter(StepEnv(world, with_leader=True))

    obs, _ = adapter.reset()
    stepped, *_ = adapter.step(
        {"fisher:0": quota_action(0.5), "fisher:1": quota_action(0.5)}
    )

    np.testing.assert_allclose(obs["fisher:0"], [0.0, 1.0])
    np.testing.assert_allclose(stepped["fisher:1"], [1.0, 2.0])


@pytest.mark.unit
def test_leader_contribution_is_absent_before_any_candidate_is_published():
    adapter = RLlibMultiAgentEnvAdapter(StepEnv(ScriptedWorld([]), with_leader=True))

    obs, _ = adapter.reset()

    np.testing.assert_allclose(obs["fisher:0"], [0.0, 0.0])
