"""Shared building blocks of the environment tests: a toy bilevel benchmark.

The toy benchmark has the shape of the fishery but nothing else of it:

- ``Fisher`` agents hold a ``Harvest`` mechanism that removes a fraction of a
  shared ``stock``; their reward is the harvest of the step, and they observe
  ``[stock, 0.0]``;
- the ``Regulator`` leader holds a ``Fee`` mechanism, searched by the outer
  optimizer, that subtracts the fee from every follower's reward and writes it
  into the second entry of the followers' observation;
- ``ToyEnv`` initialises the stock at reset and carries it to the next step.

``ScriptedWorld`` stands in for the ``World`` Ray actor: it replays fixed
answers to ``get_mechanism_by_id`` and records appended contexts. Tests that use
it make ``ray.get`` an identity function with the ``identity_ray_get`` fixture
of the module under test (``core.envs.marl_regulated.ray`` and
``core.envs.regulator.ray`` are the same ``ray`` module object).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import ray
from gymnasium import spaces

from core.agents.base import Agent, AgentConfig
from core.envs.hooks import reset, transition
from core.envs.marl_regulated import MultiAgentEnv
from core.envs.schema import EpisodeRolloutSchema
from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig

UNBOUNDED = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
FEE_SPACE = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
INITIAL_STOCK = 1.0


class ScriptedWorld:
    """World stand-in that replays answers and records appended contexts."""

    def __init__(self, answers: list | None = None) -> None:
        self.answers = list(answers or [])
        self.fetches: list[dict[str, Any]] = []
        self.contexts: list = []
        self.get_mechanism_by_id = SimpleNamespace(remote=self._fetch)
        self.append_context = SimpleNamespace(remote=self.contexts.append)

    def _fetch(self, **kwargs):
        self.fetches.append(kwargs)
        return self.answers.pop(0) if self.answers else None


class Harvest(Mechanism):
    """Remove the requested amount from the shared stock."""

    def decode(self, mdp: MDPState, action: Any) -> float:
        return float(np.asarray(action).reshape(-1)[0])

    def apply(self, mdp: MDPState, action: float) -> MDPState:
        return MDPState(state={"stock": -action})


class Fee(Mechanism):
    """Charge every follower the fee of the candidate, at each step."""

    def decode(self, mdp: MDPState, action: Any) -> float:
        return float(np.asarray(action).reshape(-1)[0])

    def apply(self, mdp: MDPState, action: float) -> MDPState:
        return MDPState(rewards={aid: -action for aid in mdp.aids})

    def observe(self, mdp: MDPState) -> MDPState:
        fee = float(np.asarray(mdp.actions[self.aid][self.id][-1]).reshape(-1)[0])
        return MDPState(
            obs={aid: np.asarray([0.0, fee], dtype=np.float32) for aid in mdp.aids}
        )


class Fisher(Agent):
    """Follower: rewarded with its harvest, observes the stock."""

    def observation(self, mdp: MDPState) -> MDPState:
        stock = mdp.state["stock"][mdp.t]
        return MDPState(obs={self.id: np.asarray([stock, 0.0], dtype=np.float32)})

    def reward(self, mdp: MDPState) -> MDPState:
        return MDPState(rewards={self.id: mdp.actions[self.id]["harvest"][mdp.t]})


class Observer(Agent):
    """Follower that needs no state: constant observation, no reward."""

    def observation(self, mdp: MDPState) -> MDPState:
        return MDPState(obs={self.id: np.asarray([7.0], dtype=np.float32)})


class HarvestConfig(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = Harvest


class FeeConfig(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = Fee


class FisherConfig(AgentConfig):
    agent_cls: ClassVar[type[Agent]] = Fisher


class ObserverConfig(AgentConfig):
    agent_cls: ClassVar[type[Agent]] = Observer


class ToyEnv(MultiAgentEnv):
    """Stock shared by the followers, carried unchanged to the next step."""

    @reset
    def init_stock(self, mdp: MDPState) -> MDPState:
        return MDPState(state={"stock": INITIAL_STOCK})

    @transition
    def carry_stock(self, mdp: MDPState) -> MDPState:
        return mdp.advance(state={"stock": mdp.state["stock"][mdp.t]})


class RecordingReporter(Reporter):
    """Reporter that keeps its label and never reports."""

    def __init__(self, label: str | None) -> None:
        self.label = label

    def _report(self, query, series) -> None:
        return None

    def close(self) -> None:
        return None


class RecordingReporterConfig(ReporterConfig):
    """Builds a ``RecordingReporter`` carrying the label it was asked for."""

    def build(self, *, label: str | None = None) -> RecordingReporter:
        return RecordingReporter(label)


def fisher_configs(count: int = 2) -> dict[str, FisherConfig]:
    return {
        f"f{i}": FisherConfig(
            id=f"f{i}",
            policy_id="fisher_policy",
            mechanisms=HarvestConfig(action_space=UNBOUNDED, id="harvest"),
        )
        for i in range(count)
    }


def regulator_configs(fee_default: float | None = None) -> dict[str, AgentConfig]:
    """The regulator leader; ``fee_default`` becomes the ``Fee`` default action."""
    default = None if fee_default is None else np.asarray([fee_default])
    return {
        "regulator": AgentConfig(
            id="regulator",
            policy_id="regulator_policy",
            mechanisms=FeeConfig(action_space=FEE_SPACE, id="fee", default=default),
        )
    }


def candidate(fee: float) -> SimpleNamespace:
    """What ``World.get_mechanism_by_id`` returns for a published candidate."""
    return SimpleNamespace(mechanism={"fee": np.asarray([fee], dtype=np.float32)})


def make_toy_env(
    world: ScriptedWorld,
    *,
    env_cls: type[MultiAgentEnv] = ToyEnv,
    followers: int = 2,
    leader: bool = True,
    **overrides: Any,
) -> MultiAgentEnv:
    """Build ``env_cls`` with the toy agents; ``overrides`` replace any argument."""
    kwargs: dict[str, Any] = {
        "world": world,
        "mechanism_id": 0,
        "env_name": "toy",
        "horizon": 3,
        "seed": 5,
        "policy_seed": 3,
        "agents_cfg_dict": fisher_configs(followers),
        "leaders_cfg_dict": regulator_configs() if leader else {},
        "reporter_cfg": RecordingReporterConfig(project="p"),
        "schema": EpisodeRolloutSchema,
    }
    kwargs.update(overrides)
    return env_cls(**kwargs)


def harvest_actions(aids: list[str], amount: float) -> dict[str, dict[str, Any]]:
    """The action dictionary RLlib hands to the adapter, then to ``mdp.update``."""
    return {aid: {"harvest": np.asarray([amount], dtype=np.float32)} for aid in aids}


@pytest.fixture
def identity_ray_get(monkeypatch) -> None:
    """Make ``ray.get`` return its argument so no Ray runtime is needed."""
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)


@pytest.fixture
def toy() -> SimpleNamespace:
    """The toy benchmark as one namespace, so test modules need no conftest import."""
    return SimpleNamespace(
        ScriptedWorld=ScriptedWorld,
        ToyEnv=ToyEnv,
        Fisher=Fisher,
        Observer=Observer,
        Harvest=Harvest,
        Fee=Fee,
        FisherConfig=FisherConfig,
        ObserverConfig=ObserverConfig,
        HarvestConfig=HarvestConfig,
        FeeConfig=FeeConfig,
        RecordingReporter=RecordingReporter,
        RecordingReporterConfig=RecordingReporterConfig,
        fisher_configs=fisher_configs,
        regulator_configs=regulator_configs,
        candidate=candidate,
        make_env=make_toy_env,
        harvest_actions=harvest_actions,
        initial_stock=INITIAL_STOCK,
    )
