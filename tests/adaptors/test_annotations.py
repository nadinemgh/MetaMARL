"""Return annotations of the adaptor methods match what the code returns."""

from __future__ import annotations

import typing
from typing import Optional, Self

import pytest
from ray.train._internal.session import _TrainingResult

from core.adaptors.ray.marl_env import RLlibMultiAgentEnvAdapter
from core.adaptors.ray.optimizer import RayOptimizer
from core.adaptors.ray.optimizer_config import RayOptimizerConfig
from core.adaptors.ray.schema import RaySchema
from core.agents.base import AgentConfig
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState


@pytest.mark.unit
def test_train_is_declared_to_return_the_ray_schema():
    hints = typing.get_type_hints(RayOptimizer.train)

    assert hints["return"] is RaySchema


@pytest.mark.unit
def test_save_stub_declares_the_training_result_it_is_meant_to_return():
    # The stub does nothing yet; its annotation states the intended contract.
    hints = typing.get_type_hints(RayOptimizer.save)

    assert hints["return"] is _TrainingResult
    assert hints["checkpoint_dir"] == Optional[str]


@pytest.mark.unit
def test_training_is_declared_to_return_self():
    hints = typing.get_type_hints(RayOptimizerConfig.training)

    assert hints["return"] is Self


@pytest.mark.unit
def test_apply_agents_is_declared_to_return_a_list_of_agent_configs():
    hints = typing.get_type_hints(RayOptimizerConfig._apply_agents_to_rllib)

    assert hints["return"] == list[AgentConfig]


@pytest.mark.unit
def test_marl_adapter_annotates_the_wrapped_environment_and_the_mdp():
    # The adapter assigns the environment to ``env`` and the shared state to
    # ``_mdp``; the class-level annotations describe those two attributes.
    annotations = RLlibMultiAgentEnvAdapter.__annotations__

    assert annotations["_mdp"] is MDPState
    assert annotations["_env"] is MultiAgentEnv
