"""Return annotations of the adaptor methods match what the code returns."""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Optional, Self

import pytest

from core.adaptors.ray.marl_env import RLlibMultiAgentEnvAdapter
from core.adaptors.ray.optimizer import RayOptimizer
from core.adaptors.ray.optimizer_config import RayOptimizerConfig
from core.adaptors.ray.schema import RaySchema
from core.agents.base import AgentConfig


@pytest.mark.unit
def test_train_is_declared_to_return_the_ray_schema():
    hints = typing.get_type_hints(RayOptimizer.train)

    assert hints["return"] is RaySchema


@pytest.mark.unit
def test_save_stub_accepts_a_path_and_declares_no_return_value():
    hints = typing.get_type_hints(RayOptimizer.save)

    assert hints["return"] is type(None)
    assert hints["checkpoint_dir"] == Optional[str | Path]


@pytest.mark.unit
def test_training_is_declared_to_return_self():
    hints = typing.get_type_hints(RayOptimizerConfig.training)

    assert hints["return"] is Self


@pytest.mark.unit
def test_apply_agents_is_declared_to_return_a_dict_of_agent_configs():
    hints = typing.get_type_hints(RayOptimizerConfig._apply_agents_to_rllib)

    assert hints["return"] == dict[str, AgentConfig]


@pytest.mark.unit
def test_marl_adapter_declares_no_env_class_annotation():
    # ``env`` is the only attribute holding the wrapped environment; a class
    # level ``_env`` annotation was never assigned.
    assert "_env" not in RLlibMultiAgentEnvAdapter.__annotations__
