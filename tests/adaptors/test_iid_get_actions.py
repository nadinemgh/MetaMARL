"""Unit tests for per-policy CPU sampling RNG in the RLlib GetActions connector."""

import pytest
import torch

import core.adaptors.ray.iid_get_actions as iid_module
from core.adaptors.ray.iid_get_actions import (
    IidGetActions,
    build_iid_module_to_env_connector,
)


@pytest.fixture
def multi_agent_episode(monkeypatch):
    """Select the multi-agent branch without constructing an RLlib episode."""

    class DummyMultiAgentEpisode:
        pass

    monkeypatch.setattr(iid_module, "MultiAgentEpisode", DummyMultiAgentEpisode)

    return DummyMultiAgentEpisode()


def install_sampler(monkeypatch, connector):
    """Replace only RLlib action generation; keep the real RNG handling."""

    def sample(module_data, rl_module, explore):
        module_data["actions"] = torch.rand(module_data["n"])

    monkeypatch.setattr(connector, "_get_actions", sample)


def expected_draws(seed, *lengths):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    return [torch.rand(n, generator=generator) for n in lengths]


@pytest.mark.unit
def test_multiagent_initializes_one_seeded_stream_per_module(
    monkeypatch, multi_agent_episode
):
    connector = IidGetActions()

    install_sampler(monkeypatch, connector)

    module_a = "fisher_m0_s123"
    module_b = "fisher_m1_s456"
    batch = {module_a: {"n": 4}, module_b: {"n": 4}}
    modules = {module_a: object(), module_b: object()}
    original_global_state = torch.get_rng_state().clone()
    returned = connector(
        rl_module=modules, batch=batch, episodes=[multi_agent_episode], explore=True
    )

    assert returned is batch
    torch.testing.assert_close(batch[module_a]["actions"], expected_draws(123, 4)[0])
    torch.testing.assert_close(batch[module_b]["actions"], expected_draws(456, 4)[0])
    assert set(connector.rng_states) == {module_a, module_b}
    assert torch.equal(torch.get_rng_state(), original_global_state)


@pytest.mark.unit
def test_multiagent_streams_advance_independently_across_calls(
    monkeypatch, multi_agent_episode
):
    connector = IidGetActions()

    install_sampler(monkeypatch, connector)

    module_a = "fisher_m0_s22"
    module_b = "fisher_m1_s22"
    modules = {module_a: object(), module_b: object()}
    a_first, a_second = expected_draws(22, 3, 5)
    b_first, b_second = expected_draws(22, 1, 4)
    first = {module_a: {"n": 3}, module_b: {"n": 1}}

    connector(rl_module=modules, batch=first, episodes=[multi_agent_episode])
    torch.testing.assert_close(first[module_a]["actions"], a_first)
    torch.testing.assert_close(first[module_b]["actions"], b_first)

    # Drawing from A must not advance B's independently saved RNG state.
    only_a = {module_a: {"n": 5}}

    connector(rl_module=modules, batch=only_a, episodes=[multi_agent_episode])
    torch.testing.assert_close(only_a[module_a]["actions"], a_second)

    only_b = {module_b: {"n": 4}}

    connector(rl_module=modules, batch=only_b, episodes=[multi_agent_episode])
    torch.testing.assert_close(only_b[module_b]["actions"], b_second)


@pytest.mark.unit
def test_matching_policy_seeds_keep_common_random_draws_aligned(
    monkeypatch, multi_agent_episode
):
    connector = IidGetActions()

    install_sampler(monkeypatch, connector)

    a, b = "fisher_m0_s300", "fisher_m1_s300"
    modules = {a: object(), b: object()}

    for n in (3, 7):
        batch = {a: {"n": n}, b: {"n": n}}

        connector(rl_module=modules, batch=batch, episodes=[multi_agent_episode])
        torch.testing.assert_close(batch[a]["actions"], batch[b]["actions"])


@pytest.mark.unit
def test_multiagent_module_order_does_not_change_samples(
    monkeypatch, multi_agent_episode
):
    a, b = "fisher_m0_s7", "fisher_m1_s8"
    modules = {a: object(), b: object()}
    forward = IidGetActions()
    reverse = IidGetActions()

    install_sampler(monkeypatch, forward)
    install_sampler(monkeypatch, reverse)

    forward_batch = {a: {"n": 3}, b: {"n": 6}}
    reverse_batch = {b: {"n": 6}, a: {"n": 3}}

    forward(rl_module=modules, batch=forward_batch, episodes=[multi_agent_episode])
    reverse(rl_module=modules, batch=reverse_batch, episodes=[multi_agent_episode])

    for module_id in (a, b):
        torch.testing.assert_close(
            forward_batch[module_id]["actions"], reverse_batch[module_id]["actions"]
        )


@pytest.mark.unit
def test_new_connector_restarts_the_seeded_stream(monkeypatch, multi_agent_episode):
    module_id = "fisher_m0_s991"
    modules = {module_id: object()}
    samples = []

    for _ in range(2):
        connector = IidGetActions()

        install_sampler(monkeypatch, connector)

        batch = {module_id: {"n": 5}}

        connector(rl_module=modules, batch=batch, episodes=[multi_agent_episode])
        samples.append(batch[module_id]["actions"])

    torch.testing.assert_close(samples[0], samples[1])


@pytest.mark.unit
def test_single_agent_delegates_without_creating_module_rng_state(monkeypatch):
    connector = IidGetActions()
    rl_module = object()
    batch = {"n": 3}
    seen = []

    def sample(module_data, module, explore):
        seen.append((module_data, module, explore))

        module_data["actions"] = torch.ones(3)

    monkeypatch.setattr(connector, "_get_actions", sample)

    returned = connector(
        rl_module=rl_module, batch=batch, episodes=[object()], explore=False
    )

    assert returned is batch
    assert seen == [(batch, rl_module, False)]
    torch.testing.assert_close(batch["actions"], torch.ones(3))
    assert connector.rng_states == {}


@pytest.mark.unit
def test_builder_replaces_rllib_get_actions_and_forwards_configuration():
    calls = []

    class Pipeline:
        def remove(self, connector_type):
            calls.append(("remove", connector_type))

        def prepend(self, connector):
            calls.append(("prepend", connector))

    pipeline = Pipeline()

    class Config:
        def build_module_to_env_connector(self, env=None, spaces=None):
            calls.append(("build", self, env, spaces))

            return pipeline

    cfg = Config()
    env, spaces = object(), object()
    result = build_iid_module_to_env_connector(cfg, env=env, spaces=spaces)

    assert result is pipeline
    assert calls[0] == ("build", cfg, env, spaces)
    assert calls[1] == ("remove", iid_module.GetActions)
    assert calls[2][0] == "prepend"
    assert isinstance(calls[2][1], IidGetActions)
