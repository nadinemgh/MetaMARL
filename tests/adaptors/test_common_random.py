"""Common random numbers across the slots of one policy seed.

The slots of one policy seed must differ only by the candidate mechanism they
train against. Two things used to break that: the pi and vf output layers drew
their initial weights from torch's global stream, so each slot started from
different heads, and the exploration draws of each slot came from that same
global stream, consumed module after module. The tests below build real RLlib
modules and connector pieces from the configuration (no algorithm, no Ray) and
check that slots of one seed build identical weights, that equal exploration
keys give equal actions whatever the slot, and that the keys are derived from
the policy seed, the env runner, the step and the agent, never from the slot.
"""

from __future__ import annotations

from types import SimpleNamespace

import cloudpickle
import numpy as np
import pytest
import torch
from gymnasium import spaces
from ray.rllib.algorithms.appo import APPO
from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import (
    DefaultAPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.connectors.common.agent_to_module_mapping import AgentToModuleMapping
from ray.rllib.connectors.common.batch_individual_items import BatchIndividualItems
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.core.columns import Columns
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

from core.adaptors.ray.common_random import (
    EXPLORATION_KEY,
    AddExplorationKeys,
    CommonRandomAPPOTorchRLModule,
    CommonRandomPPOTorchRLModule,
    SeededHeadsPPOCatalog,
    common_random_module_class,
    exploration_key,
)
from core.agents.base import AgentConfig
from core.mechanism.config import MechanismConfig
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.ppo.config import PPOptimizerConfig

OBS = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
ACT = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
ALGORITHMS = [(PPOptimizerConfig, PPO), (APPOptimizerConfig, APPO)]


def fisher(count: int = 2) -> AgentConfig:
    return AgentConfig(
        id="fisher",
        policy_id="fisher",
        mechanisms=(
            MechanismConfig(action_space=ACT, id="harvest"),
            MechanismConfig(action_space=ACT, id="restore"),
        ),
        count=count,
        observation_space=OBS,
    )


def applied(config_class=PPOptimizerConfig, algo=PPO, *, num_envs=2, seeds=(11,)):
    """Config whose ops were replayed by hand, then expanded into RLModules."""
    cfg = config_class().agents(fisher())
    cfg.seeds = list(seeds)
    cfg.rllib_cfg = algo.get_default_config().env_runners(
        num_envs_per_env_runner=num_envs
    )
    cfg._apply_agents_to_rllib()
    return cfg


def multi_spec(cfg):
    """The expanded config's MultiRLModuleSpec, as an env runner resolves it."""
    return cfg.rllib_cfg.get_multi_rl_module_spec(
        spaces={"__env__": (OBS, ACT), "__env_single__": (OBS, ACT)},
        inference_only=False,
    )


def build(cfg, module_id: str):
    """Build one RLModule from the expanded config, as an env runner would."""
    return multi_spec(cfg).rl_module_specs[module_id].build()


def state(module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


# --------------------------------------------------------------------------- #
# Seeded output layers
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("config_class, algo", ALGORITHMS)
def test_every_slot_of_one_seed_builds_identical_weights(config_class, algo):
    cfg = applied(config_class, algo, num_envs=4, seeds=(11,))

    # Draw from the global stream between builds, as the other modules of a
    # process would; the seeded layers must not depend on it.
    weights = []
    for m in range(4):
        torch.rand(17)
        weights.append(state(build(cfg, f"fisher_m{m}_s11")))

    for other in weights[1:]:
        assert other.keys() == weights[0].keys()
        for key in weights[0]:
            assert torch.equal(other[key], weights[0][key]), key


@pytest.mark.unit
def test_output_layers_follow_the_policy_seed_and_start_with_zero_bias():
    cfg = applied(num_envs=2, seeds=(11, 22))

    first = state(build(cfg, "fisher_m0_s11"))
    second = state(build(cfg, "fisher_m0_s22"))

    for head in ("pi", "vf"):
        weight = next(k for k in first if k.startswith(f"{head}.") and "weight" in k)
        bias = next(k for k in first if k.startswith(f"{head}.") and "bias" in k)
        assert not torch.equal(first[weight], second[weight])
        assert torch.count_nonzero(first[bias]) == 0


@pytest.mark.unit
@pytest.mark.parametrize("config_class, algo", ALGORITHMS)
def test_rebuilding_a_module_gives_the_same_weights(config_class, algo):
    # The learner and every env runner build their own copy of each module,
    # from the same spec or from a pickled copy of it; all copies must agree.
    cfg = applied(config_class, algo, num_envs=2, seeds=(11,))
    spec = multi_spec(cfg).rl_module_specs["fisher_m0_s11"]

    first = state(spec.build())
    again = state(spec.build())
    copied = state(cloudpickle.loads(cloudpickle.dumps(spec)).build())

    for key in first:
        assert torch.equal(again[key], first[key]), key
        assert torch.equal(copied[key], first[key]), key


@pytest.mark.unit
def test_heads_and_encoder_draw_from_separate_seed_streams():
    # Xavier-uniform weights are a scaled copy of the generator's uniform
    # stream, so two layers seeded alike share the signs of their first
    # entries. The heads must not replay the encoder's first layer.
    weights = state(build(applied(num_envs=2, seeds=(11,)), "fisher_m0_s11"))
    encoder = torch.sign(weights["encoder.actor_encoder.net.mlp.0.weight"].flatten())

    for head in ("pi", "vf"):
        layer = next(k for k in weights if k.startswith(f"{head}.") and "weight" in k)
        signs = torch.sign(weights[layer].flatten())
        size = min(signs.numel(), encoder.numel())
        assert not torch.equal(signs[:size], encoder[:size]), layer


@pytest.mark.unit
@pytest.mark.parametrize("config_class, algo", ALGORITHMS)
def test_modules_use_the_common_random_class_and_the_seeded_catalog(config_class, algo):
    cfg = applied(config_class, algo)

    spec = cfg.rllib_cfg.rl_module_spec.rl_module_specs["fisher_m0_s11"]

    expected = {PPO: CommonRandomPPOTorchRLModule, APPO: CommonRandomAPPOTorchRLModule}
    assert spec.module_class is expected[algo]
    assert spec.catalog_class is SeededHeadsPPOCatalog


@pytest.mark.unit
def test_common_random_class_is_chosen_from_the_default_module_class():
    assert (
        common_random_module_class(DefaultPPOTorchRLModule)
        is CommonRandomPPOTorchRLModule
    )
    assert (
        common_random_module_class(DefaultAPPOTorchRLModule)
        is CommonRandomAPPOTorchRLModule
    )

    with pytest.raises(ValueError, match="common random"):
        common_random_module_class(torch.nn.Module)


# --------------------------------------------------------------------------- #
# Exploration draws
# --------------------------------------------------------------------------- #


def explore(module, keys, seed: int = 0):
    """Run ``forward_exploration`` on a fixed batch of ``len(keys)`` rows."""
    generator = torch.Generator().manual_seed(seed)
    batch = {
        Columns.OBS: torch.rand(len(keys), 3, generator=generator),
        EXPLORATION_KEY: torch.as_tensor(keys, dtype=torch.int64),
    }
    return module.forward_exploration(batch)


def actions_of(output) -> torch.Tensor:
    actions = output[Columns.ACTIONS]
    return torch.cat([actions["harvest"], actions["restore"]], dim=-1)


@pytest.mark.unit
@pytest.mark.parametrize("config_class, algo", ALGORITHMS)
def test_slots_with_equal_keys_draw_equal_actions(config_class, algo):
    cfg = applied(config_class, algo, num_envs=2, seeds=(11,))
    slot0, slot1 = build(cfg, "fisher_m0_s11"), build(cfg, "fisher_m1_s11")

    torch.rand(5)  # the global stream differs between the two calls
    first = explore(slot0, [3, 4, 5])
    second = explore(slot1, [3, 4, 5])

    assert torch.equal(actions_of(first), actions_of(second))


@pytest.mark.unit
def test_distinct_keys_draw_distinct_actions_and_rows_are_independent():
    cfg = applied()
    module = build(cfg, "fisher_m0_s11")

    base = actions_of(explore(module, [3, 4, 5]))
    changed = actions_of(explore(module, [3, 9, 5]))

    assert torch.equal(base[0], changed[0])
    assert not torch.equal(base[1], changed[1])
    assert torch.equal(base[2], changed[2])


@pytest.mark.unit
def test_exploration_leaves_the_global_torch_stream_untouched():
    module = build(applied(), "fisher_m0_s11")
    before = torch.random.get_rng_state()

    explore(module, [3, 4, 5])

    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.unit
def test_logp_is_the_log_density_of_the_drawn_actions():
    module = build(applied(), "fisher_m0_s11")

    output = explore(module, [3, 4, 5])

    dist = module.get_exploration_action_dist_cls().from_logits(
        output[Columns.ACTION_DIST_INPUTS]
    )
    expected = dist.logp(output[Columns.ACTIONS])
    assert torch.allclose(output[Columns.ACTION_LOGP], expected)


@pytest.mark.unit
def test_without_keys_the_draw_is_left_to_rllib():
    module = build(applied(), "fisher_m0_s11")

    output = module.forward_exploration({Columns.OBS: torch.zeros(2, 3)})

    assert Columns.ACTIONS not in output
    assert Columns.ACTION_DIST_INPUTS in output


# --------------------------------------------------------------------------- #
# Exploration keys
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_key_depends_on_seed_worker_step_and_agent():
    base = exploration_key(policy_seed=11, worker_index=0, step=4, agent_id="f:0")

    assert base == exploration_key(
        policy_seed=11, worker_index=0, step=4, agent_id="f:0"
    )
    assert 0 <= base < 2**63
    variants = [
        exploration_key(policy_seed=22, worker_index=0, step=4, agent_id="f:0"),
        exploration_key(policy_seed=11, worker_index=1, step=4, agent_id="f:0"),
        exploration_key(policy_seed=11, worker_index=0, step=5, agent_id="f:0"),
        exploration_key(policy_seed=11, worker_index=0, step=4, agent_id="f:1"),
    ]
    assert len({base, *variants}) == 5


def vector_env(worker_index: int):
    return SimpleNamespace(
        envs=[SimpleNamespace(unwrapped=SimpleNamespace(worker_index=worker_index))]
    )


def episodes(module_ids=("fisher_m0_s11", "fisher_m1_s11")):
    obs = np.zeros(3, dtype=np.float32)
    return [
        MultiAgentEpisode(
            observations=[{"fisher:0": obs, "fisher:1": obs}],
            agent_to_module_mapping_fn=lambda aid, ep, mid=mid: mid,
        )
        for mid in module_ids
    ]


def keys_by_agent_and_module(batch) -> dict[tuple[str, str], int]:
    return {
        (agent, module): items[0]
        for (_, agent, module), items in batch[EXPLORATION_KEY].items()
    }


@pytest.mark.unit
def test_connector_gives_both_slots_the_same_keys_and_advances_its_step():
    piece = AddExplorationKeys(env=vector_env(worker_index=2))
    eps = episodes()

    first = keys_by_agent_and_module(
        piece(rl_module=None, batch={}, episodes=eps, explore=True)
    )
    second = keys_by_agent_and_module(
        piece(rl_module=None, batch={}, episodes=eps, explore=True)
    )

    for agent in ("fisher:0", "fisher:1"):
        assert first[(agent, "fisher_m0_s11")] == first[(agent, "fisher_m1_s11")]
        assert first[(agent, "fisher_m0_s11")] == exploration_key(
            policy_seed=11, worker_index=2, step=0, agent_id=agent
        )
        assert second[(agent, "fisher_m0_s11")] == exploration_key(
            policy_seed=11, worker_index=2, step=1, agent_id=agent
        )
    assert first[("fisher:0", "fisher_m0_s11")] != first[("fisher:1", "fisher_m0_s11")]


@pytest.mark.unit
def test_an_unseeded_run_shares_draws_across_slots_but_not_across_runs():
    eps = episodes(module_ids=("fisher_m0_sNone", "fisher_m1_sNone"))

    run_a = keys_by_agent_and_module(
        AddExplorationKeys(env=vector_env(worker_index=0))(
            rl_module=None, batch={}, episodes=eps, explore=True
        )
    )
    run_b = keys_by_agent_and_module(
        AddExplorationKeys(env=vector_env(worker_index=0))(
            rl_module=None, batch={}, episodes=eps, explore=True
        )
    )

    for agent in ("fisher:0", "fisher:1"):
        assert run_a[(agent, "fisher_m0_sNone")] == run_a[(agent, "fisher_m1_sNone")]
        assert run_a[(agent, "fisher_m0_sNone")] != run_b[(agent, "fisher_m0_sNone")]


@pytest.mark.unit
def test_a_runner_without_environment_builds_the_piece_but_cannot_explore():
    # RLlib builds the pipeline of a local runner that hosts no environment
    # (remote runners sample instead) with env=None.
    piece = AddExplorationKeys(env=None)

    assert piece(rl_module=None, batch={}, episodes=[], explore=False) == {}
    with pytest.raises(RuntimeError, match="without an environment"):
        piece(rl_module=None, batch={}, episodes=episodes(), explore=True)


@pytest.mark.unit
def test_sub_environments_without_runner_index_are_rejected():
    env = SimpleNamespace(envs=[SimpleNamespace(unwrapped=SimpleNamespace())])

    with pytest.raises(ValueError, match="worker_index"):
        AddExplorationKeys(env=env)


@pytest.mark.unit
def test_connector_adds_nothing_and_keeps_its_step_when_not_exploring():
    piece = AddExplorationKeys(env=vector_env(worker_index=0))
    eps = episodes()

    out = piece(rl_module=None, batch={}, episodes=eps, explore=False)
    after = keys_by_agent_and_module(
        piece(rl_module=None, batch={}, episodes=eps, explore=True)
    )

    assert EXPLORATION_KEY not in out
    assert after[("fisher:0", "fisher_m0_s11")] == exploration_key(
        policy_seed=11, worker_index=0, step=0, agent_id="fisher:0"
    )


@pytest.mark.unit
def test_keys_reach_the_module_batch_as_one_int64_row_per_agent():
    eps = episodes()
    spec = multi_spec(applied(num_envs=2, seeds=(11,)))
    multi_module = spec.build()
    shared_data: dict = {}
    batch = AddExplorationKeys(env=vector_env(worker_index=0))(
        rl_module=multi_module, batch={}, episodes=eps, explore=True
    )
    for piece in (
        AgentToModuleMapping(
            rl_module_specs=spec.rl_module_specs, agent_to_module_mapping_fn=None
        ),
        BatchIndividualItems(multi_agent=True),
        NumpyToTensor(),
    ):
        batch = piece(
            rl_module=multi_module,
            batch=batch,
            episodes=eps,
            explore=True,
            shared_data=shared_data,
        )

    for module_id in ("fisher_m0_s11", "fisher_m1_s11"):
        keys = batch[module_id][EXPLORATION_KEY]
        assert keys.dtype == torch.int64
        assert keys.shape == (2,)
    assert torch.equal(
        batch["fisher_m0_s11"][EXPLORATION_KEY].sort().values,
        batch["fisher_m1_s11"][EXPLORATION_KEY].sort().values,
    )


@pytest.mark.unit
def test_env_to_module_connector_puts_the_keys_first_and_keeps_a_user_piece():
    user_piece = object()
    cfg = PPOptimizerConfig().agents(fisher())
    cfg.seeds = [11]
    cfg.rllib_cfg = PPO.get_default_config().env_runners(
        num_envs_per_env_runner=2, env_to_module_connector=lambda env: [user_piece]
    )

    cfg._apply_agents_to_rllib()
    pieces = cfg.rllib_cfg._env_to_module_connector(vector_env(0), None, None)

    assert isinstance(pieces[0], AddExplorationKeys)
    assert pieces[1:] == [user_piece]
