"""``RayOptimizerConfig`` builders: what they record, before any RLlib config exists.

Every RLlib-facing builder only stores a deferred ``RLlibConfigOp``; the ops are
replayed on ``algo_class.get_default_config()`` when ``build_optimizer`` runs
(see ``test_optimizer_config_build.py``). These tests check what is recorded
and, by replaying each op on a ``MagicMock`` standing in for an
``AlgorithmConfig``, that it would call the matching RLlib method with the
recorded arguments. They also cover the seed bookkeeping of ``evaluation`` and
``debugging``, the episode-identity parser and the seeded initializer. No RLlib
algorithm is built and Ray is never started.

A builder called twice keeps only its last call. ``debugging`` and
``env_runners`` may be called in either order, and ``debugging`` any number of
times: the recorded ``num_envs_per_env_runner`` is always the mechanism count
times the number of training seeds.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from gymnasium import spaces

from core.adaptors.ray.optimizer import RayOptimizer
from core.adaptors.ray.optimizer_config import RayOptimizerConfig, RLlibConfigOp
from core.agents.base import AgentConfig
from core.mechanism.config import MechanismConfig
from core.metrics.schemas import MetricSchema
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.ppo.config import PPOptimizerConfig
from core.reporting.query import Query

BOX = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


class OptSchema(MetricSchema):
    """Schema for the optimizer-level reporting declaration."""


QUERY = Query(title="opt", x=("iter",), y=("iter",))


def fisher(agent_id: str = "fisher", count: int = 1) -> AgentConfig:
    return AgentConfig(
        id=agent_id,
        policy_id=agent_id,
        mechanisms=MechanismConfig(action_space=BOX, id="harvest"),
        count=count,
        observation_space=BOX,
    )


@pytest.mark.unit
def test_a_subclass_without_algo_class_is_rejected():
    class NoAlgo(RayOptimizerConfig):
        pass

    with pytest.raises(ValueError, match="NoAlgo must define `algo_class`"):
        NoAlgo()


@pytest.mark.unit
@pytest.mark.parametrize("config_cls", [PPOptimizerConfig, APPOptimizerConfig])
def test_a_fresh_config_is_unresolved_and_targets_ray_optimizer(config_cls):
    cfg = config_cls()

    assert cfg.opt_class is RayOptimizer
    assert cfg._cfg_ops == {}
    assert cfg.rllib_cfg is None
    assert cfg.world_name is None and cfg.num_mechanisms is None
    assert cfg.seeds == [] and cfg.eval_seeds is None


# ``(builder, RLlib method it replays, key of the recorded op)``
PASSTHROUGH_BUILDERS = [
    ("validate", "validate", "validate"),
    ("get_config_for_module", "get_config_for_module", "get_config_for_module"),
    ("python_environment", "python_environment", "python_environment"),
    ("resources", "resources", "resources"),
    ("framework", "framework", "framework"),
    ("api_stack", "api_stack", "api_stack"),
    ("learners", "learners", "learners"),
    ("callbacks", "callbacks", "callbacks"),
    ("offline_data", "offline_data", "offline_data"),
    ("multi_agent", "multi_agent", "multi_agent"),
    ("checkpointing", "checkpointing", "checkpointing"),
    ("fault_tolerance", "fault_tolerance", "fault_tolerance"),
    ("rl_module", "rl_module", "rl_module"),
    ("experimental", "experimental", "experimental"),
    ("_evaluation_rllib", "evaluation", "_evaluation_rllib"),
    ("_reporting_rllib", "reporting", "_reporting_rllib"),
    ("_env_runners", "env_runners", "_env_runners"),
    ("_training_rllb", "training", "_training_rllb"),
]


@pytest.mark.unit
@pytest.mark.parametrize("builder, rllib_method, op_key", PASSTHROUGH_BUILDERS)
def test_a_builder_records_a_deferred_call_that_replays_on_the_rllib_config(
    builder, rllib_method, op_key
):
    cfg = PPOptimizerConfig()

    returned = getattr(cfg, builder)(opt=1)

    assert returned is cfg
    assert list(cfg._cfg_ops) == [op_key]
    op = cfg._cfg_ops[op_key]
    assert isinstance(op, RLlibConfigOp)
    assert op.args == () and op.kwargs == {"opt": 1}

    rllib_cfg = MagicMock(name="AlgorithmConfig")
    replayed = op(rllib_cfg)

    getattr(rllib_cfg, rllib_method).assert_called_once_with(opt=1)
    assert replayed is getattr(rllib_cfg, rllib_method).return_value


@pytest.mark.unit
def test_a_builder_called_twice_keeps_only_its_last_call():
    cfg = PPOptimizerConfig().resources(num_gpus=0).resources(num_gpus=1)

    assert cfg._cfg_ops["resources"].kwargs == {"num_gpus": 1}


@pytest.mark.unit
def test_ops_replay_in_the_order_the_builders_were_first_called():
    cfg = PPOptimizerConfig().resources(num_gpus=0).framework(framework="torch")
    cfg.resources(num_gpus=2)

    assert list(cfg._cfg_ops) == ["resources", "framework"]


@pytest.mark.unit
def test_model_merges_its_kwargs_into_the_rllib_model_dict():
    cfg = PPOptimizerConfig()

    assert cfg.model(fcnet_hiddens=[8, 8]) is cfg

    rllib_cfg = MagicMock(name="AlgorithmConfig")
    rllib_cfg.model = {"existing": True}
    assert cfg._cfg_ops["model"](rllib_cfg) is rllib_cfg
    assert rllib_cfg.model == {"existing": True, "fcnet_hiddens": [8, 8]}


@pytest.mark.unit
def test_training_stores_episodes_and_forwards_only_the_rllib_hyperparameters():
    cfg = PPOptimizerConfig()

    assert cfg.training(episodes=3, lr=1e-3, gamma=0.9) is cfg

    assert cfg.episodes == 3
    assert cfg._cfg_ops["_training_rllb"].kwargs == {"lr": 1e-3, "gamma": 0.9}


@pytest.mark.unit
def test_training_without_episodes_keeps_the_previous_count():
    cfg = PPOptimizerConfig().training(episodes=4).training(lr=1e-3)

    assert cfg.episodes == 4


@pytest.mark.unit
def test_freeze_freezes_this_config_and_records_nothing_for_rllib():
    cfg = PPOptimizerConfig()

    cfg.freeze()

    assert cfg._is_frozen is True
    assert "freeze" not in cfg._cfg_ops
    with pytest.raises(AttributeError, match="frozen"):
        cfg.episodes = 99


@pytest.mark.unit
def test_copy_with_copy_frozen_returns_a_frozen_copy_and_leaves_the_original():
    cfg = PPOptimizerConfig().resources(num_gpus=0)

    frozen = cfg.copy(copy_frozen=True)

    assert frozen._is_frozen is True and cfg._is_frozen is False
    assert list(frozen._cfg_ops) == ["resources"]
    with pytest.raises(AttributeError, match="frozen"):
        frozen.episodes = 99


@pytest.mark.unit
def test_env_runners_records_the_mechanism_count_and_defaults_it_to_one():
    cfg = PPOptimizerConfig().env_runners(num_envs_per_env_runner=4)
    assert cfg.num_mechanisms == 4

    cfg = PPOptimizerConfig().env_runners(num_env_runners=2)
    assert cfg.num_mechanisms == 1
    assert cfg._cfg_ops["_env_runners"].kwargs == {"num_env_runners": 2}


@pytest.mark.unit
def test_reporting_stores_the_declaration_and_defers_the_rllib_kwargs():
    cfg = PPOptimizerConfig()

    returned = cfg.reporting(
        queries=(QUERY,), schema=OptSchema, metrics_num_episodes_for_smoothing=5
    )

    assert returned is cfg
    assert cfg._reporting_schema is OptSchema
    assert cfg._reporting_queries == (QUERY,)
    op = cfg._cfg_ops["_reporting_rllib"]
    assert op.kwargs == {"metrics_num_episodes_for_smoothing": 5}


@pytest.mark.unit
def test_reporting_with_none_keeps_the_previous_declaration():
    cfg = PPOptimizerConfig().reporting(queries=(QUERY,), schema=OptSchema)

    cfg.reporting(queries=None, schema=None)

    assert cfg._reporting_schema is OptSchema
    assert cfg._reporting_queries == (QUERY,)


@pytest.mark.unit
def test_evaluation_with_explicit_seeds_forces_manual_evaluation():
    cfg = PPOptimizerConfig()

    returned = cfg.evaluation(
        seeds=[3, "4"], evaluation_config={"env_config": {"x": 1}}
    )

    assert returned is cfg
    assert cfg.eval_seeds == [3, 4]
    kwargs = cfg._cfg_ops["_evaluation_rllib"].kwargs
    assert kwargs["evaluation_interval"] is None
    assert kwargs["evaluation_parallel_to_training"] is False
    assert kwargs["evaluation_config"]["env_config"] == {"x": 1, "mode": "eval"}


@pytest.mark.unit
def test_evaluation_seeds_win_over_base_seed():
    cfg = PPOptimizerConfig().evaluation(seeds=[9], base_seed=1, num_seeds=4)

    assert cfg.eval_seeds == [9]


@pytest.mark.unit
def test_evaluation_derives_the_seeds_from_the_base_seed():
    seeds = PPOptimizerConfig().evaluation(base_seed=42, num_seeds=3).eval_seeds
    sequence = np.random.SeedSequence(42)
    expected = [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(3)
    ]

    assert seeds == expected
    assert len(set(seeds)) == 3
    assert PPOptimizerConfig().evaluation(base_seed=43, num_seeds=3).eval_seeds != seeds
    # ``num_seeds`` defaults to one.
    assert len(PPOptimizerConfig().evaluation(base_seed=1).eval_seeds) == 1


@pytest.mark.unit
def test_evaluation_without_seeds_leaves_them_unset_but_still_marks_eval_envs():
    cfg = PPOptimizerConfig().evaluation(evaluation_duration=3)

    assert cfg.eval_seeds is None
    kwargs = cfg._cfg_ops["_evaluation_rllib"].kwargs
    assert kwargs["evaluation_config"] == {"env_config": {"mode": "eval"}}
    assert kwargs["evaluation_duration"] == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"seeds": []}, "`seeds` must contain at least one seed"),
        ({"num_seeds": 2}, "`num_seeds` requires `base_seed`"),
    ],
    ids=["empty-seeds", "orphan-num-seeds"],
)
def test_evaluation_rejects_inconsistent_seed_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PPOptimizerConfig().evaluation(**kwargs)


@pytest.mark.unit
def test_debugging_derives_training_seeds_and_scales_the_env_count():
    cfg = PPOptimizerConfig().env_runners(num_envs_per_env_runner=4)

    returned = cfg.debugging(seed=7, num_seeds=3, log_level="ERROR")

    assert returned is cfg
    assert cfg.seeds == np.random.SeedSequence(7).generate_state(3).tolist()
    assert cfg.num_mechanisms == 4
    assert cfg._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 12
    assert cfg._cfg_ops["_debugging_rllib"].kwargs == {"seed": 7, "log_level": "ERROR"}

    rllib_cfg = MagicMock(name="AlgorithmConfig")
    cfg._cfg_ops["_debugging_rllib"](rllib_cfg)
    rllib_cfg.debugging.assert_called_once_with(seed=7, log_level="ERROR")


@pytest.mark.unit
def test_debugging_before_env_runners_scales_the_env_count_all_the_same():
    cfg = PPOptimizerConfig().debugging(seed=7, num_seeds=2)
    cfg.env_runners(num_envs_per_env_runner=5)

    assert len(cfg.seeds) == 2
    assert cfg.num_mechanisms == 5
    assert cfg._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 10


@pytest.mark.unit
def test_debugging_after_env_runners_without_an_env_count_assumes_one_mechanism():
    cfg = PPOptimizerConfig().env_runners(num_env_runners=2).debugging(seed=1)

    assert cfg.num_mechanisms == 1
    assert cfg._cfg_ops["_env_runners"].kwargs == {
        "num_env_runners": 2,
        "num_envs_per_env_runner": 3,
    }


@pytest.mark.unit
def test_calling_debugging_twice_does_not_compound_the_scaling():
    cfg = PPOptimizerConfig().env_runners(num_envs_per_env_runner=4)

    cfg.debugging(seed=7, num_seeds=3).debugging(seed=7, num_seeds=3)

    assert cfg.num_mechanisms == 4
    assert cfg._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 12


@pytest.mark.unit
def test_a_later_debugging_rescales_from_the_mechanism_count():
    cfg = PPOptimizerConfig().env_runners(num_envs_per_env_runner=4)

    cfg.debugging(seed=7, num_seeds=3).debugging(seed=7, num_seeds=2)

    assert cfg._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 8


@pytest.mark.unit
def test_calling_env_runners_again_rescales_from_the_new_mechanism_count():
    cfg = PPOptimizerConfig().env_runners(num_envs_per_env_runner=4)

    cfg.debugging(seed=7, num_seeds=3).env_runners(num_envs_per_env_runner=2)

    assert cfg.num_mechanisms == 2
    assert cfg._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 6


@pytest.mark.unit
def test_debugging_without_a_seed_after_a_seeded_one_restores_the_mechanism_count():
    cfg = PPOptimizerConfig().env_runners(num_envs_per_env_runner=4)

    cfg.debugging(seed=7, num_seeds=3).debugging()

    assert cfg.seeds == []
    assert cfg._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 4


@pytest.mark.unit
def test_debugging_without_a_seed_clears_the_seeds_and_does_not_scale():
    cfg = PPOptimizerConfig().env_runners(num_envs_per_env_runner=5).debugging()

    assert cfg.seeds == []
    assert cfg._cfg_ops["_env_runners"].kwargs["num_envs_per_env_runner"] == 5


@pytest.mark.unit
def test_agents_are_stored_by_id():
    cfg = PPOptimizerConfig()
    a, b = fisher("a"), fisher("b")

    assert cfg.agents((a, b)) is cfg
    assert cfg.agents_cfgs == {"a": a, "b": b}
    assert PPOptimizerConfig().agents(a).agents_cfgs == {"a": a}


@pytest.mark.unit
def test_an_empty_agents_tuple_is_rejected():
    with pytest.raises(ValueError, match="agents cannot be empty"):
        PPOptimizerConfig().agents(())


@pytest.mark.unit
def test_episode_identity_is_parsed_into_string_fields():
    cfg = PPOptimizerConfig()

    identity = cfg._parse_episode_identity("env=3|m=1|ps=101|ss=202|raw=a=b|junk")

    # Segments without ``=`` are ignored and only the first ``=`` splits.
    assert identity == {"env": "3", "m": "1", "ps": "101", "ss": "202", "raw": "a=b"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "episode_id, missing",
    [
        ("env=3|m=1|raw=x", {"ps", "ss"}),
        ("raw-id-from-rllib", {"env", "m", "ps", "ss"}),
    ],
    ids=["partial-tag", "untagged"],
)
def test_an_untagged_episode_id_names_the_missing_keys(episode_id, missing):
    with pytest.raises(RuntimeError, match="missing identity keys") as raised:
        PPOptimizerConfig()._parse_episode_identity(episode_id)

    message = str(raised.value)
    assert episode_id in message
    assert all(key in message for key in missing)


@pytest.mark.unit
def test_an_unseeded_initializer_is_rllibs_default_name():
    assert PPOptimizerConfig()._seeded_xavier_uniform(None) == "xavier_uniform_"


@pytest.mark.unit
def test_the_seeded_initializer_is_reproducible_per_layer_and_spares_the_global_rng():
    cfg = PPOptimizerConfig()
    init_a, init_b, init_c = (cfg._seeded_xavier_uniform(s) for s in (123, 123, 124))

    torch.manual_seed(0)
    state_before = torch.random.get_rng_state()
    a1, a2, b1, b2, c1 = (torch.empty(4, 3) for _ in range(5))
    init_a(a1)
    init_a(a2)
    init_b(b1)
    init_b(b2)
    init_c(c1)

    # Same seed and layer order give the same weights; the counter advances per
    # layer and another seed gives other weights.
    assert torch.equal(a1, b1) and torch.equal(a2, b2)
    assert not torch.equal(a1, a2)
    assert not torch.equal(a1, c1)
    assert torch.equal(torch.random.get_rng_state(), state_before)

    expected = torch.empty(4, 3)
    torch.manual_seed(123)
    torch.nn.init.xavier_uniform_(expected)
    assert torch.equal(a1, expected)
