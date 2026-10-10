"""``RayOptimizerConfig``: agent expansion into RLModules and ``build_optimizer``.

``_apply_agents_to_rllib`` is exercised on a real PPO ``AlgorithmConfig`` (no
algorithm is built from it). ``build_optimizer`` runs with a fake ``World``
(same ``<method>.remote(...)`` call shape, ``ray.get`` replaced by the
identity), ``register_env`` recording the creator it is given, and
``RLlibMultiAgentEnvAdapter`` replaced by a wrapper that keeps the environment,
so the inner ``env_creator`` can be called as RLlib would call it. Tests that
need the real ``RayOptimizer`` stub its ``PolicyActor``. Ray never starts.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import ray
import torch
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

import core.adaptors.ray.optimizer_config as optimizer_config_module
import core.adaptors.ray.policy_actor as policy_actor_module
from core.adaptors.ray.optimizer import RayOptimizer
from core.agents.base import AgentConfig
from core.callbacks import _evaluate_with_fixed_duration_once
from core.mechanism.config import MechanismConfig
from core.metrics.schemas import MetricSchema
from core.optimizers.ppo.config import PPOptimizerConfig
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query

OBS = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
ACT = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
ACT_DICT = spaces.Dict({"harvest": ACT})


def fisher(agent_id: str = "fisher", count: int = 2, policy: str | None = None):
    return AgentConfig(
        id=agent_id,
        policy_id=policy or agent_id,
        mechanisms=MechanismConfig(action_space=ACT, id="harvest"),
        count=count,
        observation_space=OBS,
    )


# --------------------------------------------------------------------------- #
# _apply_agents_to_rllib
# --------------------------------------------------------------------------- #


def applied_config(*, num_envs=4, seeds=(11, 22), agents=None):
    """Config whose ops were replayed by hand, ready for ``_apply_agents_to_rllib``."""
    cfg = PPOptimizerConfig().agents(agents or (fisher(count=2),))
    cfg.seeds = list(seeds)
    cfg.rllib_cfg = PPO.get_default_config().env_runners(
        num_envs_per_env_runner=num_envs
    )

    return cfg


@pytest.mark.unit
def test_one_rl_module_is_declared_per_mechanism_and_training_seed():
    cfg = applied_config(num_envs=4, seeds=(11, 22))

    agents = cfg._apply_agents_to_rllib()

    expected = {f"fisher_m{m}_s{s}" for m in (0, 1) for s in (11, 22)}
    spec = cfg.rllib_cfg.rl_module_spec
    assert isinstance(spec, MultiRLModuleSpec)
    assert set(spec.rl_module_specs) == expected
    assert set(cfg.rllib_cfg.policies) == expected
    assert set(cfg.rllib_cfg.policies_to_train) == expected

    module = spec.rl_module_specs["fisher_m0_s11"]
    assert module.observation_space == OBS
    assert module.action_space == ACT_DICT
    assert module.model_config.vf_share_layers is False
    assert module.model_config.fcnet_bias_initializer == "zeros_"
    assert callable(module.model_config.fcnet_kernel_initializer)
    assert callable(module.model_config.head_fcnet_kernel_initializer)

    assert list(agents) == ["fisher:0", "fisher:1"]


@pytest.mark.unit
def test_every_agent_instance_becomes_a_single_agent_config():
    cfg = applied_config(agents=(fisher(count=2), fisher("boat", count=1)))

    agents = cfg._apply_agents_to_rllib()

    assert list(agents) == ["fisher:0", "fisher:1", "boat:0"]
    assert agents["fisher:1"] == replace(
        cfg.agents_cfgs["fisher"], id="fisher:1", count=1
    )
    assert all(agent.count == 1 for agent in agents.values())
    # The declaration itself is left untouched.
    assert cfg.agents_cfgs["fisher"].id == "fisher"
    assert cfg.agents_cfgs["fisher"].count == 2


@pytest.mark.unit
def test_instance_spaces_are_written_into_the_env_config():
    cfg = applied_config(agents=(fisher(count=2),))

    cfg._apply_agents_to_rllib()

    assert cfg.env_config["observation_spaces"] == {"fisher:0": OBS, "fisher:1": OBS}
    assert cfg.env_config["action_spaces"] == {
        "fisher:0": ACT_DICT,
        "fisher:1": ACT_DICT,
    }


@pytest.mark.unit
def test_two_modules_of_one_seed_start_from_identical_initializers():
    cfg = applied_config(num_envs=2, seeds=(11,))

    cfg._apply_agents_to_rllib()

    specs = cfg.rllib_cfg.rl_module_spec.rl_module_specs
    inits = [
        specs[f"fisher_m{m}_s11"].model_config.fcnet_kernel_initializer for m in (0, 1)
    ]
    assert inits[0] is not inits[1]

    first, second = torch.empty(4, 3), torch.empty(4, 3)
    inits[0](first)
    inits[1](second)
    assert torch.equal(first, second)


@pytest.mark.unit
def test_policy_mapping_routes_an_agent_to_the_module_named_by_its_episode_id():
    cfg = applied_config(
        num_envs=4, seeds=(11, 22), agents=(fisher(count=1), fisher("boat", count=1))
    )
    cfg._apply_agents_to_rllib()
    mapping = cfg.rllib_cfg.policy_mapping_fn
    episode = SimpleNamespace(id_="env=3|m=1|ps=22|ss=99|raw=abc")

    assert mapping("fisher:0", episode) == "fisher_m1_s22"
    assert mapping("boat:0", episode) == "boat_m1_s22"


@pytest.mark.unit
def test_policy_mapping_rejects_an_unknown_module_and_names_the_candidates():
    cfg = applied_config(num_envs=4, seeds=(11, 22))
    cfg._apply_agents_to_rllib()
    unknown = SimpleNamespace(id_="env=3|m=7|ps=22|ss=22|raw=abc")

    with pytest.raises(RuntimeError, match="Unknown policy") as raised:
        cfg.rllib_cfg.policy_mapping_fn("fisher:0", unknown)

    message = str(raised.value)
    assert "policy_id=fisher_m7_s22" in message
    assert "episode_id=env=3|m=7|ps=22|ss=22|raw=abc" in message
    assert "fisher_m0_s11" in message


@pytest.mark.unit
def test_policy_mapping_rejects_an_untagged_episode_id():
    cfg = applied_config()
    cfg._apply_agents_to_rllib()

    with pytest.raises(RuntimeError, match="missing identity keys"):
        cfg.rllib_cfg.policy_mapping_fn("fisher:0", SimpleNamespace(id_="plain-id"))


@pytest.mark.unit
def test_an_env_count_that_is_not_a_multiple_of_the_seeds_is_rejected():
    cfg = applied_config(num_envs=3, seeds=(11, 22))

    with pytest.raises(ValueError, match="num_envs_per_env_runner=3 .* num_seeds=2"):
        cfg._apply_agents_to_rllib()


@pytest.mark.unit
def test_a_none_seed_gives_the_default_initializer_and_a_none_module_suffix():
    cfg = applied_config(num_envs=2, seeds=(None,), agents=(fisher(count=1),))

    cfg._apply_agents_to_rllib()

    specs = cfg.rllib_cfg.rl_module_spec.rl_module_specs
    assert set(specs) == {"fisher_m0_sNone", "fisher_m1_sNone"}
    assert specs["fisher_m0_sNone"].model_config.fcnet_kernel_initializer == (
        "xavier_uniform_"
    )


# --------------------------------------------------------------------------- #
# build_optimizer
# --------------------------------------------------------------------------- #


class RecordingEnv:
    """Environment class recording the keyword arguments it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class WrappedEnv:
    """Stand-in for ``RLlibMultiAgentEnvAdapter``: keeps the wrapped env."""

    def __init__(self, env, worker_index=0):
        self.env = env
        self.worker_index = worker_index


class RecordingReporter(Reporter):
    def __init__(self, label):
        self.label = label

    def _report(self, query, x, y, **kwargs):
        pass

    def close(self):
        pass


class RecordingReporterConfig(ReporterConfig):
    """Builds ``RecordingReporter`` instances and remembers them."""

    built: list[RecordingReporter] = []

    def build(self, *, label=None):
        reporter = RecordingReporter(label)
        RecordingReporterConfig.built.append(reporter)

        return reporter


class EnvSchema(MetricSchema):
    """Env-level reporting schema."""


OPT_QUERY = Query(title="opt", x=("iter",), y=("iter",))
ENV_QUERY = Query(title="env", x=("iter",), y=("iter",))


class FakeWorld:
    """``World`` stand-in with the calls ``build_optimizer`` makes."""

    def __init__(self):
        self.opt_ids: list[str] = []
        self.get_opt_registry = SimpleNamespace(remote=lambda: set(self.opt_ids))
        self._set_new_opt_id = SimpleNamespace(remote=self._register)
        self.flush = SimpleNamespace(remote=lambda status=None: None)

    def _register(self, opt_id):
        self.opt_ids.append(opt_id)

        return opt_id


class FakeEnvContext(dict):
    """``EnvContext`` stand-in: a dict with a ``worker_index`` attribute."""

    def __init__(self, *args, worker_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_index = worker_index


class StubOptimizer:
    """``RayOptimizer`` stand-in keeping what ``build_optimizer`` hands over."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = kwargs["config"]
        self.id = None


@pytest.fixture
def registered(monkeypatch) -> dict:
    """Patch ``register_env``, the env adapter and ``ray.get``; return the registry."""
    registry: dict = {}
    monkeypatch.setattr(
        optimizer_config_module,
        "register_env",
        lambda name, fn: registry.update({name: fn}),
    )
    monkeypatch.setattr(
        optimizer_config_module, "RLlibMultiAgentEnvAdapter", WrappedEnv
    )
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)
    RecordingReporterConfig.built = []

    return registry


@pytest.fixture
def stub_optimizer(monkeypatch) -> None:
    """Replace ``RayOptimizer`` in the config module by ``StubOptimizer``."""
    monkeypatch.setattr(optimizer_config_module, "RayOptimizer", StubOptimizer)


def society(
    *, mechanisms=2, train_seeds=2, eval_seeds=(7,), reporting=False, reporter=True
) -> PPOptimizerConfig:
    cfg = (
        PPOptimizerConfig()
        .environment(
            env=RecordingEnv,
            horizon=5,
            env_config={"alpha": 0.5},
            queries=(ENV_QUERY,) if reporting else None,
            schema=EnvSchema if reporting else None,
        )
        .env_runners(num_envs_per_env_runner=mechanisms)
        .debugging(seed=3, num_seeds=train_seeds)
        .agents(fisher(count=2))
        .training(episodes=1)
    )

    if eval_seeds is not None:
        cfg.evaluation(
            seeds=list(eval_seeds),
            evaluation_duration=1,
            evaluation_config={"rollout_fragment_length": 5},
        )

    if reporting:
        cfg.reporting(queries=(OPT_QUERY,))

    if reporter:
        cfg.reporter_cfg = RecordingReporterConfig(project="unit")

    return cfg


def build(cfg, world=None, **kwargs):
    world = world or FakeWorld()

    return cfg.build_optimizer(world=world, world_name="w", **kwargs), world


@pytest.mark.unit
def test_build_resolves_the_rllib_config_and_registers_the_environment(
    registered, stub_optimizer
):
    cfg = society(mechanisms=2, train_seeds=2, eval_seeds=(7, 8))

    opt, world = build(cfg)

    # One evaluation runner per (evaluation seed, training seed), each hosting
    # one environment per mechanism and running one episode per environment.
    rllib = cfg.rllib_cfg
    assert rllib.num_envs_per_env_runner == 4
    assert rllib.evaluation_num_env_runners == 4
    assert rllib.evaluation_config["num_envs_per_env_runner"] == 2
    assert rllib.evaluation_duration == 4 * 2
    assert rllib.custom_evaluation_function is _evaluate_with_fixed_duration_once
    assert rllib.evaluation_interval is None

    (env_name,) = registered
    assert env_name.startswith("regulated_env_")
    assert rllib.env == env_name
    assert rllib.env_config["alpha"] == 0.5
    assert set(rllib.env_config["observation_spaces"]) == {"fisher:0", "fisher:1"}
    assert cfg.world_name == "w"

    assert isinstance(opt, StubOptimizer)
    assert opt.kwargs["world"] is world
    assert opt.id == world.opt_ids[-1]


@pytest.mark.unit
def test_the_optimizer_receives_a_deep_copy_of_the_config(registered, stub_optimizer):
    cfg = society()

    opt, _ = build(cfg)

    assert opt.config is not cfg
    assert opt.config.seeds == cfg.seeds
    assert opt.config.rllib_cfg is not cfg.rllib_cfg
    opt.config.seeds.append(1)
    assert len(cfg.seeds) == 2


@pytest.mark.unit
def test_each_build_draws_its_own_optimizer_id_and_environment_name(
    registered, stub_optimizer
):
    cfg = society()

    first, world = build(cfg)
    second, _ = build(cfg, world)

    assert first.id != second.id
    assert world.opt_ids == [first.id, second.id]
    assert len(registered) == 2


@pytest.mark.unit
def test_a_second_build_reuses_the_resolved_rllib_config(registered, stub_optimizer):
    cfg = society()
    build(cfg)
    resolved = cfg.rllib_cfg
    cfg._cfg_ops["resources"] = optimizer_config_module.RLlibConfigOp(
        fn=lambda c, **k: pytest.fail("ops must not be replayed twice"),
        args=(),
        kwargs={},
    )

    build(cfg)

    assert cfg.rllib_cfg is resolved


@pytest.mark.unit
def test_build_without_evaluation_leaves_the_evaluation_config_alone(
    registered, stub_optimizer
):
    cfg = society(eval_seeds=None)

    build(cfg)

    assert "_evaluation_rllib" not in cfg._cfg_ops
    assert cfg.rllib_cfg.custom_evaluation_function is None


@pytest.mark.unit
def test_build_without_a_world_name_fails_before_touching_the_world(
    registered, stub_optimizer
):
    world = FakeWorld()

    with pytest.raises(ValueError, match="world_name must be provided"):
        society().build_optimizer(world=world, world_name=None)

    assert world.opt_ids == []
    assert registered == {}


@pytest.mark.unit
def test_build_without_an_optimizer_class_is_rejected(registered):
    cfg = society()
    cfg.opt_class = None

    with pytest.raises(ValueError, match="OptimizerConfig has no opt_class"):
        build(cfg)


@pytest.mark.unit
def test_disable_env_checking_is_replayed_on_the_rllib_config(
    registered, stub_optimizer
):
    cfg = society()
    cfg.environment(env=RecordingEnv, horizon=5, disable_env_checking=True)

    build(cfg)

    assert cfg.rllib_cfg.disable_env_checking is True


@pytest.mark.unit
def test_building_without_training_seeds_fails_with_a_clear_error(
    registered, stub_optimizer
):
    # Without ``debugging(seed=...)`` the seed list is empty. One RLModule is
    # declared per training seed, so such a config would declare none; it used
    # to leak a ``ZeroDivisionError`` from the module layout.
    cfg = (
        PPOptimizerConfig()
        .environment(env=RecordingEnv, horizon=5)
        .env_runners(num_envs_per_env_runner=2)
        .agents(fisher())
    )
    cfg.reporter_cfg = RecordingReporterConfig(project="unit")

    with pytest.raises(ValueError, match=r"no training seeds.*\.debugging\(seed="):
        build(cfg)


# --- the inner env_creator ------------------------------------------------- #


def env_creator_of(registered):
    (creator,) = registered.values()

    return creator


@pytest.mark.unit
def test_training_envs_are_laid_out_by_mechanism_then_training_seed(
    registered, stub_optimizer
):
    cfg = society(mechanisms=2, train_seeds=2)
    _, world = build(cfg)
    env_creator = env_creator_of(registered)
    seeds = cfg.seeds

    # Four environments per runner: mechanism k % 2, seed index k // 2; the
    # local counter then wraps around for the next runner's worth of envs.
    expected = [
        (0, seeds[0]),
        (1, seeds[0]),
        (0, seeds[1]),
        (1, seeds[1]),
        (0, seeds[0]),
    ]
    envs = []
    for mechanism_idx, seed in expected:
        wrapped = env_creator(FakeEnvContext({"alpha": 0.5}, worker_index=1))
        assert isinstance(wrapped, WrappedEnv)
        env = wrapped.env
        envs.append(env)
        assert env.kwargs["mechanism_id"] == mechanism_idx
        assert env.kwargs["seed"] == seed
        assert env.kwargs["policy_seed"] == seed

    kwargs = envs[0].kwargs
    assert kwargs["world"] is world
    assert kwargs["opt_id"] == world.opt_ids[-1]
    assert list(kwargs["agents_cfg_dict"]) == ["fisher:0", "fisher:1"]
    assert kwargs["env_name"] == cfg.rllib_cfg.env
    assert kwargs["alpha"] == 0.5
    # ``mode`` is read from the context (default ``train``), never written back.
    assert "mode" not in kwargs


@pytest.mark.unit
def test_evaluation_envs_take_their_seeds_from_the_runner_index(
    registered, stub_optimizer
):
    cfg = society(mechanisms=2, train_seeds=2, eval_seeds=(70, 80))
    build(cfg)
    env_creator = env_creator_of(registered)
    train_seeds, eval_seeds = cfg.seeds, cfg.eval_seeds

    # Runner r = worker_index - 1 tests training seed r % 2 under evaluation
    # seed r // 2; mechanisms cycle with the per-process evaluation counter.
    cases = [
        (1, 0, train_seeds[0], eval_seeds[0]),
        (2, 1, train_seeds[1], eval_seeds[0]),
        (3, 0, train_seeds[0], eval_seeds[1]),
        (4, 1, train_seeds[1], eval_seeds[1]),
    ]
    for worker_index, mechanism_idx, policy_seed, env_seed in cases:
        wrapped = env_creator(
            FakeEnvContext({"mode": "eval"}, worker_index=worker_index)
        )
        env = wrapped.env
        # The exploration keys read the runner index from the adapter.
        assert wrapped.worker_index == worker_index
        assert env.kwargs["mode"] == "eval"
        assert env.kwargs["mechanism_id"] == mechanism_idx
        assert env.kwargs["policy_seed"] == policy_seed
        assert env.kwargs["seed"] == env_seed

    # The training counter is independent of the evaluation one.
    train_env = env_creator(FakeEnvContext({}, worker_index=1)).env
    assert train_env.kwargs["mechanism_id"] == 0


@pytest.mark.unit
def test_evaluation_envs_without_evaluation_seeds_run_unseeded(
    registered, stub_optimizer
):
    cfg = society(eval_seeds=None)
    build(cfg)

    env = env_creator_of(registered)(
        FakeEnvContext({"mode": "eval"}, worker_index=1)
    ).env

    assert env.kwargs["seed"] is None
    assert env.kwargs["policy_seed"] == cfg.seeds[0]


# --- reporters ------------------------------------------------------------- #


@pytest.mark.unit
def test_the_optimizer_level_reporter_is_built_once_and_loaded_with_the_declaration(
    registered, stub_optimizer
):
    cfg = society(reporting=True)

    opt, _ = build(cfg)

    reporter = opt.kwargs["reporting"]
    assert RecordingReporterConfig.built == [reporter]
    assert reporter.label == "StubOptimizer"
    assert reporter.queries == (OPT_QUERY,)


@pytest.mark.unit
def test_every_environment_receives_its_own_copy_of_the_reporter_config(
    registered, stub_optimizer
):
    cfg = society(reporting=True)
    build(cfg)
    env_creator = env_creator_of(registered)

    first = env_creator(FakeEnvContext({}, worker_index=1)).env
    second = env_creator(FakeEnvContext({}, worker_index=1)).env

    for env in (first, second):
        reporter_cfg = env.kwargs["reporter_cfg"]
        assert isinstance(reporter_cfg, RecordingReporterConfig)
        assert reporter_cfg is not cfg.reporter_cfg
        assert reporter_cfg.project_name == "unit"
        assert env.kwargs["queries"] == (ENV_QUERY,)
        assert env.kwargs["schema"] is EnvSchema
    assert first.kwargs["reporter_cfg"] is not second.kwargs["reporter_cfg"]
    # Environments get the config, never a built reporter.
    assert len(RecordingReporterConfig.built) == 1


@pytest.mark.unit
def test_a_config_without_reporter_builds_an_optimizer_without_reporter(
    registered, stub_optimizer
):
    # The docstring of ``build_optimizer`` says the optimizer-level reporter is
    # ``None`` when no ``reporter_cfg`` was set (reporting is optional).
    cfg = society(reporter=False)
    assert cfg.reporter_cfg is None

    opt, _ = build(cfg)

    assert "reporting" in opt.kwargs and opt.kwargs["reporting"] is None


@pytest.mark.unit
def test_the_built_optimizer_holds_the_reporter_it_was_built_with(
    registered, monkeypatch
):
    # ``build_optimizer`` documents ``opt.reporting`` as the optimizer-level
    # reporter; ``Optimizer.report_metrics`` renders the queries through it.
    actors = SimpleNamespace(remote=lambda algo_config: MagicMock(name="actor"))
    monkeypatch.setattr(policy_actor_module, "PolicyActor", actors)
    cfg = society(reporting=True)

    opt, _ = build(cfg)

    assert isinstance(opt, RayOptimizer)
    assert opt.reporting is RecordingReporterConfig.built[0]


# --- the config snapshot --------------------------------------------------- #


@pytest.mark.unit
def test_the_snapshot_handed_to_the_optimizer_is_immutable(registered, stub_optimizer):
    # ``OptimizerConfig`` promises that ``build_optimizer`` works on
    # ``copy(copy_frozen=True)``: the optimizer owns an immutable snapshot.
    opt, _ = build(society())

    with pytest.raises(AttributeError, match="frozen"):
        opt.config.episodes = 99
