"""``OptimizerConfig``: builders, freeze and copy, serialisation, ``build_optimizer``.

The base config is exercised through ``ESConfig`` (a real subclass) and through
a minimal ``RecordingConfig`` whose optimizer and environment only record how
they were constructed, so ``build_optimizer`` runs against the in-memory
``FakeWorld`` of ``conftest.py`` without Ray.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from core.agents.base import AgentConfig
from core.mechanism.config import MechanismConfig
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.optimizers.es.config import ESConfig
from core.reporting.config import ReporterConfig


class RecordingOptimizer(Optimizer):
    """Optimizer that trains nothing; keeps the keywords it was built with."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.extra = kwargs

    def train(self):
        return {}


class RecordingEnv:
    """Environment that keeps the keyword arguments of its construction."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class RecordingReporterConfig(ReporterConfig):
    """Reporter config whose ``build`` returns a label-recording reporter."""

    def __init__(self, project: str):
        super().__init__(project)
        self.built: list = []

    def build(self, *, label=None):
        reporter = SimpleNamespace(label=label, schema="unset", queries=[])
        reporter.add_query = lambda *queries: reporter.queries.extend(queries)
        self.built.append(reporter)

        return reporter


class RecordingConfig(OptimizerConfig):
    def __init__(self):
        super().__init__(opt_class=RecordingOptimizer)


def regulator(*ids: str) -> AgentConfig:
    return AgentConfig(
        id="regulator",
        policy_id="p",
        mechanisms=tuple(
            MechanismConfig(
                id=i, action_space=spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
            )
            for i in ids
        ),
    )


@pytest.fixture
def reporter_config() -> RecordingReporterConfig:
    return RecordingReporterConfig(project="p")


@pytest.mark.unit
class TestInterface:
    def test_to_dict_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ESConfig().to_dict()

    def test_defaults(self):
        cfg = RecordingConfig()

        assert cfg.opt_class is RecordingOptimizer
        assert cfg._is_frozen is False
        assert cfg.env is None and cfg.env_config == {}
        assert cfg.episodes is None and cfg.seeds == [] and cfg.base_seed is None
        assert cfg.reporter_cfg is None and cfg.agents_cfgs is None

    def test_reporter_config_round_trips_through_its_property(self):
        cfg = RecordingConfig()
        marker = object()

        cfg.reporter_cfg = marker

        assert cfg.reporter_cfg is marker
        assert cfg._reporter_cfg is marker

    def test_training_stores_the_episode_count(self):
        cfg = RecordingConfig()

        cfg.training(episodes=7)
        cfg.training()

        assert cfg.episodes == 7

    def test_training_returns_the_config_for_chaining(self):
        cfg = RecordingConfig()

        assert cfg.training(episodes=7) is cfg


@pytest.mark.unit
class TestEnvironmentBuilder:
    def test_collects_the_environment_class_and_its_keywords(self):
        space = spaces.Box(0.0, 1.0, (2,))
        cfg = ESConfig().environment(
            env=RecordingEnv,
            horizon=7,
            env_config={"a": 1},
            observation_space=space,
            action_space=space,
            disable_env_checking=True,
        )

        assert cfg.env is RecordingEnv
        assert cfg.env_config == {
            "observation_space": space,
            "action_space": space,
            "horizon": 7,
            "a": 1,
        }
        assert cfg.disable_env_checking is True

    def test_queries_and_schema_are_stored_for_the_environment_reporter(self):
        cfg = RecordingConfig().environment(queries=["q1", "q2"], schema=dict)

        assert cfg._reporting_queries_env == ("q1", "q2")
        assert cfg._reporting_schema_env is dict

    def test_a_call_without_env_keeps_the_previous_class(self):
        cfg = RecordingConfig().environment(env=RecordingEnv, env_config={"a": 1})

        cfg.environment(horizon=3)

        assert cfg.env is RecordingEnv

    def test_each_call_resets_the_environment_keywords(self):
        cfg = RecordingConfig().environment(env=RecordingEnv, env_config={"a": 1})

        cfg.environment(horizon=3)

        assert cfg.env_config == {"horizon": 3}

    def test_merge_adds_keywords_and_the_extra_wins(self):
        cfg = RecordingConfig().environment(env_config={"a": 1, "b": 2})

        returned = cfg._merge_env_config({"b": 3, "c": 4})

        assert returned is cfg
        assert cfg.env_config == {"a": 1, "b": 3, "c": 4}

    def test_merge_into_a_missing_env_config(self):
        cfg = RecordingConfig()
        cfg.env_config = None

        cfg._merge_env_config({"a": 1})

        assert cfg.env_config == {"a": 1}

    def test_env_creator_forwards_keywords_to_the_environment_class(self):
        cfg = RecordingConfig().environment(env=RecordingEnv)

        env = cfg._env_creator(world="w", train_iters=2)

        assert isinstance(env, RecordingEnv)
        assert env.kwargs == {"world": "w", "train_iters": 2}


@pytest.mark.unit
class TestSeeds:
    def test_seeds_are_the_seed_sequence_state(self):
        cfg = RecordingConfig().debugging(seed=42, num_seeds=3)

        expected = np.random.SeedSequence(42).generate_state(3).tolist()
        assert cfg.base_seed == 42
        assert cfg.seeds == expected
        assert all(type(seed) is int for seed in cfg.seeds)

    def test_default_is_three_seeds(self):
        assert len(RecordingConfig().debugging(seed=1).seeds) == 3

    def test_seed_count_is_configurable(self):
        assert len(RecordingConfig().debugging(seed=1, num_seeds=5).seeds) == 5

    def test_the_same_base_seed_gives_the_same_seeds(self):
        assert (
            RecordingConfig().debugging(seed=9).seeds
            == RecordingConfig().debugging(seed=9).seeds
        )

    def test_different_base_seeds_give_different_seeds(self):
        assert (
            RecordingConfig().debugging(seed=9).seeds
            != RecordingConfig().debugging(seed=10).seeds
        )

    def test_without_a_seed_the_seeds_are_cleared(self):
        cfg = RecordingConfig().debugging(seed=9)

        cfg.debugging()

        assert cfg.seeds == []
        assert cfg.base_seed == 9


@pytest.mark.unit
class TestReportingAndAgents:
    def test_reporting_records_schema_and_queries(self):
        cfg = RecordingConfig().reporting(queries=["q"], schema=dict)

        assert cfg._reporting_schema is dict
        assert cfg._reporting_queries == ("q",)

    def test_reporting_with_none_keeps_the_previous_declaration(self):
        cfg = RecordingConfig().reporting(queries=["q"], schema=dict)

        cfg.reporting(queries=None)

        assert cfg._reporting_schema is dict
        assert cfg._reporting_queries == ("q",)

    def test_a_single_agent_is_keyed_by_its_id(self):
        agent = regulator("quota")

        cfg = RecordingConfig().agents(agent)

        assert cfg.agents_cfgs == {"regulator": agent}

    def test_a_tuple_of_agents_is_keyed_by_id(self):
        first = AgentConfig(id="a", policy_id="p", mechanisms=regulator("m").mechanisms)
        second = AgentConfig(
            id="b", policy_id="p", mechanisms=regulator("m").mechanisms
        )

        cfg = RecordingConfig().agents((first, second))

        assert cfg.agents_cfgs == {"a": first, "b": second}

    def test_an_empty_tuple_of_agents_is_rejected(self):
        with pytest.raises(ValueError, match="agents cannot be empty"):
            RecordingConfig().agents(())


@pytest.mark.unit
class TestFreezeAndCopy:
    def test_a_frozen_config_refuses_attribute_assignment(self):
        cfg = ESConfig()

        cfg.freeze()

        with pytest.raises(AttributeError, match=r"\(sigma\).*frozen"):
            cfg.sigma = 0.1

    def test_freezing_twice_is_harmless(self):
        cfg = ESConfig()

        cfg.freeze()
        cfg.freeze()

        assert cfg._is_frozen is True

    def test_a_frozen_config_refuses_its_builders(self):
        frozen = ESConfig().copy(copy_frozen=True)

        with pytest.raises(AttributeError, match="frozen"):
            frozen.training(sigma=0.5)

    def test_copy_frozen_leaves_the_original_editable(self):
        cfg = ESConfig().training(sigma=0.3)

        frozen = cfg.copy(copy_frozen=True)
        cfg.sigma = 0.4

        assert frozen._is_frozen and not cfg._is_frozen
        assert frozen.sigma == 0.3

    def test_copy_none_keeps_the_frozen_state(self):
        frozen = ESConfig().copy(copy_frozen=True)

        assert frozen.copy()._is_frozen is True
        assert ESConfig().copy()._is_frozen is False

    def test_copy_unfrozen_makes_it_writable_again(self):
        frozen = ESConfig().training(sigma=0.3).copy(copy_frozen=True)

        thawed = frozen.copy(copy_frozen=False)
        thawed.sigma = 0.1

        assert thawed.sigma == 0.1 and frozen.sigma == 0.3

    def test_unfreezing_also_unfreezes_the_nested_evaluation_config(self):
        cfg = ESConfig()
        cfg.evaluation_config = ESConfig()
        cfg.evaluation_config.freeze()

        frozen = cfg.copy(copy_frozen=True)
        thawed = frozen.copy(copy_frozen=False)

        assert frozen.evaluation_config._is_frozen
        assert not thawed.evaluation_config._is_frozen
        thawed.evaluation_config.sigma = 0.5

    def test_the_copy_is_deep(self):
        cfg = RecordingConfig().environment(env_config={"nested": {"a": 1}})

        copy = cfg.copy()
        copy.env_config["nested"]["a"] = 2

        assert cfg.env_config["nested"]["a"] == 1

    def test_freezing_is_shallow_for_nested_objects(self):
        frozen = (
            RecordingConfig().environment(env_config={"a": 1}).copy(copy_frozen=True)
        )

        frozen.env_config["a"] = 2

        assert frozen.env_config == {"a": 2}


@pytest.mark.unit
class TestSerialisation:
    def test_from_dict_sets_known_attributes_and_skips_unknown_ones(self):
        cfg = ESConfig.from_dict({"sigma": 0.25, "unknown_key": 1})

        assert isinstance(cfg, ESConfig)
        assert cfg.sigma == 0.25
        assert not hasattr(cfg, "unknown_key")

    def test_from_yaml_reads_a_mapping(self, tmp_path):
        path = tmp_path / "es.yaml"
        path.write_text(yaml.safe_dump({"sigma": 0.11, "mean_lr": 0.2}))

        cfg = ESConfig.from_yaml(str(path))

        assert (cfg.sigma, cfg.mean_lr) == (0.11, 0.2)


@pytest.mark.unit
class TestBuildOptimizer:
    def make(self, reporter_config, **env_config):
        cfg = (
            RecordingConfig()
            .environment(env=RecordingEnv, env_config=env_config)
            .agents(regulator("quota"))
            .reporting(queries=["q"], schema=dict)
        )
        cfg.training(episodes=4)
        cfg.reporter_cfg = reporter_config

        return cfg

    def test_without_an_optimizer_class_the_build_fails(self, reporter_config):
        cfg = self.make(reporter_config)
        cfg.opt_class = None

        with pytest.raises(ValueError, match="has no opt_class"):
            cfg.build_optimizer(world=None)

    def test_the_optimizer_owns_a_frozen_snapshot_of_the_config(
        self, fake_world, reporter_config
    ):
        cfg = self.make(reporter_config)

        opt = cfg.build_optimizer(world=fake_world)

        assert isinstance(opt, RecordingOptimizer)
        assert opt.config is not cfg
        assert opt.config._is_frozen and not cfg._is_frozen
        assert opt.episodes == 4
        assert opt.world is fake_world

    def test_the_optimizer_takes_its_id_from_the_world(
        self, fake_world, reporter_config
    ):
        opt = self.make(reporter_config).build_optimizer(world=fake_world)

        assert opt.id == fake_world.opt_ids[0]
        assert isinstance(opt.id, str) and opt.id

    def test_each_build_gets_a_distinct_id(self, fake_world, reporter_config):
        cfg = self.make(reporter_config)

        first = cfg.build_optimizer(world=fake_world)
        second = cfg.build_optimizer(world=fake_world)

        assert first.id != second.id
        assert fake_world.opt_ids == [first.id, second.id]

    def test_the_environment_is_built_with_the_optimizer_context(
        self, fake_world, reporter_config
    ):
        cfg = self.make(reporter_config, horizon=9)
        inner = object()

        opt = cfg.build_optimizer(world=fake_world, inner_opt=inner)

        assert isinstance(opt.env, RecordingEnv)
        kwargs = opt.env.kwargs
        assert kwargs["world"] is fake_world
        assert kwargs["opt_id"] == opt.id
        assert kwargs["optimizer"] is inner
        assert kwargs["agents_cfgs"] is cfg.agents_cfgs
        assert kwargs["horizon"] == 9
        assert kwargs["queries"] is None and kwargs["schema"] is None

    def test_the_environment_gets_its_own_reporter_config(
        self, fake_world, reporter_config
    ):
        opt = self.make(reporter_config).build_optimizer(world=fake_world)

        env_reporter_cfg = opt.env.kwargs["reporter_cfg"]

        assert isinstance(env_reporter_cfg, RecordingReporterConfig)
        assert env_reporter_cfg is not reporter_config
        assert env_reporter_cfg is not opt.config.reporter_cfg

    def test_environment_level_queries_and_schema_reach_the_environment(
        self, fake_world, reporter_config
    ):
        cfg = self.make(reporter_config).environment(
            env=RecordingEnv, queries=["eq"], schema=list
        )

        opt = cfg.build_optimizer(world=fake_world)

        assert opt.env.kwargs["queries"] == ("eq",)
        assert opt.env.kwargs["schema"] is list

    def test_the_optimizer_level_reporter_is_built_and_configured(
        self, fake_world, reporter_config
    ):
        self.make(reporter_config).build_optimizer(world=fake_world)

        (reporter,) = reporter_config.built
        assert reporter.label == "RecordingOptimizer"
        assert reporter.schema is dict
        assert reporter.queries == ["q"]

    def test_the_built_reporter_is_attached_to_the_optimizer(
        self, fake_world, reporter_config
    ):
        opt = self.make(reporter_config).build_optimizer(world=fake_world)

        assert opt.reporting is reporter_config.built[0]

    def test_a_config_without_reporter_builds_an_optimizer_without_reporter(
        self, fake_world
    ):
        # Reporting is optional: no reporter config means no reporter at any
        # level, not a crash on ``None.build``.
        opt = self.make(None).build_optimizer(world=fake_world)

        assert opt.reporting is None
        assert opt.env.kwargs["reporter_cfg"] is None

    def test_a_build_without_a_world_leaves_the_id_unset(self, reporter_config):
        opt = self.make(reporter_config).build_optimizer(world=None)

        assert opt.opt_id is None
        assert opt.env.kwargs["opt_id"] is None
