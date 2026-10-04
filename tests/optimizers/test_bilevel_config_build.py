"""``BilevelConfig``: fluent builders and the composition root ``build_optimizer``.

``build_optimizer`` starts Ray, creates the ``World`` actor, builds both levels
and ties the ES population to the inner capacity. Here ``RayRuntime`` and
``World`` are patched in the ``core.optimizers.bilevel`` namespace and the
society (inner) and regulator (outer) configs are recording stubs, so the
wiring is checked without any actor: reporter hand-off, seed propagation,
leaders hand-off and population sizing.

``build_optimizer`` reads ``society``, ``regulator`` and ``reporter`` without
checking them, so a missing one surfaces as an ``AttributeError`` after Ray has
started; no test asserts either way (see the report).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import core.optimizers.bilevel as bilevel_module
from core.adaptors.ray.runtime import RayRuntimeConfig
from core.optimizers.bilevel import BilevelConfig, BilevelOptimizer
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig
from core.reporting.config import ReporterConfig


class StubOptimizer:
    """Built level exposing only what the composition root touches."""

    def __init__(self, batch_capacity=None):
        self.batch_capacity = batch_capacity


class StubLevelConfig:
    """Recording stand-in for the society or the regulator config.

    ``copy`` returns an independent copy that shares the built optimizer, so a
    test can tell what the composition root wrote into the copy from what it
    left in the original.
    """

    def __init__(self, *, seeds=None, batch_capacity=None, agents_cfgs=None):
        self.env_config: dict = {}
        self.seeds = seeds
        self.agents_cfgs = agents_cfgs
        self.reporter_cfg = None
        self.build_kwargs: dict | None = None
        self.built = StubOptimizer(batch_capacity)
        self.copies: list[StubLevelConfig] = []

    def copy(self, copy_frozen=None):
        duplicate = copy.copy(self)
        duplicate.env_config = dict(self.env_config)
        self.copies.append(duplicate)

        return duplicate

    def _merge_env_config(self, extra):
        self.env_config = {**self.env_config, **extra}

        return self

    def build_optimizer(self, **kwargs):
        self.build_kwargs = kwargs

        return self.built


class StubActorClass:
    """Mimics ``Actor.options(name=...).remote()`` of a Ray actor class."""

    def __init__(self, handle):
        self.handle = handle
        self.options_kwargs: dict | None = None

    def options(self, **kwargs):
        self.options_kwargs = kwargs

        return self

    def remote(self, **kwargs):
        return self.handle


class RecordingReporterConfig(ReporterConfig):
    """Reporter config whose ``build`` returns a label-recording reporter."""

    def build(self, *, label=None):
        return SimpleNamespace(label=label, world=self.world)


@pytest.fixture
def patched_actors(monkeypatch):
    """Patch Ray initialisation and the ``World`` actor class."""
    init_calls = []
    monkeypatch.setattr(
        bilevel_module.RayRuntime,
        "ensure_initialized",
        classmethod(lambda cls, cfg: init_calls.append(cfg)),
    )
    world_handle = SimpleNamespace(kind="world")
    world_cls = StubActorClass(world_handle)
    monkeypatch.setattr(bilevel_module, "World", world_cls)

    return SimpleNamespace(
        init_calls=init_calls, world_cls=world_cls, world_handle=world_handle
    )


def make_config(society, regulator) -> BilevelConfig:
    cfg = (
        BilevelConfig()
        .world(world_name="fishery")
        .society(society)
        .regulator(regulator)
        .reporter(RecordingReporterConfig(project="p"))
    )
    cfg.training(episodes=5)

    return cfg


@pytest.mark.unit
class TestBuilders:
    def test_defaults(self):
        cfg = BilevelConfig()

        assert cfg.opt_class is BilevelOptimizer
        assert cfg.inner_cfg is None and cfg.outer_cfg is None
        assert cfg.world_name is None and cfg.ray_cfg is None
        assert cfg.default_mechanism is None and cfg.output_dir is None
        assert cfg.reporter_cfg is None
        assert cfg.env is None  # inherited fields are present

    def test_a_custom_optimizer_class_is_kept(self):
        class Other(BilevelOptimizer):
            pass

        assert BilevelConfig(opt_class=Other).opt_class is Other

    def test_every_builder_returns_the_config(self):
        cfg = BilevelConfig()

        chained = (
            cfg.world(world_name="w")
            .ray(device="cpu")
            .reporter(RecordingReporterConfig(project="p"))
            .society(PPOptimizerConfig())
            .regulator(ESConfig())
        )

        assert chained is cfg

    def test_society_is_the_inner_level_and_regulator_the_outer_level(self):
        society, regulator = PPOptimizerConfig(), ESConfig()

        cfg = BilevelConfig().society(society).regulator(regulator)

        assert cfg.inner_cfg is society
        assert cfg.outer_cfg is regulator

    def test_none_keeps_the_previous_levels(self):
        society, regulator = PPOptimizerConfig(), ESConfig()
        cfg = BilevelConfig().society(society).regulator(regulator)

        cfg.society(None).regulator(None)

        assert cfg.inner_cfg is society
        assert cfg.outer_cfg is regulator

    def test_the_world_name_gets_a_random_eight_character_suffix(self):
        first = BilevelConfig().world(world_name="fishery").world_name
        second = BilevelConfig().world(world_name="fishery").world_name

        assert first.startswith("fishery_") and len(first) == len("fishery_") + 8
        assert first != second

    def test_a_missing_world_name_keeps_the_previous_one(self):
        cfg = BilevelConfig().world(world_name="w")
        name = cfg.world_name

        cfg.world(world_name=None)

        assert cfg.world_name == name

    def test_the_world_builder_ignores_extra_keywords(self):
        assert (
            BilevelConfig().world(world_name="w", unused=1).world_name.startswith("w_")
        )

    def test_the_ray_builder_records_the_runtime_configuration(self):
        cfg = BilevelConfig().ray(
            device="cpu",
            num_cpus=2,
            omp_threads=4,
            logging_level="INFO",
            runtime_env={"a": 1},
            include_dashboard=False,
        )

        assert isinstance(cfg.ray_cfg, RayRuntimeConfig)
        assert (cfg.ray_cfg.device, cfg.ray_cfg.num_cpus) == ("cpu", 2)
        assert cfg.ray_cfg.num_gpus is None
        assert cfg.ray_cfg.omp_threads == 4
        assert cfg.ray_cfg.logging_level == "INFO"
        assert cfg.ray_cfg.runtime_env == {"a": 1}
        assert cfg.ray_cfg.init_kwargs == {"include_dashboard": False}

    def test_the_ray_builder_defaults(self):
        cfg = BilevelConfig().ray()

        assert (cfg.ray_cfg.device, cfg.ray_cfg.omp_threads) == ("cpu", 1)
        assert cfg.ray_cfg.logging_level == "ERROR"
        assert cfg.ray_cfg.init_kwargs == {}

    def test_the_reporter_builder_stores_the_config(self):
        reporter_cfg = RecordingReporterConfig(project="p")

        assert BilevelConfig().reporter(reporter_cfg).reporter_cfg is reporter_cfg

    def test_the_config_can_be_frozen_and_copied(self):
        cfg = BilevelConfig()
        cfg.training(episodes=3)

        frozen = cfg.copy(copy_frozen=True)

        assert frozen.episodes == 3 and frozen._is_frozen
        with pytest.raises(AttributeError, match="frozen"):
            frozen.training(episodes=4)
        cfg.training(episodes=4)
        assert cfg.episodes == 4


@pytest.mark.unit
class TestBuildOptimizer:
    def test_ray_is_started_with_a_default_runtime_and_the_world_is_named(
        self, patched_actors
    ):
        cfg = make_config(StubLevelConfig(batch_capacity=2), StubLevelConfig())

        cfg.build_optimizer()

        assert len(patched_actors.init_calls) == 1
        assert isinstance(patched_actors.init_calls[0], RayRuntimeConfig)
        assert patched_actors.world_cls.options_kwargs == {"name": cfg.world_name}

    def test_the_configured_runtime_is_used(self, patched_actors):
        cfg = make_config(StubLevelConfig(batch_capacity=2), StubLevelConfig()).ray(
            device="cpu", num_cpus=2, include_dashboard=False
        )

        cfg.build_optimizer()

        assert patched_actors.init_calls == [cfg.ray_cfg]

    def test_both_levels_are_built_against_the_same_world(self, patched_actors):
        society = StubLevelConfig(batch_capacity=2)
        regulator = StubLevelConfig()
        cfg = make_config(society, regulator)

        opt = cfg.build_optimizer()

        (society_copy,) = society.copies
        (regulator_copy,) = regulator.copies
        assert society_copy.build_kwargs == {
            "world": patched_actors.world_handle,
            "world_name": cfg.world_name,
        }
        assert regulator_copy.build_kwargs == {
            "world": patched_actors.world_handle,
            "inner_opt": society.built,
        }
        assert opt.inner is society.built and opt.outer is regulator.built

    def test_the_original_level_configs_are_not_modified(self, patched_actors):
        society = StubLevelConfig(seeds=[1, 2], batch_capacity=2)
        regulator = StubLevelConfig(agents_cfgs={"regulator": object()})

        make_config(society, regulator).build_optimizer()

        assert society.env_config == {} and regulator.env_config == {}
        assert society.reporter_cfg is None and regulator.reporter_cfg is None

    def test_seeds_flow_from_the_society_to_the_regulator(self, patched_actors):
        society = StubLevelConfig(seeds=[1, 2], batch_capacity=2)
        regulator = StubLevelConfig()

        make_config(society, regulator).build_optimizer()

        assert regulator.copies[0].env_config["seeds"] == [1, 2]

    def test_no_seeds_are_forwarded_when_the_society_has_none(self, patched_actors):
        society = StubLevelConfig(seeds=None, batch_capacity=2)
        regulator = StubLevelConfig()

        make_config(society, regulator).build_optimizer()

        assert "seeds" not in regulator.copies[0].env_config

    def test_the_regulators_agents_are_handed_to_the_society_as_leaders(
        self, patched_actors
    ):
        leaders = {"regulator": object()}
        society = StubLevelConfig(batch_capacity=2)
        regulator = StubLevelConfig(agents_cfgs=leaders)

        make_config(society, regulator).build_optimizer()

        assert society.copies[0].env_config["leaders_cfg_dict"] is leaders

    def test_the_regulator_population_is_the_society_capacity(self, patched_actors):
        society = StubLevelConfig(batch_capacity=6)
        regulator = StubLevelConfig()

        make_config(society, regulator).build_optimizer()

        assert regulator.built.batch_capacity == 6

    def test_the_reporter_config_is_stamped_and_copied_to_each_level(
        self, patched_actors
    ):
        society = StubLevelConfig(batch_capacity=2)
        regulator = StubLevelConfig()
        cfg = make_config(society, regulator)

        opt = cfg.build_optimizer()

        assert cfg.reporter_cfg.world == cfg.world_name
        assert opt.reporting.label == "bilevel"
        assert opt.reporting.world == cfg.world_name
        society_reporter = society.copies[0].reporter_cfg
        regulator_reporter = regulator.copies[0].reporter_cfg
        for level_reporter in (society_reporter, regulator_reporter):
            assert isinstance(level_reporter, RecordingReporterConfig)
            assert level_reporter is not cfg.reporter_cfg
            assert level_reporter.world == cfg.world_name
        assert society_reporter is not regulator_reporter

    def test_the_result_is_a_bilevel_optimizer_over_the_built_levels(
        self, patched_actors
    ):
        cfg = make_config(StubLevelConfig(batch_capacity=2), StubLevelConfig())

        opt = cfg.build_optimizer()

        assert isinstance(opt, BilevelOptimizer)
        assert opt.config is cfg
        assert opt.world_name == cfg.world_name
        assert opt.episodes == 5
