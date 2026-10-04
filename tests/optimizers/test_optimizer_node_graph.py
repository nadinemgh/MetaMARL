"""``Optimizer`` as a node of the optimisation graph: identity, links, metrics.

Construction without a config and the unset batch capacity are covered by
``test_optimizer_base.py``; this file covers the rest of the base class: the
write-once identifier, the upstream and downstream links, the environment
hook, the default no-op lifecycle methods and the metric helpers.
"""

from types import SimpleNamespace

import pytest

from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig


class Leaf(Optimizer):
    """Concrete optimizer whose ``train`` returns a marker."""

    def train(self):
        return {"ran": True}


class Hooked(Leaf):
    """Leaf recording every environment handed to the ``_on_env_init`` hook."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def _on_env_init(self, env):
        self.seen.append(env)


def config_with(**attributes) -> OptimizerConfig:
    cfg = OptimizerConfig(opt_class=Leaf)
    for name, value in attributes.items():
        setattr(cfg, name, value)

    return cfg


@pytest.mark.unit
class TestIdentity:
    def test_reading_an_unset_id_is_an_error(self):
        with pytest.raises(RuntimeError, match="Optimizer ID not set"):
            Leaf().id

    def test_the_id_can_be_assigned_once(self):
        opt = Leaf()

        opt.id = "a"

        assert opt.id == "a"
        assert opt.opt_id == "a"
        with pytest.raises(RuntimeError, match="Optimizer ID already set"):
            opt.id = "b"
        assert opt.id == "a"

    def test_the_string_form_names_the_class_and_the_id(self):
        opt = Leaf()
        assert str(opt) == "Leaf(id=None)"

        opt.id = "x1"

        assert str(opt) == "Leaf(id=x1)"


@pytest.mark.unit
class TestConstruction:
    def test_episodes_and_environment_come_from_the_config(self):
        env_class = object
        cfg = config_with(episodes=12, env=env_class)

        opt = Leaf(cfg)

        assert opt.config is cfg
        assert opt.episodes == 12
        assert opt.env is env_class  # the class, until build_optimizer replaces it

    def test_world_and_reporting_come_from_the_keywords(self):
        opt = Leaf(world="world", reporting="reporting")

        assert opt.world == "world"
        assert opt.reporting == "reporting"
        assert opt.logger is None

    def test_extra_keywords_are_ignored(self):
        assert Leaf(reporter="not the reporting keyword").reporting is None

    def test_from_config_builds_the_class_with_that_config(self):
        cfg = config_with(episodes=3)

        opt = Leaf.from_config(cfg)

        assert isinstance(opt, Leaf)
        assert opt.config is cfg


@pytest.mark.unit
class TestGraph:
    def test_links_are_recorded_on_the_right_side(self):
        outer, inner = Leaf(), Leaf()

        outer.set_downstream(inner)
        inner.set_upstream(outer)

        assert outer._downstream == {inner}
        assert inner._upstream == {outer}
        assert outer._upstream == set() and inner._downstream == set()

    def test_linking_twice_keeps_one_entry(self):
        outer, inner = Leaf(), Leaf()

        outer.set_downstream(inner)
        outer.set_downstream(inner)

        assert len(outer._downstream) == 1


@pytest.mark.unit
class TestEnvironmentHook:
    def test_attaching_an_environment_fires_the_hook(self):
        opt = Hooked()

        opt.env = "env"

        assert opt.env == "env"
        assert opt.seen == ["env"]

    def test_detaching_the_environment_does_not_fire_the_hook(self):
        opt = Hooked()
        opt.env = "env"

        opt.env = None

        assert opt.env is None
        assert opt.seen == ["env"]

    def test_the_default_hook_is_a_no_op(self):
        opt = Leaf()

        opt.env = "env"

        assert opt.env == "env"


@pytest.mark.unit
class TestDefaultLifecycle:
    def test_train_of_a_subclass_runs(self):
        assert Leaf().train() == {"ran": True}

    def test_the_abstract_train_raises_when_called_through_super(self):
        class Delegating(Optimizer):
            def train(self):
                return super().train()

        with pytest.raises(NotImplementedError):
            Delegating().train()

    def test_the_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="abstract"):
            Optimizer()

    @pytest.mark.parametrize("method", ["evaluate", "save", "reset", "stop"])
    def test_optional_lifecycle_methods_do_nothing(self, method):
        assert getattr(Leaf(), method)() is None

    def test_there_is_no_default_config(self):
        with pytest.raises(NotImplementedError, match="default config explicitly"):
            Leaf.get_default_config()

    def test_there_is_no_checkpoint_loader(self):
        with pytest.raises(NotImplementedError):
            Leaf.from_checkpoint("checkpoint")


@pytest.mark.unit
class TestMetricHelpers:
    def make(self):
        calls = []
        opt = Leaf()
        opt.logger = SimpleNamespace(
            reduce=lambda: "reduced",
            reset=lambda: calls.append("reset"),
            peek=lambda: "peeked",
        )
        opt.reporting = SimpleNamespace(report=lambda m: calls.append(("report", m)))

        return opt, calls

    def test_reducing_without_a_logger_is_an_error(self):
        with pytest.raises(RuntimeError, match="Leaf has no MetricLogger"):
            Leaf().reduce_metrics()

    def test_flushing_without_a_logger_does_nothing(self):
        assert Leaf().flush_metrics() is None

    def test_reporting_without_a_logger_renders_nothing(self):
        opt = Leaf()
        opt.reporting = SimpleNamespace(report=lambda m: pytest.fail("rendered"))

        opt.report_metrics()

    def test_reporting_without_a_reporter_renders_nothing(self):
        opt, calls = self.make()
        opt.reporting = None

        opt.report_metrics()

        assert calls == []

    def test_reduce_returns_what_the_logger_reduces(self):
        opt, _ = self.make()

        assert opt.reduce_metrics() == "reduced"

    def test_flush_resets_the_logger(self):
        opt, calls = self.make()

        opt.flush_metrics()

        assert calls == ["reset"]

    def test_report_renders_a_peek_not_a_reduction(self):
        opt, calls = self.make()

        opt.report_metrics()

        assert calls == [("report", "peeked")]
