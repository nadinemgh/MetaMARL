"""``MetricLogger``: schema build, push paths, dynamic nodes, peek and reduce.

The schemas come from ``conftest.py``. The logger is built with
``MetricLogger.from_schema``; direct instantiation is forbidden on purpose, and
the first test pins that. Ported from the August suite (all tests passed
unchanged against the refactored code); the runtime-subtype test now also
checks the message of the final error.
"""

from typing import Optional

import pytest

from core.metrics.logger import MetricLogger, Node
from core.metrics.metric.base import Metric
from core.metrics.metric.mean import MeanMetric
from core.metrics.metric.series import SeriesMetric
from core.metrics.schemas import MetricSchema


@pytest.mark.unit
class TestBuild:
    def test_direct_instantiation_is_blocked(self):
        with pytest.raises(TypeError, match="from_schema"):
            MetricLogger()

    def test_static_tree_and_refs(self, metric_schemas):
        Leaf, _, _, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        tree = logger._tree
        assert isinstance(tree, Node) and tree.schema is Root and not tree.dynamic
        assert isinstance(tree["static"], Node) and tree["static"].schema is Leaf
        assert isinstance(tree["static"]["mean_value"], MeanMetric)
        assert isinstance(tree["static"]["series_value"], SeriesMetric)
        assert isinstance(
            tree["static"]["default_value"], MeanMetric
        )  # MEAN by default
        assert tree["group"]["by_id"].dynamic and tree["group"]["by_id"].schema is Leaf
        assert len(tree["group"]["by_id"]) == 0  # nothing materialized yet
        assert ("static", "mean_value") in logger._refs
        assert logger._refs[("static", "mean_value")] is tree["static"]["mean_value"]
        assert all(isinstance(m, Metric) for m in logger._refs.values())

    def test_dict_of_non_schema_is_rejected(self):
        class Bad(MetricSchema):
            by_id: dict[str, float] = {}

        with pytest.raises(TypeError, match="dict\\[ID, MetricSchema\\]"):
            MetricLogger.from_schema(Bad)


@pytest.mark.unit
class TestPush:
    def test_push_static_leaf_and_unknown_path(self, metric_schemas):
        *_, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger.push(("static", "mean_value"), 1.0)
        logger.push(("static", "mean_value"), 3.0)
        assert logger.peek_value(("static", "mean_value")) == 2.0
        with pytest.raises(KeyError, match="Unknown logger path"):
            logger.push(("static", "nope"), 1.0)
        with pytest.raises(KeyError, match="beyond metric leaf"):
            logger.push(("static", "mean_value", "deeper"), 1.0)
        with pytest.raises(KeyError, match="does not point to a metric"):
            logger.push(("static",), 1.0)

    def test_push_materializes_dynamic_ids_independently(self, metric_schemas):
        *_, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger.push(("group", "by_id", "a", "sum_value"), 1.0)
        logger.push(("group", "by_id", "a", "sum_value"), 2.0)
        logger.push(("group", "by_id", "b", "sum_value"), 10.0)
        assert logger.peek_value(("group", "by_id", "a", "sum_value")) == 3.0
        assert logger.peek_value(("group", "by_id", "b", "sum_value")) == 10.0
        with pytest.raises(KeyError, match="Expected dynamic ID"):
            logger.push(("group", "by_id"), 1.0)

    def test_push_data_static_and_skips_none(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger.push_data(
            Root(static=Leaf(mean_value=4.0, sum_value=None), group=Group())
        )
        assert logger.peek_value(("static", "mean_value")) == 4.0
        assert len(logger._refs[("static", "sum_value")]) == 0
        with pytest.raises(TypeError, match="Expected RootSchema"):
            logger.push_data(Leaf())

    def test_push_data_dynamic_and_runtime_subtype(self, metric_schemas):
        Leaf, Rich, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        payload = Root(
            static=Leaf(),
            group=Group(
                by_id={"a": Rich(mean_value=1.0, extra=5.0), "b": Leaf(mean_value=2.0)}
            ),
        )
        logger.push_data(payload)
        assert logger.peek_value(("group", "by_id", "a", "extra")) == 5.0
        assert logger.peek_value(("group", "by_id", "b", "mean_value")) == 2.0
        # a bound ID cannot silently change schema
        with pytest.raises(TypeError, match="Runtime schema changed"):
            logger.push_data(
                Root(static=Leaf(), group=Group(by_id={"a": Leaf(mean_value=1.0)}))
            )

        # unrelated schema rejected
        class Other(MetricSchema):
            x: float = 0.0

        # pydantic rejects it at construction; bypass validation to reach the guard
        bad = Root.model_construct(
            static=Leaf(), group=Group.model_construct(by_id={"c": Other()})
        )
        with pytest.raises(TypeError, match="not a subclass"):
            logger.push_data(bad)

    def test_static_nested_runtime_subtype_binding(self, metric_schemas):
        """A nested slot declared as ``MetricSchema`` is specialised by the first push.

        This is how an evolution-strategy logger ends up holding the concrete
        schemas of the inner level.
        """
        Leaf, Rich, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        assert logger._tree["inner"].schema is MetricSchema

        logger.push_data(
            Root(static=Leaf(), group=Group(), inner=Rich(series_value=1.0, extra=1.0))
        )
        assert logger._tree["inner"].schema is Rich
        assert logger.peek_value(("inner", "series_value")) == [1.0]

        logger.push_data(
            Root(static=Leaf(), group=Group(), inner=Rich(series_value=2.0))
        )
        assert logger.peek_value(("inner", "series_value")) == [1.0, 2.0]  # accumulates

        class Incompatible(MetricSchema):
            y: float = 0.0

        # a sibling subtype of MetricSchema is accepted by subclass rule but must not
        # silently reuse the Rich subtree: the declared node schema is Rich now
        with pytest.raises(TypeError, match="Incompatible is not a subclass of"):
            logger.push_data(Root(static=Leaf(), group=Group(), inner=Incompatible()))


@pytest.mark.unit
class TestPeekReduce:
    def test_peek_is_non_destructive_and_reduce_is_destructive(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        for v in (1.0, 3.0):
            logger.push(("static", "mean_value"), v)
            logger.push(("static", "series_value"), v)
        logger.push(("group", "by_id", "a", "last_value"), 7)

        peeked = logger.peek()
        assert isinstance(peeked, Root)
        assert peeked.static.mean_value == [1.0, 3.0]  # peek keeps raw history
        assert peeked.static.series_value == [1.0, 3.0]
        assert peeked.group.by_id["a"].last_value == [7]
        assert logger.peek() == peeked  # twice the same

        reduced = logger.reduce()
        assert reduced.static.mean_value == 2.0
        assert reduced.static.series_value == [1.0, 3.0]
        assert reduced.group.by_id["a"].last_value == 7
        assert reduced.static.sum_value == 0 and reduced.static.min_value is None
        assert logger.reduce().static.series_value == []  # emptied

    def test_compile_reset_flush(self, metric_schemas):
        *_, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger.push(("static", "sum_value"), 2.0)
        assert logger.compile()["static"]["sum_value"] == 2.0
        logger.push(("static", "sum_value"), 2.0)
        logger.flush(("static", "sum_value"))
        assert logger.peek_value(("static", "sum_value")) == 0
        logger.push(("static", "sum_value"), 5.0)
        logger.reset()
        assert logger.peek_value(("static", "sum_value")) == 0
        with pytest.raises(KeyError):
            logger.flush(("static", "nope"))
        with pytest.raises(KeyError):
            logger.peek_value(("static", "nope"))

    def test_optional_static_branch_is_materialized_lazily(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        # optional static schema nodes are built eagerly with their declared schema
        assert logger._tree["optional_static"].schema is Leaf
        logger.push_data(
            Root(static=Leaf(), group=Group(), optional_static=Leaf(mean_value=1.0))
        )
        assert logger.peek().optional_static.mean_value == [1.0]


@pytest.mark.unit
class TestSpecialisationKeepsPushedValues:
    """Specialising a node to a subclass must not lose what was pushed before."""

    def test_values_pushed_by_path_survive(self, metric_schemas):
        Leaf, Rich, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger.push(("optional_static", "mean_value"), 1.0)
        logger.push(("optional_static", "series_value"), 5.0)
        logger.push(("optional_static", "last_value"), 7)

        logger.push_data(
            Root(
                static=Leaf(),
                group=Group(),
                optional_static=Rich(mean_value=3.0, extra=9.0),
            )
        )

        assert logger._tree["optional_static"].schema is Rich
        peeked = logger.peek().optional_static
        assert peeked.mean_value == [1.0, 3.0]
        assert peeked.series_value == [5.0]
        assert peeked.last_value == [7]
        assert peeked.extra == [9.0]

    def test_values_pushed_by_a_declared_schema_survive(self, metric_schemas):
        Leaf, Rich, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger.push_data(
            Root(static=Leaf(), group=Group(), optional_static=Leaf(sum_value=2.0))
        )
        logger.push_data(
            Root(
                static=Leaf(),
                group=Group(),
                optional_static=Rich(sum_value=3.0, extra=1.0),
            )
        )

        assert logger.reduce().optional_static.sum_value == 5.0

    def test_values_pushed_under_a_dynamic_id_survive(self):
        from core.metrics.schemas import MetricSchema

        class Item(MetricSchema):
            value: Optional[float] = None

        class Inner(MetricSchema):
            by_id: dict[str, Item] = {}

        class RichInner(Inner):
            extra: Optional[float] = None

        class Slot(MetricSchema):
            inner: Optional[Inner] = None

        logger = MetricLogger.from_schema(Slot)
        logger.push(("inner", "by_id", "a", "value"), 1.0)
        logger.push(("inner", "by_id", "b", "value"), 10.0)

        logger.push_data(Slot(inner=RichInner(by_id={"a": Item(value=3.0)}, extra=2.0)))

        peeked = logger.peek().inner
        assert type(peeked) is RichInner
        assert peeked.by_id["a"].value == [1.0, 3.0]
        # "b" is left out of the push_data call, so it receives a gap.
        assert peeked.by_id["b"].value == [10.0, None]
        assert logger.peek_value(("inner", "by_id", "b", "value")) == 10.0

    def test_the_iteration_counter_of_an_open_slot_survives(self, metric_schemas):
        Leaf, Rich, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger.push(("inner", "iter"), 4)

        logger.push_data(Root(static=Leaf(), group=Group(), inner=Rich(iter=5)))

        assert logger.peek().inner.iter == [4, 5]
        assert logger.reduce().inner.iter == 5
