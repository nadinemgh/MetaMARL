"""``MetricLogger`` with a schema hierarchy fixed at construction.

Covers schema-tree construction, independent dynamic IDs, path-based and
schema-based pushes, sparse histories, reduction, reset, and validation.
Runtime IDs may appear or disappear, but the types of their metrics are
already declared by the configured schema. An omitted value does not imply a
new iteration or an automatic gap: callers can log time explicitly if needed.
"""

from typing import Optional

import pytest
from pydantic import Field

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

    def test_push_data_dynamic_ids_use_preconfigured_value_schema(self, metric_schemas):
        _, Rich, _, _ = metric_schemas

        class ConfiguredGroup(MetricSchema):
            by_id: dict[str, Rich] = Field(default_factory=dict)  # type: ignore

        class ConfiguredRoot(MetricSchema):
            group: ConfiguredGroup = Field(default_factory=ConfiguredGroup)

        logger = MetricLogger.from_schema(ConfiguredRoot)
        dynamic_node = logger._tree["group"]["by_id"]

        # The complete value schema is known before any dynamic ID appears.
        assert dynamic_node.schema is Rich
        assert not dynamic_node
        logger.push_data(
            ConfiguredRoot(
                group=ConfiguredGroup(
                    by_id={
                        "a": Rich(mean_value=1.0, extra=5.0),
                        "b": Rich(mean_value=2.0, extra=7.0),
                    }
                )
            )
        )
        logger.push_data(
            ConfiguredRoot(group=ConfiguredGroup(by_id={"a": Rich(mean_value=3.0)}))
        )
        assert dynamic_node.schema is Rich
        assert dynamic_node["a"].schema is Rich
        assert dynamic_node["b"].schema is Rich
        assert logger.peek().group.by_id["a"].mean_value == [1.0, 3.0]
        assert logger.peek().group.by_id["b"].mean_value == [2.0]
        assert logger.peek_value(("group", "by_id", "a", "extra")) == 5.0
        assert logger.peek_value(("group", "by_id", "b", "extra")) == 7.0

    def test_push_data_rejects_unrelated_schema_for_dynamic_id(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)

        class Other(MetricSchema):
            x: float = 0.0

        # Bypass Pydantic input validation to exercise MetricLogger's guard.
        invalid = Root.model_construct(
            static=Leaf(), group=Group.model_construct(by_id={"c": Other()})
        )

        with pytest.raises(TypeError, match="not a subclass"):
            logger.push_data(invalid)

    def test_concrete_nested_schema_is_built_before_first_push(self, metric_schemas):
        _, Rich, _, _ = metric_schemas

        class ConfiguredRoot(MetricSchema):
            inner: Rich = Field(default_factory=Rich)  # type: ignore

        logger = MetricLogger.from_schema(ConfiguredRoot)

        # No first-push specialization: the child and all its leaves exist now.
        inner_node = logger._tree["inner"]

        assert inner_node.schema is Rich
        assert ("inner", "series_value") in logger._refs
        assert ("inner", "extra") in logger._refs
        logger.push_data(ConfiguredRoot(inner=Rich(series_value=1.0, extra=4.0)))
        logger.push_data(ConfiguredRoot(inner=Rich(series_value=2.0)))
        assert logger._tree["inner"] is inner_node
        assert logger.peek().inner.series_value == [1.0, 2.0]
        assert logger.peek().inner.extra == [4.0]
        assert logger.peek_value(("inner", "series_value")) == [1.0, 2.0]


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

    def test_optional_static_branch_uses_its_declared_schema(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)

        # optional static schema nodes are built eagerly with their declared schema
        assert logger._tree["optional_static"].schema is Leaf
        logger.push_data(
            Root(static=Leaf(), group=Group(), optional_static=Leaf(mean_value=1.0))
        )
        assert logger.peek().optional_static.mean_value == [1.0]


@pytest.mark.unit
class TestFixedSchemasPreservePushedValues:
    """Values survive repeated pushes into a single configured schema tree."""

    def test_push_by_path_then_push_data_keeps_existing_values(self, metric_schemas):
        _, Rich, _, _ = metric_schemas

        class ConfiguredRoot(MetricSchema):
            inner: Rich = Field(default_factory=Rich)  # type: ignore

        logger = MetricLogger.from_schema(ConfiguredRoot)

        logger.push(("inner", "mean_value"), 1.0)
        logger.push(("inner", "series_value"), 5.0)
        logger.push(("inner", "last_value"), 7)
        logger.push_data(ConfiguredRoot(inner=Rich(mean_value=3.0, extra=9.0)))

        peeked = logger.peek().inner

        assert peeked.mean_value == [1.0, 3.0]
        assert peeked.series_value == [5.0]
        assert peeked.last_value == [7]
        assert peeked.extra == [9.0]
        assert logger._tree["inner"].schema is Rich

    def test_sum_accumulates_across_pushes_without_schema_changes(self, metric_schemas):
        _, Rich, _, _ = metric_schemas

        class ConfiguredRoot(MetricSchema):
            inner: Rich = Field(default_factory=Rich)  # type: ignore

        logger = MetricLogger.from_schema(ConfiguredRoot)

        logger.push_data(ConfiguredRoot(inner=Rich(sum_value=2.0)))
        logger.push_data(ConfiguredRoot(inner=Rich(sum_value=3.0, extra=1.0)))

        reduced = logger.reduce().inner

        assert reduced.sum_value == 5.0
        assert reduced.extra == 1.0
        assert logger._tree["inner"].schema is Rich

    def test_existing_dynamic_ids_retain_their_values_when_omitted(self):
        class Item(MetricSchema):
            value: Optional[float] = None

        class Inner(MetricSchema):
            by_id: dict[str, Item] = Field(default_factory=dict)

        class ConfiguredRoot(MetricSchema):
            inner: Inner = Field(default_factory=Inner)

        logger = MetricLogger.from_schema(ConfiguredRoot)

        logger.push(("inner", "by_id", "a", "value"), 1.0)
        logger.push(("inner", "by_id", "b", "value"), 10.0)
        logger.push_data(ConfiguredRoot(inner=Inner(by_id={"a": Item(value=3.0)})))

        peeked = logger.peek().inner

        assert peeked.by_id["a"].value == [1.0, 3.0]
        assert peeked.by_id["b"].value == [10.0]
        assert logger.peek_value(("inner", "by_id", "b", "value")) == 10.0
        assert logger._tree["inner"].schema is Inner

    def test_explicit_iteration_field_accumulates_in_fixed_schema(self, metric_schemas):
        _, Rich, _, _ = metric_schemas

        class ConfiguredRoot(MetricSchema):
            inner: Rich = Field(default_factory=Rich)  # type: ignore

        logger = MetricLogger.from_schema(ConfiguredRoot)

        logger.push(("inner", "iter"), 4)
        logger.push_data(ConfiguredRoot(inner=Rich(iter=5)))
        assert logger.peek().inner.iter == [4, 5]
        assert logger.reduce().inner.iter == 5
        assert logger._tree["inner"].schema is Rich
