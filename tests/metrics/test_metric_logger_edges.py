"""Error paths and internal guards of ``MetricLogger`` and ``Node``.

These tests reach the defensive branches that the public API never exercises
on well-formed schemas: schema-less nodes, wrong payload shapes bypassing
pydantic validation (``model_construct``), and metrics whose ``peek``/``reduce``
/``flush`` raise. Production code is not modified; where a branch is
unreachable through the public API this is stated in the test.
"""

from typing import Optional

import pytest
from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.logger import MetricLogger, Node
from core.metrics.metric.mean import MeanMetric
from core.metrics.metric.series import SeriesMetric
from core.metrics.schemas import MetricSchema


class _RaisingMetric(SeriesMetric):
    """Metric whose every accessor fails, to exercise the logger's wrappers."""

    def peek(self, compile: bool = True):
        raise RuntimeError("peek boom")

    def reduce(self, compile: bool = True):
        raise RuntimeError("reduce boom")

    def flush(self) -> None:
        raise RuntimeError("flush boom")


@pytest.mark.unit
class TestNode:
    def test_construct_without_schema_raises(self):
        node = Node()
        node["x"] = MeanMetric()
        with pytest.raises(RuntimeError, match="no schema"):
            node.construct({"x": 1.0})

    def test_dynamic_construct_maps_ids(self, metric_schemas):
        Leaf, *_ = metric_schemas
        child, _ = MetricLogger._build_from_schema(Leaf)
        node = Node(dynamic=True, schema=Leaf)
        node["a"] = child
        built = node.construct({"a": {name: None for name in child}})
        assert set(built) == {"a"} and isinstance(built["a"], Leaf)


@pytest.mark.unit
class TestBuild:
    def test_init_token_guard(self, metric_schemas):
        *_, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        # ``__init__`` carries the same guard as ``__new__``; ``from_schema`` never
        # calls it, so the only way in is an explicit call.
        with pytest.raises(TypeError, match="from_schema"):
            MetricLogger.__init__(logger)
        MetricLogger.__init__(logger, _token=MetricLogger._TOKEN)  # accepted

    def test_union_of_two_non_none_types_is_kept_as_is(self):
        class Multi(MetricSchema):
            value: Optional[int | str] = Field(
                default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
            )

        logger = MetricLogger.from_schema(Multi)
        logger.push(("value",), "x")
        assert logger.peek_value(("value",)) == "x"

    def test_dynamic_flag_on_build_skips_children(self, metric_schemas):
        """``_build_from_schema(dynamic=True)`` builds an empty dynamic node.

        The public API never sets this flag (dynamic nodes are created directly
        in the ``dict[ID, MetricSchema]`` branch), so this documents the internal
        contract: nested schemas and leaves are skipped when ``dynamic`` is set.
        """
        *_, Root = metric_schemas
        node, refs = MetricLogger._build_from_schema(Root, dynamic=True)
        assert node.dynamic and node.schema is Root
        assert "static" not in node and "iter" not in node
        assert "group" not in node
        assert refs == {}


@pytest.mark.unit
class TestResolvePath:
    def test_dynamic_node_without_schema(self, metric_schemas):
        *_, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger._tree["group"]["by_id"].schema = None
        with pytest.raises(RuntimeError, match="no schema"):
            logger.push(("group", "by_id", "a", "mean_value"), 1.0)


@pytest.mark.unit
class TestPushDataGuards:
    def test_unknown_field_when_node_lacks_it(self, metric_schemas):
        Leaf, *_ = metric_schemas
        logger = MetricLogger.from_schema(Leaf)
        node = Node(schema=Leaf)  # a node built without any field
        with pytest.raises(KeyError, match="Unknown logger field"):
            logger.push_data(Leaf(mean_value=1.0), prefix=("p",), node=node)

    def test_schema_value_on_a_metric_leaf(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        bad = Root.model_construct(static=Leaf(), group=Group(), iter=Leaf())
        with pytest.raises(TypeError, match="Expected Node"):
            logger.push_data(bad)

    def test_static_node_without_schema(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger._tree["static"].schema = None
        with pytest.raises(RuntimeError, match="no declared schema"):
            logger.push_data(Root(static=Leaf(mean_value=1.0), group=Group()))

    def test_dict_value_on_a_non_dynamic_node(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        bad = Root.model_construct(static={"a": Leaf()}, group=Group())
        with pytest.raises(TypeError, match="Expected dynamic Node"):
            logger.push_data(bad)
        bad = Root.model_construct(static=Leaf(), group=Group(), iter={"a": 1})
        with pytest.raises(TypeError, match="Expected dynamic Node"):
            logger.push_data(bad)

    def test_dynamic_node_without_schema(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger._tree["group"]["by_id"].schema = None
        with pytest.raises(RuntimeError, match="no declared schema"):
            logger.push_data(Root(static=Leaf(), group=Group(by_id={"a": Leaf()})))

    def test_dynamic_child_must_be_a_schema(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        bad = Root.model_construct(
            static=Leaf(), group=Group.model_construct(by_id={"a": 1.0})
        )
        with pytest.raises(TypeError, match="Expected MetricSchema"):
            logger.push_data(bad)

    def test_runtime_child_must_be_a_node(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger._tree["group"]["by_id"]["a"] = MeanMetric()
        with pytest.raises(TypeError, match="Expected runtime Node"):
            logger.push_data(Root(static=Leaf(), group=Group(by_id={"a": Leaf()})))

    def test_primitive_value_on_a_node(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        bad = Root.model_construct(static=1.0, group=Group())
        with pytest.raises(TypeError, match="Expected Metric"):
            logger.push_data(bad)

    def test_root_payload_must_be_exact_schema_not_subclass(self, metric_schemas):
        """The root payload must be the exact schema; nested slots accept subtypes."""
        Leaf, Rich, *_ = metric_schemas
        logger = MetricLogger.from_schema(Leaf)
        with pytest.raises(TypeError, match="Expected LeafSchema"):
            logger.push_data(Rich(mean_value=1.0))


@pytest.mark.unit
class TestFailingMetrics:
    @pytest.fixture
    def logger_with_raising_leaf(self, metric_schemas):
        *_, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        raising = _RaisingMetric()
        logger._tree["static"]["mean_value"] = raising
        logger._refs[("static", "mean_value")] = raising
        return logger

    def test_peek_wraps_errors_with_path(self, logger_with_raising_leaf):
        with pytest.raises(ValueError, match="Error peeking metric") as info:
            logger_with_raising_leaf.peek()
        assert "mean_value" in str(info.value)
        assert isinstance(info.value.__cause__, RuntimeError)

    def test_reduce_wraps_errors_with_path(self, logger_with_raising_leaf):
        with pytest.raises(ValueError, match="Error reducing metrics"):
            logger_with_raising_leaf.reduce()

    def test_reset_wraps_errors_with_path(self, logger_with_raising_leaf):
        with pytest.raises(ValueError, match="Error flushing metrics"):
            logger_with_raising_leaf.reset()

    def test_flush_wraps_errors_with_path(self, logger_with_raising_leaf):
        with pytest.raises(ValueError, match="Error flushing metric"):
            logger_with_raising_leaf.flush(("static", "mean_value"))


@pytest.mark.unit
class TestRepeatedDynamicPushes:
    def test_unknown_leaf_under_a_materialized_id(self, metric_schemas):
        *_, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        logger.push(("group", "by_id", "a", "sum_value"), 1.0)
        with pytest.raises(KeyError, match="Unknown logger path"):
            logger.push(("group", "by_id", "a", "nope"), 1.0)

    def test_push_data_twice_on_the_same_id_accumulates(self, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger = MetricLogger.from_schema(Root)
        for value in (1.0, 3.0):
            logger.push_data(
                Root(static=Leaf(), group=Group(by_id={"a": Leaf(mean_value=value)}))
            )
        assert logger.peek_value(("group", "by_id", "a", "mean_value")) == 2.0
        assert logger.reduce().group.by_id["a"].mean_value == 2.0
