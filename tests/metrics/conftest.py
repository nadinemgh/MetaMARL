"""Schemas shared by the metrics tests.

``LeafSchema`` has one field per reduction protocol so that a single logger
exercises every metric class; ``RootSchema`` nests it statically, dynamically
(``dict[str, LeafSchema]``) and through an open ``MetricSchema`` slot that a
runtime subtype can specialise. The ``Subtree*`` schemas cover the
``subtree_reduce`` override that the Ray adaptor schema relies on.
"""

from typing import Optional

import pytest
from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema


class LeafSchema(MetricSchema):
    mean_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    series_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    last_value: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    sum_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SUM}
    )
    min_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MIN}
    )
    max_value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MAX}
    )
    default_value: Optional[float] = None  # no metadata -> MEAN


class RichLeafSchema(LeafSchema):
    """Runtime subtype of ``LeafSchema`` with one extra field."""

    extra: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )


class GroupSchema(MetricSchema):
    by_id: dict[str, LeafSchema] = Field(default_factory=dict)


class RootSchema(MetricSchema):
    static: LeafSchema
    group: GroupSchema
    optional_static: Optional[LeafSchema] = None
    inner: Optional[MetricSchema] = None


class SubtreeLeafSchema(MetricSchema):
    """Leaves that would be MEAN and SERIES without a subtree override."""

    a: Optional[float] = None
    b: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )


class SubtreeNestedSchema(MetricSchema):
    """A schema one level deeper than the field that carries the override."""

    inner: Optional[SubtreeLeafSchema] = None


class SubtreeRootSchema(MetricSchema):
    """Three subtrees carry an override; the control one keeps its protocols."""

    forced: Optional[SubtreeLeafSchema] = Field(
        default=None, json_schema_extra={"subtree_reduce": ReduceProtocol.LAST}
    )
    forced_by_id: dict[str, SubtreeLeafSchema] = Field(
        default_factory=dict, json_schema_extra={"subtree_reduce": ReduceProtocol.SUM}
    )
    nested: Optional[SubtreeNestedSchema] = Field(
        default=None, json_schema_extra={"subtree_reduce": ReduceProtocol.MAX}
    )
    control: Optional[SubtreeLeafSchema] = None


@pytest.fixture
def metric_schemas():
    """Return ``(LeafSchema, RichLeafSchema, GroupSchema, RootSchema)``."""
    return LeafSchema, RichLeafSchema, GroupSchema, RootSchema


@pytest.fixture
def logger():
    """Return a fresh logger over ``RootSchema``."""
    return MetricLogger.from_schema(RootSchema)


@pytest.fixture
def subtree_schemas():
    """Return ``(SubtreeLeafSchema, SubtreeRootSchema)``."""
    return SubtreeLeafSchema, SubtreeRootSchema


@pytest.fixture
def subtree_logger():
    """Return a fresh logger over ``SubtreeRootSchema``."""
    return MetricLogger.from_schema(SubtreeRootSchema)
