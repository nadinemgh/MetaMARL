"""Typed metric accumulation driven by pydantic schemas.

A ``MetricSchema`` declares *what* is logged: each field is a leaf metric whose
reducer comes from ``Field(json_schema_extra={"reduce": ReduceProtocol.X})``
(MEAN by default), a nested ``MetricSchema`` is a static sub-tree, and a
``dict[ID, MetricSchema]`` is a *dynamic* node whose children are created on
first use (one per agent, policy, candidate, ...). :class:`MetricLogger` builds
the matching tree of :class:`~core.metrics.metric.base.Metric` objects,
accumulates values through :meth:`~MetricLogger.push` and
:meth:`~MetricLogger.push_data`, and returns populated schema instances through
:meth:`~MetricLogger.peek` (non-destructive, raw histories) or
:meth:`~MetricLogger.reduce` (destructive, reduced values). Pushing a *subclass*
of a declared schema specializes that sub-tree at runtime, which is how an ES
logger ends up holding the concrete RLlib and environment schemas of the inner
level.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, TypeAlias, get_args, get_origin

import tree  # dm_tree

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.base import Metric
from core.metrics.metric.factory import MetricFactory
from core.metrics.schemas import MetricSchema

Path: TypeAlias = tuple[str, ...]


class Node(dict[str, "Node | Metric"]):
    """One level of the metric tree built from a ``MetricSchema``.

    A node maps field names (or runtime ids for a *dynamic* node) to child
    nodes or :class:`Metric` leaves. ``schema`` is the pydantic class that the
    node rebuilds in :meth:`construct`; ``dynamic`` marks a
    ``dict[ID, MetricSchema]`` field whose children are created on first push.

    Parameters
    ----------
    *args : Any
        Positional arguments of ``dict``.
    schema : type[MetricSchema] or None, default None
        The schema class the node mirrors (the value schema of a dynamic
        node). A static node without one cannot ``construct``.
    dynamic : bool, default False
        Whether the node is the ``dict[ID, MetricSchema]`` of a schema field.
    subtree_reduce : ReduceProtocol or None, default None
        Protocol imposed on every leaf below this node, overriding the
        protocols of the schema fields.
    **kwargs : Any
        Keyword arguments of ``dict``: the initial children.

    Attributes
    ----------
    schema : type[MetricSchema] or None
        The schema class of the node.
    dynamic : bool
        Whether the children are keyed by runtime ids.
    subtree_reduce : ReduceProtocol or None
        The protocol forced on the leaves below the node, if any.

    When to use: you normally get nodes from :class:`MetricLogger`, which
    builds the tree; create one by hand only to test tree-walking code.

    Examples
    --------
    >>> from typing import Optional
    >>> from core.metrics.metric.mean import MeanMetric
    >>> from core.metrics.schemas import MetricSchema
    >>> class Rollout(MetricSchema):
    ...     reward: Optional[float] = None
    >>> node = Node(schema=Rollout)
    >>> node["reward"] = MeanMetric()
    >>> node.construct({"reward": 2.0})
    Rollout(iter=None, reward=2.0)
    """

    schema: type[MetricSchema] | None
    dynamic: bool = False
    subtree_reduce: ReduceProtocol | None = None

    def __init__(
        self,
        *args: Any,
        schema: type[MetricSchema] | None = None,
        dynamic: bool = False,
        subtree_reduce: ReduceProtocol | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.schema = schema
        self.dynamic = dynamic
        self.subtree_reduce = subtree_reduce

    def construct(self, data: dict[str, Any]) -> MetricSchema | dict[str, Any]:
        """Rebuild a schema instance from values laid out like this node.

        ``data`` mirrors the node: one entry per child, holding either a leaf
        value or the nested dictionary of a child node. A dynamic node returns
        a plain ``dict`` keyed by runtime id; a static node returns
        ``schema.model_construct(**values)``, so no pydantic validation runs.

        Parameters
        ----------
        data : dict[str, Any]
            Values laid out like the node: a key per child, holding a leaf
            value or the nested dictionary of a child node.

        Returns
        -------
        MetricSchema or dict[str, Any]
            A schema instance for a static node, a dictionary of rebuilt
            children keyed by runtime id for a dynamic node.

        Raises
        ------
        RuntimeError
            If a static node was built without a schema.
        """

        if self.dynamic:
            return {
                dynamic_id: child.construct(data[dynamic_id])
                for dynamic_id, child in self.items()
            }

        values: dict[str, Any] = {}

        for field_name, child in self.items():
            value = data[field_name]

            if isinstance(child, Metric):
                values[field_name] = value
            else:
                values[field_name] = child.construct(value)

        if self.schema is None:
            raise RuntimeError("Runtime Node has no schema.")

        return self.schema.model_construct(**values)


class MetricLogger(ABC):
    """Tree of metric accumulators mirroring a ``MetricSchema``.

    Build one with :meth:`from_schema` (direct instantiation raises
    ``TypeError``), feed it with :meth:`push` (one path) or :meth:`push_data`
    (a whole schema instance), then read it back with :meth:`peek` (raw
    histories, non-destructive) or :meth:`reduce` (reduced values, clears the
    accumulators). The logger is the second layer of the reporting stack:
    ``MetricSchema`` declares, ``MetricLogger`` accumulates,
    :class:`~core.reporting.query.Query` selects and
    :class:`~core.reporting.base.Reporter` renders.

    A path is a tuple of field names, with the runtime id after each dynamic
    node, for example ``("by_mechanism", "quota", "fitness")``. Ids are
    created the first time a value is pushed under them.

    When to use: wherever values are produced piecemeal during an iteration
    (per step, per episode, per candidate) and must be reduced once before
    being reported: the optimizers and the environments each own one.

    Examples
    --------
    >>> from typing import Optional
    >>> from pydantic import Field
    >>> from core.metrics.schemas import MetricSchema
    >>> class Rollout(MetricSchema):
    ...     reward: Optional[float] = None
    ...     steps: Optional[int] = Field(
    ...         default=None, json_schema_extra={"reduce": ReduceProtocol.SUM}
    ...     )
    >>> logger = MetricLogger.from_schema(Rollout)
    >>> logger.push(("reward",), 1.0)
    >>> logger.push_data(Rollout(reward=3.0, steps=10))
    >>> logger.peek().reward
    [1.0, 3.0]
    >>> reduced = logger.reduce()
    >>> reduced.reward, reduced.steps
    (2.0, 10)
    >>> logger.peek_value(("reward",)) is None
    True
    """

    _TOKEN: ClassVar[object] = object()
    _schema: type[MetricSchema]
    _refs: dict[Path, Metric]
    _root: str
    _tree: Node

    def __new__(cls, *, _token: object | None = None) -> "MetricLogger":
        if _token is not cls._TOKEN:
            raise TypeError(
                "MetricLogger cannot be instantiated directly. "
                + "Use MetricLogger.from_schema(schema)"
            )

        return super().__new__(cls)

    def __init__(self, *, _token: object | None = None) -> None:
        if _token is not self._TOKEN:
            raise TypeError(
                "MetricLogger cannot be instantiated directly. "
                + "Use MetricLogger.from_schema(schema)"
            )

    @classmethod
    def from_schema(cls, schema: type[MetricSchema]) -> "MetricLogger":
        """Create a logger whose tree mirrors ``schema``.

        This is the only supported constructor: it builds the ``Node`` tree and
        the flat ``path -> Metric`` index that :meth:`push` looks up first.

        Parameters
        ----------
        schema : type[MetricSchema]
            The schema class to mirror. A leaf takes the metric of its
            ``reduce`` extra (``MEAN`` by default).

        Returns
        -------
        MetricLogger
            A logger holding an empty metric per leaf; dynamic nodes have no
            child yet.

        Raises
        ------
        TypeError
            If a field is a dictionary whose values are not ``MetricSchema``
            subclasses.
        NotImplementedError
            If a field asks for a protocol without an implementation (``EMA``).
        """

        tree, refs = cls._build_from_schema(schema)
        self = cls.__new__(cls, _token=cls._TOKEN)
        self._schema = schema
        self._root = schema.__name__
        self._tree = tree
        self._refs = refs

        return self

    @classmethod
    def _build_from_schema(
        cls,
        schema: type[MetricSchema],
        *,
        prefix: Path = (),
        dynamic: bool = False,
        subtree_reduce: ReduceProtocol | None = None,
    ) -> tuple[Node, dict[Path, Metric]]:
        refs: dict[Path, Metric] = {}

        node = Node(schema=schema, dynamic=dynamic, subtree_reduce=subtree_reduce)

        for field_name, field in schema.__pydantic_fields__.items():
            path = prefix + (field_name,)
            ann = field.annotation

            # Unwrap Optional[T] / T | None.
            args = get_args(ann)

            if type(None) in args:
                non_none = tuple(arg for arg in args if arg is not type(None))

                if len(non_none) == 1:
                    ann = non_none[0]

            extra = field.json_schema_extra or {}
            field_protocol = extra.get("reduce")
            field_override = extra.get("subtree_reduce")
            reduce = subtree_reduce if subtree_reduce is not None else field_override

            if isinstance(ann, type) and issubclass(ann, MetricSchema):
                child, child_ref = cls._build_from_schema(
                    ann, prefix=path, subtree_reduce=reduce
                )

                if not node.dynamic:
                    node[field_name] = child

                    refs.update(child_ref)

                continue

            # CASE WHEN dict[ID, MetricSchema]
            if get_origin(ann) is dict:
                _, value_ann = get_args(ann)

                if not (
                    isinstance(value_ann, type) and issubclass(value_ann, MetricSchema)
                ):
                    raise TypeError(
                        f"{schema.__name__}.{field_name} must be "
                        + f"dict[ID, MetricSchema], got {ann!r}"
                    )

                node[field_name] = Node(
                    schema=value_ann, dynamic=True, subtree_reduce=reduce
                )

                continue

            protocol = (
                subtree_reduce
                if subtree_reduce is not None
                else field_protocol or ReduceProtocol.MEAN
            )
            metric = MetricFactory.create(protocol)

            if not node.dynamic:
                node[field_name] = metric
                refs[path] = metric

        return node, refs

    def _resolve_path(
        self, path: Path, *, node: Node, index: int, prefix: Path
    ) -> Metric:
        if node.dynamic:
            if index >= len(path):
                raise KeyError(f"Expected dynamic ID at path: {path}")

            if node.schema is None:
                raise RuntimeError(f"Dynamic node at {prefix} has no schema.")

            dynamic_id = path[index]
            prefix = prefix + (dynamic_id,)
            runtime_child = node.get(dynamic_id)

            if runtime_child is None:
                runtime_child, refs = self._build_from_schema(
                    node.schema, prefix=prefix, subtree_reduce=node.subtree_reduce
                )
                node[dynamic_id] = runtime_child

                self._refs.update(refs)

            return self._resolve_path(
                path=path, node=runtime_child, index=index + 1, prefix=prefix
            )

        if index >= len(path):
            raise KeyError(f"Logger path does not point to a metric: {path}")

        field_name = path[index]

        try:
            child = node[field_name]
        except KeyError:
            raise KeyError(f"Unknown logger path: {path}") from None

        child_path = prefix + (field_name,)

        # leaf
        if isinstance(child, Metric):
            if index != len(path) - 1:
                raise KeyError(f"Path continues beyond metric leaf: {path}")

            return child

        return self._resolve_path(
            node=child, path=path, index=index + 1, prefix=child_path
        )

    def _specialise(
        self, node: Node, field_name: str, schema: type[MetricSchema], path: Path
    ) -> Node:
        """Replace a nested node by the sub-tree of a subclass of its schema.

        The new sub-tree is built from ``schema`` and takes the place of
        ``node[field_name]``. Every value already held under ``path`` is pushed
        again into the matching metric of the new sub-tree, in its original
        order, creating the dynamic ids that were already bound, so that
        nothing pushed before the specialisation is lost.
        """

        old_node = node[field_name]
        carried = {
            key: metric
            for key, metric in self._refs.items()
            if key[: len(path)] == path
        }

        new_node, refs = self._build_from_schema(
            schema, prefix=path, subtree_reduce=old_node.subtree_reduce
        )
        node[field_name] = new_node

        for key in carried:
            del self._refs[key]

        self._refs.update(refs)

        for key, old_metric in carried.items():
            new_metric = self._refs.get(key)

            if new_metric is None:
                new_metric = self._resolve_path(
                    path=key, node=self._tree, index=0, prefix=()
                )

            for value in old_metric.peek(compile=False):
                new_metric.push(value)

        return new_node

    def push_data(
        self, data: MetricSchema, prefix: Path = (), node: Node | None = None
    ) -> None:
        """Push every non-``None`` leaf of a schema instance into its metric.

        Nested schemas and dynamic nodes are walked recursively, and the
        children of a dynamic node are created on first use. A nested schema
        that is a subclass of the declared one specialises that sub-tree at
        runtime (the sub-tree is rebuilt, and the values pushed earlier under the
        declared schema are carried over to it). ``prefix`` and ``node`` are the
        recursion state and keep their defaults on the first call.

        Parameters
        ----------
        data : MetricSchema
            The values to push; at the root it must be exactly the schema the
            logger was built from.
        prefix : Path, default ()
            Path of ``data`` inside the tree.
        node : Node or None, default None
            The tree node matching ``data``; the root when ``None``.

        Raises
        ------
        TypeError
            If the root is not exactly the logger's schema, a nested
            schema is not a subclass of the declared one, a dynamic id changes
            schema, or a value does not match the shape of the tree.
        KeyError
            If ``data`` has a field the tree does not know.
        RuntimeError
            If a node has no declared schema.
        """

        if node is None:
            if not prefix and type(data) is not self._schema:
                raise TypeError(
                    f"Expected {self._schema.__name__}, got {type(data).__name__}."
                )

            node = self._tree

        for field_name in type(data).model_fields:
            value = getattr(data, field_name)

            if value is None:
                continue

            path = prefix + (field_name,)

            try:
                child_node = node[field_name]
            except KeyError:
                raise KeyError(
                    f"Unknown logger field {field_name!r} at {prefix} for "
                    + f"{type(data).__name__}."
                ) from None

            if isinstance(value, MetricSchema):
                if not isinstance(child_node, Node):
                    raise TypeError(
                        f"Expected Node at {path}, got {type(child_node).__name__}."
                    )

                runtime_schema = type(value)
                declared_schema = child_node.schema

                if declared_schema is None:
                    raise RuntimeError(f"Node at {path} has no declared schema.")

                if runtime_schema is not declared_schema:
                    if not issubclass(runtime_schema, declared_schema):
                        raise TypeError(
                            f"{runtime_schema.__name__} is not a subclass of "
                            + f"{declared_schema.__name__} at {path}."
                        )

                    child_node = self._specialise(
                        node, field_name, runtime_schema, path
                    )

                self.push_data(value, prefix=path, node=child_node)
                continue

            if isinstance(value, dict):
                if not isinstance(child_node, Node) or not child_node.dynamic:
                    raise TypeError(f"Expected dynamic Node at {path}.")

                declared_schema = child_node.schema

                if declared_schema is None:
                    raise RuntimeError(
                        f"Dynamic node at {path} has no declared schema."
                    )

                for dynamic_id, dynamic_child in value.items():
                    if not isinstance(dynamic_child, MetricSchema):
                        raise TypeError(
                            f"Expected MetricSchema at {path + (dynamic_id,)}, "
                            + f"got {type(dynamic_child).__name__}."
                        )

                    runtime_schema = type(dynamic_child)

                    if not issubclass(runtime_schema, declared_schema):
                        raise TypeError(
                            f"{runtime_schema.__name__} is not a subclass of "
                            + f"{declared_schema.__name__} at {path}."
                        )

                    runtime_path = path + (dynamic_id,)
                    runtime_node = child_node.get(dynamic_id)

                    if runtime_node is None:
                        runtime_node, refs = self._build_from_schema(
                            runtime_schema,
                            prefix=runtime_path,
                            subtree_reduce=child_node.subtree_reduce,
                        )
                        child_node[dynamic_id] = runtime_node

                        self._refs.update(refs)
                    elif not isinstance(runtime_node, Node):
                        raise TypeError(
                            f"Expected runtime Node at {runtime_path}, got "
                            + f"{type(runtime_node).__name__}."
                        )
                    elif runtime_node.schema is not runtime_schema:
                        raise TypeError(
                            f"Runtime schema changed at {runtime_path}: "
                            + f"{runtime_node.schema.__name__} -> "
                            + f"{runtime_schema.__name__}."
                        )

                    self.push_data(
                        dynamic_child, prefix=runtime_path, node=runtime_node
                    )

                continue

            if not isinstance(child_node, Metric):
                raise TypeError(
                    f"Expected Metric at {path}, got {type(child_node).__name__}."
                )

            child_node.push(value)

    def push(self, key: Path, value: Any) -> None:
        """Push one value under a path.

        The leaf must exist in the schema; the id of a dynamic node is created
        if it is new.

        Parameters
        ----------
        key : Path
            Path of the leaf, with the runtime id after each dynamic node.
        value : Any
            The value to record; the metric of the leaf may reject its type.

        Raises
        ------
        KeyError
            If the path is unknown, stops before a leaf or continues past one.
        """

        metric = self._refs.get(key)

        if metric is None:
            metric = self._resolve_path(path=key, node=self._tree, index=0, prefix=())

        metric.push(value)

    def peek_value(self, key: Path) -> Any:
        # NOTE this does not work for sub trees as of now !
        """Read the compiled value of one leaf without clearing it.

        Unlike :meth:`peek`, the value is the reduction of the leaf (its mean,
        sum, ...), not its history. Only leaves that exist can be read: a path
        to a sub-tree, or under a dynamic id that has not been pushed yet, is
        unknown.

        Parameters
        ----------
        key : Path
            Path of the leaf.

        Returns
        -------
        Any
            The compiled value of the leaf (``None`` for an empty mean, max,
            min or last).

        Raises
        ------
        KeyError
            If no leaf exists at ``key``.
        """

        metric = self._refs.get(key)

        if metric is None:
            raise KeyError(f"Unknown logger path: {key}")

        return metric.peek()

    def peek(self) -> MetricSchema:
        """Return the raw history of every leaf as a schema instance.

        Nothing is reduced or cleared: each leaf of the returned schema holds
        the list of every value pushed since the last reduction, whatever its
        protocol. This is what the reporters resolve their queries against.
        The instance is built without pydantic validation.

        Returns
        -------
        MetricSchema
            An instance of the logger's schema (specialised sub-trees
            included) whose leaves are lists.

        Raises
        ------
        ValueError
            If a leaf cannot be read.
        """

        def _peek(path: Path, metric: Metric):
            try:
                return metric.peek(compile=False)
            except Exception as e:
                raise ValueError(
                    f"Error peeking metric {metric} at path {path}."
                ) from e

        peeked = tree.map_structure_with_path(_peek, self._tree)

        return self._tree.construct(peeked)

    def reduce(self) -> MetricSchema:
        """Reduce every leaf, clear the accumulators and return the result.

        Each leaf collapses its values according to its protocol (a mean, a
        sum, the last value, the history of a series...), and an empty leaf
        gives ``None`` (or ``0`` for a sum). The next push starts a new
        accumulation cycle.

        Returns
        -------
        MetricSchema
            An instance of the logger's schema holding the reduced values.

        Raises
        ------
        ValueError
            If a leaf cannot be reduced.
        """

        def _reduce(path: Path, metric: Metric):
            try:
                return metric.reduce(compile=True)

            except Exception as e:
                raise ValueError(
                    f"Error reducing metrics {metric} at path {path}."
                ) from e

        reduced = tree.map_structure_with_path(_reduce, self._tree)

        return self._tree.construct(reduced)

    def compile(self) -> dict:
        """Reduce every leaf and return the result as a dictionary.

        Equivalent to ``reduce().model_dump(serialize_as_any=True)``: like
        :meth:`reduce`, it clears the accumulators.

        Returns
        -------
        dict
            The reduced values, keyed by field name and runtime id.
        """

        return self.reduce().model_dump(serialize_as_any=True)

    def reset(self) -> None:
        """Clear every accumulator without reducing it.

        The tree keeps its shape: dynamic ids already created stay, with
        empty metrics.

        Raises
        ------
        ValueError
            If a leaf cannot be cleared.
        """

        for path, metric in self._refs.items():
            try:
                metric.flush()

            except Exception as e:
                raise ValueError(
                    f"Error flushing metrics {metric} at path {path}."
                ) from e

    def flush(self, key: Path) -> None:
        """Clear the accumulated values of the leaf at ``key``.

        Parameters
        ----------
        key : Path
            Path of the leaf.

        Raises
        ------
        KeyError
            If no leaf exists at ``key``.
        ValueError
            If the leaf cannot be cleared.
        """

        metric = self._refs.get(key)

        if metric is None:
            raise KeyError(f"Unknown logger path: {key}")

        try:
            metric.flush()
        except Exception as e:
            raise ValueError(f"Error flushing metric {metric} at path {key}.") from e
