"""Declarative YAML loader that turns a configuration file into live objects.

A configuration is a YAML mapping with an ``experiment`` section describing an
object to build and an optional ``run`` section listing method calls to apply
to it. Nodes are resolved recursively with a small set of reserved keys
(``_target_``, ``_symbol_``, ``_args_``, ``_calls_``, ``_tuple_``), in the
spirit of dependency-injection configuration files: the YAML names Python
classes by import path and supplies their arguments, so an experiment is
described without a script. Every failure is reported as a ``ConfigError``
that carries the YAML path of the failing node and chains the original
exception. The module is used by the command line in ``core.config.cli`` and
can be called directly from Python or notebooks.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import yaml

_RESERVED = {"_target_", "_symbol_", "_args_", "_calls_", "_tuple_"}


class ConfigError(ValueError):
    """Raised when a declarative config cannot be resolved.

    Every error of this module is wrapped in a ``ConfigError``: a missing file,
    invalid YAML, an import path that cannot be imported, a constructor or
    method that raises, and a malformed special form. The message names the
    failing node with a path such as ``$.experiment.inner``, and the original
    exception, when there is one, is available as ``__cause__``. It subclasses
    ``ValueError``.

    When to use: catch it around ``load_experiment`` or ``run_experiment`` to
    report a configuration problem without a stack of unrelated exceptions.

    Examples
    --------
    >>> import_symbol("fractions.Missing")
    Traceback (most recent call last):
    ...
    core.config.yaml.ConfigError: 'Missing' does not exist in 'fractions'
    >>> issubclass(ConfigError, ValueError)
    True
    """


def import_symbol(path: str) -> Any:
    """Import a class, function or other object from its dotted path.

    The path is split at the last dot: the left part is imported as a module and
    the right part is looked up on it as an attribute.

    Parameters
    ----------
    path : str
        Dotted import path of the form ``package.module.symbol``.

    Returns
    -------
    Any
        The object found at that path, not called.

    Raises
    ------
    ConfigError
        If ``path`` has no dot, if the module cannot be imported (the import
        error is chained), or if the module has no such attribute.

    When to use: to resolve a class or function named in a configuration; it is
    what ``resolve`` uses for ``_target_`` and ``_symbol_``.

    Examples
    --------
    >>> import_symbol("fractions.Fraction")
    <class 'fractions.Fraction'>
    """

    module_name, sep, symbol_name = path.rpartition(".")

    if not sep:
        raise ConfigError(
            f"Invalid import path {path!r}. Expected 'package.module.symbol'."
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ConfigError(f"Could not import module {module_name!r}") from exc

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ConfigError(f"{symbol_name!r} does not exist in {module_name!r}") from exc


def resolve(value: Any, *, path: str = "$") -> Any:
    """Recursively resolve a parsed YAML value into Python objects.

    Lists and mappings are walked; scalars are returned unchanged. A mapping
    that contains one of the reserved keys is a special form:

    ``_tuple_: [...]``
        Builds a tuple from the resolved items. The mapping must have no
        other key.
    ``_symbol_: package.module.Name``
        Imports the object without calling it. The mapping must have no other
        key.
    ``_target_: package.module.Class``
        Instantiates the target with every other key (except ``_args_`` and
        ``_calls_``) as a recursively resolved keyword argument.
    ``_args_: [...]``
        Positional arguments of the target; only recognised next to
        ``_target_`` (without it the key is kept as ordinary data).
    ``_calls_: [{method: {key: value}}]``
        Methods applied in order to the constructed object through
        ``apply_calls``; only recognised next to ``_target_``.

    ``_tuple_`` is checked first, then ``_symbol_``, then ``_target_``. A
    mapping without a reserved key is resolved value by value into a new dict.

    Parameters
    ----------
    value : Any
        A value produced by ``yaml.safe_load``: scalar, list or mapping.
    path : str, optional
        YAML path of ``value``, prefixed to error messages. Default ``"$"``.

    Returns
    -------
    Any
        The resolved object: a new list or dict for containers, a tuple, an
        imported symbol, or the result of the constructor and of its
        ``_calls_``.

    Raises
    ------
    ConfigError
        If a ``_tuple_`` or ``_symbol_`` mapping has sibling keys, if a target
        cannot be imported, or if constructing it (or applying its calls)
        fails. The message starts with the YAML path of the failing node.

    When to use: when a configuration is already parsed and only a fragment
    needs building; ``load_experiment`` calls it on the ``experiment`` section.

    Examples
    --------
    >>> resolve({"_target_": "fractions.Fraction", "_args_": [3, 6]})
    Fraction(1, 2)
    >>> resolve({"shape": {"_tuple_": [2, 3]}, "dtype": {"_symbol_": "builtins.float"}})
    {'shape': (2, 3), 'dtype': <class 'float'>}
    """

    if isinstance(value, list):
        return [resolve(item, path=f"{path}[{i}]") for i, item in enumerate(value)]

    if not isinstance(value, dict):
        return value

    if "_tuple_" in value:
        if set(value) != {"_tuple_"}:
            raise ConfigError(f"{path}: _tuple_ cannot have sibling keys")

        return tuple(
            resolve(item, path=f"{path}._tuple_[{i}]")
            for i, item in enumerate(value["_tuple_"])
        )

    if "_symbol_" in value:
        if set(value) != {"_symbol_"}:
            raise ConfigError(f"{path}: _symbol_ cannot have sibling keys")

        return import_symbol(value["_symbol_"])

    if "_target_" in value:
        target = import_symbol(value["_target_"])
        args = resolve(value.get("_args_", []), path=f"{path}._args_")
        kwargs = {
            key: resolve(item, path=f"{path}.{key}")
            for key, item in value.items()
            if key not in _RESERVED
        }

        try:
            obj = target(*args, **kwargs)
        except Exception as exc:
            raise ConfigError(
                f"{path}: failed constructing {value['_target_']!r}"
            ) from exc

        return apply_calls(obj, value.get("_calls_", []), path=f"{path}._calls_")

    return {key: resolve(item, path=f"{path}.{key}") for key, item in value.items()}


def apply_calls(
    obj: Any, calls: list[dict[str, Any]] | None, *, path: str = "$._calls_"
) -> Any:
    """Apply a list of method calls to an object, in order.

    Each entry is a mapping with exactly one key, the method name, whose value
    is a mapping of keyword arguments (``null`` means none). The special key
    ``_args_`` supplies positional arguments, and every argument is resolved
    with ``resolve`` first. When a method returns something other than
    ``None``, that result replaces the object for the following calls, which
    supports fluent configuration methods that return ``self`` as well as
    methods that return a new object or a result; a method that returns
    ``None`` leaves the object unchanged.

    Parameters
    ----------
    obj : Any
        Object the first method is looked up on.
    calls : list[dict[str, Any]] or None
        Call specifications. ``None`` or an empty list applies nothing.
    path : str, optional
        YAML path of ``calls``, prefixed to error messages. Default
        ``"$._calls_"``.

    Returns
    -------
    Any
        The object after the last call: ``obj`` itself, or the last non-``None``
        value returned by a method.

    Raises
    ------
    ConfigError
        If an entry is not a one-key mapping, if the arguments are not a
        mapping, if the method does not exist or is not callable, or if the call
        raises (the original exception is chained).

    When to use: to drive a builder-style object from a configuration, as the
    ``_calls_`` and ``run`` sections do.

    Examples
    --------
    >>> from fractions import Fraction
    >>> apply_calls(Fraction(7, 3), [{"limit_denominator": {"max_denominator": 2}}])
    Fraction(5, 2)
    """

    for i, call_spec in enumerate(calls or []):
        call_path = f"{path}[{i}]"

        if not isinstance(call_spec, dict) or len(call_spec) != 1:
            raise ConfigError(f"{call_path}: expected exactly one method name")

        method_name, params = next(iter(call_spec.items()))
        params = params or {}

        if not isinstance(params, dict):
            raise ConfigError(
                f"{call_path}.{method_name}: method arguments must be a mapping"
            )

        args = resolve(
            params.get("_args_", []), path=f"{call_path}.{method_name}._args_"
        )
        kwargs = {
            key: resolve(value, path=f"{call_path}.{method_name}.{key}")
            for key, value in params.items()
            if key != "_args_"
        }

        try:
            method = getattr(obj, method_name)
        except AttributeError as exc:
            raise ConfigError(
                f"{call_path}: {type(obj).__name__} has no method {method_name!r}"
            ) from exc

        if not callable(method):
            raise ConfigError(f"{call_path}: {method_name!r} is not callable")

        try:
            result = method(*args, **kwargs)
        except Exception as exc:
            raise ConfigError(
                f"{call_path}: call to {type(obj).__name__}.{method_name}() failed"
            ) from exc

        # Fluent config methods return `self`.
        # build_optimizer() returns an optimizer.
        # train() may return a result.
        if result is not None:
            obj = result

    return obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict with safe parsing and variable expansion.

    Environment variables (``$NAME`` and ``${NAME}``) are expanded over the
    whole file text with ``os.path.expandvars`` before parsing; a reference to
    an undefined variable is left as written. The text is then parsed with
    ``yaml.safe_load``, which refuses Python object tags.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the YAML file.

    Returns
    -------
    dict[str, Any]
        The top-level mapping of the document.

    Raises
    ------
    ConfigError
        If ``path`` is not an existing file, if the YAML is invalid (the parser
        error is chained), or if the top level is not a mapping.

    When to use: to read a configuration file without building anything.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "config.yaml"
    ...     _ = path.write_text("seeds: [1, 2]\\nname: demo\\n")
    ...     load_yaml(path)
    {'seeds': [1, 2], 'name': 'demo'}
    """

    path = Path(path)

    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    text = os.path.expandvars(path.read_text())

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML document must be a mapping")

    return data


def load_experiment(path: str | Path) -> Any:
    """Construct the configured experiment without running it.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the YAML file; its ``experiment`` section is resolved.
        The ``run`` section is ignored.

    Returns
    -------
    Any
        The object described by the ``experiment`` section.

    Raises
    ------
    ConfigError
        If the file cannot be loaded (see ``load_yaml``), if it has no
        top-level ``experiment`` key, or if resolving it fails; nested errors
        carry paths starting with ``$.experiment``.

    When to use: to validate a configuration or to obtain the experiment object
    in a notebook; the ``check`` command of ``core.config.cli`` is built on it.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "config.yaml"
    ...     _ = path.write_text(
    ...         "experiment:\\n  _target_: fractions.Fraction\\n  _args_: [3, 6]\\n"
    ...     )
    ...     load_experiment(path)
    Fraction(1, 2)
    """

    data = load_yaml(path)

    if "experiment" not in data:
        raise ConfigError("Config requires a top-level 'experiment:' section")

    return resolve(data["experiment"], path="$.experiment")


def run_experiment(path: str | Path) -> Any:
    """Construct the experiment, then apply the ``run`` section to it.

    The ``experiment`` section is resolved as in ``load_experiment``, and the
    optional top-level ``run`` list is applied with ``apply_calls`` (under the
    path ``$.run``).

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the YAML file.

    Returns
    -------
    Any
        The experiment after the last ``run`` call, or the last non-``None``
        value returned by a call (see ``apply_calls``). Without a ``run``
        section, the constructed experiment itself.

    Raises
    ------
    ConfigError
        If the file cannot be loaded, if it has no ``experiment`` key, or if
        constructing the experiment or one of the ``run`` calls fails.

    When to use: to execute a full configuration, for example a training run
    whose last call is ``train``; the ``run`` command of ``core.config.cli``
    calls it.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> text = (
    ...     "experiment:\\n  _target_: fractions.Fraction\\n  _args_: [7, 3]\\n"
    ...     "run:\\n  - limit_denominator:\\n      max_denominator: 2\\n"
    ... )
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "config.yaml"
    ...     _ = path.write_text(text)
    ...     run_experiment(path)
    Fraction(5, 2)
    """

    data = load_yaml(path)

    if "experiment" not in data:
        raise ConfigError("Config requires a top-level 'experiment:' section")

    experiment = resolve(data["experiment"], path="$.experiment")

    return apply_calls(experiment, data.get("run", []), path="$.run")
