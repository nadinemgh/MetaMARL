from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import yaml

_RESERVED = {"_target_", "_symbol_", "_args_", "_calls_", "_tuple_"}


class ConfigError(ValueError):
    """Raised when a declarative config cannot be resolved."""


def import_symbol(path: str) -> Any:
    """Import `package.module.symbol`."""

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
    """
    Recursively resolve a YAML value.

    Supported special forms:

        _symbol_: package.module.Name

    imports an object without calling it.

        _target_: package.module.Class
        key: value

    instantiates the target with recursively resolved kwargs.

        _args_: [...]

    supplies positional arguments.

        _calls_:
          - method:
              key: value

    invokes methods sequentially on the constructed object.

        _tuple_: [...]

    explicitly creates a tuple.
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
    """Apply fluent method calls to an object in YAML order."""

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
    """Read YAML using safe YAML parsing and environment expansion."""

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
    """Construct the configured experiment without running it."""

    data = load_yaml(path)

    if "experiment" not in data:
        raise ConfigError("Config requires a top-level 'experiment:' section")

    return resolve(data["experiment"], path="$.experiment")


def run_experiment(path: str | Path) -> Any:
    """Construct and execute an experiment."""

    data = load_yaml(path)

    if "experiment" not in data:
        raise ConfigError("Config requires a top-level 'experiment:' section")

    experiment = resolve(data["experiment"], path="$.experiment")

    return apply_calls(experiment, data.get("run", []), path="$.run")
