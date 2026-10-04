"""``core.config.yaml``: resolving, loading and validating declarative configs.

Every document is a small YAML text written under ``tmp_path`` or a Python
value passed to ``resolve``. Targets are standard-library classes
(``fractions.Fraction``, ``collections.OrderedDict``) and the ``Builder`` class
below, which has fluent methods for the ``_calls_`` forms. Errors carry the
YAML path of the failing node, so the tests assert on the paths too.
"""

import collections
import fractions
import textwrap

import pytest

from core.config.yaml import (
    ConfigError,
    apply_calls,
    import_symbol,
    load_experiment,
    load_yaml,
    resolve,
    run_experiment,
)

HERE = __name__


class Builder:
    """Records its method calls; ``add`` is fluent, ``finish`` returns a value."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []
        self.not_callable = 3

    def add(self, *args, **kwargs):
        self.calls.append(("add", args, kwargs))
        return self

    def note(self, **kwargs):
        self.calls.append(("note", (), kwargs))

    def finish(self):
        return "finished"

    def explode(self):
        raise RuntimeError("boom")


def write(tmp_path, text: str, name: str = "config.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


@pytest.mark.unit
def test_import_symbol_returns_the_named_object():
    assert import_symbol("fractions.Fraction") is fractions.Fraction
    assert import_symbol("os.path.join").__name__ == "join"


@pytest.mark.unit
def test_import_symbol_rejects_a_path_without_a_module():
    with pytest.raises(ConfigError, match="Invalid import path 'Fraction'"):
        import_symbol("Fraction")


@pytest.mark.unit
def test_import_symbol_names_a_module_that_cannot_be_imported():
    with pytest.raises(ConfigError, match="Could not import module 'no_such_pkg'") as e:
        import_symbol("no_such_pkg.thing")

    assert isinstance(e.value.__cause__, ImportError)


@pytest.mark.unit
def test_import_symbol_names_a_missing_symbol():
    with pytest.raises(ConfigError, match="'Missing' does not exist in 'fractions'"):
        import_symbol("fractions.Missing")


@pytest.mark.unit
@pytest.mark.parametrize("value", [3, 2.5, "text", None, True])
def test_resolve_returns_scalars_unchanged(value):
    assert resolve(value) is value


@pytest.mark.unit
def test_resolve_walks_nested_lists_and_plain_mappings():
    value = {"a": [1, {"b": [2, 3]}], "c": {"d": 4}}

    assert resolve(value) == value
    assert resolve(value) is not value


@pytest.mark.unit
def test_resolve_builds_a_tuple_from_the_tuple_form():
    resolved = resolve({"_tuple_": [1, {"_tuple_": [2, 3]}, [4]]})

    assert resolved == (1, (2, 3), [4])
    assert isinstance(resolved, tuple)


@pytest.mark.unit
def test_resolve_rejects_tuple_siblings_with_their_path():
    with pytest.raises(ConfigError, match=r"\$\.x: _tuple_ cannot have sibling keys"):
        resolve({"x": {"_tuple_": [1], "other": 2}})


@pytest.mark.unit
def test_resolve_imports_a_symbol_without_calling_it():
    assert resolve({"_symbol_": "fractions.Fraction"}) is fractions.Fraction


@pytest.mark.unit
def test_resolve_rejects_symbol_siblings_with_their_path():
    value = {"_symbol_": "fractions.Fraction", "x": 1}

    with pytest.raises(ConfigError, match=r"\$: _symbol_ cannot have sibling keys"):
        resolve(value)


@pytest.mark.unit
def test_resolve_instantiates_a_target_with_args_and_kwargs():
    value = {"_target_": f"{HERE}.Builder", "_args_": [1, 2], "name": "n", "k": [3]}

    built = resolve(value)

    assert isinstance(built, Builder)
    assert built.args == (1, 2)
    assert built.kwargs == {"name": "n", "k": [3]}


@pytest.mark.unit
def test_resolve_resolves_the_arguments_of_a_target_recursively():
    value = {
        "_target_": f"{HERE}.Builder",
        "_args_": [{"_target_": "fractions.Fraction", "_args_": [1, 4]}],
        "dtype": {"_symbol_": "fractions.Fraction"},
        "pair": {"_tuple_": [1, 2]},
    }

    built = resolve(value)

    assert built.args == (fractions.Fraction(1, 4),)
    assert built.kwargs == {"dtype": fractions.Fraction, "pair": (1, 2)}


@pytest.mark.unit
def test_resolve_reports_a_failing_construction_with_its_path_and_cause():
    value = {"a": [{"_target_": "fractions.Fraction", "_args_": [1, 0]}]}

    with pytest.raises(ConfigError, match=r"\$\.a\[0\]: failed constructing") as e:
        resolve(value)

    assert isinstance(e.value.__cause__, ZeroDivisionError)


@pytest.mark.unit
def test_resolve_reports_an_unimportable_target():
    with pytest.raises(ConfigError, match="Could not import module 'nowhere'"):
        resolve({"_target_": "nowhere.Thing"})


@pytest.mark.unit
def test_resolve_applies_the_calls_of_a_target_in_order():
    value = {
        "_target_": f"{HERE}.Builder",
        "_calls_": [
            {"add": {"_args_": [1], "x": 2}},
            {"note": {"y": {"_tuple_": [3]}}},
            {"add": None},
        ],
    }

    built = resolve(value)

    assert built.calls == [
        ("add", (1,), {"x": 2}),
        ("note", (), {"y": (3,)}),
        ("add", (), {}),
    ]


@pytest.mark.unit
def test_apply_calls_without_calls_returns_the_object():
    obj = Builder()

    assert apply_calls(obj, None) is obj
    assert apply_calls(obj, []) is obj


@pytest.mark.unit
def test_apply_calls_replaces_the_object_by_a_non_none_result():
    assert apply_calls(Builder(), [{"finish": {}}]) == "finished"


@pytest.mark.unit
def test_apply_calls_keeps_the_object_when_a_method_returns_none():
    obj = Builder()

    assert apply_calls(obj, [{"note": {"z": 1}}]) is obj
    assert obj.calls == [("note", (), {"z": 1})]


@pytest.mark.unit
@pytest.mark.parametrize(
    "calls, message",
    [
        (["add"], r"\$\._calls_\[0\]: expected exactly one method name"),
        ([{"add": {}, "note": {}}], r"expected exactly one method name"),
        ([{"add": {}}, 5], r"\$\._calls_\[1\]: expected exactly one method name"),
        ([{"add": [1, 2]}], r"\[0\]\.add: method arguments must be a mapping"),
        ([{"missing": {}}], r"\[0\]: Builder has no method 'missing'"),
        ([{"not_callable": {}}], r"\[0\]: 'not_callable' is not callable"),
        ([{"explode": {}}], r"\[0\]: call to Builder\.explode\(\) failed"),
    ],
)
def test_apply_calls_rejects_malformed_or_failing_calls(calls, message):
    with pytest.raises(ConfigError, match=message):
        apply_calls(Builder(), calls)


@pytest.mark.unit
def test_apply_calls_chains_the_original_error_of_a_failing_method():
    with pytest.raises(ConfigError) as e:
        apply_calls(Builder(), [{"explode": {}}])

    assert isinstance(e.value.__cause__, RuntimeError)
    assert str(e.value.__cause__) == "boom"


@pytest.mark.unit
def test_load_yaml_reads_a_mapping(tmp_path):
    path = write(tmp_path, "a: 1\nb: [2, 3]\n")

    assert load_yaml(path) == {"a": 1, "b": [2, 3]}
    assert load_yaml(str(path)) == {"a": 1, "b": [2, 3]}


@pytest.mark.unit
def test_load_yaml_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("METAMARL_TEST_VALUE", "42")
    path = write(tmp_path, "a: $METAMARL_TEST_VALUE\nb: ${METAMARL_TEST_VALUE}0\n")

    assert load_yaml(path) == {"a": 42, "b": 420}


@pytest.mark.unit
def test_load_yaml_names_a_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="Config file not found: .*absent.yaml"):
        load_yaml(tmp_path / "absent.yaml")


@pytest.mark.unit
def test_load_yaml_rejects_a_directory(tmp_path):
    with pytest.raises(ConfigError, match="Config file not found"):
        load_yaml(tmp_path)


@pytest.mark.unit
def test_load_yaml_wraps_a_syntax_error(tmp_path):
    path = write(tmp_path, "a: [1, 2\nb: : 3\n")

    with pytest.raises(ConfigError, match="Invalid YAML in .*config.yaml") as e:
        load_yaml(path)

    assert type(e.value.__cause__).__module__.startswith("yaml")


@pytest.mark.unit
@pytest.mark.parametrize("text", ["- 1\n- 2\n", "just a string\n", "", "42\n"])
def test_load_yaml_requires_a_top_level_mapping(tmp_path, text):
    path = write(tmp_path, text)

    with pytest.raises(ConfigError, match="Top-level YAML document must be a mapping"):
        load_yaml(path)


@pytest.mark.unit
def test_load_yaml_refuses_python_object_tags(tmp_path):
    path = write(tmp_path, "a: !!python/object/apply:os.getcwd []\n")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_yaml(path)


@pytest.mark.unit
def test_load_experiment_builds_the_experiment_section(tmp_path):
    path = write(
        tmp_path,
        """
        experiment:
          _target_: fractions.Fraction
          _args_: [3, 6]
        run:
          - limit_denominator: {}
        """,
    )

    assert load_experiment(path) == fractions.Fraction(1, 2)


@pytest.mark.unit
def test_load_experiment_does_not_run_the_run_section(tmp_path):
    path = write(
        tmp_path,
        f"""
        experiment:
          _target_: {HERE}.Builder
        run:
          - add: {{}}
        """,
    )

    assert load_experiment(path).calls == []


@pytest.mark.unit
def test_load_experiment_requires_an_experiment_section(tmp_path):
    path = write(tmp_path, "other: 1\n")

    with pytest.raises(ConfigError, match="requires a top-level 'experiment:'"):
        load_experiment(path)


@pytest.mark.unit
def test_load_experiment_gives_the_path_of_a_nested_failure(tmp_path):
    path = write(
        tmp_path,
        """
        experiment:
          _target_: collections.OrderedDict
          inner:
            _target_: fractions.Fraction
            _args_: [1, 0]
        """,
    )

    with pytest.raises(
        ConfigError, match=r"\$\.experiment\.inner: failed constructing"
    ):
        load_experiment(path)


@pytest.mark.unit
def test_run_experiment_applies_the_run_section_to_the_experiment(tmp_path):
    path = write(
        tmp_path,
        f"""
        experiment:
          _target_: {HERE}.Builder
        run:
          - add: {{_args_: [1]}}
          - add: {{_args_: [2]}}
        """,
    )

    experiment = run_experiment(path)

    assert [call[1] for call in experiment.calls] == [(1,), (2,)]


@pytest.mark.unit
def test_run_experiment_returns_the_last_non_none_result(tmp_path):
    path = write(
        tmp_path,
        f"""
        experiment:
          _target_: {HERE}.Builder
        run:
          - finish: {{}}
        """,
    )

    assert run_experiment(path) == "finished"


@pytest.mark.unit
def test_run_experiment_without_a_run_section_returns_the_experiment(tmp_path):
    path = write(
        tmp_path,
        """
        experiment:
          _target_: collections.OrderedDict
          a: 1
        """,
    )

    assert run_experiment(path) == collections.OrderedDict(a=1)


@pytest.mark.unit
def test_run_experiment_requires_an_experiment_section(tmp_path):
    path = write(tmp_path, "run: []\n")

    with pytest.raises(ConfigError, match="requires a top-level 'experiment:'"):
        run_experiment(path)


@pytest.mark.unit
def test_run_experiment_names_the_failing_run_call(tmp_path):
    path = write(
        tmp_path,
        f"""
        experiment:
          _target_: {HERE}.Builder
        run:
          - add: {{}}
          - explode: {{}}
        """,
    )

    with pytest.raises(ConfigError, match=r"\$\.run\[1\]: call to Builder.explode"):
        run_experiment(path)


@pytest.mark.unit
def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)
