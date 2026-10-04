"""Unit tests for the ``override`` decorator in ``core.annotations``.

Only the name check is effective (see the module's Notes): the decorator
returns the original method unchanged when the parent class exposes the same
attribute name and raises ``NameError`` otherwise. The subclass check of the
inner descriptor is dead code by design and is not exercised here.
"""

from __future__ import annotations

import pytest

from core.annotations import override


class _Parent:
    def run(self):
        return "parent"

    attr = 1


@pytest.mark.unit
def test_override_returns_method_unchanged():
    def run(self):
        return "child"

    decorated = override(_Parent)(run)
    assert decorated is run


@pytest.mark.unit
def test_override_in_class_body_keeps_behaviour():
    class Child(_Parent):
        @override(_Parent)
        def run(self):
            return "child"

    assert Child().run() == "child"
    assert Child.run.__name__ == "run"


@pytest.mark.unit
def test_override_accepts_non_method_attribute_names():
    def attr(self):
        return None

    assert override(_Parent)(attr) is attr


@pytest.mark.unit
def test_override_raises_name_error_for_unknown_method():
    with pytest.raises(NameError, match="missing must override"):

        class Child(_Parent):  # noqa: F841
            @override(_Parent)
            def missing(self):
                return None


@pytest.mark.unit
def test_override_does_not_check_subclass_relationship():
    class Unrelated:
        @override(_Parent)
        def run(self):
            return "unrelated"

    assert Unrelated().run() == "unrelated"


@pytest.mark.unit
def test_inner_descriptor_would_enforce_subclass_if_it_were_bound():
    """Exercise the ``OverrideCheck`` descriptor that ``override`` never binds.

    ``decorator`` builds an ``OverrideCheck`` and discards it, so the subclass
    check documented in the module Notes is unreachable through the public
    decorator. The class is pulled out of the closure here to pin down what
    it would do if it were bound: bind the function on a proper subclass and
    raise ``TypeError`` on an unrelated owner.
    """
    import inspect

    override_check = inspect.getclosurevars(override(_Parent)).nonlocals[
        "OverrideCheck"
    ]

    def run(self):
        return "bound"

    class Child(_Parent):
        pass

    override_check(run, _Parent).__set_name__(Child, "run")
    assert Child().run() == "bound"

    class Unrelated:
        pass

    with pytest.raises(TypeError, match="must be a subclass of _Parent"):
        override_check(run, _Parent).__set_name__(Unrelated, "run")
