"""The module example of ``core.reporting.query`` is a real fishery query.

The example used to show a path that no schema resolves; checking it against
the fishery's declared queries keeps it from drifting again.
"""

import doctest

import pytest

import core.reporting.query as query_module
from examples.bilevel_fishery.queries import INNER_QUERIES


@pytest.mark.unit
def test_the_module_example_is_one_of_the_fishery_queries():
    (test,) = [
        t
        for t in doctest.DocTestFinder().find(query_module)
        if t.name == query_module.__name__
    ]
    runner = doctest.DocTestRunner()
    runner.run(test, clear_globs=False)

    assert runner.failures == 0
    assert test.globs["query"] in INNER_QUERIES
