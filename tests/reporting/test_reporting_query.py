"""``Query``: construction-time validation and the ``y_paths`` normalisation.

Ported from the August ``test_query_and_reporter.py`` and ``test_reporter_edges.py``.
The August ``reduce`` argument, the ``"*"`` wildcard and ``has_wildcards`` no
longer exist: reduction is expressed by ``ReduceProtocol`` tokens inside a path
and the standard-deviation band is requested with ``error`` plus ``error_path``.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.metrics.enums import ReduceProtocol
from core.reporting.query import Query

SERIES = ReduceProtocol.SERIES
MEAN = ReduceProtocol.MEAN

pytestmark = pytest.mark.unit


class TestYPaths:
    def test_a_single_path_is_wrapped(self):
        query = Query(title="t", x=("iter",), y=("a", "b"))

        assert query.y_paths == (("a", "b"),)

    def test_a_tuple_of_paths_is_kept(self):
        query = Query(title="t", x=("iter",), y=(("a",), ("b", "c")))

        assert query.y_paths == (("a",), ("b", "c"))

    def test_a_path_may_start_with_a_reduction_token(self):
        query = Query(title="t", x=("iter",), y=(SERIES, "a"))

        assert query.y_paths == ((SERIES, "a"),)


class TestDefaultsAndImmutability:
    def test_optional_fields_default_to_no_decoration(self):
        query = Query(title="t", x=("iter",), y=("a",))

        assert query.error == "none" and query.error_path is None
        assert query.color is None and query.colorscale is None
        assert query.legend_labels is None and query.plot_modes is None
        assert query.show_group_labels is True

    def test_query_is_frozen(self):
        query = Query(title="t", x=("iter",), y=("a",))

        with pytest.raises(dataclasses.FrozenInstanceError):
            query.title = "u"  # type: ignore[misc]


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"x": (), "y": ("a",)}, "x path cannot be empty"),
            ({"x": ("iter",), "y": ()}, "y path cannot be empty"),
            (
                {"x": ("iter",), "y": (("a",), ("b",)), "legend_labels": ("only",)},
                "legend_labels must have the same length",
            ),
            (
                {"x": ("iter",), "y": ("a",), "plot_modes": ("lines", "markers")},
                "plot_modes must have the same length",
            ),
            (
                {"x": ("iter",), "y": ("a",), "color_label": "c"},
                "color_label requires a color path",
            ),
            (
                {"x": ("iter",), "y": ("a",), "colorscale": "Viridis"},
                "colorscale requires a color path",
            ),
            (
                {"x": ("iter",), "y": ("a",), "error": "var"},
                "Unsupported error statistic: 'var'",
            ),
            (
                {"x": ("iter",), "y": ("a",), "error_path": ("a",)},
                "error_path requires an error statistic",
            ),
            (
                {"x": ("iter",), "y": ("a",), "error": "std"},
                "error='std' requires error_path",
            ),
            (
                {
                    "x": ("iter",),
                    "y": ("d", MEAN, "a"),
                    "error": "std",
                    "error_path": ("d", MEAN),
                },
                "not its reduction operator",
            ),
            (
                {
                    "x": ("iter",),
                    "y": ("d", SERIES, "a"),
                    "error": "std",
                    "error_path": ("d",),
                },
                r"must be followed by ReduceProtocol.MEAN",
            ),
        ],
    )
    def test_invalid_combination_is_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            Query(title="t", **kwargs)

    def test_error_path_may_match_any_of_several_y_paths(self):
        query = Query(
            title="t",
            x=("iter",),
            y=(("loss",), ("d", MEAN, "a")),
            error="std",
            error_path=("d",),
        )

        assert query.error_path == ("d",)

    def test_matching_labels_and_modes_are_accepted(self):
        query = Query(
            title="t",
            x=("iter",),
            y=(("a",), ("b",)),
            legend_labels=("A", "B"),
            plot_modes=("lines", "markers"),
            color=("c",),
            color_label="C",
            colorscale="Viridis",
        )

        assert query.legend_labels == ("A", "B")
        assert query.colorscale == "Viridis"
